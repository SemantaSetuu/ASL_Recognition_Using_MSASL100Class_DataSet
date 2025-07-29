# train_asl_cnn_attn.py  – rev-2 (uses Stage-A backbone)
#stage_B
"""
CNN + Temporal-Attention for MS-ASL-100
──────────────────────────────────────
• MediaPipe-Holistic crop (face + hands) → 128×128 square
• MobileNet-V2 backbone **initialised from mnet_frame_pretrain.pth**
• Temporal Attention pooling (learned soft weights) instead of LSTM
• Class-balanced sampler + mini-batch Mix-Up
• Fits a 6 GB GTX 1650 @ batch-size 4  (drop to 2 if OOM)

Outputs
───────
cnn_attn_final.pth     – fine-tuned weights
hist_cnn_attn.npy      – training / validation history
"""

from __future__ import annotations
import math, time, functools, numpy as np, torch, torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision.models import mobilenet_v2
from torchvision.transforms import (Compose, Resize, ToTensor, Normalize,
                                     RandomHorizontalFlip, ColorJitter)

################ PATHS
ROOT        = Path(r"C:\Users\seman\Desktop\clg\2nd_sem\research_practicum\code\data\images")
TRAIN_DIR   = ROOT / "train"
VAL_DIR     = ROOT / "val"
BACKBONE_CKPT = Path(r"C:\Users\seman\Desktop\clg\2nd_sem\research_practicum\code\src\cnn+tsm\mnet_frame_pretrain_correct.pth")

OUT_W = "stage_b_cnn_attn_final.pth"
OUT_H = "stage_b_hist_cnn_attn.npy"

##################### HYPER-PARAMS
NUM_FRAMES   = 32
IMG_SIZE     = 160
BATCH_SZ     = 4
EPOCHS       = 30
LR_BB_FROZEN = 1e-4  # lowered
LR_FULL      = 5e-4  # lowered
FREEZE_EPOCHS= 5     # increased
VAL_EVERY    = 1
MIXUP_ALPHA  = 0.1   # slightly lowered
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"


############## Transforms
_IM_MEAN = (0.485, 0.456, 0.406)
_IM_STD  = (0.229, 0.224, 0.225)
_tx_train = Compose([
    Resize((IMG_SIZE, IMG_SIZE)),
    RandomHorizontalFlip(),
    ColorJitter(0.2,0.2,0.2,0.1),
    ToTensor(), Normalize(_IM_MEAN, _IM_STD)
])
_tx_eval  = Compose([
    Resize((IMG_SIZE, IMG_SIZE)),
    ToTensor(), Normalize(_IM_MEAN, _IM_STD)
])

################ MediaPipe crop helpers
try:
    import mediapipe as mp
    _mp_holistic = mp.solutions.holistic.Holistic(
        static_image_mode=True, model_complexity=0,
        refine_face_landmarks=False, enable_segmentation=False)
except ModuleNotFoundError:
    _mp_holistic = None  # fall back to centre-crop

_bbox_cache: dict[Path, tuple[int,int,int,int]] = {}
@functools.lru_cache(maxsize=4096)
def _detect_bbox(img_path: Path):
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    res = _mp_holistic.process(np.array(img)[:,:,::-1])

    def extract(lms):
        return [(lm.x * w, lm.y * h) for lm in lms.landmark] if lms else []

    hand_points = extract(res.left_hand_landmarks) + extract(res.right_hand_landmarks)
    face_points = extract(res.face_landmarks)

    # PRIORITIZE HANDS (Crucial for ASL)
    if hand_points:
        xs, ys = zip(*hand_points)
    elif face_points:
        xs, ys = zip(*face_points)
    else:
        return (0, 0, w, h)

    padding = 60  # generous padding around detected landmarks
    x1, y1 = max(min(xs)-padding, 0), max(min(ys)-padding, 0)
    x2, y2 = min(max(xs)+padding, w), min(max(ys)+padding, h)

    # Square bbox
    side = max(x2 - x1, y2 - y1)
    cx, cy = (x1+x2)//2, (y1+y2)//2
    half = side//2
    l, t = int(max(cx-half, 0)), int(max(cy-half, 0))
    r, b = int(min(cx+half, w)), int(min(cy+half, h))
    return l, t, r, b


def _crop(img_path: Path) -> Image.Image:
    img = Image.open(img_path).convert("RGB")
    if img_path.parent not in _bbox_cache:
        _bbox_cache[img_path.parent] = _detect_bbox(img_path)
    l,t,r,b = _bbox_cache[img_path.parent]
    if r != -1:
        img = img.crop((l,t,r,b))
    else:  # centre-crop fallback
        w,h = img.size; side = min(w,h)
        img = img.crop(((w-side)//2, (h-side)//2,
                        (w+side)//2, (h+side)//2))
    return img

############## Dataset
class ASLSeqDataset(Dataset):
    def __init__(self, root: Path, split="train", T=32):
        self.root  = Path(root)
        self.split = split.lower()
        self.T     = T
        self.cls_names = sorted(d.name for d in self.root.iterdir() if d.is_dir())
        self.cls2idx   = {c:i for i,c in enumerate(self.cls_names)}
        self.samples   = []
        for cls in self.cls_names:
            for vid in (self.root/cls).iterdir():
                if vid.is_dir():
                    frames = sorted(vid.glob("*.jpg"))
                    if len(frames) >= 2:
                        self.samples.append((frames, self.cls2idx[cls]))
        self.tx = _tx_train if split=="train" else _tx_eval

    def __len__(self): return len(self.samples)

    def _sel(self,L):
        base = np.linspace(0,L-1,self.T,dtype=int)
        if self.split=="train":
            base = np.clip(base + np.random.randint(-1,2,size=self.T), 0, L-1)
        return base

    def __getitem__(self, idx):
        frames,label = self.samples[idx]
        idxs = self._sel(len(frames))
        clip = torch.stack([ self.tx(_crop(frames[i])) for i in idxs ], dim=1)  # (3,T,H,W)
        return clip, label

# ────────── Model ──────────
class TemporalAttn(nn.Module):
    """MobileNet-V2 per-frame → soft attention → logits"""
    def __init__(self, n_cls, emb_dim=256):
        super().__init__()
        m = mobilenet_v2(weights=None)                      # <-- weights=None
        self.backbone = m.features
        self.pool  = nn.AdaptiveAvgPool2d(1)
        self.proj  = nn.Linear(1280, emb_dim, bias=False)
        self.bn    = nn.BatchNorm1d(emb_dim)
        self.attn  = nn.Linear(emb_dim, 1)
        self.fc    = nn.Linear(emb_dim, n_cls)

    def forward(self,x):                                    # x (B,3,T,H,W)
        B,C,T,H,W = x.shape
        x = x.permute(0,2,1,3,4).reshape(B*T,C,H,W)
        feat = self.pool(self.backbone(x)).flatten(1)       # (B·T,1280)
        feat = F.relu(self.bn(self.proj(feat)))             # (B·T,emb)
        feat = feat.view(B,T,-1)                            # (B,T,emb)
        attn_w = F.softmax(self.attn(feat).squeeze(-1),dim=1) # (B,T)
        emb = (attn_w.unsqueeze(-1)*feat).sum(1)            # (B,emb)
        return self.fc(emb)

############### Mix-Up helpers
def mixup(x,y,alpha=MIXUP_ALPHA):
    if alpha<=0: return x,y,None
    lam = np.random.beta(alpha,alpha)
    idx = torch.randperm(x.size(0))
    return lam*x + (1-lam)*x[idx], (y, y[idx], lam), idx

def ce_mix(ce, logits, target):
    if isinstance(target, tuple):
        y_a,y_b,lam = target
        return lam*ce(logits,y_a)+(1-lam)*ce(logits,y_b)
    return ce(logits,target)

############### Evaluation
@torch.no_grad()
def evaluate(model, loader):
    model.eval(); ce=nn.CrossEntropyLoss()
    tot=correct=loss=0.
    for clip,lab in loader:
        clip,lab = clip.to(DEVICE), lab.to(DEVICE)
        logit = model(clip)
        loss += ce(logit,lab).item()*lab.size(0)
        correct += (logit.argmax(1)==lab).sum().item()
        tot += lab.size(0)
    return loss/tot, 100*correct/tot

################### MAIN
def main():
    torch.backends.cudnn.benchmark = True

    # -------- datasets --------
    train_ds = ASLSeqDataset(TRAIN_DIR,"train",NUM_FRAMES)
    val_ds   = ASLSeqDataset(VAL_DIR,"val",  NUM_FRAMES)

    labels = np.array([lbl for _,lbl in train_ds.samples])
    counts = np.bincount(labels, minlength=len(train_ds.cls_names))
    weights = (1.0/(counts+1e-6))[labels]
    sampler = WeightedRandomSampler(weights.tolist(), len(weights), replacement=True)

    train_ld = DataLoader(train_ds, BATCH_SZ, sampler=sampler,
                          num_workers=0, pin_memory=True, drop_last=True)
    val_ld   = DataLoader(val_ds, BATCH_SZ, shuffle=False,
                          num_workers=0, pin_memory=True)

    # -------- model --------
    model = TemporalAttn(len(train_ds.cls_names)).to(DEVICE)

    # ---- load Stage-A MobileNet backbone ----
    ckpt = torch.load(BACKBONE_CKPT, map_location="cpu")
    bb_state = {k.replace("net.features.",""): v
                for k,v in ckpt.items() if k.startswith("net.features.")}
    missing,_ = model.backbone.load_state_dict(bb_state, strict=False)
    print(f"Loaded Stage-A backbone → {len(bb_state)-len(missing)}/{len(bb_state)} layers")

    # freeze backbone initially
    for p in model.backbone.parameters(): p.requires_grad = False

    opt = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": LR_BB_FROZEN},
        {"params": list(model.pool.parameters()) +
                   list(model.proj.parameters()) +
                   list(model.bn.parameters()) +
                   list(model.attn.parameters()) +
                   list(model.fc.parameters()), "lr": LR_FULL}
    ], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-6)
    ce = nn.CrossEntropyLoss(label_smoothing=0.1)

    hist = {k:[] for k in ["epoch","loss","train_acc","val_loss","val_acc"]}

    print(f"Train clips {len(train_ds)} | Val clips {len(val_ds)} | Classes {len(train_ds.cls_names)}\n")

    # -------- training loop --------
    for ep in range(1, EPOCHS + 1):
        model.train();
        tloss = correct = total = 0
        if ep == FREEZE_EPOCHS + 1:
            for p in model.backbone.parameters():
                p.requires_grad = True
            opt.param_groups[0]["lr"] = LR_FULL * 0.2  # conservative lr change

        for x, y in tqdm(train_ld, desc=f"E{ep:02d}"):
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            loss = ce(logits, y)
            opt.zero_grad();
            loss.backward();
            opt.step()
            tloss += loss.item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)

        sched.step()  # cosine LR scheduler step

        tr_acc = 100 * correct / total
        tr_loss = tloss / total
        val_acc = evaluate(model, val_ld) if ep % VAL_EVERY == 0 else float('nan')

        print(f"Epoch {ep:02d} | LR: {sched.get_last_lr()[0]:.2e} | Train Loss: {tr_loss:.4f} | Train Acc: {tr_acc:.2f}% | Val Loss: {val_acc[0]:.4f} | Val Acc: {val_acc[1]:.2f}%")

        hist["epoch"].append(ep)
        hist["loss"].append(tr_loss)
        hist["train_acc"].append(tr_acc)
        hist["val_acc"].append(val_acc)

    torch.save(model.state_dict(), OUT_W)
    np.save(OUT_H, hist)
    print(f"\nDone!  Weights to {OUT_W} | History to {OUT_H}")

# entry-point
if __name__ == "__main__":
    main()
