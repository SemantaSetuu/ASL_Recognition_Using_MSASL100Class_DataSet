from __future__ import annotations
import math, time, functools, numpy as np, torch, torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision.models import mobilenet_v2
from torchvision.transforms import (
    Compose, RandomResizedCrop, RandomHorizontalFlip, Resize,
    ColorJitter, GaussianBlur, ToTensor, Normalize, RandomErasing)
import mediapipe as mp


from typing import Dict, Any
import json  # for saving history as .json if you prefer
CKPT_FULL = Path(r"C:\Users\seman\Desktop\clg\2nd_sem\research_practicum\code\src\cnn+tsm\checkpoint for stage D for training later\stage_d_ckpt.pth" )  # ← new, holds full training state

# ────────── PATHS ──────────
ROOT        = Path(r"C:\Users\seman\Desktop\clg\2nd_sem\research_practicum\code\data\images")
TRAIN_DIR   = ROOT / "train"
VAL_DIR     = ROOT / "val"
CKPT        = r"C:\Users\seman\Desktop\clg\2nd_sem\research_practicum\code\src\cnn+tsm\stage_c_adv_cnn_attn.pth"    # <- Stage C weights
OUT_W       = r"C:\Users\seman\Desktop\clg\2nd_sem\research_practicum\code\src\cnn+tsm\stage_d_tsm_attn.pth"
OUT_H       = r"C:\Users\seman\Desktop\clg\2nd_sem\research_practicum\code\src\cnn+tsm\stage_d_tsm_hist.npy"

# ────────── HYPER-PARAMS ──────────
NUM_FRAMES    = 64
IMG_SIZE      = 160
BATCH_SZ      = 4
MAX_EPOCHS    = 50
LR_BB         = 1e-5
LR_FULL       = 3e-4
FREEZE_EPOCHS = 3
MIXUP_ALPHA   = 0.1
PATIENCE      = 7
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS   = 6

# ────────── AUGMENTATIONS ──────────
_IM_MEAN = (0.485, 0.456, 0.406)
_IM_STD  = (0.229, 0.224, 0.225)
_tr_train = Compose([
    RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    RandomHorizontalFlip(),
    ColorJitter(0.2, 0.2, 0.2, 0.1),
    GaussianBlur(kernel_size=3),
    ToTensor(), Normalize(_IM_MEAN, _IM_STD),
    RandomErasing(p=0.1)
])
_tr_eval = Compose([
    Resize((IMG_SIZE, IMG_SIZE)),
    ToTensor(), Normalize(_IM_MEAN, _IM_STD)
])

# ────────── MediaPipe cropping (hand-priority) ──────────
_mp = mp.solutions.holistic.Holistic(static_image_mode=True, model_complexity=1, refine_face_landmarks=True)
@functools.lru_cache(maxsize=4096)
def _detect_bbox(p: Path):
    img = Image.open(p).convert("RGB"); w, h = img.size
    res = _mp.process(np.array(img)[:, :, ::-1])
    hands = [(lm.x*w, lm.y*h) for s in [res.left_hand_landmarks, res.right_hand_landmarks] if s for lm in s.landmark]
    if hands:
        xs, ys = zip(*hands)
    elif res.face_landmarks:
        xs, ys = zip(*[(lm.x*w, lm.y*h) for lm in res.face_landmarks.landmark])
    else:
        return 0, 0, w, h
    pad = 60
    x1, y1 = max(min(xs)-pad, 0), max(min(ys)-pad, 0)
    x2, y2 = min(max(xs)+pad, w), min(max(ys)+pad, h)
    side   = max(x2-x1, y2-y1); cx, cy = (x1+x2)//2, (y1+y2)//2; half = side//2
    return int(max(cx-half,0)), int(max(cy-half,0)), int(min(cx+half,w)), int(min(cy+half,h))
def _crop(p: Path) -> Image.Image:
    l,t,r,b = _detect_bbox(p); return Image.open(p).convert("RGB").crop((l,t,r,b))

# ────────── DATASET ──────────
class ASLSeqDataset(Dataset):
    def __init__(self, root: Path, split="train", T=NUM_FRAMES):
        self.root  = root; self.split = split; self.T = T
        self.classes = sorted(d.name for d in root.iterdir() if d.is_dir())
        self.cls2idx = {c:i for i,c in enumerate(self.classes)}
        self.samples = [(sorted(vid.glob("*.jpg")), self.cls2idx[c])
                        for c in self.classes for vid in (root/c).iterdir()
                        if len(list(vid.glob('*.jpg'))) >= 2]
        self.tx = _tr_train if split=="train" else _tr_eval
    def __len__(self): return len(self.samples)
    def _select(self, L:int):
        idxs = np.linspace(0, L-1, self.T, dtype=int)
        if self.split=="train":
            idxs = np.clip(idxs + np.random.randint(-1,2,self.T), 0, L-1)
        return idxs
    def __getitem__(self, i):
        frames, label = self.samples[i]
        idxs = self._select(len(frames))
        clip = torch.stack([ self.tx(_crop(frames[j])) for j in idxs ], dim=1)
        return clip, label

# ────────── MODEL (TSM + Attn) ──────────
class TSM_Attn(nn.Module):
    def __init__(self, n_cls, emb_dim=256):
        super().__init__()
        m = mobilenet_v2(weights=None); self.backbone = m.features
        self.shift_div = 8
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(1280, emb_dim, bias=False)
        self.bn   = nn.BatchNorm1d(emb_dim)
        self.attn = nn.Linear(emb_dim, 1)
        self.fc   = nn.Linear(emb_dim, n_cls)
    def tsm(self, x):
        B_T, C, H, W = x.shape; B = B_T//NUM_FRAMES
        x = x.view(B, NUM_FRAMES, C, H, W)
        fold = C // self.shift_div; out = torch.zeros_like(x)
        out[:, :-1, :fold] = x[:, 1:, :fold]
        out[:, 1:, fold:2*fold] = x[:, :-1, fold:2*fold]
        out[:, :, 2*fold:] = x[:, :, 2*fold:]
        return out.view(B_T, C, H, W)
    def forward(self, x):
        B,C,T,H,W = x.shape
        x = self.tsm(x.permute(0,2,1,3,4).reshape(B*T, C, H, W))
        f = self.pool(self.backbone(x)).flatten(1)
        f = F.relu(self.bn(self.proj(f))).view(B,T,-1)
        w = F.softmax(self.attn(f).squeeze(-1), dim=1).unsqueeze(-1)
        emb = (w*f).sum(1)
        return self.fc(emb)

# MixUp, ce_mix, evaluate same as Stage C (reuse)

def mixup(x,y,alpha=MIXUP_ALPHA):
    if alpha<=0: return x,y,None
    lam = np.random.beta(alpha,alpha); idx = torch.randperm(x.size(0))
    return lam*x+(1-lam)*x[idx], (y,y[idx],lam), idx
def ce_mix(ce,logits,target):
    if isinstance(target,tuple):
        y1,y2,lam = target; return lam*ce(logits,y1)+(1-lam)*ce(logits,y2)
    return ce(logits,target)
@torch.no_grad()
def evaluate(model,loader):
    model.eval(); ce=nn.CrossEntropyLoss(); tot=correct=loss=0
    for clip,lab in loader:
        clip,lab = clip.to(DEVICE), lab.to(DEVICE)
        out = model(clip)
        loss += ce(out,lab).item()*lab.size(0)
        correct += (out.argmax(1)==lab).sum().item()
        tot += lab.size(0)
    return loss/tot, 100*correct/tot

# ────────── MAIN ──────────
def main():
    torch.backends.cudnn.benchmark = True
    train_ds = ASLSeqDataset(TRAIN_DIR, "train")
    val_ds = ASLSeqDataset(VAL_DIR, "val")

    labels = np.array([lab for _, lab in train_ds.samples])
    weights = 1.0 / (np.bincount(labels, minlength=len(train_ds.classes)) + 1e-6)
    sampler = WeightedRandomSampler(
        torch.as_tensor(weights[labels], dtype=torch.double),
        len(labels),
        replacement=True
    )

    train_ld = DataLoader(
        train_ds, BATCH_SZ, sampler=sampler,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)

    val_ld = DataLoader(
        val_ds, BATCH_SZ, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True)

    # ────────── MODEL & OPTIMIZER ──────────
    # ────────── MODEL, OPTIMISER, SCHEDULER ──────────
    model = TSM_Attn(len(train_ds.classes)).to(DEVICE)

    # Backbone warm-start (only if we’re NOT resuming a full ckpt)
    if not CKPT_FULL.exists():
        sc = torch.load(CKPT, map_location="cpu")
        bb_state = {k.replace("backbone.", ""): v
                    for k, v in sc.items() if k.startswith("backbone.")}
        model.backbone.load_state_dict(bb_state, strict=False)

    # Freeze backbone first
    for p in model.backbone.parameters():
        p.requires_grad = False

    opt = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": LR_BB},
        {"params": list(model.proj.parameters()) +
                   list(model.bn.parameters()) +
                   list(model.attn.parameters()) +
                   list(model.fc.parameters()), "lr": LR_FULL}
    ], weight_decay=1e-4)

    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=MAX_EPOCHS, eta_min=1e-6)

    ce = nn.CrossEntropyLoss(label_smoothing=0.1)

    # save the class list once so inference stays aligned
    with open("stage_d_classes.txt", "w") as f:
        f.write("\n".join(train_ds.classes))


    # ────────── (Optional) RESUME TRAINING ──────────
    start_epoch = 1
    best_loss   = math.inf
    history = {k: [] for k in ["epoch","train_loss","val_loss",
                               "train_acc","val_acc"]}

    if CKPT_FULL.exists():
        print(f"➜ Resuming from {CKPT_FULL}")
        ckpt_all: Dict[str, Any] = torch.load(CKPT_FULL, map_location=DEVICE)

        model.load_state_dict(ckpt_all["model"])
        opt.load_state_dict(ckpt_all["opt"])
        sched.load_state_dict(ckpt_all["sched"])

        best_loss   = ckpt_all["best_loss"]
        history     = ckpt_all["history"]
        start_epoch = ckpt_all["epoch"] + 1

        # Make sure LR group-0 is correct if backbone is already unfrozen
        if start_epoch > FREEZE_EPOCHS + 1:
            for p in model.backbone.parameters():
                p.requires_grad = True

    # ────────── TRAIN LOOP (early-stop on val-loss) ──────────
    epochs_no_improve = 0

    for ep in range(start_epoch, MAX_EPOCHS + 1):
        start = time.time()
        model.train()
        tloss = tcorrect = totsz = 0

        # unfreeze backbone at epoch FREEZE_EPOCHS+1
        if ep == FREEZE_EPOCHS + 1:
            for p in model.backbone.parameters():
                p.requires_grad = True
            opt.param_groups[0]["lr"] = LR_FULL * 0.3  # safe LR bump
            print(f"Backbone unfrozen at epoch {ep}")

        for clip, lab in tqdm(train_ld, desc=f"Ep{ep:02d}"):
            clip, lab = clip.to(DEVICE), lab.to(DEVICE)
            clip, tgt, _ = mixup(clip, lab)
            out = model(clip)
            loss = ce_mix(ce, out, tgt if tgt else lab)

            opt.zero_grad();
            loss.backward();
            opt.step()

            bs = lab.size(0)
            tloss += loss.item() * bs
            tcorrect += (out.argmax(1) == lab).sum().item()
            totsz += bs

        sched.step()
        tr_loss = tloss / totsz
        tr_acc = 100 * tcorrect / totsz

        v_loss, v_acc = evaluate(model, val_ld)

        # ───────────────────────── LOG THIS EPOCH ─────────────────────────
        history["epoch"].append(ep)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(v_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(v_acc)
        # ------------------------------------------------------------------

        # ---- ALWAYS SAVE *LATEST* TRAINING STATE (resume checkpoint) ----
        torch.save({
            "epoch": ep,  # last finished epoch
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "sched": sched.state_dict(),
            "best_loss": best_loss,  # keep current best
            "history": history,
        }, CKPT_FULL)

        # ---- ALSO SAVE “BEST” WEIGHTS WHEN VAL-LOSS IMPROVES ----
        if v_loss < best_loss:
            best_loss = v_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), OUT_W)  # best-weights file
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print("Early stopping triggered.")
                break

        print(f"Ep{ep:02d} | "
              f"Train {tr_acc:5.1f}% | Val {v_acc:5.1f}% | "
              f"Train-Loss {tr_loss:.3f} | Val-Loss {v_loss:.3f} | "
              f"{(time.time() - start) / 60:.1f} min")

    # save history
    np.save(OUT_H, history)
    print(f"History with {len(history['epoch'])} epochs saved to {OUT_H}")
    print(f"Best val-loss {best_loss:.4f}, weights saved to {OUT_W}")



if __name__ == "__main__":
    main()
