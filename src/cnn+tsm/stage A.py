
import numpy as np, torch, torch.nn as nn
from pathlib import Path
from torchvision.models import mobilenet_v2
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, RandomHorizontalFlip, ColorJitter
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import mediapipe as mp
import functools

# Paths
TRAIN_DIR = r"C:\Users\seman\Desktop\clg\2nd_sem\research_practicum\code\data\images\train"
VAL_DIR   = r"C:\Users\seman\Desktop\clg\2nd_sem\research_practicum\code\data\images\val"

SAVE_W = "mnet_frame_pretrain_correct.pth"
SAVE_H = "hist_mnet_frame_correct.npy"

# Hyper-params
BATCH_SZ  = 64
EPOCHS    = 10
LR        = 1e-4
VAL_EVERY = 2
IMG_SIZE  = 160
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"

# Improved Transforms
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)
_tx_train = Compose([
    Resize((IMG_SIZE, IMG_SIZE)), RandomHorizontalFlip(), ColorJitter(0.2,0.2,0.2,0.1),
    ToTensor(), Normalize(_IMAGENET_MEAN, _IMAGENET_STD)
])
_tx_eval = Compose([Resize((IMG_SIZE, IMG_SIZE)), ToTensor(), Normalize(_IMAGENET_MEAN, _IMAGENET_STD)])

# Corrected MediaPipe-based Cropping
_mp_holistic = mp.solutions.holistic.Holistic(static_image_mode=True, model_complexity=1, refine_face_landmarks=True)
_bbox_cache: dict[Path, tuple[int,int,int,int]] = {}

@functools.lru_cache(maxsize=4096)
def _detect_bbox(img_path: Path):
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    res = _mp_holistic.process(np.array(img)[:,:,::-1])

    xs, ys, detected = [], [], False
    for lm_set in [res.face_landmarks, res.left_hand_landmarks, res.right_hand_landmarks]:
        if lm_set:
            detected = True
            for lm in lm_set.landmark:
                xs.append(lm.x * w)
                ys.append(lm.y * h)

    if not detected:
        return (0, 0, w, h)

    x1, y1 = max(min(xs)-40, 0), max(min(ys)-40, 0)
    x2, y2 = min(max(xs)+40, w), min(max(ys)+40, h)

    side = max(x2 - x1, y2 - y1)
    cx, cy = (x1+x2)//2, (y1+y2)//2
    half = side//2
    l, t = int(max(cx-half, 0)), int(max(cy-half, 0))
    r, b = int(min(cx+half, w)), int(min(cy+half, h))
    return l, t, r, b

def _crop(img_path: Path) -> Image.Image:
    if img_path.parent not in _bbox_cache:
        _bbox_cache[img_path.parent] = _detect_bbox(img_path)
    l, t, r, b = _bbox_cache[img_path.parent]
    return Image.open(img_path).convert("RGB").crop((l, t, r, b))

# Dataset Class
class FrameDataset(Dataset):
    def __init__(self, root, split="train"):
        self.root = Path(root)
        self.split = split
        self.cls_names = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
        self.cls2idx = {c: i for i,c in enumerate(self.cls_names)}
        self.samples = [(f, self.cls2idx[cls]) for cls in self.cls_names for vid in (self.root/cls).iterdir() for f in vid.glob("*.jpg")]
        self.tx = _tx_train if split=="train" else _tx_eval

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        return self.tx(_crop(img_path)), label

# Model
class MobileNetFrame(nn.Module):
    def __init__(self, n_cls):
        super().__init__()
        m = mobilenet_v2(weights="IMAGENET1K_V1")
        m.classifier[1] = nn.Linear(1280, n_cls)
        self.net = m
    def forward(self, x): return self.net(x)

# Eval Function
@torch.no_grad()
def eval_top1(model, loader):
    model.eval(); correct=total=0
    for x,y in loader:
        pred = model(x.to(DEVICE)).argmax(1)
        correct += (pred==y.to(DEVICE)).sum().item(); total+=y.size(0)
    return 100*correct/total

# Main
if __name__ == "__main__":
    train_ds, val_ds = FrameDataset(TRAIN_DIR), FrameDataset(VAL_DIR, split="val")
    train_ld = DataLoader(train_ds, BATCH_SZ, shuffle=True)
    val_ld = DataLoader(val_ds, BATCH_SZ)

    model = MobileNetFrame(len(train_ds.cls_names)).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss(label_smoothing=0.1)

    hist = {k: [] for k in ["epoch","loss","train_acc","val_acc"]}

    import time

    for ep in range(1, EPOCHS + 1):
        t0 = time.time()
        model.train(); tloss = correct = total = 0

        for x,y in tqdm(train_ld, desc=f"E{ep:02d}"):
            logits = model(x.to(DEVICE))
            loss = ce(logits,y.to(DEVICE))
            opt.zero_grad(); loss.backward(); opt.step()
            tloss += loss.item()*y.size(0); correct += (logits.argmax(1)==y.to(DEVICE)).sum().item(); total+=y.size(0)

        tr_acc, tr_loss = 100*correct/total, tloss/total
        val_acc = eval_top1(model,val_ld) if ep%VAL_EVERY==0 else float('nan')

        lr = opt.param_groups[0]["lr"]
        elapsed = time.time() - t0
        print(f"Epoch {ep:02d} | LR: {lr:.2e} | Train Acc: {tr_acc:.2f}% | Val Acc: {val_acc:.2f}% | Time: {elapsed:.1f}s")

        hist["epoch"].append(ep); hist["loss"].append(tr_loss); hist["train_acc"].append(tr_acc); hist["val_acc"].append(val_acc)

    torch.save(model.state_dict(), SAVE_W)
    np.save(SAVE_H, hist)
