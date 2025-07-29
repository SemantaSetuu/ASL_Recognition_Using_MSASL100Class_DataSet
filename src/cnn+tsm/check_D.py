"""
check_sign_D.py – offline video‑file inference for Stage‑D TSM‑Attention model
------------------------------------------------------------------------------
• Loads the Stage‑D weights  ............ stage_d_tsm_attn.pth
• Loads class names from  ............... stage_d_classes.txt
• MediaPipe‑Holistic crop (hands‑first)   (same as live_test_D.py)
• Uniformly samples 64 frames per clip
• Prints TOP‑1 sign and confidence
------------------------------------------------------------------------------
Run:

    python check_sign_D.py
    # then paste the full path to an .mp4 / .avi etc.

If you want a quick run:

    python check_sign_D.py C:/path/to/video.mp4
"""

from __future__ import annotations
import sys, cv2, torch, numpy as np, mediapipe as mp
from pathlib import Path
from torchvision.transforms import (
    Compose, Resize, ToTensor, Normalize
)
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2

# ────────── CONFIG ──────────
WEIGHTS = Path(r"C:\Users\seman\Desktop\clg\2nd_sem\research_practicum\code\src\cnn+tsm\stage_d_tsm_attn.pth")
CLASSES_TXT = Path("stage_d_classes.txt")   # produced during Stage‑D training

NUM_FRAMES = 64
IMG_SIZE   = 160
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

assert WEIGHTS.exists(),    f"weights not found: {WEIGHTS}"
assert CLASSES_TXT.exists(), f"class list not found: {CLASSES_TXT}"

# ────────── model (same as training) ──────────
class TSM_Attn(nn.Module):
    def __init__(self, n_cls, emb_dim=256):
        super().__init__()
        m = mobilenet_v2(weights=None)
        self.backbone = m.features
        self.shift_div = 8
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(1280, emb_dim, bias=False)
        self.bn   = nn.BatchNorm1d(emb_dim)
        self.attn = nn.Linear(emb_dim, 1)
        self.fc   = nn.Linear(emb_dim, n_cls)

    def tsm(self, x):
        B_T,C,H,W = x.shape
        B = B_T // NUM_FRAMES
        x = x.view(B, NUM_FRAMES, C, H, W)
        fold = C // self.shift_div
        out  = torch.zeros_like(x)
        out[:, :-1, :fold]      = x[:, 1:, :fold]
        out[:, 1:, fold:2*fold] = x[:, :-1, fold:2*fold]
        out[:, :, 2*fold:]      = x[:, :, 2*fold:]
        return out.view(B_T, C, H, W)

    def forward(self, x):
        B,C,T,H,W = x.shape
        x = x.permute(0,2,1,3,4).reshape(B*T, C, H, W)
        x = self.tsm(x)
        f = self.pool(self.backbone(x)).flatten(1)
        f = F.relu(self.bn(self.proj(f))).view(B,T,-1)
        w = F.softmax(self.attn(f).squeeze(-1), dim=1).unsqueeze(-1)
        emb = (w*f).sum(1)
        return self.fc(emb)

# ────────── load classes & model ──────────
cls_names = [ln.strip() for ln in open(CLASSES_TXT) if ln.strip()]
model = TSM_Attn(len(cls_names)).to(DEVICE)
model.load_state_dict(torch.load(WEIGHTS, map_location=DEVICE))
model.eval()
print(f"✅  Loaded Stage‑D weights: {WEIGHTS}")

# ────────── transform ──────────
transform = Compose([
    Resize((IMG_SIZE, IMG_SIZE)),
    ToTensor(),
    Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
])

# ────────── MediaPipe crop helpers ──────────
mp_hol = mp.solutions.holistic.Holistic(
    static_image_mode=True, model_complexity=1, refine_face_landmarks=True
)

def detect_bbox_np(rgb: np.ndarray):
    h,w,_ = rgb.shape
    res = mp_hol.process(rgb)
    pts = []
    for hand in (res.left_hand_landmarks, res.right_hand_landmarks):
        if hand:
            pts += [(int(lm.x*w), int(lm.y*h)) for lm in hand.landmark]
    if not pts and res.face_landmarks:
        pts = [(int(lm.x*w), int(lm.y*h)) for lm in res.face_landmarks.landmark]
    if not pts: return 0,0,w,h
    xs,ys = zip(*pts); pad=60
    x1,y1 = max(min(xs)-pad,0), max(min(ys)-pad,0)
    x2,y2 = min(max(xs)+pad,w), min(max(ys)+pad,h)
    side  = max(x2-x1, y2-y1); cx,cy=(x1+x2)//2,(y1+y2)//2; half=side//2
    return max(cx-half,0), max(cy-half,0), min(cx+half,w), min(cy+half,h)

def crop_and_transform(frame_bgr: np.ndarray):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    l,t,r,b = detect_bbox_np(rgb)
    img = Image.fromarray(rgb[t:b, l:r])
    return transform(img)

# ────────── helper: load & sample frames ──────────
def make_clip(video_path: str) -> torch.Tensor:
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ok, frm = cap.read()
        if not ok: break
        frames.append(frm)
    cap.release()
    if len(frames) < 2:
        raise ValueError("Video too short / unreadable")
    idxs = np.linspace(0, len(frames)-1, NUM_FRAMES).astype(int)
    tensors = [crop_and_transform(frames[i]) for i in idxs]
    clip = torch.stack(tensors, dim=1)        # (3,T,H,W)
    return clip.unsqueeze(0).to(DEVICE)       # (1,3,T,H,W)

# ────────── predict function ──────────
def predict(video_path: str):
    print("📹 Processing video...")
    clip = make_clip(video_path)
    with torch.no_grad():
        logits = model(clip)
        probs  = torch.softmax(logits, dim=1)[0]
        pred   = probs.argmax().item()
        conf   = probs[pred].item()*100
        label  = cls_names[pred].upper()
    print(f"🧠 Predicted Sign: {label} ({conf:.1f}%)")

# ────────── CLI ──────────
if __name__ == "__main__":
    if len(sys.argv) >= 2:
        vid = sys.argv[1]
    else:
        vid = input("📁 Enter path to your video file: ").strip()
    predict(vid)
