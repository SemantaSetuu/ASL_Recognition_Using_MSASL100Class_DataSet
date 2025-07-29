"""
predict_stage_c.py
──────────────────
Loads Stage-C checkpoint + class list file and predicts a video.
"""

import os, torch, cv2, numpy as np, mediapipe as mp
from pathlib import Path
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import torch.nn as nn
import torch.nn.functional as F

# ─── paths (EDIT) ───────────────────────────────────────────
CKPT_PATH = Path(r"C:\Users\seman\Desktop\clg\2nd_sem\research_practicum\code\src\cnn+tsm\stage_c_adv_cnn_attn.pth")
CLASS_TXT = Path(r"C:\Users\seman\Desktop\clg\2nd_sem\research_practicum\code\data\images\stage_c_classes.txt")
# ────────────────────────────────────────────────────────────

NUM_FRAMES = 32
IMG_SIZE   = 160
MEAN, STD  = (0.485,0.456,0.406), (0.229,0.224,0.225)
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

# ─── load class list ───
with open(CLASS_TXT) as f:
    idx2cls = {i: line.strip() for i, line in enumerate(f)}
print(f"✅ Loaded {len(idx2cls)} classes from {CLASS_TXT}")

# ─── model definition (same as training) ───
class TemporalAttn(nn.Module):
    def __init__(self, n_cls, emb_dim=256):
        super().__init__()
        from torchvision.models import mobilenet_v2
        m = mobilenet_v2(weights=None)
        self.backbone = m.features
        self.pool  = nn.AdaptiveAvgPool2d(1)
        self.proj  = nn.Linear(1280, emb_dim, bias=False)
        self.bn    = nn.BatchNorm1d(emb_dim)
        self.drop  = nn.Dropout(0.5)
        self.attn  = nn.Linear(emb_dim, 1)
        self.fc    = nn.Linear(emb_dim, n_cls)
    def forward(self,x):
        B,C,T,H,W = x.shape
        x = x.permute(0,2,1,3,4).reshape(B*T,C,H,W)
        f = self.pool(self.backbone(x)).flatten(1)
        f = self.drop(F.relu(self.bn(self.proj(f)))).view(B,T,-1)
        w = F.softmax(self.attn(f).squeeze(-1),dim=1).unsqueeze(-1)
        emb = (w*f).sum(1)
        return self.fc(emb)

# ─── load weights ───
model = TemporalAttn(len(idx2cls)).to(DEVICE)
model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE), strict=True)
model.eval(); print("✅ Stage-C weights loaded.")

# ─── transforms ───
tx = Compose([Resize((IMG_SIZE,IMG_SIZE)),
              ToTensor(), Normalize(MEAN,STD)])

# ─── MediaPipe crop (hand priority) ───
_mp = mp.solutions.holistic.Holistic(static_image_mode=True, model_complexity=1)

def detect_bbox(img: Image.Image):
    w,h = img.size
    res = _mp.process(np.array(img)[:,:,::-1])
    hands=[(lm.x*w,lm.y*h)
           for s in (res.left_hand_landmarks,res.right_hand_landmarks)
           if s for lm in s.landmark]
    if hands:
        xs,ys = zip(*hands)
    elif res.face_landmarks:
        xs,ys = zip(*[(lm.x*w,lm.y*h) for lm in res.face_landmarks.landmark])
    else:
        return 0,0,w,h
    pad=60; x1,y1=max(min(xs)-pad,0),max(min(ys)-pad,0)
    x2,y2=min(max(xs)+pad,w),min(max(ys)+pad,h)
    side=max(x2-x1,y2-y1); cx,cy=(x1+x2)//2,(y1+y2)//2; half=side//2
    return int(max(cx-half,0)),int(max(cy-half,0)),int(min(cx+half,w)),int(min(cy+half,h))

def crop(img,bbox):
    l,t,r,b = bbox; return img.crop((l,t,r,b))

# ─── video → tensor ───
def video_to_tensor(path:str):
    cap=cv2.VideoCapture(path); frames=[]
    ok=True
    while ok:
        ok,fr=cap.read()
        if ok: frames.append(Image.fromarray(cv2.cvtColor(fr,cv2.COLOR_BGR2RGB)))
    cap.release()
    if len(frames)<2: raise ValueError("Video too short.")
    bbox = detect_bbox(frames[len(frames)//2])
    idxs = np.linspace(0,len(frames)-1,NUM_FRAMES).astype(int)
    clip = torch.stack([tx(crop(frames[i],bbox)) for i in idxs],dim=1)
    return clip.unsqueeze(0).to(DEVICE)

# ─── predict ───
@torch.no_grad()
def predict(video:str):
    clip = video_to_tensor(video)
    probs = model(clip).softmax(1)[0]
    idx   = int(probs.argmax())
    print(f"🧠 {idx2cls[idx].upper()}   ({probs[idx]*100:.1f} %)")

# ─── main ───
if __name__=="__main__":
    vid = input("Video path ➜ ").strip('"')
    if not Path(vid).is_file():
        print("❌ File not found.")
    else:
        predict(vid)
