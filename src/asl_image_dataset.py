"""
ASLImageDataset — now with MediaPipe signer crop
-----------------------------------------------
• If MediaPipe Holistic is available, detects face and both hands once per video
  folder and caches a bounding box that tightly contains them, then enlarges
  to a square and crops every frame before resizing to 224×224.
• If MediaPipe is *not* installed or detection fails, falls back to the full
  frame or centred square crop.
"""

from __future__ import annotations
import random, json, functools
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

import torchvision.transforms.functional as TF
from torchvision.transforms import ColorJitter, RandomErasing

try:
    import mediapipe as mp
    _mp_holistic = mp.solutions.holistic.Holistic(
        static_image_mode=True,
        model_complexity=0,
        enable_segmentation=False,
        refine_face_landmarks=False,
    )
except ModuleNotFoundError:  # graceful fallback
    _mp_holistic = None

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)

def _norm(t: torch.Tensor) -> torch.Tensor:
    return TF.normalize(t, _IMAGENET_MEAN, _IMAGENET_STD)

# --------------------------------------------------------------
# MediaPipe‑based signer bbox with LRU cache (video‑folder scoped)
# --------------------------------------------------------------
_bbox_cache: Dict[Path, Tuple[int,int,int,int]] = {}

@functools.lru_cache(maxsize=4096)
def _detect_bbox(img_path: Path) -> Tuple[int,int,int,int]:
    if _mp_holistic is None:
        return (0, 0, -1, -1)  # sentinel meaning "use fallback"
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    res = _mp_holistic.process(np.array(img)[:,:,::-1])
    xs, ys = [], []
    for lm_set in (res.face_landmarks,
                   res.left_hand_landmarks,
                   res.right_hand_landmarks):
        if lm_set:
            for lm in lm_set.landmark:
                xs.append(lm.x * w)
                ys.append(lm.y * h)
    if not xs:  # detection failure
        return (0, 0, -1, -1)
    x1, y1 = max(min(xs)-20, 0), max(min(ys)-20, 0)
    x2, y2 = min(max(xs)+20, w), min(max(ys)+20, h)
    # square‑pad
    side = max(x2-x1, y2-y1)
    cx, cy = (x1+x2)//2, (y1+y2)//2
    half = side//2
    left  = int(max(cx-half, 0))
    top   = int(max(cy-half, 0))
    right = int(min(cx+half, w))
    bottom= int(min(cy+half, h))
    return left, top, right, bottom

# --------------------------------------------------------------
# Dataset
# --------------------------------------------------------------
class ASLImageDataset(Dataset):
    def __init__(self, root_dir: str | Path, split="train",
                 num_frames: int = 16, size: int = 224):
        self.root = Path(root_dir)
        self.split = split.lower()
        self.T = num_frames
        self.size = size

        self.cls_names = sorted(d.name for d in self.root.iterdir() if d.is_dir())
        self.cls2idx   = {c:i for i,c in enumerate(self.cls_names)}
        self.samples: List[Tuple[Path,int]] = []
        for cls in self.cls_names:
            for vid in (self.root/cls).iterdir():
                if vid.is_dir():
                    self.samples.append((vid, self.cls2idx[cls]))

        self.cjitter = ColorJitter(0.4,0.4,0.4,0.1)
        self.erase   = RandomErasing(p=0.5, scale=(0.02,0.2))

    def _crop_resize_norm(self, img_path: Path) -> torch.Tensor:
        img = Image.open(img_path).convert("RGB")
        if img_path.parent not in _bbox_cache:
            bbox = _detect_bbox(img_path)  # detect on first frame per video
            _bbox_cache[img_path.parent] = bbox
        else:
            bbox = _bbox_cache[img_path.parent]
        if bbox[2] != -1:  # valid bbox
            img = img.crop(bbox)
        else:  # fallback: centred square
            w, h = img.size
            side = min(w, h)
            left = (w-side)//2
            top  = (h-side)//2
            img = img.crop((left, top, left+side, top+side))
        img = img.resize((self.size, self.size))
        if self.split == "train":
            img = self.cjitter(img)
        t = _norm(TF.to_tensor(img))
        if self.split == "train":
            t = self.erase(t)
        return t

    def _pad_loop(self, frames: List[Path]) -> List[Path]:
        if len(frames) >= self.T:
            return frames
        reps = (self.T + len(frames)-1)//len(frames)
        return (frames*reps)[:self.T]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        vid_dir, label = self.samples[idx]
        frames = sorted(vid_dir.glob("*.jpg"))
        frames = self._pad_loop(frames)

        if self.split == "train":
            start = random.randint(0, len(frames)-self.T)
            sel = frames[start:start+self.T]
            clip = torch.stack([self._crop_resize_norm(p) for p in sel], dim=1)
            return clip, label

        # val/test → sliding windows
        windows = []
        for s in range(0, len(frames)-self.T+1):
            sel = frames[s:s+self.T]
            clip = torch.stack([self._crop_resize_norm(p) for p in sel], dim=1)
            windows.append(clip)
        return torch.stack(windows), label
