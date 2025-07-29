import torch
from pathlib import Path

CKPT = "stage_c_adv_cnn_attn.pth"
state = torch.load(CKPT, map_location="cpu")

# ❶ list all keys
print(state.keys())

# ❷ look for something like 'cls_names', 'class_names', 'idx2cls', etc.
for k in state:
    if "cls" in k.lower():
        print(k, "→", state[k][:5], "…")   # peek at first 5 entries
