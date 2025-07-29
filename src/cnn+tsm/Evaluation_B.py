# eval_stage_b_topk.py  – Windows‑safe, no spawn errors
from __future__ import annotations
import os, torch, torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GLOG_minloglevel"]     = "3"

# ─── EDIT THESE PATHS ─────────────────────────────────────
ROOT      = Path(r"C:\Users\seman\Desktop\clg\2nd_sem\research_practicum\code")
DATA_DIR  = ROOT / "data" / "images" / "test"          # "val" or "test"
WEIGHTS   = ROOT / "src" / "cnn+tsm" / "stage_b_cnn_attn_final.pth"
CLASS_TXT = ROOT / "src" / "cnn+tsm" / "stage_b_classes.txt"
# ──────────────────────────────────────────────────────────

from stage_B import TemporalAttn, ASLSeqDataset, NUM_FRAMES, DEVICE

def evaluate_topk(model, loader, criterion: nn.Module | None = None):
    model.eval(); t1=t5=tot=loss=0.
    bar = tqdm(total=len(loader.dataset), unit="vid", desc=f"Testing {split}", ncols=80)
    with torch.no_grad():
        for clip, lab in loader:
            clip, lab = clip.to(DEVICE), lab.to(DEVICE)
            out = model(clip)
            if criterion: loss += criterion(out, lab).item() * lab.size(0)
            tot += lab.size(0)
            _, pred = out.topk(5, 1, True, True)
            corr    = pred.t().eq(lab.view(1, -1).expand_as(pred.t()))
            t1 += corr[:1].flatten().sum().item()
            t5 += corr[:5].flatten().sum().item()
            bar.update(lab.size(0))
    bar.close()
    return ((loss/tot) if criterion else None, 100*t1/tot, 100*t5/tot)

def main() -> None:
    global split                          # allow evaluate_topk to show it
    split = DATA_DIR.parts[-1]

    # -------- classes / output size -------
    if CLASS_TXT.exists():
        cls_names = [l.strip() for l in CLASS_TXT.open() if l.strip()]
        n_cls = len(cls_names)
    else:
        n_cls = torch.load(WEIGHTS, map_location="cpu")["fc.weight"].shape[0]

    # -------- dataset & loader ------------
    ds = ASLSeqDataset(DATA_DIR, split="val", T=NUM_FRAMES)
    if len(ds) == 0:
        raise RuntimeError(f"No clips in {DATA_DIR}. Run extract_frames.py first!")
    print(f"🗂️  {split.capitalize()} set : {len(ds)} videos   "
          f"{len(ds.cls_names)}/{n_cls} classes")

    ld = DataLoader(ds, batch_size=4, shuffle=False,
                    num_workers=4, pin_memory=True)

    # -------- model -----------------------
    model = TemporalAttn(n_cls).to(DEVICE)
    model.load_state_dict(torch.load(WEIGHTS, map_location=DEVICE), strict=True)

    # -------- evaluate --------------------
    loss, top1, top5 = evaluate_topk(model, ld, nn.CrossEntropyLoss())
    print(f"\n🏅  Top‑1 accuracy : {top1:6.2f}%")
    print(f"🥈  Top‑5 accuracy : {top5:6.2f}%")
    print(f"📉  {split.capitalize()} loss : {loss:.4f}")

# Windows entry‑point -------------------------------------
if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    main()
