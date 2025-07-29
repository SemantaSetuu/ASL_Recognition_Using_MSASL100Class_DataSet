# eval_stage_d_topk.py  – Stage‑D evaluator with live progress logging
# ── silence TensorFlow Lite / MediaPipe warnings ───────────
from __future__ import annotations
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

# ───────── CONFIG ─────────
ROOT      = Path(r"C:\Users\seman\Desktop\clg\2nd_sem\research_practicum\code")

# Choose which split you want to test:
#   images/val  → same set used during training
#   clips/test  → your final video test set
DATA_DIR  = ROOT / "data" / "images" / "test"      # change to ".../clips/test"
WEIGHTS   = ROOT / "src" / "cnn+tsm" / "stage_d_tsm_attn.pth"

# ───────── IMPORT TRAIN‑TIME CODE ─────────
from stage_D_new import TSM_Attn, ASLSeqDataset, NUM_FRAMES, DEVICE

# ───────── TOP‑k EVALUATION ─────────
def evaluate_topk(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: nn.Module | None = None,
    desc: str = "Eval"
) -> tuple[float | None, float, float]:
    """Return (avg_loss or None, top1%, top5%).  Prints a live tqdm bar."""
    model.eval()
    top1 = top5 = total = 0
    running_loss = 0.0

    bar = tqdm(total=len(loader.dataset), unit="vid", desc=desc, ncols=80)

    with torch.no_grad():
        for clip, lab in loader:
            clip, lab = clip.to(DEVICE), lab.to(DEVICE)
            out = model(clip)                         # (B, n_cls)

            if criterion is not None:
                running_loss += criterion(out, lab).item() * lab.size(0)

            total += lab.size(0)

            # -------- top‑k correctness --------
            _, pred = out.topk(5, dim=1, largest=True, sorted=True)
            correct = pred.t().eq(lab.view(1, -1).expand_as(pred.t()))
            top1 += correct[:1].flatten().sum().item()
            top5 += correct[:5].flatten().sum().item()

            bar.update(lab.size(0))
            bar.set_postfix_str(f"{bar.n}/{bar.total}  "
                                f"({bar.n/bar.total:5.1%})")

    bar.close()
    avg_loss = (running_loss / total) if criterion is not None else None
    return avg_loss, 100 * top1 / total, 100 * top5 / total

# ───────── MAIN (Windows‑safe) ─────────
def main() -> None:
    split_name = DATA_DIR.parts[-1]  # "val" or "test" for the printout
    ds = ASLSeqDataset(DATA_DIR, split="val", T=NUM_FRAMES)
    ld = DataLoader(
        ds,
        batch_size=4,
        shuffle=False,
        num_workers=4,        # >0 is fine behind __main__ guard
        pin_memory=True,
    )

    print(f"🗂️  {split_name.capitalize()} set : {len(ds)} videos   "
          f"{len(ds.classes)} classes")

    model = TSM_Attn(len(ds.classes)).to(DEVICE)
    model.load_state_dict(torch.load(WEIGHTS, map_location=DEVICE),
                          strict=False)

    ce = nn.CrossEntropyLoss()        # set to None if you don’t need the loss
    loss, top1, top5 = evaluate_topk(model, ld, ce,
                                     desc=f"Testing {split_name}")

    print(f"\n🏅  Top‑1 accuracy : {top1:6.2f}%")
    print(f"🥈  Top‑5 accuracy : {top5:6.2f}%")
    if loss is not None:
        print(f"📉  {split_name.capitalize()} loss : {loss:.4f}")

# ───────── ENTRY POINT ─────────
if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)  # Windows
    main()
