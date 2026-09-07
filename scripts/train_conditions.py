"""
Train the ablation conditions OUTSIDE Jupyter.

    python scripts/train_conditions.py                    # all remaining
    python scripts/train_conditions.py --only C3 C1 C2    # pick conditions
    python scripts/train_conditions.py --promote-best     # recover an interrupted run

Why not the notebook: three separate failures so far traced to Jupyter --
a stale in-browser copy, host RAM exhausted by notebook + workers, and an
interrupted cell losing its return value while the best checkpoint sat safely
on disk. Training is a batch job; it belongs in a terminal. The notebook keeps
its real job: loading the finished checkpoints and producing the tables.

Resumable: a condition whose checkpoint already exists is skipped.
"""
import argparse, json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from fsvnet import Config, celebdf_frames as cf
from fsvnet.model import build_model
from fsvnet.train import train as train_loop
from fsvnet.dataset_frames import make_frame_loaders
from fsvnet.utils import log

NUM_FRAMES, IMAGE_SIZE, NUM_WORKERS = 16, 224, 2
BATCH_SIZE, GRAD_ACCUM, LR, SEED = 2, 8, 3e-4, 42

MAN = ROOT / "data" / "manifests" / "celebdf_v2_identity.csv"
CACHE = ROOT / "data" / "processed" / f"lmf_cache_{NUM_FRAMES:02d}f.npz"
OUT = ROOT / "models" / "checkpoints"
BEST_DIR = ROOT / "checkpoints"
CONDITIONS = {"B0": (False, False), "C3": (True, True),
              "C1": (True, False), "C2": (False, True)}


def promote_best(cid):
    """Turn train()'s <run>_best.pt into the condition checkpoint the notebook reads."""
    best = BEST_DIR / f"dfe_{cid}_best.pt"
    dst = OUT / f"fsvnet_dfe_{cid}.pt"
    if not best.exists():
        return False
    blob = torch.load(best, map_location="cpu")
    tsc, lmf = CONDITIONS[cid]
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": blob["model_state_dict"], "config": blob.get("config"),
                "condition": cid, "seed": SEED, "use_tsc": tsc, "use_lmf": lmf,
                "epoch": blob.get("epoch"), "metric": blob.get("metric"),
                "monitor": blob.get("monitor"), "promoted_from": best.name}, dst)
    print(f"  promoted {best.name} (epoch {blob.get('epoch')}, "
          f"{blob.get('monitor')}={blob.get('metric'):.4f}) -> {dst.name}")
    return True


def run(cid, budget_secs, epochs, patience=2, min_delta=0.001):
    tsc, lmf = CONDITIONS[cid]
    dst = OUT / f"fsvnet_dfe_{cid}.pt"
    if dst.exists():
        print(f"[{cid}] already trained -> {dst.name}; skipping"); return True
    if lmf and not CACHE.exists():
        print(f"[{cid}] REFUSING: {CACHE.name} missing, the LMF branch would be dead.\n"
              f"        python scripts/extract_lmf.py --frames {NUM_FRAMES} --save-energy")
        return False

    cfg = Config()
    if hasattr(cfg, "auto_tune"): cfg = cfg.auto_tune()
    cfg.data.num_frames, cfg.data.img_size, cfg.data.seed = NUM_FRAMES, IMAGE_SIZE, SEED
    cfg.train.batch_size, cfg.train.grad_accum = BATCH_SIZE, GRAD_ACCUM
    cfg.train.num_workers, cfg.train.lr, cfg.train.epochs = NUM_WORKERS, LR, epochs
    cfg.train.max_seconds = budget_secs
    cfg.train.early_stop_patience = patience
    cfg.train.early_stop_min_delta = min_delta
    cfg.model.use_tsc, cfg.model.use_lmf = tsc, lmf

    df = cf.load_manifest(MAN, ROOT)
    loaders = make_frame_loaders(df, cfg, CACHE)
    cw = loaders["train"].dataset.class_weights()
    print(f"[{cid}] TSC={tsc} LMF={lmf} | {cfg.data.num_frames} frames, "
          f"batch {cfg.train.batch_size}x{cfg.train.grad_accum}, "
          f"{cfg.train.num_workers} workers | budget {budget_secs/60:.0f} min")
    print(f"[{cid}] class weights (real, fake): {cw.tolist()}")

    m = build_model(cfg)
    hist = train_loop(m, loaders, cfg, class_weights=cw, run_name=f"dfe_{cid}")
    print(f"[{cid}] {hist.get('epochs_completed')} epochs, "
          f"best {hist.get('best_metric'):.4f}"
          f"{', stopped: ' + str(hist['stopped_early']) if hist.get('stopped_early') else ''}")
    return promote_best(cid)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=list(CONDITIONS))
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--budget-hours", type=float, default=6.0)
    ap.add_argument("--patience", type=int, default=2,
                    help="stop after N epochs with no gain above --min-delta")
    ap.add_argument("--min-delta", type=float, default=0.001,
                    help="improvement below this does not reset patience")
    ap.add_argument("--promote-best", action="store_true",
                    help="only recover <run>_best.pt files from an interrupted run")
    a = ap.parse_args()

    if a.promote_best:
        for cid in CONDITIONS:
            if not (OUT / f"fsvnet_dfe_{cid}.pt").exists():
                promote_best(cid)
        sys.exit()

    torch.manual_seed(SEED); np.random.seed(SEED)
    todo = [c for c in a.only if c in CONDITIONS]
    print(f"conditions: {todo}   epochs<={a.epochs}   budget {a.budget_hours} h   "
          f"early stop: patience {a.patience}, min_delta {a.min_delta}\n")
    t0 = time.time()
    for i, cid in enumerate(todo):
        left = a.budget_hours * 3600 - (time.time() - t0)
        share = max(left / max(len(todo) - i, 1), 600.0)
        print("=" * 66)
        run(cid, share, a.epochs, a.patience, a.min_delta)
        print(f"  elapsed {(time.time()-t0)/3600:.2f} h of {a.budget_hours} h\n")
    print("DONE. Now run the notebook -- it will evaluate every checkpoint present.")
