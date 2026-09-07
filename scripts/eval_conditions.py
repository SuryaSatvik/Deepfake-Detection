"""
Evaluate the trained CNN conditions on the identity-disjoint TEST split.

    python scripts/eval_conditions.py

Why this exists: results/*.csv were written from the 1-epoch checkpoints and
are superseded by the 6-epoch runs in models/checkpoints/. The notebook's
evaluation cells do this job, but training and evaluation belong in a terminal,
not in Jupyter -- see the note at the top of train_conditions.py.

Reproduces notebook cells 31 / 33 / 35 / 43 exactly:
  * one shared evaluate() for every condition
  * threshold chosen on VALIDATION by Youden's J, applied unchanged to test
  * bootstrap CI on AUC, paired bootstrap on every ablation pair
Writes results/{metrics,predictions,comparison,calibration,ablations}.csv,
results/eval_run_meta.json and results/test_scores.npz (raw scores, so figures
can be redrawn without re-running the GPU).
"""
import json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (roc_auc_score, average_precision_score, accuracy_score,
                             f1_score, precision_score, recall_score, brier_score_loss,
                             confusion_matrix, balanced_accuracy_score, roc_curve)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fsvnet import Config, celebdf_frames as cf
from fsvnet.model import build_model
from fsvnet.dataset_frames import make_frame_loaders, lmf_pack

NUM_FRAMES, IMAGE_SIZE, BATCH_SIZE, NUM_WORKERS, SEED = 16, 224, 2, 2, 42
MAN = ROOT / "data" / "manifests" / "celebdf_v2_identity.csv"
CACHE = ROOT / "data" / "processed" / f"lmf_cache_{NUM_FRAMES:02d}f.npz"
CKPT_DIR = ROOT / "models" / "checkpoints"
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONDITIONS = [("B0", "FSV-Net conventional (no TSC, no LMF)"),
              ("C1", "FSV-Net/DFE + TSC only"),
              ("C2", "FSV-Net/DFE + LMF only"),
              ("C3", "FSV-Net/DFE full (PROPOSED)")]


def ece(y, p, bins=15):
    conf = np.maximum(p, 1 - p); correct = ((p >= .5).astype(int) == y).astype(float)
    edges = np.linspace(0, 1, bins + 1); e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum(): e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def boot_ci(y, p, n=2000, seed=0):
    rng = np.random.default_rng(seed); o = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) < 2: continue
        o.append(roc_auc_score(y[i], p[i]))
    return (round(float(np.percentile(o, 2.5)), 4), round(float(np.percentile(o, 97.5)), 4))


def evaluate(name, y, p, thr=0.5, **kw):
    y = np.asarray(y).astype(int); p = np.asarray(p, dtype=float)
    pred = (p >= thr).astype(int)
    tn, fp, fn_, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    lo, hi = boot_ci(y, p)
    row = {"Model": name, "n": len(y), "Threshold": thr,
           "Accuracy": accuracy_score(y, pred),
           "Precision": precision_score(y, pred, zero_division=0),
           "Recall": recall_score(y, pred, zero_division=0),
           "F1": f1_score(y, pred, zero_division=0),
           "ROC-AUC": roc_auc_score(y, p), "AUC lo": lo, "AUC hi": hi,
           "PR-AUC": average_precision_score(y, p),
           "Specificity": tn / max(tn + fp, 1), "Sensitivity": tp / max(tp + fn_, 1),
           "Balanced Acc.": balanced_accuracy_score(y, pred),
           "ECE": ece(y, p), "Brier": brier_score_loss(y, p),
           "TN": tn, "FP": fp, "FN": fn_, "TP": tp}
    row.update(kw)
    return row


def paired_bootstrap(preds, a, b, n=4000, seed=1):
    if a not in preds or b not in preds: return None
    ya, pa = preds[a]; yb, pb = preds[b]
    if len(ya) != len(yb) or not np.array_equal(ya, yb): return None
    rng = np.random.default_rng(seed); d = []
    for _ in range(n):
        i = rng.integers(0, len(ya), len(ya))
        if len(set(ya[i])) < 2: continue
        d.append(roc_auc_score(ya[i], pb[i]) - roc_auc_score(ya[i], pa[i]))
    d = np.array(d); lo, hi = np.percentile(d, [2.5, 97.5])
    return {"comparison": f"{b}  -  {a}", "delta AUC": round(float(d.mean()), 4),
            "CI lo": round(float(lo), 4), "CI hi": round(float(hi), 4),
            "p (two-sided)": round(float(2 * min((d <= 0).mean(), (d >= 0).mean())), 4),
            "verdict": "RESOLVED" if (lo > 0 or hi < 0) else "not resolved"}


def main():
    print(f"device: {DEVICE}")
    if DEVICE == "cuda":
        pr = torch.cuda.get_device_properties(0)
        print(f"gpu   : {pr.name} {pr.total_memory/1e9:.1f} GB")
    else:
        print("WARNING: no CUDA. This will be slow but the numbers are identical.")

    torch.manual_seed(SEED); np.random.seed(SEED)
    df = cf.load_manifest(MAN, ROOT)
    print(df.groupby(["split", "label_name"]).size().unstack(fill_value=0)
            .reindex(["train", "val", "test"]))

    rows, preds, meta = [], {}, {}
    for cid, label in CONDITIONS:
        ck = CKPT_DIR / f"fsvnet_dfe_{cid}.pt"
        if not ck.exists():
            print(f"[{cid}] NO CHECKPOINT at {ck.name} -- reported as NOT RUN"); continue
        blob = torch.load(ck, map_location=DEVICE, weights_only=False)
        c = Config()
        c.data.num_frames, c.data.img_size, c.data.seed = NUM_FRAMES, IMAGE_SIZE, SEED
        c.train.batch_size, c.train.num_workers = BATCH_SIZE, NUM_WORKERS
        c.model.use_tsc, c.model.use_lmf = blob["use_tsc"], blob["use_lmf"]
        m = build_model(c).to(DEVICE)
        m.load_state_dict(blob["state_dict"]); m.eval()
        loaders = make_frame_loaders(df, c, CACHE)

        def score(split):
            ys, ps = [], []
            with torch.no_grad():
                for b in loaders[split]:
                    out = m.predict_proba(b["clip"].to(DEVICE), lmf=lmf_pack(b))
                    ps += out[:, 1].cpu().tolist(); ys += b["label"].tolist()
            return np.array(ys), np.array(ps)

        if DEVICE == "cuda": torch.cuda.reset_peak_memory_stats()
        print(f"[{cid}] scoring val ...")
        yv, pv = score("val")
        fpr, tpr, grid = roc_curve(yv, pv)
        thr = float(grid[int(np.argmax(tpr - fpr))])
        print(f"[{cid}] val AUC {roc_auc_score(yv, pv):.4f}   threshold (Youden J) {thr:.4f}")
        print(f"[{cid}] scoring test ...")
        t0 = time.time(); ys, ps = score("test"); dt = time.time() - t0

        r = evaluate(label, ys, ps, thr=thr,
                     **{"Latency (ms)": dt * 1000 / max(len(ys), 1),
                        "Params": sum(q.numel() for q in m.parameters()),
                        "Ckpt (MB)": ck.stat().st_size / 1e6,
                        "GPU (MB)": (torch.cuda.max_memory_allocated() / 1e6)
                                    if DEVICE == "cuda" else np.nan})
        rows.append(r); preds[label] = (ys, ps)
        meta[cid] = {"label": label, "checkpoint": ck.name,
                     "epoch": blob.get("epoch"), "train_metric": blob.get("metric"),
                     "monitor": blob.get("monitor"), "use_tsc": blob["use_tsc"],
                     "use_lmf": blob["use_lmf"], "val_auc": float(roc_auc_score(yv, pv)),
                     "threshold": thr, "n_val": int(len(yv)), "n_test": int(len(ys))}
        np.savez(OUT / f"scores_{cid}.npz", y_val=yv, p_val=pv, y_test=ys, p_test=ps)
        print(f"  {label:<42} test AUC {r['ROC-AUC']:.4f} "
              f"[{r['AUC lo']:.3f},{r['AUC hi']:.3f}]  F1 {r['F1']:.4f}  "
              f"BalAcc {r['Balanced Acc.']:.4f}  ECE {r['ECE']:.4f}")
        del m; torch.cuda.empty_cache() if DEVICE == "cuda" else None

    if not rows:
        print("nothing evaluated -- no checkpoints found."); return

    RES = pd.DataFrame(rows).sort_values("ROC-AUC", ascending=False).reset_index(drop=True)
    RES.to_csv(OUT / "metrics.csv", index=False)
    pd.concat([pd.DataFrame({"ground_truth": y, "prob_fake": p,
                             "prediction": (p >= 0.5).astype(int), "model": k})
               for k, (y, p) in preds.items()]).to_csv(OUT / "predictions.csv", index=False)
    RES[["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC",
         "Balanced Acc.", "ECE", "Latency (ms)", "Params"]].to_csv(OUT / "comparison.csv", index=False)
    RES[["Model", "ECE", "Brier"]].rename(columns={"Model": "model"}).to_csv(
        OUT / "calibration.csv", index=False)

    PAIRS = [("FSV-Net conventional (no TSC, no LMF)", "FSV-Net/DFE + TSC only"),
             ("FSV-Net conventional (no TSC, no LMF)", "FSV-Net/DFE + LMF only"),
             ("FSV-Net conventional (no TSC, no LMF)", "FSV-Net/DFE full (PROPOSED)"),
             ("FSV-Net/DFE + TSC only", "FSV-Net/DFE full (PROPOSED)"),
             ("FSV-Net/DFE + LMF only", "FSV-Net/DFE full (PROPOSED)")]
    abl = [r for r in (paired_bootstrap(preds, a, b) for a, b in PAIRS) if r]
    if abl:
        pd.DataFrame(abl).to_csv(OUT / "ablations.csv", index=False)
        print("\n" + pd.DataFrame(abl).to_string(index=False))

    json.dump({"run_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "device": DEVICE, "num_frames": NUM_FRAMES, "img_size": IMAGE_SIZE,
               "seed": SEED, "conditions": meta},
              open(OUT / "eval_run_meta.json", "w"), indent=2)
    print("\nwrote results/metrics.csv, predictions.csv, comparison.csv, "
          "calibration.csv, ablations.csv, eval_run_meta.json, scores_*.npz")
    print(RES[["Model", "ROC-AUC", "AUC lo", "AUC hi", "F1", "Balanced Acc.", "ECE"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
