"""
Honest identity-disjoint probe of the LMF descriptors.

This is NOT the proposed model. It is a 7-feature logistic regression on the
closed-form LMF descriptors, fitted on train and evaluated on the held-out
identity-disjoint test split. Its only job is to answer, before any GPU time
is spent: does the landmark-conditioned motion-compensated signal survive on
real Celeb-DF-v2 across unseen identities?

A linear probe is deliberately weak. Whatever it scores is a LOWER BOUND on
what the learned branch can do, and it cannot be accused of memorising.
"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
# Descriptor files are suffixed with the frame count they were computed at.
# Mixing frame counts silently corrupts the comparison (see extract_lmf.py),
# so the frame count is part of the filename and part of every read.
PROBE_FRAMES = 8

sys.path.insert(0, str(ROOT / "src"))
from fsvnet.lmf import DESCRIPTOR_NAMES

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (roc_auc_score, average_precision_score, accuracy_score,
                             f1_score, precision_score, recall_score, brier_score_loss,
                             confusion_matrix, balanced_accuracy_score)

df = pd.read_csv(ROOT / "results" / f"lmf_descriptors_{PROBE_FRAMES:02d}f.csv",
                 dtype={"clip_name": str})
assert set(df.n_frames_used.unique()) == {PROBE_FRAMES}, "mixed frame counts"
print(f"clips with descriptors: {len(df)}")
print(df.groupby(["split", "label"]).size().unstack(fill_value=0).rename(columns={0: "real", 1: "fake"}))

tr = df[df.split == "train"]; te = df[df.split == "test"]; va = df[df.split == "val"]
assert not (set(tr.group) & set(te.group)), "GROUP LEAK between train and test"
print(f"\nidentity-group overlap train/test: {len(set(tr.group) & set(te.group))}  (must be 0)")

X = lambda d: d[DESCRIPTOR_NAMES].values
y = lambda d: d.label.values

# ---- per-descriptor univariate AUC on TEST (direction-free) --------------
print(f"\n{'descriptor':<24}{'test AUC':>10}{'|AUC-0.5|':>11}")
print("-" * 45)
uni = {}
for k in DESCRIPTOR_NAMES:
    v = te[k].values.astype(float)
    m = np.isfinite(v)
    if m.sum() < 10 or len(set(te.label.values[m])) < 2:
        continue
    a = roc_auc_score(te.label.values[m], v[m])
    uni[k] = float(a)
    print(f"{k:<24}{a:>10.4f}{abs(a-0.5):>11.4f}")

# ---- multivariate linear probe ------------------------------------------
clf = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                    LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0))
clf.fit(X(tr), y(tr))
p = clf.predict_proba(X(te))[:, 1]
yt = y(te)

def boot_auc(yt, p, n=2000, seed=0):
    rng = np.random.default_rng(seed); out = []
    for _ in range(n):
        i = rng.integers(0, len(yt), len(yt))
        if len(set(yt[i])) < 2: continue
        out.append(roc_auc_score(yt[i], p[i]))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))

thr = 0.5
pred = (p >= thr).astype(int)
tn, fp, fn, tp = confusion_matrix(yt, pred).ravel()
lo, hi = boot_auc(yt, p)
res = {
    "n_train": int(len(tr)), "n_test": int(len(te)),
    "test_real": int((yt == 0).sum()), "test_fake": int((yt == 1).sum()),
    "roc_auc": round(float(roc_auc_score(yt, p)), 4),
    "roc_auc_ci95": [round(lo, 4), round(hi, 4)],
    "pr_auc": round(float(average_precision_score(yt, p)), 4),
    "accuracy": round(float(accuracy_score(yt, pred)), 4),
    "balanced_accuracy": round(float(balanced_accuracy_score(yt, pred)), 4),
    "precision": round(float(precision_score(yt, pred, zero_division=0)), 4),
    "recall_sensitivity": round(float(recall_score(yt, pred, zero_division=0)), 4),
    "specificity": round(float(tn / max(tn + fp, 1)), 4),
    "f1": round(float(f1_score(yt, pred, zero_division=0)), 4),
    "brier": round(float(brier_score_loss(yt, p)), 4),
    "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    "univariate_test_auc": {k: round(v, 4) for k, v in uni.items()},
    "coefficients": dict(zip(DESCRIPTOR_NAMES,
                             [round(float(c), 4) for c in clf[-1].coef_[0]])),
}
print("\n" + "=" * 60)
print("LMF LINEAR PROBE — identity-disjoint test split")
print("=" * 60)
print(json.dumps(res, indent=2))

# ---- ablation: which descriptors carry it -------------------------------
print("\nleave-one-out (test AUC when that descriptor is removed):")
full = res["roc_auc"]
abl = {}
for k in DESCRIPTOR_NAMES:
    keep = [c for c in DESCRIPTOR_NAMES if c != k]
    c2 = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                       LogisticRegression(max_iter=2000, class_weight="balanced"))
    c2.fit(tr[keep].values, y(tr))
    a = roc_auc_score(yt, c2.predict_proba(te[keep].values)[:, 1])
    abl[k] = round(float(a), 4)
    print(f"  -{k:<24}{a:.4f}   ({a-full:+.4f})")
res["leave_one_out_test_auc"] = abl
res["full_test_auc"] = full

(ROOT / "results" / "baseline").mkdir(parents=True, exist_ok=True)
json.dump(res, open(ROOT / "results" / "baseline" / "lmf_linear_probe.json", "w"), indent=2)
pd.DataFrame({"clip_name": te.clip_name, "ground_truth": yt, "prob_fake": p,
              "prediction": pred, "model": "LMF-linear-probe", "split": "test"}
             ).to_csv(ROOT / "results" / "baseline" / "lmf_probe_predictions.csv", index=False)
print("\nsaved -> results/baseline/lmf_linear_probe.json")
