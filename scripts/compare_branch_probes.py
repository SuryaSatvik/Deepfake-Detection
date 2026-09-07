"""
Head-to-head CPU probe: LMF vs TSC vs both, on identical clips, identical
identity-disjoint split, identical classifiers.

This runs BEFORE any GPU time. Its purpose is to decide which forensic channel
is worth building the proposed model around, using measurement instead of
intuition. Nothing here is the proposed model; these are linear/GBDT probes on
closed-form descriptors and they are a LOWER BOUND on the learned branches.
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
from fsvnet.lmf import DESCRIPTOR_NAMES as LMF_F
from fsvnet.tsc_np import TSC_DESCRIPTOR_NAMES as TSC_F
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score

rd = lambda f: pd.read_csv(ROOT / "results" / f, dtype={"clip_name": str})
lmf = rd(f"lmf_descriptors_{PROBE_FRAMES:02d}f.csv")
tsc = rd(f"tsc_descriptors_{PROBE_FRAMES:02d}f.csv")
for _n, _d in [("lmf", lmf), ("tsc", tsc)]:
    assert set(_d.n_frames_used.unique()) == {PROBE_FRAMES}, \
        f"{_n} descriptors mix frame counts: {sorted(_d.n_frames_used.unique())}"
key = ["clip_name", "label", "split", "subset", "group"]
df = lmf[key + LMF_F].merge(tsc[key + TSC_F], on=key, how="inner")
print(f"clips with BOTH descriptor sets: {len(df)}")
print(df.groupby(["split", "label"]).size().unstack(fill_value=0).rename(columns={0: "real", 1: "fake"}))
tr, te = df[df.split == "train"], df[df.split == "test"]
assert not (set(tr.group) & set(te.group))

def boot(yt, p, n=2000, seed=0):
    rng = np.random.default_rng(seed); o = []
    for _ in range(n):
        i = rng.integers(0, len(yt), len(yt))
        if len(set(yt[i])) < 2: continue
        o.append(roc_auc_score(yt[i], p[i]))
    return round(float(np.percentile(o, 2.5)), 4), round(float(np.percentile(o, 97.5)), 4)

def run(feats, kind):
    m = (make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                       LogisticRegression(max_iter=3000, class_weight="balanced"))
         if kind == "linear" else
         make_pipeline(SimpleImputer(strategy="median"),
                       HistGradientBoostingClassifier(max_iter=300, random_state=0)))
    m.fit(tr[feats].values, tr.label.values)
    p = m.predict_proba(te[feats].values)[:, 1]
    yt = te.label.values
    lo, hi = boot(yt, p)
    return {"auc": round(float(roc_auc_score(yt, p)), 4), "ci95": [lo, hi],
            "pr_auc": round(float(average_precision_score(yt, p)), 4),
            "bal_acc": round(float(balanced_accuracy_score(yt, (p >= .5).astype(int))), 4),
            "n_features": len(feats)}, p

sets = {"LMF only": LMF_F, "TSC only": TSC_F, "LMF + TSC": LMF_F + TSC_F}
out, probs = {}, {}
print(f"\n{'channel':<12}{'model':<9}{'test AUC':>10}{'95% CI':>18}{'PR-AUC':>9}{'BalAcc':>9}")
print("-" * 68)
for name, F in sets.items():
    out[name] = {}
    for kind in ["linear", "gbdt"]:
        r, p = run(F, kind)
        out[name][kind] = r; probs[(name, kind)] = p
        print(f"{name:<12}{kind:<9}{r['auc']:>10.4f}{str(r['ci95']):>18}{r['pr_auc']:>9.4f}{r['bal_acc']:>9.4f}")

# paired bootstrap on the DIFFERENCE — the right test for two models scored on
# the same videos (their separate CIs are not independent samples)
def paired(pa, pb, yt, n=4000, seed=1):
    rng = np.random.default_rng(seed); d = []
    for _ in range(n):
        i = rng.integers(0, len(yt), len(yt))
        if len(set(yt[i])) < 2: continue
        d.append(roc_auc_score(yt[i], pb[i]) - roc_auc_score(yt[i], pa[i]))
    d = np.array(d)
    return {"delta_auc": round(float(d.mean()), 4),
            "ci95": [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)],
            "p_two_sided": round(float(2 * min((d <= 0).mean(), (d >= 0).mean())), 4)}

yt = te.label.values
print("\npaired bootstrap on the difference (gbdt):")
cmp = {}
for a, b in [("LMF only", "TSC only"), ("LMF only", "LMF + TSC"), ("TSC only", "LMF + TSC")]:
    r = paired(probs[(a, "gbdt")], probs[(b, "gbdt")], yt)
    cmp[f"{b} - {a}"] = r
    sig = "resolved" if (r["ci95"][0] > 0 or r["ci95"][1] < 0) else "NOT RESOLVED"
    print(f"  {b:<11} - {a:<11} {r['delta_auc']:+.4f}  CI {r['ci95']}  p={r['p_two_sided']:.4f}  [{sig}]")

res = {"n_clips": int(len(df)), "n_test": int(len(te)),
       "test_real": int((yt == 0).sum()), "test_fake": int((yt == 1).sum()),
       "split": "identity-disjoint (union-find over identity pairs)",
       "probe_frames_per_clip": 8, "channels": out, "paired_bootstrap": cmp}
(ROOT / "results" / "comparison").mkdir(parents=True, exist_ok=True)
json.dump(res, open(ROOT / "results" / "comparison" / "branch_probe_comparison.json", "w"), indent=2)
print("\nsaved -> results/comparison/branch_probe_comparison.json")
