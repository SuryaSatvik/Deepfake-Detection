"""
Why the identity-disjoint LMF probe scores ~0.5 while the same descriptors look
strongly separated on a same-identity sample.

Runs the SAME features and the SAME classifier under three split regimes and a
stronger nonlinear model, so the gap is measured rather than argued.
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
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score

df = pd.read_csv(ROOT / "results" / f"lmf_descriptors_{PROBE_FRAMES:02d}f.csv",
                 dtype={"clip_name": str}).dropna(subset=["label"])
assert set(df.n_frames_used.unique()) == {PROBE_FRAMES}, "mixed frame counts"
F, y = DESCRIPTOR_NAMES, "label"
rng = np.random.default_rng(0)

def fit_eval(tr, te, kind="linear"):
    if kind == "linear":
        m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                          LogisticRegression(max_iter=2000, class_weight="balanced"))
    else:
        m = make_pipeline(SimpleImputer(strategy="median"),
                          HistGradientBoostingClassifier(max_iter=300, random_state=0))
    m.fit(tr[F].values, tr[y].values)
    p = m.predict_proba(te[F].values)[:, 1]
    return roc_auc_score(te[y].values, p)

out = {}
# A. identity-disjoint (the honest one)
tr, te = df[df.split == "train"], df[df.split == "test"]
out["identity_disjoint"] = {"linear": round(fit_eval(tr, te), 4),
                            "gbdt":   round(fit_eval(tr, te, "gbdt"), 4)}
# B. random split over clips (leaks identity)
idx = rng.permutation(len(df)); k = int(0.7 * len(df))
trR, teR = df.iloc[idx[:k]], df.iloc[idx[k:]]
out["random_clip_split"] = {"linear": round(fit_eval(trR, teR), 4),
                            "gbdt":   round(fit_eval(trR, teR, "gbdt"), 4)}
# C. the same-identity slice the first probe used (id0/id1..., Celeb-real vs Celeb-synthesis)
sub = df[df.subset.isin(["Celeb-real", "Celeb-synthesis"])]
idx2 = rng.permutation(len(sub)); k2 = int(0.7 * len(sub))
out["random_split_celeb_only"] = {"linear": round(fit_eval(sub.iloc[idx2[:k2]], sub.iloc[idx2[k2:]]), 4),
                                  "gbdt":   round(fit_eval(sub.iloc[idx2[:k2]], sub.iloc[idx2[k2:]], "gbdt"), 4)}

# D. IDENTITY PROBE — can the descriptors predict WHO the person is?
#    If yes, the "real vs fake" signal above was identity, not manipulation.
from sklearn.model_selection import cross_val_score
g = df[df.subset == "Celeb-real"].copy()
top = g.group.value_counts().head(8).index
g = g[g.group.isin(top)]
if len(g) > 40:
    m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                      LogisticRegression(max_iter=3000, multi_class="auto"))
    acc = cross_val_score(m, g[F].values, g.group.values, cv=5, scoring="accuracy")
    out["identity_prediction_from_descriptors"] = {
        "n_clips": int(len(g)), "n_identities": int(g.group.nunique()),
        "chance_accuracy": round(float(g.group.value_counts(normalize=True).max()), 4),
        "cv_accuracy": round(float(acc.mean()), 4),
        "cv_std": round(float(acc.std()), 4),
    }

# E. subset confound: do the descriptors just detect YouTube-real vs Celeb-real?
r = df[df.label == 0]
if r.subset.nunique() > 1:
    m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                      LogisticRegression(max_iter=2000, class_weight="balanced"))
    yy = (r.subset == "YouTube-real").astype(int).values
    from sklearn.model_selection import cross_val_predict
    pp = cross_val_predict(m, r[F].values, yy, cv=5, method="predict_proba")[:, 1]
    out["source_subset_auc_among_reals"] = round(float(roc_auc_score(yy, pp)), 4)

print(json.dumps(out, indent=2))
json.dump(out, open(ROOT / "results" / "baseline" / "lmf_leakage_demo.json", "w"), indent=2)
