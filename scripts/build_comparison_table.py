"""
The comparison table you CAN defend.

Not accuracy — nothing is trained yet, and even when it is, a cross-test-set
accuracy comparison against published detectors would be invalid.

Instead: the axes on which this system measurably leads standard practice, with
every number pulled from a result file produced by another script in this repo.
Run `python scripts/verify_project.py` to confirm none of it is hand-typed.

    python scripts/build_comparison_table.py
"""
import json, sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
MAN = ROOT / "data" / "manifests"
RES = ROOT / "results"
OUT = RES / "comparison"
OUT.mkdir(parents=True, exist_ok=True)

# ---- load the measured facts (never hardcode a number that a script produced)
audit = json.load(open(MAN / "leakage_audit.json"))
stats = json.load(open(MAN / "celebdf_v2_stats.json"))
probe = json.load(open(OUT / "branch_probe_comparison.json"))
leak = json.load(open(RES / "baseline" / "lmf_leakage_demo.json"))

df = pd.read_csv(MAN / "celebdf_v2_identity.csv", dtype={"clip_name": str})
tr, te = set(df[df.split == "train"].group), set(df[df.split == "test"].group)
our_contamination = len(tr & te) / max(df[df.split == "test"].group.nunique(), 1)

from fsvnet import celebdf_frames as cf
official = cf.official_test_names(ROOT / "Celeb-DF-v2" / "List_of_testing_videos.txt")
m = df.clip_name.isin(official)
official_groups = set(df.loc[m, "group"])
official_contamination = float(df.group.isin(official_groups).mean())

rnd = audit["contamination_rate"]

# ==========================================================================
# TABLE 1 — evaluation integrity.  This is a QUANTITATIVE lead, and it is the
# headline: lower is better, and ours is zero by construction.
# ==========================================================================
T1 = pd.DataFrame([
    {"Evaluation protocol": "Random split over clips (common in student/course work)",
     "Test clips sharing an identity with train": f"{rnd*100:.1f}%",
     "Identity-disjoint": "no",
     "Source": "measured, scripts/build_celebdf_manifest.py"},
    {"Evaluation protocol": "Official Celeb-DF-v2 test list (used by published work)",
     "Test clips sharing an identity with train": f"{official_contamination*100:.1f}%",
     "Identity-disjoint": "no",
     "Source": "measured, this script"},
    {"Evaluation protocol": "THIS WORK — union-find identity-disjoint split",
     "Test clips sharing an identity with train": f"{our_contamination*100:.1f}%",
     "Identity-disjoint": "YES",
     "Source": "measured, asserted in scripts/verify_project.py"},
])

# ==========================================================================
# TABLE 2 — the cost of the leak, measured on identical features/classifier.
# ==========================================================================
T2 = pd.DataFrame([
    {"Split regime": "random split over clips (leaky)",
     "linear AUC": leak["random_clip_split"]["linear"],
     "GBDT AUC": leak["random_clip_split"]["gbdt"]},
    {"Split regime": "random split, Celeb subset only (leaky)",
     "linear AUC": leak["random_split_celeb_only"]["linear"],
     "GBDT AUC": leak["random_split_celeb_only"]["gbdt"]},
    {"Split regime": "identity-disjoint (honest)",
     "linear AUC": leak["identity_disjoint"]["linear"],
     "GBDT AUC": leak["identity_disjoint"]["gbdt"]},
])
T2["inflation vs honest (linear)"] = (
    T2["linear AUC"] - leak["identity_disjoint"]["linear"]).round(4)

# ==========================================================================
# TABLE 3 — capability. Qualitative, but every cell is a published fact about
# that system or a property of code in this repo. No scores, by design.
# ==========================================================================
SYS = ["CNN detector (Xception / EfficientNet)",
       "CNN + Grad-CAM (explainable baseline)",
       "Skyra / Qwen2.5-VL (VLM reasoning)",
       "THIS WORK — FSV-Net/DFE"]
CAP = [
    ("Output beyond a probability",            "no", "saliency heatmap", "prose rationale", "measurements + prose grounded in them"),
    ("Evidence is a falsifiable number",       "no", "no", "no", "YES — percentile vs training reals"),
    ("Prediction traceable to frame + timestamp", "no", "partial", "partial", "YES — video/frame/timestamp/region"),
    ("Localises evidence to face geometry",    "no", "heatmap only", "no", "YES — inner / seam / outside regions"),
    ("Measures a cue a human cannot see",      "no", "no", "no", "YES — temporal spectral coherence"),
    ("Uses facial landmark geometry",          "no", "no", "no", "YES — Procrustes-canonical residual"),
    ("Separates 'model evidence' from 'explanation'", "n/a", "no", "no", "YES — enforced in UI and notebook"),
    ("Reports calibration (ECE / Brier)",      "rarely", "rarely", "not reported", "YES"),
    ("Reports paired significance tests",      "rarely", "rarely", "no", "YES — paired bootstrap on the difference"),
    ("Publishes negative results",             "rarely", "rarely", "n/a", "YES — see docs/FINDINGS.md"),
    ("Runs on 8 GB consumer GPU",              "yes", "yes", "inference only (7B NF4)", "yes"),
    ("Refuses unsupported modalities",         "n/a", "n/a", "n/a", "YES — audio explicitly refused"),
]
T3 = pd.DataFrame([dict(zip(["Capability"] + SYS, r)) for r in CAP])

# ==========================================================================
# TABLE 4 — accuracy. DELIBERATELY EMPTY until training runs.
# ==========================================================================
T4 = pd.DataFrame([
    {"Model": "Existing baseline (EfficientNet-B0)", "Test ROC-AUC": "NOT RUN", "Balanced Acc.": "NOT RUN", "ECE": "NOT RUN"},
    {"Model": "FSV-Net conventional (no TSC, no LMF)", "Test ROC-AUC": "NOT RUN", "Balanced Acc.": "NOT RUN", "ECE": "NOT RUN"},
    {"Model": "FSV-Net/DFE + TSC only", "Test ROC-AUC": "NOT RUN", "Balanced Acc.": "NOT RUN", "ECE": "NOT RUN"},
    {"Model": "FSV-Net/DFE + LMF only", "Test ROC-AUC": "NOT RUN", "Balanced Acc.": "NOT RUN", "ECE": "NOT RUN"},
    {"Model": "FSV-Net/DFE full (PROPOSED)", "Test ROC-AUC": "NOT RUN", "Balanced Acc.": "NOT RUN", "ECE": "NOT RUN"},
    {"Model": "Skyra (official weights, zero-shot OOD)", "Test ROC-AUC": "NOT RUN", "Balanced Acc.": "NOT RUN", "ECE": "NOT RUN"},
])
_p = probe["channels"]["TSC only"]["linear"]
T4.loc[len(T4)] = {"Model": "TSC descriptors, linear probe [CPU, measured]",
                   "Test ROC-AUC": f"{_p['auc']:.4f} [{_p['ci95'][0]:.3f}-{_p['ci95'][1]:.3f}]",
                   "Balanced Acc.": f"{_p['bal_acc']:.4f}", "ECE": "n/a (probe)"}

for name, t in [("evaluation_integrity", T1), ("leakage_cost", T2),
                ("capability", T3), ("accuracy_STATUS", T4)]:
    t.to_csv(OUT / f"{name}.csv", index=False)

def show(title, t, note=""):
    print("\n" + "=" * 78); print(title); print("=" * 78)
    print(t.to_string(index=False))
    if note: print("\n" + note)

show("TABLE 1 — EVALUATION INTEGRITY  (lower contamination is better)", T1,
     "This is the quantitative claim you can defend without a trained model:\n"
     "the standard protocols for this dataset are contaminated, and ours is not.")
show("TABLE 2 — WHAT THE LEAK IS WORTH  (identical features and classifier)", T2,
     f"A leaky split inflates AUC by up to "
     f"{max(T2['inflation vs honest (linear)']):+.4f} on the SAME features.")
show("TABLE 3 — CAPABILITY  (published facts / properties of this code)", T3,
     "No scores here, by design. Systems evaluated on different benchmarks\n"
     "cannot be compared by number; they can be compared by what they DO.")
show("TABLE 4 — ACCURACY", T4,
     "Deliberately unfilled. Every row becomes a real number the moment\n"
     "notebooks/model analysis.ipynb runs on a CUDA machine.\n"
     "An unfilled cell is a status, not a gap to be filled by hand.")
print(f"\nwritten -> {OUT.relative_to(ROOT)}/*.csv")
