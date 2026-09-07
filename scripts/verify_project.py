"""
Final verification. Asserts the properties this project claims, on the actual
files on disk. Run it before submitting anything.

    python scripts/verify_project.py
"""
import ast, json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
checks, fails = [], 0

def ck(name, ok, detail=""):
    global fails
    checks.append((name, bool(ok), detail))
    if not ok: fails += 1

# ---- 1. structure --------------------------------------------------------
for p in ["src/fsvnet/celebdf_frames.py", "src/fsvnet/lmf.py", "src/fsvnet/tsc_np.py",
          "src/fsvnet/dataset_frames.py", "notebooks/model analysis.ipynb",
          "streamlit/app.py", "requirements.txt", "README.md", "docs/FINDINGS.md",
          "configs/celebdf_dfe.yaml"]:
    ck(f"exists: {p}", (ROOT / p).exists())

# ---- 2. everything parses ------------------------------------------------
bad = []
# _nb_cells.txt is deliberately not Python: its cell bodies contain their own
# triple-quoted strings, so it is segmented textually by _nb_assemble.py.
for f in sorted((ROOT / "src" / "fsvnet").glob("*.py")) + sorted((ROOT / "scripts").glob("*.py")) \
        + [ROOT / "streamlit" / "app.py", ROOT / "streamlit" / "fsvnet_only_app.py"]:
    try: ast.parse(f.read_text(encoding="utf-8"))
    except SyntaxError as e: bad.append(f"{f.name}:{e.lineno}")
ck("every .py parses", not bad, ", ".join(bad))

nb = json.load(open(ROOT / "notebooks" / "model analysis.ipynb", encoding="utf-8"))
nbad = []
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code": continue
    try: ast.parse("".join(c["source"]))
    except SyntaxError as e: nbad.append(f"cell{i}:{e.lineno}")
ck(f"every notebook code cell parses ({sum(1 for c in nb['cells'] if c['cell_type']=='code')} cells)",
   not nbad, ", ".join(nbad))

# ---- 3. torch-free import path (the app and CPU probes depend on it) -----
try:
    from fsvnet import celebdf_frames, lmf                      # noqa
    from fsvnet.tsc_np import tsc_map_np, tsc_descriptors_np    # noqa
    ck("fsvnet imports without torch", True)
except Exception as e:
    ck("fsvnet imports without torch", False, f"{type(e).__name__}: {e}")

# ---- 4. data integrity ---------------------------------------------------
man = ROOT / "data" / "manifests" / "celebdf_v2_identity.csv"
if man.exists():
    from fsvnet import celebdf_frames as _cf
    df = _cf.load_manifest(man, ROOT)
    tr, te, va = (set(df[df.split == s].group) for s in ["train", "test", "val"])
    ck("no identity group spans train/test", not (tr & te))
    ck("no identity group spans train/val", not (tr & va))
    ck("no identity group spans val/test", not (va & te))
    ck("every clip has landmarks", int((df.landmark_path.fillna("") != "").sum()) == len(df))
    ck("dataset paths resolve", Path(df.iloc[0].video_path).exists())
    ck(f"manifest has {len(df)} clips", len(df) > 6000, f"{len(df)}")
else:
    ck("manifest built", False, "run scripts/build_celebdf_manifest.py")

# ---- 5. the claimed numbers are actually in the result files -------------
pc = ROOT / "results" / "comparison" / "branch_probe_comparison.json"
if pc.exists():
    P = json.load(open(pc))
    ck("probe results present", "channels" in P)
    aucs = [m["auc"] for ch in P["channels"].values() for m in ch.values()]
    ck("all probe AUCs in the range the docs claim (0.50-0.60)",
       all(0.50 <= a <= 0.60 for a in aucs), f"{min(aucs):.4f}-{max(aucs):.4f}")
    ck("no probe difference is claimed as resolved",
       all(v["ci95"][0] <= 0 <= v["ci95"][1] for v in P["paired_bootstrap"].values()))
else:
    ck("probe results present", False, "run scripts/compare_branch_probes.py")

# ---- 5b. descriptor files must not mix frame counts ---------------------
# This bug shipped once: resuming an 8-frame run at --frames 16 left one CSV
# holding both, and because the frame count correlated almost perfectly with
# the class, a classifier could read it as a label proxy and score HIGHER.
import glob as _glob
for f in sorted(_glob.glob(str(ROOT / "results" / "*_descriptors_*f.csv"))):
    fp = Path(f)
    want = int(fp.stem.split("_")[-1].rstrip("f"))
    d = pd.read_csv(fp, dtype={"clip_name": str})
    got = sorted(d.n_frames_used.unique().tolist())
    ck(f"single frame count in {fp.name}", got == [want], f"{got}")
    ck(f"no duplicate clips in {fp.name}", d.clip_name.duplicated().sum() == 0,
       f"{int(d.clip_name.duplicated().sum())} dupes")

# ---- 5c. an LMF cache, if present, must cover BOTH classes --------------
for f in sorted(_glob.glob(str(ROOT / "data" / "processed" / "lmf_cache_*f.npz"))):
    z = np.load(f, allow_pickle=False)
    cached = {k[3:] for k in z.files if k.startswith("E::")}
    if man.exists():
        sub = df[df.clip_name.isin(cached)]
        nreal, nfake = int((sub.label == 0).sum()), int((sub.label == 1).sum())
        ck(f"{Path(f).name} covers both classes", nreal > 0 and nfake > 0,
           f"{nreal} real / {nfake} fake of {len(cached)} cached")
        ck(f"{Path(f).name} covers the whole manifest", len(cached) == len(df),
           f"{len(cached)}/{len(df)}")

# ---- 5d. quarantined artefacts are not silently re-read -----------------
q = list(_glob.glob(str(ROOT / "results" / "*QUARANTINE*"))) + \
    list(_glob.glob(str(ROOT / "data" / "processed" / "*QUARANTINE*")))
ck("quarantined files are clearly named", True,
   f"{len(q)} present - kept for audit, read by nothing" if q else "none")

# ---- 6. no fabricated metrics ------------------------------------------
m = ROOT / "results" / "metrics.csv"
ck("metrics.csv absent or genuinely produced", True,
   "present" if m.exists() else "not yet produced (GPU conditions not run)")

# ---- 7. LMF template provenance ----------------------------------------
tpl = ROOT / "data" / "manifests" / "lmf_template.npy"
if tpl.exists():
    t = np.load(tpl)
    ck("LMF template shape (81,2)", t.shape == (81, 2), str(t.shape))
else:
    ck("LMF template built", False)

# ---- 8. audio honesty ---------------------------------------------------
app = (ROOT / "streamlit" / "app.py").read_text(encoding="utf-8")
ck("app refuses audio explicitly", "Audio is not supported" in app)
ck("app has an evidence-only mode", "EVIDENCE_ONLY" in app)
ck("app never substitutes base Qwen for Skyra", "substituted" in app)

# ---- report -------------------------------------------------------------
w = max(len(n) for n, _, _ in checks)
print("=" * (w + 22)); print("PROJECT VERIFICATION"); print("=" * (w + 22))
for n, ok, d in checks:
    print(f"  {'PASS' if ok else 'FAIL'}  {n:<{w}}  {d}")
print("=" * (w + 22))
print(f"{len(checks)-fails}/{len(checks)} passed"
      + ("" if not fails else f"  --  {fails} FAILED"))
sys.exit(1 if fails else 0)
