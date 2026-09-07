"""
Build the Celeb-DF-v2 manifest and identity-disjoint splits from the
pre-extracted frames distribution. Run:  python scripts/build_celebdf_manifest.py
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from fsvnet import celebdf_frames as cf
from fsvnet import manifest as M
from fsvnet.utils import save_json

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Celeb-DF-v2"
OUT = ROOT / "data" / "manifests"
OUT.mkdir(parents=True, exist_ok=True)

print("=" * 70); print("1. LAYOUT PROBE (measured from disk)"); print("=" * 70)
import os
if (OUT/"celebdf_layout_probe.json").exists():
    probe = json.load(open(OUT/"celebdf_layout_probe.json"))
else:
    probe = cf.probe_layout(DATA)
print(json.dumps(probe, indent=2))
save_json(probe, OUT / "celebdf_layout_probe.json")

print("=" * 70); print("2. MANIFEST"); print("=" * 70)
M.ADAPTERS["celebdf_frames"] = lambda r: cf.rows_celebdf_frames(r, filelist_dir=OUT / "filelists")
df = M.build_manifest({"celebdf_frames": DATA})

print("=" * 70); print("3. IDENTITY-DISJOINT SPLIT"); print("=" * 70)
df = M.make_splits(df, split_by="identity", fracs=(0.70, 0.15, 0.15), seed=42)

print("=" * 70); print("4. LEAKAGE AUDIT"); print("=" * 70)
audit = M.leakage_audit(df)
save_json(audit, OUT / "leakage_audit.json")

print("=" * 70); print("5. OFFICIAL-PROTOCOL VARIANT"); print("=" * 70)
df_off = cf.apply_official_test_split(df, DATA / "List_of_testing_videos.txt",
                                      move_whole_group=True)
# re-split the non-test remainder into train/val
rest = df_off[df_off.split != "test"].copy()
rest = M._assign_groups(rest, (0.82, 0.18, 0.0), 42, "group")
df_off.loc[rest.index, "split"] = rest["split"]
M._report_split(df_off)

# Store paths RELATIVE to the project root. Absolute paths bake in the
# machine that built the manifest and die the moment the folder moves.
M.save_manifest(cf.to_relative(df, ROOT), OUT, "celebdf_v2_identity")
M.save_manifest(cf.to_relative(df_off, ROOT), OUT, "celebdf_v2_official")

print("=" * 70); print("6. DATASET STATISTICS"); print("=" * 70)
stats = {
    "n_clips": int(len(df)),
    "n_identity_groups": int(df.group.nunique()),
    "by_subset": df.groupby("subset").size().to_dict(),
    "by_label": {"real": int((df.label == 0).sum()), "fake": int((df.label == 1).sum())},
    "class_imbalance_fake_pct": round(float((df.label == 1).mean() * 100), 2),
    "frames_per_clip": {
        "min": int(df.frame_count.min()), "max": int(df.frame_count.max()),
        "mean": round(float(df.frame_count.mean()), 2),
        "median": float(df.frame_count.median()),
    },
    "duration_approx_s": {
        "min": float(df.duration_approx_s.min()), "max": float(df.duration_approx_s.max()),
        "mean": round(float(df.duration_approx_s.mean()), 2),
    },
    "total_frames": int(df.frame_count.sum()),
    "clips_with_landmarks": int((df.landmark_path.fillna("") != "").sum()),
    "unique_source_identities": int(df.identity.nunique()),
    "resolution": sorted(df.resolution.unique().tolist()),
    "split_identity_disjoint": df.groupby("split").size().to_dict(),
    "split_official": df_off.groupby("split").size().to_dict(),
}
print(json.dumps(stats, indent=2))
save_json(stats, OUT / "celebdf_v2_stats.json")
print("\nDONE -> data/manifests/")
