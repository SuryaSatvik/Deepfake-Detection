"""Same resumable extractor as extract_lmf.py, for the TSC descriptors, so the
two branches can be compared on identical clips and an identical split."""
import argparse, sys, time, csv
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fsvnet import celebdf_frames as cf
MAN = ROOT / "data" / "manifests" / "celebdf_v2_identity.csv"
OUT = None  # set from --frames below; see extract_lmf.py for why
COLS = ["clip_name", "label", "split", "subset", "group", "n_frames_used"] + TSC_DESCRIPTOR_NAMES

ap = argparse.ArgumentParser()
ap.add_argument("--minutes", type=float, default=0.0)
ap.add_argument("--frames", type=int, default=8)
ap.add_argument("--max-fake", type=int, default=0)
a = ap.parse_args()
OUT = ROOT / "results" / f"tsc_descriptors_{a.frames:02d}f.csv"
# Same guard as extract_lmf.py: descriptors from different frame counts are not
# comparable, and if the counts correlate with the class they create a spurious
# signal that makes the result look BETTER. Refuse rather than append.
if OUT.exists():
    _prev = pd.read_csv(OUT, dtype={"clip_name": str})
    _bad = set(_prev.n_frames_used.unique().tolist()) - {a.frames}
    if _bad:
        sys.exit(f"REFUSING TO APPEND: {OUT.name} has rows at {sorted(_bad)} frames, "
                 f"--frames={a.frames}. Delete it or match the frame count.")

df = cf.load_manifest(MAN, ROOT)
if a.max_fake:
    keep = [df[df.label == 0]]
    for s, sub in df[df.label == 1].groupby("split"):
        keep.append(sub.sample(min(a.max_fake, len(sub)), random_state=42))
    df = pd.concat(keep).sort_values("clip_name").reset_index(drop=True)

OUT.parent.mkdir(parents=True, exist_ok=True)
done = set(pd.read_csv(OUT, dtype={"clip_name": str}).clip_name) if OUT.exists() else set()
todo = df[~df.clip_name.astype(str).isin(done)]
log.info(f"{len(done)} done, {len(todo)} to go (of {len(df)})")
if not len(todo):
    print("ALL DONE"); sys.exit()

t0 = time.time(); budget = a.minutes * 60 if a.minutes else float("inf"); n = 0
hdr = not OUT.exists()
with open(OUT, "a", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=COLS)
    if hdr: w.writeheader()
    for _, r in todo.iterrows():
        if time.time() - t0 > budget: break
        try:
            fr, lm, meta = cf.load_clip(r.video_path, None, n_frames=a.frames, img_size=256)
            m, sp = tsc_map_np(fr, n_bins=64)
            row = {"clip_name": r.clip_name, "label": int(r.label), "split": r.split,
                   "subset": r.subset, "group": r.group, "n_frames_used": len(fr)}
            row.update(tsc_descriptors_np(m, sp)); w.writerow(row); n += 1
        except Exception as e:
            log.warning(f"{r.clip_name}: {type(e).__name__} {e}")
        if n % 100 == 0 and n:
            el = time.time() - t0; rate = n / el
            print(f"  {n:>5}/{len(todo)}  {rate:4.1f} clips/s  "
                  f"eta {(len(todo)-n)/max(rate,1e-9)/60:5.1f}m", flush=True)
        if n % 50 == 0 and n: fh.flush()
el = time.time() - t0
print(f"processed {n} in {el:.1f}s ({el/max(n,1):.3f}s/clip); total {len(done)+n}/{len(df)}")
