"""
Extract LMF descriptors (and optionally the region-energy cache) for the
manifest clips.

    python scripts/extract_lmf.py --frames 16 --save-energy

Resumable: interrupt and re-run freely.

THREE THINGS THIS SCRIPT NOW REFUSES TO DO, EACH BECAUSE IT WENT WRONG ONCE
---------------------------------------------------------------------------
1. **Mix frame counts in one CSV.** Descriptors computed from 8 frames and 16
   frames are not comparable. Worse, if a partial run is resumed at a
   different --frames, the frame count ends up correlated with the class
   (whatever was done first is over-represented in one label), and a
   classifier reads frame count as a proxy for the label. That produces a
   HIGHER score from a pure artifact. Output paths therefore encode the frame
   count, and an existing file with a different count is a hard error.

2. **Skip a clip whose energy is not cached.** Resume used to be keyed on the
   CSV alone, so a clip already in the CSV was skipped even when its energy
   had never been written -- silently leaving holes in the cache exactly where
   the earlier run had been. With --save-energy, a clip counts as done only if
   it is in BOTH the CSV and the cache.

3. **Hold the whole cache in memory until the end.** It is now flushed every
   `--flush-every` clips, so an interrupt costs at most that many clips.

It also prints progress, which it embarrassingly did not before.
"""
import argparse, csv, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fsvnet import lmf, celebdf_frames as cf
from fsvnet.utils import log

MAN = ROOT / "data" / "manifests" / "celebdf_v2_identity.csv"
TPL = ROOT / "data" / "manifests" / "lmf_template.npy"
COLS = ["clip_name", "label", "split", "subset", "group", "n_frames_used"] + lmf.DESCRIPTOR_NAMES


def out_paths(frames: int):
    return (ROOT / "results" / f"lmf_descriptors_{frames:02d}f.csv",
            ROOT / "data" / "processed" / f"lmf_cache_{frames:02d}f.npz")


def get_template(df, n_clips=200, n_per=4):
    if TPL.exists():
        return np.load(TPL)
    tr = df[(df.split == "train") & (df.label == 0)]
    log.info(f"building canonical template from {min(n_clips, len(tr))} TRAIN REAL clips")
    pts = []
    for lp in tr.landmark_path.head(n_clips):
        for f in sorted(Path(lp).glob("*.npy"))[:n_per]:
            pts.append(np.load(f).astype(np.float64))
    tpl = lmf.build_template(np.stack(pts), size=256)
    TPL.parent.mkdir(parents=True, exist_ok=True)
    np.save(TPL, tpl)
    return tpl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--minutes", type=float, default=0.0, help="stop after N minutes (0 = to completion)")
    ap.add_argument("--max-fake", type=int, default=0, help="cap fakes per split (0 = all)")
    ap.add_argument("--save-energy", action="store_true",
                    help="also cache the (T-1,3) region-energy series for training")
    ap.add_argument("--flush-every", type=int, default=250)
    a = ap.parse_args()

    OUT, CACHE = out_paths(a.frames)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    CACHE.parent.mkdir(parents=True, exist_ok=True)

    # Paths in the manifest are relative to the project root; resolve them
    # for THIS machine so the same manifest works on Windows and Linux.
    df_all = cf.load_manifest(MAN, ROOT)
    df = df_all
    if a.max_fake:
        keep = [df[df.label == 0]]
        for _s, sub in df[df.label == 1].groupby("split"):
            keep.append(sub.sample(min(a.max_fake, len(sub)), random_state=42))
        df = pd.concat(keep).sort_values("clip_name").reset_index(drop=True)

    # ---- guard 1: never mix frame counts in one file -----------------------
    done_csv = set()
    if OUT.exists():
        prev = pd.read_csv(OUT, dtype={"clip_name": str})
        counts = set(prev.n_frames_used.unique().tolist())
        if counts - {a.frames}:
            sys.exit(
                f"\nREFUSING TO APPEND.\n  {OUT.name} already contains rows computed at "
                f"{sorted(counts)} frames, but --frames={a.frames}.\n"
                "  Mixing frame counts makes the descriptors incomparable, and if the\n"
                "  counts correlate with the class it produces a spurious signal.\n"
                f"  Delete {OUT.name}, or re-run with --frames {sorted(counts)[0]}.\n")
        done_csv = set(prev.clip_name)

    # ---- guard 2: with --save-energy, done means in CSV *and* in cache -----
    energies, done_cache = {}, set()
    if a.save_energy and CACHE.exists():
        z = np.load(CACHE, allow_pickle=False)
        energies = {k: z[k] for k in z.files}
        done_cache = {k[3:] for k in energies if k.startswith("E::")}
    done = (done_csv & done_cache) if a.save_energy else done_csv

    todo = df[~df.clip_name.isin(done)]
    if a.save_energy:
        holes = len(done_csv - done_cache)
        if holes:
            log.warning(f"{holes} clips are in the CSV but missing from the cache — "
                        "they will be recomputed so the cache has no holes")
    log.info(f"frames={a.frames}  csv={OUT.name}  cache={CACHE.name if a.save_energy else 'off'}")
    log.info(f"{len(done)} done, {len(todo)} to go (of {len(df)})")
    if not len(todo):
        print("ALL DONE"); return

    tpl = get_template(df_all)
    t0 = time.time()
    budget = a.minutes * 60 if a.minutes else float("inf")
    n = 0
    write_header = not OUT.exists()

    def flush_cache():
        """
        Atomic-ish cache write, and never fatal.

        Two traps here, both hit in practice:
          * np.savez_compressed APPENDS ".npz" when the path does not already
            end in it. A temp name of "...npz.tmp" therefore becomes
            "...npz.tmp.npz" on disk and the rename fails. The temp name must
            end in .npz.
          * A flush failing must not kill a 50-minute run. It warns, falls back
            to writing in place, and carries on; the next flush retries.
        """
        if not a.save_energy:
            return
        tmp = CACHE.with_name(CACHE.stem + ".tmp.npz")   # ends in .npz on purpose
        try:
            np.savez_compressed(tmp, **energies)
            tmp.replace(CACHE)
        except Exception as e:
            log.warning(f"atomic cache write failed ({type(e).__name__}: {e}); "
                        "writing in place instead")
            try:
                np.savez_compressed(CACHE, **energies)
            except Exception as e2:
                log.error(f"cache write FAILED ({e2}) — continuing; will retry "
                          "at the next flush and again at the end")

    with open(OUT, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        if write_header:
            w.writeheader()
        for _, r in todo.iterrows():
            if time.time() - t0 > budget:
                break
            try:
                fr, lm, _meta = cf.load_clip(r.video_path, r.landmark_path,
                                             n_frames=a.frames, img_size=256)
                if lm is None:
                    continue
                res = lmf.analyse_clip(fr, lm, tpl, size=256)
                row = {"clip_name": r.clip_name, "label": int(r.label), "split": r.split,
                       "subset": r.subset, "group": r.group, "n_frames_used": len(fr)}
                row.update({k: float(v) for k, v in res.descriptors.items()})
                # A clip can need recomputation for the CACHE while its CSV row
                # already exists (see guard 2). Don't duplicate the row.
                if r.clip_name not in done_csv:
                    w.writerow(row)
                if a.save_energy:
                    energies[f"E::{r.clip_name}"] = res.energy.astype("float32")
                    energies[f"D::{r.clip_name}"] = np.array(
                        [res.descriptors[k] for k in lmf.DESCRIPTOR_NAMES], dtype="float32")
                n += 1
            except Exception as e:
                log.warning(f"{r.clip_name}: {type(e).__name__} {e}")
                continue

            if n % 100 == 0:
                el = time.time() - t0
                rate = n / el
                eta = (len(todo) - n) / max(rate, 1e-9)
                print(f"  {n:>5}/{len(todo)}  {rate:4.1f} clips/s  "
                      f"elapsed {el/60:5.1f}m  eta {eta/60:5.1f}m", flush=True)
            if n % a.flush_every == 0:
                fh.flush(); flush_cache()

    fh_done = len(done) + n
    flush_cache()
    el = time.time() - t0
    print(f"\nprocessed {n} clips in {el/60:.1f}m ({el/max(n,1):.3f}s/clip)")
    print(f"total done {fh_done}/{len(df)}   csv -> {OUT.relative_to(ROOT)}")
    if a.save_energy:
        print(f"cache -> {CACHE.relative_to(ROOT)}  ({len(energies)//2} clips)")


if __name__ == "__main__":
    main()
