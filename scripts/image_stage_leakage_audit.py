"""
Did the YEAR-ONE image split leak?

The FSV-Net README asserted that 17.7% of the image stage's test rows had a
near-duplicate in train, and that its honest held-out AUC was 0.5702 against
0.7998 on the random split. NONE of that is substantiated by any file in this
repository -- so it is measured here rather than repeated.

Method: 64-bit dHash per image, then for every TEST image find the smallest
Hamming distance to any TRAIN image. Frames sampled from the same source video
are near-identical, so a small distance means the model saw that content in
training. Filenames in this dataset carry no identity token, so a perceptual
hash is the only handle available after the fact -- which is itself the
lesson: record identity at collection time or you cannot audit later.

    python scripts/image_stage_leakage_audit.py --hash     # resumable, ~2 min
    python scripts/image_stage_leakage_audit.py --report
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
IMG = ROOT / "image_stage" / "data" / "processed"
OUT = ROOT / "results" / "baseline"
CACHE = OUT / "image_stage_hashes.npz"
OUT.mkdir(parents=True, exist_ok=True)
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def dhash(path, size=8):
    """64-bit difference hash. Robust to recompression and small rescales."""
    from PIL import Image
    with Image.open(path) as im:
        im = im.convert("L").resize((size + 1, size), Image.BILINEAR)
        a = np.asarray(im, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    return np.packbits(bits).view(np.uint64)[0]


def collect():
    rows = []
    for split in ["train", "val", "test"]:
        d = IMG / split
        if not d.exists():
            continue
        for p in sorted(d.rglob("*")):
            if p.suffix.lower() in EXTS:
                rows.append((split, p.parent.name, str(p)))
    return rows


def do_hash(minutes):
    rows = collect()
    print(f"{len(rows)} images found")
    have = {}
    if CACHE.exists():
        z = np.load(CACHE, allow_pickle=True)
        have = dict(zip(z["paths"].tolist(), z["hashes"].tolist()))
        print(f"resuming with {len(have)} hashed")
    todo = [r for r in rows if r[2] not in have]
    if not todo:
        print("ALL HASHED"); return
    t0, n = time.time(), 0
    budget = minutes * 60 if minutes else float("inf")
    for split, cls, p in todo:
        if time.time() - t0 > budget:
            break
        try:
            have[p] = int(dhash(p)); n += 1
        except Exception as e:
            print(f"  skip {Path(p).name}: {type(e).__name__}")
        if n % 2000 == 0 and n:
            el = time.time() - t0
            print(f"  {n}/{len(todo)}  {n/el:.0f} img/s  "
                  f"eta {(len(todo)-n)/(n/el)/60:.1f}m", flush=True)
    paths = np.array(list(have.keys()))
    hashes = np.array(list(have.values()), dtype=np.uint64)
    np.savez_compressed(CACHE, paths=paths, hashes=hashes)
    print(f"hashed {n} this run; {len(have)}/{len(rows)} total -> {CACHE.name}")


_POP = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(1).astype(np.uint8)


def hamming_min(test_h, train_h, chunk=256):
    """For each test hash, the minimum Hamming distance to any train hash."""
    out = np.full(len(test_h), 64, dtype=np.uint8)
    tr = train_h.view(np.uint8).reshape(-1, 8)
    for i in range(0, len(test_h), chunk):
        blk = test_h[i:i + chunk].view(np.uint8).reshape(-1, 8)
        d = _POP[np.bitwise_xor(blk[:, None, :], tr[None, :, :])].sum(axis=2)
        out[i:i + chunk] = d.min(axis=1)
    return out


def report():
    if not CACHE.exists():
        sys.exit("run --hash first")
    z = np.load(CACHE, allow_pickle=True)
    paths = z["paths"]; hashes = z["hashes"].astype(np.uint64)
    split = np.array([Path(p).parts[-3] for p in paths])
    cls = np.array([Path(p).parent.name for p in paths])
    tr = hashes[split == "train"]; te = hashes[split == "test"]
    te_cls = cls[split == "test"]
    print(f"train {len(tr)}  test {len(te)}")

    t0 = time.time()
    dmin = hamming_min(te, tr)
    print(f"nearest-neighbour search: {time.time()-t0:.1f}s\n")

    res = {"n_train": int(len(tr)), "n_test": int(len(te)), "hash": "dHash-64"}
    print(f"{'threshold':>10}  {'meaning':<26}{'test imgs w/ a train match':>28}")
    print("-" * 68)
    for thr, label in [(0, "byte-identical hash"), (2, "visually identical"),
                       (5, "near-duplicate"), (10, "same scene/shot")]:
        n = int((dmin <= thr).sum())
        pct = 100 * n / len(te)
        res[f"contamination_at_hamming_{thr}"] = round(pct / 100, 4)
        print(f"{thr:>10}  {label:<26}{n:>8}  ({pct:5.1f}%)")

    print(f"\n{'class':<16}{'n':>7}{'<=5 (near-dup)':>18}")
    print("-" * 42)
    by = {}
    for c in sorted(set(te_cls)):
        m = te_cls == c
        pct = 100 * float((dmin[m] <= 5).mean())
        by[c] = round(pct / 100, 4)
        print(f"{c:<16}{int(m.sum()):>7}{pct:>17.1f}%")
    res["by_class_at_hamming_5"] = by
    res["median_min_distance"] = float(np.median(dmin))
    json.dump(res, open(OUT / "image_stage_leakage.json", "w"), indent=2)
    print(f"\n-> results/baseline/image_stage_leakage.json")
    print("""
INTERPRETATION
  A test image with a near-duplicate in train is not a held-out sample. The
  year-one test accuracy was computed over this test set, so whatever fraction
  is contaminated is the fraction of that number which measured memorisation
  rather than detection.""")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hash", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--minutes", type=float, default=0.6)
    a = ap.parse_args()
    if a.hash: do_hash(a.minutes)
    if a.report: report()
