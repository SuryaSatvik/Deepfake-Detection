"""
celebdf_frames.py
=================

Adapter for the *pre-extracted* Celeb-DF-v2 distribution that this project
actually has on disk.

WHAT IS ON DISK (measured, not assumed)
---------------------------------------
    Celeb-DF-v2/
      Celeb-real/frames/id0_0000/000.png 015.png ... (~32 frames, 256x256 RGB)
      Celeb-real/landmarks/id0_0000/000.npy ...      (81, 2) uint32
      Celeb-synthesis/frames/id0_id16_0000/...       5639 clips
      YouTube-real/frames/00000/...                  300 clips
      List_of_testing_videos.txt                     official test list

    588 Celeb-real + 5639 Celeb-synthesis + 300 YouTube-real = 6527 clips.

CONSEQUENCES FOR THE PIPELINE — read these before designing anything
--------------------------------------------------------------------
1. There is NO raw video. So: no audio, no codec/bitrate metadata, no
   arbitrary temporal resampling. The temporal grid is fixed at every
   15th source frame (~0.5 s at 30 fps). `video_io.py` and `faces.py` are
   therefore NOT used on this dataset — the crops are already made.
2. The frames are ALREADY face crops at 256x256. Face detection is a no-op.
   `dataset.py`'s filmstrip cache is redundant here; this module reads the
   PNGs directly, which is why `FramesClipDataset` exists.
3. The landmarks are the resource nothing else in this project uses. They
   make `lmf.py` (landmark motion-compensated forensics) possible. That is
   the reason the proposed model looks the way it does.
4. Frame timestamps are RECONSTRUCTED as index * 15 / assumed_fps. The true
   fps is not recoverable from PNGs, so every timestamp this pipeline emits
   is labelled `approx` and `assumed_fps` is carried in the manifest. Do not
   report these as measured.

The identity semantics are unchanged from the video distribution, so the
union-find grouping in `manifest.py` applies verbatim:
    id0_0000        -> identity {celeb_id0}
    id0_id16_0000   -> identities {celeb_id0, celeb_id16}   (binds both)
    YouTube-real/00000 -> its own singleton identity
"""

from __future__ import annotations

import re
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .utils import log

# Nominal stride only -- carried in the manifest for reference. It is NOT
# constant across clips (see the module docstring); nothing in the timestamp
# path depends on it, because filenames are true source frame indices.
DEFAULT_SOURCE_STRIDE = 15
ASSUMED_FPS = 30.0

_CELEB_REAL = re.compile(r"^id(\d+)_(\d+)$")
_CELEB_FAKE = re.compile(r"^id(\d+)_id(\d+)_(\d+)$")

SUBSETS = [
    # (folder,          label, method,          dataset tag)
    ("Celeb-real",      0, "celeb_real",     "celebdf"),
    ("YouTube-real",    0, "youtube_real",   "celebdf"),
    ("Celeb-synthesis", 1, "celebdf_swap",   "celebdf"),
]


# --------------------------------------------------------------------------
# Layout probing
# --------------------------------------------------------------------------

def probe_layout(root: Path, sample: int = 40) -> dict:
    """
    Look at the data instead of trusting this docstring. Returns measured
    counts, frame-count distribution, resolution and the inferred stride.
    """
    root = Path(root)
    out: dict = {"root": str(root), "subsets": {}, "issues": []}
    strides = set()
    sizes = set()
    for sub, label, method, _ds in SUBSETS:
        fdir = root / sub / "frames"
        ldir = root / sub / "landmarks"
        if not fdir.exists():
            out["issues"].append(f"missing {fdir}")
            continue
        clips = sorted(p for p in fdir.iterdir() if p.is_dir())
        n_frames = []
        for c in clips[:sample]:
            pngs = sorted(c.glob("*.png"))
            n_frames.append(len(pngs))
            idx = [int(p.stem) for p in pngs]
            if len(idx) > 1:
                d = np.diff(idx)
                strides.update(set(d.tolist()))
            if pngs:
                from PIL import Image
                with Image.open(pngs[0]) as im:
                    sizes.add(im.size)
        out["subsets"][sub] = {
            "n_clips": len(clips),
            "label": label,
            "method": method,
            "has_landmarks": ldir.exists(),
            "n_landmark_dirs": len(list(ldir.iterdir())) if ldir.exists() else 0,
            "frames_per_clip_sampled": {
                "min": int(min(n_frames)) if n_frames else 0,
                "max": int(max(n_frames)) if n_frames else 0,
                "mean": round(float(np.mean(n_frames)), 2) if n_frames else 0.0,
            },
        }
    out["inferred_source_strides"] = sorted(int(s) for s in strides)
    out["resolutions"] = sorted(str(s) for s in sizes)
    if len(strides) > 1:
        out["issues"].append(
            f"inconsistent frame stride across clips: {sorted(strides)} — "
            "timestamps will be less reliable")
    if len(sizes) > 1:
        out["issues"].append(f"mixed resolutions: {sorted(sizes)}")
    return out


# --------------------------------------------------------------------------
# Manifest adapter  (registered into manifest.ADAPTERS as "celebdf_frames")
# --------------------------------------------------------------------------

def rows_celebdf_frames(root: Path,
                        min_frames: int = 4,
                        filelist_dir: Optional[Path] = None,
                        source_stride: int = DEFAULT_SOURCE_STRIDE,
                        assumed_fps: float = ASSUMED_FPS) -> List[dict]:
    """
    One row per CLIP DIRECTORY. `video_path` holds the frames directory, which
    keeps every downstream module (manifest, splits, evaluate, evidence)
    working unchanged — it only ever treats that column as an opaque key.
    """
    root = Path(root)
    rows: List[dict] = []
    skipped = 0
    for sub, label, method, ds in SUBSETS:
        fdir = root / sub / "frames"
        ldir = root / sub / "landmarks"
        if not fdir.exists():
            continue
        # Per-directory globbing over ~6.5k clip dirs on a mounted volume takes
        # minutes; one `find`-style walk takes seconds. `filelist_dir` lets the
        # caller supply a cached listing (one absolute PNG path per line) so the
        # scan is done once and reused. Falls back to walking if absent.
        listing = None
        if filelist_dir is not None:
            cand = Path(filelist_dir) / f"{sub}.txt"
            if cand.exists():
                listing = cand
        clips: Dict[str, List[int]] = {}
        if listing is not None:
            for line in listing.read_text().splitlines():
                line = line.strip()
                if not line.endswith(".png"):
                    continue
                q = Path(line)
                clips.setdefault(q.parent.name, []).append(int(q.stem))
            clip_items = sorted(clips.items())
        else:
            clip_items = [(c.name, sorted(int(x.stem) for x in c.glob("*.png")))
                          for c in sorted(q for q in fdir.iterdir() if q.is_dir())]
        for stem, idx in clip_items:
            idx = sorted(idx)
            if len(idx) < min_frames:
                skipped += 1
                continue
            cdir = fdir / stem
            ids: List[str]
            m = _CELEB_FAKE.match(stem)
            if m:
                ids = [f"celeb_id{m.group(1)}", f"celeb_id{m.group(2)}"]
                source_video = f"id{m.group(1)}_{m.group(3)}"
            else:
                m2 = _CELEB_REAL.match(stem)
                if m2:
                    ids = [f"celeb_id{m2.group(1)}"]
                    source_video = stem
                else:
                    ids = [f"celeb_yt_{stem}"]
                    source_video = stem
            lm = ldir / stem
            rows.append(dict(
                video_path=str(cdir),
                landmark_path=str(lm) if lm.exists() else "",
                label=label,
                label_name="fake" if label else "real",
                identities=ids,
                dataset=ds,
                method=method,
                subset=sub,
                clip_name=stem,
                source_video=source_video,
                frame_count=len(idx),
                first_source_index=idx[0],
                last_source_index=idx[-1],
                source_stride=source_stride,
                assumed_fps=assumed_fps,
                duration_approx_s=round((idx[-1] - idx[0]) / assumed_fps, 3),
                resolution="256x256",
                codec="n/a (pre-extracted PNG)",
            ))
    if skipped:
        log.warning(f"celebdf_frames: skipped {skipped} clips with < {min_frames} frames")
    log.info(f"celebdf_frames: {len(rows)} clips from {root}")
    return rows


def official_test_names(list_path: Path) -> set:
    """Clip stems named in Celeb-DF's List_of_testing_videos.txt."""
    names = set()
    for line in Path(list_path).read_text().splitlines():
        parts = line.strip().split()
        if parts:
            names.add(Path(parts[-1]).stem)
    return names


def apply_official_test_split(df: pd.DataFrame, list_path: Path,
                              move_whole_group: bool = True) -> pd.DataFrame:
    """
    Force the official Celeb-DF test clips into `test`.

    `move_whole_group=True` also moves every clip sharing an identity group
    with a listed clip. That is stricter than the published protocol and will
    give a LOWER number than papers quote — say so when you report it. Set it
    False only to reproduce the published protocol exactly, and then label the
    number as "official protocol (identity-leaky)".
    """
    names = official_test_names(list_path)
    out = df.copy()
    mask = out["clip_name"].isin(names)
    n_listed = int(mask.sum())
    if not n_listed:
        log.warning("official test list matched 0 clips — check clip_name format")
        return out
    if move_whole_group:
        groups = set(out.loc[mask, "group"])
        moved = out["group"].isin(groups)
        out.loc[moved, "split"] = "test"
        log.info(f"official test list: {n_listed} listed clips -> "
                 f"{int(moved.sum())} clips moved to test (whole identity groups)")
    else:
        out.loc[mask, "split"] = "test"
        log.info(f"official test list: {n_listed} clips moved to test (listed only)")
    return out


# --------------------------------------------------------------------------
# Reading a clip
# --------------------------------------------------------------------------

# The package lives at <root>/src/fsvnet/celebdf_frames.py
_PKG_ROOT = Path(__file__).resolve().parents[2]


def _anchor(path) -> Path:
    """
    Resolve a clip path regardless of the caller's working directory.

    Manifest paths are stored relative to the project root, but a notebook
    runs with cwd=notebooks/ and a script with cwd=<root>. Rather than making
    every caller remember to resolve, a relative path that does not exist as
    given is retried against the project root. This makes the loader immune to
    cwd, which is the single most common source of "file not found" here.
    """
    q = Path(path)
    if q.is_absolute() or q.exists():
        return q
    cand = _PKG_ROOT / q
    return cand if cand.exists() else q


def sample_indices(n_available: int, n_frames: int, mode: str = "uniform",
                   rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    Uniform temporal sampling that preserves ORDER and is deterministic for
    eval. `mode="jitter"` (train) keeps the same uniform grid but jitters each
    pick inside its own bin, so the model sees temporal variety without ever
    reordering frames.
    """
    if n_available <= 0:
        raise ValueError("no frames")
    if n_available <= n_frames:
        idx = np.arange(n_available)
        if n_available < n_frames:                      # pad by repeating last
            idx = np.concatenate([idx, np.full(n_frames - n_available, n_available - 1)])
        return idx
    edges = np.linspace(0, n_available, n_frames + 1)
    if mode == "jitter" and rng is not None:
        lo, hi = edges[:-1], edges[1:]
        picks = lo + rng.random(n_frames) * (hi - lo)
    else:
        picks = (edges[:-1] + edges[1:]) / 2.0
    return np.clip(picks.astype(int), 0, n_available - 1)


def load_clip(frames_dir, landmark_dir=None, n_frames: int = 16,
              img_size: int = 256, mode: str = "uniform",
              rng: Optional[np.random.Generator] = None,
              source_stride: int = DEFAULT_SOURCE_STRIDE,
              assumed_fps: float = ASSUMED_FPS
              ) -> Tuple[np.ndarray, Optional[np.ndarray], dict]:
    """
    Returns
        frames    uint8 (T, H, W, 3)
        landmarks float32 (T, 81, 2) in pixel coords, or None
        meta      dict with the *source* frame index and approx timestamp of
                  every returned frame, so any prediction can be traced back to
                  video -> frame -> timestamp.
    """
    from PIL import Image
    frames_dir = _anchor(frames_dir)
    pngs = sorted(frames_dir.glob("*.png"))
    if not pngs:
        # The most common cause is a RELATIVE path reaching here. Manifest
        # paths are stored relative to the project root and must be resolved
        # by `load_manifest` / `resolve_clip_paths` first. A caller that did
        # `pd.read_csv(manifest)` directly gets relative paths and lands here.
        hint = ""
        if not frames_dir.is_absolute():
            hint = ("\n  This path is RELATIVE. Manifest paths are stored "
                    "relative to the project root.\n"
                    "  Load the manifest with:\n"
                    "      df = celebdf_frames.load_manifest(MAN, PROJECT_ROOT)\n"
                    "  rather than pd.read_csv(MAN). If you are in a notebook "
                    "that already does this,\n  you are running a STALE copy: "
                    "reload it from disk and restart the kernel.")
        elif not frames_dir.exists():
            hint = ("\n  The directory does not exist. The dataset may have "
                    "moved; rebuild the manifest with\n"
                    "      python scripts/build_celebdf_manifest.py")
        raise FileNotFoundError(f"no PNG frames in {frames_dir}{hint}")
    sel = sample_indices(len(pngs), n_frames, mode, rng)

    imgs, lms = [], []
    have_lm = landmark_dir not in (None, "", float("nan"))
    ldir = _anchor(landmark_dir) if have_lm else None
    if ldir is not None and not ldir.exists():
        ldir, have_lm = None, False

    for i in sel:
        p = pngs[int(i)]
        with Image.open(p) as im:
            im = im.convert("RGB")
            if im.size != (img_size, img_size):
                im = im.resize((img_size, img_size), Image.BILINEAR)
                scale = img_size / 256.0
            else:
                scale = 1.0
            imgs.append(np.asarray(im, dtype=np.uint8))
        if have_lm:
            lp = ldir / f"{p.stem}.npy"
            if lp.exists():
                lms.append(np.asarray(np.load(lp), dtype=np.float32) * scale)
            else:
                have_lm = False

    frames = np.stack(imgs)
    landmarks = np.stack(lms) if have_lm and len(lms) == len(imgs) else None
    src_idx = [int(pngs[int(i)].stem) for i in sel]
    meta = {
        "clip": frames_dir.name,
        "frames_dir": str(frames_dir),
        "n_available": len(pngs),
        "selected_positions": [int(i) for i in sel],
        "source_frame_index": src_idx,
        "timestamp_s_approx": [round(s / assumed_fps, 3) for s in src_idx],
        "assumed_fps": assumed_fps,
        "source_stride": source_stride,
        "timestamps_are_approximate": True,
        "has_landmarks": landmarks is not None,
    }
    return frames, landmarks, meta


__all__ = ["probe_layout", "rows_celebdf_frames", "official_test_names",
           "apply_official_test_split", "sample_indices", "load_clip",
           "SUBSETS", "DEFAULT_SOURCE_STRIDE", "ASSUMED_FPS"]


# --------------------------------------------------------------------------
# Portable paths
# --------------------------------------------------------------------------
# The manifest used to store absolute paths from whichever machine built it.
# That works exactly once: rebuild on Windows and the CSV is full of
# "C:\..."; read it from anywhere else and every path is dead. Worse, it
# breaks silently if the project folder is ever moved or renamed.
#
# Paths are therefore stored RELATIVE to the project root, and resolved on
# read. `resolve_clip_paths` also accepts the old absolute form so existing
# manifests keep working.

_PATH_COLS = ("video_path", "landmark_path")


def to_relative(df: "pd.DataFrame", root) -> "pd.DataFrame":
    """Rewrite the path columns as POSIX paths relative to `root`."""
    root = Path(root).resolve()
    out = df.copy()
    for c in _PATH_COLS:
        if c not in out.columns:
            continue
        def _rel(v):
            if not isinstance(v, str) or not v:
                return v
            q = PureWindowsPath(v) if ("\\" in v and "/" not in v) else PurePosixPath(v)
            parts = list(q.parts)
            # keep everything from the dataset directory onwards
            for anchor in ("Celeb-DF-v2",):
                if anchor in parts:
                    return "/".join(parts[parts.index(anchor):])
            try:
                return Path(v).resolve().relative_to(root).as_posix()
            except Exception:
                return v
        out[c] = out[c].map(_rel)
    return out


def _is_abs(v: str) -> bool:
    return v.startswith("/") or (len(v) > 1 and v[1] == ":")


def resolve_clip_paths(df: "pd.DataFrame", root) -> "pd.DataFrame":
    """
    Turn the stored (relative or legacy-absolute) paths into absolute paths
    valid on THIS machine.

    Vectorised on purpose: a per-row Path.resolve() over a mounted volume
    costs a filesystem call each and takes minutes on 6.5k rows. String
    joining costs nothing, and correctness is checked once at the end rather
    than 6,526 times.
    """
    root = str(Path(root).resolve())
    out = df.copy()
    for c in _PATH_COLS:
        if c not in out.columns:
            continue
        col = out[c].fillna("").astype(str)
        rel = ~col.map(_is_abs) & col.ne("")
        out[c] = col.where(~rel, root + os.sep + col.str.replace("/", os.sep, regex=False))
    probe = out.loc[out["video_path"].astype(bool), "video_path"]
    if len(probe) and not Path(probe.iloc[0]).exists():
        raise FileNotFoundError(
            f"Manifest paths do not resolve under {root}.\n"
            f"  tried: {probe.iloc[0]}\n"
            "  Either the dataset moved, or the manifest was built elsewhere.\n"
            "  Rebuild it with:  python scripts/build_celebdf_manifest.py")
    return out


def load_manifest(path, root) -> "pd.DataFrame":
    """Read a manifest CSV and resolve its paths for this machine."""
    df = pd.read_csv(path, dtype={"clip_name": str})
    return resolve_clip_paths(df, root)


__all__ = __all__ + ["to_relative", "resolve_clip_paths", "load_manifest"]
