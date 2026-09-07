"""
dataset.py
==========

Two stages:

1. `preprocess_dataset()` — decode each video once, detect + stabilise the face
   box, crop, and write a **lossless PNG filmstrip** (all frames tiled
   vertically in one file) plus a small JSON sidecar. Run once; takes the
   dataset from "hours of decoding per epoch" to "milliseconds per clip".

2. `ClipDataset` — a torch Dataset that reads a filmstrip, samples a temporal
   window from it, applies the anti-shortcut augmentations, and returns a
   normalised clip tensor.

Why PNG and not JPEG for the cache
-----------------------------------
The SRM and frequency branches read exactly the high-frequency residual that
JPEG quantisation destroys. Caching to JPEG would (a) erase part of the signal
and (b) stamp *every* clip with the same new compression history, which sounds
harmless but is worse: it would hide a real difference between the classes and
make the model look more robust than it is. PNG costs ~1.5 MB per clip and
keeps the forensics honest.

Why a longer strip than the training window
--------------------------------------------
The cache stores `cache_frames` (default 2xT, min 32) frames. Training samples
a random T-frame window inside it each epoch, so there is still temporal
augmentation without re-decoding the video.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .faces import build_detector, crop_clip
from .utils import log, save_json
from .video_io import read_clip

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ==========================================================================
# Stage 1 — preprocessing / caching
# ==========================================================================

def _strip_path(cache_root: Path, video_path: str, dataset: str) -> Path:
    """Stable, collision-free cache filename derived from the full path."""
    import hashlib
    h = hashlib.md5(str(video_path).encode()).hexdigest()[:16]
    stem = Path(video_path).stem[:40]
    return cache_root / dataset / f"{stem}_{h}.png"


def save_filmstrip(frames: np.ndarray, path: Path) -> None:
    """frames (T,S,S,3) uint8 -> one tall lossless PNG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    strip = np.concatenate(list(frames), axis=0)               # (T*S, S, 3)
    ok = cv2.imwrite(str(path), cv2.cvtColor(strip, cv2.COLOR_RGB2BGR),
                     [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        raise IOError(f"could not write {path}")


def load_filmstrip(path: Path, size: int) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"could not read {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    T = img.shape[0] // size
    return img[: T * size].reshape(T, size, size, 3)


def preprocess_dataset(df: pd.DataFrame, cfg, overwrite: bool = False,
                       limit: Optional[int] = None) -> pd.DataFrame:
    """
    Decode + crop + cache every video in the manifest.

    Adds columns: cache_path, n_cached_frames, face_ratio, fps, sample_hz, ok.
    Rows that fail (unreadable file, no face at all) are kept with ok=False so
    the failure is visible in the stats rather than silently dropped.
    """
    if cv2 is None:
        raise ImportError("opencv-python is required for preprocessing")

    cache_root = Path(cfg.paths.clip_cache)
    cache_root.mkdir(parents=True, exist_ok=True)
    detector = build_detector(cfg.data.face_backend)

    cache_frames = max(cfg.data.num_frames * 2, 32)
    rows = df if limit is None else df.iloc[:limit]
    out: List[dict] = []

    try:
        from tqdm.auto import tqdm
        it = tqdm(rows.itertuples(index=False), total=len(rows), desc="preprocess")
    except ImportError:
        it = rows.itertuples(index=False)

    n_fail = 0
    for r in it:
        rec = dict(r._asdict())
        dst = _strip_path(cache_root, rec["video_path"], str(rec.get("dataset", "x")))
        meta_path = dst.with_suffix(".json")

        if dst.exists() and meta_path.exists() and not overwrite:
            meta = json.loads(meta_path.read_text())
            rec.update(meta)
            rec["cache_path"] = str(dst)
            out.append(rec)
            continue

        try:
            frames, info = read_clip(rec["video_path"], cache_frames,
                                     cfg.data.frame_stride, mode="center")
            if len(frames) == 0:
                raise RuntimeError("no frames decoded")
            crop = crop_clip(frames, detector, cfg.data.img_size,
                             cfg.data.face_margin, cfg.data.detect_every)
            save_filmstrip(crop.frames, dst)
            meta = {
                "n_cached_frames": int(len(crop.frames)),
                "face_ratio": round(float(crop.face_ratio), 4),
                "fps": round(float(info["fps"]), 3),
                "sample_hz": round(float(info["sample_hz"]), 4),
                "effective_stride": round(float(info["effective_stride"]), 3),
                "face_backend": crop.backend,
                "ok": True,
            }
            meta_path.write_text(json.dumps(meta))
            rec.update(meta)
            rec["cache_path"] = str(dst)
        except Exception as e:
            n_fail += 1
            rec.update({"cache_path": "", "n_cached_frames": 0, "face_ratio": 0.0,
                        "fps": 0.0, "sample_hz": 0.0, "ok": False,
                        "error": f"{e.__class__.__name__}: {e}"})
        out.append(rec)

    res = pd.DataFrame(out)
    n_ok = int(res["ok"].sum()) if "ok" in res else 0
    log.info(f"preprocessed {n_ok}/{len(res)} videos ({n_fail} failed)")
    if "face_ratio" in res and n_ok:
        lo = float((res.loc[res.ok, "face_ratio"] < cfg.data.min_faces_per_clip).mean())
        log.info(f"  {lo*100:.1f}% of successful videos are below the "
                 f"min_faces_per_clip={cfg.data.min_faces_per_clip} threshold")
    return res


# ==========================================================================
# Augmentation — the anti-shortcut kit
# ==========================================================================

def _jpeg_roundtrip(frames: np.ndarray, quality: int) -> np.ndarray:
    """
    Re-encode every frame at the same JPEG quality.

    This is the single most important augmentation in the whole pipeline. If
    reals and fakes in your dataset carry different compression histories — and
    in Celeb-DF and FF++ they demonstrably do — a detector will learn the codec
    instead of the generator, score beautifully in-domain, and collapse
    out-of-domain. Randomising the quality per clip removes that shortcut.
    """
    enc = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    out = np.empty_like(frames)
    for i, f in enumerate(frames):
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(f, cv2.COLOR_RGB2BGR), enc)
        out[i] = cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR),
                              cv2.COLOR_BGR2RGB) if ok else f
    return out


def _rescale(frames: np.ndarray, factor: float) -> np.ndarray:
    if factor >= 0.999:
        return frames
    T, S = frames.shape[0], frames.shape[1]
    small = int(max(32, round(S * factor)))
    out = np.empty_like(frames)
    for i, f in enumerate(frames):
        d = cv2.resize(f, (small, small), interpolation=cv2.INTER_AREA)
        out[i] = cv2.resize(d, (S, S), interpolation=cv2.INTER_CUBIC)
    return out


def augment(frames: np.ndarray, cfg, rng: np.random.Generator) -> np.ndarray:
    d = cfg.data
    if d.aug_hflip and rng.random() < 0.5:
        frames = frames[:, :, ::-1]                            # same flip all frames
    if rng.random() < d.aug_prob:
        q = int(rng.integers(d.aug_jpeg_quality[0], d.aug_jpeg_quality[1] + 1))
        frames = _jpeg_roundtrip(np.ascontiguousarray(frames), q)
    if rng.random() < d.aug_prob:
        f = float(rng.uniform(d.aug_downscale[0], d.aug_downscale[1]))
        frames = _rescale(np.ascontiguousarray(frames), f)
    return np.ascontiguousarray(frames)


def normalise(frames: np.ndarray) -> torch.Tensor:
    """(T,S,S,3) uint8 -> (T,3,S,S) float32 ImageNet-normalised."""
    x = frames.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(x.transpose(0, 3, 1, 2).copy())


# ==========================================================================
# Stage 2 — Dataset
# ==========================================================================

class ClipDataset(Dataset):
    """
    One item = one clip.

    train : random temporal window + augmentation
    eval  : `clips_per_video_eval` evenly spaced windows, no augmentation;
            evaluate.py averages their scores into one video-level prediction.
    """

    def __init__(self, df: pd.DataFrame, cfg, split: str = "train",
                 train_mode: Optional[bool] = None):
        self.cfg = cfg
        self.split = split
        self.train_mode = (split == "train") if train_mode is None else train_mode

        d = df[df.split == split] if "split" in df.columns else df
        if "ok" in d.columns:
            d = d[d.ok.astype(bool)]
        if "face_ratio" in d.columns:
            keep = d.face_ratio >= cfg.data.min_faces_per_clip
            dropped = int((~keep).sum())
            if dropped:
                log.info(f"[{split}] dropping {dropped} videos below "
                         f"min_faces_per_clip")
            d = d[keep]
        self.df = d.reset_index(drop=True)

        n_clips = (cfg.data.clips_per_video_train if self.train_mode
                   else cfg.data.clips_per_video_eval)
        self.n_clips = max(1, n_clips)
        self.index: List[Tuple[int, int]] = [
            (i, c) for i in range(len(self.df)) for c in range(self.n_clips)]

        if len(self.df) == 0:
            log.warning(f"[{split}] dataset is EMPTY after filtering")

    def __len__(self) -> int:
        return len(self.index)

    # ------------------------------------------------------------------
    def _window(self, n_cached: int, clip_id: int, rng) -> List[int]:
        T = self.cfg.data.num_frames
        if n_cached <= T:
            idx = list(range(n_cached)) + [n_cached - 1] * (T - n_cached)
            return idx
        max_start = n_cached - T
        if self.train_mode:
            start = int(rng.integers(0, max_start + 1))
        elif self.n_clips > 1:
            start = int(round(clip_id * max_start / (self.n_clips - 1)))
        else:
            start = max_start // 2
        return list(range(start, start + T))

    # ------------------------------------------------------------------
    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        row_i, clip_id = self.index[i]
        row = self.df.iloc[row_i]

        seed = (self.cfg.data.seed * 1_000_003 + row_i * 31 + clip_id)
        rng = np.random.default_rng(seed if not self.train_mode else None)

        strip = load_filmstrip(Path(row["cache_path"]), self.cfg.data.img_size)
        idx = self._window(len(strip), clip_id, rng)
        frames = strip[idx]

        if self.train_mode:
            frames = augment(frames, self.cfg, rng)

        x = normalise(frames)
        T = x.shape[0]
        # every cached frame is a real crop; the mask exists so short videos
        # padded by repetition do not dominate the attention pooling
        mask = torch.ones(T, dtype=torch.bool)
        n_real = min(T, int(row.get("n_cached_frames", T)))
        if n_real < T:
            mask[n_real:] = False

        return {
            "clip": x,
            "mask": mask,
            "label": torch.tensor(int(row["label"]), dtype=torch.long),
            "video_index": torch.tensor(row_i, dtype=torch.long),
            "clip_id": torch.tensor(clip_id, dtype=torch.long),
        }

    # ------------------------------------------------------------------
    def class_weights(self) -> torch.Tensor:
        counts = self.df["label"].value_counts().reindex([0, 1]).fillna(0).values
        w = counts.sum() / (2.0 * np.maximum(counts, 1))
        return torch.tensor(w, dtype=torch.float32)


def make_loaders(df: pd.DataFrame, cfg) -> Dict[str, DataLoader]:
    """Build train/val/test DataLoaders with sane Windows defaults."""
    loaders: Dict[str, DataLoader] = {}
    # persistent_workers only helps when num_workers>0; on Windows spawning is
    # expensive so it matters more there than on Linux.
    nw = cfg.train.num_workers
    common = dict(num_workers=nw, pin_memory=(cfg.device == "cuda"),
                  persistent_workers=nw > 0, drop_last=False)

    for split in ["train", "val", "test"]:
        ds = ClipDataset(df, cfg, split)
        if len(ds) == 0:
            continue
        loaders[split] = DataLoader(
            ds, batch_size=cfg.train.batch_size, shuffle=(split == "train"),
            **common)
        log.info(f"[{split}] {len(ds.df)} videos -> {len(ds)} clips")
    return loaders


__all__ = ["ClipDataset", "make_loaders", "preprocess_dataset", "augment",
           "normalise", "save_filmstrip", "load_filmstrip",
           "IMAGENET_MEAN", "IMAGENET_STD"]
