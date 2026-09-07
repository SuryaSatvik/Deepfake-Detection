"""
dataset_frames.py
=================

torch Dataset for the pre-extracted Celeb-DF-v2 distribution (see
`celebdf_frames.py` for what is actually on disk).

Why this exists alongside `dataset.py`
--------------------------------------
`dataset.py` decodes video, detects faces, crops, and caches a filmstrip PNG.
None of that applies here: the crops already exist at 256x256 and the
landmarks come with them. Running the old path would re-encode data that is
already in its final form and throw the landmarks away.

What one item contains
----------------------
    clip        (T, 3, S, S) float, ImageNet-normalised
    label       long
    lmf_energy  (T-1, 3) float — cached region-energy series
    lmf_desc    (D,) float     — cached closed-form descriptors
    meta        source frame indices + approximate timestamps, so any
                prediction can be traced back to video -> frame -> timestamp

THE LMF PACK IS CACHED, NOT COMPUTED HERE. Warping every frame every epoch
would make the LMF ablation a compute comparison instead of an architecture
comparison. `scripts/extract_lmf.py --save-energy` writes the cache.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from . import celebdf_frames as cf
from .lmf import DESCRIPTOR_NAMES
from .utils import log

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def normalise(frames: np.ndarray) -> torch.Tensor:
    """(T, H, W, 3) uint8 -> (T, 3, H, W) float, ImageNet-normalised."""
    x = frames.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(x.transpose(0, 3, 1, 2).copy())


def augment(frames: np.ndarray, rng: np.random.Generator,
            jpeg_quality=(40, 95), downscale=(0.5, 1.0), p: float = 0.5,
            hflip: bool = True) -> np.ndarray:
    """
    JPEG + rescale augmentation is MANDATORY, not optional.

    Celeb-DF reals and fakes have been through different processing chains. A
    model that is never shown recompression is free to learn the chain instead
    of the manipulation, and it will then collapse on any other dataset. This
    is the single cheapest defence against that and it costs almost nothing.
    """
    import cv2
    out = frames
    if hflip and rng.random() < 0.5:
        out = out[:, :, ::-1]
    if rng.random() < p:
        q = int(rng.integers(*jpeg_quality))
        enc = [int(cv2.IMWRITE_JPEG_QUALITY), q]
        out = np.stack([cv2.imdecode(cv2.imencode(".jpg", f, enc)[1], 1) for f in out])
    if rng.random() < p:
        f = float(rng.uniform(*downscale))
        h, w = out.shape[1:3]
        small = np.stack([cv2.resize(x, (max(8, int(w * f)), max(8, int(h * f)))) for x in out])
        out = np.stack([cv2.resize(x, (w, h)) for x in small])
    return np.ascontiguousarray(out)


class LMFCache:
    """
    Loads the npz written by `scripts/extract_lmf.py --save-energy`.
    Missing clips return None rather than zeros, so the model can flag
    `lmf_missing` instead of silently training on a zero vector.
    """

    def __init__(self, path, n_frames: int):
        self.n_frames = n_frames
        self.energy: Dict[str, np.ndarray] = {}
        self.desc: Dict[str, np.ndarray] = {}
        path = Path(path)
        if not path.exists():
            log.warning(f"LMF cache not found at {path} — every clip will report "
                        "lmf_missing=True. Run scripts/extract_lmf.py --save-energy")
            return
        z = np.load(path, allow_pickle=False)
        for k in z.files:
            if k.startswith("E::"):
                self.energy[k[3:]] = z[k]
            elif k.startswith("D::"):
                self.desc[k[3:]] = z[k]
        log.info(f"LMF cache: {len(self.energy)} clips from {path}")

    def get(self, clip_name: str):
        e = self.energy.get(clip_name)
        d = self.desc.get(clip_name)
        if e is None or d is None:
            return None
        # The cache was written at a fixed frame count; resample the series to
        # the loader's T-1 by linear interpolation rather than failing, so
        # NUM_FRAMES stays a free knob at train time.
        want = self.n_frames - 1
        if len(e) != want and len(e) >= 2:
            src = np.linspace(0, 1, len(e))
            dst = np.linspace(0, 1, want)
            e = np.stack([np.interp(dst, src, e[:, j]) for j in range(e.shape[1])], axis=1)
        elif len(e) < 2:
            e = np.repeat(e[:1] if len(e) else np.zeros((1, 3), np.float32), want, axis=0)
        return e.astype(np.float32), d.astype(np.float32)


class CelebDFFramesDataset(Dataset):
    def __init__(self, df: pd.DataFrame, split: str, n_frames: int = 16,
                 img_size: int = 224, augment_train: bool = True,
                 lmf_cache: Optional[LMFCache] = None, seed: int = 42):
        self.df = df[df.split == split].reset_index(drop=True)
        self.split = split
        self.n_frames = n_frames
        self.img_size = img_size
        self.augment = augment_train and split == "train"
        self.lmf = lmf_cache
        self.seed = seed
        log.info(f"{split}: {len(self.df)} clips "
                 f"({int((self.df.label == 0).sum())} real / "
                 f"{int((self.df.label == 1).sum())} fake)")

    def __len__(self):
        return len(self.df)

    def class_weights(self) -> torch.Tensor:
        n = np.bincount(self.df.label.values, minlength=2).astype(np.float64)
        w = n.sum() / (2.0 * np.maximum(n, 1))
        return torch.tensor(w, dtype=torch.float32)

    def __getitem__(self, i: int):
        r = self.df.iloc[i]
        rng = np.random.default_rng(self.seed + i if self.split == "train" else i)
        frames, _lm, meta = cf.load_clip(
            r.video_path, None, n_frames=self.n_frames, img_size=self.img_size,
            mode="jitter" if self.split == "train" else "uniform", rng=rng)
        if self.augment:
            frames = augment(frames, rng)
        item = {
            "clip": normalise(frames),
            "label": torch.tensor(int(r.label), dtype=torch.long),
            # Every frame in this distribution is already a face crop, so the
            # validity mask is all-True. It is emitted anyway so this loader is
            # drop-in compatible with the trainer written for dataset.py.
            "mask": torch.ones(self.n_frames, dtype=torch.bool),
            # One clip per video here, so the video index is the row index.
            # The trainer aggregates clip scores per video; with 1:1 that is
            # a no-op, but keeping it means video-level metrics stay correct
            # if clips_per_video is ever raised above 1.
            "video_index": torch.tensor(i, dtype=torch.long),
            "clip_name": str(r.clip_name),
            "source_frame_index": torch.tensor(meta["source_frame_index"]),
            "timestamp_s": torch.tensor(meta["timestamp_s_approx"], dtype=torch.float32),
        }
        pack = self.lmf.get(str(r.clip_name)) if self.lmf is not None else None
        if pack is None:
            item["lmf_energy"] = torch.zeros(self.n_frames - 1, 3)
            item["lmf_desc"] = torch.zeros(len(DESCRIPTOR_NAMES))
            item["lmf_ok"] = torch.tensor(False)
        else:
            e, d = pack
            item["lmf_energy"] = torch.from_numpy(e)
            item["lmf_desc"] = torch.from_numpy(d)
            item["lmf_ok"] = torch.tensor(True)
        return item


def make_frame_loaders(df: pd.DataFrame, cfg, lmf_cache_path=None) -> Dict[str, DataLoader]:
    cache = LMFCache(lmf_cache_path or cfg.paths.lmf_cache, cfg.data.num_frames)
    out = {}
    for split in ["train", "val", "test"]:
        if not (df.split == split).any():
            continue
        ds = CelebDFFramesDataset(df, split, cfg.data.num_frames, cfg.data.img_size,
                                  augment_train=True, lmf_cache=cache, seed=cfg.data.seed)
        out[split] = DataLoader(
            ds, batch_size=cfg.train.batch_size, shuffle=(split == "train"),
            num_workers=cfg.train.num_workers, pin_memory=(cfg.device == "cuda"),
            drop_last=(split == "train"))
    return out


def lmf_pack(batch) -> Dict[str, torch.Tensor]:
    """Pull the LMF inputs out of a collated batch, in the shape model.forward wants."""
    return {"energy": batch["lmf_energy"], "descriptors": batch["lmf_desc"]}


__all__ = ["CelebDFFramesDataset", "LMFCache", "make_frame_loaders", "lmf_pack",
           "normalise", "augment"]
