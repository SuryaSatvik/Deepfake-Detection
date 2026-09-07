"""
video_io.py
===========

Robust video decoding and temporal frame sampling.

Two things matter here for the forensics, and both are easy to get wrong:

1. **Temporal spacing must be uniform and known.** The TSC branch takes an FFT
   *along the time axis*, so an irregular or unknown frame spacing turns the
   temporal spectrum into mush. Every sampler here returns a constant stride
   and records it, so the temporal frequency axis has real units (cycles per
   frame at that stride).

2. **Never re-encode before feature extraction.** Decoding to RGB is fine;
   writing the frames back out as JPEG at a different quality than the source
   would erase or forge exactly the compression traces the SRM branch reads.
   The cache in preprocess.py stores PNG (lossless) for that reason.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from .utils import log

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


# --------------------------------------------------------------------------

@dataclass
class VideoMeta:
    path: str
    n_frames: int
    fps: float
    width: int
    height: int
    readable: bool
    error: str = ""

    @property
    def duration(self) -> float:
        return self.n_frames / self.fps if self.fps > 0 else 0.0


def _require_cv2() -> None:
    if cv2 is None:
        raise ImportError(
            "opencv-python is required for video decoding.\n"
            "    pip install opencv-python\n"
            "(on a headless server use opencv-python-headless instead)"
        )


def probe(path) -> VideoMeta:
    """Read a video's metadata without decoding the whole file."""
    _require_cv2()
    path = str(path)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return VideoMeta(path, 0, 0.0, 0, 0, False, "cv2 could not open the file")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    # CAP_PROP_FRAME_COUNT lies on some containers; treat <=0 as unknown
    return VideoMeta(path, max(n, 0), fps, w, h, n > 0 and w > 0,
                     "" if n > 0 else "frame count unavailable")


# --------------------------------------------------------------------------
# Frame index planning
# --------------------------------------------------------------------------

def plan_indices(n_frames: int, num_frames: int, stride: int,
                 mode: str = "center", rng: Optional[np.random.Generator] = None,
                 clip_id: int = 0, n_clips: int = 1) -> List[int]:
    """
    Choose which frame indices form one clip.

    mode:
      "center" — one window centred in the video (deterministic)
      "random" — random window start (training augmentation)
      "grid"   — clip_id-th of n_clips evenly spaced windows (evaluation)

    The returned indices are always `stride` apart, which is what keeps the
    temporal FFT interpretable.
    """
    span = (num_frames - 1) * stride + 1

    if n_frames <= 0:
        # unknown length: assume we can read sequentially and hope for the best
        return [i * stride for i in range(num_frames)]

    if n_frames < span:
        # Video is shorter than the requested window. Shrink the stride
        # rather than the clip, so the network always sees num_frames.
        stride = max(1, (n_frames - 1) // max(1, num_frames - 1))
        span = (num_frames - 1) * stride + 1

    max_start = max(0, n_frames - span)

    if mode == "random":
        rng = rng or np.random.default_rng()
        start = int(rng.integers(0, max_start + 1))
    elif mode == "grid" and n_clips > 1:
        start = int(round(clip_id * max_start / (n_clips - 1)))
    else:
        start = max_start // 2

    idx = [min(start + i * stride, max(0, n_frames - 1)) for i in range(num_frames)]
    return idx


# --------------------------------------------------------------------------
# Decoding
# --------------------------------------------------------------------------

def read_frames(path, indices: List[int]) -> Tuple[np.ndarray, List[int]]:
    """
    Decode the requested frame indices as RGB uint8.

    Returns (frames, actual_indices). Sequential reading with a skip counter is
    used rather than CAP_PROP_POS_FRAMES seeking, because seeking on
    long-GOP H.264 is both slow and inaccurate — and an inaccurate frame index
    silently corrupts the temporal axis.
    """
    _require_cv2()
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return np.empty((0,), dtype=np.uint8), []

    wanted = sorted(set(int(i) for i in indices))
    want_set = set(wanted)
    last = wanted[-1]

    grabbed: dict = {}
    pos = 0
    while pos <= last:
        ok = cap.grab()
        if not ok:
            break
        if pos in want_set:
            ok, frame = cap.retrieve()
            if ok and frame is not None:
                grabbed[pos] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pos += 1
    cap.release()

    if not grabbed:
        return np.empty((0,), dtype=np.uint8), []

    # Rebuild in the caller's requested order, repeating the nearest available
    # frame if a decode failed (rare, but a corrupt frame shouldn't kill a clip)
    keys = sorted(grabbed)
    out, actual = [], []
    for i in indices:
        k = min(keys, key=lambda kk: abs(kk - i))
        out.append(grabbed[k])
        actual.append(k)
    return np.stack(out), actual


def read_clip(path, num_frames: int, stride: int, mode: str = "center",
              rng: Optional[np.random.Generator] = None,
              clip_id: int = 0, n_clips: int = 1
              ) -> Tuple[np.ndarray, dict]:
    """
    Convenience: probe → plan → decode. Returns (T,H,W,3) uint8 and an info
    dict carrying the true temporal stride (needed by the TSC branch to label
    its frequency axis).
    """
    meta = probe(path)
    idx = plan_indices(meta.n_frames, num_frames, stride, mode, rng, clip_id, n_clips)
    frames, actual = read_frames(path, idx)
    eff_stride = float(np.mean(np.diff(actual))) if len(actual) > 1 else float(stride)
    info = {
        "path": str(path),
        "requested_indices": idx,
        "actual_indices": actual,
        "effective_stride": eff_stride,
        "fps": meta.fps,
        "sample_hz": (meta.fps / eff_stride) if eff_stride > 0 else 0.0,
        "n_decoded": int(len(frames)),
        "video_frames": meta.n_frames,
        "readable": bool(len(frames) > 0),
    }
    return frames, info


def iter_videos(root, exts=None) -> List[Path]:
    from .datasets_fetch import VIDEO_EXTS
    exts = exts or VIDEO_EXTS
    root = Path(root)
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in exts)


__all__ = ["VideoMeta", "probe", "plan_indices", "read_frames", "read_clip",
           "iter_videos"]
