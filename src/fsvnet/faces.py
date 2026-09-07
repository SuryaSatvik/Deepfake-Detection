"""
faces.py
========

Face localisation and stabilised cropping across a clip.

Why this file exists at all
---------------------------
A face-swap deepfake edits a *face region*, so a detector fed whole frames
spends most of its capacity on background. Cropping is standard. But for a
temporal model the crop has a second, less obvious requirement:

    **the crop box must be temporally smooth.**

If the box jitters by a few pixels per frame, resampling introduces its own
high-frequency flicker — and that flicker lands squarely in the band the TSC
branch measures, manufacturing exactly the signal we claim to detect. That
would be a self-inflicted shortcut, the video equivalent of the JPEG-history
trap. So: detect every `detect_every` frames, then smooth the box trajectory
with a moving average and use one common box size for the whole clip.

Backends, in order of preference (all degrade silently):
  1. mediapipe  — BlazeFace, bundled with the pip package, no download
  2. opencv     — Haar cascade shipped inside opencv-python, no download
  3. none       — centre crop; the pipeline still runs, accuracy suffers
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .utils import log

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


Box = Tuple[int, int, int, int]      # x1, y1, x2, y2


# --------------------------------------------------------------------------
# Backends
# --------------------------------------------------------------------------

class _MediaPipeDetector:
    name = "mediapipe"

    def __init__(self, min_confidence: float = 0.5):
        import mediapipe as mp
        self._mp = mp
        self.det = mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=min_confidence)

    def __call__(self, rgb: np.ndarray) -> Optional[Box]:
        res = self.det.process(rgb)
        if not res.detections:
            return None
        h, w = rgb.shape[:2]
        best = max(res.detections, key=lambda d: d.score[0])
        bb = best.location_data.relative_bounding_box
        x1 = int(bb.xmin * w)
        y1 = int(bb.ymin * h)
        x2 = int((bb.xmin + bb.width) * w)
        y2 = int((bb.ymin + bb.height) * h)
        return (x1, y1, x2, y2)


class _HaarDetector:
    name = "opencv"

    def __init__(self):
        if cv2 is None:
            raise ImportError("opencv-python not installed")
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.casc = cv2.CascadeClassifier(path)
        if self.casc.empty():
            raise RuntimeError(f"could not load Haar cascade from {path}")

    def __call__(self, rgb: np.ndarray) -> Optional[Box]:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        faces = self.casc.detectMultiScale(gray, scaleFactor=1.15,
                                           minNeighbors=5, minSize=(48, 48))
        if len(faces) == 0:
            return None
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        return (int(x), int(y), int(x + w), int(y + h))


class _NullDetector:
    name = "none"

    def __call__(self, rgb: np.ndarray) -> Optional[Box]:
        return None


def build_detector(backend: str = "auto"):
    """Return a callable(rgb)->Box|None, degrading through the backend list."""
    order = [backend] if backend != "auto" else ["mediapipe", "opencv", "none"]
    for b in order:
        try:
            if b == "mediapipe":
                d = _MediaPipeDetector()
            elif b == "opencv":
                d = _HaarDetector()
            elif b == "none":
                d = _NullDetector()
            else:
                continue
            log.info(f"face backend: {d.name}")
            return d
        except Exception as e:
            log.warning(f"face backend '{b}' unavailable ({e.__class__.__name__}: {e})")
    log.warning("no face backend available — falling back to centre crop")
    return _NullDetector()


# --------------------------------------------------------------------------
# Box maths
# --------------------------------------------------------------------------

def _square_with_margin(box: Box, margin: float, w: int, h: int) -> Box:
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    side = max(x2 - x1, y2 - y1) * (1.0 + margin)
    half = side / 2.0
    return (int(round(cx - half)), int(round(cy - half)),
            int(round(cx + half)), int(round(cy + half)))


def _centre_box(w: int, h: int) -> Box:
    side = int(min(w, h) * 0.8)
    cx, cy = w // 2, h // 2
    return (cx - side // 2, cy - side // 2, cx + side // 2, cy + side // 2)


def _crop_pad(img: np.ndarray, box: Box) -> np.ndarray:
    """Crop with reflection padding so a box running off-frame still works."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    pl, pt = max(0, -x1), max(0, -y1)
    pr, pb = max(0, x2 - w), max(0, y2 - h)
    if pl or pt or pr or pb:
        img = np.pad(img, ((pt, pb), (pl, pr), (0, 0)), mode="reflect")
        x1, x2 = x1 + pl, x2 + pl
        y1, y2 = y1 + pt, y2 + pt
    out = img[y1:y2, x1:x2]
    return out if out.size else img


def smooth_boxes(boxes: List[Optional[Box]], window: int = 5) -> List[Optional[Box]]:
    """
    Fill gaps by nearest-neighbour in time, then moving-average the centres and
    fix one common side length for the clip. This is the step that stops crop
    jitter from masquerading as a forensic signal.
    """
    n = len(boxes)
    known = [i for i, b in enumerate(boxes) if b is not None]
    if not known:
        return boxes

    filled: List[Box] = []
    for i in range(n):
        if boxes[i] is not None:
            filled.append(boxes[i])            # type: ignore[arg-type]
        else:
            j = min(known, key=lambda k: abs(k - i))
            filled.append(boxes[j])            # type: ignore[arg-type]

    arr = np.array(filled, dtype=np.float64)
    cx = (arr[:, 0] + arr[:, 2]) / 2.0
    cy = (arr[:, 1] + arr[:, 3]) / 2.0
    side = np.maximum(arr[:, 2] - arr[:, 0], arr[:, 3] - arr[:, 1])

    k = max(1, min(window, n))
    kernel = np.ones(k) / k
    pad = k // 2
    def smooth(v):
        vp = np.pad(v, (pad, pad), mode="edge")
        return np.convolve(vp, kernel, mode="valid")[:n]

    cx_s, cy_s = smooth(cx), smooth(cy)
    side_s = float(np.median(side))            # one size for the whole clip

    half = side_s / 2.0
    return [(int(round(x - half)), int(round(y - half)),
             int(round(x + half)), int(round(y + half)))
            for x, y in zip(cx_s, cy_s)]


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

@dataclass
class CropResult:
    frames: np.ndarray          # (T, S, S, 3) uint8
    face_ratio: float           # fraction of frames with a real detection
    boxes: List[Box]
    backend: str


def crop_clip(frames: np.ndarray, detector, img_size: int = 224,
              margin: float = 0.30, detect_every: int = 8) -> CropResult:
    """
    Detect on every `detect_every`-th frame, interpolate + smooth, then crop
    and resize the whole clip to (img_size, img_size).
    """
    if cv2 is None:
        raise ImportError("opencv-python is required for cropping")

    T = len(frames)
    h, w = frames[0].shape[:2]

    raw: List[Optional[Box]] = []
    for t in range(T):
        if t % detect_every == 0:
            b = detector(frames[t])
            raw.append(_square_with_margin(b, margin, w, h) if b else None)
        else:
            raw.append(None)

    n_detected = sum(1 for b in raw if b is not None)
    n_probed = max(1, math.ceil(T / detect_every))
    face_ratio = n_detected / n_probed

    if n_detected == 0:
        boxes = [_centre_box(w, h)] * T
    else:
        boxes = smooth_boxes(raw, window=5)    # type: ignore[assignment]

    out = np.empty((T, img_size, img_size, 3), dtype=np.uint8)
    for t in range(T):
        patch = _crop_pad(frames[t], boxes[t])
        # INTER_AREA when downscaling preserves the high-frequency statistics
        # far better than INTER_LINEAR, which matters for the spectral branches.
        interp = cv2.INTER_AREA if patch.shape[0] > img_size else cv2.INTER_CUBIC
        out[t] = cv2.resize(patch, (img_size, img_size), interpolation=interp)

    return CropResult(frames=out, face_ratio=float(face_ratio),
                      boxes=list(boxes), backend=getattr(detector, "name", "?"))


__all__ = ["build_detector", "crop_clip", "CropResult", "smooth_boxes", "Box"]
