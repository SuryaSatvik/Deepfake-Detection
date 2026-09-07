"""
lmf.py — Landmark-conditioned Motion-compensated Forensics
===========================================================

THE SECOND NOVEL BRANCH. Read `tsc.py` for the first one; this module is
deliberately designed to be *orthogonal* to it, which is what makes the
intra-model ablation (TSC vs LMF vs both) scientifically meaningful rather
than decorative.

THE PROBLEM WITH MEASURING TEMPORAL ARTEFACTS IN IMAGE COORDINATES
-------------------------------------------------------------------
Every temporal forensic statistic — TSC included — is computed on the pixel
grid. But between two frames of a talking-head video the pixel at (x, y) can
change for two completely different reasons:

    (a) the generator produced unstable detail        <- what we want
    (b) the head moved, so (x, y) now shows a
        different part of the face                    <- confound

(b) dominates. A person turning their head changes far more pixel energy than
any generator instability, so a temporal statistic in image coordinates is
mostly measuring head motion. Detectors cope by learning to ignore motion,
which costs capacity and generalises badly across datasets with different
motion statistics.

WHAT THIS MODULE DOES INSTEAD
------------------------------
Celeb-DF-v2 ships 81 facial landmarks per frame. That is enough to estimate,
per frame, the similarity transform taking this face to a canonical template.
Warping into that frame removes (b) *geometrically* instead of asking a
network to learn it away. Then:

    frames + landmarks
        |
        |-- 1. Procrustes alignment on motion-stable points (eyes + nose
        |      bridge) -> canonical face coordinates I~_t
        |
        |-- 2. motion-compensated residual  R_t = I~_t - I~_{t-1}
        |      In real footage this is sensor noise + non-rigid expression,
        |      and it is spatially uniform because one sensor produced the
        |      whole frame. In a swap it is NOT uniform, and the measured
        |      direction (see below) is the finding, not a prediction.
        |
        |-- 3. landmark-defined region pooling. Because we are in canonical
        |      coordinates, the face mask is a FIXED spatial locus across the
        |      whole clip, so these three regions mean the same thing in every
        |      frame:
        |          inner    = eroded convex hull of the landmarks (synthesised)
        |          boundary = the annulus around the hull      (blend seam)
        |          outer    = everything outside                (untouched)
        |      -> region energy E in R^{(T-1) x K}
        |
        |-- 4. closed-form contrast descriptors (reported as evidence, not
        |      just consumed by a network) and a small learned encoder.

WHY THIS IS A REAL CONTRIBUTION AND NOT A RESKIN
-------------------------------------------------
* It measures a property of the *generative process* of a face swap: a swap
  edits inside a mask and leaves the outside alone. `inner_outer_contrast` and
  `boundary_excess` are direct measurements of that asymmetry. No branch in
  FSV-Net, and no published frequency detector, measures it.

MEASURED DIRECTION ON CELEB-DF-V2 (30 real vs 30 fake clips, before any
training — see notebooks/deepfake_model_comparison.ipynb S5 for the full,
identity-disjoint version)

      descriptor              real     fake    Cohen's d
      inner_outer_contrast  +0.018   -0.383      -1.18
      boundary_excess       -0.084   +0.111      +1.05

  The first sign is the OPPOSITE of the naive "generator noise adds energy
  inside the mask" guess, and it is worth stating plainly because it is the
  more interesting result: motion-compensated residual energy inside the
  swapped region is *lower* than outside it. The synthesised face is
  temporally OVER-SMOOTH — it is decoded from a latent that changes smoothly,
  so consecutive frames of the fake face are more similar to each other than
  consecutive frames of real sensor output are. Head motion normally hides
  this; removing it geometrically is what exposes it.

  The second sign is the expected one: the blend seam carries *excess*
  residual energy relative to the interpolation of its two neighbours, i.e.
  the boundary flickers. The two descriptors therefore measure different
  halves of the same manipulation, which is why they are both kept.

  These are effect sizes on a small sample, not accuracy. Treat them as the
  reason the branch is worth training, not as a result.
* It is conditioned on geometry the model is *given*, not geometry it must
  infer. Neither Skyra/Qwen2.5-VL nor the FSV-Net branches use the landmarks.
* It is orthogonal to TSC by construction: TSC is frequency-domain, image
  coordinates, temporal-spectrum shaped; LMF is spatial-domain, face-canonical
  coordinates, region-contrast shaped. So `lmf_only` vs `tsc_only` vs `both`
  is a clean decomposition, not two views of one signal.
* Every descriptor is a scalar with a physical meaning and a percentile, which
  is what the evidence pack and the VLM prompt need. A grounded explanation
  can say *"the residual energy inside the face mask is 6.2x the energy
  outside it, the 99th percentile of real videos"* — falsifiable, unlike a
  bare probability.

HONEST LIMITATIONS (do not omit these from the report)
-------------------------------------------------------
* Landmark quality bounds everything. If the detector that produced the .npy
  files failed on a frame, alignment fails; `geometric_residual` is exposed
  precisely so those clips can be identified rather than silently trusted.
* Warping resamples, and resampling is itself a low-pass operation that
  changes high-frequency statistics. Both real and fake go through the
  identical warp, so the *contrast* between regions is fair — but absolute
  residual energy is not comparable to unwarped work.
* On a still head the alignment is near-identity and LMF degenerates towards a
  plain frame difference. It should help most where the head moves, which is
  the complement of TSC's weakness. That is a testable prediction and the
  notebook tests it.
* The region semantics assume the 81-point layout is dlib-68 + 13 forehead
  points. `verify_landmark_layout()` checks that on real data and refuses to
  guess if it does not hold.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import cv2
except Exception:                                       # pragma: no cover
    cv2 = None


# --------------------------------------------------------------------------
# Landmark layout
# --------------------------------------------------------------------------
# dlib-68 + 13 forehead. Indices used below; verified by verify_landmark_layout.
JAW        = list(range(0, 17))
BROW_L     = list(range(17, 22))
BROW_R     = list(range(22, 27))
NOSE_BRIDGE= list(range(27, 31))
NOSE_BASE  = list(range(31, 36))
EYE_L      = list(range(36, 42))
EYE_R      = list(range(42, 48))
MOUTH      = list(range(48, 68))
FOREHEAD   = list(range(68, 81))

# Points used to estimate the rigid/similarity transform. Chosen because they
# move least with expression: eye corners and the nose bridge. Using the mouth
# or jaw here would let a smile masquerade as head motion.
STABLE = EYE_L + EYE_R + NOSE_BRIDGE

REGION_NAMES = ["inner", "boundary", "outer"]


def verify_landmark_layout(landmarks: np.ndarray) -> dict:
    """
    Check the dlib-68+13 assumption on real data instead of trusting it.

    landmarks: (N, P, 2). Returns a report; `ok` False means the semantic
    index groups above are wrong for this dataset and the region masks would
    be nonsense. Callers should fall back to `use_convex_hull_only=True`.
    """
    lm = np.asarray(landmarks, dtype=np.float64)
    if lm.ndim == 2:
        lm = lm[None]
    P = lm.shape[1]
    rep = {"n_points": int(P), "expected": 81, "checks": {}, "ok": False}
    if P != 81:
        rep["checks"]["point_count"] = f"got {P}, expected 81"
        return rep
    m = lm.reshape(-1, P, 2).mean(axis=0)
    eye_l, eye_r = m[EYE_L].mean(0), m[EYE_R].mean(0)
    mouth, fore, jaw = m[MOUTH].mean(0), m[FOREHEAD].mean(0), m[JAW].mean(0)
    c = {
        "eyes_are_left_of_right": bool(eye_l[0] < eye_r[0]),
        "eyes_above_mouth":       bool(eye_l[1] < mouth[1] and eye_r[1] < mouth[1]),
        "forehead_above_eyes":    bool(fore[1] < min(eye_l[1], eye_r[1])),
        "jaw_below_eyes":         bool(jaw[1] > max(eye_l[1], eye_r[1])),
        "mouth_between_eyes_x":   bool(eye_l[0] < mouth[0] < eye_r[0]),
    }
    rep["checks"] = c
    rep["ok"] = all(c.values())
    if not rep["ok"]:
        rep["advice"] = ("index semantics do not hold — set use_convex_hull_only=True "
                         "so only the hull (which needs no semantics) is used")
    return rep


# --------------------------------------------------------------------------
# 1. Canonical alignment
# --------------------------------------------------------------------------

def build_template(landmarks: np.ndarray, size: int = 256,
                   inset: float = 0.78) -> np.ndarray:
    """
    Canonical face template = the Procrustes mean shape of the supplied
    landmarks, rescaled to sit inside a `size` x `size` canvas with `inset` of
    the canvas covered by the face. Build this ONCE from TRAIN reals only and
    persist it — building it from the evaluation set is a (subtle) leak.
    """
    lm = np.asarray(landmarks, dtype=np.float64).reshape(-1, np.asarray(landmarks).shape[-2], 2)
    # Generalised Procrustes: iteratively align to the running mean.
    ref = lm[0].copy()
    for _ in range(5):
        aligned = np.stack([_similarity_apply(_similarity_fit(x[STABLE], ref[STABLE]), x)
                            for x in lm])
        ref = aligned.mean(axis=0)
    lo, hi = ref.min(0), ref.max(0)
    span = float(np.max(hi - lo))
    if span <= 0:
        raise ValueError("degenerate template")
    scale = inset * size / span
    ref = (ref - (lo + hi) / 2.0) * scale + size / 2.0
    return ref.astype(np.float32)


def _similarity_fit(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """
    Least-squares similarity (rotation + uniform scale + translation) mapping
    src -> dst. Closed form (Umeyama); no RANSAC, because landmark outliers
    show up in `geometric_residual` where we want to see them, not silently
    rejected.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s0, d0 = src - mu_s, dst - mu_d
    var_s = (s0 ** 2).sum() / len(s0)
    cov = (d0.T @ s0) / len(s0)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(2)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[1, 1] = -1
    R = U @ S @ Vt
    scale = float((D * np.diag(S)).sum() / var_s) if var_s > 1e-12 else 1.0
    t = mu_d - scale * R @ mu_s
    M = np.zeros((2, 3), dtype=np.float64)
    M[:, :2] = scale * R
    M[:, 2] = t
    return M


def _similarity_apply(M: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64)
    return pts @ M[:, :2].T + M[:, 2]


def align_clip(frames: np.ndarray, landmarks: np.ndarray, template: np.ndarray,
               size: int = 256) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Warp every frame into canonical face coordinates.

    frames    (T, H, W, 3) uint8
    landmarks (T, P, 2)
    Returns
        warped   (T, size, size, 3) uint8
        fit_err  (T,)  RMS residual of the similarity fit, in canonical px.
                       This is `geometric_residual`: large values mean the
                       face did not move rigidly, which is either a real
                       expression, a landmark failure, or a warped identity.
        valid    (T,) bool — False where the transform was degenerate.
    """
    if cv2 is None:
        raise RuntimeError("lmf.align_clip needs opencv-python")
    T = len(frames)
    out = np.zeros((T, size, size, 3), dtype=np.uint8)
    fit = np.zeros(T, dtype=np.float32)
    valid = np.ones(T, dtype=bool)
    tpl_stable = template[STABLE]
    for t in range(T):
        pts = np.asarray(landmarks[t], dtype=np.float64)
        try:
            M = _similarity_fit(pts[STABLE], tpl_stable)
            pred = _similarity_apply(M, pts[STABLE])
            fit[t] = float(np.sqrt(((pred - tpl_stable) ** 2).sum(1).mean()))
            out[t] = cv2.warpAffine(frames[t], M.astype(np.float32), (size, size),
                                    flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_REPLICATE)
        except Exception:
            valid[t] = False
            out[t] = cv2.resize(frames[t], (size, size))
            fit[t] = np.nan
    return out, fit, valid


# --------------------------------------------------------------------------
# 2/3. Region masks and motion-compensated residual
# --------------------------------------------------------------------------

def region_masks(template: np.ndarray, size: int = 256,
                 band_px: int = 10, erode_px: int = 6,
                 use_convex_hull_only: bool = False) -> Dict[str, np.ndarray]:
    """
    Three masks in canonical coordinates. Because we warped, these are the
    SAME pixels in every frame of every clip — which is the whole point.

        inner     eroded convex hull of the face landmarks  (synthesised area)
        boundary  the annulus of width 2*band_px around the hull (blend seam)
        outer     outside the dilated hull                  (never edited)
    """
    if cv2 is None:
        raise RuntimeError("lmf.region_masks needs opencv-python")
    pts = np.asarray(template, dtype=np.float32)
    hull_src = pts if use_convex_hull_only else pts[JAW + FOREHEAD]
    hull = cv2.convexHull(hull_src.astype(np.int32))
    base = np.zeros((size, size), dtype=np.uint8)
    cv2.fillConvexPoly(base, hull, 1)
    k = lambda n: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * n + 1, 2 * n + 1))
    inner = cv2.erode(base, k(erode_px + band_px))
    dil = cv2.dilate(base, k(band_px))
    ero = cv2.erode(base, k(band_px))
    boundary = (dil - ero).astype(np.uint8)
    outer = (1 - dil).astype(np.uint8)
    return {"inner": inner.astype(bool),
            "boundary": boundary.astype(bool),
            "outer": outer.astype(bool)}


def motion_compensated_residual(warped: np.ndarray,
                                highpass: bool = True) -> np.ndarray:
    """
    R_t = I~_t - I~_{t-1} on the luma channel, optionally high-passed first so
    that slow illumination drift (which is benign and identical inside and
    outside the mask) does not dominate the region energies.

    warped (T, S, S, 3) uint8 -> (T-1, S, S) float32
    """
    x = warped.astype(np.float32)
    y = 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]
    if highpass and cv2 is not None:
        y = np.stack([f - cv2.GaussianBlur(f, (0, 0), 1.5) for f in y])
    return np.diff(y, axis=0).astype(np.float32)


def region_energy(residual: np.ndarray, masks: Dict[str, np.ndarray]) -> np.ndarray:
    """(T-1, S, S) -> (T-1, 3) mean squared residual per region, order REGION_NAMES."""
    out = np.zeros((residual.shape[0], len(REGION_NAMES)), dtype=np.float32)
    for j, name in enumerate(REGION_NAMES):
        m = masks[name]
        n = int(m.sum())
        out[:, j] = (residual[:, m] ** 2).mean(axis=1) if n else np.nan
    return out


# --------------------------------------------------------------------------
# 4. Closed-form descriptors
# --------------------------------------------------------------------------

@dataclass
class LMFResult:
    energy: np.ndarray                 # (T-1, 3)
    descriptors: Dict[str, float]
    fit_err: np.ndarray                # (T,)
    masks: Dict[str, np.ndarray]
    warped: Optional[np.ndarray] = None
    residual: Optional[np.ndarray] = None


def lmf_descriptors(energy: np.ndarray, fit_err: np.ndarray) -> Dict[str, float]:
    """
    Every value here is a scalar with a stated meaning, because the evidence
    pack and the VLM prompt need something falsifiable to quote.

      inner_outer_contrast  log10 of (residual energy inside the face mask) /
                            (outside it). Real footage has one sensor, so the
                            ratio sits near 0 (measured: +0.018). A swapped
                            face is decoded from a temporally smooth latent,
                            so it changes LESS between frames than real detail
                            does and the ratio goes negative (measured:
                            -0.383, Cohen's d = -1.18). This only separates
                            once head motion has been removed by the warp.
      boundary_excess       log10 of boundary energy over the interpolation of
                            inner and outer. A blend seam is a spatial
                            discontinuity that flickers; this is the only
                            descriptor that isolates it.
      inner_instability     temporal coefficient of variation of inner energy.
                            Generators are unstable in bursts; sensors are not.
      contrast_drift        slope of inner_outer_contrast over the clip.
      seam_periodicity      normalised peak of the temporal rFFT of the
                            boundary series, excluding DC. Blending pipelines
                            that re-estimate per keyframe leave a rate here.
      geometric_residual    mean Procrustes fit error, canonical px. High =>
                            the face did not move rigidly. Reported so that
                            landmark failures are visible rather than trusted.
      geometric_jitter      std of the same, frame to frame.
    """
    eps = 1e-8
    e = np.asarray(energy, dtype=np.float64)
    inner, boundary, outer = e[:, 0], e[:, 1], e[:, 2]
    ratio = np.log10((inner + eps) / (outer + eps))
    interp = 0.5 * (inner + outer)
    bex = np.log10((boundary + eps) / (interp + eps))

    def _slope(v):
        n = len(v)
        if n < 3:
            return 0.0
        t = np.arange(n, dtype=np.float64)
        return float(np.polyfit(t, v, 1)[0])

    b = boundary - boundary.mean()
    if len(b) >= 4:
        w = np.hanning(len(b))
        spec = np.abs(np.fft.rfft(b * w))[1:]
        seam = float(spec.max() / (spec.sum() + eps)) if spec.size else 0.0
    else:
        seam = 0.0

    fe = np.asarray(fit_err, dtype=np.float64)
    fe = fe[np.isfinite(fe)]
    return {
        "inner_outer_contrast": float(np.nanmean(ratio)),
        "boundary_excess": float(np.nanmean(bex)),
        "inner_instability": float(np.nanstd(inner) / (np.nanmean(inner) + eps)),
        "contrast_drift": _slope(ratio),
        "seam_periodicity": seam,
        "geometric_residual": float(fe.mean()) if fe.size else float("nan"),
        "geometric_jitter": float(fe.std()) if fe.size else float("nan"),
    }


DESCRIPTOR_NAMES = ["inner_outer_contrast", "boundary_excess", "inner_instability",
                    "contrast_drift", "seam_periodicity", "geometric_residual",
                    "geometric_jitter"]


def analyse_clip(frames: np.ndarray, landmarks: np.ndarray, template: np.ndarray,
                 size: int = 256, band_px: int = 10, erode_px: int = 6,
                 keep_maps: bool = False,
                 use_convex_hull_only: bool = False) -> LMFResult:
    """End-to-end numpy path. No torch — usable in preprocessing and on CPU."""
    warped, fit, _valid = align_clip(frames, landmarks, template, size)
    masks = region_masks(template, size, band_px, erode_px, use_convex_hull_only)
    res = motion_compensated_residual(warped)
    energy = region_energy(res, masks)
    desc = lmf_descriptors(energy, fit)
    return LMFResult(energy=energy, descriptors=desc, fit_err=fit, masks=masks,
                     warped=warped if keep_maps else None,
                     residual=res if keep_maps else None)


# --------------------------------------------------------------------------
# 5. The learned branch
# --------------------------------------------------------------------------

def _torch():
    import torch
    import torch.nn as nn
    return torch, nn


class LMFBranch:
    """
    Thin factory so this module imports without torch. Call `LMFBranch.build()`
    to get the real nn.Module.
    """

    @staticmethod
    def build(out_features: int = 256, n_regions: int = 3,
              n_descriptors: int = len(DESCRIPTOR_NAMES), hidden: int = 96):
        torch, nn = _torch()

        class _LMFBranch(nn.Module):
            """
            Consumes the CHEAP representation — per-frame region energies
            (T-1, 3) plus the closed-form descriptors — not the pixels. The
            expensive part (warping) happens once in preprocessing and is
            cached, so this branch costs almost nothing at train time and the
            ablation `-LMF` is a fair comparison rather than a compute cut.
            """

            def __init__(self):
                super().__init__()
                self.log_eps = 1e-8
                self.temporal = nn.Sequential(
                    nn.Conv1d(n_regions + 2, hidden, 5, padding=2), nn.GELU(),
                    nn.BatchNorm1d(hidden),
                    nn.Conv1d(hidden, hidden, 3, padding=1, dilation=1), nn.GELU(),
                    nn.Conv1d(hidden, hidden, 3, padding=2, dilation=2), nn.GELU(),
                )
                self.desc = nn.Sequential(
                    nn.LayerNorm(n_descriptors),
                    nn.Linear(n_descriptors, hidden), nn.GELU(),
                )
                self.head = nn.Sequential(
                    nn.Linear(hidden * 2, out_features), nn.GELU(),
                    nn.LayerNorm(out_features),
                )
                self.out_features = out_features

            def forward(self, energy, descriptors, return_evidence: bool = False):
                """
                energy      (B, T-1, 3) float
                descriptors (B, D) float
                """
                e = torch.log10(energy.clamp_min(self.log_eps))
                inner, bnd, outer = e[..., 0], e[..., 1], e[..., 2]
                ratio = (inner - outer).unsqueeze(-1)
                bex = (bnd - 0.5 * (inner + outer)).unsqueeze(-1)
                x = torch.cat([e, ratio, bex], dim=-1).transpose(1, 2)
                h = self.temporal(x)
                h = h.mean(dim=2) + h.amax(dim=2) * 0.0     # mean pool
                d = self.desc(torch.nan_to_num(descriptors))
                z = self.head(torch.cat([h, d], dim=-1))
                if return_evidence:
                    return z, {"log_inner_outer": ratio.squeeze(-1).detach(),
                               "log_boundary_excess": bex.squeeze(-1).detach()}
                return z

        return _LMFBranch()


__all__ = ["verify_landmark_layout", "build_template", "align_clip", "region_masks",
           "motion_compensated_residual", "region_energy", "lmf_descriptors",
           "analyse_clip", "LMFResult", "LMFBranch", "DESCRIPTOR_NAMES",
           "REGION_NAMES", "STABLE", "JAW", "FOREHEAD"]
