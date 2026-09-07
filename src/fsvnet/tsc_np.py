"""
tsc_np.py — torch-free numpy port of the TSC descriptors.

Kept separate from `tsc.py` (which imports torch for the learned branch) so
that CPU-only preprocessing, the descriptor probes and the Streamlit evidence
path can run on a machine with no torch installed.

MIRRORS `tsc.tsc_descriptors` EXACTLY. If you edit one, edit both — there is a
consistency assertion in scripts/smoke_test.py that will catch drift once torch
is available.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

# ==========================================================================
# numpy port of the descriptors — so the CPU probe in
# scripts/extract_tsc.py can measure TSC on the same footing as LMF without
# torch installed. Mirrors tsc_descriptors() exactly; kept adjacent so the two
# stay in step. If you edit one, edit both.
# ==========================================================================

TSC_DESCRIPTOR_NAMES = ["flicker_index", "temporal_whiteness",
                        "temporal_whiteness_coarse", "temporal_peakiness",
                        "temporal_hf_ratio", "radial_hf_ratio",
                        "fine_ring_acf1", "coarse_ring_acf1",
                        "spectral_drift", "map_energy"]


def tsc_descriptors_np(tsc_map: np.ndarray, spectra: np.ndarray,
                       eps: float = 1e-6) -> dict:
    """tsc_map (F, K), spectra (T, K) -> {name: float}"""
    Fq, K = tsc_map.shape
    lo_r, hi_r = K // 3, 2 * K // 3
    lo_t = max(1, Fq // 3)
    band = tsc_map[1:, :] if Fq > 1 else tsc_map

    e_total = band.sum() + eps
    e_hi_t = band[lo_t:, :].sum()
    e_hi_r = band[:, hi_r:].sum()
    e_hi_both = band[lo_t:, hi_r:].sum()
    e_lo_both = band[:lo_t, :lo_r].sum() + eps

    s = spectra - spectra.mean(axis=0, keepdims=True)
    num = (s[1:, :] * s[:-1, :]).sum(axis=0)
    den = (s ** 2).sum(axis=0) + eps
    acf1 = num / den

    fine = band[:, hi_r:] + eps
    whiteness = float(np.exp(np.log(fine).mean(axis=0)
                             - np.log(fine.mean(axis=0) + eps)).mean())
    coarse = band[:, :lo_r] + eps
    whiteness_c = float(np.exp(np.log(coarse).mean(axis=0)
                               - np.log(coarse.mean(axis=0) + eps)).mean())
    peak = float(fine.max(axis=0).sum() / (fine.sum() + eps))

    return {
        "flicker_index": float(np.log((e_hi_both + eps) / e_lo_both)),
        "temporal_whiteness": whiteness,
        "temporal_whiteness_coarse": whiteness_c,
        "temporal_peakiness": peak,
        "temporal_hf_ratio": float(e_hi_t / e_total),
        "radial_hf_ratio": float(e_hi_r / e_total),
        "fine_ring_acf1": float(acf1[hi_r:].mean()),
        "coarse_ring_acf1": float(acf1[:lo_r].mean()),
        "spectral_drift": float(spectra.std(axis=0).mean()),
        "map_energy": float(np.log1p(e_total)),
    }


def azimuthal_power_np(frames: np.ndarray, n_bins: int = 64) -> np.ndarray:
    """
    frames : (T, H, W, 3) uint8 or float
    -> (T, n_bins) log-compressed azimuthal power spectra
    """
    x = frames.astype(np.float64)
    if x.ndim == 4:
        x = x.mean(axis=-1)
    x = x / 255.0 if x.max() > 1.5 else x
    T, H, W = x.shape

    fy = np.fft.fftfreq(H)[:, None]
    fx = np.fft.fftfreq(W)[None, :]
    radius = np.sqrt(fx ** 2 + fy ** 2)
    idx = np.clip((radius / (radius.max() + 1e-12) * n_bins).astype(int), 0, n_bins - 1)
    counts = np.bincount(idx.ravel(), minlength=n_bins).astype(np.float64)
    counts[counts == 0] = 1.0

    out = np.zeros((T, n_bins))
    for t in range(T):
        f = np.fft.fft2(x[t], norm="ortho")
        p = (np.abs(f) ** 2).ravel()
        out[t] = np.bincount(idx.ravel(), weights=p, minlength=n_bins) / counts
    return np.log1p(out)


def tsc_map_np(frames: np.ndarray, n_bins: int = 64) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (tsc_map (F, K), spectra (T, K)) — the plotting path."""
    spectra = azimuthal_power_np(frames, n_bins)
    x = spectra - spectra.mean(axis=0, keepdims=True)
    T = x.shape[0]
    t = np.linspace(-1, 1, T)[:, None]
    x = x - (x * t).sum(0, keepdims=True) / ((t ** 2).sum() + 1e-9) * t
    if T > 2:
        x = x * np.hanning(T)[:, None]
    mag = np.abs(np.fft.rfft(x, axis=0, norm="ortho"))
    return np.log1p(mag), spectra


__all__ = ["tsc_descriptors_np", "TSC_DESCRIPTOR_NAMES",
           "azimuthal_power_np", "tsc_map_np"]
