"""
tsc.py — Temporal Spectral Coherence
=====================================

THE NOVEL COMPONENT OF THIS PROJECT.

The idea in one paragraph
-------------------------
Every frequency-domain deepfake detector so far asks: *does this frame's power
spectrum look synthetic?* On modern generators the answer is increasingly "not
obviously", and it does not transfer across generators — the previous stage of
this capstone measured 0.4683 AUC (below chance) transferring an image-level
frequency detector to Celeb-DF frames. TSC asks a different question:

    **Does the spectrum evolve over time the way a real camera's does?**

The mechanism, stated precisely
-------------------------------
In a real camera video, the *fine* spatial detail is dominated by photon and
read noise. That noise is close to independent from frame to frame, so the
energy in the fine radial rings fluctuates like **white noise along the time
axis** — a flat temporal spectrum.

In a synthesised or face-swapped video the fine detail is not sampled from a
sensor; it is *computed* by a generator from a temporally smooth latent (the
driving frame, the previous latent, a motion field). A smooth function of a
smooth input is smooth, so the fine-ring energy becomes **temporally
correlated** — a low-pass, or "coloured", temporal spectrum. Blending, colour
correction and per-frame super-resolution add the opposite defect: narrowband
spikes at the flicker rate.

So the discriminative quantity is not "more" or "less" high-frequency energy —
it is the *shape* of the temporal spectrum of each spatial frequency band:

    real  -> approximately flat (white) in the fine rings
    fake  -> coloured: low-pass (over-smooth) or spiked (flicker)

`temporal_whiteness` (spectral flatness) measures exactly that, and it is
generator-agnostic by construction: the branch never learns what a particular
generator's artefact looks like, only what a camera's temporal noise statistic
looks like. Two caveats stated up front — H.264 rate control also colours the
temporal spectrum (which is why compression augmentation is mandatory, see
dataset.py), and a static scene has little fine detail to measure.

The computation
---------------
For a clip of T frames:

  1. per frame, the azimuthally-averaged power spectrum   A ∈ R^{T×K}
     (K radial rings, exactly the descriptor FSW-Net already used per image)
  2. log-compress, then subtract the temporal mean per ring:
                                                          Ã = A − mean_t(A)
     This removes the *image content* — what is left is only how each spatial
     frequency band fluctuates in time.
  3. Hann-window and take an rFFT along the time axis:
                                       C = |rFFT_t(w·Ã)| ∈ R^{F×K}, F = T/2+1
     C is the **temporal spectral coherence map**. Its horizontal axis is
     spatial frequency (fine detail →), its vertical axis is temporal
     frequency (flicker rate →).
  4. a small CNN embeds C; a handful of closed-form scalars are also emitted
     for the report figures and the VLM evidence pack.

Reading the map
---------------
  * flat energy across temporal frequency in the FINE rings   → sensor noise,
                                                                natural
  * energy concentrated at LOW temporal frequency, all rings  → natural motion
    (removed by the linear detrend before the FFT)
  * strong roll-off with temporal frequency in the FINE rings → generated
                                                                detail
  * a narrow spike at one temporal frequency                  → flicker from
                                                                blending or
                                                                colour
                                                                correction

The last two are anomalies in opposite directions, which is why the branch
keeps a *signed* `flicker_index` and a separate `temporal_whiteness` rather
than one absolute deviation score.

Honesty note
------------
This is a hypothesis with a mechanism, not a proven result. `scripts/
smoke_test.py` and `synthetic.py` plant each of the four signatures above in
synthetic clips and assert TSC separates them — the same
validate-on-synthetic-first discipline the image project used. Whether the
signature survives H.264 rate control on real data is an empirical question
your Step-4 experiment answers, and a negative answer is a publishable finding
too.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



def _fp32_autocast():
    """
    Disable AMP for a block, across torch versions.

    The FFTs below must run in fp32: a half-precision FFT on a 224x224 frame
    loses precision exactly in the high-frequency tail this branch measures.
    """
    try:                      # torch >= 2.4
        return torch.amp.autocast("cuda", enabled=False)
    except (AttributeError, TypeError):   # older torch
        return torch.cuda.amp.autocast(enabled=False)


# ==========================================================================
# Radial binning (shared with the frequency branch)
# ==========================================================================

def build_radial_index(h: int, w: int, n_bins: int, device=None
                       ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute, once, the ring index of every pixel of a centred 2-D power
    spectrum, plus the number of pixels in each ring.

    Returns (index (h*w,) long, counts (n_bins,) float).

    FSW-Net's image version looped over 64 boolean masks per forward pass.
    Doing it with a precomputed index and one `index_add_` is ~30x faster and
    numerically identical, which matters here because we run it T times per
    clip instead of once per image.
    """
    fy = torch.fft.fftfreq(h, device=device).unsqueeze(1).expand(h, w)
    fx = torch.fft.fftfreq(w, device=device).unsqueeze(0).expand(h, w)
    radius = torch.sqrt(fx ** 2 + fy ** 2)                     # (h, w)

    r_max = float(radius.max())
    # scale to [0, n_bins) and clamp the Nyquist corner into the last ring
    idx = (radius / (r_max + 1e-12) * n_bins).long().clamp_(0, n_bins - 1)
    idx = idx.reshape(-1)

    counts = torch.zeros(n_bins, device=device)
    counts.index_add_(0, idx, torch.ones_like(idx, dtype=torch.float32))
    counts.clamp_(min=1.0)
    return idx, counts


def azimuthal_power(x: torch.Tensor, radial_idx: torch.Tensor,
                    radial_counts: torch.Tensor, n_bins: int) -> torch.Tensor:
    """
    Azimuthally-averaged power spectrum.

    x : (N, C, H, W) real
    -> (N, n_bins) log-compressed ring means
    """
    N = x.shape[0]
    fft = torch.fft.fft2(x.float(), norm="ortho")
    power = (fft.real ** 2 + fft.imag ** 2).mean(dim=1)        # (N, H, W)
    flat = power.reshape(N, -1)

    out = torch.zeros(N, n_bins, device=x.device, dtype=torch.float32)
    out.index_add_(1, radial_idx, flat)
    out = out / radial_counts.unsqueeze(0)
    return torch.log1p(out)


# ==========================================================================
# The TSC transform
# ==========================================================================

def temporal_spectral_coherence(spectra: torch.Tensor,
                                detrend: bool = True,
                                window: bool = True,
                                eps: float = 1e-6
                                ) -> torch.Tensor:
    """
    spectra : (B, T, K) per-frame azimuthal spectra (already log-compressed)
    returns : (B, F, K) temporal spectral coherence map, F = T//2 + 1

    Steps 2–3 of the docstring above.
    """
    B, T, K = spectra.shape
    x = spectra.float()

    if detrend:
        # remove the static image content AND any linear drift (a slow zoom or
        # lighting change would otherwise dump energy into low temporal bins)
        x = x - x.mean(dim=1, keepdim=True)
        t = torch.linspace(-1.0, 1.0, T, device=x.device).view(1, T, 1)
        denom = (t ** 2).sum(dim=1, keepdim=True) + eps
        slope = (x * t).sum(dim=1, keepdim=True) / denom
        x = x - slope * t

    if window and T > 2:
        w = torch.hann_window(T, periodic=False, device=x.device).view(1, T, 1)
        x = x * w

    spec = torch.fft.rfft(x, dim=1, norm="ortho")              # (B, F, K)
    mag = torch.sqrt(spec.real ** 2 + spec.imag ** 2 + eps)
    return torch.log1p(mag)


def tsc_descriptors(tsc_map: torch.Tensor, spectra: torch.Tensor,
                    eps: float = 1e-6) -> Dict[str, torch.Tensor]:
    """
    Closed-form, human-readable scalars derived from the TSC map. These are what
    go into the report figures and the VLM evidence pack — the CNN embedding is
    not interpretable, these are.

    tsc_map : (B, F, K)
    spectra : (B, T, K)
    """
    B, Fq, K = tsc_map.shape
    lo_r, hi_r = K // 3, 2 * K // 3            # radial thirds: coarse/mid/fine
    lo_t = max(1, Fq // 3)                     # skip bin 0 (the DC of time)

    band = tsc_map[:, 1:, :]                   # drop temporal DC
    if band.shape[1] == 0:
        band = tsc_map

    e_total = band.sum(dim=(1, 2)) + eps
    e_hi_t = band[:, lo_t:, :].sum(dim=(1, 2))
    e_hi_r = band[:, :, hi_r:].sum(dim=(1, 2))
    e_hi_both = band[:, lo_t:, hi_r:].sum(dim=(1, 2))
    e_lo_both = band[:, :lo_t, :lo_r].sum(dim=(1, 2)) + eps

    # lag-1 temporal autocorrelation of each ring, averaged over fine rings.
    # Real sensor noise -> low autocorrelation but non-zero; over-smoothed
    # generation -> autocorrelation near 1.
    s = spectra.float()
    s = s - s.mean(dim=1, keepdim=True)
    num = (s[:, 1:, :] * s[:, :-1, :]).sum(dim=1)
    den = (s ** 2).sum(dim=1) + eps
    acf1 = (num / den)                                          # (B, K)

    # ---- temporal whiteness (spectral flatness) -----------------------
    # Geometric mean / arithmetic mean of the temporal spectrum, computed on
    # the FINE radial rings only. White (sensor-like) -> ~1. Coloured
    # (generated, over-smooth) or spiked (flicker) -> < 1. This is the single
    # most interpretable number the branch produces and the one the report
    # should plot.
    fine = band[:, :, hi_r:] + eps                 # (B, F-1, K/3)
    log_mean = torch.log(fine).mean(dim=1)         # geometric mean, per ring
    ari_mean = torch.log(fine.mean(dim=1) + eps)
    whiteness = torch.exp(log_mean - ari_mean).mean(dim=1)          # (B,)

    coarse = band[:, :, :lo_r] + eps
    whiteness_coarse = torch.exp(
        torch.log(coarse).mean(dim=1) - torch.log(coarse.mean(dim=1) + eps)
    ).mean(dim=1)

    # peakiness: how much of the temporal spectrum sits in its single largest
    # bin — a blending/colour-correction flicker shows up here and nowhere else
    peak = fine.max(dim=1).values.sum(dim=1) / (fine.sum(dim=(1, 2)) + eps)

    out = {
        # signed: >0 means too much fine-ring flicker, <0 means too little
        "flicker_index": torch.log((e_hi_both + eps) / e_lo_both),
        "temporal_whiteness": whiteness,
        "temporal_whiteness_coarse": whiteness_coarse,
        "temporal_peakiness": peak,
        "temporal_hf_ratio": e_hi_t / e_total,
        "radial_hf_ratio": e_hi_r / e_total,
        "fine_ring_acf1": acf1[:, hi_r:].mean(dim=1),
        "coarse_ring_acf1": acf1[:, :lo_r].mean(dim=1),
        "spectral_drift": spectra.float().std(dim=1).mean(dim=1),
        "map_energy": torch.log1p(e_total),
    }
    return out


# ==========================================================================
# The branch module
# ==========================================================================

class TSCBranch(nn.Module):
    """
    Temporal Spectral Coherence branch.

    Input : (B, T, C, H, W) clip
    Output: (B, out_features) embedding, plus a dict of interpretable scalars
            and the raw map (for plotting) when `return_evidence=True`.

    Cost: one FFT2 per frame and one small FFT along time. No pretrained
    weights, ~0.35 M parameters. On an 8 GB card this branch is free relative
    to the EfficientNet trunk.
    """

    def __init__(self, out_features: int = 256, n_bins: int = 64,
                 img_size: int = 224, hidden: int = 96):
        super().__init__()
        self.n_bins = n_bins
        self.img_size = img_size

        idx, counts = build_radial_index(img_size, img_size, n_bins)
        self.register_buffer("radial_idx", idx, persistent=False)
        self.register_buffer("radial_counts", counts, persistent=False)

        # CNN over the (1, F, K) coherence map. Kernels are deliberately
        # rectangular: temporal structure is narrowband, radial structure is
        # broad, so we look further along the radial axis than the temporal one.
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 7), padding=(1, 3)),
            nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=(3, 5), padding=(1, 2)),
            nn.BatchNorm2d(64), nn.GELU(),
            nn.MaxPool2d((1, 2)),
            nn.Conv2d(64, hidden, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(hidden), nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )

        self.n_desc = 10
        self.desc_fc = nn.Sequential(
            nn.LayerNorm(self.n_desc),
            nn.Linear(self.n_desc, 64), nn.GELU(),
        )
        self.out_fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(hidden + 64, out_features),
        )
        self._desc_keys = ["flicker_index", "temporal_whiteness",
                           "temporal_whiteness_coarse", "temporal_peakiness",
                           "temporal_hf_ratio", "radial_hf_ratio",
                           "fine_ring_acf1", "coarse_ring_acf1",
                           "spectral_drift", "map_energy"]
        assert len(self._desc_keys) == self.n_desc

    # ------------------------------------------------------------------
    def spectra_of(self, clip: torch.Tensor) -> torch.Tensor:
        """(B, T, C, H, W) -> (B, T, K) per-frame azimuthal spectra."""
        B, T = clip.shape[:2]
        H, W = clip.shape[-2:]
        if (H, W) != (self.img_size, self.img_size):
            # The radial ring index is precomputed for one resolution. Silently
            # resizing here would be worse than failing: the fine rings ARE the
            # signal, and resampling rewrites them. A model trained at 112 px
            # must be run at 112 px.
            raise ValueError(
                f"TSC branch was built for {self.img_size}x{self.img_size} but "
                f"got {H}x{W}. Inference resolution must match training "
                "resolution — the frequency statistics are not scale-invariant. "
                "Set cfg.data.img_size to the value stored in the checkpoint.")
        flat = clip.reshape(B * T, *clip.shape[2:])
        # The FFT is done in fp32 regardless of AMP: half-precision FFTs on
        # 224x224 lose the very high-frequency tail we are measuring.
        with _fp32_autocast():
            a = azimuthal_power(flat.float(), self.radial_idx,
                                self.radial_counts, self.n_bins)
        return a.reshape(B, T, self.n_bins)

    # ------------------------------------------------------------------
    def forward(self, clip: torch.Tensor, return_evidence: bool = False):
        spectra = self.spectra_of(clip)                     # (B, T, K)
        with _fp32_autocast():
            tmap = temporal_spectral_coherence(spectra)     # (B, F, K)
            desc = tsc_descriptors(tmap, spectra)

        feat_map = self.cnn(tmap.unsqueeze(1)).flatten(1)   # (B, hidden)
        dvec = torch.stack([desc[k] for k in self._desc_keys], dim=1)  # (B, 7)
        dvec = torch.nan_to_num(dvec, nan=0.0, posinf=0.0, neginf=0.0)
        feat_desc = self.desc_fc(dvec.to(feat_map.dtype))

        emb = self.out_fc(torch.cat([feat_map, feat_desc], dim=1))

        if return_evidence:
            evidence = {
                "tsc_map": tmap.detach(),
                "spectra": spectra.detach(),
                "descriptors": {k: v.detach() for k, v in desc.items()},
            }
            return emb, evidence
        return emb, None


# ==========================================================================
# NumPy reference implementation (for plots and offline analysis)
# ==========================================================================

__all__ = ["TSCBranch", "temporal_spectral_coherence", "tsc_descriptors",
           "azimuthal_power", "build_radial_index", "azimuthal_power_np",
           "tsc_map_np"]


# The numpy port of these descriptors lives in `tsc_np.py` so that CPU-only
# preprocessing and the descriptor probes can run on a machine without torch.
from .tsc_np import (tsc_descriptors_np, TSC_DESCRIPTOR_NAMES,
                     azimuthal_power_np, tsc_map_np)  # noqa: E402,F401

__all__ = __all__ + ["tsc_descriptors_np", "TSC_DESCRIPTOR_NAMES",
                     "azimuthal_power_np", "tsc_map_np"]
