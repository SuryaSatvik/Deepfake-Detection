"""
synthetic.py
============

A controlled synthetic video set that (a) lets the whole pipeline be validated
today, with no download, and (b) is the **unit test for the TSC branch**.

The discipline this borrows
----------------------------
The image-level project in this capstone validated each forensic method on
synthetic data where only that method's target signal was planted, *before*
trusting it on 140k real images — and that caught a real bug cheaply. Same
idea here, with one addition that makes it a genuine control:

    **The two classes have, by construction, the SAME time-averaged spatial
    power spectrum. They differ only in the temporal structure of their fine
    detail.**

That matters. If the classes differed in their average spectrum, a per-frame
frequency detector would separate them and the test would prove nothing about
TSC. Because the marginals are matched, any separation TSC achieves *must*
come from the temporal axis. `scripts/smoke_test.py` asserts exactly that:
TSC separates, the per-frame spectral descriptor does not.

The generator
-------------
Every clip is:

    frame_t = smooth_moving_scene(t) + fine_noise_t

`smooth_moving_scene` is a low-pass random field translating at a constant
velocity — a stand-in for content. `fine_noise_t` is what differs:

  * ``real``        white in time: n_t ~ N(0, s^2), independent per frame.
                    This is what a camera sensor does.
  * ``smooth``      AR(1) in time with rho, rescaled by sqrt(1-rho^2) so the
                    per-frame variance is IDENTICAL to ``real``. This is what
                    a generator conditioned on a temporally smooth latent
                    does — the over-smooth failure mode.
  * ``flicker``     white in time but multiplied by a sinusoidal gain at one
                    temporal frequency, again variance-matched. This is
                    blending / per-frame colour correction — the spiked
                    failure mode.

`build_synthetic_dataset` writes them as real .mp4 files so the full stack
(decode -> face box -> crop -> cache -> train) is exercised, not just the
tensor maths.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .utils import log

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


# ==========================================================================
# Clip generation
# ==========================================================================

def _smooth_field(h: int, w: int, rng: np.random.Generator,
                  sigma: float = 12.0) -> np.ndarray:
    """A low-pass random field: stands in for image content."""
    base = rng.standard_normal((h, w))
    k = int(sigma * 4) | 1
    if cv2 is not None:
        base = cv2.GaussianBlur(base, (k, k), sigma)
    base -= base.min()
    base /= (base.max() + 1e-9)
    return base


def _moving_scene(T: int, size: int, rng: np.random.Generator) -> np.ndarray:
    """(T, size, size) in [0,1] — a smooth field translating at constant speed."""
    pad = size // 2
    big = _smooth_field(size + 2 * pad, size + 2 * pad, rng, sigma=size / 18)
    vx, vy = rng.uniform(-1.5, 1.5, size=2)
    out = np.empty((T, size, size), dtype=np.float32)
    for t in range(T):
        ox = int(round(pad + vx * t))
        oy = int(round(pad + vy * t))
        ox = np.clip(ox, 0, 2 * pad)
        oy = np.clip(oy, 0, 2 * pad)
        out[t] = big[oy:oy + size, ox:ox + size]
    # a gentle global illumination drift, removed by TSC's detrend step
    out *= (1.0 + 0.05 * np.sin(np.linspace(0, 2.0, T)))[:, None, None]
    # Keep the scene inside [0.15, 0.85]. Clipping at 0/1 is not neutral: it
    # attenuates large noise excursions more than small ones, which would give
    # the flicker class a *different average spectrum* and quietly break the
    # variance matching this test depends on.
    out = 0.15 + 0.70 * np.clip(out, 0, 1)
    return out


def _fine_noise(kind: str, T: int, size: int, rng: np.random.Generator,
                sigma: float = 0.045, rho: float = 0.92,
                flicker_period: float = 4.0) -> np.ndarray:
    """
    (T, size, size) fine-detail noise. All three kinds have the same per-frame
    variance sigma^2 — only the temporal correlation differs.
    """
    if kind == "real":
        return rng.standard_normal((T, size, size)).astype(np.float32) * sigma

    if kind == "smooth":
        n = np.empty((T, size, size), dtype=np.float32)
        n[0] = rng.standard_normal((size, size))
        scale = np.sqrt(1.0 - rho ** 2)
        for t in range(1, T):
            n[t] = rho * n[t - 1] + scale * rng.standard_normal((size, size))
        return n * sigma          # marginal variance == sigma^2, as for "real"

    if kind == "flicker":
        n = rng.standard_normal((T, size, size)).astype(np.float32)
        g = 1.0 + 0.9 * np.sin(2 * np.pi * np.arange(T) / flicker_period)
        g = g / np.sqrt((g ** 2).mean())            # variance-match
        return n * g[:, None, None] * sigma

    raise ValueError(f"unknown noise kind: {kind}")


def make_clip(kind: str, T: int = 32, size: int = 256,
              seed: Optional[int] = None) -> np.ndarray:
    """Return (T, size, size, 3) uint8 RGB."""
    rng = np.random.default_rng(seed)
    scene = _moving_scene(T, size, rng)
    noise = _fine_noise("real" if kind == "real" else kind, T, size, rng)
    grey = np.clip(scene + noise, 0, 1)
    # mild per-channel gain so the clip is not literally greyscale
    gains = rng.uniform(0.9, 1.1, size=3)
    rgb = np.clip(grey[..., None] * gains[None, None, None, :], 0, 1)
    return (rgb * 255).astype(np.uint8)


# ==========================================================================
# Dataset writer
# ==========================================================================

def build_synthetic_dataset(cfg, n_per_class: int = 100, T: int = 48,
                            size: int = 256, fps: int = 25,
                            fake_kinds: Tuple[str, ...] = ("smooth", "flicker"),
                            lossless: bool = False,
                            clips_per_subject: int = 1,
                            seed: int = 42) -> Path:
    """
    Write `2 * n_per_class` short videos into <data_dir>/synthetic/{real,fake}
    and return that root, ready for `manifest.build_manifest({"synthetic": root})`.

    lossless=False encodes with mp4v, which introduces a real codec into the
    loop — that is deliberate: if TSC only works on uncompressed frames it is
    useless, and you want to find that out here rather than after a 40 GB
    download. Set lossless=True to isolate the effect of the codec.

    `clips_per_subject > 1` gives several clips the same underlying scene and
    stamps a shared `sNNN` token into their filenames, so identities genuinely
    repeat across the dataset. Set it to 1 and every clip is its own identity —
    which makes the leakage audit report 0%, because with unique identities a
    random split cannot leak. Real datasets are not like that (one person
    appears in dozens of clips), so use 3 or more if you want the audit to
    demonstrate anything.

    Runtime: ~30-60 s for the default 200 clips.
    """
    if cv2 is None:
        raise ImportError("opencv-python is required to write the synthetic set")

    root = Path(cfg.paths.data_dir) / "synthetic"
    (root / "real").mkdir(parents=True, exist_ok=True)
    (root / "fake").mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*("FFV1" if lossless else "mp4v"))
    ext = ".avi" if lossless else ".mp4"

    def write(path: Path, frames: np.ndarray) -> None:
        vw = cv2.VideoWriter(str(path), fourcc, fps, (frames.shape[2], frames.shape[1]))
        if not vw.isOpened():
            raise RuntimeError(f"VideoWriter could not open {path} — try lossless=False")
        for f in frames:
            vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        vw.release()

    cps = max(1, int(clips_per_subject))
    for i in range(n_per_class):
        subj = i // cps                     # shared scene seed -> same identity
        clip_i = i % cps
        write(root / "real" / f"real_s{subj:04d}_c{clip_i}{ext}",
              make_clip("real", T, size, seed=seed + subj * 97 + clip_i))
        kind = fake_kinds[i % len(fake_kinds)]
        # the fake for a subject is built from the SAME scene seed as their real
        # clip — exactly the relationship a face swap has with its source video
        write(root / "fake" / f"fake_{kind}_s{subj:04d}_c{clip_i}{ext}",
              make_clip(kind, T, size, seed=seed + subj * 97 + clip_i))

    log.info(f"synthetic dataset written: {root} "
             f"({n_per_class} real + {n_per_class} fake, kinds={fake_kinds}, "
             f"{cps} clip(s) per subject)")
    print("\nNext:")
    print("  cfg.paths.custom_root = root            # or pass {'synthetic': root}")
    print("  cfg.data.min_faces_per_clip = 0.0       # there are no faces in these")
    print("  cfg.data.face_backend = 'none'          # skip detection, centre crop")
    return root


# ==========================================================================
# The controlled test
# ==========================================================================

def controlled_tsc_test(n: int = 40, T: int = 32, size: int = 224,
                        seed: int = 0, verbose: bool = True,
                        return_df: bool = False):
    """
    The experiment that justifies the TSC branch existing.

    Generates variance-matched real / smooth / flicker clips in memory and
    reports the AUC of:
      * TSC descriptors               (should separate)
      * per-frame spectral descriptor (should NOT — the marginals are matched)

    Returns a dict of AUCs — or, with `return_df=True`, `(aucs, per_clip_df)`
    so a caller can plot the underlying distributions without paying for a
    second generation pass. Called by scripts/smoke_test.py, which asserts on
    the numbers.
    """
    import torch
    from sklearn.metrics import roc_auc_score

    from .tsc import azimuthal_power_np, tsc_map_np

    rng = np.random.default_rng(seed)
    rows: List[dict] = []
    for i in range(n):
        for kind, label in [("real", 0), ("smooth", 1), ("flicker", 1)]:
            frames = make_clip(kind, T, size, seed=int(rng.integers(1 << 30)))
            tmap, spectra = tsc_map_np(frames, n_bins=64)

            K = spectra.shape[1]
            hi = 2 * K // 3
            band = tmap[1:] + 1e-6
            fine = band[:, hi:]
            whiteness = float(np.exp(np.log(fine).mean(0) - np.log(fine.mean(0))).mean())
            peak = float(fine.max(0).sum() / (fine.sum() + 1e-9))

            # the per-frame control: time-averaged spatial spectrum only
            marginal_hf = float(spectra.mean(axis=0)[hi:].mean())
            marginal_slope = float(np.polyfit(np.arange(K), spectra.mean(axis=0), 1)[0])

            rows.append(dict(kind=kind, label=label, whiteness=whiteness,
                             peakiness=peak, marginal_hf=marginal_hf,
                             marginal_slope=marginal_slope))


    import pandas as pd
    df = pd.DataFrame(rows)
    y = df.label.values

    def auc(col, invert=False):
        v = -df[col].values if invert else df[col].values
        try:
            return float(roc_auc_score(y, v))
        except ValueError:
            return float("nan")

    res = {
        # fakes are LESS white in time -> invert so higher score = fake
        "tsc_whiteness_auc": auc("whiteness", invert=True),
        "tsc_peakiness_auc": auc("peakiness"),
        "control_marginal_hf_auc": auc("marginal_hf"),
        "control_marginal_slope_auc": auc("marginal_slope"),
    }
    res["tsc_best_auc"] = max(res["tsc_whiteness_auc"], res["tsc_peakiness_auc"])
    res["control_best_auc"] = max(abs(res["control_marginal_hf_auc"] - 0.5),
                                  abs(res["control_marginal_slope_auc"] - 0.5)) + 0.5

    if verbose:
        print("\nControlled TSC test — variance-matched classes")
        print("-" * 58)
        print(f"  TSC temporal whiteness   AUC : {res['tsc_whiteness_auc']:.3f}")
        print(f"  TSC temporal peakiness   AUC : {res['tsc_peakiness_auc']:.3f}")
        print(f"  CONTROL marginal HF      AUC : {res['control_marginal_hf_auc']:.3f}")
        print(f"  CONTROL spectral slope   AUC : {res['control_marginal_slope_auc']:.3f}")
        print("-" * 58)
        print("  Expected: TSC near 1.0, controls near 0.5. The controls are the")
        print("  point — the classes have matched average spatial spectra, so a")
        print("  per-frame frequency detector cannot tell them apart and any")
        print("  separation TSC finds is genuinely temporal.\n")
        for k in ["real", "smooth", "flicker"]:
            sub = df[df.kind == k]
            print(f"  {k:<8} whiteness {sub.whiteness.mean():.4f} +- {sub.whiteness.std():.4f}"
                  f"   peakiness {sub.peakiness.mean():.4f}")
        print()
    return (res, df) if return_df else res


__all__ = ["make_clip", "build_synthetic_dataset", "controlled_tsc_test"]
