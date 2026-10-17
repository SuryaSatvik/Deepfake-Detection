"""
attribution.py
==============

The candidate explanation generators that `explain.py` does not have.

`explain.py` already provides Grad-CAM (where), temporal occlusion (when),
single-branch ablation (which branch) and the percentile descriptors (which
measurement). This module adds four more, chosen because each one answers a
question the existing four cannot:

| function | question | cost |
|---|---|---|
| `branch_shapley_exact` | which branch, accounting for interactions | 2^K forwards (K=5 -> 32) |
| `integrated_gradients` | which pixels, with a path baseline | 2 x steps forwards |
| `kernel_shap_superpixel` | which regions, model-agnostically | n_samples forwards |
| `counterfactual_*` | what minimal change flips the verdict | greedy |

Two things worth knowing before using any of them
-------------------------------------------------

**Exact Shapley is cheap here and it matters.** FSV-Net has at most five
branches, so there are only 2^5 = 32 coalitions: the exact Shapley value is a
closed enumeration, not a sampled estimate. It strictly dominates the
single-branch ablation in `explain.branch_attribution`, which measures
v(all) - v(all \\ i) and therefore attributes interaction effects to whichever
branch happens to be removed. When two branches are partially redundant, single
ablation makes both look unimportant; Shapley splits the credit.

**Baseline choice decides what a pixel attribution means.** A black baseline is
far outside the distribution of 256x256 face crops, so integrated gradients
against it largely explains "this is not black". The defaults here are a blurred
copy of the input and the clip mean; `zero` is available and documented as the
wrong choice for images. Every attribution carries the baseline it used, so a
figure can never lose track of it.

Resolution note: `FrequencyBranch` and `TSCBranch` precompute a radial ring index
for one input size, so nothing in this module ever changes H or W. Downscaling is
always followed by an upscale back to the original geometry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from math import factorial
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

try:
    from .utils import log
except Exception:                                     # pragma: no cover
    import logging
    log = logging.getLogger("fsvnet")


# ==========================================================================
# shared helpers
# ==========================================================================


def fake_logit(logits: torch.Tensor) -> torch.Tensor:
    """The scalar every method perturbs: the fake-vs-real margin."""
    return logits[:, 1] - logits[:, 0]


def _forward_margin(model, clip: torch.Tensor, mask=None, lmf=None,
                    ablate: Optional[Sequence[str]] = None) -> float:
    logits, _ = model(clip, mask, ablate_branches=list(ablate) if ablate else None,
                      lmf=lmf)
    return float(fake_logit(logits)[0])


def _forward_margins_batched(model, clips: torch.Tensor, mask=None, lmf=None,
                             batch: int = 8) -> np.ndarray:
    """Run many single-clip variants through the model in batches."""
    out: List[float] = []
    n = clips.shape[0]
    for i in range(0, n, batch):
        chunk = clips[i:i + batch]
        m = None
        if mask is not None:
            m = mask.expand(chunk.shape[0], -1) if mask.shape[0] == 1 else mask[i:i + batch]
        l = None
        if lmf is not None:
            l = {k: (v.expand(chunk.shape[0], *v.shape[1:]) if v.shape[0] == 1
                     else v[i:i + batch]) for k, v in lmf.items()}
        logits, _ = model(chunk, m, lmf=l)
        out.extend(fake_logit(logits).detach().float().cpu().tolist())
    return np.asarray(out, dtype=np.float64)


def make_baseline(clip: torch.Tensor, kind: str = "blur",
                  blur_sigma: float = 6.0) -> torch.Tensor:
    """
    A reference input that means "evidence removed" without meaning
    "out of distribution".

    blur  Gaussian-blurred copy. Keeps colour, lighting and rough geometry;
          destroys the fine detail every forensic branch actually measures.
          This is the default and the right one for face crops.
    mean  the clip's own per-channel mean, broadcast. Flat, in-range, but
          destroys geometry too, so attributions also pick up "there is a face
          here".
    zero  the normalised zero tensor (ImageNet mean grey). Included because it is
          the literature default; documented here as the wrong choice for this
          data, and reported alongside `blur` rather than instead of it.
    """
    if kind == "zero":
        return torch.zeros_like(clip)
    if kind == "mean":
        return clip.mean(dim=(1, 3, 4), keepdim=True).expand_as(clip).clone()
    if kind == "blur":
        return gaussian_blur_clip(clip, sigma=blur_sigma)
    raise ValueError(f"unknown baseline kind: {kind}")


def gaussian_blur_clip(clip: torch.Tensor, sigma: float = 6.0) -> torch.Tensor:
    """Separable Gaussian blur over (H, W), applied per frame. Shape-preserving."""
    B, T, C, H, W = clip.shape
    k = int(max(3, round(sigma * 4) | 1))                     # odd kernel
    x = torch.arange(k, dtype=clip.dtype, device=clip.device) - (k - 1) / 2
    g = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    flat = clip.reshape(B * T, C, H, W)
    gh = g.view(1, 1, 1, k).expand(C, 1, 1, k)
    gv = g.view(1, 1, k, 1).expand(C, 1, k, 1)
    pad_h = (k // 2, k // 2, 0, 0)
    pad_v = (0, 0, k // 2, k // 2)
    out = F.conv2d(F.pad(flat, pad_h, mode="reflect"), gh, groups=C)
    out = F.conv2d(F.pad(out, pad_v, mode="reflect"), gv, groups=C)
    return out.reshape(B, T, C, H, W)


# ==========================================================================
# 1. Exact branch Shapley
# ==========================================================================


@dataclass
class ShapleyResult:
    branches: List[str]
    values: Dict[str, float]                      # phi_i, in logit units
    normalised: Dict[str, float]                  # |phi_i| / sum|phi|
    interactions: Dict[str, float]                # "a|b" -> interaction index
    v_full: float                                 # margin with every branch on
    v_empty: float                                # margin with every branch off
    efficiency_residual: float                    # sum(phi) - (v_full - v_empty)
    n_forwards: int
    dominant: str = ""
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"branches": self.branches, "values": self.values,
                "normalised": self.normalised, "interactions": self.interactions,
                "v_full": self.v_full, "v_empty": self.v_empty,
                "efficiency_residual": self.efficiency_residual,
                "n_forwards": self.n_forwards, "dominant": self.dominant,
                "notes": self.notes}

    def top_interactions(self, k: int = 3) -> List[Tuple[str, float]]:
        return sorted(self.interactions.items(), key=lambda kv: -abs(kv[1]))[:k]


@torch.no_grad()
def branch_shapley_exact(model, clip: torch.Tensor, mask=None, lmf=None,
                         branches: Optional[Sequence[str]] = None,
                         with_interactions: bool = True) -> ShapleyResult:
    """
    Exact Shapley values over the branches, by enumerating all 2^K coalitions.

    The value function is the fake-vs-real margin with the branches *outside* the
    coalition ablated:  v(S) = margin(ablate = all \\ S).

    Why exact rather than sampled: K <= 5, so 32 forward passes settle it. There
    is no reason to approximate, and no reason to use the single-ablation
    difference either -- that quantity is v(N) - v(N\\{i}), which is only equal to
    the Shapley value when the branches do not interact. Redundant branches are
    exactly the case where they do.

    Efficiency check: sum(phi_i) must equal v(N) - v(empty). The residual is
    returned rather than asserted, because a large residual is informative -- it
    means the forward pass is not deterministic (dropout left on, non-deterministic
    kernels) and every number here should be distrusted until that is fixed.
    """
    model.eval()
    names = list(branches) if branches is not None else list(model.branch_names)
    K = len(names)
    if K == 0:
        raise ValueError("no branches to attribute over")
    if K > 12:
        raise ValueError(f"exact Shapley over {K} branches is 2^{K} forwards; "
                         "use a sampled estimator instead")

    # v(S) for every subset S, cached by frozenset
    v: Dict[frozenset, float] = {}
    n_fwd = 0
    for r in range(K + 1):
        for S in combinations(names, r):
            ablate = [b for b in names if b not in S]
            v[frozenset(S)] = _forward_margin(model, clip, mask, lmf, ablate)
            n_fwd += 1

    full, empty = frozenset(names), frozenset()
    phi: Dict[str, float] = {}
    for i in names:
        rest = [b for b in names if b != i]
        total = 0.0
        for r in range(len(rest) + 1):
            w = factorial(r) * factorial(K - r - 1) / factorial(K)
            for S in combinations(rest, r):
                fs = frozenset(S)
                total += w * (v[fs | {i}] - v[fs])
        phi[i] = float(total)

    denom = sum(abs(x) for x in phi.values()) or 1.0
    norm = {k: abs(x) / denom for k, x in phi.items()}

    inter: Dict[str, float] = {}
    if with_interactions and K >= 2:
        for i, j in combinations(names, 2):
            rest = [b for b in names if b not in (i, j)]
            total = 0.0
            for r in range(len(rest) + 1):
                w = (factorial(r) * factorial(K - r - 2) / factorial(K - 1))
                for S in combinations(rest, r):
                    fs = frozenset(S)
                    total += w * (v[fs | {i, j}] - v[fs | {i}] - v[fs | {j}] + v[fs])
            inter[f"{i}|{j}"] = float(total)

    resid = float(sum(phi.values()) - (v[full] - v[empty]))
    notes: List[str] = []
    if abs(resid) > 1e-3 * max(1.0, abs(v[full] - v[empty])):
        notes.append(f"efficiency residual {resid:+.4g} is not numerically zero. "
                     "The forward pass is not deterministic -- check that the "
                     "model is in eval() mode and that dropout is off. Do not "
                     "report these values until it is.")
    if with_interactions and inter:
        top, val = max(inter.items(), key=lambda kv: abs(kv[1]))
        if abs(val) > 0.25 * max(abs(x) for x in phi.values()):
            notes.append(f"strong interaction between {top.replace('|', ' and ')} "
                         f"({val:+.4f}): these branches are partly redundant, so "
                         "the single-ablation numbers in explain.branch_attribution "
                         "understate both of them.")

    return ShapleyResult(
        branches=names, values=phi, normalised=norm, interactions=inter,
        v_full=float(v[full]), v_empty=float(v[empty]),
        efficiency_residual=resid, n_forwards=n_fwd,
        dominant=max(norm, key=norm.get), notes=notes)


# ==========================================================================
# 2. Integrated gradients
# ==========================================================================


@torch.enable_grad()
def integrated_gradients(model, clip: torch.Tensor, mask=None, lmf=None,
                         baseline: str = "blur", steps: int = 32,
                         batch: int = 4, signed: bool = False
                         ) -> Dict[str, object]:
    """
    Integrated gradients of the fake-margin w.r.t. the input pixels.

    Returns {"attribution": (T,H,W) float, "baseline": str,
             "completeness_error": float, "delta": float}

    `completeness_error` is the axiom check: the summed attribution should equal
    margin(input) - margin(baseline). A large error means `steps` is too small
    for this input; it is returned so a figure can state it rather than imply
    the attribution is exact.
    """
    model.eval()
    base = make_baseline(clip, baseline)
    alphas = torch.linspace(1.0 / steps, 1.0, steps, device=clip.device,
                            dtype=clip.dtype)

    total = torch.zeros_like(clip)
    for i in range(0, steps, batch):
        a = alphas[i:i + batch].view(-1, 1, 1, 1, 1)
        x = (base + a * (clip - base)).detach().requires_grad_(True)
        nb = x.shape[0]
        m = mask.expand(nb, -1) if (mask is not None and mask.shape[0] == 1) else mask
        l = None
        if lmf is not None:
            l = {k: (v.expand(nb, *v.shape[1:]) if v.shape[0] == 1 else v)
                 for k, v in lmf.items()}
        model.zero_grad(set_to_none=True)
        logits, _ = model(x, m, lmf=l)
        fake_logit(logits).sum().backward()
        total = total + x.grad.detach().sum(dim=0, keepdim=True)

    avg_grad = total / steps
    attrib = ((clip - base) * avg_grad)[0]                 # (T,C,H,W)
    summed = float(attrib.sum())

    with torch.no_grad():
        d = (_forward_margin(model, clip, mask, lmf)
             - _forward_margin(model, base, mask, lmf))

    # attrib is (T,C,H,W): collapse the colour channels, keep time, so the map
    # is comparable frame-to-frame and can be ranked as spatial units.
    a_map = attrib.sum(dim=1) if signed else attrib.abs().sum(dim=1)     # (T,H,W)
    a = a_map.detach().float().cpu().numpy()

    model.zero_grad(set_to_none=True)
    return {"attribution": a, "baseline": baseline, "steps": int(steps),
            "summed_attribution": summed, "delta": float(d),
            "completeness_error": float(abs(summed - d) / (abs(d) + 1e-9)),
            "signed": bool(signed)}


# ==========================================================================
# 3. KernelSHAP over superpixels
# ==========================================================================


def _grid_segments(H: int, W: int, n_segments: int) -> np.ndarray:
    side = max(2, int(round(np.sqrt(max(n_segments, 4)))))
    ys = (np.arange(H) * side // H).clip(0, side - 1)
    xs = (np.arange(W) * side // W).clip(0, side - 1)
    return (ys[:, None] * side + xs[None, :]).astype(np.int64)


def superpixels(frame_u8: np.ndarray, n_segments: int = 50,
                min_segments: int = 4) -> np.ndarray:
    """
    SLIC segmentation of one frame -> (H,W) int label map.

    Two degenerate cases are handled explicitly rather than left to surface as a
    confusing failure downstream:

    * **scikit-image missing** -> a regular grid. Coarser and it cuts across the
      face boundary, so the log says so.
    * **SLIC collapses to one segment.** With `enforce_connectivity=True` (the
      default) SLIC can return a single label on a low-structure or very noisy
      frame. A one-segment map makes KernelSHAP meaningless -- there is nothing
      to attribute between -- so it is retried without the connectivity
      constraint and then falls back to the grid.
    """
    H, W = frame_u8.shape[:2]
    try:
        from skimage.segmentation import slic
    except Exception as e:
        log.warning(f"scikit-image unavailable ({e.__class__.__name__}); "
                    "falling back to a regular grid, which is a coarser "
                    "explanation unit than SLIC superpixels")
        return _grid_segments(H, W, n_segments)

    for kwargs in ({}, {"enforce_connectivity": False}):
        try:
            lab = slic(frame_u8, n_segments=n_segments, compactness=10.0,
                       start_label=0, channel_axis=-1, **kwargs).astype(np.int64)
        except Exception as e:
            log.warning(f"SLIC failed ({e.__class__.__name__}: {e})")
            break
        if len(np.unique(lab)) >= min_segments:
            return lab
        log.warning(f"SLIC returned {len(np.unique(lab))} segment(s) for a "
                    f"request of {n_segments} -- the frame has too little "
                    "structure for it to seed"
                    + (" ; retrying without the connectivity constraint"
                       if not kwargs else " ; falling back to a regular grid"))
    return _grid_segments(H, W, n_segments)


@torch.no_grad()
def kernel_shap_superpixel(model, clip: torch.Tensor, frames_u8: np.ndarray,
                           frame_indices: Sequence[int],
                           mask=None, lmf=None, n_segments: int = 50,
                           n_samples: int = 512, baseline: str = "blur",
                           batch: int = 8, seed: int = 42
                           ) -> Dict[str, object]:
    """
    Model-agnostic region attributions on selected frames.

    Only the named frames are perturbed; the rest of the clip is left intact.
    Perturbing the whole clip at once would need 50*T units and the same sample
    budget, which buys a much noisier fit. Pick the frames temporal occlusion
    ranks highest.

    Returns {"per_frame": {t: {"labels": (H,W), "values": (M,)}},
             "n_forwards": int, "r2": {t: float}}
    """
    model.eval()
    rng = np.random.default_rng(seed)
    base_clip = make_baseline(clip, baseline)
    out: Dict[int, Dict[str, np.ndarray]] = {}
    r2: Dict[int, float] = {}
    n_fwd = 0

    for t in frame_indices:
        t = int(t)
        labels = superpixels(frames_u8[t], n_segments)
        uniq = np.unique(labels)
        M = len(uniq)
        lab_t = torch.from_numpy(labels).to(clip.device)

        if M < 2:
            # Nothing to attribute between. Returning a single trivial value is
            # honest; fabricating a distribution over one region would not be.
            log.warning(f"frame {t} segmented into {M} region(s) -- KernelSHAP "
                        "has nothing to attribute between and is skipped")
            out[t] = {"labels": labels, "values": np.zeros(M, dtype=np.float64),
                      "segment_ids": uniq}
            r2[t] = float("nan")
            continue

        # coalitions: all-on and all-off are pinned, the rest sampled from the
        # Shapley kernel's support (subset sizes weighted towards the extremes)
        Z = np.zeros((n_samples, M), dtype=np.float64)
        Z[0, :] = 1.0
        sizes = np.arange(1, M)
        w_size = (M - 1) / (sizes * (M - sizes))
        w_size = w_size / w_size.sum()
        for i in range(1, n_samples):
            s = rng.choice(sizes, p=w_size)
            on = rng.choice(M, size=int(s), replace=False)
            Z[i, on] = 1.0

        variants = []
        for i in range(n_samples):
            x = clip.clone()
            off = np.where(Z[i] == 0.0)[0]
            if len(off):
                m2 = torch.isin(lab_t, torch.as_tensor(uniq[off], device=clip.device))
                x[0, t, :, m2] = base_clip[0, t, :, m2]
            variants.append(x)
        variants_t = torch.cat(variants, dim=0)
        y = _forward_margins_batched(model, variants_t, mask, lmf, batch=batch)
        n_fwd += n_samples

        # weighted ridge regression with the Shapley kernel
        counts = Z.sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            w = (M - 1) / (
                np.array([_nchoosek(M, int(c)) for c in counts]) * counts * (M - counts))
        w[~np.isfinite(w)] = 0.0
        big = 1e6 * (np.isclose(counts, 0) | np.isclose(counts, M))
        w = w + big                                    # pin the two endpoints

        X = np.hstack([Z, np.ones((n_samples, 1))])
        W = np.diag(w)
        reg = 1e-6 * np.eye(X.shape[1])
        beta = np.linalg.solve(X.T @ W @ X + reg, X.T @ W @ y)
        pred = X @ beta
        ss_res = float(np.sum(w * (y - pred) ** 2))
        ss_tot = float(np.sum(w * (y - np.average(y, weights=w)) ** 2))
        r2[t] = float(1.0 - ss_res / (ss_tot + 1e-12))
        out[t] = {"labels": labels, "values": beta[:M].astype(np.float64),
                  "segment_ids": uniq}

    notes = []
    poor = [t for t, v in r2.items() if v < 0.5]
    if poor:
        notes.append(f"weighted R^2 below 0.5 on frames {poor}: the additive "
                     "surrogate does not fit this model well there, so those "
                     "region values are not trustworthy. Raise n_samples or "
                     "treat the frame as unexplained.")
    return {"per_frame": out, "n_forwards": n_fwd, "r2": r2,
            "baseline": baseline, "notes": notes}


def _nchoosek(n: int, k: int) -> float:
    from math import comb
    try:
        return float(comb(n, k))
    except ValueError:
        return float("inf")


# ==========================================================================
# 4. Counterfactuals -- the minimal change that flips the verdict
# ==========================================================================


@dataclass
class Counterfactual:
    kind: str
    flipped: bool
    cost: float                       # interpretable size of the edit
    cost_unit: str
    detail: Dict[str, object] = field(default_factory=dict)
    n_forwards: int = 0
    statement: str = ""

    def to_dict(self) -> dict:
        return {"kind": self.kind, "flipped": self.flipped, "cost": self.cost,
                "cost_unit": self.cost_unit, "detail": self.detail,
                "n_forwards": self.n_forwards, "statement": self.statement}


@torch.no_grad()
def counterfactual_spectral(model, clip: torch.Tensor, mask=None, lmf=None,
                            cutoffs: Optional[Sequence[float]] = None,
                            img_size: Optional[int] = None) -> Counterfactual:
    """
    Low-pass the frames and find the highest cutoff that still flips the verdict.

    This is the most directly interpretable output in the whole module, because
    the answer is a statement about the *video* rather than about the network:
    "the evidence the detector used lives above 0.30 of Nyquist, i.e. in detail
    finer than about 3 pixels." It also connects straight to the frequency and
    TSC branches, which is what makes it worth reporting over a pixel heatmap.

    Shape is preserved throughout (the filter is applied in the Fourier domain
    and inverted), so the branches' precomputed ring indices stay valid.
    """
    model.eval()
    base_m = _forward_margin(model, clip, mask, lmf)
    want_fake = base_m > 0
    cuts = list(cutoffs) if cutoffs is not None else [0.9, 0.7, 0.5, 0.4, 0.3,
                                                     0.2, 0.15, 0.1, 0.05]
    B, T, C, H, W = clip.shape
    fy = torch.fft.fftfreq(H, device=clip.device).view(-1, 1)
    fx = torch.fft.fftfreq(W, device=clip.device).view(1, -1)
    radius = torch.sqrt(fy ** 2 + fx ** 2) / 0.5                 # 1.0 == Nyquist

    n_fwd = 1
    flip_at: Optional[float] = None
    trace: List[Dict[str, float]] = []
    for c in cuts:
        keep = (radius <= c).to(clip.dtype)
        spec = torch.fft.fft2(clip.reshape(B * T * C, H, W))
        filt = torch.fft.ifft2(spec * keep).real.reshape(B, T, C, H, W)
        m = _forward_margin(model, filt, mask, lmf)
        n_fwd += 1
        trace.append({"cutoff": float(c), "margin": float(m)})
        flipped = (m <= 0) if want_fake else (m > 0)
        if flipped:
            flip_at = float(c)
            break

    px = (1.0 / flip_at / 2.0) if flip_at else float("nan")
    statement = (
        f"Removing spatial detail finer than ~{px:.1f} px "
        f"(above {flip_at:.2f} of Nyquist) flips the verdict."
        if flip_at is not None else
        "Low-pass filtering down to 0.05 of Nyquist never flipped the verdict: "
        "the decision does not rest on fine spatial detail.")
    return Counterfactual(kind="spectral", flipped=flip_at is not None,
                          cost=float(flip_at) if flip_at else float("nan"),
                          cost_unit="fraction of Nyquist",
                          detail={"trace": trace, "base_margin": base_m,
                                  "approx_pixel_scale": px},
                          n_forwards=n_fwd, statement=statement)


@torch.no_grad()
def counterfactual_temporal(model, clip: torch.Tensor, mask=None, lmf=None,
                            order: Optional[Sequence[int]] = None,
                            max_frames: Optional[int] = None) -> Counterfactual:
    """
    Freeze frames one at a time, in order of importance, until the verdict flips.

    "Freeze" means replacing frame t with frame t-1, the same choice
    `explain.temporal_occlusion` makes and for the same reason: blanking a frame
    manufactures a discontinuity that a temporal model reacts to regardless of
    whether that frame carried evidence.

    The cost is the number of frames that had to be frozen -- a small number
    means the decision rests on a couple of moments, which is a useful thing to
    be able to say and a suspicious thing if it is 1 out of 16.
    """
    model.eval()
    T = clip.shape[1]
    base_m = _forward_margin(model, clip, mask, lmf)
    want_fake = base_m > 0
    n_fwd = 1

    if order is None:
        deltas = []
        for t in range(T):
            x = clip.clone()
            x[:, t] = clip[:, max(t - 1, 0)] if t > 0 else clip[:, min(1, T - 1)]
            deltas.append(base_m - _forward_margin(model, x, mask, lmf))
            n_fwd += 1
        order = list(np.argsort(-np.asarray(deltas)))

    cap = max_frames or T
    x = clip.clone()
    frozen: List[int] = []
    flipped = False
    for t in list(order)[:cap]:
        t = int(t)
        x[:, t] = clip[:, max(t - 1, 0)] if t > 0 else clip[:, min(1, T - 1)]
        frozen.append(t)
        m = _forward_margin(model, x, mask, lmf)
        n_fwd += 1
        if (m <= 0) if want_fake else (m > 0):
            flipped = True
            break

    statement = (
        f"Freezing {len(frozen)} of {T} frames ({frozen}) flips the verdict."
        if flipped else
        f"Freezing all {T} frames never flipped the verdict: the decision is not "
        "carried by any small set of moments.")
    return Counterfactual(kind="temporal", flipped=flipped,
                          cost=float(len(frozen)) if flipped else float("nan"),
                          cost_unit="frames frozen",
                          detail={"frames": frozen, "order": [int(i) for i in order],
                                  "base_margin": base_m, "T": int(T)},
                          n_forwards=n_fwd, statement=statement)


@torch.no_grad()
def counterfactual_region(model, clip: torch.Tensor, frames_u8: np.ndarray,
                          ranking: Sequence[Tuple[int, int]],
                          labels_per_frame: Dict[int, np.ndarray],
                          mask=None, lmf=None, baseline: str = "blur",
                          max_regions: int = 40) -> Counterfactual:
    """
    Greedily replace the highest-ranked regions with the baseline until the
    verdict flips. `ranking` is a list of (frame, segment_id) in descending
    importance, as produced by `kernel_shap_superpixel`.

    Cost is the fraction of the clip's pixels that had to be replaced -- a
    resolution-independent number, so it is comparable across settings.
    """
    model.eval()
    base_clip = make_baseline(clip, baseline)
    base_m = _forward_margin(model, clip, mask, lmf)
    want_fake = base_m > 0
    n_fwd = 1

    x = clip.clone()
    total_px = float(clip.shape[1] * clip.shape[3] * clip.shape[4])
    used_px = 0.0
    used: List[Tuple[int, int]] = []
    flipped = False

    for (t, seg) in list(ranking)[:max_regions]:
        t = int(t)
        labels = labels_per_frame.get(t)
        if labels is None:
            continue
        m2 = torch.from_numpy(labels == seg).to(clip.device)
        x[0, t, :, m2] = base_clip[0, t, :, m2]
        used_px += float(m2.sum().item())
        used.append((t, int(seg)))
        m = _forward_margin(model, x, mask, lmf)
        n_fwd += 1
        if (m <= 0) if want_fake else (m > 0):
            flipped = True
            break

    frac = used_px / total_px
    statement = (
        f"Blurring {len(used)} region(s) -- {frac:.1%} of the clip's pixels -- "
        "flips the verdict."
        if flipped else
        f"Blurring the top {len(used)} regions ({frac:.1%} of pixels) did not "
        "flip the verdict; the evidence is spatially diffuse.")
    return Counterfactual(kind="region", flipped=flipped,
                          cost=float(frac) if flipped else float("nan"),
                          cost_unit="fraction of pixels replaced",
                          detail={"regions": used, "base_margin": base_m},
                          n_forwards=n_fwd, statement=statement)


__all__ = ["fake_logit", "make_baseline", "gaussian_blur_clip",
           "ShapleyResult", "branch_shapley_exact", "integrated_gradients",
           "superpixels", "kernel_shap_superpixel", "Counterfactual",
           "counterfactual_spectral", "counterfactual_temporal",
           "counterfactual_region"]
