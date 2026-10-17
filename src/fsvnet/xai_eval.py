"""
xai_eval.py
===========

Measurement code for the explanation benchmark: does an explanation actually
account for the decision, and does it survive the controls?

How this relates to the CFX modules
-----------------------------------
`attribution.py` / `faithfulness.py` are the *runtime* explanation layer: they
run inside `CFXPipeline`, behind the confidence gate, so only clips the system
decides to explain get the expensive candidates. That is right for the product
and wrong for a benchmark — faithfulness measured only on confident clips is
selected on the outcome. This module is the *ungated* benchmark: every method on
every sampled clip, on one shared unit grid, under the same controls. It is
self-contained so its numbers are an independent check on CFX, not a re-run of
the same code.

Everything here acts on the model rather than on a picture of it. An
explanation is reduced to a RANKING over input units and scored by removing /
inserting those units and reading the model's output.

Input units
-----------
A G x G grid over the face crop (default 8 x 8 -> 64 units of 28 px at 224),
applied identically to every frame. The Celeb-DF crops are face-aligned, so a
cell covers roughly the same facial region throughout the clip, and every
method — Grad-CAM, IG, occlusion, KernelSHAP, random, centre prior — is scored
on exactly the same units.

Target quantity
---------------
p(target) = the model's probability for the class it predicted on the
unperturbed clip. "Evidence" means evidence for what the model said.

LMF is not pixel-reachable
--------------------------
LMF reads cached landmark-motion features, not pixels. Deleting pixels cannot
remove LMF evidence, so for C2/C3 every deletion curve has a floor set by LMF.
The runner reports each clip's pixel-reachable Shapley share so the curves are
read in context.

Controls
--------
* random ranking (3 seeds, per clip, paired)
* centre prior — rank units by distance from the image centre
* fills: Gaussian blur, dataset mean face, and CFX's roll fill (each frame
  replaced by the frame T/2 away). Blur and mean face are the primary pair.
  The roll fill is included so these numbers are comparable with cfx_eval.py;
  on face-aligned crops it replaces a cell with nearly the same content, so it
  is expected to remove little and is read as a diagnostic, not a verdict.
* blur-everything: how far p(target) falls when ALL fine detail goes
* model-randomisation sanity check (Adebayo et al., 2018)
"""

from __future__ import annotations

import copy
import itertools
import math
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CURVE_FRACTIONS = np.array([0.0, 0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50,
                            0.60, 0.70, 0.80, 0.90, 1.0])
REPORT_K = (0.05, 0.10, 0.20, 0.50)      # comprehensiveness / sufficiency points
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ==========================================================================
# model adapter
# ==========================================================================

def expand_lmf(lmf: Optional[Dict[str, torch.Tensor]], n: int):
    if lmf is None:
        return None
    return {k: v.expand(n, *v.shape[1:]) for k, v in lmf.items()}


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=np.float64)))


class ClipScorer:
    """
    Scores many perturbed versions of ONE clip.

    clip : (1, T, 3, H, W) normalised, on device
    lmf  : the clip's cached LMF pack or None. Passed on EVERY call — the repo's
           explain.py omits it, which silently zeroes LMF for C2/C3.
    """

    def __init__(self, model: nn.Module, clip: torch.Tensor,
                 lmf: Optional[Dict[str, torch.Tensor]] = None,
                 batch: int = 8, amp: bool = False):
        self.model = model.eval()
        self.clip = clip
        self.device = clip.device
        self.lmf = ({k: v.to(self.device) for k, v in lmf.items()}
                    if lmf is not None else None)
        self.batch = batch
        self.amp = amp and self.device.type == "cuda"
        self.n_forward = 0
        m0 = float(self.margins(clip)[0])
        self.margin0 = m0
        self.sign = 1.0 if m0 >= 0 else -1.0          # +1 => predicted fake
        self.p0 = float(sigmoid(self.sign * m0))

    @torch.no_grad()
    def margins(self, clips: torch.Tensor) -> np.ndarray:
        """fake-minus-real logit for each clip in (N,T,3,H,W)."""
        out = []
        for i in range(0, clips.shape[0], self.batch):
            x = clips[i:i + self.batch].to(self.device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=self.amp):
                logits, _ = self.model(x, None, lmf=expand_lmf(self.lmf, x.shape[0]))
            logits = logits.float()
            out.append((logits[:, 1] - logits[:, 0]).cpu().numpy())
            self.n_forward += x.shape[0]
        return np.concatenate(out)

    def p_target(self, clips: torch.Tensor) -> np.ndarray:
        return sigmoid(self.sign * self.margins(clips))


# ==========================================================================
# units and fills
# ==========================================================================

def unit_masks(h: int, w: int, grid: int) -> torch.Tensor:
    """(U, H, W) float masks, U = grid*grid, row-major."""
    ys = np.linspace(0, h, grid + 1).round().astype(int)
    xs = np.linspace(0, w, grid + 1).round().astype(int)
    m = torch.zeros(grid * grid, h, w)
    for r in range(grid):
        for c in range(grid):
            m[r * grid + c, ys[r]:ys[r + 1], xs[c]:xs[c + 1]] = 1.0
    return m


def to_unit_scores(attr: np.ndarray, masks: torch.Tensor) -> np.ndarray:
    """Pixel attribution (T,H,W) or (H,W) -> one score per unit (frame-mean, then cell-mean)."""
    a = np.asarray(attr, dtype=np.float64)
    if a.ndim == 3:
        a = a.mean(0)
    m = masks.numpy().reshape(masks.shape[0], -1)
    return (m @ a.reshape(-1)) / np.maximum(m.sum(1), 1)


def gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable blur over the last two dims of (..., 3, H, W)."""
    k = int(2 * math.ceil(3 * sigma) + 1)
    t = torch.arange(k, device=x.device, dtype=torch.float32) - k // 2
    g = torch.exp(-t ** 2 / (2 * sigma ** 2)); g = g / g.sum()
    shp = x.shape
    y = x.reshape(-1, 1, shp[-2], shp[-1]).float()
    y = F.conv2d(F.pad(y, (k // 2, k // 2, 0, 0), mode="reflect"), g.view(1, 1, 1, k))
    y = F.conv2d(F.pad(y, (0, 0, k // 2, k // 2), mode="reflect"), g.view(1, 1, k, 1))
    return y.reshape(shp).to(x.dtype)


def make_fill(kind: str, clip: torch.Tensor, mean_face: Optional[torch.Tensor] = None
              ) -> torch.Tensor:
    """A full replacement clip the same shape as `clip` (1,T,3,H,W)."""
    if kind == "blur":
        return gaussian_blur(clip, sigma=clip.shape[-1] / 28.0)
    if kind == "mean":
        if mean_face is None:
            raise ValueError("mean fill needs the dataset mean face")
        return mean_face.to(clip.device, clip.dtype).expand_as(clip).clone()
    if kind == "roll":                   # CFX faithfulness.in_distribution_fill
        return clip.roll(shifts=max(1, clip.shape[1] // 2), dims=1).clone()
    raise ValueError(kind)


def compose(clip: torch.Tensor, fill: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    """keep : (N, H, W) in {0,1}. Returns (N,T,3,H,W): clip where keep, fill elsewhere."""
    k = keep.to(clip.device, clip.dtype)[:, None, None]      # (N,1,1,H,W)
    return clip * k + fill * (1 - k)


# ==========================================================================
# attribution methods — each returns a (U,) unit score; higher = more support
# for the predicted class
# ==========================================================================

@torch.enable_grad()
def grad_cam_units(model, clip, lmf, sign: float, masks) -> Optional[np.ndarray]:
    """Grad-CAM on the last EfficientNet block, target = the predicted class."""
    if not getattr(model, "use", {}).get("spatial", False):
        return None
    acts, grads = {}, {}
    layer = model.spatial.features[-1]
    h1 = layer.register_forward_hook(lambda m, i, o: acts.__setitem__("a", o))
    h2 = layer.register_full_backward_hook(lambda m, gi, go: grads.__setitem__("g", go[0]))
    try:
        model.zero_grad(set_to_none=True)
        x = clip.clone().requires_grad_(True)       # guarantees a backward through the hook
        logits, _ = model(x, None, lmf=lmf)
        (sign * (logits[:, 1] - logits[:, 0])).sum().backward()
        a, g = acts["a"], grads["g"]
        cam = F.relu((g.mean(dim=(2, 3), keepdim=True) * a).sum(1, keepdim=True))
        cam = F.interpolate(cam.float(), size=clip.shape[-2:], mode="bilinear",
                            align_corners=False)[:, 0]
        return to_unit_scores(cam.detach().cpu().numpy(), masks)
    finally:
        h1.remove(); h2.remove(); model.zero_grad(set_to_none=True)


@torch.enable_grad()
def integrated_gradients_units(model, clip, lmf, sign: float, baseline, masks,
                               steps: int = 16, batch: int = 2
                               ) -> Tuple[np.ndarray, float]:
    """
    IG along the straight path baseline -> clip, midpoint rule.
    Returns (unit scores, completeness residual). Residual
    |sum(attr) - (f(x) - f(b))| / |f(x) - f(b)|; above ~0.1, raise steps.
    """
    alphas = (torch.arange(steps, dtype=torch.float32) + 0.5) / steps
    diff = clip - baseline
    total = torch.zeros_like(clip)
    for i in range(0, steps, batch):
        a = alphas[i:i + batch].to(clip.device).view(-1, 1, 1, 1, 1)
        x = (baseline + a * diff).detach().requires_grad_(True)
        model.zero_grad(set_to_none=True)
        logits, _ = model(x, None, lmf=expand_lmf(lmf, x.shape[0]))
        (sign * (logits[:, 1] - logits[:, 0])).sum().backward()
        total += x.grad.sum(0, keepdim=True).detach()
    attr = (diff * total / steps)[0].sum(1)                        # (T,H,W)
    with torch.no_grad():
        lo, _ = model(torch.cat([baseline, clip]), None, lmf=expand_lmf(lmf, 2))
        f = (sign * (lo[:, 1] - lo[:, 0])).float().cpu().numpy()
    delta = f[1] - f[0]
    resid = abs(float(attr.sum()) - delta) / max(abs(delta), 1e-6)
    model.zero_grad(set_to_none=True)
    return to_unit_scores(attr.float().cpu().numpy(), masks), float(resid)


def occlusion_units(scorer: ClipScorer, fill, masks) -> np.ndarray:
    """Leave-one-unit-out: support = drop in signed margin when the unit is filled."""
    m = scorer.margins(compose(scorer.clip, fill, 1 - masks))
    return scorer.sign * (scorer.margin0 - m)


def kernel_shap_units(scorer: ClipScorer, fill, masks, n_samples: int = 256,
                      seed: int = 0, ridge: float = 1e-3) -> np.ndarray:
    """
    KernelSHAP over the grid units. Coalition sizes are drawn in proportion to
    the Shapley kernel, so sampled rows get equal weight (as in the reference
    implementation); weighting them by the kernel again would count it twice.
    Paired (antithetic) sampling; empty/full coalitions pinned with a large
    weight so the efficiency constraint holds.
    """
    rng = np.random.default_rng(seed)
    U = masks.shape[0]
    sizes = np.arange(1, U)
    pk = (U - 1) / (sizes * (U - sizes)); pk = pk / pk.sum()
    Z = []
    for _ in range(n_samples // 2):
        s = rng.choice(sizes, p=pk)
        z = np.zeros(U); z[rng.choice(U, s, replace=False)] = 1
        Z.append(z); Z.append(1 - z)
    Z = np.array([np.zeros(U), np.ones(U)] + Z)
    keep = (torch.from_numpy(Z).float() @ masks.reshape(U, -1)).reshape(len(Z), *masks.shape[1:])
    y = scorer.sign * scorer.margins(compose(scorer.clip, fill, keep.clamp(0, 1)))
    s = Z.sum(1)
    w = np.where((s == 0) | (s == U), 1e6, 1.0)
    X = np.hstack([np.ones((len(Z), 1)), Z])
    A = X.T @ (X * w[:, None]) + ridge * np.eye(U + 1)
    return np.linalg.solve(A, X.T @ (w * y))[1:]


def centre_prior_units(grid: int) -> np.ndarray:
    c = (np.arange(grid) + 0.5) / grid - 0.5
    yy, xx = np.meshgrid(c, c, indexing="ij")
    return -np.sqrt(yy ** 2 + xx ** 2).reshape(-1)


# ==========================================================================
# faithfulness curves
# ==========================================================================

def ranking(scores: np.ndarray, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Units sorted by support, ties broken randomly (not by index)."""
    tie = (rng or np.random.default_rng(0)).random(len(scores)) * 1e-9
    return np.argsort(-(np.asarray(scores, dtype=np.float64) + tie))


def curves(scorer: ClipScorer, fill, masks, order: np.ndarray,
           fractions: np.ndarray = CURVE_FRACTIONS) -> Dict[str, np.ndarray]:
    """Deletion and insertion curves of p(target) along `order`."""
    U = masks.shape[0]
    ks = np.clip(np.round(fractions * U).astype(int), 0, U)
    cum = torch.zeros(len(ks), *masks.shape[1:])
    for j, k in enumerate(ks):
        if k:
            cum[j] = masks[order[:k]].sum(0).clamp(0, 1)
    dele = scorer.p_target(compose(scorer.clip, fill, 1 - cum))
    ins = scorer.p_target(compose(scorer.clip, fill, cum))
    return {"ks": ks, "fractions": fractions, "deletion": dele, "insertion": ins}


def summarise_curves(c: Dict[str, np.ndarray], p0: float) -> Dict[str, float]:
    """
    comp@k = p0 - p(top-k deleted)          (ERASER comprehensiveness; higher better)
    suff@k = p0 - p(only top-k kept)        (ERASER sufficiency; closer to 0 better)
    flip_frac = smallest deleted fraction at which the predicted class loses its majority
    """
    fr, d, i = c["fractions"], c["deletion"], c["insertion"]
    out = {"del_auc": float(np.trapezoid(d, fr)), "ins_auc": float(np.trapezoid(i, fr))}
    js = [int(np.argmin(np.abs(fr - k))) for k in REPORT_K]
    for k, j in zip(REPORT_K, js):
        out[f"comp@{k:g}"] = float(p0 - d[j])
        out[f"suff@{k:g}"] = float(p0 - i[j])
    out["comp_mean"] = float(np.mean([p0 - d[j] for j in js]))
    out["suff_mean"] = float(np.mean([p0 - i[j] for j in js]))
    out["suff_abs"] = float(np.mean([abs(p0 - i[j]) for j in js]))
    flipped = np.where(d < 0.5)[0]
    out["flip_frac"] = float(fr[flipped[0]]) if len(flipped) else float("nan")
    return out


# ==========================================================================
# stability and sanity
# ==========================================================================

def spearman(a, b) -> float:
    from scipy.stats import spearmanr
    a, b = np.asarray(a, float), np.asarray(b, float)
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(spearmanr(a, b).statistic)


def is_degenerate(a) -> bool:
    """A constant map (e.g. an all-zero Grad-CAM after the ReLU) ranks nothing."""
    return bool(np.std(np.asarray(a, float)) < 1e-12)


def topk_jaccard(a, b, frac: float = 0.10) -> float:
    """Top-k overlap. NaN if either map is constant — two all-zero maps would
    otherwise score a perfect 1.0 purely from argsort's index order."""
    if is_degenerate(a) or is_degenerate(b):
        return float("nan")
    k = max(1, int(round(frac * len(a))))
    A, B = set(np.argsort(-np.asarray(a))[:k]), set(np.argsort(-np.asarray(b))[:k])
    return len(A & B) / len(A | B)


def add_pixel_noise(clip: torch.Tensor, sigma: float = 0.01, seed: int = 0) -> torch.Tensor:
    """sigma in [0,1] pixel units, converted to ImageNet-normalised space."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    std = torch.tensor(IMAGENET_STD).view(1, 1, 3, 1, 1)
    n = torch.randn(clip.shape, generator=g) * sigma / std
    return clip + n.to(clip.device, clip.dtype)


def randomized_copy(model: nn.Module, level: str, seed: int = 0) -> nn.Module:
    """
    Cascading randomisation for the sanity check.
      'head'       — classifier re-initialised
      'head+block' — classifier + the last EfficientNet block (Grad-CAM's layer)
    """
    m = copy.deepcopy(model).eval()
    torch.manual_seed(seed)
    mods = [m.classifier]
    if level == "head+block" and getattr(m, "use", {}).get("spatial", False):
        mods.append(m.spatial.features[-1])
    for mod in mods:
        for sub in mod.modules():
            if hasattr(sub, "reset_parameters"):
                sub.reset_parameters()
    return m


# ==========================================================================
# counterfactuals
# ==========================================================================

SPECTRAL_GRID = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.05])


def lowpass(clip: torch.Tensor, r: float) -> torch.Tensor:
    """
    Keep spatial frequencies with radius <= r * Nyquist (0.5 cycles/px), per
    frame and channel. r >= 1 is the identity (it also keeps the diagonal
    corners, whose radius exceeds the per-axis Nyquist). H and W never change,
    so the frequency/TSC ring indices stay valid.
    """
    if r >= 1:
        return clip.clone()
    H, W = clip.shape[-2:]
    fy = torch.fft.fftfreq(H, device=clip.device).view(-1, 1)
    fx = torch.fft.fftfreq(W, device=clip.device).view(1, -1)
    keep = (torch.sqrt(fy ** 2 + fx ** 2) <= r * 0.5).float()
    return torch.fft.ifft2(torch.fft.fft2(clip.float()) * keep).real.to(clip.dtype)


def spectral_counterfactual(scorer: ClipScorer) -> Dict[str, object]:
    """
    Highest low-pass cutoff at which the verdict flips: "the evidence lives above
    r* x Nyquist". None when the verdict survives even 5% of Nyquist.
    `monotone` says whether every lower cutoff also flipped.
    """
    p = scorer.p_target(torch.cat([lowpass(scorer.clip, r) for r in SPECTRAL_GRID]))
    flips = p < 0.5
    if not flips.any():
        return {"cutoff_nyq": None, "monotone": True, "p_curve": p.tolist()}
    j = int(np.argmax(flips))
    return {"cutoff_nyq": float(SPECTRAL_GRID[j]), "monotone": bool(flips[j:].all()),
            "p_curve": p.tolist()}


def freeze_frames(clip: torch.Tensor, frozen: Sequence[int]) -> torch.Tensor:
    """Frozen frame t takes the nearest earlier non-frozen frame (else the nearest later)."""
    T = clip.shape[1]
    fz = set(int(t) for t in frozen)
    alive = [t for t in range(T) if t not in fz]
    if not alive:
        return clip.clone()
    x = clip.clone()
    for t in sorted(fz):
        prev = [a for a in alive if a < t]
        src = prev[-1] if prev else min(alive, key=lambda a: abs(a - t))
        x[:, t] = clip[:, src]
    return x


def temporal_scores(scorer: ClipScorer) -> np.ndarray:
    """Freeze-mode occlusion: support of frame t for the predicted class."""
    T = scorer.clip.shape[1]
    xs = torch.cat([freeze_frames(scorer.clip, [t]) for t in range(T)])
    return scorer.sign * (scorer.margin0 - scorer.margins(xs))


def temporal_curve(scorer: ClipScorer, order: np.ndarray,
                   ks: Sequence[int] = (1, 2, 4, 8)) -> np.ndarray:
    """Signed margin removed by freezing the top-k frames of `order`."""
    xs = torch.cat([freeze_frames(scorer.clip, order[:k]) for k in ks])
    return scorer.sign * (scorer.margin0 - scorer.margins(xs))


def temporal_counterfactual(scorer: ClipScorer, order: np.ndarray) -> Optional[int]:
    """Minimum number of frames (frozen in `order`) that flips the verdict."""
    T = scorer.clip.shape[1]
    p = scorer.p_target(torch.cat([freeze_frames(scorer.clip, order[:k]) for k in range(1, T)]))
    hit = np.where(p < 0.5)[0]
    return int(hit[0] + 1) if len(hit) else None


# ==========================================================================
# exact branch Shapley from cached embeddings
# ==========================================================================

@torch.no_grad()
def branch_shapley(model, embeddings: Dict[str, torch.Tensor]) -> Dict[str, object]:
    """
    Exact Shapley values of the fake-margin over branches, from the branch
    embeddings of one forward pass. Removing a branch = zeroing its pooled
    embedding before fusion, exactly as forward(ablate_branches=...) does, so it
    matches attribution.branch_shapley_exact at 1 forward instead of 2^K.
    Batched: every returned array is (B, K).
    """
    names = list(model.branch_names)
    E = [embeddings[n] for n in names]
    K, B = len(E), E[0].shape[0]
    v = {}
    for S in itertools.product([0, 1], repeat=K):
        embs = [e if s else torch.zeros_like(e) for e, s in zip(E, S)]
        fused, _ = model.fusion(*embs)
        lo = model.classifier(fused).float()
        v[S] = (lo[:, 1] - lo[:, 0]).cpu().numpy()
    phi = np.zeros((B, K))
    for i in range(K):
        for S in v:
            if S[i]:
                continue
            s = sum(S)
            w = math.factorial(s) * math.factorial(K - s - 1) / math.factorial(K)
            Si = tuple(1 if j == i else S[j] for j in range(K))
            phi[:, i] += w * (v[Si] - v[S])
    full, empty = v[tuple([1] * K)], v[tuple([0] * K)]
    return {"names": names, "phi": phi, "v_full": full, "v_empty": empty,
            "efficiency_residual": np.abs(phi.sum(1) - (full - empty))}


# ==========================================================================
# statistics
# ==========================================================================

def wmean(x, w=None) -> float:
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    if not ok.any():
        return float("nan")
    if w is None:
        return float(np.mean(x[ok]))
    w = np.asarray(w, float)
    return float(np.sum(x[ok] * w[ok]) / np.sum(w[ok]))


def paired_diff(a, b, w=None, n: int = 4000, seed: int = 0) -> Dict[str, float]:
    """
    Bootstrap over clips of the (optionally weighted) mean of a - b. Weights are
    inverse sampling probabilities, so an error-oversampled subsample still
    estimates the test population. p is two-sided.
    """
    d = np.asarray(a, float) - np.asarray(b, float)
    ok = np.isfinite(d)
    w = np.ones_like(d) if w is None else np.asarray(w, float)
    d, w = d[ok], w[ok]
    if len(d) < 3:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "p": float("nan"), "n": int(len(d))}
    rng = np.random.default_rng(seed)
    bs = np.empty(n)
    for b in range(n):
        i = rng.integers(0, len(d), len(d))
        bs[b] = np.sum(d[i] * w[i]) / np.sum(w[i])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"mean": float(np.sum(d * w) / np.sum(w)), "lo": float(lo), "hi": float(hi),
            "p": float(min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean()))),
            "n": int(len(d))}


def borda(table: Dict[str, Dict[str, float]], metrics: Dict[str, str]) -> Dict[str, float]:
    """Borda points per method; metrics maps name -> 'lower' | 'higher'. No weights."""
    methods = list(table)
    pts = {m: 0.0 for m in methods}
    for met, direction in metrics.items():
        vals = np.array([table[m].get(met, np.nan) for m in methods], float)
        vals = np.where(np.isfinite(vals), vals, np.inf if direction == "lower" else -np.inf)
        order = np.argsort(vals if direction == "lower" else -vals, kind="stable")
        for rank, idx in enumerate(order):
            pts[methods[idx]] += len(methods) - 1 - rank
    return pts


def ece_equal_mass(y, p, bins: int = 15) -> float:
    y = np.asarray(y); p = np.asarray(p, float)
    conf = np.maximum(p, 1 - p); corr = ((p >= .5).astype(int) == y).astype(float)
    order = np.argsort(conf); e = 0.0
    for chunk in np.array_split(order, bins):
        if len(chunk):
            e += len(chunk) / len(p) * abs(corr[chunk].mean() - conf[chunk].mean())
    return float(e)


__all__ = [n for n in dir() if not n.startswith("_")]
