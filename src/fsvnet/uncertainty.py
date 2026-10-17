"""
uncertainty.py
==============

Three cheap ways to ask the model how sure it is, and one expensive one.

| source | cost | what it captures |
|---|---|---|
| window spread | free | the clip windows of one video already disagree; nobody was looking |
| TTA | 6 forwards | input sensitivity -- and a robustness result in its own right |
| MC dropout | N forwards | epistemic uncertainty from the dropout the model already has |

What is deliberately NOT here
-----------------------------
A "deep ensemble" over `checkpoints/dfe_B0 / C1 / C2 / C3`. Those are ablation
variants -- they differ in which branches exist, not in random seed -- so their
disagreement measures the ablation, not the uncertainty. Averaging them would
produce a confident-looking number with no interpretation. Their disagreement is
used in `arbitrate.py` as corroboration between specialists, which is what it
actually is.

Resolution safety
-----------------
`FrequencyBranch` and `TSCBranch` precompute a radial ring index for a single
input size, and feeding them a different size raises (that was the `index_add_`
crash in an earlier session). So the scale-jitter transform downscales and then
upscales back to the original geometry: it simulates resolution loss without ever
changing H or W.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

try:
    from .utils import log
except Exception:                                     # pragma: no cover
    import logging
    log = logging.getLogger("fsvnet")

try:
    from .dataset import IMAGENET_MEAN, IMAGENET_STD
except Exception:                                     # pragma: no cover
    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _mean_std(device, dtype):
    m = torch.as_tensor(np.asarray(IMAGENET_MEAN).reshape(1, 1, 3, 1, 1),
                        device=device, dtype=dtype)
    s = torch.as_tensor(np.asarray(IMAGENET_STD).reshape(1, 1, 3, 1, 1),
                        device=device, dtype=dtype)
    return m, s


def denorm(clip: torch.Tensor) -> torch.Tensor:
    m, s = _mean_std(clip.device, clip.dtype)
    return (clip * s + m).clamp(0, 1)


def renorm(x01: torch.Tensor) -> torch.Tensor:
    m, s = _mean_std(x01.device, x01.dtype)
    return (x01 - m) / s


# ==========================================================================
# test-time augmentation
# ==========================================================================


def tta_hflip(clip): return torch.flip(clip, dims=[-1])


def tta_gamma(clip, gamma: float):
    return renorm(denorm(clip).clamp_min(1e-6) ** gamma)


def tta_downscale(clip, factor: float):
    """Downscale then restore the original size -- shape-preserving by design."""
    B, T, C, H, W = clip.shape
    h, w = max(8, int(H * factor)), max(8, int(W * factor))
    f = clip.reshape(B * T, C, H, W)
    small = F.interpolate(f, size=(h, w), mode="area")
    back = F.interpolate(small, size=(H, W), mode="bilinear", align_corners=False)
    return back.reshape(B, T, C, H, W)


def tta_jpeg(clip, quality: int):
    """
    True JPEG round-trip, which is the transform that matters most here: the
    compression shortcut is the classic way a deepfake detector cheats, so its
    sensitivity to quantisation is worth measuring rather than approximating.
    Silently skipped (returns the input) if Pillow is unavailable.
    """
    try:
        import io
        from PIL import Image
    except Exception:
        return clip
    x = (denorm(clip)[0].permute(0, 2, 3, 1).detach().cpu().numpy() * 255
         ).astype(np.uint8)
    out = np.empty_like(x)
    for t in range(x.shape[0]):
        buf = io.BytesIO()
        Image.fromarray(x[t]).save(buf, format="JPEG", quality=int(quality))
        buf.seek(0)
        out[t] = np.asarray(Image.open(buf).convert("RGB"))
    y = torch.from_numpy(out).to(clip.device, clip.dtype).permute(0, 3, 1, 2) / 255.0
    return renorm(y.unsqueeze(0))


DEFAULT_TTA: Tuple[Tuple[str, Callable], ...] = (
    ("identity", lambda c: c),
    ("hflip", tta_hflip),
    ("gamma_0.9", lambda c: tta_gamma(c, 0.9)),
    ("gamma_1.1", lambda c: tta_gamma(c, 1.1)),
    ("downscale_0.75", lambda c: tta_downscale(c, 0.75)),
    ("jpeg_60", lambda c: tta_jpeg(c, 60)),
)


# ==========================================================================
# report
# ==========================================================================


@dataclass
class UncertaintyReport:
    p_point: float                       # the single forward pass
    p_mean: float                        # mean over whatever passes were run
    window_sd: float = 0.0               # across clip windows of one video
    tta_sd: float = 0.0
    mc_sd: float = 0.0
    total_sd: float = 0.0
    n_passes: int = 1
    per_transform: Dict[str, float] = field(default_factory=dict)
    window_probs: List[float] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@torch.no_grad()
def predict_prob(model, clip, mask=None, lmf=None) -> float:
    logits, _ = model(clip, mask, lmf=lmf)
    return float(torch.softmax(logits.float(), 1)[0, 1])


def window_spread(probs: Sequence[float]) -> float:
    """
    SD of the per-window probabilities of one video.

    This is free -- `clips_per_video_eval` is already 2 and the trainer already
    averages them -- and it is the most interpretable uncertainty in the system:
    if two windows of the same video disagree, the verdict is about a moment, not
    about the video.
    """
    p = np.asarray(list(probs), dtype=np.float64)
    return float(p.std(ddof=1)) if len(p) > 1 else 0.0


@torch.no_grad()
def tta_probs(model, clip, mask=None, lmf=None,
              transforms: Sequence[Tuple[str, Callable]] = DEFAULT_TTA
              ) -> Dict[str, float]:
    model.eval()
    out: Dict[str, float] = {}
    for name, fn in transforms:
        try:
            out[name] = predict_prob(model, fn(clip), mask, lmf)
        except Exception as e:
            log.warning(f"TTA '{name}' failed ({e.__class__.__name__}: {e}) -- skipped")
    return out


def _enable_dropout(model) -> int:
    n = 0
    for m in model.modules():
        if m.__class__.__name__.startswith("Dropout"):
            m.train()
            n += 1
    return n


@torch.no_grad()
def mc_dropout_probs(model, clip, mask=None, lmf=None, n: int = 20,
                     seed: int = 0) -> List[float]:
    """
    N stochastic forward passes with dropout left on at inference.

    Returns [] and logs if the model has no dropout layers -- silently returning
    zeros would report "no epistemic uncertainty" when the truth is "not
    measured".
    """
    model.eval()
    k = _enable_dropout(model)
    if k == 0:
        log.warning("mc_dropout: the model has no dropout layers, so this "
                    "measures nothing. Returning no samples rather than zeros.")
        return []
    torch.manual_seed(seed)
    try:
        return [predict_prob(model, clip, mask, lmf) for _ in range(n)]
    finally:
        model.eval()


def estimate_uncertainty(model, clip, mask=None, lmf=None,
                         window_probs: Optional[Sequence[float]] = None,
                         do_tta: bool = True, mc_passes: int = 0,
                         transforms: Sequence[Tuple[str, Callable]] = DEFAULT_TTA
                         ) -> UncertaintyReport:
    """
    The default recommended pair is window spread + TTA: both nearly free, both
    interpretable, and TTA doubles as a reportable robustness probe. MC dropout
    is off by default (`mc_passes=0`) because 20x the forwards buys a number that
    is harder to explain than the two above.
    """
    p0 = predict_prob(model, clip, mask, lmf)
    n_passes = 1
    notes: List[str] = []

    per_t: Dict[str, float] = {}
    tta_sd = 0.0
    if do_tta:
        per_t = tta_probs(model, clip, mask, lmf, transforms)
        n_passes += len(per_t)
        if len(per_t) > 1:
            tta_sd = float(np.std(list(per_t.values()), ddof=1))
        if "jpeg_60" in per_t and abs(per_t["jpeg_60"] - p0) > 0.25:
            notes.append(f"a single JPEG round-trip at q=60 moves p(fake) by "
                         f"{per_t['jpeg_60'] - p0:+.3f}. The decision is partly "
                         "a compression artefact, not a manipulation artefact.")

    mc: List[float] = []
    if mc_passes > 0:
        mc = mc_dropout_probs(model, clip, mask, lmf, mc_passes)
        n_passes += len(mc)

    wsd = window_spread(window_probs) if window_probs else 0.0
    if window_probs and wsd > 0.2:
        notes.append(f"the clip windows of this video disagree (sd {wsd:.3f}). "
                     "The verdict describes a moment, not the video.")

    pool = [p0] + list(per_t.values()) + list(mc)
    total = float(np.sqrt(wsd ** 2 + tta_sd ** 2 +
                          (float(np.std(mc, ddof=1)) if len(mc) > 1 else 0.0) ** 2))
    return UncertaintyReport(
        p_point=p0, p_mean=float(np.mean(pool)), window_sd=wsd, tta_sd=tta_sd,
        mc_sd=float(np.std(mc, ddof=1)) if len(mc) > 1 else 0.0,
        total_sd=total, n_passes=n_passes, per_transform=per_t,
        window_probs=[float(x) for x in (window_probs or [])], notes=notes)


__all__ = ["UncertaintyReport", "predict_prob", "window_spread", "tta_probs",
           "mc_dropout_probs", "estimate_uncertainty", "DEFAULT_TTA",
           "tta_hflip", "tta_gamma", "tta_downscale", "tta_jpeg",
           "denorm", "renorm"]
