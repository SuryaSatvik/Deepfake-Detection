"""
explain.py
==========

A general explainability layer for FSV-Net: **where** in the frame, **when** in
the clip, and **which branch** drove the decision.

Four methods, each answering a different question
--------------------------------------------------

| method | question | honest limitation |
|---|---|---|
| `grad_cam` | *where* in the frame? | spatial branch only — the forensic branches have no spatial map worth showing |
| `temporal_occlusion` | *when* in the clip? | model-agnostic; costs T forward passes |
| `branch_attribution` | *which branch did the answer depend on?* | the honest version of the gate weights |
| `descriptor_report` | *which measured quantity was abnormal?* | needs the reference distribution |

Why `branch_attribution` exists, and why the gate weights are not enough
------------------------------------------------------------------------
`BranchAttentionFusion` produces a softmax weight per branch, and it is
tempting to report it as "the model used TSC 38% of the time". That is not what
it measures. The gate says what the fusion *scaled up*; it says nothing about
whether the logit would change if that branch were removed — a branch can carry
a large gate weight and contribute nothing, or a small one and be decisive,
because the classifier after the gate can rescale anything.

So this module measures dependence directly: zero one branch's embedding, re-run,
and read how far the fake-vs-real logit moved. That number is what belongs in
the report; the gate weight is a diagnostic that goes beside it, and where the
two disagree, **that disagreement is itself worth a sentence** in the write-up.

Everything here works on a single clip and never needs gradients except
`grad_cam`, so it runs comfortably inside the Streamlit app on an 8 GB card.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .dataset import IMAGENET_MEAN, IMAGENET_STD
from .utils import log


# ==========================================================================
# helpers
# ==========================================================================

def denormalise(clip: torch.Tensor) -> np.ndarray:
    """(T,3,H,W) normalised tensor -> (T,H,W,3) uint8 for display."""
    x = clip.detach().cpu().numpy().transpose(0, 2, 3, 1)
    x = x * IMAGENET_STD + IMAGENET_MEAN
    return (np.clip(x, 0, 1) * 255).astype(np.uint8)


def _fake_logit(logits: torch.Tensor) -> torch.Tensor:
    """Margin for the 'fake' class — the quantity every method perturbs."""
    return logits[:, 1] - logits[:, 0]


# ==========================================================================
# 1. Grad-CAM on the spatial branch
# ==========================================================================

@torch.enable_grad()
def grad_cam(model, clip: torch.Tensor, mask: Optional[torch.Tensor] = None,
             target: str = "fake", upsample: bool = True) -> Optional[np.ndarray]:
    """
    Class-activation map over the EfficientNet trunk, per frame.

    clip : (1, T, 3, H, W)
    returns (T, H, W) float in [0,1], or None if the spatial branch is disabled.

    Read it as "which pixels the *semantic* branch reacted to". It says nothing
    about the frequency, SRM or TSC branches — those do not have a meaningful
    spatial map, and inventing one for them would be decoration, not evidence.
    """
    if not model.use.get("spatial"):
        log.warning("grad_cam: the spatial branch is disabled — nothing to map")
        return None

    model.eval()
    acts: Dict[str, torch.Tensor] = {}
    grads: Dict[str, torch.Tensor] = {}

    layer = model.spatial.features[-1]        # last conv block

    h1 = layer.register_forward_hook(
        lambda m, i, o: acts.__setitem__("a", o))
    h2 = layer.register_full_backward_hook(
        lambda m, gi, go: grads.__setitem__("g", go[0]))

    try:
        clip = clip.clone().requires_grad_(False)
        model.zero_grad(set_to_none=True)
        logits, _ = model(clip, mask)
        score = _fake_logit(logits).sum() * (1.0 if target == "fake" else -1.0)
        score.backward()

        if "a" not in acts or "g" not in grads:
            log.warning("grad_cam: hooks captured nothing")
            return None

        a = acts["a"]                          # (T, C, h, w)
        g = grads["g"]
        weights = g.mean(dim=(2, 3), keepdim=True)          # GAP over space
        cam = F.relu((weights * a).sum(dim=1, keepdim=True))  # (T,1,h,w)

        if upsample:
            cam = F.interpolate(cam, size=clip.shape[-2:], mode="bilinear",
                                align_corners=False)
        cam = cam.squeeze(1).detach().cpu().numpy()

        # normalise per frame so frames are comparable to each other
        out = np.empty_like(cam)
        for t in range(cam.shape[0]):
            c = cam[t]
            lo, hi = float(c.min()), float(c.max())
            out[t] = (c - lo) / (hi - lo) if hi > lo else np.zeros_like(c)
        return out
    finally:
        h1.remove(); h2.remove()
        model.zero_grad(set_to_none=True)


def overlay_cam(frame_u8: np.ndarray, cam: np.ndarray, alpha: float = 0.45
                ) -> np.ndarray:
    """Blend a CAM over a frame using the project's sequential blue ramp."""
    import matplotlib
    from .viz import sequential_cmap

    heat = sequential_cmap()(np.clip(cam, 0, 1))[..., :3]
    heat = (heat * 255).astype(np.uint8)
    return (frame_u8 * (1 - alpha) + heat * alpha).astype(np.uint8)


# ==========================================================================
# 2. Temporal occlusion — *when* in the clip?
# ==========================================================================

@torch.no_grad()
def temporal_occlusion(model, clip: torch.Tensor,
                       mask: Optional[torch.Tensor] = None,
                       mode: str = "freeze") -> Dict[str, np.ndarray]:
    """
    Replace one frame at a time and measure how far the fake-logit moves.

    mode:
      "freeze" — replace frame t with frame t-1 (removes that frame's *novelty*
                 while keeping the clip statistically plausible). This is the
                 right default for a temporal model: blanking a frame would
                 create an artificial discontinuity and every frame would look
                 important.
      "blank"  — replace with the clip mean (a cruder, more standard occlusion)

    returns {"delta": (T,), "baseline": float, "occluded": (T,)}
    Positive delta = removing that frame made the clip look *less* fake, i.e.
    that frame carried fake-evidence.
    """
    model.eval()
    base_logits, _ = model(clip, mask)
    base = float(_fake_logit(base_logits)[0])

    T = clip.shape[1]
    occ = np.zeros(T, dtype=np.float32)
    mean_frame = clip.mean(dim=1, keepdim=True)

    for t in range(T):
        x = clip.clone()
        if mode == "freeze" and t > 0:
            x[:, t] = clip[:, t - 1]
        elif mode == "freeze" and t == 0 and T > 1:
            x[:, 0] = clip[:, 1]
        else:
            x[:, t] = mean_frame[:, 0]
        logits, _ = model(x, mask)
        occ[t] = float(_fake_logit(logits)[0])

    return {"baseline": base, "occluded": occ, "delta": base - occ}


# ==========================================================================
# 3. Branch attribution — *which branch did the answer depend on?*
# ==========================================================================

@torch.no_grad()
def branch_attribution(model, clip: torch.Tensor,
                       mask: Optional[torch.Tensor] = None) -> Dict[str, dict]:
    """
    For every active branch: zero its embedding, re-read the fake-logit, and
    report the drop. Also reports the gate weight beside it, and flags where
    the two tell different stories.

    Returns {branch: {"gate": float, "delta_logit": float,
                      "delta_prob": float, "share": float}}
    """
    model.eval()
    base_logits, info = model(clip, mask)
    base = float(_fake_logit(base_logits)[0])
    base_p = float(torch.softmax(base_logits.float(), 1)[0, 1])

    gates = {n: float(w) for n, w in
             zip(info["branch_names"], info["branch_weights"][0])}

    out: Dict[str, dict] = {}
    for name in model.branch_names:
        logits, _ = model(clip, mask, ablate_branches=[name])
        d = base - float(_fake_logit(logits)[0])
        dp = base_p - float(torch.softmax(logits.float(), 1)[0, 1])
        out[name] = {"gate": gates.get(name, float("nan")),
                     "delta_logit": d, "delta_prob": dp}

    total = sum(abs(v["delta_logit"]) for v in out.values()) or 1.0
    for v in out.values():
        v["share"] = abs(v["delta_logit"]) / total

    # flag gate/dependence disagreement — this is the interesting case
    for name, v in out.items():
        v["disagrees"] = bool(
            (v["gate"] > 0.35 and v["share"] < 0.10) or
            (v["gate"] < 0.15 and v["share"] > 0.40))
    return out


# ==========================================================================
# 4. Everything, in one object
# ==========================================================================

@dataclass
class Explanation:
    fake_probability: float
    verdict: str
    frames: np.ndarray                    # (T,H,W,3) uint8
    cam: Optional[np.ndarray] = None      # (T,H,W)
    occlusion: Optional[Dict[str, np.ndarray]] = None
    attribution: Dict[str, dict] = field(default_factory=dict)
    tsc_map: Optional[np.ndarray] = None
    srm_energy: Optional[np.ndarray] = None
    frame_attention: Optional[np.ndarray] = None
    descriptors: Dict[str, float] = field(default_factory=dict)
    percentiles: Dict[str, Optional[float]] = field(default_factory=dict)

    def top_frames(self, k: int = 3) -> List[int]:
        """Frames that carried the most fake-evidence, by occlusion."""
        if self.occlusion is None:
            return list(range(min(k, len(self.frames))))
        return np.argsort(-self.occlusion["delta"])[:k].tolist()

    def summary_lines(self) -> List[str]:
        """Plain-English bullets, every one traceable to a measured number."""
        lines = [f"Verdict: {self.verdict} (p(fake) = {self.fake_probability:.3f})."]
        if self.attribution:
            top = max(self.attribution, key=lambda k: self.attribution[k]["share"])
            a = self.attribution[top]
            lines.append(
                f"The decision depended most on the {top} branch: removing it moves "
                f"p(fake) by {a['delta_prob']:+.3f} ({a['share']*100:.0f}% of the "
                f"total dependence).")
            dis = [k for k, v in self.attribution.items() if v.get("disagrees")]
            if dis:
                lines.append(
                    f"Note: {', '.join(dis)} — the attention gate and the measured "
                    "dependence disagree here, so read the gate with caution.")
        if self.occlusion is not None:
            tf = self.top_frames(3)
            lines.append(f"Most informative frames: {tf} "
                         f"(max shift {float(self.occlusion['delta'].max()):+.3f} logits).")
        anomalous = [k for k, p in self.percentiles.items()
                     if p is not None and (p <= 5 or p >= 95)]
        if anomalous:
            lines.append("Measurements outside the 5th–95th percentile of real "
                         f"videos: {', '.join(anomalous)}.")
        elif self.percentiles:
            lines.append("No temporal-spectral measurement fell outside the "
                         "5th–95th percentile of real videos.")
        return lines


def explain_clip(model, clip: torch.Tensor, mask: Optional[torch.Tensor] = None,
                 reference=None, do_cam: bool = True, do_occlusion: bool = True,
                 occlusion_mode: str = "freeze") -> Explanation:
    """
    Run every explanation method on one clip: (1, T, 3, H, W).

    Cost: 1 forward for the verdict, 1 forward+backward for Grad-CAM, T forwards
    for occlusion, and one per branch for attribution. On a 4060 with T=16 that
    is a couple of seconds; set `do_occlusion=False` for an instant answer.
    """
    model.eval()
    with torch.no_grad():
        logits, info = model(clip, mask, return_evidence=True)
        p = float(torch.softmax(logits.float(), 1)[0, 1])

    ev = info.get("evidence", {})
    desc_t = ev.get("tsc_descriptors", {})
    descriptors = {k: float(v[0]) for k, v in desc_t.items()}
    percentiles = ({k: reference.percentile_of(k, v) for k, v in descriptors.items()}
                   if reference is not None else {})

    fw = info.get("frame_weights", {})
    frame_attn = (next(iter(fw.values()))[0].float().cpu().numpy()
                  if fw else None)

    return Explanation(
        fake_probability=p,
        verdict="fake" if p >= 0.5 else "real",
        frames=denormalise(clip[0]),
        cam=grad_cam(model, clip, mask) if do_cam else None,
        occlusion=(temporal_occlusion(model, clip, mask, occlusion_mode)
                   if do_occlusion else None),
        attribution=branch_attribution(model, clip, mask),
        tsc_map=(ev["tsc_tsc_map"][0].float().cpu().numpy()
                 if "tsc_tsc_map" in ev else None),
        srm_energy=(ev["srm_energy"][0].float().cpu().numpy()
                    if "srm_energy" in ev else None),
        frame_attention=frame_attn,
        descriptors=descriptors,
        percentiles=percentiles,
    )


# ==========================================================================
# Figures
# ==========================================================================

def plot_explanation(ex: Explanation, out_path=None, n_frames: int = 6):
    """One figure with every panel. Returns the matplotlib figure."""
    import matplotlib.pyplot as plt

    from . import viz
    viz._ensure_backend()
    viz.use_style()

    n_img_rows = 1 + (ex.cam is not None)
    rows = n_img_rows + 1
    # image rows are short and wide; the quantitative row needs real height,
    # so the ratios are explicit rather than equal — equal rows leave a band of
    # dead whitespace under every thumbnail strip.
    fig = plt.figure(figsize=(13, 2.35 * n_img_rows + 3.6))
    gs = fig.add_gridspec(rows, 6, hspace=0.35, wspace=0.35,
                          height_ratios=[1.0] * n_img_rows + [1.55])

    sel = ex.top_frames(n_frames) if ex.occlusion is not None else \
        list(range(min(n_frames, len(ex.frames))))
    sel = sorted(set(sel + list(range(0, len(ex.frames),
                                      max(1, len(ex.frames) // n_frames)))))[:n_frames]

    # ---- row 0: frames ------------------------------------------------
    for i, t in enumerate(sel):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(ex.frames[t]); ax.axis("off")
        ax.set_title(f"frame {t}", fontsize=9)
    fig.text(0.005, 0.995, f"Verdict: {ex.verdict.upper()}  "
             f"p(fake)={ex.fake_probability:.3f}",
             fontsize=13, fontweight="600", va="top",
             color=viz.STATUS["critical"] if ex.verdict == "fake" else viz.STATUS["good"])

    # ---- row 1: Grad-CAM ----------------------------------------------
    r = 1
    if ex.cam is not None:
        for i, t in enumerate(sel):
            ax = fig.add_subplot(gs[r, i])
            ax.imshow(overlay_cam(ex.frames[t], ex.cam[t])); ax.axis("off")
            ax.set_title("Grad-CAM" if i == 0 else "", fontsize=9)
        r += 1

    # ---- last row: the three quantitative panels -----------------------
    ax = fig.add_subplot(gs[r, 0:2])
    if ex.occlusion is not None:
        d = ex.occlusion["delta"]
        cols = [viz.STATUS["critical"] if v > 0 else viz.SERIES[0] for v in d]
        ax.bar(range(len(d)), d, color=cols, width=0.7)
        ax.axhline(0, color=viz.INK_2, lw=1)
        ax.set_title("when: per-frame occlusion (Δ fake-logit)", fontsize=10)
        ax.set_xlabel("frame")
    viz.despine(ax)

    ax = fig.add_subplot(gs[r, 2:4])
    if ex.attribution:
        names = sorted(ex.attribution, key=lambda k: ex.attribution[k]["share"])
        vals = [ex.attribution[n]["share"] for n in names]
        gates = [ex.attribution[n]["gate"] for n in names]
        y = np.arange(len(names))
        ax.barh(y - 0.19, vals, height=0.36,
                color=[viz.BRANCH_COLOR.get(n, viz.SERIES[0]) for n in names],
                label="measured dependence")
        ax.barh(y + 0.19, gates, height=0.36, color=viz.INK_MUTED,
                label="attention gate")
        ax.set_yticks(y, names)
        ax.set_title("which: branch dependence vs gate", fontsize=10)
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(axis="y", visible=False)
    viz.despine(ax)

    ax = fig.add_subplot(gs[r, 4:6])
    if ex.tsc_map is not None:
        im = ax.imshow(ex.tsc_map, aspect="auto", origin="lower",
                       cmap=viz.sequential_cmap())
        ax.set_title("TSC map (spatial freq → / temporal freq ↑)", fontsize=10)
        ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.046)
    elif ex.srm_energy is not None:
        ax.plot(ex.srm_energy, color=viz.SERIES[2], marker="o", ms=3)
        ax.set_title("SRM residual energy per frame", fontsize=10)
        viz.despine(ax)

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=140, bbox_inches="tight")
        log.info(f"figure: {out_path}")
    return fig


__all__ = ["explain_clip", "Explanation", "grad_cam", "overlay_cam",
           "temporal_occlusion", "branch_attribution", "plot_explanation",
           "denormalise"]
