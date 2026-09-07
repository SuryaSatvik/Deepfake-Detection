"""
evidence.py
===========

Builds the **forensic evidence pack** — the bridge between FSV-Net and the
reasoning layer, and the second novel component of this project.

The idea
--------
Skyra-style systems hand a VLM the raw frames and ask it to spot artefacts.
That works, but it throws away everything the forensic model measured: a VLM
cannot see a temporal spectral anomaly, and asking it to guess one produces
confident narration rather than evidence.

Here the VLM is instead **conditioned on the detector's measurements**:

    frames  +  branch attention  +  TSC descriptors with percentiles
            +  SRM residual trajectory  +  the rendered TSC map
                                    ↓
                    grounded, checkable natural-language reasoning

Every number in the pack is calibrated against a reference distribution
computed from the *training reals*, so the VLM is told "temporal whiteness
0.41, which is the 3rd percentile of real videos" rather than a bare float it
has no way to interpret. That calibration step is what makes the explanation
falsifiable: a reader can check the percentile against the reference file.

Anti-hallucination
------------------
`build_prompt` states explicitly which quantities are measured and which are
not, and instructs the model to refuse to speculate beyond them. The pack also
carries a `confidence_band` so a borderline score is described as borderline
rather than narrated as certain. A VLM given only a probability will always
invent a reason for it; given percentile-calibrated measurements it can say
"the spatial evidence is unremarkable, only the temporal statistic is
anomalous", which is both more useful and more honest.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from .utils import log, save_json

# Descriptors that get percentile-calibrated and shown to the VLM
REPORTED_DESCRIPTORS = [
    "temporal_whiteness", "temporal_peakiness", "flicker_index",
    "fine_ring_acf1", "temporal_hf_ratio", "spectral_drift",
]

# Reported for context but never flagged as evidence: these move for ordinary
# reasons (camera motion), so calling them "anomalous" would hand the reasoning
# layer a false positive to narrate.
CONTEXT_ONLY = {"spectral_drift"}

DESCRIPTOR_MEANING = {
    "temporal_whiteness":
        "how flat the temporal spectrum of fine spatial detail is. Real camera "
        "footage is close to white here (sensor noise is independent per frame); "
        "generated detail is temporally correlated, so this drops.",
    "temporal_peakiness":
        "how much of the temporal spectrum sits in a single bin. A spike means "
        "periodic flicker, typical of per-frame blending or colour correction.",
    "flicker_index":
        "signed: positive means unusually much fine-detail flicker, negative "
        "means unusually little (over-smoothed generation).",
    "fine_ring_acf1":
        "lag-1 temporal autocorrelation of fine detail. Near 0 for sensor "
        "noise, near 1 for generated detail.",
    "temporal_hf_ratio":
        "share of temporal-spectrum energy above one third of the Nyquist rate.",
    "spectral_drift":
        "how much the spatial spectrum moves over the clip; high values usually "
        "just mean camera motion, and are reported for context, not as evidence.",
}


# ==========================================================================
# Reference distribution
# ==========================================================================

@dataclass
class ReferenceStats:
    """Percentile lookup tables built from the training REAL videos."""
    quantiles: Dict[str, List[float]] = field(default_factory=dict)   # 101 values each
    n_videos: int = 0

    def percentile_of(self, key: str, value: float) -> Optional[float]:
        q = self.quantiles.get(key)
        if not q:
            return None
        # quantiles[i] is the i-th percentile of the reference reals, so the
        # insertion point IS the percentile rank.
        return float(np.clip(np.searchsorted(np.asarray(q), value), 0, 100))

    def save(self, path) -> None:
        save_json(asdict(self), path)

    @staticmethod
    def load(path) -> "ReferenceStats":
        d = json.loads(Path(path).read_text())
        return ReferenceStats(quantiles=d.get("quantiles", {}),
                              n_videos=int(d.get("n_videos", 0)))


def build_reference_stats(model, loader, cfg, max_videos: int = 400
                          ) -> ReferenceStats:
    """
    Run the model over REAL training clips and record the empirical
    distribution of each TSC descriptor. This is what turns raw floats into
    "3rd percentile of real videos".
    """
    import torch

    model.eval()
    collected: Dict[str, List[float]] = {k: [] for k in REPORTED_DESCRIPTORS}
    n = 0
    with torch.no_grad():
        for batch in loader:
            keep = batch["label"] == 0                 # reals only
            if keep.sum() == 0:
                continue
            clip = batch["clip"][keep].to(cfg.device)
            mask = batch["mask"][keep].to(cfg.device)
            _, info = model(clip, mask, return_evidence=True)
            desc = info.get("evidence", {}).get("tsc_descriptors", {})
            for k in REPORTED_DESCRIPTORS:
                if k in desc:
                    collected[k].extend(desc[k].float().cpu().numpy().tolist())
            n += int(keep.sum())
            if n >= max_videos:
                break

    MIN_SAMPLES = 20
    quantiles = {}
    for k, vals in collected.items():
        if len(vals) >= MIN_SAMPLES:
            quantiles[k] = np.percentile(np.array(vals), np.arange(101)).tolist()
    ref = ReferenceStats(quantiles=quantiles, n_videos=n)
    log.info(f"reference stats built from {n} real clips, "
             f"{len(quantiles)}/{len(REPORTED_DESCRIPTORS)} descriptors calibrated")
    if len(quantiles) < len(REPORTED_DESCRIPTORS):
        log.warning(f"fewer than {MIN_SAMPLES} real clips per descriptor — "
                    "percentiles will be unavailable and explanations will say so. "
                    "Point this at a larger training loader before reporting anything.")
    if n < 100:
        log.warning(f"only {n} real clips: percentile estimates in the tails "
                    "(<5th, >95th) are unreliable with this few samples.")
    return ref


# ==========================================================================
# Evidence pack
# ==========================================================================

@dataclass
class EvidencePack:
    video_path: str
    fake_probability: float
    verdict: str
    confidence_band: str
    branch_attention: Dict[str, float]
    descriptors: Dict[str, float]
    percentiles: Dict[str, Optional[float]]
    top_frames: List[int]
    frame_attention: List[float]
    srm_energy: List[float]
    srm_energy_cv: float
    figures: Dict[str, str] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path) -> None:
        save_json(self.to_dict(), path)


def _ordinal(n: float) -> str:
    """3 -> '3rd', 11 -> '11th'. Small thing; a report full of '3th' is not."""
    i = int(round(n))
    if 10 <= i % 100 <= 20:
        suf = "th"
    else:
        suf = {1: "st", 2: "nd", 3: "rd"}.get(i % 10, "th")
    return f"{i}{suf}"


def _band(p: float) -> str:
    d = abs(p - 0.5)
    if d < 0.10:
        return "borderline"
    if d < 0.25:
        return "weak"
    if d < 0.40:
        return "moderate"
    return "strong"


def build_evidence(model, clip, mask, video_path: str, cfg,
                   reference: Optional[ReferenceStats] = None,
                   frames_uint8: Optional[np.ndarray] = None,
                   make_figures: bool = True) -> EvidencePack:
    """
    One forward pass -> everything the reasoning layer needs.

    clip : (1, T, 3, H, W) tensor already on the right device
    """
    import torch

    model.eval()
    with torch.no_grad():
        logits, info = model(clip, mask, return_evidence=True)
        prob = float(torch.softmax(logits.float(), 1)[0, 1])

    ev = info.get("evidence", {})
    desc_t = ev.get("tsc_descriptors", {})
    descriptors = {k: float(v[0]) for k, v in desc_t.items()
                   if k in REPORTED_DESCRIPTORS}

    percentiles: Dict[str, Optional[float]] = {}
    if reference is not None:
        for k, v in descriptors.items():
            percentiles[k] = reference.percentile_of(k, v)

    bw = info["branch_weights"][0].float().cpu().numpy()
    branch_attention = {n: float(w) for n, w in zip(info["branch_names"], bw)}

    fw_all = info.get("frame_weights", {})
    fw = (next(iter(fw_all.values()))[0].float().cpu().numpy()
          if fw_all else np.zeros(clip.shape[1]))
    top_frames = np.argsort(-fw)[:3].tolist()

    srm = ev.get("srm_energy")
    srm_list = srm[0].float().cpu().numpy().tolist() if srm is not None else []
    srm_cv = (float(np.std(srm_list) / (np.mean(srm_list) + 1e-9))
              if srm_list else 0.0)

    notes: List[str] = []
    if reference is None:
        notes.append("No reference distribution loaded — descriptor values are "
                     "raw and NOT percentile-calibrated. Treat them as "
                     "uninterpretable numbers, not as evidence.")
    if abs(prob - 0.5) < 0.10:
        notes.append("Score is borderline; the model is close to undecided.")

    pack = EvidencePack(
        video_path=str(video_path),
        fake_probability=prob,
        verdict="fake" if prob >= 0.5 else "real",
        confidence_band=_band(prob),
        branch_attention=branch_attention,
        descriptors=descriptors,
        percentiles=percentiles,
        top_frames=[int(i) for i in top_frames],
        frame_attention=[float(x) for x in fw.tolist()],
        srm_energy=[float(x) for x in srm_list],
        srm_energy_cv=srm_cv,
        notes=notes,
    )

    if make_figures:
        try:
            pack.figures = render_evidence_figures(
                pack, ev, frames_uint8, Path(cfg.paths.evidence_dir))
        except Exception as e:
            log.warning(f"figure rendering failed: {e}")
    return pack


# ==========================================================================
# Figures
# ==========================================================================

def render_evidence_figures(pack: EvidencePack, ev: dict,
                            frames_uint8: Optional[np.ndarray],
                            out_dir: Path) -> Dict[str, str]:
    from .viz import _ensure_backend
    _ensure_backend()
    import matplotlib.pyplot as plt

    from .viz import SERIES, INK_2, despine, sequential_cmap, use_style
    use_style()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(pack.video_path).stem[:40]
    figs: Dict[str, str] = {}

    # --- TSC map -------------------------------------------------------
    if "tsc_tsc_map" in ev:
        tmap = ev["tsc_tsc_map"][0].float().cpu().numpy()
        fig, ax = plt.subplots(figsize=(5, 3.6))
        im = ax.imshow(tmap, aspect="auto", origin="lower",
                       cmap=sequential_cmap())
        ax.set_xlabel("spatial frequency ring →")
        ax.set_ylabel("temporal frequency →")
        ax.set_title(f"TSC map — p(fake)={pack.fake_probability:.3f}")
        fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout()
        p = out_dir / f"{stem}_tsc.png"
        fig.savefig(p, dpi=140); plt.close(fig)
        figs["tsc_map"] = str(p)

    # --- trajectories --------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(9, 3))
    if pack.srm_energy:
        axes[0].plot(pack.srm_energy, marker="o", ms=3, color=SERIES[2])
        axes[0].set_title(f"SRM residual energy (CV={pack.srm_energy_cv:.3f})")
        axes[0].set_xlabel("frame")
    axes[1].bar(range(len(pack.frame_attention)), pack.frame_attention,
                color=SERIES[6], width=0.7)
    axes[1].set_title("temporal attention per frame")
    axes[1].set_xlabel("frame")
    for a in axes:
        despine(a)
    fig.tight_layout()
    p = out_dir / f"{stem}_traj.png"
    fig.savefig(p, dpi=140); plt.close(fig)
    figs["trajectories"] = str(p)

    # --- frame strip ----------------------------------------------------
    if frames_uint8 is not None and len(frames_uint8):
        sel = pack.top_frames[:3] or [0]
        fig, axes = plt.subplots(1, len(sel), figsize=(3.2 * len(sel), 3.4))
        axes = np.atleast_1d(axes)
        for a, i in zip(axes, sel):
            i = int(np.clip(i, 0, len(frames_uint8) - 1))
            a.imshow(frames_uint8[i]); a.axis("off")
            a.set_title(f"frame {i} (w={pack.frame_attention[i]:.3f})", fontsize=9)
        fig.tight_layout()
        p = out_dir / f"{stem}_frames.png"
        fig.savefig(p, dpi=140); plt.close(fig)
        figs["frames"] = str(p)

    return figs


# ==========================================================================
# Prompt construction
# ==========================================================================

SYSTEM_PROMPT = """You are a forensic video analyst. You are given
(a) a small number of frames from a video, and
(b) quantitative measurements produced by a forensic detector, each calibrated
    as a percentile of the same measurement on known-real videos.

Your job is to write a short, grounded analysis: what the measurements say,
which of them are actually anomalous, and what physical process could produce
that pattern.

Hard rules:
- Ground every claim in either a listed measurement or something visible in the
  supplied frames. If you cannot ground it, do not write it.
- The measurements are the primary evidence; the frames are secondary context.
- Do not invent measurements that are not listed, and do not restate a
  measurement as if it were a visual observation.
- If the detector's confidence band is "borderline" or "weak", say the evidence
  is inconclusive. Do not manufacture certainty.
- If the measurements disagree with each other, say so — that is informative.
- Be concise: at most 200 words, no headers, no bullet-point padding."""


def build_prompt(pack: EvidencePack) -> str:
    """Render the evidence pack as the user turn of the VLM prompt."""
    lines: List[str] = []
    lines.append(f"Detector output: p(fake) = {pack.fake_probability:.3f} "
                 f"({pack.verdict}, {pack.confidence_band} confidence)")
    lines.append("")
    lines.append("Branch attention — which forensic signal the detector relied on:")
    for k, v in sorted(pack.branch_attention.items(), key=lambda kv: -kv[1]):
        lines.append(f"  {k:<12} {v:.3f}")
    lines.append("")
    lines.append("Temporal-spectral measurements "
                 "(percentile = where this video sits among known-real videos):")
    for k in REPORTED_DESCRIPTORS:
        if k not in pack.descriptors:
            continue
        v = pack.descriptors[k]
        p = pack.percentiles.get(k)
        pstr = f"{_ordinal(p)} pct of real" if p is not None else "NOT CALIBRATED"
        flag = ""
        if k in CONTEXT_ONLY:
            flag = "   (context only — not evidence)"
        elif p is not None and (p <= 5 or p >= 95):
            flag = "   <-- ANOMALOUS"
        lines.append(f"  {k:<22} {v:+.4f}   ({pstr}){flag}")
        lines.append(f"      meaning: {DESCRIPTOR_MEANING.get(k, '')}")
    lines.append("")
    if pack.srm_energy:
        lines.append(f"SRM residual energy across frames: coefficient of variation "
                     f"{pack.srm_energy_cv:.3f} "
                     f"(min {min(pack.srm_energy):.5f}, max {max(pack.srm_energy):.5f})")
    lines.append(f"Frames with the highest temporal attention: {pack.top_frames}")
    if pack.notes:
        lines.append("")
        lines.append("Caveats you must respect:")
        for n in pack.notes:
            lines.append(f"  - {n}")
    lines.append("")
    lines.append("Write the analysis now.")
    return "\n".join(lines)


def template_explanation(pack: EvidencePack) -> str:
    """
    Deterministic fallback used when no VLM is available. Not as readable as a
    generated explanation, but it never hallucinates — every sentence is a
    direct rendering of a measured number.
    """
    parts = [f"Verdict: {pack.verdict} (p={pack.fake_probability:.3f}, "
             f"{pack.confidence_band} confidence)."]
    top = max(pack.branch_attention, key=pack.branch_attention.get)
    parts.append(f"The detector leaned mostly on the {top} branch "
                 f"({pack.branch_attention[top]:.2f} of the attention mass).")

    calibrated = [k for k in pack.descriptors if pack.percentiles.get(k) is not None]
    anomalous = [(k, pack.descriptors[k], pack.percentiles[k]) for k in calibrated
                 if k not in CONTEXT_ONLY
                 and (pack.percentiles[k] <= 5 or pack.percentiles[k] >= 95)]

    if not calibrated:
        # Do NOT say "nothing was anomalous" here — nothing was *measurable*.
        # Conflating the two is exactly the failure this layer exists to avoid.
        parts.append("No reference distribution was available, so the temporal "
                     "measurements could not be calibrated and no statement about "
                     "whether they are anomalous is possible.")
    elif anomalous:
        for k, v, p in anomalous:
            parts.append(f"{k} = {v:.4f}, the {_ordinal(p)} percentile of real "
                         f"videos — {DESCRIPTOR_MEANING.get(k, '')}")
    else:
        parts.append(f"None of the {len(calibrated)} calibrated temporal-spectral "
                     "measurements fell outside the 5th-95th percentile range of "
                     "real videos; the temporal evidence is unremarkable.")
    if pack.confidence_band in ("borderline", "weak"):
        parts.append("Treat this as inconclusive.")
    return " ".join(parts)


__all__ = ["EvidencePack", "ReferenceStats", "build_reference_stats",
           "CONTEXT_ONLY",
           "build_evidence", "build_prompt", "template_explanation",
           "render_evidence_figures", "SYSTEM_PROMPT", "REPORTED_DESCRIPTORS"]
