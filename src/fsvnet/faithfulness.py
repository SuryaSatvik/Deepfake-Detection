"""
faithfulness.py
===============

Does the explanation describe the computation, or decorate it?

Every attribution method in this project produces a plausible-looking picture.
That is the problem: a Grad-CAM map over a face always looks like it found
something. The only way to tell an explanation from a decoration is to act on it
-- remove what it says matters and see whether the model's answer moves.

Four metrics, one aggregate, three controls
------------------------------------------
metrics
    deletion AUC        remove the top-ranked units in order; the fake-margin
                        should collapse. LOWER is better.
    insertion AUC       start from the baseline and add them back; the margin
                        should recover fast. HIGHER is better.
    comprehensiveness   margin(x) - margin(x without top-k), averaged over k.
                        HIGHER is better.
    sufficiency         margin(x) - margin(top-k only). Closer to ZERO is better
                        (the top-k alone should already carry the decision).

aggregate
    Borda count over the four per-metric rankings. Deliberately not a weighted
    sum: any weighting would be an arbitrary choice presented as a measurement,
    and it is the first thing a reviewer would attack. A rank aggregate needs no
    weights and survives the metrics being on different scales.

controls -- all three mandatory
    random ranking      the null. An explanation that does not beat a random
                        ordering of the same units is not an explanation. This
                        is the single most important number in the module and
                        the one most often missing from published XAI results.
    blur-everything     how far the margin moves when *all* the fine detail
                        goes. Bounds how much of any deletion curve is evidence
                        removal rather than image degradation.
    two fills           black or blurred fill both create inputs the model never
                        saw in training, so part of the margin drop is an
                        out-of-distribution reaction. Running every curve with a
                        blurred fill AND an in-distribution patch fill, then
                        flagging disagreement, is the cheap version of ROAD.

Stability
---------
`stability()` re-runs an explainer under reseeding and small input noise and
scores the agreement of the rankings. An explanation that changes every run is
not reporting a reason, however well it scores above -- so instability
disqualifies a candidate rather than merely lowering it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
except Exception:                                     # pragma: no cover
    torch = None                                      # numpy-only use still works

try:
    from .utils import log
except Exception:                                     # pragma: no cover
    import logging
    log = logging.getLogger("fsvnet")

from .attribution import fake_logit, make_baseline

# Fractions of units removed / inserted along each curve.
DEFAULT_STEPS: Tuple[float, ...] = (0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50,
                                    0.70, 0.90, 1.0)
# k values for comprehensiveness / sufficiency (ERASER convention).
DEFAULT_KS: Tuple[float, ...] = (0.01, 0.05, 0.10, 0.20, 0.50)

UNIT_SPACES = ("patch", "frame", "branch")


# ==========================================================================
# turning any explanation into a ranking over units
# ==========================================================================


@dataclass
class Ranking:
    """
    A ranked list of input units, which is the only thing faithfulness can score.

    space   "patch"  -> units are (frame, patch_y, patch_x) tiles
            "frame"  -> units are whole frames
            "branch" -> units are branch names
    order   unit indices, most important first
    method  which explainer produced it (carried through for the report)
    """
    space: str
    order: List[object]
    scores: Optional[np.ndarray] = None
    method: str = ""
    patch: int = 16
    shape: Optional[Tuple[int, ...]] = None

    def __len__(self) -> int:
        return len(self.order)


def patchify_map(attr: np.ndarray, patch: int = 16) -> np.ndarray:
    """
    (T,H,W) pixel attribution -> (T, H//p, W//p) patch attribution by mean.

    Deletion at single-pixel granularity is meaningless for a CNN (one pixel
    changes nothing) and costs a forward pass per pixel. Patches of ~16 px are
    the coarsest unit that still localises within a face.
    """
    a = np.asarray(attr, dtype=np.float64)
    T, H, W = a.shape
    ph, pw = H // patch, W // patch
    a = a[:, :ph * patch, :pw * patch]
    return a.reshape(T, ph, patch, pw, patch).mean(axis=(2, 4))


def ranking_from_spatial(attr: np.ndarray, patch: int = 16, method: str = "",
                         absolute: bool = True) -> Ranking:
    """(T,H,W) attribution -> patch ranking, highest first."""
    pm = patchify_map(attr, patch)
    v = np.abs(pm) if absolute else pm
    flat = v.reshape(-1)
    order = np.argsort(-flat, kind="mergesort")
    idx = [tuple(int(x) for x in np.unravel_index(i, pm.shape)) for i in order]
    return Ranking(space="patch", order=idx, scores=flat[order], method=method,
                   patch=patch, shape=pm.shape)


def ranking_from_frames(delta: np.ndarray, method: str = "") -> Ranking:
    """Per-frame importance (e.g. temporal occlusion delta) -> frame ranking."""
    d = np.asarray(delta, dtype=np.float64)
    order = np.argsort(-d, kind="mergesort")
    return Ranking(space="frame", order=[int(i) for i in order],
                   scores=d[order], method=method, shape=d.shape)


def ranking_from_branches(values: Dict[str, float], method: str = "") -> Ranking:
    """Branch attribution (Shapley or ablation) -> branch ranking."""
    items = sorted(values.items(), key=lambda kv: -abs(kv[1]))
    return Ranking(space="branch", order=[k for k, _ in items],
                   scores=np.array([abs(v) for _, v in items]), method=method)


def ranking_from_superpixels(per_frame: Dict[int, Dict[str, np.ndarray]],
                             method: str = "") -> Tuple[Ranking, Dict[int, np.ndarray]]:
    """KernelSHAP output -> (frame, segment_id) ranking plus the label maps."""
    rows: List[Tuple[float, Tuple[int, int]]] = []
    labels: Dict[int, np.ndarray] = {}
    for t, d in per_frame.items():
        labels[int(t)] = d["labels"]
        for sid, val in zip(d["segment_ids"], d["values"]):
            rows.append((abs(float(val)), (int(t), int(sid))))
    rows.sort(key=lambda r: -r[0])
    return (Ranking(space="superpixel", order=[r[1] for r in rows],
                    scores=np.array([r[0] for r in rows]), method=method),
            labels)


def random_ranking(like: Ranking, seed: int = 0) -> Ranking:
    """The null hypothesis: the same units in a random order."""
    rng = np.random.default_rng(seed)
    order = list(like.order)
    rng.shuffle(order)
    return Ranking(space=like.space, order=order, scores=None,
                   method=f"random(seed={seed})", patch=like.patch,
                   shape=like.shape)


# ==========================================================================
# applying a mask to the clip
# ==========================================================================


def _patch_slice(idx: Tuple[int, int, int], patch: int):
    t, py, px = idx
    return t, slice(py * patch, (py + 1) * patch), slice(px * patch, (px + 1) * patch)


def _apply_units(clip, fill, ranking: Ranking, units: Sequence[object],
                 keep_only: bool = False, labels=None):
    """
    Replace (or keep only) the given units.

    keep_only=False -> those units are replaced by `fill`   (deletion)
    keep_only=True  -> everything EXCEPT those units is replaced (sufficiency)
    """
    if ranking.space == "branch":
        raise ValueError("branch rankings are scored by ablation, not by masking")

    if keep_only:
        x = fill.clone()
        src, dst = clip, x
    else:
        x = clip.clone()
        src, dst = fill, x

    if ranking.space == "patch":
        for u in units:
            t, sy, sx = _patch_slice(u, ranking.patch)
            dst[0, t, :, sy, sx] = src[0, t, :, sy, sx]
    elif ranking.space == "frame":
        for u in units:
            dst[0, int(u)] = src[0, int(u)]
    elif ranking.space == "superpixel":
        if labels is None:
            raise ValueError("superpixel masking needs the label maps")
        for (t, sid) in units:
            lab = labels.get(int(t))
            if lab is None:
                continue
            m = torch.from_numpy(lab == sid).to(clip.device)
            dst[0, int(t), :, m] = src[0, int(t), :, m]
    else:
        raise ValueError(f"unknown unit space: {ranking.space}")
    return x


def in_distribution_fill(clip, frame_source: Optional[int] = None):
    """
    A second fill that is *inside* the data distribution: each frame is replaced
    by another frame of the same clip (rolled by half the clip length).

    Blurring removes evidence but produces an image the model never saw in
    training, so some of the margin drop is an out-of-distribution reaction
    rather than the evidence going away. Swapping in real content from the same
    clip removes the *local* evidence while keeping every global statistic
    plausible. Where the two fills disagree, neither number should be reported
    alone.
    """
    T = clip.shape[1]
    shift = frame_source if frame_source is not None else max(1, T // 2)
    return clip.roll(shifts=int(shift), dims=1).clone()


# ==========================================================================
# the metrics
# ==========================================================================


@dataclass
class CurveResult:
    fractions: List[float]
    margins: List[float]
    auc: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FaithfulnessScore:
    method: str
    space: str
    fill: str
    deletion_auc: float
    insertion_auc: float
    comprehensiveness: float
    sufficiency: float
    n_forwards: int
    base_margin: float
    deletion_curve: Optional[CurveResult] = None
    insertion_curve: Optional[CurveResult] = None
    notes: List[str] = field(default_factory=list)

    def metric_dict(self) -> Dict[str, float]:
        """Metrics with their orientation normalised so HIGHER is always better."""
        return {"deletion": -self.deletion_auc,
                "insertion": self.insertion_auc,
                "comprehensiveness": self.comprehensiveness,
                "sufficiency": -abs(self.sufficiency)}

    def to_dict(self) -> dict:
        d = asdict(self)
        d["deletion_curve"] = self.deletion_curve.to_dict() if self.deletion_curve else None
        d["insertion_curve"] = self.insertion_curve.to_dict() if self.insertion_curve else None
        return d


def _curve(model, clip, fill, ranking: Ranking, mask, lmf,
           fractions: Sequence[float], insertion: bool, labels=None
           ) -> Tuple[CurveResult, int]:
    n = len(ranking)
    margins: List[float] = []
    n_fwd = 0
    for f in fractions:
        k = int(round(f * n))
        units = ranking.order[:k]
        if insertion:
            x = _apply_units(clip, fill, ranking, units, keep_only=True, labels=labels)
        else:
            x = _apply_units(clip, fill, ranking, units, keep_only=False, labels=labels)
        logits, _ = model(x, mask, lmf=lmf)
        margins.append(float(fake_logit(logits)[0]))
        n_fwd += 1
    fr = np.asarray(list(fractions), dtype=np.float64)
    m = np.asarray(margins, dtype=np.float64)
    auc = float(np.trapezoid(m, fr) / (fr[-1] - fr[0] + 1e-12))
    return CurveResult(fractions=[float(x) for x in fr],
                       margins=[float(x) for x in m], auc=auc), n_fwd


def score_ranking(model, clip, ranking: Ranking, mask=None, lmf=None,
                  fill: str = "blur", fractions: Sequence[float] = DEFAULT_STEPS,
                  ks: Sequence[float] = DEFAULT_KS, labels=None
                  ) -> FaithfulnessScore:
    """
    Score one ranking with all four metrics under one fill.

    Cost: len(fractions) forwards for deletion, the same for insertion, and
    2*len(ks) for comprehensiveness/sufficiency -- about 30 forwards at the
    defaults.
    """
    model.eval()
    with torch.no_grad():
        if fill == "blur":
            fill_clip = make_baseline(clip, "blur")
        elif fill == "real_patch":
            fill_clip = in_distribution_fill(clip)
        elif fill in ("mean", "zero"):
            fill_clip = make_baseline(clip, fill)
        else:
            raise ValueError(f"unknown fill: {fill}")

        logits, _ = model(clip, mask, lmf=lmf)
        base = float(fake_logit(logits)[0])
        n_fwd = 1

        dele, k1 = _curve(model, clip, fill_clip, ranking, mask, lmf,
                          fractions, insertion=False, labels=labels)
        ins, k2 = _curve(model, clip, fill_clip, ranking, mask, lmf,
                         fractions, insertion=True, labels=labels)
        n_fwd += k1 + k2

        n = len(ranking)
        comps, suffs = [], []
        for f in ks:
            k = max(1, int(round(f * n)))
            units = ranking.order[:k]
            x_del = _apply_units(clip, fill_clip, ranking, units,
                                 keep_only=False, labels=labels)
            x_keep = _apply_units(clip, fill_clip, ranking, units,
                                  keep_only=True, labels=labels)
            l1, _ = model(x_del, mask, lmf=lmf)
            l2, _ = model(x_keep, mask, lmf=lmf)
            comps.append(base - float(fake_logit(l1)[0]))
            suffs.append(base - float(fake_logit(l2)[0]))
            n_fwd += 2

    notes: List[str] = []
    if abs(base) < 1e-3:
        notes.append("the base fake-margin is essentially zero, so every metric "
                     "here is measuring noise around an undecided prediction. "
                     "Faithfulness is not defined for a decision that was not "
                     "made.")
    return FaithfulnessScore(
        method=ranking.method, space=ranking.space, fill=fill,
        deletion_auc=dele.auc, insertion_auc=ins.auc,
        comprehensiveness=float(np.mean(comps)),
        sufficiency=float(np.mean(suffs)),
        n_forwards=n_fwd, base_margin=base,
        deletion_curve=dele, insertion_curve=ins, notes=notes)


def blur_everything_control(model, clip, mask=None, lmf=None) -> Dict[str, float]:
    """
    How far the margin moves when all fine detail is destroyed, with no
    attribution involved. This is the ceiling any deletion curve is working
    inside: if a method's full-deletion margin equals this, it has told you
    nothing that "blur the whole clip" would not have.
    """
    model.eval()
    with torch.no_grad():
        logits, _ = model(clip, mask, lmf=lmf)
        base = float(fake_logit(logits)[0])
        b = make_baseline(clip, "blur")
        logits2, _ = model(b, mask, lmf=lmf)
        blurred = float(fake_logit(logits2)[0])
    return {"base_margin": base, "blurred_margin": blurred,
            "total_available_shift": base - blurred}


# ==========================================================================
# aggregation and the random control
# ==========================================================================


def borda_aggregate(scores: Sequence[FaithfulnessScore]) -> Dict[str, float]:
    """
    Rank-aggregate several candidates over the four metrics.

    Each metric is ranked independently (higher = better after orientation
    normalisation) and a candidate's Borda score is the mean of its ranks,
    expressed in [0,1] where 1 is best. No weights, no scale assumptions.
    """
    if not scores:
        return {}
    metrics = list(scores[0].metric_dict().keys())
    n = len(scores)
    points = {s.method: 0.0 for s in scores}
    for m in metrics:
        vals = [(s.method, s.metric_dict()[m]) for s in scores]
        vals.sort(key=lambda kv: kv[1])                 # worst first
        for rank, (name, _) in enumerate(vals):
            points[name] += rank                        # 0 .. n-1
    denom = len(metrics) * max(n - 1, 1)
    return {k: v / denom for k, v in points.items()}


@dataclass
class FaithfulnessVerdict:
    method: str
    borda: float
    borda_random: float
    margin_over_random: float
    beats_random: bool
    per_metric: Dict[str, float]
    fills_agree: bool
    scores: List[FaithfulnessScore] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"method": self.method, "borda": self.borda,
                "borda_random": self.borda_random,
                "margin_over_random": self.margin_over_random,
                "beats_random": self.beats_random,
                "per_metric": self.per_metric, "fills_agree": self.fills_agree,
                "scores": [s.to_dict() for s in self.scores],
                "notes": self.notes}


def evaluate_candidate(model, clip, ranking: Ranking, mask=None, lmf=None,
                       fills: Sequence[str] = ("blur", "real_patch"),
                       n_random: int = 3, labels=None, seed: int = 0
                       ) -> FaithfulnessVerdict:
    """
    Score one candidate against random, under every fill, and decide whether it
    is admissible at all.

    `beats_random` is the gate. A candidate that does not beat a random ordering
    of the same units under *both* fills does not get shown to a user, however
    good the picture looks.
    """
    real_scores = [score_ranking(model, clip, ranking, mask, lmf, fill=f,
                                 labels=labels) for f in fills]
    rand_scores: List[FaithfulnessScore] = []
    for i in range(n_random):
        r = random_ranking(ranking, seed=seed + i)
        rand_scores.append(score_ranking(model, clip, r, mask, lmf,
                                         fill=fills[0], labels=labels))

    # Borda over [candidate (fill 0)] + [random draws]
    pool = [real_scores[0]] + rand_scores
    for i, s in enumerate(rand_scores):
        s.method = f"random_{i}"
    borda = borda_aggregate(pool)
    b_real = borda.get(real_scores[0].method, 0.0)
    b_rand = float(np.mean([borda.get(f"random_{i}", 0.0)
                            for i in range(len(rand_scores))])) if rand_scores else 0.0

    per_fill_borda = {}
    for s in real_scores:
        per_fill_borda[s.fill] = s.metric_dict()
    agree = True
    notes: List[str] = []
    if len(real_scores) >= 2:
        a, b = real_scores[0].metric_dict(), real_scores[1].metric_dict()
        for k in a:
            if np.sign(a[k]) != np.sign(b[k]) and min(abs(a[k]), abs(b[k])) > 1e-2:
                agree = False
        if not agree:
            notes.append("the two fills disagree on at least one metric: part of "
                         "the measured effect is an out-of-distribution reaction "
                         "to the fill rather than evidence removal. Report both "
                         "fills, not one.")

    beats = bool(b_real > b_rand)
    if not beats:
        notes.append(f"does NOT beat a random ordering of the same units "
                     f"(Borda {b_real:.3f} vs {b_rand:.3f}). This is not an "
                     "explanation and must not be presented as one.")

    return FaithfulnessVerdict(
        method=ranking.method, borda=b_real, borda_random=b_rand,
        margin_over_random=float(b_real - b_rand), beats_random=beats,
        per_metric=real_scores[0].metric_dict(), fills_agree=agree,
        scores=real_scores, notes=notes)


# ==========================================================================
# stability
# ==========================================================================


def rank_agreement(a: Sequence[object], b: Sequence[object],
                   top_k: Optional[int] = None) -> Dict[str, float]:
    """Spearman rho over the shared units, plus top-k Jaccard."""
    common = [u for u in a if u in set(b)]
    if len(common) < 3:
        return {"spearman": float("nan"), "jaccard": float("nan"), "n": len(common)}
    ra = {u: i for i, u in enumerate(a)}
    rb = {u: i for i, u in enumerate(b)}
    x = np.array([ra[u] for u in common], dtype=np.float64)
    y = np.array([rb[u] for u in common], dtype=np.float64)
    xc, yc = x - x.mean(), y - y.mean()
    denom = float(np.sqrt((xc ** 2).sum() * (yc ** 2).sum()))
    rho = float((xc * yc).sum() / denom) if denom > 0 else float("nan")
    k = top_k or max(1, len(common) // 10)
    sa, sb = set(list(a)[:k]), set(list(b)[:k])
    jac = len(sa & sb) / max(1, len(sa | sb))
    return {"spearman": rho, "jaccard": float(jac), "n": len(common), "top_k": k}


def stability(explain_fn: Callable[[object], Ranking], clip, n_repeats: int = 3,
              noise: float = 0.01, seed: int = 0) -> Dict[str, float]:
    """
    Re-run an explainer under small input noise and score the agreement.

    explain_fn(clip) -> Ranking. The noise is deliberately tiny (sigma=0.01 in
    normalised units, well below any visible change): an explanation that
    reorders under a perturbation the model itself is indifferent to is not
    describing the model.

    An unstable explanation is disqualified, not discounted. A method that gives
    a different reason each run is not reporting a reason.
    """
    base = explain_fn(clip)
    rhos, jacs = [], []
    g = torch.Generator(device="cpu").manual_seed(seed)
    for _ in range(n_repeats):
        pert = clip + noise * torch.randn(clip.shape, generator=g).to(
            clip.device, clip.dtype)
        r = explain_fn(pert)
        ag = rank_agreement(base.order, r.order)
        if np.isfinite(ag["spearman"]):
            rhos.append(ag["spearman"])
        if np.isfinite(ag["jaccard"]):
            jacs.append(ag["jaccard"])
    out = {"spearman_mean": float(np.mean(rhos)) if rhos else float("nan"),
           "jaccard_mean": float(np.mean(jacs)) if jacs else float("nan"),
           "n_repeats": int(n_repeats), "noise": float(noise)}
    out["stable"] = bool(np.isfinite(out["spearman_mean"])
                         and out["spearman_mean"] >= 0.5)
    return out


# ==========================================================================
# does faithfulness predict correctness?
# ==========================================================================


def faithfulness_vs_correctness(faith: Sequence[float], correct: Sequence[bool],
                                n_boot: int = 5000, seed: int = 42
                                ) -> Dict[str, float]:
    """
    The testable claim of this whole design: is the aggregate faithfulness score
    higher on clips the model got right than on clips it got wrong?

    Point-biserial correlation with a bootstrap CI. If the CI excludes zero,
    faithfulness is itself a confidence signal and belongs in the selective
    score. If it does not, that is a negative result and gets reported as one --
    it is the first thing anyone would ask about this design.
    """
    f = np.asarray(faith, dtype=np.float64)
    c = np.asarray(correct, dtype=bool)
    if len(f) != len(c) or len(f) < 8 or c.all() or (~c).all():
        return {"r": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"),
                "n": int(len(f)), "resolved": False,
                "reason": "need at least 8 clips with both outcomes present"}

    def pb(fv, cv):
        m1, m0 = fv[cv].mean(), fv[~cv].mean()
        s = fv.std(ddof=1)
        n1, n0, n = cv.sum(), (~cv).sum(), len(cv)
        return float((m1 - m0) / (s + 1e-12) * np.sqrt(n1 * n0 / (n * n)))

    obs = pb(f, c)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(f), len(f))
        if c[idx].all() or (~c[idx]).all():
            continue
        boots.append(pb(f[idx], c[idx]))
    if not boots:
        return {"r": obs, "ci_lo": float("nan"), "ci_hi": float("nan"),
                "n": int(len(f)), "resolved": False, "reason": "bootstrap degenerate"}
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"r": obs, "ci_lo": float(lo), "ci_hi": float(hi),
            "n": int(len(f)), "n_boot": len(boots),
            "resolved": bool(lo > 0 or hi < 0),
            "mean_faith_correct": float(f[c].mean()),
            "mean_faith_wrong": float(f[~c].mean())}


__all__ = ["Ranking", "patchify_map", "ranking_from_spatial",
           "ranking_from_frames", "ranking_from_branches",
           "ranking_from_superpixels", "random_ranking", "in_distribution_fill",
           "CurveResult", "FaithfulnessScore", "score_ranking",
           "blur_everything_control", "borda_aggregate", "FaithfulnessVerdict",
           "evaluate_candidate", "rank_agreement", "stability",
           "faithfulness_vs_correctness", "DEFAULT_STEPS", "DEFAULT_KS"]
