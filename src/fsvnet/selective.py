"""
selective.py
============

Knowing when not to answer.

This is the module that makes the rest of the system defensible. The honest
identity-disjoint numbers in `docs/FINDINGS.md` are 0.51-0.59 AUC for the
closed-form channels, and `results/ablations.csv` has the full model 0.111 AUC
*below* its own no-TSC/no-LMF ablation. A detector at that level should not
answer most questions. What it can usefully do is rank its own inputs by how
likely it is to be right, answer the top slice, and abstain on the rest.

That is what a risk-coverage curve measures, and it is the headline figure of
this work: "at 20% coverage the system is 85% accurate" is a real result at
0.57 AUC. A flat risk-coverage curve is *also* a real result -- it says
confidence carries no usable ranking information -- and it is reported, not
hidden.

Two things this module insists on
--------------------------------
1. **The majority-class line.** Celeb-DF-v2 is 86.4% fake, so "always say fake"
   scores 0.864 accuracy. Every accuracy figure produced here carries that
   baseline, because without it selective accuracy reads far better than it is.
2. **Class-conditional (Mondrian) thresholds.** With an 86/14 prior, a single
   global confidence threshold abstains on nearly every real clip while
   answering almost every fake one -- which looks like high coverage and is
   actually a broken system. Thresholds are fitted per predicted class.

On the conformal guarantee, honestly
------------------------------------
Split conformal gives a distribution-free bound on selective error *under
exchangeability* between the calibration set and the data you apply it to. A
cross-dataset evaluation (Celeb-DF -> FF++/DFDC) breaks exchangeability, and no
amount of code fixes that. `check_exchangeability()` exists to make the
violation loud rather than silent; it does not repair the guarantee.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from .utils import log
except Exception:                                     # pragma: no cover
    import logging
    log = logging.getLogger("fsvnet")


ABSTAIN = "abstain"


# ==========================================================================
# risk-coverage
# ==========================================================================


def confidence_from_prob(p: np.ndarray) -> np.ndarray:
    """|p - 0.5| * 2 -- 0 at the decision boundary, 1 at either extreme."""
    return np.abs(np.asarray(p, dtype=np.float64) - 0.5) * 2.0


def risk_coverage_curve(y: np.ndarray, p: np.ndarray,
                        confidence: Optional[np.ndarray] = None,
                        threshold: float = 0.5
                        ) -> Dict[str, np.ndarray]:
    """
    Sort by confidence, answer the most confident k, and record the error rate.

    Returns coverage (1/N .. 1), risk (selective error), accuracy, and the
    confidence value at each coverage level so a target can be read straight off
    the curve.
    """
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    conf = confidence_from_prob(p) if confidence is None else np.asarray(
        confidence, dtype=np.float64)
    pred = (p >= threshold).astype(np.float64)
    wrong = (pred != y).astype(np.float64)

    order = np.argsort(-conf, kind="mergesort")        # stable: ties keep order
    w = wrong[order]
    n = len(y)
    cum_err = np.cumsum(w)
    k = np.arange(1, n + 1, dtype=np.float64)
    return {"coverage": k / n, "risk": cum_err / k, "accuracy": 1.0 - cum_err / k,
            "confidence_at": conf[order], "n": n}


def aurc(y: np.ndarray, p: np.ndarray, **kw) -> float:
    """Area under the risk-coverage curve. Lower is better."""
    rc = risk_coverage_curve(y, p, **kw)
    return float(np.trapezoid(rc["risk"], rc["coverage"]) /
                 (rc["coverage"][-1] - rc["coverage"][0] + 1e-12))


def e_aurc(y: np.ndarray, p: np.ndarray, **kw) -> float:
    """
    Excess AURC: AURC minus the AURC an oracle ranking would achieve at the same
    overall error rate. Isolates the quality of the *ranking* from the quality of
    the classifier, which is what selective prediction is actually about.
    """
    rc = risk_coverage_curve(y, p, **kw)
    a = aurc(y, p, **kw)
    err = float(rc["risk"][-1])
    n = rc["n"]
    # oracle: every correct prediction ranked above every incorrect one
    n_wrong = int(round(err * n))
    w = np.concatenate([np.zeros(n - n_wrong), np.ones(n_wrong)])
    k = np.arange(1, n + 1, dtype=np.float64)
    orc = np.cumsum(w) / k
    cov = k / n
    a_orc = float(np.trapezoid(orc, cov) / (cov[-1] - cov[0] + 1e-12))
    return float(a - a_orc)


def selective_at_coverage(y: np.ndarray, p: np.ndarray,
                          coverages: Sequence[float] = (0.2, 0.4, 0.6, 0.8, 1.0),
                          **kw) -> List[Dict[str, float]]:
    """Accuracy and error at fixed coverage, with the majority baseline beside it."""
    rc = risk_coverage_curve(y, p, **kw)
    y = np.asarray(y, dtype=np.float64)
    base = max(float(y.mean()), 1.0 - float(y.mean()))
    out = []
    for c in coverages:
        idx = int(np.clip(round(c * rc["n"]), 1, rc["n"])) - 1
        out.append({"coverage": float(rc["coverage"][idx]),
                    "accuracy": float(rc["accuracy"][idx]),
                    "risk": float(rc["risk"][idx]),
                    "confidence_threshold": float(rc["confidence_at"][idx]),
                    "majority_class_baseline": base,
                    "beats_majority": bool(rc["accuracy"][idx] > base)})
    return out


def majority_class_baseline(y: np.ndarray) -> float:
    """The accuracy of 'always predict the commoner class'. Put it on every plot."""
    y = np.asarray(y, dtype=np.float64)
    return max(float(y.mean()), 1.0 - float(y.mean()))


# ==========================================================================
# conformal abstention
# ==========================================================================


def _conformal_quantile_index(n: int, alpha: float) -> int:
    """
    The finite-sample split-conformal index: ceil((n+1)(1-alpha)).

    Using the raw empirical quantile instead is optimistic by O(1/n) -- small
    here, but this is a guarantee we are quoting, so it should be the corrected
    one.
    """
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    return int(np.clip(k, 1, n))


@dataclass
class AbstentionPolicy:
    """
    A fitted two-sided abstention band.

    tau_fake : answer "fake" only when p >= tau_fake
    tau_real : answer "real" only when p <= tau_real
    Anything between the two is an abstention.

    Fitted per predicted class (Mondrian) so the 86/14 prior cannot produce a
    policy that answers every fake and abstains on every real.
    """
    alpha: float
    tau_fake: float
    tau_real: float
    class_conditional: bool = True
    n_cal: int = 0
    cal_coverage: float = 0.0
    cal_selective_risk: float = 0.0
    cal_coverage_fake: float = 0.0
    cal_coverage_real: float = 0.0
    assumed_prior: Optional[float] = None
    source: Optional[str] = None
    notes: List[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    def decide(self, p: float) -> str:
        p = float(p)
        if p >= self.tau_fake:
            return "fake"
        if p <= self.tau_real:
            return "real"
        return ABSTAIN

    def decide_many(self, p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, dtype=np.float64)
        out = np.full(p.shape, ABSTAIN, dtype=object)
        out[p >= self.tau_fake] = "fake"
        out[p <= self.tau_real] = "real"
        return out

    def band(self) -> Tuple[float, float]:
        return (self.tau_real, self.tau_fake)

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @staticmethod
    def load(path) -> "AbstentionPolicy":
        d = json.loads(Path(path).read_text())
        d.pop("notes", None)
        return AbstentionPolicy(**{**d, "notes": []})

    def __str__(self) -> str:
        return (f"AbstentionPolicy(alpha={self.alpha:.3f}) "
                f"answer 'real' if p<={self.tau_real:.3f}, "
                f"'fake' if p>={self.tau_fake:.3f}, else abstain | "
                f"cal coverage {self.cal_coverage:.1%} "
                f"(fake {self.cal_coverage_fake:.1%} / real {self.cal_coverage_real:.1%}), "
                f"selective risk {self.cal_selective_risk:.3f}")


def _one_sided_threshold(p: np.ndarray, y: np.ndarray, alpha: float,
                         side: str) -> float:
    """
    Largest-coverage threshold whose selective error is within alpha, on one side.

    side="fake": scan candidate thresholds t and answer p>=t.
    side="real": scan candidate thresholds t and answer p<=t.
    """
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(p) == 0:
        return 1.0 if side == "fake" else 0.0

    if side == "fake":
        order = np.argsort(-p, kind="mergesort")       # most confident first
        wrong = (y[order] == 0).astype(np.float64)     # said fake, was real
    else:
        order = np.argsort(p, kind="mergesort")
        wrong = (y[order] == 1).astype(np.float64)     # said real, was fake

    k = np.arange(1, len(p) + 1, dtype=np.float64)
    risk = np.cumsum(wrong) / k

    ok = np.where(risk <= alpha)[0]
    if ok.size == 0:
        # No prefix is clean enough: refuse to answer on this side at all.
        return 1.0 + 1e-9 if side == "fake" else -1e-9

    j = int(ok.max())
    # Finite-sample conservatism. `risk <= alpha` on the calibration prefix is an
    # empirical rate, which is optimistic by O(1/sqrt(n)) on new data. Shrinking
    # the answered prefix to the split-conformal index ceil((n+1)(1-alpha)) of
    # the *errors* is the standard correction; with few errors it costs almost
    # no coverage, and it keeps the quoted guarantee honest.
    n_err = int(np.cumsum(wrong)[j])
    if n_err > 0:
        allowed = _conformal_quantile_index(j + 1, alpha)
        j = min(j, max(allowed - 1, 0))
    return float(p[order][j])


def fit_abstention(p_cal: np.ndarray, y_cal: np.ndarray, alpha: float = 0.10,
                   class_conditional: bool = True,
                   assumed_prior: Optional[float] = None,
                   source: Optional[str] = None) -> AbstentionPolicy:
    """
    Fit the abstention band on a held-out CALIBRATION split.

    alpha is the target *selective* error rate: of the clips the system answers,
    at most this fraction should be wrong. It is not an overall error rate, and
    it says nothing about the clips that are abstained on.
    """
    p = np.asarray(p_cal, dtype=np.float64)
    y = np.asarray(y_cal, dtype=np.float64)
    notes: List[str] = []

    if class_conditional:
        hi = p >= 0.5
        tau_fake = _one_sided_threshold(p[hi], y[hi], alpha, "fake")
        tau_real = _one_sided_threshold(p[~hi], y[~hi], alpha, "real")
    else:
        conf_order = np.argsort(-confidence_from_prob(p), kind="mergesort")
        pred = (p >= 0.5).astype(np.float64)
        wrong = (pred[conf_order] != y[conf_order]).astype(np.float64)
        k = np.arange(1, len(p) + 1, dtype=np.float64)
        risk = np.cumsum(wrong) / k
        ok = np.where(risk <= alpha)[0]
        if ok.size == 0:
            tau_fake, tau_real = 1.0 + 1e-9, -1e-9
        else:
            c = float(confidence_from_prob(p)[conf_order][int(ok.max())])
            tau_fake, tau_real = 0.5 + c / 2.0, 0.5 - c / 2.0

    if tau_fake > 1.0 or tau_real < 0.0:
        notes.append("No threshold on at least one side reaches the target "
                     "selective error on the calibration split. The policy "
                     "abstains on that class entirely. This is the correct "
                     "behaviour for a weak detector, not a bug -- raise alpha "
                     "if you need coverage, and report that you did.")

    pol = AbstentionPolicy(alpha=float(alpha), tau_fake=float(tau_fake),
                           tau_real=float(tau_real),
                           class_conditional=class_conditional,
                           n_cal=int(len(p)), assumed_prior=assumed_prior,
                           source=source, notes=notes)

    dec = pol.decide_many(p)
    answered = dec != ABSTAIN
    pol.cal_coverage = float(answered.mean()) if len(p) else 0.0
    if answered.any():
        pred = np.where(dec[answered] == "fake", 1.0, 0.0)
        pol.cal_selective_risk = float((pred != y[answered]).mean())
    pol.cal_coverage_fake = float((dec == "fake").mean()) if len(p) else 0.0
    pol.cal_coverage_real = float((dec == "real").mean()) if len(p) else 0.0

    if pol.cal_coverage > 0.9:
        notes.append(f"calibration coverage is {pol.cal_coverage:.1%}. For a "
                     "detector whose honest AUC is near 0.57 this is "
                     "implausibly high -- check that the calibration split is "
                     "genuinely identity-disjoint from training before "
                     "believing it.")
    log.info(str(pol))
    return pol


def evaluate_policy(pol: AbstentionPolicy, p: np.ndarray, y: np.ndarray
                    ) -> Dict[str, float]:
    """Apply a fitted policy to a new split and report what it actually did."""
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    dec = pol.decide_many(p)
    answered = dec != ABSTAIN
    out: Dict[str, float] = {
        "n": float(len(p)),
        "coverage": float(answered.mean()) if len(p) else 0.0,
        "abstain_rate": float((~answered).mean()) if len(p) else 0.0,
        "majority_class_baseline": majority_class_baseline(y),
        "target_alpha": pol.alpha,
    }
    if answered.any():
        pred = np.where(dec[answered] == "fake", 1.0, 0.0)
        out["selective_accuracy"] = float((pred == y[answered]).mean())
        out["selective_risk"] = float((pred != y[answered]).mean())
        out["met_target"] = float(out["selective_risk"] <= pol.alpha)
    else:
        out["selective_accuracy"] = float("nan")
        out["selective_risk"] = float("nan")
        out["met_target"] = float("nan")
    # what the abstained-on clips would have scored, for the report
    if (~answered).any():
        pred_abs = (p[~answered] >= 0.5).astype(np.float64)
        out["accuracy_on_abstained"] = float((pred_abs == y[~answered]).mean())
    return out


# ==========================================================================
# exchangeability
# ==========================================================================


def check_exchangeability(cal_meta: Dict[str, object],
                          eval_meta: Dict[str, object],
                          keys: Sequence[str] = ("dataset", "source", "split_by",
                                                 "img_size", "num_frames"),
                          raise_on_mismatch: bool = False) -> List[str]:
    """
    Compare the provenance of the calibration data and the evaluation data.

    A conformal guarantee holds under exchangeability. Calibrating on Celeb-DF
    and evaluating on FF++ or DFDC breaks it, and so does changing the clip
    geometry (the frequency and TSC branches are not scale-invariant -- that is
    the bug that produced the index_add_ crash in an earlier session). This
    function does not repair anything; it makes the violation impossible to
    miss.
    """
    problems: List[str] = []
    for k in keys:
        a, b = cal_meta.get(k), eval_meta.get(k)
        if a is None or b is None:
            continue
        if a != b:
            problems.append(f"{k}: calibrated on {a!r}, evaluating on {b!r}")
    if problems:
        msg = ("conformal guarantee VOID -- calibration and evaluation data are "
               "not exchangeable: " + "; ".join(problems))
        if raise_on_mismatch:
            raise ValueError(msg)
        log.warning(msg)
    return problems


# ==========================================================================
# the assertion that catches a broken calibration
# ==========================================================================


def assert_coverage_sane(coverage: float, auc: Optional[float] = None,
                         max_coverage: float = 0.90,
                         strict: bool = True) -> Optional[str]:
    """
    A weak detector must abstain often. If it does not, something upstream is
    wrong -- usually a leaky split -- and the report it produces will be
    convincing and false.

    This is deliberately an assertion rather than a warning: a silently
    miscalibrated run is the single most expensive failure mode in this project,
    because its output looks like a result.
    """
    msg = None
    if coverage > max_coverage:
        msg = (f"coverage {coverage:.1%} exceeds {max_coverage:.0%}"
               + (f" while AUC is {auc:.3f}" if auc is not None else "")
               + ". A detector near chance cannot confidently answer almost "
                 "everything. Check the split for identity leakage, and check "
                 "that the calibrator was fitted on val rather than on test.")
    if msg and strict:
        raise AssertionError(msg)
    if msg:
        log.warning(msg)
    return msg


__all__ = ["ABSTAIN", "confidence_from_prob", "risk_coverage_curve", "aurc",
           "e_aurc", "selective_at_coverage", "majority_class_baseline",
           "AbstentionPolicy", "fit_abstention", "evaluate_policy",
           "check_exchangeability", "assert_coverage_sane"]
