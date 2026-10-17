"""
calibrate.py
============

Turning a logit into a probability that *means* something.

Why this module exists
----------------------
`results/calibration.csv` records ECE 0.081 for the full model. That number was
computed with equal-width bins on a dataset that is 86.4% fake, which is the
regime where equal-width ECE flatters a model: nearly every prediction lands in
two or three bins, and the bins that would expose miscalibration are empty. So
this module reports **both** binning strategies and refuses to print one alone.

It also reports the Brier decomposition, because a well-calibrated useless model
is a real possibility here. Brier = reliability - resolution + uncertainty. A
model with ECE 0.08 and resolution ~= 0 is perfectly honest about knowing
nothing. That is a publishable finding, but only if you look for it.

Everything is fit on the identity-disjoint VAL split and applied to test. Fitting
a calibrator on test and reporting test ECE measures nothing at all.

No torch dependency: this module works on numpy arrays of logit-margins, so it
can be used from a notebook, a script, or a CPU-only machine.

    from fsvnet.calibrate import TemperatureScaler, calibration_report
    cal = TemperatureScaler().fit(z_val, y_val)
    p_test = cal.transform(z_test)
    print(calibration_report(y_test, p_test))
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ==========================================================================
# primitives
# ==========================================================================


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def logit(p: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1 - eps)
    return np.log(p / (1 - p))


def fake_margin_from_logits(logits: np.ndarray) -> np.ndarray:
    """(N,2) two-class logits -> (N,) fake-vs-real margin, the quantity we scale."""
    a = np.asarray(logits, dtype=np.float64)
    if a.ndim == 1:
        return a
    if a.shape[1] != 2:
        raise ValueError(f"expected (N,2) logits, got {a.shape}")
    return a[:, 1] - a[:, 0]


def _bin_edges(p: np.ndarray, n_bins: int, strategy: str) -> np.ndarray:
    if strategy == "uniform":
        return np.linspace(0.0, 1.0, n_bins + 1)
    if strategy == "quantile":
        # Equal-MASS bins. With an 86/14 prior this is the honest choice: it
        # puts the same number of predictions in each bin instead of leaving
        # most bins empty and letting them contribute zero error.
        qs = np.linspace(0.0, 1.0, n_bins + 1)
        e = np.quantile(p, qs)
        e[0], e[-1] = 0.0, 1.0
        return np.unique(e)
    raise ValueError(f"unknown binning strategy: {strategy}")


def binned_stats(y: np.ndarray, p: np.ndarray, n_bins: int = 15,
                 strategy: str = "quantile") -> List[Dict[str, float]]:
    """Per-bin count, mean confidence and observed frequency."""
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    edges = _bin_edges(p, n_bins, strategy)
    out: List[Dict[str, float]] = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        sel = (p > lo) & (p <= hi) if i > 0 else (p >= lo) & (p <= hi)
        n = int(sel.sum())
        if n == 0:
            continue
        out.append({"lo": float(lo), "hi": float(hi), "n": n,
                    "mean_p": float(p[sel].mean()),
                    "frac_pos": float(y[sel].mean())})
    return out


def ece(y: np.ndarray, p: np.ndarray, n_bins: int = 15,
        strategy: str = "quantile") -> float:
    """Expected calibration error. Report the quantile version as the headline."""
    stats = binned_stats(y, p, n_bins, strategy)
    N = float(len(np.asarray(p)))
    if N == 0 or not stats:
        return float("nan")
    return float(sum(b["n"] / N * abs(b["mean_p"] - b["frac_pos"]) for b in stats))


def mce(y: np.ndarray, p: np.ndarray, n_bins: int = 15,
        strategy: str = "quantile", min_bin: int = 10) -> float:
    """
    Maximum calibration error, ignoring bins with fewer than `min_bin` samples
    (an n=1 bin is always 0 or 1 away from its mean and would dominate).
    """
    stats = [b for b in binned_stats(y, p, n_bins, strategy) if b["n"] >= min_bin]
    if not stats:
        return float("nan")
    return float(max(abs(b["mean_p"] - b["frac_pos"]) for b in stats))


def brier(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    return float(np.mean((p - y) ** 2))


def brier_decomposition(y: np.ndarray, p: np.ndarray, n_bins: int = 15,
                        strategy: str = "quantile") -> Dict[str, float]:
    """
    Murphy's decomposition:  Brier = reliability - resolution + uncertainty.

    reliability  how far each bin's mean prediction is from its observed rate
                 (lower is better; this is what calibration fixes)
    resolution   how far the bins' observed rates are from the base rate
                 (higher is better; this is the only term that measures whether
                 the model knows anything, and calibration cannot improve it)
    uncertainty  base_rate * (1 - base_rate), a property of the data alone
    """
    y = np.asarray(y, dtype=np.float64)
    N = float(len(y))
    base = float(y.mean()) if N else float("nan")
    stats = binned_stats(y, p, n_bins, strategy)
    rel = sum(b["n"] / N * (b["mean_p"] - b["frac_pos"]) ** 2 for b in stats)
    res = sum(b["n"] / N * (b["frac_pos"] - base) ** 2 for b in stats)
    unc = base * (1 - base)
    return {"brier": brier(y, p), "reliability": float(rel),
            "resolution": float(res), "uncertainty": float(unc),
            "base_rate": base,
            "decomposition_residual": float(brier(y, p) - (rel - res + unc))}


def nll(y: np.ndarray, p: np.ndarray, eps: float = 1e-12) -> float:
    y = np.asarray(y, dtype=np.float64)
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def reprior(p: np.ndarray, train_prior: float, deploy_prior: float) -> np.ndarray:
    """
    Shift probabilities from the training prior to a deployment prior.

    Celeb-DF-v2 is 86.4% fake. That is a property of the dataset, not of any
    situation the detector would be used in, so a probability calibrated on it
    is calibrated *for a world that is 86% fake*. This applies the prior-odds
    correction. Every output that has not been re-priored should say which prior
    it assumes -- silently reporting a Celeb-DF-calibrated probability as "the
    probability this video is fake" is a misstatement.
    """
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-7, 1 - 1e-7)
    tr = float(np.clip(train_prior, 1e-6, 1 - 1e-6))
    dp = float(np.clip(deploy_prior, 1e-6, 1 - 1e-6))
    odds = (p / (1 - p)) * ((dp / (1 - dp)) / (tr / (1 - tr)))
    return odds / (1.0 + odds)


# ==========================================================================
# calibrators
# ==========================================================================


class _Calibrator:
    name = "identity"

    def fit(self, z: np.ndarray, y: np.ndarray) -> "_Calibrator":
        return self

    def transform(self, z: np.ndarray) -> np.ndarray:
        return sigmoid(z)

    @property
    def params(self) -> Dict[str, float]:
        return {}

    def to_dict(self) -> dict:
        return {"name": self.name, "params": self.params}


class IdentityScaler(_Calibrator):
    """The 'do nothing' baseline. Every table needs its row."""
    name = "identity"


class TemperatureScaler(_Calibrator):
    """
    p = sigmoid(z / T), one parameter, fit by minimising NLL.

    One parameter is the right complexity for a 976-clip validation split.
    Temperature scaling cannot change the ranking of predictions, so it cannot
    change AUC -- which is a feature: it fixes calibration without touching
    discrimination, so any AUC change you see afterwards is a bug.
    """
    name = "temperature"

    def __init__(self, t_init: float = 1.0):
        self.T = float(t_init)
        self.fitted_ = False

    def fit(self, z: np.ndarray, y: np.ndarray) -> "TemperatureScaler":
        from scipy.optimize import minimize_scalar

        z = np.asarray(z, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if len(np.unique(y)) < 2:
            # Cannot fit a temperature on a single-class split; leaving T=1 and
            # saying so beats returning a number that means nothing.
            self.T, self.fitted_ = 1.0, False
            return self

        def obj(log_t: float) -> float:
            return nll(y, sigmoid(z / np.exp(log_t)))

        res = minimize_scalar(obj, bounds=(-3.0, 3.0), method="bounded")
        self.T = float(np.exp(res.x))
        self.fitted_ = True
        return self

    def transform(self, z: np.ndarray) -> np.ndarray:
        return sigmoid(np.asarray(z, dtype=np.float64) / self.T)

    @property
    def params(self) -> Dict[str, float]:
        return {"T": self.T}


class PlattScaler(_Calibrator):
    """
    p = sigmoid(a*z + b). Two parameters, so it can also move the decision
    threshold -- useful under an 86/14 prior, and the reason it is worth
    comparing against temperature scaling rather than assuming one is better.
    """
    name = "platt"

    def __init__(self):
        self.a, self.b = 1.0, 0.0
        self.fitted_ = False

    def fit(self, z: np.ndarray, y: np.ndarray) -> "PlattScaler":
        from scipy.optimize import minimize

        z = np.asarray(z, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if len(np.unique(y)) < 2:
            self.fitted_ = False
            return self

        def obj(theta):
            a, b = theta
            return nll(y, sigmoid(a * z + b))

        res = minimize(obj, x0=np.array([1.0, 0.0]), method="Nelder-Mead",
                       options={"xatol": 1e-6, "fatol": 1e-9, "maxiter": 2000})
        self.a, self.b = float(res.x[0]), float(res.x[1])
        self.fitted_ = True
        return self

    def transform(self, z: np.ndarray) -> np.ndarray:
        return sigmoid(self.a * np.asarray(z, dtype=np.float64) + self.b)

    @property
    def params(self) -> Dict[str, float]:
        return {"a": self.a, "b": self.b}


CALIBRATORS = {"identity": IdentityScaler, "temperature": TemperatureScaler,
               "platt": PlattScaler}


def pick_calibrator(z: np.ndarray, y: np.ndarray, k_folds: int = 5,
                    seed: int = 42, verbose: bool = True
                    ) -> Tuple[_Calibrator, Dict[str, float]]:
    """
    Choose between identity / temperature / Platt by k-fold CV **on val**.

    Selecting by test performance would make the reported test ECE a fitted
    quantity. This is the whole reason the function exists rather than a
    hard-coded choice.
    """
    z = np.asarray(z, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    folds = np.array_split(order, max(2, min(k_folds, n)))

    scores: Dict[str, float] = {}
    for name, cls in CALIBRATORS.items():
        losses = []
        for i in range(len(folds)):
            te = folds[i]
            tr = np.concatenate([folds[j] for j in range(len(folds)) if j != i])
            if len(np.unique(y[tr])) < 2 or len(te) == 0:
                continue
            cal = cls().fit(z[tr], y[tr])
            losses.append(nll(y[te], cal.transform(z[te])))
        scores[name] = float(np.mean(losses)) if losses else float("inf")

    best = min(scores, key=scores.get)
    if verbose:
        print("[calibrate] CV NLL on val: " +
              ", ".join(f"{k}={v:.4f}" for k, v in scores.items()) +
              f"  -> chose '{best}'")
    return CALIBRATORS[best]().fit(z, y), scores


# ==========================================================================
# report
# ==========================================================================


@dataclass
class CalibrationReport:
    n: int
    base_rate: float
    ece_quantile: float
    ece_uniform: float
    mce_quantile: float
    brier: float
    reliability: float
    resolution: float
    uncertainty: float
    nll: float
    auc: Optional[float] = None
    calibrator: Optional[dict] = None
    bins: List[Dict[str, float]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    def __str__(self) -> str:
        lines = [
            f"n={self.n}  base_rate={self.base_rate:.3f}"
            + (f"  AUC={self.auc:.4f}" if self.auc is not None else ""),
            f"ECE  equal-mass={self.ece_quantile:.4f}   "
            f"equal-width={self.ece_uniform:.4f}   MCE={self.mce_quantile:.4f}",
            f"Brier={self.brier:.4f} = reliability {self.reliability:.4f} "
            f"- resolution {self.resolution:.4f} + uncertainty {self.uncertainty:.4f}",
            f"NLL={self.nll:.4f}",
        ]
        lines += [f"  ! {n}" for n in self.notes]
        return "\n".join(lines)


def calibration_report(y: np.ndarray, p: np.ndarray, n_bins: int = 15,
                       calibrator: Optional[_Calibrator] = None
                       ) -> CalibrationReport:
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    dec = brier_decomposition(y, p, n_bins, "quantile")

    auc = None
    try:
        from sklearn.metrics import roc_auc_score
        if len(np.unique(y)) == 2:
            auc = float(roc_auc_score(y, p))
    except Exception:
        pass

    notes: List[str] = []
    eq, eu = ece(y, p, n_bins, "quantile"), ece(y, p, n_bins, "uniform")
    if np.isfinite(eq) and np.isfinite(eu) and eq > eu * 1.5:
        notes.append("equal-width ECE is much lower than equal-mass ECE: the "
                     "predictions are concentrated in a few bins, so the "
                     "equal-width figure understates miscalibration. Report the "
                     "equal-mass number.")
    if dec["resolution"] < 0.01:
        notes.append(f"resolution is {dec['resolution']:.4f} -- near zero. The "
                     "model is close to predicting the base rate for every "
                     "input; good calibration here does not mean good "
                     "discrimination.")
    if len(y) < 200:
        notes.append(f"only {len(y)} samples: per-bin estimates with {n_bins} "
                     "bins are noisy. Reduce n_bins or widen the split.")

    return CalibrationReport(
        n=int(len(y)), base_rate=dec["base_rate"],
        ece_quantile=eq, ece_uniform=eu,
        mce_quantile=mce(y, p, n_bins, "quantile"),
        brier=dec["brier"], reliability=dec["reliability"],
        resolution=dec["resolution"], uncertainty=dec["uncertainty"],
        nll=nll(y, p), auc=auc,
        calibrator=calibrator.to_dict() if calibrator is not None else None,
        bins=binned_stats(y, p, n_bins, "quantile"), notes=notes,
    )


def reliability_curve(y: np.ndarray, p: np.ndarray, n_bins: int = 15,
                      strategy: str = "quantile"
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(mean_confidence, observed_frequency, count) per bin, for plotting."""
    stats = binned_stats(y, p, n_bins, strategy)
    return (np.array([b["mean_p"] for b in stats]),
            np.array([b["frac_pos"] for b in stats]),
            np.array([b["n"] for b in stats]))


__all__ = ["sigmoid", "logit", "fake_margin_from_logits", "ece", "mce",
           "brier", "brier_decomposition", "nll", "reprior", "binned_stats",
           "IdentityScaler", "TemperatureScaler", "PlattScaler", "CALIBRATORS",
           "pick_calibrator", "CalibrationReport", "calibration_report",
           "reliability_curve"]
