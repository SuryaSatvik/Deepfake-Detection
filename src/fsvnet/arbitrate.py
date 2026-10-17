"""
arbitrate.py
============

Which evidence is solid, in which model, before any confidence number is given.

The problem this solves
-----------------------
`results/ablations.csv` says the full FSV-Net is 0.111 AUC *worse* than its own
no-TSC/no-LMF ablation, resolved at p~0. So "use the full model's probability" is
demonstrably the wrong answer, and "average the checkpoints" is worse -- they are
ablation variants, not seeds, so their mean has no interpretation.

What the checkpoints actually are is a panel of **specialists**:

    B0  conventional   spatial + frequency + SRM        (best honest AUC)
    C1  + TSC          adds the temporal-spectral channel
    C2  + LMF          adds the landmark-motion channel
    C3  full           everything (-0.111 AUC vs B0, resolved)

Each is authoritative for the channels it carries and silent about the rest. So
the confidence in a verdict should come from the evidence channels that have
*earned the right to speak*, weighted by how reliable that channel is in that
model, and how solid it looks on this particular clip.

Three stages
------------
1. `build_reliability_matrix` -- offline, on VAL. For every (model, channel):
   how discriminative, how faithful, how stable, how well-calibrated. Saved as
   `results/evidence_reliability.json`; it is also one of the more interesting
   tables in the report.
2. `solidity` -- per clip. reliability x extremeness x counterfactual survival x
   stability x corroboration, with every factor stored separately so the product
   can be audited rather than trusted.
3. `ConfidenceStack` -- an L2 logistic regression fitted on VAL over the
   solidity-weighted evidence, then temperature-calibrated. The coefficients ARE
   the arbitration weights: learned, printable, and checkable. No hand-tuned
   formula anywhere in this file.

A prediction, written down in advance
-------------------------------------
Given the ablation table, arbitration will most likely route most weight to
**B0** and assign TSC and LMF low reliability. That would be the dataset-level
ablation finding rediscovered automatically, per clip. It is written here before
the run so that "it came out as expected" is a check and not a story told
afterwards.

And the baseline it must beat: single-model B0 with plain temperature scaling,
compared with `benchmark.paired_bootstrap`. If arbitration loses, the honest
move is to report that and ship B0.
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

from .calibrate import TemperatureScaler, logit, sigmoid

# The five evidence channels FSV-Net can carry.
CHANNELS: Tuple[str, ...] = ("spatial", "frequency", "srm", "tsc", "lmf")

# Which channels each shipped checkpoint carries.
#
# The ids are the condition ids this repo already uses -- `scripts/train_conditions.py`
# maps them as (use_tsc, use_lmf): B0=(F,F), C1=(T,F), C2=(F,T), C3=(T,T), and
# `models/checkpoints/fsvnet_dfe_<id>.pt` carries those two flags in its blob.
# They are deliberately NOT renamed to something more descriptive: a second
# naming scheme for the same four checkpoints is how a routing table ends up
# pointing at the wrong model.
DEFAULT_SPECIALISTS: Dict[str, Tuple[str, ...]] = {
    "B0": ("spatial", "frequency", "srm"),                          # conventional
    "C1": ("spatial", "frequency", "srm", "tsc"),                   # + TSC
    "C2": ("spatial", "frequency", "srm", "lmf"),                   # + LMF
    "C3": ("spatial", "frequency", "srm", "tsc", "lmf"),            # full (PROPOSED)
}

# The labels `results/metrics.csv` and `results/ablations.csv` already use, so a
# CFX table can be joined to those without a translation step.
CONDITION_LABELS: Dict[str, str] = {
    "B0": "FSV-Net conventional (no TSC, no LMF)",
    "C1": "FSV-Net/DFE + TSC only",
    "C2": "FSV-Net/DFE + LMF only",
    "C3": "FSV-Net/DFE full (PROPOSED)",
}

# B0 is the reference baseline, because `results/ablations.csv` records the full
# model at -0.111 AUC against it, resolved at p~0. Arbitration has to beat B0 to
# be worth shipping.
BASELINE_ID = "B0"


# ==========================================================================
# 1. the reliability matrix
# ==========================================================================


@dataclass
class ChannelReliability:
    model_id: str
    channel: str
    auc: float = float("nan")            # discriminative power of this channel
    auc_ci: Tuple[float, float] = (float("nan"), float("nan"))
    faithfulness: float = float("nan")   # Borda margin over random, from faithfulness.py
    beats_random: bool = False
    stability: float = float("nan")      # Spearman across repeats
    calib_slope: float = float("nan")    # 1.0 == well-scaled on this subset
    coverage: float = float("nan")       # fraction of val clips where it dominates
    n: int = 0

    # --------------------------------------------------------------
    def score(self) -> float:
        """
        One number in [0,1] for how much this model's version of this channel
        deserves to be believed.

        Deliberately conservative and deliberately simple: the discriminative
        term is (AUC - 0.5) doubled and floored at zero, so a channel at chance
        contributes exactly nothing rather than a small positive amount. A
        channel that fails the random-attribution test is zeroed outright -- if
        its explanation is indistinguishable from noise, its evidence should not
        move a confidence number.
        """
        if not np.isfinite(self.auc):
            return 0.0
        disc = max(0.0, (self.auc - 0.5) * 2.0)
        if not self.beats_random:
            return 0.0
        faith = 0.0 if not np.isfinite(self.faithfulness) else float(
            np.clip(self.faithfulness, 0.0, 1.0))
        stab = 0.0 if not np.isfinite(self.stability) else float(
            np.clip(self.stability, 0.0, 1.0))
        # geometric mean: a channel must clear every bar, not average them out
        parts = [disc, max(faith, 1e-3), max(stab, 1e-3)]
        return float(np.exp(np.mean(np.log(parts))))

    def to_dict(self) -> dict:
        d = asdict(self)
        d["auc_ci"] = list(self.auc_ci)
        d["score"] = self.score()
        return d


@dataclass
class ReliabilityMatrix:
    entries: Dict[str, ChannelReliability] = field(default_factory=dict)  # "model|channel"
    split: str = "val"
    notes: List[str] = field(default_factory=list)

    @staticmethod
    def _key(model_id: str, channel: str) -> str:
        return f"{model_id}|{channel}"

    def put(self, e: ChannelReliability) -> None:
        self.entries[self._key(e.model_id, e.channel)] = e

    def get(self, model_id: str, channel: str) -> Optional[ChannelReliability]:
        return self.entries.get(self._key(model_id, channel))

    def score(self, model_id: str, channel: str) -> float:
        e = self.get(model_id, channel)
        return e.score() if e else 0.0

    def best_model_for(self, channel: str,
                       among: Optional[Sequence[str]] = None) -> Tuple[str, float]:
        """Which model is the authoritative source for this channel."""
        best, best_s = "", 0.0
        for k, e in self.entries.items():
            if e.channel != channel:
                continue
            if among is not None and e.model_id not in among:
                continue
            s = e.score()
            if s > best_s:
                best, best_s = e.model_id, s
        return best, best_s

    def as_table(self) -> List[Dict[str, object]]:
        return [e.to_dict() for e in self.entries.values()]

    def to_dict(self) -> dict:
        return {"split": self.split, "notes": self.notes,
                "entries": {k: e.to_dict() for k, e in self.entries.items()}}

    def save(self, path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        log.info(f"reliability matrix -> {path}")

    @staticmethod
    def load(path) -> "ReliabilityMatrix":
        d = json.loads(Path(path).read_text())
        m = ReliabilityMatrix(split=d.get("split", "val"), notes=d.get("notes", []))
        for k, e in d.get("entries", {}).items():
            e = dict(e)
            e.pop("score", None)
            e["auc_ci"] = tuple(e.get("auc_ci", (float("nan"), float("nan"))))
            m.entries[k] = ChannelReliability(**e)
        return m

    def summary(self) -> str:
        rows = sorted(self.entries.values(), key=lambda e: (-e.score(), e.channel))
        w = max([len(e.model_id) for e in rows] + [8])
        out = [f"{'model'.ljust(w)}  {'channel':<10} {'AUC':>6} {'faith':>6} "
               f"{'stab':>6} {'score':>6}  random?"]
        for e in rows:
            out.append(f"{e.model_id.ljust(w)}  {e.channel:<10} {e.auc:6.3f} "
                       f"{e.faithfulness:6.3f} {e.stability:6.3f} {e.score():6.3f}"
                       f"  {'beats' if e.beats_random else 'NO'}")
        return "\n".join(out)


def build_reliability_matrix(per_channel_results: Dict[Tuple[str, str], Dict[str, object]],
                             split: str = "val") -> ReliabilityMatrix:
    """
    Assemble the matrix from measurements you have already made.

    `per_channel_results[(model_id, channel)]` is a dict with any of:
        auc, auc_ci, faithfulness, beats_random, stability, calib_slope,
        coverage, n

    Taking measurements as input rather than computing them here is deliberate:
    it keeps this module testable without checkpoints, and it forces the caller
    (`scripts/build_reliability_matrix.py`) to be explicit about which split each
    number came from.
    """
    m = ReliabilityMatrix(split=split)
    for (model_id, channel), r in per_channel_results.items():
        m.put(ChannelReliability(
            model_id=model_id, channel=channel,
            auc=float(r.get("auc", float("nan"))),
            auc_ci=tuple(r.get("auc_ci", (float("nan"), float("nan")))),
            faithfulness=float(r.get("faithfulness", float("nan"))),
            beats_random=bool(r.get("beats_random", False)),
            stability=float(r.get("stability", float("nan"))),
            calib_slope=float(r.get("calib_slope", float("nan"))),
            coverage=float(r.get("coverage", float("nan"))),
            n=int(r.get("n", 0))))

    live = [e for e in m.entries.values() if e.score() > 0]
    if not live:
        m.notes.append("every channel scored zero: either no channel beat its "
                       "random-attribution control, or every channel AUC is at "
                       "or below chance. Arbitration cannot help here, and the "
                       "honest output is abstention on every clip.")
    else:
        top = max(live, key=lambda e: e.score())
        m.notes.append(f"highest-scoring channel: {top.channel} in {top.model_id} "
                       f"(score {top.score():.3f}).")
    return m


# ==========================================================================
# 2. per-clip solidity
# ==========================================================================


DEADBAND_PCT = 10.0      # +-10 percentiles around the median count as nothing


def extremeness(percentile: Optional[float],
                deadband: float = DEADBAND_PCT) -> float:
    """
    How far into the tail of the real-video reference distribution a measurement
    sits, mapped to [0,1].

    There is a **deadband** by design: anything from the 40th to the 60th
    percentile of real videos scores exactly zero, not merely a small number.
    The commonest failure in evidence-conditioned reasoning is narrating a
    middling measurement as though it were a finding, and a small-but-positive
    weight is precisely what lets a middling number accumulate into a confident
    story. So:

        p = 50  -> 0.0        p = 58  -> 0.0        (deadband)
        p = 75  -> 0.43       p = 95  -> 1.0
        p = 5   -> 1.0        missing -> 0.0
    """
    if percentile is None or not np.isfinite(percentile):
        return 0.0
    d = abs(float(percentile) - 50.0) - float(deadband)
    if d <= 0.0:
        return 0.0
    return float(np.clip(d / (45.0 - float(deadband)), 0.0, 1.0))


def corroboration(signals: Dict[str, float], min_models: int = 2) -> float:
    """
    Do the specialists that carry this channel agree about it?

    `signals` maps model_id -> that model's signed evidence for the channel.
    Agreement in sign with comparable magnitude -> ~1. Disagreement -> ~0.

    Disagreement between C1 and the full model about TSC is a red flag, not an
    average to take -- which is why this returns a multiplier rather than a mean.
    """
    v = np.asarray([s for s in signals.values() if np.isfinite(s)], dtype=np.float64)
    if v.size == 0:
        return 0.0
    if v.size < min_models:
        return 0.5                                  # only one witness: half credit
    same = np.abs(np.sign(v).sum()) / v.size        # 1.0 if all agree in sign
    scale = 1.0 - (v.std() / (np.abs(v).mean() + 1e-9)).clip(0, 1)
    return float(np.clip(same * scale, 0.0, 1.0))


@dataclass
class ChannelSolidity:
    channel: str
    model_id: str
    reliability: float
    extremeness: float
    counterfactual: float
    stability: float
    corroboration: float
    signed_evidence: float
    solidity: float
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def explain(self) -> str:
        return (f"{self.channel} via {self.model_id}: solidity {self.solidity:.3f} "
                f"= reliability {self.reliability:.2f} x extremeness "
                f"{self.extremeness:.2f} x counterfactual {self.counterfactual:.2f} "
                f"x stability {self.stability:.2f} x corroboration "
                f"{self.corroboration:.2f}")


def solidity(matrix: ReliabilityMatrix, channel: str,
             signed_evidence: float,
             percentile: Optional[float] = None,
             counterfactual_survived: Optional[bool] = None,
             stability_score: Optional[float] = None,
             cross_model_signals: Optional[Dict[str, float]] = None,
             among: Optional[Sequence[str]] = None) -> ChannelSolidity:
    """
    One channel's solidity on one clip.

    Every factor is in [0,1] and every factor is stored, so the product can be
    audited line by line rather than taken on faith. A zero in any factor zeroes
    the channel, which is the intended behaviour: an unreliable channel, a
    middling measurement, an explanation that does not survive its own
    counterfactual, or an unstable one, each on its own is enough reason not to
    let this evidence move a confidence number.
    """
    model_id, rel = matrix.best_model_for(channel, among=among)
    ext = extremeness(percentile)
    cf = (1.0 if counterfactual_survived else 0.0) if counterfactual_survived is not None else 0.5
    stab = float(np.clip(stability_score, 0.0, 1.0)) if stability_score is not None else 0.5
    corr = corroboration(cross_model_signals) if cross_model_signals else 0.5

    s = float(rel * ext * cf * stab * corr)
    notes: List[str] = []
    if rel == 0.0:
        notes.append(f"no model has earned reliability on the {channel} channel "
                     "(it did not beat its random control, or its AUC is at "
                     "chance), so this evidence is ignored entirely.")
    if ext == 0.0 and percentile is not None:
        notes.append(f"the measurement sits at the {percentile:.0f}th percentile "
                     "of real videos -- unremarkable, so it is not evidence.")
    if counterfactual_survived is False:
        notes.append("removing this channel's evidence did not move the logit, so "
                     "the model's answer did not depend on it.")
    return ChannelSolidity(channel=channel, model_id=model_id or "none",
                           reliability=rel, extremeness=ext, counterfactual=cf,
                           stability=stab, corroboration=corr,
                           signed_evidence=float(signed_evidence), solidity=s,
                           notes=notes)


# ==========================================================================
# 3. the confidence stack
# ==========================================================================


FEATURE_ORDER: Tuple[str, ...] = tuple(
    [f"solid_{c}" for c in CHANNELS] +
    [f"evid_{c}" for c in CHANNELS] +
    ["p_best_single", "agreement", "uncertainty", "n_solid"])


def build_features(solidities: Sequence[ChannelSolidity],
                   p_best_single: float, agreement: float,
                   uncertainty: float) -> np.ndarray:
    """
    The stack's input vector, in a fixed documented order.

    Per channel: its solidity, and its solidity-weighted signed evidence. Plus
    the best single model's calibrated probability (so the stack can only improve
    on it, never has to rediscover it), cross-model agreement, the uncertainty
    estimate, and how many channels were solid at all.

    14 features for a 976-clip validation split is comfortable for an L2 logistic
    regression. Adding more would not be.
    """
    by_ch = {s.channel: s for s in solidities}
    feats: List[float] = []
    for c in CHANNELS:
        feats.append(by_ch[c].solidity if c in by_ch else 0.0)
    for c in CHANNELS:
        s = by_ch.get(c)
        feats.append((s.solidity * s.signed_evidence) if s else 0.0)
    feats += [float(logit(np.array([p_best_single]))[0]), float(agreement),
              float(uncertainty),
              float(sum(1 for s in solidities if s.solidity > 0.1))]
    return np.asarray(feats, dtype=np.float64)


@dataclass
class StackReport:
    n_train: int
    coefficients: Dict[str, float]
    intercept: float
    temperature: float
    C: float
    val_auc: float = float("nan")
    val_nll: float = float("nan")
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:
        rows = sorted(self.coefficients.items(), key=lambda kv: -abs(kv[1]))
        head = (f"ConfidenceStack  n={self.n_train}  C={self.C}  "
                f"T={self.temperature:.3f}  val AUC={self.val_auc:.4f}")
        body = "\n".join(f"  {k:<22} {v:+.4f}" for k, v in rows)
        return head + "\n" + body


class ConfidenceStack:
    """
    L2 logistic regression over the solidity-weighted evidence, then temperature
    scaling.

    Why a fitted stack and not a formula: the coefficients are the arbitration
    weights. Learned on val, printable in the report, and falsifiable. A
    hand-written weighted sum would be an assertion dressed as a measurement.

    Standardisation is fitted on the training features and stored, so a single
    clip at inference is transformed exactly as the training clips were.
    """

    def __init__(self, C: float = 1.0, feature_names: Sequence[str] = FEATURE_ORDER):
        self.C = float(C)
        self.feature_names = list(feature_names)
        self.clf = None
        self.mu: Optional[np.ndarray] = None
        self.sd: Optional[np.ndarray] = None
        self.temp = TemperatureScaler()
        self.report_: Optional[StackReport] = None

    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray,
            calibrate_on_holdout: float = 0.3, seed: int = 42) -> "ConfidenceStack":
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != len(self.feature_names):
            raise ValueError(f"expected (n,{len(self.feature_names)}) features, "
                             f"got {X.shape}")
        if len(np.unique(y)) < 2:
            raise ValueError("need both classes present to fit the stack")

        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(y))
        n_cal = int(round(calibrate_on_holdout * len(y)))
        cal_idx, fit_idx = idx[:n_cal], idx[n_cal:]
        if len(np.unique(y[fit_idx])) < 2 or len(np.unique(y[cal_idx])) < 2:
            # too small to split: fit and calibrate on the same rows, and say so
            fit_idx, cal_idx = idx, idx

        self.mu = X[fit_idx].mean(axis=0)
        self.sd = X[fit_idx].std(axis=0)
        self.sd[self.sd < 1e-9] = 1.0

        Z = (X[fit_idx] - self.mu) / self.sd
        self.clf = LogisticRegression(C=self.C, penalty="l2", max_iter=2000)
        self.clf.fit(Z, y[fit_idx])

        z_cal = self._margin(X[cal_idx])
        self.temp.fit(z_cal, y[cal_idx])

        p_all = self.predict_proba(X)
        notes: List[str] = []
        if np.array_equal(fit_idx, cal_idx):
            notes.append("the stack was fitted and temperature-calibrated on the "
                         "same rows because the split was too small to divide. "
                         "The calibration figures from this fit are optimistic.")
        coefs = dict(zip(self.feature_names,
                         [float(c) for c in self.clf.coef_.ravel()]))
        top = max(coefs.items(), key=lambda kv: abs(kv[1]))
        if top[0] == "p_best_single":
            notes.append("the largest coefficient is on the best single model's "
                         "own probability: the stack has mostly learned to copy "
                         "it. Check the paired bootstrap against that baseline "
                         "before claiming arbitration helps.")
        try:
            auc = float(roc_auc_score(y, p_all))
        except Exception:
            auc = float("nan")
        from .calibrate import nll as _nll
        self.report_ = StackReport(
            n_train=int(len(fit_idx)), coefficients=coefs,
            intercept=float(self.clf.intercept_[0]), temperature=self.temp.T,
            C=self.C, val_auc=auc, val_nll=_nll(y, p_all), notes=notes)
        log.info(str(self.report_))
        return self

    # ------------------------------------------------------------------
    def _margin(self, X: np.ndarray) -> np.ndarray:
        if self.clf is None or self.mu is None:
            raise RuntimeError("ConfidenceStack is not fitted")
        Z = (np.atleast_2d(np.asarray(X, dtype=np.float64)) - self.mu) / self.sd
        return self.clf.decision_function(Z)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Calibrated p(fake). Single-row input returns a length-1 array."""
        return self.temp.transform(self._margin(X))

    def predict_one(self, features: np.ndarray) -> float:
        return float(self.predict_proba(np.asarray(features).reshape(1, -1))[0])

    # ------------------------------------------------------------------
    def coefficients(self) -> Dict[str, float]:
        return dict(self.report_.coefficients) if self.report_ else {}

    def to_dict(self) -> dict:
        return {"C": self.C, "feature_names": self.feature_names,
                "mu": None if self.mu is None else self.mu.tolist(),
                "sd": None if self.sd is None else self.sd.tolist(),
                "coef": None if self.clf is None else self.clf.coef_.ravel().tolist(),
                "intercept": None if self.clf is None else float(self.clf.intercept_[0]),
                "temperature": self.temp.T,
                "report": self.report_.to_dict() if self.report_ else None}

    def save(self, path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @staticmethod
    def load(path) -> "ConfidenceStack":
        from sklearn.linear_model import LogisticRegression
        d = json.loads(Path(path).read_text())
        s = ConfidenceStack(C=d.get("C", 1.0),
                            feature_names=d.get("feature_names", FEATURE_ORDER))
        s.mu = np.asarray(d["mu"], dtype=np.float64)
        s.sd = np.asarray(d["sd"], dtype=np.float64)
        clf = LogisticRegression()
        clf.coef_ = np.asarray([d["coef"]], dtype=np.float64)
        clf.intercept_ = np.asarray([d["intercept"]], dtype=np.float64)
        clf.classes_ = np.asarray([0, 1])
        s.clf = clf
        s.temp = TemperatureScaler(d.get("temperature", 1.0))
        if d.get("report"):
            r = dict(d["report"])
            s.report_ = StackReport(**r)
        return s


# ==========================================================================
# routing: which model answers (separate from how confident we are)
# ==========================================================================


@dataclass
class RoutingDecision:
    answering_model: str
    reason: str
    per_channel: List[Dict[str, object]] = field(default_factory=list)
    total_solidity: float = 0.0
    admissible_modalities: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def route(matrix: ReliabilityMatrix, solidities: Sequence[ChannelSolidity],
          specialists: Dict[str, Tuple[str, ...]] = None,
          among: Optional[Sequence[str]] = None) -> RoutingDecision:
    """
    Pick the model that answers, by summing solidity over the channels it carries.

    This is a *separate decision* from how confident the answer is -- a clip can
    be confidently routed to C1 and still land inside the abstention band. Keeping
    them apart is what makes the logs readable, and conflating them is the easiest
    way to produce a confidence number that means nothing.

    `admissible_modalities` is the modality gate: only channels with real solidity
    may be used to explain the verdict. If the spatial channel contributed nothing,
    a Grad-CAM map is not an explanation of this decision, however good it looks.
    """
    specialists = specialists or DEFAULT_SPECIALISTS
    pool = list(among) if among else list(specialists.keys())
    by_ch = {s.channel: s for s in solidities}

    totals: Dict[str, float] = {}
    for mid in pool:
        chans = specialists.get(mid, ())
        totals[mid] = float(sum(by_ch[c].solidity for c in chans if c in by_ch))

    best = max(totals, key=totals.get) if totals else "none"
    admissible = sorted([c for c, s in by_ch.items() if s.solidity > 0.1],
                        key=lambda c: -by_ch[c].solidity)

    if not totals or totals[best] <= 0.0:
        return RoutingDecision(
            answering_model=(among[0] if among else BASELINE_ID),
            reason=("no channel carried solid evidence on this clip, so routing "
                    "falls back to the model with the best honest AUC and the "
                    "verdict should be abstained on"),
            per_channel=[s.to_dict() for s in solidities],
            total_solidity=0.0, admissible_modalities=[])

    ranked = sorted(totals.items(), key=lambda kv: -kv[1])
    reason = (f"{best} carries the solid channels on this clip "
              f"(total solidity {totals[best]:.3f}; next: "
              + ", ".join(f"{m} {v:.3f}" for m, v in ranked[1:3]) + ")")
    return RoutingDecision(answering_model=best, reason=reason,
                           per_channel=[s.to_dict() for s in solidities],
                           total_solidity=totals[best],
                           admissible_modalities=admissible)


def agreement_across_models(probs: Dict[str, float]) -> float:
    """
    How much the specialists agree on the verdict, in [0,1].

    1.0 = every model on the same side of 0.5 with similar confidence.
    Used as a stack feature, never as a verdict on its own -- four models that
    share three branches will agree most of the time for reasons that have
    nothing to do with the clip.
    """
    v = np.asarray([p for p in probs.values() if np.isfinite(p)], dtype=np.float64)
    if v.size < 2:
        return 0.5
    same = float(np.abs(np.sign(v - 0.5).sum()) / v.size)
    spread = float(1.0 - np.clip(v.std(ddof=1) / 0.25, 0.0, 1.0))
    return float(np.clip(0.5 * same + 0.5 * spread, 0.0, 1.0))


def compare_to_single_model(y: Sequence[float], p_stack: Sequence[float],
                            p_single: Sequence[float], label: str = BASELINE_ID,
                            n: int = 5000) -> Dict[str, object]:
    """
    The check that decides whether arbitration ships: does the stack beat the
    best single model on the *same* clips?

    Uses the project's existing paired bootstrap, because the two models are
    scored on the same videos and their separate confidence intervals are
    therefore not independent.
    """
    from .benchmark import paired_bootstrap
    r = paired_bootstrap(np.asarray(y), np.asarray(p_stack),
                         np.asarray(p_single), n=n)
    resolved = bool(np.isfinite(r.get("p_value", np.nan)) and r["p_value"] < 0.05)
    verdict = ("arbitration beats " + label if resolved and r["delta_auc"] > 0 else
               "arbitration LOSES to " + label if resolved else
               "not resolved -- the data cannot separate them")
    if not (resolved and r["delta_auc"] > 0):
        r["recommendation"] = (f"ship {label} with temperature scaling. The stack "
                               "does not earn its complexity on this evidence, "
                               "and reporting it as an improvement would be "
                               "unsupported.")
    return {**r, "baseline": label, "verdict": verdict, "resolved": resolved}


__all__ = ["CHANNELS", "DEFAULT_SPECIALISTS", "CONDITION_LABELS", "BASELINE_ID", "ChannelReliability",
           "ReliabilityMatrix", "build_reliability_matrix", "extremeness",
           "corroboration", "ChannelSolidity", "solidity", "FEATURE_ORDER",
           "build_features", "ConfidenceStack", "StackReport", "RoutingDecision",
           "route", "agreement_across_models", "compare_to_single_model"]
