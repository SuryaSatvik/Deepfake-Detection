"""
cfx.py
======

**CFX -- Calibrated & Faithful eXplanation.** The orchestrator: one clip in, one
`ExplanationCard` out.

What it does, in order
----------------------
1. Every specialist checkpoint scores the clip (forward only).
2. Exact branch Shapley on the routed model -> where the decision came from.
3. Per-channel solidity (`arbitrate`) -> which evidence has earned a voice.
4. The confidence stack -> one calibrated probability; the abstention policy ->
   fake / real / abstain.
5. **Gate.** Only if the verdict clears the gate do the expensive explainers run
   (KernelSHAP, counterfactuals, faithfulness scoring). That is both the compute
   story on an 8 GB card and the point of the design: an unexplainable verdict is
   not worth explaining.
6. Candidates are scored for faithfulness against a random control, filtered by
   the modality gate, and the winner is chosen.
7. Both narration registers are generated once and cached on the card.

The four-quadrant output policy
-------------------------------
| verdict confidence | explanation faithfulness | output |
|---|---|---|
| high | high | verdict + explanation + both scores |
| high | low  | verdict + "no reliable explanation available" |
| low  | high | no verdict; evidence as an observation, not a conclusion |
| low  | low  | abstain entirely |

The abstraction boundary
------------------------
`card.for_user(register)` returns only what a viewer should see: verdict,
confidence phrase, one explanation, one figure. `card.for_audit()` returns
everything else -- routing trace, Shapley values, every candidate's faithfulness
and its margin over random, the conformal band, the assumed prior, the raw card.

Model selection is never a user setting. Arbitration is the contribution; handing
it to a checkbox discards it.

Honest note, on purpose
-----------------------
At the identity-disjoint numbers this project has actually measured (test AUC
~0.57, full model 0.111 below its own ablation), **most clips should land in
abstain**. `assert_coverage_sane` will fail a run where they do not. That is the
intended behaviour: a confident-looking report from a miscalibrated run is the
most expensive failure mode here, because it looks like a result.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from .utils import log
except Exception:                                     # pragma: no cover
    import logging
    log = logging.getLogger("fsvnet")

from . import arbitrate as AB
from . import faithfulness as FH
from . import narrate as NR
from .attribution import (branch_shapley_exact, counterfactual_spectral,
                          counterfactual_temporal, integrated_gradients,
                          kernel_shap_superpixel)
from .selective import ABSTAIN, AbstentionPolicy, assert_coverage_sane
from .uncertainty import estimate_uncertainty, predict_prob


# ==========================================================================
# configuration
# ==========================================================================


@dataclass
class CFXConfig:
    """
    CFX's own tunables. Kept out of `Config` deliberately: `config.py` is the
    single source of truth for the detector and the training run, and this layer
    is downstream of both. Pass an instance to `CFXPipeline`; the defaults are the
    ones the design argues for.
    """
    # --- arbitration -------------------------------------------------
    specialists: Tuple[str, ...] = tuple(AB.DEFAULT_SPECIALISTS.keys())
    fallback_model: str = AB.BASELINE_ID     # B0: the best honest AUC of the family

    # --- confidence --------------------------------------------------
    alpha: float = 0.10                  # target selective error rate
    assumed_prior: float = 0.864          # Celeb-DF-v2 fake rate; state it always
    deploy_prior: Optional[float] = None  # set to re-prior the output

    # --- gating ------------------------------------------------------
    explain_gate_confidence: float = 0.60   # |p-0.5|*2 below this -> cheap only
    min_faithfulness_margin: float = 0.0    # must beat random by more than this
    min_stability: float = 0.5
    min_solidity: float = 0.10

    # --- compute -----------------------------------------------------
    ig_steps: int = 32
    kshap_samples: int = 512
    kshap_segments: int = 50
    kshap_frames: int = 2
    faith_fills: Tuple[str, ...] = ("blur", "real_patch")
    n_random_controls: int = 3
    patch: int = 16
    do_tta: bool = True
    mc_passes: int = 0
    cheap_only: bool = False             # CPU fallback: skip KernelSHAP etc.

    # --- narration ---------------------------------------------------
    lexicon_path: Optional[str] = None
    narrate: bool = True

    def cpu_fallback(self) -> "CFXConfig":
        """The documented CPU profile: ~400 forwards per clip instead of ~2,800."""
        self.cheap_only = True
        self.ig_steps = 16
        self.kshap_samples = 0
        self.faith_fills = ("blur",)
        self.n_random_controls = 2
        self.do_tta = False
        self.mc_passes = 0
        return self

    def to_dict(self) -> dict:
        return asdict(self)


# ==========================================================================
# the card
# ==========================================================================


@dataclass
class Verdict:
    p_fake_raw: float
    p_fake_calibrated: float
    band: Tuple[float, float]
    uncertainty: float
    decision: str
    alpha: float
    assumed_prior: float
    answering_model: str
    routing_reason: str = ""
    routing_trace: Dict[str, object] = field(default_factory=dict)
    per_model_prob: Dict[str, float] = field(default_factory=dict)
    agreement: float = 0.5
    confident: bool = False

    def to_dict(self) -> dict:
        d = asdict(self)
        d["band"] = list(self.band)
        return d


@dataclass
class ScoredExplanation:
    method: str
    modality: str
    faithfulness: Dict[str, float] = field(default_factory=dict)
    margin_over_random: float = float("nan")
    beats_random: bool = False
    stability: float = float("nan")
    solidity: float = 0.0
    admissible: bool = False
    grade: str = "withheld"
    statement: str = ""
    payload_kind: str = ""
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ExplanationCard:
    clip_id: str
    verdict: Verdict
    best: Optional[ScoredExplanation] = None
    all_scored: List[ScoredExplanation] = field(default_factory=list)
    shapley: Dict[str, object] = field(default_factory=dict)
    evidence_items: List[Dict[str, object]] = field(default_factory=list)
    narration: Optional[NR.Narration] = None
    withheld_reason: Optional[str] = None
    quadrant: str = ""
    figures: Dict[str, str] = field(default_factory=dict)
    n_forwards: int = 0
    payloads: Dict[str, object] = field(default_factory=dict, repr=False)
    notes: List[str] = field(default_factory=list)

    # ---------------------------------------------------------------
    def for_user(self, register: str = NR.LAYMAN) -> Dict[str, object]:
        """
        Exactly what a viewer sees. Note what is absent: no model name, no
        Shapley values, no faithfulness numbers, no alpha. The confidence NUMBER
        is present and so is its band -- a confidence without a band is the one
        thing this layer must never show.
        """
        v = self.verdict
        return {
            "decision": v.decision,
            "confidence_number": round(v.p_fake_calibrated, 3),
            "confidence_band": [round(v.band[0], 3), round(v.band[1], 3)],
            "text": self.narration.text(register) if self.narration else "",
            "explanation_available": self.best is not None,
            "explanation_of": self.best.statement if self.best else
                             (self.withheld_reason or ""),
            "figure": self.figures.get("primary", ""),
            "register": register,
        }

    def for_audit(self) -> Dict[str, object]:
        """Everything the front surface hides. Collapsed in the UI, never removed."""
        return {
            "verdict": self.verdict.to_dict(),
            "routing": self.verdict.routing_trace,
            "shapley": self.shapley,
            "candidates": [s.to_dict() for s in self.all_scored],
            "evidence_items": self.evidence_items,
            "narration": self.narration.to_dict() if self.narration else None,
            "quadrant": self.quadrant,
            "n_forwards": self.n_forwards,
            "notes": self.notes,
        }

    def to_dict(self) -> dict:
        return {"clip_id": self.clip_id, **self.for_audit(),
                "best": self.best.to_dict() if self.best else None,
                "withheld_reason": self.withheld_reason,
                "figures": self.figures}

    def save(self, path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, default=str),
                             encoding="utf-8")


# ==========================================================================
# the pipeline
# ==========================================================================


class CFXPipeline:
    """
        pipe = CFXPipeline(models={"C1_conventional": m1, "C2_tsc": m2, ...},
                           matrix=ReliabilityMatrix.load(...),
                           stack=ConfidenceStack.load(...),
                           policy=AbstentionPolicy.load(...),
                           cfx=CFXConfig(), reasoner=EvidenceReasoner(cfg))
        card = pipe.analyse(clip, mask, lmf, frames_u8, clip_id="id12_0003")
        print(card.for_user("layman")["text"])

    Every component is optional. Without a fitted stack the routed model's own
    probability is used and the card says so; without a policy nothing is ever
    abstained on and the card says that too. Missing pieces degrade the output and
    are recorded -- they never silently change the meaning of a number.
    """

    def __init__(self, models: Dict[str, object],
                 matrix: Optional[AB.ReliabilityMatrix] = None,
                 stack: Optional[AB.ConfidenceStack] = None,
                 policy: Optional[AbstentionPolicy] = None,
                 cfx: Optional[CFXConfig] = None,
                 reasoner=None,
                 reference=None,
                 specialists: Optional[Dict[str, Tuple[str, ...]]] = None):
        if not models:
            raise ValueError("CFXPipeline needs at least one model")
        self.models = dict(models)
        self.matrix = matrix or AB.ReliabilityMatrix()
        self.stack = stack
        self.policy = policy
        self.cfx = cfx or CFXConfig()
        self.reasoner = reasoner
        self.reference = reference
        self.specialists = specialists or AB.DEFAULT_SPECIALISTS
        self.lex = NR.Lexicon.load(self.cfx.lexicon_path)

    # ------------------------------------------------------------------
    def _primary(self, model_id: str):
        return self.models.get(model_id) or next(iter(self.models.values()))

    # ------------------------------------------------------------------
    def _score_all(self, clip, mask, lmf) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for mid, m in self.models.items():
            try:
                out[mid] = predict_prob(m, clip, mask, lmf)
            except Exception as e:
                log.warning(f"specialist '{mid}' failed on this clip "
                            f"({e.__class__.__name__}: {e}) -- excluded")
        return out

    # ------------------------------------------------------------------
    def _evidence_from_model(self, model, clip, mask, lmf
                             ) -> Tuple[Dict[str, float], Dict[str, Optional[float]]]:
        """TSC descriptors and their percentiles against the real-video reference."""
        import torch
        with torch.no_grad():
            _, info = model(clip, mask, lmf=lmf, return_evidence=True)
        ev = info.get("evidence", {}) or {}
        desc_t = ev.get("tsc_descriptors", {}) or {}
        desc = {k: float(v[0]) for k, v in desc_t.items()}
        pct: Dict[str, Optional[float]] = {}
        if self.reference is not None:
            for k, v in desc.items():
                pct[k] = self.reference.percentile_of(k, v)
        return desc, pct

    # ------------------------------------------------------------------
    def _solidities(self, shap: "object", desc, pct, per_model,
                    stability_hint: float) -> List[AB.ChannelSolidity]:
        """
        Channel solidity, built from what was actually measured on this clip.

        The signed evidence for a channel is its Shapley value (how far it moved
        the fake-margin). Extremeness comes from the descriptor percentiles where
        a channel has descriptors, and from the Shapley magnitude where it does
        not. Counterfactual survival is the Shapley value being non-negligible --
        a channel whose removal does not move the logit did not carry the answer,
        whatever the gate said about it.
        """
        # map channel -> the descriptors that belong to it
        tsc_keys = [k for k in desc if k.startswith(("temporal_", "flicker_",
                                                     "fine_", "spectral_"))]
        sols: List[AB.ChannelSolidity] = []
        phi = dict(getattr(shap, "values", {}) or {})
        norm = dict(getattr(shap, "normalised", {}) or {})
        for ch in AB.CHANNELS:
            signed = float(phi.get(ch, 0.0))
            share = float(norm.get(ch, 0.0))
            if ch == "tsc" and tsc_keys and pct:
                ps = [pct[k] for k in tsc_keys if pct.get(k) is not None]
                # the most extreme calibrated descriptor speaks for the channel
                p_use = (max(ps, key=lambda x: abs(x - 50.0)) if ps else None)
            else:
                # no reference distribution for this channel: fall back to the
                # Shapley share, mapped onto the same 0-100 percentile scale so
                # `extremeness` has something meaningful to read
                p_use = 50.0 + 50.0 * min(share / 0.5, 1.0) if share > 0 else 50.0
            sols.append(AB.solidity(
                self.matrix, ch, signed_evidence=signed, percentile=p_use,
                counterfactual_survived=bool(share > 0.10),
                stability_score=stability_hint,
                cross_model_signals={m: (p - 0.5) for m, p in per_model.items()
                                     if ch in self.specialists.get(m, ())},
                among=list(self.cfx.specialists)))
        return sols

    # ------------------------------------------------------------------
    def _calibrated_probability(self, sols, per_model, routed_id, unc
                                ) -> Tuple[float, List[str]]:
        notes: List[str] = []
        p_single = per_model.get(routed_id, float(np.mean(list(per_model.values()))))
        agree = AB.agreement_across_models(per_model)
        if self.stack is None:
            notes.append("no fitted confidence stack: the routed model's own "
                         "probability is reported, uncalibrated by arbitration. "
                         "Run scripts/fit_calibration.py before reporting any "
                         "number from this card.")
            return float(p_single), notes
        feats = AB.build_features(sols, p_best_single=p_single, agreement=agree,
                                 uncertainty=float(unc.total_sd))
        return float(self.stack.predict_one(feats)), notes

    # ------------------------------------------------------------------
    def _candidates(self, model, clip, mask, lmf, frames_u8, shap, cheap: bool
                    ) -> Tuple[List[Tuple[FH.Ranking, str, object]], int]:
        """
        Build the candidate rankings. `cheap=True` (gate not cleared, or CPU
        profile) keeps only the methods that cost a handful of forwards.
        """
        from .explain import temporal_occlusion

        cands: List[Tuple[FH.Ranking, str, object]] = []
        n_fwd = 0

        r_branch = FH.ranking_from_branches(dict(shap.values), method="branch_shapley")
        cands.append((r_branch, "branch", None))
        n_fwd += int(getattr(shap, "n_forwards", 0))

        occ = temporal_occlusion(model, clip, mask)
        cands.append((FH.ranking_from_frames(occ["delta"],
                                             method="temporal_occlusion"),
                      "temporal", None))
        n_fwd += int(clip.shape[1]) + 1

        ig = integrated_gradients(model, clip, mask, lmf, baseline="blur",
                                 steps=self.cfx.ig_steps)
        cands.append((FH.ranking_from_spatial(ig["attribution"], self.cfx.patch,
                                              method="integrated_gradients"),
                      "spatial", ig))
        n_fwd += 2 * self.cfx.ig_steps

        if cheap:
            return cands, n_fwd

        top = list(np.argsort(-occ["delta"])[:max(1, self.cfx.kshap_frames)])
        if self.cfx.kshap_samples > 0 and frames_u8 is not None and len(frames_u8):
            ks = kernel_shap_superpixel(
                model, clip, frames_u8, top, mask, lmf,
                n_segments=self.cfx.kshap_segments,
                n_samples=self.cfx.kshap_samples)
            r_sp, labels = FH.ranking_from_superpixels(ks["per_frame"],
                                                       method="kernel_shap")
            cands.append((r_sp, "spatial", {"kshap": ks, "labels": labels}))
            n_fwd += int(ks["n_forwards"])

        cf_s = counterfactual_spectral(model, clip, mask, lmf)
        cf_t = counterfactual_temporal(model, clip, mask, lmf,
                                      order=list(np.argsort(-occ["delta"])))
        n_fwd += cf_s.n_forwards + cf_t.n_forwards
        cands.append((FH.ranking_from_frames(occ["delta"],
                                             method="counterfactual_temporal"),
                      "temporal", cf_t))
        cands.append((FH.ranking_from_spatial(ig["attribution"], self.cfx.patch,
                                              method="counterfactual_spectral"),
                      "spectral", cf_s))
        return cands, n_fwd

    # ------------------------------------------------------------------
    def analyse(self, clip, mask=None, lmf=None, frames_u8=None,
                clip_id: str = "clip", window_probs: Optional[Sequence[float]] = None
                ) -> ExplanationCard:
        n_fwd = 0
        notes: List[str] = []

        # ---- 1. every specialist scores the clip -----------------------
        per_model = self._score_all(clip, mask, lmf)
        n_fwd += len(per_model)
        if not per_model:
            raise RuntimeError("no specialist produced a score for this clip")

        # ---- 2. uncertainty, on the best honest single model ------------
        primary_id = (self.cfx.fallback_model if self.cfx.fallback_model in self.models
                      else next(iter(self.models)))
        unc = estimate_uncertainty(self._primary(primary_id), clip, mask, lmf,
                                   window_probs=window_probs,
                                   do_tta=self.cfx.do_tta,
                                   mc_passes=self.cfx.mc_passes)
        n_fwd += unc.n_passes
        notes += unc.notes

        # ---- 3. Shapley on the primary, then route ---------------------
        shap = branch_shapley_exact(self._primary(primary_id), clip, mask, lmf)
        n_fwd += shap.n_forwards
        notes += shap.notes

        desc, pct = self._evidence_from_model(self._primary(primary_id), clip,
                                              mask, lmf)
        n_fwd += 1
        if self.reference is None:
            notes.append("no reference distribution loaded, so the descriptors "
                         "are not percentile-calibrated. Extremeness falls back "
                         "to the Shapley share, which is a weaker signal -- and "
                         "no claim that a measurement is 'anomalous' is possible.")

        stability_hint = 0.5
        sols = self._solidities(shap, desc, pct, per_model, stability_hint)
        routing = AB.route(self.matrix, sols, self.specialists,
                           among=list(self.cfx.specialists))
        model = self._primary(routing.answering_model)

        # ---- 4. calibrated probability + decision ----------------------
        p_cal, cal_notes = self._calibrated_probability(sols, per_model,
                                                        routing.answering_model, unc)
        notes += cal_notes
        if self.cfx.deploy_prior is not None:
            from .calibrate import reprior
            p_cal = float(reprior(np.array([p_cal]), self.cfx.assumed_prior,
                                  self.cfx.deploy_prior)[0])
            notes.append(f"probability re-priored from {self.cfx.assumed_prior:.3f} "
                         f"to {self.cfx.deploy_prior:.3f}.")

        if self.policy is not None:
            decision = self.policy.decide(p_cal)
            band = self.policy.band()
            alpha = self.policy.alpha
        else:
            decision = "fake" if p_cal >= 0.5 else "real"
            band = (0.5, 0.5)
            alpha = float("nan")
            notes.append("no abstention policy fitted, so nothing is abstained "
                         "on and the four-quadrant policy collapses to two. Fit "
                         "one with selective.fit_abstention before reporting.")

        conf = abs(p_cal - 0.5) * 2.0
        confident = bool(decision != ABSTAIN and conf >= self.cfx.explain_gate_confidence)

        verdict = Verdict(
            p_fake_raw=float(per_model.get(routing.answering_model, p_cal)),
            p_fake_calibrated=float(p_cal), band=band,
            uncertainty=float(unc.total_sd), decision=decision, alpha=alpha,
            assumed_prior=self.cfx.assumed_prior,
            answering_model=routing.answering_model,
            routing_reason=routing.reason,
            routing_trace={**routing.to_dict(),
                           "uncertainty": unc.to_dict(),
                           "descriptors": desc, "percentiles": pct},
            per_model_prob=per_model,
            agreement=AB.agreement_across_models(per_model), confident=confident)

        card = ExplanationCard(clip_id=clip_id, verdict=verdict,
                               shapley=shap.to_dict(), notes=notes)

        # ---- 5. the gate ----------------------------------------------
        if decision == ABSTAIN:
            card.quadrant = "low-confidence"
            card.withheld_reason = ("the verdict falls inside the abstention "
                                    "band, so no explanation is offered: there "
                                    "is no decision to explain")
            card.evidence_items = self._evidence_items(sols, desc, pct)
            card.narration = self._narrate(card, [])
            card.n_forwards = n_fwd
            return card

        cheap = self.cfx.cheap_only or not confident
        if cheap and not self.cfx.cheap_only:
            notes.append(f"confidence {conf:.2f} below the explain gate "
                         f"({self.cfx.explain_gate_confidence:.2f}): only the "
                         "cheap explainers ran. This is the designed behaviour, "
                         "not a truncated run.")

        cands, k = self._candidates(model, clip, mask, lmf, frames_u8, shap, cheap)
        n_fwd += k

        # ---- 6. score the candidates, apply the modality gate ----------
        solid_by_ch = {s.channel: s.solidity for s in sols}
        modality_solidity = {
            "spatial": solid_by_ch.get("spatial", 0.0),
            "spectral": max(solid_by_ch.get("frequency", 0.0),
                            solid_by_ch.get("tsc", 0.0)),
            "temporal": max(solid_by_ch.get("tsc", 0.0),
                            solid_by_ch.get("lmf", 0.0)),
            "branch": max(solid_by_ch.values()) if solid_by_ch else 0.0,
        }

        scored: List[ScoredExplanation] = []
        for ranking, modality, payload in cands:
            sol = float(modality_solidity.get(modality, 0.0))
            se = ScoredExplanation(method=ranking.method, modality=modality,
                                   solidity=sol,
                                   payload_kind=type(payload).__name__)
            if modality != "branch" and sol < self.cfx.min_solidity:
                se.admissible = False
                se.grade = "withheld"
                se.notes.append(
                    f"the {modality} channels carried only {sol:.3f} of the "
                    "decision on this clip, so a "
                    f"{ranking.method} map would be explaining almost none of "
                    "it. Withheld by the modality gate, not by its own score.")
                scored.append(se)
                continue

            if ranking.space == "branch":
                # branch rankings are validated by the Shapley efficiency check,
                # not by masking pixels -- there is nothing to occlude.
                se.admissible = True
                se.beats_random = True
                se.margin_over_random = float("nan")
                se.faithfulness = {"shapley_efficiency_residual":
                                   float(shap.efficiency_residual)}
                se.grade = "A" if abs(shap.efficiency_residual) < 1e-3 else "C"
                se.statement = self._branch_statement(shap)
                scored.append(se)
                continue

            labels = payload.get("labels") if isinstance(payload, dict) else None
            try:
                verd = FH.evaluate_candidate(
                    model, clip, ranking, mask, lmf,
                    fills=tuple(self.cfx.faith_fills),
                    n_random=self.cfx.n_random_controls, labels=labels)
                n_fwd += sum(s.n_forwards for s in verd.scores)
            except Exception as e:
                se.notes.append(f"faithfulness scoring failed "
                                f"({e.__class__.__name__}: {e})")
                scored.append(se)
                continue

            se.faithfulness = {**verd.per_metric, "borda": verd.borda,
                               "borda_random": verd.borda_random}
            se.margin_over_random = verd.margin_over_random
            se.beats_random = verd.beats_random
            se.notes += verd.notes
            se.admissible = bool(verd.beats_random and
                                 verd.margin_over_random > self.cfx.min_faithfulness_margin)
            se.grade = ("A" if se.admissible and verd.fills_agree else
                        "B" if se.admissible else "withheld")
            if isinstance(payload, object) and hasattr(payload, "statement"):
                se.statement = getattr(payload, "statement", "")
            scored.append(se)

        card.all_scored = scored
        admissible = [s for s in scored
                      if s.admissible and s.modality != "branch"]
        best = max(admissible, key=lambda s: (s.margin_over_random
                                              if np.isfinite(s.margin_over_random)
                                              else -1e9), default=None)
        if best is None:
            branch_ok = next((s for s in scored if s.modality == "branch"
                              and s.admissible), None)
            best = branch_ok

        card.best = best
        card.evidence_items = self._evidence_items(sols, desc, pct)

        # ---- 7. the quadrant -------------------------------------------
        if confident and best is not None:
            card.quadrant = "confident-explainable"
        elif confident and best is None:
            card.quadrant = "confident-unexplainable"
            card.withheld_reason = ("no candidate explanation beat a random "
                                    "ordering of the same units, so none is "
                                    "shown. The verdict stands; the reason for "
                                    "it does not")
        elif not confident and best is not None:
            card.quadrant = "unconfident-explainable"
            card.withheld_reason = ("the evidence below is an observation, not a "
                                    "conclusion: the verdict did not clear the "
                                    "confidence gate")
        else:
            card.quadrant = "unconfident-unexplainable"
            card.withheld_reason = "neither the verdict nor any explanation cleared its test"

        card.narration = self._narrate(card, card.evidence_items)
        card.n_forwards = n_fwd
        return card

    # ------------------------------------------------------------------
    def _branch_statement(self, shap) -> str:
        top = shap.dominant
        share = shap.normalised.get(top, 0.0)
        s = (f"the {top} channel carried {share:.0%} of the decision "
             f"(exact Shapley, {shap.n_forwards} coalitions)")
        ti = shap.top_interactions(1)
        if ti and abs(ti[0][1]) > 1e-3:
            a, b = ti[0][0].split("|")
            s += f"; {a} and {b} interact ({ti[0][1]:+.3f})"
        return s

    # ------------------------------------------------------------------
    def _evidence_items(self, sols, desc, pct) -> List[Dict[str, object]]:
        """
        The evidence the narration layer is allowed to mention: only channels
        that cleared the solidity floor, and only descriptors that are actually
        calibrated. What is not here cannot be narrated.
        """
        items: List[Dict[str, object]] = []
        for s in sols:
            if s.solidity < self.cfx.min_solidity:
                continue
            items.append({
                "id": f"ch_{s.channel}",
                "term": s.channel,
                "layman": self.lex.phrase(s.channel, NR.LAYMAN),
                "how_unusual": f"solidity {s.solidity:.2f}",
                "solidity": s.solidity,
                "signed_evidence": s.signed_evidence,
                "model": s.model_id,
            })
        for k, v in desc.items():
            p = pct.get(k)
            if p is None or abs(p - 50.0) < 40.0:        # not in the tails
                continue
            items.append({
                "id": f"desc_{k}",
                "term": k,
                "layman": self.lex.phrase(k, NR.LAYMAN),
                "how_unusual": f"{p:.0f}th percentile of real videos",
                "value": float(v), "percentile": float(p),
            })
        return items

    # ------------------------------------------------------------------
    def _technical_text(self, card: ExplanationCard) -> Tuple[str, List[float]]:
        v = card.verdict
        nums: List[float] = [round(v.p_fake_calibrated, 3), round(v.band[0], 3),
                             round(v.band[1], 3), round(v.uncertainty, 3)]
        bits = [f"Routed to {v.answering_model}: {v.routing_reason}."]
        if card.shapley:
            top = card.shapley.get("dominant", "")
            share = (card.shapley.get("normalised", {}) or {}).get(top, 0.0)
            bits.append(f"Exact branch Shapley attributes {share:.3f} of the "
                        f"decision to the {top} channel.")
            nums.append(round(float(share), 3))
        if card.best is not None:
            m = card.best.margin_over_random
            bits.append(f"Best explanation: {card.best.method} "
                        f"(grade {card.best.grade}"
                        + (f", Borda margin over random {m:+.3f}"
                           if np.isfinite(m) else "") + ").")
            if np.isfinite(m):
                nums.append(round(float(m), 3))
            if card.best.statement:
                bits.append(card.best.statement.capitalize() + ".")
        elif card.withheld_reason:
            bits.append(card.withheld_reason.capitalize() + ".")
        bits.append(f"Conformal band [{v.band[0]:.3f}, {v.band[1]:.3f}], "
                    f"uncertainty {v.uncertainty:.3f}, assuming a fake prior of "
                    f"{v.assumed_prior:.3f}.")
        nums.append(round(float(v.assumed_prior), 3))
        # every integer appearing in a statement (frame indices, counts) is
        # already in the card; collect them so numeral_fidelity does not flag them
        for s in card.all_scored:
            for val in s.faithfulness.values():
                if isinstance(val, (int, float)) and np.isfinite(val):
                    nums.append(round(float(val), 3))
        return " ".join(bits), nums

    # ------------------------------------------------------------------
    def _narrate(self, card: ExplanationCard, items) -> NR.Narration:
        if not self.cfx.narrate:
            return NR.Narration(technical="", layman="", backend="disabled")
        tech, nums = self._technical_text(card)
        return NR.narrate(evidence_items=items, technical_text=tech,
                          allowed_numerals=nums,
                          decision=card.verdict.decision,
                          p_calibrated=card.verdict.p_fake_calibrated,
                          lex=self.lex, reasoner=self.reasoner)

    # ------------------------------------------------------------------
    def analyse_many(self, items: Sequence[dict], strict_coverage: bool = True
                     ) -> Tuple[List[ExplanationCard], Dict[str, object]]:
        """
        Run over a list of {clip, mask, lmf, frames_u8, clip_id} dicts and check
        the coverage assertion at the end.

        `strict_coverage=True` RAISES when the system confidently answers more
        than 90% of clips. At this project's honest AUC that cannot be right, and
        a run that produces a convincing report from a leaky split is worse than
        a run that fails.
        """
        cards = [self.analyse(**it) for it in items]
        answered = [c for c in cards if c.verdict.decision != ABSTAIN]
        coverage = len(answered) / max(1, len(cards))
        msg = assert_coverage_sane(coverage, strict=strict_coverage)
        return cards, {
            "n": len(cards), "coverage": coverage,
            "n_explained": sum(1 for c in cards if c.best is not None),
            "quadrants": {q: sum(1 for c in cards if c.quadrant == q)
                          for q in ("confident-explainable",
                                    "confident-unexplainable",
                                    "unconfident-explainable",
                                    "unconfident-unexplainable",
                                    "low-confidence")},
            "template_fallback_rate": float(np.mean(
                [1.0 if (c.narration and c.narration.fell_back_to_template) else 0.0
                 for c in cards])) if cards else float("nan"),
            "mean_fk_grade": float(np.mean(
                [c.narration.fk_grade for c in cards if c.narration])) if cards else float("nan"),
            "coverage_warning": msg,
        }


__all__ = ["CFXConfig", "Verdict", "ScoredExplanation", "ExplanationCard",
           "CFXPipeline"]
