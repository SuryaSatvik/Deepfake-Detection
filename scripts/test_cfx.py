#!/usr/bin/env python
"""
test_cfx.py
===========

Weight-free test suite for the CFX layer (calibrate / selective / attribution /
faithfulness / arbitrate / narrate / cfx).

Same pattern as `scripts/test_qwen_path.py`: everything that does not need a
trained checkpoint is tested against a tiny stub model whose behaviour is known
in advance, so the tests can assert the *right answer* rather than merely "it
ran". The stub is deliberately crafted so that:

  * one branch (frequency) is decisive and one (lmf) is pure noise, so Shapley,
    the modality gate and the solidity factors have a ground truth to hit;
  * the decisive evidence lives in a specific frame and a specific image region,
    so deletion/insertion and the counterfactuals have a ground truth too;
  * the forward pass is exactly deterministic, so the Shapley efficiency residual
    must be ~0 and a non-zero residual is a real failure.

Run:  python scripts/test_cfx.py
Exit code is non-zero if any check fails, so it works as a pre-commit gate.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

PASS: List[str] = []
FAIL: List[str] = []


def check(name: str, cond: bool, detail: str = "") -> bool:
    (PASS if cond else FAIL).append(name if not detail else f"{name} -- {detail}")
    print(f"  {'ok  ' if cond else 'FAIL'} {name}" + (f"  [{detail}]" if detail else ""))
    return bool(cond)


def section(title: str) -> None:
    print(f"\n=== {title} ===")


# ==========================================================================
# the stub model
# ==========================================================================


class StubFSVNet(nn.Module):
    """
    Mimics `FSVNet`'s public surface with hand-built, deterministic behaviour.

    The fake-margin is a weighted sum of per-branch signals:
      frequency : mean of a 32x32 patch at (16,16) of frame 3  -> the decisive one
      spatial   : global mean brightness                        -> weak
      srm       : high-frequency energy                         -> weak
      tsc       : frame-to-frame difference energy              -> moderate
      lmf       : a constant                                    -> no information

    Ablating a branch zeroes its contribution, exactly as `FSVNet.forward` zeroes
    the embedding before fusion.

    The bias is tuned so that a clip WITH the planted patch reads fake and the
    same clip with frame 3 frozen reads real. That makes the counterfactuals
    and the deletion curves testable against a known answer instead of merely
    checking that they return a number.
    """

    W = {"spatial": 0.6, "frequency": 3.0, "srm": 0.4, "tsc": 1.2, "lmf": 0.0}

    def __init__(self, branches=("spatial", "frequency", "srm", "tsc", "lmf"),
                 bias: float = -2.0):
        super().__init__()
        self.use = {b: (b in branches) for b in
                    ("spatial", "frequency", "srm", "tsc", "lmf")}
        self.branch_names = [b for b in
                             ("spatial", "frequency", "srm", "tsc", "lmf")
                             if self.use[b]]
        self.branch_dim = 8
        self.bias = float(bias)
        # a real parameter so .eval()/.modules() behave normally
        self.dummy = nn.Linear(4, 2)
        self.drop = nn.Dropout(0.3)

    # --------------------------------------------------------------
    def _signals(self, clip: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, T, C, H, W = clip.shape
        y0, x0 = H // 4, W // 4
        s = H // 4
        patch = clip[:, min(3, T - 1), :, y0:y0 + s, x0:x0 + s]
        d = (clip[:, 1:] - clip[:, :-1]).abs().mean(dim=(1, 2, 3, 4))
        hf = (clip[..., 1:] - clip[..., :-1]).abs().mean(dim=(1, 2, 3, 4))
        return {
            "spatial": clip.mean(dim=(1, 2, 3, 4)),
            "frequency": patch.mean(dim=(1, 2, 3)),
            "srm": hf,
            "tsc": d,
            "lmf": torch.full((B,), 0.5, device=clip.device, dtype=clip.dtype),
        }

    # --------------------------------------------------------------
    def forward(self, clip, frame_mask=None, return_evidence: bool = False,
                ablate_branches: Optional[List[str]] = None, lmf=None):
        ablate = set(ablate_branches or [])
        sig = self._signals(clip)
        B, T = clip.shape[:2]
        margin = torch.full((B,), self.bias, device=clip.device, dtype=clip.dtype)
        for b in self.branch_names:
            if b not in ablate:
                margin = margin + self.W[b] * sig[b]

        logits = torch.stack([-margin / 2.0, margin / 2.0], dim=1)

        n = len(self.branch_names)
        bw = torch.full((B, n), 1.0 / n, device=clip.device, dtype=clip.dtype)
        fw = torch.softmax(clip.mean(dim=(2, 3, 4)), dim=1)      # (B,T)

        info: Dict[str, object] = {
            "branch_weights": bw,
            "branch_names": list(self.branch_names),
            "frame_weights": {"spatial": fw},
        }
        if return_evidence:
            desc = {
                "temporal_whiteness": sig["tsc"].detach(),
                "fine_ring_acf1": (1.0 - sig["tsc"]).detach(),
                "flicker_index": sig["srm"].detach(),
            }
            info["evidence"] = {"tsc_descriptors": desc,
                                "srm_energy": clip.mean(dim=(2, 3, 4)).detach()}
        return logits, info

    @torch.no_grad()
    def predict_proba(self, clip, frame_mask=None, lmf=None):
        logits, _ = self.forward(clip, frame_mask, lmf=lmf)
        return torch.softmax(logits, dim=1)


def make_clip(T: int = 8, H: int = 64, W: int = 64, hot: bool = True,
              seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    clip = torch.rand(1, T, 3, H, W, generator=g) * 0.4
    if hot:
        # plant the decisive evidence where the stub looks for it
        y0, x0, s = H // 4, W // 4, H // 4
        clip[:, min(3, T - 1), :, y0:y0 + s, x0:x0 + s] += 0.9
    return clip


def frames_u8(clip: torch.Tensor) -> np.ndarray:
    x = clip[0].permute(0, 2, 3, 1).numpy()
    x = (np.clip(x, 0, 1) * 255).astype(np.uint8)
    return x


# ==========================================================================
# tests
# ==========================================================================


def test_calibrate() -> None:
    section("calibrate")
    from fsvnet.calibrate import (brier_decomposition, calibration_report, ece,
                                  pick_calibrator, reprior, sigmoid,
                                  TemperatureScaler, PlattScaler)

    rng = np.random.default_rng(0)
    n = 4000
    z_true = rng.normal(0, 2.0, n)
    y = (rng.random(n) < sigmoid(z_true)).astype(float)

    # a deliberately over-confident model: logits scaled up by 3
    z_bad = z_true * 3.0
    p_bad = sigmoid(z_bad)
    cal = TemperatureScaler().fit(z_bad, y)
    p_fix = cal.transform(z_bad)

    check("temperature recovers the scale (T ~ 3)", abs(cal.T - 3.0) < 0.35,
          f"T={cal.T:.3f}")
    check("temperature reduces ECE", ece(y, p_fix) < ece(y, p_bad),
          f"{ece(y, p_bad):.4f} -> {ece(y, p_fix):.4f}")

    # temperature scaling must not change the ranking, hence not the AUC
    from sklearn.metrics import roc_auc_score
    check("temperature leaves AUC untouched",
          abs(roc_auc_score(y, p_bad) - roc_auc_score(y, p_fix)) < 1e-9)

    # equal-width vs equal-mass on a skewed, piled-up distribution
    p_pile = np.clip(rng.normal(0.9, 0.02, 2000), 0, 1)
    y_pile = (rng.random(2000) < 0.6).astype(float)
    check("equal-mass ECE >= equal-width ECE when predictions pile up",
          ece(y_pile, p_pile, strategy="quantile")
          >= ece(y_pile, p_pile, strategy="uniform") - 1e-12,
          f"mass={ece(y_pile, p_pile, 15, 'quantile'):.4f} "
          f"width={ece(y_pile, p_pile, 15, 'uniform'):.4f}")

    d = brier_decomposition(y, sigmoid(z_true))
    check("Brier decomposition closes", abs(d["decomposition_residual"]) < 5e-3,
          f"residual={d['decomposition_residual']:.2e}")

    # a model that predicts the base rate for everyone has ~zero resolution
    d0 = brier_decomposition(y, np.full(n, y.mean()))
    check("zero-information model has ~zero resolution", d0["resolution"] < 1e-6,
          f"resolution={d0['resolution']:.2e}")

    pr = reprior(np.array([0.9]), train_prior=0.864, deploy_prior=0.5)
    check("reprior moves a Celeb-DF probability down at a 50/50 prior",
          pr[0] < 0.9, f"0.900 -> {pr[0]:.3f}")
    check("reprior is identity when the priors match",
          abs(reprior(np.array([0.73]), 0.6, 0.6)[0] - 0.73) < 1e-9)

    best, scores = pick_calibrator(z_bad, y, verbose=False)
    check("CV picks a real calibrator over identity on a miscalibrated model",
          best.name in ("temperature", "platt"), f"chose {best.name}")

    rep = calibration_report(y, p_bad)
    check("report flags nothing false on a sane input", rep.n == n)
    rep0 = calibration_report(y, np.full(n, y.mean()))
    check("report warns when resolution is ~0",
          any("resolution" in s for s in rep0.notes))


def test_selective() -> None:
    section("selective")
    from fsvnet.selective import (ABSTAIN, assert_coverage_sane, aurc,
                                  check_exchangeability, evaluate_policy,
                                  fit_abstention, majority_class_baseline,
                                  risk_coverage_curve, selective_at_coverage)

    rng = np.random.default_rng(1)
    n = 2000
    y = (rng.random(n) < 0.864).astype(float)             # the real class balance
    # a weak but non-trivial detector: AUC ~0.65
    p = np.clip(0.5 + 0.12 * (y - 0.5) * 2 + rng.normal(0, 0.12, n), 0.01, 0.99)

    rc = risk_coverage_curve(y, p)
    check("risk-coverage is monotone in coverage at the ends",
          rc["risk"][-1] >= rc["risk"][int(0.1 * n)] - 1e-9,
          f"risk@10%={rc['risk'][int(0.1 * n)]:.3f} risk@100%={rc['risk'][-1]:.3f}")
    check("a perfect ranking gives AURC ~0",
          aurc(y, np.where(y == 1, 0.99, 0.01)) < 1e-6)

    rows = selective_at_coverage(y, p)
    check("majority baseline is reported at every coverage",
          all("majority_class_baseline" in r for r in rows))
    check("majority baseline equals the class prior",
          abs(majority_class_baseline(y) - max(y.mean(), 1 - y.mean())) < 1e-12)

    half = n // 2
    pol = fit_abstention(p[:half], y[:half], alpha=0.10)
    ev = evaluate_policy(pol, p[half:], y[half:])
    check("fitted policy abstains on something", ev["coverage"] < 1.0,
          f"coverage={ev['coverage']:.1%}")
    check("selective risk is at or near the target on held-out data",
          not np.isfinite(ev["selective_risk"]) or ev["selective_risk"] <= 0.10 + 0.05,
          f"risk={ev['selective_risk']:.3f} target=0.100")
    check("band is a real interval", pol.tau_real <= pol.tau_fake,
          f"[{pol.tau_real:.3f}, {pol.tau_fake:.3f}]")
    check("a probability inside the band abstains",
          pol.decide((pol.tau_real + pol.tau_fake) / 2) == ABSTAIN)

    # class-conditional matters at this prior
    pol_g = fit_abstention(p[:half], y[:half], alpha=0.10, class_conditional=False)
    check("Mondrian and global policies differ at an 86/14 prior",
          abs(pol.tau_real - pol_g.tau_real) > 1e-6
          or abs(pol.tau_fake - pol_g.tau_fake) > 1e-6)

    probs = check_exchangeability({"dataset": "celebdf", "img_size": 112},
                                  {"dataset": "ffpp", "img_size": 224})
    check("exchangeability violation is detected", len(probs) == 2, str(probs))
    check("no false alarm when provenance matches",
          check_exchangeability({"dataset": "celebdf"}, {"dataset": "celebdf"}) == [])

    raised = False
    try:
        assert_coverage_sane(0.97, auc=0.57)
    except AssertionError:
        raised = True
    check("coverage assertion fires at 97% coverage / 0.57 AUC", raised)
    check("coverage assertion silent at 40%",
          assert_coverage_sane(0.40, auc=0.57) is None)


def test_attribution() -> None:
    section("attribution")
    from fsvnet.attribution import (branch_shapley_exact, counterfactual_spectral,
                                    counterfactual_temporal, gaussian_blur_clip,
                                    integrated_gradients, make_baseline,
                                    superpixels, kernel_shap_superpixel)

    m = StubFSVNet().eval()
    clip = make_clip()

    for kind in ("blur", "mean", "zero"):
        b = make_baseline(clip, kind)
        check(f"baseline '{kind}' preserves shape", b.shape == clip.shape)
    blur = gaussian_blur_clip(clip, sigma=4.0)
    check("blur reduces high-frequency energy",
          float((blur[..., 1:] - blur[..., :-1]).abs().mean())
          < float((clip[..., 1:] - clip[..., :-1]).abs().mean()))

    sh = branch_shapley_exact(m, clip)
    check("Shapley enumerates 2^5 coalitions", sh.n_forwards == 32,
          f"{sh.n_forwards}")
    check("Shapley efficiency holds exactly on a deterministic model",
          abs(sh.efficiency_residual) < 1e-4,
          f"residual={sh.efficiency_residual:.2e}")
    check("Shapley finds the decisive branch", sh.dominant == "frequency",
          f"dominant={sh.dominant}, values="
          + ", ".join(f"{k}={v:+.3f}" for k, v in sh.values.items()))
    check("Shapley gives the uninformative branch ~zero credit",
          abs(sh.values["lmf"]) < 1e-5, f"lmf={sh.values['lmf']:.2e}")
    check("no spurious interactions in an additive model",
          max(abs(v) for v in sh.interactions.values()) < 1e-4,
          f"max|I|={max(abs(v) for v in sh.interactions.values()):.2e}")

    ig = integrated_gradients(m, clip, steps=32, baseline="blur")
    check("IG returns a (T,H,W) map", ig["attribution"].shape == (8, 64, 64),
          str(ig["attribution"].shape))
    check("IG satisfies completeness to within 5%",
          ig["completeness_error"] < 0.05, f"err={ig['completeness_error']:.4f}")
    hot_frame = int(np.argmax(ig["attribution"].sum(axis=(1, 2))))
    check("IG localises the decisive frame", hot_frame == 3, f"frame {hot_frame}")
    a3 = ig["attribution"][3]
    q = a3[16:32, 16:32].mean() / (a3.mean() + 1e-12)
    check("IG localises the decisive region", q > 2.0, f"region/mean = {q:.2f}x")

    cf = counterfactual_temporal(m, clip)
    check("temporal counterfactual flips", cf.flipped, cf.statement)
    check("it needs only the decisive frame", cf.cost <= 2.0,
          f"{cf.cost} frames: {cf.detail['frames']}")

    cs = counterfactual_spectral(m, clip)
    check("spectral counterfactual returns a statement", bool(cs.statement),
          cs.statement)

    fu = frames_u8(clip)
    lab = superpixels(fu[3], n_segments=24)
    check("superpixels segment the frame", len(np.unique(lab)) >= 4,
          f"{len(np.unique(lab))} segments")
    flat = np.zeros((32, 32, 3), dtype=np.uint8)
    check("a featureless frame still yields usable units (grid fallback)",
          len(np.unique(superpixels(flat, n_segments=16))) >= 4,
          f"{len(np.unique(superpixels(flat, n_segments=16)))} segments")
    ks = kernel_shap_superpixel(m, clip, fu, [3], n_segments=24, n_samples=200)
    check("KernelSHAP fits the surrogate well on an additive model",
          ks["r2"][3] > 0.6, f"R2={ks['r2'][3]:.3f}")
    vals = ks["per_frame"][3]["values"]
    labs = ks["per_frame"][3]["labels"]
    top_seg = ks["per_frame"][3]["segment_ids"][int(np.argmax(np.abs(vals)))]
    ys, xs = np.where(labs == top_seg)
    inside = (16 <= ys.mean() <= 32) and (16 <= xs.mean() <= 32)
    check("KernelSHAP's top region lies in the planted patch", inside,
          f"centroid=({ys.mean():.0f},{xs.mean():.0f}) expected ~(24,24)")


def test_faithfulness() -> None:
    section("faithfulness")
    from fsvnet.faithfulness import (borda_aggregate, blur_everything_control,
                                     evaluate_candidate,
                                     faithfulness_vs_correctness,
                                     patchify_map, rank_agreement,
                                     random_ranking, ranking_from_frames,
                                     ranking_from_spatial, score_ranking,
                                     stability)
    from fsvnet.attribution import integrated_gradients

    m = StubFSVNet().eval()
    clip = make_clip()

    pm = patchify_map(np.ones((8, 64, 64)), patch=16)
    check("patchify gives the right grid", pm.shape == (8, 4, 4), str(pm.shape))

    ig = integrated_gradients(m, clip, steps=16)
    good = ranking_from_spatial(ig["attribution"], patch=16, method="ig")
    bad = random_ranking(good, seed=7)
    bad.method = "random_ref"

    s_good = score_ranking(m, clip, good, fill="blur")
    s_bad = score_ranking(m, clip, bad, fill="blur")
    check("a real attribution deletes faster than a random one",
          s_good.deletion_auc < s_bad.deletion_auc,
          f"{s_good.deletion_auc:.3f} vs {s_bad.deletion_auc:.3f}")
    check("a real attribution is more comprehensive than random",
          s_good.comprehensiveness > s_bad.comprehensiveness,
          f"{s_good.comprehensiveness:.3f} vs {s_bad.comprehensiveness:.3f}")

    b = borda_aggregate([s_good, s_bad])
    check("Borda ranks the real attribution first",
          b[s_good.method] > b[s_bad.method], str({k: round(v, 3) for k, v in b.items()}))

    verd = evaluate_candidate(m, clip, good, n_random=3, fills=("blur",))
    check("evaluate_candidate says a real attribution beats random",
          verd.beats_random, f"margin={verd.margin_over_random:+.3f}")

    verd_r = evaluate_candidate(m, clip, random_ranking(good, seed=99),
                               n_random=3, fills=("blur",))
    check("and says a random attribution does NOT beat random",
          not verd_r.beats_random, f"margin={verd_r.margin_over_random:+.3f}")
    check("the failure is recorded in the notes, not swallowed",
          any("random" in n for n in verd_r.notes))

    ctrl = blur_everything_control(m, clip)
    check("blur-everything control bounds the available shift",
          ctrl["total_available_shift"] > 0,
          f"{ctrl['total_available_shift']:.3f} logits")

    ag = rank_agreement(good.order, good.order)
    check("a ranking agrees perfectly with itself", abs(ag["spearman"] - 1.0) < 1e-9)
    ag2 = rank_agreement(good.order, list(reversed(good.order)))
    check("a reversed ranking gives rho ~ -1", ag2["spearman"] < -0.99,
          f"rho={ag2['spearman']:.3f}")

    st = stability(lambda c: ranking_from_spatial(
        integrated_gradients(m, c, steps=8)["attribution"], 16, method="ig"),
        clip, n_repeats=2)
    check("IG is stable under sigma=0.01 noise", st["stable"],
          f"rho={st['spearman_mean']:.3f}")

    rng = np.random.default_rng(3)
    corr = rng.random(200) < 0.6
    faith = np.where(corr, rng.normal(0.7, 0.1, 200), rng.normal(0.4, 0.1, 200))
    r = faithfulness_vs_correctness(faith, corr)
    check("faithfulness-vs-correctness resolves on a planted effect",
          r["resolved"] and r["r"] > 0, f"r={r['r']:.3f} CI=[{r['ci_lo']:.3f},{r['ci_hi']:.3f}]")
    r0 = faithfulness_vs_correctness(rng.random(200), corr)
    check("and does not resolve when there is no effect", not r0["resolved"],
          f"r={r0['r']:.3f} CI=[{r0['ci_lo']:.3f},{r0['ci_hi']:.3f}]")


def test_arbitrate() -> None:
    section("arbitrate")
    from fsvnet.arbitrate import (agreement_across_models, build_features,
                                  build_reliability_matrix, compare_to_single_model,
                                  ConfidenceStack, corroboration, extremeness,
                                  FEATURE_ORDER, route, solidity)

    check("extremeness: 50th percentile is not evidence",
          extremeness(50.0) == 0.0)
    check("extremeness: the deadband makes a 58th percentile exactly zero",
          extremeness(58.0) == 0.0, "not merely small -- zero")
    check("extremeness: 3rd percentile is full evidence",
          extremeness(3.0) == 1.0)
    check("extremeness: missing percentile contributes nothing",
          extremeness(None) == 0.0)

    check("corroboration rewards agreement",
          corroboration({"a": 0.4, "b": 0.42}) > 0.8)
    check("corroboration punishes disagreement",
          corroboration({"a": 0.4, "b": -0.38}) < 0.2)
    check("a single witness gets half credit",
          corroboration({"a": 0.4}) == 0.5)

    m = build_reliability_matrix({
        ("B0", "spatial"):   dict(auc=0.60, faithfulness=0.5,
                                               beats_random=True, stability=0.8),
        ("B0", "frequency"): dict(auc=0.66, faithfulness=0.7,
                                               beats_random=True, stability=0.9),
        ("B0", "srm"):       dict(auc=0.54, faithfulness=0.3,
                                               beats_random=True, stability=0.6),
        ("C1", "tsc"):                dict(auc=0.52, faithfulness=0.1,
                                               beats_random=False, stability=0.4),
        ("C2", "lmf"):                dict(auc=0.51, faithfulness=0.05,
                                               beats_random=False, stability=0.3),
        ("C3", "frequency"):  dict(auc=0.58, faithfulness=0.4,
                                               beats_random=True, stability=0.7),
    })
    check("a channel that fails its random control scores zero",
          m.score("C1", "tsc") == 0.0)
    best, s = m.best_model_for("frequency")
    check("the authoritative source for frequency is B0, not the full model",
          best == "B0", f"{best} ({s:.3f})")
    check("a chance-level channel contributes nothing even if it beats random",
          m.score("B0", "srm") < m.score("B0", "frequency"))

    sols = [
        solidity(m, "frequency", signed_evidence=1.2, percentile=3.0,
                 counterfactual_survived=True, stability_score=0.9,
                 cross_model_signals={"B0": 0.3, "C3": 0.28}),
        solidity(m, "spatial", signed_evidence=0.2, percentile=52.0,
                 counterfactual_survived=True, stability_score=0.8,
                 cross_model_signals={"B0": 0.1}),
        solidity(m, "tsc", signed_evidence=0.9, percentile=2.0,
                 counterfactual_survived=True, stability_score=0.9,
                 cross_model_signals={"C1": 0.3}),
    ]
    by = {s.channel: s for s in sols}
    check("an unremarkable measurement has zero solidity even when reliable",
          by["spatial"].solidity == 0.0,
          f"spatial solidity={by['spatial'].solidity:.3f} (52nd percentile)")
    check("an extreme measurement on an unreliable channel still has zero solidity",
          by["tsc"].solidity == 0.0,
          "TSC failed its random control, so a 2nd-percentile value buys nothing")
    check("the solid channel survives", by["frequency"].solidity > 0.3,
          f"{by['frequency'].solidity:.3f}")
    check("every factor is retained for audit",
          all(k in by["frequency"].to_dict() for k in
              ("reliability", "extremeness", "counterfactual", "stability",
               "corroboration")))

    r = route(m, sols)
    check("routing prefers a model carrying the solid channel",
          r.answering_model in ("B0", "C1", "C2",
                                "C3"), r.answering_model)
    check("the modality gate admits only solid channels",
          r.admissible_modalities == ["frequency"], str(r.admissible_modalities))
    r0 = route(m, [solidity(m, "spatial", 0.0, percentile=50.0)])
    check("with no solid evidence, routing falls back and says so",
          r0.total_solidity == 0.0 and "abstain" in r0.reason)

    feats = build_features(sols, p_best_single=0.8, agreement=0.9, uncertainty=0.05)
    check("feature vector matches the documented order",
          feats.shape == (len(FEATURE_ORDER),), f"{feats.shape} vs {len(FEATURE_ORDER)}")

    rng = np.random.default_rng(5)
    n = 600
    y = (rng.random(n) < 0.5).astype(float)
    X = rng.normal(0, 1, (n, len(FEATURE_ORDER)))
    X[:, 1] += 1.5 * y                                    # solid_frequency carries signal
    stack = ConfidenceStack(C=1.0).fit(X, y)
    p = stack.predict_proba(X)
    from sklearn.metrics import roc_auc_score
    check("the stack learns a planted signal", roc_auc_score(y, p) > 0.75,
          f"AUC={roc_auc_score(y, p):.3f}")
    check("the stack's coefficients are printable as arbitration weights",
          len(stack.coefficients()) == len(FEATURE_ORDER))
    check("stack output is a probability", bool(np.all((p >= 0) & (p <= 1))))

    p_single = np.clip(0.5 + 0.05 * (y - 0.5), 0, 1) + rng.normal(0, 0.2, n)
    cmp = compare_to_single_model(y, p, np.clip(p_single, 0, 1), n=400)
    check("comparison against the single-model baseline returns a verdict",
          "verdict" in cmp and "delta_auc" in cmp, cmp["verdict"])
    cmp_same = compare_to_single_model(y, p, p, n=200)
    check("comparing a model against itself is not resolved",
          not cmp_same["resolved"], f"p={cmp_same['p_value']:.3f}")
    check("a non-win carries an explicit 'ship the baseline' recommendation",
          "recommendation" in cmp_same)

    check("agreement is high when models concur",
          agreement_across_models({"a": 0.8, "b": 0.82}) > 0.7)
    check("agreement is low when they split",
          agreement_across_models({"a": 0.8, "b": 0.2}) < 0.5)


def test_narrate() -> None:
    section("narrate")
    from fsvnet.narrate import (Claim, LAYMAN, TECHNICAL, Lexicon, abstain_text,
                                back_translation_check, claim_link_filter,
                                confidence_phrase, flesch_kincaid_grade,
                                forbidden_check, layman_template, narrate,
                                natural_frequency, numeral_fidelity,
                                parse_claim_json)

    lex = Lexicon.load()
    check("lexicon loads", len(lex.terms) > 10, f"{len(lex.terms)} terms")
    check("every confidence band has both registers",
          all("layman" in e and "technical" in e for e in lex.confidence))

    check("natural frequency reads correctly",
          natural_frequency(0.71) == "about 7 times in 10",
          natural_frequency(0.71))
    check("the same probability always yields the same words",
          confidence_phrase(0.8, "fake", lex) == confidence_phrase(0.8, "fake", lex))
    check("a layman confidence phrase keeps a frequency framing",
          "times in 10" in confidence_phrase(0.8, "fake", lex, LAYMAN))
    check("abstain short-circuits the confidence phrase",
          "can't tell" in confidence_phrase(0.52, "abstain", lex))

    nf = numeral_fidelity("p is 0.573 and the share is 0.410", [0.573, 0.41])
    check("numeral fidelity passes on exact numbers", nf["ok"], str(nf))
    nf2 = numeral_fidelity("p is about 0.6", [0.573])
    check("numeral fidelity catches a rounded measurement", not nf2["ok"],
          str(nf2["unsupported"]))

    check("forbidden wording is detected",
          forbidden_check("this is clearly a deepfake", lex.forbidden["layman"]))
    check("acceptable wording passes",
          not forbidden_check("the fine grain looks unusual", lex.forbidden["layman"]))

    raw = ('```json\n[{"sentence":"The fine grain flickers oddly.",'
           '"supports":["desc_temporal_whiteness"]},'
           '{"sentence":"The eyes never blink.","supports":[]}]\n```')
    claims, err = parse_claim_json(raw)
    check("claim JSON parses through a code fence", err is None and len(claims) == 2,
          str(err))
    kept, dropped = claim_link_filter(claims, ["desc_temporal_whiteness"], lex)
    check("a supported sentence is kept", len(kept) == 1)
    check("an unsupported sentence is dropped, not rewritten",
          len(dropped) == 1 and kept[0].sentence != claims[1].sentence,
          dropped[0]["reason"])

    ungrounded = [Claim("The lighting on the cheek is inconsistent.",
                        ["desc_temporal_whiteness"])]
    k2, d2 = claim_link_filter(ungrounded, ["desc_temporal_whiteness"], lex)
    check("a cited-but-unmeasurable claim is still dropped",
          len(k2) == 0 and "unmeasurable" in d2[0]["reason"], str(d2))

    bt = back_translation_check(
        "The fine grain of the face changes from frame to frame in a way that real "
        "camera footage usually does not.",
        ["desc_temporal_whiteness"], {"desc_temporal_whiteness": "temporal_whiteness"},
        lex)
    check("back-translation recovers the cited evidence", bt["ok"],
          f"recovered={bt['recovered']:.0%}")
    bt2 = back_translation_check("Something looked odd about the video.",
                                 ["desc_temporal_whiteness"],
                                 {"desc_temporal_whiteness": "temporal_whiteness"}, lex)
    check("back-translation fails on drifted text", not bt2["ok"])

    at = abstain_text(lex, LAYMAN)
    check("abstain text leads with 'I can't tell'", at.startswith("I can't tell"))
    check("abstain text does not read as a soft yes",
          not forbidden_check(at, lex.forbidden["layman"]))
    check("abstain text is plain enough to read",
          flesch_kincaid_grade(at) <= 12.0, f"FK={flesch_kincaid_grade(at):.1f}")

    items = [{"id": "desc_temporal_whiteness", "term": "temporal_whiteness",
              "layman": lex.phrase("temporal_whiteness", LAYMAN),
              "how_unusual": "3rd percentile of real videos"}]
    n = narrate(items, "Routed to B0. Share 0.410.",
                allowed_numerals=[0.41, 0.73], decision="fake",
                p_calibrated=0.73, lex=lex, reasoner=None)
    check("narration produces both registers", bool(n.technical) and bool(n.layman))
    check("no model available -> template fallback, recorded",
          n.fell_back_to_template and n.backend == "template")
    check("the toggle is a pure switch between cached strings",
          n.text(TECHNICAL) == n.technical and n.text(LAYMAN) == n.layman)
    check("the layman register keeps the confidence words",
          "times in 10" in n.layman)
    check("the layman register avoids overconfident words",
          not forbidden_check(n.layman, lex.forbidden["layman"]), n.layman)

    n_ab = narrate(items, "tech", [0.5], decision="abstain", p_calibrated=0.51,
                   lex=lex)
    check("an abstained clip is templated in both registers",
          n_ab.fell_back_to_template and "can't tell" in n_ab.layman)

    # a hostile generator: cites nothing, overclaims, invents a number
    class BadBackend:
        mode = "bad"
        def generate(self, system, user, images):
            return '[{"sentence":"This is clearly a deepfake (99%).","supports":[]}]'

    class BadReasoner:
        def load(self): return BadBackend()

    n_bad = narrate(items, "tech", [0.41], decision="fake", p_calibrated=0.73,
                    lex=lex, reasoner=BadReasoner())
    check("a hostile generator cannot reach the user",
          n_bad.fell_back_to_template and "clearly" not in n_bad.layman.lower(),
          f"backend={n_bad.backend}")
    check("and the rejection is recorded", len(n_bad.notes) > 0, str(n_bad.notes[:1]))

    # a well-behaved generator
    class GoodBackend:
        mode = "qwen_4bit"
        def generate(self, system, user, images):
            p = lex.phrase("temporal_whiteness", LAYMAN)
            return ('[{"sentence":"' + p.capitalize() +
                    '.","supports":["desc_temporal_whiteness"]}]')

    class GoodReasoner:
        def load(self): return GoodBackend()

    n_ok = narrate(items, "tech", [0.41], decision="fake", p_calibrated=0.73,
                   lex=lex, reasoner=GoodReasoner())
    check("a well-behaved generator is used", not n_ok.fell_back_to_template,
          f"backend={n_ok.backend}")
    check("groundedness is recorded", n_ok.groundedness > 0.0,
          f"{n_ok.groundedness:.2f}")

    check("template layman text mentions the measurement in plain words",
          "fine grain" in layman_template(items, lex))


def test_cfx_pipeline() -> None:
    section("cfx pipeline")
    from fsvnet.arbitrate import (build_reliability_matrix, ConfidenceStack,
                                  FEATURE_ORDER)
    from fsvnet.cfx import CFXConfig, CFXPipeline
    from fsvnet.narrate import LAYMAN, TECHNICAL
    from fsvnet.selective import fit_abstention

    models = {
        "B0": StubFSVNet(("spatial", "frequency", "srm")).eval(),
        "C1": StubFSVNet(("spatial", "frequency", "srm", "tsc")).eval(),
        "C2": StubFSVNet(("spatial", "frequency", "srm", "lmf")).eval(),
        "C3": StubFSVNet().eval(),
    }
    matrix = build_reliability_matrix({
        ("B0", "spatial"):   dict(auc=0.60, faithfulness=0.5,
                                               beats_random=True, stability=0.8),
        ("B0", "frequency"): dict(auc=0.68, faithfulness=0.7,
                                               beats_random=True, stability=0.9),
        ("B0", "srm"):       dict(auc=0.55, faithfulness=0.3,
                                               beats_random=True, stability=0.7),
        ("C1", "tsc"):                dict(auc=0.52, faithfulness=0.1,
                                               beats_random=False, stability=0.4),
        ("C2", "lmf"):                dict(auc=0.50, faithfulness=0.0,
                                               beats_random=False, stability=0.2),
    })

    # An abstention policy fitted on plausible calibration data: confident
    # extremes are reliable, the middle is not, which is the shape a weak
    # detector actually has.
    rng = np.random.default_rng(11)
    n = 800
    y = (rng.random(n) < 0.5).astype(float)
    p_cal = np.clip(0.5 + 0.30 * (y - 0.5) * 2 + rng.normal(0, 0.12, n), 0.01, 0.99)
    policy = fit_abstention(p_cal, y, alpha=0.10)

    cfg = CFXConfig(kshap_samples=120, kshap_segments=16, ig_steps=8,
                    n_random_controls=2, do_tta=False,
                    faith_fills=("blur",))

    # No stack here on purpose: with `stack=None` the routed model's own
    # probability is used, which is what exercises the confident/explainable
    # path. The stack is tested on its own in test_arbitrate, where its inputs
    # can be controlled; feeding it stub features would test the fixture, not
    # the pipeline.
    pipe = CFXPipeline(models=models, matrix=matrix, policy=policy, cfx=cfg)

    clip = make_clip()
    card = pipe.analyse(clip, frames_u8=frames_u8(clip), clip_id="stub_hot")
    check("a clip with planted evidence clears the confidence gate",
          card.verdict.confident and card.verdict.decision == "fake",
          f"p={card.verdict.p_fake_calibrated:.3f} "
          f"decision={card.verdict.decision} confident={card.verdict.confident}")
    check("clearing the gate runs the expensive explainers",
          card.quadrant == "confident-explainable", card.quadrant)
    check("the winning explanation beat its random control",
          card.best is not None and card.best.beats_random,
          card.best.method if card.best else "none")
    check("analyse returns a card", card.clip_id == "stub_hot")
    check("a quadrant is always assigned", bool(card.quadrant), card.quadrant)
    check("the card records how much compute it spent", card.n_forwards > 30,
          f"{card.n_forwards} forwards")
    check("Shapley is on the card", "dominant" in card.shapley,
          str(card.shapley.get("dominant")))

    user = card.for_user(LAYMAN)
    check("the user surface hides the model name",
          "answering_model" not in user and
          not any("B0" in str(v) for v in user.values()),
          str(list(user.keys())))
    check("the user surface hides faithfulness numbers",
          "faithfulness" not in user)
    check("but always shows the confidence number AND its band",
          "confidence_number" in user and "confidence_band" in user,
          f"p={user['confidence_number']} band={user['confidence_band']}")
    check("the toggle switches register without recomputing",
          card.for_user(TECHNICAL)["text"] != card.for_user(LAYMAN)["text"]
          or card.verdict.decision == "abstain")

    audit = card.for_audit()
    check("the audit surface exposes the routing trace",
          "routing" in audit and "per_channel" in audit["routing"])
    check("the audit surface lists every candidate with its verdict",
          isinstance(audit["candidates"], list) and len(audit["candidates"]) >= 3,
          f"{len(audit['candidates'])} candidates")
    check("withheld candidates say why",
          all(c["admissible"] or c["notes"] or not np.isfinite(
              c["margin_over_random"]) for c in audit["candidates"]))

    withheld = [c for c in card.all_scored if not c.admissible]
    check("every withheld candidate carries a reason",
          all(c.notes or not np.isfinite(c.margin_over_random)
              for c in withheld),
          f"{len(withheld)} withheld of {len(card.all_scored)}")

    # a clip with no planted evidence should be less confident
    cold = make_clip(hot=False, seed=4)
    card_cold = pipe.analyse(cold, frames_u8=frames_u8(cold), clip_id="stub_cold")
    check("a clip with no planted evidence scores lower",
          card_cold.verdict.p_fake_calibrated < card.verdict.p_fake_calibrated,
          f"cold={card_cold.verdict.p_fake_calibrated:.3f} "
          f"hot={card.verdict.p_fake_calibrated:.3f}")

    # the abstention path: a probability inside the band must short-circuit
    # A uniform clip of 0.5556 makes the stub's margin exactly zero
    # (-2.0 + 0.6*v + 3.0*v = 0), so p(fake) lands on 0.5 and the clip must be
    # abstained on. Solving for the fixture rather than guessing a value keeps
    # this test meaningful if the stub's weights ever change.
    mid = torch.full((1, 8, 3, 64, 64), 2.0 / 3.6)
    band_lo, band_hi = policy.band()
    card_mid = pipe.analyse(mid, frames_u8=frames_u8(mid), clip_id="stub_mid")
    if band_lo <= card_mid.verdict.p_fake_calibrated <= band_hi:
        check("a probability inside the band abstains and explains nothing",
              card_mid.verdict.decision == "abstain" and card_mid.best is None,
              f"p={card_mid.verdict.p_fake_calibrated:.3f} "
              f"band=[{band_lo:.3f},{band_hi:.3f}]")
        check("the abstain card says why nothing is explained",
              bool(card_mid.withheld_reason), card_mid.withheld_reason or "")
        check("both registers are templated when abstaining",
              card_mid.narration.fell_back_to_template
              and "can't tell" in card_mid.narration.layman)
        check("an abstain card spends far less compute than an explained one",
              card_mid.n_forwards < card.n_forwards,
              f"{card_mid.n_forwards} vs {card.n_forwards} forwards")
    else:
        check("abstention path reachable in this fixture", False,
              f"p={card_mid.verdict.p_fake_calibrated:.3f} outside "
              f"[{band_lo:.3f},{band_hi:.3f}] -- fixture needs adjusting")

    # summary over a batch
    cards, summary = pipe.analyse_many(
        [{"clip": clip, "frames_u8": frames_u8(clip), "clip_id": "a"},
         {"clip": cold, "frames_u8": frames_u8(cold), "clip_id": "b"},
         {"clip": mid, "frames_u8": frames_u8(mid), "clip_id": "c"}],
        strict_coverage=False)
    check("analyse_many reports coverage and quadrants",
          "coverage" in summary and "quadrants" in summary,
          f"coverage={summary['coverage']:.0%} {summary['quadrants']}")
    check("template fallback rate is tracked",
          np.isfinite(summary["template_fallback_rate"]),
          f"{summary['template_fallback_rate']:.0%}")
    check("mean reading level is tracked", np.isfinite(summary["mean_fk_grade"]),
          f"FK={summary['mean_fk_grade']:.1f}")

    # serialisation
    out = Path("/tmp/cfx_card.json")
    card.save(out)
    check("the card serialises to JSON", out.exists() and out.stat().st_size > 200,
          f"{out.stat().st_size} bytes")

    # a fitted stack must be used when present, and its absence recorded
    nrow = 400
    ys = (rng.random(nrow) < 0.5).astype(float)
    Xs = rng.normal(0, 1, (nrow, len(FEATURE_ORDER)))
    Xs[:, 1] += 1.2 * ys
    stack = ConfidenceStack().fit(Xs, ys)
    piped = CFXPipeline(models=models, matrix=matrix, stack=stack,
                        policy=policy, cfx=cfg)
    card_s = piped.analyse(clip, frames_u8=frames_u8(clip), clip_id="stacked")
    check("with a stack fitted, the card no longer warns about a missing one",
          not any("confidence stack" in n for n in card_s.notes))
    check("the stack changes the reported probability",
          abs(card_s.verdict.p_fake_calibrated
              - card.verdict.p_fake_calibrated) > 1e-6,
          f"stack={card_s.verdict.p_fake_calibrated:.3f} "
          f"routed={card.verdict.p_fake_calibrated:.3f}")
    check("the raw routed probability is kept beside the calibrated one",
          abs(card_s.verdict.p_fake_raw - card.verdict.p_fake_raw) < 1e-9)

    # no-stack / no-policy degradation must be recorded, not silent
    bare = CFXPipeline(models=models, matrix=matrix, cfx=cfg)
    card_b = bare.analyse(clip, frames_u8=frames_u8(clip), clip_id="bare")
    check("running without a fitted stack is recorded on the card",
          any("confidence stack" in n for n in card_b.notes))
    check("running without an abstention policy is recorded on the card",
          any("abstention policy" in n for n in card_b.notes))


# ==========================================================================


def main() -> int:
    print("CFX test suite -- stub model, no checkpoints required")
    for fn in (test_calibrate, test_selective, test_attribution,
               test_faithfulness, test_arbitrate, test_narrate,
               test_cfx_pipeline):
        try:
            fn()
        except Exception:
            FAIL.append(f"{fn.__name__} raised")
            print(f"  FAIL {fn.__name__} raised:")
            traceback.print_exc()

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("\nfailures:")
        for f in FAIL:
            print(f"  - {f}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
