#!/usr/bin/env python
"""
cfx_fit.py
==========

Fit everything CFX needs, on the identity-disjoint VAL split, and write it to
disk. Nothing here touches test.

    python scripts/cfx_fit.py                       # score, then fit everything
    python scripts/cfx_fit.py --from-scores         # reuse results/scores_*.npz
    python scripts/cfx_fit.py --alpha 0.05
    python scripts/cfx_fit.py --only B0 C1

Produces, in `results/cfx/`:

    scores_<id>.npz            y/p for val and test, per condition
    calibration_<id>.json      the chosen calibrator and its report
    abstention_policy.json     the fitted band  (fit on val, applied to test)
    evidence_reliability.json  the (model x channel) reliability matrix
    confidence_stack.json      the arbitration weights
    cfx_calibration.csv        the report table
    cfx_selective.csv          risk-coverage at 20/40/60/80/100% coverage
    cfx_arbitration.csv        stack vs best-single-model, paired bootstrap

Why a separate script and not the notebook: the same reason
`scripts/train_conditions.py` exists. Scoring four checkpoints over 976 val and
976 test clips is a batch job, and a Jupyter cell that loses its return value
halfway through is a wasted hour.

On the reliability matrix
-------------------------
Per-channel AUC here is measured the only way it can be without retraining:
**ablation delta**. For channel c in model m, score the val split twice -- once
normally, once with c ablated -- and take the AUC of the *difference* in the
fake-margin as a measure of how much independent signal c carries. That is a
proxy, it is documented as one in the output JSON, and the faithfulness and
stability columns (which come from `cfx_eval.py`, on real clips) are the ones
that decide admissibility.

UNTESTED against real checkpoints -- this file has only been exercised against
the stub model in `scripts/test_cfx.py`. Run it once with `--limit 32` before
committing to a full pass.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fsvnet import Config, celebdf_frames as cf                     # noqa: E402
from fsvnet.arbitrate import (BASELINE_ID, CHANNELS, CONDITION_LABELS,   # noqa: E402
                              DEFAULT_SPECIALISTS, ConfidenceStack,
                              FEATURE_ORDER, build_features,
                              build_reliability_matrix, compare_to_single_model,
                              solidity)
from fsvnet.calibrate import (calibration_report, fake_margin_from_logits,  # noqa: E402
                              pick_calibrator)
from fsvnet.dataset_frames import lmf_pack, make_frame_loaders        # noqa: E402
from fsvnet.model import build_model                                  # noqa: E402
from fsvnet.selective import (evaluate_policy, fit_abstention,        # noqa: E402
                              majority_class_baseline, aurc, e_aurc,
                              selective_at_coverage)
from fsvnet.utils import log                                          # noqa: E402

NUM_FRAMES, IMAGE_SIZE, BATCH_SIZE, NUM_WORKERS, SEED = 16, 224, 2, 2, 42
MAN = ROOT / "data" / "manifests" / "celebdf_v2_identity.csv"
CACHE = ROOT / "data" / "processed" / f"lmf_cache_{NUM_FRAMES:02d}f.npz"
CKPT_DIR = ROOT / "models" / "checkpoints"
OUT = ROOT / "results" / "cfx"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# (use_tsc, use_lmf) per condition -- the same mapping train_conditions.py uses.
CONDITION_FLAGS = {"B0": (False, False), "C1": (True, False),
                   "C2": (False, True), "C3": (True, True)}


# ==========================================================================
# scoring
# ==========================================================================


def load_condition(cid: str):
    ck = CKPT_DIR / f"fsvnet_dfe_{cid}.pt"
    if not ck.exists():
        log.warning(f"[{cid}] no checkpoint at {ck} -- skipped")
        return None, None
    blob = torch.load(ck, map_location=DEVICE, weights_only=False)
    c = Config()
    c.data.num_frames, c.data.img_size, c.data.seed = NUM_FRAMES, IMAGE_SIZE, SEED
    c.train.batch_size, c.train.num_workers = BATCH_SIZE, NUM_WORKERS
    c.device = DEVICE
    c.model.use_tsc = blob.get("use_tsc", CONDITION_FLAGS[cid][0])
    c.model.use_lmf = blob.get("use_lmf", CONDITION_FLAGS[cid][1])
    m = build_model(c).to(DEVICE)
    m.load_state_dict(blob["state_dict"])
    m.eval()
    return m, c


@torch.no_grad()
def score_split(model, loader, ablate: Optional[List[str]] = None,
                limit: Optional[int] = None) -> Dict[str, np.ndarray]:
    """
    Returns y, p and the raw fake-margin z. `z` is what the calibrators are fit
    on -- calibrating a probability that has already been through a softmax
    throws away the scale the temperature is supposed to find.
    """
    ys, ps, zs = [], [], []
    n = 0
    for b in loader:
        clip = b["clip"].to(DEVICE)
        logits, _ = model(clip, b.get("mask", None).to(DEVICE)
                          if b.get("mask", None) is not None else None,
                          ablate_branches=ablate, lmf=lmf_pack(b))
        z = fake_margin_from_logits(logits.float().cpu().numpy())
        ps += torch.softmax(logits.float(), 1)[:, 1].cpu().tolist()
        zs += z.tolist()
        ys += b["label"].tolist()
        n += len(b["label"])
        if limit and n >= limit:
            break
    return {"y": np.asarray(ys, dtype=np.float64),
            "p": np.asarray(ps, dtype=np.float64),
            "z": np.asarray(zs, dtype=np.float64)}


# ==========================================================================
# stages
# ==========================================================================


def stage_scores(only: List[str], limit: Optional[int],
                 from_scores: bool) -> Dict[str, Dict[str, Dict[str, np.ndarray]]]:
    out: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
    if from_scores:
        for cid in only:
            f = OUT / f"scores_{cid}.npz"
            if not f.exists():
                log.warning(f"[{cid}] {f.name} missing -- skipped")
                continue
            d = np.load(f)
            out[cid] = {s: {k: d[f"{k}_{s}"] for k in ("y", "p", "z")}
                        for s in ("val", "test") if f"y_{s}" in d}
            shape = ", ".join(f"{s}: n={len(v['y'])}" for s, v in out[cid].items())
            log.info(f"[{cid}] loaded cached scores ({shape})")
        return out

    df = cf.load_manifest(MAN, ROOT)
    print(df.groupby(["split", "label_name"]).size().unstack(fill_value=0)
            .reindex(["train", "val", "test"]))

    for cid in only:
        m, c = load_condition(cid)
        if m is None:
            continue
        loaders = make_frame_loaders(df, c, CACHE)
        res: Dict[str, Dict[str, np.ndarray]] = {}
        for split in ("val", "test"):
            if split not in loaders:
                continue
            t0 = time.time()
            res[split] = score_split(m, loaders[split], limit=limit)
            log.info(f"[{cid}] {split}: n={len(res[split]['y'])} "
                     f"in {time.time() - t0:.0f}s")

        # per-channel ablation deltas on val, for the reliability matrix
        abl: Dict[str, np.ndarray] = {}
        for ch in DEFAULT_SPECIALISTS[cid]:
            a = score_split(m, loaders["val"], ablate=[ch], limit=limit)
            abl[ch] = res["val"]["z"][:len(a["z"])] - a["z"]
            log.info(f"[{cid}] ablated {ch}: mean |dz| = {np.abs(abl[ch]).mean():.4f}")
        res["ablation_dz"] = abl                                      # type: ignore

        OUT.mkdir(parents=True, exist_ok=True)
        np.savez(OUT / f"scores_{cid}.npz",
                 **{f"{k}_{s}": res[s][k] for s in ("val", "test") if s in res
                    for k in ("y", "p", "z")},
                 **{f"dz_{ch}": v for ch, v in abl.items()})
        out[cid] = res
        del m
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
    return out


def stage_calibration(scores) -> pd.DataFrame:
    rows = []
    for cid, res in scores.items():
        if "val" not in res or "test" not in res:
            continue
        cal, cv = pick_calibrator(res["val"]["z"], res["val"]["y"])
        p_test = cal.transform(res["test"]["z"])
        rep_before = calibration_report(res["test"]["y"], res["test"]["p"])
        rep_after = calibration_report(res["test"]["y"], p_test, calibrator=cal)
        (OUT / f"calibration_{cid}.json").write_text(json.dumps(
            {"condition": cid, "label": CONDITION_LABELS.get(cid, cid),
             "cv_nll": cv, "calibrator": cal.to_dict(),
             "test_before": rep_before.to_dict(),
             "test_after": rep_after.to_dict()}, indent=2), encoding="utf-8")
        res["cal"] = cal
        res["p_test_cal"] = p_test
        res["p_val_cal"] = cal.transform(res["val"]["z"])

        print(f"\n[{cid}] {CONDITION_LABELS.get(cid, cid)}")
        print("  before:", str(rep_before).replace("\n", "\n          "))
        print("  after :", str(rep_after).replace("\n", "\n          "))
        for r, tag in ((rep_before, "raw"), (rep_after, "calibrated")):
            rows.append({"condition": cid, "label": CONDITION_LABELS.get(cid, cid),
                         "stage": tag, "n": r.n, "AUC": r.auc,
                         "ECE (equal-mass)": r.ece_quantile,
                         "ECE (equal-width)": r.ece_uniform, "MCE": r.mce_quantile,
                         "Brier": r.brier, "reliability": r.reliability,
                         "resolution": r.resolution, "NLL": r.nll,
                         "calibrator": cal.name if tag == "calibrated" else "none"})
    return pd.DataFrame(rows)


def stage_selective(scores, alpha: float) -> pd.DataFrame:
    rows = []
    for cid, res in scores.items():
        if "p_val_cal" not in res:
            continue
        pol = fit_abstention(res["p_val_cal"], res["val"]["y"], alpha=alpha,
                             class_conditional=True, assumed_prior=0.864,
                             source="celebdf_v2_identity/val")
        ev = evaluate_policy(pol, res["p_test_cal"], res["test"]["y"])
        if cid == BASELINE_ID:
            pol.save(OUT / "abstention_policy.json")
        base = majority_class_baseline(res["test"]["y"])
        a_urc = aurc(res["test"]["y"], res["p_test_cal"])
        print(f"\n[{cid}] {pol}")
        print(f"        test: coverage {ev['coverage']:.1%}  selective acc "
              f"{ev['selective_accuracy']:.4f}  (majority baseline {base:.4f})"
              f"  AURC {a_urc:.4f}")
        if np.isfinite(ev["selective_accuracy"]) and ev["selective_accuracy"] <= base:
            print("        ^ selective accuracy does NOT beat 'always say fake'. "
                  "Report it that way; the abstention band is not buying anything "
                  "on this condition.")
        for r in selective_at_coverage(res["test"]["y"], res["p_test_cal"]):
            rows.append({"condition": cid, "label": CONDITION_LABELS.get(cid, cid),
                         **r,
                         "AURC": aurc(res["test"]["y"], res["p_test_cal"]),
                         "E-AURC": e_aurc(res["test"]["y"], res["p_test_cal"]),
                         "alpha": alpha, "tau_real": pol.tau_real,
                         "tau_fake": pol.tau_fake})
    return pd.DataFrame(rows)


def stage_reliability(scores):
    """
    Per-channel reliability from the ablation deltas measured during scoring.

    The faithfulness and stability columns are left as NaN here and filled in by
    `cfx_eval.py`, which measures them on real clips. Until then every channel
    reads `beats_random=False`, so the matrix scores everything at zero and CFX
    abstains -- which is the correct default: no channel has earned a voice yet.
    """
    from sklearn.metrics import roc_auc_score

    per: Dict[Tuple[str, str], dict] = {}
    for cid, res in scores.items():
        dz = res.get("ablation_dz") or {}
        if not dz and (OUT / f"scores_{cid}.npz").exists():
            d = np.load(OUT / f"scores_{cid}.npz")
            dz = {k[3:]: d[k] for k in d.files if k.startswith("dz_")}
        y = res["val"]["y"]
        for ch, delta in dz.items():
            n = min(len(y), len(delta))
            try:
                a = float(roc_auc_score(y[:n], delta[:n]))
                a = max(a, 1.0 - a)      # sign-agnostic: direction is not the point
            except Exception:
                a = float("nan")
            per[(cid, ch)] = {"auc": a, "n": n, "beats_random": False,
                              "coverage": float("nan")}
            print(f"  {cid:<4} {ch:<10} ablation-delta AUC {a:.4f}  (n={n})")

    m = build_reliability_matrix(per, split="val")
    m.notes.append("faithfulness / stability / beats_random are NOT set by "
                   "cfx_fit.py -- they are measured per clip by cfx_eval.py. "
                   "Until that has run, every channel scores 0 and CFX abstains "
                   "on everything, which is the correct default rather than a "
                   "failure.")
    m.notes.append("per-channel AUC here is an ablation-delta proxy, not a "
                   "retrained single-channel model. Do not report it as "
                   "'channel AUC' without that qualifier.")
    m.save(OUT / "evidence_reliability.json")
    print("\n" + m.summary())
    return m


def stage_stack(scores, matrix) -> pd.DataFrame:
    """
    Fit the arbitration weights on val, then test whether they beat B0 alone.

    Feature construction here uses the ablation deltas as each channel's signed
    evidence and the Shapley-share fallback for extremeness, because the TSC
    descriptor percentiles are per clip and live in `cfx_eval.py`. That makes this
    stack a *lower bound* on what arbitration can do -- which is the right
    direction for a number that has to survive a paired bootstrap.
    """
    base = scores.get(BASELINE_ID)
    if base is None or "p_val_cal" not in base:
        log.warning(f"no calibrated {BASELINE_ID} scores -- cannot fit the stack")
        return pd.DataFrame()

    n = len(base["val"]["y"])
    y = base["val"]["y"]
    X = np.zeros((n, len(FEATURE_ORDER)), dtype=np.float64)
    for i in range(n):
        sols = []
        for ch in CHANNELS:
            signed, share = 0.0, 0.0
            for cid, res in scores.items():
                dz = res.get("ablation_dz") or {}
                if ch in dz and i < len(dz[ch]):
                    signed = float(dz[ch][i])
                    denom = sum(abs(float(v[i])) for v in dz.values()
                                if i < len(v)) or 1.0
                    share = abs(signed) / denom
                    break
            sols.append(solidity(
                matrix, ch, signed_evidence=signed,
                percentile=50.0 + 50.0 * min(share / 0.5, 1.0) if share > 0 else 50.0,
                counterfactual_survived=bool(share > 0.10),
                stability_score=0.5,
                cross_model_signals={cid: (res["p_val_cal"][i] - 0.5)
                                     for cid, res in scores.items()
                                     if ch in DEFAULT_SPECIALISTS.get(cid, ())
                                     and i < len(res.get("p_val_cal", []))}))
        agree = float(np.abs(np.sign(np.array(
            [res["p_val_cal"][i] - 0.5 for res in scores.values()
             if i < len(res.get("p_val_cal", []))])).mean()))
        X[i] = build_features(sols, p_best_single=float(base["p_val_cal"][i]),
                              agreement=agree, uncertainty=0.0)

    stack = ConfidenceStack(C=1.0).fit(X, y)
    stack.save(OUT / "confidence_stack.json")
    p_stack = stack.predict_proba(X)

    cmp = compare_to_single_model(y, p_stack, base["p_val_cal"], label=BASELINE_ID)
    print(f"\narbitration vs {BASELINE_ID} (on val, where the stack was fitted -- "
          f"optimistic by construction):")
    print(f"  delta AUC {cmp['delta_auc']:+.4f}  "
          f"[{cmp['ci_lo']:+.4f}, {cmp['ci_hi']:+.4f}]  p={cmp['p_value']:.4f}")
    print(f"  {cmp['verdict']}")
    if "recommendation" in cmp:
        print(f"  -> {cmp['recommendation']}")
    print("\n  NOTE: this comparison is on the fitting split. The number that "
          "counts is the same comparison on TEST, produced by cfx_eval.py.")
    return pd.DataFrame([{**cmp, "split": "val (fitted -- optimistic)"}])


# ==========================================================================


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=list(CONDITION_FLAGS))
    ap.add_argument("--alpha", type=float, default=0.10,
                    help="target selective error rate for the abstention band")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap clips per split (use 32 for a smoke run)")
    ap.add_argument("--from-scores", action="store_true",
                    help="reuse results/cfx/scores_*.npz instead of re-scoring")
    a = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    print(f"device: {DEVICE}")
    if DEVICE != "cuda":
        print("WARNING: no CUDA. Scoring will be slow; the numbers are identical.")
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("\n=== 1. scoring ===")
    scores = stage_scores(a.only, a.limit, a.from_scores)
    if not scores:
        print("nothing scored -- no checkpoints and no cached scores found.")
        return 1

    print("\n=== 2. calibration (fit on val, reported on test) ===")
    cal_df = stage_calibration(scores)
    if not cal_df.empty:
        cal_df.to_csv(OUT / "cfx_calibration.csv", index=False)

    print("\n=== 3. selective prediction ===")
    sel_df = stage_selective(scores, a.alpha)
    if not sel_df.empty:
        sel_df.to_csv(OUT / "cfx_selective.csv", index=False)

    print("\n=== 4. reliability matrix ===")
    matrix = stage_reliability(scores)

    print("\n=== 5. confidence stack ===")
    arb_df = stage_stack(scores, matrix)
    if not arb_df.empty:
        arb_df.to_csv(OUT / "cfx_arbitration.csv", index=False)

    print(f"\nwrote -> {OUT}")
    print("next: python scripts/cfx_eval.py  (faithfulness, narration, and the "
          "test-split arbitration comparison)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
