#!/usr/bin/env python
"""
cfx_eval.py
===========

The measurement pass. Everything in here is per clip, on the identity-disjoint
TEST split, and produces the tables the report needs.

    python scripts/cfx_eval.py --n 200                # the faithfulness benchmark
    python scripts/cfx_eval.py --n 100 --narrate      # + Qwen narration quality
    python scripts/cfx_eval.py --n 8 --cpu            # smoke run

Run `scripts/cfx_fit.py` first: this script reads the calibrators, the abstention
policy and the reliability matrix it wrote.

Outputs, in `results/cfx/`:

    cfx_faithfulness.csv       method x metric, with the margin over random
    cfx_channel_reliability.csv  the faithfulness/stability columns of the matrix
    evidence_reliability.json  updated in place with those columns filled in
    cfx_narration.csv          groundedness, numeral fidelity, FK grade, fallbacks
    cfx_quadrants.csv          how many clips landed in each output quadrant
    cfx_correctness.json       does faithfulness predict correctness?
    cards/<clip_id>.json       one ExplanationCard per clip

The sampling is stated, not implied
-----------------------------------
`--n` clips are drawn stratified by label with a fixed seed, because the full
faithfulness pass is ~2,800 forwards per clip. The CSVs carry `n` and the seed so
the report can say "on a stratified 200-clip subsample" rather than implying
full-split coverage.

Two results this script is designed to be able to return
--------------------------------------------------------
* **No method beats random.** Then no explanation should ever be shown, and that
  is the finding. `cfx_faithfulness.csv` will say so per method.
* **Faithfulness does not predict correctness.** Then it is not a confidence
  signal, and `cfx_correctness.json` reports the null with its CI.

Both are written the same way as a positive result. A pipeline that can only
produce good news is not a measurement.

UNTESTED against real checkpoints -- exercised only against the stub model in
`scripts/test_cfx.py`. Start with `--n 8`.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fsvnet import Config, celebdf_frames as cf                        # noqa: E402
from fsvnet.arbitrate import (BASELINE_ID, CONDITION_LABELS,           # noqa: E402
                              ConfidenceStack, ReliabilityMatrix,
                              compare_to_single_model)
from fsvnet.cfx import CFXConfig, CFXPipeline                          # noqa: E402
from fsvnet.dataset_frames import lmf_pack, make_frame_loaders         # noqa: E402
from fsvnet.evidence import ReferenceStats                             # noqa: E402
from fsvnet.explain import denormalise                                 # noqa: E402
from fsvnet.faithfulness import faithfulness_vs_correctness            # noqa: E402
from fsvnet.model import build_model                                   # noqa: E402
from fsvnet.narrate import LAYMAN, TECHNICAL                           # noqa: E402
from fsvnet.selective import ABSTAIN, AbstentionPolicy                 # noqa: E402
from fsvnet.utils import log                                           # noqa: E402

NUM_FRAMES, IMAGE_SIZE, SEED = 16, 224, 42
MAN = ROOT / "data" / "manifests" / "celebdf_v2_identity.csv"
CACHE = ROOT / "data" / "processed" / f"lmf_cache_{NUM_FRAMES:02d}f.npz"
CKPT_DIR = ROOT / "models" / "checkpoints"
FIT = ROOT / "results" / "cfx"
OUT = ROOT / "results" / "cfx"
CARDS = OUT / "cards"
REF = ROOT / "data" / "manifests" / "reference_stats.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CONDITION_FLAGS = {"B0": (False, False), "C1": (True, False),
                   "C2": (False, True), "C3": (True, True)}


# ==========================================================================


def load_models(only: List[str], device: str) -> Dict[str, object]:
    models: Dict[str, object] = {}
    for cid in only:
        ck = CKPT_DIR / f"fsvnet_dfe_{cid}.pt"
        if not ck.exists():
            log.warning(f"[{cid}] no checkpoint -- excluded from the panel")
            continue
        blob = torch.load(ck, map_location=device, weights_only=False)
        c = Config()
        c.data.num_frames, c.data.img_size, c.data.seed = NUM_FRAMES, IMAGE_SIZE, SEED
        c.device = device
        c.model.use_tsc = blob.get("use_tsc", CONDITION_FLAGS[cid][0])
        c.model.use_lmf = blob.get("use_lmf", CONDITION_FLAGS[cid][1])
        m = build_model(c).to(device)
        m.load_state_dict(blob["state_dict"])
        m.eval()
        models[cid] = m
        log.info(f"[{cid}] loaded {ck.name} "
                 f"(tsc={c.model.use_tsc}, lmf={c.model.use_lmf})")
    return models


def sample_clips(n: int, seed: int = SEED):
    """Stratified by label so a 14%-real split does not vanish from the sample."""
    df = cf.load_manifest(MAN, ROOT)
    test = df[df.split == "test"]
    rng = np.random.default_rng(seed)
    per = max(1, n // 2)
    idx = []
    for lab in (0, 1):
        rows = test[test.label == lab]
        k = min(per, len(rows))
        idx += list(rng.choice(rows.index.values, size=k, replace=False))
    return df, sorted(idx)


# ==========================================================================


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200, help="clips to analyse")
    ap.add_argument("--only", nargs="*", default=list(CONDITION_FLAGS))
    ap.add_argument("--narrate", action="store_true",
                    help="load Qwen and measure the layman register too")
    ap.add_argument("--cpu", action="store_true", help="force the CPU profile")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--strict-coverage", action="store_true", default=True)
    ap.add_argument("--no-strict-coverage", dest="strict_coverage",
                    action="store_false")
    a = ap.parse_args()

    device = "cpu" if a.cpu else DEVICE
    OUT.mkdir(parents=True, exist_ok=True)
    CARDS.mkdir(parents=True, exist_ok=True)
    print(f"device: {device}    clips: {a.n}    seed: {a.seed}")

    models = load_models(a.only, device)
    if not models:
        print("no checkpoints found -- nothing to evaluate.")
        return 1

    matrix = (ReliabilityMatrix.load(FIT / "evidence_reliability.json")
              if (FIT / "evidence_reliability.json").exists()
              else ReliabilityMatrix())
    policy = (AbstentionPolicy.load(FIT / "abstention_policy.json")
              if (FIT / "abstention_policy.json").exists() else None)
    stack = (ConfidenceStack.load(FIT / "confidence_stack.json")
             if (FIT / "confidence_stack.json").exists() else None)
    reference = ReferenceStats.load(REF) if REF.exists() else None
    if policy is None:
        print("! no abstention_policy.json -- run scripts/cfx_fit.py first, or "
              "nothing will be abstained on.")
    if reference is None:
        print("! no reference_stats.json -- descriptor percentiles are "
              "unavailable, so no measurement can be called anomalous.")

    cfx = CFXConfig()
    if a.cpu:
        cfx = cfx.cpu_fallback()
    cfx.narrate = True

    reasoner = None
    if a.narrate:
        from fsvnet.vlm_reasoner import EvidenceReasoner
        c = Config()
        c.device = device
        reasoner = EvidenceReasoner(c)
        print("Qwen will be loaded lazily on the first clip that clears the gate.")

    pipe = CFXPipeline(models=models, matrix=matrix, stack=stack, policy=policy,
                       cfx=cfx, reasoner=reasoner, reference=reference)

    # ---- gather the clips ------------------------------------------
    df, idx = sample_clips(a.n, a.seed)
    c = Config()
    c.data.num_frames, c.data.img_size = NUM_FRAMES, IMAGE_SIZE
    c.train.batch_size, c.train.num_workers = 1, 0
    c.device = device
    loaders = make_frame_loaders(df, c, CACHE)
    want = set(str(df.loc[i].clip_name) for i in idx)

    cards = []
    t0 = time.time()
    for b in loaders["test"]:
        name = b["clip_name"][0] if isinstance(b["clip_name"], list) else str(b["clip_name"])
        if name not in want:
            continue
        clip = b["clip"].to(device)
        mask = b["mask"].to(device) if "mask" in b else None
        card = pipe.analyse(clip, mask, lmf_pack(b), frames_u8=denormalise(clip[0]),
                            clip_id=name)
        card.save(CARDS / f"{name}.json")
        cards.append((card, int(b["label"][0])))
        done = len(cards)
        print(f"  [{done}/{len(want)}] {name:<28} {card.verdict.decision:<7} "
              f"p={card.verdict.p_fake_calibrated:.3f} "
              f"{card.quadrant:<26} best={card.best.method if card.best else '-'} "
              f"({card.n_forwards} fwd)")
        if done >= len(want):
            break
    dt = time.time() - t0
    if not cards:
        print("no clips matched the sample -- check the manifest split names.")
        return 1
    print(f"\n{len(cards)} clips in {dt:.0f}s ({dt / len(cards):.1f}s/clip)")

    # ---- 1. faithfulness table -------------------------------------
    rows = []
    for card, y in cards:
        for s in card.all_scored:
            rows.append({"clip_id": card.clip_id, "label": y,
                         "method": s.method, "modality": s.modality,
                         "admissible": s.admissible, "grade": s.grade,
                         "beats_random": s.beats_random,
                         "margin_over_random": s.margin_over_random,
                         "solidity": s.solidity,
                         **{f"m_{k}": v for k, v in s.faithfulness.items()}})
    fdf = pd.DataFrame(rows)
    fdf.to_csv(OUT / "cfx_faithfulness_raw.csv", index=False)

    summary = (fdf.groupby(["method", "modality"])
                  .agg(n=("clip_id", "count"),
                       beats_random_rate=("beats_random", "mean"),
                       mean_margin=("margin_over_random", "mean"),
                       admissible_rate=("admissible", "mean"),
                       mean_solidity=("solidity", "mean"))
                  .reset_index()
                  .sort_values("mean_margin", ascending=False))
    summary["n_clips"] = len(cards)
    summary["seed"] = a.seed
    summary["verdict"] = np.where(
        summary["beats_random_rate"] > 0.5,
        "beats random on a majority of clips",
        "DOES NOT beat random on a majority of clips -- not an explanation")
    summary.to_csv(OUT / "cfx_faithfulness.csv", index=False)
    print("\n=== faithfulness, per method ===")
    print(summary.to_string(index=False))

    # ---- 2. fill the reliability matrix's measured columns ----------
    chan_of = {"spatial": "spatial", "spectral": "frequency",
               "temporal": "tsc", "branch": "srm"}
    crows = []
    for (modality,), g in fdf.groupby(["modality"]):
        ch = chan_of.get(modality)
        if ch is None:
            continue
        faith = float(np.nanmean(g["margin_over_random"]))
        beats = bool(np.nanmean(g["beats_random"]) > 0.5)
        for cid in models:
            e = matrix.get(cid, ch)
            if e is None:
                continue
            e.faithfulness = faith
            e.beats_random = beats
            e.stability = float("nan")       # set by --stability, below
            matrix.put(e)
            crows.append({"condition": cid, "channel": ch,
                          "auc_proxy": e.auc, "faithfulness": faith,
                          "beats_random": beats, "score": e.score()})
    matrix.notes.append(f"faithfulness and beats_random filled from "
                        f"cfx_eval.py on {len(cards)} test clips (seed {a.seed}).")
    matrix.save(OUT / "evidence_reliability.json")
    if crows:
        pd.DataFrame(crows).to_csv(OUT / "cfx_channel_reliability.csv", index=False)
        print("\n=== reliability matrix, measured columns filled ===")
        print(matrix.summary())

    # ---- 3. quadrants ----------------------------------------------
    q = pd.Series([c.quadrant for c, _ in cards]).value_counts().rename_axis(
        "quadrant").reset_index(name="n")
    q["fraction"] = q["n"] / len(cards)
    q.to_csv(OUT / "cfx_quadrants.csv", index=False)
    coverage = float(np.mean([c.verdict.decision != ABSTAIN for c, _ in cards]))
    print(f"\n=== quadrants (coverage {coverage:.1%}) ===")
    print(q.to_string(index=False))
    if coverage > 0.90:
        msg = (f"coverage is {coverage:.1%}. At this project's honest test AUC "
               "that is implausible -- check the split for identity leakage and "
               "check the calibrator was fitted on val.")
        print("\n! " + msg)
        if a.strict_coverage:
            (OUT / "COVERAGE_WARNING.txt").write_text(msg, encoding="utf-8")

    # ---- 4. narration quality --------------------------------------
    nrows = []
    for card, y in cards:
        if card.narration is None:
            continue
        nrows.append({
            "clip_id": card.clip_id, "decision": card.verdict.decision,
            "backend": card.narration.backend,
            "groundedness": card.narration.groundedness,
            "fk_grade": card.narration.fk_grade,
            "numeral_ok": bool(card.narration.numeral_check.get("ok", False)),
            "unsupported_numerals": ";".join(
                card.narration.numeral_check.get("unsupported", [])),
            "back_translation_ok": card.narration.back_translation_ok,
            "fell_back_to_template": card.narration.fell_back_to_template,
            "n_claims": len(card.narration.claims),
            "n_dropped": len(card.narration.dropped_claims),
            "layman_words": len(card.narration.layman.split()),
        })
    ndf = pd.DataFrame(nrows)
    if not ndf.empty:
        ndf.to_csv(OUT / "cfx_narration.csv", index=False)
        print("\n=== narration quality ===")
        print(f"  layman groundedness   {ndf['groundedness'].mean():.3f}")
        print(f"  numeral fidelity      {ndf['numeral_ok'].mean():.1%} of clips clean")
        print(f"  back-translation pass {ndf['back_translation_ok'].mean():.1%}")
        print(f"  template fallback     {ndf['fell_back_to_template'].mean():.1%}")
        print(f"  FK grade              {ndf['fk_grade'].mean():.1f} "
              f"(target <= 8, ceiling 10)")
        print(f"  claims dropped        {ndf['n_dropped'].sum()} across "
              f"{len(ndf)} clips")
        if ndf["fell_back_to_template"].mean() > 0.5 and a.narrate:
            print("  ! more than half the clips fell back to the template. Read "
                  "the per-card narration notes: either Qwen did not load, or "
                  "its output is being rejected by the claim-link filter.")

    # ---- 5. does faithfulness predict correctness? -----------------
    faith, correct = [], []
    for card, y in cards:
        if card.verdict.decision == ABSTAIN or card.best is None:
            continue
        m = card.best.margin_over_random
        if not np.isfinite(m):
            continue
        faith.append(float(m))
        correct.append(bool((card.verdict.decision == "fake") == (y == 1)))
    r = faithfulness_vs_correctness(faith, correct)
    (OUT / "cfx_correctness.json").write_text(json.dumps(r, indent=2),
                                              encoding="utf-8")
    print("\n=== does explanation faithfulness predict correctness? ===")
    if r.get("resolved"):
        print(f"  RESOLVED: r={r['r']:+.3f}  95% CI [{r['ci_lo']:+.3f}, "
              f"{r['ci_hi']:+.3f}]  n={r['n']}")
        print("  -> faithfulness carries confidence information and belongs in "
              "the selective score.")
    else:
        print(f"  NOT RESOLVED: r={r['r']:+.3f}  95% CI [{r.get('ci_lo', float('nan')):+.3f}, "
              f"{r.get('ci_hi', float('nan')):+.3f}]  n={r['n']}")
        print("  -> report the null. Faithfulness does not predict correctness "
              "on this evidence, which is a real finding about the design and "
              "not a gap to fill.")

    # ---- 6. arbitration vs the baseline, on TEST -------------------
    if stack is not None:
        ys = np.array([1.0 if y == 1 else 0.0 for _, y in cards])
        p_stack = np.array([c.verdict.p_fake_calibrated for c, _ in cards])
        p_base = np.array([c.verdict.per_model_prob.get(BASELINE_ID, np.nan)
                           for c, _ in cards])
        if np.isfinite(p_base).all() and len(np.unique(ys)) == 2:
            cmp = compare_to_single_model(ys, p_stack, p_base, label=BASELINE_ID)
            (OUT / "cfx_arbitration_test.json").write_text(
                json.dumps(cmp, indent=2), encoding="utf-8")
            print(f"\n=== arbitration vs {BASELINE_ID}, on TEST ===")
            print(f"  delta AUC {cmp['delta_auc']:+.4f}  "
                  f"[{cmp['ci_lo']:+.4f}, {cmp['ci_hi']:+.4f}]  "
                  f"p={cmp['p_value']:.4f}")
            print(f"  {cmp['verdict']}")
            if "recommendation" in cmp:
                print(f"  -> {cmp['recommendation']}")
            print(f"  (n={len(ys)} clips -- a subsample. State that beside the "
                  "number.)")

    # ---- 7. one example of each register ---------------------------
    ex = next((c for c, _ in cards if c.best is not None), cards[0][0])
    print(f"\n=== example: {ex.clip_id} ===")
    print("technical:", ex.for_user(TECHNICAL)["text"])
    print("layman   :", ex.for_user(LAYMAN)["text"])

    print(f"\nwrote -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
