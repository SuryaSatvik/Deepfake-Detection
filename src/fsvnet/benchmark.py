"""
benchmark.py
============

Two things your report needs, and one thing it must not contain.

**Must not contain:** a table claiming this model beats Skyra. Skyra's weights
are not released, we have no numbers on any shared benchmark, and the two
systems do not perform the same task. A number invented to fill that cell is
the single fastest way to lose an examiner's trust.

**Does contain:**

1. `capability_table()` — an honest, *qualitative* positioning against Skyra,
   Effort and the conventional baselines. Every cell is either a published
   fact with a citation or a property of our own code. Nothing is a score.
2. `run_comparison()` — a harness that produces a *quantitative* table against
   baselines you can actually execute, on one identity-disjoint split, with
   video-level metrics, bootstrap confidence intervals and a **paired**
   significance test.

The paired test is the part people skip. Two models evaluated on the same test
videos are not independent samples: comparing their separate confidence
intervals is a weaker and wronger test than resampling the videos jointly and
looking at the distribution of the *difference*. `paired_bootstrap` does the
latter, and it is what lets you write "0.83 vs 0.79, p = 0.04" instead of
"0.83 vs 0.79" and hoping nobody asks.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .utils import log, save_json


# ==========================================================================
# 1. Capability comparison — qualitative, cited, no invented numbers
# ==========================================================================

CAPABILITIES: List[Dict[str, str]] = [
    {
        "property": "Primary task",
        "Skyra (arXiv 2512.15693)": "grounded artefact reasoning + spatio-temporal localisation",
        "Effort (ICML 2025)": "binary AI-generated image detection, applied frame-wise",
        "FSV-Net (ours)": "binary video detection + calibrated forensic evidence",
        "FSV-Net + Qwen-QLoRA (ours)": "binary video detection by a VLM conditioned on forensic measurements",
    },
    {
        "property": "Backbone",
        "Skyra (arXiv 2512.15693)": "Qwen2.5-VL-7B",
        "Effort (ICML 2025)": "CLIP ViT-L/14, SVD-orthogonal adaptation",
        "FSV-Net (ours)": "EfficientNet-B0 + 3 forensic branches + temporal transformer",
        "FSV-Net + Qwen-QLoRA (ours)": "Qwen2.5-VL-3B, 4-bit, LoRA adapters",
    },
    {
        "property": "Trainable parameters",
        "Skyra (arXiv 2512.15693)": "~7 B (full SFT + GRPO)",
        "Effort (ICML 2025)": "small adapter over a frozen CLIP",
        "FSV-Net (ours)": "9.1 M (TSC branch: 129 K, 1.4%)",
        "FSV-Net + Qwen-QLoRA (ours)": "LoRA only, ~10-20 M of a 3 B model",
    },
    {
        "property": "Supervision required",
        "Skyra (arXiv 2512.15693)": "ViF-CoT-4K — human-annotated artefact rationales",
        "Effort (ICML 2025)": "image labels",
        "FSV-Net (ours)": "video labels only",
        "FSV-Net + Qwen-QLoRA (ours)": "video labels only (rationales optional)",
    },
    {
        "property": "Uses temporal evidence",
        "Skyra (arXiv 2512.15693)": "yes — via the VLM's own video understanding",
        "Effort (ICML 2025)": "no — frame-level",
        "FSV-Net (ours)": "yes — explicit temporal spectral statistic",
        "FSV-Net + Qwen-QLoRA (ours)": "yes — the statistic is injected as text",
    },
    {
        "property": "Sees sub-perceptual signal",
        "Skyra (arXiv 2512.15693)": "no — reasons over human-perceivable artefacts",
        "Effort (ICML 2025)": "implicitly, in learned features",
        "FSV-Net (ours)": "yes — spectral, by construction",
        "FSV-Net + Qwen-QLoRA (ours)": "yes — measured externally, given to the model",
    },
    {
        "property": "Output",
        "Skyra (arXiv 2512.15693)": "verdict + localisation + chain-of-thought",
        "Effort (ICML 2025)": "probability",
        "FSV-Net (ours)": "probability + percentile-calibrated evidence pack",
        "FSV-Net + Qwen-QLoRA (ours)": "probability + written rationale",
    },
    {
        "property": "Explanations are checkable",
        "Skyra (arXiv 2512.15693)": "against annotated artefact regions",
        "Effort (ICML 2025)": "n/a",
        "FSV-Net (ours)": "yes — every claim maps to a numbered percentile",
        "FSV-Net + Qwen-QLoRA (ours)": "yes — same measurements, in the prompt",
    },
    {
        "property": "Inference VRAM",
        "Skyra (arXiv 2512.15693)": "~16 GB+ (7B)",
        "Effort (ICML 2025)": "~2 GB",
        "FSV-Net (ours)": "< 1 GB",
        "FSV-Net + Qwen-QLoRA (ours)": "~4 GB (3B, NF4)",
    },
    {
        "property": "Weights available to us",
        "Skyra (arXiv 2512.15693)": "NO — paper only, cannot be run or compared against",
        "Effort (ICML 2025)": "yes (public repo)",
        "FSV-Net (ours)": "yes — trained here",
        "FSV-Net + Qwen-QLoRA (ours)": "yes — base weights public, adapters trained here",
    },
    {
        "property": "Reported benchmark",
        "Skyra (arXiv 2512.15693)": "ViF-Bench, 91.02% acc (their own benchmark)",
        "Effort (ICML 2025)": "94.05 AUC on AIGVDBench closed-source generators",
        "FSV-Net (ours)": "NOT YET MEASURED on any public benchmark",
        "FSV-Net + Qwen-QLoRA (ours)": "NOT YET MEASURED on any public benchmark",
    },
]


def capability_table(as_markdown: bool = False):
    """The qualitative comparison. Returns a DataFrame, or markdown."""
    df = pd.DataFrame(CAPABILITIES).set_index("property")
    return df.to_markdown() if as_markdown else df


HONESTY_NOTE = """
How to phrase this in the report
--------------------------------
CORRECT:   "Skyra reports 91.02% on its own ViF-Bench. We do not evaluate on
            ViF-Bench and therefore make no comparative accuracy claim; the
            comparison below is one of capability and cost, not performance."
CORRECT:   "Against baselines we can run on our own split, FSV-Net reaches
            <x> AUC versus <y> for the conventional configuration
            (paired bootstrap p = <p>)."
INCORRECT: "Our model outperforms Skyra."
INCORRECT: any table with a number for us and a number for Skyra in the same
            column, on different data.

The distinction that makes the capability table worth printing: Skyra reasons
over artefacts a *human could see*. A temporal spectral anomaly is not one of
those — it lives in the relationship between frames at a frequency scale no
tokeniser represents. That is a difference in what the two systems can perceive
at all, and it is defensible without a shared benchmark.
"""


# ==========================================================================
# 2. The protocol
# ==========================================================================

PROTOCOL = {
    "split": "identity-disjoint (union-find over every identity a manipulation touches)",
    "unit_of_evaluation": "video (clip scores averaged); never frame or clip",
    "clips_per_video": 2,
    "threshold": "Youden's J chosen on val, applied unchanged to test",
    "primary_metric": "ROC-AUC",
    "secondary_metrics": ["AP", "EER", "balanced accuracy"],
    "uncertainty": "2000-sample bootstrap 95% CI; paired bootstrap between models",
    "augmentation": "JPEG quality 40-95 and rescale 0.5-1.0, applied to BOTH classes",
    "compression_control": "report with and without the augmentation",
    "forbidden": [
        "random splits", "frame-level metrics", "threshold chosen on test",
        "training on the cross-dataset test set",
        "evaluating on watermarked generated video without stripping the watermark",
    ],
}


def check_protocol(df: pd.DataFrame, verbose: bool = True) -> Dict[str, object]:
    """
    Validate a manifest against the protocol before any number is produced.
    Cheap, and it catches the mistakes that invalidate everything downstream.
    """
    from .manifest import _group_overlap

    rep: Dict[str, object] = {"ok": True, "problems": [], "warnings": []}

    if "split" not in df.columns:
        rep["ok"] = False
        rep["problems"].append("no split column — run manifest.make_splits first")
        return _report(rep, verbose)

    overlap = _group_overlap(df)
    if overlap:
        rep["ok"] = False
        rep["problems"].append(
            f"{overlap['n_groups_in_multiple_splits']} identity groups appear in "
            "more than one split — this is the leak, and every metric below it "
            "is meaningless")

    for s in ("train", "val", "test"):
        sub = df[df.split == s]
        if len(sub) == 0:
            rep["ok"] = False
            rep["problems"].append(f"split '{s}' is empty")
            continue
        if sub.label.nunique() < 2:
            rep["ok"] = False
            rep["problems"].append(f"split '{s}' has only one class")
        n_groups = sub["group"].nunique() if "group" in sub else 0
        if n_groups < 5:
            rep["warnings"].append(
                f"split '{s}' has only {n_groups} identity groups — the metrics "
                "will be very noisy, and the CI will show it")
        frac_fake = float((sub.label == 1).mean())
        if not 0.2 <= frac_fake <= 0.8:
            rep["warnings"].append(
                f"split '{s}' is {frac_fake*100:.0f}% fake — lead with AP and "
                "balanced accuracy, not accuracy")

    if "dataset" in df.columns and df.dataset.nunique() > 1:
        per = df.groupby(["dataset", "label"]).size().unstack(fill_value=0)
        for ds, row in per.iterrows():
            if 0 in row.index and 1 in row.index and (row[0] == 0 or row[1] == 0):
                rep["warnings"].append(
                    f"dataset '{ds}' contributes only one class — the model can "
                    "learn to recognise the dataset instead of the manipulation")
    return _report(rep, verbose)


def _report(rep, verbose):
    if verbose:
        print("PROTOCOL CHECK:", "PASS" if rep["ok"] else "FAIL")
        for p in rep["problems"]:
            print(f"  BLOCKER {p}")
        for w in rep["warnings"]:
            print(f"  WARN    {w}")
    return rep


# ==========================================================================
# 3. Paired significance test
# ==========================================================================

def paired_bootstrap(y_true: np.ndarray, score_a: np.ndarray, score_b: np.ndarray,
                     n: int = 5000, seed: int = 42) -> Dict[str, float]:
    """
    Resample the *videos* (not the models) and look at the AUC difference.

    Returns the observed difference, a CI for it, and a two-sided p-value for
    "the difference is zero". Use this whenever two models were evaluated on
    the same test videos — which is every row of `run_comparison`.
    """
    from sklearn.metrics import roc_auc_score

    y_true = np.asarray(y_true)
    a, b = np.asarray(score_a), np.asarray(score_b)
    obs = roc_auc_score(y_true, a) - roc_auc_score(y_true, b)

    rng = np.random.default_rng(seed)
    diffs = []
    N = len(y_true)
    for _ in range(n):
        idx = rng.integers(0, N, N)
        if len(set(y_true[idx].tolist())) < 2:
            continue
        diffs.append(roc_auc_score(y_true[idx], a[idx])
                     - roc_auc_score(y_true[idx], b[idx]))
    diffs = np.array(diffs)
    if diffs.size == 0:
        return {"delta_auc": float(obs), "p_value": float("nan"),
                "ci_lo": float("nan"), "ci_hi": float("nan"), "n_resamples": 0}

    # two-sided p: how often the resampled difference crosses zero
    p = 2.0 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return {"delta_auc": float(obs),
            "ci_lo": float(np.percentile(diffs, 2.5)),
            "ci_hi": float(np.percentile(diffs, 97.5)),
            "p_value": float(min(1.0, p)),
            "n_resamples": int(diffs.size)}


# ==========================================================================
# 4. Baselines you can actually run
# ==========================================================================

BASELINES: Dict[str, dict] = {
    # Every one of these is a branch configuration of the same codebase, so the
    # preprocessing, split and evaluation are identical — the only thing that
    # differs is the model. That is what makes the comparison fair.
    "frame_cnn": dict(use_spatial=True, use_frequency=False, use_srm=False,
                      use_tsc=False, temporal_layers=1),
    "fsw_net_video": dict(use_spatial=True, use_frequency=True, use_srm=True,
                          use_tsc=False),
    "tsc_only": dict(use_spatial=False, use_frequency=False, use_srm=False,
                     use_tsc=True),
    "fsv_net_full": dict(use_spatial=True, use_frequency=True, use_srm=True,
                         use_tsc=True),
}


def run_comparison(df: pd.DataFrame, cfg, variants: Optional[Dict[str, dict]] = None,
                   epochs: Optional[int] = None, reference: str = "fsw_net_video",
                   out_dir: Optional[str] = None) -> Dict[str, object]:
    """
    Train every variant on the same split, evaluate at video level, and compare
    each against `reference` with a paired bootstrap.

    Returns {"table": DataFrame, "scores": {name: (y_true, y_score)},
             "comparisons": DataFrame}.
    """
    from .dataset import make_loaders
    from .evaluate import evaluate_model, full_metrics
    from .model import build_model
    from .train import evaluate_loader, load_checkpoint, train
    from .utils import free_cuda

    check_protocol(df)
    variants = variants or BASELINES
    out_dir = Path(out_dir or cfg.paths.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[dict] = []
    scores: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    for name, flags in variants.items():
        log.info("=" * 70)
        log.info(f"BASELINE: {name}  {flags}")
        log.info("=" * 70)

        c = copy.deepcopy(cfg)
        for k, v in flags.items():
            target = c.model if hasattr(c.model, k) else c.train
            setattr(target, k, v)
        if epochs:
            c.train.epochs = epochs
        c.run_name = f"cmp_{name}"

        loaders = make_loaders(df, c)
        model = build_model(c)
        train(model, loaders, c, class_weights=loaders["train"].dataset.class_weights(),
              run_name=c.run_name)
        model, _ = load_checkpoint(Path(c.paths.ckpt_dir) / f"{c.run_name}_best.pt",
                                   model, c.device)
        res = evaluate_model(model, loaders, c, ("val", "test"), tag=name)
        out = evaluate_loader(model, loaders["test"], c.device, c.train.amp)
        scores[name] = (out["y_true"], out["y_score"])

        row = {"model": name, **{k: v for k, v in flags.items()},
               "params": sum(p.numel() for p in model.parameters())}
        for k in ("auc", "ap", "eer", "acc", "balanced_acc",
                  "auc_ci_lo", "auc_ci_hi"):
            row[k] = res.get("test", {}).get(k)
        rows.append(row)

        del model
        free_cuda()

    table = pd.DataFrame(rows)

    # ---- paired comparisons against the reference model ----------------
    comps: List[dict] = []
    if reference in scores:
        y, ref_s = scores[reference]
        for name, (yy, s) in scores.items():
            if name == reference:
                continue
            if not np.array_equal(y, yy):
                log.warning(f"{name}: test order differs from the reference — "
                            "skipping the paired test")
                continue
            pb = paired_bootstrap(y, s, ref_s)
            comps.append({"model": name, "vs": reference, **pb,
                          "verdict": ("better" if pb["p_value"] < 0.05 and pb["delta_auc"] > 0
                                      else "worse" if pb["p_value"] < 0.05
                                      else "not resolved")})
    comparisons = pd.DataFrame(comps)

    table.to_csv(out_dir / "comparison_table.csv", index=False)
    if len(comparisons):
        comparisons.to_csv(out_dir / "comparison_paired.csv", index=False)
    save_json({"protocol": PROTOCOL, "variants": variants}, out_dir / "protocol.json")

    print("\n" + table.to_string(index=False))
    if len(comparisons):
        print("\nPaired comparisons (same test videos, 5000 resamples):")
        print(comparisons.to_string(index=False))
        print("\n'not resolved' means the data cannot distinguish the two models — "
              "\nwhich is a finding, not a gap to be filled with a bigger claim.")
    return {"table": table, "scores": scores, "comparisons": comparisons}


def print_capability_report() -> None:
    """Everything the report's comparison section needs, in one call."""
    print("=" * 78)
    print("CAPABILITY COMPARISON  (qualitative — no scores, by design)")
    print("=" * 78)
    print(capability_table(as_markdown=True))
    print(HONESTY_NOTE)
    print("=" * 78)
    print("EVALUATION PROTOCOL")
    print("=" * 78)
    for k, v in PROTOCOL.items():
        if isinstance(v, list):
            print(f"  {k}:")
            for i in v:
                print(f"      - {i}")
        else:
            print(f"  {k}: {v}")


__all__ = ["CAPABILITIES", "capability_table", "HONESTY_NOTE", "PROTOCOL",
           "check_protocol", "paired_bootstrap", "BASELINES", "run_comparison",
           "print_capability_report"]
