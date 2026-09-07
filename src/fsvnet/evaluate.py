"""
evaluate.py
===========

Video-level metrics, the ablation runner, and the figures.

Everything here reports at **video level**. Frame-level and clip-level numbers
are reported nowhere, because they are not the quantity anyone cares about and
they are systematically optimistic (clips from one video are not independent).

Metrics
-------
AUC, AP, EER, accuracy at the val-selected threshold, plus a bootstrap 95%
confidence interval. The CI is not decoration: with ~900 test videos an AUC
difference of 0.02 between two ablations is inside the noise, and a table of
point estimates without intervals invites you to over-read it.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .utils import log, save_json


# ==========================================================================
# Metrics
# ==========================================================================

def equal_error_rate(y_true: np.ndarray, y_score: np.ndarray) -> float:
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fpr[i] + fnr[i]) / 2)


def best_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Youden's J on the given (validation!) split."""
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(y_true, y_score)
    return float(thr[int(np.argmax(tpr - fpr))])


def bootstrap_auc(y_true: np.ndarray, y_score: np.ndarray, n: int = 2000,
                  seed: int = 42) -> Dict[str, float]:
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    N = len(y_true)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, N, N)
        if len(set(y_true[idx].tolist())) < 2:
            continue
        vals.append(roc_auc_score(y_true[idx], y_score[idx]))
    if not vals:
        return {"lo": float("nan"), "hi": float("nan")}
    return {"lo": float(np.percentile(vals, 2.5)),
            "hi": float(np.percentile(vals, 97.5))}


def full_metrics(y_true: np.ndarray, y_score: np.ndarray,
                 threshold: float = 0.5, ci: bool = True) -> Dict[str, float]:
    from sklearn.metrics import (accuracy_score, average_precision_score,
                                 balanced_accuracy_score, f1_score,
                                 roc_auc_score)
    y_pred = (y_score >= threshold).astype(int)
    two_class = len(set(y_true.tolist())) > 1
    m = {
        "n": int(len(y_true)),
        "n_fake": int(y_true.sum()),
        "auc": float(roc_auc_score(y_true, y_score)) if two_class else float("nan"),
        "ap": float(average_precision_score(y_true, y_score)) if two_class else float("nan"),
        "eer": equal_error_rate(y_true, y_score) if two_class else float("nan"),
        "acc": float(accuracy_score(y_true, y_pred)),
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "threshold": float(threshold),
    }
    if ci and two_class:
        m.update({f"auc_ci_{k}": v for k, v in bootstrap_auc(y_true, y_score).items()})
    return m


# ==========================================================================
# Evaluation entry points
# ==========================================================================

def evaluate_model(model, loaders: Dict[str, object], cfg,
                   splits: Sequence[str] = ("val", "test"),
                   save_predictions: bool = True,
                   tag: str = "") -> Dict[str, Dict[str, float]]:
    """
    Evaluate on the given splits. The decision threshold is chosen on `val`
    and *applied* to `test` — choosing it on test is a small but real form of
    peeking, and it is the kind of thing a reviewer checks.
    """
    from .train import evaluate_loader

    results: Dict[str, Dict[str, float]] = {}
    thr = 0.5
    raw: Dict[str, dict] = {}

    for split in splits:
        if split not in loaders:
            continue
        out = evaluate_loader(model, loaders[split], cfg.device, cfg.train.amp)
        raw[split] = out
        if split == "val" and len(set(out["y_true"].tolist())) > 1:
            thr = best_threshold(out["y_true"], out["y_score"])
            log.info(f"threshold selected on val: {thr:.3f}")

    for split, out in raw.items():
        m = full_metrics(out["y_true"], out["y_score"], thr)
        m["branch_weights_mean"] = out["branch_weights_mean"]
        results[split] = m
        log.info(f"[{split}] AUC {m['auc']:.4f} "
                 f"(95% CI {m.get('auc_ci_lo', float('nan')):.3f}"
                 f"-{m.get('auc_ci_hi', float('nan')):.3f})  "
                 f"AP {m['ap']:.4f}  EER {m['eer']:.4f}  acc {m['acc']:.4f}")

        if save_predictions:
            pred_dir = Path(cfg.paths.pred_dir)
            pred_dir.mkdir(parents=True, exist_ok=True)
            name = f"{cfg.run_name}{('_' + tag) if tag else ''}_{split}.csv"
            pd.DataFrame({"video_index": out["video_ids"],
                          "y_true": out["y_true"],
                          "y_score": out["y_score"]}).to_csv(pred_dir / name,
                                                             index=False)
    return results


def cross_dataset_eval(model, df_other: pd.DataFrame, cfg,
                       name: str = "cross") -> Dict[str, float]:
    """
    Score a model on a dataset it never trained on — the only number that says
    anything about generalisation. Expect it to be much lower than the
    in-domain number; that gap IS the result, not a failure.
    """
    from torch.utils.data import DataLoader

    from .dataset import ClipDataset
    from .train import evaluate_loader

    df = df_other.copy()
    df["split"] = "test"
    ds = ClipDataset(df, cfg, "test")
    if len(ds) == 0:
        log.warning(f"[{name}] no usable videos — did you preprocess this set?")
        return {}
    loader = DataLoader(ds, batch_size=cfg.train.batch_size, shuffle=False,
                        num_workers=cfg.train.num_workers)
    out = evaluate_loader(model, loader, cfg.device, cfg.train.amp)
    m = full_metrics(out["y_true"], out["y_score"])
    log.info(f"[cross-dataset: {name}] AUC {m['auc']:.4f}  acc {m['acc']:.4f}  "
             f"n={m['n']}")
    return m


# ==========================================================================
# Ablation
# ==========================================================================

ABLATIONS = {
    "spatial_only":        dict(use_spatial=True,  use_frequency=False, use_srm=False, use_tsc=False),
    "spatial_freq_srm":    dict(use_spatial=True,  use_frequency=True,  use_srm=True,  use_tsc=False),
    "tsc_only":            dict(use_spatial=False, use_frequency=False, use_srm=False, use_tsc=True),
    "full":                dict(use_spatial=True,  use_frequency=True,  use_srm=True,  use_tsc=True),
}


def run_ablation(df: pd.DataFrame, cfg, variants: Optional[Dict[str, dict]] = None,
                 epochs: Optional[int] = None) -> pd.DataFrame:
    """
    Train one model per branch configuration and tabulate test AUC.

    `spatial_freq_srm` vs `full` is the row that decides whether the TSC branch
    earned its place; `tsc_only` says whether it carries signal by itself.
    Report all four, including the ones that make the contribution look small.
    """
    from .dataset import make_loaders
    from .model import build_model
    from .train import train

    variants = variants or ABLATIONS
    rows = []

    for name, flags in variants.items():
        log.info("=" * 70)
        log.info(f"ABLATION: {name}  {flags}")
        log.info("=" * 70)

        c = copy.deepcopy(cfg)
        for k, v in flags.items():
            setattr(c.model, k, v)
        if epochs:
            c.train.epochs = epochs
        c.run_name = f"{cfg.run_name}_{name}"

        loaders = make_loaders(df, c)
        model = build_model(c)
        train_ds = loaders["train"].dataset
        train(model, loaders, c, class_weights=train_ds.class_weights(),
              run_name=c.run_name)

        from .train import load_checkpoint
        best = Path(c.paths.ckpt_dir) / f"{c.run_name}_best.pt"
        model, _ = load_checkpoint(best, model, c.device)
        res = evaluate_model(model, loaders, c, ("val", "test"), tag=name)

        row = {"variant": name, **{k: v for k, v in flags.items()}}
        for split in ("val", "test"):
            if split in res:
                row[f"{split}_auc"] = res[split]["auc"]
                row[f"{split}_acc"] = res[split]["acc"]
                row[f"{split}_eer"] = res[split].get("eer")
        if "test" in res:
            row["test_auc_ci"] = (f"{res['test'].get('auc_ci_lo', float('nan')):.3f}"
                                  f"-{res['test'].get('auc_ci_hi', float('nan')):.3f}")
        rows.append(row)

        del model
        from .utils import free_cuda
        free_cuda()

    table = pd.DataFrame(rows)
    out = Path(cfg.paths.out_dir) / f"{cfg.run_name}_ablation.csv"
    table.to_csv(out, index=False)
    log.info(f"ablation table saved: {out}")
    print("\n" + table.to_string(index=False))
    return table


# ==========================================================================
# Figures
# ==========================================================================

def plot_roc(results: Dict[str, dict], out_path, title: str = "FSV-Net ROC"):
    from .viz import _ensure_backend
    _ensure_backend()
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    from .viz import SERIES, INK_MUTED, despine, use_style
    use_style()

    fig, ax = plt.subplots(figsize=(5.2, 5))
    for i, (name, r) in enumerate(results.items()):
        y, s = np.asarray(r["y_true"]), np.asarray(r["y_score"])
        if len(set(y.tolist())) < 2:
            continue
        fpr, tpr, _ = roc_curve(y, s)
        from sklearn.metrics import roc_auc_score
        ax.plot(fpr, tpr, lw=2, color=SERIES[i % len(SERIES)],
                label=f"{name} (AUC {roc_auc_score(y, s):.3f})")
    ax.plot([0, 1], [0, 1], ls=(0, (4, 3)), lw=1.2, color=INK_MUTED,
            label="chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=9)
    despine(ax)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info(f"figure: {out_path}")


def plot_training_curves(history: List[dict], out_path,
                         branch_names: Optional[List[str]] = None):
    from .viz import _ensure_backend
    _ensure_backend()
    import matplotlib.pyplot as plt

    from .viz import SERIES, BRANCH_COLOR, INK_MUTED, despine, use_style
    use_style()

    ep = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    axes[0].plot(ep, [h["train_loss"] for h in history], marker="o",
                 color=SERIES[0])
    axes[0].set_title("train loss"); axes[0].set_xlabel("epoch")

    if "val_auc" in history[0]:
        axes[1].plot(ep, [h.get("val_auc", np.nan) for h in history],
                     marker="o", color=SERIES[2])
        axes[1].axhline(0.5, ls=(0, (4, 3)), lw=1.2, color=INK_MUTED)
        axes[1].set_title("val AUC (video level)"); axes[1].set_xlabel("epoch")

    bw = [h.get("val_branch_weights") for h in history if h.get("val_branch_weights")]
    if bw:
        arr = np.array(bw)
        names = branch_names or [f"branch {i}" for i in range(arr.shape[1])]
        for i in range(arr.shape[1]):
            nm = names[i] if i < len(names) else f"branch {i}"
            axes[2].plot(ep[:len(arr)], arr[:, i], marker="o", label=nm,
                         color=BRANCH_COLOR.get(nm, SERIES[i % len(SERIES)]))
        axes[2].set_title("branch attention"); axes[2].set_xlabel("epoch")
        axes[2].legend(fontsize=8)
    for a in axes:
        despine(a)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info(f"figure: {out_path}")


def plot_tsc_comparison(real_frames: np.ndarray, fake_frames: np.ndarray,
                        out_path, n_bins: int = 64):
    """
    The money figure: side-by-side TSC maps for a real and a fake clip, with
    the temporal-whiteness profile underneath. This is what goes in the poster.
    """
    from .viz import _ensure_backend
    _ensure_backend()
    import matplotlib.pyplot as plt

    from .tsc import tsc_map_np

    from .viz import SERIES, INK_MUTED, despine, sequential_cmap, use_style
    use_style()
    cmap = sequential_cmap()

    maps = [tsc_map_np(f, n_bins) for f in (real_frames, fake_frames)]
    vmax = max(float(m[0].max()) for m in maps)          # ONE shared scale:
    vmin = min(float(m[0].min()) for m in maps)          # two panels are only
                                                          # comparable if the
                                                          # colour means the
                                                          # same in both
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for col, (frames, name) in enumerate([(real_frames, "REAL"), (fake_frames, "FAKE")]):
        tmap, spectra = maps[col]
        im = axes[0, col].imshow(tmap, aspect="auto", origin="lower", cmap=cmap,
                                 vmin=vmin, vmax=vmax)
        axes[0, col].set_title(f"{name} — TSC map")
        axes[0, col].set_xlabel("spatial frequency ring →")
        axes[0, col].set_ylabel("temporal frequency →")
        fig.colorbar(im, ax=axes[0, col], fraction=0.046)

        band = tmap[1:] + 1e-6
        white = np.exp(np.log(band).mean(0) - np.log(band.mean(0)))
        axes[1, col].plot(white, lw=2, color=SERIES[0 if col == 0 else 1])
        axes[1, col].set_ylim(0, 1)
        axes[1, col].axhline(white[2 * n_bins // 3:].mean(), ls=(0, (4, 3)),
                             lw=1.2, color=INK_MUTED,
                             label=f"fine-ring mean {white[2*n_bins//3:].mean():.3f}")
        axes[1, col].set_title(f"{name} — temporal whiteness per ring")
        axes[1, col].set_xlabel("spatial frequency ring")
        axes[1, col].legend(fontsize=8)
        despine(axes[1, col])

    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info(f"figure: {out_path}")


__all__ = ["full_metrics", "equal_error_rate", "best_threshold", "bootstrap_auc",
           "evaluate_model", "cross_dataset_eval", "run_ablation", "ABLATIONS",
           "plot_roc", "plot_training_curves", "plot_tsc_comparison"]
