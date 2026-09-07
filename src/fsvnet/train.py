"""
train.py
========

Training loop for FSV-Net.

Built for an 8 GB card:
  * gradient accumulation (batch 2 x accum 8 = effective 16)
  * AMP autocast, with the FFT paths forced back to fp32 (see tsc.py)
  * optional gradient checkpointing on the EfficientNet trunk
  * a warm-up epoch with the spatial backbone frozen, so the three new
    branches get gradient before the pretrained trunk starts moving
  * every checkpoint carries its config, so a run is reproducible from the
    .pt alone

Model selection is on **video-level val AUC**, not clip accuracy. Clip
accuracy is the wrong quantity: a video is one decision, and two clips from
the same video are not independent samples.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .losses import FSVLoss
from .utils import (AverageMeter, cuda_mem_summary, free_cuda, log, oom_guard,
                    save_json, seed_everything)


# --------------------------------------------------------------------------

def _amp_ctx(enabled: bool, device: str):
    """torch.amp.autocast across versions; a no-op on CPU."""
    if not enabled or device != "cuda":
        from contextlib import nullcontext
        return nullcontext()
    try:
        return torch.amp.autocast("cuda", dtype=torch.float16)
    except (AttributeError, TypeError):
        return torch.cuda.amp.autocast()


def _make_scaler(enabled: bool, device: str):
    if not enabled or device != "cuda":
        return None
    try:
        return torch.amp.GradScaler("cuda")
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler()


def build_optimizer(model, cfg):
    """
    Two parameter groups: the pretrained trunk gets a 10x smaller LR than the
    randomly-initialised branches. Training them at the same rate wrecks the
    ImageNet features in the first few hundred steps.
    """
    backbone, head = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (backbone if name.startswith("spatial.features") else head).append(p)

    groups = [{"params": head, "lr": cfg.train.lr}]
    if backbone:
        groups.append({"params": backbone, "lr": cfg.train.backbone_lr})
    return torch.optim.AdamW(groups, weight_decay=cfg.train.weight_decay)


def build_scheduler(optimizer, cfg, steps_per_epoch: int):
    total = max(1, cfg.train.epochs * steps_per_epoch)
    warm = max(1, cfg.train.warmup_epochs * steps_per_epoch)

    def lr_lambda(step):
        if step < warm:
            return (step + 1) / warm
        prog = (step - warm) / max(1, total - warm)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# --------------------------------------------------------------------------

@torch.no_grad()
def evaluate_loader(model, loader: DataLoader, device: str,
                    amp: bool = False) -> Dict[str, object]:
    """
    Run a loader and aggregate clip scores into **video-level** predictions by
    averaging the fake-probability of every clip belonging to the same video.
    """
    from sklearn.metrics import accuracy_score, roc_auc_score

    model.eval()
    vid_scores: Dict[int, List[float]] = {}
    vid_labels: Dict[int, int] = {}
    branch_w: List[np.ndarray] = []

    for batch in loader:
        clip, mask, lmf = _batch_inputs(batch, device)
        with _amp_ctx(amp, device):
            logits, info = model(clip, mask, lmf=lmf)
        prob = torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy()
        branch_w.append(info["branch_weights"].float().cpu().numpy())

        vidx = (batch["video_index"].numpy() if "video_index" in batch
                else np.arange(len(prob)) + len(vid_scores))
        for p, v, y in zip(prob, vidx, batch["label"].numpy()):
            vid_scores.setdefault(int(v), []).append(float(p))
            vid_labels[int(v)] = int(y)

    vids = sorted(vid_scores)
    y_true = np.array([vid_labels[v] for v in vids])
    y_score = np.array([float(np.mean(vid_scores[v])) for v in vids])

    out: Dict[str, object] = {
        "n_videos": len(vids),
        "y_true": y_true,
        "y_score": y_score,
        "video_ids": vids,
        "branch_weights_mean": (np.concatenate(branch_w).mean(0).tolist()
                                if branch_w else []),
    }
    if len(set(y_true.tolist())) > 1:
        out["auc"] = float(roc_auc_score(y_true, y_score))
    else:
        out["auc"] = float("nan")
        log.warning("only one class present — AUC undefined for this split")
    out["acc"] = float(accuracy_score(y_true, (y_score >= 0.5).astype(int)))
    return out


# --------------------------------------------------------------------------

def _batch_inputs(batch, device):
    """
    Pull model inputs out of a batch, whichever loader produced it.

    Two loaders feed this trainer: `dataset.py` (video decode + filmstrip
    cache) and `dataset_frames.py` (the pre-extracted Celeb-DF-v2 crops).
    They agree on "clip" and "label" and differ elsewhere:

      * "mask" is absent for the frames loader -- every frame there IS a face
        crop, so the mask is all-True and is synthesised here rather than
        stored 6,526 times.
      * the LMF pack ("lmf_energy" / "lmf_desc") exists only for the frames
        loader. It MUST be forwarded: without it the model's LMF branch
        receives None, zeroes its embedding, and the -LMF ablation silently
        compares two identical models. That bug would not crash; it would
        just quietly report no effect.
    """
    clip = batch["clip"].to(device, non_blocking=True)
    if "mask" in batch:
        mask = batch["mask"].to(device, non_blocking=True)
    else:
        mask = torch.ones(clip.shape[:2], dtype=torch.bool, device=device)
    lmf = None
    if "lmf_energy" in batch and "lmf_desc" in batch:
        lmf = {"energy": batch["lmf_energy"].to(device, non_blocking=True),
               "descriptors": batch["lmf_desc"].to(device, non_blocking=True)}
    return clip, mask, lmf



def train(model, loaders: Dict[str, DataLoader], cfg,
          class_weights: Optional[torch.Tensor] = None,
          run_name: Optional[str] = None) -> Dict[str, object]:
    """
    Train and return a history dict. Saves `<ckpt_dir>/<run_name>_best.pt` and
    `<run_name>_history.json`.
    """
    seed_everything(cfg.data.seed)
    device = cfg.device
    run_name = run_name or cfg.run_name
    model = model.to(device)

    if "train" not in loaders:
        raise ValueError("no train loader — check your manifest and splits")

    criterion = FSVLoss(
        class_weights.to(device) if class_weights is not None else None,
        gamma=cfg.train.focal_gamma,
        label_smoothing=cfg.train.label_smoothing,
    ).to(device)

    optimizer = build_optimizer(model, cfg)
    steps_per_epoch = max(1, len(loaders["train"]) // cfg.train.grad_accum)
    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch)
    scaler = _make_scaler(cfg.train.amp, device)

    ckpt_dir = Path(cfg.paths.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / f"{run_name}_best.pt"

    history: List[dict] = []
    best_metric = -np.inf
    run_started = time.time()
    _budget = float(getattr(cfg.train, 'max_seconds', 0.0) or 0.0)
    stopped_early = None
    bad_epochs = 0

    log.info(f"training {run_name} on {device} | branches={model.branch_names} "
             f"| effective batch {cfg.train.batch_size * cfg.train.grad_accum}")

    for epoch in range(cfg.train.epochs):
        # ---- backbone freeze schedule ---------------------------------
        if cfg.model.freeze_spatial_epochs:
            model.set_backbone_frozen(epoch < cfg.model.freeze_spatial_epochs)
            if epoch == cfg.model.freeze_spatial_epochs:
                optimizer = build_optimizer(model, cfg)   # pick up new params
                scheduler = build_scheduler(optimizer, cfg, steps_per_epoch)

        model.train()
        loss_m, cls_m, div_m = AverageMeter(), AverageMeter(), AverageMeter()
        t0 = time.time()
        optimizer.zero_grad(set_to_none=True)

        try:
            from tqdm.auto import tqdm
            pbar = tqdm(loaders["train"], desc=f"epoch {epoch+1}/{cfg.train.epochs}",
                        leave=False)
        except ImportError:
            pbar = loaders["train"]

        for step, batch in enumerate(pbar):
            clip, mask, lmf = _batch_inputs(batch, device)
            y = batch["label"].to(device, non_blocking=True)

            with oom_guard(f"epoch {epoch}, step {step}"):
                with _amp_ctx(cfg.train.amp, device):
                    logits, info = model(clip, mask, lmf=lmf)
                    out = criterion(logits, y, info["branch_weights"], epoch)
                    loss = out["loss"] / cfg.train.grad_accum

                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                if (step + 1) % cfg.train.grad_accum == 0:
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(),
                                             cfg.train.max_grad_norm)
                    if scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()

            # --- mid-epoch wall-clock guard --------------------------------
            # The end-of-epoch check is useless if the EPOCH is what runs long.
            # Checked every 50 steps: cheap, and it bounds the damage from a
            # sudden slowdown (throttling, power-profile change, another
            # process) to at most a fraction of one epoch.
            if _budget > 0 and step % 50 == 0 and step > 0:
                spent_now = time.time() - run_started
                if spent_now > _budget * 1.25:
                    log.warning(
                        f"mid-epoch budget breach: {spent_now/60:.1f} min spent "
                        f"against a {_budget/60:.1f} min budget. Abandoning "
                        f"epoch {epoch+1} and keeping the best checkpoint so far.")
                    stopped_early = "budget_midepoch"
                    break

            n = y.size(0)
            loss_m.update(out["loss"].item(), n)
            cls_m.update(out["cls"].item(), n)
            div_m.update(out["div"].item(), n)
            if hasattr(pbar, "set_postfix"):
                pbar.set_postfix(loss=f"{loss_m.avg:.4f}")

        # ---- validation ------------------------------------------------
        rec = {"epoch": epoch + 1, "train_loss": loss_m.avg,
               "train_cls": cls_m.avg, "train_div": div_m.avg,
               "lr": optimizer.param_groups[0]["lr"],
               "secs": round(time.time() - t0, 1)}

        if "val" in loaders:
            val = evaluate_loader(model, loaders["val"], device, cfg.train.amp)
            rec.update({"val_auc": val["auc"], "val_acc": val["acc"],
                        "val_branch_weights": val["branch_weights_mean"]})
        history.append(rec)

        bw = rec.get("val_branch_weights") or []
        bw_str = ("  gate " + " ".join(f"{n[:4]}={w:.2f}" for n, w
                                       in zip(model.branch_names, bw))) if bw else ""
        log.info(f"epoch {epoch+1:>2}  loss {loss_m.avg:.4f}  "
                 f"val_auc {rec.get('val_auc', float('nan')):.4f}  "
                 f"val_acc {rec.get('val_acc', float('nan')):.4f}  "
                 f"{rec['secs']:.0f}s  [{cuda_mem_summary()}]{bw_str}")

        if stopped_early == "budget_midepoch":
            break

        # ---- checkpoint / early stop -----------------------------------
        metric = rec.get(cfg.train.monitor, None)
        if metric is None or (isinstance(metric, float) and np.isnan(metric)):
            metric = -rec["train_loss"]

        min_delta = float(getattr(cfg.train, "early_stop_min_delta", 0.0) or 0.0)
        improved = metric > best_metric              # worth KEEPING
        meaningful = metric > best_metric + min_delta  # worth CONTINUING for

        if improved:
            best_metric = metric
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": cfg.to_dict(),
                "epoch": epoch + 1,
                "metric": float(metric),
                "monitor": cfg.train.monitor,
                "branch_names": model.branch_names,
            }, best_path)
            log.info(f"         new best ({cfg.train.monitor}={metric:.4f}) -> {best_path.name}")

        if meaningful:
            bad_epochs = 0
        else:
            bad_epochs += 1
            log.info(f"         no meaningful gain (>{min_delta:g}) "
                     f"[{bad_epochs}/{cfg.train.early_stop_patience}]")
            if bad_epochs >= cfg.train.early_stop_patience:
                log.info(f"early stop: {bad_epochs} epochs without a gain above "
                         f"{min_delta:g}. Best {cfg.train.monitor}={best_metric:.4f}.")
                stopped_early = "patience"
                break

        # ---- wall-clock budget -----------------------------------------
        budget = _budget
        if budget > 0:
            spent = time.time() - run_started
            if spent >= budget:
                log.info(f"wall-clock budget reached ({spent/60:.1f} min of "
                         f"{budget/60:.1f}); stopping after epoch {epoch+1} of "
                         f"{cfg.train.epochs}. Best checkpoint is saved.")
                stopped_early = "budget"
                break
            per_epoch = spent / (epoch + 1)
            if spent + per_epoch > budget:
                log.info(f"stopping now: another epoch (~{per_epoch/60:.1f} min) "
                         f"would exceed the {budget/60:.1f} min budget. "
                         f"Completed {epoch+1} of {cfg.train.epochs} epochs.")
                stopped_early = "budget"
                break

    free_cuda()
    hist_path = ckpt_dir / f"{run_name}_history.json"
    save_json({"history": history, "best_metric": float(best_metric),
               "best_checkpoint": str(best_path),
               "epochs_completed": len(history),
               "epochs_requested": int(cfg.train.epochs),
               "stopped_early": stopped_early,
               "total_secs": round(time.time() - run_started, 1),
               "config": cfg.to_dict()},
              hist_path)
    log.info(f"history saved: {hist_path}")

    return {"history": history, "best_metric": float(best_metric),
            "epochs_completed": len(history), "stopped_early": stopped_early,
            "best_checkpoint": str(best_path)}


def load_checkpoint(path, model=None, device: str = "cpu"):
    """Restore a checkpoint; builds the model from the stored config if needed."""
    ck = torch.load(path, map_location=device)
    if model is None:
        from .config import Config
        from .model import build_model
        cfg = Config()
        # rehydrate only the fields build_model reads
        d = ck.get("config", {})
        for section in ["model", "data", "train"]:
            for k, v in (d.get(section) or {}).items():
                if hasattr(getattr(cfg, section), k):
                    try:
                        setattr(getattr(cfg, section), k, v)
                    except Exception:
                        pass
        cfg.model.pretrained = False        # weights come from the checkpoint
        cfg.model.init_from_fswnet = None
        model = build_model(cfg)
    model.load_state_dict(ck["model_state_dict"], strict=False)
    model.to(device).eval()
    log.info(f"loaded {path} (epoch {ck.get('epoch')}, "
             f"{ck.get('monitor')}={ck.get('metric')})")
    return model, ck


__all__ = ["train", "evaluate_loader", "build_optimizer", "build_scheduler",
           "load_checkpoint"]
