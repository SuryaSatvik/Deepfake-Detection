"""
utils.py
========

Seeding, logging, device handling and small helpers used everywhere else.
Nothing here imports torch at module level except lazily, so the manifest and
dataset-prep scripts can run on a machine with no torch installed.
"""

from __future__ import annotations

import json
import logging
import os
import random
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np

_LOGGER_NAME = "fsvnet"


# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------

def get_logger(name: str = _LOGGER_NAME, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", "%H:%M:%S"))
        logger.addHandler(h)
        logger.setLevel(level)
        logger.propagate = False
    return logger


log = get_logger()


@contextmanager
def timed(label: str):
    """with timed('preprocessing'): ..."""
    t0 = time.time()
    yield
    log.info(f"{label} took {time.time() - t0:.1f}s")


# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------

def seed_everything(seed: int = 42, deterministic: bool = True) -> None:
    """
    Seed python / numpy / torch. `deterministic=True` also pins cuDNN, which
    costs a little speed and buys reproducible numbers — worth it for a report
    where every figure has to be regenerable.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.benchmark = True
    except ImportError:
        pass


# --------------------------------------------------------------------------
# Device
# --------------------------------------------------------------------------

def resolve_device(preferred: str = "cuda"):
    """Return a torch.device, degrading to CPU with a loud message."""
    import torch
    if preferred.startswith("cuda") and torch.cuda.is_available():
        return torch.device(preferred)
    if preferred.startswith("cuda"):
        log.warning("CUDA requested but unavailable — falling back to CPU.")
    return torch.device("cpu")


def cuda_mem_summary() -> str:
    try:
        import torch
        if not torch.cuda.is_available():
            return "cpu"
        alloc = torch.cuda.memory_allocated() / 1024 ** 3
        reserved = torch.cuda.max_memory_reserved() / 1024 ** 3
        total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        return f"{alloc:.2f}G alloc / {reserved:.2f}G peak reserved / {total:.1f}G total"
    except Exception:
        return "n/a"


def free_cuda() -> None:
    try:
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


@contextmanager
def oom_guard(hint: str = ""):
    """
    Wrap a block so a CUDA OOM produces an actionable message instead of a
    wall of stack trace. Re-raises after printing.
    """
    try:
        yield
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            log.error("CUDA out of memory. " + (hint or ""))
            log.error("Try, in this order:")
            log.error("  1. cfg.train.grad_checkpointing = True")
            log.error("  2. cfg.data.num_frames = 8")
            log.error("  3. cfg.train.batch_size = 1  (raise grad_accum to compensate)")
            log.error("  4. cfg.model.use_spatial = False  (ablation run, tiny memory)")
            free_cuda()
        raise


# --------------------------------------------------------------------------
# Small IO helpers
# --------------------------------------------------------------------------

def save_json(obj: Any, path: os.PathLike) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def default(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        return str(o)

    path.write_text(json.dumps(obj, indent=2, default=default), encoding="utf-8")


def load_json(path: os.PathLike) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def human_count(n: int) -> str:
    for unit in ["", "K", "M", "B"]:
        if abs(n) < 1000:
            return f"{n:.0f}{unit}" if unit else f"{n}"
        n /= 1000.0
    return f"{n:.1f}T"


def count_params(module) -> Dict[str, int]:
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}


def chunked(it: Iterable, n: int):
    buf = []
    for x in it:
        buf.append(x)
        if len(buf) == n:
            yield buf
            buf = []
    if buf:
        yield buf


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.sum += float(val) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / self.count if self.count else 0.0


__all__ = [
    "get_logger", "log", "timed", "seed_everything", "resolve_device",
    "cuda_mem_summary", "free_cuda", "oom_guard", "save_json", "load_json",
    "human_count", "count_params", "chunked", "AverageMeter",
]
