"""
FSV-Net — Frequency-Spatial-Watermark Video Network
====================================================

A video deepfake detector that extends the image-level FSW-Net
(../DeepfakeProject) with two contributions:

  1. **Temporal Spectral Coherence (TSC)** — a clip-level branch that measures
     how the spatial power spectrum evolves *in time*. Real camera detail is
     temporally white (sensor noise); generated detail is temporally coloured.
     See `fsvnet.tsc`.

  2. **Evidence-conditioned reasoning** — a Skyra-style VLM layer that is given
     the detector's percentile-calibrated forensic measurements, not just the
     frames, so its explanations are grounded and checkable.
     See `fsvnet.evidence` and `fsvnet.vlm_reasoner`.

Typical use (see notebooks/FSV_Net_main.ipynb):

    from fsvnet import Config, build_model, make_loaders, train
    cfg = Config().auto_tune()
    ...

Module map
----------
    config          all tunables + hardware auto-tuning
    utils           seeding, logging, device, OOM guard
    datasets_fetch  dataset acquisition guide, Kaggle fetch, layout verification
    synthetic       controlled synthetic clips + the TSC unit test
    video_io        decoding and uniform temporal sampling
    faces           face detection with a stabilised crop trajectory
    manifest        identity-disjoint splits (the anti-leakage machinery)
    dataset         preprocessing cache + torch Dataset + augmentation
    tsc             *** the novel branch ***
    branches        spatial / frequency / SRM branches + temporal encoder
    model           FSV-Net assembly, FSW-Net warm start
    losses          focal loss + branch-diversity regulariser
    train           training loop (AMP, accumulation, video-level selection)
    evaluate        video-level metrics, ablations, figures
    evidence        *** the forensic evidence pack ***
    vlm_reasoner    Qwen2.5-VL / Claude / template reasoning with fallbacks
    inference       single-video end-to-end pipeline

CFX -- the explanation layer (see docs/CFX.md)
----------------------------------------------
    calibrate       temperature/Platt scaling, adaptive ECE, Brier decomposition
    uncertainty     window spread, TTA, MC dropout
    selective       conformal abstention, risk-coverage, the coverage assertion
    attribution     exact branch Shapley, IG, KernelSHAP, counterfactuals
    faithfulness    deletion/insertion/comprehensiveness + the random control
    arbitrate       *** which evidence is solid in which model ***
    narrate         the technical/layman registers and their guard-rails
    cfx             *** CFXPipeline -> ExplanationCard ***
"""

__version__ = "1.0.0"

from .config import Config, DataConfig, ModelConfig, PathConfig, TrainConfig, VLMConfig
from .utils import log, seed_everything

__all__ = ["Config", "PathConfig", "DataConfig", "ModelConfig", "TrainConfig",
           "VLMConfig", "seed_everything", "log", "__version__"]


def __getattr__(name: str):
    """
    Lazy re-exports so `import fsvnet` stays cheap and does not pull torch in
    on a machine that only needs the manifest tools.
    """
    lazy = {
        "build_model": ("fsvnet.model", "build_model"),
        "FSVNet": ("fsvnet.model", "FSVNet"),
        "make_loaders": ("fsvnet.dataset", "make_loaders"),
        "preprocess_dataset": ("fsvnet.dataset", "preprocess_dataset"),
        "train": ("fsvnet.train", "train"),
        "evaluate_model": ("fsvnet.evaluate", "evaluate_model"),
        "run_ablation": ("fsvnet.evaluate", "run_ablation"),
        "build_manifest": ("fsvnet.manifest", "build_manifest"),
        "make_splits": ("fsvnet.manifest", "make_splits"),
        "VideoInferencePipeline": ("fsvnet.inference", "VideoInferencePipeline"),
        "EvidenceReasoner": ("fsvnet.vlm_reasoner", "EvidenceReasoner"),
        # ---- CFX (docs/CFX.md) ----
        "CFXPipeline": ("fsvnet.cfx", "CFXPipeline"),
        "CFXConfig": ("fsvnet.cfx", "CFXConfig"),
        "ExplanationCard": ("fsvnet.cfx", "ExplanationCard"),
        "TemperatureScaler": ("fsvnet.calibrate", "TemperatureScaler"),
        "calibration_report": ("fsvnet.calibrate", "calibration_report"),
        "fit_abstention": ("fsvnet.selective", "fit_abstention"),
        "AbstentionPolicy": ("fsvnet.selective", "AbstentionPolicy"),
        "ReliabilityMatrix": ("fsvnet.arbitrate", "ReliabilityMatrix"),
        "ConfidenceStack": ("fsvnet.arbitrate", "ConfidenceStack"),
        "branch_shapley_exact": ("fsvnet.attribution", "branch_shapley_exact"),
        "Lexicon": ("fsvnet.narrate", "Lexicon"),
    }
    if name in lazy:
        import importlib
        mod, attr = lazy[name]
        return getattr(importlib.import_module(mod), attr)
    raise AttributeError(f"module 'fsvnet' has no attribute '{name}'")
