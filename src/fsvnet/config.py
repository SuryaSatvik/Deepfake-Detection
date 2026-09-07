"""
config.py
=========

Single source of truth for every tunable in the FSV-Net video pipeline.

Everything is a plain dataclass so the notebook can override any field in one
line and print the whole config for the report:

    from fsvnet.config import Config
    cfg = Config()
    cfg.train.epochs = 20
    print(cfg)

Hardware policy (this project targets an RTX 4060 8 GB with a CPU safety net):
  * `Config.auto_tune()` inspects the machine and rewrites the memory-sensitive
    fields (clip length, batch size, AMP, VLM size) to something that will
    actually fit.  It never silently upgrades past what the card can hold.
  * Every module calls `cfg.device` rather than hard-coding "cuda", so the same
    code runs on CPU (slowly, but it runs).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

# .../video_detector/src/fsvnet/config.py  ->  .../video_detector
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class PathConfig:
    project_root: Path = PROJECT_ROOT

    # Where the raw downloaded datasets live (you point these at your disk).
    # Each entry is a dataset root; see fsvnet.manifest for the layout each
    # adapter expects.
    celebdf_root: Optional[Path] = None      # .../Celeb-DF-v2 (raw video distribution)
    # The pre-extracted frames+landmarks distribution this project actually has.
    # If set, celebdf_frames.py is used and video_io/faces are bypassed.
    celebdf_frames_root: Optional[Path] = PROJECT_ROOT / "Celeb-DF-v2"
    lmf_cache: Path = PROJECT_ROOT / "data" / "processed" / "lmf_cache.npz"
    lmf_template: Path = PROJECT_ROOT / "data" / "manifests" / "lmf_template.npy"
    ffpp_root: Optional[Path] = None         # .../FaceForensics++
    dfdc_root: Optional[Path] = None         # .../dfdc_preview
    custom_root: Optional[Path] = None       # any folder with real/ and fake/

    # Everything the pipeline generates
    data_dir: Path = PROJECT_ROOT / "data"
    clip_cache: Path = PROJECT_ROOT / "data" / "clips"          # cached face crops
    manifest_dir: Path = PROJECT_ROOT / "data" / "manifests"
    ckpt_dir: Path = PROJECT_ROOT / "checkpoints"
    out_dir: Path = PROJECT_ROOT / "outputs"
    plot_dir: Path = PROJECT_ROOT / "outputs" / "plots"
    pred_dir: Path = PROJECT_ROOT / "outputs" / "predictions"
    evidence_dir: Path = PROJECT_ROOT / "outputs" / "evidence"

    def mkdirs(self) -> None:
        for p in [self.data_dir, self.clip_cache, self.manifest_dir, self.ckpt_dir,
                  self.out_dir, self.plot_dir, self.pred_dir, self.evidence_dir]:
            Path(p).mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# Data / preprocessing
# --------------------------------------------------------------------------

@dataclass
class DataConfig:
    # ---- clip geometry -------------------------------------------------
    num_frames: int = 16          # T — frames per clip (temporal window)
    frame_stride: int = 4         # sample every Nth decoded frame
    img_size: int = 224           # face crop resolution fed to the network
    face_margin: float = 0.30     # expand the detected box by this fraction

    # ---- clip sampling -------------------------------------------------
    clips_per_video_train: int = 1    # random temporal crop each epoch
    clips_per_video_eval: int = 2     # fixed, evenly spaced, then averaged
    min_faces_per_clip: float = 0.6   # reject a clip if fewer than 60% of its
                                      # frames had a usable face

    # ---- splits --------------------------------------------------------
    # IDENTITY-DISJOINT by default. This is not a stylistic choice: the
    # previous image project measured 0.80 AUC on a random split and 0.57 on a
    # genuinely held-out one, because 17.7% of its test rows had a near
    # duplicate in train. Video datasets are far worse — the same person
    # appears in dozens of clips. Never set this to "random" for a reported
    # number.
    split_by: str = "identity"        # "identity" | "video" | "random"
    train_frac: float = 0.70
    val_frac: float = 0.15
    test_frac: float = 0.15
    seed: int = 42

    # ---- compression augmentation --------------------------------------
    # The JPEG-shortcut trap: if reals and fakes carry different compression
    # histories, the model learns the codec, not the generator. These
    # augmentations randomise that history so it cannot be used as a shortcut.
    aug_jpeg_quality: tuple = (40, 95)
    aug_downscale: tuple = (0.5, 1.0)
    aug_prob: float = 0.5
    aug_hflip: bool = True

    # ---- face detection -------------------------------------------------
    face_backend: str = "auto"    # "auto" | "mediapipe" | "opencv" | "none"
    detect_every: int = 8         # re-detect every N frames, track in between


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

@dataclass
class ModelConfig:
    num_classes: int = 2                 # real / fake at video level
    branch_dim: int = 256
    azimuthal_bins: int = 64             # radial bins in the power spectrum
    dropout: float = 0.4

    # temporal encoder over per-frame embeddings
    temporal_layers: int = 2
    temporal_heads: int = 4
    temporal_ff: int = 512

    # which branches are active — this is the ablation switchboard
    use_spatial: bool = True
    use_frequency: bool = True
    use_srm: bool = True
    use_tsc: bool = True                 # <-- novel branch 1 (frequency/time)
    use_lmf: bool = True                 # <-- novel branch 2 (geometry/space)

    spatial_backbone: str = "efficientnet_b0"
    pretrained: bool = True
    freeze_spatial_epochs: int = 1       # warm up the new branches first

    # Optional: initialise the frame-level branches from the image-level
    # FSW-Net checkpoint in ../DeepfakeProject/notebooks/best_fsw_net.pth
    init_from_fswnet: Optional[str] = None


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------

@dataclass
class TrainConfig:
    epochs: int = 12
    batch_size: int = 2                  # clips per step (8 GB card)
    grad_accum: int = 8                  # -> effective batch 16
    lr: float = 3e-4
    backbone_lr: float = 3e-5            # lower LR for pretrained EfficientNet
    weight_decay: float = 1e-4
    warmup_epochs: int = 1
    amp: bool = True                     # fp16 autocast on CUDA
    grad_checkpointing: bool = False     # turn on if you still OOM
    num_workers: int = 4
    focal_gamma: float = 2.0
    label_smoothing: float = 0.05
    early_stop_patience: int = 2
    # An epoch counts as "improved" only if it beats the best by more than
    # this. Without it, a +0.0002 wobble resets the patience counter and early
    # stopping never fires on a noisy validation metric. The BEST checkpoint is
    # still saved on any improvement, however small -- min_delta governs
    # patience only, never what gets kept.
    early_stop_min_delta: float = 0.001
    # Wall-clock cap for one training run, in seconds. 0 = unlimited.
    # Exists so an overnight run cannot overrun a deadline: training stops at
    # the end of the epoch that crosses the budget, and the best checkpoint so
    # far is already on disk. An interrupted run is still a usable result; a
    # run you had to kill halfway is not.
    max_seconds: float = 0.0
    monitor: str = "val_auc"
    max_grad_norm: float = 1.0


# --------------------------------------------------------------------------
# VLM reasoning layer
# --------------------------------------------------------------------------

@dataclass
class VLMConfig:
    """
    Skyra-style grounded artifact reasoning.

    NOTE ON HONESTY: Skyra (arXiv 2512.15693) has no released weights. This is
    a Skyra-*style* module built on base Qwen2.5-VL — same idea (grounded
    artifact reasoning instead of a bare probability), not the same model.
    Describe it that way in the report.
    """
    enabled: bool = True
    model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    load_in_4bit: bool = True            # ~3.5 GB on an 8 GB card
    max_new_tokens: int = 512
    num_evidence_frames: int = 4         # frames shown to the VLM
    temperature: float = 0.2

    # Fallback ladder, tried in order, so this never hard-fails:
    #   qwen_4bit -> qwen_fp16 -> qwen_cpu -> anthropic_api -> template
    fallback_chain: List[str] = field(default_factory=lambda: [
        "qwen_4bit", "qwen_fp16", "qwen_cpu", "anthropic_api", "template"
    ])
    anthropic_model: str = "claude-sonnet-4-5"
    anthropic_api_key_env: str = "ANTHROPIC_API_KEY"


# --------------------------------------------------------------------------
# Top-level
# --------------------------------------------------------------------------

@dataclass
class Config:
    paths: PathConfig = field(default_factory=PathConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    vlm: VLMConfig = field(default_factory=VLMConfig)

    device: str = "cuda"
    run_name: str = "fsvnet_v1"

    # ----------------------------------------------------------------
    def auto_tune(self, verbose: bool = True) -> "Config":
        """
        Inspect the machine and rewrite memory-sensitive fields so the run
        actually fits. Called by the notebook before anything is built.
        """
        try:
            import torch
        except ImportError:
            if verbose:
                print("[config] torch not importable — leaving defaults, CPU assumed.")
            self.device = "cpu"
            return self

        if not torch.cuda.is_available():
            self.device = "cpu"
            self.train.amp = False
            self.train.batch_size = 1
            self.train.grad_accum = 4
            self.data.num_frames = 8
            self.train.num_workers = 2
            self.vlm.load_in_4bit = False
            if verbose:
                print("[config] No CUDA device. CPU safety net engaged:")
                print("         frames=8, batch=1, amp=off. Training will be slow;")
                print("         use this path to validate the pipeline, not to train.")
            return self

        self.device = "cuda"
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3

        if vram < 6:
            self.data.num_frames = 8
            self.train.batch_size = 1
            self.train.grad_accum = 16
            self.train.grad_checkpointing = True
            self.vlm.model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
        elif vram < 10:                      # RTX 4060 8 GB lands here
            self.data.num_frames = 16
            self.train.batch_size = 2
            self.train.grad_accum = 8
            self.vlm.model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
            self.vlm.load_in_4bit = True
        elif vram < 18:
            self.data.num_frames = 24
            self.train.batch_size = 4
            self.train.grad_accum = 4
            self.vlm.model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
            self.vlm.load_in_4bit = True
        else:
            self.data.num_frames = 32
            self.train.batch_size = 8
            self.train.grad_accum = 2
            self.vlm.model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
            self.vlm.load_in_4bit = False

        self.train.amp = True
        if verbose:
            print(f"[config] GPU: {name} ({vram:.1f} GB)")
            print(f"[config] frames={self.data.num_frames} batch={self.train.batch_size} "
                  f"accum={self.train.grad_accum} amp={self.train.amp}")
            print(f"[config] VLM: {self.vlm.model_id} (4bit={self.vlm.load_in_4bit})")
            print( "[config] If you still hit CUDA OOM: set cfg.train.grad_checkpointing=True,")
            print( "         then cfg.data.num_frames=8, then cfg.train.batch_size=1.")
        return self

    # ----------------------------------------------------------------
    def active_branches(self) -> List[str]:
        names = []
        if self.model.use_spatial:   names.append("spatial")
        if self.model.use_frequency: names.append("frequency")
        if self.model.use_srm:       names.append("srm")
        if self.model.use_tsc:       names.append("tsc")
        return names

    def to_dict(self) -> dict:
        def conv(o):
            if isinstance(o, Path):
                return str(o)
            if isinstance(o, tuple):
                return list(o)
            return o
        return json.loads(json.dumps(asdict(self), default=conv))

    def save(self, path: os.PathLike) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    def __str__(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


__all__ = ["Config", "PathConfig", "DataConfig", "ModelConfig",
           "TrainConfig", "VLMConfig", "PROJECT_ROOT"]
