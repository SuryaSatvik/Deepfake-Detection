"""
branches.py
===========

The three per-frame forensic branches, carried over from the image-level
FSW-Net (`../DeepfakeProject/src/fsw_net.py`) and adapted for clips, plus the
temporal encoder that turns per-frame embeddings into one clip embedding.

What changed relative to the image version, and why
---------------------------------------------------
* **Frame-batched execution.** A clip of shape (B, T, 3, H, W) is folded to
  (B*T, 3, H, W) for the per-frame branches and unfolded afterwards. One
  kernel launch instead of T, and the EfficientNet BatchNorm sees B*T samples
  instead of B, which stabilises training at batch_size=2 on an 8 GB card.
* **Precomputed radial index** in the frequency branch (see tsc.py) — the
  image version looped over 64 boolean masks per forward pass, which is fine
  once per image and painful T times per clip.
* **rfft2 replaced by fft2 for the azimuthal path.** rfft2 returns a half
  spectrum whose radial geometry is not symmetric, so binning it by radius
  double-counts one half of the plane. The image version's CNN path used
  rfft2 + interpolate, which is a *feature map*, not a spectrum, so it was
  unaffected — but the azimuthal descriptor genuinely needs the full plane.
* **Temporal encoder** — new. Per-frame embeddings go through a small
  Transformer with learned positional encoding and attention pooling, so the
  clip embedding can weight informative frames instead of averaging blindly.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tsc import azimuthal_power, build_radial_index, _fp32_autocast


# ==========================================================================
# Branch 1 — Spatial
# ==========================================================================

class SpatialBranch(nn.Module):
    """EfficientNet-B0 (ImageNet) per frame -> (B, T, out_features)."""

    def __init__(self, out_features: int = 256, pretrained: bool = True,
                 backbone: str = "efficientnet_b0"):
        super().__init__()
        from torchvision import models

        if backbone == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            net = models.efficientnet_b0(weights=weights)
            self.features = net.features
            feat_dim = 1280
        elif backbone == "efficientnet_b3":
            weights = models.EfficientNet_B3_Weights.DEFAULT if pretrained else None
            net = models.efficientnet_b3(weights=weights)
            self.features = net.features
            feat_dim = 1536
        elif backbone == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            net = models.resnet50(weights=weights)
            self.features = nn.Sequential(*list(net.children())[:-2])
            feat_dim = 2048
        else:
            raise ValueError(f"unsupported backbone: {backbone}")

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(nn.Dropout(0.3), nn.Linear(feat_dim, out_features))
        self.feat_dim = feat_dim
        self._ckpt = False

    def set_grad_checkpointing(self, enabled: bool = True) -> None:
        self._ckpt = enabled

    def set_frozen(self, frozen: bool) -> None:
        for p in self.features.parameters():
            p.requires_grad = not frozen

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """frames: (N, 3, H, W) -> (N, out_features)"""
        if self._ckpt and self.training:
            from torch.utils.checkpoint import checkpoint
            f = checkpoint(self.features, frames, use_reentrant=False)
        else:
            f = self.features(frames)
        f = self.pool(f).flatten(1)
        return self.proj(f)


# ==========================================================================
# Branch 2 — Frequency (per frame)
# ==========================================================================

class FrequencyBranch(nn.Module):
    """
    Per-frame 2-D FFT log-magnitude CNN + azimuthal power spectrum MLP.

    Detects the *static* spectral signature: GAN up-convolution checkerboards,
    diffusion high-frequency noise floors, upsampling replication peaks.
    TSC handles the temporal side; this is the frame-level counterpart and the
    two are deliberately kept separate so the ablation can attribute credit.
    """

    def __init__(self, out_features: int = 256, n_bins: int = 64,
                 img_size: int = 224):
        super().__init__()
        self.n_bins = n_bins
        self.img_size = img_size

        idx, counts = build_radial_index(img_size, img_size, n_bins)
        self.register_buffer("radial_idx", idx, persistent=False)
        self.register_buffer("radial_counts", counts, persistent=False)

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.spec_fc = nn.Sequential(nn.Linear(n_bins, 64), nn.ReLU(inplace=True))
        self.out_fc = nn.Linear(256 + 64, out_features)
        self.dropout = nn.Dropout(0.3)

    def _log_magnitude(self, x: torch.Tensor) -> torch.Tensor:
        with _fp32_autocast():
            fft = torch.fft.fftshift(torch.fft.fft2(x.float(), norm="ortho"),
                                     dim=(-2, -1))
            mag = torch.log1p(torch.abs(fft))
        return mag.to(x.dtype)

    def forward(self, frames: torch.Tensor,
                spectra: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        frames  : (N, 3, H, W)
        spectra : (N, n_bins) precomputed azimuthal spectra. The TSC branch
                  already computes these for the whole clip; passing them in
                  avoids doing every FFT twice.
        """
        H, W = frames.shape[-2:]
        if (H, W) != (self.img_size, self.img_size) and spectra is None:
            raise ValueError(
                f"frequency branch was built for {self.img_size}x{self.img_size} "
                f"but got {H}x{W}; inference resolution must match training.")
        mag = self._log_magnitude(frames)
        cnn_feat = self.cnn(mag).flatten(1)                       # (N, 256)

        if spectra is None:
            with _fp32_autocast():
                spectra = azimuthal_power(frames.float(), self.radial_idx,
                                          self.radial_counts, self.n_bins)
        spec_feat = self.spec_fc(spectra.to(cnn_feat.dtype))      # (N, 64)

        return self.out_fc(self.dropout(torch.cat([cnn_feat, spec_feat], 1)))


# ==========================================================================
# Branch 3 — SRM residual
# ==========================================================================

class SRMBranch(nn.Module):
    """
    Six frozen Steganalysis-Rich-Model high-pass kernels suppress image content
    and expose the generator's noise residual. Identical kernel set to the
    image-level FSW-Net so a checkpoint can be transferred; the CNN on top is
    the same shape, which is what makes `init_from_fswnet` possible.
    """

    def __init__(self, out_features: int = 256):
        super().__init__()
        kern = self._build_kernels()                              # (6, 1, 5, 5)
        K = kern.shape[0]
        self.srm_conv = nn.Conv2d(3, 3 * K, 5, padding=2, bias=False, groups=3)
        with torch.no_grad():
            self.srm_conv.weight.copy_(kern.repeat(3, 1, 1, 1))
        for p in self.srm_conv.parameters():
            p.requires_grad = False

        self.cnn = nn.Sequential(
            nn.Conv2d(3 * K, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.out_fc = nn.Linear(256, out_features)
        self.dropout = nn.Dropout(0.3)

    @staticmethod
    def _build_kernels() -> torch.Tensor:
        def z():
            return torch.zeros(5, 5)
        k1 = z(); k1[2, 1], k1[2, 2], k1[2, 3] = -0.5, 1.0, -0.5      # horizontal
        k2 = z(); k2[1, 2], k2[2, 2], k2[3, 2] = -0.5, 1.0, -0.5      # vertical
        k3 = z(); k3[1, 1], k3[2, 2], k3[3, 3] = -0.5, 1.0, -0.5      # diagonal
        k4 = z(); k4[1, 3], k4[2, 2], k4[3, 1] = -0.5, 1.0, -0.5      # anti-diag
        k5 = z(); k5[1:4, 1:4] = -1.0 / 8; k5[2, 2] = 1.0             # avg residual
        k6 = z(); k6[2, 1] = k6[2, 3] = k6[1, 2] = k6[3, 2] = -1.0; k6[2, 2] = 4.0
        return torch.stack([k1, k2, k3, k4, k5, k6]).unsqueeze(1)

    def residuals(self, frames: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.srm_conv(frames)

    def forward(self, frames: torch.Tensor,
                return_residual_energy: bool = False):
        res = torch.tanh(self.srm_conv(frames) * 3.0)                # (N, 18, H, W)
        feat = self.out_fc(self.dropout(self.cnn(res).flatten(1)))
        if return_residual_energy:
            # per-frame residual energy — its temporal trajectory is one of the
            # signals the evidence pack reports alongside the TSC map
            energy = res.pow(2).mean(dim=(1, 2, 3))                  # (N,)
            return feat, energy
        return feat, None


# ==========================================================================
# Temporal encoder
# ==========================================================================

class TemporalEncoder(nn.Module):
    """
    Small Transformer over the T per-frame embeddings of one branch, followed
    by attention pooling to a single clip embedding.

    Attention pooling (rather than mean) matters for face swaps: the artefact
    is often concentrated in the few frames where the head turns or the mouth
    opens, and a mean over 16 frames dilutes it by 16x.
    """

    def __init__(self, dim: int = 256, layers: int = 2, heads: int = 4,
                 ff: int = 512, dropout: float = 0.1, max_len: int = 64):
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(1, max_len, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=ff, dropout=dropout,
            batch_first=True, norm_first=True, activation="gelu")
        # enable_nested_tensor=False: it is incompatible with norm_first=True
        # and only emits a warning otherwise. norm_first is worth keeping —
        # pre-norm transformers are far more stable at the tiny batch sizes an
        # 8 GB card forces on us.
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers,
                                             enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.attn_pool = nn.Sequential(nn.Linear(dim, dim // 2), nn.Tanh(),
                                       nn.Linear(dim // 2, 1))

    def forward(self, x: torch.Tensor,
                mask: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x    : (B, T, D)
        mask : (B, T) bool, True where the frame is valid
        returns (clip_embedding (B, D), frame_weights (B, T))
        """
        B, T, D = x.shape
        x = x + self.pos[:, :T]
        pad_mask = (~mask) if mask is not None else None
        h = self.encoder(x, src_key_padding_mask=pad_mask)
        h = self.norm(h)

        logits = self.attn_pool(h).squeeze(-1)                       # (B, T)
        if mask is not None:
            logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        w = torch.softmax(logits, dim=1)
        pooled = (h * w.unsqueeze(-1)).sum(dim=1)
        return pooled, w


# ==========================================================================
# Branch attention fusion (clip level)
# ==========================================================================

class BranchAttentionFusion(nn.Module):
    """
    Learnable softmax gate over branch embeddings — the same mechanism as the
    image FSW-Net, generalised to N branches so the ablation can switch any of
    them off without changing the head's shape logic.

    The returned weights are the model's own statement about which forensic
    signal it trusted for this clip, and they go straight into the VLM evidence
    pack.
    """

    def __init__(self, branch_dim: int = 256, num_branches: int = 4,
                 hidden: int = 128):
        super().__init__()
        self.num_branches = num_branches
        self.gate = nn.Sequential(
            nn.Linear(branch_dim * num_branches, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, num_branches))

    def forward(self, *branches) -> Tuple[torch.Tensor, torch.Tensor]:
        cat = torch.cat(branches, dim=1)
        weights = torch.softmax(self.gate(cat), dim=1)
        weighted = [b * weights[:, i:i + 1] for i, b in enumerate(branches)]
        return torch.cat(weighted, dim=1), weights


__all__ = ["SpatialBranch", "FrequencyBranch", "SRMBranch", "TemporalEncoder",
           "BranchAttentionFusion"]
