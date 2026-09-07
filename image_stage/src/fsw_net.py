"""
FSW-Net: Frequency-Spatial-Watermark Network for AI-Generated Image Detection
==============================================================================

Novel tri-branch architecture that detects AI-generated images by jointly
analyzing three complementary signal domains:

  Branch 1 — Spatial (EfficientNet-B0):
      Captures high-level semantic and texture features.

  Branch 2 — Frequency (2-D FFT log-magnitude):
      Exposes GAN upsampling checkerboard artifacts and diffusion-model
      high-frequency noise patterns that are invisible in the spatial domain.
      Also computes a 1-D radially-averaged power spectrum (azimuthal mean)
      as a compact spectral fingerprint.

  Branch 3 — Watermark / Artifact (SRM residuals):
      Applies six fixed Steganalysis Rich Model (SRM) high-pass kernels that
      suppress image content and amplify the microscopic residual traces that
      generative models leave behind — acting as an invisible-watermark detector.

Fusion — Learnable Branch-Attention:
      A lightweight attention gate computes per-image softmax weights over the
      three branch embeddings, so the network learns which signal to trust for
      each sample rather than treating branches equally.

References:
  - Fridrich & Kodovsky, "Rich Models for Steganalysis of Digital Images," 2012
  - Corvi et al., "Detection of Diffusion-Model-Generated Images," 2022
  - Frank et al., "Leveraging Frequency Analysis for Deep Fake Image Recognition," 2020
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


# ---------------------------------------------------------------------------
# Branch 2 — Frequency
# ---------------------------------------------------------------------------

class FrequencyBranch(nn.Module):
    """
    2-D FFT log-magnitude spectrum analysis.

    AI-generated images show three characteristic frequency signatures:
      * GAN up-convolutions → periodic checkerboard artifacts at f = N/4
      * Diffusion denoisers → elevated high-frequency noise floor
      * Nearest/bilinear upsampling → spectral replication peaks

    In addition, a radially-averaged power spectrum (azimuthal mean) is
    computed and concatenated as a 1-D spectral fingerprint — real camera
    images follow a 1/f^2 roll-off, while generated images deviate.
    """

    AZIMUTHAL_BINS = 64  # number of radial bins for the 1-D spectrum

    def __init__(self, out_features: int = 256, img_size: int = 224):
        super().__init__()
        self.img_size = img_size

        # Lightweight CNN operating on the 3-channel log-magnitude map
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                                 # 112
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                                 # 56
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                                 # 28
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),                                         # (B, 256, 1, 1)
        )

        # MLP that processes the azimuthal power spectrum
        self.spec_fc = nn.Sequential(
            nn.Linear(self.AZIMUTHAL_BINS, 64),
            nn.ReLU(inplace=True),
        )

        self.out_fc = nn.Linear(256 + 64, out_features)
        self.dropout = nn.Dropout(0.3)

    # ------------------------------------------------------------------
    def _fft_log_magnitude(self, x: torch.Tensor) -> torch.Tensor:
        """Return the log-magnitude of the 2-D FFT, zero-centred."""
        # x: (B, C, H, W)  — values after ImageNet normalisation
        fft = torch.fft.rfft2(x, norm="ortho")          # (B, C, H, W//2+1)
        mag = torch.abs(fft)
        log_mag = torch.log1p(mag)
        # Resize back to H×W so the CNN sees the full spatial resolution
        log_mag = F.interpolate(log_mag, size=(self.img_size, self.img_size),
                                mode="bilinear", align_corners=False)
        return log_mag                                   # (B, 3, H, W)

    def _azimuthal_mean(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the radially-averaged (azimuthal mean) power spectrum.
        Returns a (B, AZIMUTHAL_BINS) descriptor.
        """
        B, C, H, W = x.shape
        fft = torch.fft.fft2(x, norm="ortho")           # complex (B, C, H, W)
        power = (torch.abs(fft) ** 2).mean(dim=1)       # average over channels → (B, H, W)

        # Build radius grid (same device/dtype as x)
        fy = torch.fft.fftfreq(H, device=x.device).unsqueeze(1).expand(H, W)
        fx = torch.fft.fftfreq(W, device=x.device).unsqueeze(0).expand(H, W)
        radius = torch.sqrt(fx ** 2 + fy ** 2)          # (H, W), values in [0, ~0.7]

        max_r = radius.max()
        bins = self.AZIMUTHAL_BINS
        bin_edges = torch.linspace(0, max_r.item(), bins + 1, device=x.device)

        azimuthal = torch.zeros(B, bins, device=x.device, dtype=x.dtype)
        for k in range(bins):
            mask = (radius >= bin_edges[k]) & (radius < bin_edges[k + 1])
            if mask.sum() > 0:
                azimuthal[:, k] = power[:, mask].mean(dim=1)

        # Log-compress
        azimuthal = torch.log1p(azimuthal)
        return azimuthal                                 # (B, AZIMUTHAL_BINS)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        log_mag = self._fft_log_magnitude(x)            # (B, 3, H, W)
        cnn_feat = self.cnn(log_mag).flatten(1)         # (B, 256)

        azimuthal = self._azimuthal_mean(x)             # (B, AZIMUTHAL_BINS)
        spec_feat = self.spec_fc(azimuthal)             # (B, 64)

        combined = torch.cat([cnn_feat, spec_feat], dim=1)  # (B, 320)
        combined = self.dropout(combined)
        return self.out_fc(combined)                    # (B, out_features)


# ---------------------------------------------------------------------------
# Branch 3 — Watermark / Artifact (SRM)
# ---------------------------------------------------------------------------

class SRMWatermarkBranch(nn.Module):
    """
    Steganalysis Rich Model (SRM) residual analysis.

    Six fixed high-pass kernels suppress visible image content and amplify
    the microscopic noise residuals that generative models leave behind.
    These residuals act as invisible "watermarks":

      * Horizontal / vertical / diagonal 2nd-order difference kernels
        catch ringing artefacts from GAN upsampling and diffusion sampling.
      * 3×3 local average residual detects smoothing introduced by
        neural post-processing (super-resolution, inpainting).
      * Laplacian catches sharp discontinuities at patch/block boundaries.

    The SRM weights are frozen — only the downstream CNN is trained.
    This design keeps the forensic filters interpretable.
    """

    def __init__(self, out_features: int = 256):
        super().__init__()

        srm_kernels = self._build_srm_kernels()         # (K, 1, 5, 5)
        K = srm_kernels.shape[0]

        # grouped conv: each input channel gets its own copy of all K filters
        # → output channels = 3 * K
        self.srm_conv = nn.Conv2d(
            in_channels=3,
            out_channels=3 * K,
            kernel_size=5,
            padding=2,
            bias=False,
            groups=3,
        )
        with torch.no_grad():
            # shape needed: (3*K, 1, 5, 5)  (groups=3 splits input into 3 groups)
            weight = srm_kernels.repeat(3, 1, 1, 1)    # (3*K, 1, 5, 5)
            self.srm_conv.weight.copy_(weight)
        for p in self.srm_conv.parameters():
            p.requires_grad = False                     # freeze forensic filters

        in_ch = 3 * K
        self.cnn = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.out_fc = nn.Linear(256, out_features)
        self.dropout = nn.Dropout(0.3)

    # ------------------------------------------------------------------
    @staticmethod
    def _build_srm_kernels() -> torch.Tensor:
        """Six SRM-inspired 5×5 forensic high-pass kernels."""
        def z():
            return torch.zeros(5, 5)

        # 1. Horizontal 2nd-order difference
        k1 = z(); k1[2, 1], k1[2, 2], k1[2, 3] = -0.5, 1.0, -0.5

        # 2. Vertical 2nd-order difference
        k2 = z(); k2[1, 2], k2[2, 2], k2[3, 2] = -0.5, 1.0, -0.5

        # 3. Main-diagonal 2nd-order difference
        k3 = z(); k3[1, 1], k3[2, 2], k3[3, 3] = -0.5, 1.0, -0.5

        # 4. Anti-diagonal 2nd-order difference
        k4 = z(); k4[1, 3], k4[2, 2], k4[3, 1] = -0.5, 1.0, -0.5

        # 5. 3×3 local-average residual (detects smoothing)
        k5 = z(); k5[1:4, 1:4] = -1.0 / 8
        k5[2, 2] = 1.0

        # 6. Laplacian (edge / boundary artifact detector)
        k6 = z()
        k6[2, 1], k6[2, 3], k6[1, 2], k6[3, 2] = -1, -1, -1, -1
        k6[2, 2] = 4.0

        kernels = torch.stack([k1, k2, k3, k4, k5, k6])  # (6, 5, 5)
        return kernels.unsqueeze(1)                        # (6, 1, 5, 5)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residuals = self.srm_conv(x)               # (B, 18, H, W)
        residuals = torch.tanh(residuals * 3.0)    # bound residuals ∈ (−1, 1)
        feat = self.cnn(residuals).flatten(1)      # (B, 256)
        feat = self.dropout(feat)
        return self.out_fc(feat)                   # (B, out_features)

    # ------------------------------------------------------------------
    def get_residual_maps(self, x: torch.Tensor) -> torch.Tensor:
        """Return the raw SRM residual maps for visualisation."""
        with torch.no_grad():
            return self.srm_conv(x)                # (B, 18, H, W)


# ---------------------------------------------------------------------------
# Attention Fusion
# ---------------------------------------------------------------------------

class BranchAttentionFusion(nn.Module):
    """
    Learnable soft-attention over three branch embeddings.

    A small MLP reads the concatenated embeddings and outputs three
    softmax-normalised weights — one per branch.  Each branch embedding is
    scaled by its weight, so the network learns which forensic signal to
    trust on a per-image basis.

    The weights are also returned for interpretability.
    """

    def __init__(self, branch_dim: int = 256, num_branches: int = 3):
        super().__init__()
        total = branch_dim * num_branches
        self.gate = nn.Sequential(
            nn.Linear(total, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_branches),
            nn.Softmax(dim=1),
        )
        self.num_branches = num_branches

    def forward(self, *branches) -> tuple:
        """
        Args:
            *branches: num_branches tensors of shape (B, branch_dim)
        Returns:
            fused   : (B, branch_dim * num_branches)
            weights : (B, num_branches) — attention weights per branch
        """
        cat = torch.cat(branches, dim=1)           # (B, total)
        weights = self.gate(cat)                   # (B, num_branches)

        weighted = [feat * weights[:, i:i+1] for i, feat in enumerate(branches)]
        fused = torch.cat(weighted, dim=1)         # (B, total)
        return fused, weights


# ---------------------------------------------------------------------------
# FSW-Net
# ---------------------------------------------------------------------------

class FSWNet(nn.Module):
    """
    FSW-Net — Frequency-Spatial-Watermark Network.

    Parameters
    ----------
    num_classes : int
        Number of output classes (default 4: ai_generated, deepfake, edited, real).
    branch_dim : int
        Embedding size produced by each branch before fusion.
    dropout : float
        Dropout probability in the classifier head.
    freeze_spatial_backbone : bool
        If True, EfficientNet-B0 feature layers are frozen for the first
        few epochs and then unfrozen via unfreeze_spatial_backbone().
    """

    def __init__(
        self,
        num_classes: int = 4,
        branch_dim: int = 256,
        dropout: float = 0.4,
        freeze_spatial_backbone: bool = False,
    ):
        super().__init__()

        # ------------------------------------------------------------------
        # Branch 1 — Spatial (EfficientNet-B0, pretrained)
        # ------------------------------------------------------------------
        eff = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        self.spatial_features = eff.features       # (B, 1280, 7, 7)
        self.spatial_pool = nn.AdaptiveAvgPool2d(1)
        self.spatial_proj = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(1280, branch_dim),
        )

        if freeze_spatial_backbone:
            for p in self.spatial_features.parameters():
                p.requires_grad = False

        # ------------------------------------------------------------------
        # Branch 2 — Frequency
        # ------------------------------------------------------------------
        self.freq_branch = FrequencyBranch(out_features=branch_dim)

        # ------------------------------------------------------------------
        # Branch 3 — Watermark / Artifact (SRM)
        # ------------------------------------------------------------------
        self.srm_branch = SRMWatermarkBranch(out_features=branch_dim)

        # ------------------------------------------------------------------
        # Attention Fusion
        # ------------------------------------------------------------------
        self.fusion = BranchAttentionFusion(branch_dim=branch_dim, num_branches=3)

        # ------------------------------------------------------------------
        # Classifier head
        # ------------------------------------------------------------------
        total_dim = branch_dim * 3
        self.classifier = nn.Sequential(
            nn.LayerNorm(total_dim),
            nn.Dropout(dropout),
            nn.Linear(total_dim, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

        self.num_classes = num_classes
        self.branch_dim = branch_dim

    # ------------------------------------------------------------------
    def unfreeze_spatial_backbone(self):
        """Unfreeze the EfficientNet backbone for fine-tuning."""
        for p in self.spatial_features.parameters():
            p.requires_grad = True

    # ------------------------------------------------------------------
    def _spatial_embed(self, x: torch.Tensor) -> torch.Tensor:
        f = self.spatial_features(x)               # (B, 1280, 7, 7)
        f = self.spatial_pool(f).flatten(1)        # (B, 1280)
        return self.spatial_proj(f)                # (B, branch_dim)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor):
        """
        Returns
        -------
        logits  : (B, num_classes)
        weights : (B, 3) — branch attention weights [spatial, freq, srm]
        """
        spatial_feat = self._spatial_embed(x)
        freq_feat    = self.freq_branch(x)
        srm_feat     = self.srm_branch(x)

        fused, weights = self.fusion(spatial_feat, freq_feat, srm_feat)
        logits = self.classifier(fused)
        return logits, weights

    # ------------------------------------------------------------------
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience wrapper — returns only class logits."""
        logits, _ = self.forward(x)
        return logits

    # ------------------------------------------------------------------
    def get_branch_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Return branch attention weights for a batch (for analysis)."""
        with torch.no_grad():
            _, weights = self.forward(x)
        return weights                             # (B, 3)

    # ------------------------------------------------------------------
    def get_srm_residuals(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw SRM residual maps for a batch (for visualisation)."""
        return self.srm_branch.get_residual_maps(x)


# ---------------------------------------------------------------------------
# Focal Loss
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """
    Focal Loss with optional per-class weighting.
    Reduces the relative loss for well-classified examples and focuses
    training on hard, misclassified ones.
    """

    def __init__(self, alpha=None, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(weight=alpha, reduction="none")

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = self.ce(inputs, targets)         # (B,)
        pt = torch.exp(-ce_loss)
        focal = ((1 - pt) ** self.gamma) * ce_loss
        return focal.mean()


# ---------------------------------------------------------------------------
# Quick sanity-check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = FSWNet(num_classes=4, branch_dim=256).to(device)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params   : {total:,}")
    print(f"Trainable params: {trainable:,}")

    dummy = torch.randn(4, 3, 224, 224).to(device)
    logits, weights = model(dummy)
    print(f"Logits shape   : {logits.shape}")    # (4, 4)
    print(f"Weights shape  : {weights.shape}")   # (4, 3)
    print(f"Branch weights (mean): {weights.mean(0).detach().cpu().numpy()}")
    print("Sanity check passed.")
