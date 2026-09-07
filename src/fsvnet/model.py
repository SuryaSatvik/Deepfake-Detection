"""
model.py
========

FSV-Net — Frequency-Spatial-Watermark **Video** Network.

Architecture
------------

    clip (B, T, 3, 224, 224)
      │
      ├─ per-frame ────────────────────────────────────┐
      │   spatial  : EfficientNet-B0  ─┐               │
      │   frequency: FFT CNN + ring MLP ├─ (B,T,256) ──┼─ TemporalEncoder ×3
      │   srm      : frozen SRM + CNN  ─┘               │   (transformer +
      │                                                 │    attention pool)
      └─ clip-level (two orthogonal forensic channels) ─┤
          TSC : temporal spectral coherence  (B,256) ───┤
                frequency domain, image coordinates     │
          LMF : landmark motion-compensated  (B,256) ───┘
                spatial domain, FACE-CANONICAL coords   │
                (cached energy series + descriptors)    │
                                                        │
                                    BranchAttentionFusion (B, 5)
                                                        │
                                                 classifier → logits (B, 2)

Design notes
------------
* Every branch is switchable from the config. `use_tsc=False, use_lmf=False`
  reproduces a conventional video FSW-Net, which is the ablation baseline both
  novel branches are measured against — you need that number in the report.
* TSC and LMF are deliberately orthogonal: TSC asks "does fine detail evolve
  in time the way a camera's does?" in the frequency domain on the raw pixel
  grid; LMF asks "is the frame-to-frame change spatially bound to the face
  geometry?" in the spatial domain after warping head motion away. Because
  they share no operator and no coordinate frame, `tsc_only` / `lmf_only` /
  `both` is a genuine decomposition. Their CPU descriptor probes both landed
  at 0.51-0.59 AUC identity-disjoint (see docs/FINDINGS.md §3) — so this
  ablation is a real open question, not a formality.
* The three per-frame branches each get their **own** temporal encoder rather
  than sharing one. They measure different physics: the spatial branch's
  frames are correlated by content, the SRM branch's by noise. Sharing the
  positional encoding across them mixed the two in early experiments; keeping
  them separate costs ~1 M parameters and is cleaner to interpret.
* `forward(..., return_evidence=True)` returns everything the evidence pack and
  the VLM need, in one pass, so inference never runs the model twice.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .branches import (BranchAttentionFusion, FrequencyBranch, SpatialBranch,
                       SRMBranch, TemporalEncoder)
from .tsc import TSCBranch
from .lmf import LMFBranch, DESCRIPTOR_NAMES as LMF_DESCRIPTOR_NAMES
from .utils import count_params, log


class FSVNet(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        branch_dim: int = 256,
        n_bins: int = 64,
        img_size: int = 224,
        dropout: float = 0.4,
        temporal_layers: int = 2,
        temporal_heads: int = 4,
        temporal_ff: int = 512,
        use_spatial: bool = True,
        use_frequency: bool = True,
        use_srm: bool = True,
        use_tsc: bool = True,
        use_lmf: bool = True,
        pretrained: bool = True,
        backbone: str = "efficientnet_b0",
        max_frames: int = 64,
    ):
        super().__init__()
        if not any([use_spatial, use_frequency, use_srm, use_tsc, use_lmf]):
            raise ValueError("at least one branch must be enabled")

        self.use = dict(spatial=use_spatial, frequency=use_frequency,
                        srm=use_srm, tsc=use_tsc, lmf=use_lmf)
        self.branch_dim = branch_dim
        self.num_classes = num_classes
        self.n_bins = n_bins

        # ---- per-frame branches ----------------------------------------
        if use_spatial:
            self.spatial = SpatialBranch(branch_dim, pretrained, backbone)
            self.temp_spatial = TemporalEncoder(branch_dim, temporal_layers,
                                                temporal_heads, temporal_ff,
                                                max_len=max_frames)
        if use_frequency:
            self.frequency = FrequencyBranch(branch_dim, n_bins, img_size)
            self.temp_freq = TemporalEncoder(branch_dim, temporal_layers,
                                             temporal_heads, temporal_ff,
                                             max_len=max_frames)
        if use_srm:
            self.srm = SRMBranch(branch_dim)
            self.temp_srm = TemporalEncoder(branch_dim, temporal_layers,
                                            temporal_heads, temporal_ff,
                                            max_len=max_frames)
        # ---- clip-level branch ------------------------------------------
        if use_tsc:
            self.tsc = TSCBranch(branch_dim, n_bins, img_size)
        # LMF consumes the CACHED region-energy series + descriptors, not
        # pixels — the warp happens once in preprocessing. So enabling it costs
        # almost no train-time compute, which is what makes the -LMF ablation a
        # fair comparison rather than a compute cut.
        if use_lmf:
            self.lmf = LMFBranch.build(branch_dim,
                                       n_descriptors=len(LMF_DESCRIPTOR_NAMES))

        # ---- fusion + head ----------------------------------------------
        self.branch_names: List[str] = [k for k, v in self.use.items() if v]
        n_active = len(self.branch_names)
        self.fusion = BranchAttentionFusion(branch_dim, n_active)

        total = branch_dim * n_active
        self.classifier = nn.Sequential(
            nn.LayerNorm(total),
            nn.Dropout(dropout),
            nn.Linear(total, 256), nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    # ------------------------------------------------------------------
    def set_backbone_frozen(self, frozen: bool) -> None:
        if self.use["spatial"]:
            self.spatial.set_frozen(frozen)
            log.info(f"spatial backbone {'frozen' if frozen else 'unfrozen'}")

    def set_grad_checkpointing(self, enabled: bool = True) -> None:
        if self.use["spatial"]:
            self.spatial.set_grad_checkpointing(enabled)

    def param_report(self) -> Dict[str, Dict[str, int]]:
        rep = {"TOTAL": count_params(self)}
        for name in self.branch_names:
            mod = getattr(self, name if name != "tsc" else "tsc")
            rep[name] = count_params(mod)
        return rep

    # ------------------------------------------------------------------
    def forward(self, clip: torch.Tensor,
                frame_mask: Optional[torch.Tensor] = None,
                return_evidence: bool = False,
                ablate_branches: Optional[List[str]] = None,
                lmf: Optional[Dict[str, torch.Tensor]] = None):
        """
        clip            : (B, T, 3, H, W) ImageNet-normalised
        frame_mask      : (B, T) bool, True where the frame had a real face
        ablate_branches : zero these branches' embeddings before fusion.
        lmf             : {"energy": (B, T-1, 3), "descriptors": (B, D)} from
                          the cached LMF pass. Required when use_lmf=True;
                          None substitutes zeros and sets info["lmf_missing"].

        `ablate_branches` is what makes honest attribution possible. The gate
        weights say what the model *attended to*; they do not say what the
        decision *depended on* — a branch can carry a large gate weight and
        still be irrelevant to the logit. Zeroing it and re-reading the logit
        measures the dependence directly. See explain.py.

        returns : logits (B, num_classes), info dict
        """
        ablate = set(ablate_branches or [])
        B, T = clip.shape[:2]
        flat = clip.reshape(B * T, *clip.shape[2:])

        evidence: Dict[str, torch.Tensor] = {}
        embeddings: List[torch.Tensor] = []
        frame_weights: Dict[str, torch.Tensor] = {}

        # ---- TSC first: it produces the azimuthal spectra the frequency
        #      branch would otherwise recompute -------------------------
        spectra_flat = None
        if self.use["tsc"]:
            tsc_emb, tsc_ev = self.tsc(clip, return_evidence=True)
            spectra_flat = tsc_ev["spectra"].reshape(B * T, self.n_bins)
            if return_evidence:
                evidence.update({f"tsc_{k}": v for k, v in tsc_ev.items()
                                 if k != "descriptors"})
                evidence["tsc_descriptors"] = tsc_ev["descriptors"]

        # ---- per-frame branches -----------------------------------------
        if self.use["spatial"]:
            e = self.spatial(flat).reshape(B, T, -1)
            pooled, w = self.temp_spatial(e, frame_mask)
            embeddings.append(pooled * 0 if "spatial" in ablate else pooled)
            frame_weights["spatial"] = w

        if self.use["frequency"]:
            e = self.frequency(flat, spectra_flat).reshape(B, T, -1)
            pooled, w = self.temp_freq(e, frame_mask)
            embeddings.append(pooled * 0 if "frequency" in ablate else pooled)
            frame_weights["frequency"] = w

        if self.use["srm"]:
            e, energy = self.srm(flat, return_residual_energy=True)
            e = e.reshape(B, T, -1)
            pooled, w = self.temp_srm(e, frame_mask)
            embeddings.append(pooled * 0 if "srm" in ablate else pooled)
            frame_weights["srm"] = w
            if return_evidence:
                evidence["srm_energy"] = energy.reshape(B, T).detach()

        if self.use["tsc"]:
            embeddings.append(tsc_emb * 0 if "tsc" in ablate else tsc_emb)

        if self.use["lmf"]:
            if lmf is None:
                # A missing LMF pack is a DATA problem (no landmarks for this
                # clip), not a modelling choice. Substituting zeros silently
                # would make the branch look useless in evaluation for the
                # wrong reason, so it is logged and flagged in the info dict.
                lmf_emb = torch.zeros(B, self.branch_dim, device=clip.device,
                                      dtype=clip.dtype)
                info_lmf_missing = True
            else:
                lmf_emb = self.lmf(lmf["energy"].to(clip.device, clip.dtype),
                                   lmf["descriptors"].to(clip.device, clip.dtype))
                info_lmf_missing = False
            embeddings.append(lmf_emb * 0 if "lmf" in ablate else lmf_emb)

        # ---- fuse --------------------------------------------------------
        fused, branch_w = self.fusion(*embeddings)
        logits = self.classifier(fused)

        info: Dict[str, object] = {
            "branch_weights": branch_w,
            "branch_names": self.branch_names,
            "frame_weights": frame_weights,
        }
        if self.use["lmf"]:
            info["lmf_missing"] = info_lmf_missing
        if return_evidence:
            info["evidence"] = evidence
            info["embeddings"] = {n: e.detach() for n, e
                                  in zip(self.branch_names, embeddings)}
        return logits, info

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict_proba(self, clip: torch.Tensor,
                      frame_mask: Optional[torch.Tensor] = None,
                      lmf: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        logits, _ = self.forward(clip, frame_mask, lmf=lmf)
        return torch.softmax(logits, dim=1)


# ==========================================================================
# Transfer from the image-level FSW-Net
# ==========================================================================

def init_from_fswnet(model: FSVNet, ckpt_path: str, strict: bool = False,
                     verbose: bool = True) -> Dict[str, int]:
    """
    Warm-start FSV-Net's per-frame branches from the image FSW-Net checkpoint
    at ../DeepfakeProject/notebooks/best_fsw_net.pth.

    Only the shape-compatible tensors transfer: the SRM CNN and the frequency
    CNN map one-to-one; the EfficientNet trunk maps one-to-one; the classifier
    head does NOT (4 classes there, 2 here) and is deliberately left random.

    Returns a count of what moved. This is worth doing — the image model has
    already learnt useful low-level filters on 21k images — but note in the
    report that it is a *warm start*, not a result: the image model's honest
    held-out AUC was 0.5702, so nothing about its accuracy transfers, only its
    features.
    """
    import re

    raw = torch.load(ckpt_path, map_location="cpu")
    state = raw.get("model_state_dict", raw.get("state_dict", raw))
    if not isinstance(state, dict):
        raise ValueError(f"unrecognised checkpoint format: {type(raw)}")

    rename = [
        (r"^spatial_features\.", "spatial.features."),
        (r"^spatial_proj\.", "spatial.proj."),
        (r"^freq_branch\.", "frequency."),
        (r"^srm_branch\.", "srm."),
    ]
    mapped = {}
    for k, v in state.items():
        nk = k
        for pat, rep in rename:
            nk = re.sub(pat, rep, nk)
        mapped[nk] = v

    own = model.state_dict()
    moved, skipped = {}, []
    for k, v in mapped.items():
        if k in own and own[k].shape == v.shape:
            moved[k] = v
        else:
            skipped.append(k)

    missing = model.load_state_dict(moved, strict=False)
    report = {"transferred": len(moved), "skipped": len(skipped),
              "left_random": len(own) - len(moved)}
    if verbose:
        log.info(f"FSW-Net warm start: {report}")
        if skipped:
            log.info(f"  (skipped, e.g.: {skipped[:5]})")
        _ = missing
    return report


def build_model(cfg) -> FSVNet:
    """Construct FSV-Net from a Config object."""
    m = FSVNet(
        num_classes=cfg.model.num_classes,
        branch_dim=cfg.model.branch_dim,
        n_bins=cfg.model.azimuthal_bins,
        img_size=cfg.data.img_size,
        dropout=cfg.model.dropout,
        temporal_layers=cfg.model.temporal_layers,
        temporal_heads=cfg.model.temporal_heads,
        temporal_ff=cfg.model.temporal_ff,
        use_spatial=cfg.model.use_spatial,
        use_frequency=cfg.model.use_frequency,
        use_srm=cfg.model.use_srm,
        use_tsc=cfg.model.use_tsc,
        use_lmf=getattr(cfg.model, 'use_lmf', True),
        pretrained=cfg.model.pretrained,
        backbone=cfg.model.spatial_backbone,
        max_frames=max(64, cfg.data.num_frames),
    )
    if cfg.model.init_from_fswnet:
        try:
            init_from_fswnet(m, cfg.model.init_from_fswnet)
        except Exception as e:
            log.warning(f"FSW-Net warm start failed ({e}) — training from scratch")
    if cfg.train.grad_checkpointing:
        m.set_grad_checkpointing(True)
    return m


__all__ = ["FSVNet", "build_model", "init_from_fswnet"]
