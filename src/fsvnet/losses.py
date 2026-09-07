"""
losses.py
=========

Focal loss (carried over from the image FSW-Net) plus two regularisers that
exist specifically to keep the branch-attention fusion honest.

Why the extra terms
-------------------
A gated fusion has a well-known failure mode: the gate collapses onto whichever
branch converges fastest — usually the ImageNet-pretrained spatial trunk — and
the other branches never receive useful gradient. You then report a "four-branch
model" whose ablation table shows the other three contribute nothing, and you
cannot tell whether that is a real finding or an optimisation artefact.

`BranchDiversityLoss` adds a small entropy bonus on the gate weights for the
first few epochs, which keeps every branch alive long enough to be judged on
its merits. Set `weight=0` to turn it off — and do report both settings, since
"the gate collapses onto spatial even when regularised" is itself a result.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal loss with optional class weights and label smoothing.

    gamma=0 reduces to weighted cross-entropy, which is the ablation baseline.
    """

    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0,
                 label_smoothing: float = 0.0):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("alpha", alpha if alpha is not None else torch.empty(0))
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        weight = self.alpha if self.alpha.numel() else None
        ce = F.cross_entropy(logits, target, weight=weight,
                             label_smoothing=self.label_smoothing,
                             reduction="none")
        if self.gamma == 0:
            return ce.mean()
        # pt from the *unsmoothed* probability of the true class
        with torch.no_grad():
            pt = torch.softmax(logits, dim=1).gather(1, target.view(-1, 1)).squeeze(1)
        return (((1 - pt) ** self.gamma) * ce).mean()


class BranchDiversityLoss(nn.Module):
    """
    Negative entropy of the branch-attention distribution, averaged over the
    batch. Minimising it pushes the gate towards using every branch.

    Applied with a schedule: full weight for `warm_epochs`, then decayed to
    zero, so late training is free to specialise.
    """

    def __init__(self, weight: float = 0.02, warm_epochs: int = 3):
        super().__init__()
        self.weight = weight
        self.warm_epochs = warm_epochs

    def current_weight(self, epoch: int) -> float:
        if self.weight <= 0:
            return 0.0
        if epoch < self.warm_epochs:
            return self.weight
        return self.weight * max(0.0, 1.0 - (epoch - self.warm_epochs) / 3.0)

    def forward(self, branch_weights: torch.Tensor, epoch: int = 0) -> torch.Tensor:
        w = self.current_weight(epoch)
        if w == 0:
            return branch_weights.sum() * 0.0
        p = branch_weights.clamp_min(1e-8)
        entropy = -(p * p.log()).sum(dim=1).mean()
        max_entropy = torch.log(torch.tensor(float(p.shape[1]), device=p.device))
        return w * (max_entropy - entropy)


class FSVLoss(nn.Module):
    """Focal + diversity, in one object so train.py stays readable."""

    def __init__(self, class_weights: Optional[torch.Tensor] = None,
                 gamma: float = 2.0, label_smoothing: float = 0.05,
                 diversity_weight: float = 0.02, diversity_warm: int = 3):
        super().__init__()
        self.focal = FocalLoss(class_weights, gamma, label_smoothing)
        self.diversity = BranchDiversityLoss(diversity_weight, diversity_warm)

    def forward(self, logits: torch.Tensor, target: torch.Tensor,
                branch_weights: Optional[torch.Tensor] = None,
                epoch: int = 0) -> dict:
        cls = self.focal(logits, target)
        div = (self.diversity(branch_weights, epoch)
               if branch_weights is not None else logits.sum() * 0.0)
        return {"loss": cls + div, "cls": cls.detach(), "div": div.detach()}


__all__ = ["FocalLoss", "BranchDiversityLoss", "FSVLoss"]
