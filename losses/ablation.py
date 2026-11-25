"""
Losses used only for the loss ablation. They reuse the original implementations so that
nothing but the studied factor changes.

  sfmi          1 - sFMI                      HFM without the hard-sigmoid gate
  hard_afm_l0u1 (1 - sFMI) * (1 - sFMI)       HFM with gate bounds l = 0, u = 1
  focal_a75     Focal, gamma = 2, alpha = 0.75 flood pixels weighted 0.75 instead of 0.25
"""
import torch

from .focal import FocalLoss
from .hard_afm import HardAFMLoss


class SFMILoss(HardAFMLoss):
    """Same smoothed statistics as HardAFMLoss (pooled over the mini-batch), no gate."""

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs).view(-1)
        targets = targets.view(-1)

        tp = (inputs * targets).sum()
        fp = (inputs * (1 - targets)).sum()
        fn = ((1 - inputs) * targets).sum()

        precision = (tp + self.smooth) / (tp + fp + self.smooth)
        recall = (tp + self.smooth) / (tp + fn + self.smooth)
        return 1 - precision * recall


def build_sfmi(num_classes=1):
    return SFMILoss(num_classes=num_classes)


def build_hard_afm_l0u1(num_classes=1):
    return HardAFMLoss(hs_low=0.0, hs_high=1.0, num_classes=num_classes)


def build_focal_a75(num_classes=1):
    return FocalLoss(alpha=0.75, gamma=2.0, num_classes=num_classes)
