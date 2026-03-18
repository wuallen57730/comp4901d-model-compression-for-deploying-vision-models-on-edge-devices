"""
Channel Adapters for Knowledge Distillation.

When the Teacher and Student have different channel dimensions at corresponding
feature map locations, we use a 1x1 convolution to project the Student's
feature maps into the Teacher's channel space before computing the
distillation loss.

Example:
    Teacher YOLOv8l P3 has 256 channels.
    Student YOLOv8n P3 has 128 channels.
    Adapter: Conv2d(128, 256, kernel_size=1) aligns them.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Dict


class ChannelAdapter(nn.Module):
    """
    A set of 1x1 convolutions that align Student feature map channels
    to match Teacher feature map channels.

    Contains one Conv2d per feature level (P3, P4, P5).
    """

    def __init__(
        self,
        student_channels: Dict[str, int],
        teacher_channels: Dict[str, int],
    ):
        """
        Args:
            student_channels: Dict mapping feature name -> student channel count.
                              e.g. {"P3": 128, "P4": 256, "P5": 256}
            teacher_channels: Dict mapping feature name -> teacher channel count.
                              e.g. {"P3": 256, "P4": 512, "P5": 512}
        """
        super().__init__()

        self.adapters = nn.ModuleDict()
        for name in student_channels:
            s_ch = student_channels[name]
            t_ch = teacher_channels[name]
            if s_ch != t_ch:
                # 1x1 conv to project student channels -> teacher channels
                self.adapters[name] = nn.Sequential(
                    nn.Conv2d(s_ch, t_ch, kernel_size=1, bias=False),
                    nn.BatchNorm2d(t_ch),
                )
            else:
                # If channels match, use identity (no projection needed)
                self.adapters[name] = nn.Identity()

        self._init_weights()

    def _init_weights(self):
        """Initialize adapter weights with Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(
        self, student_features: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Project student features to match teacher channel dimensions.

        Args:
            student_features: Dict of student feature maps {name: (B, C_s, H, W)}.

        Returns:
            Dict of adapted feature maps {name: (B, C_t, H, W)}.
        """
        adapted = {}
        for name, feat in student_features.items():
            if name in self.adapters:
                adapted[name] = self.adapters[name](feat)
            else:
                adapted[name] = feat
        return adapted


