"""
Distillation Loss Functions for Feature-Based Knowledge Distillation.

Implements:
1. Feature Imitation Loss (L2/MSE between aligned feature maps)
2. Optional Response-Based KD Loss (KL divergence on soft logits)

Total Distillation Loss = alpha * Feature_Loss + beta * Response_Loss
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class FeatureDistillationLoss(nn.Module):
    """
    Computes the feature imitation loss between Teacher and Student
    feature maps after channel alignment.

    Loss = (1/N) * sum over all feature levels of MSE(adapted_student, teacher)

    Supports:
        - L2 (MSE) loss: standard feature imitation
        - Normalized L2: L2 on L2-normalized features (cosine-like)
    """

    def __init__(self, loss_type: str = "mse", normalize: bool = False):
        """
        Args:
            loss_type: "mse" for Mean Squared Error, "l1" for L1 loss.
            normalize: If True, L2-normalize features before computing loss.
        """
        super().__init__()
        self.loss_type = loss_type
        self.normalize = normalize

        if loss_type == "mse":
            self.criterion = nn.MSELoss(reduction="mean")
        elif loss_type == "l1":
            self.criterion = nn.L1Loss(reduction="mean")
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}. Use 'mse' or 'l1'.")

    def forward(
        self,
        student_features: Dict[str, torch.Tensor],
        teacher_features: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute the total feature distillation loss across all feature levels.

        Args:
            student_features: Adapted student features {name: (B, C_t, H, W)}.
                              Must already be channel-aligned to teacher dims.
            teacher_features: Teacher features {name: (B, C_t, H, W)}.

        Returns:
            Scalar loss tensor.
        """
        total_loss = torch.tensor(0.0, device=next(iter(teacher_features.values())).device)
        num_levels = 0

        for name in teacher_features:
            if name not in student_features:
                continue

            s_feat = student_features[name]
            t_feat = teacher_features[name].detach()  # Teacher is frozen

            # Ensure spatial dimensions match (should be same if same input)
            if s_feat.shape[2:] != t_feat.shape[2:]:
                s_feat = F.interpolate(
                    s_feat, size=t_feat.shape[2:], mode="bilinear", align_corners=False
                )

            if self.normalize:
                s_feat = F.normalize(s_feat, p=2, dim=1)
                t_feat = F.normalize(t_feat, p=2, dim=1)

            total_loss = total_loss + self.criterion(s_feat, t_feat)
            num_levels += 1

        if num_levels > 0:
            total_loss = total_loss / num_levels

        return total_loss


class ResponseDistillationLoss(nn.Module):
    """
    Response-based Knowledge Distillation loss using KL Divergence.

    Softens the logits from Teacher and Student using a temperature parameter,
    then computes KL divergence between the softened distributions.

    Loss = T^2 * KL(softmax(student_logits/T) || softmax(teacher_logits/T))
    """

    def __init__(self, temperature: float = 4.0):
        """
        Args:
            temperature: Temperature for softening logits. Higher = softer.
        """
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute KL divergence loss between softened Teacher and Student logits.

        Args:
            student_logits: Raw student classification logits (B, num_classes, ...).
            teacher_logits: Raw teacher classification logits (B, num_classes, ...).

        Returns:
            Scalar KL divergence loss (scaled by T^2).
        """
        T = self.temperature

        soft_student = F.log_softmax(student_logits / T, dim=1)
        soft_teacher = F.softmax(teacher_logits / T, dim=1)

        kl_loss = F.kl_div(soft_student, soft_teacher, reduction="batchmean")

        # Scale by T^2 to make gradients comparable to hard-label loss
        return kl_loss * (T * T)


class CombinedDistillationLoss(nn.Module):
    """
    Combined loss for Knowledge Distillation training.

    Total Loss = detection_loss + alpha * feature_loss + beta * response_loss

    Where:
        - detection_loss: Standard YOLO detection loss (cls + box + obj)
        - feature_loss: L2 loss between aligned feature maps
        - response_loss: KL divergence on softened logits (optional)
    """

    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.0,
        feature_loss_type: str = "mse",
        temperature: float = 4.0,
        normalize_features: bool = False,
    ):
        """
        Args:
            alpha: Weight for feature distillation loss.
            beta: Weight for response distillation loss (0 to disable).
            feature_loss_type: "mse" or "l1" for feature loss.
            temperature: Temperature for response KD.
            normalize_features: Whether to L2-normalize features before MSE.
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta

        self.feature_loss = FeatureDistillationLoss(
            loss_type=feature_loss_type,
            normalize=normalize_features,
        )
        self.response_loss = (
            ResponseDistillationLoss(temperature=temperature)
            if beta > 0
            else None
        )

    def forward(
        self,
        student_features: Dict[str, torch.Tensor],
        teacher_features: Dict[str, torch.Tensor],
        student_logits: Optional[torch.Tensor] = None,
        teacher_logits: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined distillation losses.

        Returns:
            Dict with keys "feature_loss", "response_loss", "total_distill_loss".
        """
        result = {}

        # Feature imitation loss
        feat_loss = self.feature_loss(student_features, teacher_features)
        result["feature_loss"] = feat_loss

        total = self.alpha * feat_loss

        # Response KD loss (optional)
        if self.response_loss is not None and student_logits is not None:
            resp_loss = self.response_loss(student_logits, teacher_logits)
            result["response_loss"] = resp_loss
            total = total + self.beta * resp_loss
        else:
            result["response_loss"] = torch.tensor(0.0, device=feat_loss.device)

        result["total_distill_loss"] = total
        return result


