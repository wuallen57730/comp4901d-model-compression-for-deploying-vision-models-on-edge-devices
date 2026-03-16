"""
Custom Distillation Trainer for YOLO Knowledge Distillation.

Extends the standard Ultralytics training loop to incorporate:
1. Teacher model forward pass (frozen) to extract feature maps
2. Student model forward pass with feature extraction hooks
3. Combined loss = YOLO detection loss + alpha * feature distillation loss

Usage:
    from distill.trainer import DistillationTrainer
    trainer = DistillationTrainer(
        teacher_weights="yolov8l.pt",
        student_weights="yolov8n.pt",
        data="coco.yaml",
        ...
    )
    trainer.train()
"""

from __future__ import annotations

import os
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import Dict, Optional, Tuple
from tqdm import tqdm

from ultralytics import YOLO
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.utils import LOGGER, RANK
from ultralytics.cfg import get_cfg

from .hooks import FeatureExtractor, detect_feature_channels_from_yolo
from .adapters import ChannelAdapter
from .losses import CombinedDistillationLoss


class DistillationTrainer:
    """
    Feature-Based Knowledge Distillation trainer for YOLO models.

    Implements the training loop where:
    - Teacher (large YOLO) is frozen, only used for feature extraction
    - Student (small YOLO) is trained with combined detection + distillation loss
    - 1x1 Conv adapters align channel dimensions between teacher/student features
    """

    def __init__(
        self,
        teacher_weights: str = "yolov8l.pt",
        student_weights: str = "yolov8n.pt",
        data: str = "coco.yaml",
        imgsz: int = 640,
        epochs: int = 100,
        batch_size: int = 16,
        alpha: float = 0.5,
        beta: float = 0.0,
        temperature: float = 4.0,
        lr0: float = 0.01,
        momentum: float = 0.937,
        weight_decay: float = 0.0005,
        warmup_epochs: int = 3,
        device: str = "",
        project: str = "runs/distill",
        name: str = "exp",
        save_period: int = 10,
        val_period: int = 5,
        workers: int = 8,
        feature_loss_type: str = "mse",
        normalize_features: bool = False,
    ):
        """
        Args:
            teacher_weights: Path to teacher model weights.
            student_weights: Path to student model weights (initial).
            data: Path to dataset YAML config.
            imgsz: Input image size.
            epochs: Total training epochs.
            batch_size: Batch size for training.
            alpha: Weight for feature distillation loss.
            beta: Weight for response KD loss (0 = feature-only).
            temperature: Temperature for response KD.
            lr0: Initial learning rate.
            momentum: SGD momentum.
            weight_decay: Weight decay.
            warmup_epochs: Number of warmup epochs.
            device: Device string ("", "0", "cpu", etc.).
            project: Output project directory.
            name: Experiment name.
            save_period: Save checkpoint every N epochs.
            val_period: Run validation every N epochs.
            workers: DataLoader workers.
            feature_loss_type: "mse" or "l1".
            normalize_features: Whether to normalize features before loss.
        """
        self.teacher_weights = teacher_weights
        self.student_weights = student_weights
        self.data = data
        self.imgsz = imgsz
        self.epochs = epochs
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature
        self.lr0 = lr0
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.device_str = device
        self.project = project
        self.name = name
        self.save_period = save_period
        self.val_period = val_period
        self.workers = workers
        self.feature_loss_type = feature_loss_type
        self.normalize_features = normalize_features

        # Will be initialized in setup()
        self.teacher_model = None
        self.student_model = None
        self.teacher_extractor = None
        self.student_extractor = None
        self.adapter = None
        self.distill_loss_fn = None
        self.optimizer = None
        self.scheduler = None
        self.device = None
        self.save_dir = None
        self.best_map = 0.0

    def setup(self):
        """Initialize models, hooks, adapters, optimizer, and data loaders."""
        # ---- Device ----
        if self.device_str == "" or self.device_str == "auto":
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        elif self.device_str == "cpu":
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(f"cuda:{self.device_str}")

        LOGGER.info(f"[KD] Using device: {self.device}")

        # ---- Save directory ----
        self.save_dir = Path(self.project) / self.name
        self.save_dir.mkdir(parents=True, exist_ok=True)
        (self.save_dir / "weights").mkdir(exist_ok=True)

        # ---- Load Teacher (frozen) ----
        LOGGER.info(f"[KD] Loading Teacher: {self.teacher_weights}")
        self.teacher_yolo = YOLO(self.teacher_weights)
        self.teacher_model = self.teacher_yolo.model.to(self.device)
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False

        # ---- Load Student (trainable) ----
        LOGGER.info(f"[KD] Loading Student: {self.student_weights}")
        self.student_yolo = YOLO(self.student_weights)
        self.student_model = self.student_yolo.model.to(self.device)
        self.student_model.train()

        # ---- Detect channel dimensions ----
        LOGGER.info("[KD] Detecting feature channel dimensions...")
        teacher_channels = detect_feature_channels_from_yolo(
            self.teacher_yolo, imgsz=self.imgsz, device=str(self.device)
        )
        student_channels = detect_feature_channels_from_yolo(
            self.student_yolo, imgsz=self.imgsz, device=str(self.device)
        )
        LOGGER.info(f"[KD] Teacher channels: {teacher_channels}")
        LOGGER.info(f"[KD] Student channels: {student_channels}")

        # ---- Feature Extractors (hooks) ----
        teacher_inner = self.teacher_model.model  # nn.Sequential
        student_inner = self.student_model.model  # nn.Sequential

        self.teacher_extractor = FeatureExtractor(teacher_inner)
        self.student_extractor = FeatureExtractor(student_inner)
        self.teacher_extractor.register_hooks()
        self.student_extractor.register_hooks()

        # ---- Channel Adapters ----
        self.adapter = ChannelAdapter(student_channels, teacher_channels).to(self.device)

        # ---- Distillation Loss ----
        self.distill_loss_fn = CombinedDistillationLoss(
            alpha=self.alpha,
            beta=self.beta,
            feature_loss_type=self.feature_loss_type,
            temperature=self.temperature,
            normalize_features=self.normalize_features,
        ).to(self.device)

        # ---- Optimizer (Student params + Adapter params) ----
        params = list(self.student_model.parameters()) + list(self.adapter.parameters())
        self.optimizer = optim.SGD(
            params,
            lr=self.lr0,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
            nesterov=True,
        )

        # ---- LR Scheduler (Cosine Annealing) ----
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.epochs,
            eta_min=self.lr0 * 0.01,
        )

        LOGGER.info("[KD] Setup complete.")
        LOGGER.info(f"[KD] Teacher: {sum(p.numel() for p in self.teacher_model.parameters()):,} params (frozen)")
        LOGGER.info(f"[KD] Student: {sum(p.numel() for p in self.student_model.parameters()):,} params (trainable)")
        LOGGER.info(f"[KD] Adapter: {sum(p.numel() for p in self.adapter.parameters()):,} params")
        LOGGER.info(f"[KD] alpha={self.alpha}, beta={self.beta}, T={self.temperature}")

    def train(self):
        """
        Main training loop with Knowledge Distillation.

        Uses the standard Ultralytics YOLO training for the student,
        but adds the feature distillation loss to each training step.
        """
        self.setup()

        LOGGER.info(f"\n[KD] Starting Knowledge Distillation Training")
        LOGGER.info(f"[KD] Epochs: {self.epochs}, Batch: {self.batch_size}, ImgSz: {self.imgsz}")
        LOGGER.info(f"[KD] Saving to: {self.save_dir}")

        # Use Ultralytics' built-in training with distillation callback approach
        # We train the student model through Ultralytics API and add distillation
        # as a custom loss callback.
        #
        # Strategy: We use the student YOLO's train() method but inject our
        # distillation logic by:
        # 1. Training student normally (this handles data loading, augmentation, etc.)
        # 2. At each batch, also run teacher forward and compute feature loss
        #
        # Implementation: Patch the student's loss computation.

        self._train_with_distillation()

    def _train_with_distillation(self):
        """
        Train the student model with distillation using Ultralytics internals.

        Strategy:
        - Use Ultralytics DetectionTrainer for the student (handles data, aug, etc.)
        - Patch model.loss() to inject feature distillation loss
        - The teacher forward runs inside the patched loss to capture features
        - 1x1 Conv adapters align student→teacher channel dims before L2 loss

        Callback lifecycle in Ultralytics 8.x:
          _setup_train() → on_pretrain_routine_end  (model & optimizer ready)
          on_train_start                            (before first epoch)
          for each epoch:
            on_train_epoch_start
            for each batch:
              model(batch) → model.loss(batch) → [OUR PATCH]
              backward, optimizer step
              on_train_batch_end
            on_train_epoch_end
          on_train_end
        """
        from ultralytics.models.yolo.detect import DetectionTrainer

        trainer = DetectionTrainer(
            overrides={
                "model": self.student_weights,
                "data": self.data,
                "epochs": self.epochs,
                "imgsz": self.imgsz,
                "batch": self.batch_size,
                "device": self.device_str if self.device_str else None,
                "project": self.project,
                "name": self.name,
                "lr0": self.lr0,
                "momentum": self.momentum,
                "weight_decay": self.weight_decay,
                "warmup_epochs": self.warmup_epochs,
                "workers": self.workers,
                "save_period": self.save_period,
                "val": True,
                "plots": True,
                "exist_ok": True,
            }
        )

        # ---- Closure references ----
        teacher_model = self.teacher_model
        teacher_extractor = self.teacher_extractor
        adapter = self.adapter
        distill_loss_fn = self.distill_loss_fn

        # Mutable holder so inner callbacks can share the student extractor
        _student_ext = [None]

        # ------------------------------------------------------------------
        # Callback: on_pretrain_routine_end
        #   Fires at the END of _setup_train(), after model & optimizer exist.
        #   Perfect place to:
        #     1) register hooks on the trainer's student model
        #     2) move teacher / adapter to the correct device
        #     3) monkey-patch model.loss to add distillation
        #     4) add adapter params to the optimizer
        # ------------------------------------------------------------------
        def on_pretrain_routine_end(trainer_instance):
            # --- unwrap DDP / DataParallel ---
            model = trainer_instance.model
            if hasattr(model, "module"):
                model = model.module

            # 1. Register student feature hooks on the *trainer's* model
            student_ext = FeatureExtractor(model.model)
            student_ext.register_hooks()
            _student_ext[0] = student_ext

            # 2. Move teacher / adapter / loss to the same device as student
            dev = next(model.parameters()).device
            teacher_model.to(dev)
            adapter.to(dev)
            distill_loss_fn.to(dev)

            # Re-register teacher hooks (model objects survive .to())
            teacher_extractor.register_hooks()

            # 3. Monkey-patch model.loss
            #    In Ultralytics 8.x the training loop calls:
            #        self.loss, self.loss_items = self.model(batch)
            #    BaseModel.forward(batch_dict) → self.loss(batch)
            #    BaseModel.loss() runs student forward & criterion.
            #    We wrap that to add the teacher forward + distillation loss.
            _original_loss = model.loss          # bound method

            def _kd_loss(batch, preds=None):
                # a) original detection loss (student forward happens here)
                det_loss, det_loss_items = _original_loss(batch, preds)

                # b) student features were captured by hooks in (a)
                s_feats = student_ext.get_features()
                if s_feats:
                    # c) teacher forward (frozen, no grad)
                    with torch.no_grad():
                        teacher_model(batch["img"])
                    t_feats = teacher_extractor.get_features()

                    # d) adapt channels & compute feature distillation loss
                    adapted_s = adapter(s_feats)
                    d_out = distill_loss_fn(adapted_s, t_feats)

                    return det_loss + d_out["total_distill_loss"], det_loss_items

                return det_loss, det_loss_items

            model.loss = _kd_loss

            # 4. Add adapter parameters to the optimizer AND update scheduler
            #    "initial_lr" is required by Ultralytics' warmup / LR logic.
            adapter_lr = trainer_instance.args.lr0
            trainer_instance.optimizer.add_param_group({
                "params": list(adapter.parameters()),
                "lr": adapter_lr,
                "initial_lr": adapter_lr,
            })

            # The LR scheduler (LambdaLR) was built before we added the
            # extra param-group, so its internal lists are one element short.
            # Extend them so scheduler.step() won't crash.
            sched = trainer_instance.scheduler
            if sched is not None:
                if hasattr(sched, "lr_lambdas") and len(sched.lr_lambdas) < len(
                    trainer_instance.optimizer.param_groups
                ):
                    # Re-use the first group's lambda for the adapter
                    sched.lr_lambdas.append(sched.lr_lambdas[0])
                if hasattr(sched, "base_lrs") and len(sched.base_lrs) < len(
                    trainer_instance.optimizer.param_groups
                ):
                    sched.base_lrs.append(adapter_lr)
                # Also extend _last_lr if it exists (used by get_last_lr())
                if hasattr(sched, "_last_lr") and len(sched._last_lr) < len(
                    trainer_instance.optimizer.param_groups
                ):
                    sched._last_lr.append(adapter_lr)

            LOGGER.info(
                f"[KD] Distillation active — patched model.loss, "
                f"adapter ({sum(p.numel() for p in adapter.parameters()):,} params) "
                f"added to optimizer."
            )

        # ------------------------------------------------------------------
        def on_train_epoch_end(trainer_instance):
            LOGGER.info(
                f"[KD] Epoch {trainer_instance.epoch + 1}/{self.epochs} — "
                f"LR: {trainer_instance.optimizer.param_groups[0]['lr']:.6f}"
            )

        def on_train_end(trainer_instance):
            if _student_ext[0]:
                _student_ext[0].remove_hooks()
            teacher_extractor.remove_hooks()
            LOGGER.info("[KD] Training complete. Hooks removed.")

        # ---- register callbacks ----
        trainer.add_callback("on_pretrain_routine_end", on_pretrain_routine_end)
        trainer.add_callback("on_train_epoch_end", on_train_epoch_end)
        trainer.add_callback("on_train_end", on_train_end)

        # ---- run ----
        trainer.train()

        LOGGER.info(f"\n[KD] Training complete! Results saved to {self.save_dir}")
        LOGGER.info(f"[KD] Best model: {trainer.best}")

        return trainer

    def cleanup(self):
        """Remove hooks and free memory."""
        if self.teacher_extractor:
            self.teacher_extractor.remove_hooks()
        if self.student_extractor:
            self.student_extractor.remove_hooks()

