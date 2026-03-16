"""
Knowledge Distillation Training Script (Milestone 2)

Trains a student YOLO model (YOLOv8n) using Feature-Based Knowledge
Distillation from a teacher model (YOLOv8l).

This script:
1. Loads teacher (frozen) and student (trainable) models
2. Registers forward hooks to extract P3/P4/P5 feature maps
3. Uses 1x1 conv adapters to align channel dimensions
4. Trains with combined loss: Detection Loss + alpha * Feature Distillation Loss
5. Saves best and last checkpoints

Usage:
    # Debug run with COCO128 (quick, ~5 min)
    python train_distill.py --data coco128.yaml --epochs 5 --batch 8

    # Full training on COCO 2017 (hours/days depending on GPU)
    python train_distill.py --data coco.yaml --epochs 100 --batch 16

    # Use config file
    python train_distill.py --config configs/distill_config.yaml

    # Custom teacher/student
    python train_distill.py --teacher yolov8m.pt --student yolov8n.pt
"""

import argparse
import sys
from pathlib import Path

import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from distill.trainer import DistillationTrainer


def parse_args():
    parser = argparse.ArgumentParser(
        description="YOLO Feature-Based Knowledge Distillation Training"
    )

    # Config file (overrides individual args if provided)
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file")

    # Model
    parser.add_argument("--teacher", type=str, default="yolov8l.pt",
                        help="Teacher model weights (default: yolov8l.pt)")
    parser.add_argument("--student", type=str, default="yolov8n.pt",
                        help="Student model weights (default: yolov8n.pt)")

    # Data
    parser.add_argument("--data", type=str, default="coco.yaml",
                        help="Dataset config YAML (default: coco.yaml)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Input image size (default: 640)")

    # Training
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs (default: 100)")
    parser.add_argument("--batch", type=int, default=16,
                        help="Batch size (default: 16, reduce if OOM)")
    parser.add_argument("--lr0", type=float, default=0.01,
                        help="Initial learning rate (default: 0.01)")
    parser.add_argument("--momentum", type=float, default=0.937,
                        help="SGD momentum (default: 0.937)")
    parser.add_argument("--weight-decay", type=float, default=0.0005,
                        help="Weight decay (default: 0.0005)")
    parser.add_argument("--warmup-epochs", type=int, default=3,
                        help="Warmup epochs (default: 3)")
    parser.add_argument("--workers", type=int, default=8,
                        help="DataLoader workers (default: 8)")

    # Distillation
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Feature distillation loss weight (default: 0.5)")
    parser.add_argument("--beta", type=float, default=0.0,
                        help="Response KD loss weight (default: 0.0, feature-only)")
    parser.add_argument("--temperature", type=float, default=4.0,
                        help="KD temperature (default: 4.0)")
    parser.add_argument("--feature-loss", type=str, default="mse",
                        choices=["mse", "l1"],
                        help="Feature loss type (default: mse)")
    parser.add_argument("--normalize-features", action="store_true",
                        help="L2-normalize features before computing loss")

    # Output
    parser.add_argument("--project", type=str, default="runs/distill",
                        help="Output project directory")
    parser.add_argument("--name", type=str, default="yolov8l_to_yolov8n",
                        help="Experiment name")
    parser.add_argument("--save-period", type=int, default=10,
                        help="Save checkpoint every N epochs")

    # Device
    parser.add_argument("--device", type=str, default="",
                        help="Device: '' (auto), '0' (GPU 0), 'cpu'")

    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def main():
    args = parse_args()

    # Load config file if provided
    if args.config:
        print(f"Loading config from: {args.config}")
        config = load_config(args.config)
    else:
        config = {}

    # Merge: config file values < command-line args (CLI takes precedence)
    trainer_kwargs = {
        "teacher_weights": config.get("teacher_weights", args.teacher),
        "student_weights": config.get("student_weights", args.student),
        "data": config.get("data", args.data),
        "imgsz": config.get("imgsz", args.imgsz),
        "epochs": config.get("epochs", args.epochs),
        "batch_size": config.get("batch_size", args.batch),
        "alpha": config.get("alpha", args.alpha),
        "beta": config.get("beta", args.beta),
        "temperature": config.get("temperature", args.temperature),
        "lr0": config.get("lr0", args.lr0),
        "momentum": config.get("momentum", args.momentum),
        "weight_decay": config.get("weight_decay", args.weight_decay),
        "warmup_epochs": config.get("warmup_epochs", args.warmup_epochs),
        "device": config.get("device", args.device),
        "project": config.get("project", args.project),
        "name": config.get("name", args.name),
        "save_period": config.get("save_period", args.save_period),
        "workers": config.get("workers", args.workers),
        "feature_loss_type": config.get("feature_loss_type", args.feature_loss),
        "normalize_features": config.get("normalize_features", args.normalize_features),
    }

    print("=" * 60)
    print("  COMP 4901D - Knowledge Distillation Training")
    print("=" * 60)
    print(f"\n  Teacher:    {trainer_kwargs['teacher_weights']}")
    print(f"  Student:    {trainer_kwargs['student_weights']}")
    print(f"  Dataset:    {trainer_kwargs['data']}")
    print(f"  Epochs:     {trainer_kwargs['epochs']}")
    print(f"  Batch:      {trainer_kwargs['batch_size']}")
    print(f"  Image Size: {trainer_kwargs['imgsz']}")
    print(f"  Alpha:      {trainer_kwargs['alpha']}")
    print(f"  Beta:       {trainer_kwargs['beta']}")
    print(f"  Device:     {trainer_kwargs['device'] or 'auto'}")
    print(f"  Output:     {trainer_kwargs['project']}/{trainer_kwargs['name']}")
    print("=" * 60)

    # Create and run trainer
    trainer = DistillationTrainer(**trainer_kwargs)
    trainer.train()

    print("\nTraining complete!")


if __name__ == "__main__":
    main()

