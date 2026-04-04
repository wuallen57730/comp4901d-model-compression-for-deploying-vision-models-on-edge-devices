"""
Quick Pipeline Verification Script

Runs a fast end-to-end test of the entire pipeline using COCO128:
1. Load pre-trained YOLOv8n
2. Quick-train on COCO128 (3 epochs)
3. Validate mAP
4. Test inference
5. Export to ONNX
6. Test feature extraction hooks (for KD)

This is designed to verify that everything works before starting
the actual long-running training.

Usage:
    python verify_pipeline.py
    python verify_pipeline.py --device cpu    # Force CPU
"""

import argparse
import sys
import os
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description="Quick Pipeline Verification")
    parser.add_argument("--device", type=str, default="",
                        help="Device: '' (auto), '0' (GPU), 'cpu'")
    args = parser.parse_args()

    device = args.device or ("0" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print("  Pipeline Verification (COCO128 Quick Test)")
    print("=" * 60)
    print(f"  Device: {device}")
    print(f"  CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print()

    # ---- Step 1: Load model ----
    print("[1/6] Loading YOLOv8n pre-trained model...")
    from ultralytics import YOLO
    model = YOLO("yolov8n.pt")
    print("  OK - Model loaded")
    print(f"  Parameters: {sum(p.numel() for p in model.model.parameters()):,}")

    # ---- Step 2: Quick train ----
    print("\n[2/6] Quick training on COCO128 (3 epochs)...")
    model.train(
        data="coco128.yaml",
        epochs=3,
        imgsz=320,  # Smaller for speed
        batch=8,
        device=device,
        project="runs/verify",
        name="quick_test",
        exist_ok=True,
        verbose=False,
    )
    print("  OK - Training complete")

    # ---- Step 3: Validate ----
    print("\n[3/6] Validating on COCO128...")
    metrics = model.val(data="coco128.yaml", imgsz=320, device=device, verbose=False)
    print(f"  mAP50: {metrics.box.map50:.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}")

    # ---- Step 4: Inference ----
    print("\n[4/6] Testing inference...")
    results = model.predict(
        source="https://ultralytics.com/images/bus.jpg",
        imgsz=320,
        device=device,
        save=True,
        project="runs/verify",
        name="predict_test",
        exist_ok=True,
        verbose=False,
    )
    print(f"  OK - Detected {len(results[0].boxes)} objects")

    # ---- Step 5: ONNX export ----
    print("\n[5/6] Exporting to ONNX...")
    export_path = model.export(format="onnx", imgsz=320)
    if export_path and os.path.exists(export_path):
        size_mb = os.path.getsize(export_path) / (1024 * 1024)
        print(f"  OK - Exported to {export_path} ({size_mb:.2f} MB)")
    else:
        print(f"  OK - Export returned: {export_path}")

    # ---- Step 6: Test feature extraction hooks ----
    print("\n[6/6] Testing feature extraction hooks (for KD)...")
    sys.path.insert(0, str(Path(__file__).parent))
    from distill.hooks import FeatureExtractor, detect_feature_channels_from_yolo

    # Test with student model
    student = YOLO("yolov8n.pt")
    channels = detect_feature_channels_from_yolo(student, imgsz=320, device=device)
    print(f"  Student (YOLOv8n) channels: {channels}")

    # Test with a larger model as teacher
    teacher = YOLO("yolov8s.pt")  # Use 's' instead of 'l' for speed
    t_channels = detect_feature_channels_from_yolo(teacher, imgsz=320, device=device)
    print(f"  Teacher (YOLOv8s) channels: {t_channels}")

    # Test adapter
    from distill.adapters import ChannelAdapter
    adapter = ChannelAdapter(channels, t_channels)
    print(f"  Adapter parameters: {sum(p.numel() for p in adapter.parameters()):,}")

    # Test loss computation
    from distill.losses import FeatureDistillationLoss
    loss_fn = FeatureDistillationLoss(loss_type="mse")

    # Create dummy features for testing
    dev = torch.device(f"cuda:{device}" if device != "cpu" and torch.cuda.is_available() else "cpu")
    adapter = adapter.to(dev)

    dummy_student_feats = {
        name: torch.randn(1, ch, 40, 40, device=dev)
        for name, ch in channels.items()
    }
    dummy_teacher_feats = {
        name: torch.randn(1, ch, 40, 40, device=dev)
        for name, ch in t_channels.items()
    }

    adapted = adapter(dummy_student_feats)
    loss = loss_fn(adapted, dummy_teacher_feats)
    print(f"  Feature distillation loss: {loss.item():.4f}")
    print("  OK - KD pipeline verified!")

    # ---- Summary ----
    print(f"\n{'=' * 60}")
    print("  ALL CHECKS PASSED!")
    print("  Pipeline is ready for full training.")
    print(f"{'=' * 60}")
    print("\nNext steps:")
    print("  1. Run baseline evaluation:")
    print("     python eval_baseline.py --data coco.yaml")
    print("  2. Start KD training:")
    print("     python train_distill.py --data coco.yaml --epochs 100")
    print("  3. Or quick KD test first:")
    print("     python train_distill.py --data coco128.yaml --epochs 5 --batch 8")


if __name__ == "__main__":
    main()


