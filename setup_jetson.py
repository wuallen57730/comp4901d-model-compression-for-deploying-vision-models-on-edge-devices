"""
Jetson Orin NX Environment Setup & Verification Script

Run this script ON the Jetson to verify all dependencies and hardware
acceleration are properly configured.

Usage (on Jetson):
    python setup_jetson.py
"""

import sys
import os
import subprocess


def check_python():
    """Check Python version."""
    v = sys.version_info
    print(f"  Python: {v.major}.{v.minor}.{v.micro}")
    if v.major < 3 or (v.major == 3 and v.minor < 8):
        print("    [WARN] Python >= 3.8 recommended")
    else:
        print("    [OK]")


def check_pytorch():
    """Check PyTorch installation and CUDA support."""
    try:
        import torch
        print(f"  PyTorch: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
            print("    [OK]")
        else:
            print("    [WARN] CUDA not available - check JetPack installation")
    except ImportError:
        print("  PyTorch: NOT INSTALLED")
        print("    [ERROR] Install PyTorch for Jetson from NVIDIA forums")


def check_tensorrt():
    """Check TensorRT installation."""
    try:
        import tensorrt
        print(f"  TensorRT: {tensorrt.__version__}")
        print("    [OK]")
    except ImportError:
        print("  TensorRT: NOT INSTALLED")
        print("    [INFO] TensorRT should be pre-installed with JetPack")


def check_ultralytics():
    """Check Ultralytics YOLOv8 installation."""
    try:
        import ultralytics
        print(f"  Ultralytics: {ultralytics.__version__}")
        print("    [OK]")
    except ImportError:
        print("  Ultralytics: NOT INSTALLED")
        print("    [ACTION] Run: pip install ultralytics")


def check_opencv():
    """Check OpenCV installation."""
    try:
        import cv2
        print(f"  OpenCV: {cv2.__version__}")
        print("    [OK]")
    except ImportError:
        print("  OpenCV: NOT INSTALLED")
        print("    [ACTION] Run: pip install opencv-python")


def check_onnx():
    """Check ONNX and ONNX Runtime."""
    try:
        import onnx
        print(f"  ONNX: {onnx.__version__}")
    except ImportError:
        print("  ONNX: NOT INSTALLED")

    try:
        import onnxruntime
        print(f"  ONNX Runtime: {onnxruntime.__version__}")
        providers = onnxruntime.get_available_providers()
        print(f"  ORT Providers: {providers}")
    except ImportError:
        print("  ONNX Runtime: NOT INSTALLED")


def check_jetson_stats():
    """Check Jetson power/thermal monitoring tools."""
    try:
        result = subprocess.run(["tegrastats", "--help"], capture_output=True, text=True)
        print("  tegrastats: available")
        print("    [OK] Use 'tegrastats' to monitor GPU/CPU during benchmarks")
    except FileNotFoundError:
        print("  tegrastats: not found")
        print("    [INFO] Install jetson-stats: pip install jetson-stats")


def check_trtexec():
    """Check trtexec availability."""
    try:
        result = subprocess.run(["trtexec", "--help"], capture_output=True, text=True)
        print("  trtexec: available")
        print("    [OK] Use trtexec for TensorRT engine building and profiling")
    except FileNotFoundError:
        print("  trtexec: not found")
        print("    [INFO] trtexec should be at /usr/src/tensorrt/bin/trtexec")


def quick_inference_test():
    """Run a quick YOLOv8n inference to verify the full pipeline."""
    print("\n[Quick Inference Test]")
    try:
        import torch
        from ultralytics import YOLO

        model = YOLO("yolov8n.pt")
        dummy = torch.randn(1, 3, 640, 640)

        if torch.cuda.is_available():
            dummy = dummy.cuda()

        # Run inference
        import time
        start = time.time()
        results = model.predict(
            source=dummy.cpu().numpy().transpose(0, 2, 3, 1)[0],
            verbose=False,
        )
        elapsed = time.time() - start
        print(f"  Inference test: {elapsed * 1000:.1f} ms")
        print("  [OK] Pipeline works!")
    except Exception as e:
        print(f"  Inference test failed: {e}")
        print("  [WARN] Check model and environment")


def main():
    print("=" * 60)
    print("  Jetson Orin NX Environment Verification")
    print("=" * 60)

    print("\n[System]")
    check_python()

    print("\n[Deep Learning Framework]")
    check_pytorch()

    print("\n[Inference Engines]")
    check_tensorrt()
    check_trtexec()

    print("\n[ML Libraries]")
    check_ultralytics()
    check_opencv()
    check_onnx()

    print("\n[Monitoring Tools]")
    check_jetson_stats()

    quick_inference_test()

    print("\n" + "=" * 60)
    print("  Setup verification complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

