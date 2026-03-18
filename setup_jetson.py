"""
Jetson Orin Nano Environment Setup & Verification Script

Run this script ON the Jetson Orin Nano to verify all dependencies,
hardware acceleration, and available memory for on-device KD training.

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


def check_system_memory():
    """Check system RAM (shared with GPU on Jetson)."""
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    kb = int(line.split()[1])
                    gb = kb / (1024 ** 2)
                    print(f"  System RAM: {gb:.1f} GB (shared with GPU)")
                    if gb < 7.5:
                        print("    [WARN] < 8 GB — use batch_size=2 and teacher_half=true")
                    else:
                        print("    [OK]")
                    break
    except FileNotFoundError:
        print("  System RAM: (unable to read /proc/meminfo — not Linux?)")


def check_swap():
    """Check swap space (important for training on 8 GB Jetson)."""
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if line.startswith("SwapTotal"):
                    kb = int(line.split()[1])
                    gb = kb / (1024 ** 2)
                    print(f"  Swap: {gb:.1f} GB")
                    if gb < 4.0:
                        print("    [WARN] Swap < 4 GB. For KD training, recommend >= 8 GB swap.")
                        print("    [ACTION] sudo fallocate -l 8G /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile")
                    else:
                        print("    [OK]")
                    break
    except FileNotFoundError:
        print("  Swap: (unable to read /proc/meminfo)")


def check_jetson_power_mode():
    """Check current Jetson power mode (MAXN recommended for training)."""
    try:
        result = subprocess.run(["nvpmodel", "-q"], capture_output=True, text=True)
        print(f"  Power mode:\n    {result.stdout.strip()}")
        if "MAXN" not in result.stdout.upper():
            print("    [WARN] Not in MAXN mode. For training, run: sudo nvpmodel -m 0")
        else:
            print("    [OK] MAXN mode — maximum performance")
    except FileNotFoundError:
        print("  Power mode: nvpmodel not found (not a Jetson?)")


def check_pytorch():
    """Check PyTorch installation and CUDA support."""
    try:
        import torch
        print(f"  PyTorch: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            props = torch.cuda.get_device_properties(0)
            print(f"  GPU Memory: {props.total_mem / 1e9:.1f} GB")
            print(f"  CUDA Cores: {props.multi_processor_count} SM")

            if props.total_mem / 1e9 < 7.5:
                print("    [INFO] 8 GB shared memory — use configs/distill_config_jetson.yaml")
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


def quick_training_memory_estimate():
    """Estimate memory for KD training with YOLOv8s teacher + YOLOv8n student."""
    print("\n[KD Training Memory Estimate]")
    try:
        import torch
        if not torch.cuda.is_available():
            print("  Skipping (no CUDA)")
            return

        torch.cuda.reset_peak_memory_stats()
        from ultralytics import YOLO

        teacher = YOLO("yolov8s.pt").model.cuda().half().eval()
        student = YOLO("yolov8n.pt").model.cuda().train()

        dummy = torch.randn(4, 3, 640, 640, device="cuda")
        with torch.no_grad():
            teacher(dummy.half())
        student(dummy)

        peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"  Peak GPU memory (batch=4, imgsz=640): {peak:.0f} MB")

        total = torch.cuda.get_device_properties(0).total_mem / (1024 ** 2)
        remaining = total - peak
        print(f"  Remaining: {remaining:.0f} MB of {total:.0f} MB")

        if remaining < 500:
            print("    [WARN] Very tight! Use batch=2 or imgsz=416")
        else:
            print("    [OK] Should be enough for training")

        del teacher, student, dummy
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"  Memory estimate failed: {e}")
        print("  [INFO] This is fine — actual training handles OOM gracefully")


def main():
    print("=" * 60)
    print("  Jetson Orin Nano — Environment Verification")
    print("=" * 60)

    print("\n[System]")
    check_python()
    check_system_memory()
    check_swap()
    check_jetson_power_mode()

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
    quick_training_memory_estimate()

    print("\n" + "=" * 60)
    print("  Setup verification complete!")
    print("  To start KD training:")
    print("    python train_distill.py --config configs/distill_config_jetson.yaml")
    print("=" * 60)


if __name__ == "__main__":
    main()


