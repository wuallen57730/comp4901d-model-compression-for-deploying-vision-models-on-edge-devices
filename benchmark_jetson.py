"""
Comprehensive Benchmark Script for Jetson Orin NX (Milestone 4)

Benchmarks all model variants and records:
    - mAP50, mAP50-95 (accuracy)
    - Latency (ms) with P50/P95/P99 percentiles
    - FPS (throughput)
    - Peak GPU memory (MB)
    - Model file size (MB)

Model variants to benchmark:
    1. Baseline YOLOv8n (FP32, PyTorch)
    2. Baseline YOLOv8n (FP16, TensorRT)
    3. Distilled Student (FP32, PyTorch)
    4. Distilled Student (FP16, TensorRT)
    5. Distilled Student (INT8, TensorRT)

Usage (on Jetson Orin NX):
    python benchmark_jetson.py --models-dir runs/export --data coco.yaml

    # Quick test with fewer iterations
    python benchmark_jetson.py --iterations 50 --data coco128.yaml
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO


def benchmark_model(
    model_path: str,
    data: str = "coco.yaml",
    imgsz: int = 640,
    batch: int = 1,
    device: str = "0",
    iterations: int = 200,
    warmup: int = 20,
) -> dict:
    """
    Run a complete benchmark on a single model variant.

    Returns dict with accuracy, latency, memory metrics.
    """
    print(f"\n{'=' * 50}")
    print(f"  Benchmarking: {model_path}")
    print(f"{'=' * 50}")

    model = YOLO(model_path)
    result = {
        "model_path": model_path,
        "model_size_mb": os.path.getsize(model_path) / (1024 * 1024) if os.path.exists(model_path) else 0,
    }

    # ---- Accuracy (mAP) ----
    print("  [1/3] Running validation for mAP...")
    try:
        metrics = model.val(data=data, imgsz=imgsz, batch=batch, device=device, verbose=False)
        result["mAP50"] = float(metrics.box.map50)
        result["mAP50-95"] = float(metrics.box.map)
        print(f"    mAP50: {result['mAP50']:.4f}, mAP50-95: {result['mAP50-95']:.4f}")
    except Exception as e:
        print(f"    Validation failed: {e}")
        result["mAP50"] = 0.0
        result["mAP50-95"] = 0.0

    # ---- Latency ----
    print(f"  [2/3] Benchmarking latency ({iterations} iterations)...")
    use_cuda = device != "cpu" and torch.cuda.is_available()
    dev = torch.device(f"cuda:{device}" if use_cuda else "cpu")

    dummy = torch.randn(batch, 3, imgsz, imgsz, device=dev)

    # Try to get the inner model for direct benchmarking
    try:
        inner_model = model.model.to(dev)
        inner_model.eval()

        # Warmup
        with torch.no_grad():
            for _ in range(warmup):
                inner_model(dummy)

        if use_cuda:
            torch.cuda.synchronize()

        # Measure
        latencies = []
        with torch.no_grad():
            for _ in range(iterations):
                if use_cuda:
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                inner_model(dummy)
                if use_cuda:
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                latencies.append((t1 - t0) * 1000)

        latencies = np.array(latencies)
        result["latency"] = {
            "avg_ms": float(np.mean(latencies)),
            "p50_ms": float(np.percentile(latencies, 50)),
            "p95_ms": float(np.percentile(latencies, 95)),
            "p99_ms": float(np.percentile(latencies, 99)),
            "std_ms": float(np.std(latencies)),
        }
        result["fps"] = float(1000.0 / np.mean(latencies))
    except Exception as e:
        print(f"    Latency benchmark failed: {e}")
        result["latency"] = {"avg_ms": 0, "p50_ms": 0, "p95_ms": 0, "p99_ms": 0, "std_ms": 0}
        result["fps"] = 0

    print(f"    Avg: {result['latency']['avg_ms']:.2f} ms, FPS: {result['fps']:.1f}")

    # ---- Memory ----
    print("  [3/3] Measuring memory...")
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            inner_model(dummy)
        result["peak_memory_mb"] = float(torch.cuda.max_memory_allocated() / (1024 ** 2))
        print(f"    Peak GPU memory: {result['peak_memory_mb']:.1f} MB")
    else:
        result["peak_memory_mb"] = 0
        print("    GPU not available, skipping memory measurement.")

    return result


def main():
    parser = argparse.ArgumentParser(description="Comprehensive Model Benchmarking")
    parser.add_argument("--data", type=str, default="coco.yaml",
                        help="Dataset config for mAP evaluation")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Input image size")
    parser.add_argument("--device", type=str, default="0",
                        help="Device for benchmarking")
    parser.add_argument("--iterations", type=int, default=200,
                        help="Latency benchmark iterations")
    parser.add_argument("--output", type=str, default="runs/benchmark",
                        help="Output directory for results")

    # Model paths -- provide each variant explicitly
    parser.add_argument("--baseline-pt", type=str, default="yolov8n.pt",
                        help="Baseline PyTorch model (YOLOv8n pre-trained)")
    parser.add_argument("--baseline-engine-fp16", type=str, default="",
                        help="Baseline TensorRT FP16 engine path")
    parser.add_argument("--distilled-pt", type=str, default="",
                        help="Distilled student PyTorch model path")
    parser.add_argument("--distilled-engine-fp16", type=str, default="",
                        help="Distilled student TensorRT FP16 engine path")
    parser.add_argument("--distilled-engine-int8", type=str, default="",
                        help="Distilled student TensorRT INT8 engine path")

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  COMP 4901D - Comprehensive Model Benchmarking")
    print("=" * 60)

    # Collect all model variants
    models = {}
    if args.baseline_pt:
        models["Baseline_YOLOv8n_FP32_PyTorch"] = args.baseline_pt
    if args.baseline_engine_fp16:
        models["Baseline_YOLOv8n_FP16_TensorRT"] = args.baseline_engine_fp16
    if args.distilled_pt:
        models["Distilled_Student_FP32_PyTorch"] = args.distilled_pt
    if args.distilled_engine_fp16:
        models["Distilled_Student_FP16_TensorRT"] = args.distilled_engine_fp16
    if args.distilled_engine_int8:
        models["Distilled_Student_INT8_TensorRT"] = args.distilled_engine_int8

    print(f"\nModels to benchmark: {len(models)}")
    for name, path in models.items():
        print(f"  - {name}: {path}")

    # Benchmark each model
    all_results = {}
    for name, path in models.items():
        if not os.path.exists(path) and not path.endswith(".pt"):
            print(f"\n  Skipping {name}: {path} not found")
            continue
        try:
            result = benchmark_model(
                path,
                data=args.data,
                imgsz=args.imgsz,
                device=args.device,
                iterations=args.iterations,
            )
            result["variant_name"] = name
            all_results[name] = result
        except Exception as e:
            print(f"\n  Error benchmarking {name}: {e}")

    # Save results
    results_path = output_dir / "benchmark_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Print summary table
    print(f"\n\n{'=' * 90}")
    print(f"  BENCHMARK SUMMARY")
    print(f"{'=' * 90}")
    header = f"{'Variant':<40} {'mAP50-95':>10} {'Latency(ms)':>12} {'FPS':>8} {'Memory(MB)':>12} {'Size(MB)':>10}"
    print(header)
    print("-" * 90)
    for name, r in all_results.items():
        print(
            f"{name:<40} "
            f"{r.get('mAP50-95', 0):>10.4f} "
            f"{r['latency']['avg_ms']:>12.2f} "
            f"{r['fps']:>8.1f} "
            f"{r['peak_memory_mb']:>12.1f} "
            f"{r['model_size_mb']:>10.2f}"
        )
    print(f"{'=' * 90}")
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()

