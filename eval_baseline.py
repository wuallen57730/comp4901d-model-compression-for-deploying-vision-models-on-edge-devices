"""
Baseline Evaluation Script (Milestone 1)

Evaluates the pre-trained YOLOv8n model on COCO val2017 to establish
baseline metrics before any compression.

Recorded metrics:
    - mAP50, mAP50-95
    - Inference latency (ms per image)
    - Model file size (MB)
    - Parameter count
    - FLOPs

Usage:
    python eval_baseline.py                         # Full evaluation on COCO val2017
    python eval_baseline.py --data coco128.yaml     # Quick debug with COCO128
    python eval_baseline.py --model yolov8s.pt      # Evaluate a different model
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
from ultralytics import YOLO


def get_model_size_mb(model_path: str) -> float:
    """Get model file size in MB."""
    return os.path.getsize(model_path) / (1024 * 1024)


def count_parameters(model: torch.nn.Module) -> int:
    """Count total number of parameters."""
    return sum(p.numel() for p in model.parameters())


def benchmark_latency(
    model: YOLO,
    imgsz: int = 640,
    iterations: int = 100,
    warmup: int = 10,
    device: str = "0",
) -> dict:
    """
    Benchmark inference latency by running dummy inputs.

    Returns:
        Dict with avg_latency_ms, fps, p50_ms, p95_ms, p99_ms.
    """
    import numpy as np

    # Create dummy input
    use_cuda = device != "cpu" and torch.cuda.is_available()
    dev = torch.device(f"cuda:{device}" if use_cuda else "cpu")

    dummy = torch.randn(1, 3, imgsz, imgsz, device=dev)

    # Move model to device
    yolo_model = model.model.to(dev)
    yolo_model.eval()

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            yolo_model(dummy)

    if use_cuda:
        torch.cuda.synchronize()

    # Benchmark
    latencies = []
    with torch.no_grad():
        for _ in range(iterations):
            if use_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            yolo_model(dummy)
            if use_cuda:
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000)  # ms

    latencies = np.array(latencies)
    return {
        "avg_latency_ms": float(np.mean(latencies)),
        "fps": float(1000.0 / np.mean(latencies)),
        "p50_ms": float(np.percentile(latencies, 50)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
        "std_ms": float(np.std(latencies)),
    }


def main():
    parser = argparse.ArgumentParser(description="Baseline Model Evaluation")
    parser.add_argument("--model", type=str, default="yolov8n.pt",
                        help="Model weights path (default: yolov8n.pt)")
    parser.add_argument("--data", type=str, default="coco.yaml",
                        help="Dataset config (default: coco.yaml, use coco128.yaml for debug)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Input image size (default: 640)")
    parser.add_argument("--batch", type=int, default=16,
                        help="Batch size for validation (default: 16)")
    parser.add_argument("--device", type=str, default="0",
                        help="Device: '0' for GPU, 'cpu' for CPU")
    parser.add_argument("--iterations", type=int, default=100,
                        help="Number of iterations for latency benchmark")
    parser.add_argument("--output", type=str, default="runs/baseline",
                        help="Output directory for results")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  COMP 4901D - Baseline Model Evaluation")
    print("=" * 60)

    # ---- Load Model ----
    print(f"\n[1/5] Loading model: {args.model}")
    model = YOLO(args.model)
    model_path = args.model

    # ---- Model Info ----
    print(f"\n[2/5] Model Information:")
    param_count = count_parameters(model.model)
    model_size = get_model_size_mb(model_path)
    print(f"  - Parameters:  {param_count:,}")
    print(f"  - Model size:  {model_size:.2f} MB")

    # ---- COCO Validation ----
    print(f"\n[3/5] Running validation on {args.data}...")
    metrics = model.val(
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        plots=True,
        save_json=True,
    )

    map50 = metrics.box.map50
    map50_95 = metrics.box.map
    print(f"  - mAP50:       {map50:.4f}")
    print(f"  - mAP50-95:    {map50_95:.4f}")

    # ---- Latency Benchmark ----
    print(f"\n[4/5] Benchmarking inference latency ({args.iterations} iterations)...")
    latency = benchmark_latency(
        model, imgsz=args.imgsz, iterations=args.iterations, device=args.device
    )
    print(f"  - Avg latency: {latency['avg_latency_ms']:.2f} ms")
    print(f"  - FPS:         {latency['fps']:.1f}")
    print(f"  - P50:         {latency['p50_ms']:.2f} ms")
    print(f"  - P95:         {latency['p95_ms']:.2f} ms")
    print(f"  - P99:         {latency['p99_ms']:.2f} ms")

    # ---- Memory Usage ----
    print(f"\n[5/5] Memory usage:")
    if torch.cuda.is_available() and args.device != "cpu":
        mem_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)
        mem_reserved = torch.cuda.max_memory_reserved() / (1024 ** 2)
        print(f"  - Peak GPU allocated: {mem_allocated:.1f} MB")
        print(f"  - Peak GPU reserved:  {mem_reserved:.1f} MB")
    else:
        mem_allocated = 0
        mem_reserved = 0
        print("  - GPU not available, skipping memory measurement.")

    # ---- Save Results ----
    results = {
        "model": args.model,
        "data": args.data,
        "imgsz": args.imgsz,
        "parameters": param_count,
        "model_size_mb": model_size,
        "mAP50": map50,
        "mAP50-95": map50_95,
        "latency": latency,
        "memory": {
            "peak_allocated_mb": mem_allocated,
            "peak_reserved_mb": mem_reserved,
        },
        "device": args.device,
        "notes": "Baseline evaluation (no compression applied)",
    }

    results_path = output_dir / "baseline_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"  Results saved to: {results_path}")
    print(f"{'=' * 60}")

    # ---- Summary Table ----
    print(f"\n{'Metric':<25} {'Value':<20}")
    print("-" * 45)
    print(f"{'Model':<25} {args.model:<20}")
    print(f"{'Parameters':<25} {param_count:,}")
    print(f"{'Model Size (MB)':<25} {model_size:.2f}")
    print(f"{'mAP50':<25} {map50:.4f}")
    print(f"{'mAP50-95':<25} {map50_95:.4f}")
    print(f"{'Avg Latency (ms)':<25} {latency['avg_latency_ms']:.2f}")
    print(f"{'FPS':<25} {latency['fps']:.1f}")

    return results


if __name__ == "__main__":
    main()


