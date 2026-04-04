"""
Complete Export + Benchmark + Pareto Analysis Pipeline

Runs all steps in sequence:
1. Export missing INT8 TensorRT engines
2. Benchmark ALL model variants (FP32 PyTorch, FP16 TRT, INT8 TRT)
3. Generate consolidated results and Pareto analysis plots
"""

import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_YAML = "coco.yaml"
IMGSZ = 640
DEVICE = "0"
ITERATIONS = 200
WARMUP = 20

OUTPUT_DIR = Path("runs/full_benchmark")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ALL_MODELS = {
    "Baseline_FP32_PyTorch": "yolov8n.pt",
    "Baseline_FP16_TensorRT": "yolov8n.engine",
    "Baseline_INT8_TensorRT": "runs/quantize/baseline_int8_int8.engine",
    "KD_s2n_FP32_PyTorch": "best_s2n.pt",
    "KD_s2n_FP16_TensorRT": "best_s2n.engine",
    "KD_s2n_INT8_TensorRT": "runs/quantize/best_s2n_int8.engine",
    "KD_m2n_FP32_PyTorch": "best_m2n.pt",
    "KD_m2n_FP16_TensorRT": "best_m2n.engine",
    "KD_m2n_INT8_TensorRT": "runs/quantize/best_m2n_int8.engine",
    "KD_l2n_FP32_PyTorch": "best_l2n.pt",
    "KD_l2n_FP16_TensorRT": "best_l2n.engine",
    "KD_l2n_INT8_TensorRT": "runs/quantize/best_l2n_int8.engine",
}


def export_int8_engine(weights_path, output_path):
    """Export a .pt model to INT8 TensorRT engine."""
    if os.path.exists(output_path):
        print(f"  [SKIP] {output_path} already exists")
        return output_path

    print(f"  [EXPORT] {weights_path} -> INT8 TensorRT ...")
    model = YOLO(weights_path)
    engine_path = model.export(
        format="engine",
        imgsz=IMGSZ,
        batch=1,
        int8=True,
        data=DATA_YAML,
        device=DEVICE,
    )
    engine_path = Path(str(engine_path))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if engine_path.resolve() != Path(output_path).resolve():
        os.replace(str(engine_path), output_path)

    print(f"  [DONE] Saved to {output_path}")
    return output_path


def benchmark_single(model_path, variant_name):
    """Benchmark a single model variant: mAP + latency + memory."""
    print(f"\n{'='*60}")
    print(f"  Benchmarking: {variant_name}")
    print(f"  Path: {model_path}")
    print(f"{'='*60}")

    if not os.path.exists(model_path):
        print(f"  [SKIP] File not found: {model_path}")
        return None

    model = YOLO(model_path)
    result = {
        "variant_name": variant_name,
        "model_path": model_path,
        "model_size_mb": os.path.getsize(model_path) / (1024 * 1024),
    }

    # --- mAP ---
    print("  [1/3] Validation (mAP)...")
    try:
        metrics = model.val(data=DATA_YAML, imgsz=IMGSZ, batch=1, device=DEVICE, verbose=False)
        result["mAP50"] = float(metrics.box.map50)
        result["mAP50-95"] = float(metrics.box.map)
        print(f"    mAP50={result['mAP50']:.4f}  mAP50-95={result['mAP50-95']:.4f}")
    except Exception as e:
        print(f"    Validation FAILED: {e}")
        result["mAP50"] = 0.0
        result["mAP50-95"] = 0.0

    # --- Latency ---
    print(f"  [2/3] Latency ({ITERATIONS} iters)...")
    use_cuda = torch.cuda.is_available()
    dev = torch.device(f"cuda:{DEVICE}" if use_cuda else "cpu")

    dummy = torch.randn(1, 3, IMGSZ, IMGSZ, device=dev)
    try:
        inner = model.model.to(dev)
        inner.eval()

        with torch.no_grad():
            for _ in range(WARMUP):
                inner(dummy)
        if use_cuda:
            torch.cuda.synchronize()

        latencies = []
        with torch.no_grad():
            for _ in range(ITERATIONS):
                if use_cuda:
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                inner(dummy)
                if use_cuda:
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                latencies.append((t1 - t0) * 1000)

        lat = np.array(latencies)
        result["latency"] = {
            "avg_ms": float(np.mean(lat)),
            "p50_ms": float(np.percentile(lat, 50)),
            "p95_ms": float(np.percentile(lat, 95)),
            "p99_ms": float(np.percentile(lat, 99)),
            "std_ms": float(np.std(lat)),
        }
        result["fps"] = float(1000.0 / np.mean(lat))
        print(f"    Avg={result['latency']['avg_ms']:.2f}ms  FPS={result['fps']:.1f}")
    except Exception as e:
        print(f"    Latency FAILED: {e}")
        result["latency"] = {"avg_ms": 0, "p50_ms": 0, "p95_ms": 0, "p99_ms": 0, "std_ms": 0}
        result["fps"] = 0

    # --- Memory ---
    print("  [3/3] Memory...")
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            try:
                inner(dummy)
            except Exception:
                pass
        result["peak_memory_mb"] = float(torch.cuda.max_memory_allocated() / (1024 ** 2))
        print(f"    Peak GPU memory: {result['peak_memory_mb']:.1f} MB")
    else:
        result["peak_memory_mb"] = 0

    return result


def generate_pareto_plots(results, output_dir):
    """Generate Pareto and comparison plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    valid = {k: v for k, v in results.items() if v.get("mAP50-95", 0) > 0 and v.get("fps", 0) > 0}
    if not valid:
        print("  No valid results for plotting")
        return

    # Color scheme by model family
    family_colors = {
        "Baseline": "#2196F3",
        "KD_s2n": "#FF9800",
        "KD_m2n": "#4CAF50",
        "KD_l2n": "#9C27B0",
    }
    format_markers = {
        "FP32_PyTorch": "o",
        "FP16_TensorRT": "s",
        "INT8_TensorRT": "^",
    }

    def get_family(name):
        for f in ["Baseline", "KD_s2n", "KD_m2n", "KD_l2n"]:
            if name.startswith(f):
                return f
        return "Other"

    def get_format(name):
        for f in ["FP32_PyTorch", "FP16_TensorRT", "INT8_TensorRT"]:
            if f in name:
                return f
        return "Other"

    # --- 1) Pareto: mAP vs Latency ---
    fig, ax = plt.subplots(figsize=(12, 8))
    points = []
    for name, r in valid.items():
        lat = r["latency"]["avg_ms"]
        acc = r["mAP50-95"]
        fam = get_family(name)
        fmt = get_format(name)
        c = family_colors.get(fam, "#666666")
        m = format_markers.get(fmt, "o")

        ax.scatter(lat, acc, c=c, marker=m, s=200, zorder=5,
                   edgecolors="black", linewidths=0.5)
        ax.annotate(name.replace("_", "\n"), (lat, acc),
                    textcoords="offset points", xytext=(8, 8), fontsize=6, ha="left")
        points.append((lat, acc, name))

    # Pareto frontier (minimize latency, maximize accuracy)
    sorted_pts = sorted(points, key=lambda p: p[0])
    frontier = []
    max_acc = -float("inf")
    for lat, acc, name in sorted_pts:
        if acc >= max_acc:
            frontier.append((lat, acc, name))
            max_acc = acc
    if len(frontier) >= 2:
        ax.plot([p[0] for p in frontier], [p[1] for p in frontier],
                "r--", alpha=0.6, linewidth=2, label="Pareto Frontier")

    # Legend entries
    for fam, c in family_colors.items():
        ax.scatter([], [], c=c, s=100, label=fam.replace("_", " "))
    for fmt, m in format_markers.items():
        ax.scatter([], [], c="gray", marker=m, s=100, label=fmt.replace("_", " "))

    ax.set_xlabel("Latency (ms)", fontsize=14)
    ax.set_ylabel("mAP50-95", fontsize=14)
    ax.set_title("Accuracy vs Latency Trade-off (Pareto Analysis)\nJetson Orin NX", fontsize=14)
    ax.legend(loc="lower left", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "pareto_accuracy_vs_latency.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: pareto_accuracy_vs_latency.png")

    # --- 2) Pareto: mAP vs Model Size ---
    fig, ax = plt.subplots(figsize=(12, 8))
    for name, r in valid.items():
        sz = r["model_size_mb"]
        acc = r["mAP50-95"]
        fam = get_family(name)
        fmt = get_format(name)
        c = family_colors.get(fam, "#666666")
        m = format_markers.get(fmt, "o")
        ax.scatter(sz, acc, c=c, marker=m, s=200, zorder=5,
                   edgecolors="black", linewidths=0.5)
        ax.annotate(name.replace("_", "\n"), (sz, acc),
                    textcoords="offset points", xytext=(8, 8), fontsize=6, ha="left")

    for fam, c in family_colors.items():
        ax.scatter([], [], c=c, s=100, label=fam.replace("_", " "))
    for fmt, m in format_markers.items():
        ax.scatter([], [], c="gray", marker=m, s=100, label=fmt.replace("_", " "))

    ax.set_xlabel("Model Size (MB)", fontsize=14)
    ax.set_ylabel("mAP50-95", fontsize=14)
    ax.set_title("Accuracy vs Model Size\nJetson Orin NX", fontsize=14)
    ax.legend(loc="lower left", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "pareto_accuracy_vs_size.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: pareto_accuracy_vs_size.png")

    # --- 3) Grouped bar chart: mAP by model x format ---
    families = ["Baseline", "KD_s2n", "KD_m2n", "KD_l2n"]
    formats = ["FP32_PyTorch", "FP16_TensorRT", "INT8_TensorRT"]
    x = np.arange(len(families))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 7))
    for i, fmt in enumerate(formats):
        vals = []
        for fam in families:
            key = f"{fam}_{fmt}"
            vals.append(valid[key]["mAP50-95"] if key in valid else 0)
        bars = ax.bar(x + i * width, vals, width, label=fmt.replace("_", " "),
                      edgecolor="black", linewidth=0.5)
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                        f"{v:.4f}", ha="center", va="bottom", fontsize=7, rotation=45)
    ax.set_xticks(x + width)
    ax.set_xticklabels([f.replace("_", " ") for f in families], fontsize=10)
    ax.set_ylabel("mAP50-95", fontsize=14)
    ax.set_title("Detection Accuracy by Model & Precision", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    if any(valid.get(f"{fam}_{fmt}", {}).get("mAP50-95", 0) > 0
           for fam in families for fmt in formats):
        all_vals = [valid.get(f"{fam}_{fmt}", {}).get("mAP50-95", 0)
                    for fam in families for fmt in formats]
        nonzero = [v for v in all_vals if v > 0]
        if nonzero:
            ax.set_ylim(min(nonzero) - 0.02, max(nonzero) + 0.02)
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_grouped_bar.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: accuracy_grouped_bar.png")

    # --- 4) FPS comparison bar chart ---
    fig, ax = plt.subplots(figsize=(14, 7))
    names_sorted = sorted(valid.keys(), key=lambda n: valid[n]["fps"], reverse=True)
    fps_vals = [valid[n]["fps"] for n in names_sorted]
    colors = [family_colors.get(get_family(n), "#666666") for n in names_sorted]
    short_labels = [n.replace("_", "\n") for n in names_sorted]

    bars = ax.bar(range(len(names_sorted)), fps_vals, color=colors,
                  edgecolor="black", linewidth=0.5)
    for bar, v in zip(bars, fps_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{v:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.set_xticks(range(len(names_sorted)))
    ax.set_xticklabels(short_labels, fontsize=7)
    ax.set_ylabel("FPS", fontsize=14)
    ax.set_title("Inference Throughput (FPS) Comparison\nJetson Orin NX", fontsize=14)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "fps_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: fps_comparison.png")

    # --- 5) Latency comparison ---
    fig, ax = plt.subplots(figsize=(14, 7))
    names_sorted = sorted(valid.keys(), key=lambda n: valid[n]["latency"]["avg_ms"])
    lat_vals = [valid[n]["latency"]["avg_ms"] for n in names_sorted]
    p95_vals = [valid[n]["latency"]["p95_ms"] for n in names_sorted]
    colors = [family_colors.get(get_family(n), "#666666") for n in names_sorted]
    short_labels = [n.replace("_", "\n") for n in names_sorted]

    bars = ax.bar(range(len(names_sorted)), lat_vals, color=colors,
                  edgecolor="black", linewidth=0.5)
    ax.errorbar(range(len(names_sorted)), lat_vals,
                yerr=[np.array(lat_vals) - np.array(lat_vals),
                      np.array(p95_vals) - np.array(lat_vals)],
                fmt="none", ecolor="black", capsize=3)
    for bar, v in zip(bars, lat_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f"{v:.2f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.set_xticks(range(len(names_sorted)))
    ax.set_xticklabels(short_labels, fontsize=7)
    ax.set_ylabel("Latency (ms)", fontsize=14)
    ax.set_title("Inference Latency (avg + P95 error bar)\nJetson Orin NX", fontsize=14)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "latency_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: latency_comparison.png")

    # --- 6) Memory comparison ---
    fig, ax = plt.subplots(figsize=(14, 7))
    names_sorted = sorted(valid.keys(), key=lambda n: valid[n].get("peak_memory_mb", 0))
    mem_vals = [valid[n].get("peak_memory_mb", 0) for n in names_sorted]
    colors = [family_colors.get(get_family(n), "#666666") for n in names_sorted]
    short_labels = [n.replace("_", "\n") for n in names_sorted]

    bars = ax.bar(range(len(names_sorted)), mem_vals, color=colors,
                  edgecolor="black", linewidth=0.5)
    for bar, v in zip(bars, mem_vals):
        if v > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{v:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.set_xticks(range(len(names_sorted)))
    ax.set_xticklabels(short_labels, fontsize=7)
    ax.set_ylabel("Peak GPU Memory (MB)", fontsize=14)
    ax.set_title("Memory Footprint Comparison\nJetson Orin NX", fontsize=14)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "memory_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: memory_comparison.png")


def generate_markdown_report(results, output_dir):
    """Generate a comprehensive Markdown report."""
    output_dir = Path(output_dir)
    valid = {k: v for k, v in results.items() if v.get("mAP50-95", 0) > 0}

    lines = [
        "# COMP 4901D - Full Benchmark Report",
        f"## Jetson Orin NX | TensorRT 10.3 | CUDA 12.6\n",
        "## Complete Results Table\n",
        "| Variant | mAP50 | mAP50-95 | Latency (ms) | FPS | Memory (MB) | Size (MB) |",
        "|---------|-------|----------|-------------|-----|-------------|-----------|",
    ]

    for name in sorted(valid.keys()):
        r = valid[name]
        lines.append(
            f"| {name} | {r.get('mAP50', 0):.4f} | {r.get('mAP50-95', 0):.4f} | "
            f"{r['latency']['avg_ms']:.2f} | {r['fps']:.1f} | "
            f"{r.get('peak_memory_mb', 0):.1f} | {r['model_size_mb']:.2f} |"
        )

    # Analysis section
    baseline_fp32 = valid.get("Baseline_FP32_PyTorch", {})
    baseline_map = baseline_fp32.get("mAP50-95", 0)
    baseline_lat = baseline_fp32.get("latency", {}).get("avg_ms", 0)
    baseline_fps = baseline_fp32.get("fps", 0)

    lines.append("\n## Key Findings\n")

    if baseline_map > 0:
        lines.append(f"### Baseline Performance")
        lines.append(f"- Baseline YOLOv8n FP32: mAP50-95 = {baseline_map:.4f}, "
                      f"Latency = {baseline_lat:.2f}ms, FPS = {baseline_fps:.1f}\n")

        lines.append(f"### Knowledge Distillation Impact (FP32)")
        for kd in ["KD_s2n", "KD_m2n", "KD_l2n"]:
            key = f"{kd}_FP32_PyTorch"
            if key in valid:
                r = valid[key]
                delta = r["mAP50-95"] - baseline_map
                lines.append(f"- {kd}: mAP50-95 = {r['mAP50-95']:.4f} "
                              f"(Δ = {delta:+.4f})")

        lines.append(f"\n### Quantization Speedup")
        for family in ["Baseline", "KD_s2n", "KD_m2n", "KD_l2n"]:
            fp32_key = f"{family}_FP32_PyTorch"
            fp16_key = f"{family}_FP16_TensorRT"
            int8_key = f"{family}_INT8_TensorRT"
            if fp32_key in valid:
                fp32_fps = valid[fp32_key]["fps"]
                parts = [f"{family} FP32: {fp32_fps:.1f} FPS"]
                if fp16_key in valid:
                    fp16_fps = valid[fp16_key]["fps"]
                    parts.append(f"FP16: {fp16_fps:.1f} FPS ({fp16_fps/fp32_fps:.2f}x)")
                if int8_key in valid:
                    int8_fps = valid[int8_key]["fps"]
                    parts.append(f"INT8: {int8_fps:.1f} FPS ({int8_fps/fp32_fps:.2f}x)")
                lines.append(f"- {' | '.join(parts)}")

    with open(output_dir / "full_benchmark_report.md", "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved: full_benchmark_report.md")


def main():
    print("=" * 70)
    print("  COMP 4901D - Full Export + Benchmark + Pareto Pipeline")
    print("  Jetson Orin NX")
    print("=" * 70)

    # === Step 1: Export missing INT8 engines ===
    print("\n" + "=" * 70)
    print("  STEP 1: Export Missing INT8 TensorRT Engines")
    print("=" * 70)

    int8_exports = {
        "best_s2n.pt": "runs/quantize/best_s2n_int8.engine",
        "best_m2n.pt": "runs/quantize/best_m2n_int8.engine",
        "best_l2n.pt": "runs/quantize/best_l2n_int8.engine",
    }
    for weights, output in int8_exports.items():
        if os.path.exists(weights):
            export_int8_engine(weights, output)
        else:
            print(f"  [SKIP] Weights not found: {weights}")

    # === Step 2: Benchmark ALL variants ===
    print("\n" + "=" * 70)
    print("  STEP 2: Comprehensive Benchmark")
    print("=" * 70)

    all_results = {}
    for variant_name, model_path in ALL_MODELS.items():
        if not os.path.exists(model_path):
            print(f"\n  [SKIP] {variant_name}: {model_path} not found")
            continue
        try:
            result = benchmark_single(model_path, variant_name)
            if result:
                all_results[variant_name] = result
        except Exception as e:
            print(f"\n  [ERROR] {variant_name}: {e}")

        # Save intermediate results after each model
        with open(OUTPUT_DIR / "benchmark_results.json", "w") as f:
            json.dump(all_results, f, indent=2)

    # === Step 3: Generate Plots and Report ===
    print("\n" + "=" * 70)
    print("  STEP 3: Pareto Analysis & Visualization")
    print("=" * 70)

    generate_pareto_plots(all_results, OUTPUT_DIR)
    generate_markdown_report(all_results, OUTPUT_DIR)

    # Print final summary
    print("\n" + "=" * 90)
    print("  FINAL SUMMARY")
    print("=" * 90)
    header = f"{'Variant':<35} {'mAP50-95':>10} {'Latency(ms)':>12} {'FPS':>8} {'Mem(MB)':>10} {'Size(MB)':>10}"
    print(header)
    print("-" * 90)
    for name in sorted(all_results.keys()):
        r = all_results[name]
        print(
            f"{name:<35} "
            f"{r.get('mAP50-95', 0):>10.4f} "
            f"{r['latency']['avg_ms']:>12.2f} "
            f"{r['fps']:>8.1f} "
            f"{r.get('peak_memory_mb', 0):>10.1f} "
            f"{r['model_size_mb']:>10.2f}"
        )
    print("=" * 90)
    print(f"\nAll results saved to: {OUTPUT_DIR}")
    print(f"  - benchmark_results.json")
    print(f"  - full_benchmark_report.md")
    print(f"  - pareto_accuracy_vs_latency.png")
    print(f"  - pareto_accuracy_vs_size.png")
    print(f"  - accuracy_grouped_bar.png")
    print(f"  - fps_comparison.png")
    print(f"  - latency_comparison.png")
    print(f"  - memory_comparison.png")


if __name__ == "__main__":
    main()
