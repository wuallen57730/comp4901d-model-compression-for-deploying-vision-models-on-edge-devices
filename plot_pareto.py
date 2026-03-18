"""
Pareto Frontier Analysis and Visualization (Milestone 4)

Reads benchmark results and plots:
1. Accuracy (mAP50-95) vs Latency (ms) scatter plot with Pareto frontier
2. Accuracy vs Model Size comparison bar chart
3. FPS comparison bar chart
4. Memory usage comparison bar chart

Usage:
    python plot_pareto.py --results runs/benchmark/benchmark_results.json
    python plot_pareto.py --results runs/benchmark/benchmark_results.json --output runs/plots
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# Use non-interactive backend if no display
matplotlib.use("Agg")

# Color palette for different model variants
COLORS = {
    "Baseline_YOLOv8n_FP32_PyTorch": "#2196F3",       # Blue
    "Baseline_YOLOv8n_FP16_TensorRT": "#4CAF50",      # Green
    "Distilled_Student_FP32_PyTorch": "#FF9800",       # Orange
    "Distilled_Student_FP16_TensorRT": "#9C27B0",     # Purple
    "Distilled_Student_INT8_TensorRT": "#F44336",      # Red
}

MARKERS = {
    "Baseline_YOLOv8n_FP32_PyTorch": "o",
    "Baseline_YOLOv8n_FP16_TensorRT": "s",
    "Distilled_Student_FP32_PyTorch": "^",
    "Distilled_Student_FP16_TensorRT": "D",
    "Distilled_Student_INT8_TensorRT": "*",
}

SHORT_NAMES = {
    "Baseline_YOLOv8n_FP32_PyTorch": "Baseline\n(FP32 PyTorch)",
    "Baseline_YOLOv8n_FP16_TensorRT": "Baseline\n(FP16 TRT)",
    "Distilled_Student_FP32_PyTorch": "Distilled\n(FP32 PyTorch)",
    "Distilled_Student_FP16_TensorRT": "Distilled\n(FP16 TRT)",
    "Distilled_Student_INT8_TensorRT": "Distilled\n(INT8 TRT)",
}


def compute_pareto_frontier(points):
    """
    Compute Pareto frontier for maximizing accuracy and minimizing latency.

    Args:
        points: List of (latency, accuracy, name) tuples.

    Returns:
        List of points on the Pareto frontier, sorted by latency.
    """
    # Sort by latency (ascending)
    sorted_pts = sorted(points, key=lambda p: p[0])

    frontier = []
    max_acc = -float("inf")

    # Sweep from lowest latency to highest
    # A point is Pareto-optimal if no other point has both lower latency AND higher accuracy
    for lat, acc, name in sorted_pts:
        if acc >= max_acc:
            frontier.append((lat, acc, name))
            max_acc = acc

    return frontier


def plot_pareto(results: dict, output_dir: Path):
    """Create the Pareto frontier plot: mAP vs Latency."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    points = []
    for name, r in results.items():
        lat = r["latency"]["avg_ms"]
        acc = r.get("mAP50-95", 0)
        color = COLORS.get(name, "#666666")
        marker = MARKERS.get(name, "o")
        short = SHORT_NAMES.get(name, name)

        ax.scatter(lat, acc, c=color, marker=marker, s=200, zorder=5,
                   edgecolors="black", linewidths=0.5, label=short.replace("\n", " "))

        # Annotate
        ax.annotate(
            short,
            (lat, acc),
            textcoords="offset points",
            xytext=(10, 10),
            fontsize=8,
            ha="left",
        )

        points.append((lat, acc, name))

    # Draw Pareto frontier
    if len(points) >= 2:
        frontier = compute_pareto_frontier(points)
        if len(frontier) >= 2:
            f_lats = [p[0] for p in frontier]
            f_accs = [p[1] for p in frontier]
            ax.plot(f_lats, f_accs, "r--", alpha=0.5, linewidth=2, label="Pareto Frontier")

    ax.set_xlabel("Latency (ms)", fontsize=14)
    ax.set_ylabel("mAP50-95", fontsize=14)
    ax.set_title("Accuracy vs Latency Trade-off (Pareto Analysis)", fontsize=16)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)

    # Add arrow indicating optimization direction
    ax.annotate(
        "Better",
        xy=(0.05, 0.95),
        xycoords="axes fraction",
        fontsize=10,
        color="green",
        ha="left",
        va="top",
        arrowprops=dict(arrowstyle="->", color="green"),
        xytext=(0.15, 0.85),
    )

    plt.tight_layout()
    save_path = output_dir / "pareto_accuracy_vs_latency.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_fps_comparison(results: dict, output_dir: Path):
    """Bar chart comparing FPS across model variants."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    names = list(results.keys())
    fps_values = [results[n]["fps"] for n in names]
    colors = [COLORS.get(n, "#666666") for n in names]
    short_names = [SHORT_NAMES.get(n, n) for n in names]

    bars = ax.bar(range(len(names)), fps_values, color=colors, edgecolor="black", linewidth=0.5)

    # Add value labels on bars
    for bar, val in zip(bars, fps_values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(short_names, fontsize=9)
    ax.set_ylabel("FPS (Frames Per Second)", fontsize=14)
    ax.set_title("Inference Throughput Comparison", fontsize=16)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    save_path = output_dir / "fps_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_memory_comparison(results: dict, output_dir: Path):
    """Bar chart comparing peak memory usage."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    names = list(results.keys())
    mem_values = [results[n].get("peak_memory_mb", 0) for n in names]
    colors = [COLORS.get(n, "#666666") for n in names]
    short_names = [SHORT_NAMES.get(n, n) for n in names]

    bars = ax.bar(range(len(names)), mem_values, color=colors, edgecolor="black", linewidth=0.5)

    for bar, val in zip(bars, mem_values):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{val:.0f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(short_names, fontsize=9)
    ax.set_ylabel("Peak GPU Memory (MB)", fontsize=14)
    ax.set_title("Memory Footprint Comparison", fontsize=16)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    save_path = output_dir / "memory_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_accuracy_comparison(results: dict, output_dir: Path):
    """Bar chart comparing mAP50-95 accuracy."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    names = list(results.keys())
    acc_values = [results[n].get("mAP50-95", 0) for n in names]
    colors = [COLORS.get(n, "#666666") for n in names]
    short_names = [SHORT_NAMES.get(n, n) for n in names]

    bars = ax.bar(range(len(names)), acc_values, color=colors, edgecolor="black", linewidth=0.5)

    for bar, val in zip(bars, acc_values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{val:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(short_names, fontsize=9)
    ax.set_ylabel("mAP50-95", fontsize=14)
    ax.set_title("Detection Accuracy Comparison", fontsize=16)
    ax.grid(True, axis="y", alpha=0.3)

    # Set y-axis to start slightly below minimum
    if acc_values:
        min_acc = min(v for v in acc_values if v > 0) if any(v > 0 for v in acc_values) else 0
        ax.set_ylim(max(0, min_acc - 0.05), max(acc_values) + 0.05)

    plt.tight_layout()
    save_path = output_dir / "accuracy_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def generate_summary_table(results: dict, output_dir: Path):
    """Generate a Markdown summary table."""
    lines = [
        "# COMP 4901D - Model Compression Benchmark Results\n",
        "## Summary Table\n",
        "| Variant | mAP50-95 | Latency (ms) | FPS | Memory (MB) | Size (MB) |",
        "|---------|----------|-------------|-----|-------------|-----------|",
    ]

    for name, r in results.items():
        short = SHORT_NAMES.get(name, name).replace("\n", " ")
        lines.append(
            f"| {short} | {r.get('mAP50-95', 0):.4f} | "
            f"{r['latency']['avg_ms']:.2f} | {r['fps']:.1f} | "
            f"{r.get('peak_memory_mb', 0):.0f} | {r['model_size_mb']:.2f} |"
        )

    lines.append("\n## Key Findings\n")
    lines.append("- **Accuracy**: [Fill in after benchmarking]")
    lines.append("- **Speedup**: [Fill in after benchmarking]")
    lines.append("- **Memory Savings**: [Fill in after benchmarking]")
    lines.append("- **Optimal Deployment**: [Fill in after benchmarking]")

    table_path = output_dir / "benchmark_summary.md"
    with open(table_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved: {table_path}")


def main():
    parser = argparse.ArgumentParser(description="Pareto Analysis Visualization")
    parser.add_argument("--results", type=str, default="runs/benchmark/benchmark_results.json",
                        help="Path to benchmark results JSON")
    parser.add_argument("--output", type=str, default="runs/plots",
                        help="Output directory for plots")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  COMP 4901D - Pareto Analysis & Visualization")
    print("=" * 60)

    # Load results
    with open(args.results, "r") as f:
        results = json.load(f)

    print(f"\nLoaded {len(results)} model variants from {args.results}\n")

    # Generate all plots
    print("Generating plots...")
    plot_pareto(results, output_dir)
    plot_fps_comparison(results, output_dir)
    plot_memory_comparison(results, output_dir)
    plot_accuracy_comparison(results, output_dir)
    generate_summary_table(results, output_dir)

    print(f"\nAll plots saved to: {output_dir}")


if __name__ == "__main__":
    main()


