"""
Final 8-Point Pareto Analysis for COMP 4901D

Combines benchmark results from:
  - 1 Baseline (yolov8n.pt, FP32 pretrained)
  - 3 KD FP32 variants (s→n, m→n, l→n)
  - 1 Quantization-only (yolov8n INT8, no KD)
  - 3 KD + INT8 TensorRT variants (s→n, m→n, l→n)

Usage:
    python3 plot_pareto_final.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


RESULT_FILES = {
    "s2n": "runs/benchmark_s2n_v2/benchmark_results.json",
    "m2n": "runs/benchmark_m2n_v2/benchmark_results.json",
    "l2n": "runs/benchmark_l2n_v2/benchmark_results.json",
    "int8_baseline": "runs/benchmark_int8_baseline/benchmark_results.json",
    "int8_s2n": "runs/benchmark_int8_s2n_v2/benchmark_results.json",
    "int8_m2n": "runs/benchmark_int8_m2n_v2/benchmark_results.json",
    "int8_l2n": "runs/benchmark_int8_l2n_v2/benchmark_results.json",
}

MODEL_DISPLAY = {
    "Baseline_FP32":   {"label": "Baseline (FP32)",       "color": "#2196F3", "marker": "o"},
    "KD_s2n_FP32":     {"label": "KD: s→n (FP32)",        "color": "#4CAF50", "marker": "s"},
    "KD_m2n_FP32":     {"label": "KD: m→n (FP32)",        "color": "#FF9800", "marker": "^"},
    "KD_l2n_FP32":     {"label": "KD: l→n (FP32)",        "color": "#9C27B0", "marker": "D"},
    "Baseline_INT8":   {"label": "Baseline INT8 (Q only)", "color": "#00BCD4", "marker": "h"},
    "KD_s2n_INT8":     {"label": "KD: s→n + INT8",        "color": "#E91E63", "marker": "*"},
    "KD_m2n_INT8":     {"label": "KD: m→n + INT8",        "color": "#F44336", "marker": "P"},
    "KD_l2n_INT8":     {"label": "KD: l→n + INT8",        "color": "#795548", "marker": "X"},
}


def load_and_combine():
    """Load all benchmark JSONs and combine into 7 data points."""
    combined = {}

    # --- Baseline from s2n run (same across all runs) ---
    with open(RESULT_FILES["s2n"]) as f:
        s2n_data = json.load(f)
    combined["Baseline_FP32"] = s2n_data["Baseline_YOLOv8n_FP32_PyTorch"]
    combined["Baseline_FP32"]["display_name"] = "Baseline (FP32)"

    # --- 3 KD FP32 models ---
    for tag, key in [("s2n", "s2n"), ("m2n", "m2n"), ("l2n", "l2n")]:
        with open(RESULT_FILES[key]) as f:
            data = json.load(f)
        entry = data["Distilled_Student_FP32_PyTorch"]
        combined[f"KD_{tag}_FP32"] = entry
        combined[f"KD_{tag}_FP32"]["display_name"] = MODEL_DISPLAY[f"KD_{tag}_FP32"]["label"]

    # --- Baseline INT8 (Quantization only, no KD) ---
    fpath = RESULT_FILES["int8_baseline"]
    if Path(fpath).exists():
        with open(fpath) as f:
            data = json.load(f)
        key = next((k for k in data if "INT8" in k or "int8" in k.lower()), None)
        if key is None:
            key = list(data.keys())[0]
        combined["Baseline_INT8"] = data[key]
        combined["Baseline_INT8"]["display_name"] = MODEL_DISPLAY["Baseline_INT8"]["label"]
    else:
        print(f"  WARNING: {fpath} not found, skipping Baseline_INT8")

    # --- 3 KD INT8 models ---
    for tag in ["s2n", "m2n", "l2n"]:
        fpath = RESULT_FILES[f"int8_{tag}"]
        if not Path(fpath).exists():
            print(f"  WARNING: {fpath} not found, skipping KD_{tag}_INT8")
            continue
        with open(fpath) as f:
            data = json.load(f)
        key = "Distilled_Student_INT8_TensorRT"
        if key not in data:
            for k in data:
                if "INT8" in k or "int8" in k.lower():
                    key = k
                    break
            else:
                print(f"  WARNING: no INT8 entry in {fpath}, keys: {list(data.keys())}")
                continue
        entry = data[key]
        combined[f"KD_{tag}_INT8"] = entry
        combined[f"KD_{tag}_INT8"]["display_name"] = MODEL_DISPLAY[f"KD_{tag}_INT8"]["label"]

    return combined


def compute_pareto_frontier(points):
    """Points on Pareto frontier: minimize latency, maximize accuracy."""
    sorted_pts = sorted(points, key=lambda p: p[0])
    frontier = []
    max_acc = -float("inf")
    for lat, acc, name in sorted_pts:
        if acc >= max_acc:
            frontier.append((lat, acc, name))
            max_acc = acc
    return frontier


def plot_pareto_7pt(combined, output_dir):
    """Scatter plot: mAP50-95 vs Latency with Pareto frontier."""
    fig, ax = plt.subplots(figsize=(12, 8))

    points = []
    for model_key, r in combined.items():
        lat = r["latency"]["avg_ms"]
        acc = r.get("mAP50-95", 0)
        if lat == 0 or acc == 0:
            continue
        style = MODEL_DISPLAY.get(model_key, {"label": model_key, "color": "#666", "marker": "o"})
        ax.scatter(lat, acc, c=style["color"], marker=style["marker"], s=250, zorder=5,
                   edgecolors="black", linewidths=0.8, label=style["label"])
        ax.annotate(style["label"], (lat, acc), textcoords="offset points",
                    xytext=(12, 8), fontsize=8.5, ha="left",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))
        points.append((lat, acc, model_key))

    if len(points) >= 2:
        frontier = compute_pareto_frontier(points)
        if len(frontier) >= 2:
            f_lats = [p[0] for p in frontier]
            f_accs = [p[1] for p in frontier]
            ax.plot(f_lats, f_accs, "r--", alpha=0.6, linewidth=2.5, label="Pareto Frontier")

    ax.set_xlabel("Latency (ms)", fontsize=14, fontweight="bold")
    ax.set_ylabel("mAP50-95", fontsize=14, fontweight="bold")
    n = len(points)
    ax.set_title(f"Accuracy vs Latency Trade-off — {n} Model Variants\n"
                 "(Baseline / KD / Q-only / KD+INT8 on Jetson Orin NX)", fontsize=15, fontweight="bold")
    ax.legend(loc="lower left", fontsize=9, framealpha=0.9,
              ncol=2, labelspacing=1.2, columnspacing=1.5, handletextpad=0.8)
    ax.grid(True, alpha=0.3)

    ax.annotate("← Lower latency, Higher accuracy →\n       (Better)", xy=(0.02, 0.97),
                xycoords="axes fraction", fontsize=9, color="green", ha="left", va="top")

    plt.tight_layout()
    save_path = output_dir / "pareto_accuracy_vs_latency.png"
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_accuracy_bar(combined, output_dir):
    """Bar chart: mAP50-95 for all 7 models."""
    fig, ax = plt.subplots(figsize=(14, 7))

    names = list(combined.keys())
    labels = [MODEL_DISPLAY.get(n, {}).get("label", n) for n in names]
    accs = [combined[n].get("mAP50-95", 0) for n in names]
    colors = [MODEL_DISPLAY.get(n, {}).get("color", "#666") for n in names]

    bars = ax.bar(range(len(names)), accs, color=colors, edgecolor="black", linewidth=0.5, width=0.6)
    for bar, val in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                f"{val:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(labels, fontsize=9, rotation=15, ha="right")
    ax.set_ylabel("mAP50-95", fontsize=13, fontweight="bold")
    ax.set_title(f"Detection Accuracy Comparison — {len(names)} Model Variants", fontsize=15, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)

    if accs:
        valid = [v for v in accs if v > 0]
        if valid:
            ax.set_ylim(min(valid) - 0.02, max(valid) + 0.02)

    plt.tight_layout()
    save_path = output_dir / "accuracy_comparison.png"
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_fps_bar(combined, output_dir):
    """Bar chart: FPS for all 7 models."""
    fig, ax = plt.subplots(figsize=(14, 7))

    names = list(combined.keys())
    labels = [MODEL_DISPLAY.get(n, {}).get("label", n) for n in names]
    fps_vals = [combined[n].get("fps", 0) for n in names]
    colors = [MODEL_DISPLAY.get(n, {}).get("color", "#666") for n in names]

    bars = ax.bar(range(len(names)), fps_vals, color=colors, edgecolor="black", linewidth=0.5, width=0.6)
    for bar, val in zip(bars, fps_vals):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(labels, fontsize=9, rotation=15, ha="right")
    ax.set_ylabel("FPS", fontsize=13, fontweight="bold")
    ax.set_title(f"Inference Throughput — {len(names)} Model Variants", fontsize=15, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    save_path = output_dir / "fps_comparison.png"
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_latency_bar(combined, output_dir):
    """Bar chart: Latency for all 7 models."""
    fig, ax = plt.subplots(figsize=(14, 7))

    names = list(combined.keys())
    labels = [MODEL_DISPLAY.get(n, {}).get("label", n) for n in names]
    lats = [combined[n]["latency"]["avg_ms"] for n in names]
    colors = [MODEL_DISPLAY.get(n, {}).get("color", "#666") for n in names]

    bars = ax.bar(range(len(names)), lats, color=colors, edgecolor="black", linewidth=0.5, width=0.6)
    for bar, val in zip(bars, lats):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(labels, fontsize=9, rotation=15, ha="right")
    ax.set_ylabel("Latency (ms)", fontsize=13, fontweight="bold")
    ax.set_title(f"Inference Latency — {len(names)} Model Variants", fontsize=15, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    save_path = output_dir / "latency_comparison.png"
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_memory_bar(combined, output_dir):
    """Bar chart: Memory for all 7 models."""
    fig, ax = plt.subplots(figsize=(14, 7))

    names = list(combined.keys())
    labels = [MODEL_DISPLAY.get(n, {}).get("label", n) for n in names]
    mems = [combined[n].get("peak_memory_mb", 0) for n in names]
    colors = [MODEL_DISPLAY.get(n, {}).get("color", "#666") for n in names]

    bars = ax.bar(range(len(names)), mems, color=colors, edgecolor="black", linewidth=0.5, width=0.6)
    for bar, val in zip(bars, mems):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(labels, fontsize=9, rotation=15, ha="right")
    ax.set_ylabel("Peak GPU Memory (MB)", fontsize=13, fontweight="bold")
    ax.set_title(f"Memory Footprint — {len(names)} Model Variants", fontsize=15, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    save_path = output_dir / "memory_comparison.png"
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_model_size_bar(combined, output_dir):
    """Bar chart: Model file size for all 7 models."""
    fig, ax = plt.subplots(figsize=(14, 7))

    names = list(combined.keys())
    labels = [MODEL_DISPLAY.get(n, {}).get("label", n) for n in names]
    sizes = [combined[n].get("model_size_mb", 0) for n in names]
    colors = [MODEL_DISPLAY.get(n, {}).get("color", "#666") for n in names]

    bars = ax.bar(range(len(names)), sizes, color=colors, edgecolor="black", linewidth=0.5, width=0.6)
    for bar, val in zip(bars, sizes):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(labels, fontsize=9, rotation=15, ha="right")
    ax.set_ylabel("Model Size (MB)", fontsize=13, fontweight="bold")
    ax.set_title(f"Model File Size — {len(names)} Model Variants", fontsize=15, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    save_path = output_dir / "model_size_comparison.png"
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def generate_summary(combined, output_dir):
    """Generate markdown summary table."""
    lines = [
        "# COMP 4901D — Final 7-Point Benchmark Results\n",
        "## Summary Table\n",
        "| # | Model | Format | mAP50-95 | mAP50 | Latency (ms) | FPS | Memory (MB) | Size (MB) |",
        "|---|-------|--------|----------|-------|-------------|-----|-------------|-----------|",
    ]
    for i, (key, r) in enumerate(combined.items(), 1):
        label = MODEL_DISPLAY.get(key, {}).get("label", key)
        fmt = "INT8 TRT" if "INT8" in key else "FP32 PyTorch"
        lines.append(
            f"| {i} | {label} | {fmt} | "
            f"{r.get('mAP50-95', 0):.4f} | {r.get('mAP50', 0):.4f} | "
            f"{r['latency']['avg_ms']:.2f} | {r.get('fps', 0):.1f} | "
            f"{r.get('peak_memory_mb', 0):.1f} | {r.get('model_size_mb', 0):.2f} |"
        )

    baseline_acc = combined.get("Baseline_FP32", {}).get("mAP50-95", 0)
    baseline_lat = combined.get("Baseline_FP32", {}).get("latency", {}).get("avg_ms", 0)

    lines.append("\n## Key Findings\n")
    for key, r in combined.items():
        if key == "Baseline_FP32":
            continue
        label = MODEL_DISPLAY.get(key, {}).get("label", key)
        acc = r.get("mAP50-95", 0)
        lat = r["latency"]["avg_ms"]
        if baseline_acc > 0 and acc > 0:
            acc_drop = (baseline_acc - acc) / baseline_acc * 100
            speedup = baseline_lat / lat if lat > 0 else 0
            lines.append(f"- **{label}**: mAP drop = {acc_drop:.2f}%, Speedup = {speedup:.2f}x")

    out_path = output_dir / "final_benchmark_summary.md"
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved: {out_path}")

    combined_path = output_dir / "combined_results.json"
    with open(combined_path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"  Saved: {combined_path}")


def main():
    output_dir = Path("runs/plots_final_7pt")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  COMP 4901D — Final Pareto Analysis")
    print("=" * 60)

    print("\nLoading and combining results...")
    combined = load_and_combine()
    print(f"  Loaded {len(combined)} model variants\n")

    for key, r in combined.items():
        label = MODEL_DISPLAY.get(key, {}).get("label", key)
        print(f"  {label:25s}  mAP50-95={r.get('mAP50-95',0):.4f}  "
              f"Latency={r['latency']['avg_ms']:.2f}ms  FPS={r.get('fps',0):.1f}")

    print("\nGenerating plots...")
    plot_pareto_7pt(combined, output_dir)
    plot_accuracy_bar(combined, output_dir)
    plot_fps_bar(combined, output_dir)
    plot_latency_bar(combined, output_dir)
    plot_memory_bar(combined, output_dir)
    plot_model_size_bar(combined, output_dir)
    generate_summary(combined, output_dir)

    print(f"\nAll plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
