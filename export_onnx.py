"""
ONNX & TensorRT Export Script (Milestone 3 preparation)

Exports trained YOLO models to ONNX format (can be done on PC)
and optionally to TensorRT FP16 engine format (requires Jetson/TensorRT).

This script handles FP32/FP16 exports only.  For INT8 quantization
(ONNX Runtime PTQ or TensorRT INT8), use ``int8_ptq.py`` instead.

Supported export variants for Pareto analysis:
1. Baseline YOLOv8n (pre-trained, no distillation)
2. Distilled Student (after KD training)
3. FP16 TensorRT engine (on Jetson)

Usage:
    # Export distilled model to ONNX (PC)
    python export_onnx.py --weights runs/distill/yolov8l_to_yolov8n/weights/best.pt --format onnx

    # Export baseline to ONNX
    python export_onnx.py --weights yolov8n.pt --format onnx --name baseline

    # Export to TensorRT FP16 (on Jetson only)
    python export_onnx.py --weights best.pt --format engine --half
"""

import argparse
import json
import os
from pathlib import Path

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Export YOLO Model to ONNX/TensorRT")
    parser.add_argument("--weights", type=str, required=True,
                        help="Path to model weights (.pt file)")
    parser.add_argument("--format", type=str, default="onnx",
                        choices=["onnx", "engine", "torchscript"],
                        help="Export format (default: onnx)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Input image size (default: 640)")
    parser.add_argument("--half", action="store_true",
                        help="FP16 export (for TensorRT)")
    parser.add_argument("--simplify", action="store_true", default=True,
                        help="Simplify ONNX model (default: True)")
    parser.add_argument("--opset", type=int, default=17,
                        help="ONNX opset version (default: 17)")
    parser.add_argument("--batch", type=int, default=1,
                        help="Batch size for export (default: 1)")
    parser.add_argument("--device", type=str, default="0",
                        help="Device for export")
    parser.add_argument("--name", type=str, default="",
                        help="Custom name suffix for output file")
    parser.add_argument("--output", type=str, default="runs/export",
                        help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  COMP 4901D - Model Export")
    print("=" * 60)
    print(f"  Weights:  {args.weights}")
    print(f"  Format:   {args.format}")
    print(f"  ImgSz:    {args.imgsz}")
    print(f"  FP16:     {args.half}")
    print("=" * 60)

    # Load model
    model = YOLO(args.weights)

    export_kwargs = {
        "format": args.format,
        "imgsz": args.imgsz,
        "half": args.half,
        "simplify": args.simplify and args.format == "onnx",
        "opset": args.opset if args.format == "onnx" else None,
        "batch": args.batch,
        "device": args.device,
    }

    export_kwargs = {k: v for k, v in export_kwargs.items() if v is not None}

    print(f"\nExporting with kwargs: {export_kwargs}")
    exported_path = model.export(**export_kwargs)

    print(f"\nExported model: {exported_path}")

    # Record export info
    export_info = {
        "source_weights": args.weights,
        "export_format": args.format,
        "exported_path": str(exported_path),
        "imgsz": args.imgsz,
        "half": args.half,
        "opset": args.opset if args.format == "onnx" else None,
        "batch": args.batch,
    }

    if exported_path and os.path.exists(exported_path):
        export_info["file_size_mb"] = os.path.getsize(exported_path) / (1024 * 1024)
        print(f"Exported file size: {export_info['file_size_mb']:.2f} MB")

    info_path = output_dir / f"export_info_{args.name or 'model'}.json"
    with open(info_path, "w") as f:
        json.dump(export_info, f, indent=2)

    print(f"Export info saved to: {info_path}")


if __name__ == "__main__":
    main()


