"""
INT8 Post-Training Quantization (PTQ) — Unified Entry Point

This is the **single entry point** for all INT8 quantization workflows in the
COMP 4901D compression pipeline.  For FP32/FP16 exports (ONNX, TensorRT FP16,
TorchScript), use ``export_onnx.py`` instead.

Supported backends
------------------
1. **onnx**      – ONNX Runtime static INT8 quantization (PC / dev machine)
2. **tensorrt**  – TensorRT INT8 engine export (Jetson via Ultralytics)

ONNX backend details:
- Per-channel INT8 weight quantization
- Entropy calibration for activations
- QDQ format for broad runtime compatibility

TensorRT backend details:
- Delegates to Ultralytics ``model.export(format="engine", int8=True, ...)``
- Uses the same calibration dataset YAML used elsewhere in the project

Examples
--------
::

    # 1) Export FP32 ONNX from distilled weights, then quantize to INT8 ONNX
    python int8_ptq.py \\
        --weights runs/distill/yolov8s_to_yolov8n/weights/best.pt \\
        --backend onnx \\
        --data coco.yaml \\
        --split val \\
        --calib-size 300

    # 2) Quantize an existing ONNX model directly
    python int8_ptq.py \\
        --onnx runs/export/distilled.onnx \\
        --backend onnx \\
        --data coco.yaml \\
        --imgsz 640

    # 3) Build an INT8 TensorRT engine on Jetson
    python int8_ptq.py \\
        --weights runs/distill/yolov8s_to_yolov8n/weights/best.pt \\
        --backend tensorrt \\
        --data coco.yaml \\
        --device 0

    # 4) Use a YAML config (e.g. on Jetson)
    python int8_ptq.py --config configs/ptq_config_jetson.yaml
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml


IMAGE_EXTENSIONS = {".bmp", ".dng", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="INT8 PTQ for YOLO edge deployment")
    parser.add_argument("--config", type=str, default=None,
                        help="Optional YAML config file for PTQ settings")

    source = parser.add_mutually_exclusive_group(required=False)
    source.add_argument("--weights", type=str, default=None,
                        help="Source YOLO weights (.pt) for export/quantization")
    source.add_argument("--onnx", type=str, default=None,
                        help="Existing ONNX model path to quantize")

    parser.add_argument("--backend", type=str, default=None,
                        choices=["onnx", "tensorrt"],
                        help="PTQ backend: 'onnx' for static INT8 ONNX, 'tensorrt' for Jetson engine export")
    parser.add_argument("--data", type=str, default=None,
                        help="Dataset YAML used for calibration")
    parser.add_argument("--split", type=str, default=None,
                        choices=["train", "val", "test"],
                        help="Dataset split for calibration images")
    parser.add_argument("--imgsz", type=int, default=None,
                        help="Model input image size")
    parser.add_argument("--batch", type=int, default=None,
                        help="Calibration/export batch size")
    parser.add_argument("--calib-size", type=int, default=None,
                        help="Maximum number of calibration images")
    parser.add_argument("--device", type=str, default=None,
                        help="Device for export/engine build")
    parser.add_argument("--opset", type=int, default=None,
                        help="ONNX opset for temporary export")
    parser.add_argument("--workspace", type=float, default=None,
                        help="TensorRT workspace size in GB (optional)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory")
    parser.add_argument("--name", type=str, default=None,
                        help="Optional name prefix for output files")
    parser.add_argument("--keep-fp32-onnx", action="store_true",
                        help="Keep the intermediate FP32 ONNX when quantizing from .pt")

    return parser.parse_args()


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_config(config_path: str) -> Dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f) or {}


def resolve_dataset_yaml(data_yaml: str) -> Dict:
    with open(data_yaml, "r") as f:
        data = yaml.safe_load(f)

    yaml_path = Path(data_yaml).resolve()
    base_path = Path(data.get("path", yaml_path.parent)).expanduser()
    if not base_path.is_absolute():
        base_path = (yaml_path.parent / base_path).resolve()

    data["_yaml_path"] = yaml_path
    data["_base_path"] = base_path
    return data


def resolve_split_paths(data_cfg: Dict, split: str) -> List[Path]:
    if split not in data_cfg:
        raise ValueError(f"Dataset YAML does not define split '{split}'")

    split_value = data_cfg[split]
    if isinstance(split_value, (list, tuple)):
        values = split_value
    else:
        values = [split_value]

    resolved = []
    for value in values:
        path = Path(str(value)).expanduser()
        if not path.is_absolute():
            path = (data_cfg["_base_path"] / path).resolve()
        resolved.append(path)
    return resolved


def collect_image_files(path: Path) -> List[Path]:
    if path.is_file():
        if path.suffix.lower() == ".txt":
            with open(path, "r") as f:
                files = [Path(line.strip()) for line in f if line.strip()]
            resolved = []
            for item in files:
                item = item.expanduser()
                if not item.is_absolute():
                    item = (path.parent / item).resolve()
                else:
                    item = item.resolve()
                resolved.append(item)
            return resolved
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            return [path.resolve()]
        raise ValueError(f"Unsupported calibration source file: {path}")

    if not path.exists():
        raise FileNotFoundError(f"Calibration path not found: {path}")

    files = [p for p in path.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS]
    return sorted(files)


def letterbox(
    image: np.ndarray,
    new_shape: Tuple[int, int],
    color: Tuple[int, int, int] = (114, 114, 114),
) -> Tuple[np.ndarray, float, Tuple[float, float]]:
    import cv2

    shape = image.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    ratio = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * ratio)), int(round(shape[0] * ratio)))
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)

    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))

    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return image, ratio, (dw, dh)


def preprocess_image(image_path: Path, imgsz: int) -> np.ndarray:
    import cv2

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")

    image, _, _ = letterbox(image, (imgsz, imgsz))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)
    return np.ascontiguousarray(image)


class YOLOCalibrationDataReader:
    """ONNX Runtime calibration reader for YOLO image tensors."""

    def __init__(self, image_paths: Sequence[Path], input_name: str, imgsz: int):
        self.image_paths = list(image_paths)
        self.input_name = input_name
        self.imgsz = imgsz
        self._index = 0

    def get_next(self) -> Optional[Dict[str, np.ndarray]]:
        if self._index >= len(self.image_paths):
            return None

        image_path = self.image_paths[self._index]
        self._index += 1
        return {self.input_name: preprocess_image(image_path, self.imgsz)}

    def rewind(self) -> None:
        self._index = 0


def maybe_export_fp32_onnx(
    weights_path: str,
    output_dir: Path,
    name: str,
    imgsz: int,
    batch: int,
    opset: int,
    device: str,
) -> Path:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "Ultralytics is required to export a .pt model to ONNX. Install with `pip install ultralytics`."
        ) from exc

    model = YOLO(weights_path)
    exported_path = model.export(
        format="onnx",
        imgsz=imgsz,
        batch=batch,
        simplify=True,
        opset=opset,
        device=device,
    )

    exported_path = Path(str(exported_path))
    target_name = name or exported_path.stem
    target_path = output_dir / f"{target_name}_fp32.onnx"
    ensure_parent_dir(target_path)

    if exported_path.resolve() != target_path.resolve():
        os.replace(exported_path, target_path)

    return target_path


def quantize_onnx_model(
    onnx_path: Path,
    data_yaml: str,
    split: str,
    imgsz: int,
    calib_size: int,
    output_dir: Path,
    name: str,
) -> Tuple[Path, Dict]:
    try:
        import onnx
        from onnxruntime.quantization import (
            CalibrationMethod,
            QuantFormat,
            QuantType,
            quantize_static,
        )
    except ImportError as exc:
        raise ImportError(
            "ONNX, ONNX Runtime, and onnxruntime.quantization are required for static INT8 PTQ. "
            "Install with `pip install onnx onnxruntime`."
        ) from exc

    model = onnx.load(str(onnx_path))
    if not model.graph.input:
        raise ValueError(f"No inputs found in ONNX model: {onnx_path}")
    input_name = model.graph.input[0].name

    data_cfg = resolve_dataset_yaml(data_yaml)
    split_paths = resolve_split_paths(data_cfg, split)
    image_files: List[Path] = []
    for split_path in split_paths:
        image_files.extend(collect_image_files(split_path))
    if not image_files:
        raise ValueError(f"No calibration images found for split '{split}' in {data_yaml}")

    selected_images = image_files[:calib_size]
    reader = YOLOCalibrationDataReader(selected_images, input_name, imgsz)

    output_name = name or onnx_path.stem.replace("_fp32", "")
    quantized_path = output_dir / f"{output_name}_int8.onnx"
    ensure_parent_dir(quantized_path)

    quantize_static(
        model_input=str(onnx_path),
        model_output=str(quantized_path),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        calibrate_method=CalibrationMethod.Entropy,
    )

    metadata = {
        "backend": "onnxruntime",
        "source_onnx": str(onnx_path),
        "quantized_onnx": str(quantized_path),
        "data": data_yaml,
        "split": split,
        "split_paths": [str(p) for p in split_paths],
        "imgsz": imgsz,
        "calibration_images": len(selected_images),
        "calibration_method": "entropy",
        "weight_quantization": "per-channel int8",
        "activation_quantization": "per-tensor uint8",
        "quant_format": "QDQ",
    }
    return quantized_path, metadata


def export_tensorrt_int8(
    weights_path: str,
    data_yaml: str,
    imgsz: int,
    batch: int,
    device: str,
    workspace: Optional[float],
    output_dir: Path,
    name: str,
) -> Tuple[Path, Dict]:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "Ultralytics is required for TensorRT INT8 export. Install with `pip install ultralytics` "
            "on the Jetson environment with TensorRT available."
        ) from exc

    model = YOLO(weights_path)
    export_kwargs = {
        "format": "engine",
        "imgsz": imgsz,
        "batch": batch,
        "int8": True,
        "data": data_yaml,
        "device": device,
    }
    if workspace is not None:
        export_kwargs["workspace"] = workspace

    engine_path = Path(str(model.export(**export_kwargs)))
    target_name = name or engine_path.stem
    target_path = output_dir / f"{target_name}.engine"
    ensure_parent_dir(target_path)

    if engine_path.resolve() != target_path.resolve():
        os.replace(engine_path, target_path)

    metadata = {
        "backend": "tensorrt",
        "source_weights": weights_path,
        "engine_path": str(target_path),
        "data": data_yaml,
        "imgsz": imgsz,
        "batch": batch,
        "device": device,
        "workspace_gb": workspace,
        "calibration_method": "TensorRT INT8 calibration",
    }
    return target_path, metadata


def save_metadata(output_dir: Path, name: str, metadata: Dict) -> Path:
    info_path = output_dir / f"{name}_ptq_info.json"
    with open(info_path, "w") as f:
        json.dump(metadata, f, indent=2)
    return info_path


def main() -> None:
    args = parse_args()

    config = load_config(args.config) if args.config else {}
    if config:
        for key, value in config.items():
            if getattr(args, key, None) is None:
                setattr(args, key, value)

    defaults = {
        "backend": "onnx",
        "split": "val",
        "imgsz": 640,
        "batch": 1,
        "calib_size": 300,
        "device": "0",
        "opset": 17,
        "output": "runs/quantize",
    }
    for key, value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, value)

    if not args.weights and not args.onnx:
        raise ValueError("Provide either `--weights` or `--onnx`, or set it in the config file.")
    if not args.data:
        raise ValueError("Provide `--data` or set `data` in the config file.")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    artifact_name = args.name
    if not artifact_name:
        source_name = Path(args.weights or args.onnx).stem
        artifact_name = source_name.replace("_fp32", "")

    print("=" * 60)
    print("  COMP 4901D - INT8 Post-Training Quantization")
    print("=" * 60)
    print(f"  Backend:      {args.backend}")
    print(f"  Source:       {args.weights or args.onnx}")
    print(f"  Dataset:      {args.data}")
    print(f"  Split:        {args.split}")
    print(f"  ImgSz:        {args.imgsz}")
    print(f"  Calib Images: {args.calib_size}")
    print(f"  Output Dir:   {output_dir}")
    print("=" * 60)

    metadata: Dict[str, object] = {
        "project": "COMP 4901D model compression for edge deployment",
        "artifact_name": artifact_name,
    }

    if args.backend == "onnx":
        if args.onnx:
            fp32_onnx_path = Path(args.onnx).resolve()
        else:
            fp32_onnx_path = maybe_export_fp32_onnx(
                weights_path=args.weights,
                output_dir=output_dir,
                name=artifact_name,
                imgsz=args.imgsz,
                batch=args.batch,
                opset=args.opset,
                device=args.device,
            )
            print(f"[Export] FP32 ONNX: {fp32_onnx_path}")

        quantized_path, backend_metadata = quantize_onnx_model(
            onnx_path=fp32_onnx_path,
            data_yaml=args.data,
            split=args.split,
            imgsz=args.imgsz,
            calib_size=args.calib_size,
            output_dir=output_dir,
            name=artifact_name,
        )
        metadata.update(backend_metadata)

        if not args.keep_fp32_onnx and args.weights and fp32_onnx_path.exists():
            fp32_onnx_path.unlink()
            metadata["intermediate_fp32_onnx_removed"] = True
        else:
            metadata["intermediate_fp32_onnx_removed"] = False

        print(f"[PTQ] INT8 ONNX saved to: {quantized_path}")

    else:
        if args.onnx:
            raise ValueError("TensorRT export currently expects `--weights` (.pt), not `--onnx`.")

        engine_path, backend_metadata = export_tensorrt_int8(
            weights_path=args.weights,
            data_yaml=args.data,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workspace=args.workspace,
            output_dir=output_dir,
            name=f"{artifact_name}_int8",
        )
        metadata.update(backend_metadata)
        print(f"[PTQ] TensorRT engine saved to: {engine_path}")

    info_path = save_metadata(output_dir, artifact_name, metadata)
    print(f"[Info] PTQ metadata saved to: {info_path}")


if __name__ == "__main__":
    main()
