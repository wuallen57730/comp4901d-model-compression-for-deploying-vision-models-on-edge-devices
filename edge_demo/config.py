from __future__ import annotations

import argparse
import os
from dataclasses import dataclass


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


@dataclass(frozen=True)
class DemoConfig:
    baseline_model: str
    compressed_model: str
    device: str
    camera_index: int
    host: str
    port: int
    imgsz: int
    conf: float
    iou: float
    camera_width: int
    camera_height: int
    camera_fps: int
    jpeg_quality: int
    line_width: int
    stream_interval_ms: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Jetson realtime side-by-side YOLO demo backend"
    )
    parser.add_argument(
        "--baseline-model",
        type=str,
        default=_env("EDGE_DEMO_BASELINE_MODEL", "yolov8n.pt"),
        help="Baseline model artifact (.pt, .onnx, .engine)",
    )
    parser.add_argument(
        "--compressed-model",
        type=str,
        default=_env("EDGE_DEMO_COMPRESSED_MODEL", ""),
        help="Compressed deployment artifact, typically an INT8 TensorRT engine",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=_env("EDGE_DEMO_DEVICE", "0"),
        help="Inference device, e.g. '0' for Jetson GPU or 'cpu'",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=_env_int("EDGE_DEMO_CAMERA_INDEX", 0),
        help="USB webcam index",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=_env("EDGE_DEMO_HOST", "0.0.0.0"),
        help="Host interface for the FastAPI server",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=_env_int("EDGE_DEMO_PORT", 8000),
        help="Port for the FastAPI server",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=_env_int("EDGE_DEMO_IMGSZ", 640),
        help="Inference image size",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=_env_float("EDGE_DEMO_CONF", 0.25),
        help="Confidence threshold",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=_env_float("EDGE_DEMO_IOU", 0.45),
        help="NMS IoU threshold",
    )
    parser.add_argument(
        "--camera-width",
        type=int,
        default=_env_int("EDGE_DEMO_CAMERA_WIDTH", 1280),
        help="Requested camera capture width",
    )
    parser.add_argument(
        "--camera-height",
        type=int,
        default=_env_int("EDGE_DEMO_CAMERA_HEIGHT", 720),
        help="Requested camera capture height",
    )
    parser.add_argument(
        "--camera-fps",
        type=int,
        default=_env_int("EDGE_DEMO_CAMERA_FPS", 30),
        help="Requested camera capture FPS",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=_env_int("EDGE_DEMO_JPEG_QUALITY", 85),
        help="JPEG quality for snapshot and MJPEG endpoints",
    )
    parser.add_argument(
        "--line-width",
        type=int,
        default=_env_int("EDGE_DEMO_LINE_WIDTH", 2),
        help="Bounding box line width",
    )
    parser.add_argument(
        "--stream-interval-ms",
        type=int,
        default=_env_int("EDGE_DEMO_STREAM_INTERVAL_MS", 80),
        help="Delay between MJPEG frames served to clients",
    )
    return parser


def parse_config(argv=None) -> DemoConfig:
    args = build_parser().parse_args(argv)
    return DemoConfig(
        baseline_model=args.baseline_model,
        compressed_model=args.compressed_model,
        device=args.device,
        camera_index=args.camera_index,
        host=args.host,
        port=args.port,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        camera_width=args.camera_width,
        camera_height=args.camera_height,
        camera_fps=args.camera_fps,
        jpeg_quality=max(1, min(100, args.jpeg_quality)),
        line_width=max(1, args.line_width),
        stream_interval_ms=max(10, args.stream_interval_ms),
    )
