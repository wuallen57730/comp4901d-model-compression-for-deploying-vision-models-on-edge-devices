from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": round(self.confidence, 4),
            "x1": round(self.x1, 2),
            "y1": round(self.y1, 2),
            "x2": round(self.x2, 2),
            "y2": round(self.y2, 2),
        }


@dataclass
class RawFrame:
    frame_id: int
    timestamp: float
    frame: Any


@dataclass
class WorkerSnapshot:
    role: str
    model_name: str
    artifact_path: str
    annotated_frame: Optional[Any] = None
    detections: List[Detection] = field(default_factory=list)
    latency_ms: Optional[float] = None
    avg_latency_ms: Optional[float] = None
    fps: Optional[float] = None
    last_frame_id: Optional[int] = None
    last_timestamp: Optional[float] = None
    loaded: bool = False
    error: Optional[str] = None

    def metrics_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "artifact_path": self.artifact_path,
            "fps": round(self.fps, 2) if self.fps is not None else None,
            "avg_latency_ms": round(self.avg_latency_ms, 2)
            if self.avg_latency_ms is not None
            else None,
            "last_frame_id": self.last_frame_id,
            "last_timestamp": self.last_timestamp,
            "loaded": self.loaded,
            "error": self.error,
        }

    def detections_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "artifact_path": self.artifact_path,
            "loaded": self.loaded,
            "error": self.error,
            "detections": [item.to_dict() for item in self.detections],
        }


class AppState:
    def __init__(self, baseline_path: str, compressed_path: str):
        self._lock = threading.RLock()
        self.start_time = time.time()
        self.latest_raw_frame = None  # type: Optional[RawFrame]
        self.latest_composed_frame = None  # type: Optional[Any]
        self.camera_open = False
        self.camera_error = None  # type: Optional[str]
        self.camera_fps = 0.0
        self.shutdown_event = threading.Event()
        self.baseline_snapshot = WorkerSnapshot(
            role="baseline",
            model_name="baseline",
            artifact_path=baseline_path,
        )
        self.compressed_snapshot = WorkerSnapshot(
            role="compressed",
            model_name="compressed",
            artifact_path=compressed_path,
        )

    def stop(self) -> None:
        self.shutdown_event.set()

    def uptime_s(self) -> float:
        return time.time() - self.start_time

    def set_camera_status(self, is_open: bool, error: Optional[str] = None) -> None:
        with self._lock:
            self.camera_open = is_open
            self.camera_error = error

    def update_camera_frame(self, raw_frame: RawFrame, fps_value: float) -> None:
        with self._lock:
            self.latest_raw_frame = raw_frame
            self.camera_open = True
            self.camera_error = None
            self.camera_fps = fps_value

    def get_latest_raw_frame(self) -> Optional[RawFrame]:
        with self._lock:
            return self.latest_raw_frame

    def update_worker_snapshot(self, role: str, snapshot: WorkerSnapshot) -> None:
        with self._lock:
            if role == "baseline":
                self.baseline_snapshot = snapshot
            else:
                self.compressed_snapshot = snapshot

    def get_worker_snapshot(self, role: str) -> WorkerSnapshot:
        with self._lock:
            if role == "baseline":
                return self.baseline_snapshot
            return self.compressed_snapshot

    def update_composed_frame(self, frame: Any) -> None:
        with self._lock:
            self.latest_composed_frame = frame

    def get_composed_frame(self) -> Optional[Any]:
        with self._lock:
            return self.latest_composed_frame

    def health_dict(self) -> Dict[str, Any]:
        with self._lock:
            baseline_loaded = self.baseline_snapshot.loaded
            compressed_loaded = self.compressed_snapshot.loaded
            if self.camera_open and baseline_loaded and compressed_loaded:
                status = "healthy"
            elif self.camera_open or baseline_loaded or compressed_loaded:
                status = "degraded"
            else:
                status = "error"

            return {
                "status": status,
                "uptime_s": round(self.uptime_s(), 2),
                "camera_open": self.camera_open,
                "camera_error": self.camera_error,
                "baseline_loaded": baseline_loaded,
                "compressed_loaded": compressed_loaded,
                "baseline_error": self.baseline_snapshot.error,
                "compressed_error": self.compressed_snapshot.error,
            }

    def metrics_dict(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "camera_fps": round(self.camera_fps, 2),
                "baseline": self.baseline_snapshot.metrics_dict(),
                "compressed": self.compressed_snapshot.metrics_dict(),
            }

    def detections_dict(self) -> Dict[str, Any]:
        with self._lock:
            timestamps = [
                item
                for item in (
                    self.baseline_snapshot.last_timestamp,
                    self.compressed_snapshot.last_timestamp,
                    self.latest_raw_frame.timestamp if self.latest_raw_frame else None,
                )
                if item is not None
            ]
            frame_ids = [
                item
                for item in (
                    self.baseline_snapshot.last_frame_id,
                    self.compressed_snapshot.last_frame_id,
                    self.latest_raw_frame.frame_id if self.latest_raw_frame else None,
                )
                if item is not None
            ]
            return {
                "frame_id": max(frame_ids) if frame_ids else None,
                "source_timestamp": max(timestamps) if timestamps else None,
                "baseline": self.baseline_snapshot.detections_dict(),
                "compressed": self.compressed_snapshot.detections_dict(),
            }
