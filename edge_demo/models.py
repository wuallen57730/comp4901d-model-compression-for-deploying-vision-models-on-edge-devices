from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List

from .overlay import annotate_detections
from .state import Detection, RawFrame, WorkerSnapshot


class BaseModelRunner:
    def __init__(
        self,
        role: str,
        artifact_path: str,
        device: str,
        imgsz: int,
        conf: float,
        iou: float,
        line_width: int,
    ):
        self.role = role
        self.artifact_path = artifact_path
        self.device = device
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.line_width = line_width
        self.model = None
        self.load_error = None
        self.model_name = self._build_model_name()

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            self.load_error = (
                "Ultralytics is not installed. Run `pip install ultralytics`."
            )
            self._import_error = exc
            return

        try:
            if not artifact_path:
                raise ValueError("No artifact path provided.")
            self.model = YOLO(artifact_path)
            self.model_name = Path(artifact_path).name
        except Exception as exc:
            self.load_error = str(exc)

    @property
    def loaded(self) -> bool:
        return self.model is not None and self.load_error is None

    def bootstrap_snapshot(self) -> WorkerSnapshot:
        return WorkerSnapshot(
            role=self.role,
            model_name=self.model_name,
            artifact_path=self.artifact_path,
            loaded=self.loaded,
            error=self.load_error,
        )

    def run(self, raw_frame: RawFrame) -> WorkerSnapshot:
        if not self.loaded:
            snapshot = self.bootstrap_snapshot()
            snapshot.last_frame_id = raw_frame.frame_id
            snapshot.last_timestamp = raw_frame.timestamp
            return snapshot

        start = time.perf_counter()
        results = self.model.predict(
            source=raw_frame.frame,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            verbose=False,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        result = results[0]
        detections = self._extract_detections(result)
        annotated = raw_frame.frame.copy()
        annotate_detections(annotated, detections, self.line_width)

        return WorkerSnapshot(
            role=self.role,
            model_name=self.model_name,
            artifact_path=self.artifact_path,
            annotated_frame=annotated,
            detections=detections,
            latency_ms=elapsed_ms,
            last_frame_id=raw_frame.frame_id,
            last_timestamp=raw_frame.timestamp,
            loaded=True,
        )

    def _build_model_name(self) -> str:
        if self.artifact_path:
            return Path(self.artifact_path).name
        return self.role

    def _extract_detections(self, result) -> List[Detection]:
        detections = []
        boxes = getattr(result, "boxes", None)
        names = self._resolve_names(result)
        if boxes is None:
            return detections

        for box in boxes:
            coords = box.xyxy[0].tolist()
            class_id = int(box.cls.item())
            confidence = float(box.conf.item())
            detections.append(
                Detection(
                    class_id=class_id,
                    class_name=names.get(class_id, str(class_id)),
                    confidence=confidence,
                    x1=float(coords[0]),
                    y1=float(coords[1]),
                    x2=float(coords[2]),
                    y2=float(coords[3]),
                )
            )
        return detections

    def _resolve_names(self, result) -> Dict[int, str]:
        names = getattr(result, "names", None)
        if isinstance(names, dict):
            return names
        if isinstance(names, list):
            return dict((idx, name) for idx, name in enumerate(names))
        model_names = getattr(self.model, "names", {})
        if isinstance(model_names, dict):
            return model_names
        if isinstance(model_names, list):
            return dict((idx, name) for idx, name in enumerate(model_names))
        return {}


class BaselineModelRunner(BaseModelRunner):
    def __init__(self, artifact_path: str, device: str, imgsz: int, conf: float, iou: float, line_width: int):
        super().__init__(
            role="baseline",
            artifact_path=artifact_path,
            device=device,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            line_width=line_width,
        )


class CompressedModelRunner(BaseModelRunner):
    def __init__(self, artifact_path: str, device: str, imgsz: int, conf: float, iou: float, line_width: int):
        super().__init__(
            role="compressed",
            artifact_path=artifact_path,
            device=device,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            line_width=line_width,
        )
