from __future__ import annotations

from typing import List, Optional, Tuple

from .state import Detection, WorkerSnapshot


BOX_COLOR = (78, 205, 196)
TEXT_COLOR = (255, 255, 255)
BANNER_COLOR = (33, 33, 33)
ERROR_COLOR = (60, 76, 231)
PANEL_HEADER_HEIGHT = 86


def annotate_detections(frame, detections: List[Detection], line_width: int) -> None:
    import cv2

    for detection in detections:
        x1 = int(round(detection.x1))
        y1 = int(round(detection.y1))
        x2 = int(round(detection.x2))
        y2 = int(round(detection.y2))
        label = "{name} {score:.2f}".format(
            name=detection.class_name,
            score=detection.confidence,
        )

        cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, line_width)
        (text_w, text_h), _ = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            2,
        )
        text_origin_y = max(y1 - 8, text_h + 8)
        cv2.rectangle(
            frame,
            (x1, text_origin_y - text_h - 8),
            (x1 + text_w + 8, text_origin_y),
            BOX_COLOR,
            -1,
        )
        cv2.putText(
            frame,
            label,
            (x1 + 4, text_origin_y - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (15, 15, 15),
            2,
            cv2.LINE_AA,
        )


def draw_panel_overlay(panel, title: str, snapshot: WorkerSnapshot) -> None:
    import cv2

    cv2.rectangle(panel, (0, 0), (panel.shape[1], PANEL_HEADER_HEIGHT), BANNER_COLOR, -1)

    status_text = "loaded" if snapshot.loaded else "unavailable"
    cv2.putText(
        panel,
        title,
        (16, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        TEXT_COLOR,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        "{model} | {status}".format(model=snapshot.model_name, status=status_text),
        (16, 54),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        TEXT_COLOR,
        2,
        cv2.LINE_AA,
    )

    stats = "FPS: {fps} | Latency: {latency} ms | Dets: {count}".format(
        fps=_format_number(snapshot.fps),
        latency=_format_number(snapshot.avg_latency_ms),
        count=len(snapshot.detections),
    )
    cv2.putText(
        panel,
        stats,
        (16, 76),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.56,
        TEXT_COLOR,
        1,
        cv2.LINE_AA,
    )

    if snapshot.error:
        cv2.rectangle(
            panel,
            (0, panel.shape[0] - 38),
            (panel.shape[1], panel.shape[0]),
            ERROR_COLOR,
            -1,
        )
        cv2.putText(
            panel,
            snapshot.error[: max(10, int(panel.shape[1] / 9))],
            (16, panel.shape[0] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            TEXT_COLOR,
            2,
            cv2.LINE_AA,
        )


def draw_composite_header(
    frame,
    frame_id: Optional[int],
    timestamp: Optional[float],
    left_title: str,
    right_title: str,
) -> None:
    import cv2
    import datetime as dt

    header_height = 52
    cv2.rectangle(frame, (0, 0), (frame.shape[1], header_height), (20, 20, 20), -1)
    timestamp_text = "n/a"
    if timestamp is not None:
        timestamp_text = dt.datetime.fromtimestamp(timestamp).strftime("%H:%M:%S")
    summary = "Frame: {frame_id} | Time: {timestamp}".format(
        frame_id=frame_id if frame_id is not None else "n/a",
        timestamp=timestamp_text,
    )
    labels = "{left}  <->  {right}".format(left=left_title, right=right_title)

    cv2.putText(
        frame,
        "COMP4901D Edge Demo",
        (16, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        TEXT_COLOR,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        summary,
        (16, 44),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        TEXT_COLOR,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        labels,
        (frame.shape[1] // 2, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        TEXT_COLOR,
        1,
        cv2.LINE_AA,
    )


def make_placeholder_frame(size: Tuple[int, int], title: str, subtitle: str):
    import cv2
    import numpy as np

    width, height = size
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:] = (28, 28, 28)
    cv2.putText(
        canvas,
        title,
        (24, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.95,
        TEXT_COLOR,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        subtitle,
        (24, 82),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        TEXT_COLOR,
        2,
        cv2.LINE_AA,
    )
    return canvas


def make_panel_header(width: int, title: str, snapshot: WorkerSnapshot):
    import numpy as np

    panel = np.zeros((PANEL_HEADER_HEIGHT, width, 3), dtype=np.uint8)
    draw_panel_overlay(panel, title, snapshot)
    return panel


def _format_number(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return "{:.1f}".format(value)
