from __future__ import annotations

import threading
import time

from .overlay import draw_composite_header, draw_panel_overlay, make_placeholder_frame
from .state import AppState, RawFrame, WorkerSnapshot


class CompositeWorker(threading.Thread):
    def __init__(self, state: AppState):
        super().__init__(name="compositor", daemon=True)
        self.state = state

    def run(self) -> None:
        while not self.state.shutdown_event.is_set():
            raw_frame = self.state.get_latest_raw_frame()
            baseline = self.state.get_worker_snapshot("baseline")
            compressed = self.state.get_worker_snapshot("compressed")
            frame = build_composite_frame(raw_frame, baseline, compressed)
            self.state.update_composed_frame(frame)
            time.sleep(0.03)


def build_composite_frame(raw_frame: RawFrame, baseline: WorkerSnapshot, compressed: WorkerSnapshot):
    import cv2
    import numpy as np

    if raw_frame is not None:
        height, width = raw_frame.frame.shape[:2]
        panel_size = (width, height)
    else:
        panel_size = (960, 540)

    left = _prepare_panel(raw_frame, baseline, panel_size, "Baseline")
    right = _prepare_panel(raw_frame, compressed, panel_size, "Compressed")

    header_height = 52
    combined = np.zeros((panel_size[1] + header_height, panel_size[0] * 2, 3), dtype=np.uint8)
    combined[header_height:, : panel_size[0]] = left
    combined[header_height:, panel_size[0] :] = right

    frame_id = raw_frame.frame_id if raw_frame is not None else None
    timestamp = raw_frame.timestamp if raw_frame is not None else None
    draw_composite_header(combined, frame_id, timestamp, "Baseline", "Compressed")
    cv2.line(
        combined,
        (panel_size[0], header_height),
        (panel_size[0], combined.shape[0]),
        (255, 255, 255),
        2,
    )
    return combined


def _prepare_panel(
    raw_frame: RawFrame,
    snapshot: WorkerSnapshot,
    panel_size,
    title: str,
):
    import cv2

    width, height = panel_size

    if snapshot.annotated_frame is not None:
        panel = snapshot.annotated_frame.copy()
    elif raw_frame is not None:
        panel = raw_frame.frame.copy()
    else:
        panel = make_placeholder_frame(panel_size, title, "Waiting for camera input...")

    panel = cv2.resize(panel, (width, height))
    if not snapshot.loaded and snapshot.error:
        placeholder = make_placeholder_frame(panel_size, title, snapshot.error)
        panel = placeholder
    draw_panel_overlay(panel, title, snapshot)
    return panel
