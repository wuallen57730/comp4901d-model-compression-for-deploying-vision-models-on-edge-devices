from __future__ import annotations

import threading
import time

from .overlay import (
    PANEL_HEADER_HEIGHT,
    draw_composite_header,
    make_panel_header,
    make_placeholder_frame,
)
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

    left_header, left_frame = _prepare_panel(raw_frame, baseline, panel_size, "Baseline")
    right_header, right_frame = _prepare_panel(raw_frame, compressed, panel_size, "Compressed")

    header_height = 52
    panel_block_height = PANEL_HEADER_HEIGHT + panel_size[1]
    combined = np.zeros((header_height + panel_block_height, panel_size[0] * 2, 3), dtype=np.uint8)
    combined[header_height:header_height + PANEL_HEADER_HEIGHT, : panel_size[0]] = left_header
    combined[header_height:header_height + PANEL_HEADER_HEIGHT, panel_size[0]:] = right_header
    combined[header_height + PANEL_HEADER_HEIGHT:, : panel_size[0]] = left_frame
    combined[header_height + PANEL_HEADER_HEIGHT:, panel_size[0]:] = right_frame

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
        frame = snapshot.annotated_frame.copy()
    elif raw_frame is not None:
        frame = raw_frame.frame.copy()
    else:
        frame = make_placeholder_frame(panel_size, title, "Waiting for camera input...")

    frame = cv2.resize(frame, (width, height))
    if not snapshot.loaded and snapshot.error:
        placeholder = make_placeholder_frame(panel_size, title, snapshot.error)
        frame = placeholder
    header = make_panel_header(width, title, snapshot)
    return header, frame
