from __future__ import annotations

import threading
import time
from collections import deque

from .state import AppState, RawFrame


class CameraStream(threading.Thread):
    def __init__(self, state: AppState, camera_index: int, width: int, height: int, fps: int):
        super().__init__(name="camera-stream", daemon=True)
        self.state = state
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps
        self._frame_id = 0

    def run(self) -> None:
        try:
            import cv2
        except ImportError:
            self.state.set_camera_status(False, "OpenCV is not installed.")
            return

        capture = cv2.VideoCapture(self.camera_index)
        if not capture.isOpened():
            self.state.set_camera_status(
                False,
                "Failed to open webcam index {idx}".format(idx=self.camera_index),
            )
            return

        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        capture.set(cv2.CAP_PROP_FPS, self.fps)
        timestamps = deque(maxlen=30)

        try:
            while not self.state.shutdown_event.is_set():
                ok, frame = capture.read()
                if not ok:
                    self.state.set_camera_status(False, "Camera frame grab failed.")
                    time.sleep(0.1)
                    continue

                self._frame_id += 1
                now = time.time()
                timestamps.append(now)
                fps_value = 0.0
                if len(timestamps) >= 2:
                    elapsed = timestamps[-1] - timestamps[0]
                    if elapsed > 0:
                        fps_value = float(len(timestamps) - 1) / elapsed

                self.state.update_camera_frame(
                    RawFrame(frame_id=self._frame_id, timestamp=now, frame=frame),
                    fps_value=fps_value,
                )
        finally:
            capture.release()
            self.state.set_camera_status(False, "Camera stopped.")
