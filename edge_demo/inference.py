from __future__ import annotations

import threading
import time
from collections import deque

from .models import BaseModelRunner
from .state import AppState, WorkerSnapshot


class InferenceWorker(threading.Thread):
    def __init__(self, state: AppState, runner: BaseModelRunner):
        super().__init__(name="{role}-worker".format(role=runner.role), daemon=True)
        self.state = state
        self.runner = runner
        self._latencies = deque(maxlen=30)
        self._completions = deque(maxlen=30)

    def run(self) -> None:
        last_frame_id = None

        while not self.state.shutdown_event.is_set():
            raw_frame = self.state.get_latest_raw_frame()
            if raw_frame is None or raw_frame.frame_id == last_frame_id:
                time.sleep(0.01)
                continue

            try:
                snapshot = self.runner.run(raw_frame)
                snapshot = self._enrich_snapshot(snapshot)
                self.state.update_worker_snapshot(self.runner.role, snapshot)
            except Exception as exc:
                failed = WorkerSnapshot(
                    role=self.runner.role,
                    model_name=self.runner.model_name,
                    artifact_path=self.runner.artifact_path,
                    loaded=False,
                    error=str(exc),
                    last_frame_id=raw_frame.frame_id,
                    last_timestamp=raw_frame.timestamp,
                )
                self.state.update_worker_snapshot(self.runner.role, failed)

            last_frame_id = raw_frame.frame_id

    def _enrich_snapshot(self, snapshot: WorkerSnapshot) -> WorkerSnapshot:
        if snapshot.latency_ms is not None:
            self._latencies.append(snapshot.latency_ms)
            snapshot.avg_latency_ms = sum(self._latencies) / len(self._latencies)

        self._completions.append(time.time())
        if len(self._completions) >= 2:
            elapsed = self._completions[-1] - self._completions[0]
            if elapsed > 0:
                snapshot.fps = float(len(self._completions) - 1) / elapsed
        return snapshot
