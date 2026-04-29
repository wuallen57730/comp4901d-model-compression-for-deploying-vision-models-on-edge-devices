from __future__ import annotations

from typing import List

from .camera import CameraStream
from .compositor import CompositeWorker
from .config import DemoConfig, parse_config
from .inference import InferenceWorker
from .models import BaselineModelRunner, CompressedModelRunner
from .server import create_app
from .state import AppState


class EdgeDemoRuntime:
    def __init__(self, config: DemoConfig):
        self.config = config
        self.state = AppState(
            baseline_path=config.baseline_model,
            compressed_path=config.compressed_model,
        )
        self.camera = CameraStream(
            state=self.state,
            camera_index=config.camera_index,
            width=config.camera_width,
            height=config.camera_height,
            fps=config.camera_fps,
        )
        self.compositor = CompositeWorker(self.state)
        self.workers = []  # type: List[InferenceWorker]

        baseline_runner = BaselineModelRunner(
            artifact_path=config.baseline_model,
            device=config.device,
            imgsz=config.imgsz,
            conf=config.conf,
            iou=config.iou,
            line_width=config.line_width,
        )
        compressed_runner = CompressedModelRunner(
            artifact_path=config.compressed_model,
            device=config.device,
            imgsz=config.imgsz,
            conf=config.conf,
            iou=config.iou,
            line_width=config.line_width,
        )

        self.state.update_worker_snapshot("baseline", baseline_runner.bootstrap_snapshot())
        self.state.update_worker_snapshot("compressed", compressed_runner.bootstrap_snapshot())

        if baseline_runner.loaded:
            self.workers.append(InferenceWorker(self.state, baseline_runner))
        if compressed_runner.loaded:
            self.workers.append(InferenceWorker(self.state, compressed_runner))

    def start(self) -> None:
        self.camera.start()
        self.compositor.start()
        for worker in self.workers:
            worker.start()

    def stop(self) -> None:
        self.state.stop()
        self.camera.join(timeout=1.0)
        self.compositor.join(timeout=1.0)
        for worker in self.workers:
            worker.join(timeout=1.0)


def main(argv=None) -> None:
    config = parse_config(argv)
    runtime = EdgeDemoRuntime(config)
    try:
        app = create_app(runtime.state, config)
    except ImportError as exc:
        raise ImportError(
            "FastAPI dependencies are missing. Install them with `pip install -r requirements.txt`."
        ) from exc

    print("=" * 68)
    print("  COMP4901D - Jetson Edge Demo Backend")
    print("=" * 68)
    print("  Baseline model:   {path}".format(path=config.baseline_model))
    print("  Compressed model: {path}".format(path=config.compressed_model or "<missing>"))
    print("  Camera index:     {idx}".format(idx=config.camera_index))
    print("  Listen address:   http://{host}:{port}".format(host=config.host, port=config.port))
    print("=" * 68)

    runtime.start()
    try:
        try:
            import uvicorn
        except ImportError as exc:
            raise ImportError(
                "Uvicorn is not installed. Install it with `pip install -r requirements.txt`."
            ) from exc

        uvicorn.run(
            app,
            host=config.host,
            port=config.port,
            log_level="info",
        )
    finally:
        runtime.stop()


if __name__ == "__main__":
    main()
