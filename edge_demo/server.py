from __future__ import annotations

import time
from typing import Iterator, Optional

from .config import DemoConfig
from .state import AppState


def create_app(state: AppState, config: DemoConfig):
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse, Response, StreamingResponse

    app = FastAPI(title="COMP4901D Edge Demo", version="1.0.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health():
        return state.health_dict()

    @app.get("/metrics")
    def metrics():
        return state.metrics_dict()

    @app.get("/detections/latest")
    def detections_latest():
        return state.detections_dict()

    @app.get("/snapshot.jpg")
    def snapshot():
        payload = _encode_jpeg(state, config.jpeg_quality)
        if payload is None:
            return JSONResponse(
                status_code=503,
                content={"detail": "No composed frame available yet."},
            )
        return Response(content=payload, media_type="image/jpeg")

    @app.get("/stream/combined.mjpg")
    def stream_combined():
        return StreamingResponse(
            _mjpeg_stream(state, config),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    return app


def _mjpeg_stream(state: AppState, config: DemoConfig) -> Iterator[bytes]:
    while not state.shutdown_event.is_set():
        payload = _encode_jpeg(state, config.jpeg_quality)
        if payload is not None:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + payload + b"\r\n"
            )
        time.sleep(config.stream_interval_ms / 1000.0)


def _encode_jpeg(state: AppState, jpeg_quality: int) -> Optional[bytes]:
    import cv2

    frame = state.get_composed_frame()
    if frame is None:
        return None

    ok, encoded = cv2.imencode(
        ".jpg",
        frame,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
    )
    if not ok:
        return None
    return encoded.tobytes()
