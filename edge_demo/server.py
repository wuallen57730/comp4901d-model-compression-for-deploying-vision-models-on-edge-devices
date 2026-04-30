from __future__ import annotations

import time
from typing import Iterator, Optional

from .config import DemoConfig
from .state import AppState


def create_app(state: AppState, config: DemoConfig):
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse

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

    @app.get("/viewer", response_class=HTMLResponse)
    def viewer():
        return HTMLResponse(_viewer_html())

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


def _viewer_html() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>COMP4901D Edge Demo Viewer</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #0b0b0b;
      --panel: #111111;
      --text: #f4f4f4;
      --muted: #b8b8b8;
      --accent: #d7df23;
    }
    * {
      box-sizing: border-box;
    }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Arial, Helvetica, sans-serif;
      min-height: 100vh;
      display: grid;
      grid-template-rows: auto 1fr;
    }
    header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 16px;
      padding: 14px 18px;
      background: rgba(17, 17, 17, 0.92);
      border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    }
    h1 {
      margin: 0;
      font-size: 20px;
      letter-spacing: 0.02em;
    }
    .hint {
      color: var(--muted);
      font-size: 13px;
    }
    main {
      display: grid;
      place-items: center;
      padding: 12px;
    }
    .stage {
      width: min(98vw, 2200px);
      height: calc(100vh - 86px);
      display: grid;
      place-items: center;
      background: #050505;
      border-radius: 12px;
      overflow: hidden;
      box-shadow: 0 18px 48px rgba(0, 0, 0, 0.35);
    }
    img {
      width: 100%;
      height: 100%;
      object-fit: contain;
      background: #000;
      display: block;
    }
    .accent {
      color: var(--accent);
      font-weight: 700;
    }
  </style>
</head>
<body>
  <header>
    <h1>COMP4901D Edge Demo Viewer</h1>
    <div class="hint">Open browser fullscreen for the biggest display. Stream source: <span class="accent">/stream/combined.mjpg</span></div>
  </header>
  <main>
    <div class="stage">
      <img src="/stream/combined.mjpg" alt="COMP4901D Edge Demo Stream" />
    </div>
  </main>
</body>
</html>
"""
