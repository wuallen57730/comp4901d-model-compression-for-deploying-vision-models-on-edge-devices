# Edge Demo Backend

This folder contains the Jetson-side realtime inference backend for the COMP4901D live demo. It captures a USB webcam stream, runs baseline and compressed YOLO inference in parallel, composes the results side by side, and exposes frames plus metadata to a UI running on another machine over LAN.

## Architecture
- `webcam`
- `camera buffer (latest frame only)`
- `baseline worker`
- `compressed worker`
- `side-by-side compositor`
- `FastAPI endpoints`

In words: `webcam -> camera buffer -> baseline worker + compressed worker -> compositor -> FastAPI endpoints`

The backend uses an asynchronous "latest frame wins" policy. If one model is slower, old frames are dropped so the demo stays responsive.

## Files
- `config.py`: CLI and environment-based runtime configuration
- `state.py`: thread-safe shared state for frames, metrics, detections, and health
- `camera.py`: USB webcam capture thread
- `models.py`: baseline/compressed YOLO wrappers
- `inference.py`: model worker threads and rolling FPS/latency tracking
- `overlay.py`: drawing utilities for boxes, stats, and status banners
- `compositor.py`: side-by-side frame composition
- `server.py`: FastAPI app and MJPEG/JSON endpoints
- `main.py`: backend entrypoint

## Expected Artifacts
- Baseline model: `yolov8n.pt`
- Compressed model: distilled `INT8 .engine`

Example compressed artifact path:

```bash
runs/quantize/distilled_student_jetson.engine
```

## Jetson Setup
1. Make sure the project and model artifacts are on the Jetson.
2. Verify the Jetson environment:

```bash
python setup_jetson.py
```

3. Install Python dependencies from this repo:

```bash
pip install -r requirements.txt
```

4. If the compressed TensorRT engine does not exist yet, build it on the Jetson:

```bash
python int8_ptq.py --config configs/ptq_config_jetson.yaml
```

## Launch
Run the backend directly from the project root:

```bash
python -m edge_demo.main \
  --baseline-model yolov8n.pt \
  --compressed-model runs/quantize/distilled_student_jetson.engine \
  --host 0.0.0.0 \
  --port 8000 \
  --camera-index 0
```

You can also set environment variables instead of passing CLI args:

```bash
export EDGE_DEMO_BASELINE_MODEL=yolov8n.pt
export EDGE_DEMO_COMPRESSED_MODEL=runs/quantize/distilled_student_jetson.engine
export EDGE_DEMO_HOST=0.0.0.0
export EDGE_DEMO_PORT=8000
python -m edge_demo.main
```

## Endpoints
- `GET /health`
  - Returns overall backend status, camera status, and model load status
- `GET /metrics`
  - Returns camera FPS plus per-model FPS, average latency, and last processed frame info
- `GET /detections/latest`
  - Returns the latest baseline/compressed detections in JSON form
- `GET /snapshot.jpg`
  - Returns the latest side-by-side composed frame as JPEG
- `GET /stream/combined.mjpg`
  - Returns a live MJPEG stream of the composed demo frame

## UI Handoff
Your UI teammate can start with:
- `http://<jetson-ip>:8000/snapshot.jpg`
- `http://<jetson-ip>:8000/stream/combined.mjpg`
- `http://<jetson-ip>:8000/metrics`
- `http://<jetson-ip>:8000/detections/latest`

This lets them build the frontend without needing to touch YOLO, TensorRT, or the webcam code.

## Demo-Day Checklist
- Confirm the USB webcam is plugged into the Jetson.
- Confirm the correct webcam index. If unsure, try `0` first.
- Confirm the Jetson IP address on the presentation network.
- Confirm `yolov8n.pt` exists on the Jetson.
- Confirm the compressed `.engine` loads successfully.
- Open `http://<jetson-ip>:8000/health` from the UI laptop.
- Open `http://<jetson-ip>:8000/snapshot.jpg` from the UI laptop.
- Open `http://<jetson-ip>:8000/stream/combined.mjpg` from the UI laptop.
- Let the stream run for several minutes before the presentation.

## Troubleshooting
### Camera not opening
- Check `GET /health` for `camera_open` and `camera_error`.
- Verify the webcam is connected to the Jetson, not the UI laptop.
- Try a different `--camera-index`.

### TensorRT engine missing or incompatible
- Check `GET /health` for `compressed_loaded` and `compressed_error`.
- Rebuild the engine on the target Jetson using the current JetPack/TensorRT setup.
- Confirm the `.engine` path passed to `--compressed-model` is correct.

### One panel is frozen
- Check `GET /metrics` and `GET /detections/latest`.
- If a model fails after startup, its latest error will be reported while the server keeps running.
- Restart the backend after fixing the failing artifact or dependency.

### UI laptop cannot connect over LAN
- Make sure the backend is launched with `--host 0.0.0.0`.
- Check the Jetson IP address and port.
- Confirm both devices are on the same network.
- Check firewall or hotspot isolation settings on the demo network.

## Notes
- The current v1 comparison is `baseline PyTorch .pt` vs `compressed INT8 TensorRT .engine`.
- This is intentionally optimized for a smooth live demo rather than a perfectly backend-matched benchmark.
