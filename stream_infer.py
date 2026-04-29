"""
Compatibility launcher for the Jetson realtime edge demo backend.

The implementation now lives under the `edge_demo/` package.
Run either:

    python stream_infer.py --baseline-model yolov8n.pt --compressed-model path/to/model.engine

or:

    python -m edge_demo.main --baseline-model yolov8n.pt --compressed-model path/to/model.engine
"""

from edge_demo.main import main


if __name__ == "__main__":
    main()
