# COMP4901D — Model Compression for Deploying Vision Models on Edge Devices

Knowledge Distillation + INT8 Quantization pipeline for deploying YOLOv8 on NVIDIA Jetson Orin Nano.

## Project Overview

Optimize YOLOv8 for edge deployment by navigating the accuracy-efficiency trade-off through:
1. **Knowledge Distillation (KD)** — Feature-based Teacher→Student training
2. **INT8 Post-Training Quantization (PTQ)** — Via ONNX Runtime / TensorRT
3. **Pareto Analysis** — Accuracy vs Latency trade-off on Jetson hardware

## Repository Structure

```
comp4901d/
├── train_distill.py          # KD training entry point
├── eval_baseline.py          # Evaluate model mAP/latency/size
├── export_onnx.py            # Export to ONNX / TensorRT FP16
├── int8_ptq.py               # INT8 quantization (ONNX RT / TRT)
├── benchmark_jetson.py       # Full benchmark on Jetson (FP32/FP16/INT8)
├── plot_pareto.py            # Generate Pareto frontier plots
├── setup_jetson.py           # Jetson environment validation
├── verify_pipeline.py        # End-to-end pipeline verification
├── distill/                  # KD core module
│   ├── trainer.py            #   Custom DistillationTrainer (extends Ultralytics)
│   ├── losses.py             #   Feature MSE + Response KL-div losses
│   ├── hooks.py              #   Feature extraction hooks
│   └── adapters.py           #   Channel alignment adapters
├── configs/                  # Training & quantization configs
│   ├── distill_s2n_v2.yaml   #   YOLOv8s → YOLOv8n (alpha=0.01)
│   ├── distill_m2n_v2.yaml   #   YOLOv8m → YOLOv8n (alpha=0.01)
│   ├── distill_l2n_v2.yaml   #   YOLOv8l → YOLOv8n (alpha=0.01)
│   └── ...
├── results/                  # Training results (curves, metrics)
│   ├── s2n_v2/               #   YOLOv8s → n results
│   ├── m2n_v2/               #   YOLOv8m → n results
│   └── l2n_v2/               #   YOLOv8l → n results
├── best_s2n_v2.pt            # Distilled weight: s→n
├── best_m2n_v2.pt            # Distilled weight: m→n
├── best_l2n_v2.pt            # Distilled weight: l→n
├── yolov8n.pt                # Pretrained baseline (uncompressed)
└── requirements.txt
```

## Current Progress

### Completed

- [x] Feature-based KD training on GPU server (3 teacher variants × 100 epochs on COCO val2017)
- [x] Hyperparameter tuning: reduced `alpha` from 0.5 → 0.01 (v2 experiments)
- [x] Distilled model weights exported (`best_*_v2.pt`)

### KD Training Results (v2, alpha=0.01, 100 epochs on COCO)

| Model | Teacher | mAP@50 | mAP@50-95 | Notes |
|-------|---------|--------|-----------|-------|
| **YOLOv8n (pretrained baseline)** | — | 0.528 | 0.373 | No retraining |
| KD: s→n (`best_s2n_v2.pt`) | YOLOv8s | 0.518 | 0.367 | -0.6% mAP50-95 |
| KD: m→n (`best_m2n_v2.pt`) | YOLOv8m | 0.517 | 0.367 | -0.6% mAP50-95 |
| KD: l→n (`best_l2n_v2.pt`) | YOLOv8l | 0.516 | 0.367 | -0.6% mAP50-95 |

> The ~0.6% mAP drop is expected — the student (YOLOv8n) capacity is the bottleneck. All 3 teacher variants converge to nearly identical student performance. The primary benefit of KD will show after INT8 quantization, where distilled models tend to be more robust to quantization degradation.

### NOT Completed (Jetson Orin Nano was unreachable)

- [ ] Transfer weights to Jetson Orin Nano
- [ ] Export to ONNX → TensorRT FP16 engine
- [ ] INT8 quantization on Jetson
- [ ] Run `benchmark_jetson.py` (mAP, latency, FPS, memory)
- [ ] Generate Pareto frontier plots (`plot_pareto.py`)

## Setup

### GPU Server (Training)

```bash
python3 -m venv comp4901d_env
source comp4901d_env/bin/activate
pip install -r requirements.txt
```

Dataset: COCO val2017 is expected at `datasets/coco/` (download separately, ~52GB).

### Jetson Orin Nano (Benchmarking)

```
Account: group1
IP: 10.89.68.233
```

```bash
# 1. Transfer files from GPU server to local, then to Jetson
scp ycwu@<GPU_SERVER_IP>:~/comp4901d/best_*_v2.pt .
scp ycwu@<GPU_SERVER_IP>:~/comp4901d/yolov8n.pt .
scp best_*_v2.pt yolov8n.pt group1@10.89.68.233:~/comp4901d/

# Also transfer all Python scripts and configs
scp ycwu@<GPU_SERVER_IP>:~/comp4901d/*.py .
scp ycwu@<GPU_SERVER_IP>:~/comp4901d/requirements.txt .
scp -r ycwu@<GPU_SERVER_IP>:~/comp4901d/configs .
scp -r ycwu@<GPU_SERVER_IP>:~/comp4901d/distill .

scp *.py requirements.txt group1@10.89.68.233:~/comp4901d/
scp -r configs distill group1@10.89.68.233:~/comp4901d/

# 2. On Jetson: setup environment
ssh group1@10.89.68.233
cd ~/comp4901d
python3 setup_jetson.py
pip install -r requirements.txt

# 3. Export to ONNX + TensorRT FP16
python3 export_onnx.py --weights best_s2n_v2.pt --format engine --half
python3 export_onnx.py --weights best_m2n_v2.pt --format engine --half
python3 export_onnx.py --weights best_l2n_v2.pt --format engine --half
python3 export_onnx.py --weights yolov8n.pt --format engine --half

# 4. INT8 quantization
python3 int8_ptq.py --weights best_s2n_v2.pt --data coco.yaml --output runs/quantize/
python3 int8_ptq.py --weights best_m2n_v2.pt --data coco.yaml --output runs/quantize/
python3 int8_ptq.py --weights best_l2n_v2.pt --data coco.yaml --output runs/quantize/
python3 int8_ptq.py --weights yolov8n.pt --data coco.yaml --output runs/quantize/

# 5. Full benchmark (repeat for each model)
python3 benchmark_jetson.py \
    --data coco.yaml \
    --baseline-pt yolov8n.pt \
    --distilled-pt best_s2n_v2.pt \
    --distilled-engine-int8 runs/quantize/best_s2n_v2_int8.engine \
    --output runs/benchmark_s2n_v2

# 6. Pareto plot
python3 plot_pareto.py --results-dir runs/ --output runs/pareto/
```

## Baseline Definition

The baseline is the **pretrained YOLOv8n.pt** evaluated directly on COCO val2017 without any retraining. This represents the "uncompressed baseline" mentioned in the project spec. KD and INT8 variants are compared against this baseline on the Pareto frontier.

## Hardware

| Device | Purpose |
|--------|---------|
| GPU Server (A100/V100) | KD training (100 epochs, COCO) |
| NVIDIA Jetson Orin Nano (8GB) | Inference benchmarking (TensorRT) |

## Key Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `alpha` (feature KD weight) | 0.01 | v1 used 0.5 (too high, overwhelmed detection loss) |
| `beta` (response KD weight) | 0.0 | Feature-based KD only |
| Epochs | 100 | |
| Image size | 640 | |
| Optimizer | SGD (Ultralytics default) | |
| Student init | Pretrained YOLOv8n.pt | |
