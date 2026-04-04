# COMP 4901D — Final 7-Point Benchmark Results

## Summary Table

| # | Model | Format | mAP50-95 | mAP50 | Latency (ms) | FPS | Memory (MB) | Size (MB) |
|---|-------|--------|----------|-------|-------------|-----|-------------|-----------|
| 1 | Baseline (FP32) | FP32 PyTorch | 0.3714 | 0.5212 | 15.44 | 64.8 | 35.6 | 6.25 |
| 2 | KD: s→n (FP32) | FP32 PyTorch | 0.3670 | 0.5172 | 15.46 | 64.7 | 35.6 | 6.23 |
| 3 | KD: m→n (FP32) | FP32 PyTorch | 0.3670 | 0.5165 | 16.61 | 60.2 | 35.6 | 6.23 |
| 4 | KD: l→n (FP32) | FP32 PyTorch | 0.3673 | 0.5157 | 16.59 | 60.3 | 35.6 | 6.23 |
| 5 | Baseline INT8 (Q only) | INT8 TRT | 0.3535 | 0.4995 | 4.54 | 220.2 | 12.1 | 4.99 |
| 6 | KD: s→n + INT8 | INT8 TRT | 0.3532 | 0.5044 | 4.41 | 226.9 | 12.1 | 4.97 |
| 7 | KD: m→n + INT8 | INT8 TRT | 0.3497 | 0.4996 | 4.63 | 216.1 | 12.1 | 4.99 |
| 8 | KD: l→n + INT8 | INT8 TRT | 0.3454 | 0.4978 | 4.62 | 216.3 | 12.1 | 4.99 |

## Key Findings

- **KD: s→n (FP32)**: mAP drop = 1.18%, Speedup = 1.00x
- **KD: m→n (FP32)**: mAP drop = 1.19%, Speedup = 0.93x
- **KD: l→n (FP32)**: mAP drop = 1.11%, Speedup = 0.93x
- **Baseline INT8 (Q only)**: mAP drop = 4.81%, Speedup = 3.40x
- **KD: s→n + INT8**: mAP drop = 4.90%, Speedup = 3.50x
- **KD: m→n + INT8**: mAP drop = 5.85%, Speedup = 3.34x
- **KD: l→n + INT8**: mAP drop = 7.00%, Speedup = 3.34x