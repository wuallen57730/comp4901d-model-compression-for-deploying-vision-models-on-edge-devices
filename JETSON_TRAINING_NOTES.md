# Jetson Orin Nano — On-Device KD Training 筆記

## 硬體限制

| 項目 | 規格 |
|------|------|
| GPU | NVIDIA Ampere (1024 CUDA cores) |
| 記憶體 | **8 GB 共享**（CPU + GPU 共用同一塊 LPDDR5） |
| 儲存 | microSD / NVMe SSD |
| 電源模式 | 建議 MAXN (`sudo nvpmodel -m 0`) |

> 8 GB 共享記憶體是最大瓶頸，需要同時放 Teacher + Student + Optimizer states + Feature maps。

---

## Teacher 模型選擇

| 模型 | Params | mAP50-95 (COCO) | FP32 記憶體 | FP16 記憶體 |
|------|--------|-----------------|------------|------------|
| YOLOv8l | 43.7M | 52.9 | ~175 MB | ~87 MB |
| YOLOv8m | 25.9M | 50.2 | ~104 MB | ~52 MB |
| **YOLOv8s** | **11.2M** | **44.9** | **~45 MB** | **~22 MB** |
| YOLOv8n (Student) | 3.2M | 37.3 | ~13 MB | ~6 MB |

**決定：使用 YOLOv8s 作為 Teacher**

- YOLOv8l 在 8 GB 下同時跑 teacher + student forward 會 OOM
- YOLOv8s 的 mAP (44.9) 比 YOLOv8n (37.3) 高 7.6 個點，蒸餾仍有足夠的知識差距
- FP16 下 Teacher 只佔 ~22 MB，遠比 YOLOv8l 的 ~87 MB 友好

---

## 記憶體優化策略

### 1. Teacher FP16 (`teacher_half: true`)

Teacher 是凍結的（`requires_grad=False`），不需要 FP32 精度。轉 FP16 省約 50% 記憶體。

```python
# trainer.py setup() 中
self.teacher_model.half()

# _kd_loss() 中，輸入圖片也要轉 half
img = batch["img"].half()
with torch.no_grad():
    teacher_model(img)
# teacher features cast 回 FP32 再算 loss
t_feats = {k: v.float() for k, v in t_feats.items()}
```

### 2. AMP 混合精度訓練 (`amp: true`)

Ultralytics 內建 AMP 支援，Student 的 forward/backward 自動在 FP16 下執行，梯度更新保持 FP32 master weights。

### 3. 梯度累積 (`nbs: 32`)

Jetson 上 batch_size 只能開 2-4，透過梯度累積模擬大 batch：

| batch_size | nbs | accumulate 步數 | effective batch |
|-----------|-----|----------------|-----------------|
| 2 | 32 | 16 | 32 |
| 4 | 32 | 8 | 32 |
| 4 | 64 | 16 | 64 |

### 4. 記憶體清理

- 每個 batch：計算完 distillation loss 後立即清理 teacher/student features
- 每個 epoch：呼叫 `torch.cuda.empty_cache()` 歸還未使用的 CUDA cache
- 訓練結束：移除所有 hooks + `gc.collect()`

---

## Jetson 專用配置

檔案：`configs/distill_config_jetson.yaml`

```yaml
teacher_weights: "yolov8s.pt"     # 小 Teacher，8 GB 友好
student_weights: "yolov8n.pt"
epochs: 50
batch_size: 4                     # Orin Nano 建議 2-4
workers: 2                        # 記憶體有限，不要開太多
teacher_half: true                # Teacher FP16
amp: true                         # AMP 混合精度
nbs: 32                           # gradient accumulation
device: "0"
```

---

## 訓練指令

```bash
# 1. 環境驗證
python setup_jetson.py

# 2. 快速測試（COCO128, ~10 min）
python train_distill.py --config configs/distill_config_jetson.yaml \
    --data coco128.yaml --epochs 5

# 3. 正式訓練
python train_distill.py --config configs/distill_config_jetson.yaml

# 4. 如果 OOM，降低 batch 或圖片大小
python train_distill.py --config configs/distill_config_jetson.yaml \
    --batch 2 --imgsz 416
```

---

## 修改的檔案清單

| 檔案 | 改動內容 |
|------|---------|
| `distill/trainer.py` | 新增 `teacher_half`, `amp`, `nbs`, `cache` 參數；FP16 teacher forward；記憶體清理；GPU 記憶體監控 |
| `configs/distill_config.yaml` | Teacher 改為 YOLOv8s；新增記憶體優化參數；加 Jetson 說明註解 |
| `configs/distill_config_jetson.yaml` | 新建，Jetson Orin Nano 專用配置 |
| `train_distill.py` | 新增 `--teacher-half`, `--amp`, `--nbs`, `--cache` CLI 參數 |
| `setup_jetson.py` | 改為 Orin Nano；新增 RAM/Swap/電源模式檢查；KD 訓練記憶體估算 |

---

## 注意事項

- Jetson Orin Nano **不需要額外 GPU**，直接在板上的 GPU 訓練即可
- 建議設定 **>= 8 GB swap**，避免記憶體不足時直接被 OOM killer 殺掉
- 訓練前用 `sudo nvpmodel -m 0` 切到 MAXN 模式，確保最大效能
- 訓練時可用 `tegrastats` 或 `jtop` 監控 GPU/記憶體使用率
