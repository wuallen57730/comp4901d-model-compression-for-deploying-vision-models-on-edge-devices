# Knowledge Distillation module for YOLO Feature-Based KD
from .hooks import FeatureExtractor
from .adapters import ChannelAdapter
from .losses import FeatureDistillationLoss
from .trainer import DistillationTrainer

__all__ = [
    "FeatureExtractor",
    "ChannelAdapter",
    "FeatureDistillationLoss",
    "DistillationTrainer",
]


