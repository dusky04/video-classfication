from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Dict, Tuple


def get_config() -> Dict[str, Any]:
    return {
        "batch_size": 16,
        "num_epochs": 10,
        "num_frames": 10,
        "lr": 1e-4,  # usually big learning rate at the start but gradually decrease it with each epoch
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "experiment_name": "runs/tmodel",
    }


@dataclass
class Config:
    NUM_FRAMES: int = 16
    EXPERIMENT_NAME: str = "CNN-LSTM-Model"
    FRAME_SIZE: Tuple[int, int] = (224, 224)
    DATASET_NAME: str = "CricShot10"
    BATCH_SIZE: int = 10
    NUM_CLASSES: int = 10
    TRAIN_SIZE: float = 0.8
    NUM_WORKERS: int = os.cpu_count() or 0
    LSTM_HIDDEN_DIM: int = 256
    LSTM_NUM_LAYERS: int = 1
    MODEL_FOLDER: str = "weights"
    DOWNLOAD_DIR: Path = Path("zipped-data")
    TO_DIR: Path = Path("dataset")
    LR: float = 1e-3
    NUM_EPOCHS: int = 20
    WEIGHT_DECAY: float = 5e-4
