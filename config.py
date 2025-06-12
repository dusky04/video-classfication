import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    NUM_FRAMES = 16
    EXPERIMENT_NAME = "CNN-LSTM-Model"
    DATASET_NAME = "CricShot10"
    DOWNLOAD_DIR = Path("zipped-data")
    MODEL_FOLDER = "weights"
    FRAME_SIZE = (224, 224)
    BATCH_SIZE = 10
    NUM_CLASSES = 9
    TRAIN_SIZE = 0.8
    NUM_WORKERS = min(6, os.cpu_count() or 0)
    LSTM_HIDDEN_DIM = 256
    LSTM_NUM_LAYERS = 1
    # TO_DIR = Path("dataset")
    LR = 1e-3
    NUM_EPOCHS = 20
    WEIGHT_DECAY = 5e-4
    PREFETCH_FACTOR = 10
