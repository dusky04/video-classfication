# video-classfication

todo:

- maybe resize all the videos to 224, 224 to the dataset only and not resize during training
- try random frame sampling perhaps

```python
@dataclass
class Config:
    NUM_FRAMES = 16
    EXPERIMENT_NAME = "CNN-LSTM-Model"
    DATASET_NAME = "CricShot10"
    DOWNLOAD_DIR = Path("zipped-data")
    MODEL_FOLDER = "weights"
    FRAME_SIZE = (224, 224)
    BATCH_SIZE = 10
    NUM_CLASSES = 10
    TRAIN_SIZE = 0.8
    NUM_WORKERS = min(6, os.cpu_count() or 0)
    LSTM_HIDDEN_DIM = 256
    LSTM_NUM_LAYERS = 1
    # TO_DIR = Path("dataset")
    LR = 1e-3
    NUM_EPOCHS = 20
    WEIGHT_DECAY = 5e-4
    PREFETCH_FACTOR = 10
    LSTM_DROPOUT = 0.5
```

Best Training Accuracy : 92.31

Best Validation Accuracy : 76.67

```python
@dataclass
class Config:
    NUM_FRAMES = 32
    EXPERIMENT_NAME = "CNN-LSTM-Model"
    DATASET_NAME = "CricShot10"
    DOWNLOAD_DIR = Path("zipped-data")
    MODEL_FOLDER = "weights2"
    FRAME_SIZE = (224, 224)
    BATCH_SIZE = 32
    NUM_CLASSES = 10
    TRAIN_SIZE = 0.8
    NUM_WORKERS = min(6, os.cpu_count() or 0)
    LSTM_HIDDEN_DIM = 512
    LSTM_NUM_LAYERS = 2
    # TO_DIR = Path("dataset")
    LR = 1e-3
    NUM_EPOCHS = 20
    WEIGHT_DECAY = 5e-4
    PREFETCH_FACTOR = 20
    LSTM_DROPOUT = 0.5
```
