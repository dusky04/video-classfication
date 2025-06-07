from typing import Any, Dict


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
