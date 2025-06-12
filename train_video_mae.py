from pathlib import Path

import numpy as np
import torch
from transformers import (
    Trainer,
    TrainingArguments,
    VideoMAEForVideoClassification,
    VideoMAEImageProcessor,
)

from config import Config
from dataset import build_dataset  # your provided dataset setup

# Load dataset
config = Config()
train_loader, val_loader = build_dataset(
    config, download_dir=Path("zipped-data"), to_dir=Path("dataset")
)

# Load feature extractor
feature_extractor = VideoMAEImageProcessor.from_pretrained(
    "MCG-NJU/videomae-base-finetuned-kinetics"
)


def collate_hf(batch):
    processed_videos = []
    labels = []
    num_frames_required = 16  # typically what VideoMAE expects

    for item in batch:
        video_tensor = item[0]  # (C, T, H, W)
        label = item[1]

        # Convert to (T, H, W, C)
        frames = video_tensor.permute(1, 2, 3, 0).numpy()

        # Sample or truncate to 16 frames
        if frames.shape[0] >= num_frames_required:
            # Uniformly sample 16 frames
            indices = np.linspace(
                0, frames.shape[0] - 1, num_frames_required, dtype=int
            )
            selected_frames = [frames[i] for i in indices]
        else:
            # Pad by repeating last frame
            last_frame = frames[-1]
            padding = [last_frame] * (num_frames_required - frames.shape[0])
            selected_frames = list(frames) + padding

        processed_videos.append(selected_frames)
        labels.append(label)

    # Now processed_videos is: List[List[ndarray of shape (H, W, C)]], one per video
    pixel_values = feature_extractor(images=processed_videos, return_tensors="pt")[
        "pixel_values"
    ]

    return {"pixel_values": pixel_values, "labels": torch.tensor(labels)}


# Custom dataset wrapper for Hugging Face
class HFWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)


# Wrap datasets, not dataloaders
train_dataset = HFWrapper(train_loader.dataset)
val_dataset = HFWrapper(val_loader.dataset)

# Load pretrained model
model = VideoMAEForVideoClassification.from_pretrained(
    "MCG-NJU/videomae-base-finetuned-kinetics",
    num_labels=10,
    ignore_mismatched_sizes=True,
)

# Define training args
training_args = TrainingArguments(
    output_dir="./videomae-cricshot10",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    eval_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=5,
    logging_dir="./logs",
    fp16=torch.cuda.is_available(),
    report_to="none",
)

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=feature_extractor,
    data_collator=collate_hf,
)

# Start training
trainer.train()
