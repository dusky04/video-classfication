from pathlib import Path

import torch
from torch import nn
from models.resnet_lstm import build_resnet_lstm_model, ResnetLSTModel
from dataset import build_dataset, CricShot
from config import Config
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.utils.data import DataLoader


def train_model(config: Config):
    # get the device to train on
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using Device: ", device)

    Path(config.MODEL_FOLDER).mkdir(parents=True, exist_ok=True)

    # get the dataset
    train_dataloader, test_dataloader = build_dataset(
        config, download_dir=config.DOWNLOAD_DIR, to_dir=config.TO_DIR
    )

    # setup tensorboard
    writer: SummaryWriter = SummaryWriter(config.EXPERIMENT_NAME)

    model: ResnetLSTModel = build_resnet_lstm_model(
        hidden_dim=config.LSTM_HIDDEN_DIM,
        num_lstm_layers=config.LSTM_NUM_LAYERS,
        num_classes=config.NUM_CLASSES,
    ).to(device)

    loss_fn = nn.CrossEntropyLoss().to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY
    )
    # TODO: Add lr scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=3, factor=0.5
    )

    # training loop
    for epoch in range(config.NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        batch_iterator: tqdm[DataLoader[CricShot]] = tqdm(
            train_dataloader, desc=f"Processing Epoch: {epoch + 1:02d}"
        )
        for batch_idx, (videos, labels) in enumerate(batch_iterator):
            videos = videos.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(videos)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * videos.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            batch_iterator.set_postfix(
                {
                    "loss": loss.item(),
                    "acc": 100.0 * correct / total if total > 0 else 0,
                }
            )

        epoch_loss = running_loss / total if total > 0 else 0
        epoch_acc = 100.0 * correct / total if total > 0 else 0
        writer.add_scalar("Loss/train", epoch_loss, epoch)
        writer.add_scalar("Accuracy/train", epoch_acc, epoch)

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.inference_mode():
            for videos, labels in test_dataloader:
                videos = videos.to(device)
                labels = labels.to(device)
                outputs = model(videos)
                loss = loss_fn(outputs, labels)
                val_loss += loss.item() * videos.size(0)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
        val_epoch_loss = val_loss / val_total if val_total > 0 else 0
        val_epoch_acc = 100.0 * val_correct / val_total if val_total > 0 else 0
        writer.add_scalar("Loss/val", val_epoch_loss, epoch)
        writer.add_scalar("Accuracy/val", val_epoch_acc, epoch)
        scheduler.step(val_epoch_acc)

        print(
            f"Epoch {epoch + 1}/{config.NUM_EPOCHS} | Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.2f}% | Val Loss: {val_epoch_loss:.4f} | Val Acc: {val_epoch_acc:.2f}%"
        )

        # Save model checkpoint
        torch.save(
            model.state_dict(),
            Path(config.MODEL_FOLDER)
            / f"{config.EXPERIMENT_NAME}_epoch{epoch + 1}.pth",
        )

    writer.close()
    print("Training complete.")
    return model


if __name__ == "__main__":
    config = Config()
    train_model(config)
