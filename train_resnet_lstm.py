from pathlib import Path

import torch
from torchvision import transforms

from config import Config
from dataset import get_dataloaders
from models.resnet_lstm import build_resnet_lstm_model
from train import train_model
from utils import download_dataset, unzip_files

DATA_LINKS = {
    "CricShot10": "https://drive.google.com/file/d/1MOGU7xC2uw9qUmB9Zjez2sl3a2E7oFnS/view?usp=sharing",
    "SoccerAct10": "https://drive.google.com/file/d/1vD3DqGxxuuxCZaj10dhst-4St08MECVI/view?usp=sharing",
}


if __name__ == "__main__":
    config = Config()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not Path(config.DATASET_NAME).exists():
        download_dataset(config.DOWNLOAD_DIR, DATA_LINKS[config.DATASET_NAME])
        unzip_files(config.DOWNLOAD_DIR, config.DATASET_NAME)

    train_transform = transforms.Compose(
        [
            transforms.ConvertImageDtype(torch.float32),
            transforms.RandomApply([transforms.RandomRotation(10)], p=0.3),
            transforms.RandomApply(
                [transforms.RandomAffine(0, translate=(0.1, 0.1))], p=0.3
            ),
            transforms.RandomApply([transforms.ColorJitter(0.2, 0.2, 0.2, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=(5, 5))], p=0.5
            ),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    train_dataloader, test_dataloader = get_dataloaders(
        config, train_transform=train_transform, test_transform=test_transform
    )

    model = build_resnet_lstm_model(
        config.LSTM_HIDDEN_DIM, config.LSTM_NUM_LAYERS, config.NUM_CLASSES
    ).to(device)

    # finally train the model
    train_model(
        config,
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        device=device,
    )
