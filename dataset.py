# File containing functions based on dataset
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import torch
from gdown import download
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from decord import VideoReader

from config import Config
from utils import get_classes, setup_dataset_structure, unzip_files

# DATA_LINKS: Dict[str, List[str]] = {
#     "CricShot10": [
#         "https://drive.google.com/file/d/11ieVDcfewzLnJHEjuXs22w5F_Iv9R4rb/view?usp=sharing",
#         "https://drive.google.com/file/d/15g-mUAgUZl2EIHMWbUnaaRONQxLfhezC/view?usp=sharing",
#         "https://drive.google.com/file/d/1-EwcjRFLAniNlR4L2MUUH2ndTJGcsAWa/view?usp=sharing",
#         "https://drive.google.com/file/d/1U-xqokSjtKeNBlnxYcZUio1GVONyJnQs/view?usp=sharing",
#         "https://drive.google.com/file/d/1-U8Jec5SnSzGmEPe6kjrhiXJx_HuvqNV/view?usp=sharing",
#         "https://drive.google.com/file/d/1E8LTOhghJ4skOtCLucAWYi2Jta_4r67p/view?usp=sharing",
#         "https://drive.google.com/file/d/1S-CQlDuHENpADcW4ATm2buGOiWMRuT0S/view?usp=sharing",
#         "https://drive.google.com/file/d/1iT7MQluDQG9hoXKVeVOOFGxhDj4I-i56/view?usp=sharing",
#         "https://drive.google.com/file/d/1fyPCxbi2FF1rpuJpO9HPFggHfDRh845s/view?usp=sharing",
#         "https://drive.google.com/file/d/1om1VVIE2WmkMHwZVB44ell-kgfBGlL7H/view?usp=sharing",
#     ]
# }

DATA_LINKS = {
    "CricShot10": [
        "https://drive.google.com/file/d/1xjOJqHEUaYFQKrF1aqwBVr5JZcqMuGOS/view?usp=sharing"
    ]
}

# TODO: Think about adding Optical Flow
# TODO: Apparently each frame in a video has different transforms applied??
# TODO: Update this dataset class


class CricShot(Dataset):
    def __init__(
        self,
        dir: Path,
        transform: transforms.Compose,
        num_frames: int = 16,
        frame_size: Tuple[int, int] = (224, 224),
    ) -> None:
        # directory which contains the dataset
        self.dataset_dir: Path = dir
        # paths of all the videos present in the dataset - label / video.avi
        self.video_paths: List[Path] = list(dir.glob("*/*.avi"))
        # transform to apply to the videos for data augmentation
        self.transform: transforms.Compose = transform
        # number of frames to sample from the video
        self.num_frames: int = num_frames
        # size of each frame
        self.frame_size: Tuple[int, int] = frame_size
        # class_names
        self.class_names, self.class_to_idx = get_classes(dir)

    def __len__(self) -> int:
        return len(self.video_paths)

    def get_last_or_blank_frame(self, video_frames: List[torch.Tensor]) -> torch.Tensor:
        return (
            video_frames[-1]
            if video_frames
            else torch.zeros((*self.frame_size, 3), dtype=torch.float32)
        )

    # def load_video_frames(self, idx: int) -> torch.Tensor:
    #     video_path: Path = self.video_paths[idx]

    #     cap: cv2.VideoCapture = cv2.VideoCapture(str(video_path))

    #     total_frame_count: float = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    #     # uniformly sample the number of frames
    #     indices: List[int] = torch.linspace(
    #         0, total_frame_count - 1, self.num_frames, dtype=torch.float32
    #     ).tolist()

    #     video_frames = []
    #     for idx in indices:
    #         # read the frame at index 'idx'
    #         cap.set(cv2.CAP_PROP_POS_FRAMES, idx)

    #         ret, frame = cap.read()
    #         if not ret:
    #             last_frame = self.get_last_or_blank_frame(video_frames)
    #             video_frames.append(last_frame)
    #             continue

    #         # convert the frame to (x, y, 3) -> (224, 224, 3)
    #         frame = cv2.resize(frame, self.frame_size)
    #         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #         frame = torch.from_numpy(frame)
    #         video_frames.append(frame)
    #     cap.release()

    #     # Pad with copies of the last frame
    #     if len(video_frames) < self.num_frames:
    #         last_frame = self.get_last_or_blank_frame(video_frames)
    #         video_frames.extend([last_frame] * (self.num_frames - len(video_frames)))

    #     # stack the tensor along the number of frames of a video
    #     # (224, 224, 3) -> (num_frames, 224, 224, 3)

    #     video_frames = torch.stack(
    #         [self.transform(torch.permute(frame, (2, 1, 0))) for frame in video_frames]
    #         if self.transform
    #         else video_frames,
    #         dim=0,
    #     )
    #     # torch requires in the form (C, D, H, W)
    #     # video_frames = torch.permute(video_frames, (0, 3, 1, 2))
    #     # print(video_frames.shape)
    #     video_frames = video_frames / 255.0
    #     return video_frames

    def load_video_frames(self, idx: int) -> torch.Tensor:
        vr = VideoReader(str(self.video_paths[idx]))
        indices: List[int] = torch.linspace(
            0, len(vr) - 1, self.num_frames, dtype=torch.float32
        ).tolist()
        frames = torch.from_numpy(vr.get_batch(indices=indices).asnumpy()).permute(
            0, 3, 1, 2
        )
        if self.transform:
            return self.transform(frames)
        return frames

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        if self.num_frames > 32:
            raise Exception("CANT HANDLE 32 FRAMES")

        video: torch.Tensor = self.load_video_frames(idx)
        label: str = self.video_paths[idx].parent.name
        label_idx: int = self.class_to_idx[label]
        return video, label_idx


def load_dataset(download_dir: Path, dataset_name: str = "CricShot10"):
    download_dir.mkdir(exist_ok=True)
    for idx, url in enumerate(DATA_LINKS[dataset_name]):
        download(url, output=str(download_dir / f"{idx}.zip"), fuzzy=True, quiet=True)


def build_dataset(
    config: Config,
    download_dir: Path,
    to_dir: Path,
    dataset_name: str = "CricShot10",
) -> Tuple[DataLoader[CricShot], DataLoader[CricShot]]:
    if not to_dir.exists():
        # download the dataset from google drive to 'download_dir'
        load_dataset(download_dir, dataset_name)
        # unzip the files
        unzip_files(download_dir, config.DATASET_NAME)

        # get the minimum number of videos per label to handle class imbalance
        min_samples_per_class = min(
            (
                len(list(path.glob("*.avi")))
                for path in Path(dataset_name).iterdir()
                if path.is_dir()
            )
        )

        # CricShot10/
        #     train/
        #         flick/
        #             flick_0001.avi
        #             flick_0002.avi
        #             ...
        setup_dataset_structure(
            Path(dataset_name),
            to_dir,
            config.TRAIN_SIZE,
            samples_per_class=min_samples_per_class,
        )

    # creating the train and test directories
    train_dir: Path = to_dir / "train"
    test_dir: Path = to_dir / "test"

    # defining the transforms
    # train_transform: transforms.Compose = transforms.Compose(
    #     [
    #         transforms.RandomHorizontalFlip(p=0.5),
    #         transforms.RandomResizedCrop((224, 224), scale=(0.8, 1.0)),
    #         transforms.ColorJitter(
    #             brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
    #         ),
    #         transforms.RandomRotation(degrees=10),
    #         transforms.ConvertImageDtype(torch.float32),
    #         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #     ]
    # )

    # test_transform: transforms.Compose = transforms.Compose(
    #     [
    #         transforms.Resize((224, 224)),
    #         transforms.ConvertImageDtype(torch.float32),
    #         transforms.Normalize(
    #             mean=[0.485, 0.456, 0.406],  # ImageNet-like normalization
    #             std=[0.229, 0.224, 0.225],
    #         ),
    #     ]
    # )

    # REMOVE Normalization and ConvertImageDtype

    # In dataset.py - modify train_transform
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

    # creating the datasets
    train_dataset: CricShot = CricShot(
        dir=train_dir,
        transform=train_transform,
        num_frames=config.NUM_FRAMES,
        frame_size=config.FRAME_SIZE,
    )
    test_dataset: CricShot = CricShot(
        dir=test_dir,
        transform=test_transform,
        num_frames=config.NUM_FRAMES,
        frame_size=config.FRAME_SIZE,
    )

    # creating the dataloaders
    train_dataloader: DataLoader[CricShot] = DataLoader(
        train_dataset,
        shuffle=True,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
    )
    test_dataloader: DataLoader[CricShot] = DataLoader(
        test_dataset,
        shuffle=False,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
    )

    return train_dataloader, test_dataloader
