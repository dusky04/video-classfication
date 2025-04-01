import os
from pathlib import Path
from typing import Dict, List, Tuple
from zipfile import ZipFile


# CricShot10/
#     train/
#         flick/
#             flick_0001.avi
#             flick_0002.avi
#             ...


def unzip_files(path: str, dataset_name: str) -> None:
    zip_path = Path(path)
    if not zip_path.exists():
        raise FileNotFoundError(f"Directory does not exist: {zip_path}")

    dataset_path = Path(dataset_name)
    Path.mkdir(dataset_path, exist_ok=True)
    print(f"Creating directory: {dataset_path}")

    print("Unzipping all the files: ")
    for zip_file_path in zip_path.glob("*.zip"):
        with ZipFile(zip_file_path) as zip_file:
            print("Unzipping:", zip_file_path.name)
            zip_file.extractall(dataset_path)

    print(f"All files unzipped to: `{dataset_path.absolute()}`")


def get_classes(root_dir: str) -> Tuple[List[str], Dict[str, int]]:
    class_names = sorted(
        [entry.name for entry in list(os.scandir(root_dir)) if entry.is_dir()]
    )
    if not class_names:  # checks if class_names list is empty
        raise FileNotFoundError(
            f"Couldn't find any classes in {root_dir}. Check File Structure."
        )
    class_to_idx = {s: i for i, s in enumerate(class_names)}
    return class_names, class_to_idx


def create_label_folders(path: Path, labels: List[str]) -> None:
    assert path.exists(), f"Provided path: {path} does not exist."
    # Create the label folders
    for label in labels:
        Path.mkdir(path / label, exist_ok=True)


def move_files(files: List[Path], destination: Path) -> None:
    for file in files:
        file_dest_path = destination / file.name
        os.rename(file, file_dest_path)


def create_and_populate_train_test_dirs(
    from_dir: str,
    to_dir: str,
    train_ratio: float,
    samples_per_class: int,
) -> None:
    from_dir = Path(from_dir)
    to_dir = Path(to_dir)
    train_dir = to_dir / "train"
    test_dir = to_dir / "test"

    assert 0 <= train_ratio <= 1, "train_ratio must be between 0 and 1."
    assert samples_per_class >= 0, "samples_per_class must be a positive integer."
    assert from_dir.is_dir(), f"from_dir: {from_dir} must be a valid directory"

    train_sample_count = int(samples_per_class * train_ratio)

    all_class_dirs = from_dir.glob("*")
    for class_dir in all_class_dirs:
        video_files = list(class_dir.glob("*.avi"))
        if len(video_files) < samples_per_class:
            print(
                f"Warning: Only found {len(video_files)} videos in {class_dir}, expected {samples_per_class}."
            )
        video_files = video_files[:samples_per_class]

        # Create train and test sets
        train_videos, test_videos = (
            video_files[:train_sample_count],
            video_files[train_sample_count:],
        )

        train_class_dir = train_dir / class_dir.stem
        test_class_dir = test_dir / class_dir.stem

        move_files(train_videos, train_class_dir)
        move_files(test_videos, test_class_dir)


def setup_dataset_structure(
    from_dir: str, to_dir: str, train_ratio: float, samples_per_class: int
) -> None:
    from_dir: Path = Path(from_dir)
    if not from_dir.exists():
        print(f"Dataset Path: {from_dir} does not exist")
        return

    print(f"Found dataset at: {from_dir}")

    # Create root directory
    root_dir = Path(to_dir)
    print(f"LOG: Creating directory '{root_dir}'")
    Path.mkdir(root_dir, exist_ok=True)

    train_dir = root_dir / "train"
    test_dir = root_dir / "test"

    # Get all class names
    labels, _ = get_classes(from_dir)

    Path.mkdir(train_dir, exist_ok=True)
    print(f"LOG: Creating directory '{train_dir}'")
    Path.mkdir(test_dir, exist_ok=True)
    print(f"LOG: Creating directory '{test_dir}'")

    create_label_folders(train_dir, labels)
    create_label_folders(test_dir, labels)

    create_and_populate_train_test_dirs(
        from_dir=from_dir,
        to_dir=root_dir,
        train_ratio=train_ratio,
        samples_per_class=samples_per_class,
    )
