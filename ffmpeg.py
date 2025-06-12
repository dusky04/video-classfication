import os
from pathlib import Path

# remove the scorecard from the bottom of every image
# to reduce unnecessary noise

root = Path("dataset")
cropped_ds = Path("cropped_dir")
Path.mkdir(cropped_ds, parents=True, exist_ok=True)


video_paths = list(root.rglob("*/*/*.avi"))
for video_path in video_paths:
    new_video_path = cropped_ds / video_path.relative_to(root).parent
    new_video_path.mkdir(exist_ok=True, parents=True)
    # command = f'ffmpeg -i "{video_path}" -vf "crop=iw:ih*0.80:0:0" -an {new_video_path / video_path.name}'
    command = f'ffmpeg -i "{video_path}" -s 224x224 -vf "crop=iw:ih*0.80:0:0" -an -loglevel error {new_video_path / video_path.name}'
    os.system(command)
