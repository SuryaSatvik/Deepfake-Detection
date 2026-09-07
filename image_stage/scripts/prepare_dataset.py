import os
import cv2
import random
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ---------------- CONFIG ----------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

CLASSES = ["real", "deepfake", "ai_generated", "edited"]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42

# For videos
FRAMES_PER_VIDEO = 8   # increase later if needed
MIN_FRAME_GAP = 5      # avoid near-identical frames
# ----------------------------------------


def ensure_dirs():
    for split in ["train", "val", "test"]:
        for cls in CLASSES:
            os.makedirs(PROCESSED_DIR / split / cls, exist_ok=True)


def is_image(path: Path):
    return path.suffix.lower() in IMAGE_EXTS


def is_video(path: Path):
    return path.suffix.lower() in VIDEO_EXTS


def list_files(folder: Path):
    if not folder.exists():
        return []
    return [p for p in folder.iterdir() if p.is_file()]


def copy_image(src_path: Path, dst_path: Path):
    shutil.copy2(src_path, dst_path)


def extract_frames_from_video(video_path: Path, output_dir: Path, frames_per_video=8):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[WARNING] Could not open video: {video_path}")
        return 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        print(f"[WARNING] No frames found in: {video_path}")
        cap.release()
        return 0

    # Uniformly sample frame indices
    candidate_indices = []
    step = max(total_frames // frames_per_video, MIN_FRAME_GAP)
    idx = 0
    while idx < total_frames and len(candidate_indices) < frames_per_video:
        candidate_indices.append(idx)
        idx += step

    saved = 0
    video_stem = video_path.stem

    for i, frame_idx in enumerate(candidate_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue

        out_name = f"{video_stem}_frame_{i:03d}.jpg"
        out_path = output_dir / out_name
        cv2.imwrite(str(out_path), frame)
        saved += 1

    cap.release()
    return saved


def collect_class_samples(class_name: str):
    """
    Returns a list of tuples:
    ("image", image_path) or ("video", video_path)
    """
    class_dir = RAW_DIR / class_name
    files = list_files(class_dir)

    samples = []
    for f in files:
        if is_image(f):
            samples.append(("image", f))
        elif is_video(f):
            samples.append(("video", f))
        else:
            print(f"[SKIP] Unsupported file: {f.name}")
    return samples


def split_samples(samples):
    if len(samples) < 3:
        raise ValueError("Need at least 3 samples in a class to make train/val/test split.")

    train_samples, temp_samples = train_test_split(
        samples,
        test_size=(1 - TRAIN_RATIO),
        random_state=RANDOM_SEED,
        shuffle=True
    )

    # temp = val + test
    relative_test_ratio = TEST_RATIO / (VAL_RATIO + TEST_RATIO)
    val_samples, test_samples = train_test_split(
        temp_samples,
        test_size=relative_test_ratio,
        random_state=RANDOM_SEED,
        shuffle=True
    )

    return train_samples, val_samples, test_samples


def process_split(samples, split_name, class_name):
    split_dir = PROCESSED_DIR / split_name / class_name
    saved_count = 0

    for sample_type, src_path in tqdm(samples, desc=f"{class_name}-{split_name}"):
        if sample_type == "image":
            dst_name = src_path.name
            dst_path = split_dir / dst_name
            copy_image(src_path, dst_path)
            saved_count += 1

        elif sample_type == "video":
            saved_count += extract_frames_from_video(
                src_path,
                split_dir,
                frames_per_video=FRAMES_PER_VIDEO
            )

    return saved_count


def main():
    random.seed(RANDOM_SEED)
    ensure_dirs()

    print(f"RAW_DIR: {RAW_DIR}")
    print(f"PROCESSED_DIR: {PROCESSED_DIR}")

    for class_name in CLASSES:
        print(f"\n=== Processing class: {class_name} ===")
        samples = collect_class_samples(class_name)

        if len(samples) == 0:
            print(f"[WARNING] No files found in {RAW_DIR / class_name}")
            continue

        print(f"Found {len(samples)} source files")

        if len(samples) < 3:
            print(f"[WARNING] Not enough files in {class_name} for split. Add more data.")
            continue

        train_samples, val_samples, test_samples = split_samples(samples)

        train_saved = process_split(train_samples, "train", class_name)
        val_saved = process_split(val_samples, "val", class_name)
        test_saved = process_split(test_samples, "test", class_name)

        print(f"[DONE] {class_name}")
        print(f"  Train saved: {train_saved}")
        print(f"  Val saved:   {val_saved}")
        print(f"  Test saved:  {test_saved}")

    print("\nDataset preparation complete.")


if __name__ == "__main__":
    main()