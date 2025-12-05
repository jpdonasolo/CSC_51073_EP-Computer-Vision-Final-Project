import cv2
from pathlib import Path

"""
split video into 5 seconds 13s -> 5 | 5 | 3 (if the rest < 8 flames -> delete)
writes the clips into the SAME directory and DELETE original ones
"""

REPO_ROOT = Path(__file__).resolve().parents[2]
# Root directory that contains "train" and "val"
ROOT = REPO_ROOT / "finetuning"

# Segment length and minimum frames required
SEGMENT_LENGTH = 5.0  # seconds
MIN_FRAMES = 8

# Video extensions to process
VIDEO_EXTS = {".mp4", ".mov"}

DELETE_ORIGINAL = True  # set to False if you want to keep originals

def is_video_file(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_EXTS

def split_video(in_path: Path):
    """
    Split a video into ~5-second segments using OpenCV only.

    - Reads the video frame by frame.
    - Groups frames into chunks that correspond to SEGMENT_LENGTH seconds.
    - Writes each chunk to a new file: <stem>_segXXX<suffix>.
    - If a chunk has fewer than MIN_FRAMES frames, the corresponding file is deleted.
    """
    # Do not re-process files that already look like segments
    if "_seg" in in_path.stem:
        print(f"Skip already segmented file: {in_path}")
        return

    cap = cv2.VideoCapture(str(in_path))
    if not cap.isOpened():
        print(f"Could not open video: {in_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    if fps <= 0:
        # Fallback to a reasonable fps if metadata is broken
        print(f"Warning: FPS not available for {in_path}, using 30.0 as default.")
        fps = 30.0

    duration = frame_count / fps if frame_count > 0 else 0
    if duration <= 0:
        print(f"Invalid duration for {in_path}, skipping.")
        cap.release()
        return

    print(
        f"Processing {in_path} "
        f"(~{duration:.2f}s, fps ~{fps:.2f}, frames ~{int(frame_count)})"
    )

    # Number of frames per SEGMENT_LENGTH-second segment (rounded)
    frames_per_segment = int(round(SEGMENT_LENGTH * fps))

    # Get frame size and codec for writing
    ret, sample_frame = cap.read()
    if not ret:
        print(f"No frames in {in_path}, skipping.")
        cap.release()
        return

    height, width = sample_frame.shape[:2]
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # rewind to start

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    segment_index = 1
    segment_writer = None
    segment_frame_count = 0
    current_out_path = None

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            # End of video
            break

        if segment_writer is None:
            # Start a new segment
            out_name = f"{in_path.stem}_seg{segment_index:03d}{in_path.suffix}"
            current_out_path = in_path.parent / out_name
            segment_writer = cv2.VideoWriter(
                str(current_out_path),
                fourcc,
                fps,
                (width, height)
            )
            segment_frame_count = 0
            print(f"  Starting new segment: {current_out_path.name}")

        # Write frame to the current segment
        segment_writer.write(frame)
        segment_frame_count += 1
        frame_idx += 1

        # If we reached the frame limit for this segment, close it and
        # decide whether to keep or delete based on MIN_FRAMES
        if segment_frame_count >= frames_per_segment:
            segment_writer.release()
            segment_writer = None

            if segment_frame_count < MIN_FRAMES:
                # Too short -> remove file
                print(
                    f"  Deleting short segment {current_out_path.name} "
                    f"({segment_frame_count} frames < {MIN_FRAMES})"
                )
                current_out_path.unlink(missing_ok=True)
            else:
                print(
                    f"  Finished segment {current_out_path.name} "
                    f"({segment_frame_count} frames)"
                )

            segment_index += 1
            current_out_path = None

    # Handle last open segment (if video ended in the middle of a segment)
    if segment_writer is not None:
        segment_writer.release()
        if segment_frame_count < MIN_FRAMES:
            print(
                f"  Deleting last short segment {current_out_path.name} "
                f"({segment_frame_count} frames < {MIN_FRAMES})"
            )
            current_out_path.unlink(missing_ok=True)
        else:
            print(
                f"  Finished last segment {current_out_path.name} "
                f"({segment_frame_count} frames)"
            )

    cap.release()

    # Optionally delete original video if at least one segment was created
    if DELETE_ORIGINAL and segment_index > 1:
        print(f"Deleting original file: {in_path}")
        in_path.unlink()


def main():
    """
    Walk through finetuning/train and finetuning/val,
    and split all video files for each class directory.
    """
    print(f"Dataset root: {ROOT}")
    for split in ["train", "val"]:
        split_dir = ROOT / split
        if not split_dir.exists():
            print(f"Skip: {split_dir} does not exist")
            continue

        for class_dir in split_dir.iterdir():
            if not class_dir.is_dir():
                continue

            print(f"\n=== Split videos in {class_dir} ===")
            for video_file in class_dir.iterdir():
                if video_file.is_file() and is_video_file(video_file):
                    split_video(video_file)

    print("\nDone.")


if __name__ == "__main__":
    main()