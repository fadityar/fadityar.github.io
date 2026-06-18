from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import cv2
import imageio_ffmpeg


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "assets" / "videos"
TMP_DIR = ROOT / ".tmp-video"


def natural_key(path: Path) -> tuple[int, str]:
    stem = path.stem
    if stem == "video-makkah":
        return (0, path.name)
    suffix = stem.replace("video-makkah", "").strip()
    return (int(suffix or 0), path.name)


def expand_box(x: int, y: int, w: int, h: int, width: int, height: int) -> tuple[int, int, int, int]:
    pad_x = int(w * 0.28)
    pad_y = int(h * 0.32)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(width, x + w + pad_x)
    y2 = min(height, y + h + pad_y)
    return x1, y1, x2, y2


def blur_region(frame, box: tuple[int, int, int, int]) -> None:
    x1, y1, x2, y2 = box
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return

    small_w = max(1, roi.shape[1] // 12)
    small_h = max(1, roi.shape[0] // 12)
    pixelated = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
    pixelated = cv2.resize(pixelated, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_NEAREST)
    kernel = max(21, (min(roi.shape[:2]) // 2) | 1)
    frame[y1:y2, x1:x2] = cv2.GaussianBlur(pixelated, (kernel, kernel), 0)


def detect_faces(frame, frontal, profile) -> list[tuple[int, int, int, int]]:
    height, width = frame.shape[:2]
    target_width = min(320, width)
    scale = target_width / width
    resized = cv2.resize(frame, (target_width, int(height * scale)))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    detections: list[tuple[int, int, int, int]] = []
    min_size = max(18, int(target_width * 0.04))

    for cascade in (frontal, profile):
        faces = cascade.detectMultiScale(
            gray,
            scaleFactor=1.08,
            minNeighbors=4,
            flags=cv2.CASCADE_SCALE_IMAGE,
            minSize=(min_size, min_size),
        )
        detections.extend(tuple(map(int, face)) for face in faces)

    flipped = cv2.flip(gray, 1)
    for x, y, w, h in profile.detectMultiScale(
        flipped,
        scaleFactor=1.08,
        minNeighbors=4,
        flags=cv2.CASCADE_SCALE_IMAGE,
        minSize=(min_size, min_size),
    ):
        detections.append((target_width - x - w, y, w, h))

    boxes = []
    for x, y, w, h in detections:
        boxes.append(
            expand_box(
                int(x / scale),
                int(y / scale),
                int(w / scale),
                int(h / scale),
                width,
                height,
            )
        )
    return boxes


def ffprobe_has_audio(ffmpeg: str, input_path: Path) -> bool:
    command = [
        ffmpeg,
        "-hide_banner",
        "-i",
        str(input_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    return "Audio:" in (result.stderr or "")


def mux_with_ffmpeg(ffmpeg: str, temp_video: Path, source: Path, output: Path) -> None:
    has_audio = ffprobe_has_audio(ffmpeg, source)
    command = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(temp_video),
        "-i",
        str(source),
        "-map",
        "0:v:0",
    ]
    if has_audio:
        command += ["-map", "1:a:0?", "-c:a", "aac", "-b:a", "128k"]
    command += [
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "30",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-shortest",
        str(output),
    ]
    subprocess.run(command, check=True)


def process_video(input_path: Path, output_path: Path, frontal, profile, ffmpeg: str) -> dict:
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    temp_path = TMP_DIR / f"{input_path.stem}.tmp.mp4"
    writer = cv2.VideoWriter(
        str(temp_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create temp video for {input_path}")

    frame_count = 0
    faces_count = 0
    last_boxes: list[tuple[int, int, int, int]] = []
    detection_interval = 20
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_count % detection_interval == 0:
            last_boxes = detect_faces(frame, frontal, profile)
        for box in last_boxes:
            blur_region(frame, box)
        faces_count += len(last_boxes)
        writer.write(frame)
        frame_count += 1

    cap.release()
    writer.release()
    mux_with_ffmpeg(ffmpeg, temp_path, input_path, output_path)
    temp_path.unlink(missing_ok=True)

    return {
        "input": input_path.name,
        "output": output_path.as_posix(),
        "frames": frame_count,
        "expected_frames": frames_total,
        "faces_blurred": faces_count,
        "width": width,
        "height": height,
        "fps": round(fps, 2),
        "size_bytes": output_path.stat().st_size,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    cascade_dir = Path(cv2.data.haarcascades)
    frontal = cv2.CascadeClassifier(str(cascade_dir / "haarcascade_frontalface_default.xml"))
    profile = cv2.CascadeClassifier(str(cascade_dir / "haarcascade_profileface.xml"))
    if frontal.empty() or profile.empty():
        raise RuntimeError("OpenCV face cascades are not available.")

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    videos = sorted(ROOT.glob("video-makkah*.mp4"), key=natural_key)
    results = []

    for index, input_path in enumerate(videos, start=1):
        output_path = OUTPUT_DIR / f"makkah-footage-{index:02d}.mp4"
        if output_path.exists() and output_path.stat().st_size > 100_000:
            print(f"Skipping existing {output_path.relative_to(ROOT)}", flush=True)
            continue
        print(f"Processing {input_path.name} -> {output_path.relative_to(ROOT)}", flush=True)
        results.append(process_video(input_path, output_path, frontal, profile, ffmpeg))

    shutil.rmtree(TMP_DIR, ignore_errors=True)
    (OUTPUT_DIR / "blur-report.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
