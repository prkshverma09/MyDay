"""ffmpeg helpers for practice clips, uploads, frames, and an optional camera."""

from __future__ import annotations

import subprocess
from pathlib import Path

from reminisce.settings import CLIPS_DIR, MAX_CLIP_SECONDS, MIN_CLIP_SECONDS

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


class MediaError(RuntimeError):
    pass


def _run(command: list[str]) -> str:
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "ffmpeg failed").strip()
        raise MediaError(detail[-500:])
    return (completed.stdout or "").strip()


def make_title_clip(path: Path, *, color: str, title: str, subtitle: str, seconds: int = 6) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    video_filter = (
        "drawbox=x=60:y=90:w=520:h=180:color=black@0.45:t=fill,"
        f"drawtext=fontfile={FONT}:text='{title}':fontsize=52:fontcolor=white:"
        "x=(w-text_w)/2:y=120,"
        f"drawtext=fontfile={FONT}:text='{subtitle}':fontsize=26:fontcolor=white:"
        "x=(w-text_w)/2:y=190"
    )
    _run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"color=c={color}:s=640x360:d={seconds}:r=8",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=r=44100:cl=mono",
            "-filter:v",
            video_filter,
            "-shortest",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            str(path),
        ]
    )


def probe_duration(path: Path) -> float:
    raw = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "csv=p=0",
            str(path),
        ]
    )
    try:
        return float(raw)
    except ValueError as exc:
        raise MediaError(f"Could not read duration of {path.name}") from exc


def validate_duration(seconds: float) -> None:
    if seconds < MIN_CLIP_SECONDS - 0.2 or seconds > MAX_CLIP_SECONDS + 0.2:
        raise MediaError(
            f"Clips need to be between {MIN_CLIP_SECONDS} and {MAX_CLIP_SECONDS} seconds "
            f"(this one is {seconds:.1f}s)."
        )


def extract_frame(video: Path, image: Path) -> Path:
    image.parent.mkdir(parents=True, exist_ok=True)
    _run(["ffmpeg", "-y", "-ss", "1", "-i", str(video), "-frames:v", "1", str(image)])
    return image


def to_wav(source: Path, wav: Path) -> Path:
    wav.parent.mkdir(parents=True, exist_ok=True)
    _run(["ffmpeg", "-y", "-i", str(source), "-ar", "16000", "-ac", "1", str(wav)])
    return wav


def normalize_upload(source: Path, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(source),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            str(dest),
        ]
    )
    return dest


def record_camera(dest: Path, seconds: int = 8) -> Path | None:
    if not Path("/dev/video0").exists():
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "v4l2",
            "-t",
            str(seconds),
            "-i",
            "/dev/video0",
            "-vf",
            "scale=640:360",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(dest),
        ]
    )
    return dest


def clips_dir() -> Path:
    CLIPS_DIR.mkdir(parents=True, exist_ok=True)
    return CLIPS_DIR
