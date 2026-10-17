from pathlib import Path

from video_chunker.models import (
    MODE_EXACT,
    MODE_FAST,
    CutError,
    MediaInfo,
)

FAST_VIDEO = {"h264", "hevc", "mpeg4"}
FAST_AUDIO = {"aac", "mp3", "ac3"}
EXACT_VIDEO_ARGS = [
    "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
]
EXACT_AUDIO_ARGS = ["-c:a", "aac", "-b:a", "128k"]


def assert_fast_mode_compatible(info: MediaInfo) -> None:
    if info.video_codec not in FAST_VIDEO:
        raise CutError(
            f"fast mode cannot copy video codec {info.video_codec} into mp4; use --mode exact"
        )
    if info.audio_codec is not None and info.audio_codec not in FAST_AUDIO:
        raise CutError(
            f"fast mode cannot copy audio codec {info.audio_codec} into mp4; use --mode exact"
        )


def build_ffmpeg_command(
    *,
    ffmpeg_bin: str,
    source: Path,
    output_pattern: Path,
    chunk_s: float,
    mode: str,
    has_audio: bool,
) -> list[str]:
    cmd = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-map",
        "0:v:0",
    ]
    if has_audio:
        cmd.extend(["-map", "0:a:0"])
    cmd.extend([
        "-f",
        "segment",
        "-segment_time",
        f"{chunk_s:.6f}",
        "-reset_timestamps",
        "1",
    ])
    if mode == MODE_EXACT:
        cmd.extend(EXACT_VIDEO_ARGS)
        if has_audio:
            cmd.extend(EXACT_AUDIO_ARGS)
    elif mode == MODE_FAST:
        cmd.extend(["-c", "copy"])
    else:
        raise CutError(f"unknown mode {mode}")
    cmd.append(str(output_pattern))
    return cmd
