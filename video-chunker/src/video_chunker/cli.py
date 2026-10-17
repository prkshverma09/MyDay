import argparse
import os
import shutil
import sys
from pathlib import Path

from video_chunker.cut import chunk_video
from video_chunker.models import MODES, CutError, CutFailed, ProbeError

def resolve_bin(explicit: str | None, name: str) -> str:
    if explicit:
        candidate = Path(explicit)
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
        raise CutError(f"{name} not executable: {explicit}")
    found = shutil.which(name)
    if not found:
        raise CutError(
            f"{name} not found. Install ffmpeg 5 or newer, or pass --{name}."
        )
    return found

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="video-chunker",
        description="Cut a video into timed MP4 chunks.",
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("--duration", type=float, default=15.0)
    parser.add_argument("--output", type=Path, default=Path("chunks"))
    parser.add_argument("--mode", choices=MODES, default="exact")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--ffmpeg", default=None)
    parser.add_argument("--ffprobe", default=None)
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code) if exc.code is not None else 2
    if args.duration <= 0:
        print("duration must be > 0 seconds", file=sys.stderr)
        return 2
    try:
        ffmpeg_bin = resolve_bin(args.ffmpeg, "ffmpeg")
        ffprobe_bin = resolve_bin(args.ffprobe, "ffprobe")
        if not args.input.is_file():
            raise CutError(f"{args.input} is not a file")
        chunks = chunk_video(
            args.input,
            args.output,
            chunk_s=args.duration,
            mode=args.mode,
            overwrite=args.overwrite,
            ffmpeg_bin=ffmpeg_bin,
            ffprobe_bin=ffprobe_bin,
        )
    except CutFailed as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except (CutError, ProbeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    for chunk in chunks:
        print(chunk.path.resolve())
    return 0
