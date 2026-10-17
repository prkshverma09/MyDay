import subprocess
from pathlib import Path

from video_chunker.models import (
    MODE_EXACT,
    MODE_FAST,
    ChunkRecord,
    CutError,
    CutFailed,
    MediaInfo,
)
from video_chunker.output import (
    chunk_filename,
    existing_chunk_paths,
    prepare_output_dir,
    write_manifest,
)
from video_chunker.plan import duration_tolerance_s, plan_segments
from video_chunker.probe import probe_media

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
        cmd.extend([
            "-force_key_frames",
            f"expr:gte(t,n_forced*{chunk_s:.6f})",
        ])
        if has_audio:
            cmd.extend(EXACT_AUDIO_ARGS)
    elif mode == MODE_FAST:
        cmd.extend(["-c", "copy"])
    else:
        raise CutError(f"unknown mode {mode}")
    cmd.append(str(output_pattern))
    return cmd


def drop_extra_frame_tail(paths, segment_count, frame_rate, ffprobe_bin):
    """Drop one unplanned file when it is only a frame or two long.

    The segment muxer can emit that extra file when the encoded length runs
    a little past the cut, which happens with Safari camera recordings. The
    picture is already in the previous file. A real remainder is its own
    planned segment and is kept.
    """
    if len(paths) != segment_count + 1:
        return paths
    tail = paths[-1]
    if probe_media(tail, ffprobe_bin=ffprobe_bin).duration_s > duration_tolerance_s(frame_rate):
        return paths
    tail.unlink()
    return paths[:-1]


def _assign_times(paths, segments, mode, ffprobe_bin, frame_rate):
    records = []
    cursor = 0.0
    tolerance = duration_tolerance_s(frame_rate)
    for index, path in enumerate(paths):
        measured = probe_media(path, ffprobe_bin=ffprobe_bin).duration_s
        if mode == MODE_FAST:
            start = cursor
            end = cursor + measured
            cursor = end
        else:
            start = segments[index].start_s
            end = segments[index].end_s
            if abs(measured - (end - start)) > tolerance:
                raise CutFailed(
                    f"{path.name} is {measured:.3f}s; planned {(end - start):.3f}s"
                )
        records.append(ChunkRecord(index, path, start, end))
    return records


def chunk_video(
    source: Path,
    output_dir: Path,
    *,
    chunk_s: float = 15.0,
    mode: str = MODE_EXACT,
    overwrite: bool = False,
    ffmpeg_bin: str = "ffmpeg",
    ffprobe_bin: str = "ffprobe",
) -> list[ChunkRecord]:
    if not source.is_file():
        raise CutError(f"{source} is not a file")
    chunk_filename(source, 0)
    info = probe_media(source, ffprobe_bin=ffprobe_bin)
    if chunk_s < (1.0 / info.frame_rate):
        raise CutError(
            f"chunk duration {chunk_s}s is shorter than one frame "
            f"({1.0 / info.frame_rate:.4f}s)"
        )
    if info.duration_s / chunk_s > 10_000:
        raise CutError("chunk duration would create more than 10000 files")
    if mode == MODE_FAST:
        assert_fast_mode_compatible(info)
    elif mode != MODE_EXACT:
        raise CutError(f"unknown mode {mode}")
    segments = plan_segments(
        info.duration_s,
        chunk_s,
        min_tail_s=1.0 / info.frame_rate,
    )
    prepare_output_dir(output_dir, source, overwrite=overwrite)
    pattern = output_dir / f"{source.stem}_%04d.mp4"
    cmd = build_ffmpeg_command(
        ffmpeg_bin=ffmpeg_bin,
        source=source,
        output_pattern=pattern,
        chunk_s=chunk_s,
        mode=mode,
        has_audio=info.audio_codec is not None,
    )
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        detail = result.stderr.strip() or "ffmpeg failed"
        raise CutFailed(detail)
    paths = existing_chunk_paths(output_dir, source)
    paths = drop_extra_frame_tail(
        paths,
        len(segments),
        info.frame_rate,
        ffprobe_bin,
    )
    if mode == MODE_EXACT and len(paths) != len(segments):
        raise CutFailed(f"expected {len(segments)} chunks, found {len(paths)}")
    if not paths:
        raise CutFailed("ffmpeg produced no chunks")
    records = _assign_times(paths, segments, mode, ffprobe_bin, info.frame_rate)
    if mode == MODE_FAST:
        total = sum(record.end_s - record.start_s for record in records)
        if abs(total - info.duration_s) > 0.5:
            raise CutFailed(
                f"fast mode duration {total:.3f}s is far from source {info.duration_s:.3f}s"
            )
    write_manifest(output_dir, source, chunk_s, mode, records)
    return records
