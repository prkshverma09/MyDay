import json
import subprocess
from pathlib import Path

from video_chunker.models import MediaInfo, ProbeError

def _parse_rate(value: str) -> float:
    if "/" in value:
        num, den = value.split("/", 1)
        den_f = float(den)
        if den_f == 0:
            return 0.0
        return float(num) / den_f
    return float(value)

def parse_probe_json(payload: dict, source: Path) -> MediaInfo:
    streams = payload.get("streams") or []
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise ProbeError(f"{source} has no video stream")
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    fmt = payload.get("format") or {}
    raw_duration = fmt.get("duration") or video.get("duration")
    if raw_duration is None:
        raise ProbeError(f"{source} has no duration")
    duration_s = float(raw_duration)
    if duration_s <= 0:
        raise ProbeError(f"{source} has a non-positive duration")
    frame_rate = _parse_rate(str(video.get("r_frame_rate") or "0/1"))
    if frame_rate <= 0:
        frame_rate = 25.0
    audio_name = audio.get("codec_name") if audio else None
    return MediaInfo(
        path=source,
        duration_s=duration_s,
        video_codec=str(video.get("codec_name") or "unknown"),
        audio_codec=str(audio_name) if audio_name else None,
        format_name=str(fmt.get("format_name") or "unknown"),
        frame_rate=frame_rate,
    )

def probe_media(source: Path, *, ffprobe_bin: str = "ffprobe") -> MediaInfo:
    result = subprocess.run(
        [
            ffprobe_bin,
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(source),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or "ffprobe failed"
        raise ProbeError(detail)
    return parse_probe_json(json.loads(result.stdout), source)
