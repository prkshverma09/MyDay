from dataclasses import dataclass
from pathlib import Path

MODE_EXACT = "exact"
MODE_FAST = "fast"
MODES = (MODE_EXACT, MODE_FAST)

class ProbeError(Exception):
    pass

class CutError(Exception):
    pass

class CutFailed(CutError):
    """ffmpeg ran, or the outputs did not match the plan."""

@dataclass(frozen=True)
class Segment:
    index: int
    start_s: float
    end_s: float

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s

@dataclass(frozen=True)
class MediaInfo:
    path: Path
    duration_s: float
    video_codec: str
    audio_codec: str | None
    format_name: str
    frame_rate: float

@dataclass(frozen=True)
class ChunkRecord:
    index: int
    path: Path
    start_s: float
    end_s: float
