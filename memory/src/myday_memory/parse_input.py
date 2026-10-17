from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class YoloObject:
    label: str
    count: int
    confidence: float
    position: str


@dataclass(frozen=True)
class LlmInput:
    text: str
    start_sec: float | None
    end_sec: float | None
    camera_id: str
    location: str
    sentences: list[str]
    objects: list[YoloObject] = field(default_factory=list)


_WINDOW_RE = re.compile(
    r"(?:Time window|Chunk\s+\d+)\s*:?\s*([0-9.]+)\s*-\s*([0-9.]+)s",
    re.IGNORECASE,
)
_YOLO_RE = re.compile(
    r"^-\s+(.+?)\s+count=(\d+)\s+confidence=([0-9.]+)\s+position=(\S+)\s*$"
)


def parse_llm_input(text: str) -> LlmInput:
    """Split one llm_input block. The model still receives the raw text."""

    camera = ""
    location = ""
    start = end = None
    sentences: list[str] = []
    objects: list[YoloObject] = []
    section = ""
    window = _WINDOW_RE.search(text)
    if window:
        start = float(window.group(1))
        end = float(window.group(2))
    for raw in text.splitlines():
        line = raw.strip()
        if line.lower().startswith("camera:"):
            camera = line.split(":", 1)[1].strip()
            continue
        if line.lower().startswith("location:"):
            location = line.split(":", 1)[1].strip()
            continue
        if line.lower() == "cosmos:":
            section = "cosmos"
            continue
        if line.lower() == "yolo:":
            section = "yolo"
            continue
        if section == "cosmos" and line.startswith("- ") and line[2:].strip().lower() != "none":
            sentences.append(line[2:].strip())
            continue
        if section == "yolo":
            match = _YOLO_RE.match(line)
            if match:
                objects.append(
                    YoloObject(
                        label=match.group(1).strip(),
                        count=int(match.group(2)),
                        confidence=float(match.group(3)),
                        position=match.group(4),
                    )
                )
    return LlmInput(
        text=text,
        start_sec=start,
        end_sec=end,
        camera_id=camera,
        location=location,
        sentences=sentences,
        objects=objects,
    )
