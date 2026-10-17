from __future__ import annotations

from dataclasses import asdict, dataclass, field


QUESTION_IDS = ("people", "lost_objects", "door_lock", "stove", "appointment")
PERSONAL_OBJECTS = ("keys", "glasses", "hearing aid")


@dataclass(frozen=True)
class ChunkText:
    """One clip after captioning and transcription.

    Pipelines 2 and 3 can leave `detections` empty. Times are ISO-8601 so
    later sightings sort after earlier ones.
    """

    clip_id: str
    camera: str
    place: str
    t_start: str
    t_end: str
    file: str
    caption: str = ""
    transcript: str = ""
    detections: list[str] = field(default_factory=list)
    resident: list[str] = field(default_factory=lambda: ["Margaret"])
    pets: list[str] = field(default_factory=list)
    summary: str = ""
    highlights: list[str] = field(default_factory=list)
    key_moment: bool | None = None

    @classmethod
    def from_dict(cls, raw: dict) -> ChunkText:
        resident = raw.get("resident") or ["Margaret"]
        flagged = raw.get("key_moment")
        return cls(
            clip_id=str(raw["clip_id"]),
            camera=str(raw.get("camera") or ""),
            place=str(raw.get("place") or ""),
            t_start=str(raw.get("t_start") or ""),
            t_end=str(raw.get("t_end") or ""),
            file=str(raw.get("file") or ""),
            caption=str(raw.get("caption") or ""),
            transcript=str(raw.get("transcript") or ""),
            detections=[str(item) for item in raw.get("detections") or []],
            resident=[str(item) for item in resident],
            pets=[str(item) for item in raw.get("pets") or []],
            summary=str(raw.get("summary") or ""),
            highlights=[str(item) for item in raw.get("highlights") or []],
            key_moment=flagged if isinstance(flagged, bool) else None,
        )

    @property
    def source_text(self) -> str:
        return "\n".join(part for part in (self.caption, self.transcript) if part).strip()


@dataclass(frozen=True)
class Answer:
    text: str
    clip_id: str
    t_start: str
    confidence: float
    abstain: bool
    question_id: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MemoryEntry:
    clip_id: str
    t_start: str
    t_end: str
    place: str
    people: list[str]
    action: str
    objects: list[str]
    statement: str
    transcript: str
    answers: list[Answer]
    importance: str
    camera: str = ""
    file: str = ""
    object_details: dict[str, dict[str, str]] = field(default_factory=dict)
    key_moment: bool = False
    highlights: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["answers"] = [answer.to_dict() for answer in self.answers]
        payload["summary"] = self.statement
        return payload


@dataclass
class LastSeen:
    object: str
    place: str
    detail: str
    prep: str
    clip_id: str
    t_start: str
    t_end: str
    camera: str
    statement: str
    file: str

    def to_dict(self) -> dict:
        return asdict(self)


def day_key(timestamp: str) -> str:
    if len(timestamp) >= 10 and timestamp[4] == "-" and timestamp[7] == "-":
        return timestamp[:10]
    return "undated"
