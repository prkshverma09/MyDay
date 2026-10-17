"""Local journal of clips. VastDB is the shared index; this file is the working copy."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from reminisce.settings import DEFAULT_PREFS, JOURNAL_PATH, PREFS_PATH


@dataclass
class Clip:
    clip_id: str
    owner_id: str
    recorded_at: str
    duration_s: float
    cosmos_description: str
    activity_tags: list[str]
    embedding_id: str
    embedding: list[float]
    shared_with: list[str] = field(default_factory=list)
    source_citation: str = ""
    media_relpath: str = ""
    description_source: str = "cosmos"

    def __post_init__(self) -> None:
        if not self.source_citation:
            self.source_citation = self.clip_id


def clip_from_dict(row: dict) -> Clip:
    tags = row.get("activity_tags") or []
    shared = row.get("shared_with") or []
    if isinstance(tags, str):
        tags = [part for part in tags.split(",") if part]
    if isinstance(shared, str):
        shared = [part for part in shared.split(",") if part]
    embedding = row.get("embedding") or []
    if isinstance(embedding, str):
        embedding = json.loads(embedding) if embedding else []
    return Clip(
        clip_id=row["clip_id"],
        owner_id=row.get("owner_id") or "user_123",
        recorded_at=row.get("recorded_at") or "",
        duration_s=float(row.get("duration_s") or 0),
        cosmos_description=row.get("cosmos_description") or "",
        activity_tags=list(tags),
        embedding_id=row.get("embedding_id") or f"emb_{row['clip_id']}",
        embedding=[float(value) for value in embedding],
        shared_with=list(shared),
        source_citation=row.get("source_citation") or row["clip_id"],
        media_relpath=row.get("media_relpath") or "",
        description_source=row.get("description_source") or "cosmos",
    )


def public_dict(clip: Clip) -> dict:
    row = asdict(clip)
    row.pop("embedding", None)
    return row


def _read_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text())


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2))
    temporary.replace(path)


def load_clips(path: Path | None = None) -> list[Clip]:
    path = path or JOURNAL_PATH
    rows = _read_json(path, [])
    clips = [clip_from_dict(row) for row in rows]
    clips.sort(key=lambda clip: clip.recorded_at)
    return clips


def save_clips(clips: list[Clip], path: Path | None = None) -> None:
    path = path or JOURNAL_PATH
    ordered = sorted(clips, key=lambda clip: clip.recorded_at)
    _write_json(path, [asdict(clip) for clip in ordered])


def upsert_clip(clip: Clip, path: Path | None = None) -> list[Clip]:
    path = path or JOURNAL_PATH
    clips = [item for item in load_clips(path) if item.clip_id != clip.clip_id]
    clips.append(clip)
    save_clips(clips, path)
    return clips


def set_shared(clip_id: str, carer_id: str, shared: bool, path: Path | None = None) -> Clip:
    path = path or JOURNAL_PATH
    clips = load_clips(path)
    found = None
    for clip in clips:
        if clip.clip_id == clip_id:
            found = clip
            break
    if found is None:
        raise KeyError(clip_id)
    others = [carer for carer in found.shared_with if carer != carer_id]
    if shared:
        others.append(carer_id)
    found.shared_with = others
    save_clips(clips, path)
    return found


def next_clip_id(clips: list[Clip], when: datetime | None = None) -> str:
    """Name a clip by when it was taken, e.g. clip_20261017_144612."""
    base = f"clip_{(when or datetime.now().astimezone()):%Y%m%d_%H%M%S}"
    taken = {clip.clip_id for clip in clips}
    clip_id, n = base, 2
    while clip_id in taken:
        clip_id, n = f"{base}_{n}", n + 1
    return clip_id


def load_prefs(path: Path | None = None) -> dict:
    path = path or PREFS_PATH
    prefs = dict(DEFAULT_PREFS)
    prefs.update(_read_json(path, {}))
    return prefs


def save_prefs(prefs: dict, path: Path | None = None) -> None:
    path = path or PREFS_PATH
    merged = dict(DEFAULT_PREFS)
    merged.update(prefs)
    _write_json(path, merged)


def human_when(iso_timestamp: str, now: datetime | None = None) -> str:
    if not iso_timestamp:
        return "an unknown time"
    try:
        moment = datetime.fromisoformat(iso_timestamp)
    except ValueError:
        return iso_timestamp
    clock = moment.strftime("%H:%M")
    today = (now or datetime.now().astimezone()).date()
    if moment.date() == today:
        return f"today {clock}"
    return f"{moment.strftime('%d %b')} {clock}"


def day_part(iso_timestamp: str) -> str:
    try:
        hour = datetime.fromisoformat(iso_timestamp).hour
    except ValueError:
        return "Day"
    if hour < 12:
        return "Morning"
    if hour < 17:
        return "Afternoon"
    return "Evening"
