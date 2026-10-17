"""One clip in: frame, Cosmos description, embedding, journal, VastDB."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from reminisce.models import ModelError, describe_image, embed_text
from reminisce.settings import CLIPS_DIR, ROOT
from reminisce.store import Clip, upsert_clip
from reminisce import vast_index
from reminisce.media import extract_frame, probe_duration, validate_duration


def fallback_description(tags: list[str], subtitle: str = "") -> str:
    visible = subtitle or ", ".join(tags) or "no words were added"
    return f"Cosmos did not describe this frame. On-screen label: {visible}."


def ingest_video(
    video: Path,
    *,
    clip_id: str,
    owner_id: str,
    recorded_at: str,
    tags: list[str],
    subtitle: str = "",
    shared_with: list[str] | None = None,
    journal_path: Path | None = None,
    mirror: bool = True,
) -> Clip:
    duration = probe_duration(video)
    validate_duration(duration)
    frame = CLIPS_DIR / "frames" / f"{clip_id}.jpg"
    extract_frame(video, frame)
    try:
        description = describe_image(frame)
        source = "cosmos"
    except ModelError:
        description = fallback_description(tags, subtitle)
        source = "fallback"
    vector, _model = embed_text(description)
    try:
        relpath = str(video.resolve().relative_to(ROOT))
    except ValueError:
        relpath = str(video)
    clip = Clip(
        clip_id=clip_id,
        owner_id=owner_id,
        recorded_at=recorded_at,
        duration_s=round(duration, 2),
        cosmos_description=description,
        activity_tags=list(tags),
        embedding_id=f"emb_{clip_id}",
        embedding=vector,
        shared_with=list(shared_with or []),
        source_citation=clip_id,
        media_relpath=relpath,
        description_source=source,
    )
    upsert_clip(clip, journal_path)
    if mirror:
        try:
            vast_index.mirror_clip(clip)
        except Exception:
            # The local journal is already saved. vast_index records a redacted status.
            pass
    return clip


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")
