"""Save one closed upload and publish the chunker's MP4s into the session."""

from __future__ import annotations

import json
import re
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path

from stream_chunk.sessions import atomic_write_json

_WINDOW_NAME = re.compile(r"^window_(\d{4})\.[^.]+$")


class MediaRejected(Exception):
    """The upload cannot be cut. Nothing is published. HTTP 400."""

    def __init__(self, error: str, detail: str) -> None:
        super().__init__(detail)
        self.status_code = 400
        self.error = error
        self.detail = detail


class MediaFailed(Exception):
    """ffmpeg or the chunker failed. Nothing is published. HTTP 500."""

    def __init__(self, error: str, detail: str) -> None:
        super().__init__(detail)
        self.status_code = 500
        self.error = error
        self.detail = detail


@dataclass(frozen=True)
class ChunkRow:
    index: int
    file: str
    start_s: float
    end_s: float
    size_bytes: int
    window_index: int
    gap_before_s: float

    def as_manifest(self) -> dict:
        return {
            "index": self.index,
            "file": self.file,
            "start_s": self.start_s,
            "end_s": self.end_s,
            "bytes": self.size_bytes,
            "window_index": self.window_index,
            "gap_before_s": self.gap_before_s,
        }

    def as_media(self) -> dict:
        row = self.as_manifest()
        row.pop("window_index")
        return row


@dataclass(frozen=True)
class IngestResult:
    window_index: int
    probed_duration_s: float
    split: bool
    chunks: list[ChunkRow]

    def as_response(self, session_id: str) -> dict:
        return {
            "session_id": session_id,
            "window_index": self.window_index,
            "probed_duration_s": self.probed_duration_s,
            "split": self.split,
            "chunks": [row.as_media() for row in self.chunks],
        }


def video_chunker_available() -> bool:
    try:
        from video_chunker.cut import chunk_video
    except ImportError:
        return False
    return callable(chunk_video)


def _detail(exc: BaseException, limit: int = 2000) -> str:
    text = str(exc).strip() or exc.__class__.__name__
    if len(text) > limit:
        return text[:limit]
    return text


def _load_chunker():
    from video_chunker.cut import chunk_video
    from video_chunker.models import CutError, CutFailed, ProbeError
    from video_chunker.probe import probe_media

    return chunk_video, probe_media, ProbeError, CutError, CutFailed


def _fail_from_chunker(exc: Exception, probe_error: type, cut_error: type, cut_failed: type) -> None:
    if isinstance(exc, probe_error):
        raise MediaRejected("unreadable_media", "ffprobe could not read a duration") from exc
    text = str(exc).strip()
    if isinstance(exc, cut_error) and not isinstance(exc, cut_failed) and "fast mode" in text.lower():
        if "exact" not in text.lower():
            text = f"{text}; use exact"
        raise MediaRejected("unsupported_codec", text) from exc
    raise MediaFailed("cut_failed", _detail(exc)) from exc


def _next_session_index(chunks_dir: Path, used: set[int], start: int) -> int:
    index = start
    while index in used or (chunks_dir / f"session_{index:04d}.mp4").exists():
        index += 1
        if index > 9999:
            raise MediaFailed("cut_failed", "session has too many chunks")
    return index


def _publish(
    session_dir: Path,
    session: dict,
    records: list,
    *,
    gap_before_s: float,
    window_index: int,
) -> list[ChunkRow]:
    """Move staged MP4s onto the next free session indexes and rewrite the manifest."""
    chunks_dir = session_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    manifest_file = chunks_dir / "session.chunks.json"
    if manifest_file.is_file():
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    else:
        manifest = {
            "session_id": session["id"],
            "chunk_duration_s": session["chunk_duration_s"],
            "slack_s": session["slack_s"],
            "mode": session["mode"],
            "chunks": [],
        }
    existing = [row for row in manifest.get("chunks") or [] if isinstance(row, dict)]
    offset = float(existing[-1]["end_s"]) if existing else 0.0
    used = {int(row["index"]) for row in existing if "index" in row}
    next_index = _next_session_index(chunks_dir, used, 0)

    rows: list[ChunkRow] = []
    moved: list[Path] = []
    try:
        for record in sorted(records, key=lambda item: item.index):
            next_index = _next_session_index(chunks_dir, used, next_index)
            name = f"session_{next_index:04d}.mp4"
            dest = chunks_dir / name
            Path(record.path).replace(dest)
            moved.append(dest)
            row = ChunkRow(
                index=next_index,
                file=name,
                start_s=offset + float(record.start_s),
                end_s=offset + float(record.end_s),
                size_bytes=dest.stat().st_size,
                window_index=window_index,
                gap_before_s=gap_before_s,
            )
            rows.append(row)
            used.add(next_index)
            next_index += 1
        manifest = {
            "session_id": session["id"],
            "chunk_duration_s": session["chunk_duration_s"],
            "slack_s": session["slack_s"],
            "mode": session["mode"],
            "chunks": existing + [row.as_manifest() for row in rows],
        }
        atomic_write_json(manifest_file, manifest)
    except Exception:
        for path in moved:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        raise
    return rows


def _remove_stage(stage_dir: Path) -> None:
    shutil.rmtree(stage_dir, ignore_errors=True)
    staging_root = stage_dir.parent
    try:
        if staging_root.name == ".staging" and staging_root.is_dir() and not any(staging_root.iterdir()):
            staging_root.rmdir()
    except OSError:
        pass


def accept_media(session_dir: Path, upload_path: Path, *, gap_before_s: float) -> IngestResult:
    """Probe ``upload_path``, apply the slack rule, and publish session MP4s.

    ``chunk_video`` always restarts names at 0000, so each call writes into a
    private stage directory. Publish renames the results to the next free
    ``session_XXXX.mp4`` indexes and rewrites ``session.chunks.json`` by
    temp file and rename. If probing or cutting fails, publish does not run.
    """
    session = json.loads((session_dir / "session.json").read_text(encoding="utf-8"))
    match = _WINDOW_NAME.fullmatch(upload_path.name)
    if match is None:
        raise MediaFailed("cut_failed", "internal window name was not recognized")
    window_index = int(match.group(1))

    chunk_video, probe_media, probe_error, cut_error, cut_failed = _load_chunker()
    try:
        info = probe_media(Path(upload_path))
    except probe_error as exc:
        raise MediaRejected("unreadable_media", "ffprobe could not read a duration") from exc

    configured = float(session["chunk_duration_s"])
    slack = float(session["slack_s"])
    # A window that only overran the cut by the slack stays one file.
    if info.duration_s <= configured + slack:
        chunk_s = info.duration_s
    else:
        chunk_s = configured

    stage_dir = session_dir / ".staging" / uuid.uuid4().hex
    stage_dir.mkdir(parents=True, exist_ok=True)
    try:
        try:
            records = chunk_video(
                Path(upload_path),
                stage_dir,
                chunk_s=chunk_s,
                mode=session["mode"],
                overwrite=True,
            )
        except (probe_error, cut_error) as exc:
            _fail_from_chunker(exc, probe_error, cut_error, cut_failed)
            raise
        rows = _publish(
            session_dir,
            session,
            records,
            gap_before_s=gap_before_s,
            window_index=window_index,
        )
    finally:
        _remove_stage(stage_dir)

    return IngestResult(
        window_index=window_index,
        probed_duration_s=info.duration_s,
        split=len(rows) > 1,
        chunks=rows,
    )
