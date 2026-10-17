"""Session folders, ids, and the shared chunks folder."""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from stream_chunk.config import data_dir, default_seconds, slack_seconds

# New folders sort by time: 2026-10-17_14-46-12_a1b2c3
# Older recordings used a 12-character hex id and stay readable.
_ID_RE = re.compile(
    r"^(?:[0-9a-f]{12}|[0-9]{4}-[0-9]{2}-[0-9]{2}_[0-9]{2}-[0-9]{2}-[0-9]{2}_[0-9a-f]{6})$"
)
_WINDOW_RE = re.compile(r"^window_(\d{4})\.[A-Za-z0-9]+$")
# Current files: 2026-10-17_15-24-01_001.mp4. Older sessions used session_0000.mp4.
_CHUNK_RE = re.compile(
    r"^(?:session_\d{4}|\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_\d{3})\.mp4$"
)
_STAMPED_CHUNK = re.compile(
    r"^(?P<stamp>\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})_(?P<seq>\d{3})\.mp4$"
)
_SESSION_STAMP = re.compile(r"^(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})_[0-9a-f]{6}$")
_LEGACY_CHUNK = re.compile(r"^session_(\d{4})\.mp4$")

_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()
_chunk_names = threading.Lock()
logger = logging.getLogger(__name__)


class APIError(Exception):
    def __init__(self, status_code: int, error: str, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.error = error
        self.detail = detail


class UnknownSession(APIError):
    def __init__(self) -> None:
        super().__init__(404, "unknown_session", "session not found")


class UnknownChunk(APIError):
    def __init__(self) -> None:
        super().__init__(404, "unknown_chunk", "chunk is not in this session")


class SessionClosed(APIError):
    def __init__(self) -> None:
        super().__init__(
            409,
            "session_complete",
            "media posted after the session was completed",
        )


class AlreadyComplete(APIError):
    def __init__(self) -> None:
        super().__init__(409, "already_complete", "session is already complete")


def session_lock(session_id: str) -> threading.Lock:
    """Serialize media and complete for one session. Reads do not take it."""
    with _locks_guard:
        lock = _locks.get(session_id)
        if lock is None:
            lock = threading.Lock()
            _locks[session_id] = lock
        return lock


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON by creating a temp file in the same directory, then renaming."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def _require_under_data(path: Path) -> Path:
    root = data_dir().resolve()
    resolved = path.resolve()
    if resolved != root and root not in resolved.parents:
        raise APIError(500, "invalid_path", "path escapes the data directory")
    return resolved


def valid_session_id(session_id: str) -> bool:
    return isinstance(session_id, str) and _ID_RE.fullmatch(session_id) is not None


def new_session_id() -> str:
    """A folder name that sorts in time order, unique within the same second."""
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return f"{stamp}_{uuid.uuid4().hex[:6]}"


def session_directory(session_id: str) -> Path:
    if not valid_session_id(session_id):
        raise UnknownSession()
    root = data_dir().resolve()
    path = _require_under_data(root / "sessions" / session_id)
    if path.parent != (root / "sessions").resolve():
        raise UnknownSession()
    return path


def _session_json(session_id: str) -> Path:
    return session_directory(session_id) / "session.json"


def create_session(
    *,
    chunk_duration_s: float | None = None,
    mode: str = "exact",
    label: str = "",
) -> dict[str, Any]:
    if chunk_duration_s is None:
        chunk_duration_s = default_seconds()
    root = data_dir().resolve()
    sessions = root / "sessions"
    sessions.mkdir(parents=True, exist_ok=True)
    for _ in range(8):
        session_id = new_session_id()
        path = _require_under_data(sessions / session_id)
        try:
            path.mkdir(parents=False, exist_ok=False)
        except FileExistsError:
            continue
        payload = {
            "id": session_id,
            "status": "open",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "chunk_duration_s": chunk_duration_s,
            "slack_s": slack_seconds(),
            "mode": mode,
            "label": label,
        }
        atomic_write_json(path / "session.json", payload)
        return payload
    raise APIError(500, "session_id", "could not allocate a session id")


def get_session(session_id: str) -> dict[str, Any] | None:
    if not valid_session_id(session_id):
        return None
    try:
        path = _session_json(session_id)
    except APIError:
        return None
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return None
    return payload


def set_status(session_id: str, status: str) -> dict[str, Any]:
    payload = get_session(session_id)
    if payload is None:
        raise UnknownSession()
    payload["status"] = status
    atomic_write_json(_session_json(session_id), payload)
    return payload


def chunks_directory() -> Path:
    """The one folder every published chunk is written into."""
    path = _require_under_data(data_dir().resolve() / "chunks")
    path.mkdir(parents=True, exist_ok=True)
    return path


@contextmanager
def reserved_chunk_paths(count: int) -> Iterator[list[Path]]:
    """Reserve timestamped names in the shared chunks folder.

    The lock is held until the caller finishes moving files into those paths,
    so two recordings cannot take the same name.
    """
    if count < 1:
        yield []
        return
    directory = chunks_directory()
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    with _chunk_names:
        used = _used_sequences(directory, stamp)
        paths: list[Path] = []
        seq = 1
        while len(paths) < count:
            if seq not in used:
                paths.append(_require_under_data(directory / f"{stamp}_{seq:03d}.mp4"))
                used.add(seq)
            seq += 1
            if seq > 999:
                raise APIError(500, "too_many_chunks", "too many chunks in the same second")
        yield paths


def _used_sequences(directory: Path, stamp: str) -> set[int]:
    used: set[int] = set()
    for entry in directory.iterdir():
        match = _STAMPED_CHUNK.fullmatch(entry.name)
        if match and match.group("stamp") == stamp:
            used.add(int(match.group("seq")))
    return used


def adopt_session_chunks() -> int:
    """Move chunk MP4s out of per-session folders into the shared folder.

    A session named ``2026-10-17_15-24-01_ab12cd`` keeps that time in the
    filenames (``..._001.mp4``, ``..._002.mp4``). The session manifest is
    updated to the new names. Files already in the shared folder stay put.
    """
    root = data_dir().resolve()
    sessions = root / "sessions"
    if not sessions.is_dir():
        return 0
    moved = 0
    for session_dir in sorted(path for path in sessions.iterdir() if path.is_dir()):
        try:
            moved += _adopt_one_session(session_dir)
        except OSError:
            logger.exception("could not move chunks from %s", session_dir.name)
    return moved


def _adopt_one_session(session_dir: Path) -> int:
    chunks = session_dir / "chunks"
    if not chunks.is_dir():
        return 0
    sources = sorted(
        path for path in chunks.iterdir() if path.is_file() and _LEGACY_CHUNK.fullmatch(path.name)
    )
    if not sources:
        return 0
    stamp_match = _SESSION_STAMP.fullmatch(session_dir.name)
    if stamp_match:
        stamp = stamp_match.group(1)
    else:
        stamp = datetime.fromtimestamp(sources[0].stat().st_mtime).strftime("%Y-%m-%d_%H-%M-%S")
    directory = chunks_directory()
    renamed: dict[str, str] = {}
    with _chunk_names:
        for source in sources:
            seq = int(_LEGACY_CHUNK.fullmatch(source.name).group(1)) + 1
            while True:
                dest = directory / f"{stamp}_{seq:03d}.mp4"
                if not dest.exists():
                    break
                seq += 1
                if seq > 999:
                    raise OSError(f"no free chunk name for {source.name}")
            _require_under_data(dest)
            source.replace(dest)
            renamed[source.name] = dest.name
    manifest_file = chunks / "session.chunks.json"
    if manifest_file.is_file() and renamed:
        payload = json.loads(manifest_file.read_text(encoding="utf-8"))
        rows = payload.get("chunks") if isinstance(payload, dict) else None
        if isinstance(rows, list):
            for row in rows:
                if isinstance(row, dict) and row.get("file") in renamed:
                    row["file"] = renamed[row["file"]]
            atomic_write_json(manifest_file, payload)
    return len(renamed)


def manifest_path(session_id: str) -> Path:
    return session_directory(session_id) / "chunks" / "session.chunks.json"


def load_chunks(session_id: str) -> list[dict[str, Any]]:
    path = manifest_path(session_id)
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    chunks = payload.get("chunks") if isinstance(payload, dict) else None
    if not isinstance(chunks, list):
        return []
    return [row for row in chunks if isinstance(row, dict)]


def write_window(session_id: str, extension: str, body: bytes) -> Path:
    directory = session_directory(session_id)
    windows = directory / "windows"
    windows.mkdir(parents=True, exist_ok=True)
    highest = -1
    for entry in windows.iterdir():
        match = _WINDOW_RE.fullmatch(entry.name)
        if match:
            highest = max(highest, int(match.group(1)))
    next_index = highest + 1
    if next_index > 9999:
        raise APIError(500, "too_many_windows", "session has too many windows")
    if not extension.startswith("."):
        extension = f".{extension}"
    path = windows / f"window_{next_index:04d}{extension}"
    _require_under_data(path)
    path.write_bytes(body)
    return path


def resolve_chunk_file(session_id: str, filename: str) -> Path:
    if get_session(session_id) is None:
        raise UnknownSession()
    if not isinstance(filename, str) or _CHUNK_RE.fullmatch(filename) is None:
        raise UnknownChunk()
    if not any(row.get("file") == filename for row in load_chunks(session_id)):
        raise UnknownChunk()
    shared = chunks_directory()
    path = _require_under_data(shared / filename)
    if path.parent == shared and path.is_file():
        return path
    legacy_dir = _require_under_data(session_directory(session_id) / "chunks")
    legacy = _require_under_data(legacy_dir / filename)
    if legacy.parent == legacy_dir and legacy.is_file():
        return legacy
    raise UnknownChunk()
