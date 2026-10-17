"""Environment configuration for the stream-chunk API."""

from __future__ import annotations

import os
from pathlib import Path


def stream_chunk_root() -> Path:
    """The stream-chunk directory that contains backend/ and frontend/.

    This file lives at backend/src/stream_chunk/config.py, so the project
    root is three parents up. Relative data paths use that directory, not
    the process working directory, so tests and uvicorn agree.
    """
    return Path(__file__).resolve().parents[3]


def data_dir() -> Path:
    """Root for session folders.

    ``STREAM_CHUNK_DATA`` defaults to ``stream-chunk/data``. An absolute
    value is used as given. A relative value is resolved against the
    stream-chunk directory.
    """
    raw = os.environ.get("STREAM_CHUNK_DATA")
    root = stream_chunk_root()
    if raw is None or raw.strip() == "":
        path = root / "data"
    else:
        candidate = Path(raw.strip())
        path = candidate if candidate.is_absolute() else root / candidate
    return path.resolve()


def default_seconds() -> float:
    return float(os.environ.get("STREAM_CHUNK_DEFAULT_SECONDS", "15"))


def slack_seconds() -> float:
    return float(os.environ.get("STREAM_CHUNK_SLACK_SECONDS", "0.75"))


def max_upload_bytes() -> int:
    megabytes = float(os.environ.get("STREAM_CHUNK_MAX_UPLOAD_MB", "200"))
    if megabytes < 0:
        return 0
    return int(megabytes * 1024 * 1024)


def host() -> str:
    value = os.environ.get("STREAM_CHUNK_HOST", "127.0.0.1").strip()
    return value or "127.0.0.1"


def port() -> int:
    return int(os.environ.get("STREAM_CHUNK_PORT", "8765"))


def cors_origins() -> list[str]:
    raw = os.environ.get("STREAM_CHUNK_CORS_ORIGIN", "http://127.0.0.1:8766")
    return [part.strip() for part in raw.split(",") if part.strip()]


def serve_frontend() -> bool:
    return os.environ.get("STREAM_CHUNK_SERVE_FRONTEND", "").strip() == "1"
