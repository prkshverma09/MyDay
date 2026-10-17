"""Fixtures for the stream-chunk API tests.

Generated clips stay tiny: 160x120, 10 fps, no audio, so exact-mode
re-encodes finish quickly. Nothing here reads sample-chunks/.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


@pytest.fixture
def data_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "data"
    root.mkdir()
    monkeypatch.setenv("STREAM_CHUNK_DATA", str(root))
    return root


@pytest.fixture
def client(data_root: Path):
    from fastapi.testclient import TestClient
    from stream_chunk.app import app

    test_client = TestClient(app)
    # Exact-mode re-encodes of the longer clips outlast httpx's 5s default.
    test_client.timeout = 600
    return test_client


def generate_clip(path: Path, seconds: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=160x120:rate=10",
            "-t",
            str(seconds),
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-preset",
            "ultrafast",
            str(path),
        ],
        check=True,
        timeout=180,
    )


def probe_duration(path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return float(result.stdout.strip())


def post_media(client, session_id: str, clip: Path, *, gap_ms: int | None = None, content_type: str = "video/mp4"):
    headers = {"Content-Type": content_type}
    if gap_ms is not None:
        headers["X-Gap-Before-Ms"] = str(gap_ms)
    return client.post(
        f"/api/sessions/{session_id}/media",
        content=clip.read_bytes(),
        headers=headers,
    )
