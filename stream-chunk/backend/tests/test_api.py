"""HTTP contract for the stream-chunk backend."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest

from tests.conftest import generate_clip, post_media, probe_duration

_ID_RE = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}_[0-9]{2}-[0-9]{2}-[0-9]{2}_[0-9a-f]{6}$"
)


def _create(client, **payload):
    response = client.post("/api/sessions", json=payload)
    assert response.status_code == 200, response.text
    return response.json()


def _session_dir(data_root: Path, session_id: str) -> Path:
    return data_root / "sessions" / session_id


def _chunk_mp4s(data_root: Path, session_id: str) -> list[Path]:
    chunks = _session_dir(data_root, session_id) / "chunks"
    if not chunks.is_dir():
        return []
    return sorted(chunks.glob("session_*.mp4"))


def test_health_reports_local_tools(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"ffmpeg": True, "ffprobe": True, "video_chunker": True}


def test_relative_data_dir_ignores_process_cwd(monkeypatch, tmp_path):
    from stream_chunk.config import data_dir, stream_chunk_root

    monkeypatch.delenv("STREAM_CHUNK_DATA", raising=False)
    assert data_dir() == (stream_chunk_root() / "data").resolve()
    monkeypatch.chdir(tmp_path)
    assert data_dir() == (stream_chunk_root() / "data").resolve()
    monkeypatch.setenv("STREAM_CHUNK_DATA", str(tmp_path / "absolute"))
    assert data_dir() == (tmp_path / "absolute").resolve()


def test_frontend_is_not_mounted_by_default(client):
    response = client.get("/")
    assert response.status_code == 404


def test_create_get_complete_without_media(client, data_root):
    created = _create(
        client,
        chunk_duration_s=15,
        mode="exact",
        label="../not-a-path",
    )
    assert set(created) == {
        "id",
        "status",
        "created_at",
        "chunk_duration_s",
        "slack_s",
        "mode",
        "label",
    }
    session_id = created["id"]
    assert _ID_RE.fullmatch(session_id)
    assert created["status"] == "open"
    assert created["chunk_duration_s"] == 15
    assert created["slack_s"] == 0.75
    assert created["mode"] == "exact"
    assert created["label"] == "../not-a-path"
    names = sorted(path.name for path in (data_root / "sessions").iterdir())
    assert names == [session_id]
    assert (data_root / "sessions" / session_id / "session.json").is_file()

    defaults = _create(client)
    assert defaults["chunk_duration_s"] == 15
    assert defaults["mode"] == "exact"
    assert defaults["label"] == ""
    assert defaults["slack_s"] == 0.75

    fetched = client.get(f"/api/sessions/{session_id}")
    assert fetched.status_code == 200
    body = fetched.json()
    assert set(body) == {
        "id",
        "status",
        "created_at",
        "chunk_duration_s",
        "slack_s",
        "mode",
        "label",
        "chunks",
    }
    assert body["status"] == "open"
    assert body["chunks"] == []
    assert body["label"] == "../not-a-path"

    done = client.post(f"/api/sessions/{session_id}/complete")
    assert done.status_code == 200
    assert done.json() == {"id": session_id, "status": "complete", "chunk_count": 0}
    assert client.get(f"/api/sessions/{session_id}").json()["status"] == "complete"

    again = client.post(f"/api/sessions/{session_id}/complete")
    assert again.status_code == 409
    assert set(again.json()) == {"error", "detail"}


def test_rejects_bad_session_json(client):
    bad = client.post(
        "/api/sessions",
        content=b"{",
        headers={"Content-Type": "application/json"},
    )
    assert bad.status_code == 400
    assert bad.json()["error"] == "invalid_json"

    zero = client.post("/api/sessions", json={"chunk_duration_s": 0})
    assert zero.status_code == 400
    negative = client.post("/api/sessions", json={"chunk_duration_s": -3})
    assert negative.status_code == 400
    unknown = client.post("/api/sessions", json={"mode": "copy"})
    assert unknown.status_code == 400
    assert unknown.json()["error"] == "unknown_mode"


def test_unknown_session_is_404(client):
    missing = "0123456789ab"
    assert client.get(f"/api/sessions/{missing}").status_code == 404
    assert client.post(f"/api/sessions/{missing}/complete").status_code == 404
    media = client.post(
        f"/api/sessions/{missing}/media",
        content=b"hello",
        headers={"Content-Type": "video/mp4"},
    )
    assert media.status_code == 404
    assert media.json()["error"] == "unknown_session"
    assert client.get(f"/api/sessions/not-a-valid-id").status_code == 404


def test_slack_keeps_a_15_3_second_clip_as_one_file(client, data_root, tmp_path):
    clip = tmp_path / "over.mp4"
    generate_clip(clip, 15.3)
    duration = probe_duration(clip)
    assert duration <= 15.75

    created = _create(client, chunk_duration_s=15, label="slack")
    session_id = created["id"]
    response = post_media(client, session_id, clip)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["split"] is False
    assert body["window_index"] == 0
    assert body["probed_duration_s"] == pytest.approx(duration, abs=0.05)
    assert len(body["chunks"]) == 1
    chunk = body["chunks"][0]
    assert set(chunk) == {"index", "file", "start_s", "end_s", "bytes", "gap_before_s"}
    assert chunk["index"] == 0
    assert chunk["file"] == "session_0000.mp4"
    assert chunk["start_s"] == pytest.approx(0)
    assert chunk["end_s"] == pytest.approx(body["probed_duration_s"])
    assert chunk["gap_before_s"] == 0
    assert chunk["bytes"] > 0

    stored = _chunk_mp4s(data_root, session_id)
    assert [path.name for path in stored] == ["session_0000.mp4"]
    assert stored[0].stat().st_size == chunk["bytes"]
    manifest = json.loads(
        (_session_dir(data_root, session_id) / "chunks" / "session.chunks.json").read_text()
    )
    assert manifest["chunks"][0]["end_s"] == pytest.approx(duration, abs=0.05)
    assert manifest["chunks"][0]["window_index"] == 0
    assert client.get(f"/api/sessions/{session_id}").json()["status"] == "open"
    assert not (_session_dir(data_root, session_id) / ".staging").exists()


def test_long_clip_splits_into_four_files(client, data_root, tmp_path):
    clip = tmp_path / "long.mp4"
    generate_clip(clip, 47)
    duration = probe_duration(clip)
    assert duration > 45.2

    created = _create(client, chunk_duration_s=15, mode="exact", label="curl")
    session_id = created["id"]
    response = post_media(client, session_id, clip, gap_ms=0)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["split"] is True
    chunks = body["chunks"]
    assert [row["file"] for row in chunks] == [
        "session_0000.mp4",
        "session_0001.mp4",
        "session_0002.mp4",
        "session_0003.mp4",
    ]
    assert [row["index"] for row in chunks] == [0, 1, 2, 3]
    assert chunks[0]["start_s"] == pytest.approx(0)
    assert chunks[0]["end_s"] == pytest.approx(15)
    assert chunks[1]["start_s"] == pytest.approx(15)
    assert chunks[1]["end_s"] == pytest.approx(30)
    assert chunks[2]["start_s"] == pytest.approx(30)
    assert chunks[2]["end_s"] == pytest.approx(45)
    assert chunks[3]["start_s"] == pytest.approx(45)
    assert chunks[3]["end_s"] == pytest.approx(body["probed_duration_s"])
    assert body["probed_duration_s"] == pytest.approx(duration, abs=0.05)
    assert all(row["gap_before_s"] == 0 for row in chunks)

    manifest_path = _session_dir(data_root, session_id) / "chunks" / "session.chunks.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["session_id"] == session_id
    assert manifest["chunk_duration_s"] == 15
    assert manifest["slack_s"] == 0.75
    assert manifest["mode"] == "exact"
    assert len(manifest["chunks"]) == 4
    for response_row, stored_row in zip(chunks, manifest["chunks"]):
        assert stored_row["file"] == response_row["file"]
        assert stored_row["start_s"] == pytest.approx(response_row["start_s"])
        assert stored_row["end_s"] == pytest.approx(response_row["end_s"])
        assert stored_row["window_index"] == 0
        assert stored_row["gap_before_s"] == 0
        assert stored_row["bytes"] == response_row["bytes"]
    assert manifest["chunks"][-1]["end_s"] == pytest.approx(duration, abs=0.05)
    assert [path.name for path in _chunk_mp4s(data_root, session_id)] == [
        "session_0000.mp4",
        "session_0001.mp4",
        "session_0002.mp4",
        "session_0003.mp4",
    ]

    listed = client.get(f"/api/sessions/{session_id}")
    assert listed.status_code == 200
    assert listed.json()["chunks"] == manifest["chunks"]

    download = client.get(f"/api/sessions/{session_id}/chunks/session_0000.mp4")
    assert download.status_code == 200
    assert download.headers["content-type"].startswith("video/mp4")
    assert download.content[4:8] == b"ftyp"
    assert len(download.content) == chunks[0]["bytes"]

    window = client.get(f"/api/sessions/{session_id}/chunks/window_0000.mp4")
    assert window.status_code == 404
    escaped = client.get(f"/api/sessions/{session_id}/chunks/../session.json")
    assert escaped.status_code == 404
    other = client.get(f"/api/sessions/{session_id}/chunks/session_0009.mp4")
    assert other.status_code == 404

    done = client.post(f"/api/sessions/{session_id}/complete")
    assert done.status_code == 200
    assert done.json()["chunk_count"] == 4
    assert done.json()["status"] == "complete"


def test_two_posts_get_contiguous_indexes(client, data_root, tmp_path):
    first_clip = tmp_path / "a.mp4"
    second_clip = tmp_path / "b.mp4"
    generate_clip(first_clip, 1.0)
    generate_clip(second_clip, 1.2)
    created = _create(client, chunk_duration_s=15)
    session_id = created["id"]

    first = post_media(client, session_id, first_clip)
    assert first.status_code == 200, first.text
    first_body = first.json()
    assert first_body["split"] is False
    assert first_body["chunks"][0]["file"] == "session_0000.mp4"
    assert first_body["chunks"][0]["index"] == 0
    assert first_body["window_index"] == 0

    second = post_media(client, session_id, second_clip, gap_ms=400)
    assert second.status_code == 200, second.text
    second_body = second.json()
    assert second_body["split"] is False
    assert second_body["window_index"] == 1
    assert second_body["chunks"][0]["file"] == "session_0001.mp4"
    assert second_body["chunks"][0]["index"] == 1
    assert second_body["chunks"][0]["start_s"] == pytest.approx(first_body["chunks"][0]["end_s"])
    assert second_body["chunks"][0]["gap_before_s"] == pytest.approx(0.4)

    manifest = json.loads(
        (_session_dir(data_root, session_id) / "chunks" / "session.chunks.json").read_text()
    )
    assert [row["file"] for row in manifest["chunks"]] == ["session_0000.mp4", "session_0001.mp4"]
    assert manifest["chunks"][0]["window_index"] == 0
    assert manifest["chunks"][0]["gap_before_s"] == 0
    assert manifest["chunks"][1]["window_index"] == 1
    assert manifest["chunks"][1]["gap_before_s"] == pytest.approx(0.4)
    assert [path.name for path in _chunk_mp4s(data_root, session_id)] == [
        "session_0000.mp4",
        "session_0001.mp4",
    ]


def test_status_is_cutting_while_media_is_processed(client, data_root, tmp_path, monkeypatch):
    clip = tmp_path / "short.mp4"
    generate_clip(clip, 1.0)
    created = _create(client, chunk_duration_s=15)
    session_id = created["id"]
    seen: dict[str, str] = {}
    from stream_chunk.ingest import accept_media as real_accept

    def wrapped(session_dir, upload_path, *, gap_before_s):
        payload = json.loads((Path(session_dir) / "session.json").read_text(encoding="utf-8"))
        seen["status"] = payload["status"]
        return real_accept(session_dir, upload_path, gap_before_s=gap_before_s)

    monkeypatch.setattr("stream_chunk.app.accept_media", wrapped)
    response = post_media(client, session_id, clip)
    assert response.status_code == 200, response.text
    assert seen["status"] == "cutting"
    assert client.get(f"/api/sessions/{session_id}").json()["status"] == "open"


def test_empty_and_unreadable_media(client, data_root):
    created = _create(client)
    session_id = created["id"]
    empty = client.post(
        f"/api/sessions/{session_id}/media",
        content=b"",
        headers={"Content-Type": "video/mp4"},
    )
    assert empty.status_code == 400
    assert empty.json()["error"] == "empty_body"
    assert _chunk_mp4s(data_root, session_id) == []

    text = client.post(
        f"/api/sessions/{session_id}/media",
        content=b"this is plain text, not a video",
        headers={"Content-Type": "video/mp4"},
    )
    assert text.status_code == 400
    assert text.json() == {
        "error": "unreadable_media",
        "detail": "ffprobe could not read a duration",
    }
    assert _chunk_mp4s(data_root, session_id) == []
    windows = _session_dir(data_root, session_id) / "windows"
    assert [path.name for path in windows.iterdir()] == ["window_0000.mp4"]
    chunks = _session_dir(data_root, session_id) / "chunks"
    assert not chunks.exists() or list(chunks.iterdir()) == []
    assert client.get(f"/api/sessions/{session_id}").json()["status"] == "open"

    wrong_type = client.post(
        f"/api/sessions/{session_id}/media",
        content=b"hello",
        headers={"Content-Type": "text/plain"},
    )
    assert wrong_type.status_code == 400


def test_media_after_complete_is_409(client):
    created = _create(client)
    session_id = created["id"]
    assert client.post(f"/api/sessions/{session_id}/complete").status_code == 200
    response = client.post(
        f"/api/sessions/{session_id}/media",
        content=b"not-empty",
        headers={"Content-Type": "video/mp4"},
    )
    assert response.status_code == 409
    assert set(response.json()) == {"error", "detail"}


def test_upload_over_the_limit_is_413(client, data_root, monkeypatch):
    monkeypatch.setenv("STREAM_CHUNK_MAX_UPLOAD_MB", "0.0001")
    created = _create(client)
    session_id = created["id"]
    response = client.post(
        f"/api/sessions/{session_id}/media",
        content=b"x" * 4096,
        headers={"Content-Type": "video/mp4"},
    )
    assert response.status_code == 413
    assert set(response.json()) == {"error", "detail"}
    assert _chunk_mp4s(data_root, session_id) == []


def test_fast_mode_rejects_vp8_and_mentions_exact(client, data_root, tmp_path):
    clip = tmp_path / "vp8.webm"
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
            "1",
            "-an",
            "-c:v",
            "libvpx",
            "-deadline",
            "realtime",
            "-cpu-used",
            "8",
            "-b:v",
            "200k",
            "-f",
            "webm",
            str(clip),
        ],
        check=True,
        timeout=60,
    )
    created = _create(client, mode="fast", chunk_duration_s=15)
    session_id = created["id"]
    response = post_media(client, session_id, clip, content_type="video/webm")
    assert response.status_code == 400, response.text
    body = response.json()
    assert "exact" in body["detail"].lower()
    assert _chunk_mp4s(data_root, session_id) == []
