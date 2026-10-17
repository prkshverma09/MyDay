import json
from pathlib import Path

import pytest

from tests.conftest import make_sample, requires_ffmpeg
from video_chunker.cut import chunk_video
from video_chunker.models import CutError, MediaInfo
from video_chunker.probe import probe_media

@requires_ffmpeg
def test_exact_chunks_match_duration(tmp_path):
    source = tmp_path / "clip.mp4"
    make_sample(source, 2.0)
    out = tmp_path / "out"
    chunks = chunk_video(source, out, chunk_s=0.5, mode="exact")
    assert [c.path.name for c in chunks] == [
        "clip_0000.mp4",
        "clip_0001.mp4",
        "clip_0002.mp4",
        "clip_0003.mp4",
    ]
    measured = [probe_media(c.path).duration_s for c in chunks]
    for seconds in measured:
        assert seconds == pytest.approx(0.5, abs=0.08)
    manifest = json.loads((out / "clip.chunks.json").read_text(encoding="utf-8"))
    assert manifest["mode"] == "exact"
    assert manifest["chunk_s"] == 0.5

@requires_ffmpeg
def test_short_video_writes_one_file(tmp_path):
    source = tmp_path / "short.mp4"
    make_sample(source, 2.0)
    chunks = chunk_video(source, tmp_path / "out", chunk_s=15, mode="exact")
    assert len(chunks) == 1
    assert probe_media(chunks[0].path).duration_s == pytest.approx(2.0, abs=0.08)

@requires_ffmpeg
def test_remainder_chunk_is_kept(tmp_path):
    source = tmp_path / "tail.mp4"
    make_sample(source, 2.2)
    chunks = chunk_video(source, tmp_path / "out", chunk_s=1.0, mode="exact")
    assert len(chunks) == 3
    assert probe_media(chunks[-1].path).duration_s == pytest.approx(0.2, abs=0.08)

@requires_ffmpeg
def test_second_run_without_overwrite_keeps_bytes(tmp_path):
    source = tmp_path / "clip.mp4"
    make_sample(source, 2.0)
    out = tmp_path / "out"
    first = chunk_video(source, out, chunk_s=0.5, mode="exact")
    original = first[0].path.read_bytes()
    with pytest.raises(CutError, match="already exists"):
        chunk_video(source, out, chunk_s=0.5, mode="exact")
    assert first[0].path.read_bytes() == original

@requires_ffmpeg
def test_overwrite_removes_stale_higher_indexes(tmp_path):
    source = tmp_path / "clip.mp4"
    make_sample(source, 2.0)
    out = tmp_path / "out"
    chunk_video(source, out, chunk_s=0.5, mode="exact")
    chunk_video(source, out, chunk_s=15, mode="exact", overwrite=True)
    names = sorted(p.name for p in out.glob("clip_*.mp4"))
    assert names == ["clip_0000.mp4"]

def test_chunk_video_fast_reject_writes_nothing(tmp_path, monkeypatch):
    source = tmp_path / "film.webm"
    source.write_bytes(b"fake")
    info = MediaInfo(
        path=source,
        duration_s=10.0,
        video_codec="vp9",
        audio_codec="opus",
        format_name="webm",
        frame_rate=25.0,
    )
    monkeypatch.setattr("video_chunker.cut.probe_media", lambda *args, **kwargs: info)
    out = tmp_path / "out"
    with pytest.raises(CutError, match="vp9"):
        chunk_video(source, out, chunk_s=15, mode="fast")
    assert list(out.glob("*.mp4")) == [] if out.exists() else True

def test_chunk_video_rejects_tiny_duration(tmp_path, monkeypatch):
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"fake")
    info = MediaInfo(source, 10.0, "h264", "aac", "mp4", 25.0)
    monkeypatch.setattr("video_chunker.cut.probe_media", lambda *args, **kwargs: info)
    with pytest.raises(CutError, match="one frame"):
        chunk_video(source, tmp_path / "out", chunk_s=0.01, mode="exact")

@requires_ffmpeg
def test_cut_equal_to_duration_stays_one_file(tmp_path):
    source = tmp_path / "whole.mp4"
    make_sample(source, 2.0)
    duration = probe_media(source).duration_s
    chunks = chunk_video(source, tmp_path / "out", chunk_s=duration, mode="exact")
    assert len(chunks) == 1


SAFARI_WINDOW = (
    Path(__file__).resolve().parents[2]
    / "stream-chunk/data/sessions/3618576334a3/windows/window_0000.mp4"
)


@requires_ffmpeg
@pytest.mark.skipif(not SAFARI_WINDOW.is_file(), reason="local Safari capture is not present")
def test_safari_camera_window_is_one_chunk(tmp_path):
    info = probe_media(SAFARI_WINDOW)
    chunks = chunk_video(SAFARI_WINDOW, tmp_path / "out", chunk_s=info.duration_s, mode="exact")
    assert len(chunks) == 1
    assert probe_media(chunks[0].path).duration_s == pytest.approx(info.duration_s, abs=0.15)


def test_chunk_video_rejects_too_many_chunks(tmp_path, monkeypatch):
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"fake")
    info = MediaInfo(source, 10_001.0, "h264", "aac", "mp4", 25.0)
    monkeypatch.setattr("video_chunker.cut.probe_media", lambda *args, **kwargs: info)
    with pytest.raises(CutError, match="10000"):
        chunk_video(source, tmp_path / "out", chunk_s=1.0, mode="exact")
