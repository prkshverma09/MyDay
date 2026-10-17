from pathlib import Path

import pytest

from video_chunker.cli import main
from video_chunker.models import ChunkRecord, CutError, CutFailed

def test_cli_rejects_non_positive_duration(capsys):
    code = main(["clip.mp4", "--duration", "0"])
    assert code == 2
    assert "duration" in capsys.readouterr().err

def test_cli_rejects_unknown_mode(capsys):
    code = main(["clip.mp4", "--mode", "fancy"])
    assert code == 2
    assert "mode" in capsys.readouterr().err

def test_cli_missing_file(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("video_chunker.cli.resolve_bin", lambda explicit, name: name)
    missing = tmp_path / "missing.mp4"
    code = main([str(missing), "--output", str(tmp_path / "out")])
    assert code == 2
    assert "not a file" in capsys.readouterr().err

def test_cli_missing_binary_creates_nothing(tmp_path, monkeypatch, capsys):
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"x")
    out = tmp_path / "out"

    def missing(explicit, name):
        raise CutError(f"{name} not found. Install ffmpeg 5 or newer, or pass --{name}.")

    monkeypatch.setattr("video_chunker.cli.resolve_bin", missing)
    code = main([str(source), "--output", str(out)])
    assert code == 2
    err = capsys.readouterr().err
    assert "ffmpeg" in err or "ffprobe" in err
    assert list(out.glob("*.mp4")) == [] if out.exists() else True

def test_cli_prints_chunk_paths(tmp_path, monkeypatch, capsys):
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"x")
    out = tmp_path / "out"
    monkeypatch.setattr("video_chunker.cli.resolve_bin", lambda explicit, name: name)

    def fake_chunk(source, output_dir, **kwargs):
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "clip_0000.mp4"
        path.write_bytes(b"chunk")
        return [ChunkRecord(0, path, 0.0, 15.0)]

    monkeypatch.setattr("video_chunker.cli.chunk_video", fake_chunk)
    code = main([str(source), "--output", str(out), "--duration", "15"])
    assert code == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines == [str((out / "clip_0000.mp4").resolve())]

def test_cli_cut_error_is_exit_1(tmp_path, monkeypatch, capsys):
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"x")
    monkeypatch.setattr("video_chunker.cli.resolve_bin", lambda explicit, name: name)

    def boom(*args, **kwargs):
        raise CutFailed("ffmpeg failed")

    monkeypatch.setattr("video_chunker.cli.chunk_video", boom)
    code = main([str(source), "--output", str(tmp_path / "out")])
    assert code == 1
    assert "ffmpeg failed" in capsys.readouterr().err
