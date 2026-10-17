# tests/test_probe.py
import json
from pathlib import Path

import pytest

from video_chunker.models import ProbeError
from video_chunker.probe import parse_probe_json, probe_media

def test_parse_video_and_audio():
    payload = {
        "format": {"duration": "62.5", "format_name": "mov,mp4,m4a,3gp,3g2,mj2"},
        "streams": [
            {"codec_type": "video", "codec_name": "h264", "r_frame_rate": "30000/1001"},
            {"codec_type": "audio", "codec_name": "aac"},
        ],
    }
    info = parse_probe_json(payload, Path("holiday.mp4"))
    assert info.duration_s == pytest.approx(62.5)
    assert info.video_codec == "h264"
    assert info.audio_codec == "aac"
    assert info.frame_rate == pytest.approx(30000 / 1001)

def test_parse_video_without_audio():
    payload = {
        "format": {"duration": "4", "format_name": "mp4"},
        "streams": [
            {"codec_type": "video", "codec_name": "hevc", "r_frame_rate": "25/1"},
        ],
    }
    info = parse_probe_json(payload, Path("silent.mp4"))
    assert info.audio_codec is None
    assert info.frame_rate == pytest.approx(25)

def test_parse_rejects_audio_only():
    payload = {
        "format": {"duration": "4", "format_name": "mp3"},
        "streams": [{"codec_type": "audio", "codec_name": "mp3"}],
    }
    with pytest.raises(ProbeError, match="no video stream"):
        parse_probe_json(payload, Path("song.mp3"))

def test_parse_rejects_missing_duration():
    payload = {
        "format": {"format_name": "mp4"},
        "streams": [{"codec_type": "video", "codec_name": "h264", "r_frame_rate": "25/1"}],
    }
    with pytest.raises(ProbeError, match="no duration"):
        parse_probe_json(payload, Path("bad.mp4"))

def test_probe_media_uses_ffprobe_json(tmp_path, monkeypatch):
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"not really")
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        class Result:
            returncode = 0
            stdout = json.dumps({
                "format": {"duration": "2", "format_name": "mp4"},
                "streams": [
                    {"codec_type": "video", "codec_name": "h264", "r_frame_rate": "25/1"},
                ],
            })
            stderr = ""
        return Result()

    monkeypatch.setattr("video_chunker.probe.subprocess.run", fake_run)
    info = probe_media(source, ffprobe_bin="ffprobe")
    assert captured["cmd"][0] == "ffprobe"
    assert str(source) in captured["cmd"]
    assert info.duration_s == pytest.approx(2)

def test_probe_media_raises_on_ffprobe_failure(tmp_path, monkeypatch):
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"x")

    def fake_run(cmd, **kwargs):
        class Result:
            returncode = 1
            stdout = ""
            stderr = "Invalid data"
        return Result()

    monkeypatch.setattr("video_chunker.probe.subprocess.run", fake_run)
    with pytest.raises(ProbeError, match="Invalid data"):
        probe_media(source)
