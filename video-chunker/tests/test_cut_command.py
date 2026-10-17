from pathlib import Path

import pytest

from video_chunker.cut import assert_fast_mode_compatible, build_ffmpeg_command
from video_chunker.models import CutError, MediaInfo


def _info(**overrides) -> MediaInfo:
    base = dict(
        path=Path("clip.mp4"),
        duration_s=62.0,
        video_codec="h264",
        audio_codec="aac",
        format_name="mp4",
        frame_rate=25.0,
    )
    base.update(overrides)
    return MediaInfo(**base)


def test_exact_command_reencodes_h264_aac():
    cmd = build_ffmpeg_command(
        ffmpeg_bin="ffmpeg",
        source=Path("holiday.mp4"),
        output_pattern=Path("out/holiday_%04d.mp4"),
        chunk_s=15,
        mode="exact",
        has_audio=True,
    )
    assert cmd[:4] == ["ffmpeg", "-hide_banner", "-loglevel", "error"]
    assert "-c" not in cmd or cmd[cmd.index("-c:v") + 1] == "libx264"
    assert "copy" not in cmd
    assert "-map" in cmd and "0:a:0" in cmd
    assert cmd[-1] == "out/holiday_%04d.mp4"
    assert cmd[cmd.index("-segment_time") + 1] == "15.000000"


def test_exact_command_omits_audio_when_absent():
    cmd = build_ffmpeg_command(
        ffmpeg_bin="ffmpeg",
        source=Path("silent.mp4"),
        output_pattern=Path("out/silent_%04d.mp4"),
        chunk_s=15,
        mode="exact",
        has_audio=False,
    )
    assert "0:a:0" not in cmd
    assert "aac" not in cmd


def test_fast_command_copies_packets():
    cmd = build_ffmpeg_command(
        ffmpeg_bin="/usr/local/bin/ffmpeg",
        source=Path("holiday.mp4"),
        output_pattern=Path("out/holiday_%04d.mp4"),
        chunk_s=15,
        mode="fast",
        has_audio=True,
    )
    assert cmd[0] == "/usr/local/bin/ffmpeg"
    assert cmd[cmd.index("-c") + 1] == "copy"
    assert "libx264" not in cmd


def test_fast_mode_rejects_vp9():
    with pytest.raises(CutError, match="vp9"):
        assert_fast_mode_compatible(_info(video_codec="vp9"))


def test_fast_mode_rejects_vorbis_audio():
    with pytest.raises(CutError, match="vorbis"):
        assert_fast_mode_compatible(_info(audio_codec="vorbis"))


def test_fast_mode_allows_silent_h264():
    assert_fast_mode_compatible(_info(audio_codec=None))


def test_unknown_mode_is_rejected():
    with pytest.raises(CutError, match="unknown mode"):
        build_ffmpeg_command(
            ffmpeg_bin="ffmpeg",
            source=Path("a.mp4"),
            output_pattern=Path("out/a_%04d.mp4"),
            chunk_s=15,
            mode="fancy",
            has_audio=False,
        )
