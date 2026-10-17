import pytest

from video_chunker.plan import duration_tolerance_s, plan_segments

def test_exact_multiple_has_no_extra_tail():
    segments = plan_segments(45.0, 15.0)
    assert [(s.index, s.start_s, s.end_s) for s in segments] == [
        (0, 0.0, 15.0),
        (1, 15.0, 30.0),
        (2, 30.0, 45.0),
    ]

def test_remainder_is_its_own_chunk():
    segments = plan_segments(62.0, 15.0)
    assert len(segments) == 5
    assert segments[-1].start_s == 60.0
    assert segments[-1].end_s == 62.0
    assert segments[-1].duration_s == 2.0

def test_shorter_than_chunk_is_one_segment():
    segments = plan_segments(4.0, 15.0)
    assert len(segments) == 1
    assert segments[0].start_s == 0.0
    assert segments[0].end_s == 4.0

def test_sub_frame_tail_folds_into_previous():
    segments = plan_segments(30.01, 15.0, min_tail_s=0.04)
    assert len(segments) == 2
    assert segments[-1].end_s == 30.01

def test_rejects_non_positive_chunk():
    with pytest.raises(ValueError, match="chunk_s"):
        plan_segments(10, 0)

def test_rejects_non_positive_duration():
    with pytest.raises(ValueError, match="duration_s"):
        plan_segments(0, 15)

def test_tolerance_is_four_frames_with_a_floor():
    assert duration_tolerance_s(25) == pytest.approx(0.16)
    assert duration_tolerance_s(120) == pytest.approx(0.04)
    assert duration_tolerance_s(0) == pytest.approx(0.08)


def test_ntsc_tolerance_covers_a_few_frames_but_not_a_second():
    # 29.97 fps (30000/1001). A real ffmpeg cut was 0.071s long; ~1s must still fail.
    tolerance = duration_tolerance_s(30000 / 1001)
    assert tolerance > 0.071
    assert tolerance < 0.5
