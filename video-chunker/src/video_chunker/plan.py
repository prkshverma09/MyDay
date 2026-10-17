from video_chunker.models import Segment

def plan_segments(
    duration_s: float,
    chunk_s: float,
    *,
    min_tail_s: float = 0.001,
) -> list[Segment]:
    if duration_s <= 0:
        raise ValueError("duration_s must be > 0")
    if chunk_s <= 0:
        raise ValueError("chunk_s must be > 0")
    if min_tail_s < 0:
        raise ValueError("min_tail_s must be >= 0")

    segments: list[Segment] = []
    start = 0.0
    index = 0
    while True:
        remaining = duration_s - start
        if remaining <= min_tail_s:
            if segments:
                previous = segments[-1]
                segments[-1] = Segment(previous.index, previous.start_s, duration_s)
            break
        end = start + chunk_s
        if end >= duration_s - min_tail_s:
            end = duration_s
        segments.append(Segment(index, start, end))
        if end >= duration_s:
            break
        start = end
        index += 1
    if not segments:
        raise ValueError("duration_s is shorter than min_tail_s")
    return segments

def duration_tolerance_s(frame_rate: float) -> float:
    if frame_rate <= 0:
        return 0.08
    return max(2.0 / frame_rate, 0.04)
