#!/usr/bin/env python3
"""Turn a video into a JSONL payload for a later LLM.

Cosmos Reason and YOLO11 are the perception step. This program does not call
a summarizer. Each record's ``llm_input`` is the text the next stage should read.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

DEFAULT_COSMOS_URL = "http://166.19.38.112:8001"
DEFAULT_YOLO_URL = "http://166.19.38.112:8002"
FFMPEG = "/usr/bin/ffmpeg" if Path("/usr/bin/ffmpeg").is_file() else "ffmpeg"
FFPROBE = "/usr/bin/ffprobe" if Path("/usr/bin/ffprobe").is_file() else "ffprobe"

COSMOS_PROMPT = (
    "Describe only what is clearly visible, in short sentences. "
    "Each sentence needs an actor, an action, an object with color or type, "
    "and where it is relative to another visible thing. "
    "Example shape, do not copy the example: "
    "'A person in a black coat stands beside a yellow taxi.' "
    "Skip weather and general setting. Do not invent objects or actions."
)
EXAMPLE_SENTENCE = "a person in a black coat stands beside a yellow taxi."
ABSURD_LABELS = {"toilet", "airplane", "train"}
LOW_CONFIDENCE = 0.50

_MODEL_CACHE: str | None = None


def _env(name: str, default: str | None = None) -> str:
    value = os.environ.get(name, default)
    if not value:
        raise SystemExit(f"missing environment variable {name}")
    return value


def cosmos_base() -> str:
    return _env("COSMOS_URL", DEFAULT_COSMOS_URL).rstrip("/")


def yolo_base() -> str:
    return _env("YOLO_URL", DEFAULT_YOLO_URL).rstrip("/")


def _auth_headers() -> dict[str, str]:
    return {
        "Authorization": "Bearer " + _env("GPU_BEARER_TOKEN"),
        "Content-Type": "application/json",
    }


def cosmos_model() -> str:
    global _MODEL_CACHE
    configured = os.environ.get("COSMOS_MODEL")
    if configured:
        return configured
    if _MODEL_CACHE:
        return _MODEL_CACHE
    response = requests.get(
        cosmos_base() + "/v1/models", headers=_auth_headers(), timeout=(10, 30)
    )
    response.raise_for_status()
    _MODEL_CACHE = response.json()["data"][0]["id"]
    return _MODEL_CACHE


def probe_duration(path: str) -> float:
    result = subprocess.run(
        [
            FFPROBE,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nokey=1:noprint_wrappers=1",
            path,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(result.stdout.strip())


def iter_windows(duration: float, window: float):
    start = 0.0
    while start < duration - 0.05:
        end = min(duration, start + window)
        if end - start < 0.4:
            break
        yield start, end
        start += window


def transcode_window(src: str, start: float, length: float, dest: str) -> None:
    """640px wide, 4 fps, no audio. This is the measured Cosmos detail window."""
    cmd = [
        FFMPEG,
        "-y",
        "-ss",
        f"{start:.3f}",
        "-i",
        src,
        "-t",
        f"{length:.3f}",
        "-vf",
        "scale=640:-2,fps=4",
        "-an",
        "-c:v",
        "libx264",
        "-crf",
        "28",
        "-preset",
        "veryfast",
        "-movflags",
        "+faststart",
        dest,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not Path(dest).is_file():
        tail = (result.stderr or "")[-400:]
        raise RuntimeError(f"ffmpeg failed: {tail}")


def _post_json(url: str, payload: dict, timeout: tuple[int, int]) -> dict:
    response = requests.post(url, headers=_auth_headers(), json=payload, timeout=timeout)
    if response.status_code >= 400:
        snippet = response.text[:240].replace("\n", " ")
        raise RuntimeError(f"HTTP {response.status_code} from {url}: {snippet}")
    return response.json()


def _sentences(text: str) -> list[str]:
    parts: list[str] = []
    for line in (text or "").splitlines():
        line = re.sub(r"^[\s\-\*\u2022\d\.\)\(]+", "", line).strip()
        if not line:
            continue
        for chunk in re.split(r"(?<=[.!?])\s+", line):
            chunk = chunk.strip()
            if not chunk:
                continue
            if chunk[-1] not in ".!?":
                chunk += "."
            if chunk.lower().strip(" .") == EXAMPLE_SENTENCE.strip(" ."):
                continue
            if parts and parts[-1] == chunk:
                continue
            parts.append(chunk)
    return parts[:5]


def _needs_retry(text: str) -> bool:
    lowered = (text or "").lower()
    if EXAMPLE_SENTENCE.strip(" .") in lowered:
        return True
    raw = [s.strip() for s in re.split(r"(?<=[.!?])\s+", (text or "").strip()) if s.strip()]
    if not raw:
        return True
    counts: dict[str, int] = defaultdict(int)
    for sentence in raw:
        counts[sentence.lower()] += 1
    if any(n >= 3 for n in counts.values()):
        return True
    return False


def call_cosmos(video_path: str) -> dict:
    video_b64 = _b64(video_path)
    content = [
        {"type": "text", "text": COSMOS_PROMPT},
        {"type": "video_url", "video_url": {"url": "data:video/mp4;base64," + video_b64}},
    ]
    attempts = (
        {"temperature": 0.2, "max_tokens": 400},
        {"temperature": 0.2, "max_tokens": 180, "repetition_penalty": 1.15},
    )
    last_error = None
    chosen = None
    for extra in attempts:
        payload = {
            "model": cosmos_model(),
            "messages": [{"role": "user", "content": content}],
            **extra,
        }
        started = time.perf_counter()
        try:
            body = _post_json(
                cosmos_base() + "/v1/chat/completions", payload, timeout=(10, 180)
            )
        except RuntimeError as exc:
            last_error = exc
            if "repetition_penalty" in payload and "HTTP 400" in str(exc):
                payload.pop("repetition_penalty", None)
                try:
                    body = _post_json(
                        cosmos_base() + "/v1/chat/completions", payload, timeout=(10, 180)
                    )
                except RuntimeError as exc2:
                    last_error = exc2
                    continue
            else:
                continue
        elapsed = time.perf_counter() - started
        message = body["choices"][0]["message"]
        text = message.get("content") or message.get("reasoning_content") or ""
        usage = body.get("usage") or {}
        candidate = {
            "sentences": _sentences(text),
            "inference_s": round(elapsed, 3),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
        }
        if chosen is None or (candidate["sentences"] and _needs_retry(text) is False):
            chosen = candidate
        if candidate["sentences"] and not _needs_retry(text):
            break
    if chosen is None:
        raise RuntimeError(f"cosmos failed: {last_error}")
    if not chosen["sentences"]:
        raise RuntimeError("cosmos returned no usable sentences")
    return chosen


def _b64(path: str) -> str:
    import base64

    return base64.b64encode(Path(path).read_bytes()).decode("ascii")


def _position(bbox: list[float], shape: list[int]) -> str:
    height, width = shape
    x1, y1, x2, y2 = bbox
    if width <= 0 or height <= 0 or x2 <= x1 or y2 <= y1:
        return "center/far"
    cx = ((x1 + x2) / 2.0) / width
    if cx < 0.33:
        horizontal = "left"
    elif cx > 0.67:
        horizontal = "right"
    else:
        horizontal = "center"
    depth = "near" if ((y2 - y1) / height) >= 0.18 else "far"
    return f"{horizontal}/{depth}"


def call_yolo(video_path: str) -> dict:
    payload = {
        "video_base64": _b64(video_path),
        "filename": Path(video_path).name,
        "include_frames": True,
    }
    started = time.perf_counter()
    body = _post_json(yolo_base() + "/v1/infer", payload, timeout=(10, 120))
    elapsed = time.perf_counter() - started
    frames = body.get("frames") or []
    stats: dict[str, dict] = {}
    for frame in frames:
        shape = frame.get("shape") or [1, 1]
        for det in frame.get("detections") or []:
            label = det.get("label") or "unknown"
            bucket = stats.setdefault(
                label,
                {
                    "hits": 0,
                    "max_confidence": 0.0,
                    "frames": set(),
                    "best_bbox": None,
                    "best_shape": shape,
                },
            )
            bucket["hits"] += 1
            bucket["frames"].add(frame.get("frame_index"))
            conf = float(det.get("confidence") or 0.0)
            if conf >= bucket["max_confidence"]:
                bucket["max_confidence"] = conf
                bucket["best_bbox"] = det.get("bbox")
                bucket["best_shape"] = shape

    objects = []
    ignored = []
    for label, bucket in sorted(stats.items(), key=lambda item: -item[1]["hits"]):
        record = {
            "label": label,
            "hits": bucket["hits"],
            "max_confidence": round(bucket["max_confidence"], 4),
            "position": _position(bucket["best_bbox"] or [0, 0, 0, 0], bucket["best_shape"]),
        }
        reasons = []
        if label in ABSURD_LABELS:
            if bucket["max_confidence"] < LOW_CONFIDENCE:
                reasons.append("low_confidence")
            if bucket["hits"] == 1:
                reasons.append("hit_count_1")
            elif len(bucket["frames"]) <= 1:
                reasons.append("one_frame")
        if reasons:
            ignored.append(
                {
                    "label": label,
                    "hits": bucket["hits"],
                    "max_confidence": record["max_confidence"],
                    "reason": "+".join(reasons),
                }
            )
            continue
        objects.append(record)

    return {
        "inference_s": round(elapsed, 3),
        "frames": len(frames),
        "objects": objects,
        "ignored": ignored,
        "perception_ok": bool(body.get("perception_ok") or body.get("ok")),
    }


def perceive(window_mp4: str) -> tuple[dict, dict]:
    """Run Cosmos and YOLO at the same time on one prepared window."""
    with ThreadPoolExecutor(max_workers=2) as pool:
        cosmos_future = pool.submit(call_cosmos, window_mp4)
        yolo_future = pool.submit(call_yolo, window_mp4)
        cosmos = cosmos_future.result()
        yolo = yolo_future.result()
    return cosmos, yolo


def _fmt_sec(value: float) -> str:
    text = f"{float(value):.2f}"
    if text.endswith("00"):
        return text[:-1]
    return text.rstrip("0")


def build_llm_input(
    window: dict,
    cosmos: dict,
    yolo: dict,
    camera_id: str = "",
    location: str = "",
) -> str:
    lines = [
        f"Time window: {_fmt_sec(window['start_sec'])}-{_fmt_sec(window['end_sec'])}s",
    ]
    if camera_id:
        lines.append(f"Camera: {camera_id}")
    if location:
        lines.append(f"Location: {location}")
    lines.append("Cosmos:")
    if cosmos.get("sentences"):
        lines.extend(f"- {sentence}" for sentence in cosmos["sentences"])
    else:
        lines.append("- none")
    lines.append("YOLO:")
    if yolo.get("objects"):
        for obj in yolo["objects"]:
            lines.append(
                f"- {obj['label']} count={obj['hits']} "
                f"confidence={obj['max_confidence']:.2f} position={obj['position']}"
            )
    else:
        lines.append("- none")
    return "\n".join(lines)


def make_record(
    *,
    ann_id: str,
    source: str,
    camera_id: str,
    location: str,
    window: dict,
    cosmos: dict,
    yolo: dict,
) -> dict:
    return {
        "id": ann_id,
        "source": source,
        "camera_id": camera_id,
        "location": location,
        "window": {
            "start_sec": round(float(window["start_sec"]), 2),
            "end_sec": round(float(window["end_sec"]), 2),
        },
        "cosmos": cosmos,
        "yolo": yolo,
        "llm_input": build_llm_input(window, cosmos, yolo, camera_id, location),
    }


def describe_file(
    video_path: str,
    *,
    window_s: float = 5.0,
    source: str = "",
    camera_id: str = "",
    location: str = "",
    id_start: int = 1,
) -> list[dict]:
    duration = probe_duration(video_path)
    records = []
    with tempfile.TemporaryDirectory(prefix="v2t-") as tmp:
        for index, (start, end) in enumerate(iter_windows(duration, window_s)):
            dest = str(Path(tmp) / f"window_{index:03d}.mp4")
            transcode_window(video_path, start, end - start, dest)
            cosmos, yolo = perceive(dest)
            window = {"start_sec": start, "end_sec": end}
            records.append(
                make_record(
                    ann_id=f"ann_{id_start + index:04d}",
                    source=source or video_path,
                    camera_id=camera_id,
                    location=location,
                    window=window,
                    cosmos=cosmos,
                    yolo=yolo,
                )
            )
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Video to structured text for a later LLM")
    parser.add_argument("video", help="Local video file")
    parser.add_argument("--window", type=float, default=5.0, help="Window length in seconds")
    parser.add_argument("-o", "--output", help="JSONL path (default stdout)")
    parser.add_argument("--source", default="", help="Source URI stored on each record")
    parser.add_argument("--camera-id", default="")
    parser.add_argument("--location", default="")
    parser.add_argument("--id-start", type=int, default=1)
    args = parser.parse_args(argv)

    records = describe_file(
        args.video,
        window_s=args.window,
        source=args.source,
        camera_id=args.camera_id,
        location=args.location,
        id_start=args.id_start,
    )
    handle = open(args.output, "w", encoding="utf-8") if args.output else sys.stdout
    try:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
    finally:
        if args.output:
            handle.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
