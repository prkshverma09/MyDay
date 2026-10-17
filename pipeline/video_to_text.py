#!/usr/bin/env python3
"""Turn a video into a JSONL payload for a later LLM, one chunk at a time.

Cuts follow video-chunker: a 10 second chunk length yields 0–10, then 10–20,
and so on. Cosmos Reason and YOLO11 run on that chunk as soon as it is cut.
This program does not call a summarizer. Each record's ``llm_input`` is the
text the next stage should read.
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
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "video-chunker" / "src"))
from video_chunker.plan import plan_segments  # noqa: E402
from video_chunker.probe import probe_media  # noqa: E402

DEFAULT_COSMOS_URL = "http://166.19.38.112:8001"
DEFAULT_YOLO_URL = "http://166.19.38.112:8002"
DEFAULT_CHUNK_S = 10.0
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


def chunk_plan(video_path: str, chunk_s: float):
    """Same non-overlapping cuts as video-chunker: one chunk every ``chunk_s`` seconds."""
    if chunk_s <= 0:
        raise SystemExit("chunk length must be > 0 seconds")
    info = probe_media(Path(video_path), ffprobe_bin=FFPROBE)
    segments = plan_segments(
        info.duration_s,
        chunk_s,
        min_tail_s=1.0 / info.frame_rate,
    )
    return info, segments


def transcode_window(src: str, start: float, length: float, dest: str) -> None:
    """640px wide, 4 fps, no audio, for one chunk."""
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
    chunk_index = window.get("index")
    if chunk_index is None:
        span = f"Time window: {_fmt_sec(window['start_sec'])}-{_fmt_sec(window['end_sec'])}s"
    else:
        span = (
            f"Chunk {int(chunk_index)}: "
            f"{_fmt_sec(window['start_sec'])}-{_fmt_sec(window['end_sec'])}s"
        )
    lines = [span]
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
    start = round(float(window["start_sec"]), 2)
    end = round(float(window["end_sec"]), 2)
    chunk = {
        "index": int(window["index"]),
        "start_sec": start,
        "end_sec": end,
        "duration_sec": round(end - start, 2),
    }
    return {
        "id": ann_id,
        "source": source,
        "camera_id": camera_id,
        "location": location,
        "chunk": chunk,
        "window": {"start_sec": start, "end_sec": end},
        "cosmos": cosmos,
        "yolo": yolo,
        "llm_input": build_llm_input(
            {**window, "index": chunk["index"]}, cosmos, yolo, camera_id, location
        ),
    }


def _proxy_and_perceive(src: str, start: float, length: float, dest: str) -> tuple[dict, dict]:
    transcode_window(src, start, length, dest)
    return perceive(dest)


def iter_chunk_records(
    video_path: str,
    *,
    chunk_s: float = DEFAULT_CHUNK_S,
    source: str = "",
    camera_id: str = "",
    location: str = "",
    id_start: int = 1,
) -> Iterator[dict]:
    """Cut one chunk, describe it, then move to the next chunk.

    A 10 second chunk length means 0–10, then 10–20, and so on. Cosmos and YOLO
    see that whole chunk. Nothing is held back for a later summary.
    """
    _info, segments = chunk_plan(video_path, chunk_s)
    with tempfile.TemporaryDirectory(prefix="v2t-") as tmp:
        for offset, segment in enumerate(segments):
            dest = str(Path(tmp) / f"chunk_{segment.index:04d}.mp4")
            cosmos, yolo = _proxy_and_perceive(
                video_path, segment.start_s, segment.duration_s, dest
            )
            yield make_record(
                ann_id=f"ann_{id_start + offset:04d}",
                source=source or video_path,
                camera_id=camera_id,
                location=location,
                window={
                    "index": segment.index,
                    "start_sec": segment.start_s,
                    "end_sec": segment.end_s,
                },
                cosmos=cosmos,
                yolo=yolo,
            )


def iter_manifest_records(
    manifest_path: str,
    *,
    camera_id: str = "",
    location: str = "",
    id_start: int = 1,
) -> Iterator[dict]:
    """Describe chunks already written by video-chunker, in index order."""
    payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    source = str(payload.get("source") or manifest_path)
    chunks = sorted(payload.get("chunks") or [], key=lambda item: int(item["index"]))
    with tempfile.TemporaryDirectory(prefix="v2t-") as tmp:
        for offset, chunk in enumerate(chunks):
            path = Path(chunk["path"])
            if not path.is_file():
                raise SystemExit(f"missing chunk file {path}")
            dest = str(Path(tmp) / f"chunk_{int(chunk['index']):04d}.mp4")
            cosmos, yolo = _proxy_and_perceive(
                str(path), 0.0, float(chunk["end_s"]) - float(chunk["start_s"]), dest
            )
            yield make_record(
                ann_id=f"ann_{id_start + offset:04d}",
                source=source,
                camera_id=camera_id,
                location=location,
                window={
                    "index": int(chunk["index"]),
                    "start_sec": float(chunk["start_s"]),
                    "end_sec": float(chunk["end_s"]),
                },
                cosmos=cosmos,
                yolo=yolo,
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Video chunks to structured text for a later LLM")
    parser.add_argument("video", nargs="?", help="Local video file")
    parser.add_argument(
        "--chunk",
        type=float,
        default=DEFAULT_CHUNK_S,
        help="Chunk length in seconds. 10 means a 10s clip every 10s (default 10)",
    )
    parser.add_argument(
        "--window",
        type=float,
        default=None,
        help="Alias for --chunk",
    )
    parser.add_argument(
        "--manifest",
        help="video-chunker manifest (.chunks.json). Describes those chunk files in order",
    )
    parser.add_argument("-o", "--output", help="JSONL path (default stdout)")
    parser.add_argument("--source", default="", help="Source URI stored on each record")
    parser.add_argument("--camera-id", default="")
    parser.add_argument("--location", default="")
    parser.add_argument("--id-start", type=int, default=1)
    args = parser.parse_args(argv)
    chunk_s = args.window if args.window is not None else args.chunk

    if args.manifest:
        records = iter_manifest_records(
            args.manifest,
            camera_id=args.camera_id,
            location=args.location,
            id_start=args.id_start,
        )
    else:
        if not args.video:
            parser.error("pass a video file or --manifest")
        records = iter_chunk_records(
            args.video,
            chunk_s=chunk_s,
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
