#!/usr/bin/env python3
"""Describe 10-second chunks of the demo video as the playhead asks for them.

The page at / plays demo_data/IMG_2517.mp4 and requests /chunk?t=seconds.
Cosmos runs once per chunk index and stays in memory. The bearer token is read
from the environment or /config/team-20.config and is never written to a response.
"""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))
import video_to_text as v2t  # noqa: E402

DEFAULT_VIDEO = ROOT / "demo_data" / "IMG_2517.mp4"
ANNOTATIONS = ROOT / "demo_data" / "annotations.jsonl"
PAGE = ROOT / "examples" / "ui-demo" / "index.html"
HOST = "0.0.0.0"
PORT = 8787
DEFAULT_CHUNK_S = 10.0

_cache: dict[tuple, dict] = {}
_key_locks: dict[tuple, threading.Lock] = {}
_guard = threading.Lock()


def load_credentials() -> None:
    os.environ.setdefault("COSMOS_URL", "http://166.19.38.112:8001")
    os.environ.setdefault("YOLO_URL", "http://166.19.38.112:8002")
    if os.environ.get("GPU_BEARER_TOKEN"):
        return
    config = Path("/config/team-20.config")
    if not config.is_file():
        raise SystemExit("missing /config/team-20.config")
    for raw in config.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if not line.startswith("GPU_BEARER_TOKEN="):
            continue
        value = line.split("=", 1)[1].strip().strip('"').strip("'")
        if len(value) < 8:
            raise SystemExit("GPU_BEARER_TOKEN is empty")
        os.environ["GPU_BEARER_TOKEN"] = value
        return
    raise SystemExit("GPU_BEARER_TOKEN missing")


def resolve_video(raw: str | None) -> Path:
    if not raw:
        path = DEFAULT_VIDEO
    else:
        candidate = Path(raw)
        path = candidate if candidate.is_absolute() else ROOT / candidate
    path = path.resolve()
    root = ROOT.resolve()
    if root != path and root not in path.parents:
        raise ValueError("video path is outside the repo")
    if not path.is_file():
        raise FileNotFoundError(path.name)
    return path


def segment_at(segments, time_s: float):
    if not segments:
        raise ValueError("video has no chunks")
    moment = max(0.0, time_s)
    for index, segment in enumerate(segments):
        last = index == len(segments) - 1
        if last or (segment.start_s <= moment < segment.end_s):
            if moment >= segment.start_s or last:
                return segment
    return segments[-1]


def _lock_for(key: tuple) -> threading.Lock:
    with _guard:
        lock = _key_locks.get(key)
        if lock is None:
            lock = threading.Lock()
            _key_locks[key] = lock
        return lock


def _saved_rows(video: Path) -> list[dict] | None:
    """Precomputed Cosmos and YOLO rows for the default demo clip."""
    if video.resolve() != DEFAULT_VIDEO.resolve():
        return None
    if not ANNOTATIONS.is_file():
        return None
    rows: list[dict] = []
    for raw in ANNOTATIONS.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        row = json.loads(line)
        if "IMG_2517" not in str(row.get("source") or ""):
            return None
        rows.append(row)
    if not rows:
        return None
    rows.sort(key=lambda row: float(row["chunk"]["start_sec"]))
    return rows


def _payload_from_rows(rows: list[dict], time_s: float) -> dict:
    moment = max(0.0, time_s)
    chosen = rows[-1]
    for index, row in enumerate(rows):
        start = float(row["chunk"]["start_sec"])
        end = float(row["chunk"]["end_sec"])
        last = index == len(rows) - 1
        if last or start <= moment < end:
            if moment >= start or last:
                chosen = row
                break
    start = float(chosen["chunk"]["start_sec"])
    end = float(chosen["chunk"]["end_sec"])
    duration = float(rows[-1]["chunk"]["end_sec"])
    cosmos = chosen.get("cosmos") or {}
    text = (cosmos.get("text") or "").strip() or " ".join(cosmos.get("sentences") or [])
    yolo = chosen.get("yolo") or {}
    return {
        "index": int(chosen["chunk"]["index"]),
        "start_sec": round(start, 3),
        "end_sec": round(end, 3),
        "seconds_ago": max(0, math.floor(duration - start)),
        "is_last": int(chosen["chunk"]["index"]) == int(rows[-1]["chunk"]["index"]),
        "duration_sec": round(duration, 3),
        "cosmos_text": text,
        "yolo": yolo.get("objects") or [],
    }


def describe_time(video: Path, time_s: float, chunk_s: float) -> dict:
    saved = _saved_rows(video)
    if saved is not None:
        return _payload_from_rows(saved, time_s)
    info, segments = v2t.chunk_plan(str(video), chunk_s)
    segment = segment_at(segments, time_s)
    key = (str(video), int(segment.index), float(chunk_s))
    with _lock_for(key):
        cached = _cache.get(key)
        if cached is not None:
            return cached
        with tempfile.TemporaryDirectory(prefix="chunk-") as tmp:
            window = str(Path(tmp) / f"chunk_{segment.index:04d}.mp4")
            v2t.transcode_window(str(video), segment.start_s, segment.duration_s, window)
            with ThreadPoolExecutor(max_workers=2) as pool:
                cosmos_future = pool.submit(v2t.call_cosmos, window)
                yolo_future = pool.submit(v2t.call_yolo, window)
                cosmos = cosmos_future.result()
                try:
                    yolo = yolo_future.result()
                except Exception:
                    yolo = {"objects": []}
        text = (cosmos.get("text") or "").strip() or " ".join(cosmos.get("sentences") or [])
        payload = {
            "index": int(segment.index),
            "start_sec": round(float(segment.start_s), 3),
            "end_sec": round(float(segment.end_s), 3),
            "seconds_ago": max(0, math.floor(float(info.duration_s) - float(segment.start_s))),
            "is_last": segment.index == segments[-1].index,
            "duration_sec": round(float(info.duration_s), 3),
            "cosmos_text": text,
            "yolo": yolo.get("objects") or [],
        }
        _cache[key] = payload
        return payload


def _public_error(exc: Exception) -> str:
    text = str(exc).replace("\n", " ")
    lowered = text.lower()
    if "bearer" in lowered or "token" in lowered or "gpu_bearer" in lowered:
        return "description failed"
    return text[:240] or exc.__class__.__name__


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args) -> None:
        message = fmt % args
        if "bearer" in message.lower() or "token" in message.lower():
            return
        sys.stderr.write("%s %s\n" % (self.address_string(), message))

    def _cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self._cors()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            self._send_bytes(PAGE.read_bytes(), "text/html; charset=utf-8")
            return
        if parsed.path == "/video":
            self._send_video(parsed.query)
            return
        if parsed.path == "/chunk":
            self._send_chunk(parsed.query)
            return
        if parsed.path == "/health":
            self._send_json({"ok": True, "video": DEFAULT_VIDEO.name})
            return
        self._send_json({"error": "not found"}, status=404)

    def _send_chunk(self, query: str) -> None:
        params = parse_qs(query)
        try:
            time_s = float((params.get("t") or ["0"])[0])
            chunk_s = float((params.get("chunk") or [str(DEFAULT_CHUNK_S)])[0])
            video = resolve_video((params.get("video") or [None])[0])
            if chunk_s <= 0:
                raise ValueError("chunk length must be > 0")
            payload = describe_time(video, time_s, chunk_s)
        except Exception as exc:
            self._send_json({"error": _public_error(exc)}, status=400 if isinstance(exc, (ValueError, FileNotFoundError)) else 500)
            return
        self._send_json(payload)

    def _send_video(self, query: str) -> None:
        params = parse_qs(query)
        try:
            path = resolve_video((params.get("video") or [None])[0])
        except Exception as exc:
            self._send_json({"error": _public_error(exc)}, status=404)
            return
        size = path.stat().st_size
        range_header = self.headers.get("Range")
        start, end = 0, size - 1
        status = 200
        if range_header and range_header.startswith("bytes="):
            spec = range_header.split("=", 1)[1].split(",", 1)[0].strip()
            left, _, right = spec.partition("-")
            try:
                start = int(left) if left else 0
                end = int(right) if right else size - 1
            except ValueError:
                start, end = 0, size - 1
            end = min(end, size - 1)
            start = max(0, start)
            if start > end:
                start, end = 0, size - 1
            status = 206
        length = end - start + 1
        try:
            self.send_response(status)
            self._cors()
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(length))
            if status == 206:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            if self.command == "HEAD":
                return
            with path.open("rb") as handle:
                handle.seek(start)
                remaining = length
                while remaining > 0:
                    block = handle.read(min(256 * 1024, remaining))
                    if not block:
                        break
                    self.wfile.write(block)
                    remaining -= len(block)
        except (BrokenPipeError, ConnectionResetError):
            return

    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send_bytes(body, "application/json; charset=utf-8", status=status)

    def _send_bytes(self, body: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self._cors()
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)


def main() -> None:
    load_credentials()
    if not DEFAULT_VIDEO.is_file():
        raise SystemExit(f"missing demo video {DEFAULT_VIDEO}")
    if not PAGE.is_file():
        raise SystemExit(f"missing player page {PAGE}")
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"listening http://{HOST}:{PORT}/", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
