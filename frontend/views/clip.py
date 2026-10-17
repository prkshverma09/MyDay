"""The demo clip player shown at the top of My day.

The player and the seconds-ago clock run in the browser. Streamlit does not
rerun as the video plays. Chunk text is fetched from the description backend.
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import quote

import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VIDEO = ROOT / "demo_data" / "IMG_2517.mp4"
_UPLOAD_DIR = ROOT / "data" / "clips" / "uploads"

_PLAYER = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <style>
    html, body { height: 100%; margin: 0; }
    body {
      font-family: "Segoe UI", "Helvetica Neue", sans-serif;
      color: #2b241c;
      background: transparent;
    }
    main {
      height: 100%;
      display: grid;
      grid-template-columns: minmax(0, 1.45fr) minmax(240px, 0.9fr);
      gap: 14px;
    }
    .stage, .panel {
      min-height: 0;
      background: #fffdfb;
      border: 1px solid #e3d7cb;
      border-radius: 16px;
    }
    .stage {
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 10px;
    }
    video {
      width: 100%;
      max-height: 100%;
      border-radius: 10px;
      background: #161616;
    }
    .panel { display: flex; flex-direction: column; }
    .meta {
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      gap: 12px;
      padding: 14px 16px 10px;
      border-bottom: 1px solid #e3d7cb;
    }
    #range { font-weight: 700; font-variant-numeric: tabular-nums; font-size: 18px; }
    #ago {
      color: #3d6b54;
      font-weight: 700;
      font-variant-numeric: tabular-nums;
      font-size: 22px;
      white-space: nowrap;
    }
    #text {
      overflow: auto;
      margin: 0;
      padding: 12px 16px 16px;
      font-size: 16px;
      line-height: 1.45;
      white-space: pre-wrap;
    }
    #text.pending { color: #6e6258; }
    @media (max-width: 820px) {
      main { grid-template-columns: 1fr; grid-template-rows: auto minmax(0, 1fr); }
      .stage { padding: 6px; }
    }
  </style>
</head>
<body>
  <main>
    <section class="stage">
      <video id="video" controls playsinline preload="metadata"></video>
    </section>
    <section class="panel">
      <div class="meta">
        <div id="range">…</div>
        <div id="ago"></div>
      </div>
      <p id="text" class="pending">Describing this moment…</p>
    </section>
  </main>
  <script>
    function backendBase() {
      try {
        const host = window.parent.location.hostname;
        if (host) return "http://" + host + ":8787";
      } catch (error) {}
      return "http://127.0.0.1:8787";
    }
    const BACKEND = backendBase();
    const VIDEO_EXTRA = "__VIDEO_EXTRA__";
    const video = document.getElementById("video");
    video.src = BACKEND + "/video" + (VIDEO_EXTRA ? ("?" + VIDEO_EXTRA.slice(1)) : "");
    const range = document.getElementById("range");
    const ago = document.getElementById("ago");
    const text = document.getElementById("text");
    let shown = null;
    let inflight = false;
    let queued = false;

    function integerSecond(value) {
      const number = Number(value);
      if (!Number.isFinite(number)) return 0;
      return Math.max(0, Math.floor(number));
    }

    function secondsAgo() {
      const duration = video.duration;
      const now = Number.isFinite(video.currentTime) ? video.currentTime : 0;
      if (!Number.isFinite(duration) || duration <= 0) return null;
      return Math.max(0, Math.floor(duration - now));
    }

    function paintClock() {
      const value = secondsAgo();
      ago.textContent = value === null ? "" : value + " seconds ago";
    }

    function covers(record, time) {
      if (!record) return false;
      const start = Number(record.start_sec);
      const end = Number(record.end_sec);
      if (time < start - 0.05) return false;
      if (record.is_last) return time <= end + 0.05;
      return time < end - 0.001;
    }

    function paintChunk(record) {
      range.textContent = integerSecond(record.start_sec) + "–" + integerSecond(record.end_sec) + "s";
      text.classList.remove("pending");
      text.textContent = record.cosmos_text || "No description.";
    }

    async function syncChunk() {
      const time = Number.isFinite(video.currentTime) ? video.currentTime : 0;
      if (covers(shown, time)) return;
      if (inflight) {
        queued = true;
        return;
      }
      inflight = true;
      queued = false;
      if (!shown) {
        text.classList.add("pending");
        text.textContent = "Describing this moment…";
      }
      try {
        const response = await fetch(BACKEND + "/chunk?t=" + encodeURIComponent(time.toFixed(3)) + "&chunk=10" + VIDEO_EXTRA);
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.error || ("HTTP " + response.status));
        const now = Number.isFinite(video.currentTime) ? video.currentTime : 0;
        if (covers(payload, now)) {
          shown = payload;
          paintChunk(payload);
        } else {
          queued = true;
        }
      } catch (error) {
        text.classList.add("pending");
        text.textContent = error.message || "Could not describe this chunk.";
      } finally {
        inflight = false;
        if (queued) syncChunk();
      }
    }

    function onTime() {
      paintClock();
      syncChunk();
    }

    video.addEventListener("timeupdate", onTime);
    video.addEventListener("seeked", onTime);
    video.addEventListener("loadedmetadata", onTime);
    video.addEventListener("play", onTime);
    video.addEventListener("pause", paintClock);
    setInterval(paintClock, 200);
    onTime();
  </script>
</body>
</html>
"""


def _save_upload(upload) -> Path:
    _UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    suffix = Path(upload.name).suffix.lower()
    if suffix not in {".mp4", ".mov", ".webm", ".mkv", ".avi"}:
        suffix = ".mp4"
    dest = _UPLOAD_DIR / f"player-upload{suffix}"
    dest.write_bytes(upload.getvalue())
    return dest


def demo_player(upload=None) -> None:
    """The demo video with the live chunk text beside it. `upload` plays a different file instead."""
    extra = ""
    if upload is not None:
        saved = _save_upload(upload)
        extra = "&video=" + quote(saved.relative_to(ROOT).as_posix())
    elif not DEFAULT_VIDEO.is_file():
        st.info("demo_data/IMG_2517.mp4 is missing.")
        return
    components.html(_PLAYER.replace("__VIDEO_EXTRA__", extra), height=600, scrolling=False)
