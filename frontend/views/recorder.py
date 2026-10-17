"""Browser recording that sends 15 second pieces to the stream-chunk service."""

from __future__ import annotations

import streamlit as st

# The page is served by Streamlit. The chunk API stays on its own port.
# Same hostname so a laptop tunnel that forwards both ports reaches the VM.
_RECORDER_HTML = r"""
<style>
  .myday-recorder { position: relative; min-height: 0; }
  #myday-record {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.45rem;
    width: 100%;
    min-height: 56px;
    margin: 0;
    border-radius: 999px;
    border: 1.5px solid rgba(255, 255, 255, 0.85);
    background: rgba(253, 248, 245, 0.72);
    color: #44341D;
    font-family: "Outfit", "Segoe UI", "Helvetica Neue", sans-serif;
    font-size: 22px;
    font-weight: 600;
    line-height: 1;
    cursor: pointer;
    box-shadow: none;
  }
  .myday-cam {
    font-family: "Material Symbols Rounded";
    font-size: 22px;
    font-weight: 400;
    line-height: 1;
  }
  #myday-record:disabled { opacity: 0.55; cursor: default; }
  #myday-stage { margin-top: 0.85rem; }
  #myday-stage[hidden] { display: none !important; }
  #myday-preview {
    display: block;
    width: 100%;
    max-height: 360px;
    border-radius: 22px;
    background: #ECE0D4;
    object-fit: cover;
  }
  #myday-tools {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin-top: 0.7rem;
  }
  #myday-timer {
    margin: 0;
    padding: 0.28rem 0.85rem;
    border-radius: 999px;
    background: #ECE0D4;
    color: #44341D;
    font-family: "Outfit", "Segoe UI", sans-serif;
    font-size: 18px;
    font-weight: 700;
  }
  #myday-stop, #myday-flip {
    min-height: 56px;
    border-radius: 999px;
    font-family: "Outfit", "Segoe UI", sans-serif;
    font-size: 22px;
    font-weight: 600;
    cursor: pointer;
    box-shadow: none;
  }
  #myday-stop {
    flex: 1;
    border: 1.5px solid #52735E;
    background: #52735E;
    color: #FDF8F5;
  }
  #myday-flip {
    border: 1.5px solid rgba(255, 255, 255, 0.85);
    background: rgba(253, 248, 245, 0.72);
    color: #44341D;
    padding: 0 1.1rem;
  }
  #myday-stop:disabled, #myday-flip:disabled { opacity: 0.55; cursor: default; }
  #myday-status, #myday-error {
    margin: 0.45rem 0 0;
    color: #44341D;
    font-family: "Outfit", "Segoe UI", sans-serif;
    font-size: 20px;
    line-height: 1.4;
  }
  #myday-error:empty { display: none; }
  #myday-chunks {
    list-style: none;
    margin: 0.7rem 0 0;
    padding: 0;
  }
  #myday-chunks li { margin: 0 0 0.8rem; }
  #myday-chunks p {
    margin: 0 0 0.25rem;
    color: #44341D;
    font-family: "Outfit", "Segoe UI", sans-serif;
    font-size: 18px;
  }
  #myday-chunks video {
    width: 100%;
    border-radius: 22px;
    background: #ECE0D4;
  }
</style>
<div class="myday-recorder">
<button id="myday-record" type="button">
  <span class="myday-cam" aria-hidden="true">videocam</span>
  <span id="myday-record-label">Record</span>
</button>
<div id="myday-stage" hidden>
  <video id="myday-preview" playsinline autoplay muted></video>
  <div id="myday-tools">
    <p id="myday-timer">0:00</p>
    <button id="myday-stop" type="button">Stop</button>
    <button id="myday-flip" type="button">Flip</button>
  </div>
  <p id="myday-status"></p>
  <p id="myday-error" role="alert"></p>
  <ol id="myday-chunks"></ol>
</div>
</div>
<script>
(() => {
  const CHUNK_DURATION_S = 15;
  const CHUNK_PUBLIC = "https://mechanics-western-updating-winston.trycloudflare.com";
  const API_BASE = (location.hostname === "127.0.0.1" || location.hostname === "localhost")
    ? location.protocol + "//" + location.hostname + ":8765"
    : CHUNK_PUBLIC;
  const recordBtn = document.getElementById("myday-record");
  const recordLabel = document.getElementById("myday-record-label");
  const stage = document.getElementById("myday-stage");
  const preview = document.getElementById("myday-preview");
  const timerEl = document.getElementById("myday-timer");
  const stopBtn = document.getElementById("myday-stop");
  const flipBtn = document.getElementById("myday-flip");
  const statusEl = document.getElementById("myday-status");
  const errorEl = document.getElementById("myday-error");
  const chunkList = document.getElementById("myday-chunks");
  if (!recordBtn || recordBtn.dataset.bound === "1") return;
  recordBtn.dataset.bound = "1";

  const liveSupported = typeof MediaRecorder !== "undefined"
    && navigator.mediaDevices
    && typeof navigator.mediaDevices.getUserMedia === "function";

  if (window.__mydayStream) {
    window.__mydayStream.getTracks().forEach((track) => track.stop());
    window.__mydayStream = null;
  }

  let facingMode = "user";
  let stream = null;
  let recorder = null;
  let chosenMime = "";
  let sessionId = null;
  let sessionDone = false;
  let chunkSeconds = CHUNK_DURATION_S;
  let phase = "idle";
  let previousStopAt = null;
  let windowStartedAt = 0;
  let windowGeneration = 0;
  let timerId = 0;
  let ending = false;
  let uploadChain = Promise.resolve();

  function showStatus(text) { statusEl.textContent = text; }
  function clearError() { errorEl.textContent = ""; }
  function detailText(detail) {
    if (detail == null || detail === "") return "";
    if (typeof detail === "string") return detail;
    try { return JSON.stringify(detail); } catch (err) { return String(detail); }
  }
  function fail(error, detail) {
    const err = new Error(detail || error);
    err.error = error;
    err.detail = detail || "";
    return err;
  }
  function showApiError(err) {
    const error = err && err.error != null && err.error !== "" ? String(err.error) : (err && err.name) || "Error";
    const detail = detailText(err && err.detail != null && err.detail !== "" ? err.detail : err && err.message);
    errorEl.textContent = detail && detail !== error ? error + ": " + detail : error;
    stage.hidden = false;
  }
  function apiUrl(path) { return API_BASE + path; }
  async function request(path, options) {
    let response;
    try {
      response = await fetch(apiUrl(path), Object.assign({ cache: "no-store" }, options));
    } catch (err) {
      throw fail("network", err && err.message ? err.message : "Cannot reach the chunk server");
    }
    const text = await response.text();
    let payload = null;
    if (text) {
      try { payload = JSON.parse(text); } catch (err) { payload = null; }
    }
    if (!response.ok) {
      const error = payload && payload.error != null ? String(payload.error) : "HTTP " + response.status;
      const detail = payload && Object.prototype.hasOwnProperty.call(payload, "detail")
        ? payload.detail
        : (payload ? "" : text);
      throw fail(error, detailText(detail));
    }
    return payload;
  }
  function contentTypeFor(mime) {
    const base = String(mime || "").trim().toLowerCase().split(";")[0].trim();
    if (base === "video/mp4" || base === "video/webm" || base === "video/quicktime") return base;
    return "video/webm";
  }
  function chooseMime() {
    if (typeof MediaRecorder === "undefined" || typeof MediaRecorder.isTypeSupported !== "function") return "";
    const candidates = ["video/mp4", "video/webm;codecs=vp8,opus", "video/webm"];
    for (const candidate of candidates) {
      if (MediaRecorder.isTypeSupported(candidate)) return candidate;
    }
    return "";
  }
  function formatSeconds(value) {
    const n = Number(value);
    if (!Number.isFinite(n)) return String(value);
    return String(Math.round(n * 100) / 100);
  }
  function renderTimer(ms) {
    const seconds = Math.max(0, Math.floor(ms / 1000));
    timerEl.textContent = Math.floor(seconds / 60) + ":" + String(seconds % 60).padStart(2, "0");
  }
  function clearWindowTimer() {
    if (timerId) {
      clearTimeout(timerId);
      timerId = 0;
    }
  }
  function placeStage() {
    const column = recordBtn.closest('[data-testid="stColumn"]');
    const row = column && column.parentElement;
    if (!column || !row) return;
    const shift = column.getBoundingClientRect().left - row.getBoundingClientRect().left;
    if (shift < 8) {
      stage.style.width = "";
      stage.style.marginLeft = "";
      return;
    }
    stage.style.width = row.getBoundingClientRect().width + "px";
    stage.style.marginLeft = (-shift) + "px";
  }
  function setControls() {
    const busy = phase === "starting" || phase === "recording" || phase === "stopping";
    recordBtn.disabled = !liveSupported || busy;
    recordLabel.textContent = phase === "recording" ? "Recording" : "Record";
    stopBtn.disabled = phase !== "recording";
    flipBtn.disabled = !liveSupported || busy;
  }
  function appendChunks(id, rows) {
    for (const row of rows) {
      if (!row || !row.file) continue;
      const item = document.createElement("li");
      const name = document.createElement("p");
      name.textContent = row.file + "  ·  " + formatSeconds(row.start_s) + "s–" + formatSeconds(row.end_s) + "s";
      const video = document.createElement("video");
      video.controls = true;
      video.playsInline = true;
      video.setAttribute("playsinline", "");
      video.preload = "metadata";
      video.src = apiUrl("/api/sessions/" + encodeURIComponent(id) + "/chunks/" + encodeURIComponent(row.file));
      item.append(name, video);
      chunkList.append(item);
    }
  }
  function stopTracks(media) {
    if (!media) return;
    for (const track of media.getTracks()) track.stop();
  }
  function streamIsLive(media) {
    return !!(media && media.getVideoTracks().some((track) => track.readyState === "live"));
  }
  async function startCamera(facing) {
    const video = { facingMode: facing, width: { ideal: 1280 }, height: { ideal: 720 } };
    let next;
    try {
      next = await navigator.mediaDevices.getUserMedia({ video, audio: true });
    } catch (withAudio) {
      try {
        next = await navigator.mediaDevices.getUserMedia({ video, audio: false });
      } catch (videoOnly) {
        throw videoOnly || withAudio;
      }
    }
    stopTracks(stream);
    stream = next;
    window.__mydayStream = next;
    preview.srcObject = stream;
    preview.muted = true;
    const played = preview.play();
    if (played && typeof played.catch === "function") played.catch(() => {});
  }
  function enqueue(work) {
    const pending = uploadChain.then(work);
    uploadChain = pending.then(() => {}, () => {});
    return pending;
  }
  async function ensureSession() {
    if (sessionId && !sessionDone) return;
    const data = await request("/api/sessions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ chunk_duration_s: CHUNK_DURATION_S, mode: "exact", label: "myday" }),
    });
    if (!data || !data.id) throw fail("bad_response", "Session response did not include id");
    sessionId = data.id;
    sessionDone = false;
    chunkSeconds = Number(data.chunk_duration_s) > 0 ? Number(data.chunk_duration_s) : CHUNK_DURATION_S;
    previousStopAt = null;
    chunkList.replaceChildren();
    showStatus("Recording in " + chunkSeconds + " second pieces");
  }
  async function postMedia(id, blob, gapMs, contentType) {
    const data = await request("/api/sessions/" + encodeURIComponent(id) + "/media", {
      method: "POST",
      headers: { "Content-Type": contentType, "X-Gap-Before-Ms": String(gapMs) },
      body: blob,
    });
    if (data && Array.isArray(data.chunks)) appendChunks(id, data.chunks);
    if (phase === "recording") showStatus("Recording · " + chunkList.children.length + " chunks");
    return data;
  }
  async function finishSession() {
    const id = sessionId;
    const data = await request("/api/sessions/" + encodeURIComponent(id) + "/complete", { method: "POST" });
    sessionDone = true;
    phase = "idle";
    recorder = null;
    renderTimer(0);
    const count = data && data.chunk_count != null ? data.chunk_count : chunkList.children.length;
    showStatus("Saved " + count + " chunks · " + id);
    setControls();
  }
  function onUploadError(err) {
    showApiError(err);
    if (phase === "recording") {
      showStatus("Recording · last piece was not saved");
      return;
    }
    if (phase !== "stopping") setControls();
  }
  function onFinishError(err) {
    showApiError(err);
    phase = "halted";
    setControls();
  }
  function watchWindow(startedAt, generation) {
    const limit = chunkSeconds * 1000;
    const tick = () => {
      if (generation !== windowGeneration || phase !== "recording") return;
      const elapsed = performance.now() - startedAt;
      if (elapsed >= limit) {
        renderTimer(limit);
        endWindow();
        return;
      }
      renderTimer(elapsed);
      timerId = setTimeout(tick, Math.min(200, limit - elapsed));
    };
    tick();
  }
  function endWindow() {
    if (ending || !recorder || recorder.state !== "recording") return;
    ending = true;
    clearWindowTimer();
    windowGeneration += 1;
    try { recorder.stop(); } finally { ending = false; }
  }
  function startRecorder() {
    clearWindowTimer();
    const generation = ++windowGeneration;
    const startedAt = performance.now();
    const gapMs = previousStopAt == null ? 0 : Math.max(0, Math.round(startedAt - previousStopAt));
    const mime = chosenMime;
    const rec = mime ? new MediaRecorder(stream, { mimeType: mime }) : new MediaRecorder(stream);
    const type = contentTypeFor(rec.mimeType || mime);
    const parts = [];
    const id = sessionId;
    rec.ondataavailable = (event) => {
      if (event.data && event.data.size > 0) parts.push(event.data);
    };
    rec.onerror = (event) => {
      const err = (event && event.error) || fail("recorder", "Recorder error");
      if (!err.error) err.error = err.name || "recorder";
      if (!err.detail) err.detail = err.message || "Recorder error";
      showApiError(err);
      if (phase === "recording") {
        phase = "halted";
        endWindow();
      }
    };
    rec.onstop = () => {
      const stoppedAt = performance.now();
      const blob = new Blob(parts, { type });
      const continueRecording = phase === "recording";
      previousStopAt = stoppedAt;
      if (!continueRecording) {
        stopTracks(stream);
        stream = null;
        window.__mydayStream = null;
      }
      if (continueRecording) {
        try { startRecorder(); } catch (err) {
          phase = "halted";
          showApiError(err);
        }
      }
      const upload = blob.size > 0 ? enqueue(() => postMedia(id, blob, gapMs, type)) : Promise.resolve();
      if (phase === "stopping") {
        upload.catch(showApiError).then(() => enqueue(() => finishSession())).catch(onFinishError);
      } else {
        upload.catch(onUploadError);
      }
    };
    recorder = rec;
    windowStartedAt = startedAt;
    rec.start();
    watchWindow(startedAt, generation);
    showStatus("Recording");
  }
  async function onRecordClick() {
    if (!liveSupported || phase === "starting" || phase === "recording" || phase === "stopping") return;
    clearError();
    phase = "starting";
    stage.hidden = false;
    placeStage();
    setControls();
    try {
      if (!streamIsLive(stream)) await startCamera(facingMode);
      await ensureSession();
      chosenMime = chooseMime();
      phase = "recording";
      startRecorder();
      setControls();
    } catch (err) {
      phase = sessionId && !sessionDone ? "halted" : "idle";
      stopTracks(stream);
      stream = null;
      window.__mydayStream = null;
      showApiError(err);
      setControls();
    }
  }
  async function onFlipClick() {
    if (!liveSupported || phase === "starting" || phase === "recording" || phase === "stopping") return;
    clearError();
    facingMode = facingMode === "environment" ? "user" : "environment";
    flipBtn.disabled = true;
    try { await startCamera(facingMode); } catch (err) { showApiError(err); } finally { setControls(); }
  }
  function onStopClick() {
    if (phase === "starting" || phase === "stopping") return;
    if (phase !== "recording" && !(sessionId && !sessionDone)) return;
    clearError();
    phase = "stopping";
    setControls();
    showStatus("Stopping");
    if (recorder && recorder.state === "recording") {
      renderTimer(Math.max(0, performance.now() - windowStartedAt));
      endWindow();
      return;
    }
    enqueue(() => finishSession()).catch(onFinishError);
  }
  preview.muted = true;
  preview.playsInline = true;
  window.addEventListener("resize", placeStage);
  recordBtn.addEventListener("click", onRecordClick);
  flipBtn.addEventListener("click", onFlipClick);
  stopBtn.addEventListener("click", onStopClick);
  if (!liveSupported) {
    stage.hidden = false;
    errorEl.textContent = "This browser cannot record the camera.";
  }
  setControls();
})();
</script>
"""


def render_recorder() -> None:
    """Record button. Clicking it records the browser camera in 15 second pieces."""

    st.html(_RECORDER_HTML, unsafe_allow_javascript=True)
