// Same host that served this page, so a laptop browser using the VM tunnel
// talks to the VM server instead of a hardcoded address.
const API_BASE = location.origin;

const CHUNK_DURATION_S = 15;

const preview = document.querySelector("#preview");
const timerEl = document.querySelector("#timer");
const recordBtn = document.querySelector("#record");
const flipBtn = document.querySelector("#flip");
const stopBtn = document.querySelector("#stop");
const fileInput = document.querySelector("#file");
const liveSection = document.querySelector("#live");
const liveUnavailable = document.querySelector("#live-unavailable");
const statusEl = document.querySelector("#status");
const errorEl = document.querySelector("#error");
const chunkList = document.querySelector("#chunks");
const emptyChunks = document.querySelector("#empty-chunks");

const liveSupported = typeof MediaRecorder !== "undefined"
  && navigator.mediaDevices
  && typeof navigator.mediaDevices.getUserMedia === "function";

let facingMode = "environment";
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

function apiUrl(path) {
  return API_BASE + path;
}

function detailText(detail) {
  if (detail == null || detail === "") return "";
  if (typeof detail === "string") return detail;
  try {
    return JSON.stringify(detail);
  } catch (err) {
    return String(detail);
  }
}

function showStatus(text) {
  statusEl.textContent = text;
}

function clearError() {
  errorEl.textContent = "";
}

function showApiError(err) {
  const error = err && err.error != null && err.error !== ""
    ? String(err.error)
    : (err && err.name) || "Error";
  const detail = detailText(
    err && err.detail != null && err.detail !== ""
      ? err.detail
      : err && err.message
  );
  errorEl.textContent = detail && detail !== error ? error + ": " + detail : error;
}

function fail(error, detail) {
  const err = new Error(detail || error);
  err.error = error;
  err.detail = detail || "";
  return err;
}

async function request(path, options) {
  let response;
  try {
    response = await fetch(apiUrl(path), Object.assign({ cache: "no-store" }, options));
  } catch (err) {
    throw fail("network", err && err.message ? err.message : "Cannot reach the API");
  }

  const text = await response.text();
  let payload = null;
  if (text) {
    try {
      payload = JSON.parse(text);
    } catch (err) {
      payload = null;
    }
  }

  if (!response.ok) {
    const error = payload && payload.error != null
      ? String(payload.error)
      : "HTTP " + response.status;
    const detail = payload && Object.prototype.hasOwnProperty.call(payload, "detail")
      ? payload.detail
      : (payload ? "" : text);
    throw fail(error, detailText(detail));
  }

  return payload;
}

function contentTypeFor(mime) {
  const raw = String(mime || "").trim().toLowerCase();
  const base = raw.split(";")[0].trim();
  if (base === "video/mp4" || base === "video/webm" || base === "video/quicktime") {
    return base;
  }
  return "video/webm";
}

function contentTypeForFile(file) {
  const base = String(file.type || "").split(";")[0].trim().toLowerCase();
  if (base === "video/mp4" || base === "video/webm" || base === "video/quicktime") {
    return base;
  }
  const name = String(file.name || "").toLowerCase();
  if (name.endsWith(".webm")) return "video/webm";
  if (name.endsWith(".mov") || name.endsWith(".qt")) return "video/quicktime";
  return "video/mp4";
}

function chooseMime() {
  if (typeof MediaRecorder === "undefined" || typeof MediaRecorder.isTypeSupported !== "function") {
    return "";
  }
  const candidates = [
    "video/mp4",
    "video/webm;codecs=vp8,opus",
    "video/webm",
  ];
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
  const minutes = Math.floor(seconds / 60);
  const remain = seconds % 60;
  timerEl.textContent = minutes + ":" + String(remain).padStart(2, "0");
}

function clearWindowTimer() {
  if (timerId) {
    clearTimeout(timerId);
    timerId = 0;
  }
}

function setControls() {
  const busy = phase === "starting" || phase === "recording" || phase === "stopping";
  recordBtn.disabled = !liveSupported || busy;
  recordBtn.textContent = phase === "recording" ? "Recording" : "Record";
  flipBtn.disabled = !liveSupported || busy;
  const canStop = phase === "recording"
    || (!!sessionId && !sessionDone && phase !== "starting" && phase !== "stopping");
  stopBtn.disabled = !canStop;
  fileInput.disabled = busy;
}

function clearChunks() {
  chunkList.replaceChildren();
  emptyChunks.hidden = false;
}

function appendChunks(id, rows) {
  for (const row of rows) {
    if (!row || !row.file) continue;
    const item = document.createElement("li");
    const name = document.createElement("p");
    name.className = "chunk-name";
    name.textContent = row.file;
    const time = document.createElement("p");
    time.className = "chunk-time";
    time.textContent = "start_s " + formatSeconds(row.start_s) + " · end_s " + formatSeconds(row.end_s);
    const video = document.createElement("video");
    video.controls = true;
    video.playsInline = true;
    video.setAttribute("playsinline", "");
    video.preload = "metadata";
    video.src = apiUrl(
      "/api/sessions/" + encodeURIComponent(id) + "/chunks/" + encodeURIComponent(row.file)
    );
    item.append(name, time, video);
    chunkList.append(item);
  }
  emptyChunks.hidden = chunkList.children.length > 0;
}

function stopTracks(media) {
  if (!media) return;
  for (const track of media.getTracks()) track.stop();
}

function streamIsLive(media) {
  return !!(media && media.getVideoTracks().some((track) => track.readyState === "live"));
}

async function startCamera(facing) {
  const video = {
    facingMode: facing,
    width: { ideal: 1280 },
    height: { ideal: 720 },
  };
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
  preview.srcObject = stream;
  preview.muted = true;
  const played = preview.play();
  if (played && typeof played.catch === "function") played.catch(() => {});
}

function enqueue(work) {
  const pending = uploadChain.then(work);
  uploadChain = pending.then(
    () => {},
    () => {}
  );
  return pending;
}

async function ensureSession(label) {
  if (sessionId && !sessionDone) return;
  const data = await request("/api/sessions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      chunk_duration_s: CHUNK_DURATION_S,
      mode: "exact",
      label,
    }),
  });
  if (!data || !data.id) {
    throw fail("bad_response", "Session response did not include id");
  }
  sessionId = data.id;
  sessionDone = false;
  chunkSeconds = Number(data.chunk_duration_s) > 0 ? Number(data.chunk_duration_s) : CHUNK_DURATION_S;
  previousStopAt = null;
  clearChunks();
  showStatus("Session open");
}

async function postMedia(id, blob, gapMs, contentType) {
  const data = await request("/api/sessions/" + encodeURIComponent(id) + "/media", {
    method: "POST",
    headers: {
      "Content-Type": contentType,
      "X-Gap-Before-Ms": String(gapMs),
    },
    body: blob,
  });
  if (data && Array.isArray(data.chunks)) appendChunks(id, data.chunks);
  if (phase === "recording") {
    showStatus("Recording · " + chunkList.children.length + " chunks");
  } else if (phase !== "stopping") {
    showStatus("Uploaded. Stop finishes the session.");
  }
  return data;
}

async function finishSession() {
  const id = sessionId;
  const data = await request("/api/sessions/" + encodeURIComponent(id) + "/complete", {
    method: "POST",
  });
  sessionDone = true;
  phase = "idle";
  recorder = null;
  renderTimer(0);
  const count = data && data.chunk_count != null ? data.chunk_count : chunkList.children.length;
  const status = data && data.status ? data.status : "complete";
  showStatus("Session " + status + " · " + count + " chunks");
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
  try {
    recorder.stop();
  } finally {
    ending = false;
  }
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
    // Start the next recorder before the upload returns. The gap is only
    // stop-to-start; the POST queue waits so the next body is not sent early.
    if (continueRecording) {
      try {
        startRecorder();
      } catch (err) {
        phase = "halted";
        showApiError(err);
      }
    }
    const upload = blob.size > 0
      ? enqueue(() => postMedia(id, blob, gapMs, type))
      : Promise.resolve();
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
  setControls();
  try {
    if (!streamIsLive(stream)) await startCamera(facingMode);
    await ensureSession("phone");
    chosenMime = chooseMime();
    phase = "recording";
    startRecorder();
    setControls();
  } catch (err) {
    phase = sessionId && !sessionDone ? "halted" : "idle";
    showApiError(err);
    setControls();
  }
}

async function onFlipClick() {
  if (!liveSupported || phase === "starting" || phase === "recording" || phase === "stopping") return;
  clearError();
  facingMode = facingMode === "environment" ? "user" : "environment";
  flipBtn.disabled = true;
  try {
    await startCamera(facingMode);
  } catch (err) {
    showApiError(err);
  } finally {
    setControls();
  }
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

async function onFileChange() {
  const file = fileInput.files && fileInput.files[0];
  fileInput.value = "";
  if (!file || phase === "starting" || phase === "recording" || phase === "stopping") return;
  clearError();
  phase = "starting";
  setControls();
  try {
    await ensureSession("file");
    showStatus("Uploading");
    await enqueue(() => postMedia(sessionId, file, 0, contentTypeForFile(file)));
    if (phase === "starting") phase = "idle";
  } catch (err) {
    if (phase === "starting") phase = sessionId && !sessionDone ? "halted" : "idle";
    showApiError(err);
  } finally {
    setControls();
  }
}

function init() {
  preview.muted = true;
  preview.playsInline = true;
  if (!liveSupported) {
    liveSection.hidden = true;
    liveUnavailable.hidden = false;
  }
  recordBtn.addEventListener("click", onRecordClick);
  flipBtn.addEventListener("click", onFlipClick);
  stopBtn.addEventListener("click", onStopClick);
  fileInput.addEventListener("change", onFileChange);
  setControls();
}

init();
