"""FastAPI routes for stream-chunk sessions."""

from __future__ import annotations

import asyncio
import logging
import re
import shutil
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from stream_chunk.config import (
    cors_origins,
    default_seconds,
    host,
    max_upload_bytes,
    port,
    serve_frontend,
    stream_chunk_root,
)
from stream_chunk.ingest import MediaFailed, MediaRejected, accept_media, video_chunker_available
from stream_chunk.sessions import (
    APIError,
    AlreadyComplete,
    SessionClosed,
    UnknownSession,
    create_session,
    get_session,
    load_chunks,
    resolve_chunk_file,
    session_directory,
    session_lock,
    set_status,
    valid_session_id,
    write_window,
)

logger = logging.getLogger(__name__)

_MEDIA_EXTENSIONS = {
    "video/webm": ".webm",
    "video/mp4": ".mp4",
    "video/quicktime": ".mov",
}
_GAP_RE = re.compile(r"\d+")
_MODES = ("exact", "fast")


def json_error(status_code: int, error: str, detail: str) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"error": error, "detail": detail})


def _positive_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if number <= 0 or number != number or number == float("inf"):
        return None
    return number


def _session_view(session: dict[str, Any], *, include_chunks: bool) -> dict[str, Any]:
    view = {
        "id": session["id"],
        "status": session["status"],
        "created_at": session.get("created_at") or "",
        "chunk_duration_s": session["chunk_duration_s"],
        "slack_s": session["slack_s"],
        "mode": session["mode"],
        "label": session.get("label") or "",
    }
    if include_chunks:
        view["chunks"] = load_chunks(session["id"])
    return view


def _parse_create(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise APIError(400, "invalid_json", "request body must be a JSON object")
    if "chunk_duration_s" in payload and payload["chunk_duration_s"] is not None:
        duration = _positive_number(payload["chunk_duration_s"])
        if duration is None:
            raise APIError(400, "invalid_duration", "chunk_duration_s must be greater than 0")
    else:
        duration = default_seconds()
    if "mode" in payload and payload["mode"] is not None:
        mode = payload["mode"]
        if mode not in _MODES:
            raise APIError(400, "unknown_mode", "mode must be exact or fast")
    else:
        mode = "exact"
    if "label" in payload and payload["label"] is not None:
        label = payload["label"]
        if not isinstance(label, str):
            raise APIError(400, "invalid_label", "label must be a string")
    else:
        label = ""
    return create_session(chunk_duration_s=duration, mode=mode, label=label)


def _gap_before_s(request: Request) -> float:
    raw = request.headers.get("x-gap-before-ms")
    if raw is None or raw.strip() == "":
        return 0.0
    text = raw.strip()
    if _GAP_RE.fullmatch(text) is None:
        raise APIError(400, "invalid_gap", "X-Gap-Before-Ms must be a non-negative integer")
    return int(text) / 1000.0


def _media_extension(content_type: str | None) -> str | None:
    if not content_type:
        return None
    base = content_type.split(";", 1)[0].strip().lower()
    return _MEDIA_EXTENSIONS.get(base)


def _process_media(session_id: str, body: bytes, extension: str, gap_before_s: float):
    with session_lock(session_id):
        session = get_session(session_id)
        if session is None:
            raise UnknownSession()
        if session["status"] == "complete":
            raise SessionClosed()
        window_path = write_window(session_id, extension, body)
        set_status(session_id, "cutting")
        try:
            return accept_media(
                session_directory(session_id),
                window_path,
                gap_before_s=gap_before_s,
            )
        finally:
            current = get_session(session_id)
            if current is not None and current["status"] != "complete":
                set_status(session_id, "open")


def _complete(session_id: str) -> dict[str, Any]:
    with session_lock(session_id):
        session = get_session(session_id)
        if session is None:
            raise UnknownSession()
        if session["status"] == "complete":
            raise AlreadyComplete()
        updated = set_status(session_id, "complete")
        return {
            "id": updated["id"],
            "status": "complete",
            "chunk_count": len(load_chunks(session_id)),
        }


def create_app() -> FastAPI:
    application = FastAPI(
        title="stream-chunk",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins(),
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    @application.get("/api/health")
    def health() -> dict[str, bool]:
        return {
            "ffmpeg": shutil.which("ffmpeg") is not None,
            "ffprobe": shutil.which("ffprobe") is not None,
            "video_chunker": video_chunker_available(),
        }

    @application.post("/api/sessions")
    async def post_session(request: Request) -> JSONResponse:
        raw = await request.body()
        if not raw:
            payload: Any = {}
        else:
            try:
                payload = await request.json()
            except Exception:
                return json_error(400, "invalid_json", "request body is not valid JSON")
        try:
            created = _parse_create(payload)
        except APIError as exc:
            return json_error(exc.status_code, exc.error, exc.detail)
        return JSONResponse(_session_view(created, include_chunks=False))

    @application.get("/api/sessions/{session_id}")
    def get_session_route(session_id: str) -> JSONResponse:
        session = get_session(session_id)
        if session is None:
            return json_error(404, "unknown_session", "session not found")
        return JSONResponse(_session_view(session, include_chunks=True))

    @application.post("/api/sessions/{session_id}/media")
    async def post_media(session_id: str, request: Request) -> JSONResponse:
        if not valid_session_id(session_id) or get_session(session_id) is None:
            return json_error(404, "unknown_session", "session not found")
        extension = _media_extension(request.headers.get("content-type"))
        if extension is None:
            return json_error(
                400,
                "unsupported_media_type",
                "content type must be video/webm, video/mp4, or video/quicktime",
            )
        try:
            gap_before_s = _gap_before_s(request)
        except APIError as exc:
            return json_error(exc.status_code, exc.error, exc.detail)

        limit = max_upload_bytes()
        announced = request.headers.get("content-length")
        if announced is not None and announced.strip().isdigit() and int(announced) > limit:
            return json_error(413, "payload_too_large", "body larger than the configured limit")
        body = await request.body()
        if len(body) > limit:
            return json_error(413, "payload_too_large", "body larger than the configured limit")
        if not body:
            return json_error(400, "empty_body", "request body is empty")

        try:
            result = await asyncio.to_thread(_process_media, session_id, body, extension, gap_before_s)
        except (APIError, MediaRejected, MediaFailed) as exc:
            return json_error(exc.status_code, exc.error, exc.detail)
        except OSError:
            logger.exception("storage failure while accepting media")
            return json_error(500, "storage_failed", "the server could not write the session files")
        return JSONResponse(result.as_response(session_id))

    @application.post("/api/sessions/{session_id}/complete")
    async def post_complete(session_id: str) -> JSONResponse:
        if not valid_session_id(session_id):
            return json_error(404, "unknown_session", "session not found")
        try:
            payload = await asyncio.to_thread(_complete, session_id)
        except APIError as exc:
            return json_error(exc.status_code, exc.error, exc.detail)
        return JSONResponse(payload)

    @application.get("/api/sessions/{session_id}/chunks/{filename}", response_model=None)
    def get_chunk(session_id: str, filename: str) -> FileResponse | JSONResponse:
        try:
            path = resolve_chunk_file(session_id, filename)
        except APIError as exc:
            return json_error(exc.status_code, exc.error, exc.detail)
        return FileResponse(path, media_type="video/mp4")

    if serve_frontend():
        frontend_dir = stream_chunk_root() / "frontend"
        if frontend_dir.is_dir():
            application.mount(
                "/",
                StaticFiles(directory=frontend_dir, html=True),
                name="frontend",
            )
    return application


app = create_app()


def main() -> None:
    import uvicorn

    uvicorn.run(app, host=host(), port=port())


if __name__ == "__main__":
    main()
