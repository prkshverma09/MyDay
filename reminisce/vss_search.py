"""Video search via POST /api/v1/search.

The login JWT stays in this process. Passwords, tokens, and hosts are never logged.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import requests

from reminisce.settings import CLIPS_DIR, ROOT
from reminisce.store import Clip
from reminisce.vectors import CLIP_ID_RE

_TOKEN: str | None = None
TOP_K = 5
# Hybrid scores on the team archive sit around 0.13 to 0.30, so 0.3 discards real matches.
MIN_SIMILARITY = 0.15
SEGMENT_CACHE = CLIPS_DIR / "vss"
MAX_SEGMENT_BYTES = 40 * 1024 * 1024


class VssSearchError(RuntimeError):
    """Safe to show or store. The message is a status, never a credential."""


@dataclass
class VssHit:
    citation_id: str
    description: str
    similarity: float
    source: str = ""
    clip_id: str = ""
    preview_source: str = ""
    recorded_at: str = ""
    place: str = ""


def cached_token() -> str:
    return _TOKEN or ""


def clear_token() -> None:
    global _TOKEN
    _TOKEN = None


def _env(name: str) -> str:
    return os.environ.get(name, "").strip()


def credentials_ready() -> bool:
    return bool(_env("INGRESS_URL") and _env("USERNAME") and _env("PASSWORD"))


def _base() -> str:
    base = _env("INGRESS_URL").rstrip("/")
    if not base or not _env("USERNAME") or not _env("PASSWORD"):
        raise VssSearchError("search credentials are not set")
    return base


def _login(*, force: bool = False) -> str:
    global _TOKEN
    if _TOKEN and not force:
        return _TOKEN
    try:
        response = requests.post(
            f"{_base()}/api/v1/auth/login",
            json={"username": _env("USERNAME"), "password": _env("PASSWORD")},
            headers={"Content-Type": "application/json"},
            timeout=30,
        )
    except requests.RequestException:
        raise VssSearchError("search unavailable") from None
    if response.status_code >= 400:
        raise VssSearchError(f"search HTTP {response.status_code}")
    try:
        body = response.json()
    except ValueError:
        raise VssSearchError("search unavailable") from None
    token = body.get("access_token") if isinstance(body, dict) else ""
    if not isinstance(token, str) or not token:
        raise VssSearchError("search unavailable")
    _TOKEN = token
    return token


def _authorized(method: str, path: str, *, json_body: dict | None = None, params: dict | None = None, retry: bool = True):
    token = _login()
    headers = {"Authorization": f"Bearer {token}"}
    try:
        response = requests.request(
            method,
            f"{_base()}{path}",
            json=json_body,
            params=params,
            headers=headers,
            timeout=45,
        )
    except requests.RequestException:
        raise VssSearchError("search unavailable") from None
    if response.status_code == 401 and retry:
        clear_token()
        _login(force=True)
        return _authorized(method, path, json_body=json_body, params=params, retry=False)
    return response


def _text(value) -> str:
    if isinstance(value, str):
        return value.strip()
    return ""


def _clip_id_in(value: str) -> str:
    found = CLIP_ID_RE.findall(value or "")
    if len(found) == 1:
        return found[0]
    return ""


def _explicit_clip_id(row: dict) -> str:
    """A journal id only when a field is that id, or a source path names it once."""
    for key in ("clip_id", "source_citation"):
        raw = _text(row.get(key))
        if CLIP_ID_RE.fullmatch(raw):
            return raw
    meta = row.get("metadata")
    if isinstance(meta, dict):
        for key in ("clip_id", "source_citation"):
            raw = _text(meta.get(key))
            if CLIP_ID_RE.fullmatch(raw):
                return raw
    for key in ("source", "preview_source", "original_video", "segment_id"):
        found = _clip_id_in(_text(row.get(key)))
        if found:
            return found
    return ""


def _segment_id(row: dict) -> str:
    for key in ("segment_id", "id", "chunk_id", "result_id"):
        raw = row.get(key)
        if isinstance(raw, str) and raw.strip():
            return raw.strip()[:200]
        if isinstance(raw, int) and not isinstance(raw, bool):
            return str(raw)
    for key in ("source", "preview_source", "original_video"):
        raw = _text(row.get(key))
        if not raw:
            continue
        tail = raw.rstrip("/").split("/")[-1]
        if tail:
            return tail[:200]
    return ""


def _description(row: dict) -> str:
    for key in ("reasoning_content", "caption", "description", "text", "content"):
        raw = _text(row.get(key))
        if raw:
            return raw[:400]
    nested = row.get("best_match")
    if isinstance(nested, dict):
        return _description(nested)
    return ""


def _similarity(row: dict) -> float:
    for key in ("similarity_score", "similarity", "score"):
        raw = row.get(key)
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            return float(raw)
    return 0.0


def _hit_from_row(row: dict) -> VssHit | None:
    clip_id = _explicit_clip_id(row)
    source = _text(row.get("source")) or _text(row.get("original_video"))
    preview = _text(row.get("preview_source"))
    citation = clip_id or _segment_id(row)
    description = _description(row)
    if not citation:
        return None
    if not description and not clip_id:
        return None
    return VssHit(
        citation_id=citation[:200],
        description=description,
        similarity=_similarity(row),
        source=source or preview,
        clip_id=clip_id,
        preview_source=preview,
        recorded_at=_text(row.get("upload_timestamp")),
        place=_text(row.get("location")).replace("_", " "),
    )


def parse_search_payload(payload: dict) -> list[VssHit]:
    """Turn a search response into hits. Unknown shapes yield an empty list."""
    if not isinstance(payload, dict):
        return []
    rows = payload.get("results")
    if not isinstance(rows, list) or not rows:
        rows = payload.get("chunk_results") if isinstance(payload.get("chunk_results"), list) else []
    hits: list[VssHit] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        nested = row.get("results") if "source" not in row and "clip_id" not in row else None
        pieces = nested if isinstance(nested, list) else [row]
        for piece in pieces:
            if not isinstance(piece, dict):
                continue
            hit = _hit_from_row(piece)
            if hit is None or hit.citation_id in seen:
                continue
            seen.add(hit.citation_id)
            hits.append(hit)
    return hits


def search_archive(query: str, *, top_k: int = TOP_K, min_similarity: float = MIN_SIMILARITY) -> list[VssHit]:
    """Natural-language search. Raises VssSearchError with a status-only message."""
    response = _authorized(
        "POST",
        "/api/v1/search",
        json_body={
            "query": query,
            "top_k": top_k,
            "min_similarity": min_similarity,
        },
    )
    if response.status_code >= 400:
        raise VssSearchError(f"search HTTP {response.status_code}")
    try:
        payload = response.json()
    except ValueError:
        raise VssSearchError("search unavailable") from None
    return parse_search_payload(payload if isinstance(payload, dict) else {})


def fetch_segment(source: str) -> Path | None:
    """Download one returned segment into the local cache and return its path.

    The JWT goes only in this server-side request, so it never reaches the browser.
    """
    source = (source or "").strip()
    if not source.startswith("s3://"):
        return None
    SEGMENT_CACHE.mkdir(parents=True, exist_ok=True)
    target = SEGMENT_CACHE / (hashlib.sha1(source.encode()).hexdigest()[:20] + ".mp4")
    if target.is_file() and target.stat().st_size > 0:
        return target
    for attempt in (0, 1):
        try:
            token = _login(force=attempt == 1)
            response = requests.get(
                f"{_base()}/api/v1/videos/stream",
                params={"source": source, "token": token},
                timeout=60,
                stream=True,
            )
        except (requests.RequestException, VssSearchError):
            return None
        if response.status_code == 401 and attempt == 0:
            clear_token()
            continue
        if response.status_code >= 400:
            return None
        temporary = target.with_suffix(".part")
        size = 0
        try:
            with temporary.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1 << 16):
                    size += len(chunk)
                    if size > MAX_SEGMENT_BYTES:
                        raise VssSearchError("segment too large")
                    handle.write(chunk)
        except (requests.RequestException, VssSearchError, OSError):
            temporary.unlink(missing_ok=True)
            return None
        if size == 0:
            temporary.unlink(missing_ok=True)
            return None
        temporary.replace(target)
        return target
    return None


def match_journal_hits(items: list[VssHit], clips: list[Clip]) -> list[tuple[VssHit, Clip]]:
    """Keep hits whose id is a clip this viewer is allowed to see."""
    by_id = {clip.clip_id: clip for clip in clips}
    paired: list[tuple[VssHit, Clip]] = []
    seen: set[str] = set()
    for item in items:
        clip_id = item.clip_id
        if not clip_id:
            found = CLIP_ID_RE.search(item.citation_id or "")
            clip_id = found.group(0) if found else ""
        clip = by_id.get(clip_id)
        if clip is None or clip.clip_id in seen:
            continue
        seen.add(clip.clip_id)
        paired.append((item, clip))
    return paired


def external_clips(items: list[VssHit], *, owner_id: str) -> list[Clip]:
    """Build clips from hits that are not journal rows.

    The citation is the segment id the API returned and the description is the
    API's Cosmos text. Nothing is invented. Media is attached later by
    attach_media, only for the clips that are actually shown.
    """
    built: list[Clip] = []
    seen: set[str] = set()
    for item in items:
        if item.clip_id or not item.citation_id or not item.description:
            continue
        if item.citation_id in seen:
            continue
        seen.add(item.citation_id)
        built.append(
            Clip(
                clip_id=item.citation_id,
                owner_id=owner_id,
                recorded_at=item.recorded_at,
                duration_s=0,
                cosmos_description=item.description,
                activity_tags=[item.place] if item.place else [],
                embedding_id="",
                embedding=[],
                source_citation=item.citation_id,
                media_relpath="",
                description_source="vss",
            )
        )
    return built


def attach_media(clip: Clip, source: str, fetch=None) -> Clip:
    """Point a search clip at its cached segment file, if it can be fetched."""
    fetcher = fetch if fetch is not None else fetch_segment
    try:
        path = fetcher(source) if source else None
    except VssSearchError:
        path = None
    if path:
        path = Path(path)
        try:
            clip.media_relpath = str(path.resolve().relative_to(ROOT))
        except ValueError:
            clip.media_relpath = str(path)
    return clip


def probe_search(query: str) -> dict:
    """Status and hit count only. Safe to print."""
    try:
        hits = search_archive(query)
    except VssSearchError as exc:
        status = 0
        for part in str(exc).split():
            if part.isdigit():
                status = int(part)
                break
        return {"status": status, "count": 0}
    return {"status": 200, "count": len(hits)}
