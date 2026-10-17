"""Video search via POST /api/v1/search.

The login JWT stays in this process. Passwords, tokens, and hosts are never logged.
"""

from __future__ import annotations

import hashlib
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta
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
THUMB_CACHE = SEGMENT_CACHE / "thumbs"
MAX_SEGMENT_BYTES = 40 * 1024 * 1024
FILTER_FIELDS = ("location", "camera_id", "capture_type", "object_classes")
TIME_WINDOWS = ("all", "1h", "24h", "7d", "30d")
SCHEMA_TTL_S = 600

_SCHEMA: dict[str, list[str]] = {}
_SCHEMA_AT = 0.0


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


@dataclass
class VssMoment:
    """One parent video, pointed at the segment that matched best."""

    citation_id: str
    description: str
    similarity: float
    source: str
    original_video: str = ""
    start_sec: float = 0.0
    end_sec: float = 0.0
    matched_segments: int = 0
    recorded_at: str = ""
    place: str = ""
    camera: str = ""


@dataclass
class SearchFilters:
    time_window: str = "all"
    metadata: dict[str, str] = field(default_factory=dict)

    def request_fields(self, now: datetime | None = None) -> dict:
        fields: dict = {}
        metadata = {key: value for key, value in self.metadata.items() if key in FILTER_FIELDS and value}
        if metadata:
            fields["metadata_filters"] = metadata
        window = self.time_window if self.time_window in TIME_WINDOWS else "all"
        if window == "30d":
            end = now or datetime.now()
            fields["time_filter"] = "custom"
            fields["custom_start_date"] = (end - timedelta(days=30)).strftime("%Y-%m-%dT%H:%M:%S")
            fields["custom_end_date"] = end.strftime("%Y-%m-%dT%H:%M:%S")
        elif window != "all":
            fields["time_filter"] = window
        return fields

    def active(self) -> bool:
        return self.time_window != "all" or any(self.metadata.values())


@dataclass
class SearchResult:
    query: str
    filters: SearchFilters
    hits: list[VssHit]
    moments: list[VssMoment]
    summary: str = ""
    summary_model: str = ""
    total: int = 0
    search_ms: float = 0.0


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


def _number(value) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return 0.0


def parse_moments(payload: dict) -> list[VssMoment]:
    """`chunk_results`: one entry per parent video, with its best-matching segment."""
    rows = payload.get("chunk_results") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        return []
    moments: list[VssMoment] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        source = _text(row.get("preview_source"))
        description = _description(row)
        if not source or not description:
            continue
        citation = _clip_id_in(source) or source.rstrip("/").split("/")[-1][:200]
        if not citation or citation in seen:
            continue
        seen.add(citation)
        moments.append(
            VssMoment(
                citation_id=citation,
                description=description,
                similarity=_similarity(row),
                source=source,
                original_video=_text(row.get("original_video")),
                start_sec=_number(row.get("best_match_start_sec")),
                end_sec=_number(row.get("best_match_end_sec")),
                matched_segments=int(_number(row.get("matched_segment_count"))),
                recorded_at=_text(row.get("upload_timestamp")),
                place=_text(row.get("location")).replace("_", " "),
                camera=_text(row.get("camera_id")),
            )
        )
    return moments


def parse_summary(payload: dict) -> tuple[str, str]:
    """The search model's synthesis text and model name, or empty strings."""
    block = payload.get("llm_synthesis") if isinstance(payload, dict) else None
    if not isinstance(block, dict) or block.get("error"):
        return "", ""
    return _text(block.get("response"))[:4000], _text(block.get("model"))


def search(
    query: str,
    filters: SearchFilters | None = None,
    *,
    top_k: int = TOP_K,
    min_similarity: float = MIN_SIMILARITY,
) -> SearchResult:
    """Natural-language search. Raises VssSearchError with a status-only message."""
    filters = filters or SearchFilters()
    body = {"query": query, "top_k": top_k, "min_similarity": min_similarity}
    body.update(filters.request_fields())
    response = _authorized("POST", "/api/v1/search", json_body=body)
    if response.status_code >= 400:
        raise VssSearchError(f"search HTTP {response.status_code}")
    try:
        payload = response.json()
    except ValueError:
        raise VssSearchError("search unavailable") from None
    if not isinstance(payload, dict):
        payload = {}
    summary, model = parse_summary(payload)
    hits = parse_search_payload(payload)
    return SearchResult(
        query=query,
        filters=filters,
        hits=hits,
        moments=parse_moments(payload),
        summary=summary,
        summary_model=model,
        total=int(_number(payload.get("total"))) or len(hits),
        search_ms=_number(payload.get("search_time_ms")),
    )


def search_archive(query: str, *, top_k: int = TOP_K, min_similarity: float = MIN_SIMILARITY) -> list[VssHit]:
    return search(query, top_k=top_k, min_similarity=min_similarity).hits


def metadata_options(*, refresh: bool = False) -> dict[str, list[str]]:
    """Filterable fields and their values from /metadata/schema. Empty when search is down."""
    global _SCHEMA, _SCHEMA_AT
    if _SCHEMA and not refresh and time.monotonic() - _SCHEMA_AT < SCHEMA_TTL_S:
        return _SCHEMA
    try:
        response = _authorized("GET", "/api/v1/metadata/schema")
        payload = response.json() if response.status_code < 400 else {}
    except (VssSearchError, ValueError):
        return _SCHEMA
    options: dict[str, list[str]] = {}
    for column in payload.get("schema", []) if isinstance(payload, dict) else []:
        if not isinstance(column, dict) or column.get("name") not in FILTER_FIELDS:
            continue
        values = [str(value) for value in column.get("options") or [] if isinstance(value, (str, int))]
        if values:
            options[column["name"]] = sorted(values)
    _SCHEMA, _SCHEMA_AT = options, time.monotonic()
    return options


_TIME_PHRASES = (
    (re.compile(r"\b(?:in\s+the\s+)?(?:last|past)\s+hour\b", re.I), "1h"),
    (re.compile(r"\b(?:today|(?:in\s+the\s+)?(?:last|past)\s+(?:day|24\s+hours))\b", re.I), "24h"),
    (re.compile(r"\b(?:this\s+week|(?:in\s+the\s+)?(?:last|past)\s+(?:week|7\s+days))\b", re.I), "7d"),
    (re.compile(r"\b(?:this\s+month|(?:in\s+the\s+)?(?:last|past)\s+(?:month|30\s+days))\b", re.I), "30d"),
)


def parse_query(text: str, options: dict[str, list[str]] | None = None) -> tuple[str, SearchFilters]:
    """Split "a dog in new york this week" into the query words plus a time window and place.

    Only time phrases and place names are lifted out. Cameras, capture types, and
    objects are chosen explicitly, because words like "traffic" or "person" are
    usually part of what is being searched for.
    """
    remaining = text or ""
    filters = SearchFilters()
    for pattern, window in _TIME_PHRASES:
        if pattern.search(remaining):
            filters.time_window = window
            remaining = pattern.sub(" ", remaining)
            break
    places = sorted((options or {}).get("location", []), key=len, reverse=True)
    for place in places:
        words = r"\s+".join(re.escape(part) for part in place.replace("_", " ").split())
        pattern = re.compile(rf"\b(?:(?:in|at|from|near|around)\s+(?:the\s+)?)?{words}\b", re.I)
        if pattern.search(remaining):
            filters.metadata["location"] = place
            remaining = pattern.sub(" ", remaining)
            break
    cleaned = " ".join(remaining.split()).strip(" ,.?!")
    return (cleaned or " ".join((text or "").split())), filters


def segment_thumbnail(source: str) -> Path | None:
    """One still from a returned segment, cached beside the downloaded segment."""
    from reminisce.media import MediaError, extract_frame

    target = THUMB_CACHE / (hashlib.sha1((source or "").encode()).hexdigest()[:20] + ".jpg")
    if target.is_file() and target.stat().st_size > 0:
        return target
    video = fetch_segment(source)
    if video is None:
        return None
    try:
        extract_frame(video, target)
    except MediaError:
        return None
    return target if target.is_file() else None


def segment_thumbnails(sources: list[str], *, workers: int = 4) -> dict[str, Path | None]:
    unique = list(dict.fromkeys(source for source in sources if source))
    if not unique:
        return {}
    try:
        _login()
    except VssSearchError:
        return {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return dict(zip(unique, pool.map(segment_thumbnail, unique)))


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
