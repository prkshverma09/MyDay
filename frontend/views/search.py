"""Search results in an Ask reply, as small picture rows: more saved clips, then the camera archive."""

from __future__ import annotations

import hashlib
import html

import streamlit as st

from frontend.views.common import _first_upper, _frame_path, _media, _send_button
from reminisce.agent import clips_since, find_clips
from reminisce.store import Clip, human_when
from reminisce.vss_search import (
    SearchFilters,
    SearchResult,
    VssMoment,
    VssSearchError,
    fetch_segment,
    metadata_options,
    parse_query,
    search,
)

WHEN = {
    "Any time": "all",
    "Last hour": "1h",
    "Last 24 hours": "24h",
    "Last 7 days": "7d",
    "Last 30 days": "30d",
}
WHEN_LABEL = {value: label for label, value in WHEN.items()}
PAGE = 10
MAX_RESULTS = 50


def _pretty(value: str) -> str:
    return value.replace("_", " ")


def _safe_key(value: str) -> str:
    return hashlib.sha1(value.encode()).hexdigest()[:16]


def _clock(seconds: float) -> str:
    whole = int(max(0.0, seconds))
    return f"{whole // 60}:{whole % 60:02d}"


def build_filters(query: str) -> tuple[str, SearchFilters]:
    """Words for the archive, plus the time and place phrases lifted from the question."""
    return parse_query(query, metadata_options())


def archive_search(words: str, filters: SearchFilters, *, top_k: int) -> tuple[SearchResult | None, str]:
    try:
        return search(words, filters, top_k=top_k), ""
    except VssSearchError as exc:
        return None, str(exc)


def journal_matches(words: str, clips: list[Clip], vector: list[float], filters: SearchFilters, owner_id: str):
    if filters.metadata:
        return []
    return find_clips(words, clips_since(clips, filters.time_window), vector, owner_id=owner_id)


SPOKEN = {
    "location": "in {}",
    "camera_id": "from camera {}",
    "capture_type": "{} footage",
    "object_classes": "showing a {}",
}


def filter_parts(filters: SearchFilters) -> str:
    """"the last 24 hours, in New York", or empty when nothing narrows the search."""
    rest = []
    for field_name, value in filters.metadata.items():
        shown = _pretty(value).title() if field_name == "location" else _pretty(value)
        rest.append(SPOKEN.get(field_name, "{}").format(shown))
    if filters.time_window != "all":
        head = "the " + WHEN_LABEL.get(filters.time_window, filters.time_window).lower()
    elif rest:
        head = "videos"
    else:
        return ""
    return " ".join([head, ", ".join(rest)]).strip()


def _short(text: str, limit: int = 150) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[: limit - 1].rsplit(" ", 1)[0] + "…"


def _row(
    key: str,
    open_id: str,
    thumb,
    *,
    pill: str,
    line: str,
    details: str,
    source: str,
    load_video,
    send=None,
) -> None:
    """One compact result: a small picture on the left, words and buttons on the right."""
    playing = st.session_state.get("search_open") == open_id
    with st.container(border=True, key=f"shown-row-{key}"):
        picture, words = st.columns([1, 2.4], gap="small", vertical_alignment="center")
        with picture:
            if thumb:
                st.image(str(thumb), width="stretch")
            else:
                st.markdown('<div class="thumb-missing">No picture</div>', unsafe_allow_html=True)
        with words:
            parts = []
            if pill:
                parts.append(f'<span class="row-time">{html.escape(pill)}</span>')
            if line:
                parts.append(f'<div class="row-line">{html.escape(_short(line))}</div>')
            if details:
                parts.append(f'<div class="row-meta">{html.escape(details)}</div>')
            parts.append(f'<div class="row-source" title="{html.escape(source)}">Source: {html.escape(source)}</div>')
            st.markdown(f'<div class="row-words">{"".join(parts)}</div>', unsafe_allow_html=True)
            if send is None:
                _play_button(open_id, key=key)
            else:
                play_col, send_col = st.columns(2, gap="small")
                with play_col:
                    _play_button(open_id, key=key)
                with send_col:
                    send()
        if playing:
            with st.spinner("Loading video…"):
                path = load_video()
            if path is not None:
                st.video(str(path), autoplay=True)
            else:
                st.info("This video could not be loaded.")


def clip_row(clip: Clip, prefs: dict, *, turn: int, section: str, description: bool = True, send: bool = True) -> None:
    open_id = f"{turn}:{clip.clip_id}"
    key = _safe_key(f"{section}:{open_id}")
    _row(
        key,
        open_id,
        _frame_path(clip),
        pill=_first_upper(human_when(clip.recorded_at)),
        line=clip.cosmos_description if description else "",
        details="",
        source=clip.source_citation or clip.clip_id,
        load_video=lambda: _media(clip),
        send=(lambda: _send_button(clip, prefs, key=f"{section}-{key}")) if send else None,
    )


def _moment_row(moment: VssMoment, thumb, *, turn: int) -> None:
    open_id = f"{turn}:{moment.citation_id}"
    key = _safe_key(f"archive:{open_id}")
    when = _first_upper(human_when(moment.recorded_at))
    span = f"{_clock(moment.start_sec)}–{_clock(moment.end_sec)}" if moment.end_sec > moment.start_sec else ""
    details = [part for part in (_first_upper(moment.place), moment.camera) if part]
    details.append(f"match {moment.similarity:.2f}")
    if moment.matched_segments > 1:
        details.append(f"{moment.matched_segments} moments in this video")
    _row(
        key,
        open_id,
        thumb,
        pill=" · ".join(part for part in (when, span) if part),
        line=moment.description,
        details=" · ".join(details),
        source=moment.citation_id,
        load_video=lambda: fetch_segment(moment.source),
    )


def show_journal(hits, filters: SearchFilters, prefs: dict, *, skip: set[str], turn: int) -> None:
    hits = [hit for hit in hits if hit.clip.clip_id not in skip]
    if filters.metadata or not hits:
        return
    st.markdown('<p class="result-head">More from your clips</p>', unsafe_allow_html=True)
    for hit in hits:
        clip_row(hit.clip, prefs, turn=turn, section="journal")


def show_archive(
    result: SearchResult | None,
    filters: SearchFilters,
    *,
    skip: set[str],
    top_k: int,
    latency: float,
    thumbs: dict[str, str],
    turn: int,
    allow_more: bool,
) -> bool:
    """Draw the archive results. Returns True when the person asked for more."""
    st.markdown('<p class="result-head">Camera archive</p>', unsafe_allow_html=True)
    if result is None:
        st.markdown('<p class="quiet-note">The camera archive cannot be searched right now.</p>', unsafe_allow_html=True)
        return False
    moments = [moment for moment in result.moments if moment.citation_id not in skip]
    if not moments:
        tail = " for that time or place" if filters.active() else ""
        st.markdown(f'<p class="quiet-note">Nothing else in the archive matches{tail}.</p>', unsafe_allow_html=True)
        return False
    for moment in moments:
        _moment_row(moment, thumbs.get(moment.source), turn=turn)
    st.markdown(
        f'<p class="quiet-note">{len(moments)} videos from {result.total} matching moments, in {latency:.1f} s.</p>',
        unsafe_allow_html=True,
    )
    more = False
    if allow_more and result.total >= top_k and top_k < MAX_RESULTS:
        more = st.button("Show more", key=f"search-more-{turn}", icon=":material/expand_more:", width="stretch")
    if result.summary:
        with st.expander("What the search model says"):
            st.caption(
                f"Written by {result.summary_model or 'the search model'} from the top results. "
                "It is not a memory. Check it against the clips."
            )
            st.markdown(result.summary)
    return more


def _play_button(open_id: str, *, key: str) -> None:
    playing = st.session_state.get("search_open") == open_id
    if st.button(
        "Close" if playing else "Play",
        key=f"search-play-{key}",
        icon=":material/close:" if playing else ":material/play_arrow:",
        type="secondary" if playing else "primary",
        width="stretch",
    ):
        st.session_state["search_open"] = "" if playing else open_id
        st.rerun()
