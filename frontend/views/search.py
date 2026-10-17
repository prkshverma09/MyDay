"""Search results shown under an answer on Ask: more saved clips, then the shared camera archive."""

from __future__ import annotations

import hashlib
import html

import streamlit as st

from frontend.views.common import _clip_picture, _clip_words, _first_upper, _frame_path, _send_button
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


def show_journal(hits, filters: SearchFilters, prefs: dict, *, skip: set[str]) -> None:
    hits = [hit for hit in hits if hit.clip.clip_id not in skip]
    if filters.metadata:
        st.markdown('<p class="day-part">More from your clips</p>', unsafe_allow_html=True)
        st.markdown(
            '<p class="quiet-note">Place, camera, kind, and object filters only apply to the camera archive.</p>',
            unsafe_allow_html=True,
        )
        return
    if not hits:
        return
    st.markdown('<p class="day-part">More from your clips</p>', unsafe_allow_html=True)
    for hit in hits:
        clip = hit.clip
        key = _safe_key(clip.clip_id)
        with st.container(border=True, key=f"shown-search-{key}"):
            if st.session_state.get("search_open") == clip.clip_id:
                _clip_picture(clip)
            else:
                frame = _frame_path(clip)
                if frame is not None:
                    st.image(str(frame), width="stretch")
            _clip_words(clip, description=True)
            _play_button(clip.clip_id, key=f"journal-{key}")
            _send_button(clip, prefs, key=f"search-{key}")


def show_archive(result: SearchResult | None, filters: SearchFilters, *, skip: set[str], top_k: int, latency: float) -> bool:
    """Draw the archive section. Returns True when the person asked for more results."""
    st.markdown('<p class="day-part">Camera archive</p>', unsafe_allow_html=True)
    if result is None:
        st.info("The camera archive cannot be searched right now.")
        return False
    moments = [moment for moment in result.moments if moment.citation_id not in skip]
    if not moments:
        tail = " with those filters" if filters.active() else ""
        st.markdown(f'<p class="quiet-note">Nothing else in the archive matches{tail}.</p>', unsafe_allow_html=True)
        return False
    for moment in moments:
        _moment_card(moment)
    st.markdown(
        f'<p class="quiet-note">{len(moments)} videos from {result.total} matching moments, in {latency:.1f} s.</p>',
        unsafe_allow_html=True,
    )
    more = False
    if result.total >= top_k and top_k < MAX_RESULTS:
        more = st.button("Show more", key="search-more", icon=":material/expand_more:", width="stretch")
    if result.summary:
        with st.expander("What the search model says"):
            st.caption(
                f"Written by {result.summary_model or 'the search model'} from the top results. "
                "It is not a memory. Check it against the clips."
            )
            st.markdown(result.summary)
    return more


def _moment_card(moment: VssMoment) -> None:
    key = _safe_key(moment.citation_id)
    with st.container(border=True, key=f"shown-archive-{key}"):
        if st.session_state.get("search_open") == moment.citation_id:
            with st.spinner("Loading video…"):
                path = fetch_segment(moment.source)
            if path is not None:
                st.video(str(path))
            else:
                st.info("This video could not be loaded.")
        when = _first_upper(human_when(moment.recorded_at))
        span = f"{_clock(moment.start_sec)}–{_clock(moment.end_sec)}" if moment.end_sec > moment.start_sec else ""
        pill = " · ".join(part for part in (when, span) if part)
        st.markdown(f'<p class="clip-time">{html.escape(pill)}</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="clip-line">{html.escape(" ".join(moment.description.split()))}</p>', unsafe_allow_html=True)
        details = [part for part in (_first_upper(moment.place), moment.camera) if part]
        details.append(f"match {moment.similarity:.2f}")
        if moment.matched_segments > 1:
            details.append(f"{moment.matched_segments} moments in this video")
        st.markdown(f'<p class="quiet-note">{html.escape(" · ".join(details))}</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="clip-source">Source: {html.escape(moment.citation_id)}</p>', unsafe_allow_html=True)
        _play_button(moment.citation_id, key=f"archive-{key}")


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
