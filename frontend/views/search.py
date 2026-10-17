"""Results in an Ask reply: small picture rows of moments, or a 2×2 grid when nothing answers."""

from __future__ import annotations

import base64
import hashlib
import html
from pathlib import Path

import streamlit as st

from frontend.views.common import _first_upper, _frame_path, _media, _send_button
from reminisce.agent import clips_since, find_clips
from reminisce.demo import is_demo, key_sentence, span_label
from reminisce.store import Clip, human_when
from reminisce.vss_search import SearchFilters, parse_query

WHEN_LABEL = {
    "1h": "Last hour",
    "24h": "Last 24 hours",
    "7d": "Last 7 days",
    "30d": "Last 30 days",
}


def _safe_key(value: str) -> str:
    return hashlib.sha1(value.encode()).hexdigest()[:16]


def build_filters(query: str) -> tuple[str, SearchFilters]:
    """The question words, plus any time phrase lifted out of them ("this morning", "today")."""
    return parse_query(query)


def journal_matches(words: str, clips: list[Clip], vector: list[float], filters: SearchFilters, owner_id: str):
    return find_clips(words, clips_since(clips, filters.time_window), vector, owner_id=owner_id)


def filter_parts(filters: SearchFilters) -> str:
    """"the last 24 hours", or empty when the question names no time."""
    if filters.time_window == "all":
        return ""
    return "the " + WHEN_LABEL.get(filters.time_window, filters.time_window).lower()


def _short(text: str, limit: int = 150) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[: limit - 1].rsplit(" ", 1)[0] + "…"


def when_label(clip: Clip) -> str:
    """'0:10–0:20' for a moment inside the demo video, otherwise when the clip was recorded."""
    return span_label(clip) if is_demo(clip) else _first_upper(human_when(clip.recorded_at))


def _row(key: str, open_id: str, thumb, *, pill: str, line: str, source: str, load_video, start: float = 0.0, send=None) -> None:
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
            _play_video(load_video, start)


def _play_video(load_video, start: float) -> None:
    path = load_video()
    if path is not None:
        st.video(str(path), start_time=int(start), autoplay=True)
    else:
        st.info("This video could not be loaded.")


def clip_row(clip: Clip, prefs: dict, *, turn: int, section: str, line: str | None = None) -> None:
    open_id = f"{turn}:{clip.clip_id}"
    key = _safe_key(f"{section}:{open_id}")
    _row(
        key,
        open_id,
        _frame_path(clip),
        pill=when_label(clip),
        line=clip.cosmos_description if line is None else line,
        source=clip.source_citation or clip.clip_id,
        load_video=lambda: _media(clip),
        start=clip.start_sec,
        send=None if is_demo(clip) else (lambda: _send_button(clip, prefs, key=f"{section}-{key}")),
    )


def show_journal(hits, prefs: dict, *, skip: set[str], turn: int, question: str) -> None:
    hits = [hit for hit in hits if hit.clip.clip_id not in skip]
    if not hits:
        return
    st.markdown('<p class="result-head">Other moments</p>', unsafe_allow_html=True)
    for hit in hits:
        clip_row(hit.clip, prefs, turn=turn, section="journal", line=key_sentence(hit.clip, question))


def _tile_background(key: str, thumb) -> str:
    """Per-tile CSS that paints the picture behind that tile's play button."""
    try:
        data = base64.b64encode(Path(thumb).read_bytes()).decode() if thumb else ""
    except OSError:
        data = ""
    if not data:
        return ""
    return (
        f'<style>.st-key-tile-play-{key} button {{'
        f'background-image: url("data:image/jpeg;base64,{data}") !important;}}</style>'
    )


def show_tiles(clips: list[Clip], *, turn: int, count: int = 4) -> None:
    """When nothing answers the question: a few moments as a small 2×2 picture grid."""
    clips = clips[:count]
    if not clips:
        return
    st.markdown('<p class="result-head">From your video</p>', unsafe_allow_html=True)
    playing = None
    for start in range(0, len(clips), 2):
        columns = st.columns(2, gap="small")
        for column, clip in zip(columns, clips[start : start + 2]):
            open_id = f"{turn}:{clip.clip_id}"
            key = _safe_key(f"tile:{open_id}")
            is_open = st.session_state.get("search_open") == open_id
            with column, st.container(key=f"shown-tile-{key}"):
                st.markdown(_tile_background(key, _frame_path(clip)), unsafe_allow_html=True)
                if st.button(
                    "Close" if is_open else "Play",
                    key=f"tile-play-{key}",
                    icon=":material/close:" if is_open else ":material/play_arrow:",
                    width="stretch",
                ):
                    st.session_state["search_open"] = "" if is_open else open_id
                    st.rerun()
                st.markdown(f'<p class="tile-when">{html.escape(when_label(clip))}</p>', unsafe_allow_html=True)
            if is_open:
                playing = clip
    if playing is not None:
        _play_video(lambda: _media(playing), playing.start_sec)


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
