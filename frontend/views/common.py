"""Shared My Day widgets: navigation, thumbnails, and clip playback."""

from __future__ import annotations

import html
import time
from datetime import datetime
from pathlib import Path

import streamlit as st

from reminisce.agent import answer_question
from reminisce.media import MediaError, extract_frame
from reminisce.models import ChatModel, ModelError, embed_text, load_llm_status
from reminisce.settings import CLIPS_DIR, ROOT, load_settings
from reminisce.store import Clip, day_part, human_when, load_clips, load_prefs, set_shared
from reminisce.tracing import safe_error, trace_from_answer
from reminisce import vast_index

PREVIEW = 4

NAV = (
    ("My day", "nav-day", ":material/home:"),
    ("Ask", "nav-ask", ":material/chat_bubble:"),
    ("Story", "nav-story", ":material/auto_stories:"),
)

def _prefs() -> dict:
    return load_prefs()

def _greeting() -> str:
    hour = datetime.now().astimezone().hour
    if hour < 12:
        return "Good morning"
    if hour < 17:
        return "Good afternoon"
    return "Good evening"

def _clear_open() -> None:
    st.session_state["open_clip"] = ""
    if "open" in st.query_params:
        del st.query_params["open"]

def _nav() -> str:
    if "nav" not in st.session_state:
        st.session_state["nav"] = "My day"
    with st.container(key="quiet-eval"):
        if st.button("Eval", key="nav-eval", type="tertiary", icon=":material/bar_chart:"):
            st.session_state["nav"] = "Eval"
            _clear_open()
            st.rerun()
    st.title("My Day")
    st.markdown('<p class="tagline">Make every day count.</p>', unsafe_allow_html=True)
    columns = st.columns(len(NAV))
    for column, (label, key, icon) in zip(columns, NAV):
        selected = st.session_state["nav"] == label
        with column:
            if st.button(
                label,
                key=key,
                icon=icon,
                type="primary" if selected else "secondary",
                width="stretch",
            ):
                st.session_state["nav"] = label
                if label != "My day":
                    _clear_open()
                st.rerun()
    return st.session_state["nav"]

def _flash() -> None:
    note = st.session_state.pop("note", None)
    if note:
        st.success(note)

def _media(clip: Clip) -> Path | None:
    if not clip.media_relpath:
        return None
    path = Path(clip.media_relpath)
    if not path.is_absolute():
        path = ROOT / path
    return path if path.exists() else None

def _share(clip: Clip, prefs: dict, share: bool) -> str | None:
    updated = set_shared(clip.clip_id, prefs["carer_id"], share)
    try:
        vast_index.mirror_clip(updated)
    except Exception as exc:
        return f"Saved on this computer. The copy did not update ({safe_error(exc)})."
    return None

def _clip_picture(clip: Clip) -> None:
    path = _media(clip)
    if path is not None:
        st.video(str(path))
    else:
        st.info("Video is not on this computer.")


def _first_upper(text: str) -> str:
    return text[:1].upper() + text[1:]


def _clip_words(clip: Clip, *, description: bool) -> None:
    when = html.escape(_first_upper(human_when(clip.recorded_at)))
    st.markdown(f'<p class="clip-time">{when}</p>', unsafe_allow_html=True)
    if description:
        line = html.escape(" ".join(clip.cosmos_description.split()))
        st.markdown(f'<p class="clip-line">{line}</p>', unsafe_allow_html=True)
    st.markdown(
        f'<p class="clip-source">Source: {html.escape(clip.source_citation)}</p>',
        unsafe_allow_html=True,
    )

def _send_button(clip: Clip, prefs: dict, *, key: str) -> None:
    shared = prefs["carer_id"] in clip.shared_with
    if shared:
        if st.button("Take it back", key=f"unshare-{key}", icon=":material/undo:", width="stretch"):
            st.session_state["note"] = _share(clip, prefs, False) or "Taken back."
            st.rerun()
        return
    if st.button("Send", key=f"share-{key}", icon=":material/send:", type="primary", width="stretch"):
        st.session_state["note"] = _share(clip, prefs, True) or f"Sent to {prefs['carer_name']}."
        st.rerun()

def _frame_path(clip: Clip) -> Path | None:
    frame = CLIPS_DIR / "frames" / f"{clip.clip_id}.jpg"
    if frame.is_file() and frame.stat().st_size > 0:
        return frame
    media = _media(clip)
    if media is None:
        return None
    try:
        extract_frame(media, frame)
    except MediaError:
        return None
    return frame if frame.is_file() else None

def _query_open() -> str:
    raw = st.query_params.get("open", "")
    if isinstance(raw, (list, tuple)):
        raw = raw[0] if raw else ""
    return str(raw or "")

def _thumb_slot(clip: Clip | None, index: int) -> None:
    with st.container(key=f"thumb-slot-{index}"):
        if clip is None:
            st.markdown('<p class="thumb-label"> </p>', unsafe_allow_html=True)
            return
        frame = _frame_path(clip)
        if frame is not None:
            st.image(str(frame), width="stretch", link=f"?open={clip.clip_id}")
        else:
            st.info("No picture")
        part = html.escape(day_part(clip.recorded_at))
        when = html.escape(_first_upper(human_when(clip.recorded_at)))
        st.markdown(f'<p class="thumb-label">{part} · {when}</p>', unsafe_allow_html=True)
        if st.button(
            "Play",
            key=f"play-{clip.clip_id}",
            icon=":material/play_arrow:",
            type="primary",
            width="stretch",
        ):
            st.session_state["open_clip"] = clip.clip_id
            st.query_params["open"] = clip.clip_id
            st.rerun()

def _clip_detail(clip: Clip, prefs: dict) -> None:
    if st.button("Back", key="clip-back", icon=":material/arrow_back:", width="stretch"):
        _clear_open()
        st.rerun()
    with st.container(border=True, key=f"feed-{clip.clip_id.replace('_', '-')}"):
        _clip_picture(clip)
        _clip_words(clip, description=True)
        _send_button(clip, prefs, key=clip.clip_id)

def _connection_text() -> str:
    status = load_llm_status()
    if status.get("fallback"):
        return "The language model did not answer, so this shows the closest clip."
    model = str(status.get("model") or "")
    if model and model != "template":
        return f"Connected to {model}"
    return ""

def _paint_connection(slot) -> None:
    text = _connection_text()
    if text:
        slot.markdown(f'<p class="llm-line">{html.escape(text)}</p>', unsafe_allow_html=True)
    else:
        slot.empty()

def _where_from(answer) -> str:
    if answer.retrieval_path == "vss":
        return "Found in your videos"
    if answer.retrieval_note:
        return "Video search is unavailable, so this is from your saved clips"
    return "From your saved clips"


def _ask(question: str, prefs: dict, *, role: str) -> None:
    from reminisce.vss_search import search_archive

    settings = load_settings()
    started = time.perf_counter()
    try:
        with st.spinner("Looking…"):
            vector, _model = embed_text(question, settings)
            answer = answer_question(
                question,
                load_clips(),
                vector,
                role=role,
                owner_id=prefs["owner_id"],
                carer_id=prefs["carer_id"],
                prompt_version="warm",
                llm=ChatModel(settings),
                searcher=search_archive if role == "owner" else None,
            )
    except (ModelError, RuntimeError) as exc:
        st.error(safe_error(exc))
        return
    trace_from_answer("answer" if role == "owner" else "carer_answer", question, answer, time.perf_counter() - started)
    _show_answer(answer, prefs, role=role)


def _show_answer(answer, prefs: dict, *, role: str) -> None:
    if not answer.grounded:
        st.error("This answer was withheld because it was not tied to a real clip.")
        return
    if answer.citations:
        st.markdown(f'<p class="llm-line">{html.escape(_where_from(answer))}</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="answer-line">{html.escape(answer.text)}</p>', unsafe_allow_html=True)
    for clip in answer.clips:
        safe_id = "".join(ch if ch.isalnum() else "-" for ch in clip.clip_id)[:60]
        with st.container(border=True, key=f"shown-{role}-{safe_id}"):
            _clip_picture(clip)
            _clip_words(clip, description=False)
            if role == "owner":
                _send_button(clip, prefs, key=f"ask-{safe_id}")


