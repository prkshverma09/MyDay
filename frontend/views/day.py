"""My day: 2×2 thumbnail gallery."""

from __future__ import annotations

import html

import streamlit as st

from frontend.views.common import PREVIEW, _clip_detail, _greeting, _query_open, _thumb_slot
from reminisce.catalog import SAMPLE_IDS
from reminisce.ingest import ingest_video, now_iso
from reminisce.media import MediaError, normalize_upload, record_camera
from reminisce.models import ModelError
from reminisce.settings import CLIPS_DIR, MAX_CLIP_SECONDS, MIN_CLIP_SECONDS
from reminisce.store import load_clips, next_clip_id
from reminisce.tracing import safe_error

def _add_box(prefs: dict) -> None:
    with st.container(border=True, key="add-moment"):
        tags = st.text_input("Words", placeholder="walk, park")
        upload = st.file_uploader(
            "Clip",
            type=["mp4", "mov", "webm", "mkv", "avi"],
            help=f"Between {MIN_CLIP_SECONDS} and {MAX_CLIP_SECONDS} seconds.",
        )
        st.markdown(
            f'<p class="quiet-note">{MIN_CLIP_SECONDS}–{MAX_CLIP_SECONDS} seconds.</p>',
            unsafe_allow_html=True,
        )
        add_col, record_col = st.columns(2)
        with add_col:
            add = st.button(
                "Add",
                disabled=upload is None,
                icon=":material/add:",
                type="primary",
                width="stretch",
            )
        with record_col:
            record = st.button("Record", icon=":material/videocam:", width="stretch")
    tag_list = [part.strip().lower() for part in tags.split(",") if part.strip()]
    if add and upload is not None:
        _accept_upload(upload, tag_list, prefs)
    if record:
        _record(tag_list, prefs)

def _page_day(prefs: dict) -> None:
    st.markdown(f'<p class="greeting">{html.escape(_greeting())}</p>', unsafe_allow_html=True)
    clips = [clip for clip in load_clips() if clip.owner_id == prefs["owner_id"]]
    adding = bool(st.session_state.get("adding"))
    if st.button(
        "Close" if adding else "New clip",
        key="toggle-add",
        icon=":material/close:" if adding else ":material/add:",
        width="stretch",
    ):
        st.session_state["adding"] = not adding
        st.rerun()
    if adding or not clips:
        _add_box(prefs)
    if not clips:
        st.info("No clips yet.")
        return
    if any(clip.clip_id in SAMPLE_IDS for clip in clips):
        st.markdown('<p class="quiet-note">Practice clips.</p>', unsafe_allow_html=True)
    chosen = _query_open()
    if chosen:
        st.session_state["open_clip"] = chosen
    open_id = str(st.session_state.get("open_clip") or "")
    opened = next((clip for clip in clips if clip.clip_id == open_id), None)
    if opened is not None:
        _clip_detail(opened, prefs)
        return
    page = int(st.session_state.get("gallery_page") or 0)
    page_count = max(1, (len(clips) + PREVIEW - 1) // PREVIEW)
    if page < 0 or page >= page_count:
        page = 0
        st.session_state["gallery_page"] = 0
    window = clips[page * PREVIEW : page * PREVIEW + PREVIEW]
    with st.container(key="gallery"):
        top = st.columns(2, gap="medium")
        bottom = st.columns(2, gap="medium")
        cells = (top[0], top[1], bottom[0], bottom[1])
        for index, cell in enumerate(cells):
            clip = window[index] if index < len(window) else None
            with cell:
                _thumb_slot(clip, index)
    if len(clips) > PREVIEW:
        back_col, next_col = st.columns(2)
        with back_col:
            if st.button(
                "Back",
                key="gallery-back",
                icon=":material/arrow_back:",
                disabled=page == 0,
                width="stretch",
            ):
                st.session_state["gallery_page"] = page - 1
                st.rerun()
        with next_col:
            if st.button(
                "Next",
                key="gallery-next",
                icon=":material/arrow_forward:",
                disabled=page >= page_count - 1,
                width="stretch",
            ):
                st.session_state["gallery_page"] = page + 1
                st.rerun()

def _accept_upload(upload, tags: list[str], prefs: dict) -> None:
    if upload.size > 25 * 1024 * 1024:
        st.error("That file is larger than 25 MB. Use a shorter clip.")
        return
    raw_dir = CLIPS_DIR / "uploads"
    raw_dir.mkdir(parents=True, exist_ok=True)
    source = raw_dir / f"raw-{upload.name}"
    source.write_bytes(upload.getvalue())
    clip_id = next_clip_id(load_clips())
    dest = raw_dir / f"{clip_id}.mp4"
    try:
        with st.spinner("Saving…"):
            normalize_upload(source, dest)
            ingest_video(
                dest,
                clip_id=clip_id,
                owner_id=prefs["owner_id"],
                recorded_at=now_iso(),
                tags=tags,
            )
    except (MediaError, ModelError, RuntimeError) as exc:
        st.error(safe_error(exc))
        return
    st.session_state["note"] = "Saved."
    st.rerun()

def _record(tags: list[str], prefs: dict) -> None:
    clip_id = next_clip_id(load_clips())
    dest = CLIPS_DIR / "uploads" / f"{clip_id}.mp4"
    try:
        with st.spinner("Recording…"):
            recorded = record_camera(dest, seconds=8)
        if recorded is None:
            st.info("No camera. Add a clip instead.")
            return
        with st.spinner("Saving…"):
            ingest_video(
                recorded,
                clip_id=clip_id,
                owner_id=prefs["owner_id"],
                recorded_at=now_iso(),
                tags=tags,
            )
    except (MediaError, ModelError, RuntimeError) as exc:
        st.error(safe_error(exc))
        return
    st.session_state["note"] = "Saved."
    st.rerun()

