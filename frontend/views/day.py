"""My day: 2×2 thumbnail gallery."""

from __future__ import annotations

import html
import time
from datetime import datetime

import streamlit as st

from frontend.views.common import PREVIEW, _clip_detail, _first_upper, _greeting, _query_open, _thumb_slot
from frontend.views.clip import demo_player
from frontend.views.progress import NOTE, Steps
from frontend.views.recorder import render_recorder
from reminisce.demo import DEMO_VIDEO, demo_name, recorded_base
from reminisce.ingest import ingest_video
from reminisce.media import MediaError, normalize_upload
from reminisce.models import ModelError
from reminisce.settings import CLIPS_DIR, MAX_CLIP_SECONDS, MIN_CLIP_SECONDS
from reminisce.store import human_when, load_clips, next_clip_id
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
            render_recorder()
    tag_list = [part.strip().lower() for part in tags.split(",") if part.strip()]
    if add and upload is not None:
        _accept_upload(upload, tag_list, prefs)

def _demo_card() -> None:
    if not DEMO_VIDEO.is_file():
        return
    when = _first_upper(human_when(recorded_base().isoformat()))
    st.markdown(
        f'<p class="demo-head"><span>{html.escape(demo_name())}</span> · {html.escape(when)}</p>'
        '<p class="quiet-note">The text follows the video, 10 seconds at a time. Ask searches this video.</p>',
        unsafe_allow_html=True,
    )
    demo_player()


def _page_day(prefs: dict) -> None:
    st.markdown(f'<p class="greeting">{html.escape(_greeting())}</p>', unsafe_allow_html=True)
    _demo_card()
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
    if adding:
        _add_box(prefs)
    if not clips:
        return
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
    when = datetime.now().astimezone()
    clip_id = next_clip_id(load_clips(), when)
    dest = raw_dir / f"{clip_id}.mp4"
    steps = Steps("Adding your clip")
    try:
        steps.start("Preparing the video")
        normalize_upload(source, dest)
        steps.done("Video ready")
        _ingest(steps, dest, clip_id, when, tags, prefs)
    except (MediaError, ModelError, RuntimeError) as exc:
        steps.finish("The clip could not be added", ok=False)
        st.error(safe_error(exc))
        return
    _saved(steps)

def _ingest(steps: Steps, video, clip_id: str, when: datetime, tags: list[str], prefs: dict) -> None:
    ingest_video(
        video,
        clip_id=clip_id,
        owner_id=prefs["owner_id"],
        recorded_at=when.isoformat(timespec="seconds"),
        tags=tags,
        progress=_ingest_progress(steps),
    )

def _ingest_progress(steps: Steps):
    """Turn ingest events into plain words about what happens to the video."""

    def on_event(event: str, **info) -> None:
        if event == "measuring":
            steps.start("Checking how long the clip is")
        elif event == "measured":
            steps.done(f"The clip is {round(info.get('seconds', 0))} seconds long")
        elif event == "frame":
            steps.start("Taking a still picture from the clip")
        elif event == "framed":
            steps.done("Picture taken")
        elif event == "describing":
            steps.start("Cosmos is looking at the picture")
        elif event == "described":
            if info.get("source") == "cosmos":
                words = " ".join(str(info.get("description", "")).split())
                short = words if len(words) <= 140 else words[:137].rstrip() + "…"
                steps.done(f"Cosmos sees: “{short}”")
            else:
                steps.done("Cosmos did not answer, so your words describe this clip", icon=NOTE)
        elif event == "embedding":
            steps.start("Making the clip easy to find when you ask")
        elif event == "embedded":
            steps.done("Ready to be found")
        elif event == "saving":
            steps.start("Saving it to your day")
        elif event == "saved":
            steps.done("Saved on this computer")
        elif event == "mirroring":
            steps.start("Keeping a copy in the team database")
        elif event == "mirrored":
            if info.get("ok"):
                steps.done("Copy kept in the team database")
            else:
                steps.done("The team copy did not update. Your clip is still saved here", icon=NOTE)

    return on_event

def _saved(steps: Steps) -> None:
    steps.finish("Added to your day")
    time.sleep(1.2)
    st.session_state["note"] = "Saved."
    st.rerun()

