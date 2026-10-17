"""Streamlit pages for the person, their carer, and the team eval."""

from __future__ import annotations

import html
import time
from datetime import datetime
from pathlib import Path

import streamlit as st

from reminisce.agent import answer_question, daily_digest, story
from reminisce.catalog import SAMPLE_IDS
from reminisce.ingest import ingest_video, now_iso
from reminisce.media import MediaError, extract_frame, normalize_upload, record_camera, to_wav
from reminisce.models import ChatModel, ModelError, embed_text, load_llm_status, transcribe_wav
from reminisce.settings import CLIPS_DIR, EVAL_PATH, MAX_CLIP_SECONDS, MIN_CLIP_SECONDS, ROOT, load_settings
from reminisce.store import Clip, day_part, human_when, load_clips, load_prefs, next_clip_id, save_prefs, set_shared
from reminisce.tracing import safe_error, trace_from_answer
from reminisce import vast_index

PREVIEW = 4
NAV = (
    ("My day", "nav-day", ":material/home:"),
    ("Ask", "nav-ask", ":material/chat_bubble:"),
    ("Story", "nav-story", ":material/auto_stories:"),
    ("Carer", "nav-carer", ":material/favorite:"),
)

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,650&family=Outfit:wght@400;500;600;700&display=swap');

html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
  background:
    radial-gradient(920px 440px at 50% -90px, rgba(255, 252, 248, 0.96), rgba(249, 240, 233, 0) 64%),
    linear-gradient(180deg, #FBF7F2 0%, #F3E6DA 100%) !important;
  color: #44341D;
  font-family: "Outfit", "Segoe UI", "Helvetica Neue", sans-serif;
  font-size: 22px;
}
header[data-testid="stHeader"] {
  background: #F9F0E9 !important;
  height: 0 !important;
  min-height: 0 !important;
  overflow: hidden !important;
}
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"],
[data-testid="stAppDeployButton"],
#MainMenu,
footer {
  display: none !important;
}
section[data-testid="stSidebar"],
[data-testid="stSidebarCollapsedControl"] {
  display: none !important;
}
.block-container {
  position: relative;
  max-width: 760px;
  padding-top: 1.4rem;
  padding-bottom: 3rem;
}
h1 {
  font-family: "Outfit", "Segoe UI", sans-serif !important;
  font-size: 28px !important;
  font-weight: 700 !important;
  letter-spacing: 0.22em !important;
  text-transform: uppercase !important;
  text-align: center !important;
  color: #44341D !important;
  line-height: 1.2 !important;
  margin: 0 0 0.15rem !important;
}
h1 a { color: #44341D !important; text-decoration: none !important; }
.tagline {
  margin: 0 0 0.85rem;
  text-align: center;
  font-size: 22px;
  line-height: 1.35;
  color: #44341D;
}
[data-testid="stMarkdownContainer"] p.greeting {
  font-family: "Fraunces", Georgia, "Times New Roman", serif !important;
  font-size: 52px !important;
  font-weight: 650 !important;
  letter-spacing: -0.03em !important;
  line-height: 1.12 !important;
  color: #44341D !important;
  margin: 0.7rem 0 0.2rem !important;
}
[data-testid="stMarkdownContainer"] p.day-part {
  margin: 1.2rem 0 0.1rem !important;
  font-size: 30px !important;
  font-weight: 700 !important;
  letter-spacing: -0.02em !important;
  color: #44341D !important;
}
[data-testid="stMarkdownContainer"] p.quiet-note,
[data-testid="stMarkdownContainer"] p.llm-line,
[data-testid="stMarkdownContainer"] p.clip-source,
[data-testid="stMarkdownContainer"] p.tagline {
  margin: 0.15rem 0 0.35rem !important;
  color: #44341D !important;
  font-size: 22px !important;
  line-height: 1.4 !important;
}
[data-testid="stMarkdownContainer"] p.tagline {
  text-align: center !important;
  margin-bottom: 0.85rem !important;
}
[data-testid="stMarkdownContainer"] p.llm-line { font-weight: 500 !important; }
[data-testid="stMarkdownContainer"] p.clip-time {
  display: inline-block !important;
  margin: 0.7rem 0 0.35rem !important;
  padding: 0.28rem 0.85rem !important;
  border-radius: 999px !important;
  background: #ECE0D4 !important;
  color: #44341D !important;
  font-size: 18px !important;
  font-weight: 700 !important;
  line-height: 1.3 !important;
}
[data-testid="stMarkdownContainer"] p.clip-line,
[data-testid="stMarkdownContainer"] p.answer-line {
  margin: 0.15rem 0 0.35rem !important;
  color: #44341D !important;
  font-size: 22px !important;
  line-height: 1.45 !important;
}
[data-testid="stMarkdownContainer"] p.answer-line { font-size: 26px !important; }
[class*="st-key-quiet-eval"] {
  position: absolute;
  top: 0.15rem;
  right: 0.15rem;
  z-index: 20;
  width: 7.2rem;
}
button[data-testid^="stBaseButton-"] {
  min-height: 56px !important;
  border-radius: 999px !important;
  font-family: "Outfit", "Segoe UI", sans-serif !important;
  font-size: 22px !important;
  font-weight: 600 !important;
  padding: 0.4rem 1rem !important;
  box-shadow: none !important;
}
button[data-testid="stBaseButton-primary"] {
  background: #52735E !important;
  color: #FDF8F5 !important;
  border: 1.5px solid #52735E !important;
}
button[data-testid="stBaseButton-primary"] p,
button[data-testid="stBaseButton-primary"] span,
button[data-testid="stBaseButton-primary"] div {
  color: #FDF8F5 !important;
}
button[data-testid="stBaseButton-secondary"] {
  background: rgba(253, 248, 245, 0.72) !important;
  color: #44341D !important;
  border: 1.5px solid rgba(255, 255, 255, 0.85) !important;
}
button[data-testid="stBaseButton-secondary"] p,
button[data-testid="stBaseButton-secondary"] span {
  color: #44341D !important;
}
button[data-testid="stBaseButton-tertiary"] {
  background: transparent !important;
  color: #44341D !important;
  border: none !important;
  font-size: 20px !important;
}
button[data-testid="stBaseButton-tertiary"] p,
button[data-testid="stBaseButton-tertiary"] span {
  color: #44341D !important;
}
button[data-testid^="stBaseButton-"]:disabled {
  opacity: 0.55;
}
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] label,
[data-testid="stMarkdownContainer"] p {
  color: #44341D;
}
[data-testid="stWidgetLabel"] p {
  font-size: 22px !important;
  font-weight: 600 !important;
}
[data-testid="stTextInput"] input {
  min-height: 56px;
  border-radius: 18px;
  background: rgba(253, 248, 245, 0.78);
  color: #44341D;
  border: 1.5px solid rgba(255, 255, 255, 0.8);
  font-size: 22px !important;
}
[data-testid="stCaptionContainer"] p {
  color: #44341D !important;
  font-size: 20px !important;
}
[data-testid="stAlert"] {
  background: rgba(253, 248, 245, 0.72) !important;
  color: #44341D !important;
  border: 1px solid rgba(255, 255, 255, 0.8) !important;
  border-radius: 20px !important;
  backdrop-filter: blur(16px);
}
[data-testid="stAlert"] p,
[data-testid="stAlert"] div {
  color: #44341D !important;
  font-size: 22px !important;
}
[class*="st-key-feed-"],
[class*="st-key-add-moment"],
[class*="st-key-shown-"],
[class*="st-key-thumb-slot-"] {
  background: rgba(253, 248, 245, 0.58) !important;
  border: 1px solid rgba(255, 255, 255, 0.78) !important;
  border-radius: 28px !important;
  padding: 0.55rem !important;
  box-shadow:
    inset 0 1px 0 rgba(255, 255, 255, 0.88),
    0 16px 36px rgba(68, 52, 29, 0.08);
  backdrop-filter: blur(20px) saturate(150%);
  -webkit-backdrop-filter: blur(20px) saturate(150%);
}
[class*="st-key-gallery"] {
  gap: 16px !important;
}
[class*="st-key-gallery"] [data-testid="stHorizontalBlock"] {
  gap: 16px !important;
}
[class*="st-key-thumb-slot-"] {
  min-height: 248px;
}
[class*="st-key-thumb-slot-"] a {
  display: block;
  line-height: 0;
  border-radius: 20px;
}
[class*="st-key-thumb-slot-"] [data-testid="stImage"] {
  width: 100%;
}
[class*="st-key-thumb-slot-"] [data-testid="stImage"] img {
  width: 100% !important;
  height: 168px !important;
  object-fit: cover !important;
  border-radius: 20px !important;
}
[data-testid="stMarkdownContainer"] p.thumb-label {
  margin: 0.4rem 0 0.2rem !important;
  text-align: center !important;
  color: #44341D !important;
  font-size: 22px !important;
  font-weight: 700 !important;
  line-height: 1.3 !important;
}
[class*="st-key-thumb-slot-"] button,
[class*="st-key-gallery-back"] button,
[class*="st-key-gallery-next"] button,
[class*="st-key-clip-back"] button {
  min-height: 64px !important;
}
[data-testid="stVideo"],
[data-testid="stVideo"] video {
  border-radius: 22px;
  overflow: hidden;
  background: #ECE0D4;
}
[data-testid="stFileUploader"] section {
  background: rgba(253, 248, 245, 0.72);
  border-radius: 20px;
  border: 1.5px dashed rgba(228, 213, 196, 0.95);
}
[data-testid="stFileUploaderDropzoneInstructions"],
[data-testid="stFileUploader"] small {
  display: none !important;
}
[data-testid="stFileUploader"] button {
  min-height: 56px !important;
  width: 100% !important;
  font-size: 22px !important;
}
[data-testid="stMetricValue"],
[data-testid="stMetricLabel"] {
  color: #44341D !important;
}
[data-testid="stMetricValue"] { font-size: 32px !important; }
[data-testid="stMetricLabel"] { font-size: 18px !important; }
[data-testid="stMain"] [data-testid="stVerticalBlock"] { gap: 0.75rem; }
</style>
"""


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
    columns = st.columns(4)
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


def _clip_words(clip: Clip, *, description: bool) -> None:
    when = html.escape(human_when(clip.recorded_at).capitalize())
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
        when = html.escape(human_when(clip.recorded_at).capitalize())
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


def _ask(question: str, prefs: dict, *, role: str) -> None:
    settings = load_settings()
    started = time.perf_counter()
    try:
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
        )
    except (ModelError, RuntimeError) as exc:
        st.error(safe_error(exc))
        return
    trace_from_answer("answer" if role == "owner" else "carer_answer", question, answer, time.perf_counter() - started)
    if not answer.grounded:
        st.error("This answer was withheld because it was not tied to a real clip.")
        return
    st.markdown(f'<p class="answer-line">{html.escape(answer.text)}</p>', unsafe_allow_html=True)
    for clip in answer.clips:
        with st.container(border=True, key=f"shown-{role}-{clip.clip_id.replace('_', '-')}"):
            _clip_picture(clip)
            _clip_words(clip, description=False)
            if role == "owner":
                _send_button(clip, prefs, key=f"ask-{clip.clip_id}")


def _page_ask(prefs: dict) -> None:
    st.markdown('<p class="greeting">Ask</p>', unsafe_allow_html=True)
    link = st.empty()
    audio = st.file_uploader("Voice", type=["wav", "mp3", "m4a", "webm"])
    if audio is not None and st.button("Use voice", icon=":material/mic:", width="stretch"):
        _transcribe(audio)
    question = st.text_input("Question", placeholder="did I have lunch today?", key="owner_question")
    if st.button("Ask", key="ask-go", type="primary", icon=":material/chat_bubble:", width="stretch") and question.strip():
        _ask(question.strip(), prefs, role="owner")
    _paint_connection(link)


def _transcribe(audio) -> None:
    folder = CLIPS_DIR / "uploads"
    folder.mkdir(parents=True, exist_ok=True)
    source = folder / "question-audio"
    source.write_bytes(audio.getvalue())
    wav = folder / "question.wav"
    try:
        to_wav(source, wav)
        text = transcribe_wav(wav)
    except (MediaError, ModelError) as exc:
        st.info(f"Voice is unavailable. Type the question. ({safe_error(exc)})")
        return
    st.session_state["owner_question"] = text
    st.rerun()


def _page_story(prefs: dict) -> None:
    st.markdown('<p class="greeting">Story</p>', unsafe_allow_html=True)
    if st.button("Tell story", key="tell-story", type="primary", icon=":material/auto_stories:", width="stretch"):
        started = time.perf_counter()
        answer = story(load_clips(), owner_id=prefs["owner_id"])
        trace_from_answer("story", "story", answer, time.perf_counter() - started)
        st.markdown(f'<p class="answer-line">{html.escape(answer.text)}</p>', unsafe_allow_html=True)
        for clip in answer.clips:
            with st.container(border=True, key=f"shown-story-{clip.clip_id.replace('_', '-')}"):
                _clip_picture(clip)
                _clip_words(clip, description=False)


def _page_carer(prefs: dict) -> None:
    st.markdown('<p class="greeting">Carer</p>', unsafe_allow_html=True)
    name = st.text_input("Name", value=prefs.get("carer_name") or "my daughter")
    digest = st.toggle("Daily note", value=bool(prefs.get("daily_digest")))
    updated = dict(prefs)
    updated["carer_name"] = name.strip() or "my daughter"
    updated["daily_digest"] = digest
    if updated != prefs:
        save_prefs(updated)
        prefs = updated
    if prefs.get("daily_digest"):
        started = time.perf_counter()
        note = daily_digest(
            load_clips(),
            owner_id=prefs["owner_id"],
            carer_id=prefs["carer_id"],
            carer_name=prefs["carer_name"],
        )
        trace_from_answer("daily_digest", "daily digest", note, time.perf_counter() - started)
        st.markdown('<p class="day-part">Daily note</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="answer-line">{html.escape(note.text)}</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p class="quiet-note">Only clips you send.</p>', unsafe_allow_html=True)
    clips = [
        clip
        for clip in load_clips()
        if clip.owner_id == prefs["owner_id"] and prefs["carer_id"] in clip.shared_with
    ]
    if not clips:
        st.info("Nothing shared.")
    for clip in clips:
        with st.container(border=True, key=f"shown-carer-{clip.clip_id.replace('_', '-')}"):
            _clip_picture(clip)
            _clip_words(clip, description=True)
    st.markdown('<p class="day-part">Ask</p>', unsafe_allow_html=True)
    link = st.empty()
    question = st.text_input("Question", placeholder="when did mum last go for a walk?", key="carer_question")
    if st.button("Ask", key="carer-ask", type="primary", icon=":material/chat_bubble:", width="stretch") and question.strip():
        _ask(question.strip(), prefs, role="carer")
    _paint_connection(link)


def _page_eval() -> None:
    st.markdown('<p class="greeting">Eval</p>', unsafe_allow_html=True)
    if not EVAL_PATH.exists():
        st.info("No eval yet.")
        return
    import json

    report = json.loads(EVAL_PATH.read_text())
    n = report["n"]
    warm = report["prompts"]["warm"]
    strict = report["prompts"]["strict"]
    one, two, three, four = st.columns(4)
    one.metric("Embedding", f"{report['embedding_only_at_1']}/{n}")
    two.metric("Hybrid", f"{report['hybrid_at_1']}/{n}")
    three.metric("Grounded", f"{warm['served_grounded']}/{n}")
    four.metric("Warm prompt", f"{warm['raw_selection']}/{n}")
    st.markdown(
        "<p class='quiet-note'>"
        f"Cosmos descriptions {report['cosmos_descriptions']}/{n}. "
        f"Strict prompt {strict['raw_selection']}/{n}. "
        f"Warm latency {warm['mean_latency_s']}s. "
        f"W&amp;B: {html.escape(str(report.get('wandb') or 'not logged'))}."
        "</p>",
        unsafe_allow_html=True,
    )
    rows = []
    for row in warm["rows"]:
        rows.append(
            {
                "question": row["question"],
                "expected": row["expected"],
                "embedding": row["embedding_top"],
                "hybrid": row["hybrid_top"],
                "served": "yes" if row["served_ok"] else "no",
                "grounded": "yes" if row["grounded"] else "no",
                "latency_s": row["latency_s"],
            }
        )
    st.dataframe(rows, width="stretch", hide_index=True)


def main() -> None:
    st.set_page_config(page_title="My Day", page_icon="☕", layout="centered", initial_sidebar_state="collapsed")
    st.markdown(_CSS, unsafe_allow_html=True)
    prefs = _prefs()
    page = _nav()
    _flash()
    if page == "My day":
        _page_day(prefs)
    elif page == "Ask":
        _page_ask(prefs)
    elif page == "Story":
        _page_story(prefs)
    elif page == "Carer":
        _page_carer(prefs)
    else:
        _page_eval()
