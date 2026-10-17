"""Ask page."""

from __future__ import annotations

import streamlit as st

from frontend.views.common import _ask, _paint_connection
from reminisce.media import MediaError, to_wav
from reminisce.models import ModelError, transcribe_wav
from reminisce.settings import CLIPS_DIR
from reminisce.tracing import safe_error

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

