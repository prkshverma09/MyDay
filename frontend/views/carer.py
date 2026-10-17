"""Carer page."""

from __future__ import annotations

import html
import time

import streamlit as st

from frontend.views.common import _ask, _clip_picture, _clip_words, _paint_connection
from reminisce.agent import daily_digest
from reminisce.store import load_clips, save_prefs
from reminisce.tracing import trace_from_answer

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

