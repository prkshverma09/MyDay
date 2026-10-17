"""Story page."""

from __future__ import annotations

import html
import time

import streamlit as st

from frontend.views.common import _clip_picture, _clip_words
from reminisce.agent import story
from reminisce.store import load_clips
from reminisce.tracing import trace_from_answer

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

