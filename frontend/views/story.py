"""Story page: pick a day, read its key moments and what happened as a short timeline."""

from __future__ import annotations

import html
import time
from datetime import date, timedelta

import streamlit as st

from reminisce.agent import day_story
from reminisce.day_memory import load_key_moments
from reminisce.demo import demo_clips
from reminisce.models import ChatModel
from reminisce.store import load_clips
from reminisce.tracing import append_trace

TEA_DEMO_DAY = date(2026, 10, 16)


def _shift(days: int) -> None:
    current = st.session_state.get("story-day") or TEA_DEMO_DAY
    st.session_state["story-day"] = min(date.today(), current + timedelta(days=days))


def _page_story(prefs: dict) -> None:
    st.markdown('<p class="greeting">Story</p>', unsafe_allow_html=True)
    st.session_state.setdefault("story-day", TEA_DEMO_DAY)
    with st.container(key="story-picker"):
        back, picker, forward = st.columns([1, 4, 1], gap="small", vertical_alignment="center")
        with back:
            st.button("", key="story-prev", icon=":material/chevron_left:", on_click=_shift, args=(-1,), width="stretch")
        with picker:
            day = st.date_input(
                "Day",
                key="story-day",
                max_value=date.today(),
                format="DD/MM/YYYY",
                label_visibility="collapsed",
            )
        with forward:
            st.button(
                "",
                key="story-next",
                icon=":material/chevron_right:",
                on_click=_shift,
                args=(1,),
                disabled=day >= date.today(),
                width="stretch",
            )
    _show_key_moments(day)
    _show_day(day, prefs)


def _show_key_moments(day: date) -> None:
    moments = load_key_moments(day.isoformat())
    if not moments:
        return
    rows = "".join(
        '<div class="story-row">'
        + (f'<span class="story-time">{html.escape(moment["span"])}</span>' if moment.get("span") else "")
        + f'<span class="story-text">{html.escape(moment["statement"])}</span></div>'
        for moment in moments
    )
    st.markdown(
        '<div class="story-card"><p class="story-date">Key moments</p>'
        f'<div class="story-line">{rows}</div>'
        '<p class="story-source">From your day\'s memory</p></div>',
        unsafe_allow_html=True,
    )


def _show_day(day: date, prefs: dict) -> None:
    clips = load_clips() + demo_clips(prefs["owner_id"])
    cache = st.session_state.setdefault("story_cache", {})
    signature = (day.isoformat(), tuple(sorted((clip.clip_id, clip.cosmos_description) for clip in clips)))
    story = cache.get(signature)
    if story is None:
        started = time.perf_counter()
        with st.spinner("Writing your day…"):
            story = day_story(clips, day, owner_id=prefs["owner_id"], llm=ChatModel())
        cache[signature] = story
        append_trace(
            {
                "kind": "story",
                "question": f"story {story.day}",
                "citations": [clip.clip_id for _, _, clip in story.moments],
                "grounded": True,
                "model": story.model,
                "latency_s": round(time.perf_counter() - started, 3),
            }
        )
    title = "Today" if day == date.today() else ("Yesterday" if day == date.today() - timedelta(days=1) else "")
    heading = f"{title} · {day:%A %-d %B}" if title else f"{day:%A %-d %B}"
    if not story.moments:
        st.markdown(
            f'<div class="story-card"><p class="story-date">{html.escape(heading)}</p>'
            '<p class="story-empty">Nothing was recorded on this day.</p></div>',
            unsafe_allow_html=True,
        )
        return
    count = len(story.moments)
    first, last = story.moments[0][0], story.moments[-1][0]
    span = f"from {first} to {last}" if count > 1 else f"at {first}"
    rows = "".join(
        f'<div class="story-row"><span class="story-time">{html.escape(clock)}</span>'
        f'<span class="story-text">{html.escape(sentence)}</span></div>'
        for clock, sentence, _clip in story.moments
    )
    source = "Written from your clips" if story.model == "template" else f"Written by {story.model} from your clips"
    st.markdown(
        f'<div class="story-card"><p class="story-date">{html.escape(heading)}</p>'
        f'<p class="story-intro">{count} moment{"s" if count != 1 else ""}, {span}.</p>'
        f'<div class="story-line">{rows}</div>'
        f'<p class="story-source">{html.escape(source)}</p></div>',
        unsafe_allow_html=True,
    )
