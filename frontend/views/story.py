"""Story page: the day's key moments."""

from __future__ import annotations

import html
from datetime import date

import streamlit as st

from reminisce.day_memory import filed_days, load_key_moments

TEA_DEMO_DAY = "2026-10-16"


def _page_story(_prefs: dict) -> None:
    days = filed_days()
    day = _chosen_day(days)
    moments = load_key_moments(day.isoformat())
    st.markdown('<p class="greeting">Story</p>', unsafe_allow_html=True)
    st.markdown(
        f'<p class="tagline">{html.escape(_day_label(day))}</p>',
        unsafe_allow_html=True,
    )
    if len(days) > 1:
        st.selectbox(
            "Day",
            days,
            format_func=lambda value: _day_label(date.fromisoformat(value)),
            key="story_day",
            label_visibility="collapsed",
        )
    if not moments:
        st.info("No key moments for this day yet.")
        return
    for index, moment in enumerate(moments):
        with st.container(border=True, key=f"key-moment-{index}"):
            caption = _caption(moment)
            if caption:
                st.markdown(f'<p class="clip-time">{html.escape(caption)}</p>', unsafe_allow_html=True)
            st.markdown(
                f'<p class="answer-line">{html.escape(moment["statement"])}</p>',
                unsafe_allow_html=True,
            )


def _chosen_day(days: list[str]) -> date:
    if not days:
        return date.today()
    if "story_day" not in st.session_state:
        st.session_state["story_day"] = TEA_DEMO_DAY if TEA_DEMO_DAY in days else days[-1]
    selected = st.session_state.get("story_day")
    if selected not in days:
        selected = TEA_DEMO_DAY if TEA_DEMO_DAY in days else days[-1]
        st.session_state["story_day"] = selected
    return date.fromisoformat(selected)


def _day_label(day: date) -> str:
    return f"{day.strftime('%A')} {day.day} {day.strftime('%B')}"


def _caption(moment: dict) -> str:
    return str(moment.get("span") or "").strip()
