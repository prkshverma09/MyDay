"""Story page: the day's key moments."""

from __future__ import annotations

import html
from datetime import date

import streamlit as st

from reminisce.day_memory import load_key_moments


def _page_story(_prefs: dict) -> None:
    day = date.today()
    moments = load_key_moments(day.isoformat())
    st.markdown('<p class="greeting">Story</p>', unsafe_allow_html=True)
    st.markdown(
        f'<p class="tagline">{html.escape(_day_label(day))}</p>',
        unsafe_allow_html=True,
    )
    if not moments:
        st.info("No key moments for today yet.")
        return
    for index, moment in enumerate(moments):
        with st.container(border=True, key=f"key-moment-{index}"):
            place = _place(moment["location"])
            if place:
                st.markdown(f'<p class="clip-time">{html.escape(place)}</p>', unsafe_allow_html=True)
            st.markdown(
                f'<p class="answer-line">{html.escape(moment["statement"])}</p>',
                unsafe_allow_html=True,
            )


def _day_label(day: date) -> str:
    return f"{day.strftime('%A')} {day.day} {day.strftime('%B')}"


def _place(location: str) -> str:
    text = " ".join(location.split())
    if not text:
        return ""
    return text[:1].upper() + text[1:]
