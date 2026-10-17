"""Streamlit entry for My Day. Run: streamlit run frontend/app.py"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from frontend.style import CSS
from frontend.views.ask import _page_ask
from frontend.views.clip import _page_clip
from frontend.views.common import _flash, _nav, _prefs
from frontend.views.day import _page_day
from frontend.views.eval import _page_eval
from frontend.views.story import _page_story


def main() -> None:
    st.set_page_config(page_title="My Day", page_icon="☕", layout="centered", initial_sidebar_state="collapsed")
    st.markdown(CSS, unsafe_allow_html=True)
    prefs = _prefs()
    page = _nav()
    _flash()
    if page == "Ask":
        _page_ask(prefs)
    elif page == "Story":
        _page_story(prefs)
    elif page == "Clip":
        st.markdown(
            "<style>.block-container{max-width:1180px !important;}</style>",
            unsafe_allow_html=True,
        )
        _page_clip(prefs)
    elif page == "Eval":
        _page_eval()
    else:
        _page_day(prefs)


main()
