"""Ask page: a grounded answer first, then more matching clips and the camera archive."""

from __future__ import annotations

import html
import time

import streamlit as st

from frontend.views.common import _paint_connection, _show_answer
from frontend.views.search import (
    MAX_RESULTS,
    PAGE,
    archive_search,
    build_filters,
    filter_line,
    filters_box,
    journal_matches,
    show_archive,
    show_journal,
)
from reminisce.agent import answer_question, clips_since
from reminisce.media import MediaError, to_wav
from reminisce.models import ChatModel, ModelError, embed_text, transcribe_wav
from reminisce.settings import CLIPS_DIR, load_settings
from reminisce.store import load_clips
from reminisce.tracing import append_trace, safe_error, trace_from_answer
from reminisce.vss_search import VssSearchError


def _page_ask(prefs: dict) -> None:
    st.markdown('<p class="greeting">Ask</p>', unsafe_allow_html=True)
    link = st.empty()
    audio = st.file_uploader("Voice", type=["wav", "mp3", "m4a", "webm"])
    if audio is not None and st.button("Use voice", icon=":material/mic:", width="stretch"):
        _transcribe(audio)
    question = st.text_input("Question", placeholder="did I have lunch today?", key="owner_question")
    window, picked = filters_box()
    if st.button("Ask", key="ask-go", type="primary", icon=":material/chat_bubble:", width="stretch") and question.strip():
        _run(question.strip(), window, picked, prefs, top_k=PAGE)
    state = st.session_state.get("ask_state")
    if state:
        _show(state, prefs)
    _paint_connection(link)


def _run(question: str, window: str, picked: dict[str, str], prefs: dict, *, top_k: int, answer=None) -> None:
    settings = load_settings()
    words, filters = build_filters(question, window, picked)
    started = time.perf_counter()
    with st.spinner("Looking…"):
        try:
            vector, _model = embed_text(question, settings)
        except ModelError:
            vector = []
        result, note = archive_search(words, filters, top_k=top_k)
        clips = load_clips()
        if answer is None:

            def searcher(_query: str):
                if result is None:
                    raise VssSearchError(note or "search unavailable")
                return result.hits

            try:
                answer = answer_question(
                    question,
                    clips_since(clips, filters.time_window),
                    vector,
                    role="owner",
                    owner_id=prefs["owner_id"],
                    carer_id=prefs["carer_id"],
                    prompt_version="warm",
                    llm=ChatModel(settings),
                    searcher=searcher,
                )
            except (ModelError, RuntimeError) as exc:
                st.error(safe_error(exc))
                return
            trace_from_answer("answer", question, answer, time.perf_counter() - started)
        journal = journal_matches(words, clips, vector, filters, prefs["owner_id"])
    latency = time.perf_counter() - started
    st.session_state["ask_state"] = {
        "question": question,
        "window": window,
        "picked": picked,
        "words": words,
        "filters": filters,
        "answer": answer,
        "journal": journal,
        "result": result,
        "top_k": top_k,
        "latency": latency,
    }
    st.session_state["search_open"] = ""
    moments = [moment.citation_id for moment in (result.moments if result else [])]
    append_trace(
        {
            "kind": "search",
            "question": question,
            "query": words,
            "time_window": filters.time_window,
            "metadata_filters": dict(filters.metadata),
            "tools": ["vss_search", "find_clips"],
            "citations": [hit.clip.clip_id for hit in journal] + moments,
            "grounded": True,
            "journal_hits": [hit.clip.clip_id for hit in journal],
            "archive_moments": moments,
            "archive_total": result.total if result else 0,
            "retrieval_note": note,
            "latency_s": round(latency, 3),
        }
    )


def _show(state: dict, prefs: dict) -> None:
    line = filter_line(state["question"], state["words"], state["filters"])
    if line:
        st.markdown(f'<p class="quiet-note">Looking for {html.escape(line)}</p>', unsafe_allow_html=True)
    answer = state["answer"]
    _show_answer(answer, prefs, role="owner")
    skip = set(answer.citations)
    show_journal(state["journal"], state["filters"], prefs, skip=skip)
    wants_more = show_archive(
        state["result"],
        state["filters"],
        skip=skip,
        top_k=state["top_k"],
        latency=state["latency"],
    )
    if wants_more:
        _run(
            state["question"],
            state["window"],
            state["picked"],
            prefs,
            top_k=min(MAX_RESULTS, state["top_k"] + PAGE),
            answer=answer,
        )
        st.rerun()


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
