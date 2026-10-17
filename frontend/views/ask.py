"""Ask page: a conversation about the demo video. Each reply shows its steps, a grounded answer, and moments."""

from __future__ import annotations

import html
import time

import streamlit as st

from frontend.views.common import _paint_connection
from frontend.views.progress import NOTE, Steps, plural, replay, type_out
from frontend.views.search import (
    build_filters,
    clip_row,
    filter_parts,
    journal_matches,
    show_journal,
    show_tiles,
)
from reminisce.agent import answer_question, clips_since
from reminisce.demo import demo_clips, demo_name, key_sentence, span_label
from reminisce.models import ChatModel, ModelError, embed_text
from reminisce.settings import load_settings
from reminisce.tracing import append_trace, safe_error, trace_from_answer


SUGGESTIONS = (
    "Where did I put my phone?",
    "Did I take my medicine?",
    "Where did I put the can?",
    "Did I have pizza?",
    "Who was in the meeting room?",
    "Did I use my laptop?",
)


def _pick_suggestion() -> None:
    picked = st.session_state.get("ask-suggestion")
    if picked:
        st.session_state["ask_pending"] = picked
    st.session_state["ask-suggestion"] = None


def _page_ask(prefs: dict) -> None:
    st.markdown('<p class="greeting">Ask</p>', unsafe_allow_html=True)
    link = st.empty()
    chat: list[dict] = st.session_state.setdefault("ask_chat", [])
    # st.chat_input is pinned to the bottom wherever it is called, so read it before drawing.
    typed = st.chat_input("Ask about your day…", key="ask-input")
    question = (typed or st.session_state.pop("ask_pending", "") or "").strip()
    for index, turn in enumerate(chat):
        _show_turn(index, turn, prefs)
    # Always claim this slot, so the welcome is wiped at once instead of fading out while a reply runs.
    welcome = st.empty()
    if not chat and not question:
        with welcome.container(), st.chat_message("assistant"):
            st.markdown(
                f'<p class="answer-line">Hello. Ask me about your day in {html.escape(demo_name())}, '
                "and I will find the moment.</p>",
                unsafe_allow_html=True,
            )
            st.pills("Try asking", SUGGESTIONS, key="ask-suggestion", on_change=_pick_suggestion, width="stretch")
    if question:
        _ask_turn(question, prefs, chat)
    elif chat and st.button("Start again", key="ask-clear", icon=":material/refresh:", type="tertiary"):
        st.session_state["ask_chat"] = []
        st.session_state["search_open"] = ""
        st.rerun()
    _paint_connection(link)


def _question_bubble(question: str) -> None:
    with st.chat_message("user"):
        st.markdown(f'<p class="chat-question">{html.escape(question)}</p>', unsafe_allow_html=True)


def _ask_turn(question: str, prefs: dict, chat: list[dict]) -> None:
    st.session_state["search_open"] = ""
    _question_bubble(question)
    with st.chat_message("assistant"):
        turn = _run(question, prefs)
        if turn is None:
            return
        chat.append(turn)
        _show_reply(len(chat) - 1, turn, prefs, stream=True)


def _show_turn(index: int, turn: dict, prefs: dict) -> None:
    _question_bubble(turn["question"])
    with st.chat_message("assistant"):
        replay(turn.get("steps_label") or "How I found this", turn.get("steps") or [])
        _show_reply(index, turn, prefs, stream=False)


def _show_reply(index: int, turn: dict, prefs: dict, *, stream: bool) -> None:
    answer = turn["answer"]
    question = turn["question"]
    if not answer.grounded:
        st.error("This answer was withheld because it was not tied to a real clip.")
        return
    if not answer.citations:
        st.markdown(f'<p class="answer-line">{html.escape(answer.text)}</p>', unsafe_allow_html=True)
        show_tiles([hit.clip for hit in turn["journal"]] or turn["moments"], turn=index)
        return
    for clip in answer.clips:
        sentence = key_sentence(clip, question)
        st.markdown(
            f'<p class="llm-line">At {html.escape(span_label(clip))} in {html.escape(demo_name())}</p>',
            unsafe_allow_html=True,
        )
        if stream:
            type_out(st.empty(), sentence)
        else:
            st.markdown(f'<p class="answer-line">{html.escape(sentence)}</p>', unsafe_allow_html=True)
        clip_row(clip, prefs, turn=index, section="answer", line="")
    show_journal(turn["journal"], skip=set(answer.citations), turn=index)


def _agent_progress(steps: Steps):
    """Turn agent events into lines a person would say out loud."""

    def on_event(event: str, **info) -> None:
        if event == "ranking":
            steps.start(f"Looking through {demo_name()}, 10 seconds at a time")
        elif event == "ranked":
            count = info.get("count", 0)
            steps.done(f"Found {plural(count, 'moment')} that might match" if count else "No moment in the video matches")
        elif event == "choosing":
            count = info.get("count", 0)
            steps.start("Choosing the moment that answers your question" if count > 1 else "Looking closely at that moment")
        elif event == "chosen":
            if info.get("by_model"):
                steps.done(f"Picked {plural(info.get('count', 1), 'moment')}")
            else:
                steps.done("Picked the closest moment")
        elif event == "checking":
            steps.start("Making sure the answer matches the video")
        elif event == "checked":
            if not info.get("count"):
                steps.done("Nothing in the video matches that")
            elif info.get("grounded"):
                steps.done("The answer matches the video")
            else:
                steps.done("The answer did not match the video, so it is held back", icon=NOTE)

    return on_event


def _run(question: str, prefs: dict) -> dict | None:
    settings = load_settings()
    words, filters = build_filters(question)
    started = time.perf_counter()
    steps = Steps("Looking for that moment")
    steps.start("Reading your question")
    try:
        vector, _model = embed_text(question, settings)
        steps.done("Understood your question")
    except ModelError:
        vector = []
        steps.done("Reading your question word by word", icon=NOTE)
    narrowed = filter_parts(filters)
    if narrowed:
        steps.note(f"Looking only at {narrowed}")
    moments = clips_since(demo_clips(prefs["owner_id"]), filters.time_window)
    try:
        answer = answer_question(
            question,
            moments,
            vector,
            role="owner",
            owner_id=prefs["owner_id"],
            carer_id=prefs["carer_id"],
            prompt_version="warm",
            llm=ChatModel(settings),
            searcher=None,
            progress=_agent_progress(steps),
        )
    except (ModelError, RuntimeError) as exc:
        steps.finish("Something went wrong", ok=False)
        st.error(safe_error(exc))
        return None
    trace_from_answer("answer", question, answer, time.perf_counter() - started)
    journal = journal_matches(words, moments, vector, filters, prefs["owner_id"])
    outcome = "Found it" if answer.citations else "Could not find that moment"
    steps_label = f"{outcome} · {time.perf_counter() - started:.1f} s"
    steps.finish(steps_label)
    latency = time.perf_counter() - started
    append_trace(
        {
            "kind": "search",
            "question": question,
            "query": words,
            "time_window": filters.time_window,
            "tools": ["search_day", "find_clips"],
            "video": demo_name(),
            "citations": [hit.clip.clip_id for hit in journal],
            "grounded": True,
            "latency_s": round(latency, 3),
        }
    )
    return {
        "question": question,
        "answer": answer,
        "journal": journal,
        "moments": moments,
        "latency": latency,
        "steps": steps.lines,
        "steps_label": steps_label,
    }
