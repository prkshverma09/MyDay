"""Ask page: a grounded answer first, then more matching clips and the camera archive."""

from __future__ import annotations

import time

import streamlit as st

from frontend.views.common import _paint_connection, _show_answer
from frontend.views.progress import NOTE, Steps, plural, replay
from frontend.views.search import (
    MAX_RESULTS,
    PAGE,
    archive_search,
    build_filters,
    filter_parts,
    journal_matches,
    show_archive,
    show_journal,
)
from reminisce.agent import answer_question, clips_since
from reminisce.models import ChatModel, ModelError, embed_text
from reminisce.settings import load_settings
from reminisce.store import load_clips
from reminisce.tracing import append_trace, safe_error, trace_from_answer
from reminisce.vss_search import VssSearchError


SUGGESTIONS = (
    "Where did I put my phone?",
    "Did I take my medicine?",
    "Did I have lunch today?",
    "When did I last go for a walk?",
    "Who visited me today?",
    "Did I lock the front door?",
    "What did I do this afternoon?",
)


def _pick_suggestion() -> None:
    picked = st.session_state.get("ask-suggestion")
    if picked:
        st.session_state["owner_question"] = picked
        st.session_state["ask_pending"] = True
    st.session_state["ask-suggestion"] = None


def _page_ask(prefs: dict) -> None:
    st.markdown('<p class="greeting">Ask</p>', unsafe_allow_html=True)
    link = st.empty()
    question = st.text_area(
        "Question",
        placeholder="Ask about your day, like “did I have lunch today?”",
        key="owner_question",
        height=130,
        label_visibility="collapsed",
    )
    asked = st.button("Ask", key="ask-go", type="primary", icon=":material/search:", width="stretch")
    st.pills(
        "Try asking",
        SUGGESTIONS,
        key="ask-suggestion",
        on_change=_pick_suggestion,
        width="stretch",
    )
    if (asked or st.session_state.pop("ask_pending", False)) and question.strip():
        _run(question.strip(), prefs, top_k=PAGE)
    state = st.session_state.get("ask_state")
    if state:
        _show(state, prefs)
    _paint_connection(link)


def _agent_progress(steps: Steps):
    """Turn agent events into lines a person would say out loud."""

    videos = {"asked": 0, "ready": 0}

    def on_event(event: str, **info) -> None:
        if videos["asked"] and event not in ("fetching", "fetched"):
            asked, ready = videos["asked"], videos["ready"]
            if asked == 1:
                steps.done("Video ready" if ready else "The video could not be loaded, so only the words are shown")
            else:
                steps.done(f"{ready} of {asked} videos ready")
            videos["asked"] = videos["ready"] = 0
        if event == "ranking":
            steps.start("Looking through your saved clips")
        elif event == "ranked":
            count = info.get("count", 0)
            steps.done(f"Found {plural(count, 'clip')} that might match" if count else "None of your saved clips match")
        elif event == "using_archive":
            steps.note(f"Using {plural(info.get('count', 0), 'close match', 'close matches')} from the camera archive")
        elif event == "choosing":
            count = info.get("count", 0)
            steps.start("Choosing the moment that answers your question" if count > 1 else "Looking closely at that moment")
        elif event == "chosen":
            if info.get("by_model"):
                steps.done(f"Picked {plural(info.get('count', 1), 'moment')}")
            else:
                steps.done("Picked the closest moment")
        elif event == "fetching":
            if not videos["asked"]:
                steps.start("Fetching the video")
            videos["asked"] += 1
        elif event == "fetched":
            videos["ready"] += 1 if info.get("ok") else 0
        elif event == "checking":
            steps.start("Making sure the answer matches the clip")
        elif event == "checked":
            if not info.get("count"):
                steps.done("Nothing in your day matches that")
            elif info.get("grounded"):
                steps.done("The answer matches the clip" if info.get("count") == 1 else "The answer matches the clips")
            else:
                steps.done("The answer did not match the clip, so it is held back", icon=NOTE)

    return on_event


def _run(question: str, prefs: dict, *, top_k: int, answer=None) -> None:
    settings = load_settings()
    words, filters = build_filters(question)
    started = time.perf_counter()
    fresh = answer is None
    previous = st.session_state.get("ask_state") or {}
    if not fresh:
        with st.spinner("Looking for more…"):
            result, note = archive_search(words, filters, top_k=top_k)
        journal = previous.get("journal", [])
        steps_label, step_lines = previous.get("steps_label", ""), previous.get("steps", [])
    else:
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
        steps.start("Searching the camera archive")
        result, note = archive_search(words, filters, top_k=top_k)
        if result is None:
            steps.done("The camera archive is not reachable right now", icon=NOTE)
        elif result.moments:
            steps.done(f"Found {plural(len(result.moments), 'video')} in the camera archive")
        else:
            steps.done("Nothing in the camera archive matches")
        clips = load_clips()

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
                progress=_agent_progress(steps),
            )
        except (ModelError, RuntimeError) as exc:
            steps.finish("Something went wrong", ok=False)
            st.error(safe_error(exc))
            return
        trace_from_answer("answer", question, answer, time.perf_counter() - started)
        journal = journal_matches(words, clips, vector, filters, prefs["owner_id"])
        others = len([hit for hit in journal if hit.clip.clip_id not in set(answer.citations)])
        if others:
            steps.done(f"{plural(others, 'other saved clip')} also {'matches' if others == 1 else 'match'}")
        outcome = "Found it" if answer.citations else "Could not find that moment"
        steps_label = f"{outcome} · {time.perf_counter() - started:.1f} s"
        steps.finish(steps_label)
        step_lines = steps.lines
    latency = time.perf_counter() - started
    st.session_state["ask_state"] = {
        "question": question,
        "words": words,
        "filters": filters,
        "answer": answer,
        "journal": journal,
        "result": result,
        "top_k": top_k,
        "latency": latency,
        "steps": step_lines,
        "steps_label": steps_label,
        "fresh": fresh,
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
    fresh = state.pop("fresh", False)
    if not fresh:
        replay(state.get("steps_label") or "How I found this", state.get("steps") or [])
    answer = state["answer"]
    _show_answer(answer, prefs, role="owner", stream=fresh)
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
            prefs,
            top_k=min(MAX_RESULTS, state["top_k"] + PAGE),
            answer=answer,
        )
        st.rerun()
