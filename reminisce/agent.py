"""Agent tools. The words shown to a person always come from a real clip description."""

from __future__ import annotations

from dataclasses import dataclass, field

from reminisce.models import ChatModel, ModelError, remember_llm
from reminisce.store import Clip, human_when
from reminisce.vectors import Hit, extract_clip_ids, rank_clips, select_hits

PROMPTS = {
    "strict": (
        "You pick video clips that answer a question about one person's day. "
        "Reply with only clip ids from the list, one per line. "
        "If none answer the question, reply with the single word NONE. "
        "Do not describe events that are not in the list."
    ),
    "warm": (
        "You help an older adult find a moment from their own day. "
        "Choose only clip ids from the list that answer the question, one per line. "
        "If none match, reply NONE. "
        "Never invent a meal, a visitor, or any other memory."
    ),
}

ABSTAIN = "I couldn't find that moment in your day."


@dataclass
class Answer:
    text: str
    citations: list[str]
    clips: list[Clip]
    hits: list[Hit]
    raw_model: str
    model: str
    prompt_version: str
    selection: str
    grounded: bool
    tools: list[str] = field(default_factory=list)


def clips_for_viewer(clips: list[Clip], *, role: str, owner_id: str, carer_id: str) -> list[Clip]:
    owned = [clip for clip in clips if clip.owner_id == owner_id]
    if role == "owner":
        return owned
    if role == "carer":
        return [clip for clip in owned if carer_id in clip.shared_with]
    return []


def search_day(query: str, clips: list[Clip], query_vector: list[float]) -> list[Hit]:
    return select_hits(rank_clips(query, clips, query_vector))


def get_clip(clips: list[Clip], clip_id: str) -> Clip | None:
    for clip in clips:
        if clip.clip_id == clip_id:
            return clip
    return None


def render_clips(clips: list[Clip]) -> str:
    parts = []
    for clip in clips:
        parts.append(f"At {human_when(clip.recorded_at)}: {clip.cosmos_description} Source: {clip.clip_id}.")
    return " ".join(parts)


def is_grounded(text: str, cited: list[Clip], corpus_ids: set[str]) -> bool:
    found = set(extract_clip_ids(text))
    if not found <= corpus_ids:
        return False
    expected = {clip.clip_id for clip in cited}
    if found != expected:
        return False
    for clip in cited:
        if clip.cosmos_description not in text or clip.clip_id not in text:
            return False
    return True


def _prompt_user(question: str, hits: list[Hit]) -> str:
    lines = [f"Question: {question}", "Clips:"]
    for hit in hits:
        clip = hit.clip
        tags = ", ".join(clip.activity_tags)
        lines.append(f"{clip.clip_id} | {tags} | {clip.cosmos_description}")
    return "\n".join(lines)


def answer_question(
    question: str,
    clips: list[Clip],
    query_vector: list[float],
    *,
    role: str,
    owner_id: str,
    carer_id: str,
    prompt_version: str = "warm",
    llm: ChatModel | None = None,
) -> Answer:
    tools = ["search_day"]
    visible = clips_for_viewer(clips, role=role, owner_id=owner_id, carer_id=carer_id)
    corpus_ids = {clip.clip_id for clip in visible}
    ranked = rank_clips(question, visible, query_vector)
    hits = select_hits(ranked)
    system = PROMPTS.get(prompt_version, PROMPTS["warm"])
    raw = ""
    model_name = "template"
    selection = "abstain"

    if not hits:
        text = ABSTAIN
        return Answer(
            text=text,
            citations=[],
            clips=[],
            hits=ranked[:3],
            raw_model=raw,
            model=model_name,
            prompt_version=prompt_version,
            selection=selection,
            grounded=is_grounded(text, [], corpus_ids),
            tools=tools,
        )

    chosen: list[Clip] = []
    if llm is not None:
        tools.append("select_clips")
        try:
            raw = llm.complete(system, _prompt_user(question, hits))
            model_name = getattr(llm, "model", "llm")
            raw_ids = [clip_id for clip_id in extract_clip_ids(raw) if clip_id in corpus_ids]
            by_id = {hit.clip.clip_id: hit.clip for hit in hits}
            chosen = [by_id[clip_id] for clip_id in raw_ids if clip_id in by_id]
            selection = "model" if chosen else "fallback_top1"
        except ModelError:
            raw = ""
            selection = "fallback_top1"
            model_name = "template"
            remember_llm("", fallback=True)
    else:
        selection = "fallback_top1"

    if not chosen:
        chosen = [hits[0].clip]
        if selection == "abstain":
            selection = "fallback_top1"

    confirmed: list[Clip] = []
    for clip in chosen:
        tools.append("get_clip")
        loaded = get_clip(visible, clip.clip_id)
        if loaded is not None:
            confirmed.append(loaded)

    text = render_clips(confirmed) if confirmed else ABSTAIN
    citations = [clip.clip_id for clip in confirmed]
    return Answer(
        text=text,
        citations=citations,
        clips=confirmed,
        hits=ranked[:3],
        raw_model=raw,
        model=model_name,
        prompt_version=prompt_version,
        selection=selection,
        grounded=is_grounded(text, confirmed, corpus_ids),
        tools=tools,
    )


def daily_digest(clips: list[Clip], *, owner_id: str, carer_id: str, carer_name: str) -> Answer:
    shared = clips_for_viewer(clips, role="carer", owner_id=owner_id, carer_id=carer_id)
    shared = sorted(shared, key=lambda clip: clip.recorded_at)
    corpus_ids = {clip.clip_id for clip in clips}
    if not shared:
        text = f"Nothing has been shared with {carer_name}."
        cited: list[Clip] = []
    else:
        text = f"Shared with {carer_name}. " + render_clips(shared)
        cited = shared
    return Answer(
        text=text,
        citations=[clip.clip_id for clip in cited],
        clips=cited,
        hits=[],
        raw_model="",
        model="template",
        prompt_version="digest",
        selection="shared_only",
        grounded=is_grounded(text, cited, corpus_ids),
        tools=["daily_digest", "get_clip"],
    )


def story(clips: list[Clip], *, owner_id: str) -> Answer:
    owned = clips_for_viewer(clips, role="owner", owner_id=owner_id, carer_id="")
    ordered = sorted(owned, key=lambda clip: clip.recorded_at)
    if len(ordered) > 5:
        step = (len(ordered) - 1) / 4
        picked = [ordered[round(index * step)] for index in range(5)]
    else:
        picked = ordered
    labels = ["First", "Then", "Next", "After that", "Later"]
    if not picked:
        text = "There are no clips to tell a story from yet."
        cited: list[Clip] = []
    else:
        sentences = []
        for label, clip in zip(labels, picked):
            sentences.append(f"{label}, {clip.cosmos_description} Source: {clip.clip_id}.")
        text = " ".join(sentences)
        cited = picked
    corpus_ids = {clip.clip_id for clip in owned}
    return Answer(
        text=text,
        citations=[clip.clip_id for clip in cited],
        clips=cited,
        hits=[],
        raw_model="",
        model="template",
        prompt_version="story",
        selection="time_order",
        grounded=is_grounded(text, cited, corpus_ids) if cited else True,
        tools=["story", "get_clip"],
    )
