"""Agent tools. The words shown to a person always come from a real clip description."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from reminisce.models import ChatModel, ModelError, remember_llm
from reminisce.store import Clip, human_when
from reminisce.vectors import CLIP_ID_RE, Hit, extract_clip_ids, rank_clips, select_hits
from reminisce.vss_search import (
    VssSearchError,
    attach_media,
    external_clips,
    match_journal_hits,
)

ALIAS_RE = re.compile(r"hit_(\d+)")
MAX_VSS_HITS = 5
# The shared archive is public camera footage, so a weak hit is a stranger's street, not her day.
ARCHIVE_MIN_SIMILARITY = 0.3

PROMPTS = {
    "strict": (
        "You pick video clips that answer a question about one person's day. "
        "Reply with only ids from the list, one per line. "
        "If none answer the question, reply with the single word NONE. "
        "Do not describe events that are not in the list."
    ),
    "warm": (
        "You help an older adult find a moment from their own day. "
        "Choose only ids from the list that answer the question, one per line. "
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
    retrieval_path: str = ""
    retrieval_note: str = ""


def clips_for_viewer(clips: list[Clip], *, role: str, owner_id: str, carer_id: str) -> list[Clip]:
    owned = [clip for clip in clips if clip.owner_id == owner_id]
    if role == "owner":
        return owned
    if role == "carer":
        return [clip for clip in owned if carer_id in clip.shared_with]
    return []


def search_day(query: str, clips: list[Clip], query_vector: list[float]) -> list[Hit]:
    return select_hits(rank_clips(query, clips, query_vector))


WINDOW_SPANS = {"1h": timedelta(hours=1), "24h": timedelta(days=1), "7d": timedelta(days=7), "30d": timedelta(days=30)}


def clips_since(clips: list[Clip], window: str, now: datetime | None = None) -> list[Clip]:
    """Clips recorded inside a search time window. "all" or an unknown window keeps every clip."""
    span = WINDOW_SPANS.get(window)
    if span is None:
        return list(clips)
    cutoff = (now or datetime.now().astimezone()) - span
    kept = []
    for clip in clips:
        try:
            moment = datetime.fromisoformat(clip.recorded_at)
        except ValueError:
            continue
        if moment.tzinfo is None:
            moment = moment.astimezone()
        if moment >= cutoff:
            kept.append(clip)
    return kept


STOPWORDS = frozenset(
    "a an and are at did do does for from had has have i in is it me my of on or the this to was were what "
    "when where which who with you your today".split()
)
WORD_RE = re.compile(r"[a-z0-9]+")


def _content_words(text: str) -> set[str]:
    return {word for word in WORD_RE.findall((text or "").lower()) if len(word) >= 3 and word not in STOPWORDS}


def words_match(query: str, clip: Clip) -> bool:
    """A search word shares a stem with a tag or a description word ("walking" finds "walk")."""
    wanted = _content_words(query)
    have = _content_words(" ".join(clip.activity_tags) + " " + clip.cosmos_description)
    for word in wanted:
        for other in have:
            if word == other or (min(len(word), len(other)) >= 4 and (word.startswith(other) or other.startswith(word))):
                return True
    return False


def find_clips(query: str, clips: list[Clip], query_vector: list[float], *, owner_id: str, limit: int = 4) -> list[Hit]:
    """Owned clips whose tags or description share a word with the search, best first.

    Cosine only orders the matches. Some title-card embeddings sit close to every
    query, so a clip with no shared word is never shown on the strength of cosine alone.
    """
    owned = clips_for_viewer(clips, role="owner", owner_id=owner_id, carer_id="")
    ranked = rank_clips(query, owned, query_vector)
    matched = [hit for hit in ranked if hit.tag_match or words_match(query, hit.clip)]
    matched.sort(key=lambda hit: (hit.tag_match, hit.cosine), reverse=True)
    return matched[:limit]


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
    cited_ids = {clip.clip_id for clip in cited}
    if not found <= (set(corpus_ids) | cited_ids):
        return False
    expected = {clip_id for clip_id in cited_ids if CLIP_ID_RE.fullmatch(clip_id)}
    if found != expected:
        return False
    for clip in cited:
        if clip.cosmos_description not in text or clip.clip_id not in text:
            return False
    return True


def choose_ids(raw: str, candidate_ids: list[str]) -> list[str]:
    """Ids the model named that are actually in the retrieved list, in reply order."""
    allowed = set(candidate_ids)
    chosen: list[str] = []
    for clip_id in extract_clip_ids(raw):
        if clip_id in allowed and clip_id not in chosen:
            chosen.append(clip_id)
    indexed: list[tuple[int, str]] = []
    for cid in candidate_ids:
        if not cid or CLIP_ID_RE.fullmatch(cid):
            continue
        position = (raw or "").find(cid)
        if position >= 0:
            indexed.append((position, cid))
    for _, cid in sorted(indexed):
        if cid not in chosen:
            chosen.append(cid)
    return chosen


def _prompt_user(question: str, hits: list[Hit], aliases: list[str] | None = None) -> str:
    lines = [f"Question: {question}", "Clips:"]
    for index, hit in enumerate(hits):
        clip = hit.clip
        tags = ", ".join(clip.activity_tags)
        label = aliases[index] if aliases else clip.clip_id
        lines.append(f"{label} | {tags} | {clip.cosmos_description}")
    return "\n".join(lines)


def choose_aliases(raw: str, aliases: list[str]) -> list[str]:
    """Short labels like hit_2 that the model named, in reply order."""
    allowed = set(aliases)
    chosen: list[str] = []
    for number in ALIAS_RE.findall(raw or ""):
        label = f"hit_{number}"
        if label in allowed and label not in chosen:
            chosen.append(label)
    return chosen


def _journal_hits(question: str, visible: list[Clip], query_vector: list[float]) -> tuple[list[Hit], list[Hit]]:
    ranked = rank_clips(question, visible, query_vector)
    return ranked, select_hits(ranked)


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
    searcher=None,
    fetch=None,
    progress=None,
) -> Answer:
    """Answer from the journal, using the video archive only when it adds something real.

    Order: an archive hit that names a journal clip, then the journal rank, then an
    archive-only hit at or above ARCHIVE_MIN_SIMILARITY. Only the owner searches the
    archive. A carer searches clips that were sent to them. `searcher` is omitted by
    the eval and the unit tests that rank the journal directly.

    `progress(event, **info)` is called as each step starts and ends, so a UI can
    show the work as it happens. Events: ranking, ranked, using_archive, choosing,
    chosen, fetching, fetched, checking, checked.
    """

    def emit(event: str, **info) -> None:
        if progress is not None:
            progress(event, **info)

    visible = clips_for_viewer(clips, role=role, owner_id=owner_id, carer_id=carer_id)
    corpus_ids = {clip.clip_id for clip in visible}
    tools: list[str] = []
    retrieval_path = "journal"
    retrieval_note = ""
    vss_items = []

    if searcher is not None and role == "owner":
        tools.append("vss_search")
        try:
            vss_items = list(searcher(question) or [])
        except VssSearchError as exc:
            vss_items = []
            retrieval_note = str(exc)

    matched = match_journal_hits(vss_items, visible) if vss_items else []
    strong = [item for item in vss_items if float(item.similarity or 0.0) >= ARCHIVE_MIN_SIMILARITY]
    outside = external_clips(strong, owner_id=owner_id) if strong else []
    sources = {item.citation_id: item.source or item.preview_source for item in vss_items}
    similarity = {item.citation_id: float(item.similarity or 0.0) for item in vss_items}

    def _vss_ranked(include_outside: bool) -> list[Hit]:
        found = [
            Hit(clip=clip, cosine=float(item.similarity or 0.0), tag_match=False, combined=float(item.similarity or 0.0))
            for item, clip in matched
        ]
        if include_outside:
            for clip in outside:
                score = similarity.get(clip.clip_id, 0.0)
                found.append(Hit(clip=clip, cosine=score, tag_match=False, combined=score))
        return found[:MAX_VSS_HITS]

    emit("ranking", clips=len(visible))
    if matched:
        retrieval_path = "vss"
        ranked = hits = _vss_ranked(include_outside=False)
        emit("ranked", count=len(hits))
    else:
        tools.append("search_day")
        ranked, hits = _journal_hits(question, visible, query_vector)
        emit("ranked", count=len(hits))
        if not hits and outside:
            retrieval_path = "vss"
            ranked = hits = _vss_ranked(include_outside=True)
            emit("using_archive", count=len(hits))

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
            retrieval_path=retrieval_path,
            retrieval_note=retrieval_note,
        )

    chosen: list[Clip] = []
    candidate_ids = [hit.clip.clip_id for hit in hits]
    by_id = {hit.clip.clip_id: hit.clip for hit in hits}
    aliases = [f"hit_{index + 1}" for index in range(len(hits))] if retrieval_path == "vss" else None
    emit("choosing", count=len(hits), model=getattr(llm, "model", "") if llm is not None else "")
    if llm is not None:
        tools.append("select_clips")
        try:
            raw = llm.complete(system, _prompt_user(question, hits, aliases))
            model_name = getattr(llm, "model", "llm")
            raw_ids: list[str] = []
            if aliases:
                by_alias = dict(zip(aliases, candidate_ids))
                raw_ids = [by_alias[label] for label in choose_aliases(raw, aliases)]
            for clip_id in choose_ids(raw, candidate_ids):
                if clip_id not in raw_ids:
                    raw_ids.append(clip_id)
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
    emit("chosen", count=len(chosen), by_model=selection == "model")

    confirmed: list[Clip] = []
    for clip in chosen:
        tools.append("get_clip")
        if clip.clip_id in {item.clip_id for item in visible}:
            loaded = get_clip(visible, clip.clip_id)
        elif retrieval_path == "vss":
            loaded = by_id.get(clip.clip_id)
            if loaded is not None and not loaded.media_relpath:
                tools.append("fetch_segment")
                emit("fetching")
                attach_media(loaded, sources.get(loaded.clip_id, ""), fetch)
                emit("fetched", ok=bool(loaded.media_relpath))
        else:
            loaded = None
        if loaded is not None:
            confirmed.append(loaded)

    emit("checking")
    text = render_clips(confirmed) if confirmed else ABSTAIN
    citations = [clip.clip_id for clip in confirmed]
    grounded = is_grounded(text, confirmed, corpus_ids)
    emit("checked", grounded=grounded, count=len(confirmed))
    return Answer(
        text=text,
        citations=citations,
        clips=confirmed,
        hits=ranked[:3],
        raw_model=raw,
        model=model_name,
        prompt_version=prompt_version,
        selection=selection,
        grounded=grounded,
        tools=tools,
        retrieval_path=retrieval_path,
        retrieval_note=retrieval_note,
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
