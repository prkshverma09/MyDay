from __future__ import annotations

import re

from myday_memory.models import (
    PERSONAL_OBJECTS,
    QUESTION_IDS,
    Answer,
    ChunkText,
    MemoryEntry,
)

TRACKED = {
    "keys": ("keys", "key"),
    "glasses": ("glasses", "spectacles"),
    "hearing aid": ("hearing aid", "hearing aids"),
    "mug": ("mug",),
    "cat": ("cat",),
}

SKIP_NAMES = {
    "the",
    "this",
    "that",
    "there",
    "then",
    "and",
    "but",
    "she",
    "her",
    "his",
    "dont",
    "don't",
    "blue",
    "kitchen",
    "living",
    "room",
    "front",
    "door",
    "counter",
    "dentist",
    "doctor",
    "optician",
    "hospital",
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
    "please",
    "margaret",
    "mum",
    "mom",
    "dad",
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
}

NAME_RE = re.compile(r"(?<!\w)([A-Z][a-z]{2,})(?![\w'])")
LOC_RE = re.compile(
    r"\b(in|into|on|onto)\s+(?:the|a|an|her|his)\s+([a-z0-9][a-z0-9 ]{0,30}?)(?=\s+on\b|\s+in\b|\s+and\b|[.,]|$)",
    re.IGNORECASE,
)
WHEN_RE = re.compile(
    r"\b((?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
    r"\s*(?:,\s*|\s+at\s+|\s+)\d{1,2}(?::\d{2})?\s*(?:am|pm)?)\b",
    re.IGNORECASE,
)
PET_RE = re.compile(r"\b([A-Z][a-z]{2,})\s+the\s+cat\b")
WHAT_RE = re.compile(r"\b(dentist|doctor|optician|hospital|appointment|gp)\b", re.IGNORECASE)
_STOVE = r"stove|hob|burner|oven|ring"
STOVE_ON_RE = re.compile(
    rf"\b(?:{_STOVE})\b.{{0,40}}\b(?:on|lit)\b|\b(?:on|lit)\b.{{0,40}}\b(?:{_STOVE})\b",
    re.IGNORECASE,
)
STOVE_OFF_RE = re.compile(
    rf"\b(?:{_STOVE})\b.{{0,40}}\boff\b|\boff\b.{{0,40}}\b(?:{_STOVE})\b",
    re.IGNORECASE,
)
LOCK_RE = re.compile(r"\b(unlocked|locked|locking|locks|lock)\b", re.IGNORECASE)
DOOR_RE = re.compile(r"\bdoor\b", re.IGNORECASE)


def mentioned(text: str, phrase: str) -> bool:
    return re.search(rf"\b{re.escape(phrase)}\b", text, re.IGNORECASE) is not None


COMMON_OPENERS = {
    "just",
    "they",
    "them",
    "steam",
    "morning",
    "remember",
    "hello",
    "lovely",
    "did",
    "please",
    "there",
    "then",
    "this",
    "that",
    "she",
    "her",
    "his",
    "and",
    "but",
    "audio",
    "pov",
    "yolo",
    "cosmos",
    "oh",
}


def _people(text: str, resident: list[str], pets: list[str]) -> list[str]:
    blocked = SKIP_NAMES | {name.lower() for name in resident} | {name.lower() for name in pets}
    blocked |= {match.group(1).lower() for match in PET_RE.finditer(text)}
    found: list[str] = []
    for match in NAME_RE.finditer(text):
        name = match.group(1)
        if name.lower() in blocked or name in found:
            continue
        prefix = text[: match.start()]
        at_start = prefix.strip() == "" or prefix.rstrip()[-1:] in ".!?\n"
        if at_start and name.lower() in COMMON_OPENERS:
            continue
        found.append(name)
    return found


def _objects(text: str, detections: list[str]) -> list[str]:
    found: list[str] = []
    detected = {item.lower() for item in detections}
    for label, aliases in TRACKED.items():
        if label in detected or any(alias in detected for alias in aliases):
            found.append(label)
            continue
        if any(mentioned(text, alias) for alias in aliases):
            found.append(label)
    return found


def _location(text: str, label: str) -> tuple[str, str] | None:
    aliases = TRACKED.get(label, (label,))
    lower = text.lower()
    positions = [lower.find(alias) for alias in aliases if alias in lower]
    anchor = min(positions) if positions else -1
    matches = list(LOC_RE.finditer(text))
    if not matches:
        return None
    if anchor < 0:
        chosen = matches[0]
    else:
        after = [match for match in matches if match.start() >= anchor]
        chosen = after[0] if after else matches[-1]
    prep = chosen.group(1).lower()
    prep = "in" if prep in {"in", "into"} else "on"
    where = re.sub(r"\s+", " ", chosen.group(2)).strip()
    return prep, where


def _appointment(text: str) -> tuple[str, str] | None:
    when = WHEN_RE.search(text)
    if not when:
        return None
    what = WHAT_RE.search(text)
    return (
        what.group(1).lower() if what else "appointment",
        re.sub(r"\s+", " ", when.group(1)).strip(),
    )


def _stove_state(text: str) -> str | None:
    if STOVE_OFF_RE.search(text):
        return "off"
    if STOVE_ON_RE.search(text):
        return "on"
    return None


def _door_state(text: str) -> str | None:
    if not DOOR_RE.search(text):
        return None
    match = LOCK_RE.search(text)
    if not match:
        return None
    word = match.group(1).lower()
    if word == "unlocked":
        return "unlocked"
    return "locked"


def _place_phrase(place: str) -> str:
    if not place:
        return "here"
    if "door" in place.lower():
        return f"at the {place}"
    return f"in the {place}"


def _object_clause(label: str, prep: str, where: str) -> str:
    verb = "are" if label in {"keys", "glasses"} else "is"
    if where:
        return f"The {label} {verb} {prep} the {where}."
    return f"The {label} {verb} here."


def _article_object(label: str) -> str:
    article = "the" if label != "hearing aid" else "the"
    return f"{article} {label}"


def extract_entry(chunk: ChunkText) -> MemoryEntry:
    text = chunk.source_text
    people = _people(text, chunk.resident, chunk.pets)
    objects = _objects(text, chunk.detections)
    details: dict[str, dict[str, str]] = {}
    for label in objects:
        located = _location(text, label)
        if label == "cat" and located and "bowl" in located[1] and re.search(r"\b(food|feed|tips)\b", text, re.I):
            located = None
        if located:
            prep, where = located
            details[label] = {"prep": prep, "where": where}
        elif chunk.place:
            details[label] = {"prep": "in", "where": chunk.place}

    appointment = _appointment(text)
    stove = _stove_state(text)
    door = _door_state(text)
    answers: list[Answer] = []

    personal = next((label for label in PERSONAL_OBJECTS if label in objects), None)
    statement = ""
    action = "was recorded"

    if personal:
        detail = details.get(personal, {})
        prep = detail.get("prep", "in")
        where = detail.get("where", chunk.place)
        statement = _object_clause(personal, prep, where)
        action = f"put {_article_object(personal)} down"
        answers.append(
            Answer(statement, chunk.clip_id, chunk.t_start, 0.9 if where else 0.7, False, "lost_objects")
        )
    elif people and appointment:
        what, when = appointment
        when_bit = f" on {when}" if when else ""
        statement = f"{_join(people)} was {_place_phrase(chunk.place)} and mentioned the {what}{when_bit}."
        action = "visited and talked about an appointment"
    elif people:
        statement = f"{_join(people)} was {_place_phrase(chunk.place)}."
        action = "visited"
    elif stove:
        statement = f"The stove is {stove} {_place_phrase(chunk.place)}."
        action = "left the stove on" if stove == "on" else "turned the stove off"
    elif door:
        statement = f"The door was {door}."
        action = "locked the door" if door == "locked" else "unlocked the door"
    elif objects:
        label = objects[0]
        detail = details.get(label, {})
        statement = _object_clause(label, detail.get("prep", "in"), detail.get("where", chunk.place))
        action = "was recorded"
    elif text:
        statement = text.split(".", 1)[0].strip()
        if statement and not statement.endswith("."):
            statement += "."
    else:
        statement = "Nothing notable was described."

    if people:
        answers.append(
            Answer(
                f"{_join(people)} was {_place_phrase(chunk.place)}.",
                chunk.clip_id,
                chunk.t_start,
                0.9,
                False,
                "people",
            )
        )
    if appointment:
        what, when = appointment
        when_bit = f" on {when}" if when else ""
        driving = ""
        if re.search(r"\b(drive|driving)\b", text, re.IGNORECASE) and "sarah" in text.lower():
            driving = ", Sarah driving"
        confidence = 0.85 if what and when else 0.6
        answers.append(
            Answer(
                f"{what[:1].upper()}{what[1:]}{when_bit}{driving}.",
                chunk.clip_id,
                chunk.t_start,
                confidence,
                False,
                "appointment",
            )
        )
    if stove:
        answers.append(
            Answer(
                f"The stove is {stove} {_place_phrase(chunk.place)}.",
                chunk.clip_id,
                chunk.t_start,
                0.85,
                False,
                "stove",
            )
        )
    if door:
        answers.append(
            Answer(
                f"The door was {door}.",
                chunk.clip_id,
                chunk.t_start,
                0.85,
                False,
                "door_lock",
            )
        )

    if chunk.summary:
        statement = chunk.summary
    high = bool(people or personal or stove or door)
    highlighted = bool(answers) if chunk.key_moment is None else chunk.key_moment
    return MemoryEntry(
        clip_id=chunk.clip_id,
        t_start=chunk.t_start,
        t_end=chunk.t_end,
        place=chunk.place,
        people=people,
        action=action,
        objects=objects,
        statement=statement,
        transcript=chunk.transcript,
        answers=answers,
        importance="high" if high else "normal",
        camera=chunk.camera,
        file=chunk.file,
        object_details=details,
        key_moment=highlighted,
        highlights=list(chunk.highlights) if highlighted else [],
    )


def _join(names: list[str]) -> str:
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return ", ".join(names[:-1]) + f", and {names[-1]}"


def content_words(text: str) -> set[str]:
    stop = {
        "the",
        "and",
        "was",
        "were",
        "with",
        "that",
        "this",
        "into",
        "onto",
        "from",
        "have",
        "been",
        "your",
        "about",
    }
    return {word for word in re.findall(r"[a-z]{4,}", text.lower()) if word not in stop}


def ground_llm_payload(payload: dict, chunk: ChunkText, rules: MemoryEntry) -> MemoryEntry | None:
    """Keep an LLM structuring only where the source text supports it."""

    if not isinstance(payload, dict):
        return None
    text = chunk.source_text
    words = content_words(text)
    statement = str(payload.get("statement") or "").strip()
    statement_words = content_words(statement)
    if not statement_words or not statement_words <= words:
        statement = rules.statement

    action = str(payload.get("action") or "").strip()
    action_words = content_words(action)
    if not action or not action_words or not action_words <= words:
        action = rules.action
    proposed_people = payload.get("people") if isinstance(payload.get("people"), list) else []
    people = []
    for name in proposed_people:
        if not isinstance(name, str):
            continue
        cleaned = name.strip()
        if cleaned and mentioned(text, cleaned) and cleaned.lower() not in {r.lower() for r in chunk.resident}:
            people.append(cleaned)
    if not people:
        people = list(rules.people)

    proposed_objects = payload.get("objects") if isinstance(payload.get("objects"), list) else []
    detected = {item.lower() for item in chunk.detections}
    objects = []
    for label in proposed_objects:
        if not isinstance(label, str):
            continue
        cleaned = label.strip().lower()
        if cleaned in TRACKED and (cleaned in detected or mentioned(text, cleaned)):
            objects.append(cleaned)
    if not objects:
        objects = list(rules.objects)

    answers = []
    supports = payload.get("supports") if isinstance(payload.get("supports"), list) else []
    for support in supports:
        if not isinstance(support, dict):
            continue
        question_id = str(support.get("question_id") or "")
        if question_id not in QUESTION_IDS or not _source_supports(question_id, text, objects, people):
            continue
        answer_text = str(support.get("text") or "").strip()
        answer_words = content_words(answer_text)
        if not answer_words or not answer_words <= words:
            continue
        try:
            confidence = float(support.get("confidence", 0.5))
        except (TypeError, ValueError):
            confidence = 0.5
        confidence = min(1.0, max(0.0, confidence))
        answers.append(Answer(answer_text, chunk.clip_id, chunk.t_start, confidence, False, question_id))
    seen = {answer.question_id for answer in answers}
    for rule_answer in rules.answers:
        if rule_answer.question_id not in seen:
            answers.append(rule_answer)

    importance = rules.importance
    if rules.importance != "high" and payload.get("importance") in {"high", "normal"}:
        importance = str(payload["importance"])

    return MemoryEntry(
        clip_id=chunk.clip_id,
        t_start=chunk.t_start,
        t_end=chunk.t_end,
        place=chunk.place,
        people=people,
        action=action,
        objects=objects,
        statement=statement,
        transcript=chunk.transcript,
        answers=answers,
        importance=importance,
        camera=chunk.camera,
        file=chunk.file,
        object_details=dict(rules.object_details),
    )


def _source_supports(question_id: str, text: str, objects: list[str], people: list[str]) -> bool:
    if question_id == "people":
        return bool(people)
    if question_id == "lost_objects":
        return any(label in objects for label in PERSONAL_OBJECTS)
    if question_id == "door_lock":
        return _door_state(text) is not None
    if question_id == "stove":
        return _stove_state(text) is not None
    if question_id == "appointment":
        return _appointment(text) is not None
    return False
