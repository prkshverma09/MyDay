"""File one llm_input window into general memory or key memory."""

from __future__ import annotations

import json
import re
from pathlib import Path

from myday_memory.parse_input import LlmInput, parse_llm_input

INSTRUCTION_PATH = Path(__file__).resolve().parents[2] / "INSTRUCTION.md"

KEY_REASONS = (
    "visitor or conversation",
    "appliance on or off",
    "object put down or picked up",
    "front door",
    "door locked",
    "unusual event",
)
IMPORTANCE = {
    "visitor or conversation": 0.9,
    "appliance on or off": 0.9,
    "object put down or picked up": 0.8,
    "front door": 0.9,
    "door locked": 0.9,
    "unusual event": 0.95,
}
TRACKED = ("keys", "glasses", "hearing aid", "phone", "handbag")
_APPLIANCE = re.compile(r"\b(stove|oven|microwave|kettle|hob|burner|ring)\b", re.IGNORECASE)
_ON_OFF = re.compile(r"\b(on|off|lit)\b", re.IGNORECASE)
_PLACE_VERB = re.compile(
    r"\b(put|puts|set|sets|drop|drops|dropped|pick|picks|picked|left|hang|hangs|hung)\b",
    re.IGNORECASE,
)
_TRACKED = re.compile(r"\b(keys|key|glasses|hearing aid|phone|handbag)\b", re.IGNORECASE)
_DOOR = re.compile(r"\b(front door|the door)\b", re.IGNORECASE)
_DOOR_ACT = re.compile(
    r"\b(open|opens|opened|comes home|returns home|leaves home|goes out)\b",
    re.IGNORECASE,
)
_LOCK = re.compile(
    r"\b(unlocked|unlocks|unlocking|unlock|locked|locks|locking|lock)\b",
    re.IGNORECASE,
)
_UNUSUAL = re.compile(
    r"\b(fall|fell|fallen|spill|spilled|broken|distress|alert|crying|collapsed)\b",
    re.IGNORECASE,
)
_SPEECH = re.compile(r"\b(say|says|said|ask|asks|asked|calling|on the phone)\b", re.IGNORECASE)
_QUOTE = re.compile(r'"([^"]+)"|“([^”]+)”')
_WHEN = re.compile(
    r"\b((?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
    r"\s*(?:,\s*|\s+at\s+|\s+)\d{1,2}(?::\d{2})?\s*(?:am|pm)?)\b",
    re.IGNORECASE,
)
_WHAT = re.compile(r"\b(dentist|doctor|optician|hospital|appointment|gp)\b", re.IGNORECASE)
_NAME = re.compile(r"(?<!\w)([A-Z][a-z]{2,})(?![\w'])")
_SKIP_NAMES = {
    "the", "this", "that", "there", "then", "she", "her", "his", "and", "but",
    "margaret", "mum", "mom", "kitchen", "hallway", "living", "room", "front",
    "door", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday",
    "sunday", "january", "february", "march", "april", "june", "july", "august",
    "september", "october", "november", "december", "just", "they", "steam",
    "morning", "remember", "hello", "lovely", "please", "cosmos", "yolo",
}
_MOVE = re.compile(
    r"\b(walk|walks|walked|goes|went|into the|to the)\b",
    re.IGNORECASE,
)


def load_instruction() -> str:
    return INSTRUCTION_PATH.read_text(encoding="utf-8").strip()


def file_record(record: dict, llm=None) -> dict:
    """Return one memory entry. Identity fields are copied off the record."""

    text = str(record.get("llm_input") or "")
    parsed = parse_llm_input(text)
    proposed = None
    if llm is not None:
        try:
            proposed = llm.complete(text)
        except Exception:
            proposed = None
        if not isinstance(proposed, dict):
            proposed = None
    entry = _from_rules(record, parsed)
    if proposed is not None:
        _apply_wording(entry, proposed, parsed)
    return entry


def file_records(records: list[dict], llm=None) -> dict[str, list[dict]]:
    ordered: list[dict] = []
    for record in records:
        entry = file_record(record, llm=llm)
        parsed = parse_llm_input(str(record.get("llm_input") or ""))
        if entry["store"] == "general" and _is_reading(parsed.sentences):
            entry["_reading"] = True
            entry["_sentences"] = parsed.sentences
        ordered.append(entry)
    ordered = _collapse_reading(ordered)
    general = [entry for entry in ordered if entry["store"] != "key"]
    key = [entry for entry in ordered if entry["store"] == "key"]
    return {"general": general, "key": key, "last_seen": last_seen_from(key)}


def _is_reading(sentences: list[str]) -> bool:
    blob = " ".join(sentences)
    if re.search(r"\b(page|pages)\b", blob, re.IGNORECASE) and re.search(
        r"\b(flip|flips|flipped|turn|turns|turned)\b", blob, re.IGNORECASE
    ):
        return True
    if re.search(r"\b(read|reads|reading)\b", blob, re.IGNORECASE) and re.search(r"\bbook\b", blob, re.IGNORECASE):
        return True
    if re.search(r"\b(pick|picks|picked|open|opens)\b.{0,40}\bbook\b", blob, re.IGNORECASE):
        return True
    return False


def _book_name(sentences: list[str]) -> str | None:
    blob = " ".join(sentences)
    if re.search(r"\bbook\b", blob, re.IGNORECASE):
        quoted = _QUOTE.search(blob)
        if quoted:
            return (quoted.group(1) or quoted.group(2)).strip()
    named = re.search(
        r"\bbook\s+(?:called|titled|named)\s+([A-Z][\w'-]+(?:\s+[A-Z][\w'-]+)*)",
        blob,
    )
    if named:
        return named.group(1)
    titled = re.search(r"\b(?:the|a)\s+book\s+([A-Z][\w'-]+)", blob)
    if titled:
        return titled.group(1)
    return None


def _collapse_reading(entries: list[dict]) -> list[dict]:
    """A run of picking up a book and turning pages becomes one general memory."""

    collapsed: list[dict] = []
    group: list[dict] = []

    def flush() -> None:
        if not group:
            return
        if len(group) == 1:
            _drop_private(group[0])
            collapsed.append(group[0])
        else:
            collapsed.append(_merge_reading(group))
        group.clear()

    for entry in entries:
        if entry.get("_reading") and _same_reading_scene(group, entry):
            group.append(entry)
            continue
        flush()
        if entry.get("_reading"):
            group.append(entry)
        else:
            _drop_private(entry)
            collapsed.append(entry)
    flush()
    return collapsed


def _same_reading_scene(group: list[dict], entry: dict) -> bool:
    if not group:
        return True
    previous = group[-1]
    if previous.get("location") != entry.get("location"):
        return False
    return previous.get("time_frame", {}).get("source") == entry.get("time_frame", {}).get("source")


def _merge_reading(group: list[dict]) -> dict:
    first = group[0]
    last = group[-1]
    frame = dict(first["time_frame"])
    frame["start_sec"] = first["time_frame"]["start_sec"]
    frame["end_sec"] = last["time_frame"]["end_sec"]
    sentences: list[str] = []
    for entry in group:
        sentences.extend(entry.get("_sentences") or [])
    name = _book_name(sentences)
    statement = f"You read {name}." if name else "You read a book."
    merged = {
        "id": first["id"],
        "camera_id": first["camera_id"],
        "location": first["location"],
        "window": {"start_sec": frame["start_sec"], "end_sec": frame["end_sec"]},
        "time_frame": frame,
        "store": "general",
        "key_reason": None,
        "statement": statement,
        "actors": [],
        "objects": _union_objects(group),
        "appointment": None,
        "importance": 0.4,
    }
    return merged


def _drop_private(entry: dict) -> None:
    entry.pop("_reading", None)
    entry.pop("_sentences", None)


def _union_objects(group: list[dict]) -> list[dict]:
    seen: set[str] = set()
    chosen: list[dict] = []
    for entry in group:
        for obj in entry.get("objects") or []:
            label = str(obj.get("label") or "")
            if not label or label in seen:
                continue
            seen.add(label)
            chosen.append({"label": label, "position": obj.get("position") or ""})
    return chosen


def last_seen_from(key_entries: list[dict]) -> list[dict]:
    latest: dict[str, dict] = {}
    for entry in key_entries:
        if entry.get("key_reason") != "object put down or picked up":
            continue
        for obj in entry.get("objects") or []:
            canonical = _canonical_object(str(obj.get("label") or ""))
            if canonical is None:
                continue
            end = float(entry["window"]["end_sec"])
            current = latest.get(canonical)
            if current is not None and float(current["window"]["end_sec"]) > end:
                continue
            latest[canonical] = {
                "label": canonical,
                "position": obj.get("position") or "",
                "location": entry["location"],
                "id": entry["id"],
                "camera_id": entry["camera_id"],
                "window": entry["window"],
                "time_frame": entry["time_frame"],
                "statement": entry["statement"],
            }
    return [latest[label] for label in sorted(latest)]


def write_stores(filed: dict[str, list[dict]], directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    _write_jsonl(directory / "general-memory.jsonl", filed["general"])
    _write_jsonl(directory / "key-memory.jsonl", filed["key"])
    (directory / "last-seen.json").write_text(
        json.dumps({"objects": filed["last_seen"]}, indent=2) + "\n",
        encoding="utf-8",
    )


def load_records(path: Path) -> list[dict]:
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return []
    if raw.startswith("["):
        data = json.loads(raw)
        return data if isinstance(data, list) else [data]
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


def _from_rules(record: dict, parsed: LlmInput) -> dict:
    reason = _key_reason(parsed.sentences)
    names = _other_names(parsed.sentences)
    quotes = _speech_quotes(parsed.sentences)
    location = str(record.get("location") or parsed.location or "")
    statement = _statement(parsed.sentences, location, reason, names, quotes)
    actors = list(names)
    if not actors:
        actors = _attribute_actors(parsed.sentences)
    window = record.get("window") or {}
    start = window.get("start_sec", parsed.start_sec if parsed.start_sec is not None else 0.0)
    end = window.get("end_sec", parsed.end_sec if parsed.end_sec is not None else 0.0)
    importance = IMPORTANCE[reason] if reason else _routine_importance(parsed.sentences)
    frame = _time_frame(record, start, end)
    return {
        "id": record.get("id") or "",
        "camera_id": record.get("camera_id") or parsed.camera_id,
        "location": location,
        "window": {"start_sec": frame["start_sec"], "end_sec": frame["end_sec"]},
        "time_frame": frame,
        "store": "key" if reason else "general",
        "key_reason": reason,
        "statement": statement,
        "actors": actors,
        "objects": _objects_for(parsed, reason),
        "appointment": _appointment(parsed.sentences),
        "importance": importance,
    }


def _apply_wording(entry: dict, proposed: dict, parsed: LlmInput) -> None:
    source = " ".join(parsed.sentences)
    words = _content_words(source)
    statement = str(proposed.get("statement") or "").strip()
    statement_words = _content_words(statement)
    if statement_words and statement_words <= words:
        entry["statement"] = statement
    elif statement_words and statement_words <= words | _content_words(entry["statement"]):
        entry["statement"] = statement
    actors = []
    for actor in proposed.get("actors") or []:
        if not isinstance(actor, str):
            continue
        actor_words = _content_words(actor)
        if actor_words and actor_words <= words:
            actors.append(actor.strip())
    if actors:
        entry["actors"] = actors
    appointment = proposed.get("appointment")
    if isinstance(appointment, str) and appointment.strip():
        if _content_words(appointment) <= words:
            entry["appointment"] = appointment.strip()


def _time_frame(record: dict, start: float, end: float) -> dict:
    """Video span for this moment. Copied from the pipeline record, never from the model."""

    frame: dict = {"start_sec": start, "end_sec": end}
    source = record.get("source")
    if source:
        frame["source"] = source
    chunk = record.get("chunk") if isinstance(record.get("chunk"), dict) else {}
    if chunk.get("index") is not None:
        frame["chunk"] = int(chunk["index"])
    return frame


def _key_reason(sentences: list[str]) -> str | None:
    if _visitor(sentences):
        return "visitor or conversation"
    if any(_appliance(sentence) for sentence in sentences):
        return "appliance on or off"
    if any(_placement(sentence) for sentence in sentences):
        return "object put down or picked up"
    if any(_locked(sentence) for sentence in sentences):
        return "door locked"
    if any(_door(sentence) for sentence in sentences):
        return "front door"
    if any(_UNUSUAL.search(sentence) for sentence in sentences):
        return "unusual event"
    return None


def _appliance(sentence: str) -> bool:
    return _APPLIANCE.search(sentence) is not None and _ON_OFF.search(sentence) is not None


def _placement(sentence: str) -> bool:
    return _TRACKED.search(sentence) is not None and _PLACE_VERB.search(sentence) is not None


def _door(sentence: str) -> bool:
    return _DOOR.search(sentence) is not None and _DOOR_ACT.search(sentence) is not None


def _locked(sentence: str) -> bool:
    return _DOOR.search(sentence) is not None and _LOCK.search(sentence) is not None


def _visitor(sentences: list[str]) -> bool:
    if _other_names(sentences) or _speech_quotes(sentences):
        return True
    blob = " ".join(sentences)
    if re.search(r"\b(another person|a visitor|someone else|on the phone)\b", blob, re.IGNORECASE):
        return True
    if re.search(r"\b(a man|a woman|a person)\b.{0,40}\b(at the door|in the doorway|on the step)\b", blob, re.IGNORECASE):
        return True
    return False


def _other_names(sentences: list[str]) -> list[str]:
    found: list[str] = []
    for sentence in sentences:
        visible = _QUOTE.sub(" ", sentence)
        visible = re.sub(
            r"\bbook\s+(?:(?:called|titled|named)\s+)?[A-Z][\w'-]+(?:\s+[A-Z][\w'-]+)*",
            "book",
            visible,
        )
        for name in _NAME.findall(visible):
            if name.lower() in _SKIP_NAMES or name in found:
                continue
            found.append(name)
    return found


def _speech_quotes(sentences: list[str]) -> list[str]:
    quotes: list[str] = []
    for sentence in sentences:
        if not _SPEECH.search(sentence) and not _other_names([sentence]):
            continue
        for match in _QUOTE.finditer(sentence):
            quote = match.group(1) or match.group(2)
            if quote and quote not in quotes:
                quotes.append(quote)
    return quotes


def _attribute_actors(sentences: list[str]) -> list[str]:
    found: list[str] = []
    pattern = re.compile(
        r"\b(a person (?:in|wearing) [^.,]+|a man (?:in|wearing) [^.,]+|a woman (?:in|wearing) [^.,]+)",
        re.IGNORECASE,
    )
    for sentence in sentences:
        if not _visitor([sentence]):
            continue
        for match in pattern.finditer(sentence):
            text = match.group(1).strip()
            if text not in found:
                found.append(text)
    return found


def _appointment(sentences: list[str]) -> str | None:
    blob = " ".join(sentences)
    when = _WHEN.search(blob)
    if not when:
        return None
    what = _WHAT.search(blob)
    label = what.group(1).lower() if what else "appointment"
    when_text = re.sub(r"\s+", " ", when.group(1)).strip()
    return f"{label[:1].upper()}{label[1:]}, {when_text}"


def _statement(
    sentences: list[str],
    location: str,
    reason: str | None,
    names: list[str],
    quotes: list[str],
) -> str:
    place = f" in the {location}" if location else ""
    if names and quotes:
        return f"{names[0]} said \"{quotes[0]}\"."
    if names:
        where = f" at the {location}" if "door" in location.lower() else place
        return f"{names[0]} was{where}."
    if reason == "appliance on or off":
        state = "off" if any(re.search(r"\boff\b", sentence, re.IGNORECASE) for sentence in sentences) else "on"
        name = "stove"
        for sentence in sentences:
            match = _APPLIANCE.search(sentence)
            if match:
                name = "stove" if match.group(1).lower() == "ring" else match.group(1).lower()
                break
        return f"The {name} is {state}{place}."
    if reason == "object put down or picked up":
        label = _placed_label(sentences) or "it"
        return f"You put your {label} down{place}."
    if reason == "door locked":
        blob = " ".join(sentences).lower()
        if re.search(r"\bunlock", blob):
            return "You unlocked the front door."
        return "You locked the front door."
    if reason == "front door":
        blob = " ".join(sentences).lower()
        if "comes home" in blob or "returns home" in blob:
            return "You came home."
        if "leaves home" in blob or "goes out" in blob:
            return "You left home."
        return "You opened the front door."
    if reason == "unusual event":
        return _first_sentence(sentences)
    if any(re.search(r"\b(cat|biscuit)\b", sentence, re.IGNORECASE) for sentence in sentences):
        return "You fed the cat."
    if any(re.search(r"\b(tea|kettle|mug)\b", sentence, re.IGNORECASE) for sentence in sentences):
        return f"You made a cup of tea{place}."
    if _MOVE.search(" ".join(sentences)):
        return f"You walked into the {location}." if location else "You walked into another room."
    if sentences:
        return _as_you(_first_sentence(sentences))
    return "You were home."


def _placed_label(sentences: list[str]) -> str | None:
    for sentence in sentences:
        if not _placement(sentence):
            continue
        match = _TRACKED.search(sentence)
        if match:
            return _canonical_object(match.group(1))
    return None


def _canonical_object(label: str) -> str | None:
    text = label.lower()
    if text in {"key", "keys"}:
        return "keys"
    if text in TRACKED:
        return text
    return None


def _objects_for(parsed: LlmInput, reason: str | None) -> list[dict]:
    cosmos = " ".join(parsed.sentences).lower()
    chosen: list[dict] = []
    seen: set[str] = set()
    for obj in parsed.objects:
        if obj.label.lower() == "person":
            continue
        canonical = _canonical_object(obj.label)
        mentioned = obj.label.lower() in cosmos or (canonical or "") in cosmos
        if reason == "object put down or picked up" and canonical:
            mentioned = True
        if not mentioned:
            continue
        label = canonical or obj.label
        if label in seen:
            continue
        seen.add(label)
        chosen.append({"label": label, "position": obj.position})
    if reason == "object put down or picked up":
        label = _placed_label(parsed.sentences)
        if label and label not in seen:
            position = next((obj.position for obj in parsed.objects if _canonical_object(obj.label) == label), "")
            chosen.append({"label": label, "position": position})
    return chosen


def _routine_importance(sentences: list[str]) -> float:
    if _MOVE.search(" ".join(sentences)) and not any(
        re.search(r"\b(tea|kettle|cat|mug|food)\b", sentence, re.IGNORECASE) for sentence in sentences
    ):
        return 0.15
    return 0.4


def _first_sentence(sentences: list[str]) -> str:
    sentence = sentences[0].strip()
    if sentence and sentence[-1] not in ".!?":
        sentence += "."
    return sentence


def _as_you(sentence: str) -> str:
    text = re.sub(r"^(Her hands|Her hand|She|Margaret)\b", "You", sentence, count=1, flags=re.IGNORECASE)
    if text == sentence and not text.lower().startswith("you"):
        return sentence
    return text[:1].upper() + text[1:]


def _content_words(text: str) -> set[str]:
    stop = {"the", "and", "was", "were", "with", "that", "this", "your", "you", "her", "his", "she"}
    return {word for word in re.findall(r"[a-z0-9']{4,}", text.lower()) if word not in stop}


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    body = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    path.write_text(body, encoding="utf-8")
