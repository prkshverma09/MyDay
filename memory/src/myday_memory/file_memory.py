"""File one llm_input window into general memory or key memory."""

from __future__ import annotations

import json
import re
from dataclasses import replace
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
    "reading",
)
IMPORTANCE = {
    "visitor or conversation": 0.9,
    "appliance on or off": 0.9,
    "object put down or picked up": 0.8,
    "front door": 0.9,
    "door locked": 0.9,
    "unusual event": 0.95,
    "reading": 0.7,
}
TRACKED = ("keys", "glasses", "hearing aid", "phone", "handbag")
_APPLIANCE = re.compile(r"\b(stove|oven|microwave|kettle|hob|burner|ring)\b", re.IGNORECASE)
_APPLIANCE_STATE = re.compile(
    r"\b(?:turn(?:s|ed|ing)?|switch(?:es|ed|ing)?|click(?:s|ed|ing)?|"
    r"left|stays|remains|comes|goes|going|shuts|shut)\s+(?:on|off|lit)\b"
    r"|\b(?:is|was|are|were|been|still|left)\s+(?:still\s+)?(?:on|off|lit)\b",
    re.IGNORECASE,
)
_NEW_FACT = re.compile(
    r"\b(locked|unlocks|unlocked|unlocking|unlock|front door|"
    r"falls|fell|fallen|spill|spilled|dentist|doctor|optician|"
    r"hospital|appointment|broken|crying|collapsed|visitor)\b",
    re.IGNORECASE,
)
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
    "the", "this", "that", "there", "then", "she", "her", "his", "you", "and", "but",
    "margaret", "mum", "mom", "kitchen", "hallway", "living", "room", "front",
    "door", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday",
    "sunday", "january", "february", "march", "april", "june", "july", "august",
    "september", "october", "november", "december", "just", "they", "steam",
    "morning", "remember", "hello", "lovely", "please", "cosmos", "yolo",
}
_MOVE = re.compile(
    r"\b(walk|walks|walked|goes|went|enter|enters|entered)\b",
    re.IGNORECASE,
)
_PLACE_WORDS = {
    "home_kitchen": "the kitchen",
    "home_living_room": "the living room",
    "home_hallway": "the hallway",
    "home_bedroom": "the bedroom",
    "home_garden": "the garden",
    "kitchen": "the kitchen",
    "living_room": "the living room",
    "living room": "the living room",
    "hallway": "the hallway",
    "bedroom": "the bedroom",
    "garden": "the garden",
    "front_door": "the front door",
    "front door": "the front door",
}
_RAW_PLACE = re.compile(r"\bhome_[a-z0-9_]+\b", re.IGNORECASE)
_KETTLE_SWITCH = re.compile(
    r"\b(press|presses|pressed|push|pushes|pushed)\b.{0,40}\bswitch\b"
    r"|\bswitch\b.{0,40}\b(press|presses|pressed|push|pushes|pushed)\b",
    re.IGNORECASE,
)
_ACTION = re.compile(
    r"\b(fill|fills|filled|pour|pours|poured|walk|walks|walked|pick|picks|picked|"
    r"open|opens|opened|press|presses|pressed|set|sets|drop|drops|dropped|"
    r"carry|carries|carried|turn|turns|turned|flip|flips|flipped|feed|feeds|"
    r"put|puts|lock|locks|unlock|unlocks|read|reads|reading|enter|enters|"
    r"entered|run|runs|running|click|clicks|tip|tips|tipped|lift|lifts|"
    r"lifted|reach|reaches|reached|knock|knocks|knocked)\b",
    re.IGNORECASE,
)
_DETAIL = re.compile(
    r"\b(rests|resting|leans|leaning|looks|looking|waits|waiting|"
    r"stands still|empty|raised|holding the kettle in place)\b",
    re.IGNORECASE,
)
_PAST = {
    "fill": "filled",
    "fills": "filled",
    "pick": "picked",
    "picks": "picked",
    "press": "pressed",
    "presses": "pressed",
    "set": "set",
    "sets": "set",
    "drop": "dropped",
    "drops": "dropped",
    "open": "opened",
    "opens": "opened",
    "carry": "carried",
    "carries": "carried",
    "walk": "walked",
    "walks": "walked",
    "enter": "entered",
    "enters": "entered",
    "pour": "poured",
    "pours": "poured",
    "take": "took",
    "takes": "took",
    "turn": "turned",
    "turns": "turned",
    "flip": "flipped",
    "flips": "flipped",
    "lock": "locked",
    "locks": "locked",
    "unlock": "unlocked",
    "unlocks": "unlocked",
    "put": "put",
    "puts": "put",
    "read": "read",
    "reads": "read",
    "sit": "sat",
    "sits": "sat",
    "lift": "lifted",
    "lifts": "lifted",
    "reach": "reached",
    "reaches": "reached",
    "tip": "tipped",
    "tips": "tipped",
    "rest": "rested",
    "rests": "rested",
    "knock": "knocked",
    "knocks": "knocked",
}


def load_instruction() -> str:
    return INSTRUCTION_PATH.read_text(encoding="utf-8").strip()


def file_record(record: dict, llm=None) -> dict:
    """Return the first memory entry for a window. Identity fields are copied off the record."""

    return file_window(record, llm=llm)[0]


def file_window(record: dict, llm=None) -> list[dict]:
    """Return one entry per distinct event. Every entry keeps this window's video span."""

    from myday_memory.llm import memory_response

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
    entries = _from_rules_many(record, parsed)
    if proposed is not None:
        _apply_proposals(entries, memory_response(proposed)["entries"])
    _assign_ids(entries, str(record.get("id") or ""))
    return entries


def file_records(records: list[dict], llm=None) -> dict[str, list[dict]]:
    ordered: list[dict] = []
    for record in records:
        day = record_day(record)
        for entry in file_window(record, llm=llm):
            if day:
                entry["day"] = day
            ordered.append(entry)
    ordered = _collapse_reading(ordered)
    ordered = _collapse_episodes(ordered)
    general = [entry for entry in ordered if entry["store"] != "key"]
    key = [entry for entry in ordered if entry["store"] == "key"]
    return {
        "general": general,
        "key": key,
        "last_seen": last_seen_from(key),
        "entries": ordered,
    }


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
    """A run of reading a book becomes one key memory. One page turn stays general."""

    collapsed: list[dict] = []
    group: list[dict] = []

    def flush() -> None:
        if not group:
            return
        if len(group) == 1:
            entry = group[0]
            sentences = list(entry.get("_sentences") or [])
            _drop_private(entry)
            if not _page_turn_only(sentences):
                _mark_reading(entry)
            collapsed.append(entry)
        else:
            collapsed.append(_merge_reading(group))
        group.clear()

    for entry in entries:
        joins = entry.get("_reading") or (entry.get("_with_reading") and group)
        if joins and _same_reading_scene(group, entry):
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
    previous_frame = previous.get("time_frame") or {}
    frame = entry.get("time_frame") or {}
    previous_end = float(previous_frame.get("end_sec", 0))
    start = float(frame.get("start_sec", 0))
    same_span = abs(start - float(previous_frame.get("start_sec", 0))) <= 0.05 and abs(
        float(frame.get("end_sec", 0)) - previous_end
    ) <= 0.05
    return same_span or -0.05 <= start - previous_end <= 1.0


def _merge_reading(group: list[dict]) -> dict:
    first = group[0]
    frame = _span_frame(group)
    sentences: list[str] = []
    for entry in group:
        sentences.extend(entry.get("_sentences") or [])
    name = _book_name(sentences)
    statement = f"You read {name}." if name else "You read a book."
    merged = {
        "id": first["id"],
        "window_id": first.get("window_id"),
        "window_ids": _window_ids(group),
        "day": first.get("day"),
        "camera_id": first["camera_id"],
        "location": first["location"],
        "window": {"start_sec": frame["start_sec"], "end_sec": frame["end_sec"]},
        "time_frame": frame,
        "store": "key",
        "key_reason": "reading",
        "statement": statement,
        "actors": [],
        "objects": _union_objects(group),
        "appointment": None,
        "importance": IMPORTANCE["reading"],
    }
    return merged


def _mark_reading(entry: dict) -> None:
    entry["store"] = "key"
    entry["key_reason"] = "reading"
    entry["importance"] = IMPORTANCE["reading"]


def _page_turn_only(sentences: list[str]) -> bool:
    blob = " ".join(sentences)
    turning = re.search(r"\b(page|pages)\b", blob, re.IGNORECASE) and re.search(
        r"\b(flip|flips|flipped|turn|turns|turned)\b", blob, re.IGNORECASE
    )
    if not turning:
        return False
    if re.search(r"\b(read|reads|reading)\b", blob, re.IGNORECASE) and re.search(r"\bbook\b", blob, re.IGNORECASE):
        return False
    if re.search(r"\b(pick|picks|picked|open|opens)\b.{0,40}\bbook\b", blob, re.IGNORECASE):
        return False
    return _book_name(sentences) is None


def _collapse_episodes(entries: list[dict]) -> list[dict]:
    """A reading run, including sitting down and a sip, is one memory.

    Tea steps stay separate. A kettle switching on or off is its own key entry.
    """

    collapsed: list[dict] = []
    for place in _place_groups(entries):
        labels = _absorb_settling([_episode_kind(entry) for entry in place])
        start = 0
        for index in range(1, len(place) + 1):
            if index < len(place) and labels[index] == labels[start]:
                continue
            chunk = place[start:index]
            kind = labels[start]
            if len(chunk) > 1 and kind == "reading":
                collapsed.append(_merge_episode_reading(chunk))
            else:
                collapsed.extend(chunk)
            start = index
    return _collapse_same_statement(collapsed)


def _collapse_same_statement(entries: list[dict]) -> list[dict]:
    """Consecutive windows with the same sentence are one span."""

    collapsed: list[dict] = []
    for entry in entries:
        if collapsed and _same_action(collapsed[-1], entry):
            previous = collapsed[-1]
            merged = _merge_span([previous, entry], str(previous.get("statement") or ""))
            merged["store"] = previous.get("store") or "general"
            merged["key_reason"] = previous.get("key_reason")
            merged["importance"] = previous.get("importance")
            collapsed[-1] = merged
            continue
        collapsed.append(entry)
    return collapsed


def _same_action(previous: dict, entry: dict) -> bool:
    if str(previous.get("statement") or "") != str(entry.get("statement") or ""):
        return False
    if previous.get("location") != entry.get("location"):
        return False
    if previous.get("store") != entry.get("store"):
        return False
    previous_end = float((previous.get("time_frame") or previous.get("window") or {}).get("end_sec") or 0)
    start = float((entry.get("time_frame") or entry.get("window") or {}).get("start_sec") or 0)
    return -0.05 <= start - previous_end <= 1.0


def _place_groups(entries: list[dict]) -> list[list[dict]]:
    groups: list[list[dict]] = []
    for entry in entries:
        if groups and _same_reading_scene(groups[-1], entry):
            groups[-1].append(entry)
            continue
        groups.append([entry])
    return groups


def _episode_kind(entry: dict) -> str:
    text = str(entry.get("statement") or "")
    if entry.get("key_reason") == "reading" or re.search(r"\b(read|reading|book)\b", text, re.IGNORECASE):
        return "reading"
    if re.search(r"\b(kettle|poured|pouring|milk)\b", text, re.IGNORECASE) or re.search(
        r"\bmade a cup of tea\b", text, re.IGNORECASE
    ):
        return "tea"
    if entry.get("key_reason") == "appliance on or off" and re.search(r"\bkettle\b", text, re.IGNORECASE):
        return "tea"
    if re.search(r"\b(walked|walks|sat|sits|sitting|sofa|couch|enters|enter)\b", text, re.IGNORECASE):
        return "settle"
    return "other"


def _absorb_settling(labels: list[str]) -> list[str]:
    absorbed = list(labels)
    for index, label in enumerate(absorbed):
        if label != "settle":
            continue
        previous = next((absorbed[cursor] for cursor in range(index - 1, -1, -1) if absorbed[cursor] != "settle"), None)
        following = next((absorbed[cursor] for cursor in range(index + 1, len(absorbed)) if absorbed[cursor] != "settle"), None)
        if previous in {"tea", "reading"} and following in {None, previous}:
            absorbed[index] = previous
        elif following in {"tea", "reading"} and previous in {None, following}:
            absorbed[index] = following
    return absorbed


def _merge_episode_reading(group: list[dict]) -> dict:
    name = None
    for entry in group:
        found = _book_name([str(entry.get("statement") or "")])
        if found:
            name = found
            break
    statement = f"You read {name}." if name else "You read a book."
    merged = _merge_span(group, statement)
    merged["store"] = "key"
    merged["key_reason"] = "reading"
    merged["importance"] = IMPORTANCE["reading"]
    return merged


def _merge_span(group: list[dict], statement: str) -> dict:
    first = group[0]
    frame = _span_frame(group)
    return {
        "id": first.get("id") or "",
        "window_id": first.get("window_id"),
        "window_ids": _window_ids(group),
        "day": first.get("day"),
        "camera_id": first.get("camera_id") or "",
        "location": first.get("location") or "",
        "window": {"start_sec": frame["start_sec"], "end_sec": frame["end_sec"]},
        "time_frame": frame,
        "store": first.get("store") or "general",
        "key_reason": first.get("key_reason"),
        "statement": statement,
        "actors": [],
        "objects": _union_objects(group),
        "appointment": None,
        "importance": first.get("importance") or 0.4,
    }


def _span_frame(group: list[dict]) -> dict:
    first = group[0].get("time_frame") or group[0].get("window") or {}
    last = group[-1].get("time_frame") or group[-1].get("window") or {}
    frame = dict(first)
    frame["start_sec"] = first.get("start_sec", 0)
    frame["end_sec"] = last.get("end_sec", frame["start_sec"])
    sources: list[str] = []
    for entry in group:
        moment = entry.get("time_frame") or {}
        source = moment.get("source")
        if source and source not in sources:
            sources.append(source)
        for extra in moment.get("sources") or []:
            if extra not in sources:
                sources.append(extra)
    if len(sources) > 1:
        frame["sources"] = sources
        frame["source"] = sources[0]
    elif len(sources) == 1:
        frame["source"] = sources[0]
    return frame


def _drop_private(entry: dict) -> None:
    entry.pop("_reading", None)
    entry.pop("_with_reading", None)
    entry.pop("_sentences", None)


def _window_ids(group: list[dict]) -> list[str]:
    found: list[str] = []
    for entry in group:
        raw = entry.get("window_ids") or [entry.get("window_id")]
        for window_id in raw:
            if window_id and window_id not in found:
                found.append(window_id)
    return found


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


_DAY = re.compile(r"^(\d{4}-\d{2}-\d{2})")
_MEMORY_LINE = re.compile(r"^(\d{4}-\d{2}-\d{2})  (\d+):(\d{2})–(\d+):(\d{2})  (.*)$")


def record_day(record: dict) -> str | None:
    """Calendar day for a pipeline record, as YYYY-MM-DD."""

    for key in ("day", "date", "recorded_at"):
        value = record.get(key)
        if isinstance(value, str):
            match = _DAY.match(value.strip())
            if match:
                return match.group(1)
    return None


def memory_line(entry: dict) -> str:
    """The one line a person or an agent reads for this memory."""

    day = str(entry.get("day") or "undated")
    frame = entry.get("time_frame") or entry.get("window") or {}
    start = _stamp(float(frame.get("start_sec") or 0))
    end = _stamp(float(frame.get("end_sec") or 0))
    sentence = str(entry.get("statement") or "").strip()
    return f"{day}  {start}–{end}  {sentence}".rstrip()


def _stamp(seconds: float) -> str:
    total = int(round(seconds))
    return f"{total // 60}:{total % 60:02d}"


def render_memory(entries: list[dict]) -> str:
    """Plain memory file: one timestamp and sentence per line."""

    lines = [memory_line(entry) for entry in entries if memory_line(entry)]
    if not lines:
        return ""
    return "\n".join(lines) + "\n"


def read_memory(path: Path) -> list[dict]:
    """Read a memory file back. Each row has the line, the span, and the sentence."""

    rows = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        match = _MEMORY_LINE.match(line)
        if not match:
            continue
        start = int(match.group(2)) * 60 + int(match.group(3))
        end = int(match.group(4)) * 60 + int(match.group(5))
        rows.append(
            {
                "text": line,
                "day": match.group(1),
                "start_sec": start,
                "end_sec": end,
                "statement": match.group(6),
            }
        )
    return rows


def write_stores(filed: dict[str, list[dict]], directory: Path, day: str | None = None) -> list[Path]:
    """Write one folder per calendar day: text to read, JSON to cross-check."""

    directory.mkdir(parents=True, exist_ok=True)
    entries = list(filed.get("entries") or [*filed["general"], *filed["key"]])
    buckets: dict[str, list[dict]] = {}
    for entry in entries:
        day_name = str(entry.get("day") or day or "undated")
        entry["day"] = day_name
        buckets.setdefault(day_name, []).append(entry)
    written: list[Path] = []
    for day_name in sorted(buckets):
        subset = buckets[day_name]
        general = [entry for entry in subset if entry.get("store") != "key"]
        key = [entry for entry in subset if entry.get("store") == "key"]
        target = directory / day_name
        _write_day(target, general, key, last_seen_from(key))
        written.append(target)
    return written


def _write_day(directory: Path, general: list[dict], key: list[dict], last_seen: list[dict]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    _write_json(directory / "general_memory.json", [_with_text(entry) for entry in general])
    _write_json(directory / "key_moments.json", [_with_text(entry) for entry in key])
    (directory / "last-seen.json").write_text(
        json.dumps({"objects": last_seen}, indent=2) + "\n",
        encoding="utf-8",
    )
    (directory / "general_memory.txt").write_text(render_memory(general), encoding="utf-8")
    (directory / "key_moments.txt").write_text(render_memory(key), encoding="utf-8")


def load_records(path: Path) -> list[dict]:
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return []
    if raw.startswith("["):
        data = json.loads(raw)
        return data if isinstance(data, list) else [data]
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


def _from_rules_many(record: dict, parsed: LlmInput) -> list[dict]:
    slices = _event_slices(parsed.sentences)
    return [_entry_for(record, parsed, sentences, reason) for sentences, reason in slices]


def _event_slices(sentences: list[str]) -> list[tuple[list[str], str | None]]:
    slices: list[tuple[list[str], str | None]] = []
    for sentence in sentences:
        reasons = _reasons_in(sentence)
        if reasons:
            for reason in reasons:
                slices.append(([sentence], reason))
            continue
        if not _is_detail(sentence):
            slices.append(([sentence], None))
    if not slices and sentences:
        slices.append((list(sentences), _key_reason(sentences)))
    return slices


def _reasons_in(sentence: str) -> list[str]:
    """Distinct key events in one sentence, in the order they appear."""

    found: list[tuple[int, str]] = []

    def add(reason: str, match: re.Match | None) -> None:
        if match is not None:
            found.append((match.start(), reason))

    if _kettle_switch_on(sentence) or (_APPLIANCE.search(sentence) and _APPLIANCE_STATE.search(sentence)):
        add("appliance on or off", _APPLIANCE.search(sentence) or re.search(r"\bswitch\b", sentence, re.IGNORECASE))
    if _placement(sentence):
        add("object put down or picked up", _TRACKED.search(sentence))
    if _locked(sentence):
        add("door locked", _LOCK.search(sentence))
    elif _door(sentence):
        add("front door", _DOOR.search(sentence))
    if _UNUSUAL.search(sentence):
        add("unusual event", _UNUSUAL.search(sentence))
    if _is_reading([sentence]) and not _page_turn_only([sentence]):
        add("reading", re.search(r"\b(read|reads|reading|book)\b", sentence, re.IGNORECASE))
    if _visitor([sentence]) and not found:
        visible = _QUOTE.sub(" ", sentence)
        add("visitor or conversation", _NAME.search(visible) or _SPEECH.search(sentence) or _QUOTE.search(sentence))
    found.sort(key=lambda item: item[0])
    reasons: list[str] = []
    for _, reason in found:
        if reason not in reasons:
            reasons.append(reason)
    return reasons


def _kettle_switch_on(sentence: str) -> bool:
    return re.search(r"\bkettle\b", sentence, re.IGNORECASE) is not None and _KETTLE_SWITCH.search(sentence) is not None


def _is_detail(sentence: str) -> bool:
    """A pose or an object just sitting there is not its own event."""

    if _reasons_in(sentence):
        return False
    if _ACTION.search(sentence) and not _DETAIL.search(sentence):
        return False
    return True


def _entry_for(record: dict, parsed: LlmInput, sentences: list[str], reason: str | None) -> dict:
    names = _other_names(sentences)
    quotes = _speech_quotes(sentences)
    location = str(record.get("location") or parsed.location or "")
    statement = _statement(sentences, location, reason, names, quotes)
    actors = list(names)
    if not actors:
        actors = _attribute_actors(sentences)
    window = record.get("window") or {}
    start = window.get("start_sec", parsed.start_sec if parsed.start_sec is not None else 0.0)
    end = window.get("end_sec", parsed.end_sec if parsed.end_sec is not None else 0.0)
    importance = IMPORTANCE[reason] if reason else _routine_importance(sentences)
    frame = _time_frame(record, start, end)
    focused = replace(parsed, sentences=list(sentences))
    entry = {
        "id": record.get("id") or "",
        "window_id": record.get("id") or "",
        "window_ids": [record.get("id") or ""],
        "camera_id": record.get("camera_id") or parsed.camera_id,
        "location": location,
        "window": {"start_sec": frame["start_sec"], "end_sec": frame["end_sec"]},
        "time_frame": frame,
        "store": "key" if reason else "general",
        "key_reason": reason,
        "statement": statement,
        "actors": actors,
        "objects": _objects_for(focused, reason),
        "appointment": _appointment(sentences),
        "importance": importance,
        "_sentences": list(sentences),
    }
    blob = " ".join(sentences)
    if _is_reading(sentences) or _page_turn_only(sentences):
        entry["_reading"] = True
    elif re.search(r"\bsip\b", blob, re.IGNORECASE) and re.search(r"\bbook\b", blob, re.IGNORECASE):
        entry["_with_reading"] = True
    return entry


def _assign_ids(entries: list[dict], window_id: str) -> None:
    if len(entries) == 1:
        entries[0]["id"] = window_id
        entries[0]["window_id"] = window_id
        entries[0]["window_ids"] = [window_id]
        return
    for index, entry in enumerate(entries, start=1):
        entry["id"] = f"{window_id}#{index}"
        entry["window_id"] = window_id
        entry["window_ids"] = [window_id]


def _apply_proposals(entries: list[dict], proposals: list[dict]) -> None:
    if not proposals:
        return
    used: set[int] = set()
    for entry in entries:
        best = None
        best_score = -1
        for index, proposal in enumerate(proposals):
            if index in used:
                continue
            score = _proposal_score(entry, proposal)
            if score > best_score:
                best = index
                best_score = score
        if best is None:
            continue
        if best_score <= 0 and not (len(entries) == 1 and len(proposals) == 1):
            continue
        used.add(best)
        _apply_wording(entry, proposals[best])


def _proposal_score(entry: dict, proposal: dict) -> int:
    source = " ".join(entry.get("_sentences") or [])
    statement = _translate_places(str(proposal.get("statement") or ""))
    score = len(_content_words(statement) & _content_words(source))
    reason = proposal.get("key_reason")
    if reason and reason == entry.get("key_reason"):
        score += 3
    return score


def _apply_wording(entry: dict, proposed: dict) -> None:
    source = " ".join(entry.get("_sentences") or [])
    words = _content_words(source)
    statement = _translate_places(str(proposed.get("statement") or "").strip())
    if _RAW_PLACE.search(statement):
        statement = ""
    statement_words = _content_words(statement)
    if _statement_supported(statement, statement_words, source, words, entry["statement"]):
        entry["statement"] = _with_place(statement, str(entry.get("location") or ""))
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
    return _APPLIANCE.search(sentence) is not None and _APPLIANCE_STATE.search(sentence) is not None


def _statement_supported(
    statement: str,
    statement_words: set[str],
    source: str,
    words: set[str],
    rules_statement: str,
) -> bool:
    """Keep a model sentence that stays on this window. A paraphrase may add ordinary words."""

    if not statement or not statement_words:
        return False
    source_text = source.lower()
    for match in _NEW_FACT.finditer(statement):
        if match.group(1).lower() not in source_text:
            return False
    for name in _NAME.findall(statement):
        if name.lower() in _SKIP_NAMES:
            continue
        if name.lower() not in source_text:
            return False
    if _APPLIANCE.search(statement) and _APPLIANCE_STATE.search(statement):
        if not _source_has_appliance_state(source):
            return False
    if statement_words <= words or statement_words <= words | _content_words(rules_statement):
        return True
    return bool(statement_words & words)


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
            if _sentence_starts_with(visible, name) and not _person_context(sentence):
                continue
            found.append(name)
    return found


def _sentence_starts_with(sentence: str, name: str) -> bool:
    stripped = sentence.strip().lstrip("\"“'")
    return stripped.startswith(name) and (
        len(stripped) == len(name) or not stripped[len(name)].isalnum()
    )


def _person_context(sentence: str) -> bool:
    if _SPEECH.search(sentence) or _QUOTE.search(sentence):
        return True
    return re.search(
        r"\b(on the step|at the door|in the doorway|front door|a visitor|someone else|knocks|knocked)\b",
        sentence,
        re.IGNORECASE,
    ) is not None


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


def _plain_place(location: str) -> str:
    text = " ".join(location.strip().split())
    if not text:
        return ""
    key = text.lower()
    if key in _PLACE_WORDS:
        return _PLACE_WORDS[key]
    underscored = key.replace(" ", "_")
    if underscored in _PLACE_WORDS:
        return _PLACE_WORDS[underscored]
    if underscored.startswith("home_"):
        rest = underscored[len("home_"):].replace("_", " ")
        return f"the {rest}" if rest else ""
    return ""


def _with_place(statement: str, location: str) -> str:
    phrase = _plain_place(location)
    if not phrase:
        return statement
    if phrase.lower() in statement.lower():
        return statement
    bare = phrase[4:] if phrase.lower().startswith("the ") else phrase
    if re.search(rf"\b{re.escape(bare)}\b", statement, re.IGNORECASE):
        return statement
    body = statement.rstrip()
    if body.endswith("."):
        body = body[:-1]
    return f"{body} in {phrase}."


def _translate_places(statement: str) -> str:
    for token, phrase in _PLACE_WORDS.items():
        if " " in token:
            continue
        bare = phrase[4:] if phrase.lower().startswith("the ") else phrase
        statement = re.sub(
            rf"\bthe\s+{re.escape(token)}\b",
            f"the {bare}",
            statement,
            flags=re.IGNORECASE,
        )
        statement = re.sub(rf"\b{re.escape(token)}\b", phrase, statement, flags=re.IGNORECASE)
    statement = re.sub(r"\bthe\s+the\b", "the", statement, flags=re.IGNORECASE)
    return statement


def _in_place(location: str) -> str:
    phrase = _plain_place(location)
    return f" in {phrase}" if phrase else ""


def _source_has_appliance_state(source: str) -> bool:
    if _kettle_switch_on(source):
        return True
    return _APPLIANCE.search(source) is not None and _APPLIANCE_STATE.search(source) is not None


def _statement(
    sentences: list[str],
    location: str,
    reason: str | None,
    names: list[str],
    quotes: list[str],
) -> str:
    place = _in_place(location)
    if names and quotes:
        return f"{names[0]} said \"{quotes[0]}\"."
    if names:
        where = f" at the {location}" if "door" in location.lower() else place
        return f"{names[0]} was{where}."
    if reason == "reading":
        name = _book_name(sentences)
        return _with_place(f"You read {name}." if name else "You read a book.", location)
    if reason == "appliance on or off":
        blob = " ".join(sentences)
        if _kettle_switch_on(blob):
            return f"You switched the kettle on{place}."
        state = "off" if re.search(r"\boff\b", blob, re.IGNORECASE) else "on"
        name = "stove"
        match = _APPLIANCE.search(blob)
        if match:
            name = "stove" if match.group(1).lower() == "ring" else match.group(1).lower()
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
        return _with_place(_first_sentence(sentences), location)
    if any(re.search(r"\b(cat|biscuit)\b", sentence, re.IGNORECASE) for sentence in sentences):
        return _with_place("You fed the cat.", location)
    if _MOVE.search(" ".join(sentences)) and not any(
        re.search(r"\b(kettle|fill|fills|filled|pour|pours|pick|picks|set|sets)\b", sentence, re.IGNORECASE)
        for sentence in sentences
    ):
        phrase = _plain_place(location)
        if phrase:
            return f"You walked into {phrase}."
        return "You walked into another room."
    if sentences:
        return _with_place(_as_you(_first_sentence(sentences)), location)
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
    text = re.sub(
        r"^(Her (?:left |right )?hands?|Her finger|She|Margaret|The woman|A woman)\b",
        "You",
        sentence.strip(),
        count=1,
        flags=re.IGNORECASE,
    )
    match = re.match(r"^You\s+(\w+)(.*)$", text, flags=re.IGNORECASE)
    if match:
        verb = _PAST.get(match.group(1).lower(), match.group(1))
        text = f"You {verb}{match.group(2)}"
    if text and text[-1] not in ".!?":
        text += "."
    return text


def _content_words(text: str) -> set[str]:
    stop = {"the", "and", "was", "were", "with", "that", "this", "your", "you", "her", "his", "she"}
    return {word for word in re.findall(r"[a-z0-9']{4,}", text.lower()) if word not in stop}


def _video(entry: dict) -> dict:
    frame = entry.get("time_frame") or entry.get("window") or {}
    video = {
        "camera_id": entry.get("camera_id") or "",
        "start_sec": frame.get("start_sec"),
        "end_sec": frame.get("end_sec"),
    }
    if frame.get("source"):
        video["source"] = frame["source"]
    if frame.get("sources"):
        video["sources"] = list(frame["sources"])
    if frame.get("chunk") is not None:
        video["chunk"] = frame["chunk"]
    return video


def _with_text(entry: dict) -> dict:
    stored = {"text": memory_line(entry), "video": _video(entry)}
    stored.update(entry)
    stored["text"] = memory_line(entry)
    stored["video"] = _video(entry)
    return stored


def _write_json(path: Path, rows: list[dict]) -> None:
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
