from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from myday_memory.extract import extract_entry, ground_llm_payload
from myday_memory.models import QUESTION_IDS, ChunkText, LastSeen, MemoryEntry, day_key


@dataclass
class MemoryStore:
    entries: list[MemoryEntry] = field(default_factory=list)
    last_seen: dict[str, LastSeen] = field(default_factory=dict)
    days: dict[str, dict] = field(default_factory=dict)

    def add(self, entry: MemoryEntry) -> None:
        self.entries.append(entry)
        self._update_last_seen(entry)
        self._update_coverage(entry)

    def _update_last_seen(self, entry: MemoryEntry) -> None:
        for label in entry.objects:
            detail = entry.object_details.get(label, {})
            current = self.last_seen.get(label)
            if current is not None and current.t_end > entry.t_end:
                continue
            where = detail.get("where", "")
            prep = detail.get("prep", "")
            if label in {"keys", "glasses"}:
                verb = "are"
            else:
                verb = "is"
            if where:
                statement = f"The {label} {verb} {prep} the {where}."
            else:
                statement = entry.statement
            self.last_seen[label] = LastSeen(
                object=label,
                place=entry.place,
                detail=where,
                prep=prep,
                clip_id=entry.clip_id,
                t_start=entry.t_start,
                t_end=entry.t_end,
                camera=entry.camera,
                statement=statement,
                file=entry.file,
            )

    def _update_coverage(self, entry: MemoryEntry) -> None:
        day = day_key(entry.t_start)
        bucket = self.days.setdefault(
            day,
            {
                "day": day,
                "clip_count": 0,
                "places": [],
                "cameras": [],
                "span_start": entry.t_start,
                "span_end": entry.t_end,
                "intervals": [],
                "questions": {
                    question_id: {"recorded": False, "clip_ids": []} for question_id in QUESTION_IDS
                },
            },
        )
        bucket["clip_count"] += 1
        if entry.place and entry.place not in bucket["places"]:
            bucket["places"].append(entry.place)
        if entry.camera and entry.camera not in bucket["cameras"]:
            bucket["cameras"].append(entry.camera)
        if entry.t_start and (not bucket["span_start"] or entry.t_start < bucket["span_start"]):
            bucket["span_start"] = entry.t_start
        if entry.t_end and entry.t_end > bucket["span_end"]:
            bucket["span_end"] = entry.t_end
        bucket["intervals"].append(
            {
                "clip_id": entry.clip_id,
                "camera": entry.camera,
                "place": entry.place,
                "t_start": entry.t_start,
                "t_end": entry.t_end,
                "file": entry.file,
            }
        )
        for answer in entry.answers:
            question = bucket["questions"][answer.question_id]
            question["recorded"] = True
            if entry.clip_id not in question["clip_ids"]:
                question["clip_ids"].append(entry.clip_id)
            if answer.question_id == "people":
                names = question.setdefault("names", [])
                for name in entry.people:
                    if name not in names:
                        names.append(name)
            elif answer.question_id == "lost_objects":
                seen = question.setdefault("objects", [])
                for label in entry.objects:
                    if label in {"keys", "glasses", "hearing aid"} and label not in seen:
                        seen.append(label)
            elif answer.question_id == "stove":
                question["state"] = "off" if "off" in answer.text else "on"
                question["text"] = answer.text
            elif answer.question_id == "appointment":
                question["text"] = answer.text
            elif answer.question_id == "door_lock":
                question["state"] = "unlocked" if "unlocked" in answer.text else "locked"
                question["text"] = answer.text
        bucket["unrecorded"] = [
            question_id
            for question_id in QUESTION_IDS
            if not bucket["questions"][question_id]["recorded"]
        ]

    def to_dict(self) -> dict:
        days = [self.days[key] for key in sorted(self.days)]
        return {
            "memory_entries": [entry.to_dict() for entry in self.entries],
            "last_seen": [self.last_seen[key].to_dict() for key in sorted(self.last_seen)],
            "coverage": {"days": days},
        }

    def write(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        payload = self.to_dict()
        _write_json(directory / "memory_entries.json", payload["memory_entries"])
        _write_json(directory / "last_seen.json", {"objects": payload["last_seen"]})
        _write_json(directory / "coverage.json", payload["coverage"])
        _write_json(
            directory / "key_moments.json",
            [entry for entry in payload["memory_entries"] if entry.get("key_moment")],
        )


def build_store(chunks: list[ChunkText], llm=None) -> MemoryStore:
    store = MemoryStore()
    for chunk in chunks:
        rules = extract_entry(chunk)
        entry = rules
        if llm is not None:
            try:
                payload = llm.structure(chunk)
            except Exception:
                payload = None
            if isinstance(payload, dict):
                grounded = ground_llm_payload(payload, chunk, rules)
                if grounded is not None:
                    entry = grounded
        store.add(entry)
    return store


def load_chunks(path: Path) -> list[ChunkText]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    rows = raw["chunks"] if isinstance(raw, dict) else raw
    return [ChunkText.from_dict(row) for row in rows]


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
