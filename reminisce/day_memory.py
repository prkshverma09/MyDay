"""Read a day's filed key moments from data/memory."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from reminisce.settings import ROOT

MEMORY_DIR = ROOT / "data" / "memory"


def load_key_moments(day: str | None = None, directory: Path | None = None) -> list[dict]:
    """Key moments for one calendar day, in the order they were filed.

    Real filings live in ``data/memory/<YYYY-MM-DD>/``. A missing day is an empty list.
    """

    folder = (directory or MEMORY_DIR) / (day or date.today().isoformat())
    path = folder / "key_moments.json"
    if path.is_file():
        return _from_json(path)
    return _from_text(folder / "key_moments.txt")


def _from_json(path: Path) -> list[dict]:
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(rows, list):
        return []
    moments = []
    for row in rows:
        if isinstance(row, dict):
            moment = _moment(row.get("statement"), row.get("text"), row.get("day"), row.get("location"))
            if moment is not None:
                moments.append(moment)
    return moments


def _from_text(path: Path) -> list[dict]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    moments = []
    for line in lines:
        day, statement = _split_line(line)
        moment = _moment(statement, line, day, "")
        if moment is not None:
            moments.append(moment)
    return moments


def _moment(statement, text, day, location) -> dict | None:
    sentence = str(statement or "").strip()
    if not sentence:
        _day, sentence = _split_line(str(text or ""))
        day = day or _day
    if not sentence:
        return None
    return {
        "statement": sentence,
        "day": str(day or "").strip(),
        "location": str(location or "").strip(),
    }


def _split_line(line: str) -> tuple[str, str]:
    text = line.strip()
    if "  " not in text:
        return "", text
    day, sentence = text.split("  ", 1)
    return day.strip(), sentence.strip()
