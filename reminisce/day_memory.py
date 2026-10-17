"""Read a day's filed key moments from data/memory."""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path

from reminisce.settings import ROOT

MEMORY_DIR = ROOT / "data" / "memory"


def filed_days(directory: Path | None = None) -> list[str]:
    """Calendar days that already have a memory folder, oldest first."""

    root = directory or MEMORY_DIR
    if not root.is_dir():
        return []
    return sorted(
        path.name
        for path in root.iterdir()
        if path.is_dir() and re.fullmatch(r"\d{4}-\d{2}-\d{2}", path.name)
    )


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
    folder_day = path.parent.name
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", folder_day):
        folder_day = ""
    moments = []
    for line in lines:
        day, _span, statement = _split_line(line)
        moment = _moment(statement, line, day or folder_day, "")
        if moment is not None:
            moments.append(moment)
    return moments


def _moment(statement, text, day, location) -> dict | None:
    parsed_day, span, parsed_sentence = _split_line(str(text or ""))
    sentence = str(statement or "").strip() or parsed_sentence
    if not sentence:
        return None
    return {
        "statement": sentence,
        "day": str(day or parsed_day or "").strip(),
        "location": str(location or "").strip(),
        "span": span,
    }


_DAY_PREFIX = re.compile(r"^(\d{4}-\d{2}-\d{2})\s+")
_SPAN_PREFIX = re.compile(r"^(\d+:\d{2}–\d+:\d{2})\s+")


def _split_line(line: str) -> tuple[str, str, str]:
    text = line.strip()
    day = ""
    span = ""
    day_match = _DAY_PREFIX.match(text)
    if day_match:
        day = day_match.group(1)
        text = text[day_match.end():].strip()
    span_match = _SPAN_PREFIX.match(text)
    if span_match:
        span = span_match.group(1)
        text = text[span_match.end():].strip()
    return day, span, text
