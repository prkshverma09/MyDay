"""The pre-loaded demo video: demo_data/IMG_2517.mp4 and its 10-second Cosmos and YOLO annotations.

Each annotated chunk becomes one searchable moment (a Clip with a start time inside the video).
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timedelta

from reminisce.models import ModelError, embed_text
from reminisce.settings import JOURNAL_DIR, ROOT
from reminisce.store import Clip

DEMO_VIDEO = ROOT / "demo_data" / "IMG_2517.mp4"
DEMO_ANNOTATIONS = ROOT / "demo_data" / "annotations.jsonl"
EMBED_CACHE = JOURNAL_DIR / "demo_embeddings.json"
MIN_OBJECT_HITS = 3
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"[a-z0-9]+")
_FILLER = frozenset(
    "a an and are at did do does for from had has have i in is it me my of on or the this to was were what "
    "when where which who with you your today put".split()
)


def demo_name() -> str:
    return DEMO_VIDEO.stem


def recorded_base() -> datetime:
    """When the demo counts as recorded: the video file's time, so it always belongs to the day it was loaded."""
    try:
        return datetime.fromtimestamp(DEMO_VIDEO.stat().st_mtime).astimezone().replace(microsecond=0)
    except OSError:
        return datetime.now().astimezone().replace(microsecond=0)


def _rows() -> list[dict]:
    if not DEMO_ANNOTATIONS.is_file():
        return []
    rows = []
    for line in DEMO_ANNOTATIONS.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return sorted(rows, key=lambda row: float(row["chunk"]["start_sec"]))


def _load_cache() -> dict[str, list[float]]:
    try:
        return json.loads(EMBED_CACHE.read_text())
    except (OSError, ValueError):
        return {}


def _embedding(text: str, cache: dict[str, list[float]]) -> list[float]:
    key = hashlib.sha1(text.encode()).hexdigest()
    if key not in cache:
        try:
            cache[key], _model = embed_text(text)
        except ModelError:
            return []
        EMBED_CACHE.parent.mkdir(parents=True, exist_ok=True)
        EMBED_CACHE.write_text(json.dumps(cache))
    return cache[key]


def demo_clips(owner_id: str = "user_123", *, embed: bool = True) -> list[Clip]:
    """One Clip per annotated chunk of the demo video, in time order."""
    if not DEMO_VIDEO.is_file():
        return []
    base = recorded_base()
    cache = _load_cache() if embed else {}
    clips = []
    for row in _rows():
        chunk = row["chunk"]
        start = float(chunk["start_sec"])
        end = float(chunk["end_sec"])
        text = " ".join(" ".join(row.get("cosmos", {}).get("sentences") or []).split())
        if not text:
            continue
        objects = [
            item["label"]
            for item in (row.get("yolo") or {}).get("objects") or []
            if int(item.get("hits") or 0) >= MIN_OBJECT_HITS and item["label"] != "person"
        ]
        clip_id = f"clip_{demo_name().removeprefix('IMG_')}_{int(start):03d}"
        clips.append(
            Clip(
                clip_id=clip_id,
                owner_id=owner_id,
                recorded_at=(base + timedelta(seconds=start)).isoformat(),
                duration_s=round(end - start, 2),
                cosmos_description=text,
                activity_tags=objects,
                embedding_id=f"emb_{clip_id}",
                embedding=_embedding(text, cache) if embed else [],
                source_citation=clip_id,
                media_relpath=str(DEMO_VIDEO.relative_to(ROOT)),
                start_sec=start,
            )
        )
    return clips


def is_demo(clip: Clip) -> bool:
    return clip.media_relpath == str(DEMO_VIDEO.relative_to(ROOT))


def span_label(clip: Clip) -> str:
    def clock(seconds: float) -> str:
        whole = int(max(0.0, seconds))
        return f"{whole // 60}:{whole % 60:02d}"

    return f"{clock(clip.start_sec)}–{clock(clip.start_sec + clip.duration_s)}"


def _words(text: str) -> set[str]:
    return {word for word in _WORD_RE.findall((text or "").lower()) if len(word) >= 3 and word not in _FILLER}


def _related(word: str, other: str) -> bool:
    if word == other:
        return True
    if min(len(word), len(other)) < 4:
        return False
    return word.startswith(other) or other.startswith(word) or word in other or other in word


def key_sentence(clip: Clip, question: str) -> str:
    """The one sentence of the chunk's own description that best answers the question.

    Sentences where the person acts ("I place…", "places the can…") win ties, because
    questions are usually about what they did.
    """
    sentences = [part.strip() for part in _SENTENCE_RE.split(clip.cosmos_description) if part.strip()]
    if not sentences:
        return clip.cosmos_description
    wanted = _words(question)
    asks_where = bool(re.search(r"\b(put|leave|left|place|placed|where)\b", question.lower()))

    def score(sentence: str) -> tuple[int, int, int]:
        lowered = sentence.lower()
        have = _words(sentence)
        overlap = sum(1 for word in wanted if any(_related(word, other) for other in have))
        placing = 1 if asks_where and re.search(r"\b(plac(e|es|ed|ing)|put|puts|left|sets? down)\b", lowered) else 0
        acting = 1 if re.search(r"\b(i|places?|put|puts|picks?|holds?|reach(es)?|walks?|takes?)\b", lowered) else 0
        return overlap, placing, acting

    best = max(sentences, key=score)
    return best if score(best)[0] else sentences[0]
