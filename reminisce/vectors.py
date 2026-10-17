"""Ranking over Cosmos description embeddings, with activity tags as a fallback."""

from __future__ import annotations

import re
from dataclasses import dataclass

from reminisce.settings import MIN_COSINE, TAG_BONUS
from reminisce.store import Clip

CLIP_ID_RE = re.compile(r"clip_\d+")


@dataclass
class Hit:
    clip: Clip
    cosine: float
    tag_match: bool
    combined: float


def cosine(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for a_value, b_value in zip(left, right):
        dot += a_value * b_value
        left_norm += a_value * a_value
        right_norm += b_value * b_value
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / ((left_norm ** 0.5) * (right_norm ** 0.5))


def tag_match(query: str, tags: list[str]) -> bool:
    haystack = query.lower()
    for tag in tags:
        if re.search(rf"\b{re.escape(tag.lower())}\b", haystack):
            return True
    return False


def extract_clip_ids(text: str) -> list[str]:
    seen: list[str] = []
    for clip_id in CLIP_ID_RE.findall(text or ""):
        if clip_id not in seen:
            seen.append(clip_id)
    return seen


def rank_clips(query: str, clips: list[Clip], query_vector: list[float]) -> list[Hit]:
    hits: list[Hit] = []
    for clip in clips:
        matched = tag_match(query, clip.activity_tags)
        score = cosine(query_vector, clip.embedding)
        bonus = TAG_BONUS if matched else 0.0
        hits.append(Hit(clip=clip, cosine=score, tag_match=matched, combined=min(1.0, score + bonus)))
    hits.sort(key=lambda hit: (hit.combined, hit.cosine), reverse=True)
    return hits


def select_hits(hits: list[Hit], limit: int = 3) -> list[Hit]:
    """Keep a clear winner, plus near ties. Drop weak matches that have no tag."""
    eligible = [hit for hit in hits if hit.tag_match or hit.cosine >= MIN_COSINE]
    if not eligible:
        return []
    best = eligible[0].combined
    close = [hit for hit in eligible if best - hit.combined <= 0.05]
    return close[:limit]
