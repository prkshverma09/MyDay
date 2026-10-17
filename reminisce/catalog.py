"""The ten labelled practice moments and the questions that should find them.

Practice clips are synthetic title cards, not recordings of a real person.
"""

from __future__ import annotations

SAMPLE_CLIPS: list[dict] = [
    {
        "clip_id": "clip_001",
        "question": "did I have lunch today?",
        "tags": ["lunch", "soup", "kitchen"],
        "title": "Lunch",
        "subtitle": "Soup at the kitchen table",
        "color": "0xC47B4A",
        "hour": 12,
        "minute": 30,
    },
    {
        "clip_id": "clip_002",
        "question": "when did my daughter visit?",
        "tags": ["daughter", "visitor"],
        "title": "Visitor",
        "subtitle": "Daughter came to visit",
        "color": "0x4C6B8A",
        "hour": 15,
        "minute": 10,
    },
    {
        "clip_id": "clip_003",
        "question": "when did I last go for a walk?",
        "tags": ["walk"],
        "title": "Walk",
        "subtitle": "A walk outside",
        "color": "0x3E6B5A",
        "hour": 10,
        "minute": 15,
    },
    {
        "clip_id": "clip_004",
        "question": "did I take my medication this morning?",
        "tags": ["medication"],
        "title": "Medication",
        "subtitle": "Morning medication",
        "color": "0x7A6A58",
        "hour": 8,
        "minute": 5,
    },
    {
        "clip_id": "clip_005",
        "question": "did I have breakfast?",
        "tags": ["breakfast", "tea", "toast"],
        "title": "Breakfast",
        "subtitle": "Tea and toast",
        "color": "0xC4922A",
        "hour": 8,
        "minute": 40,
    },
    {
        "clip_id": "clip_006",
        "question": "did anyone phone me today?",
        "tags": ["phone"],
        "title": "Phone call",
        "subtitle": "A phone call at home",
        "color": "0x2F6F7E",
        "hour": 11,
        "minute": 20,
    },
    {
        "clip_id": "clip_007",
        "question": "was I in the garden?",
        "tags": ["garden"],
        "title": "Garden",
        "subtitle": "Time in the garden",
        "color": "0x6E8B4A",
        "hour": 16,
        "minute": 0,
    },
    {
        "clip_id": "clip_008",
        "question": "did I watch television?",
        "tags": ["television"],
        "title": "Television",
        "subtitle": "Watching television",
        "color": "0x3C3A55",
        "hour": 19,
        "minute": 10,
    },
    {
        "clip_id": "clip_009",
        "question": "did I rest this afternoon?",
        "tags": ["rest"],
        "title": "Rest",
        "subtitle": "Resting this afternoon",
        "color": "0x6E7C99",
        "hour": 14,
        "minute": 0,
    },
    {
        "clip_id": "clip_010",
        "question": "did I open the front door?",
        "tags": ["door"],
        "title": "Front door",
        "subtitle": "Opening the front door",
        "color": "0x8A5A44",
        "hour": 9,
        "minute": 30,
    },
]

SAMPLE_IDS = {row["clip_id"] for row in SAMPLE_CLIPS}


def question_for(clip_id: str) -> str:
    for row in SAMPLE_CLIPS:
        if row["clip_id"] == clip_id:
            return row["question"]
    raise KeyError(clip_id)
