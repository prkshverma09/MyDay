import json
from pathlib import Path

from myday_memory.extract import extract_entry, ground_llm_payload
from myday_memory.models import ChunkText
from myday_memory.store import build_store, load_chunks

MORNING = Path(__file__).parents[1] / "examples" / "morning.json"


def test_morning_demo_builds_the_three_stores():
    store = build_store(load_chunks(MORNING))
    by_id = {entry.clip_id: entry for entry in store.entries}

    sarah = by_id["door-0905"]
    assert sarah.people == ["Sarah"]
    assert "dentist" in sarah.statement.lower()
    assert {answer.question_id for answer in sarah.answers} == {"people", "appointment"}
    assert all(answer.abstain is False for answer in sarah.answers)

    keys = by_id["kitchen-0912"]
    assert keys.objects == ["keys"]
    assert keys.statement == "The keys are in the blue bowl."
    assert keys.importance == "high"

    seen = {row.object: row for row in store.last_seen.values()}
    assert seen["keys"].detail == "blue bowl"
    assert seen["keys"].place == "kitchen"
    assert seen["keys"].clip_id == "kitchen-0912"
    assert seen["cat"].detail == "sofa"
    assert seen["cat"].prep == "on"

    day = store.days["2026-10-17"]
    assert day["unrecorded"] == ["door_lock"]
    assert day["questions"]["people"]["names"] == ["Sarah"]
    assert day["questions"]["lost_objects"]["objects"] == ["keys"]
    assert day["questions"]["stove"]["state"] == "on"
    assert "Thursday at 10" in day["questions"]["appointment"]["text"]
    assert day["clip_count"] == 4


def test_later_sighting_replaces_last_seen_even_if_it_arrives_first():
    later = ChunkText(
        clip_id="later",
        camera="hall",
        place="hall",
        t_start="2026-10-17T15:00:00",
        t_end="2026-10-17T15:00:10",
        file="hall.mp4",
        caption="The glasses are on the side table.",
    )
    earlier = ChunkText(
        clip_id="earlier",
        camera="bedroom",
        place="bedroom",
        t_start="2026-10-17T08:00:00",
        t_end="2026-10-17T08:00:10",
        file="bedroom.mp4",
        caption="The glasses are on the dresser.",
    )
    store = build_store([later, earlier])
    assert store.last_seen["glasses"].clip_id == "later"
    assert store.last_seen["glasses"].detail == "side table"


def test_empty_clip_is_coverage_only():
    chunk = ChunkText(
        clip_id="quiet",
        camera="kitchen",
        place="kitchen",
        t_start="2026-10-17T12:00:00",
        t_end="2026-10-17T12:00:10",
        file="quiet.mp4",
    )
    entry = extract_entry(chunk)
    assert entry.statement == "Nothing notable was described."
    assert entry.answers == []
    store = build_store([chunk])
    assert store.days["2026-10-17"]["unrecorded"] == [
        "people",
        "lost_objects",
        "door_lock",
        "stove",
        "appointment",
    ]


def test_hearing_aid_and_locked_door():
    chunk = ChunkText(
        clip_id="bath",
        camera="bathroom",
        place="bathroom",
        t_start="2026-10-17T07:30:00",
        t_end="2026-10-17T07:30:20",
        file="bath.mp4",
        caption="Margaret locks the door, then sets her hearing aid on the shelf.",
    )
    entry = extract_entry(chunk)
    assert "hearing aid" in entry.objects
    assert entry.object_details["hearing aid"]["where"] == "shelf"
    assert any(answer.question_id == "door_lock" and "locked" in answer.text for answer in entry.answers)


def test_dont_is_not_a_person():
    chunk = ChunkText(
        clip_id="talk",
        camera="door",
        place="front door",
        t_start="2026-10-17T09:05:00",
        t_end="2026-10-17T09:05:10",
        file="door.mp4",
        transcript="Don't forget the dentist on Thursday at 10.",
    )
    assert extract_entry(chunk).people == []


class _Scripted:
    def __init__(self, payload):
        self.payload = payload

    def structure(self, chunk):
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload


def test_llm_wording_is_kept_when_the_source_supports_it():
    chunk = load_chunks(MORNING)[1]
    payload = {
        "people": [],
        "action": "set the keys down",
        "objects": ["keys"],
        "statement": "Margaret put the keys in the blue bowl.",
        "importance": "normal",
        "supports": [
            {
                "question_id": "lost_objects",
                "text": "The keys are in the blue bowl.",
                "confidence": 0.95,
            }
        ],
    }
    store = build_store([chunk], llm=_Scripted(payload))
    entry = store.entries[0]
    assert entry.statement == "Margaret put the keys in the blue bowl."
    assert entry.importance == "high"
    assert entry.clip_id == "kitchen-0912"


def test_llm_cannot_invent_a_locked_door():
    chunk = load_chunks(MORNING)[0]
    payload = {
        "people": ["Sarah", "Nobody"],
        "action": "locked up",
        "objects": ["keys"],
        "statement": "Sarah locked the front door.",
        "importance": "high",
        "supports": [
            {"question_id": "door_lock", "text": "The door was locked.", "confidence": 0.99}
        ],
    }
    rules = extract_entry(chunk)
    grounded = ground_llm_payload(payload, chunk, rules)
    assert grounded is not None
    assert "Nobody" not in grounded.people
    assert "keys" not in grounded.objects
    assert "locked" not in grounded.statement.lower()
    assert all(answer.question_id != "door_lock" for answer in grounded.answers)
    assert any(answer.question_id == "people" for answer in grounded.answers)


MARGARET = Path(__file__).parents[1] / "examples" / "margaret-day.json"


def test_margaret_day_is_saved_with_key_moments_highlighted():
    store = build_store(load_chunks(MARGARET))
    by_id = {entry.clip_id: entry for entry in store.entries}
    highlighted = [entry.clip_id for entry in store.entries if entry.key_moment]

    assert highlighted == [
        "keys-0815",
        "sarah-0905",
        "glasses-1040",
        "stove-on-1205",
        "stove-off-1207",
        "evening-1820",
    ]
    assert by_id["tea-0802"].key_moment is False
    assert by_id["cat-0920"].people == []
    assert by_id["keys-0815"].statement == "You put your keys in the blue bowl in the hallway at 08:15."
    assert by_id["keys-0815"].highlights[0].startswith("0:05-0:10")
    assert store.last_seen["keys"].detail == "blue ceramic bowl"
    assert store.last_seen["keys"].place == "hallway"
    assert store.last_seen["glasses"].detail == "counter"
    assert store.last_seen["glasses"].clip_id == "glasses-1040"
    assert "Biscuit" not in store.last_seen

    day = store.days["2026-10-17"]
    assert day["unrecorded"] == ["door_lock"]
    assert day["questions"]["people"]["names"] == ["Sarah"]
    assert day["questions"]["stove"]["state"] == "off"
    assert day["questions"]["stove"]["clip_ids"] == ["stove-on-1205", "stove-off-1207"]
    assert day["questions"]["appointment"]["text"] == "Dentist on Thursday, 10am, Sarah driving."
    assert by_id["sarah-0905"].to_dict()["summary"] == "Sarah visited at 09:05."


def test_llm_failure_falls_back_to_rules(tmp_path):
    store = build_store(load_chunks(MORNING), llm=_Scripted(RuntimeError("down")))
    assert store.entries[1].statement == "The keys are in the blue bowl."
    store.write(tmp_path)
    written = json.loads((tmp_path / "last_seen.json").read_text(encoding="utf-8"))
    labels = [row["object"] for row in written["objects"]]
    assert labels == ["cat", "keys"]
