import json
from pathlib import Path

from myday_memory.file_memory import file_record, file_records, load_records, read_memory, write_stores
from myday_memory.parse_input import parse_llm_input

HOME = Path(__file__).parents[1] / "examples" / "home-windows.jsonl"


class _Scripted:
    def __init__(self, payload):
        self.payload = payload

    def complete(self, llm_input):
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload


def test_home_windows_split_into_general_and_key(tmp_path):
    filed = file_records(load_records(HOME))
    general = {entry["id"]: entry for entry in filed["general"]}
    key = {entry["id"]: entry for entry in filed["key"]}

    assert set(general) == {"home_0001", "home_0004", "home_0006"}
    assert set(key) == {"home_0002", "home_0003", "home_0005"}
    assert general["home_0001"]["statement"] == "You made a cup of tea in the kitchen."
    assert general["home_0001"]["importance"] == 0.4
    assert general["home_0004"]["statement"] == "You walked into the living room."
    assert general["home_0004"]["importance"] == 0.15
    assert key["home_0002"]["key_reason"] == "object put down or picked up"
    assert key["home_0002"]["objects"] == [{"label": "keys", "position": "center/near"}]
    assert key["home_0003"]["actors"] == ["Sarah"]
    assert key["home_0003"]["appointment"] == "Dentist, Thursday at 10am"
    assert "Thursday at 10am" in key["home_0003"]["statement"]
    assert key["home_0005"]["key_reason"] == "appliance on or off"
    assert key["home_0005"]["window"] == {"start_sec": 20.0, "end_sec": 25.0}
    assert key["home_0002"]["time_frame"] == {"start_sec": 5.0, "end_sec": 10.0}
    assert general["home_0004"]["time_frame"]["start_sec"] == 10.0

    seen = {row["label"]: row for row in filed["last_seen"]}
    assert seen["keys"]["id"] == "home_0002"
    assert seen["keys"]["location"] == "hallway"
    assert seen["keys"]["time_frame"] == {"start_sec": 5.0, "end_sec": 10.0}

    write_stores(filed, tmp_path)
    assert (tmp_path / "general-memory.jsonl").read_text(encoding="utf-8").count("\n") == 3
    assert (tmp_path / "key-memory.jsonl").read_text(encoding="utf-8").count("\n") == 3
    general_text = (tmp_path / "general_memory.txt").read_text(encoding="utf-8")
    key_text = (tmp_path / "key_moments.txt").read_text(encoding="utf-8")
    assert general_text.splitlines()[0] == "0:00–0:05  You made a cup of tea in the kitchen."
    assert "0:05–0:10  You put your keys down in the hallway." in key_text.splitlines()
    assert "keys" not in general_text
    saved = json.loads((tmp_path / "general-memory.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert next(iter(saved)) == "text"
    assert saved["text"] == general_text.splitlines()[0]
    assert saved["video"]["camera_id"] == "pov"
    assert saved["video"]["start_sec"] == 0.0
    assert saved["video"]["end_sec"] == 5.0
    assert saved["time_frame"]["start_sec"] == 0.0
    assert read_memory(tmp_path / "general_memory.txt")[0]["statement"] == "You made a cup of tea in the kitchen."


def test_yolo_person_count_is_not_a_visitor():
    record = {
        "id": "ann_0001",
        "camera_id": "pie_cam-3",
        "location": "toronto",
        "source": "s3://archive/segment.mp4",
        "chunk": {"index": 3, "start_sec": 0.0, "end_sec": 5.0},
        "window": {"start_sec": 0.0, "end_sec": 5.0},
        "llm_input": (
            "Time window: 0.0-5.0s\n"
            "Camera: pie_cam-3\n"
            "Location: toronto\n"
            "Cosmos:\n"
            "- A blue minivan is parked on the street, facing away from the camera.\n"
            "YOLO:\n"
            "- car count=20 confidence=0.94 position=center/near\n"
            "- person count=2 confidence=0.71 position=right/far\n"
        ),
    }
    entry = file_record(record)
    assert entry["store"] == "general"
    assert entry["key_reason"] is None
    assert entry["actors"] == []
    assert entry["id"] == "ann_0001"
    assert entry["window"] == {"start_sec": 0.0, "end_sec": 5.0}
    assert entry["time_frame"] == {
        "start_sec": 0.0,
        "end_sec": 5.0,
        "source": "s3://archive/segment.mp4",
        "chunk": 3,
    }


def test_quoted_sign_is_not_speech():
    parsed = parse_llm_input(
        'Cosmos:\n- A white building labeled "Bank America" is on the left.\nYOLO:\n- none\n'
    )
    entry = file_record(
        {
            "id": "sign",
            "camera_id": "street",
            "location": "san_francisco",
            "window": {"start_sec": 0.0, "end_sec": 5.0},
            "llm_input": parsed.text,
        }
    )
    assert entry["store"] == "general"
    assert entry["appointment"] is None


def test_model_wording_is_kept_and_identity_is_not():
    record = load_records(HOME)[1]
    payload = {
        "id": "nope",
        "store": "general",
        "key_reason": None,
        "statement": "Her hand drops the keys into a blue bowl.",
        "actors": ["Nobody"],
        "objects": [{"label": "truck", "position": "left/far"}],
        "appointment": "Dentist, Thursday at 10am",
        "importance": 0.1,
        "window": {"start_sec": 1, "end_sec": 2},
    }
    entry = file_record(record, llm=_Scripted(payload))
    assert entry["id"] == "home_0002"
    assert entry["window"] == {"start_sec": 5.0, "end_sec": 10.0}
    assert entry["time_frame"]["start_sec"] == 5.0
    assert entry["time_frame"]["end_sec"] == 10.0
    assert entry["store"] == "key"
    assert entry["statement"] == "Her hand drops the keys into a blue bowl."
    assert entry["actors"] == []
    assert entry["appointment"] is None
    assert entry["objects"] == [{"label": "keys", "position": "center/near"}]


def test_model_cannot_invent_a_locked_door():
    record = load_records(HOME)[0]
    payload = {
        "store": "key",
        "key_reason": "front door",
        "statement": "You locked the front door.",
        "actors": [],
        "importance": 0.99,
    }
    entry = file_record(record, llm=_Scripted(payload))
    assert entry["store"] == "general"
    assert "locked" not in entry["statement"].lower()


def _clip(clip_id, start, end, sentence, location="living room"):
    return {
        "id": clip_id,
        "camera_id": "pov",
        "location": location,
        "source": "day.mp4",
        "window": {"start_sec": start, "end_sec": end},
        "llm_input": (
            f"Time window: {start}-{end}s\n"
            "Camera: pov\n"
            f"Location: {location}\n"
            "Cosmos:\n"
            f"- {sentence}\n"
            "YOLO:\n"
            "- book count=2 confidence=0.70 position=center/near\n"
        ),
    }


def test_locking_the_door_is_key():
    locked = file_record(_clip("lock", 0, 5, "Her hand locks the front door.", "front door"))
    unlocked = file_record(_clip("unlock", 5, 10, "Her hand unlocks the front door.", "front door"))
    assert locked["store"] == "key"
    assert locked["key_reason"] == "door locked"
    assert locked["statement"] == "You locked the front door."
    assert unlocked["statement"] == "You unlocked the front door."
    assert locked["importance"] == 0.9


def test_page_flips_collapse_to_one_reading():
    records = [
        _clip("read-1", 0, 5, "Her hand picks up the book Bluebirds."),
        _clip("read-2", 5, 10, "She flips a page."),
        _clip("read-3", 10, 15, "She flips a page."),
        _clip("read-4", 15, 20, "She turns a page."),
    ]
    filed = file_records(records)
    assert filed["general"] == []
    assert len(filed["key"]) == 1
    entry = filed["key"][0]
    assert entry["statement"] == "You read Bluebirds."
    assert entry["store"] == "key"
    assert entry["key_reason"] == "reading"
    assert entry["time_frame"]["start_sec"] == 0
    assert entry["time_frame"]["end_sec"] == 20
    assert entry["time_frame"]["source"] == "day.mp4"
    assert "_reading" not in entry


def test_one_page_turn_stays_one_moment():
    filed = file_records([_clip("page", 0, 5, "She flips a page.")])
    assert filed["key"] == []
    assert len(filed["general"]) == 1
    assert filed["general"][0]["statement"] != "You read a book."
    assert filed["general"][0]["time_frame"]["end_sec"] == 5


def test_a_visitor_breaks_the_reading_run():
    records = [
        _clip("read-1", 0, 5, "Her hand picks up the book Bluebirds."),
        _clip("read-2", 5, 10, "She flips a page."),
        _clip("visit", 10, 15, 'Sarah says "hello."', "front door"),
        _clip("read-3", 15, 20, "She flips a page."),
        _clip("read-4", 20, 25, "She flips a page."),
    ]
    filed = file_records(records)
    assert filed["general"] == []
    assert [entry["statement"] for entry in filed["key"]] == [
        "You read Bluebirds.",
        'Sarah said "hello.".',
        "You read a book.",
    ]
    assert filed["key"][1]["actors"] == ["Sarah"]
    assert filed["key"][0]["key_reason"] == "reading"
    assert filed["key"][2]["key_reason"] == "reading"


def test_kettle_resting_on_its_base_is_general():
    entry = file_record(_clip("base", 0, 5, "She sets the kettle on its base on the counter.", "kitchen"))
    assert entry["store"] == "general"
    assert entry["key_reason"] is None


def test_a_hand_on_the_kettle_is_not_the_kettle_switching_on():
    entry = file_record(_clip("lid", 0, 5, "Her left hand rests on the kettle lid.", "kitchen"))
    assert entry["store"] == "general"
    assert entry["key_reason"] is None


def test_model_cannot_invent_the_kettle_switching_on():
    entry = file_record(
        _clip("switch", 0, 5, "Her finger presses the switch at the top of the kettle.", "kitchen"),
        llm=_Scripted({"statement": "You turned on the kettle.", "store": "key", "key_reason": "appliance on or off"}),
    )
    assert entry["store"] == "general"
    assert "turned on" not in entry["statement"].lower()


def test_kettle_clicks_off_is_key():
    entry = file_record(
        _clip("off", 0, 5, "The kettle clicks off, steam rising from its spout.", "kitchen")
    )
    assert entry["store"] == "key"
    assert entry["key_reason"] == "appliance on or off"
    assert entry["statement"] == "The kettle is off in the kitchen."


def test_sentence_opening_is_not_a_visitor():
    water = file_record(_clip("water", 0, 5, "Water runs from the tap into the kettle.", "kitchen"))
    only = file_record(_clip("only", 5, 10, "Only her eyes move along the page."))
    assert water["store"] == "general"
    assert water["actors"] == []
    assert only["store"] == "general"
    assert only["actors"] == []


def test_model_paraphrase_is_kept_when_it_adds_no_fact():
    record = load_records(HOME)[0]
    entry = file_record(
        record,
        llm=_Scripted(
            {
                "store": "key",
                "statement": "You filled the kettle at the sink.",
                "actors": ["Sarah"],
                "importance": 0.1,
            }
        ),
    )
    assert entry["statement"] == "You filled the kettle at the sink."
    assert entry["store"] == "general"
    assert entry["actors"] == []


def test_reading_run_collapses_across_segment_files():
    sentences = [
        "Her hand picks up the book Bluebirds.",
        "She flips a page.",
        "She turns a page.",
    ]
    records = []
    for index, sentence in enumerate(sentences):
        record = _clip(f"read-{index}", index * 5, index * 5 + 5, sentence)
        record["source"] = f"segment_{index}.mp4"
        records.append(record)
    filed = file_records(records)
    assert filed["general"] == []
    assert len(filed["key"]) == 1
    assert filed["key"][0]["statement"] == "You read Bluebirds."
    assert filed["key"][0]["key_reason"] == "reading"
    assert filed["key"][0]["time_frame"]["end_sec"] == 15


def test_tea_and_kettle_off_is_one_memory():
    records = [
        _clip("walk", 0, 5, "She walks into the kitchen.", "kitchen"),
        _clip("fill", 5, 10, "Her hands fill a kettle at the sink.", "kitchen"),
        _clip("off", 10, 15, "The kettle clicks off.", "kitchen"),
        _clip("pour", 15, 20, "She pours hot water from the kettle into the mug.", "kitchen"),
    ]
    filed = file_records(records)
    assert [entry["statement"] for entry in filed["key"]] == [
        "You made a cup of tea and switched off the kettle.",
    ]
    assert filed["key"][0]["time_frame"]["start_sec"] == 0
    assert filed["key"][0]["time_frame"]["end_sec"] == 20
    assert any(entry["statement"] == "You walked into the kitchen." for entry in filed["general"])


def test_the_same_action_in_a_row_is_one_span():
    records = [
        _clip("tea-1", 20, 25, "She makes a cup of tea.", "home_kitchen"),
        _clip("tea-2", 25, 30, "She makes a cup of tea.", "home_kitchen"),
    ]
    filed = file_records(records)
    assert len(filed["general"]) == 1
    entry = filed["general"][0]
    assert entry["statement"] == "You made a cup of tea in the home_kitchen."
    assert entry["time_frame"]["start_sec"] == 20
    assert entry["time_frame"]["end_sec"] == 30


def test_reading_with_a_sip_in_the_middle_is_one_memory():
    records = [
        _clip("arrive", 0, 5, "She walks into the living room.", "living room"),
        _clip("read-1", 5, 10, "She sits reading, the open book in her hands.", "living room"),
        _clip("sip", 10, 15, "She takes a sip of tea and returns to the book.", "living room"),
        _clip("read-2", 15, 20, "She turns a page and keeps reading the book.", "living room"),
    ]
    filed = file_records(records)
    assert filed["general"] == []
    assert len(filed["key"]) == 1
    entry = filed["key"][0]
    assert entry["statement"] == "You read a book."
    assert entry["key_reason"] == "reading"
    assert entry["time_frame"]["start_sec"] == 0
    assert entry["time_frame"]["end_sec"] == 20


def test_model_failure_uses_the_text(tmp_path):
    filed = file_records(load_records(HOME), llm=_Scripted(RuntimeError("down")))
    assert filed["key"][0]["statement"].startswith("You put your keys")
    write_stores(filed, tmp_path)
    saved = json.loads((tmp_path / "last-seen.json").read_text(encoding="utf-8"))
    assert saved["objects"][0]["label"] == "keys"
