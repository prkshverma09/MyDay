import json
from pathlib import Path

from myday_memory.file_memory import file_record, file_records, file_window, load_records, read_memory, write_stores
from myday_memory.llm import _parse_json_object, memory_response
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
    assert general["home_0001"]["statement"] == "You filled a kettle at the sink in the kitchen."
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

    write_stores(filed, tmp_path, day="2026-10-17")
    day_dir = tmp_path / "2026-10-17"
    general_rows = json.loads((day_dir / "general_memory.json").read_text(encoding="utf-8"))
    key_rows = json.loads((day_dir / "key_moments.json").read_text(encoding="utf-8"))
    assert len(general_rows) == 3
    assert len(key_rows) == 3
    general_text = (day_dir / "general_memory.txt").read_text(encoding="utf-8")
    key_text = (day_dir / "key_moments.txt").read_text(encoding="utf-8")
    assert general_text.splitlines()[0] == "2026-10-17  You filled a kettle at the sink in the kitchen."
    assert "2026-10-17  You put your keys down in the hallway." in key_text.splitlines()
    assert "keys" not in general_text
    saved = general_rows[0]
    assert next(iter(saved)) == "text"
    assert saved["text"] == general_text.splitlines()[0]
    assert saved["video"]["camera_id"] == "pov"
    assert saved["video"]["start_sec"] == 0.0
    assert saved["video"]["end_sec"] == 5.0
    assert saved["time_frame"]["start_sec"] == 0.0
    assert read_memory(day_dir / "general_memory.txt")[0]["statement"] == "You filled a kettle at the sink in the kitchen."
    covered = set()
    for entry in filed["entries"]:
        covered.update(entry.get("window_ids") or [])
    assert covered == {record["id"] for record in load_records(HOME)}
    assert read_memory(day_dir / "general_memory.txt")[0]["day"] == "2026-10-17"


def test_two_days_are_separate_folders(tmp_path):
    friday = _clip("fri", 0, 5, "She walks into the kitchen.")
    friday["recorded_at"] = "2026-10-16T09:00:00+01:00"
    saturday = _clip("sat", 0, 5, "She walks into the kitchen.")
    saturday["day"] = "2026-10-17"
    write_stores(file_records([friday, saturday]), tmp_path)
    friday_text = (tmp_path / "2026-10-16" / "general_memory.txt").read_text(encoding="utf-8")
    saturday_text = (tmp_path / "2026-10-17" / "general_memory.txt").read_text(encoding="utf-8")
    assert friday_text.splitlines() == ["2026-10-16  You walked into the living room."]
    assert saturday_text.splitlines() == ["2026-10-17  You walked into the living room."]
    reference = json.loads((tmp_path / "2026-10-17" / "general_memory.json").read_text(encoding="utf-8"))
    assert reference[0]["text"] == saturday_text.strip()
    assert reference[0]["video"]["start_sec"] == 0.0


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
    assert entry["statement"] == "Her hand drops the keys into a blue bowl in the hallway."
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


def test_pressing_the_kettle_switch_is_switching_it_on():
    entry = file_record(
        _clip("switch", 0, 5, "Her finger presses the switch at the top of the kettle.", "kitchen"),
        llm=_Scripted({"statement": "You turned on the kettle.", "store": "general", "key_reason": None}),
    )
    assert entry["store"] == "key"
    assert entry["key_reason"] == "appliance on or off"
    assert entry["statement"] == "You turned on the kettle in the kitchen."
    assert "home_kitchen" not in entry["statement"]


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
    assert entry["statement"] == "You filled the kettle at the sink in the kitchen."
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


def test_tea_steps_stay_separate_and_the_kettle_off_is_key():
    records = [
        _clip("walk", 0, 5, "She walks into the kitchen.", "kitchen"),
        _clip("fill", 5, 10, "Her hands fill a kettle at the sink.", "kitchen"),
        _clip("off", 10, 15, "The kettle clicks off.", "kitchen"),
        _clip("pour", 15, 20, "She pours hot water from the kettle into the mug.", "kitchen"),
    ]
    filed = file_records(records)
    assert [entry["statement"] for entry in filed["key"]] == ["The kettle is off in the kitchen."]
    assert [entry["statement"] for entry in filed["general"]] == [
        "You walked into the kitchen.",
        "You filled a kettle at the sink in the kitchen.",
        "You poured hot water from the kettle into the mug in the kitchen.",
    ]
    assert all("made a cup of tea" not in entry["statement"] for entry in filed["entries"])


def test_the_same_action_in_a_row_is_one_span():
    records = [
        _clip("tea-1", 20, 25, "She fills the kettle at the sink.", "home_kitchen"),
        _clip("tea-2", 25, 30, "She fills the kettle at the sink.", "home_kitchen"),
    ]
    filed = file_records(records)
    assert len(filed["general"]) == 1
    entry = filed["general"][0]
    assert entry["statement"] == "You filled the kettle at the sink in the kitchen."
    assert "home_kitchen" not in entry["statement"]
    assert entry["time_frame"]["start_sec"] == 20
    assert entry["time_frame"]["end_sec"] == 30
    assert entry["window_ids"] == ["tea-1", "tea-2"]


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


def test_two_key_events_in_one_window():
    entries = file_window(_window(
        "door-keys",
        "front door",
        [
            "Sarah knocks and Margaret opens the front door.",
            "Margaret puts her keys in the blue bowl by the door.",
        ],
        ["person count=8 confidence=0.90 position=center/near", "keys count=4 confidence=0.88 position=center/near"],
    ))
    assert [entry["key_reason"] for entry in entries] == ["front door", "object put down or picked up"]
    assert all(entry["store"] == "key" for entry in entries)
    assert entries[0]["actors"] == ["Sarah"]
    assert entries[1]["objects"] == [{"label": "keys", "position": "center/near"}]
    assert [entry["id"] for entry in entries] == ["door-keys#1", "door-keys#2"]
    assert entries[0]["window_id"] == entries[1]["window_id"] == "door-keys"
    assert entries[0]["time_frame"]["start_sec"] == entries[1]["time_frame"]["start_sec"] == 0.0


def test_a_routine_step_and_a_key_event_both_stay():
    entries = file_window(_window(
        "ann_h009",
        "home_kitchen",
        [
            "The kettle clicks off, steam rising steadily from its spout.",
            "The woman picks the kettle up by its handle again.",
        ],
        ["person count=20 confidence=0.93 position=center/near", "cup count=16 confidence=0.80 position=center/near"],
    ))
    assert len(entries) == 2
    assert entries[0]["store"] == "key"
    assert entries[0]["key_reason"] == "appliance on or off"
    assert entries[0]["statement"] == "The kettle is off in the kitchen."
    assert entries[1]["store"] == "general"
    assert "home_kitchen" not in entries[1]["statement"]
    assert "made a cup of tea" not in entries[1]["statement"]
    assert entries[1]["id"] == "ann_h009#2"


def test_kettle_switch_window_names_the_switch():
    path = Path(__file__).parents[2] / "library" / "example-descriptions" / "home-tea-reading.annotations.jsonl"
    window = next(record for record in load_records(path) if record["id"] == "ann_h005")
    filed = file_records([window])
    assert len(filed["key"]) == 1
    key = filed["key"][0]
    assert key["key_reason"] == "appliance on or off"
    assert key["statement"] == "You switched the kettle on in the kitchen."
    for entry in filed["entries"]:
        assert "made a cup of tea" not in entry["statement"].lower()
        assert "home_kitchen" not in entry["statement"]
    assert any(entry["store"] == "general" for entry in filed["entries"])


def test_tea_reading_windows_keep_every_window_and_no_raw_place():
    path = Path(__file__).parents[2] / "library" / "example-descriptions" / "home-tea-reading.annotations.jsonl"
    records = load_records(path)
    filed = file_records(records)
    covered = set()
    for entry in filed["entries"]:
        covered.update(entry.get("window_ids") or [])
        assert "home_kitchen" not in entry["statement"]
        assert "home_living_room" not in entry["statement"]
        assert "made a cup of tea" not in entry["statement"].lower()
    assert covered == {record["id"] for record in records}


def test_model_response_accepts_entries_and_a_bare_object():
    bare = memory_response(
        {
            "store": "key",
            "key_reason": "front door",
            "statement": "You opened the front door.",
            "actors": ["Sarah"],
            "objects": [],
            "appointment": None,
            "importance": 0.9,
        }
    )
    assert len(bare["entries"]) == 1
    assert bare["entries"][0]["actors"] == ["Sarah"]
    wrapped = memory_response(
        {
            "entries": [
                bare["entries"][0],
                {"store": "nope", "key_reason": "not a reason", "statement": "You walked.", "importance": 4},
            ]
        }
    )
    assert wrapped["entries"][1]["store"] == "general"
    assert wrapped["entries"][1]["key_reason"] is None
    assert wrapped["entries"][1]["importance"] == 1.0
    parsed = _parse_json_object(
        'Sure. {"entries": [{"store": "general", "key_reason": null, "statement": "You waited.", '
        '"actors": [], "objects": [], "appointment": null, "importance": 0.2}]}'
    )
    assert parsed["entries"][0]["statement"] == "You waited."


def test_a_bare_model_object_does_not_drop_the_second_event():
    record = _window(
        "door-keys",
        "front door",
        [
            "Sarah knocks and Margaret opens the front door.",
            "Margaret puts her keys in the blue bowl by the door.",
        ],
        ["person count=8 confidence=0.90 position=center/near", "keys count=4 confidence=0.88 position=center/near"],
    )
    entries = file_window(
        record,
        llm=_Scripted({"store": "key", "key_reason": "front door", "statement": "You opened the front door."}),
    )
    assert len(entries) == 2
    assert entries[1]["key_reason"] == "object put down or picked up"


def _window(window_id, location, sentences, yolo):
    cosmos = "\n".join(f"- {sentence}" for sentence in sentences)
    boxes = "\n".join(f"- {line}" for line in yolo)
    return {
        "id": window_id,
        "camera_id": "door",
        "location": location,
        "window": {"start_sec": 0.0, "end_sec": 5.0},
        "llm_input": (
            "Time window: 0.0-5.0s\n"
            "Camera: door\n"
            f"Location: {location}\n"
            "Cosmos:\n"
            f"{cosmos}\n"
            "YOLO:\n"
            f"{boxes}\n"
        ),
    }


def test_model_failure_uses_the_text(tmp_path):
    filed = file_records(load_records(HOME), llm=_Scripted(RuntimeError("down")))
    assert filed["key"][0]["statement"].startswith("You put your keys")
    write_stores(filed, tmp_path, day="2026-10-17")
    saved = json.loads((tmp_path / "2026-10-17" / "last-seen.json").read_text(encoding="utf-8"))
    assert saved["objects"][0]["label"] == "keys"
