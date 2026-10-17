# Memory store

Pipeline 4. `pipeline/video_to_text.py` writes `llm_input`. This stage reads only that text, asks W&B to phrase the memory, and files each distinct event in the window. A routine step and a key event in the same window both stay. Two key events in one window stay two memories.

The instruction sent to W&B is [`INSTRUCTION.md`](INSTRUCTION.md).

```bash
cd memory
PYTHONPATH=src python -m myday_memory.cli file path/to/real.jsonl
```

A real run writes to `data/memory/` at the repo root, one folder per day. The folder name is the day, `2026-10-17`. A record can carry that day in `day`, `date`, or `recorded_at`. Otherwise the command uses `--day`, or today's date. Pass `-o` to send a test run somewhere else, such as `/tmp`.

Each day folder contains:

- `general_memory.txt` — routine memories, one line each: `2026-10-17  You filled the kettle at the sink in the kitchen.`
- `key_moments.txt` — the key memories in that same text form
- `general_memory.json` and `key_moments.json` — the same lines, plus the video span for cross-checking
- `last-seen.json` — the latest place a tracked object was put down, taken from the key file

Each object uses `"store": "general"` or `"store": "key"`. `camera_id`, `location`, and `window` are copied from the pipeline record. `time_frame` and `video` keep the same span (`start_sec`, `end_sec`, camera, and the segment `source` when the record has one), so a text line can be checked against the clip. The model cannot change that. When a window files more than one event, the ids are `ann_h009#1` and `ann_h009#2`. `window_id` stays the pipeline id. Pass `--rules` to file from the text without calling W&B.

A statement names the step in that window, in plain place words: `home_kitchen` is "the kitchen". A raw location token never appears in a sentence. "You made a cup of tea" is not a window statement. Switching the kettle on or off is its own key memory. Consecutive windows with the same sentence share one span.

A run of reading (pick up a book, then turn pages, including a sip in the middle) is one key memory, "You read [the book]", and `time_frame` covers the whole run. One page turn stays one general window. `examples/home-windows.jsonl` is a few made-up home windows in the pipeline's format. The 100 archive records in `library/example-descriptions/` are format examples only. Do not file them.

## Older caption draft

```
clip text  →  memory entry  →  last-seen
                           →  coverage
```

A memory entry is one moment: who was there, what they did, which objects were visible, one plain sentence, and the reassurance questions that moment can answer. Every answer points at the clip. This step never writes an abstention. If a question was not recorded, coverage says so, and pipeline 5 is the piece that replies "I don't have that recorded."

## Input

A JSON list, or `{"chunks": [...]}`. One object per clip:

| Field | Meaning |
| --- | --- |
| `clip_id`, `camera`, `place`, `t_start`, `t_end`, `file` | The segment. Times are ISO-8601. |
| `caption` | Cosmos description of the clip. |
| `transcript` | Speech in the clip. Empty if nobody spoke. |
| `detections` | Optional object labels from YOLO. |
| `resident` | People who live here, so they are not counted as visitors. Defaults to Margaret. |

`examples/morning.json` is a short morning. `examples/margaret-day.json` is the filmed day from the team prompt sheet: tea, keys, Sarah's visit, Biscuit, glasses, the stove left on and then turned off, the plant, the garden, and Sarah's evening call. The front door lock is not in any clip. `examples/personal-memory/` is that day saved, and `key_moments.json` is the highlighted subset.

## Run

```bash
cd memory
python -m myday_memory.cli build examples/morning.json -o memory-store
```

That writes:

- `memory_entries.json` — one entry per clip
- `last_seen.json` — the latest place each object was seen
- `coverage.json` — the day's intervals, and which of the five questions have evidence

Add `--llm` to ask Weights & Biases inference for the sentence, then keep only the parts the caption or transcript actually supports. If the call fails, the rules result is kept. Add `--vastdb` to insert the same rows into VastDB schema `myday` (tables `memory_entries`, `last_seen`, `coverage`) on this team's bucket.

## What pipeline 5 should read

- "Where are my keys?" — `last_seen.objects` for `keys`. No row means it was not recorded.
- "Has Sarah been today?" — `coverage.days[].questions.people.names`.
- "Did I lock the door?" / "Is the stove off?" / "When is my appointment?" — `coverage.days[].questions.<id>.recorded`. The unanswered ids are listed in `unrecorded`.
- The clip to play is `clip_id` plus `file`, `t_start`, and `t_end` on the entry or the last-seen row.

## Test

```bash
python -m pytest -q
```
