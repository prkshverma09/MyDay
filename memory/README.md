# Memory store

Pipeline 4. `pipeline/video_to_text.py` writes `llm_input`. This stage reads only that text, asks W&B to phrase the memory, and files each window into one store.

The instruction sent to W&B is [`INSTRUCTION.md`](INSTRUCTION.md).

```bash
cd memory
PYTHONPATH=src python -m myday_memory.cli file examples/home-windows.jsonl -o memory-files
```

That writes:

- `general-memory.jsonl` — routine moments, one JSON object per window
- `key-memory.jsonl` — visitors, appliance on/off, objects put down or picked up, the front door (including locking and unlocking), and anything unusual
- `last-seen.json` — the latest place a tracked object was put down, taken from the key file

Each object uses `"store": "general"` or `"store": "key"`. `id`, `camera_id`, `location`, and `window` are copied from the pipeline record. `time_frame` is the same span in the video (`start_sec`, `end_sec`, plus `source` and chunk index when the pipeline record has them), so a moment can be played back. The model cannot change it. Pass `--rules` to file from the text without calling W&B.

A run of ordinary reading windows (pick up a book, then turn pages) is written as one general memory, "You read [the book]", and `time_frame` covers the whole run. One page turn stays one window. `examples/home-windows.jsonl` is a few made-up home windows in the pipeline's format. The 100 archive records in `library/example-descriptions/` are format examples only. Do not file them.

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
