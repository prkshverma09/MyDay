# Example video descriptions

One hundred 5-second descriptions sampled from footage already in the team index. Each line of `annotations.jsonl` is one window. This file is the input contract for the next LLM stage. Nothing here was produced by that LLM.

Segments come from the indexed archive. One segment per parent chunk, spread across cameras, so the library is not a hundred slices of one minute. Every clip was normalized to 640px wide, 4 fps, no audio before Cosmos Reason and YOLO11, including high-resolution clips under 2MB. At that size Cosmos stays near 2400 prompt tokens. A raw 2048px neighborhood clip is only about 1MB but lands near 12000 tokens.

## What this file contains

100 windows, 100 parent chunks, 12 cameras, 6 locations:

| Camera | Location | Windows |
|---|---|---|
| neighborhood_cam-1 | neighborhood | 9 |
| nyc_bike_gopro-1 | new_york | 9 |
| nyc_streets_cam-1 | new_york | 9 |
| nyc_streets_cam-2 | new_york | 9 |
| pie_cam-3 | toronto | 9 |
| sdg_warehouse_cam-2 | warehouse3 | 8 |
| sf_streets_cam-1 | san_francisco | 8 |
| sf_streets_cam-2 | san_francisco | 8 |
| sf_streets_cam-3 | san_francisco | 7 |
| sf_streets_cam-4 | san_francisco | 8 |
| sf_streets_cam-5 | san_francisco | 8 |
| smartspace_cam-1 | indoor | 8 |

Ninety-nine windows are 5.0 seconds. `ann_0100` is 0.0–3.15s because that San Francisco chunk is the tail of the recording and the index only has 3.15 seconds of it.

Cosmos writes short relational sentences: an actor, a concrete action, an object with color or type, and where that object sits relative to something else that is visible. YOLO adds class, count, confidence, and a coarse position. The two are fused. The model is not asked to invent keys, cups, brands, or actions that are not in the frame.

## Record schema

```json
{
  "id": "ann_0001",
  "source": "s3://…/segment….mp4",
  "camera_id": "nyc_bike_gopro-1",
  "location": "new_york",
  "window": {"start_sec": 0.0, "end_sec": 5.0},
  "cosmos": {
    "sentences": ["A cyclist in a yellow jacket rides ahead of a white van."],
    "inference_s": 1.2,
    "prompt_tokens": 2402,
    "completion_tokens": 80
  },
  "yolo": {
    "inference_s": 0.5,
    "frames": 20,
    "objects": [
      {"label": "person", "hits": 12, "max_confidence": 0.8, "position": "right/near"}
    ],
    "ignored": [
      {"label": "toilet", "hits": 1, "max_confidence": 0.3, "reason": "hit_count_1"}
    ],
    "perception_ok": true
  },
  "llm_input": "Time window, Cosmos sentences, and YOLO objects with positions. Plain text."
}
```

| Field | Meaning |
|---|---|
| `id` | `ann_` plus a 4-digit number |
| `source` | Segment URI in the index. The clip is already about 5 seconds |
| `camera_id`, `location` | From the index row for the parent chunk |
| `window` | `segment_start_sec` / `segment_end_sec` inside that parent chunk |
| `cosmos.sentences` | At most five sentences. The example sentence in the prompt is discarded if the model copies it |
| `cosmos.inference_s`, token counts | Timing and usage for that call |
| `yolo.objects[].hits` | Detection boxes of that class across the window |
| `yolo.objects[].position` | Horizontal third of the clearest box, plus `near` or `far` from its height |
| `yolo.ignored` | Absurd classes (`toilet`, `airplane`, `train`) dropped for low confidence, a single hit, or a single frame. Omitted from `llm_input` |
| `llm_input` | The only text the next LLM should read. No chain-of-thought |

`window` on a file you pass to `pipeline/video_to_text.py` is instead relative to that file. The library uses the index timestamps so a later stage can place the moment on the parent chunk.
