# Video to text

Perception only. A later LLM reads `llm_input`. This pipeline never calls that LLM.

## Why 5-second windows

Cosmos Reason keeps the most useful detail on a 5-second clip. Thirty seconds is the longest window that still ingests new frames. Sixty seconds saturates (about the same prompt tokens as 30 seconds) and the model invents the second half. A one-minute video is twelve windows, not one 60-second call.

Each window is transcoded to 640px wide, 4 fps, no audio, H.264 CRF 28 before inference. On that input, Cosmos is about 1–3 seconds and ~2400 prompt tokens. YOLO11 is about 0.4–1.0 seconds over the 20 frames.

## What runs

For every window, Cosmos Reason and YOLO11 run at the same time. Their results are fused into one JSON object:

- `cosmos.sentences` — up to five short relational sentences (actor, action, object, where it is relative to something else)
- `yolo.objects` — class, hit count, max confidence, coarse position (`left`/`center`/`right` and `near`/`far`)
- `llm_input` — a compact plain-text block: time window, Cosmos sentences, YOLO objects with positions

One-frame absurd YOLO labels (`toilet`, `airplane`, `train`) are dropped when confidence is below 0.5, the hit count is 1, or every hit sits in a single frame. Dropped labels are listed under `yolo.ignored` and are left out of `llm_input`.

## Entrypoint

Requires `ffmpeg` and the packages in `requirements.txt`. Auth and endpoints come from the environment, not from a config file:

| Variable | Role |
|---|---|
| `GPU_BEARER_TOKEN` | Bearer token for both services |
| `COSMOS_URL` | Default `http://166.19.38.112:8001` |
| `YOLO_URL` | Default `http://166.19.38.112:8002` |
| `COSMOS_MODEL` | Optional. Otherwise the id from `GET /v1/models` |

```bash
python pipeline/video_to_text.py path/to/video.mp4 -o descriptions.jsonl
python pipeline/video_to_text.py path/to/video.mp4 --window 5 --camera-id door --location kitchen
```

Omit `-o` to write JSONL to stdout. `--window` defaults to 5 seconds.

## Next stage

`llm_input` is the contract for the memory LLM (people, objects, what happened). That call is not part of this repo.
