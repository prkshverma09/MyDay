# Video to text

Perception only. A later LLM reads `llm_input`. This pipeline never calls that LLM.

## Chunks, same as video-chunker

A stream is a sequence of equal chunks. With `--chunk 10`, the file is cut at 0–10s, 10–20s, 20–30s, and so on. The last piece keeps whatever time remains. Those bounds come from `video_chunker.plan.plan_segments`, the same planner `video-chunker` uses.

Each chunk is described as soon as it is cut. Cosmos and YOLO run together on that piece, one JSONL line is written, then the next chunk starts. A one-minute video at 10 seconds is six records, not one call on the whole minute.

Before inference the chunk is scaled to 640px wide, 4 fps, no audio, H.264 CRF 28. Cosmos still sees the whole chunk. Thirty seconds is the longest chunk that still adds new visual tokens. A 60 second chunk saturates and the model invents the second half, so keep the stream at 10–30 seconds.

## What runs

For every window, Cosmos Reason and YOLO11 run at the same time. Their results are fused into one JSON object:

- `cosmos.sentences` — up to five short relational sentences (actor, action, object, where it is relative to something else)
- `yolo.objects` — class, hit count, max confidence, coarse position (`left`/`center`/`right` and `near`/`far`)
- `chunk` — index, start, end, and duration, matching the video-chunker manifest
- `llm_input` — a compact plain-text block: chunk time, Cosmos sentences, YOLO objects with positions

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
python pipeline/video_to_text.py path/to/video.mp4 --chunk 10 -o descriptions.jsonl
python pipeline/video_to_text.py path/to/video.mp4 --chunk 10 --camera-id door --location kitchen
python pipeline/video_to_text.py --manifest chunks/day.chunks.json -o descriptions.jsonl
```

`--chunk` defaults to 10 seconds. `--window` is an alias. Omit `-o` to write JSONL to stdout. `--manifest` describes chunk files already written by `video-chunker`, in index order, and uses the times stored in that manifest.

## Next stage

`llm_input` is the contract for the memory LLM (people, objects, what happened). That call is not part of this repo.
