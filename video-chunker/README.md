# video-chunker

Cut one local video into MP4 pieces of a chosen length.

## Install

ffmpeg 5 or newer is required (`brew install ffmpeg` on macOS).

```bash
python -m pip install -e ".[dev]"
```

## Use

```bash
video-chunker holiday.mp4 --duration 15 --output ./chunks
```

That writes `chunks/holiday_0000.mp4`, then `holiday_0001.mp4`, and so on,
plus `chunks/holiday.chunks.json`. The last file holds any remainder.

`--mode fast` copies packets and can cut on a later keyframe.
`--overwrite` replaces previous chunks for the same file name.

Stdout prints one absolute chunk path per line.

| Exit | Meaning |
| --- | --- |
| 0 | Chunks and the manifest are written. |
| 2 | Bad arguments, missing binary, unreadable media, or existing files. |
| 1 | ffmpeg failed, or a chunk length did not match the plan. |

## Test

```bash
pytest -v
```
