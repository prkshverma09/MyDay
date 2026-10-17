#!/usr/bin/env python3
"""Create the ten synthetic practice clips. No real person is recorded."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reminisce.catalog import SAMPLE_CLIPS
from reminisce.media import make_title_clip, probe_duration
from reminisce.settings import CLIPS_DIR, MANIFEST_PATH, QUESTIONS_PATH


def main() -> None:
    today = datetime.now().astimezone()
    manifest = []
    for sample in SAMPLE_CLIPS:
        path = CLIPS_DIR / f"{sample['clip_id']}.mp4"
        print(f"writing {path.name}", flush=True)
        make_title_clip(
            path,
            color=sample["color"],
            title=sample["title"],
            subtitle=sample["subtitle"],
            seconds=6,
        )
        recorded = today.replace(hour=sample["hour"], minute=sample["minute"], second=0, microsecond=0)
        manifest.append(
            {
                "clip_id": sample["clip_id"],
                "path": str(path),
                "tags": sample["tags"],
                "subtitle": sample["subtitle"],
                "recorded_at": recorded.isoformat(timespec="seconds"),
                "duration_s": round(probe_duration(path), 2),
                "question": sample["question"],
            }
        )
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    QUESTIONS_PATH.write_text(
        json.dumps(
            [{"question": row["question"], "expected_clip_id": row["clip_id"]} for row in manifest],
            indent=2,
        )
    )
    print(f"wrote {len(manifest)} clips", flush=True)


if __name__ == "__main__":
    main()
