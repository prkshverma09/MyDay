#!/usr/bin/env python3
"""Describe practice clips with Cosmos, embed them, and mirror rows into VAST."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reminisce.ingest import ingest_video
from reminisce.settings import MANIFEST_PATH, VAST_STATUS_PATH
from reminisce.store import load_clips


def main() -> None:
    parser = argparse.ArgumentParser(description="Index practice clips into the journal and VAST")
    parser.add_argument("--force", action="store_true", help="Describe and embed again even if a clip is already indexed")
    parser.add_argument("--clip", help="Index only this clip id")
    args = parser.parse_args()
    if not MANIFEST_PATH.exists():
        print("No manifest. Run scripts/make_sample_clips.py first.", file=sys.stderr)
        sys.exit(1)
    manifest = json.loads(MANIFEST_PATH.read_text())
    existing = {clip.clip_id: clip for clip in load_clips()}
    failures = 0
    for row in manifest:
        if args.clip and row["clip_id"] != args.clip:
            continue
        current = existing.get(row["clip_id"])
        if current and current.embedding and not args.force:
            print(f"skip {row['clip_id']}", flush=True)
            continue
        print(f"indexing {row['clip_id']}", flush=True)
        try:
            clip = ingest_video(
                Path(row["path"]),
                clip_id=row["clip_id"],
                owner_id="user_123",
                recorded_at=row["recorded_at"],
                tags=row["tags"],
                subtitle=row["subtitle"],
                shared_with=list(current.shared_with) if current else [],
            )
        except Exception as exc:
            failures += 1
            print(f"failed {row['clip_id']}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            continue
        print(f"{clip.clip_id} {clip.description_source} dims={len(clip.embedding)} {clip.cosmos_description[:90]}", flush=True)
    if VAST_STATUS_PATH.exists():
        print(VAST_STATUS_PATH.read_text().strip(), flush=True)
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
