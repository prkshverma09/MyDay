from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

from myday_memory.file_memory import file_records, load_records, write_stores
from myday_memory.llm import WandbMemoryModel, WandbStructurer
from myday_memory.store import build_store, load_chunks

# Real filings land here. Test runs pass -o to a temporary directory.
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MEMORY_DIR = REPO_ROOT / "data" / "memory"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="myday-memory",
        description="Turn llm_input windows into general memory and key memory.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    file_cmd = sub.add_parser("file", help="read pipeline JSONL and file each llm_input")
    file_cmd.add_argument("records", type=Path, help="JSONL from pipeline/video_to_text.py, or a JSON list")
    file_cmd.add_argument("-o", "--out", type=Path, default=DEFAULT_MEMORY_DIR)
    file_cmd.add_argument(
        "--day",
        default=None,
        help="calendar day YYYY-MM-DD when the records do not carry one",
    )
    file_cmd.add_argument(
        "--rules",
        action="store_true",
        help="file from the text only, without calling W&B",
    )

    build = sub.add_parser("build", help="older caption/transcript draft")
    build.add_argument("chunks", type=Path, help="JSON list of captioned clips, or an object with a chunks array")
    build.add_argument("-o", "--out", type=Path, default=Path("memory-store"))
    build.add_argument("--llm", action="store_true", help="structure each clip with W&B inference, then ground it")
    build.add_argument("--vastdb", action="store_true", help="also insert the stores into VastDB schema myday")
    args = parser.parse_args(argv)

    if args.command == "file":
        model = None if args.rules else WandbMemoryModel()
        filed = file_records(load_records(args.records), llm=model)
        written = write_stores(filed, args.out, day=args.day or date.today().isoformat())
        for index, directory in enumerate(written):
            if index:
                sys.stdout.write("\n")
            general = (directory / "general_memory.txt").read_text(encoding="utf-8")
            key = (directory / "key_moments.txt").read_text(encoding="utf-8")
            sys.stdout.write(general)
            if general and key:
                sys.stdout.write("\n")
            sys.stdout.write(key)
        return 0

    chunks = load_chunks(args.chunks)
    store = build_store(chunks, llm=WandbStructurer() if args.llm else None)
    store.write(args.out)
    summary = {
        "memory_entries": len(store.entries),
        "last_seen": len(store.last_seen),
        "days": sorted(store.days),
        "out": str(args.out),
    }
    if args.vastdb:
        from myday_memory.vastdb_store import write_vastdb

        summary["vastdb"] = write_vastdb(args.out)
    json.dump(summary, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
