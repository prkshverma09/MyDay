#!/usr/bin/env python3
"""Run the ten-question eval and print the counts."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reminisce.evaluate import evaluate


def main() -> None:
    report = evaluate()
    n = report["n"]
    warm = report["prompts"]["warm"]
    strict = report["prompts"]["strict"]
    print(f"cosmos descriptions: {report['cosmos_descriptions']}/{n}", flush=True)
    print(f"fallback descriptions: {report['fallback_descriptions']}", flush=True)
    if report["missing_clips"]:
        print(f"missing: {', '.join(report['missing_clips'])}", flush=True)
    print(f"embedding-only retrieval: {report['embedding_only_at_1']}/{n}", flush=True)
    print(f"hybrid retrieval: {report['hybrid_at_1']}/{n}", flush=True)
    print(
        f"strict raw selection: {strict['raw_selection']}/{n}  "
        f"served: {strict['served_cites_expected']}/{n}  "
        f"grounded: {strict['served_grounded']}/{n}  "
        f"hallucinations: {strict['raw_hallucinations']}  "
        f"latency: {strict['mean_latency_s']}s",
        flush=True,
    )
    print(
        f"warm raw selection: {warm['raw_selection']}/{n}  "
        f"served: {warm['served_cites_expected']}/{n}  "
        f"grounded: {warm['served_grounded']}/{n}  "
        f"hallucinations: {warm['raw_hallucinations']}  "
        f"latency: {warm['mean_latency_s']}s",
        flush=True,
    )
    print(f"model: {report['llm_model']}", flush=True)
    print(f"wandb: {report['wandb']}", flush=True)
    grounded_ok = strict["served_grounded"] == n and warm["served_grounded"] == n
    if not grounded_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
