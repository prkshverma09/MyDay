"""Local traces for every agent run, plus a best-effort W&B / Weave log."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

from reminisce.settings import TRACES_PATH, load_settings, redact

_WANDB_DISABLED = False
_WEAVE_READY = False


def append_trace(event: dict) -> None:
    TRACES_PATH.parent.mkdir(parents=True, exist_ok=True)
    row = dict(event)
    row.setdefault("ts", datetime.now(timezone.utc).isoformat())
    with TRACES_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")
    _wandb(row)
    _weave(row)


def _wandb(event: dict) -> None:
    global _WANDB_DISABLED
    if _WANDB_DISABLED:
        return
    settings = load_settings()
    if not settings.wandb_ready:
        _WANDB_DISABLED = True
        return
    try:
        os.environ.setdefault("WANDB_SILENT", "true")
        import wandb

        wandb.init(
            entity=settings.wandb_team or None,
            project=settings.wandb_project,
            job_type="reminisce-agent",
            name="reminisce-agent",
            reinit=False,
            settings=wandb.Settings(silent=True, init_timeout=20),
        )
        wandb.log(
            {
                "grounded": 1 if event.get("grounded") else 0,
                "latency_s": float(event.get("latency_s") or 0),
                "citations": len(event.get("citations") or []),
            }
        )
    except Exception:
        _WANDB_DISABLED = True


def _weave(event: dict) -> None:
    global _WEAVE_READY
    settings = load_settings()
    if not settings.wandb_ready:
        return
    try:
        import weave

        if not _WEAVE_READY:
            target = settings.wandb_project
            if settings.wandb_team:
                target = f"{settings.wandb_team}/{settings.wandb_project}"
            weave.init(target)
            _WEAVE_READY = True
        weave.publish({"reminisce_agent_run": {key: event[key] for key in event if key != "embedding"}})
    except Exception:
        return


def trace_from_answer(kind: str, question: str, answer, latency_s: float) -> None:
    append_trace(
        {
            "kind": kind,
            "question": question,
            "tools": answer.tools,
            "prompt_version": answer.prompt_version,
            "model": answer.model,
            "selection": answer.selection,
            "raw_model": answer.raw_model,
            "served": answer.text,
            "citations": answer.citations,
            "grounded": answer.grounded,
            "retrieval_path": getattr(answer, "retrieval_path", ""),
            "retrieval_note": getattr(answer, "retrieval_note", ""),
            "latency_s": round(latency_s, 3),
            "retrieved": [
                {
                    "clip_id": hit.clip.clip_id,
                    "cosine": round(hit.cosine, 4),
                    "combined": round(hit.combined, 4),
                    "tag_match": hit.tag_match,
                }
                for hit in answer.hits
            ],
        }
    )


def safe_error(exc: Exception) -> str:
    return redact(f"{type(exc).__name__}: {exc}")
