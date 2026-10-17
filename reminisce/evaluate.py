"""Run the ten labelled questions and write counted results."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from reminisce.agent import answer_question
from reminisce.catalog import SAMPLE_CLIPS
from reminisce.models import ChatModel, embed_text
from reminisce.settings import EVAL_PATH, load_settings, redact
from reminisce.store import Clip, load_clips
from reminisce.vectors import extract_clip_ids, rank_clips


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 3)


def evaluate(clips: list[Clip] | None = None, llm: ChatModel | None = None) -> dict:
    import time

    clips = clips if clips is not None else load_clips()
    by_id = {clip.clip_id: clip for clip in clips}
    settings = load_settings()
    prefs_owner = "user_123"
    carer_id = "carer_daughter"
    llm = llm if llm is not None else ChatModel(settings)
    cosmos_count = sum(1 for row in SAMPLE_CLIPS if by_id.get(row["clip_id"]) and by_id[row["clip_id"]].description_source == "cosmos")
    missing = [row["clip_id"] for row in SAMPLE_CLIPS if row["clip_id"] not in by_id]

    prompt_reports: dict[str, dict] = {}
    corpus_ids = {clip.clip_id for clip in clips}

    for prompt_name in ("strict", "warm"):
        rows = []
        raw_hits = 0
        served_hits = 0
        grounded_hits = 0
        hallucinations = 0
        latencies: list[float] = []
        for sample in SAMPLE_CLIPS:
            question = sample["question"]
            expected = sample["clip_id"]
            started = time.perf_counter()
            vector, embed_model = embed_text(question, settings)
            ranked = rank_clips(question, clips, vector)
            answer = answer_question(
                question,
                clips,
                vector,
                role="owner",
                owner_id=prefs_owner,
                carer_id=carer_id,
                prompt_version=prompt_name,
                llm=llm,
            )
            latency = time.perf_counter() - started
            latencies.append(latency)
            embedding_top = ""
            hybrid_top = ranked[0].clip.clip_id if ranked else ""
            if ranked:
                embedding_top = sorted(ranked, key=lambda hit: hit.cosine, reverse=True)[0].clip.clip_id
            raw_ids = extract_clip_ids(answer.raw_model)
            invented = [clip_id for clip_id in raw_ids if clip_id not in corpus_ids]
            if expected in raw_ids:
                raw_hits += 1
            if expected in answer.citations:
                served_hits += 1
            if answer.grounded:
                grounded_hits += 1
            hallucinations += len(invented)
            top = ranked[0] if ranked else None
            rows.append(
                {
                    "question": question,
                    "expected": expected,
                    "embedding_top": embedding_top,
                    "hybrid_top": hybrid_top,
                    "embedding_ok": embedding_top == expected,
                    "hybrid_ok": hybrid_top == expected,
                    "cosine": round(top.cosine, 4) if top else 0,
                    "raw_model": answer.raw_model,
                    "raw_ok": expected in raw_ids,
                    "served": answer.text,
                    "served_ok": expected in answer.citations,
                    "grounded": answer.grounded,
                    "selection": answer.selection,
                    "hallucinated_ids": invented,
                    "latency_s": round(latency, 3),
                    "description_source": by_id[expected].description_source if expected in by_id else "missing",
                }
            )
        prompt_reports[prompt_name] = {
            "raw_selection": raw_hits,
            "served_cites_expected": served_hits,
            "served_grounded": grounded_hits,
            "raw_hallucinations": hallucinations,
            "mean_latency_s": _mean(latencies),
            "rows": rows,
        }

    n = len(SAMPLE_CLIPS)
    # Embedding and hybrid counts were incremented once per prompt. Keep a single pass.
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n": n,
        "missing_clips": missing,
        "cosmos_descriptions": cosmos_count,
        "fallback_descriptions": n - cosmos_count - len(missing),
        "embedding_only_at_1": prompt_reports["strict"]["rows"] and sum(
            1 for row in prompt_reports["strict"]["rows"] if row["embedding_ok"]
        ),
        "hybrid_at_1": sum(1 for row in prompt_reports["strict"]["rows"] if row["hybrid_ok"]),
        "llm_model": llm.model,
        "embed_model": embed_model if SAMPLE_CLIPS else "",
        "prompts": prompt_reports,
        "wandb": "pending",
    }
    report["wandb"] = _log_wandb(report, settings)
    EVAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    EVAL_PATH.write_text(json.dumps(report, indent=2))
    return report


def _log_wandb(report: dict, settings) -> str:
    if not settings.wandb_ready:
        return "skipped: no api key"
    try:
        import wandb

        run = wandb.init(
            entity=settings.wandb_team or None,
            project=settings.wandb_project,
            job_type="reminisce-eval",
            name="reminisce-eval",
            reinit=True,
            settings=wandb.Settings(silent=True, init_timeout=30),
        )
        columns = [
            "prompt",
            "question",
            "expected",
            "embedding_top",
            "hybrid_top",
            "served_ok",
            "grounded",
            "latency_s",
        ]
        table = wandb.Table(columns=columns)
        for prompt_name, block in report["prompts"].items():
            for row in block["rows"]:
                table.add_data(
                    prompt_name,
                    row["question"],
                    row["expected"],
                    row["embedding_top"],
                    row["hybrid_top"],
                    row["served_ok"],
                    row["grounded"],
                    row["latency_s"],
                )
        wandb.log(
            {
                "embedding_only_at_1": report["embedding_only_at_1"],
                "hybrid_at_1": report["hybrid_at_1"],
                "strict_grounded": report["prompts"]["strict"]["served_grounded"],
                "warm_grounded": report["prompts"]["warm"]["served_grounded"],
                "strict_raw_selection": report["prompts"]["strict"]["raw_selection"],
                "warm_raw_selection": report["prompts"]["warm"]["raw_selection"],
                "results": table,
            }
        )
        url = run.url or ""
        run.finish()
        return url or "logged"
    except Exception as exc:
        return "skipped: " + redact(f"{type(exc).__name__}: {exc}")
