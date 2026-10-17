from __future__ import annotations

import json
import os
import re
import urllib.request

from myday_memory.models import ChunkText

DEFAULT_MODEL = "meta-llama/Llama-3.3-70B-Instruct"
INFERENCE_URL = "https://api.inference.wandb.ai/v1/chat/completions"

SYSTEM_PROMPT = """You structure one home-video clip into a memory entry for someone with early memory loss.
Use only the caption and transcript. Do not invent people, objects, places, or events.
Reply with one JSON object and no other text, using this shape:
{
  "people": ["names actually present"],
  "action": "short verb phrase",
  "objects": ["keys", "glasses", "hearing aid", "mug", "cat"],
  "statement": "one plain sentence of what happened",
  "importance": "high or normal",
  "supports": [
    {"question_id": "people|lost_objects|door_lock|stove|appointment", "text": "one grounded sentence", "confidence": 0.0}
  ]
}
Omit a support when the clip does not show that question. importance is high for a visitor, keys, glasses, a hearing aid, the stove, or the door lock."""


class WandbStructurer:
    def __init__(self, model: str | None = None):
        self.model = model or os.environ.get("MYDAY_LLM_MODEL") or DEFAULT_MODEL

    def structure(self, chunk: ChunkText) -> dict:
        user = {
            "clip_id": chunk.clip_id,
            "place": chunk.place,
            "camera": chunk.camera,
            "t_start": chunk.t_start,
            "t_end": chunk.t_end,
            "caption": chunk.caption,
            "transcript": chunk.transcript,
            "detections": chunk.detections,
        }
        return chat_json(SYSTEM_PROMPT, json.dumps(user), model=self.model)


class WandbMemoryModel:
    """W&B call for one llm_input block. The system text is memory/INSTRUCTION.md."""

    def __init__(self, model: str | None = None):
        self.model = model or os.environ.get("MYDAY_LLM_MODEL") or DEFAULT_MODEL

    def complete(self, llm_input: str) -> dict:
        from myday_memory.file_memory import load_instruction

        return chat_json(load_instruction(), llm_input, model=self.model)


def chat_json(system: str, user: str, model: str | None = None) -> dict:
    api_key = os.environ.get("WANDB_API_KEY", "")
    if not api_key:
        raise RuntimeError("WANDB_API_KEY is not set")
    team = os.environ.get("WANDB_TEAM", "")
    project = os.environ.get("WANDB_PROJECT", "")
    body = {
        "model": model or os.environ.get("MYDAY_LLM_MODEL") or DEFAULT_MODEL,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    request = urllib.request.Request(
        INFERENCE_URL,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "OpenAI-Project": f"{team}/{project}" if team and project else project,
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        raw = json.loads(response.read().decode("utf-8"))
    return _parse_json_object(raw["choices"][0]["message"]["content"])


def _parse_json_object(content: str) -> dict:
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if not match:
            raise
        parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("model response was not a JSON object")
    return parsed
