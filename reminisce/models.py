"""Cosmos Reason, Cosmos Embed1, Canary, and the W&B inference model."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import requests

from reminisce.settings import EMBED_DIM, LLM_STATUS_PATH, TRACES_PATH, Settings, load_settings, redact

DESCRIBE_PROMPT = (
    "Describe this short memory-journal frame in one sentence. "
    "Say only what is visible, including any words on screen. "
    "Do not invent people, places, or events that are not shown. "
    "Do not give medical advice."
)


class ModelError(RuntimeError):
    pass


def _auth_headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _post_json(url: str, payload: dict, headers: dict[str, str], timeout: int = 90) -> dict:
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=timeout)
    except requests.RequestException as exc:
        raise ModelError(redact(str(exc))) from exc
    if response.status_code >= 400:
        raise ModelError(f"HTTP {response.status_code}: {redact(response.text)}")
    try:
        return response.json()
    except json.JSONDecodeError as exc:
        raise ModelError("Model returned a response that was not JSON") from exc


def _get_json(url: str, headers: dict[str, str], timeout: int = 20) -> dict:
    try:
        response = requests.get(url, headers=headers, timeout=timeout)
    except requests.RequestException as exc:
        raise ModelError(redact(str(exc))) from exc
    if response.status_code >= 400:
        raise ModelError(f"HTTP {response.status_code}: {redact(response.text)}")
    return response.json()


def _first_model_id(url: str, headers: dict[str, str]) -> str:
    payload = _get_json(f"{url.rstrip('/')}/v1/models", headers)
    data = payload.get("data") or []
    if not data or not data[0].get("id"):
        raise ModelError(f"No model id at {url}")
    return data[0]["id"]


def describe_image(image_path: Path, settings: Settings | None = None) -> str:
    settings = settings or load_settings()
    if not settings.gpu_ready:
        raise ModelError("GPU token is not set")
    headers = _auth_headers(settings.gpu_bearer_token)
    model = _first_model_id(settings.cosmos_reason_url, headers)
    encoded = base64.b64encode(image_path.read_bytes()).decode()
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": DESCRIBE_PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}},
                ],
            }
        ],
        "max_tokens": 120,
        "temperature": 0,
    }
    body = _post_json(
        f"{settings.cosmos_reason_url.rstrip('/')}/v1/chat/completions",
        payload,
        {**headers, "Content-Type": "application/json"},
    )
    message = body["choices"][0]["message"]
    text = (message.get("content") or message.get("reasoning_content") or "").strip()
    if not text:
        raise ModelError("Cosmos returned an empty description")
    return text[:400]


def embed_text(text: str, settings: Settings | None = None) -> tuple[list[float], str]:
    settings = settings or load_settings()
    if not settings.gpu_ready:
        raise ModelError("GPU token is not set")
    headers = _auth_headers(settings.gpu_bearer_token)
    model = _first_model_id(settings.cosmos_embed_url, headers)
    payload = {
        "input": text,
        "model": model,
        "request_type": "query",
        "encoding_format": "float",
    }
    body = _post_json(
        f"{settings.cosmos_embed_url.rstrip('/')}/v1/embeddings",
        payload,
        {**headers, "Content-Type": "application/json"},
        timeout=60,
    )
    vector = body["data"][0]["embedding"]
    if len(vector) != EMBED_DIM:
        raise ModelError(f"Expected a {EMBED_DIM}-dim embedding, got {len(vector)}")
    return [float(value) for value in vector], model


def transcribe_wav(wav_path: Path, settings: Settings | None = None) -> str:
    settings = settings or load_settings()
    if not settings.gpu_ready:
        raise ModelError("GPU token is not set")
    headers = _auth_headers(settings.gpu_bearer_token)
    url = f"{settings.canary_url.rstrip('/')}/v1/audio/transcriptions"
    try:
        with wav_path.open("rb") as handle:
            response = requests.post(
                url,
                headers=headers,
                files={"file": (wav_path.name, handle, "audio/wav")},
                data={"model": "nvidia/canary-1b"},
                timeout=90,
            )
    except requests.RequestException as exc:
        raise ModelError(redact(str(exc))) from exc
    if response.status_code >= 400:
        raise ModelError(f"HTTP {response.status_code}: {redact(response.text)}")
    try:
        body = response.json()
    except json.JSONDecodeError:
        text = response.text.strip()
        if not text:
            raise ModelError("Canary returned an empty transcript")
        return text
    if isinstance(body, str):
        return body.strip()
    for key in ("text", "transcript", "transcription"):
        if body.get(key):
            return str(body[key]).strip()
    raise ModelError("Canary response had no transcript field")


def remember_llm(model: str, *, fallback: bool = False) -> None:
    """Record whether the last check or answer used the language model. No secrets."""
    try:
        LLM_STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
        LLM_STATUS_PATH.write_text(
            json.dumps({"model": model, "fallback": fallback}),
            encoding="utf-8",
        )
    except OSError:
        return


def load_llm_status() -> dict:
    """Last language-model outcome, from the status file or the latest trace."""
    if LLM_STATUS_PATH.exists():
        try:
            data = json.loads(LLM_STATUS_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            data = None
        if isinstance(data, dict):
            return data
    if not TRACES_PATH.exists():
        return {}
    lines: list[str] = []
    try:
        with TRACES_PATH.open(encoding="utf-8") as handle:
            lines = [line for line in handle if line.strip()]
    except OSError:
        return {}
    for line in reversed(lines):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        model = str(row.get("model") or "")
        if row.get("selection") == "fallback_top1" and model == "template":
            return {"model": "", "fallback": True}
        if model and model != "template":
            return {"model": model, "fallback": False}
    return {}


class ChatModel:
    """W&B serverless chat. complete() raises ModelError; callers decide the fallback."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or load_settings()
        self.model = self.settings.llm_model
        self.backend = "wandb" if self.settings.wandb_ready else "template"

    def complete(self, system: str, user: str) -> str:
        if not self.settings.wandb_ready:
            raise ModelError("W&B API key is not set")
        headers = {
            "Authorization": f"Bearer {self.settings.wandb_api_key}",
            "Content-Type": "application/json",
        }
        if self.settings.wandb_team and self.settings.wandb_project:
            headers["OpenAI-Project"] = f"{self.settings.wandb_team}/{self.settings.wandb_project}"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": 80,
            "temperature": 0,
        }
        body = _post_json("https://api.inference.wandb.ai/v1/chat/completions", payload, headers, timeout=60)
        message = body["choices"][0]["message"]
        text = (message.get("content") or message.get("reasoning_content") or "").strip()
        if not text:
            raise ModelError("W&B model returned an empty reply")
        remember_llm(self.model, fallback=False)
        return text
