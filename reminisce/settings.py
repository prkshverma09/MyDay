"""Environment configuration. Values stay in memory and are never logged."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CLIPS_DIR = DATA / "clips"
JOURNAL_DIR = DATA / "journal"
JOURNAL_PATH = JOURNAL_DIR / "clips.json"
PREFS_PATH = JOURNAL_DIR / "prefs.json"
TRACES_PATH = JOURNAL_DIR / "traces.jsonl"
LLM_STATUS_PATH = JOURNAL_DIR / "llm_status.json"
EVAL_PATH = JOURNAL_DIR / "eval_latest.json"
VAST_STATUS_PATH = JOURNAL_DIR / "vast_status.json"
MANIFEST_PATH = CLIPS_DIR / "manifest.json"
QUESTIONS_PATH = DATA / "questions.json"

# Host and ports from the builders GPU skill. Override with environment variables.
_GPU_HOST = os.environ.get("GPU_HOST", "166.19.38.112")

DEFAULT_PREFS = {
    "owner_id": "user_123",
    "carer_id": "carer_daughter",
    "carer_name": "my daughter",
    "daily_digest": False,
}

MIN_CLIP_SECONDS = 5
MAX_CLIP_SECONDS = 30
EMBED_DIM = 256
MIN_COSINE = 0.22
TAG_BONUS = 0.25


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


@dataclass(frozen=True)
class Settings:
    s3_endpoint: str
    access_key: str
    secret_key: str
    vastdb_bucket: str
    gpu_bearer_token: str
    cosmos_reason_url: str
    cosmos_embed_url: str
    canary_url: str
    wandb_api_key: str
    wandb_team: str
    wandb_project: str
    llm_model: str

    @property
    def vast_ready(self) -> bool:
        return bool(self.s3_endpoint and self.access_key and self.secret_key and self.vastdb_bucket)

    @property
    def gpu_ready(self) -> bool:
        return bool(self.gpu_bearer_token)

    @property
    def wandb_ready(self) -> bool:
        return bool(self.wandb_api_key and self.wandb_project)


def load_settings() -> Settings:
    return Settings(
        s3_endpoint=_env("S3_ENDPOINT"),
        access_key=_env("ACCESS_KEY"),
        secret_key=_env("SECRET_KEY"),
        vastdb_bucket=_env("VASTDB_BUCKET"),
        gpu_bearer_token=_env("GPU_BEARER_TOKEN"),
        cosmos_reason_url=_env("COSMOS3_REASON_URL", f"http://{_GPU_HOST}:8001"),
        cosmos_embed_url=_env("COSMOS_EMBED1_URL", f"http://{_GPU_HOST}:8003"),
        canary_url=_env("CANARY_1B_URL", f"http://{_GPU_HOST}:8004"),
        wandb_api_key=_env("WANDB_API_KEY"),
        wandb_team=_env("WANDB_TEAM"),
        wandb_project=_env("WANDB_PROJECT"),
        llm_model=_env("REMINISCE_LLM_MODEL", "meta-llama/Llama-3.1-8B-Instruct"),
    )


def redact(text: str) -> str:
    """Remove credential material from an error string before it is stored or printed."""
    import re

    cleaned = text or ""
    settings = load_settings()
    secrets = [
        settings.access_key,
        settings.secret_key,
        settings.gpu_bearer_token,
        settings.wandb_api_key,
        settings.s3_endpoint,
        os.environ.get("PASSWORD", ""),
        os.environ.get("INGRESS_URL", ""),
    ]
    try:
        from reminisce.vss_search import cached_token

        secrets.append(cached_token())
    except Exception:
        pass
    for secret in secrets:
        if secret:
            cleaned = cleaned.replace(secret, "[redacted]")
    cleaned = re.sub(r"Bearer\s+\S+", "Bearer [redacted]", cleaned, flags=re.IGNORECASE)
    return cleaned[:500]
