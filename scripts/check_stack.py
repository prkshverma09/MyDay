#!/usr/bin/env python3
"""Check Cosmos, Embed, Canary, W&B, and VAST without printing secrets."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import requests

from reminisce.models import embed_text
from reminisce.settings import load_settings, redact
from reminisce import vast_index


def main() -> None:
    settings = load_settings()
    failed = False

    try:
        response = requests.get(
            f"{settings.cosmos_reason_url.rstrip('/')}/v1/models",
            headers={"Authorization": f"Bearer {settings.gpu_bearer_token}"},
            timeout=20,
        )
        response.raise_for_status()
        model = response.json()["data"][0]["id"]
        print(f"cosmos: ok model={model}", flush=True)
    except Exception as exc:
        failed = True
        print(f"cosmos: fail {redact(str(exc))}", flush=True)

    try:
        vector, model = embed_text("a person walking", settings)
        print(f"embed: ok dim={len(vector)} model={model}", flush=True)
    except Exception as exc:
        failed = True
        print(f"embed: fail {redact(str(exc))}", flush=True)

    try:
        response = requests.get(
            f"{settings.canary_url.rstrip('/')}/v1/health/ready",
            headers={"Authorization": f"Bearer {settings.gpu_bearer_token}"},
            timeout=20,
        )
        print(f"canary: {'ok' if response.status_code == 200 else 'fail'} http={response.status_code}", flush=True)
    except Exception as exc:
        print(f"canary: optional fail {redact(str(exc))}", flush=True)

    if not settings.wandb_ready:
        failed = True
        print("wandb: fail missing api key or project", flush=True)
    else:
        try:
            response = requests.get(
                "https://api.inference.wandb.ai/v1/models",
                headers={
                    "Authorization": f"Bearer {settings.wandb_api_key}",
                    "OpenAI-Project": f"{settings.wandb_team}/{settings.wandb_project}",
                },
                timeout=30,
            )
            response.raise_for_status()
            ids = [item.get("id") for item in response.json().get("data", [])]
            present = settings.llm_model in ids
            print(f"wandb: ok model={settings.llm_model} listed={present}", flush=True)
            if not present:
                failed = True
        except Exception as exc:
            failed = True
            print(f"wandb: fail {redact(str(exc))}", flush=True)

    try:
        print(f"vast: {vast_index.ping(settings)}", flush=True)
    except Exception as exc:
        failed = True
        print(f"vast: fail {redact(str(exc))}", flush=True)

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
