from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa

PROTECTED = frozenset({"vss-collection", "vss-prompts-events"})
SCHEMA_NAME = "myday"


def write_vastdb(store_dir: Path) -> dict:
    """Insert the three JSON stores into schema `myday` on this team's bucket."""

    import os

    import vastdb

    env = _team_env()
    access = env.get("VAST_ACCESS_KEY") or env.get("ACCESS_KEY", "")
    secret = env.get("VAST_SECRET_KEY") or env.get("SECRET_KEY", "")
    endpoint = env.get("S3_ENDPOINT", "")
    bucket = env.get("VASTDB_BUCKET", "")
    if not access or not secret or not endpoint or not bucket:
        raise RuntimeError("team config is missing the VastDB endpoint or this team's bucket")

    tables = {
        "memory_entries": json.loads((store_dir / "memory_entries.json").read_text(encoding="utf-8")),
        "last_seen": json.loads((store_dir / "last_seen.json").read_text(encoding="utf-8"))["objects"],
        "coverage": _coverage_rows(store_dir),
    }
    if not endpoint.startswith(("http://", "https://")):
        endpoint = f"http://{endpoint}"
    session = vastdb.connect(endpoint=endpoint, access=access, secret=secret, ssl_verify=False)
    written = {}
    with session.transaction() as tx:
        held = tx.bucket(bucket)
        schema = held.schema(SCHEMA_NAME, fail_if_missing=False)
        if schema is None:
            schema = held.create_schema(SCHEMA_NAME, fail_if_exists=False)
        for name, rows in tables.items():
            if name in PROTECTED:
                raise RuntimeError(f"refusing to write protected table {name}")
            flat = [_flatten(row) for row in rows]
            written[name] = len(flat)
            if not flat:
                continue
            arrow_schema = _arrow_schema(flat)
            table = schema.table(name, fail_if_missing=False)
            if table is None:
                table = schema.create_table(name, columns=arrow_schema, fail_if_exists=False)
            table.insert(pa.Table.from_pylist(flat, schema=arrow_schema))
    return {"schema": SCHEMA_NAME, "tables": written}


def _team_env() -> dict[str, str]:
    import os

    override = os.environ.get("VAST_ENV_FILE")
    if override:
        path = Path(override)
    else:
        configs = sorted(Path("/config").glob("*.config"))
        if len(configs) != 1:
            raise RuntimeError(f"expected exactly one /config/*.config; found {len(configs)}")
        path = configs[0]
    env: dict[str, str] = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        env[key.strip()] = value.strip()
    return env


def _coverage_rows(store_dir: Path) -> list[dict]:
    payload = json.loads((store_dir / "coverage.json").read_text(encoding="utf-8"))
    rows = []
    for day in payload.get("days", []):
        for question_id, question in day["questions"].items():
            extra = {k: v for k, v in question.items() if k not in {"recorded", "clip_ids"}}
            rows.append(
                {
                    "day": day["day"],
                    "question_id": question_id,
                    "recorded": bool(question.get("recorded")),
                    "clip_ids": json.dumps(question.get("clip_ids") or []),
                    "detail": json.dumps(extra),
                    "span_start": day.get("span_start") or "",
                    "span_end": day.get("span_end") or "",
                }
            )
    return rows


def _flatten(row: dict) -> dict:
    flat = {}
    for key, value in row.items():
        if isinstance(value, (dict, list)):
            flat[key] = json.dumps(value)
        elif isinstance(value, bool) or value is None or isinstance(value, (int, float, str)):
            flat[key] = value
        else:
            flat[key] = str(value)
    return flat


def _arrow_schema(rows: list[dict]):
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    fields = []
    for key in keys:
        sample = next((row[key] for row in rows if row.get(key) is not None), None)
        if isinstance(sample, bool):
            fields.append((key, pa.bool_()))
        elif isinstance(sample, int) and not isinstance(sample, bool):
            fields.append((key, pa.int64()))
        elif isinstance(sample, float):
            fields.append((key, pa.float32()))
        else:
            fields.append((key, pa.utf8()))
    return pa.schema(fields)
