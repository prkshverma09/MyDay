"""Write and read Reminisce rows in this team's VastDB bucket.

The shared VSS tables are never created, dropped, or rewritten.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from reminisce.settings import VAST_STATUS_PATH, Settings, load_settings, redact
from reminisce.store import Clip, clip_from_dict

SCHEMA_NAME = "reminisce"
TABLE_NAME = "clips"


def _status(ok: bool, detail: str) -> None:
    VAST_STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    VAST_STATUS_PATH.write_text(
        json.dumps({"ok": ok, "detail": redact(detail), "schema": SCHEMA_NAME, "table": TABLE_NAME})
    )


def _endpoint(settings: Settings) -> str:
    endpoint = settings.s3_endpoint
    if endpoint.startswith(("http://", "https://")):
        return endpoint
    return f"http://{endpoint}"


def _connect(settings: Settings):
    import logging
    import urllib3
    import vastdb

    logging.getLogger("vastdb").setLevel(logging.ERROR)
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    return vastdb.connect(
        endpoint=_endpoint(settings),
        access=settings.access_key,
        secret=settings.secret_key,
        ssl_verify=False,
    )


def _arrow_schema():
    import pyarrow as pa

    return pa.schema(
        [
            ("clip_id", pa.utf8()),
            ("owner_id", pa.utf8()),
            ("recorded_at", pa.utf8()),
            ("duration_s", pa.float64()),
            ("cosmos_description", pa.utf8()),
            ("activity_tags", pa.utf8()),
            ("embedding_id", pa.utf8()),
            ("embedding_json", pa.utf8()),
            ("shared_with", pa.utf8()),
            ("source_citation", pa.utf8()),
            ("media_relpath", pa.utf8()),
            ("description_source", pa.utf8()),
            ("inserted_at", pa.utf8()),
        ]
    )


def _row(clip: Clip) -> dict:
    return {
        "clip_id": clip.clip_id,
        "owner_id": clip.owner_id,
        "recorded_at": clip.recorded_at,
        "duration_s": float(clip.duration_s),
        "cosmos_description": clip.cosmos_description,
        "activity_tags": ",".join(clip.activity_tags),
        "embedding_id": clip.embedding_id,
        "embedding_json": json.dumps(clip.embedding),
        "shared_with": ",".join(clip.shared_with),
        "source_citation": clip.source_citation,
        "media_relpath": clip.media_relpath,
        "description_source": clip.description_source,
        "inserted_at": datetime.now(timezone.utc).isoformat(),
    }


def mirror_clip(clip: Clip, settings: Settings | None = None) -> None:
    """Insert one clip row. The latest inserted_at wins on read."""
    settings = settings or load_settings()
    if not settings.vast_ready:
        _status(False, "VAST credentials are not set")
        raise RuntimeError("VAST credentials are not set")
    import pyarrow as pa

    columns = _arrow_schema()
    arrow = pa.table({key: [value] for key, value in _row(clip).items()}, schema=columns)
    try:
        session = _connect(settings)
        with session.transaction() as tx:
            bucket = tx.bucket(settings.vastdb_bucket)
            schema = bucket.schema(SCHEMA_NAME, fail_if_missing=False)
            if schema is None:
                schema = bucket.create_schema(SCHEMA_NAME, fail_if_exists=False)
            table = schema.table(TABLE_NAME, fail_if_missing=False)
            if table is None:
                table = schema.create_table(TABLE_NAME, columns=columns, fail_if_exists=False)
            table.insert(arrow)
    except Exception as exc:
        _status(False, f"{type(exc).__name__}: {exc}")
        raise
    _status(True, f"mirrored {clip.clip_id}")


def load_clips(settings: Settings | None = None) -> list[Clip]:
    """Return the latest row for each clip_id."""
    settings = settings or load_settings()
    if not settings.vast_ready:
        return []
    session = _connect(settings)
    with session.transaction() as tx:
        schema = tx.bucket(settings.vastdb_bucket).schema(SCHEMA_NAME, fail_if_missing=False)
        if schema is None:
            return []
        table = schema.table(TABLE_NAME, fail_if_missing=False)
        if table is None:
            return []
        columns = [field.name for field in _arrow_schema()]
        rows = table.select(columns=columns).read_all().to_pylist()
    latest: dict[str, dict] = {}
    for row in rows:
        current = latest.get(row["clip_id"])
        if current is None or row["inserted_at"] >= current["inserted_at"]:
            latest[row["clip_id"]] = row
    clips = []
    for row in latest.values():
        row = dict(row)
        row["embedding"] = row.pop("embedding_json", "[]")
        clips.append(clip_from_dict(row))
    clips.sort(key=lambda clip: clip.recorded_at)
    return clips


def ping(settings: Settings | None = None) -> str:
    settings = settings or load_settings()
    if not settings.vast_ready:
        return "missing credentials"
    session = _connect(settings)
    with session.transaction() as tx:
        bucket = tx.bucket(settings.vastdb_bucket)
        schema = bucket.schema(SCHEMA_NAME, fail_if_missing=False)
    if schema is None:
        return "connected table=not_created"
    return "connected table=ready"
