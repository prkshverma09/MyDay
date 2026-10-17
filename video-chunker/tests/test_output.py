import json
from pathlib import Path

import pytest

from video_chunker.models import ChunkRecord, CutError
from video_chunker.output import (
    chunk_filename,
    existing_chunk_paths,
    manifest_path,
    prepare_output_dir,
    write_manifest,
)

def test_filename_is_zero_padded():
    assert chunk_filename(Path("/vids/holiday.mp4"), 0) == "holiday_0000.mp4"
    assert chunk_filename(Path("/vids/holiday.mp4"), 12) == "holiday_0012.mp4"

def test_filename_rejects_comma_in_stem():
    with pytest.raises(CutError, match="comma"):
        chunk_filename(Path("my,trip.mp4"), 0)

def test_prepare_refuses_existing_chunk(tmp_path):
    source = Path("holiday.mp4")
    kept = tmp_path / "holiday_0000.mp4"
    kept.write_bytes(b"original")
    with pytest.raises(CutError, match="already exists"):
        prepare_output_dir(tmp_path, source, overwrite=False)
    assert kept.read_bytes() == b"original"

def test_prepare_overwrite_clears_stale_indexes(tmp_path):
    source = Path("holiday.mp4")
    (tmp_path / "holiday_0000.mp4").write_bytes(b"a")
    (tmp_path / "holiday_0003.mp4").write_bytes(b"stale")
    (tmp_path / "holiday.chunks.json").write_text("{}", encoding="utf-8")
    (tmp_path / "other_0000.mp4").write_bytes(b"leave")
    prepare_output_dir(tmp_path, source, overwrite=True)
    assert existing_chunk_paths(tmp_path, source) == []
    assert not (tmp_path / "holiday.chunks.json").exists()
    assert (tmp_path / "other_0000.mp4").read_bytes() == b"leave"

def test_prepare_rejects_output_path_that_is_a_file(tmp_path):
    blocked = tmp_path / "not-a-dir"
    blocked.write_text("x", encoding="utf-8")
    with pytest.raises(CutError, match="not a directory"):
        prepare_output_dir(blocked, Path("holiday.mp4"), overwrite=False)

def test_write_manifest_uses_stem_name(tmp_path):
    source = Path("/vids/holiday.mp4")
    record = ChunkRecord(
        index=0,
        path=tmp_path / "holiday_0000.mp4",
        start_s=0.0,
        end_s=15.0,
    )
    path = write_manifest(tmp_path, source, 15.0, "exact", [record])
    assert path == manifest_path(tmp_path, source)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["chunk_s"] == 15.0
    assert payload["mode"] == "exact"
    assert payload["chunks"][0]["file"] == "holiday_0000.mp4"
    assert payload["chunks"][0]["start_s"] == 0.0
    assert payload["source"].endswith("holiday.mp4")
