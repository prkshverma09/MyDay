import json
import re
from pathlib import Path

from video_chunker.models import ChunkRecord, CutError

def chunk_filename(source: Path, index: int) -> str:
    if "," in source.stem:
        raise CutError(f"{source.name} has a comma in its name; rename it before chunking")
    return f"{source.stem}_{index:04d}.mp4"

def chunk_path(output_dir: Path, source: Path, index: int) -> Path:
    return output_dir / chunk_filename(source, index)

def manifest_path(output_dir: Path, source: Path) -> Path:
    if "," in source.stem:
        raise CutError(f"{source.name} has a comma in its name; rename it before chunking")
    return output_dir / f"{source.stem}.chunks.json"

def existing_chunk_paths(output_dir: Path, source: Path) -> list[Path]:
    if not output_dir.is_dir():
        return []
    pattern = re.compile(rf"^{re.escape(source.stem)}_\d{{4}}\.mp4$")
    return sorted(p for p in output_dir.iterdir() if p.is_file() and pattern.match(p.name))

def prepare_output_dir(output_dir: Path, source: Path, *, overwrite: bool) -> None:
    if output_dir.exists() and not output_dir.is_dir():
        raise CutError(f"{output_dir} is not a directory")
    existing = existing_chunk_paths(output_dir, source)
    manifest = output_dir / f"{source.stem}.chunks.json" if output_dir.is_dir() else None
    if existing and not overwrite:
        raise CutError(
            f"{existing[0].name} already exists in {output_dir}; "
            f"pass --overwrite to replace chunks for {source.stem}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for path in existing:
            path.unlink()
        if manifest is not None and manifest.exists():
            manifest.unlink()

def write_manifest(
    output_dir: Path,
    source: Path,
    chunk_s: float,
    mode: str,
    chunks: list[ChunkRecord],
) -> Path:
    path = manifest_path(output_dir, source)
    payload = {
        "source": str(source.resolve()),
        "chunk_s": chunk_s,
        "mode": mode,
        "chunks": [
            {
                "index": chunk.index,
                "file": chunk.path.name,
                "path": str(chunk.path.resolve()),
                "start_s": chunk.start_s,
                "end_s": chunk.end_s,
            }
            for chunk in chunks
        ],
    }
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path
