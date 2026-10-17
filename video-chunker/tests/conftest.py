import shutil
import subprocess
import sys
from pathlib import Path

import pytest

# pytest puts tests/ on sys.path, not the project root, so `import tests` fails
# unless this directory is a package. Keep the brief's `from tests.conftest` import.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None

def make_sample(path: Path, duration: float) -> None:
    subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", f"testsrc=size=160x120:rate=25:duration={duration}",
            "-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=44100:duration={duration}",
            "-shortest",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            str(path),
        ],
        check=True,
    )

requires_ffmpeg = pytest.mark.skipif(
    not ffmpeg_available(),
    reason="ffmpeg and ffprobe are required",
)
