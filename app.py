"""Previous entry point. The UI lives in frontend/app.py."""

from __future__ import annotations

import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).resolve().parent / "frontend" / "app.py"), run_name="__main__")
