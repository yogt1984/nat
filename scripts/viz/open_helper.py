"""Cross-platform file opener for viz outputs.

Shared by the visualization commands (`nat viz render`, `nat viz3d`, `nat 15m
viz`) so `--open` behaves identically everywhere and degrades gracefully to
"printed path only" when no opener is available (e.g. headless servers).
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


def open_path(path) -> bool:
    """Open ``path`` with the platform's default handler.

    Returns True if an opener was launched, False otherwise (caller should then
    just rely on the printed path). Never raises.
    """
    p = Path(path)
    if not p.exists():
        return False

    try:
        if sys.platform == "darwin" and shutil.which("open"):
            cmd = ["open", str(p)]
        elif sys.platform.startswith("win"):
            # os.startfile is Windows-only; use it directly.
            import os
            os.startfile(str(p))  # type: ignore[attr-defined]
            return True
        elif shutil.which("xdg-open"):
            cmd = ["xdg-open", str(p)]
        else:
            return False
        # Detach: don't block the CLI on the viewer process.
        subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False


def open_all(paths) -> int:
    """Open several paths; return how many openers launched."""
    return sum(1 for p in paths if open_path(p))
