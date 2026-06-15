"""Provenance — git SHA + data fingerprint for reproducible research (plan T2).

The canonical source (with `scripts/utils/costs.py` it completes T2's shared
foundations) for:
  - ``get_provenance()`` -> {git_sha, git_sha_full, branch, dirty, generated_at}
  - ``data_fingerprint(data_dir, start_date, end_date)`` -> cheap deterministic
    sha256 over the (relative-path, byte-size) of the parquet inputs.

Several modules already *expect* this module and fall back to inline logic until
it lands — `scripts/processes/base.py:get_provenance`,
`scripts/processes/runner.py:_data_fingerprint`,
`scripts/signal_lifecycle.py:_provenance`. Creating it unifies them automatically.
Shared with P-track P1.5 (preprint reproducibility).
"""

from __future__ import annotations

import hashlib
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent      # repo root
_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
_GIT_CACHE: dict | None = None                       # git facts are fixed per process


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], cwd=_ROOT,
            capture_output=True, text=True, timeout=5,
        )
        return (out.stdout.strip() or None) if out.returncode == 0 else None
    except Exception:
        return None


def get_provenance(*, refresh: bool = False) -> dict:
    """Reproducibility stamp: git SHA (short + full), branch, dirty flag, timestamp.

    The git facts are computed once per process and cached (they don't change
    mid-run); only ``generated_at`` is re-stamped each call. Pass ``refresh=True``
    to recompute (e.g. after a commit within the same process).
    """
    global _GIT_CACHE
    if _GIT_CACHE is None or refresh:
        _GIT_CACHE = {
            "git_sha": _git("rev-parse", "--short", "HEAD"),
            "git_sha_full": _git("rev-parse", "HEAD"),
            "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(_git("status", "--porcelain")),
        }
    return {**_GIT_CACHE, "generated_at": datetime.now(timezone.utc).isoformat()}


def _file_date(path: Path) -> str | None:
    m = _DATE_RE.search(str(path))
    return m.group(0) if m else None


def data_fingerprint(
    data_dir: Path | str,
    start_date: str | None = None,
    end_date: str | None = None,
    *,
    pattern: str = "**/*.parquet",
) -> str:
    """Cheap, deterministic fingerprint of a parquet input set.

    sha256 over the sorted ``(relative-path, byte-size)`` of every matching file,
    optionally restricted to files whose path carries a ``YYYY-MM-DD`` within
    ``[start_date, end_date]`` (ISO dates compare lexically). Same file set -> same
    16-char digest, independent of enumeration order and absolute location. Uses
    size (not content) for speed — matches the process-runner fingerprint it
    replaces; a rewrite that changes a file's size changes the digest.
    """
    root = Path(data_dir)
    h = hashlib.sha256()
    for f in sorted(root.glob(pattern)):
        rel = f.relative_to(root)
        d = _file_date(rel)
        if d is not None:
            if start_date and d < start_date:
                continue
            if end_date and d > end_date:
                continue
        try:
            size = f.stat().st_size
        except OSError:
            continue
        h.update(f"{rel}:{size}\n".encode())
    return h.hexdigest()[:16]
