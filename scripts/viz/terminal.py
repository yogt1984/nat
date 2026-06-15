"""Terminal-first render primitives for the `nat viz` group (plan T7 / NAT3).

Pure helpers — sparklines, IC coloring, bars, a live-refresh loop — shared by every
`nat viz <unit>` command. No matplotlib; this is the in-terminal layer (the PNG layer
lives in the sibling viz modules used by `nat visualize`).
"""

from __future__ import annotations

import math
import os
import time
from typing import Callable, Optional, Sequence

# ANSI
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
GREEN, RED, YELLOW, BLUE, GREY = (
    "\033[32m", "\033[31m", "\033[33m", "\033[34m", "\033[90m")

_SPARK = "▁▂▃▄▅▆▇█"


def _isnan(v) -> bool:
    try:
        return v is None or math.isnan(v)
    except (TypeError, ValueError):
        return False


def sparkline(values: Sequence[float]) -> str:
    """Unicode sparkline of *values*. NaNs render as a gap; empty/all-NaN -> ''."""
    vals = list(values)
    finite = [v for v in vals if not _isnan(v)]
    if not finite:
        return ""
    lo, hi = min(finite), max(finite)
    span = hi - lo
    out = []
    for v in vals:
        if _isnan(v):
            out.append(" ")
        elif span == 0:
            out.append(_SPARK[len(_SPARK) // 2])
        else:
            idx = int((v - lo) / span * (len(_SPARK) - 1))
            out.append(_SPARK[max(0, min(len(_SPARK) - 1, idx))])
    return "".join(out)


def ic_color(ic: Optional[float]) -> str:
    """ANSI color by IC sign + magnitude (green long edge, red short, grey weak)."""
    if _isnan(ic):
        return GREY
    a = abs(ic)
    if a < 0.02:
        return GREY
    if ic > 0:
        return (BOLD + GREEN) if a >= 0.05 else GREEN
    return (BOLD + RED) if a >= 0.05 else RED


def colorize(text: str, color: str) -> str:
    return f"{color}{text}{RESET}"


def bar(value: float, lo: float = -1.0, hi: float = 1.0, width: int = 12) -> str:
    """A filled horizontal bar for *value* clamped to [lo, hi]."""
    if _isnan(value) or hi <= lo:
        return " " * width
    frac = max(0.0, min(1.0, (value - lo) / (hi - lo)))
    filled = int(round(frac * width))
    return "█" * filled + "·" * (width - filled)


def live_refresh(render: Callable[[], str], interval: float = 1.0,
                 iterations: Optional[int] = None, clear: bool = True) -> None:
    """Clear-and-redraw ``render()`` every *interval*s until Ctrl-C (or *iterations*)."""
    n = 0
    try:
        while iterations is None or n < iterations:
            if clear:
                os.system("clear" if os.name != "nt" else "cls")
            print(render())
            n += 1
            if iterations is not None and n >= iterations:
                break
            time.sleep(interval)
    except KeyboardInterrupt:
        pass
