"""Shared test configuration.

With `pip install -e scripts/`, Python packaging handles imports.
This file is kept as a fallback for running tests without the editable install.
"""

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

EXPLORATION_DIR = SCRIPTS_DIR / "exploration"
if EXPLORATION_DIR.exists() and str(EXPLORATION_DIR) not in sys.path:
    sys.path.insert(0, str(EXPLORATION_DIR))
