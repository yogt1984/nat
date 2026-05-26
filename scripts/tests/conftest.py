"""Shared test configuration — sets up import paths for all tests.

With `pip install -e scripts/`, this file is technically unnecessary,
but it serves as a fallback for running tests without the package installed.
"""

import sys
from pathlib import Path

# Ensure scripts/ is on the import path (for running without pip install -e)
SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
