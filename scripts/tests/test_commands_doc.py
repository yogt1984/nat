"""DOCS-3 — the CLI reference must not drift from the CLI.

`docs/commands.md` was hand-maintained until 2026-08-07, by which point **26 command groups
were missing** and the headline count was stale by 80 (`~260` against 340). A reference that
disagrees with the CLI is worse than none, because it is trusted — someone reads it, does
not find `nat xs`, and concludes the feature does not exist.

The fix is generation rather than discipline: `scripts/ops/gen_commands_doc.py` derives the
doc from the live argparse tree. This test is what makes that stick.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_commands_doc_is_not_stale():
    """Regenerating must be a no-op. If it isn't, the doc lies about the CLI."""
    out = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "ops" / "gen_commands_doc.py"), "--check"],
        cwd=str(ROOT), capture_output=True, text=True,
    )
    assert out.returncode == 0, (
        "docs/commands.md is stale — regenerate with "
        "`python scripts/ops/gen_commands_doc.py`.\n" + out.stdout + out.stderr
    )


def test_the_doc_declares_it_is_generated():
    """A hand-edit is the failure mode; the file has to say so on its face."""
    text = (ROOT / "docs" / "commands.md").read_text()
    assert "Generated" in text and "Do not edit by hand" in text


def test_the_headline_count_matches_the_cli():
    """The number that was wrong by 80 for months."""
    import json
    live = json.loads(subprocess.run([str(ROOT / "nat"), "--json", "commands"],
                                     cwd=str(ROOT), capture_output=True,
                                     text=True).stdout)["count"]
    assert f"**{live} commands**" in (ROOT / "docs" / "commands.md").read_text()
