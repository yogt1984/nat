"""
Process result persistence: full JSON records + SQLite index.

Full ProcessResult records are written atomically (write-then-rename, the
`research_output` pattern) to data/research/processes/{run_id}.json; derived
series from transform processes go to data/research/processes/derived/.
A summary row lands in the `process_results` index table in nat.db so runs
are queryable across time (`nat process results`) even if JSON is pruned.

The SQLite index write is best-effort and non-fatal — a locked or missing
DB never loses the JSON record (mirrors `agent.research_output._get_store`).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from agent.research_output import _write_record

from .base import ProcessResult

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUT_DIR = ROOT / "data" / "research" / "processes"
DEFAULT_DB_PATH = ROOT / "data" / "nat.db"


def save_result(
    result: ProcessResult,
    out_dir: Path | str = DEFAULT_OUT_DIR,
    db_path: Path | str | None = DEFAULT_DB_PATH,
) -> Path:
    """Persist one run: JSON record (authoritative) + index row (best-effort).

    Returns the JSON path.
    """
    out_dir = Path(out_dir)
    record = result.to_dict()
    _write_record(out_dir, result.run_id, record)
    json_path = out_dir / f"{result.run_id}.json"

    if db_path is not None:
        try:
            from data.state import StateStore
            store = StateStore(Path(db_path))
            try:
                top = (result.summary.get("top") or [{}])[0]
                store.insert_process_result({
                    "run_id": result.run_id,
                    "process": result.process,
                    "kind": result.kind,
                    "symbol": result.symbol,
                    "timeframe": result.timeframe,
                    "start_date": result.data.get("start_date"),
                    "end_date": result.data.get("end_date"),
                    "n_tested": result.summary.get("n_tested", 0),
                    "n_informative": result.summary.get("n_informative", 0),
                    "top_feature": top.get("feature"),
                    "top_metric": top.get("metric"),
                    "top_value": top.get("value"),
                    "json_path": str(json_path),
                    "git_sha": result.provenance.get("git_sha"),
                    "data_fingerprint": result.data.get("fingerprint"),
                    "created_at": result.provenance.get("generated_at"),
                })
            finally:
                store.close()
        except Exception:
            log.warning("process_results index write failed (JSON record kept: %s)",
                        json_path, exc_info=True)

    return json_path


def save_derived(
    result: ProcessResult,
    derived_df,
    out_dir: Path | str = DEFAULT_OUT_DIR,
) -> Path:
    """Persist a transform process's derived series as parquet."""
    derived_dir = Path(out_dir) / "derived"
    derived_dir.mkdir(parents=True, exist_ok=True)
    path = derived_dir / f"{result.run_id}.parquet"
    derived_df.to_parquet(path, index=False)
    return path


def load_result(run_id: str, out_dir: Path | str = DEFAULT_OUT_DIR) -> dict:
    """Load one full JSON record by run id."""
    path = Path(out_dir) / f"{run_id}.json"
    with open(path) as f:
        return json.load(f)


def list_results(
    process: Optional[str] = None,
    symbol: Optional[str] = None,
    limit: int = 50,
    db_path: Path | str = DEFAULT_DB_PATH,
) -> list[dict]:
    """Query the SQLite index, newest first. Empty list if no DB yet."""
    db_path = Path(db_path)
    if not db_path.exists():
        return []
    from data.state import StateStore
    store = StateStore(db_path)
    try:
        return store.list_process_results(process=process, symbol=symbol, limit=limit)
    finally:
        store.close()
