"""Persistence tests: migration idempotence, JSON + index round-trip,
JSON survival when the DB write fails."""

import json
from pathlib import Path

from data.state import StateStore

from processes import get_process
from processes.base import get_provenance
from processes.persistence import list_results, load_result, save_result
from processes.synthetic import make_planted_frame, make_test_context


def _run_result():
    df = make_planted_frame(n=600, ic=0.25, horizon=4)
    proc = get_process("ic_horizon", min_breakeven_bps=0.0)
    result = proc.evaluate(df, make_test_context())
    result.provenance = get_provenance()
    result.data = {"dir": "synthetic", "start_date": "2026-01-01",
                   "end_date": "2026-01-07", "n_rows": 600, "n_bars": 600,
                   "fingerprint": None}
    return result


def test_migration_idempotent(tmp_path):
    db = tmp_path / "nat.db"
    s1 = StateStore(db)
    s1.close()
    s2 = StateStore(db)  # re-open re-runs _run_migrations — must be a no-op
    rows = s2._conn.execute(
        "SELECT name FROM _migrations WHERE name LIKE 'create_process_results%'"
    ).fetchall()
    s2.close()
    assert {r["name"] for r in rows} == {
        "create_process_results", "create_process_results_index",
    }


def test_save_and_round_trip(tmp_path):
    result = _run_result()
    out_dir = tmp_path / "processes"
    db = tmp_path / "nat.db"

    json_path = save_result(result, out_dir=out_dir, db_path=db)
    assert json_path.exists()

    record = load_result(result.run_id, out_dir=out_dir)
    assert record["run_id"] == result.run_id
    assert record["schema_version"] == 1
    assert record["provenance"]["generated_at"]
    assert record["findings"]

    rows = list_results(db_path=db)
    assert len(rows) == 1
    row = rows[0]
    assert row["run_id"] == result.run_id
    assert row["process"] == "ic_horizon"
    assert row["n_tested"] == record["summary"]["n_tested"]
    assert row["json_path"] == str(json_path)

    # Filters
    assert list_results(process="ic_horizon", db_path=db)
    assert not list_results(process="mi_ksg", db_path=db)
    assert list_results(symbol="SYN", db_path=db)


def test_insert_or_replace_same_run_id(tmp_path):
    result = _run_result()
    out_dir, db = tmp_path / "p", tmp_path / "nat.db"
    save_result(result, out_dir=out_dir, db_path=db)
    save_result(result, out_dir=out_dir, db_path=db)
    assert len(list_results(db_path=db)) == 1


def test_json_survives_db_failure(tmp_path):
    result = _run_result()
    out_dir = tmp_path / "processes"
    bad_db = tmp_path / "not_a_dir" / "nested"  # parent created, but...
    bad_db.mkdir(parents=True)                  # ...db path IS a directory -> sqlite fails

    json_path = save_result(result, out_dir=out_dir, db_path=bad_db)
    assert json_path.exists()
    assert json.loads(json_path.read_text())["run_id"] == result.run_id


def test_json_write_is_atomic(tmp_path):
    result = _run_result()
    out_dir = tmp_path / "processes"
    save_result(result, out_dir=out_dir, db_path=None)
    assert not list(Path(out_dir).glob("*.tmp"))
