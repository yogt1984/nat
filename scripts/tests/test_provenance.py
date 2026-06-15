"""Contract test for scripts/provenance.py (plan T2).

get_provenance() shape + git stamp; data_fingerprint() deterministic,
order-independent, size-sensitive, date-windowed.
"""

from __future__ import annotations

from provenance import data_fingerprint, get_provenance


def test_get_provenance_shape():
    p = get_provenance()
    assert {"git_sha", "git_sha_full", "branch", "dirty", "generated_at"} <= set(p)
    assert isinstance(p["dirty"], bool)
    # tests run inside the repo -> a short sha is present
    assert p["git_sha"] and len(p["git_sha"]) >= 7


def test_get_provenance_caches_git_but_restamps_time():
    a, b = get_provenance(), get_provenance()
    assert a["git_sha"] == b["git_sha"]          # cached git facts
    assert a["generated_at"] <= b["generated_at"]  # fresh timestamp each call


def _mk(root, rel, content: bytes):
    f = root / rel
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_bytes(content)
    return f


def test_fingerprint_deterministic(tmp_path):
    _mk(tmp_path, "2026-06-13/a.parquet", b"aaa")
    _mk(tmp_path, "2026-06-13/b.parquet", b"bbbb")
    fp = data_fingerprint(tmp_path)
    assert fp == data_fingerprint(tmp_path)
    assert len(fp) == 16


def test_fingerprint_size_sensitive(tmp_path):
    _mk(tmp_path, "d/a.parquet", b"aa")
    before = data_fingerprint(tmp_path)
    _mk(tmp_path, "d/a.parquet", b"aaaaaa")      # same path, bigger
    assert data_fingerprint(tmp_path) != before


def test_fingerprint_date_window(tmp_path):
    _mk(tmp_path, "2026-06-10/a.parquet", b"a")
    _mk(tmp_path, "2026-06-17/b.parquet", b"b")
    full = data_fingerprint(tmp_path)
    windowed = data_fingerprint(tmp_path, start_date="2026-06-15", end_date="2026-06-20")
    assert windowed != full                       # Jun-10 file excluded
    assert data_fingerprint(tmp_path, start_date="2026-06-01", end_date="2026-06-30") == full


def test_fingerprint_empty_is_stable(tmp_path):
    assert data_fingerprint(tmp_path) == data_fingerprint(tmp_path)
