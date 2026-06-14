# P1.5 — Data Provenance Tracking for Reproducibility

**Phase**: P1 — Preprint Completion
**Priority**: 3 (shared with Q1.4)
**Status**: NOT STARTED
**Effort**: ~4h (subset of Arch-p.3)
**Depends on**: None

---

## Objective

Attach git SHA + data fingerprint to every research output so that any published result can be traced back to the exact code version and data snapshot that produced it. Required for PhD-grade reproducibility.

## Context

Academic papers require reproducible results. Currently, research outputs (IC scans, walk-forward results, alpha screens) have no provenance metadata. If a reviewer asks "can you reproduce Table 3?", there's no way to identify which code version and data produced those numbers.

This is a subset of Arch-p.3 that directly serves the PhD path. It also benefits the quant path (backtest provenance for debugging regressions).

## Prerequisites

- None (can implement immediately)

## Scope

**In scope**:
- `git rev-parse HEAD` attached to every research output record
- SHA-256 fingerprint of input Parquet files
- Provenance metadata in JSON research output and backtest results
- `data_version` field format: `{git_sha_short}_{data_hash_short}`

**Out of scope**:
- Full Arch-p.3 (pyproject.toml, script reorg — separate task Q1.4)
- Version control of data files themselves (too large for git)
- Containerized reproducibility (Docker snapshot)

## Steps

1. Create `scripts/provenance.py`:
   ```python
   def get_provenance() -> dict:
       git_sha = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()
       return {
           'git_sha': git_sha,
           'git_sha_short': git_sha[:8],
           'timestamp': datetime.utcnow().isoformat(),
           'python_version': sys.version,
       }
   
   def data_fingerprint(parquet_paths: list[Path]) -> str:
       h = hashlib.sha256()
       for p in sorted(parquet_paths):
           h.update(p.name.encode())
           h.update(str(p.stat().st_size).encode())
       return h.hexdigest()[:12]
   ```
2. Integrate into `scripts/agent/research_output.py`:
   - Add `provenance` field to every hypothesis record
   - Add `data_fingerprint` from input Parquet files
3. Integrate into backtest output:
   - `scripts/backtest/walk_forward.py` — add provenance to result JSON
4. Integrate into alpha screening output:
   - `reports/alpha_screen.json` — add provenance header
5. Create verification script:
   - Given a provenance record, check if current git SHA matches
   - Warn if code has changed since the result was produced

## Acceptance Criteria

- [ ] Every JSON output from research agents contains `provenance.git_sha`
- [ ] Every JSON output contains `provenance.data_fingerprint`
- [ ] `data_fingerprint` is deterministic: same input files → same hash
- [ ] `scripts/provenance.py` exists with `get_provenance()` and `data_fingerprint()`
- [ ] Backtest results include provenance metadata
- [ ] Verification script can check: "was this result produced with the current code?"
- [ ] No existing tests break

## Testing / Verification

```bash
# 1. Provenance module works
python3 -c "
from scripts.provenance import get_provenance, data_fingerprint
from pathlib import Path

prov = get_provenance()
print(f'Git SHA: {prov[\"git_sha_short\"]}')
assert len(prov['git_sha']) == 40

# Test fingerprint determinism
paths = list(Path('data/features').glob('**/*.parquet'))[:5]
if paths:
    fp1 = data_fingerprint(paths)
    fp2 = data_fingerprint(paths)
    assert fp1 == fp2, 'Fingerprint not deterministic'
    print(f'Data fingerprint: {fp1}')
"

# 2. Research output includes provenance
python3 -c "
import json
from pathlib import Path
files = list(Path('data/research/hypotheses').glob('*.json'))
if files:
    with open(files[-1]) as f:
        record = json.load(f)
    assert 'provenance' in record, 'Missing provenance'
    assert 'git_sha' in record['provenance'], 'Missing git_sha'
    print(f'Provenance: {record[\"provenance\"]}')"

# 3. Verification script
python3 scripts/provenance.py --verify reports/alpha_screen.json
# Should print: "Result produced with git SHA abc123, current SHA: abc123 [MATCH]"

# 4. Existing tests pass
nat test agent
```

## Key Files

- `scripts/provenance.py` — provenance module (new)
- `scripts/agent/research_output.py` — add provenance to records
- `scripts/backtest/walk_forward.py` — add provenance to results

## References

- Arch-p.3 provenance spec: `docs/Arch-p.3.md`
- Shared with Q1.4 (cost model unification)
