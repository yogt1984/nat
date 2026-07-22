# Detailed Task Descriptions (Implementation Order)

All tasks from the engineering backlog with precise file paths, subtasks, and acceptance criteria. P1-1 has its own spec at `P1-1_unified_data_layer.md`.

Implementation order follows the critical path:
```
P1-1 (data layer) ──→ P1-4 (SQLite) ──→ P2-1 (structured output) ──→ P2-2 (API) ──→ P3-1 (scaffold) ──→ P3-2 (dashboard)
P1-2 (daemons)    ─┐
P1-3 (runners)    ─┤ parallel with P1-1
P1-5 (config)     ─┘
P1-6 (logging)    ── after P1-2/P1-3 (needs consolidated code)
P1-7 (cache)      ── independent, any time
P1-8 (tests)      ── after P1-2/P1-3/P1-4
```

---

## Phase 1: Foundation

---

### P1-2. Consolidate Agent Daemons

**Priority**: High | **Effort**: 2-3 days | **Depends on**: Nothing | **Parallel with**: P1-1, P1-3

#### Problem

Three daemon files share ~70% identical code:

| File | LOC | Config section |
|---|---|---|
| `scripts/agent/daemon.py` | 583 | `[agent]` |
| `scripts/agent/mf_daemon.py` | 422 | `[agent_mf]` |
| `scripts/agent/macro_daemon.py` | 420 | `[agent_macro]` |

Every daemon reimplements these identical methods with only generator names, state paths, and config keys differing:

| Duplicated method | daemon.py line | mf_daemon.py line | macro_daemon.py line |
|---|---|---|---|
| `load_config()` | 63 | 70 | 69 |
| `agent_type` property | 103 | 94 | 93 |
| `root` / `state_path` / `queue_path` / `stats_path` | 112-124 | 102-114 | 101-113 |
| `get_generator()` | 134-157 | 124-138 | 123-137 |
| `create_runner()` | 159-161 | 140-142 | 139-141 |
| `pre_execute()` | 163-168 | 144-149 | 143-148 |
| `on_fdr_reject()` | 182-184 | 151-153 | 150-152 |
| `run_monitor()` | 186-287 | 155-212 | 154-211 |
| `_remove_from_registry()` | 292-301 | 217-225 | 216-224 |
| `_compute_adaptive_ic()` | 377-400 | 227-243 | 226-242 |
| `_compute_rolling_ic()` | 402-462 | 245-321 | 244-319 |
| `print_report()` | 464-531 | 323-375 | 321-373 |
| `main()` | 542-579 | 382-418 | 380-416 |

**The only actual differences**:
1. Generator names and import paths (`agent.generators.{name}` vs `agent.generators.medium_freq.{name}` vs `agent.generators.macro.{name}`)
2. State directory (`data/agent/` vs `data/agent_mf/` vs `data/agent_macro/`)
3. TOML config section name (`agent` vs `agent_mf` vs `agent_macro`)
4. Adaptive IC floor defaults (0.10 vs 0.08 vs 0.07)
5. Runner class import (`MicrostructureRunner` vs `MediumFrequencyRunner` vs `MacroRunner`)

#### Files to modify

| File | Action |
|---|---|
| `scripts/agent/base.py` | Add shared daemon methods to `ResearchAgent` |
| `scripts/agent/daemon.py` | Reduce to thin subclass (~80 LOC) |
| `scripts/agent/mf_daemon.py` | Reduce to thin subclass (~50 LOC) |
| `scripts/agent/macro_daemon.py` | Reduce to thin subclass (~50 LOC) |

#### Subtasks

- [ ] Move `load_config()` into `ResearchAgent` base class, parameterized by `config_section` property
- [ ] Move path properties (`root`, `state_path`, `queue_path`, `stats_path`) into base class using `agent_type` as directory suffix
- [ ] Move `get_generator()` into base class with `generator_module_prefix` property (e.g., `"agent.generators"`, `"agent.generators.medium_freq"`)
- [ ] Move `run_monitor()` into base class (identical across all three)
- [ ] Move `_remove_from_registry()`, `_compute_adaptive_ic()`, `_compute_rolling_ic()` into base class
- [ ] Move `print_report()` into base class
- [ ] Move `main()` logic into base class as `cli_main()`, parameterized by agent class
- [ ] Reduce `daemon.py` `MicrostructureAgent` to: `config_section = "agent"`, `agent_type = "microstructure"`, `generator_module_prefix = "agent.generators"`, `default_generators`, `create_runner()`
- [ ] Reduce `mf_daemon.py` `MediumFrequencyAgent` to same pattern with MF-specific values
- [ ] Reduce `macro_daemon.py` `MacroAgent` to same pattern with macro-specific values
- [ ] Verify all three CLI entry points still work: `python -m scripts.agent.daemon start`, `python -m scripts.agent.mf_daemon start`, `python -m scripts.agent.macro_daemon start`
- [ ] Run `make test_agent` — all 101 tests pass

#### Acceptance criteria

- Total daemon LOC drops from 1,425 to ~600
- `base.py` grows by ~400 LOC (net reduction of ~425 LOC)
- Adding a 4th agent requires only a new TOML section + ~40 LOC subclass
- All existing CLI commands work identically
- All 101 agent tests pass

#### Tests

**File**: `scripts/tests/test_daemon_consolidation.py`

**Fixtures** (reuse existing patterns from `test_agent_base.py`):
```python
@pytest.fixture
def stub_agent(tmp_path):
    """StubAgent with tmp_path-based state — identical to test_agent_base.py pattern."""
    agent = MicrostructureAgent.__new__(MicrostructureAgent)
    agent._root = tmp_path
    agent._config = load_test_config("agent")
    return agent

@pytest.fixture
def all_agents(tmp_path):
    """Instantiate all 3 daemon subclasses in tmp_path to verify shared behavior."""
    return [
        MicrostructureAgent(config_path=tmp_path / "agent.toml"),
        MediumFrequencyAgent(config_path=tmp_path / "agent.toml"),
        MacroAgent(config_path=tmp_path / "agent.toml"),
    ]
```

**Test functions**:

| Test | Description | Assertion |
|---|---|---|
| `test_config_section_routing` | Each subclass loads from correct TOML section | micro→`[agent]`, mf→`[agent_mf]`, macro→`[agent_macro]` |
| `test_path_properties_use_agent_type` | `root`, `state_path`, `queue_path` derive from `agent_type` | `agent.root == data/{agent_type}/` |
| `test_generator_module_prefix` | `get_generator("foo")` imports from correct module path | micro→`agent.generators.foo`, mf→`agent.generators.medium_freq.foo` |
| `test_run_monitor_identical_behavior` | All three agents produce same monitoring logic given same IC data | Same retirement decisions across agents |
| `test_cli_main_micro` | `cli_main(MicrostructureAgent)` parses `start`/`stop`/`status` args | No exceptions, correct agent instantiated |
| `test_cli_main_mf` | Same for MF | Same assertions |
| `test_cli_main_macro` | Same for macro | Same assertions |
| `test_subclass_loc_under_80` | Count lines in `daemon.py` post-refactor | `<= 80` |
| `test_existing_tests_pass` | Metacheck: all 101 tests from `make test_agent` still green | `subprocess.run(["make", "test_agent"]).returncode == 0` |

**Mocking**:
- `monkeypatch.setattr` on `importlib.import_module` for generator module resolution tests
- `patch.object(ResearchAgent, "load_config")` with test TOML fixture

**Run**: Add to `make test_agent` target in Makefile.

---

### P1-3. Consolidate Agent Runners

**Priority**: High | **Effort**: 1-2 days | **Depends on**: Nothing | **Parallel with**: P1-1, P1-2

#### Problem

Three runner files total 1,163 LOC with ~85% identical code:

| File | LOC | Gates | Timeframe |
|---|---|---|---|
| `scripts/agent/runner.py` | 630 | 5-gate | tick/1min |
| `scripts/agent/mf_runner.py` | 270 | 4-gate | 5min |
| `scripts/agent/macro_runner.py` | 263 | 4-gate | 1h |

`mf_runner.py` imports 6 functions from `runner.py` (line 20-27) but still reimplements:
- `_check_gates()` (mf line 230-241, macro line 223-234) — identical logic
- `_extract_features()` (mf line 243-250, macro line 236-243) — only feature list differs
- `_extract_ic_from_results()` (mf line 252-262, macro line 245-255) — identical logic
- `_load_registry()` (mf line 265-270, macro line 258-263) — only path differs

Gate threshold differences (all from `config/agent.toml`):

| Threshold | Micro | MF | Macro |
|---|---|---|---|
| `min_ic` | 0.10 | 0.08 | 0.07 |
| `min_dIC` | 0.05 | 0.03 | 0.02 |
| `fdr_q` | 0.05 | 0.05 | 0.05 |
| `min_oos_dates` | 2 | 2 | 2 |
| `min_symbols` | 2 | 2 | 2 |

Protocol difference: micro has 5 gates (includes `cost` and `walkforward`), MF/macro have 4 gates.

#### Files to modify

| File | Action |
|---|---|
| `scripts/agent/base.py` | Add configurable gate logic to `BaseRunner` |
| `scripts/agent/runner.py` | Keep as `MicrostructureRunner`, remove duplicatable methods |
| `scripts/agent/mf_runner.py` | Reduce to thin subclass (~60 LOC) |
| `scripts/agent/macro_runner.py` | Reduce to thin subclass (~60 LOC) |

#### Subtasks

- [ ] Move `_check_gates()` into `BaseRunner`, reading thresholds from config (not hardcoded)
- [ ] Move `_extract_ic_from_results()` into `BaseRunner` (identical across all three)
- [ ] Move `_load_registry()` into `BaseRunner` using `registry_path` property from daemon
- [ ] Make `_extract_features()` abstract in `BaseRunner` with per-runner feature list
- [ ] Make `steps()` in `BaseRunner` configurable: accept gate list from config (`["discovery", "temporal", "symbol", "correlation"]` vs `["discovery", "temporal", "symbol", "cost", "walkforward"]`)
- [ ] Keep shared utility functions in `runner.py`: `run_nat_cached()`, `parse_report()`, `check_ic_gate()`, `check_dIC_gate()`, `check_correlation_gate()`
- [ ] Reduce `mf_runner.py` to: `TIMEFRAME = "5min"`, `SIGNAL_FEATURES = [...]`, `steps()` returning 4-gate list
- [ ] Reduce `macro_runner.py` to: `TIMEFRAME = "1h"`, `SIGNAL_FEATURES = [...]`, `steps()` returning 4-gate list
- [ ] Run `make test_agent` — all 101 tests pass

#### Acceptance criteria

- Total runner LOC drops from 1,163 to ~500
- Gate thresholds live in config, not code
- Adding a new gate to all runners requires one base class change
- All 101 agent tests pass

#### Tests

**File**: `scripts/tests/test_runner_consolidation.py`

**Fixtures** (extend existing `StubRunner` from `test_agent_base.py`):
```python
@pytest.fixture
def micro_runner(tmp_path, monkeypatch):
    """MicrostructureRunner with patched data paths and test config."""
    monkeypatch.setenv("NAT_DATA_DIR", str(tmp_path))
    config = {"gates": {"min_ic": 0.10, "min_dIC": 0.05, "fdr_q": 0.05}}
    return MicrostructureRunner(hypothesis=stub_hypothesis(), config=config)

@pytest.fixture
def mf_runner(tmp_path, monkeypatch):
    """MediumFrequencyRunner with lower thresholds."""
    monkeypatch.setenv("NAT_DATA_DIR", str(tmp_path))
    config = {"gates": {"min_ic": 0.08, "min_dIC": 0.03, "fdr_q": 0.05}}
    return MediumFrequencyRunner(hypothesis=stub_hypothesis(), config=config)

@pytest.fixture
def macro_runner(tmp_path, monkeypatch):
    """MacroRunner with lowest thresholds."""
    monkeypatch.setenv("NAT_DATA_DIR", str(tmp_path))
    config = {"gates": {"min_ic": 0.07, "min_dIC": 0.02, "fdr_q": 0.05}}
    return MacroRunner(hypothesis=stub_hypothesis(), config=config)
```

**Test functions**:

| Test | Description | Assertion |
|---|---|---|
| `test_gate_thresholds_from_config` | Runners read thresholds from config, not hardcoded | Change config → different gate pass/fail |
| `test_check_gates_identical_logic` | Same IC data + same thresholds → same result across all 3 runners | `micro._check_gates(data) == mf._check_gates(data)` when thresholds match |
| `test_extract_ic_shared` | `_extract_ic_from_results()` produces same output from base class for all runners | All three return identical IC dict |
| `test_load_registry_path_varies` | `_load_registry()` reads from agent-specific directory | micro→`data/agent/`, mf→`data/agent_mf/` |
| `test_5_gate_protocol_micro` | Microstructure runner has `["discovery", "temporal", "symbol", "cost", "walkforward"]` | `len(micro_runner.steps()) == 5` |
| `test_4_gate_protocol_mf` | MF runner has `["discovery", "temporal", "symbol", "correlation"]` | `len(mf_runner.steps()) == 4` |
| `test_4_gate_protocol_macro` | Macro runner has same 4-gate as MF | `macro_runner.steps() == mf_runner.steps()` |
| `test_extract_features_differ` | Each runner extracts its own feature set | `micro.SIGNAL_FEATURES != mf.SIGNAL_FEATURES` |
| `test_run_full_uses_base_logic` | `run_full()` in base class drives gate sequence correctly | StubRunner pattern with `step_results=[True, True, False]` → stops at gate 3 |
| `test_backward_compat_imports` | `from scripts.agent.runner import run_nat_cached` still works | Import does not raise |

**Mocking**:
- `StubRunner(step_results=[...])` pattern from `test_agent_base.py` (line 23)
- `patch("scripts.agent.runner.run_nat_cached")` for subprocess isolation
- `monkeypatch` for environment and paths

**Run**: Add to `make test_agent` target.

---

### P1-4. Replace JSON State Files with SQLite

**Priority**: High | **Effort**: 2-3 days | **Depends on**: P1-1 (data layer for schema patterns) | **Blocks**: P2-1

#### Problem

Five independent JSON files with no coordination:

| File | Owner | Keys |
|---|---|---|
| `data/pipeline_state.json` | `PipelineState` in `pipeline_runner.py` (line 80-136) | state, timestamps, pid, health counters, decision, history[] |
| `data/alpha/pipeline_state.json` | `AlphaPipelineState` in `alpha_pipeline.py` (line 114-196) | phase, current_step, gates{}, artifacts{}, step_outputs{}, history[] |
| `data/agent/agent_state.json` | `AgentState` in `base.py` (line 42-92) | phase, cycle_count, totals, current_hypothesis, history[] |
| `data/agent_mf/agent_state.json` | Same `AgentState` class, different path | identical schema |
| `data/agent_macro/agent_state.json` | Same `AgentState` class, different path | identical schema |

Additional JSON files that should migrate:
| File | Content |
|---|---|
| `data/agent/hypotheses.json` | `HypothesisQueue` (line 44-123 in `hypothesis_queue.py`) — list of Hypothesis dicts with status, IC, gates |
| `data/agent/registry.json` | Registered signals with IC history |
| `data/agent/generator_stats.json` | Per-generator success/failure counts |
| Same pattern for `agent_mf/` and `agent_macro/` | Three copies of each |

Problems: no atomic writes (crash mid-write = corrupt state), no cross-agent queries, no queryable history.

#### Files to create

| File | Purpose |
|---|---|
| `scripts/data/state.py` | `StateStore` class wrapping SQLite |

#### Files to modify

| File | Change |
|---|---|
| `scripts/pipeline_runner.py` | Replace `PipelineState` JSON with `StateStore` |
| `scripts/alpha/alpha_pipeline.py` | Replace `AlphaPipelineState` JSON with `StateStore` |
| `scripts/agent/base.py` | Replace `AgentState` JSON with `StateStore` |
| `scripts/agent/hypothesis_queue.py` | Replace JSON-backed queue with SQLite table |

#### SQLite Schema

```sql
-- Pipeline runs (pipeline_runner + alpha_pipeline)
CREATE TABLE pipeline_runs (
    id INTEGER PRIMARY KEY,
    pipeline_type TEXT NOT NULL,       -- 'ingest', 'alpha', 'discovery'
    state TEXT NOT NULL,               -- enum: IDLE, BUILDING, INGESTING, ...
    started_at TEXT,
    finished_at TEXT,
    metadata TEXT,                     -- JSON blob for pipeline-specific fields
    error TEXT
);

-- State transitions (replaces all history[] arrays)
CREATE TABLE state_transitions (
    id INTEGER PRIMARY KEY,
    entity_type TEXT NOT NULL,         -- 'pipeline', 'agent', 'alpha'
    entity_id TEXT NOT NULL,           -- 'ingest', 'microstructure', 'mf', 'macro'
    from_state TEXT,
    to_state TEXT NOT NULL,
    message TEXT,
    timestamp TEXT NOT NULL
);

-- Agent cycles
CREATE TABLE agent_cycles (
    id INTEGER PRIMARY KEY,
    agent_type TEXT NOT NULL,          -- 'microstructure', 'mf', 'macro'
    cycle_number INTEGER NOT NULL,
    started_at TEXT,
    finished_at TEXT,
    hypotheses_tested INTEGER DEFAULT 0,
    hypotheses_passed INTEGER DEFAULT 0,
    fdr_budget_remaining REAL
);

-- Hypotheses (replaces hypotheses.json for all agents)
CREATE TABLE hypotheses (
    id TEXT PRIMARY KEY,               -- e.g., 'h_20260525_001'
    agent_type TEXT NOT NULL,
    generator TEXT NOT NULL,
    claim TEXT NOT NULL,
    status TEXT NOT NULL,              -- QUEUED, RUNNING, REGISTERED, GRAVEYARD
    priority REAL DEFAULT 0,
    ic REAL,
    p_value REAL,
    gates TEXT,                        -- JSON: per-gate results
    created_at TEXT NOT NULL,
    tested_at TEXT,
    metadata TEXT                      -- JSON: generator-specific data
);

-- Registered signals (replaces registry.json for all agents)
CREATE TABLE signals (
    id TEXT PRIMARY KEY,
    hypothesis_id TEXT REFERENCES hypotheses(id),
    agent_type TEXT NOT NULL,
    features TEXT,                     -- JSON array of feature names
    ic_current REAL,
    ic_history TEXT,                   -- JSON array of {date, ic} entries
    sharpe REAL,
    status TEXT NOT NULL,              -- ACTIVE, DECAYING, RETIRED
    registered_at TEXT NOT NULL,
    retired_at TEXT
);

-- Generator statistics (replaces generator_stats.json)
CREATE TABLE generator_stats (
    agent_type TEXT NOT NULL,
    generator TEXT NOT NULL,
    total_generated INTEGER DEFAULT 0,
    total_passed INTEGER DEFAULT 0,
    total_failed INTEGER DEFAULT 0,
    last_success_at TEXT,
    PRIMARY KEY (agent_type, generator)
);
```

#### Subtasks

- [ ] Create `scripts/data/state.py` with `StateStore` class
  - [ ] `__init__(db_path)` — create tables if not exist
  - [ ] `pipeline_transition(pipeline_type, new_state, message)` — atomic state change
  - [ ] `get_pipeline_state(pipeline_type) -> dict`
  - [ ] `agent_transition(agent_type, new_phase, message)` — atomic agent state change
  - [ ] `get_agent_state(agent_type) -> dict`
  - [ ] `push_hypothesis(hypothesis_dict)` — insert with dedup by claim
  - [ ] `pop_hypothesis(agent_type) -> dict` — highest priority queued hypothesis
  - [ ] `update_hypothesis(id, **fields)` — update status, IC, gates
  - [ ] `register_signal(signal_dict)` — insert registered signal
  - [ ] `query_hypotheses(agent_type=None, status=None, generator=None, limit=50) -> list`
  - [ ] `query_signals(agent_type=None, status=None) -> list`
  - [ ] `update_generator_stats(agent_type, generator, passed: bool)`
  - [ ] `system_status() -> dict` — one-call summary of all pipeline states, agent states, hypothesis counts
- [ ] Migrate `PipelineState` in `pipeline_runner.py` (lines 80-136) to use `StateStore`
- [ ] Migrate `AlphaPipelineState` in `alpha_pipeline.py` (lines 114-196) to use `StateStore`
- [ ] Migrate `AgentState` in `base.py` (lines 42-92) to use `StateStore`
- [ ] Migrate `HypothesisQueue` in `hypothesis_queue.py` to use `StateStore`
- [ ] Keep JSON export methods for backward compatibility with dashboard (read from SQLite, write JSON)
- [ ] Add `status` CLI: `python -m scripts.data.state status` prints all states
- [ ] Write crash-recovery test: begin transaction, kill, verify rollback
- [ ] Write unit tests for all `StateStore` methods

#### Acceptance criteria

- Single `data/nat.db` file replaces 12+ JSON files
- `python -m scripts.data.state status` outputs system-wide status
- All pipeline/agent/hypothesis operations are atomic (no corrupt state on crash)
- Dashboard still works (JSON export layer)
- `make test_agent` passes

#### Tests

**File**: `scripts/tests/test_state_store.py`

**Fixtures**:
```python
@pytest.fixture
def store(tmp_path):
    """Fresh SQLite StateStore in tmp dir."""
    return StateStore(db_path=tmp_path / "test.db")

@pytest.fixture
def populated_store(store):
    """StateStore pre-loaded with representative data."""
    store.pipeline_transition("ingest", "INGESTING", "started")
    store.agent_transition("microstructure", "EXECUTE", "cycle 42")
    store.push_hypothesis({"id": "h_001", "agent_type": "microstructure",
        "generator": "spectral", "claim": "test", "status": "QUEUED", "priority": 1.0})
    store.push_hypothesis({"id": "h_002", "agent_type": "mf",
        "generator": "funding", "claim": "test2", "status": "REGISTERED", "priority": 0.5})
    return store
```

**Test functions**:

| Test | Description | Assertion |
|---|---|---|
| `test_create_tables` | `StateStore(path)` creates all 6 tables | `SELECT name FROM sqlite_master` returns 6 table names |
| `test_pipeline_transition_atomic` | State change writes both `pipeline_runs` and `state_transitions` | Both tables have 1 row after single call |
| `test_get_pipeline_state` | Returns latest state for given pipeline type | `state["state"] == "INGESTING"` |
| `test_agent_transition` | `agent_transition()` creates agent_cycles row + state_transition | Verify cycle_number increments |
| `test_push_hypothesis_dedup` | Push same claim twice → only 1 row | `len(query_hypotheses()) == 1` |
| `test_pop_hypothesis_priority` | Pop returns highest priority QUEUED item | `popped["id"] == "h_001"` (priority 1.0) |
| `test_pop_hypothesis_updates_status` | Popped hypothesis moves to RUNNING | `query(id="h_001")["status"] == "RUNNING"` |
| `test_update_hypothesis` | `update_hypothesis("h_001", ic=0.05, status="REGISTERED")` | Fields updated in DB |
| `test_register_signal` | Insert signal with hypothesis FK | Signal queryable, hypothesis_id matches |
| `test_query_hypotheses_filters` | Filter by agent_type, status, generator | Each filter narrows correctly |
| `test_query_signals_by_status` | `query_signals(status="ACTIVE")` | Only active signals returned |
| `test_generator_stats_increment` | `update_generator_stats(micro, spectral, True)` twice | `total_passed == 2` |
| `test_system_status_summary` | `system_status()` returns all subsystems | Dict has keys: pipelines, agents, hypotheses, signals |
| `test_crash_recovery_rollback` | Begin write, simulate crash (close mid-transaction) | State unchanged on reopen |
| `test_concurrent_reads` | Multiple readers don't block each other | `threading.Thread` reads in parallel succeed |
| `test_json_export_compat` | Export to JSON matches old format | Dashboard parser accepts output |

**Mocking**:
- No mocking needed — SQLite in `tmp_path` is the test double
- `monkeypatch.setattr` on `StateStore.DB_PATH` for integration tests with other modules

**Additional file**: `scripts/tests/test_state_migration.py`

| Test | Description |
|---|---|
| `test_migrate_pipeline_state_json` | Read existing `data/pipeline_state.json`, insert into SQLite, verify equivalence |
| `test_migrate_hypotheses_json` | Read existing `data/agent/hypotheses.json`, insert all, verify counts and fields |
| `test_migrate_registry_json` | Read `data/agent/registry.json`, insert as signals, verify IC history preserved |

**Run**: `pytest scripts/tests/test_state_store.py -v` and add to `make test_agent`.

---

### P1-5. Config Inheritance and Deduplication

**Priority**: Medium | **Effort**: 1 day | **Depends on**: Nothing | **Parallel with**: P1-2

#### Problem

In `config/agent.toml`, these keys are duplicated identically across `[agent]`, `[agent_mf]`, `[agent_macro]`:

| Key | Value (identical in all 3) | Lines |
|---|---|---|
| `symbols.primary` | `["BTC", "ETH", "SOL"]` | agent:52, mf:88, macro:122 |
| `paths.data_dir` | `"data/features"` | agent:56, mf:91, macro:125 |
| `gates.fdr_q` | `0.05` | duplicated |
| `gates.min_oos_dates` | `2` | duplicated |
| `gates.min_symbols` | `2` | duplicated |
| `decay.ic_decay_ratio` | `0.5` | duplicated |
| `decay.consecutive_days_limit` | `14` | duplicated |

#### Files to modify

| File | Change |
|---|---|
| `config/agent.toml` | Add `[agent_base]` section, remove duplicates from per-agent sections |
| `scripts/agent/base.py` | Load `[agent_base]` first, merge per-agent section on top |

#### Subtasks

- [ ] Add `[agent_base]` section at top of `agent.toml` with all shared keys
- [ ] Remove duplicated keys from `[agent]`, `[agent_mf]`, `[agent_macro]` — keep only overrides
- [ ] Modify `load_config()` in `ResearchAgent` (after P1-2, this is in base class) to:
  1. Load `[agent_base]` as defaults
  2. Deep-merge per-agent section on top
- [ ] Add config validation: log warning for unknown keys, error for missing required keys
- [ ] Add comment block at top of `agent.toml` documenting the inheritance model
- [ ] Verify all three agents load correct merged config (write test)

#### Acceptance criteria

- `agent.toml` shrinks by ~40 lines
- Adding a 4th agent requires only keys that differ from base
- Unknown key in config → warning in log
- All agent tests pass

#### Tests

**File**: `scripts/tests/test_config_inheritance.py`

**Fixtures**:
```python
@pytest.fixture
def config_file(tmp_path):
    """Write a test agent.toml with [agent_base] + per-agent overrides."""
    toml_content = '''
    [agent_base]
    symbols = ["BTC", "ETH", "SOL"]
    data_dir = "data/features"
    fdr_q = 0.05
    min_oos_dates = 2

    [agent]
    min_ic = 0.10
    generators = ["spectral", "entropy"]

    [agent_mf]
    min_ic = 0.08
    generators = ["funding", "trend"]

    [agent_macro]
    min_ic = 0.07
    generators = ["macro_flow"]
    '''
    p = tmp_path / "agent.toml"
    p.write_text(toml_content)
    return p
```

**Test functions**:

| Test | Description | Assertion |
|---|---|---|
| `test_base_keys_inherited` | Micro agent config has `symbols` from `[agent_base]` | `config["symbols"] == ["BTC", "ETH", "SOL"]` |
| `test_per_agent_overrides` | `min_ic` differs per agent | micro=0.10, mf=0.08, macro=0.07 |
| `test_deep_merge_nested` | Nested dict in base merges with per-agent nested dict | Verify nested key accessible |
| `test_unknown_key_warns` | Add `bogus_key = 123` to TOML → log warning | `caplog` contains "unknown key" |
| `test_missing_required_errors` | Remove `symbols` from both base and per-agent → error | Raises `ConfigError` |
| `test_override_base_value` | Per-agent `fdr_q = 0.10` overrides base `fdr_q = 0.05` | `config["fdr_q"] == 0.10` |
| `test_all_agents_get_base` | Load config for all 3 agents → all have `data_dir` | All three configs have same `data_dir` |

**Mocking**: None — test directly with TOML files in `tmp_path`.

**Run**: Add to `make test_agent`.

---

### P1-6. Structured Logging

**Priority**: Medium | **Effort**: 2-3 days | **Depends on**: P1-2 (consolidated daemons) | **After**: P1-2/P1-3

#### Problem

27 files call `logging.basicConfig()` independently. 2,057 `print()` calls across codebase. No JSON logging, no correlation IDs, no structlog.

#### Files to create

| File | Purpose |
|---|---|
| `scripts/logging_config.py` | Centralized logging setup |

#### Files to modify (critical path first)

| File | print() count | Priority |
|---|---|---|
| `scripts/agent/daemon.py` (post-P1-2: `base.py`) | ~50 | High (production daemon) |
| `scripts/agent/runner.py` | ~40 | High (gate execution) |
| `scripts/alpha/alpha_pipeline.py` | ~35 | High (pipeline orchestrator) |
| `scripts/pipeline_runner.py` | ~30 | High (pipeline orchestrator) |
| `scripts/execution/signal_bridge.py` | ~25 | High (live execution) |
| `scripts/discovery_orchestrator.py` | ~20 | Medium |

#### Subtasks

- [ ] Create `scripts/logging_config.py`:
  ```python
  def setup_logging(name: str, level: str = "INFO", log_dir: Path = None) -> logging.Logger:
      """Configure JSON-formatted logging with file + console handlers.
      
      Log format: {"ts": "...", "level": "INFO", "logger": "agent.micro", 
                    "msg": "gate_passed", "hypothesis_id": "h_001", "gate": "G2", "ic": 0.042}
      """
  ```
  - JSON formatter for file output (`data/logs/YYYY-MM-DD.jsonl`)
  - Human-readable formatter for console (colored, concise)
  - Log rotation: daily files, keep 30 days
  - Context injection: `bind(hypothesis_id=..., cycle_id=...)` for correlation
- [ ] Replace `logging.basicConfig()` in consolidated daemon base class (post-P1-2) with `setup_logging()`
- [ ] Convert `print()` to `log.info/debug/warning` in daemon base class cycle loop
- [ ] Convert `print()` to logging in runner gate execution (critical: hypothesis_id in every log line)
- [ ] Convert `print()` to logging in `alpha_pipeline.py`
- [ ] Convert `print()` to logging in `pipeline_runner.py`
- [ ] Convert `print()` to logging in `signal_bridge.py` (critical: live execution needs audit trail)
- [ ] Add hypothesis_id and cycle_id as correlation IDs in agent log context

#### Acceptance criteria

- `grep hypothesis_id data/logs/2026-05-25.jsonl` traces a single hypothesis through all gates
- No new `print()` calls in modified files (add to CI lint later)
- Console output remains human-readable
- Log files rotate daily at `data/logs/`

#### Tests

**File**: `scripts/tests/test_logging_config.py`

**Fixtures**:
```python
@pytest.fixture
def log_dir(tmp_path):
    """Temporary log directory."""
    d = tmp_path / "logs"
    d.mkdir()
    return d

@pytest.fixture
def logger(log_dir):
    """Configured logger writing to tmp log dir."""
    return setup_logging("test.agent", level="DEBUG", log_dir=log_dir)
```

**Test functions**:

| Test | Description | Assertion |
|---|---|---|
| `test_json_format_file_output` | Log a message, read file, parse JSON | `json.loads(line)` has keys: `ts`, `level`, `logger`, `msg` |
| `test_context_binding` | `log.bind(hypothesis_id="h_001")` then log | JSON line includes `"hypothesis_id": "h_001"` |
| `test_console_human_readable` | Console handler produces non-JSON output | No `{` prefix in stderr capture |
| `test_daily_rotation` | Log with fake dates spanning 2 days | 2 files: `2026-05-24.jsonl`, `2026-05-25.jsonl` |
| `test_correlation_id_propagation` | Bind `cycle_id`, log from different module | Both log lines share same `cycle_id` |
| `test_level_filtering` | Set level=WARNING, log DEBUG | DEBUG message not in file |
| `test_no_print_in_modified_files` | AST parse critical files, check for `print()` calls | `count == 0` for each file in modified list |

**Mocking**:
- `monkeypatch.setattr("time.strftime", ...)` for rotation tests
- `capsys` / `capfd` for console output verification

**Lint check** (add to CI):
```bash
# scripts/tests/test_no_print.py
def test_no_print_in_daemons():
    """Ensure print() is not used in production daemon code."""
    import ast
    for path in DAEMON_FILES:
        tree = ast.parse(Path(path).read_text())
        prints = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                  and getattr(n.func, 'id', '') == 'print']
        assert len(prints) == 0, f"{path} has {len(prints)} print() calls"
```

**Run**: `pytest scripts/tests/test_logging_config.py -v` and add to `make test_agent`.

---

### P1-7. Dashboard Caching

**Priority**: Low | **Effort**: Half day | **Depends on**: Nothing | **Independent**

#### Problem

`scripts/agent_dashboard.py` (1,003 LOC) reads and parses JSON from disk on every HTTP request across 12 endpoints. Frontend auto-refreshes every 10s (line 900). `read_state()` (lines 40-45) opens and parses `agent_state.json` on every call.

#### Files to modify

| File | Change |
|---|---|
| `scripts/agent_dashboard.py` | Add in-memory cache with TTL |

#### Subtasks

- [ ] Add module-level cache dict with timestamps: `_cache = {}`, `_cache_ts = {}`
- [ ] Add `_cached_read(key, loader_fn, ttl_seconds=60)` helper function
- [ ] Wrap `read_state()` (line 40), `build_heatmap_data()`, and all data-loading functions with cache
- [ ] Add `Cache-Control: max-age=60` header to API responses
- [ ] Add basic request logging to `do_GET()`: `log.info(f"{method} {path} {status} {latency_ms}ms")`
- [ ] Replace bare `except:` handlers (multiple locations) with `except Exception as e: log.error(...)`

#### Acceptance criteria

- Disk reads drop from 6/sec to 1/min per endpoint
- Dashboard still reflects updates within 60s
- Request log shows method, path, status, latency

#### Tests

**File**: `scripts/tests/test_dashboard_cache.py` (extend existing `test_agent_dashboard.py` pattern)

**Fixtures** (reuse from `test_agent_dashboard.py` — real HTTPServer on random port):
```python
@pytest.fixture
def dashboard_server(tmp_path):
    """Start dashboard server with mock state files in tmp_path."""
    # Pattern from test_agent_dashboard.py: real stdlib HTTPServer
    state_file = tmp_path / "agent_state.json"
    state_file.write_text(json.dumps({"phase": "SLEEPING", "cycle_count": 42}))
    server = start_dashboard(port=0, state_dir=tmp_path)  # port=0 for random
    yield server
    server.shutdown()
```

**Test functions**:

| Test | Description | Assertion |
|---|---|---|
| `test_cache_hit_no_disk_read` | Request same endpoint twice within TTL | Second request doesn't call `open()` on state file (mock `builtins.open`) |
| `test_cache_expires_after_ttl` | Request, wait > TTL, request again | Second request reads from disk |
| `test_cache_control_header` | Any GET response | `Cache-Control: max-age=60` header present |
| `test_request_logging` | Make a request, check log output | Log line contains method, path, status code, latency_ms |
| `test_concurrent_requests_no_stampede` | 10 concurrent requests after TTL expiry | Only 1 disk read (cache stampede prevention) |
| `test_state_update_reflected_within_ttl` | Write new state, wait TTL, query | New state visible after TTL |

**Mocking**:
- `patch("builtins.open", wraps=builtins.open)` to count disk reads
- `monkeypatch.setattr(time, "time", ...)` to simulate TTL expiry without sleeping
- `urllib.request.urlopen` for HTTP requests (pattern from `test_agent_dashboard.py`)

**Run**: Add to `make test_dashboard`.

---

### P1-8. Integration Tests for Daemon Cycles

**Priority**: Medium | **Effort**: 2 days | **Depends on**: P1-2, P1-3, P1-4

#### Problem

56+ test files exist but no tests for full daemon cycles. No crash-recovery tests. No multi-agent coordination tests. Existing fixtures in `scripts/algorithms/tests/conftest.py` (line 74) provide `make_synthetic_ticks()` which can be reused.

#### Files to create

| File | Purpose |
|---|---|
| `scripts/tests/test_daemon_integration.py` | Full cycle tests |
| `scripts/tests/test_multi_agent.py` | Cross-agent coordination |
| `scripts/tests/conftest.py` | Shared fixtures (if not already present at this level) |

#### Subtasks

- [ ] Create synthetic parquet fixture: 3 dates, 2 symbols (BTC, ETH), 20 columns, deterministic seed
  - Reuse `make_synthetic_ticks()` from `scripts/algorithms/tests/conftest.py`
  - Write to temp directory as `data/features/YYYY-MM-DD/*.parquet`
- [ ] Test single-agent full cycle (microstructure):
  - MANIFEST scan finds synthetic data
  - Generator produces at least 1 hypothesis
  - Runner executes 5-gate protocol
  - Hypothesis reaches REGISTERED or GRAVEYARD (deterministic with seed)
  - State persisted correctly in SQLite
- [ ] Test state persistence + resume:
  - Run half a cycle (complete MANIFEST + GENERATE, stop before EXECUTE)
  - Restart agent
  - Verify it resumes from correct phase (not re-running completed steps)
- [ ] Test FDR control:
  - Inject 20 hypotheses with known p-values (10 significant, 10 not)
  - Run BH correction
  - Verify expected number of survivors
- [ ] Test multi-agent dedup:
  - Run micro + MF agents on same data
  - Both discover a correlated signal
  - Meta Agent detects correlation > 0.3 and deduplicates
- [ ] Test config validation:
  - Missing required key → error
  - Unknown key → warning
  - Invalid threshold (negative IC) → error

#### Acceptance criteria

- `pytest scripts/tests/test_daemon_integration.py -v` passes
- Tests complete in < 30 seconds (synthetic data, no real API calls)
- CI catches state machine regressions

#### Tests

**Note**: P1-8 IS the integration test task. The subtasks above define the tests. Below specifies the implementation details.

**Files**: `scripts/tests/test_daemon_integration.py`, `scripts/tests/test_multi_agent.py`, `scripts/tests/conftest.py`

**Shared conftest** (`scripts/tests/conftest.py`):
```python
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np

@pytest.fixture
def synthetic_parquet_dir(tmp_path):
    """Create 3 dates × 2 symbols × 20 columns of deterministic parquet data.
    
    Reuses column distributions from scripts/algorithms/tests/conftest.py
    make_synthetic_ticks() but writes to parquet directory structure.
    """
    rng = np.random.default_rng(42)
    dates = ["2026-05-20", "2026-05-21", "2026-05-22"]
    symbols = ["BTC", "ETH"]
    for date in dates:
        date_dir = tmp_path / "features" / date
        date_dir.mkdir(parents=True)
        for symbol in symbols:
            # 1000 ticks per symbol per date
            df = _make_ticks(rng, symbol, date, n=1000)
            pq.write_table(pa.Table.from_pandas(df), date_dir / f"{symbol}_{date}_00_00.parquet")
    return tmp_path / "features"

@pytest.fixture
def stub_agent_with_data(tmp_path, synthetic_parquet_dir, monkeypatch):
    """Full agent with synthetic data, patched to avoid real subprocess calls."""
    monkeypatch.setenv("NAT_DATA_DIR", str(synthetic_parquet_dir.parent))
    agent = MicrostructureAgent(config_path=_test_config_path(tmp_path))
    return agent
```

**Test function details** for `test_daemon_integration.py`:

| Test | Setup | Action | Assertion |
|---|---|---|---|
| `test_full_cycle_manifest_to_sleep` | `stub_agent_with_data` + `StubRunner(step_results=[True]*5)` | `agent.run_one_cycle()` | State transitions: MANIFEST→GENERATE→EXECUTE→FDR→MONITOR→SLEEP |
| `test_cycle_produces_hypothesis` | `stub_agent_with_data` | `agent.manifest()` + `agent.generate()` | `hypothesis_queue.size() >= 1` |
| `test_runner_executes_gates` | `StubRunner(step_results=[True, True, False, _, _])` | `runner.run_full(hypothesis)` | hypothesis status = GRAVEYARD, failed at gate 3 |
| `test_state_persists_to_sqlite` | `stub_agent_with_data` + `StateStore(tmp_path)` | Run half cycle | `store.get_agent_state("microstructure")["phase"] == "EXECUTE"` |
| `test_resume_from_interrupted` | Run to EXECUTE, stop, restart | `agent.resume()` | Does NOT re-run MANIFEST or GENERATE |
| `test_fdr_bh_correction` | Inject 20 hypotheses: 10 with p<0.01, 10 with p>0.5 | `agent.fdr_control()` | ~10 survivors (BH at q=0.05) |

**Test function details** for `test_multi_agent.py`:

| Test | Setup | Action | Assertion |
|---|---|---|---|
| `test_cross_agent_correlation_dedup` | Run micro + MF, both register correlated signal (r>0.3) | `meta_agent.correlation_check()` | One signal retired, one kept |
| `test_budget_allocation` | 3 agents with different pass rates | `meta_agent.allocate_budget()` | Higher pass rate → more budget (Thompson sampling) |
| `test_portfolio_weights_sum_one` | Register 3 signals from different agents | `meta_portfolio.compute_weights()` | `sum(weights) == 1.0` |
| `test_config_validation_missing_key` | Remove `min_ic` from config | Load config | Raises `ConfigError` with helpful message |
| `test_config_validation_unknown_key` | Add `bogus = 1` to config | Load config | `caplog` contains warning |

**Mocking patterns**:
- `StubRunner(step_results=[...])` for deterministic gate outcomes
- `patch("scripts.agent.runner.subprocess.run")` to prevent real `nat` binary calls
- `monkeypatch.setenv("NAT_DATA_DIR", ...)` for data path isolation
- `patch.object(ResearchAgent, "run_nat_cached", return_value={"ic": 0.05})` for IC injection

**Run**: `make test_integration` (new target) = `pytest scripts/tests/test_daemon_integration.py scripts/tests/test_multi_agent.py -v`

---

## Phase 2: Research API

---

### P2-1. Structured Hypothesis Output

**Priority**: Critical | **Effort**: 2 days | **Depends on**: P1-4 (SQLite state) | **Blocks**: P2-2

#### Problem

Hypothesis results are scattered across JSON files without a consistent schema for the website to consume. Each agent stores results differently. No math derivation field. No structured gate detail.

#### Files to modify

| File | Change |
|---|---|
| `scripts/agent/base.py` | Add `to_research_json()` method to `BaseRunner` |
| `scripts/agent/runner.py` | Emit structured JSON per hypothesis |
| `scripts/agent/mf_runner.py` | Same |
| `scripts/agent/macro_runner.py` | Same |

#### Files to create

| File | Purpose |
|---|---|
| `scripts/data/research.py` | Research artifact schema and I/O |

#### Hypothesis JSON Schema

```python
{
    "id": "h_20260525_001",
    "agent": "microstructure",
    "generator": "spectral",
    "claim": "Leading eigenvalue of 5-level OBI covariance predicts 5s returns",
    "math": {
        "model": "IC = corr(\\lambda_1(\\Sigma_{OBI}), r_{t+50})",  # LaTeX
        "derivation": "Eigendecomposition of L5 order book imbalance...",
        "references": ["Cont2014", "Kyle1985"]
    },
    "gates": {
        "G1_discovery": {
            "metric": "ic", "value": 0.042, "threshold": 0.03,
            "p_value": 0.008, "result": "PASS"
        },
        "G2_temporal": {
            "metric": "oos_dates_passed", "value": 4, "threshold": 2,
            "dates_tested": ["2026-05-20", "2026-05-21", "2026-05-22", "2026-05-23", "2026-05-24"],
            "result": "PASS"
        },
        ...
    },
    "symbols_tested": ["BTC", "ETH", "SOL"],
    "symbols_passed": ["BTC", "ETH"],
    "timeframe": "5min",
    "status": "REGISTERED",
    "created_at": "2026-05-25T14:32:00Z",
    "tested_at": "2026-05-25T14:35:42Z"
}
```

#### Subtasks

- [ ] Define `ResearchHypothesis` dataclass in `scripts/data/research.py` with all fields above
- [ ] Add `math_spec` property to each generator: returns `{"model": "...", "derivation": "...", "references": [...]}`
  - Each generator already knows its claim — add LaTeX model string
- [ ] Modify `BaseRunner.run_full()` to collect gate results into structured dict (metric, value, threshold, result)
- [ ] After each hypothesis completes (pass or fail), write to SQLite `hypotheses` table (P1-4) with full gate detail in JSON `gates` column
- [ ] Also write to `data/research/hypotheses/h_{id}.json` for standalone access
- [ ] Emit cycle summary to `data/research/cycles/cycle_{agent}_{date}_{n}.json` after each cycle
- [ ] Add `to_dict()` method on `ResearchHypothesis` for JSON serialization
- [ ] Write validation test: generate a hypothesis, serialize, deserialize, verify all fields

#### Acceptance criteria

- Every hypothesis (pass or fail) produces a self-contained JSON with math and gate details
- `data/research/hypotheses/` contains one JSON per hypothesis
- SQLite `hypotheses` table has queryable gate results
- LaTeX strings in `math.model` render correctly in KaTeX (manual verification)

#### Tests

**File**: `scripts/tests/test_research_output.py`

**Fixtures**:
```python
@pytest.fixture
def research_hypothesis():
    """A complete ResearchHypothesis with all fields populated."""
    return ResearchHypothesis(
        id="h_test_001",
        agent="microstructure",
        generator="spectral",
        claim="Leading eigenvalue predicts 5s returns",
        math=MathSpec(
            model=r"IC = \text{corr}(\lambda_1(\Sigma_{OBI}), r_{t+50})",
            derivation="Eigendecomposition of L5 order book imbalance...",
            references=["Cont2014", "Kyle1985"]
        ),
        gates={"G1_discovery": {"metric": "ic", "value": 0.042, "threshold": 0.03, "result": "PASS"}},
        symbols_tested=["BTC", "ETH", "SOL"],
        symbols_passed=["BTC", "ETH"],
        timeframe="5min",
        status="REGISTERED",
    )

@pytest.fixture
def runner_with_store(tmp_path):
    """Runner wired to write structured output to StateStore + JSON files."""
    store = StateStore(db_path=tmp_path / "test.db")
    runner = StubRunner(hypothesis=stub_hypothesis(), step_results=[True]*5)
    runner._store = store
    runner._output_dir = tmp_path / "research" / "hypotheses"
    runner._output_dir.mkdir(parents=True)
    return runner, store
```

**Test functions**:

| Test | Description | Assertion |
|---|---|---|
| `test_to_dict_roundtrip` | `ResearchHypothesis.to_dict()` → JSON → `from_dict()` | All fields preserved |
| `test_json_schema_valid` | Serialize hypothesis, validate against JSON schema | No schema violations |
| `test_math_spec_latex_present` | Hypothesis has `math.model` with LaTeX | Contains `\\text{corr}` |
| `test_gate_detail_structure` | Each gate in `gates` dict has metric, value, threshold, result | All 4 keys present per gate |
| `test_runner_writes_json_file` | Run full hypothesis through StubRunner | `data/research/hypotheses/h_test_001.json` exists and is valid JSON |
| `test_runner_writes_sqlite` | Run full hypothesis | `store.query_hypotheses(id="h_test_001")` returns 1 result with gates |
| `test_failed_hypothesis_also_written` | StubRunner with `step_results=[True, False]` | JSON file written with `status=GRAVEYARD`, gate G2 has `result=FAIL` |
| `test_cycle_summary_emitted` | Complete a full cycle | `data/research/cycles/cycle_microstructure_2026-05-25_1.json` exists |
| `test_generator_math_spec` | Each generator class has `math_spec` property | Returns dict with `model`, `derivation`, `references` |
| `test_latex_no_unescaped_braces` | All `math.model` strings | No raw `{` without LaTeX command prefix |

**Mocking**:
- `StubRunner` with controlled `step_results` for deterministic gate outcomes
- `patch("scripts.data.research.datetime")` for deterministic timestamps in output files

**Run**: `pytest scripts/tests/test_research_output.py -v` and add to `make test_agent`.

---

### P2-2. Research REST Endpoints

**Priority**: High | **Effort**: 2-3 days | **Depends on**: P2-1 | **Blocks**: P3-2

#### Problem

No API to query hypothesis history, signal registry, or cycle reports. The existing Axum API (`rust/api/src/main.rs`, lines 41-63) only serves live features and market data.

#### Files to create

| File | Purpose |
|---|---|
| `rust/api/src/routes/research.rs` | Research endpoint handlers |
| `rust/api/src/db.rs` | SQLite reader (read-only, Python writes) |

#### Files to modify

| File | Change |
|---|---|
| `rust/api/src/main.rs` | Add research routes to router |
| `rust/api/src/routes/mod.rs` | Add `pub mod research;` |
| `rust/api/src/lib.rs` | Add `pub mod db;` |
| `rust/api/src/state.rs` | Add `db: SqlitePool` to `AppState` |

#### Endpoints

| Method | Path | Handler | Returns |
|---|---|---|---|
| `GET` | `/api/research/hypotheses` | `list_hypotheses` | Paginated list, filterable by agent/generator/status/date |
| `GET` | `/api/research/hypotheses/:id` | `get_hypothesis` | Full detail including math and gate results |
| `GET` | `/api/research/signals` | `list_signals` | Registered signals with IC history |
| `GET` | `/api/research/cycles` | `list_cycles` | Cycle summaries per agent |
| `GET` | `/api/research/stats` | `get_stats` | Aggregate: total tested/passed, FDR budget, per-gate pass rates |
| `GET` | `/api/research/heatmap` | `get_heatmap` | Feature x horizon IC matrix |

#### Subtasks

- [ ] Add `rusqlite` dependency to `rust/api/Cargo.toml` (read-only access to `data/nat.db`)
- [ ] Create `rust/api/src/db.rs`: open SQLite in read-only mode, query functions for hypotheses/signals/cycles
- [ ] Add `SqlitePool` (or single connection) to `AppState` in `state.rs`
- [ ] Implement `list_hypotheses` with query params: `?agent=`, `?generator=`, `?status=`, `?since=`, `?limit=`, `?offset=`
- [ ] Implement `get_hypothesis` returning full JSON including gates and math
- [ ] Implement `list_signals` with IC history arrays
- [ ] Implement `list_cycles` with per-cycle hypothesis counts
- [ ] Implement `get_stats` aggregating across all agents
- [ ] Implement `get_heatmap` building feature × horizon IC matrix from hypothesis data
- [ ] Add routes to router in `main.rs`
- [ ] Add integration test: insert test data into SQLite, query via HTTP, verify response

#### Acceptance criteria

- `curl localhost:3000/api/research/hypotheses?agent=microstructure&status=REGISTERED` returns JSON
- All endpoints return proper pagination (`total`, `offset`, `limit`, `items`)
- Response times < 50ms for typical queries
- `cargo test --package api` passes

#### Tests

**File**: `rust/api/tests/research_api.rs` (Rust integration tests)

**Setup**: Pre-populate test SQLite DB with known hypothesis/signal data.

```rust
fn test_db(tmp: &TempDir) -> SqlitePool {
    let db = StateStore::new(tmp.path().join("test.db"));
    db.insert_hypothesis(test_hypothesis("h_001", "microstructure", "REGISTERED"));
    db.insert_hypothesis(test_hypothesis("h_002", "mf", "GRAVEYARD"));
    db.insert_signal(test_signal("s_001", "h_001", "ACTIVE"));
    db.pool()
}
```

**Test functions**:

| Test | Description | Assertion |
|---|---|---|
| `test_list_hypotheses_no_filter` | `GET /api/research/hypotheses` | 200, returns both hypotheses, has `total`, `offset`, `limit`, `items` |
| `test_list_hypotheses_filter_agent` | `?agent=microstructure` | Only `h_001` returned |
| `test_list_hypotheses_filter_status` | `?status=GRAVEYARD` | Only `h_002` returned |
| `test_list_hypotheses_pagination` | `?limit=1&offset=0` then `?limit=1&offset=1` | Each returns 1 item, different IDs |
| `test_get_hypothesis_detail` | `GET /api/research/hypotheses/h_001` | 200, full JSON with gates and math |
| `test_get_hypothesis_404` | `GET /api/research/hypotheses/nonexistent` | 404 |
| `test_list_signals` | `GET /api/research/signals` | Returns signal with IC history array |
| `test_list_cycles` | `GET /api/research/cycles` | Returns cycle summaries |
| `test_get_stats` | `GET /api/research/stats` | `total_tested`, `total_registered`, `fdr_budget` present |
| `test_get_heatmap` | `GET /api/research/heatmap` | Returns matrix with feature rows and horizon columns |
| `test_response_time_under_50ms` | Benchmark `list_hypotheses` with 1000 rows | p99 < 50ms |
| `test_readonly_no_writes` | Attempt POST/PUT/DELETE | 405 Method Not Allowed |

**Run**: `cargo test --package api` (existing target).

**Also add Python smoke test** in `scripts/tests/test_research_api_smoke.py`:

| Test | Description |
|---|---|
| `test_hypotheses_endpoint_reachable` | `urllib.request.urlopen("http://localhost:3000/api/research/hypotheses")` returns 200 |
| `test_response_is_valid_json` | Parse response, verify `items` key exists |

**Run**: `make test_api` (new target, requires running API server — mark as `@pytest.mark.integration`).

---

### P2-3. WebSocket Research Stream

**Priority**: Medium | **Effort**: 1 day | **Depends on**: P2-2

#### Problem

No real-time updates when hypotheses complete. Dashboard must poll.

#### Files to modify

| File | Change |
|---|---|
| `rust/api/src/routes/ws.rs` | Add `research_websocket_handler` |
| `rust/api/src/main.rs` | Add `/ws/research` route |
| `scripts/agent/base.py` | Publish events to Redis on hypothesis completion |

#### Event Types

```json
{"event": "hypothesis_started", "id": "h_001", "agent": "micro", "claim": "..."}
{"event": "gate_passed", "id": "h_001", "gate": "G2_temporal", "ic": 0.042}
{"event": "gate_failed", "id": "h_001", "gate": "G3_symbol", "reason": "only 1/3 symbols passed"}
{"event": "hypothesis_registered", "id": "h_001", "agent": "micro", "ic": 0.042}
{"event": "cycle_completed", "agent": "micro", "tested": 8, "passed": 1, "cycle": 42}
```

#### Subtasks

- [ ] Add `publish_research_event(event_type, payload)` function in `scripts/data/research.py`
  - Publishes to Redis channel `nat:research:events` (same Redis instance as feature streaming)
- [ ] Call `publish_research_event()` from `BaseRunner` at: hypothesis start, each gate pass/fail, registration, cycle end
- [ ] Add `research_websocket_handler` in `rust/api/src/routes/ws.rs` subscribing to `nat:research:events`
- [ ] Add route `/ws/research` to router
- [ ] Test: start WebSocket client, run one agent cycle, verify events received in order

#### Acceptance criteria

- WebSocket client at `ws://localhost:3000/ws/research` receives live events
- Events arrive within 1s of hypothesis completion
- Disconnected clients can reconnect without missing state (query REST API for current state)

#### Tests

**File (Rust)**: `rust/api/tests/research_ws.rs`

```rust
#[tokio::test]
async fn test_ws_receives_events() {
    // Start test server, connect WS client
    // Publish event to Redis nat:research:events
    // Assert WS client receives matching JSON within 1s
}
```

| Test | Description | Assertion |
|---|---|---|
| `test_ws_connect` | Connect to `ws://localhost:3000/ws/research` | Connection established, no error frame |
| `test_hypothesis_started_event` | Publish `hypothesis_started` to Redis | WS client receives JSON with `event`, `id`, `agent` |
| `test_gate_passed_event` | Publish `gate_passed` | WS receives with `gate` and `ic` fields |
| `test_gate_failed_event` | Publish `gate_failed` | WS receives with `gate` and `reason` fields |
| `test_cycle_completed_event` | Publish `cycle_completed` | WS receives with `tested`, `passed`, `cycle` |
| `test_multiple_clients` | 2 clients connected simultaneously | Both receive same event |
| `test_disconnect_reconnect` | Disconnect, publish event, reconnect, query REST | Event accessible via REST after reconnect |

**File (Python)**: `scripts/tests/test_research_events.py`

| Test | Description | Assertion |
|---|---|---|
| `test_publish_event_to_redis` | `publish_research_event("hypothesis_started", {...})` | Redis `SUBSCRIBE` receives message |
| `test_runner_emits_on_gate_pass` | `StubRunner` with `step_results=[True]` | Redis received `gate_passed` event |
| `test_runner_emits_on_gate_fail` | `StubRunner` with `step_results=[False]` | Redis received `gate_failed` event |
| `test_cycle_end_emits_summary` | Complete full agent cycle | Redis received `cycle_completed` |

**Mocking**:
- `fakeredis` for Python tests (no real Redis needed)
- `redis-mock` crate or actual Redis in Docker for Rust tests

**Run**: `cargo test --package api -- ws` + `pytest scripts/tests/test_research_events.py -v`

---

## Phase 3: Research Website

---

### P3-1. Project Scaffolding

**Priority**: Critical | **Effort**: 1 day | **Depends on**: P2-2 (API must exist) | **Blocks**: all P3-*

#### Subtasks

- [ ] Initialize Next.js project: `npx create-next-app@latest web --typescript --tailwind --app --src-dir`
- [ ] Install dependencies: `plotly.js`, `react-plotly.js`, `d3`, `katex`, `react-katex`
- [ ] Configure API proxy in `next.config.js`: `/api/*` → `http://localhost:3000/api/*`
- [ ] Configure WebSocket proxy: `/ws/*` → `ws://localhost:3000/ws/*`
- [ ] Create layout: sidebar nav (Dashboard, Explorer, Signals, Heatmap, Math Lab, Graveyard), header with status dot
- [ ] Create `web/src/lib/api.ts` — typed fetch wrappers for all `/api/research/*` endpoints
- [ ] Create `web/src/lib/ws.ts` — WebSocket hook for `/ws/research`
- [ ] Add to `docker-compose.yml`: `web` service (node, port 3001)
- [ ] Add Makefile targets: `make web_dev` (next dev), `make web_build` (next build)
- [ ] Verify: `make web_dev` starts on localhost:3001, API proxy works

#### Acceptance criteria

- `make web_dev` starts frontend with hot reload
- Navigating to `/` shows layout with sidebar
- `fetch('/api/research/stats')` from frontend returns JSON from Axum backend

#### Tests

**File**: `web/src/__tests__/setup.test.ts` (Vitest or Jest)

| Test | Description | Assertion |
|---|---|---|
| `test_layout_renders` | Render `<Layout />` | Sidebar with 7 nav items visible |
| `test_api_client_types` | Import `api.ts` functions | TypeScript compiles without errors |
| `test_api_proxy_config` | Read `next.config.js` | `/api/*` rewrites to `http://localhost:3000/api/*` |
| `test_ws_hook_connects` | Mount `useResearchWs()` hook with mock WS | `readyState === OPEN` after connect |

**E2E** (Playwright, `web/e2e/scaffold.spec.ts`):

| Test | Description |
|---|---|
| `test_homepage_loads` | Navigate to `/`, assert sidebar visible |
| `test_navigation_works` | Click each nav item, assert URL changes |
| `test_api_proxy_functional` | Mock `/api/research/stats`, verify frontend displays data |

**Run**: `cd web && npm test` (unit), `cd web && npx playwright test` (e2e).

---

### P3-2. Dashboard Page

**Priority**: Critical | **Effort**: 2-3 days | **Depends on**: P3-1

#### Components

- [ ] **Agent Status Cards** (3 cards: micro, MF, macro): current phase, cycle count, last cycle timestamp, queue depth. Color-coded: green=SLEEPING, blue=EXECUTE, yellow=GENERATE, red=ERROR
- [ ] **Cycle Progress Ring**: SVG animated arc showing current phase within MANIFEST→GENERATE→EXECUTE→MONITOR→SLEEP cycle
- [ ] **Recent Hypothesis Feed**: last 20 hypotheses, status badges (green PASS, red FAIL, yellow TESTING). Auto-updates via WebSocket `hypothesis_registered` / `gate_failed` events
- [ ] **Aggregate Stats Bar**: total tested / total registered / FDR budget remaining / active signals count
- [ ] **System Health**: ingestor uptime, data freshness per symbol, last parquet timestamp. Reads from `/api/features/BTC` existing endpoint
- [ ] Connect to `GET /api/research/stats` on mount, `WS /ws/research` for live updates

#### Tests

**File**: `web/src/__tests__/dashboard.test.tsx`

| Test | Description | Assertion |
|---|---|---|
| `test_agent_cards_render` | Mock `/api/research/stats` → render Dashboard | 3 agent status cards visible (micro, MF, macro) |
| `test_card_color_by_phase` | Agent in SLEEPING → green, EXECUTE → blue, ERROR → red | Card has correct CSS class |
| `test_hypothesis_feed_populates` | Mock 20 hypotheses from API | Feed shows 20 rows with status badges |
| `test_ws_updates_feed` | Send `hypothesis_registered` via mock WS | New row appears in feed without refresh |
| `test_aggregate_stats_display` | Mock stats endpoint | Total tested, registered, FDR budget shown |
| `test_system_health_freshness` | Mock features endpoint with stale timestamp | Warning indicator for stale data |

**E2E** (`web/e2e/dashboard.spec.ts`):

| Test | Description |
|---|---|
| `test_dashboard_loads_with_data` | Navigate to `/`, mock API, verify cards + feed visible |
| `test_live_update_animation` | Send WS event, verify slide-in animation on new hypothesis |

**Run**: `cd web && npm test -- dashboard` (unit), `cd web && npx playwright test dashboard` (e2e).

---

### P3-3. Hypothesis Explorer Page

**Priority**: High | **Effort**: 2-3 days | **Depends on**: P3-1

#### Components

- [ ] **Sortable Table**: columns = ID, agent, generator, claim (truncated), IC, p-value, status, date. Server-side pagination via `?limit=&offset=`
- [ ] **Filter Bar**: dropdowns for agent type, generator, status (REGISTERED/GRAVEYARD/QUEUED). Date range picker. IC range slider
- [ ] **Gate Funnel Chart** (Plotly): animated funnel N tested → N pass G1 → N pass G2 → ... → N registered. Updates on filter change
- [ ] **Sankey Diagram** (Plotly): generator → gate outcome flows. Shows which generators produce which failure modes
- [ ] **Row Click → Detail**: navigate to `/hypotheses/:id` (P3-4)
- [ ] Connect to `GET /api/research/hypotheses` with query params from filters

#### Tests

**File**: `web/src/__tests__/explorer.test.tsx`

| Test | Description | Assertion |
|---|---|---|
| `test_table_renders_hypotheses` | Mock 10 hypotheses | Table has 10 rows with correct columns |
| `test_table_sortable` | Click IC column header | Rows reorder by IC descending |
| `test_filter_by_agent` | Select "microstructure" dropdown | API called with `?agent=microstructure`, table updates |
| `test_filter_by_status` | Select "REGISTERED" | Only registered hypotheses shown |
| `test_date_range_filter` | Set date range | API called with `?since=...` param |
| `test_pagination_controls` | Mock 50 hypotheses, limit=20 | Next/prev buttons, page indicator shows "1 of 3" |
| `test_funnel_chart_renders` | Mock gate pass/fail counts | Plotly funnel SVG present in DOM |
| `test_sankey_diagram_renders` | Mock generator-to-gate flow data | Plotly sankey SVG present |
| `test_row_click_navigates` | Click hypothesis row | `router.push("/hypotheses/h_001")` called |

**E2E** (`web/e2e/explorer.spec.ts`):

| Test | Description |
|---|---|
| `test_filter_and_sort` | Apply filters, sort, verify URL params + table content |
| `test_pagination_navigation` | Click through pages, verify different data per page |

**Run**: `cd web && npm test -- explorer`.

---

### P3-4. Signal Detail Page

**Priority**: High | **Effort**: 3-4 days | **Depends on**: P3-1

#### Components

- [ ] **Header**: hypothesis ID, claim text, agent badge, generator badge, status badge (REGISTERED=green, GRAVEYARD=red), timestamps
- [ ] **Gate Waterfall Chart** (Plotly): horizontal bars per gate, bar length = metric value, vertical line = threshold. Green if PASS, red if FAIL
- [ ] **Math Derivation Panel** (KaTeX): render `hypothesis.math.model` and `hypothesis.math.derivation` fields. Expandable sections for full derivation. References as clickable links
- [ ] **IC Time Series** (Plotly): line chart of IC over time (from signal IC history). Confidence bands. Horizontal reference line at retirement threshold
- [ ] **Walk-Forward Equity Curve** (Plotly): overlaid IS (dashed) and OOS (solid) equity curves. Gap between them = overfitting indicator
- [ ] **Per-Symbol Breakdown**: small multiples (3 mini-charts) showing IC and Sharpe per BTC/ETH/SOL
- [ ] **Related Hypotheses**: table of hypotheses with correlation > 0.3 (from orthogonality gate data)
- [ ] Connect to `GET /api/research/hypotheses/:id`

#### Tests

**File**: `web/src/__tests__/signal-detail.test.tsx`

| Test | Description | Assertion |
|---|---|---|
| `test_header_displays_metadata` | Mock hypothesis with all fields | ID, claim, agent badge, status badge visible |
| `test_gate_waterfall_chart` | Mock 5 gates with mixed PASS/FAIL | Plotly bar chart with 5 bars, green/red coloring |
| `test_math_katex_renders` | Mock hypothesis with LaTeX `math.model` | KaTeX-rendered DOM element present (no raw LaTeX) |
| `test_math_derivation_expandable` | Click "Show derivation" | Derivation text appears |
| `test_ic_time_series` | Mock IC history with 30 data points | Line chart with 30 points, threshold reference line |
| `test_equity_curve_is_oos` | Mock IS and OOS equity data | Two lines: dashed (IS) + solid (OOS) |
| `test_per_symbol_breakdown` | Mock 3 symbols | 3 mini-charts with symbol labels |
| `test_related_hypotheses_table` | Mock correlated hypotheses | Table with correlation > 0.3 entries |
| `test_404_nonexistent_id` | Navigate to `/hypotheses/bad_id` | "Not found" message displayed |

**E2E** (`web/e2e/signal-detail.spec.ts`):

| Test | Description |
|---|---|
| `test_full_detail_page_render` | Navigate to `/hypotheses/h_001`, verify all sections present |
| `test_katex_no_error_spans` | Verify no `katex-error` class elements in DOM |

**Run**: `cd web && npm test -- signal-detail`.

---

### P3-5. IC Heatmap Page

**Priority**: High | **Effort**: 2 days | **Depends on**: P3-1

#### Components

- [ ] **Interactive Heatmap** (Plotly): rows = features (217), columns = horizons (5s, 1min, 5min, 1h, 24h). Cells = mean IC across tested hypotheses. Diverging colorscale (red=negative, blue=positive)
- [ ] **Row Clustering**: group features by category (flow, entropy, trend, etc.) with collapsible groups
- [ ] **Click-to-drill**: click cell → navigate to Hypothesis Explorer filtered to that feature + horizon
- [ ] **Toggle**: raw IC vs FDR-adjusted significance vs cost-adjusted net IC
- [ ] **Export**: PNG/SVG download button
- [ ] Connect to `GET /api/research/heatmap`

#### Tests

**File**: `web/src/__tests__/heatmap.test.tsx`

| Test | Description | Assertion |
|---|---|---|
| `test_heatmap_renders` | Mock heatmap API with 20 features × 5 horizons | Plotly heatmap SVG present |
| `test_correct_dimensions` | Mock 217 features × 5 horizons | Heatmap has 217 rows and 5 columns |
| `test_diverging_colorscale` | Mock positive and negative IC values | Red cells for negative, blue for positive |
| `test_row_clustering_toggles` | Click category group header | Group collapses/expands |
| `test_cell_click_navigates` | Click cell at (flow_trade_count, 5min) | Navigates to explorer with `?feature=flow_trade_count&horizon=5min` |
| `test_toggle_raw_vs_fdr` | Click "FDR-adjusted" toggle | Different values displayed (some cells grey out) |
| `test_export_button` | Click "Export PNG" | Download triggered (mock `URL.createObjectURL`) |

**Run**: `cd web && npm test -- heatmap`.

---

### P3-6. Signal Registry Page

**Priority**: Medium | **Effort**: 2-3 days | **Depends on**: P3-1

#### Components

- [ ] **Registry Table**: signal ID, agent, features used, IC (current), Sharpe, status (ACTIVE/DECAYING/RETIRED)
- [ ] **IC Decay Curves** (Plotly): per-signal line chart. Horizontal dashed line = retirement threshold. Signal crosses it → line turns red. Clickable legend to toggle signals
- [ ] **Correlation Matrix** (Plotly heatmap): pairwise correlation between registered signals. Highlights correlated pairs (> 0.3)
- [ ] **Portfolio Weights Treemap** (Plotly): proportional area per signal, color = agent type (micro=blue, MF=purple, macro=orange)
- [ ] Connect to `GET /api/research/signals`

#### Tests

**File**: `web/src/__tests__/registry.test.tsx`

| Test | Description | Assertion |
|---|---|---|
| `test_registry_table_renders` | Mock 5 signals | Table with 5 rows, columns: ID, agent, IC, Sharpe, status |
| `test_status_color_coding` | ACTIVE=green, DECAYING=yellow, RETIRED=red | Correct badge color per status |
| `test_ic_decay_curves_render` | Mock 3 signals with IC history | 3 lines on Plotly line chart |
| `test_decay_threshold_line` | Mock threshold at 0.02 | Horizontal dashed line at y=0.02 |
| `test_correlation_matrix_renders` | Mock 4 signals with pairwise correlations | 4×4 heatmap present |
| `test_correlation_highlight` | Pair with r>0.3 | Cell highlighted/bordered |
| `test_treemap_renders` | Mock 5 signals with weights | Plotly treemap with 5 blocks |
| `test_treemap_color_by_agent` | Micro=blue, MF=purple, macro=orange | Correct fill colors |

**Run**: `cd web && npm test -- registry`.

---

### P3-7. Math Lab Page

**Priority**: Medium | **Effort**: 2-3 days | **Depends on**: P3-1

#### Content (KaTeX-rendered, no backend dependency — static content)

- [ ] **Feature Definitions**: closed-form formula for each feature category. Source from `FEATURES.md` (already exists with formulas)
  - Raw: midprice, spread, microprice definitions
  - Entropy: permutation entropy formula, tick entropy
  - Illiquidity: Kyle lambda, Amihud, Hasbrouck derivations
  - Toxicity: VPIN, adverse selection
- [ ] **Gate Protocols**:
  - IC definition: $IC = \text{corr}(s_t, r_{t+\tau})$
  - FDR: Benjamini-Hochberg procedure with step-up formula
  - Temporal replication: bootstrap confidence interval
  - Orthogonality: $R^2$ threshold derivation
- [ ] **Convolver Math**: SVD decomposition, cosine similarity scoring, kernel selection. Source from `docs/convolver_method.tex`
- [ ] **Liquidity Signal**: z-score normalization, composite construction, percentile thresholds
- [ ] **Position Sizing**: Kelly criterion derivation, cost adjustment formula
- [ ] **Searchable**: full-text search on LaTeX source strings
- [ ] **Paper References**: DOI links for cited papers (Cont 2014, Kyle 1985, Cartea 2015, etc.)

#### Tests

**File**: `web/src/__tests__/math-lab.test.tsx`

| Test | Description | Assertion |
|---|---|---|
| `test_katex_renders_all_sections` | Render Math Lab page | No `katex-error` class elements in DOM |
| `test_feature_definitions_present` | Check for each feature category header | "Raw", "Entropy", "Illiquidity", "Toxicity" sections exist |
| `test_gate_protocol_formulas` | IC definition section | Contains rendered `corr(s_t, r_{t+τ})` |
| `test_fdr_bh_formula` | FDR section | Contains step-up formula rendered |
| `test_convolver_svd_section` | Convolver section | Contains "SVD", "cosine similarity" |
| `test_search_functionality` | Type "entropy" in search box | Only entropy-related sections visible |
| `test_references_have_links` | Paper references section | Each reference is a clickable `<a>` with DOI href |
| `test_static_no_api_calls` | Render full page | Zero network requests (all content static) |

**Run**: `cd web && npm test -- math-lab`.

---

### P3-8. Graveyard Page

**Priority**: Low | **Effort**: 1-2 days | **Depends on**: P3-1

#### Components

- [ ] **Failure Mode Distribution** (Plotly pie/donut): proportion of failures per gate (G1, G2, G3, G4, G5)
- [ ] **Generator Failure Rates** (Plotly bar): per-generator success/failure counts
- [ ] **Near Misses Table**: hypotheses that passed 4/5 gates (closest to promotion). Sortable by IC
- [ ] **Failure Trends** (Plotly line): failures per cycle over time. Are generators improving?
- [ ] **Recyclable Hypotheses**: failed on replication but IC > threshold — candidates for parameter tuning
- [ ] Connect to `GET /api/research/hypotheses?status=GRAVEYARD`

#### Tests

**File**: `web/src/__tests__/graveyard.test.tsx`

| Test | Description | Assertion |
|---|---|---|
| `test_failure_distribution_chart` | Mock failures across 5 gates | Donut chart with 5 segments |
| `test_generator_failure_bars` | Mock 4 generators with different failure rates | Bar chart with 4 bars |
| `test_near_misses_table` | Mock hypotheses passing 4/5 gates | Table shows only near-misses, sorted by IC |
| `test_failure_trends_line` | Mock failures per cycle for 10 cycles | Line chart with 10 data points |
| `test_recyclable_hypotheses` | Mock hypotheses: failed replication but IC > threshold | Separate table with "recyclable" label |
| `test_filter_by_gate` | Click on G2 in donut chart | Table filters to G2 failures only |

**Run**: `cd web && npm test -- graveyard`.

---

### P3-9. Feature Interaction Network

**Priority**: Low | **Effort**: 2-3 days | **Depends on**: P3-1

#### Components

- [ ] **D3 Force-Directed Graph**: nodes = features, edges = conditional mutual information > threshold
- [ ] **Node sizing**: proportional to number of hypotheses using that feature
- [ ] **Node color**: by category (flow=blue, entropy=green, trend=orange, etc.)
- [ ] **Edge thickness**: proportional to mutual information strength
- [ ] **Cluster highlights**: visual grouping of co-discovered features
- [ ] **Click-to-filter**: click node → navigate to Hypothesis Explorer filtered to that feature
- [ ] **Toggle**: registered-only features vs all tested features

#### Tests

**File**: `web/src/__tests__/network.test.tsx`

| Test | Description | Assertion |
|---|---|---|
| `test_graph_renders_nodes` | Mock 10 features with MI edges | 10 SVG circle elements (D3 nodes) |
| `test_edges_above_threshold` | Mock MI matrix, threshold=0.1 | Only edges with MI > 0.1 drawn |
| `test_node_size_proportional` | Feature with 10 hypotheses vs feature with 1 | Larger node radius for high-count feature |
| `test_node_color_by_category` | Flow features vs entropy features | Different fill colors |
| `test_edge_thickness` | Strong MI (0.5) vs weak MI (0.15) | Thicker stroke-width for stronger edge |
| `test_click_node_navigates` | Click a feature node | Navigates to explorer with `?feature=...` |
| `test_toggle_registered_only` | Toggle switch | Graph updates with fewer nodes (only features in registered hypotheses) |
| `test_cluster_highlighting` | Hover on node | Connected cluster highlighted, rest dimmed |

**Run**: `cd web && npm test -- network`.

---

## Phase 4: Polish

---

### P4-1. Real-Time WebSocket Updates on Frontend

**Priority**: Medium | **Effort**: 1-2 days | **Depends on**: P2-3, P3-2

#### Subtasks

- [ ] Connect Dashboard to `WS /ws/research` via React hook
- [ ] Animate new hypotheses in feed (slide-in CSS transition)
- [ ] Flash agent status card border on gate pass (green pulse) / fail (red pulse)
- [ ] Update aggregate stats counter in real-time (no full page refresh)
- [ ] Connection status indicator in header: green dot = connected, yellow = reconnecting, red = disconnected
- [ ] Auto-reconnect with exponential backoff

#### Tests

**File**: `web/src/__tests__/realtime.test.tsx`

| Test | Description | Assertion |
|---|---|---|
| `test_ws_connect_on_mount` | Mount Dashboard | WebSocket connection opened to `/ws/research` |
| `test_hypothesis_feed_animates` | Send `hypothesis_registered` event via mock WS | New row with CSS transition class appears |
| `test_card_flash_on_pass` | Send `gate_passed` event | Agent card border pulses green (CSS class applied) |
| `test_card_flash_on_fail` | Send `gate_failed` event | Agent card border pulses red |
| `test_stats_counter_updates` | Send `cycle_completed` with new counts | Stats bar numbers change without page refresh |
| `test_connection_indicator_green` | WS connected | Header shows green dot |
| `test_connection_indicator_red` | Close mock WS server | Header shows red dot after timeout |
| `test_auto_reconnect` | Close WS, wait | WS reopened with exponential backoff (verify attempt count) |
| `test_no_duplicate_events` | Send same event twice | Feed shows only 1 new entry |

**Run**: `cd web && npm test -- realtime`.

---

### P4-2. PDF Export per Hypothesis

**Priority**: Low | **Effort**: 2 days | **Depends on**: P3-4

#### Subtasks

- [ ] Add "Export PDF" button on Signal Detail page
- [ ] Create LaTeX template in `docs/templates/hypothesis_report.tex`
- [ ] Server-side endpoint `GET /api/research/hypotheses/:id/pdf`:
  - Fill template with hypothesis data (claim, math, gates, IC chart)
  - Compile via `pdflatex` (same toolchain as existing docs)
  - Return PDF as `Content-Type: application/pdf`
- [ ] Alternative: generate PDF client-side with `jsPDF` + KaTeX-rendered math (avoid server-side LaTeX dependency)

#### Tests

**File**: `rust/api/tests/pdf_export.rs` (if server-side) or `web/src/__tests__/pdf-export.test.tsx` (if client-side)

**Server-side tests** (Rust):

| Test | Description | Assertion |
|---|---|---|
| `test_pdf_endpoint_returns_pdf` | `GET /api/research/hypotheses/h_001/pdf` | `Content-Type: application/pdf`, body starts with `%PDF` |
| `test_pdf_contains_hypothesis_claim` | Extract text from PDF | Contains hypothesis claim text |
| `test_pdf_contains_math` | Verify LaTeX compiled | No `\text{corr}` literal in output (means LaTeX rendered) |
| `test_pdf_missing_hypothesis_404` | `GET /api/research/hypotheses/bad_id/pdf` | 404 |
| `test_pdf_template_compiles` | Run `pdflatex` on template with test data | Exit code 0, non-empty PDF |

**Client-side tests** (if using jsPDF):

| Test | Description | Assertion |
|---|---|---|
| `test_export_button_triggers_download` | Click "Export PDF" | `URL.createObjectURL` called with Blob of type `application/pdf` |
| `test_pdf_includes_gate_table` | Generate PDF from mock hypothesis | PDF text contains gate names |
| `test_pdf_math_rendered` | Hypothesis with LaTeX math | KaTeX SVG included in PDF |

**Run**: `cargo test --package api -- pdf` or `cd web && npm test -- pdf-export`.

---

### P4-3. CI/CD for Frontend

**Priority**: Medium | **Effort**: 1 day | **Depends on**: P3-1

#### Subtasks

- [ ] Add `web/` lint + type-check to CI: `cd web && npm run lint && npm run build`
- [ ] Create `web/Dockerfile` for production: multi-stage build (node build → nginx serve)
- [ ] Add `web` service to `docker-compose.yml` (nginx on port 80, proxy `/api` to Axum)
- [ ] Add Makefile target: `make web_deploy` (docker build + restart)
- [ ] Add `pytest` config at project root (`pyproject.toml` or `pytest.ini`) — currently none exists

#### Tests

**File**: `web/e2e/ci.spec.ts` + CI pipeline validation

| Test | Description | Assertion |
|---|---|---|
| `test_lint_passes` | `npm run lint` | Exit code 0, no errors |
| `test_type_check_passes` | `npm run build` (includes tsc) | Exit code 0 |
| `test_docker_build_succeeds` | `docker build -f web/Dockerfile .` | Image builds, exit code 0 |
| `test_nginx_serves_static` | Start container, `curl localhost:80` | 200 with HTML content |
| `test_api_proxy_in_docker` | Start full docker-compose, `curl localhost:80/api/research/stats` | Proxied to Axum, returns JSON |
| `test_pytest_ini_discovery` | `pytest --collect-only scripts/tests/` | All test files discovered |

**Run**: CI pipeline (GitHub Actions or equivalent). Local: `make web_build && make docker_build`.

---

### P4-4. Alert Integration

**Priority**: Low | **Effort**: 1 day | **Depends on**: P2-3

#### Subtasks

- [ ] Telegram notification on `hypothesis_registered` event (use existing `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID`)
- [ ] Telegram notification when registered signal IC crosses decay threshold
- [ ] Telegram daily digest (cron or agent cycle): hypotheses tested/registered/retired in last 24h
- [ ] Wire into existing alert infrastructure in `rust/api/src/telegram.rs`
- [ ] Make alerts configurable: `config/agent.toml` `[alerts]` section with `telegram_enabled`, `digest_hour`

#### Tests

**File**: `scripts/tests/test_alerts.py`

**Fixtures**:
```python
@pytest.fixture
def mock_telegram(monkeypatch):
    """Mock Telegram API calls, capture sent messages."""
    sent = []
    def fake_send(chat_id, text):
        sent.append({"chat_id": chat_id, "text": text})
    monkeypatch.setattr("scripts.alerts.send_telegram", fake_send)
    return sent

@pytest.fixture
def alert_config(tmp_path):
    """Config with alerts enabled."""
    config = {"alerts": {"telegram_enabled": True, "digest_hour": 8, 
              "chat_id": "123", "bot_token": "fake"}}
    return config
```

**Test functions**:

| Test | Description | Assertion |
|---|---|---|
| `test_hypothesis_registered_alert` | Emit `hypothesis_registered` event | `mock_telegram` has 1 message containing hypothesis ID and claim |
| `test_ic_decay_alert` | Signal IC crosses threshold | Alert sent with signal ID and current IC |
| `test_daily_digest_content` | Trigger digest with 5 tested, 1 registered, 0 retired | Message contains "5 tested, 1 registered, 0 retired" |
| `test_alerts_disabled` | Set `telegram_enabled = False` | `mock_telegram` stays empty after events |
| `test_config_section_parsed` | Load `[alerts]` from TOML | `config["alerts"]["digest_hour"] == 8` |
| `test_missing_token_skips_gracefully` | Unset `TELEGRAM_BOT_TOKEN` | No crash, warning logged |
| `test_telegram_api_error_retries` | Mock API returns 500 once, then 200 | Message eventually sent (1 retry) |

**Mocking**:
- `monkeypatch.setattr` on Telegram send function (no real API calls)
- `fakeredis` for event subscription tests
- `monkeypatch.delenv("TELEGRAM_BOT_TOKEN")` for missing-token test

**Run**: `pytest scripts/tests/test_alerts.py -v` and add to `make test_agent`.

---

## Test Infrastructure Summary

### New test files created across all phases

| Phase | Test File | Framework | Target |
|---|---|---|---|
| P1-2 | `scripts/tests/test_daemon_consolidation.py` | pytest | `make test_agent` |
| P1-3 | `scripts/tests/test_runner_consolidation.py` | pytest | `make test_agent` |
| P1-4 | `scripts/tests/test_state_store.py` | pytest | `make test_agent` |
| P1-4 | `scripts/tests/test_state_migration.py` | pytest | `make test_agent` |
| P1-5 | `scripts/tests/test_config_inheritance.py` | pytest | `make test_agent` |
| P1-6 | `scripts/tests/test_logging_config.py` | pytest | `make test_agent` |
| P1-7 | `scripts/tests/test_dashboard_cache.py` | pytest | `make test_dashboard` |
| P1-8 | `scripts/tests/test_daemon_integration.py` | pytest | `make test_integration` |
| P1-8 | `scripts/tests/test_multi_agent.py` | pytest | `make test_integration` |
| P1-8 | `scripts/tests/conftest.py` | pytest fixture | shared |
| P2-1 | `scripts/tests/test_research_output.py` | pytest | `make test_agent` |
| P2-2 | `rust/api/tests/research_api.rs` | cargo test | `cargo test --package api` |
| P2-2 | `scripts/tests/test_research_api_smoke.py` | pytest (integration) | `make test_api` |
| P2-3 | `rust/api/tests/research_ws.rs` | cargo test (tokio) | `cargo test --package api` |
| P2-3 | `scripts/tests/test_research_events.py` | pytest + fakeredis | `make test_agent` |
| P3-* | `web/src/__tests__/*.test.tsx` | Vitest/Jest + RTL | `cd web && npm test` |
| P3-* | `web/e2e/*.spec.ts` | Playwright | `cd web && npx playwright test` |
| P4-4 | `scripts/tests/test_alerts.py` | pytest | `make test_agent` |

### Makefile targets (new)

```makefile
test_integration:    ## Integration tests (daemon cycles, multi-agent)
	pytest scripts/tests/test_daemon_integration.py scripts/tests/test_multi_agent.py -v

test_api:            ## API smoke tests (requires running server)
	pytest scripts/tests/test_research_api_smoke.py -v -m integration

test_web:            ## Frontend unit + e2e tests
	cd web && npm test && npx playwright test

test_all:            ## All tests: Rust + Python + Frontend
	make test && make test_agent && make test_integration && make test_api && make test_web
```

### Shared patterns across all test files

| Pattern | Source | Reuse |
|---|---|---|
| `StubRunner(step_results=[...])` | `test_agent_base.py:23` | P1-2, P1-3, P1-8, P2-1 |
| `StubAgent(tmp_path)` | `test_agent_base.py:45` | P1-2, P1-8 |
| `make_synthetic_ticks(n, columns, seed)` | `algorithms/tests/conftest.py:74` | P1-8, integration tests |
| `HTTPServer on random port` | `test_agent_dashboard.py:20` | P1-7 |
| `monkeypatch.setenv("NAT_DATA_DIR")` | new pattern | P1-3, P1-8, P2-1 |
| `StateStore(tmp_path / "test.db")` | new pattern | P1-4, P1-8, P2-1, P2-2 |
| `fakeredis.FakeRedis()` | new dependency | P2-3, P4-4 |
| `pytest.mark.integration` | new marker | P2-2, P1-8 (skip in CI without server) |

---

## Dependency Graph

```
P1-1 (data layer) ────────────────────────────┐
P1-2 (daemons) ─────┐                         │
P1-3 (runners) ─────┤                         │
P1-5 (config) ──────┘                         │
      │                                        │
      ├──→ P1-6 (logging, needs consolidated code)
      │                                        │
      └──→ P1-8 (integration tests)           │
                                               │
P1-7 (dashboard cache) ── independent          │
                                               │
P1-4 (SQLite state) ←─────────────────────────┘
      │
      ├──→ P2-1 (structured hypothesis output)
      │         │
      │         ├──→ P2-2 (REST API)
      │         │         │
      │         │         ├──→ P2-3 (WebSocket stream)
      │         │         │         │
      │         │         │         ├──→ P4-1 (real-time frontend)
      │         │         │         └──→ P4-4 (alerts)
      │         │         │
      │         │         └──→ P3-1 (frontend scaffold)
      │         │                   │
      │         │                   ├──→ P3-2 (dashboard page)
      │         │                   ├──→ P3-3 (explorer page)
      │         │                   ├──→ P3-4 (detail page)
      │         │                   │         └──→ P4-2 (PDF export)
      │         │                   ├──→ P3-5 (heatmap page)
      │         │                   ├──→ P3-6 (registry page)
      │         │                   ├──→ P3-7 (math lab page)
      │         │                   ├──→ P3-8 (graveyard page)
      │         │                   ├──→ P3-9 (network page)
      │         │                   └──→ P4-3 (CI/CD)
```
