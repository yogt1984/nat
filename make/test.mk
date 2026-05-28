# Testing & validation
.PHONY: test test_verbose test_hypotheses \
        validate validate_all validate_api validate_positions validate_whales validate_entropy \
        validate_data validate_data_recent validate-config \
        test_api test_redis test_integration \
        test_cluster_quality test_cluster_quality_cov \
        test_pipeline test_pipeline_cov test_pipeline_runner \
        test_dashboard test_15m test_agent test_serving \
        test_backtest test_backtest_cov \
        eamm_test eamm_test_integration

# --- Rust unit tests ---

test:
	@echo "╔══════════════════════════════════════════════════════════════════╗"
	@echo "║                    RUNNING ALL UNIT TESTS                        ║"
	@echo "╚══════════════════════════════════════════════════════════════════╝"
	@echo ""
	cd rust && cargo test --package ing
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════════════╗"
	@echo "║                    ALL UNIT TESTS PASSED                         ║"
	@echo "╚══════════════════════════════════════════════════════════════════╝"

test_verbose:
	@echo "Running all tests (verbose)..."
	cd rust && cargo test --package ing -- --nocapture

# --- Live API validation ---

validate: release
	@echo "╔══════════════════════════════════════════════════════════════════╗"
	@echo "║           RUNNING ALL VALIDATIONS (SKEPTICAL MODE)               ║"
	@echo "╚══════════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "[1/4] API Connection Validation..."
	@cd rust && ./target/release/validate_api
	@echo ""
	@echo "[2/4] Position Tracking Validation..."
	@cd rust && ./target/release/validate_positions
	@echo ""
	@echo "[3/4] Whale Identification Validation..."
	@cd rust && ./target/release/validate_whales
	@echo ""
	@echo "[4/4] Tick Entropy Validation..."
	@cd rust && ./target/release/validate_entropy
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════════════╗"
	@echo "║                  ALL VALIDATIONS COMPLETE                        ║"
	@echo "╚══════════════════════════════════════════════════════════════════╝"

validate_all: validate

validate_api: release
	@echo "Running API connection validation..."
	cd rust && ./target/release/validate_api

validate_positions: release
	@echo "Running position tracking validation..."
	cd rust && ./target/release/validate_positions

validate_whales: release
	@echo "Running whale identification validation..."
	cd rust && ./target/release/validate_whales

validate_entropy: release
	@echo "Running tick entropy feature validation..."
	cd rust && ./target/release/validate_entropy

# --- Data validation ---

validate-config:
	$(PYTHON) -m scripts.utils.validate_config

validate_data:
	@echo "╔══════════════════════════════════════════════════════════════════╗"
	@echo "║                 VALIDATING COLLECTED DATA                        ║"
	@echo "╚══════════════════════════════════════════════════════════════════╝"
	$(PYTHON) scripts/validate_data.py ./data/features --verbose

validate_data_recent:
	@echo "Validating last $(HOURS) hours of data..."
	$(PYTHON) scripts/validate_data.py ./data/features --hours $(HOURS) --verbose

# --- Hypothesis tests ---

test_hypotheses: release
	@echo "╔══════════════════════════════════════════════════════════════════╗"
	@echo "║         RUNNING HYPOTHESIS TESTING ON COLLECTED DATA             ║"
	@echo "╚══════════════════════════════════════════════════════════════════╝"
	@echo ""
	cd rust && ./target/release/test_hypotheses ../$(DATA)

# --- API / integration tests ---

test_api:
	@echo "Testing API endpoints..."
	bash scripts/test_api.sh

test_redis:
	@echo "Testing Redis connection..."
	@redis-cli ping && echo "✓ Redis is running" || (echo "✗ Redis not running" && exit 1)
	@echo ""
	@echo "Checking cached symbols..."
	@redis-cli KEYS "nat:latest:*" || true
	@echo ""
	@echo "To subscribe to features: redis-cli SUBSCRIBE nat:features:BTC"
	@echo "To subscribe to alerts:   redis-cli SUBSCRIBE nat:alerts"

test_integration:
	@echo "Running integration tests..."
	bash scripts/test_integration.sh

# --- Python test suites ---

test_cluster_quality:
	@echo "╔══════════════════════════════════════════════════════════════════╗"
	@echo "║          TESTING CLUSTER QUALITY METRICS                         ║"
	@echo "╚══════════════════════════════════════════════════════════════════╝"
	@echo ""
	$(PYTHON) -m pytest scripts/cluster_quality/tests/ -v

test_cluster_quality_cov:
	@echo "Running cluster quality tests with coverage..."
	$(PYTHON) -m pytest scripts/cluster_quality/tests/ -v --cov=cluster_quality --cov-report=term-missing

test_pipeline:
	@echo "╔══════════════════════════════════════════════════════════════════╗"
	@echo "║             RUNNING CLUSTER PIPELINE TESTS                      ║"
	@echo "╚══════════════════════════════════════════════════════════════════╝"
	@echo ""
	cd scripts && $(PYTHON) -m pytest tests/test_cluster_loader.py tests/test_cluster_preprocess.py tests/test_cluster_engine.py tests/test_cluster_reduce.py tests/test_cluster_viz.py -v

test_pipeline_cov:
	@echo "Running cluster pipeline tests with coverage..."
	cd scripts && $(PYTHON) -m pytest tests/test_cluster_loader.py tests/test_cluster_preprocess.py tests/test_cluster_engine.py tests/test_cluster_reduce.py tests/test_cluster_viz.py -v --cov=cluster_pipeline --cov-report=term-missing

test_pipeline_runner:
	@echo "╔══════════════════════════════════════════════════════════════════╗"
	@echo "║          TESTING PIPELINE RUNNER                                 ║"
	@echo "╚══════════════════════════════════════════════════════════════════╝"
	@echo ""
	cd scripts && $(PYTHON) -m pytest tests/test_pipeline_runner.py -v

test_dashboard:
	@echo "╔══════════════════════════════════════════════════════════════════╗"
	@echo "║          TESTING PIPELINE DASHBOARD                              ║"
	@echo "╚══════════════════════════════════════════════════════════════════╝"
	@echo ""
	cd scripts && $(PYTHON) -m pytest tests/test_dashboard.py -v

test_15m:
	@echo "╔══════════════════════════════════════════════════════════════════╗"
	@echo "║          TESTING 15-MINUTE SMOKE TEST                            ║"
	@echo "╚══════════════════════════════════════════════════════════════════╝"
	@echo ""
	cd scripts && $(PYTHON) -m pytest tests/test_15m_test.py -v

test_agent:
	@echo "╔══════════════════════════════════════════════════════════════════╗"
	@echo "║          TESTING AGENT SUBSYSTEM                                 ║"
	@echo "╚══════════════════════════════════════════════════════════════════╝"
	@echo ""
	cd scripts && $(PYTHON) -m pytest tests/test_agent_base.py tests/test_agent_cache.py tests/test_agent_dashboard.py tests/test_agent_monitor.py tests/test_agent_ensemble.py tests/test_mf_agent.py tests/test_macro_agent.py tests/test_meta_agent.py -v

test_serving:
	@echo "╔══════════════════════════════════════════════════════════════════╗"
	@echo "║          TESTING MODEL SERVING API                               ║"
	@echo "╚══════════════════════════════════════════════════════════════════╝"
	@echo ""
	$(PYTHON) -m pytest scripts/tests/test_model_serving.py -v

test_backtest:
	@echo "Running backtest tests..."
	cd scripts && $(PYTHON) -m pytest backtest/tests/ -v

test_backtest_cov:
	@echo "Running backtest tests with coverage..."
	cd scripts && $(PYTHON) -m pytest backtest/tests/ -v --cov=backtest --cov-report=term-missing

eamm_test:
	@echo "Running EAMM test suite..."
	cd scripts && $(PYTHON) -m pytest eamm/tests/ -v

eamm_test_integration:
	@echo "Running EAMM integration tests..."
	cd scripts && $(PYTHON) -m pytest eamm/tests/test_integration.py -v
