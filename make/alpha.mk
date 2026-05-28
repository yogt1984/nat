# Alpha pipeline, backtesting, signal testing, model training, OOS validation
.PHONY: alpha_pipeline alpha_pipeline_resume alpha_pipeline_force \
        alpha_pipeline_status alpha_pipeline_gates alpha_pipeline_step \
        signal_test signal_test_all \
        backtest backtest_validate backtest_ml backtest_ml_validate backtest_ml_quantile \
        backtest_list backtest_ml_tracked run_ml_workflow \
        train_gmm train_gmm_auto \
        train_baseline list_models score_data score_and_save \
        experiments_list experiments_list_stage experiments_get experiments_compare experiments_best \
        analyze_clusters analyze_clusters_gmm analyze_all_symbols \
        test_alg1 test_alg1_paper test_alg1_live test_oos30 \
        oos_validate oos_watch oos_report

# --- Alpha pipeline ---

alpha_pipeline:
	@echo "╔══════════════════════════════════════════════════════════════════╗"
	@echo "║          STARTING ALPHA PIPELINE                                ║"
	@echo "║  SCREEN → COMBINE → SIZE → VALIDATE → REGIME → PORTFOLIO → … ║"
	@echo "╚══════════════════════════════════════════════════════════════════╝"
	@echo ""
	$(PYTHON) scripts/alpha/alpha_pipeline.py --config $(ALPHA_CONFIG) start

alpha_pipeline_resume:
	@echo "Resuming alpha pipeline from saved state..."
	$(PYTHON) scripts/alpha/alpha_pipeline.py --config $(ALPHA_CONFIG) resume

alpha_pipeline_force:
	@echo "Resuming alpha pipeline (forcing past failed gate)..."
	$(PYTHON) scripts/alpha/alpha_pipeline.py --config $(ALPHA_CONFIG) resume --force-gate

alpha_pipeline_status:
	@$(PYTHON) scripts/alpha/alpha_pipeline.py --config $(ALPHA_CONFIG) status

alpha_pipeline_gates:
	@$(PYTHON) scripts/alpha/alpha_pipeline.py --config $(ALPHA_CONFIG) gates

alpha_pipeline_step:
	$(PYTHON) scripts/alpha/alpha_pipeline.py --config $(ALPHA_CONFIG) run-step $(STEP)

# --- Signal testing ---

signal_test:
	@echo "╔══════════════════════════════════════════════════════════════════╗"
	@echo "║          PHASE 1: SIGNAL EXISTENCE TEST                          ║"
	@echo "╚══════════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "  Symbol:       $(SYMBOL)"
	@echo "  Horizon:      $(HORIZON) rows ($$(echo "$(HORIZON) * 0.1" | bc)s)"
	@echo "  Spread:       $(SPREAD_BPS) bps"
	@echo "  Remove leaky: $(REMOVE_LEAKY)"
	@echo ""
	$(PYTHON) scripts/phase1_signal_test.py \
		--symbol $(SYMBOL) \
		--horizon $(HORIZON) \
		--spread-bps $(SPREAD_BPS) \
		$(if $(filter 1,$(REMOVE_LEAKY)),--remove-leaky,)

signal_test_all:
	@echo "╔══════════════════════════════════════════════════════════════════╗"
	@echo "║          PHASE 1: FULL SIGNAL SWEEP (all symbols + configs)      ║"
	@echo "╚══════════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "=== BTC — all features ===" && \
	$(PYTHON) scripts/phase1_signal_test.py --symbol BTC --horizon $(HORIZON) --spread-bps $(SPREAD_BPS) && \
	echo "" && \
	echo "=== BTC — leaky removed ===" && \
	$(PYTHON) scripts/phase1_signal_test.py --symbol BTC --horizon $(HORIZON) --spread-bps $(SPREAD_BPS) --remove-leaky && \
	echo "" && \
	echo "=== ETH — all features ===" && \
	$(PYTHON) scripts/phase1_signal_test.py --symbol ETH --horizon $(HORIZON) --spread-bps $(SPREAD_BPS) && \
	echo "" && \
	echo "=== SOL — all features ===" && \
	$(PYTHON) scripts/phase1_signal_test.py --symbol SOL --horizon $(HORIZON) --spread-bps $(SPREAD_BPS)

# --- Backtesting ---

backtest:
	@echo "╔══════════════════════════════════════════════════════════════════╗"
	@echo "║                    RUNNING BACKTEST                              ║"
	@echo "╚══════════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "Strategy: $(STRATEGY)"
	@echo "Data: $(DATA)"
	@echo ""
	$(PYTHON) scripts/run_backtest.py --data-dir $(DATA) --symbol $(SYMBOL) --strategy $(STRATEGY)

backtest_validate:
	@echo "╔══════════════════════════════════════════════════════════════════╗"
	@echo "║              WALK-FORWARD VALIDATION                             ║"
	@echo "╚══════════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "Strategy: $(STRATEGY)"
	@echo "Data: $(DATA)"
	@echo ""
	$(PYTHON) scripts/run_backtest.py --data-dir $(DATA) --symbol $(SYMBOL) --strategy $(STRATEGY) --walk-forward

backtest_ml:
	@echo "╔══════════════════════════════════════════════════════════════════╗"
	@echo "║                 ML MODEL BACKTEST                                ║"
	@echo "╚══════════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "Predictions: $(ML_PREDICTIONS)"
	@echo "Entry Threshold: $(ML_ENTRY)"
	@echo "Exit Threshold: $(ML_EXIT)"
	@echo "Data: $(DATA)"
	@echo ""
	$(PYTHON) scripts/run_backtest.py --data-dir $(DATA) --symbol $(SYMBOL) \
		--ml-predictions $(ML_PREDICTIONS) \
		--ml-entry-threshold $(ML_ENTRY) \
		--ml-exit-threshold $(ML_EXIT)

backtest_ml_validate:
	@echo "╔══════════════════════════════════════════════════════════════════╗"
	@echo "║          ML MODEL WALK-FORWARD VALIDATION                        ║"
	@echo "╚══════════════════════════════════════════════════════════════════╝"
	@echo ""
	$(PYTHON) scripts/run_backtest.py --data-dir $(DATA) --symbol $(SYMBOL) \
		--ml-predictions $(ML_PREDICTIONS) \
		--ml-entry-threshold $(ML_ENTRY) \
		--ml-exit-threshold $(ML_EXIT) \
		--walk-forward

backtest_ml_quantile:
	@echo "╔══════════════════════════════════════════════════════════════════╗"
	@echo "║            ML MODEL BACKTEST (QUANTILE)                          ║"
	@echo "╚══════════════════════════════════════════════════════════════════╝"
	@echo ""
	$(PYTHON) scripts/run_backtest.py --data-dir $(DATA) --symbol $(SYMBOL) \
		--ml-predictions $(ML_PREDICTIONS) \
		--ml-quantile \
		--ml-entry-threshold $(ML_ENTRY_Q) \
		--ml-exit-threshold $(ML_EXIT_Q) \
		--walk-forward

backtest_list:
	@$(PYTHON) scripts/run_backtest.py --list-strategies

backtest_ml_tracked:
	@echo "╔══════════════════════════════════════════════════════════════════╗"
	@echo "║          ML BACKTEST WITH TRACKING                               ║"
	@echo "╚══════════════════════════════════════════════════════════════════╝"
	@echo ""
	$(PYTHON) scripts/run_backtest_tracked.py \
		--ml-predictions $(ML_PREDICTIONS) \
		--ml-entry-threshold $(ML_ENTRY) \
		--ml-exit-threshold $(ML_EXIT) \
		--ml-direction $(ML_DIRECTION) \
		--walk-forward \
		--output $(BACKTEST_JSON)

run_ml_workflow:
	@echo "╔══════════════════════════════════════════════════════════════════╗"
	@echo "║          COMPLETE ML WORKFLOW WITH TRACKING                      ║"
	@echo "╚══════════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "Step 1: Training model..."
	$(MAKE) train_baseline SNAPSHOT=$(SNAPSHOT) MODEL_TYPE=$(MODEL_TYPE)
	@echo ""
	@echo "Step 2: Generating predictions..."
	$(MAKE) score_and_save MODEL_PATH=models/$(MODEL_TYPE)_*.* PREDICTIONS=$(PREDICTIONS)
	@echo ""
	@echo "Step 3: Running backtest with tracking..."
	$(PYTHON) scripts/run_backtest_tracked.py \
		--ml-predictions $(PREDICTIONS) \
		--ml-entry-threshold $(ML_ENTRY) \
		--ml-exit-threshold $(ML_EXIT) \
		--walk-forward \
		--output $(BACKTEST_JSON)
	@echo ""
	@echo "═══════════════════════════════════════════════════════════════════"
	@echo "WORKFLOW COMPLETE - All stages tracked automatically"
	@echo "═══════════════════════════════════════════════════════════════════"
	@echo ""
	@echo "View experiment:"
	@echo "  make experiments_list"

# --- Model training ---

train_gmm:
	@echo "╔══════════════════════════════════════════════════════════════════╗"
	@echo "║               TRAINING GMM REGIME CLASSIFIER                     ║"
	@echo "╚══════════════════════════════════════════════════════════════════╝"
	@echo ""
	@mkdir -p models
	$(PYTHON) scripts/train_regime_gmm.py --data-dir $(DATA) --output-dir models

train_gmm_auto:
	@echo "Training GMM with auto-selected components..."
	@mkdir -p models
	$(PYTHON) scripts/train_regime_gmm.py --data-dir $(DATA) --output-dir models --auto-select

train_baseline:
	@echo "╔══════════════════════════════════════════════════════════════════╗"
	@echo "║              TRAINING BASELINE MODEL                             ║"
	@echo "╚══════════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "Snapshot: $(SNAPSHOT)"
	@echo "Model: $(MODEL_TYPE)"
	@echo "Output: $(MODEL_DIR)"
	@echo ""
	@mkdir -p $(MODEL_DIR)
	$(PYTHON) scripts/train_baseline.py --snapshot $(SNAPSHOT) --model $(MODEL_TYPE) --output-dir $(MODEL_DIR)

list_models:
	@echo "Listing saved models..."
	$(PYTHON) scripts/list_models.py --model-dir $(MODEL_DIR)

score_data:
	@echo "╔══════════════════════════════════════════════════════════════════╗"
	@echo "║              SCORING DATA WITH TRAINED MODEL                     ║"
	@echo "╚══════════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "Model: $(MODEL_PATH)"
	@echo "Data: $(DATA)"
	@echo ""
	$(PYTHON) scripts/score_data.py --model $(MODEL_PATH) --data $(DATA) --evaluate

score_and_save:
	@echo "Scoring and saving predictions..."
	$(PYTHON) scripts/score_data.py --model $(MODEL_PATH) --data $(DATA) --output $(PREDICTIONS) --evaluate

# --- Experiment tracking ---

experiments_list:
	@echo "╔══════════════════════════════════════════════════════════════════╗"
	@echo "║                   TRACKED EXPERIMENTS                            ║"
	@echo "╚══════════════════════════════════════════════════════════════════╝"
	@echo ""
	$(PYTHON) scripts/experiment_tracking.py list

experiments_list_stage:
	@echo "Listing experiments at stage: $(STAGE)"
	$(PYTHON) scripts/experiment_tracking.py list --stage $(STAGE)

experiments_get:
	@echo "Getting experiment details: $(EXP_ID)"
	$(PYTHON) scripts/experiment_tracking.py get $(EXP_ID)

experiments_compare:
	@echo "Comparing experiments..."
	$(PYTHON) scripts/experiment_tracking.py compare $(EXP_IDS)

experiments_best:
	@echo "Finding best experiment by $(METRIC)..."
	$(PYTHON) scripts/experiment_tracking.py best --metric $(METRIC)

# --- Cluster analysis ---

analyze_clusters:
	@echo "╔══════════════════════════════════════════════════════════════════╗"
	@echo "║              ANALYZING CLUSTER QUALITY                           ║"
	@echo "╚══════════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "Symbol: $(SYMBOL)"
	@echo "Data: $(DATA)"
	@echo ""
	$(PYTHON) scripts/analyze_clusters.py --data-dir $(DATA) --symbol $(SYMBOL) --hours $(HOURS)

analyze_clusters_gmm:
	@echo "Analyzing with trained GMM model..."
	$(PYTHON) scripts/analyze_clusters.py --data-dir $(DATA) --symbol $(SYMBOL) --model models/regime_gmm.json

analyze_all_symbols:
	@for sym in BTC ETH SOL; do \
		echo "Analyzing $$sym..."; \
		$(PYTHON) scripts/analyze_clusters.py --data-dir $(DATA) --symbol $$sym --output reports/cluster_$$sym.txt; \
	done

# --- ALG1: MF 3-Feature Liquidity Signal ---

test_alg1:
	@echo "═══ ALG1: MF 3-Feature Liquidity Signal (100min) ═══"
	@echo ""
	@echo "Step 1: Backtest on latest data..."
	$(PYTHON) scripts/analysis/mf_liquidity_backtest.py --features both --save
	@echo ""
	@echo "Step 2: Starting signal bridge (dry-run, Ctrl-C to stop)..."
	$(PYTHON) scripts/execution/signal_bridge.py --mode dry-run --cycle 300

test_alg1_paper:
	@echo "═══ ALG1: Paper Trading Mode ═══"
	$(PYTHON) scripts/alpha/paper_trader.py batch --save
	@echo ""
	@echo "Starting paper trader watch (Ctrl-C to stop)..."
	$(PYTHON) scripts/alpha/paper_trader.py watch --symbol BTC --poll 300

test_alg1_live:
	@echo "═══ ALG1: LIVE MODE (requires HL_PRIVATE_KEY) ═══"
	$(PYTHON) scripts/execution/signal_bridge.py --mode live --cycle 300

# --- OOS validation ---

test_oos30:
	@echo "═══ OOS30: 30-Day Out-of-Sample Validation ═══"
	@echo ""
	@echo "Step 1/3: 3f liquidity signal..."
	$(PYTHON) scripts/alpha/paper_trader.py batch --save
	@echo ""
	@echo "Step 2/3: Generic algorithms (jump_detector, funding_reversion, optimal_entry)..."
	$(PYTHON) scripts/alpha/paper_trader_generic.py --algorithms jump_detector funding_reversion optimal_entry --save
	@echo ""
	@echo "Step 3/3: Surprise signal..."
	$(PYTHON) scripts/alpha/paper_trader_surprise.py batch --save
	@echo ""
	@echo "Done. Reports saved to reports/"

oos_validate:
	@echo "Running OOS validation (4 winning algos)..."
	$(PYTHON) scripts/oos_validate.py batch

oos_watch:
	@echo "Starting OOS validation watcher..."
	$(PYTHON) scripts/oos_validate.py watch

oos_report:
	$(PYTHON) scripts/oos_terminal.py
