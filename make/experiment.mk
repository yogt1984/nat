# Experiment runner, EAMM, 15-min smoke test, trade visualization
.PHONY: exp_start exp_stop exp_status exp_check exp_midweek exp_analyze \
        exp_dashboard exp_tunnel explore \
        15m 15m_offline 15m_viz trade_viz \
        eamm_run eamm_regime eamm_backtest

# --- Experiment runner ---

exp_start: release
	@$(PYTHON) scripts/run_experiment.py start

exp_stop:
	@$(PYTHON) scripts/run_experiment.py stop

exp_status:
	@$(PYTHON) scripts/run_experiment.py status

exp_check:
	@$(PYTHON) scripts/run_experiment.py check --hours $(HOURS)

exp_midweek:
	@$(PYTHON) scripts/run_experiment.py midweek

exp_analyze:
	@$(PYTHON) scripts/run_experiment.py analyze

exp_dashboard:
	@$(PYTHON) scripts/run_experiment.py dashboard

exp_tunnel:
	@echo "Exposing dashboard at localhost:8050 via Cloudflare tunnel..."
	cloudflared tunnel --url http://localhost:8050

explore:
	@echo "Launching feature exploration notebook..."
	jupyter notebook notebooks/explore_features.ipynb

# --- 15-minute smoke test ---

15m:
	@echo "╔══════════════════════════════════════════════════════════════════╗"
	@echo "║          15-MINUTE EXPERIMENT                                    ║"
	@echo "╚══════════════════════════════════════════════════════════════════╝"
	@echo ""
	nat 15m

15m_offline:
	@echo "╔══════════════════════════════════════════════════════════════════╗"
	@echo "║          15-MINUTE ANALYSIS (OFFLINE)                            ║"
	@echo "╚══════════════════════════════════════════════════════════════════╝"
	@echo ""
	$(PYTHON) scripts/15m_test.py run --data-dir $(SMOKE_DATA) --output $(SMOKE_OUTPUT) -v

15m_viz:
	@echo "╔══════════════════════════════════════════════════════════════════╗"
	@echo "║          15-MINUTE VISUAL HEALTH CHECK                           ║"
	@echo "╚══════════════════════════════════════════════════════════════════╝"
	@echo ""
	$(PYTHON) scripts/15m_visualize.py \
		$(if $(filter command line,$(origin SMOKE_DATA)),--data-dir $(SMOKE_DATA),--latest) \
		--symbol $(SYMBOL) $(if $(WINDOW),--window $(WINDOW)) -v

trade_viz:
	@echo "╔══════════════════════════════════════════════════════════════════╗"
	@echo "║          PAPER TRADE VISUALIZATION                               ║"
	@echo "╚══════════════════════════════════════════════════════════════════╝"
	@echo ""
	$(PYTHON) scripts/trade_visualize.py \
		$(if $(TRADE_DATE),--date $(TRADE_DATE),--latest) \
		--symbol $(TRADE_SYMBOL) -v

# --- EAMM ---

eamm_run:
	@echo "Running EAMM full pipeline ($(EAMM_SYMBOL), horizon=$(EAMM_HORIZON))..."
	cd scripts && $(PYTHON) -m eamm.cli run --symbol $(EAMM_SYMBOL) --horizon $(EAMM_HORIZON) --mode $(EAMM_MODE)

eamm_regime:
	@echo "Running EAMM regime analysis ($(EAMM_SYMBOL))..."
	cd scripts && $(PYTHON) -m eamm.cli regime --symbol $(EAMM_SYMBOL) --horizon $(EAMM_HORIZON)

eamm_backtest:
	@echo "Running EAMM backtest ($(EAMM_SYMBOL), gamma=$(EAMM_GAMMA))..."
	cd scripts && $(PYTHON) -m eamm.cli backtest --symbol $(EAMM_SYMBOL) --horizon $(EAMM_HORIZON) --gamma $(EAMM_GAMMA) --q-max $(EAMM_QMAX)
