# Automated pipeline & cluster pipeline targets
.PHONY: pipeline_start pipeline_resume pipeline_analyze pipeline_stop pipeline_status \
        dashboard scan_schema

# --- Automated pipeline state machine ---

pipeline_start:
	@echo "╔══════════════════════════════════════════════════════════════════╗"
	@echo "║          STARTING AUTOMATED PIPELINE                             ║"
	@echo "║  State machine: IDLE → BUILD → INGEST → COLLECT → ANALYZE → DONE║"
	@echo "╚══════════════════════════════════════════════════════════════════╝"
	@echo ""
	$(PYTHON) scripts/pipeline_runner.py --config $(PIPELINE_CONFIG) start

pipeline_resume:
	@echo "Resuming pipeline from saved state..."
	$(PYTHON) scripts/pipeline_runner.py --config $(PIPELINE_CONFIG) resume

pipeline_analyze:
	@echo "╔══════════════════════════════════════════════════════════════════╗"
	@echo "║          ANALYZING EXISTING DATA (skip ingestion)                ║"
	@echo "╚══════════════════════════════════════════════════════════════════╝"
	@echo ""
	$(PYTHON) scripts/pipeline_runner.py --config $(PIPELINE_CONFIG) analyze

pipeline_stop:
	@echo "Stopping ingestor..."
	$(PYTHON) scripts/pipeline_runner.py --config $(PIPELINE_CONFIG) stop

pipeline_status:
	@$(PYTHON) scripts/pipeline_runner.py --config $(PIPELINE_CONFIG) status

# --- Dashboard ---

dashboard:
	@echo "╔══════════════════════════════════════════════════════════════════╗"
	@echo "║           NAT PIPELINE DASHBOARD                                ║"
	@echo "╚══════════════════════════════════════════════════════════════════╝"
	$(PYTHON) scripts/dashboard.py --config $(PIPELINE_CONFIG) --port $(DASHBOARD_PORT)

# --- Schema scan ---

scan_schema:
	@echo "╔══════════════════════════════════════════════════════════════════╗"
	@echo "║            SCANNING PARQUET SCHEMA & VECTOR COVERAGE            ║"
	@echo "╚══════════════════════════════════════════════════════════════════╝"
	@echo ""
	cd scripts && $(PYTHON) -c "from cluster_pipeline.loader import print_schema_summary; print_schema_summary('../$(DATA)')"
