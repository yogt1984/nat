# Run, serve, and deploy targets
.PHONY: all run stop run_and_serve tunnel \
        show show_fast show_hft \
        api alerts serve_all \
        web_dev web_build web_install \
        docker_build docker_up docker_down docker_logs \
        serve_models serve_models_dev serve_best \
        agent_start agent_stop agent_status agent_report agent_dashboard \
        agent_watchdog_install agent_watchdog_remove \
        discovery_start discovery_once discovery_status discovery_stop \
        cascade_start cascade_once cascade_status cascade_stop cascade_report \
        monitor

all: run

# --- Ingestor ---

stop:
	@if [ -f .ing.pid ]; then \
		PID=$$(cat .ing.pid); \
		if kill -0 $$PID 2>/dev/null; then \
			echo "Stopping ingestor (PID $$PID)..."; \
			kill $$PID; \
			sleep 2; \
			kill -0 $$PID 2>/dev/null && kill -9 $$PID; \
		fi; \
		rm -f .ing.pid; \
	else \
		echo "No .ing.pid file found — ingestor may not be running"; \
	fi

run: release
	@$(MAKE) stop
	@echo "Running ingestor..."
	cd rust && ./target/release/ing ../config/ing.toml & echo $$! > ../.ing.pid
	@echo "Ingestor started (PID $$(cat .ing.pid))"

run_and_serve: release
	@$(MAKE) stop
	@echo "╔══════════════════════════════════════════════════════════════════╗"
	@echo "║           STARTING INGESTOR WITH LIVE DASHBOARD                  ║"
	@echo "╚══════════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "Dashboard: http://localhost:8080"
	@echo ""
	@echo "To expose to internet, run in another terminal:"
	@echo "  make tunnel"
	@echo ""
	cd rust && ING_DASHBOARD_ENABLED=true ./target/release/ing ../config/ing.toml & echo $$! > ../.ing.pid
	@echo "Ingestor started (PID $$(cat .ing.pid))"

tunnel:
	@echo "Starting cloudflare tunnel to localhost:8080..."
	@echo "Press Ctrl+C to stop"
	cloudflared tunnel --url http://localhost:8080

# --- Real-time monitoring ---

show: release
	@echo "Starting real-time feature display..."
	@echo "  Symbol: $(SYMBOL)"
	@echo "  Frequency: $(FREQ) Hz"
	@echo "Press Ctrl+C to stop"
	@echo ""
	cd rust && exec ./target/release/show_features $(SYMBOL) $(FREQ)

show_fast: FREQ=10
show_fast: show

show_hft: FREQ=50
show_hft: show

monitor:
	$(PYTHON) scripts/monitor.py

# --- API server ---

api: release_api
	@echo "Starting NAT API server..."
	@echo "  REST API: http://localhost:3000"
	@echo "  WebSocket: ws://localhost:3000/ws/stream/:symbol"
	@echo ""
	cd rust && exec ./target/release/nat-api

alerts: release_api
	@echo "Starting Telegram Alert Service..."
	@echo ""
	@echo "Required environment variables:"
	@echo "  TELEGRAM_BOT_TOKEN - Bot token from @BotFather"
	@echo "  TELEGRAM_CHAT_ID   - Your chat ID"
	@echo ""
	cd rust && exec ./target/release/alert-service

serve_all: release release_api
	@echo "╔══════════════════════════════════════════════════════════════════╗"
	@echo "║              STARTING FULL NAT STACK                             ║"
	@echo "╚══════════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "Services:"
	@echo "  - Ingestor:  Publishing features to Redis"
	@echo "  - API:       http://localhost:3000"
	@echo "  - Alerts:    Sending to Telegram"
	@echo ""
	@echo "Prerequisites:"
	@echo "  - Redis running on localhost:6379"
	@echo "  - TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID set"
	@echo ""
	@echo "Starting in tmux session 'nat'..."
	@echo "  (Use 'tmux attach -t nat' to reconnect)"
	@echo ""
	@tmux kill-session -t nat 2>/dev/null || true
	@tmux new-session -d -s nat -n ingestor 'cd rust && ./target/release/ing ../config/ing.toml; read'
	@tmux new-window -t nat -n api 'cd rust && ./target/release/nat-api; read'
	@tmux new-window -t nat -n alerts 'cd rust && ./target/release/alert-service; read'
	@tmux attach -t nat

# --- Frontend ---

web_dev:
	cd web && npm run dev

web_build:
	cd web && npm run build

web_install:
	cd web && npm install

# --- Docker ---

docker_build:
	@echo "Building Docker images..."
	docker-compose build

docker_up:
	@echo "Starting NAT stack with Docker..."
	docker-compose up -d
	@echo ""
	@echo "Services:"
	@echo "  - API: http://localhost:3000"
	@echo "  - Redis: localhost:6379"
	@echo ""
	@echo "View logs: make docker_logs"

docker_down:
	@echo "Stopping NAT stack..."
	docker-compose down

docker_logs:
	docker-compose logs -f

# --- Model serving ---

serve_models:
	@echo "╔══════════════════════════════════════════════════════════════════╗"
	@echo "║          STARTING MODEL SERVING API                              ║"
	@echo "╚══════════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "Server: http://$(HOST):$(PORT)"
	@echo "Health: http://$(HOST):$(PORT)/health"
	@echo "Docs:   http://$(HOST):$(PORT)/docs"
	@echo ""
	$(PYTHON) scripts/model_serving.py \
		--host $(HOST) \
		--port $(PORT) \
		--cache-size $(CACHE_SIZE)

serve_models_dev:
	@echo "╔══════════════════════════════════════════════════════════════════╗"
	@echo "║          MODEL SERVING API (DEV MODE - HOT RELOAD)               ║"
	@echo "╚══════════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "Server: http://$(HOST):$(PORT)"
	@echo "Docs:   http://$(HOST):$(PORT)/docs"
	@echo ""
	@echo "Changes to model_serving.py will auto-reload"
	@echo ""
	uvicorn scripts.model_serving:app \
		--reload \
		--host $(HOST) \
		--port $(PORT)

serve_best:
	@echo "╔══════════════════════════════════════════════════════════════════╗"
	@echo "║          SERVING BEST MODEL BY $(METRIC)                         ║"
	@echo "╚══════════════════════════════════════════════════════════════════╝"
	@echo ""
	$(PYTHON) scripts/model_serving.py \
		--host $(HOST) \
		--port $(PORT) \
		--serve-best \
		--metric $(METRIC) \
		--cache-size $(CACHE_SIZE)

# --- Research agent ---

agent_start:
	@echo "╔══════════════════════════════════════════════════════════════════╗"
	@echo "║          STARTING NAT RESEARCH AGENT                             ║"
	@echo "╚══════════════════════════════════════════════════════════════════╝"
	@echo ""
	@if [ -f .microstructure_agent.pid ]; then \
		PID=$$(cat .microstructure_agent.pid); \
		if kill -0 $$PID 2>/dev/null; then kill $$PID; sleep 1; fi; \
		rm -f .microstructure_agent.pid; \
	fi
	@tmux kill-session -t nat-agent 2>/dev/null || true
	tmux new-session -d -s nat-agent '$(PYTHON) scripts/agent/daemon.py start; read'
	@echo "Agent running in tmux session 'nat-agent'"
	@echo "  attach: tmux attach -t nat-agent"
	@echo "  status: make agent_status"

agent_stop:
	@if [ -f .microstructure_agent.pid ]; then \
		PID=$$(cat .microstructure_agent.pid); \
		if kill -0 $$PID 2>/dev/null; then \
			echo "Stopping NAT agent (PID $$PID)..."; \
			kill $$PID; sleep 2; \
			kill -0 $$PID 2>/dev/null && kill -9 $$PID; \
		fi; \
		rm -f .microstructure_agent.pid; \
		echo "Agent stopped"; \
	else \
		echo "No .microstructure_agent.pid — agent not running"; \
	fi

agent_status:
	@$(PYTHON) scripts/agent/daemon.py status

agent_report:
	@$(PYTHON) scripts/agent/daemon.py report

agent_dashboard:
	@echo "╔══════════════════════════════════════════════════════════════════╗"
	@echo "║          NAT AGENT DASHBOARD                                     ║"
	@echo "╚══════════════════════════════════════════════════════════════════╝"
	$(PYTHON) scripts/agent_dashboard.py --port $(AGENT_DASHBOARD_PORT)

agent_watchdog_install:
	@echo "Installing agent watchdog cron..."
	@(crontab -l 2>/dev/null | grep -v 'agent/daemon.py'; \
	  echo "*/5 * * * * pgrep -f 'agent/daemon.py start' >/dev/null || cd $(CURDIR) && $(PYTHON) scripts/agent/daemon.py start >> data/agent/watchdog.log 2>&1") | crontab -
	@echo "Watchdog installed (checks every 5 minutes)"

agent_watchdog_remove:
	@echo "Removing agent watchdog cron..."
	@(crontab -l 2>/dev/null | grep -v 'agent/daemon.py') | crontab -
	@echo "Watchdog removed"

# --- Discovery orchestrator ---

discovery_start:
	@echo "Starting Alpha Discovery Orchestrator..."
	@if [ -f .discovery_agent.pid ]; then \
		PID=$$(cat .discovery_agent.pid); \
		if kill -0 $$PID 2>/dev/null; then kill $$PID; sleep 1; fi; \
		rm -f .discovery_agent.pid; \
	fi
	@tmux kill-session -t nat-discovery 2>/dev/null || true
	tmux new-session -d -s nat-discovery '$(PYTHON) scripts/discovery_orchestrator.py --config $(DISCOVERY_CONFIG) start; read'
	@echo "Orchestrator running in tmux session 'nat-discovery'"

discovery_once:
	$(PYTHON) scripts/discovery_orchestrator.py --config $(DISCOVERY_CONFIG) once

discovery_status:
	@$(PYTHON) scripts/discovery_orchestrator.py --config $(DISCOVERY_CONFIG) status

discovery_stop:
	@if [ -f .discovery_agent.pid ]; then \
		PID=$$(cat .discovery_agent.pid); \
		if kill -0 $$PID 2>/dev/null; then \
			echo "Stopping Discovery Orchestrator (PID $$PID)..."; \
			kill $$PID; sleep 2; \
			kill -0 $$PID 2>/dev/null && kill -9 $$PID; \
		fi; \
		rm -f .discovery_agent.pid; \
		echo "Stopped"; \
	else \
		echo "No .discovery_agent.pid — orchestrator not running"; \
	fi

# --- Cascade validation agent ---

cascade_start:
	@echo "Starting Cascade Validation Agent..."
	@if [ -f .cascade_agent.pid ]; then \
		PID=$$(cat .cascade_agent.pid); \
		if kill -0 $$PID 2>/dev/null; then kill $$PID; sleep 1; fi; \
		rm -f .cascade_agent.pid; \
	fi
	@tmux kill-session -t nat-cascade 2>/dev/null || true
	tmux new-session -d -s nat-cascade '$(PYTHON) scripts/agent/cascade_daemon.py start; read'
	@echo "Cascade agent running in tmux session 'nat-cascade'"

cascade_once:
	$(PYTHON) scripts/agent/cascade_daemon.py once

cascade_status:
	@$(PYTHON) scripts/agent/cascade_daemon.py status

cascade_stop:
	@if [ -f .cascade_agent.pid ]; then \
		PID=$$(cat .cascade_agent.pid); \
		if kill -0 $$PID 2>/dev/null; then \
			echo "Stopping Cascade Agent (PID $$PID)..."; \
			kill $$PID; sleep 2; \
			kill -0 $$PID 2>/dev/null && kill -9 $$PID; \
		fi; \
		rm -f .cascade_agent.pid; \
		echo "Stopped"; \
	else \
		echo "No .cascade_agent.pid — cascade agent not running"; \
	fi

cascade_report:
	@$(PYTHON) scripts/agent/cascade_daemon.py report
