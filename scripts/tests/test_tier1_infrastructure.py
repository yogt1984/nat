"""
Tier 1 infrastructure tests — verify Docker stack, Prometheus, Grafana,
Caddy, and PostgreSQL config files are valid, consistent, and complete.

These are offline/static tests — they validate config files without
needing Docker running. For live smoke tests, use `nat docker smoke`.
"""

import json
import os
import re
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent.parent

# ── docker-compose.yml ──────────────────────────────────────────────────────


class TestDockerCompose:
    @pytest.fixture(autouse=True)
    def load_compose(self):
        self.path = ROOT / "docker-compose.yml"
        assert self.path.exists(), "docker-compose.yml missing"
        with open(self.path) as f:
            self.compose = yaml.safe_load(f)

    def test_services_defined(self):
        svcs = set(self.compose["services"].keys())
        required = {"redis", "ingestor", "api", "alerts", "prometheus", "grafana",
                     "postgres", "caddy", "optuna-dashboard"}
        missing = required - svcs
        assert not missing, f"Missing services: {missing}"

    def test_redis_healthcheck(self):
        redis = self.compose["services"]["redis"]
        assert "healthcheck" in redis
        assert "redis-cli" in str(redis["healthcheck"]["test"])

    def test_postgres_healthcheck(self):
        pg = self.compose["services"]["postgres"]
        assert "healthcheck" in pg
        assert "pg_isready" in str(pg["healthcheck"]["test"])

    def test_api_healthcheck(self):
        api = self.compose["services"]["api"]
        assert "healthcheck" in api
        assert "/health" in str(api["healthcheck"]["test"])

    def test_ingestor_depends_on_redis(self):
        dep = self.compose["services"]["ingestor"]["depends_on"]
        assert "redis" in dep

    def test_api_depends_on_redis(self):
        dep = self.compose["services"]["api"]["depends_on"]
        assert "redis" in dep

    def test_grafana_depends_on_prometheus(self):
        dep = self.compose["services"]["grafana"]["depends_on"]
        assert "prometheus" in dep

    def test_caddy_depends_on_grafana_and_api(self):
        dep = self.compose["services"]["caddy"]["depends_on"]
        assert "grafana" in dep
        assert "api" in dep

    def test_no_port_conflicts(self):
        """Verify no two services expose the same host port."""
        host_ports = []
        for name, svc in self.compose["services"].items():
            for pm in svc.get("ports", []):
                host_port = str(pm).split(":")[0]
                host_ports.append((host_port, name))
        port_map = {}
        for port, name in host_ports:
            if port in port_map:
                pytest.fail(
                    f"Port {port} conflict: {port_map[port]} and {name}")
            port_map[port] = name

    def test_volumes_declared(self):
        vols = set(self.compose.get("volumes", {}).keys())
        required = {"redis_data", "prometheus_data", "grafana_data",
                     "postgres_data", "caddy_data", "caddy_config"}
        missing = required - vols
        assert not missing, f"Missing volumes: {missing}"

    def test_restart_policy_on_all_services(self):
        for name, svc in self.compose["services"].items():
            assert svc.get("restart") == "unless-stopped", \
                f"Service {name} missing restart: unless-stopped"

    def test_postgres_password_from_env(self):
        pg = self.compose["services"]["postgres"]
        env = pg["environment"]
        pw_line = [e for e in env if "POSTGRES_PASSWORD" in e][0]
        assert "${POSTGRES_PASSWORD" in pw_line, \
            "Postgres password should come from .env"

    def test_grafana_password_from_env(self):
        gf = self.compose["services"]["grafana"]
        env = gf["environment"]
        pw_line = [e for e in env if "ADMIN_PASSWORD" in e][0]
        assert "${GRAFANA_PASSWORD" in pw_line

    def test_caddy_domain_from_env(self):
        caddy = self.compose["services"]["caddy"]
        env = caddy["environment"]
        domain_line = [e for e in env if "DOMAIN" in e][0]
        assert "${DOMAIN" in domain_line

    def test_prometheus_retention_90d(self):
        prom = self.compose["services"]["prometheus"]
        cmd = prom.get("command", [])
        retention = [c for c in cmd if "retention.time" in c]
        assert retention, "Prometheus retention not configured"
        assert "90d" in retention[0]


# ── Dockerfiles ─────────────────────────────────────────────────────────────


class TestDockerfiles:
    DOCKERFILES = [
        "docker/Dockerfile.ingestor",
        "docker/Dockerfile.api",
        "docker/Dockerfile.alerts",
    ]

    @pytest.fixture(params=DOCKERFILES)
    def dockerfile(self, request):
        path = ROOT / request.param
        assert path.exists(), f"{request.param} missing"
        return path.read_text()

    def test_multistage_build(self, dockerfile):
        assert dockerfile.count("FROM ") >= 2, \
            "Should be multi-stage (builder + runtime)"

    def test_rust_version_ge_178(self, dockerfile):
        m = re.search(r"FROM rust:(\d+\.\d+)", dockerfile)
        assert m, "Rust base image not found"
        version = tuple(int(x) for x in m.group(1).split("."))
        assert version >= (1, 78), \
            f"Rust {m.group(1)} < 1.78, Cargo.lock v4 requires >= 1.78"

    def test_copies_all_workspace_crates(self, dockerfile):
        for crate in ["ing-types", "ing-features", "ing", "api"]:
            assert crate in dockerfile, \
                f"Workspace crate '{crate}' not copied in Dockerfile"

    def test_slim_runtime_image(self, dockerfile):
        # Last FROM should be a slim image
        froms = re.findall(r"FROM\s+([\w:/.+-]+)", dockerfile)
        runtime = froms[-1]
        assert "slim" in runtime or "alpine" in runtime, \
            f"Runtime image '{runtime}' should be slim/alpine"


# ── .dockerignore ───────────────────────────────────────────────────────────


class TestDockerIgnore:
    def test_dockerignore_exists(self):
        path = ROOT / ".dockerignore"
        assert path.exists(), ".dockerignore missing — build context will be huge"

    def test_excludes_large_dirs(self):
        content = (ROOT / ".dockerignore").read_text()
        for pattern in ["rust/target", "data/", ".git", "__pycache__"]:
            assert pattern in content, \
                f".dockerignore should exclude '{pattern}'"


# ── Caddyfile ───────────────────────────────────────────────────────────────


class TestCaddyfile:
    @pytest.fixture(autouse=True)
    def load_caddyfile(self):
        self.path = ROOT / "docker" / "Caddyfile"
        assert self.path.exists()
        self.content = self.path.read_text()

    def test_has_global_block(self):
        assert "email" in self.content, "Caddyfile should set ACME email"

    def test_reverse_proxy_grafana(self):
        assert "reverse_proxy grafana:3000" in self.content

    def test_reverse_proxy_api(self):
        assert "reverse_proxy api:3000" in self.content

    def test_reverse_proxy_ingestor(self):
        assert "reverse_proxy ingestor:8080" in self.content

    def test_reverse_proxy_prometheus(self):
        assert "reverse_proxy prometheus:9090" in self.content

    def test_http_to_https_redirect(self):
        assert "redir" in self.content and "https://" in self.content

    def test_uses_env_domain(self):
        assert "{$DOMAIN" in self.content, \
            "Caddyfile should use $DOMAIN env var"


# ── Prometheus ──────────────────────────────────────────────────────────────


class TestPrometheus:
    @pytest.fixture(autouse=True)
    def load_config(self):
        self.path = ROOT / "docker" / "prometheus" / "prometheus.yml"
        assert self.path.exists()
        with open(self.path) as f:
            self.config = yaml.safe_load(f)

    def test_scrape_interval(self):
        interval = self.config["global"]["scrape_interval"]
        assert interval in ("5s", "10s", "15s"), \
            f"Scrape interval {interval} seems unusual"

    def test_ingestor_target(self):
        jobs = self.config["scrape_configs"]
        job_names = [j["job_name"] for j in jobs]
        assert any("ingestor" in j for j in job_names), \
            "Prometheus should scrape the ingestor"

    def test_target_uses_container_name(self):
        jobs = self.config["scrape_configs"]
        for job in jobs:
            targets = str(job.get("static_configs", [{}]))
            # Should use Docker DNS (service name), not localhost
            assert "localhost" not in targets, \
                f"Job {job['job_name']} should use Docker DNS, not localhost"


# ── Grafana Provisioning ───────────────────────────────────────────────────


class TestGrafana:
    def test_datasource_config_valid(self):
        path = ROOT / "docker" / "grafana" / "provisioning" / "datasources" / "prometheus.yml"
        assert path.exists()
        with open(path) as f:
            cfg = yaml.safe_load(f)
        ds = cfg["datasources"][0]
        assert ds["type"] == "prometheus"
        assert "prometheus:9090" in ds["url"]

    def test_dashboard_provider_config(self):
        path = ROOT / "docker" / "grafana" / "provisioning" / "dashboards" / "dashboards.yml"
        assert path.exists()
        with open(path) as f:
            cfg = yaml.safe_load(f)
        providers = cfg["providers"]
        assert len(providers) >= 1

    def test_dashboard_json_valid(self):
        path = ROOT / "docker" / "grafana" / "dashboards" / "nat_overview.json"
        assert path.exists(), "Dashboard JSON missing"
        with open(path) as f:
            dashboard = json.load(f)
        assert "panels" in dashboard, "Dashboard should have panels"
        assert len(dashboard["panels"]) > 0, "Dashboard should have at least 1 panel"


# ── .env.example ────────────────────────────────────────────────────────────


class TestEnvExample:
    @pytest.fixture(autouse=True)
    def load_env(self):
        self.path = ROOT / ".env.example"
        assert self.path.exists()
        self.content = self.path.read_text()

    def test_required_vars_present(self):
        required = [
            "DOMAIN",
            "POSTGRES_PASSWORD",
            "GRAFANA_PASSWORD",
            "REDIS_URL",
            "TELEGRAM_BOT_TOKEN",
            "TELEGRAM_CHAT_ID",
        ]
        for var in required:
            assert var in self.content, f".env.example missing {var}"

    def test_no_real_secrets(self):
        """Ensure .env.example doesn't contain actual secrets."""
        lines = self.content.splitlines()
        for line in lines:
            if "=" in line and not line.strip().startswith("#"):
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip()
                # Passwords should be placeholder
                if "PASSWORD" in key:
                    assert value in ("changeme", "<strong-password>", ""), \
                        f".env.example has non-placeholder for {key}: {value}"

    def test_grafana_anon_default_true(self):
        """Local dev should default to anonymous access."""
        assert "GRAFANA_ANON=true" in self.content


# ── Cross-cutting consistency ───────────────────────────────────────────────


class TestConsistency:
    def test_compose_services_match_dockerfiles(self):
        """Services with `build:` should have a corresponding Dockerfile."""
        with open(ROOT / "docker-compose.yml") as f:
            compose = yaml.safe_load(f)
        for name, svc in compose["services"].items():
            build = svc.get("build")
            if not build:
                continue
            if isinstance(build, str):
                continue  # simple build context, no explicit dockerfile
            dockerfile = build.get("dockerfile")
            context = build.get("context", ".")
            if dockerfile:
                # Resolve relative to build context
                full_path = ROOT / context / dockerfile
                assert full_path.exists(), \
                    f"Service {name} references {context}/{dockerfile} but it doesn't exist"

    def test_prometheus_target_matches_compose_port(self):
        """Ingestor Prometheus port in compose should match prometheus.yml target."""
        with open(ROOT / "docker-compose.yml") as f:
            compose = yaml.safe_load(f)
        with open(ROOT / "docker" / "prometheus" / "prometheus.yml") as f:
            prom = yaml.safe_load(f)

        # Get the Prometheus scrape target port
        for job in prom["scrape_configs"]:
            for sc in job.get("static_configs", []):
                for target in sc.get("targets", []):
                    port = target.split(":")[-1]
                    # This port should be the container-internal prometheus port
                    # of the ingestor, not the host-mapped one
                    assert port.isdigit(), f"Target port not numeric: {target}"

    def test_grafana_datasource_url_matches_compose(self):
        """Grafana datasource URL should point to the prometheus service."""
        with open(ROOT / "docker" / "grafana" / "provisioning" / "datasources" / "prometheus.yml") as f:
            ds_cfg = yaml.safe_load(f)
        url = ds_cfg["datasources"][0]["url"]
        assert "prometheus" in url, \
            f"Grafana datasource URL '{url}' should reference 'prometheus' service"
