"""Monitoring/observability (T18).

`monitoring.metrics_exporter` is the keystone that makes NAT's SQLite/JSON state
(lifecycle funnel, live P&L, paper performance) scrapable by Prometheus, so the
Grafana dashboards have a datasource. Grafana cannot query SQLite/JSON directly.
"""
