"""Operational monitoring daemons.

`ops.gap_alert` (T0b) pages via Telegram within minutes when feature ingestion
stalls — the real-time complement to the next-day nightly report. Data
continuity is the project's binding constraint, so a silent ingestion gap is the
most expensive failure to miss.
"""
