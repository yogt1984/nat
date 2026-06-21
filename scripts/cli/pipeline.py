"""`nat pipeline` — automated pipeline state machine (build/ingest/analyze)."""

from __future__ import annotations

import sys

from cli.common import ROOT, PIPE_CFG, R, W, _banner, _py, _p, _json_mode, _output


def _pipe_cfg(args):
    return getattr(args, 'config', None) or str(PIPE_CFG)


def cmd_pipeline_start(args):
    _banner("Starting automated pipeline")
    _py(f"scripts/pipeline_runner.py --config {_pipe_cfg(args)} start")


def cmd_pipeline_resume(args):
    _py(f"scripts/pipeline_runner.py --config {_pipe_cfg(args)} resume")


def cmd_pipeline_analyze(args):
    _banner("Analyzing existing data")
    _py(f"scripts/pipeline_runner.py --config {_pipe_cfg(args)} analyze")


def cmd_pipeline_stop(args):
    _py(f"scripts/pipeline_runner.py --config {_pipe_cfg(args)} stop")


def cmd_pipeline_status(args):
    if _json_mode(args):
        p = ROOT / "data" / "pipeline_state.json"
        if p.exists():
            print(p.read_text())
        else:
            _output({"error": "No pipeline state file", "state": "UNKNOWN"}, args)
        return
    _py(f"scripts/pipeline_runner.py --config {_pipe_cfg(args)} status")


def cmd_pipeline_dashboard(args):
    try:
        import dash  # noqa: F401
    except ImportError:
        _p("x", R, "Missing dash dependency. Install with:")
        _p(" ", W, "pip install dash>=2.14.0")
        sys.exit(1)

    port = getattr(args, 'port', 8050)
    _banner("Pipeline dashboard")
    _py(f"scripts/dashboard.py --config {_pipe_cfg(args)} --port {port}")


def register(sub):
    pipe_p = sub.add_parser('pipeline', help='Automated pipeline')
    pipe_p.add_argument('--config', default=str(PIPE_CFG), help='Pipeline config TOML')
    pipe_p.set_defaults(func=lambda a: pipe_p.print_help())
    psub = pipe_p.add_subparsers(dest='subcmd')
    psub.add_parser('start', help='Start pipeline').set_defaults(func=cmd_pipeline_start)
    psub.add_parser('resume', help='Resume from saved state').set_defaults(func=cmd_pipeline_resume)
    psub.add_parser('analyze', help='Analyze existing data').set_defaults(func=cmd_pipeline_analyze)
    psub.add_parser('stop', help='Stop pipeline').set_defaults(func=cmd_pipeline_stop)
    psub.add_parser('status', help='Show state').set_defaults(func=cmd_pipeline_status)
    pd = psub.add_parser('dashboard', help='Pipeline dashboard')
    pd.add_argument('--port', type=int, default=8050, help='Dashboard port (default: 8050)')
    pd.set_defaults(func=cmd_pipeline_dashboard)


__all__ = ["cmd_pipeline_start", "cmd_pipeline_resume", "cmd_pipeline_analyze",
           "cmd_pipeline_stop", "cmd_pipeline_status", "cmd_pipeline_dashboard",
           "register"]
