"""`nat api` — REST/WebSocket API server + Telegram alert services."""

from __future__ import annotations

from cli.common import RUST, ING_CFG, _ensure_release, _cargo, _banner, _exec


def cmd_api_start(args):
    _ensure_release()
    _cargo("build --locked --release --bin nat-api")
    _banner("NAT API server")
    print("  REST: http://localhost:3000")
    print("  WebSocket: ws://localhost:3000/ws/stream/:symbol\n")
    _exec("./target/release/nat-api", cwd=RUST)


def cmd_api_alerts(args):
    _cargo("build --locked --release --bin nat-api")
    _banner("Telegram alert service")
    _exec("./target/release/alert-service", cwd=RUST)


def cmd_api_serve_all(args):
    _ensure_release()
    _cargo("build --locked --release --bin nat-api")
    _banner("Starting full NAT stack in tmux")
    _exec("tmux kill-session -t nat 2>/dev/null || true")
    _exec(f"tmux new-session -d -s nat -n ingestor "
          f"'cd {RUST} && ./target/release/ing {ING_CFG}; read'")
    _exec(f"tmux new-window -t nat -n api 'cd {RUST} && ./target/release/nat-api; read'")
    _exec(f"tmux new-window -t nat -n alerts 'cd {RUST} && ./target/release/alert-service; read'")
    _exec("tmux attach -t nat")


def register(sub):
    api_p = sub.add_parser('api', help='API & alert services')
    api_p.set_defaults(func=lambda a: api_p.print_help())
    asub = api_p.add_subparsers(dest='subcmd')
    asub.add_parser('start', help='Start API server').set_defaults(func=cmd_api_start)
    asub.add_parser('alerts', help='Telegram alerts').set_defaults(func=cmd_api_alerts)
    asub.add_parser('serve-all', help='Full stack in tmux').set_defaults(func=cmd_api_serve_all)


__all__ = ["cmd_api_start", "cmd_api_alerts", "cmd_api_serve_all", "register"]
