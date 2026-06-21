"""`nat config` — configuration inspection (show/get/validate/paths)."""

from __future__ import annotations

from cli.common import CONFIG_DIR, BOLD, W, R, G, nat_paths, _json_mode, _output, _banner, _p


def cmd_config_show(args):
    """Show full merged configuration."""
    config_dir = CONFIG_DIR
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib

    merged = {}
    for toml_file in sorted(config_dir.glob("*.toml")):
        try:
            with open(toml_file, "rb") as f:
                data = tomllib.load(f)
            merged[toml_file.stem] = data
        except Exception as e:
            merged[toml_file.stem] = {"_error": str(e)}

    if _json_mode(args):
        _output(merged, args)
        return

    _banner("Configuration")
    for name, data in merged.items():
        print(f"  {BOLD}[{name}]{W} ({config_dir / name}.toml)")
        if isinstance(data, dict) and "_error" in data:
            _p("x", R, f"Parse error: {data['_error']}")
        else:
            for key, val in data.items():
                if isinstance(val, dict):
                    print(f"    [{key}]")
                    for k2, v2 in val.items():
                        print(f"      {k2} = {v2}")
                else:
                    print(f"    {key} = {val}")
        print()


def cmd_config_get(args):
    """Get specific config value by dot-notation key."""
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib

    key_parts = args.key.split(".")
    if len(key_parts) < 2:
        print(f"  {R}Error:{W} Use file.section.key format, e.g. 'agent.agent.ic_threshold'")
        return 1

    config_file = CONFIG_DIR / f"{key_parts[0]}.toml"
    if not config_file.exists():
        print(f"  {R}Error:{W} Config file not found: {config_file}")
        return 1

    with open(config_file, "rb") as f:
        data = tomllib.load(f)

    val = data
    for part in key_parts[1:]:
        if isinstance(val, dict) and part in val:
            val = val[part]
        else:
            print(f"  {R}Error:{W} Key '{args.key}' not found")
            return 1

    if _json_mode(args):
        _output({"key": args.key, "value": val}, args)
    else:
        print(f"  {args.key} = {val}")


def cmd_config_paths(args):
    """Show the resolved data/config/log/report locations (and how each was chosen)."""
    info = nat_paths.describe()

    def _human(d):
        print(f"\n  {BOLD}Resolved paths{W}  (source: {d['source']})\n")
        order = ["install_root", "home", "data_root", "features_dir", "trades_dir",
                 "config_dir", "reports_dir", "log_dir", "db_path"]
        for k in order:
            print(f"    {k:<14} {d[k]}")
        print(f"\n  Override with NAT_HOME (master) or NAT_DATA/NAT_CONFIG/NAT_REPORTS/NAT_LOG.\n")

    _output(info, args, _human)
    return 0


def cmd_config_validate(args):
    """Validate all TOML config files for syntax errors."""
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib

    config_dir = CONFIG_DIR
    results = []
    for toml_file in sorted(config_dir.glob("*.toml")):
        try:
            with open(toml_file, "rb") as f:
                tomllib.load(f)
            results.append({"file": toml_file.name, "valid": True})
        except Exception as e:
            results.append({"file": toml_file.name, "valid": False, "error": str(e)})

    if _json_mode(args):
        _output({"configs": results, "all_valid": all(r["valid"] for r in results)}, args)
        return

    for r in results:
        if r["valid"]:
            _p("*", G, f"{r['file']}: OK")
        else:
            _p("x", R, f"{r['file']}: {r['error']}")


def register(sub):
    cfg_p = sub.add_parser('config', help='Configuration inspection')
    cfg_p.set_defaults(func=cmd_config_show)
    cfgsub = cfg_p.add_subparsers(dest='subcmd')
    cfgsub.add_parser('show', help='Full config dump (all TOML files merged)').set_defaults(func=cmd_config_show)
    cfg_get = cfgsub.add_parser('get', help='Get specific config value (file.section.key)')
    cfg_get.add_argument('key', help='Config key in dot notation (e.g., agent.agent.ic_threshold)')
    cfg_get.set_defaults(func=cmd_config_get)
    cfgsub.add_parser('validate', help='Check all config files for syntax errors').set_defaults(func=cmd_config_validate)
    cfg_paths = cfgsub.add_parser('paths', help='Show resolved data/config/log/report locations')
    cfg_paths.add_argument('--json', action='store_true', help='JSON output')
    cfg_paths.set_defaults(func=cmd_config_paths)


__all__ = ["cmd_config_show", "cmd_config_get", "cmd_config_paths", "cmd_config_validate", "register"]
