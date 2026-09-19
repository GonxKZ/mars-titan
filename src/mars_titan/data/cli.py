"""Órdenes locales para preparar y verificar datos, sin entrenar modelos por defecto."""

import argparse
import json
import sys
from pathlib import Path

from .inventory import inventory
from .storage import atomic_json, sha256


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("inventory", help="Inventariar toda la copia sin modificarla")
    audit.add_argument("--source", type=Path, default=Path("dataset"))
    audit.add_argument(
        "--database", type=Path, default=Path("data/interim/source-inventory.sqlite")
    )
    audit.add_argument("--report", type=Path)
    audit.add_argument("--verify", action="store_true", help="Recalcular todas las huellas")
    panels = commands.add_parser("select", help="Seleccionar un panel técnico por sector y volumen")
    panels.add_argument("--source", type=Path, default=Path("dataset"))
    panels.add_argument(
        "--database", type=Path, default=Path("data/interim/source-inventory.sqlite")
    )
    panels.add_argument("--market", choices=["US", "CN"], required=True)
    panels.add_argument("--per-sector", type=int, default=2)
    panels.add_argument("--output", type=Path, required=True)
    prepare = commands.add_parser(
        "prepare", help="Normalizar el panel, sin afirmar aptitud de entrenamiento"
    )
    prepare.add_argument("--source", type=Path, default=Path("dataset"))
    prepare.add_argument("--panel", type=Path, required=True)
    prepare.add_argument("--output", type=Path, default=Path("data/processed/phase1"))
    prepare.add_argument("--limit", type=int)
    prepare.add_argument("--start")
    prepare.add_argument("--end", default="2026-01-01")
    args = parser.parse_args()
    from .storage import outside_source

    if args.command == "inventory":
        result = inventory(args.source, args.database, verify=args.verify)
        if args.report:
            outside_source(args.source, args.report)
            atomic_json(args.report, result)
    elif args.command == "select":
        from .preparation import candidates, select_panel

        outside_source(args.source, args.output)
        pool = candidates(args.source, args.database, args.market)
        selected = select_panel(pool, per_sector=args.per_sector)
        result = {
            "market": args.market,
            "purpose": "engineering_profile_only",
            "seed": 42,
            "selection": "median_then_90th_percentile_source_bytes_per_current_sector",
            "candidate_assets": len(pool),
            "assets": selected,
            "source_database_sha256": sha256(args.database),
        }
        atomic_json(args.output, result)
    else:
        from .preparation import prepare_asset
        from .temporal import MarketClock

        panel = json.loads(args.panel.read_text())
        start = args.start or ("1990-01-01" if panel["market"] == "US" else "2000-01-01")
        clock = MarketClock(panel["market"], start, args.end)
        assets = panel["assets"][: args.limit] if args.limit is not None else panel["assets"]
        if args.limit is not None and args.limit < 1:
            raise ValueError("limit must be positive")
        summaries = []
        for asset in assets:
            item = prepare_asset(args.source, args.output, asset, clock)
            brief = {k: item[k] for k in ("symbol", "counts", "elapsed_seconds", "reused")}
            summaries.append(brief)
            print(json.dumps(brief), file=sys.stderr, flush=True)
        result = {"market": panel["market"], "prepared": summaries, "training_ready": False}
    print(json.dumps(result, ensure_ascii=True, indent=2))
    return 2 if result.get("errors") else 0


if __name__ == "__main__":
    raise SystemExit(main())
