"""Órdenes locales para preparar y verificar datos, sin entrenar modelos por defecto."""

import argparse
import json
from pathlib import Path

from .inventory import inventory
from .storage import atomic_json


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
    args = parser.parse_args()
    result = inventory(args.source, args.database, verify=args.verify)
    if args.report:
        from .storage import outside_source

        outside_source(args.source, args.report)
        atomic_json(args.report, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 2 if result.get("errors") else 0


if __name__ == "__main__":
    raise SystemExit(main())
