"""Comparar los seis controles E2 de una ventana completada y declarada previamente."""

import argparse
import json
from pathlib import Path

from mars_titan.evaluation.spaced_comparison import analyze_spaced_runs
from mars_titan.memory.spaced_study import ARMS


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, metavar="VARIANTE=INFORME")
    for name in ("study", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument(
        "--partition", choices=("validation", "calibration", "evaluation"), required=True
    )
    parser.add_argument("--batch-size", type=int, default=4096)
    args = parser.parse_args(argv)
    reports = {}
    for entry in args.run:
        name, separator, path = entry.partition("=")
        if not separator or not path or name not in ARMS or name in reports:
            parser.error("Cada --run necesita una variante E2 distinta y su informe")
        reports[name] = Path(path)
    result = analyze_spaced_runs(
        reports,
        args.study,
        partition=args.partition,
        output=args.output,
        batch_size=args.batch_size,
    )
    print(json.dumps(dict(status=result["status"], output=str(args.output)), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
