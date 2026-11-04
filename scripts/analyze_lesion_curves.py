"""Analizar lesiones B4 de una ventana completada, sin abrir el test final."""

import argparse
import json
from pathlib import Path

from mars_titan.evaluation.lesion_curves import analyze_lesion_run


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("report", "study", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument(
        "--partition", choices=("validation", "calibration", "evaluation"), required=True
    )
    parser.add_argument("--batch-size", type=int, default=4096)
    args = parser.parse_args(argv)
    result = analyze_lesion_run(
        args.report,
        args.study,
        partition=args.partition,
        output=args.output,
        batch_size=args.batch_size,
    )
    print(
        json.dumps(
            dict(status=result["status"], rows=result["rows"], output=str(args.output)),
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
