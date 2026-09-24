"""Ejecutar los comparadores financieros con datos confirmados y test cerrado."""

import argparse
import json
from pathlib import Path

from mars_titan.data.cohort_files import read_manifest
from mars_titan.simulation.campaign import run_campaign
from mars_titan.simulation.storage import read_tape
from mars_titan.training.checkpoints import StopRequest
from mars_titan.training.experiment_resources import GpuLease


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-tape", type=Path, required=True)
    parser.add_argument("--validation-tape", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/simulation/comparators.json"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--diagnostic", action="store_true", help="Prueba técnica explícita en CPU, hasta 32 pasos"
    )
    args = parser.parse_args()
    config, _ = read_manifest(args.config, 1024**2)
    train, validation = (
        read_tape(args.train_tape),
        [read_tape(path) for path in args.validation_tape],
    )
    with StopRequest() as stop:
        options = dict(resume=args.resume, diagnostic=args.diagnostic, stop=lambda: stop.requested)
        if args.diagnostic:
            report = run_campaign(train, validation, args.output, config, **options)
        else:
            with GpuLease() as lease:
                report = run_campaign(
                    train, validation, args.output, config, lease=lease, **options
                )
    print(
        json.dumps(
            {k: report[k] for k in ("status", "domain", "training_runs", "evaluations")},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
