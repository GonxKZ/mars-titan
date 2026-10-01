"""Preparar mundos técnicos de adaptación con una reserva independiente de auditoría."""

import argparse
import json
from pathlib import Path

from mars_titan.data.cohort_files import read_manifest
from mars_titan.simulation.adaptation_scenarios import prepare_adaptation_scenarios


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/simulation/adaptation-scenarios.json")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--fit-hmm", action="store_true", help="Necesita uv run --with hmmlearn==0.3.3"
    )
    args = parser.parse_args()
    settings, _ = read_manifest(args.config, 1024**2)
    result = prepare_adaptation_scenarios(settings, args.output, fit_markov=args.fit_hmm)
    print(
        json.dumps(
            dict(
                status=result["status"],
                worlds=len(result["records"]),
                analysis_domain="technical",
                final_test_opened=False,
            )
        )
    )


if __name__ == "__main__":
    main()
