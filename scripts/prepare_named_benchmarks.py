"""Describir o preparar benchmarks identificados, sin entrenamiento ni evaluación."""

import argparse
import json
from pathlib import Path

from mars_titan.data.cohort_files import read_manifest
from mars_titan.simulation.benchmark_catalog import benchmark_catalog, prepare_benchmark_profile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/benchmarks/scenarios.json"))
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--describe", action="store_true")
    mode.add_argument("--output", type=Path)
    parser.add_argument("--profile")
    args = parser.parse_args()
    if args.describe and args.profile or args.output and not args.profile:
        parser.error("Usa --describe sin perfil o --output con --profile")
    settings, _ = read_manifest(args.config, 64 * 1024)
    if args.describe:
        result = benchmark_catalog(settings)
    else:
        report = prepare_benchmark_profile(settings, args.profile, args.output)
        result = {key: report[key] for key in ("suite_id", "profile", "status", "identity_sha256")}
        result.update(worlds=len(report["records"]), scientific_evaluation_executed=False)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
