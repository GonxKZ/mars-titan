"""Ejecutar piloto, comparación y auditoría congelada dentro de 168 horas activas."""

import argparse
import json
import signal
from pathlib import Path

from mars_titan.simulation.adaptive_campaign import DEFAULT_CONFIG, run_adaptive_campaign
from mars_titan.training.gpu_supervisor import StopFlag


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--build-identity", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--pilot-only", action="store_true")
    args = parser.parse_args(argv)
    stop = StopFlag()
    handlers = {name: signal.signal(name, stop.request) for name in (signal.SIGINT, signal.SIGTERM)}
    try:
        result = run_adaptive_campaign(
            args.scenarios,
            args.binary,
            args.output,
            config=args.config,
            build_identity=args.build_identity,
            resume=args.resume,
            pilot_only=args.pilot_only,
            stop=lambda: stop.requested,
        )
        print(json.dumps(result, ensure_ascii=False))
        return 0 if result["status"] == "completed" else 2 if result["status"] == "paused" else 1
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Error: {error}")
        return 1
    finally:
        for name, handler in handlers.items():
            signal.signal(name, handler)


if __name__ == "__main__":
    raise SystemExit(main())
