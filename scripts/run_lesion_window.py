"""Ejecutar una ventana de corrección existente con diagnósticos B4 independientes."""

import argparse
import json
from contextlib import nullcontext
from pathlib import Path

from mars_titan.training.learning_hold import require_learning_allowed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("view", "parent", "study", "lesions", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--search-case", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--indices", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    args = parser.parse_args(argv)
    require_learning_allowed("la ventana de diagnóstico B4")

    import torch

    from mars_titan.memory.lesion_study import freeze_study, load_study, select_arm
    from mars_titan.training.checkpoints import StopRequest
    from mars_titan.training.experiment_resources import GpuLease
    from mars_titan.training.mars_titan_correction import run_correction_window

    study, digest = load_study(args.lesions)
    recipe, components, arms_digest = select_arm(args.study, args.arm)
    freeze_study(
        args.output,
        study=study,
        study_sha256=digest,
        arms_sha256=arms_digest,
        arm=args.arm,
        seed=args.seed,
        search_case=args.search_case,
        recipe=recipe,
    )
    torch.backends.mha.set_fastpath_enabled(False)
    lease = GpuLease() if args.device == "cuda:0" else nullcontext()
    with StopRequest() as stop, lease:
        report = run_correction_window(
            args.view,
            args.parent,
            recipe,
            components=components,
            seed=args.seed,
            output=args.output,
            search_case=args.search_case,
            indices=args.indices,
            device=args.device,
            stop=stop,
            lesions=study.protocol,
        )
    if report["status"] not in ("completed", "paused"):
        raise ValueError("La ventana B4 devolvió un estado desconocido")
    print(
        json.dumps(
            dict(status=report["status"], run_id=report.get("run_id"), output=str(args.output)),
            ensure_ascii=False,
        )
    )
    return 0 if report["status"] == "completed" else 3


if __name__ == "__main__":
    raise SystemExit(main())
