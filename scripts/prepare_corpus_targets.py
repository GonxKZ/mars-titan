"""Preparar objetivos residuales con un único proceso por destino. Por defecto solo comprueba."""

import argparse
import fcntl
import json
import os
from collections import Counter
from pathlib import Path

from mars_titan.data.cohort_files import read_manifest, safe_destination
from mars_titan.data.input_policy import INPUT_POLICIES, STRICT_INPUTS
from mars_titan.data.storage import sha256
from mars_titan.training.cohort_contract import cohort_identity
from mars_titan.training.corpus_targets import prepare_corpus_targets

BUSY = 3


def lock_path(output):
    """Situar el cerrojo junto al destino para no alterar su contenido reanudable."""
    return output.parent / f".{output.name}.lock"


def _lock_state(path):
    if not path.exists():
        return "free"
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return "busy"
    finally:
        os.close(descriptor)
    return "free"


def check(manifest, prepared, output, *, input_policy, target_factors=None):
    """Describir entradas y progreso sin escribir archivos ni calcular etiquetas."""
    meta, digest = read_manifest(manifest, 8 * 1024**2)
    cohort_identity(meta, input_policy=input_policy)
    safe_destination(output)
    if not prepared.is_dir() or (target_factors is not None and not target_factors.is_file()):
        raise ValueError("Falta el directorio preparado o la revisión de factores")
    labels = output / "labels"
    return dict(
        mode="check",
        input_policy=input_policy,
        manifest_sha256=digest,
        markets=dict(Counter(asset["market"] for asset in meta["assets"])),
        assets=len(meta["assets"]),
        target_factors_sha256=None if target_factors is None else sha256(target_factors),
        output_exists=output.exists(),
        configuration_exists=(output / "configuration.json").is_file(),
        confirmed_manifest=(output / "manifest.json").is_file(),
        receipts=sum(1 for _ in labels.glob("*/*/receipt.json")) if labels.is_dir() else 0,
        lock=_lock_state(lock_path(output)),
        targets_written=False,
    )


def execute(manifest, prepared, output, *, backend, input_policy, target_factors=None):
    """Retener el destino durante toda la preparación. Un segundo proceso se rechaza."""
    path = lock_path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return None
        result = prepare_corpus_targets(
            manifest,
            prepared,
            output,
            backend=backend,
            input_policy=input_policy,
            target_factors=target_factors,
        )
    finally:
        os.close(descriptor)
    return dict(
        mode="execute",
        input_policy=input_policy,
        counts=result["counts"],
        assets=len(result["assets"]),
        reused_assets=result["reused_assets"],
        final_test_opened=result["final_test_opened"],
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "prepared", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--target-factors", type=Path)
    parser.add_argument("--backend", choices=("numpy", "reference"), default="numpy")
    parser.add_argument("--input-policy", choices=INPUT_POLICIES, default=STRICT_INPUTS)
    parser.add_argument(
        "--execute", action="store_true", help="Calcular y confirmar las etiquetas del destino"
    )
    args = parser.parse_args(argv)
    options = dict(input_policy=args.input_policy, target_factors=args.target_factors)
    if not args.execute:
        report = check(args.manifest, args.prepared, args.output, **options)
        print(json.dumps(report, ensure_ascii=False))
        return BUSY if report["lock"] == "busy" else 0
    report = execute(args.manifest, args.prepared, args.output, backend=args.backend, **options)
    if report is None:
        print(json.dumps(dict(mode="execute", lock="busy"), ensure_ascii=False))
        return BUSY
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
