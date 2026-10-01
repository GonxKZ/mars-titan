"""Activar la comparativa tabular solo tras confirmar la campaña neuronal completa."""

import argparse
from pathlib import Path

from mars_titan.data.cohort_files import read_manifest, safe_destination
from mars_titan.data.storage import atomic_json, outside_source, sha256

from .checkpoints import StopRequest
from .tabular_search import _artifact, run_tabular_search


def reference_view(path, arm="US"):
    """Verificar recibos y artefactos antes de compartir la población de un brazo."""
    path = Path(path)
    summary, digest = read_manifest(path, 8 * 1024**2)
    if (
        arm not in {"US", "CN", "US+CN"}
        or summary.get("kind") != "reference_search"
        or summary.get("status") != "completed"
        or summary.get("scope") != "full_corpus"
        or summary.get("cohort_complete") is not True
        or summary.get("final_test_opened") is not False
        or type(summary.get("planned_runs")) is not int
        or not 1 <= summary["planned_runs"] <= 512
        or summary.get("completed_runs") != summary["planned_runs"]
        or not isinstance(summary.get("runs"), list)
        or len(summary["runs"]) != summary["planned_runs"]
        or not all(item.get("status") == "completed" for item in summary["runs"])
    ):
        raise ValueError("La campaña neuronal todavía no confirma todos sus resultados")
    root = path.parent
    view = root / "views" / f"{arm}.json"
    safe_destination(view)
    population, population_hash = read_manifest(view, 8 * 1024**2)
    if (
        population.get("scope") != "full_corpus"
        or population.get("cohort_complete") is not True
        or population.get("final_test_opened", False) is not False
        or population.get("source_manifest_sha256") != summary["identity"]["manifest_sha256"]
    ):
        raise ValueError("La vista no conserva el origen y alcance de la campaña neuronal")
    found, seen = 0, set()
    for item in summary["runs"]:
        relative = Path(item["path"])
        if relative.is_absolute() or ".." in relative.parts or item["path"] in seen:
            raise ValueError("Una ejecución tiene una ruta ajena o duplicada")
        seen.add(item["path"])
        _artifact(root, dict(path=str(relative / "run.json"), sha256=item["report_sha256"]))
        folder = root / relative
        report, _ = read_manifest(folder / "run.json", 8 * 1024**2)
        if (
            report.get("status") != "completed"
            or report.get("final_test_opened") is not False
            or set(report.get("predictions", {})) != {"train", "validation"}
        ):
            raise ValueError("Un resultado neuronal no está terminado o utiliza el test")
        _artifact(folder, report["checkpoint"])
        for record in report["predictions"].values():
            _artifact(folder, record)
        if item["arm"] == arm and item["weighting"] == "natural":
            if (
                report["identity"]["manifest_sha256"] != population_hash
                or report["identity"].get("weighting", "natural") != "natural"
                or report.get("samples") != population["counts"]
                or report.get("scope") != "full_corpus"
                or report.get("cohort_complete") is not True
            ):
                raise ValueError("La población de la referencia y de su vista no coincide")
            found += 1
    if not found or sha256(path) != digest or sha256(view) != population_hash:
        raise ValueError("Faltan referencias naturales del brazo o cambiaron sus recibos")
    return dict(
        manifest=str(view.resolve()),
        manifest_sha256=population_hash,
        reference_summary_sha256=digest,
        confirmed_runs=summary["planned_runs"],
        matched_runs=found,
        arm=arm,
        counts=population["counts"],
        final_test_opened=False,
    )


def run_queue(config, reference, output, *, arm="US", stop=None):
    reference, output = Path(reference), Path(output)
    safe_destination(output)
    outside_source(reference.parent, output)
    outside_source(output, reference.parent)
    proof = reference_view(reference, arm)
    receipt = output.with_name(output.name + "-gate.json")
    safe_destination(receipt)
    if receipt.exists() and read_manifest(receipt)[0] != proof:
        raise ValueError("La cola corresponde a otra campaña o población neuronal")
    atomic_json(receipt, proof)
    return run_tabular_search(
        config,
        Path(proof["manifest"]),
        output,
        stop=stop,
        resume=output.exists(),
        expected_source_hash=proof["manifest_sha256"],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "reference", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--arm", choices=("US", "CN", "US+CN"), default="US")
    args = parser.parse_args()
    with StopRequest() as stop:
        result = run_queue(args.config, args.reference, args.output, arm=args.arm, stop=stop)
    print(f"Estado: {result['status']}. Casos terminados: {result['completed_runs']}")


if __name__ == "__main__":
    main()
