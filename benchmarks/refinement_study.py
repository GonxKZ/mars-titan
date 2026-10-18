"""Ensayo por etapas de rutas de consulta sobre un padre elegido antes de 2022."""

import argparse
import copy
import hashlib
import json
import platform
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

from mars_titan.data.cohort_files import read_manifest, safe_destination
from mars_titan.data.input_policy import HISTORICAL_MASKED
from mars_titan.data.storage import atomic_json, outside_source, sha256
from mars_titan.training.learning_hold import require_learning_allowed

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/experiments/refinement-development-v1.json"
FOLDS = {"fold-017": 2022, "fold-018": 2023}


def configuration():
    return read_manifest(CONFIG)[0]


def _window(source, fold):
    year = FOLDS.get(fold)
    if year is None:
        raise ValueError("La ventana no pertenece al estudio de desarrollo")
    expected = dict(
        id=fold,
        train=["2000-01-01", f"{year - 1}-04-01"],
        validation=[f"{year - 1}-04-01", f"{year - 1}-10-01"],
        calibration=[f"{year - 1}-10-01", f"{year}-01-01"],
        evaluation=[f"{year}-01-01", f"{year + 1}-01-01"],
    )
    temporal = source.get("temporal_view", {})
    if (
        source.get("final_test_opened") is not False
        or source.get("input_policy") != HISTORICAL_MASKED
        or temporal.get("fold") != expected
        or temporal.get("protocol", {}).get("market") != "US"
    ):
        raise ValueError("La fuente no conserva la ventana histórica declarada")


def _assets(source):
    assets = source["assets"]
    by_symbol = {a["symbol"]: a for a in assets}
    if len(assets) != len(by_symbol) or any(a["market"] != "US" for a in assets):
        raise ValueError("Hay activos duplicados o ajenos al mercado US")
    return by_symbol


def choose_assets(source, count):
    """Elegir por huella fija entre activos con historia de ajuste anterior a abril de 2021.

    La elegibilidad usa el recuento de filas de ajuste aceptadas de la vista. No usa
    su valor objetivo ni la cobertura de validación o de años posteriores.
    """
    _window(source, "fold-017")
    assets = _assets(source)
    if any(
        type(a["counts"]["train"]) is not int or a["counts"]["train"] < 0 for a in assets.values()
    ):
        raise ValueError("Los activos necesitan recuentos de ajuste enteros y no negativos")
    candidates = [symbol for symbol, a in assets.items() if a["counts"]["train"] > 0]
    if type(count) is not int or not 1 <= count <= min(64, len(candidates)):
        raise ValueError("No hay un número válido y suficiente de activos")
    salt = configuration()["cohort_salt"]
    candidates.sort(
        key=lambda symbol: (hashlib.sha256(f"{salt}/US/{symbol}".encode()).digest(), symbol)
    )
    return tuple(sorted(candidates[:count]))


def project_view(source, symbols, fold, epochs):
    """Proyectar activos completos sin filtrar filas por presencia, resultado o cobertura futura."""
    _window(source, fold)
    original = _assets(source)
    if (
        not symbols
        or len(symbols) != len(set(symbols))
        or not set(symbols) <= set(original)
        or type(epochs) is not int
        or not 1 <= epochs <= 10
    ):
        raise ValueError("Los activos y las épocas no corresponden a una proyección admitida")
    result = copy.deepcopy(source)
    assets = [copy.deepcopy(original[s]) for s in sorted(symbols)]
    result["assets"] = assets
    result["coverage"] = [
        r for r in result["coverage"] if r["market"] == "US" and r["symbol"] in symbols
    ]
    result.update(
        scope="development_snapshot",
        cohort_complete=False,
        training_ready=False,
        candidate_count=len(assets),
        samples=sum(a["samples"] for a in assets),
    )
    for field in ("counts", "purged_by_boundary"):
        result[field] = {phase: sum(a[field][phase] for a in assets) for phase in source["counts"]}
    result["temporal_view"]["protocol"]["selection"]["max_epochs"] = epochs
    return result


def jobs():
    """Catálogo finito. La primera semilla completa permite estimar coste antes de ampliar."""
    result = []
    config = configuration()
    for seed in config["seeds"]:
        for projection in config["projections"]:
            stem = f"{projection}-{seed}"
            base = dict(projection=projection, seed=seed)
            core = f"{stem}/core"
            result.append(dict(base, id=core, kind="core", fold="fold-017"))
            result.append(
                dict(base, id=f"{stem}/reference", kind="reference", parent=core, fold="fold-018")
            )
            result.append(
                dict(base, id=f"{stem}/m0-reader", kind="m0_reader", parent=core, fold="fold-017")
            )
            result.append(
                dict(
                    base,
                    id=f"{stem}/m0-reference",
                    kind="m0_reference",
                    parent=f"{stem}/m0-reader",
                    fold="fold-018",
                )
            )
            for depth in config["reader_depths"]:
                reader = f"{stem}/reader-k{depth}"
                result.append(
                    dict(base, id=reader, kind="reader", parent=core, depth=depth, fold="fold-017")
                )
                result.append(
                    dict(
                        base,
                        id=f"{stem}/collect-k{depth}",
                        kind="collect",
                        parent=reader,
                        depth=depth,
                        fold="fold-018",
                    )
                )
                selected = f"{stem}/select-k{depth}"
                result.append(
                    dict(
                        base,
                        id=selected,
                        kind="select",
                        parent=f"{stem}/collect-k{depth}",
                        reader=reader,
                        depth=depth,
                        fold="fold-018",
                    )
                )
                result.append(
                    dict(
                        base,
                        id=f"{stem}/evaluate-k{depth}",
                        kind="evaluate",
                        parent=selected,
                        reader=reader,
                        depth=depth,
                        fold="fold-018",
                    )
                )
    return result


def code_identity():
    """Huellas del ejecutor y de los módulos que fijan su comportamiento numérico y temporal."""
    paths = [
        Path(__file__),
        *(
            ROOT / "benchmarks" / name
            for name in (
                "refinement_study_traces.py",
                "refinement_study_policy.py",
                "refinement_study_controls.py",
            )
        ),
    ]
    paths.extend(
        (ROOT / "uv.lock", ROOT / "pyproject.toml", ROOT / "src/mars_titan/models/quantile_head.py")
    )
    for package in (
        "models/titans",
        "training",
        "memory",
        "posttraining",
        "calibration",
        "evaluation",
        "data",
    ):
        paths.extend(sorted((ROOT / "src/mars_titan" / package).glob("*.py")))
    return {str(path.relative_to(ROOT)): sha256(path) for path in paths}


def environment_identity():
    return dict(
        python=platform.python_version(),
        packages={name: version(name) for name in ("torch", "numpy", "pyarrow")},
    )


def _recipes(work, config):
    paths = []
    for name in config["projections"]:
        core = read_manifest(ROOT / "configs/titans/chronological-training-historical-masked.json")[
            0
        ]
        core["recipe"].update(epochs=config["epochs"], accumulation_rows=None)
        core["predictor"]["attention_layout"] = config["attention_layout"]
        if name == "mac_paper":
            core["predictor"]["memory_projections"] = "titans_mac_paper_projections_v2"
        core["pending"] = ["Receta propia del estudio de refinamientos. No sustituye la campaña."]
        path = work / "recipes" / f"{name}.json"
        atomic_json(path, core)
        paths.append(path)
    reader = read_manifest(ROOT / "configs/titans/episodic-readout-historical-masked.json")[0]
    reader["recipe"].update(
        epochs=config["epochs"],
        bank_capacity=config["bank_capacity"],
        refinement_initialization="zero_refinement",
    )
    reader["pending"] = ["Lector M1 elegido antes del ajuste de la política de rutas."]
    path = work / "recipes/reader.json"
    atomic_json(path, reader)
    return paths + [path]


def prepare(sources, work):
    """Preparar dos vistas de la misma cohorte y acreditar la disjunción de las filas de puerta."""
    from mars_titan.posttraining.chronological_windows import staged_window
    from mars_titan.posttraining.staged_rows import fit_rows_proof
    from mars_titan.training.corpus_inputs import CorpusDataset
    from mars_titan.training.label_maturity import FIT_PARTITIONS, label_maturity

    sources, work = Path(sources), Path(work)
    safe_destination(work)
    outside_source(sources, work)
    if work.exists():
        raise FileExistsError("El estudio necesita un directorio nuevo")
    config = configuration()
    originals = {name: read_manifest(sources / name / "manifest.json")[0] for name in FOLDS}
    selected = choose_assets(originals["fold-017"], config["cohort_size"])
    views = {
        name: project_view(value, selected, name, config["epochs"])
        for name, value in originals.items()
    }
    parent_assets = _assets(views["fold-017"])
    if any(
        a["samples_sha256"] != parent_assets[a["symbol"]]["samples_sha256"]
        for a in views["fold-018"]["assets"]
    ):
        raise ValueError("Las vistas no comparten los mismos archivos de muestras")
    for view in views.values():
        for root in view["roots"].values():
            outside_source(Path(root), work)
    work.mkdir(parents=True)
    artifacts, datasets = [], {}
    for name, view in views.items():
        path = work / name / "view.json"
        protocol = path.with_name("protocol.json")
        atomic_json(path, view)
        atomic_json(protocol, view["temporal_view"]["protocol"])
        artifacts.extend((path, protocol))
        datasets[name] = CorpusDataset(
            path, input_policy=HISTORICAL_MASKED, cache_bytes=64 * 1024**2
        )
    parent_view, view = (work / name / "view.json" for name in FOLDS)
    fold, phases, placement = staged_window(parent_view, view, 12)
    latest, _ = label_maturity(parent_view, FIT_PARTITIONS)
    proof = fit_rows_proof(
        datasets["fold-017"],
        datasets["fold-018"],
        parent_fold=views["fold-017"]["temporal_view"]["fold"],
        fold=fold,
        parent_labels_until=latest,
    )
    path = work / "staged.json"
    atomic_json(path, dict(fold=fold, phases=phases, placement=placement, fit_rows_proof=proof))
    artifacts.extend([path, *_recipes(work, config)])
    receipt = dict(
        schema_version=1,
        kind=config["kind"],
        configuration=config,
        configuration_sha256=sha256(CONFIG),
        jobs=jobs(),
        code=code_identity(),
        environment=environment_identity(),
        artifacts={str(p.relative_to(work)): sha256(p) for p in artifacts},
        source_manifests={
            name: dict(
                path=str((sources / name / "manifest.json").resolve()),
                sha256=sha256(sources / name / "manifest.json"),
            )
            for name in FOLDS
        },
        assets=list(selected),
        counts={name: value["counts"] for name, value in views.items()},
        cohort_selection=dict(
            rule="sha256_among_parent_train_positive",
            information_before="2021-04-01",
            targets_used=False,
            future_coverage_used=False,
        ),
        final_test_opened=False,
        prepared_at_utc=datetime.now(UTC).isoformat(),
    )
    atomic_json(work / "prepared.json", receipt)
    return receipt


def verify_prepared(work):
    work = Path(work).resolve()
    receipt = read_manifest(work / "prepared.json")[0]
    if (
        receipt["configuration_sha256"] != sha256(CONFIG)
        or receipt["configuration"] != configuration()
        or receipt["jobs"] != jobs()
        or receipt["code"] != code_identity()
        or receipt["environment"] != environment_identity()
    ):
        raise ValueError("La configuración, código o catálogo cambió después de preparar")
    for name, digest in receipt["artifacts"].items():
        path = work / name
        safe_destination(path)
        if not path.resolve().is_relative_to(work) or sha256(path) != digest:
            raise ValueError("Un artefacto preparado cambió o sale del estudio")
    for record in receipt["source_manifests"].values():
        if sha256(Path(record["path"])) != record["sha256"]:
            raise ValueError("Un manifiesto de origen cambió después de preparar")
    return receipt


def run_job(work, job_id):
    require_learning_allowed("el estudio de rutas de refinamiento")
    from .refinement_study_traces import execute_job

    return execute_job(Path(work).resolve(), job_id)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "verify", "run"))
    parser.add_argument("--work", required=True, type=Path)
    parser.add_argument("--sources", type=Path)
    parser.add_argument("--job")
    args = parser.parse_args()
    if args.command == "prepare":
        if args.sources is None or args.job is not None:
            parser.error("prepare necesita --sources y no admite --job")
        result = prepare(args.sources, args.work)
    elif args.command == "verify":
        if args.sources is not None or args.job is not None:
            parser.error("verify solo necesita --work")
        result = verify_prepared(args.work)
    else:
        if args.job is None or args.sources is not None:
            parser.error("run necesita --job y no admite --sources")
        result = run_job(args.work, args.job)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
