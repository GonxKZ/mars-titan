"""Cohortes de la criba elegidas con la historia de ajuste de cada origen.

La preparación solo lee manifiestos. No abre muestras, etiquetas ni modelos y no acredita
la integridad de esos archivos. El lector del corpus los comprueba antes de utilizarlos.
"""

import argparse
import copy
import hashlib
import json
import re
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from mars_titan.data.cohort_files import read_manifest, safe_destination
from mars_titan.data.input_policy import HISTORICAL_MASKED
from mars_titan.data.macro_coverage import _publish_directory
from mars_titan.data.storage import atomic_json, outside_source, sha256
from mars_titan.evaluation.splits import PARTITIONS

from .cohort_contract import cohort_identity
from .temporal_contract import temporal_contracts

ORIGIN_FOLDS = {2019: "fold-014", 2020: "fold-015", 2021: "fold-016", 2022: "fold-017"}
COHORT_SALT = "mars-titan-variant-screening-v1"
KIND = "screening_origin_cohorts_v1"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _source(source, origin):
    _require(type(origin) is int and origin in ORIGIN_FOLDS, "El origen no pertenece a la criba")
    _require(
        isinstance(source, dict)
        and source.get("kind") == "corpus_supervision"
        and source.get("scope") == "full_corpus"
        and source.get("cohort_complete") is True
        and source.get("failed_assets") == 0
        and source.get("markets") == ["US"]
        and "data_subset" not in source,
        "El origen necesita la cohorte histórica completa de US, sin filtros adicionales",
    )
    cohort_identity(source, input_policy=HISTORICAL_MASKED)
    contracts = temporal_contracts(source, input_policy=HISTORICAL_MASKED)
    _require(set(contracts) == {"US"}, "El origen necesita un único calendario US")
    fold = contracts["US"]["fold"]
    _require(
        fold
        == dict(
            id=ORIGIN_FOLDS[origin],
            train=["2000-01-01", f"{origin - 1}-04-01"],
            validation=[f"{origin - 1}-04-01", f"{origin - 1}-10-01"],
            calibration=[f"{origin - 1}-10-01", f"{origin}-01-01"],
            evaluation=[f"{origin}-01-01", f"{origin + 1}-01-01"],
        ),
        "La ventana no corresponde al origen declarado",
    )
    roots = source.get("roots", {})
    _require(
        isinstance(roots, dict)
        and set(roots) == {"prepared", "samples", "labels"}
        and all(isinstance(p, str) and Path(p).is_absolute() for p in roots.values()),
        "Las raíces de los datos deben ser rutas absolutas",
    )
    for asset in source["assets"]:
        symbol = asset.get("symbol")
        _require(
            asset.get("market") == "US"
            and isinstance(symbol, str)
            and symbol not in {".", ".."}
            and re.fullmatch(r"[A-Z0-9.^_=\-]{1,64}", symbol),
            "El activo no tiene una identidad US válida",
        )
        for field in ("counts", "purged_by_boundary"):
            values = asset.get(field, {})
            _require(
                isinstance(values, dict)
                and set(values) == set(PARTITIONS)
                and all(type(n) is int and n >= 0 for n in values.values()),
                "Los recuentos por activo deben ser enteros no negativos",
            )
    for field in ("counts", "purged_by_boundary"):
        totals = source.get(field)
        _require(
            isinstance(totals, dict)
            and all(type(n) is int and n >= 0 for n in totals.values())
            and totals == {p: sum(a[field][p] for a in source["assets"]) for p in PARTITIONS},
            "Los recuentos del origen no concilian con sus activos",
        )
    return fold


def _rank(symbol):
    return hashlib.sha256(f"{COHORT_SALT}/US/{symbol}".encode()).digest(), symbol


def project_origin_cohort(source, *, origin, size=256):
    """Elegir por huella fija entre activos con al menos una etiqueta admitida en ajuste.

    Ni el valor del objetivo ni el tamaño o la cobertura futura ordenan los candidatos.
    Cada activo conserva todas sus referencias, máscaras y filas. Los recuentos pertenecen
    al manifiesto temporal recibido y no reconstruyen un universo histórico point-in-time.
    """
    fold = _source(source, origin)
    _require(
        type(size) is int and 1 <= size <= 256, "El número de activos debe estar entre 1 y 256"
    )
    candidates = [a for a in source["assets"] if a["counts"]["train"] > 0]
    _require(len(candidates) >= size, "No hay suficientes activos con historia de ajuste")
    symbols = sorted(
        a["symbol"] for a in sorted(candidates, key=lambda a: _rank(a["symbol"]))[:size]
    )
    selected = set(symbols)
    view = copy.deepcopy(source)
    view["assets"] = sorted(
        (a for a in view["assets"] if a["symbol"] in selected), key=lambda a: a["symbol"]
    )
    view["coverage"] = sorted(
        (a for a in view["coverage"] if a["symbol"] in selected), key=lambda a: a["symbol"]
    )
    view.update(
        scope="development_snapshot",
        cohort_complete=False,
        candidate_count=size,
        samples=sum(a["samples"] for a in view["assets"]),
    )
    for field in ("counts", "purged_by_boundary"):
        view[field] = {p: sum(a[field][p] for a in view["assets"]) for p in PARTITIONS}
    selection = dict(
        origin=origin,
        fold=fold["id"],
        rule="sha256_among_origin_train_positive",
        salt=COHORT_SALT,
        information_before=fold["train"][1],
        eligible_count=len(candidates),
        size=size,
        assets=symbols,
        assets_sha256=hashlib.sha256(
            json.dumps([["US", symbol] for symbol in symbols], separators=(",", ":")).encode()
        ).hexdigest(),
        targets_used=False,
        future_coverage_used=False,
    )
    return view, selection


def prepare(sources, output, *, size=256):
    """Publicar cuatro vistas y sus huellas juntas, sin sustituir ningún directorio existente."""
    sources, output = Path(sources), Path(output)
    safe_destination(output)
    outside_source(sources, output)
    if output.exists():
        raise FileExistsError("La preparación necesita un directorio nuevo")
    views, records = {}, {}
    for origin, fold in ORIGIN_FOLDS.items():
        path = sources / fold / "manifest.json"
        source, digest = read_manifest(path, 8 * 1024**2)
        view, selection = project_origin_cohort(source, origin=origin, size=size)
        for root in source["roots"].values():
            outside_source(Path(root), output)
        views[origin] = view
        records[str(origin)] = dict(
            source=dict(path=str(path.resolve()), sha256=digest), selection=selection
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as staging:
        staged = Path(staging)
        artifacts = {}
        for origin, view in views.items():
            path = staged / str(origin) / "manifest.json"
            atomic_json(path, view)
            artifacts[str(path.relative_to(staged))] = sha256(path)
        receipt = dict(
            schema_version=1,
            kind=KIND,
            size=size,
            origins=records,
            artifacts=artifacts,
            code_sha256=sha256(Path(__file__)),
            prepared_at_utc=datetime.now(UTC).isoformat(),
            verification_scope="manifests_only",
            row_files_opened=False,
            optimizer_steps=0,
            final_test_opened=False,
        )
        for record in records.values():
            _require(
                sha256(Path(record["source"]["path"])) == record["source"]["sha256"],
                "Cambió la huella del origen durante la preparación",
            )
        atomic_json(staged / "prepared.json", receipt)
        _publish_directory(staged, output)
    return receipt


def verify(work):
    """Comprobar huellas y repetir la selección, sin leer los archivos de muestras."""
    work = Path(work)
    safe_destination(work)
    receipt, _ = read_manifest(work / "prepared.json")
    _require(
        type(receipt.get("schema_version")) is int
        and receipt.get("schema_version") == 1
        and receipt.get("kind") == KIND
        and receipt.get("code_sha256") == sha256(Path(__file__))
        and set(receipt.get("origins", {})) == {str(y) for y in ORIGIN_FOLDS}
        and set(receipt.get("artifacts", {})) == {f"{y}/manifest.json" for y in ORIGIN_FOLDS}
        and receipt.get("verification_scope") == "manifests_only"
        and type(receipt.get("optimizer_steps")) is int
        and receipt.get("optimizer_steps") == 0
        and receipt.get("row_files_opened") is False
        and receipt.get("final_test_opened") is False,
        "El recibo no conserva el contrato de la preparación",
    )
    for origin in ORIGIN_FOLDS:
        record = receipt["origins"][str(origin)]
        source, digest = read_manifest(Path(record["source"]["path"]), 8 * 1024**2)
        _require(digest == record["source"]["sha256"], "Ha cambiado la huella del origen")
        path = work / str(origin) / "manifest.json"
        safe_destination(path)
        actual, digest = read_manifest(path, 8 * 1024**2)
        _require(
            digest == receipt["artifacts"][str(path.relative_to(work))],
            "Ha cambiado la huella de la vista preparada",
        )
        expected, selection = project_origin_cohort(source, origin=origin, size=receipt["size"])
        _require(
            expected == actual and selection == record["selection"],
            "La selección o sus filas no corresponden al origen",
        )
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare", help="Preparar las cuatro cohortes")
    prepare_parser.add_argument("--sources", required=True, type=Path)
    prepare_parser.add_argument("--output", required=True, type=Path)
    prepare_parser.add_argument("--size", type=int, default=256)
    verify_parser = commands.add_parser("verify", help="Comprobar una preparación existente")
    verify_parser.add_argument("--work", required=True, type=Path)
    args = parser.parse_args()
    result = (
        prepare(args.sources, args.output, size=args.size)
        if args.command == "prepare"
        else verify(args.work)
    )
    print(json.dumps({k: result[k] for k in ("kind", "size", "verification_scope")}))


if __name__ == "__main__":
    main()
