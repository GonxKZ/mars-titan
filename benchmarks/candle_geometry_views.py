"""Vistas emparejadas que reutilizan etiquetas y cambian solo la representación de velas."""

import argparse
import copy
import json
from pathlib import Path

from mars_titan.data.candle_geometry_edition import compare_representations
from mars_titan.data.cohort_files import read_manifest, safe_destination
from mars_titan.data.input_policy import HISTORICAL_MASKED
from mars_titan.data.storage import atomic_json, outside_source, sha256
from mars_titan.training.cohort_contract import representation_hash, representation_identity
from mars_titan.training.corpus_inputs import CorpusDataset

FOLDS = ("fold-017", "fold-018")


def project_samples(
    source, *, original, original_digest, geometry, receipts, representation, representation_digest
):
    """Proyectar activos completos sin recalcular etiquetas ni alterar sus metadatos."""
    assets = source.get("assets", [])
    by_id = {(a["market"], a["symbol"]): a for a in assets}
    if (
        source.get("kind") != "corpus_supervision"
        or source.get("input_policy") != HISTORICAL_MASKED
        or source.get("final_test_opened") is not False
        or source.get("configuration", {}).get("source_manifest_sha256") != original_digest
        or Path(source["roots"]["samples"]).resolve() != (original / "samples").resolve()
        or len(by_id) != len(assets)
        or not receipts
        or not set(receipts) <= set(by_id)
        or any(receipts[key]["samples"] != by_id[key]["samples"] for key in receipts)
    ):
        raise ValueError("La supervisión no corresponde al origen y a la población de las velas")
    result = copy.deepcopy(source)
    result["assets"] = [
        dict(
            copy.deepcopy(by_id[key]),
            samples_sha256=receipts[key]["samples_sha256"],
            representation_sha256=representation_digest,
        )
        for key in sorted(receipts)
    ]
    result["roots"]["samples"] = str((geometry / "samples").resolve())
    result["representation"] = copy.deepcopy(representation)
    result["coverage"] = [
        row for row in result["coverage"] if (row["market"], row["symbol"]) in receipts
    ]
    for field in ("counts", "purged_by_boundary"):
        if field in result:
            result[field] = {
                phase: sum(a[field][phase] for a in result["assets"]) for phase in result[field]
            }
    result.update(
        scope="development_snapshot",
        cohort_complete=False,
        training_ready=False,
        candidate_count=len(receipts),
        samples=sum(a["samples"] for a in result["assets"]),
    )
    return result


def _stat(path):
    value = path.stat()
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def _read(path, sources):
    safe_destination(path)
    before = _stat(path)
    value, digest = read_manifest(path, 8 * 1024**2)
    if before != _stat(path):
        raise ValueError("Una fuente cambió mientras se leía")
    sources[path] = digest, before
    return value, digest


def validate_source_samples(source, receipts, representation):
    """Rechazar una vista ajena antes de sustituir sus huellas de representación."""
    assets = {(a["market"], a["symbol"]): a for a in source["assets"]}
    if source["representation"] != representation or any(
        key not in assets or assets[key]["samples_sha256"] != receipt["samples_sha256"]
        for key, receipt in receipts.items()
    ):
        raise ValueError("La vista no acredita las muestras originales de la comparación")


def prepare(source_views, geometry, output):
    """Comprobar las modalidades y publicar ambas vistas sin abrir valores de etiquetas.

    La configuración y los fingerprints de supervisión siguen describiendo la generación
    original de etiquetas. `representation_derivation` acredita por separado las nuevas
    muestras. Los Parquet de precios y etiquetas se referencian sin copiarlos ni escribirlos.
    """
    source_views, geometry, output = (
        Path(path).resolve() for path in (source_views, geometry, output)
    )
    safe_destination(output)
    if output.exists():
        raise FileExistsError("Las vistas derivadas ya existen")
    tracked = {}
    encoded, geometry_digest = _read(geometry / "manifest.json", tracked)
    origin = encoded["chart_derivation"]["source"]
    original = Path(origin["source"]).resolve()
    old, original_digest = _read(original / "manifest.json", tracked)
    if original_digest != origin["manifest_sha256"]:
        raise ValueError("La edición original no conserva su huella")
    views = {}
    parent_path = parent_digest = None
    for fold in FOLDS:
        path = source_views / fold / "manifest.json"
        view, digest = _read(path, tracked)
        temporal = view["temporal_view"]
        expected_parent = Path(temporal["parent_manifest"]).resolve()
        if (
            temporal["fold"]["id"] != fold
            or temporal["protocol"]["final_test_start"] != "2024-01-01"
            or temporal["fold"]["evaluation"][1] > "2024-01-01"
            or (parent_path is not None and expected_parent != parent_path)
            or (parent_digest is not None and temporal["parent_sha256"] != parent_digest)
        ):
            raise ValueError("Las vistas necesitan el mismo padre y ventanas anteriores a 2024")
        parent_path, parent_digest = expected_parent, temporal["parent_sha256"]
        views[fold] = view, digest
    parent, observed_parent = _read(parent_path, tracked)
    if observed_parent != parent_digest or "temporal_view" in parent:
        raise ValueError("El padre anual no conserva su procedencia")
    for root in (source_views, original, geometry, *(Path(v) for v in parent["roots"].values())):
        outside_source(root, output)
        outside_source(output, root)
    comparison = compare_representations(original, geometry)
    receipts = {"baseline": {}, "geometry": {}}
    representations = {}
    for variant, root in (("baseline", original), ("geometry", geometry)):
        for asset in encoded["assets"]:
            key = asset["market"], asset["symbol"]
            receipt, _ = _read(root / "samples" / key[0] / key[1] / "manifest.json", tracked)
            identity = representation_identity(receipt, input_policy=HISTORICAL_MASKED)
            if key in receipts[variant] or representations.get(variant, identity) != identity:
                raise ValueError("Los activos repiten identidad o mezclan representaciones")
            representations[variant] = identity
            receipts[variant][key] = receipt
    for source in (parent, *(value for value, _ in views.values())):
        validate_source_samples(source, receipts["baseline"], representations["baseline"])
    outputs = {}
    for variant, root in (("baseline", original), ("geometry", geometry)):
        identity = representations[variant]
        options = dict(
            original=original,
            original_digest=original_digest,
            geometry=root,
            receipts=receipts[variant],
            representation=identity,
            representation_digest=representation_hash(identity, input_policy=HISTORICAL_MASKED),
        )
        derivation = dict(
            schema_version=1,
            kind="candle_geometry_paired_views",
            variant=variant,
            original_encoding_sha256=original_digest,
            geometry_encoding_sha256=geometry_digest,
            comparison=comparison,
            labels="same_original_parquet_without_recalculation",
            code_sha256=sha256(Path(__file__)),
        )
        projected = project_samples(parent, **options)
        projected["representation_derivation"] = dict(
            derivation, source_manifest=str(parent_path), source_manifest_sha256=parent_digest
        )
        new_parent = output / variant / "parent.json"
        atomic_json(new_parent, projected)
        CorpusDataset(new_parent, cache_bytes=0, input_policy=HISTORICAL_MASKED)
        outputs[variant] = {}
        for fold, (view, digest) in views.items():
            projected = project_samples(view, **options)
            projected["temporal_view"].update(
                parent_manifest=str(new_parent), parent_sha256=sha256(new_parent)
            )
            projected["representation_derivation"] = dict(
                derivation,
                source_manifest=str(source_views / fold / "manifest.json"),
                source_manifest_sha256=digest,
            )
            path = output / variant / fold / "manifest.json"
            atomic_json(path, projected)
            dataset = CorpusDataset(path, cache_bytes=0, input_policy=HISTORICAL_MASKED)
            outputs[variant][fold] = dict(
                path=str(path), sha256=dataset.identity, counts=dataset.manifest["counts"]
            )
    for path, (digest, signature) in tracked.items():
        if _stat(path) != signature or sha256(path) != digest:
            raise ValueError("Una fuente cambió durante la preparación de vistas")
    report = dict(
        schema_version=1,
        status="paired_representation_views_prepared",
        source_manifests={str(p): digest for p, (digest, _) in tracked.items()},
        comparison=comparison,
        views=outputs,
        final_test_opened=False,
        training_ready=False,
        labels_recalculated=False,
    )
    atomic_json(output / "prepared.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-views", type=Path, required=True)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(prepare(args.source_views, args.geometry, args.output), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
