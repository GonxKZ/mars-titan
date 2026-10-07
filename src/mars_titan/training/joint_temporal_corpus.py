"""Unión de ventanas US y CN con etiquetas y calendarios propios."""

import argparse
import copy
import json
import os
import tempfile
from pathlib import Path

from mars_titan.data.cohort_files import read_manifest, safe_destination
from mars_titan.data.macro_coverage import _publish_directory
from mars_titan.data.storage import atomic_json, outside_source, sha256
from mars_titan.evaluation.splits import PARTITIONS, build_folds

from .corpus_inputs import CorpusDataset
from .reference_campaign import campaign_views
from .temporal_contract import temporal_contracts
from .temporal_corpus import prepare_temporal_corpus


def prepare_joint_temporal_corpus(parent, sources, output, *, recover_annual_boundaries=False):
    """Preparar cada calendario y publicar su unión sin intersectar decisiones."""
    parent, output = Path(parent), Path(output)
    safe_destination(output)
    if output.exists():
        raise FileExistsError("Las ventanas conjuntas ya existen")
    if not isinstance(sources, dict) or set(sources) != {"US", "CN"}:
        raise ValueError("Se requieren protocolo, macro y admisión de US y CN")
    protocols, paths, signatures = {}, {}, {}
    for market, record in sources.items():
        if not isinstance(record, dict) or set(record) != {"protocol", "macro", "admission"}:
            raise ValueError("Cada mercado necesita sus tres fuentes temporales")
        paths[market] = {name: Path(path) for name, path in record.items()}
        protocol, signature = read_manifest(paths[market]["protocol"], 64 * 1024)
        signatures[paths[market]["protocol"]] = signature
        build_folds(protocol)
        if protocol["market"] != market or protocol["final_test_start"] != "2024-01-01":
            raise ValueError("El protocolo no conserva su mercado y la reserva de 2024")
        protocols[market] = protocol
    if {k: v for k, v in protocols["US"].items() if k != "market"} != {
        k: v for k, v in protocols["CN"].items() if k != "market"
    }:
        raise ValueError("Los protocolos conjuntos deben compartir cortes, semillas y margen")
    dataset = CorpusDataset(parent, cache_bytes=0)
    signatures[parent] = dataset.identity
    if dataset.temporal is not None or {a["market"] for a in dataset.assets} != {"US", "CN"}:
        raise ValueError("Se necesita el padre anual común de los dos mercados")
    if dataset.manifest.get("final_test_opened") is not False:
        raise ValueError("El padre debe conservar cerrado el test final")
    for path in (parent, *(p for record in paths.values() for p in record.values())):
        safe_destination(path)
        if not path.is_file():
            raise ValueError("Falta una fuente temporal regular")
        signature = sha256(path)
        if path in signatures and signature != signatures[path]:
            raise ValueError("Una fuente temporal cambió durante su lectura")
        signatures[path] = signature
    for protected in (*dataset.roots.values(), *signatures, Path("dataset")):
        outside_source(protected, output)
        outside_source(output, protected)
    code = {
        name: sha256(Path(__file__).with_name(name))
        for name in (
            "joint_temporal_corpus.py",
            "temporal_corpus.py",
            "temporal_contract.py",
            "reference_campaign.py",
        )
    }
    projections = campaign_views(parent, ["US", "CN"])
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}.", dir=output.parent) as temporary:
        stage = Path(temporary) / "views"
        stage.mkdir()
        local_reports = {}
        for market in ("US", "CN"):
            projection = stage / "parents" / f"{market}.json"
            atomic_json(projection, projections[market])
            local_reports[market] = prepare_temporal_corpus(
                projection,
                paths[market]["protocol"],
                paths[market]["macro"],
                paths[market]["admission"],
                stage / "markets" / market,
                recover_annual_boundaries=recover_annual_boundaries,
            )
        summaries = []
        for index, fold in enumerate(build_folds(protocols["US"])):
            views = {}
            label_root = stage / fold["id"] / "labels"
            label_root.mkdir(parents=True)
            for market in ("US", "CN"):
                folder = stage / "markets" / market / fold["id"]
                view, signature = read_manifest(folder / "manifest.json", 8 * 1024**2)
                record = local_reports[market]["folds"][index]
                if record["id"] != fold["id"] or signature != record["manifest_sha256"]:
                    raise ValueError("La ventana local cambió durante la unión")
                # Solo se reubican etiquetas nuevas de este intento, nunca fuentes del padre.
                (folder / "labels" / market).rename(label_root / market)
                view["roots"]["labels"] = str((output / fold["id"] / "labels").resolve())
                view["temporal_view"]["parent_manifest"] = str(
                    (output / "parents" / f"{market}.json").resolve()
                )
                atomic_json(folder / "manifest.json", view)
                record["manifest_sha256"] = sha256(folder / "manifest.json")
                views[market] = view
            joined = copy.deepcopy(dataset.manifest)
            joined.update(
                roots=dict(
                    dataset.manifest["roots"],
                    labels=str((output / fold["id"] / "labels").resolve()),
                ),
                assets=[asset for view in views.values() for asset in view["assets"]],
                counts={
                    name: sum(view["counts"][name] for view in views.values())
                    for name in PARTITIONS
                },
                temporal_views={market: view["temporal_view"] for market, view in views.items()},
                final_test_opened=False,
            )
            if recover_annual_boundaries:
                joined["label_admission"] = views["US"]["label_admission"]
                joined["recovered_annual_labels"] = sum(
                    view["recovered_annual_labels"] for view in views.values()
                )
            temporal_contracts(joined)
            destination = stage / fold["id"] / "manifest.json"
            atomic_json(destination, joined)
            if destination.stat().st_size > 8 * 1024**2:
                raise ValueError("El manifiesto conjunto supera el presupuesto de 8 MiB")
            summaries.append(
                dict(
                    id=fold["id"],
                    counts=joined["counts"],
                    market_counts={market: view["counts"] for market, view in views.items()},
                    manifest_sha256=sha256(destination),
                    has_all_partitions=all(all(view["counts"].values()) for view in views.values()),
                )
            )
        evidence = {}
        for market, report in local_reports.items():
            relative = f"markets/{market}/report.json"
            atomic_json(stage / relative, report)
            evidence[market] = dict(path=relative, sha256=sha256(stage / relative))
        for asset in dataset.assets:
            for kind in ("prices", "samples", "labels"):
                dataset._file(asset, kind)
        if any(sha256(path) != signature for path, signature in signatures.items()) or any(
            sha256(Path(__file__).with_name(name)) != signature for name, signature in code.items()
        ):
            raise ValueError("Las fuentes o el código cambiaron durante la unión temporal")
        report = dict(
            schema_version=2,
            kind="joint_temporal_views",
            status="temporal_views_prepared",
            parent_sha256=dataset.identity,
            sources=evidence,
            folds=summaries,
            scope=dataset.manifest["scope"],
            cohort_complete=dataset.manifest["cohort_complete"],
            code_sha256=code,
            scientific_training_started=False,
            final_test_opened=False,
        )
        atomic_json(stage / "report.json", report)
        _publish_directory(stage, output)
        descriptor = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "parent",
        "output",
        *(
            f"{market}-{field}"
            for market in ("us", "cn")
            for field in ("protocol", "macro", "admission")
        ),
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--recover-annual-boundaries", action="store_true")
    args = parser.parse_args(argv)
    sources = {
        market.upper(): {
            field: getattr(args, f"{market}_{field}")
            for field in ("protocol", "macro", "admission")
        }
        for market in ("us", "cn")
    }
    result = prepare_joint_temporal_corpus(
        args.parent, sources, args.output, recover_annual_boundaries=args.recover_annual_boundaries
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
