"""Materializar una representación US/CN común sin convertir monedas ni recalcular etiquetas."""

import argparse
import copy
import fcntl
import json
import os
import re
import resource
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from mars_titan.training.cohort_contract import representation_hash, representation_identity
from mars_titan.training.corpus_inputs import CorpusDataset, _availability, _price_contexts

from .batches import atomic_parquet_batches
from .chinese_samples import CNY_CONCEPTS
from .cohort_files import read_manifest, safe_destination
from .currency_samples import COMMON_CONCEPTS, _checked, _metadata, _times
from .joint_projection import project_numeric_context
from .storage import atomic_json, outside_source, sha256

CONCEPTS = (*COMMON_CONCEPTS, *CNY_CONCEPTS)
_MARKETS = ("US", "CN")
_LIMIT = 64 * 1024**2
_COMMON = (
    "macro_indicators",
    "encoders",
    "text_aggregation",
    "context_sessions",
    "news_lookback_sessions",
)
_CODE = (
    "data/joint_corpus.py",
    "data/joint_projection.py",
    "data/currency_samples.py",
    "data/chinese_samples.py",
    "data/cohort_files.py",
    "data/batches.py",
    "data/storage.py",
    "training/corpus_inputs.py",
    "training/cohort_contract.py",
    "training/temporal_contract.py",
)


def _same(first, second):
    return json.dumps(first, sort_keys=True, allow_nan=False) == json.dumps(
        second, sort_keys=True, allow_nan=False
    )


def _verify(readers, code_root, code, config, signature):
    for reader in readers.values():
        _checked(reader.path, reader.identity)
        for asset in reader.assets:
            for kind in ("prices", "samples", "labels"):
                reader._file(asset, kind)
    if any(sha256(code_root / name) != digest for name, digest in code.items()):
        raise ValueError("El código cambió durante la materialización")
    _checked(config, signature)


def _preflight(path, market):
    meta, signature = read_manifest(path, 8 * 1024**2)
    assets, roots = meta.get("assets"), meta.get("roots")
    if (
        not isinstance(assets, list)
        or not 1 <= len(assets) <= 10_000
        or not isinstance(roots, dict)
        or set(roots) != {"prepared", "samples", "labels"}
        or any(not isinstance(value, str) or not value for value in roots.values())
        or type(meta.get("samples")) is not int
        or not 1 <= meta["samples"] <= 10_000_000
    ):
        raise ValueError("La fuente supera el presupuesto o no declara sus artefactos")
    total = 0
    for asset in assets:
        if (
            not isinstance(asset, dict)
            or asset.get("market") != market
            or not isinstance(asset.get("symbol"), str)
            or not re.fullmatch(r"[A-Z0-9.^_=\-]{1,64}", asset["symbol"])
            or asset["symbol"] in {".", ".."}
        ):
            raise ValueError("La fuente no identifica un activo dentro del mercado")
        for kind in ("prices", "samples", "labels"):
            root = Path(roots["prepared" if kind == "prices" else kind])
            artifact = root / market / asset["symbol"] / (kind + ".parquet")
            safe_destination(artifact)
            if not artifact.is_file() or artifact.stat().st_size > _LIMIT:
                raise ValueError("El artefacto no es regular o supera 64 MiB")
            total += artifact.stat().st_size
    return signature, total


def _sources(paths):
    if not isinstance(paths, dict) or set(paths) != set(_MARKETS):
        raise ValueError("La unión necesita exactamente una edición US y otra CN")
    sources, representations, total_bytes = {}, {}, 0
    for market in _MARKETS:
        path = Path(paths[market])
        safe_destination(path)
        signature, size = _preflight(path, market)
        total_bytes += size
        if total_bytes > 32 * 1024**3:
            raise ValueError("Las fuentes superan el presupuesto de 32 GiB")
        dataset = CorpusDataset(path, cache_bytes=0)
        if dataset.identity != signature:
            raise ValueError("La fuente cambió durante la admisión")
        meta = dataset.manifest
        expected = list(COMMON_CONCEPTS if market == "US" else CNY_CONCEPTS)
        representation = representation_identity(meta.get("representation", {}))
        if (
            dataset.cohort is None
            or dataset.temporal is not None
            or "temporal_views" in meta
            or meta.get("markets") != [market]
            or meta.get("final_test_opened") is not False
            or any(a["market"] != market for a in dataset.assets)
            or representation["fundamental_concepts"] != expected
            or len(representation["macro_indicators"]) != 140
            or representation["context_sessions"] != dataset.context
        ):
            raise ValueError("La fuente no conserva mercado, conceptos, modalidades y reserva")
        sources[market], representations[market] = dataset, representation
    first, second = (sources[m].manifest for m in _MARKETS)
    if first["cohort_id"] != second["cohort_id"] or any(
        representations["US"][key] != representations["CN"][key] for key in _COMMON
    ):
        raise ValueError("Los mercados no comparten cohorte, codificadores y contexto semántico")
    if (
        sum(len(source.assets) for source in sources.values()) > 10_000
        or sum(source.manifest["samples"] for source in sources.values()) > 10_000_000
    ):
        raise ValueError("La unión supera el presupuesto de activos")
    return sources, representations


def _copy(source, destination, signature):
    safe_destination(destination)
    if destination.exists():
        _checked(destination, signature)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    os.close(descriptor)
    try:
        shutil.copyfile(source, temporary)
        _checked(Path(temporary), signature)
        with open(temporary, "rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        parent = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _project(source, destination, source_concepts, expected_rows, cohort, *, reader, asset):
    last = None
    with pq.ParquetFile(source) as file:
        if file.metadata.num_rows != expected_rows or len(file.schema_arrow.names) != len(
            set(file.schema_arrow.names)
        ):
            raise ValueError("La tabla no conserva sus filas o contiene columnas repetidas")
        widths = dict(news=384, charts=512, macro=420, fundamentals=3 * len(source_concepts))
        if not set(widths) | {"cohort_id", "prediction_at", "price_end_index"} <= set(
            file.schema_arrow.names
        ):
            raise ValueError("Faltan columnas necesarias para conservar las modalidades")
        for name, width in widths.items():
            kind = file.schema_arrow.field(name).type
            if not (
                pa.types.is_fixed_size_list(kind)
                and kind.list_size == width
                and kind.value_type == pa.float32()
            ):
                raise ValueError("Las dimensiones o los tipos de una modalidad no son válidos")
        _metadata(file)
        labels = [
            reader._labels(asset, partition, expected_rows) for partition in reader.partitions
        ]
        prices, price_available = reader._prices(asset)

        def batches():
            nonlocal last
            offset = 0
            for batch in file.iter_batches(batch_size=4096, use_threads=False):
                table = pa.Table.from_batches([batch])
                moments = _times(table["prediction_at"])
                if last is not None and moments[0] <= last:
                    raise ValueError("Las muestras no conservan un orden temporal único")
                last = moments[-1]
                if table.nbytes > _LIMIT or table["cohort_id"].to_pylist() != [cohort] * len(table):
                    raise ValueError("La tabla excede su presupuesto o mezcla cohortes")
                bounds, valid = _availability(table)
                if bounds is None or not valid.all() or np.any(bounds > moments):
                    raise ValueError("Falta disponibilidad de una modalidad o invade el futuro")
                column = table["price_end_index"]
                if not pa.types.is_integer(column.type) or column.null_count:
                    raise ValueError("La muestra necesita un índice de precios entero y completo")
                ends = column.to_numpy()
                for start in range(0, len(ends), 256):
                    _price_contexts(prices, ends[start : start + 256], reader.context)
                if np.any(price_available[ends] > moments):
                    raise ValueError("Una muestra utiliza precios posteriores a su decisión")
                for positions, prediction, _, _ in labels:
                    first, end = np.searchsorted(positions, [offset, offset + len(table)])
                    if not np.array_equal(
                        moments[positions[first:end] - offset], prediction[first:end]
                    ):
                        raise ValueError("La etiqueta no corresponde a la fecha de su muestra")
                offset += len(table)
                for name, width in (("news", 384), ("charts", 512), ("macro", 420)):
                    column = table[name]
                    if column.null_count:
                        raise ValueError(
                            "Falta una modalidad completa con las dimensiones acordadas"
                        )
                    array = column.combine_chunks()
                    values = array.values.slice(array.offset * width, len(array) * width)
                    if values.null_count or not np.isfinite(values.to_numpy()).all():
                        raise ValueError("Una modalidad contiene valores ausentes o no finitos")
                    if name == "macro" and not np.all(
                        values.to_numpy().reshape(-1, width)[:, 140:280] == 1
                    ):
                        raise ValueError("La muestra no conserva los 140 indicadores completos")
                projected = project_numeric_context(
                    table["fundamentals"], source_concepts, CONCEPTS
                )
                field = table.schema.field("fundamentals").with_type(projected.type)
                yield table.set_column(
                    table.schema.get_field_index("fundamentals"), field, projected
                )

        return atomic_parquet_batches(destination, batches())


def _projected_asset(source, asset, representation, sample_hash):
    result = {key: copy.deepcopy(value) for key, value in asset.items() if key != "source_lineage"}
    result.update(
        samples_sha256=sample_hash,
        representation_sha256=representation_hash(representation),
        source_lineage=dict(
            source_manifest_sha256=source.identity,
            source_representation_sha256=asset["representation_sha256"],
            source_artifacts={k: asset[k + "_sha256"] for k in ("prices", "samples", "labels")},
            transformation="disjoint_accounting_channels_v1",
        ),
    )
    return result


def _asset(source, asset, output, representation, config_hash):
    market, symbol = asset["market"], asset["symbol"]
    paths = {kind: source._file(asset, kind) for kind in ("prices", "samples", "labels")}
    destinations = dict(
        prices=output / "prepared" / market / symbol / "prices.parquet",
        samples=output / "encoded/samples" / market / symbol / "samples.parquet",
        labels=output / "supervised/labels" / market / symbol / "labels.parquet",
    )
    marker = destinations["samples"].with_name("projection.json")
    safe_destination(marker)
    identity = dict(
        configuration_sha256=config_hash,
        source_manifest_sha256=source.identity,
        market=market,
        symbol=symbol,
        source_artifacts={kind: asset[kind + "_sha256"] for kind in paths},
    )
    if marker.exists():
        record, _ = read_manifest(marker, 1024**2)
        if not _same(record.get("identity"), identity):
            raise ValueError("La proyección confirmada pertenece a otra fuente o configuración")
        declared = record.get("asset")
        if (
            not isinstance(declared, dict)
            or not isinstance(declared.get("samples_sha256"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", declared["samples_sha256"])
            or not _same(
                declared,
                _projected_asset(source, asset, representation, declared["samples_sha256"]),
            )
        ):
            raise ValueError("El recibo del activo no conserva la población y representación")
        for kind, path in destinations.items():
            _checked(path, record["asset"][kind + "_sha256"])
        return record["asset"]
    for path in destinations.values():
        safe_destination(path)
    for kind in ("prices", "labels"):
        _copy(paths[kind], destinations[kind], asset[kind + "_sha256"])
    _project(
        paths["samples"],
        destinations["samples"],
        source.manifest["representation"]["fundamental_concepts"],
        asset["samples"],
        source.cohort,
        reader=source,
        asset=asset,
    )
    for kind, path in paths.items():
        _checked(path, asset[kind + "_sha256"])
    result = _projected_asset(source, asset, representation, sha256(destinations["samples"]))
    atomic_json(marker, dict(identity=identity, asset=result))
    return result


def prepare_joint_corpus(sources, output):
    """Proyectar una vez, conservar etiquetas y recuperar activos ya confirmados."""
    started = time.perf_counter()
    readers, representations = _sources(sources)
    output = Path(output)
    safe_destination(output)
    for reader in readers.values():
        for protected in (reader.path, *reader.roots.values()):
            outside_source(protected, output)
            outside_source(output, protected)
    outside_source(Path("dataset"), output)
    code_root = Path(__file__).parents[1]
    code = {name: sha256(code_root / name) for name in _CODE}
    representation = copy.deepcopy(representations["US"])
    representation["fundamental_concepts"] = list(CONCEPTS)
    representation["representation_code"] = dict(
        **code, **{f"source_{m}": representation_hash(r) for m, r in representations.items()}
    )
    factor_code = representations["US"]["representation_code"].get("company_factors.py")
    if factor_code is not None:
        representation["representation_code"]["company_factors.py"] = factor_code
    identity = dict(
        schema_version=1,
        policy="disjoint_accounting_channels_v1",
        code=code,
        sources={
            m: dict(
                path=str(r.path.resolve()),
                sha256=r.identity,
                scope=r.manifest["scope"],
                cohort_complete=r.manifest["cohort_complete"],
            )
            for m, r in readers.items()
        },
        representation=representation,
    )
    output.mkdir(parents=True, exist_ok=True)
    lock = output / ".edition.lock"
    safe_destination(lock)
    descriptor = os.open(lock, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = output / "configuration.json"
        safe_destination(config)
        if not config.exists():
            if any(path.name != lock.name for path in output.iterdir()):
                raise ValueError("La salida contiene archivos sin configuración confirmada")
            atomic_json(config, identity)
        configured, signature = read_manifest(config)
        if not _same(configured, identity):
            raise ValueError("La salida contiene otra edición o versión del código")
        complete = all(r.manifest["cohort_complete"] for r in readers.values())
        scope = (
            "full_corpus"
            if complete and all(r.manifest["scope"] == "full_corpus" for r in readers.values())
            else "development_snapshot"
        )
        expected = dict(
            schema_version=1,
            kind="joint_market_corpus",
            status="completed",
            scope=scope,
            cohort_complete=complete,
            assets=sum(len(r.assets) for r in readers.values()),
            samples=sum(r.manifest["samples"] for r in readers.values()),
            counts={
                p: sum(r.manifest["counts"][p] for r in readers.values())
                for p in ("train", "validation")
            },
            configuration_sha256=signature,
            new_inference=False,
            scientific_training_started=False,
            final_test_opened=False,
        )
        previous = output / "report.json"
        safe_destination(previous)
        if previous.exists():
            report, _ = read_manifest(previous)
            if not _same({k: report.get(k) for k in expected}, expected) or set(
                report.get("artifacts", {})
            ) != {"encoded/manifest.json", "supervised/manifest.json"}:
                raise ValueError("El recibo no conserva configuración, población y reserva")
            for relative, digest in report["artifacts"].items():
                if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                    raise ValueError("Falta una huella válida de los manifiestos confirmados")
                _checked(output / relative, digest)
            CorpusDataset(output / "supervised/manifest.json", cache_bytes=0)
            _verify(readers, code_root, code, config, signature)
            return report
        assets = [
            _asset(reader, asset, output, representation, signature)
            for reader in readers.values()
            for asset in reader.assets
        ]
        coverage = [
            copy.deepcopy(row) for reader in readers.values() for row in reader.manifest["coverage"]
        ]
        first = readers["US"].manifest
        counts = {part: sum(a["counts"][part] for a in assets) for part in ("train", "validation")}
        common = dict(
            schema_version=2,
            cohort_id=first["cohort_id"],
            news_content_policy=first["news_content_policy"],
            scope=scope,
            cohort_complete=complete,
            context_sessions=representation["context_sessions"],
            markets=sorted(_MARKETS),
            coverage=coverage,
            candidate_count=len(coverage),
            samples=sum(r.manifest["samples"] for r in readers.values()),
            failed_assets=sum(r.manifest["failed_assets"] for r in readers.values()),
            final_test_opened=False,
            representation=representation,
            market_factors={
                m: copy.deepcopy(r.manifest["market_factors"][m]) for m, r in readers.items()
            },
            joint_projection=dict(configuration_sha256=signature, sources=identity["sources"]),
        )
        encoded = output / "encoded/manifest.json"
        atomic_json(
            encoded,
            dict(
                common,
                kind="projected_corpus_encoding",
                assets=assets,
                samples_root=str((output / "encoded/samples").resolve()),
                configuration=dict(
                    encoders=representation["encoders"], joint_projection_sha256=signature
                ),
            ),
        )
        supervised = output / "supervised/manifest.json"
        atomic_json(
            supervised,
            dict(
                common,
                kind="corpus_supervision",
                assets=assets,
                counts=counts,
                roots={
                    name: str((output / relative).resolve())
                    for name, relative in (
                        ("prepared", "prepared"),
                        ("samples", "encoded/samples"),
                        ("labels", "supervised/labels"),
                    )
                },
                configuration=dict(
                    source_manifest_sha256=sha256(encoded), joint_projection_sha256=signature
                ),
            ),
        )
        CorpusDataset(supervised, cache_bytes=0)
        report = dict(
            **expected,
            source_counts={
                m: dict(
                    samples=r.manifest["samples"],
                    candidate_count=r.manifest["candidate_count"],
                    assets=len(r.assets),
                )
                for m, r in readers.items()
            },
            artifacts={str(p.relative_to(output)): sha256(p) for p in (encoded, supervised)},
            elapsed_seconds=time.perf_counter() - started,
            process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        )
        _verify(readers, code_root, code, config, signature)
        atomic_json(previous, report)
        return report
    finally:
        os.close(descriptor)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("us", "cn", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    report = prepare_joint_corpus({"US": args.us, "CN": args.cn}, args.output)
    print(f"Unión comprobada: {report['samples']} muestras de {report['assets']} activos")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
