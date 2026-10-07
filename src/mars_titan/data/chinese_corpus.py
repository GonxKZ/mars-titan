"""Unir activos chinos revisados conservando sus representaciones y procedencia."""

import fcntl
import json
import os
import re
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from mars_titan.training.cohort_contract import (
    cohort_identity,
    representation_identity,
    validate_cohort_rows,
)
from mars_titan.training.corpus_inputs import (
    VECTORS,
    CorpusDataset,
    _availability,
    _price_contexts,
    _times,
)
from mars_titan.training.corpus_targets import _asset_sources, prepare_corpus_targets

from .batches import read_bounded_table
from .chinese_samples import CNY_CONCEPTS
from .cohort_files import read_manifest, safe_destination
from .cohort_samples import _digest, _schema
from .macro_coverage import _read_catalog
from .storage import atomic_json, outside_source, sha256

_MAX_EDITIONS = 1024
_MAX_SAMPLES = 1_000_000
_MAX_BYTES = 64 * 1024**2
_MAX_ROWS = 200_000
_UNSPECIFIED = object()
_ARTIFACTS = {
    "prices.parquet",
    "fundamentals.parquet",
    "news/news.parquet",
    "news/excluded.parquet",
    "news/manifest.json",
}
_STAGES = {"preparation.json", "encoded/manifest.json", "supervised/manifest.json"}
_POLICY = "copied_reviewed_chinese_editions_v1"
_CODE = (
    "data/chinese_corpus.py",
    "data/chinese_samples.py",
    "data/cohort_files.py",
    "data/cohort_samples.py",
    "data/macro_coverage.py",
    "data/batches.py",
    "data/storage.py",
    "data/temporal.py",
    "data/budget_targets.py",
    "data/residual_arrays.py",
    "training/cohort_contract.py",
    "training/corpus_inputs.py",
    "training/corpus_targets.py",
)


def _same(actual, expected):
    """Comparar JSON sin identificar booleanos con enteros ni enteros con decimales."""
    return json.dumps(actual, sort_keys=True, allow_nan=False) == json.dumps(
        expected, sort_keys=True, allow_nan=False
    )


def _has(value, **expected):
    return _same({key: value.get(key) for key in expected}, expected)


def _read(path, sources, expected=_UNSPECIFIED):
    safe_destination(path)
    if expected is not _UNSPECIFIED and (
        not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected)
    ):
        raise ValueError("Falta una huella obligatoria válida")
    value, digest = read_manifest(path, maximum=8 * 1024**2)
    if not isinstance(value, dict) or expected is not _UNSPECIFIED and digest != expected:
        raise ValueError("Un recibo no conserva la huella o estructura declaradas")
    if path in sources and sources[path] != digest:
        raise ValueError("Un recibo cambió durante su lectura")
    sources[path] = digest
    return value


def _file(path, sources, expected):
    safe_destination(path)
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError("Falta una huella SHA-256 válida")
    if not path.is_file() or path.stat().st_size > _MAX_BYTES:
        raise ValueError("Un archivo no es regular o supera el presupuesto")
    if path not in sources:
        sources[path] = sha256(path)
    if sources[path] != expected:
        raise ValueError("Ha cambiado un artefacto de origen")


def _bounded(path, *, count=None, column=None, schema=None):
    with path.open("rb") as stream:
        stream.seek(-8, 2)
        footer = stream.read(8)
    if footer[4:] != b"PAR1" or int.from_bytes(footer[:4], "little") > 8 * 1024**2:
        raise ValueError("El footer Parquet no es válido o supera el presupuesto")
    with pq.ParquetFile(path) as file:
        meta = file.metadata
        if (
            meta.num_rows > _MAX_ROWS
            or count is not None
            and meta.num_rows != count
            or meta.num_row_groups > max(1, meta.num_rows)
            or sum(meta.row_group(i).total_byte_size for i in range(meta.num_row_groups))
            > _MAX_BYTES
            or schema is not None
            and file.schema_arrow != schema
        ):
            raise ValueError("El Parquet no conserva su esquema, población o presupuesto")
        if column is not None:
            values = file.read(columns=[column], use_threads=False)[column]
            if (
                not pa.types.is_timestamp(values.type)
                or values.type.tz != "UTC"
                or values.null_count
            ):
                raise ValueError("Las fechas necesitan instantes UTC completos")
            latest = pc.max(values).as_py()
            if latest is not None and latest.year >= 2024:
                raise ValueError("Un artefacto abre la reserva final")
            return values.to_pylist()


def _factor(factors, sources):
    if not isinstance(factors, dict) or set(factors) != {"CN"}:
        raise ValueError("La unión requiere un factor propio de CN")
    specification = factors["CN"]
    if specification.get("market") != "CN" or not specification.get("symbol"):
        raise ValueError("El factor no identifica su mercado CN")
    path = Path(specification["prices_path"])
    _file(path, sources, specification["prices_sha256"])
    _bounded(path, column="available_at")
    return factors


def _catalog(configuration, sources, macro_hash, indicators):
    declared = configuration.get("sources")
    if not isinstance(declared, dict) or not 1 <= len(declared) <= 128:
        raise ValueError("Las fuentes de la edición no están acotadas")
    admissions = []
    for name, digest in declared.items():
        path = Path(name)
        if not path.is_absolute():
            raise ValueError("Las fuentes necesitan rutas explícitas")
        _file(path, sources, digest)
        if path.suffix == ".json":
            value = _read(path, sources, digest)
            if value.get("policy") == "all_catalog_indicators_valid":
                admissions.append(value)
    if len(admissions) != 1:
        raise ValueError("La edición no identifica una única admisión macro")
    admission = admissions[0]
    digest = admission.get("catalog_sha256")
    catalogs = [
        Path(name)
        for name, value in declared.items()
        if value == digest and Path(name).suffix == ".csv"
    ]
    if (
        admission.get("market") != "CN"
        or admission.get("source_sha256") != macro_hash
        or not _same(admission.get("required_indicator_count"), 140)
        or admission.get("required_indicator_ids") != indicators
        or len(catalogs) != 1
        or sorted(_read_catalog(catalogs[0])) != indicators
    ):
        raise ValueError("El catálogo y el panel macro no conservan la misma identidad")
    return digest


def _validate_samples(path, dataset):
    """Comprobar todas las muestras, incluidas las que el factor anterior excluyó."""
    prices, price_available = dataset._prices(dataset.assets[0])
    with pq.ParquetFile(path) as file:
        for batch in file.iter_batches(batch_size=256, use_threads=False):
            table = pa.Table.from_batches([batch])
            validate_cohort_rows(table, "original_audited")
            prediction = _times(table["prediction_at"])
            ends = table["price_end_index"].to_numpy()
            _price_contexts(prices, ends, 64)
            available, valid = _availability(table)
            if (
                available is None
                or not valid.all()
                or (available > prediction).any()
                or (price_available[ends] > prediction).any()
            ):
                raise ValueError("La disponibilidad de una modalidad es ausente o futura")
            for name in VECTORS:
                column = table[name].combine_chunks()
                if column.null_count or column.values.null_count:
                    raise ValueError("Una modalidad contiene valores ausentes")
                values = column.values.to_numpy()
                if not np.isfinite(values).all():
                    raise ValueError("Una modalidad contiene valores no finitos")
                if name == "macro" and not (values.reshape(len(table), 420)[:, 140:280] == 1).all():
                    raise ValueError("Cada muestra requiere los 140 indicadores macro presentes")


def _source(edition, sources):
    report = _read(edition / "report.json", sources)
    if (
        not _has(
            report,
            schema_version=1,
            status="completed",
            scope="development_snapshot",
            cohort_complete=False,
            training_ready=False,
            final_test_opened=False,
        )
        or set(report.get("artifacts", {})) != _STAGES
    ):
        raise ValueError("La edición china no está completa con su reserva cerrada")
    symbol = report.get("symbol")
    if not isinstance(symbol, str) or not re.fullmatch(r"\d{6}\.(?:SZ|SS|SH)", symbol):
        raise ValueError("El símbolo de la edición no es válido")
    config = _read(edition / "configuration.json", sources, report["configuration_sha256"])
    stages = {
        name: _read(edition / name, sources, digest) for name, digest in report["artifacts"].items()
    }
    selection, encoded, supervised = (
        stages[name]
        for name in ("preparation.json", "encoded/manifest.json", "supervised/manifest.json")
    )
    asset = dict(market="CN", symbol=symbol, cohort_id="original_audited")
    parent = selection.get("parent_preparation", {})
    if (
        not _has(
            selection,
            schema_version=1,
            kind="prepared_cohort",
            status="completed",
            scope="reviewed_asset_subset",
            candidate_count=1,
            failed_assets=0,
            cohort_id="original_audited",
        )
        or len(selection.get("assets", [])) != 1
        or selection["assets"][0].get("symbol") != symbol
        or selection["assets"][0].get("market") != "CN"
        or selection["assets"][0].get("state") != "prepared"
        or Path(selection["prepared_root"]).resolve() != edition / "prepared"
    ):
        raise ValueError("La selección preparada no corresponde al activo")
    parent_path = Path(parent["path"])
    original = _read(parent_path, sources, parent.get("sha256"))
    selected = [
        item
        for item in original.get("assets", [])
        if (item.get("market"), item.get("symbol")) == ("CN", symbol)
    ]
    if len(selected) != 1 or selected[0].get("state") != "prepared":
        raise ValueError("El activo no pertenece al censo padre preparado")
    population = original.get("candidate_count")
    if (
        type(population) is not int
        or population != len(original.get("assets", []))
        or not _same(population, parent.get("candidate_count"))
        or not _same(population, report.get("parent_candidate_count"))
        or not _same(population, config.get("parent_candidate_count"))
    ):
        raise ValueError("El censo padre no concilia")
    for meta, kind in ((encoded, "materialized_corpus"), (supervised, "corpus_supervision")):
        if (
            cohort_identity(meta) != "original_audited"
            or not _has(
                meta, schema_version=2, candidate_count=1, failed_assets=0, context_sessions=64
            )
            or meta.get("kind") != kind
            or meta.get("scope") != "development_snapshot"
            or meta.get("cohort_complete") is not False
            or meta.get("final_test_opened") is not False
            or meta.get("context_sessions") != 64
            or meta.get("markets") != ["CN"]
            or len(meta.get("assets", [])) != 1
            or meta.get("candidate_count") != 1
            or meta.get("failed_assets") != 0
            or any(meta["assets"][0].get(k) != v for k, v in asset.items())
            or not _same(meta.get("parent_preparation"), parent)
        ):
            raise ValueError("La población o representación de la edición no concilia")
    prepared, samples = edition / "prepared", edition / "encoded/samples"
    roots = dict(
        prepared=str(prepared), samples=str(samples), labels=str(edition / "supervised/labels")
    )
    if (
        supervised.get("roots") != roots
        or Path(encoded["samples_root"]).resolve() != samples
        or encoded["configuration"].get("preparation_sha256")
        != sources[edition / "preparation.json"]
        or supervised["configuration"].get("source_manifest_sha256")
        != sources[edition / "encoded/manifest.json"]
        or config.get("policy") != "reviewed_cn_asset_complete_macro_v1"
        or config.get("symbol") != symbol
        or not _same(config.get("context_sessions"), 64)
    ):
        raise ValueError("Los enlaces de etapas o el contexto no corresponden a la edición")
    for name, meta in (("encoded", encoded), ("supervised", supervised)):
        if not _same(_read(edition / name / "configuration.json", sources), meta["configuration"]):
            raise ValueError("La configuración de una etapa no conserva su recibo confirmado")
    prepared_path, sample_path = prepared / "CN" / symbol, samples / "CN" / symbol
    origin = _read(
        prepared_path / "manifest.json", sources, selection["assets"][0]["manifest_sha256"]
    )
    receipt = _read(sample_path / "manifest.json", sources)
    sample_config = _read(sample_path / "configuration.json", sources)
    representation = representation_identity(receipt)
    if (
        not _same(representation, representation_identity(supervised["representation"]))
        or representation["fundamental_concepts"] != list(CNY_CONCEPTS)
        or len(representation["macro_indicators"]) != 140
        or not _same(representation["context_sessions"], 64)
        or receipt.get("source_unit") != "CNY"
        or receipt.get("company_factors_sha256") is not None
        or not _same(representation["encoders"], config.get("encoders"))
        or not _same(representation["encoders"], encoded["configuration"].get("encoders"))
        or config.get("fundamental_concepts") != list(CNY_CONCEPTS)
        or config.get("source_unit") != "CNY"
        or receipt.get("fingerprint") != _digest(sample_config)
        or sample_config.get("prepared_manifest_sha256") != sources[prepared_path / "manifest.json"]
        or sample_config.get("calendar") != origin.get("policy", {}).get("calendar")
    ):
        raise ValueError("La representación no conserva el codificador y los conceptos CNY")
    macro_hash = receipt.get("macro_sha256")
    if (
        encoded["configuration"].get("macro_sha256") != {"CN": macro_hash}
        or sample_config.get("macro_sha256") != macro_hash
    ):
        raise ValueError("La representación utiliza otro panel macro")
    catalog_hash = _catalog(config, sources, macro_hash, representation["macro_indicators"])
    count = receipt.get("samples")
    if (
        type(count) is not int
        or not 1 <= count <= _MAX_ROWS
        or any(not _same(meta.get("samples"), count) for meta in (report, encoded, supervised))
    ):
        raise ValueError("Las muestras de la edición no concilian")
    if (
        set(origin.get("artifacts", {})) != _ARTIFACTS
        or not _same(origin.get("schema_version"), 3)
        or origin.get("training_ready") is not False
        or set(origin.get("counts", {})) != {"prices", "news", "fundamentals"}
        or any(type(n) is not int or n < 0 for n in origin["counts"].values())
    ):
        raise ValueError("La preparación no conserva sus artefactos")
    copies = {}
    for name, digest in origin["artifacts"].items():
        path = prepared_path / name
        _file(path, sources, digest)
        if path.suffix == ".parquet":
            expected = {
                "prices.parquet": "prices",
                "fundamentals.parquet": "fundamentals",
                "news/news.parquet": "news",
            }.get(name)
            _bounded(path, count=origin["counts"][expected] if expected else None)
        copies[Path("prepared/CN") / symbol / name] = path
    copies[Path("prepared/CN") / symbol / "manifest.json"] = prepared_path / "manifest.json"
    _file(sample_path / "samples.parquet", sources, receipt["samples_sha256"])
    times = _bounded(
        sample_path / "samples.parquet",
        count=count,
        column="prediction_at",
        schema=_schema(CNY_CONCEPTS, representation["macro_indicators"]),
    )
    for name in ("samples.parquet", "manifest.json", "configuration.json"):
        copies[Path("encoded/samples/CN") / symbol / name] = sample_path / name
    _, _, _, checked_representation = _asset_sources(
        asset, prepared, samples, 64, "original_audited"
    )
    if (
        not _same(checked_representation, representation)
        or encoded["coverage"][0].get("fingerprint") != receipt["fingerprint"]
    ):
        raise ValueError("La cobertura no corresponde a la representación preparada")
    if not _same(supervised["market_factors"], encoded["market_factors"]):
        raise ValueError("Las etapas declaran factores distintos")
    factors = _factor(supervised["market_factors"], sources)
    label_path = edition / "supervised/labels/CN" / symbol
    label_receipt = _read(label_path / "receipt.json", sources)
    if not _same(label_receipt, supervised["assets"][0]) or not _same(
        report["counts"], supervised["counts"]
    ):
        raise ValueError("La supervisión no concilia con sus recibos")
    _file(label_path / "labels.parquet", sources, label_receipt["labels_sha256"])
    _bounded(label_path / "labels.parquet", count=count, column="prediction_at")
    labels = read_bounded_table(
        label_path / "labels.parquet", max_rows=_MAX_ROWS, max_bytes=_MAX_BYTES
    )
    if (
        labels["sample_row"].to_pylist() != list(range(count))
        or labels["prediction_at"].to_pylist() != times
    ):
        raise ValueError("Las etiquetas no corresponden a las posiciones y fechas de las muestras")
    excluded = Counter(
        reason
        for part, reason in zip(
            labels["partition"].to_pylist(), labels["reason"].to_pylist(), strict=True
        )
        if part is None
    )
    if not _same(dict(excluded), label_receipt.get("excluded_reasons")):
        raise ValueError("Las exclusiones del recibo no concilian")
    dataset = CorpusDataset(edition / "supervised/manifest.json", cache_bytes=0)
    dataset._labels(dataset.assets[0], "train", count)
    _validate_samples(sample_path / "samples.parquet", dataset)
    return dict(
        symbol=symbol,
        samples=count,
        representation=representation,
        factors=factors,
        copies=copies,
        parent={key: parent[key] for key in ("path", "sha256", "candidate_count")},
        macro_sha256=macro_hash,
        catalog_sha256=catalog_hash,
        calendar_start=encoded["calendar_start"],
        calendar=sample_config["calendar"],
        coverage=encoded["coverage"][0],
        asset=asset,
        lineage=dict(
            edition=str(edition),
            symbol=symbol,
            samples=count,
            report_sha256=sources[edition / "report.json"],
            configuration_sha256=sources[edition / "configuration.json"],
            encoded_sha256=sources[edition / "encoded/manifest.json"],
            supervised_sha256=sources[edition / "supervised/manifest.json"],
        ),
    )


def _verify(sources, code):
    for path, digest in {**sources, **code}.items():
        safe_destination(path)
        if not path.is_file() or sha256(path) != digest:
            raise ValueError("Una fuente, configuración o código cambió durante la unión")


def _write(path, value):
    safe_destination(path)
    if not path.exists():
        atomic_json(path, value)
    return _confirmed(path, value)


def _confirmed(path, value):
    expected = {key: item for key, item in value.items() if key != "reused_assets"}
    consumed = {}
    actual = _read(path, consumed)
    if not _same(actual, expected):
        raise ValueError("El recibo no conserva el contenido y los tipos confirmados")
    return consumed[path]


def _copy(source, destination, digest, output):
    safe_destination(destination)
    if destination.exists():
        if sha256(destination) != digest:
            raise ValueError("Ha cambiado una copia ya confirmada")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name("." + destination.name + ".partial")
    safe_destination(temporary)
    try:
        shutil.copyfile(source, temporary)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        if sha256(temporary) != digest:
            raise ValueError("El artefacto cambió durante la copia")
        os.replace(temporary, destination)
        directory = destination.parent
        while directory.is_relative_to(output):
            fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
            directory = directory.parent
    finally:
        if temporary.exists():
            temporary.unlink()


def combine_chinese_samples(editions, output, *, market_factors=None):
    """Copiar activos compatibles y recalcular su supervisión con un factor CN común."""
    if not isinstance(editions, (list, tuple)) or not 1 <= len(editions) <= _MAX_EDITIONS:
        raise ValueError("El número de ediciones supera el presupuesto")
    for path in (*editions, output):
        safe_destination(Path(path))
    editions, output = [Path(path).resolve() for path in editions], Path(output).resolve()
    if len(set(editions)) != len(editions):
        raise ValueError("Hay ediciones duplicadas")
    for path in (*editions, Path("dataset").resolve()):
        outside_source(path, output)
        outside_source(output, path)
    package = Path(__file__).parents[1]
    code = {package / name: sha256(package / name) for name in _CODE}
    sources = {}
    items = sorted(
        (_source(edition, sources) for edition in editions), key=lambda item: item["symbol"]
    )
    first = items[0]
    if len({item["symbol"] for item in items}) != len(items):
        raise ValueError("La unión contiene símbolos duplicados")
    for item in items[1:]:
        if not _same(item["representation"], first["representation"]):
            raise ValueError("Las ediciones tienen distinta semántica de representación")
        if any(
            not _same(item[key], first[key]) for key in ("parent", "calendar", "calendar_start")
        ):
            raise ValueError("Las ediciones no comparten padre y calendario")
        if any(item[key] != first[key] for key in ("macro_sha256", "catalog_sha256")):
            raise ValueError("Las ediciones no comparten panel y catálogo macro")
    count = sum(item["samples"] for item in items)
    if count > _MAX_SAMPLES:
        raise ValueError("La unión supera el presupuesto de muestras")
    factors = first["factors"]
    if market_factors is None:
        if any(not _same(item["factors"], factors) for item in items):
            raise ValueError("Los factores difieren y requieren un descriptor común explícito")
    else:
        factors = _factor(_read(Path(market_factors).resolve(), sources), sources)
    for path in sources:
        outside_source(output, path)
        outside_source(path, output)
    lineage = [item["lineage"] for item in items]
    identity = dict(
        policy=_POLICY,
        sources={str(path): digest for path, digest in sources.items()},
        code={str(path.relative_to(package)): digest for path, digest in code.items()},
        lineage=lineage,
        representation=first["representation"],
        market_factors=factors,
        limits=dict(
            editions=_MAX_EDITIONS, samples=_MAX_SAMPLES, file_bytes=_MAX_BYTES, file_rows=_MAX_ROWS
        ),
    )
    _verify(sources, code)
    output.mkdir(parents=True, exist_ok=True)
    lock_path = output / ".edition.lock"
    safe_destination(lock_path)
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        configuration = output / "configuration.json"
        if not configuration.exists() and any(
            path.name != ".edition.lock" for path in output.iterdir()
        ):
            raise ValueError("La salida contiene una edición sin identidad")
        config_hash = _write(configuration, identity)
        final_path = output / "report.json"
        if final_path.exists():
            previous = _read(final_path, {})
            if previous.get("configuration_sha256") != config_hash:
                raise ValueError("El recibo final pertenece a otra configuración")
            for name, digest in previous.get("artifacts", {}).items():
                if name not in {"encoded/manifest.json", "supervised/manifest.json"}:
                    raise ValueError("El recibo contiene una etapa desconocida")
                _read(output / name, {}, digest)
            CorpusDataset(output / "supervised/manifest.json", cache_bytes=0)
        copied = {}
        for item in items:
            for relative, source in item["copies"].items():
                destination = output / relative
                _copy(source, destination, sources[source], output)
                copied[destination] = sources[source]
        parent = {
            **first["parent"],
            "selection": "reviewed_asset_union",
            "selected_symbols": [item["symbol"] for item in items],
            "editions": lineage,
        }
        encoded = dict(
            schema_version=2,
            kind="materialized_corpus",
            scope="development_snapshot",
            cohort_complete=False,
            cohort_id="original_audited",
            news_content_policy="source_audited_not_external",
            markets=["CN"],
            preparation_scope="reviewed_asset_union",
            parent_preparation=parent,
            context_sessions=64,
            calendar_start=first["calendar_start"],
            samples_root=str(output / "encoded/samples"),
            assets=[item["asset"] for item in items],
            coverage=[item["coverage"] for item in items],
            candidate_count=len(items),
            failed_assets=0,
            samples=count,
            market_factors=factors,
            macro_catalog_sha256=first["catalog_sha256"],
            configuration=dict(
                policy=_POLICY,
                union_configuration_sha256=config_hash,
                encoders=first["representation"]["encoders"],
                macro_sha256={"CN": first["macro_sha256"]},
            ),
            training_ready=False,
            final_test_opened=False,
        )
        cohort_identity(encoded)
        encoded_path = output / "encoded/manifest.json"
        encoded_hash = _write(encoded_path, encoded)
        supervised = prepare_corpus_targets(
            encoded_path, output / "prepared", output / "supervised"
        )
        supervised_path = output / "supervised/manifest.json"
        supervised_hash = _confirmed(supervised_path, supervised)
        report = dict(
            schema_version=1,
            kind="chinese_corpus_union",
            status="completed",
            scope="development_snapshot",
            cohort_complete=False,
            candidate_count=len(items),
            samples=count,
            counts=supervised["counts"],
            lineage=lineage,
            parent_candidate_count=first["parent"]["candidate_count"],
            configuration_sha256=config_hash,
            artifacts={
                "encoded/manifest.json": encoded_hash,
                "supervised/manifest.json": supervised_hash,
            },
            training_ready=False,
            final_test_opened=False,
        )
        _verify(
            {
                **sources,
                **copied,
                configuration: config_hash,
                encoded_path: encoded_hash,
                supervised_path: supervised_hash,
            },
            code,
        )
        _write(final_path, report)
    return report
