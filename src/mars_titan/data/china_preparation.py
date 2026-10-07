"""Derivar un activo preparado CN a partir de una edición contable revisada."""

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .china_fundamentals import _SCHEMA
from .china_sources import CONCEPTS, _day, _provenance
from .cohort_files import read_manifest, safe_destination
from .cohort_samples import _prepared
from .macro_coverage import _publish_directory
from .preparation import atomic_parquet
from .storage import atomic_json, outside_source, sha256
from .temporal import MarketClock

_MAX_BYTES = 64 * 1024**2
_MAX_FACTS = 100_000
_MAX_EDITIONS = 16
_POLICY = "reviewed_chinese_facts_into_empty_preparation_v1"
_HISTORY_POLICY = "reviewed_chinese_fact_history_into_empty_preparation_v1"
_CODE = (
    "china_preparation.py",
    "china_fundamentals.py",
    "china_sources.py",
    "cohort_samples.py",
    "cohort_preparation.py",
    "cohort_files.py",
    "cohort_news.py",
    "macro_coverage.py",
    "preparation.py",
    "storage.py",
    "temporal.py",
)


def _hash(path):
    safe_destination(path)
    if not path.is_file() or path.stat().st_size > _MAX_BYTES:
        raise ValueError("El archivo no es regular o supera el presupuesto")
    return sha256(path)


def _table(path, *, count=None, facts=False):
    with pq.ParquetFile(path) as file:
        limit = _MAX_FACTS if facts else 200_000
        meta = file.metadata
        if (
            meta.num_rows > limit
            or count is not None
            and meta.num_rows != count
            or meta.num_row_groups > max(1, meta.num_rows)
            or sum(meta.row_group(i).total_byte_size for i in range(meta.num_row_groups))
            > _MAX_BYTES
        ):
            raise ValueError("El Parquet no conserva su población o supera el presupuesto")
        if facts:
            if file.schema_arrow != _SCHEMA:
                raise ValueError("El esquema contable no conserva la procedencia revisada")
            table = file.read(use_threads=False)
            if table.nbytes > _MAX_BYTES:
                raise ValueError("Los hechos decodificados superan el presupuesto")
            return table.to_pylist()


def _facts(edition, origin, clock, cutoff):
    path = edition / "report.json"
    report, signature = read_manifest(path, maximum=1024**2)
    identity = report.get("identity", {})
    count = report.get("facts")
    if (
        report.get("schema_version") != 1
        or report.get("kind") != "reconciled_chinese_fundamentals"
        or report.get("training_ready") is not False
        or report.get("final_test_opened") is not False
        or identity.get("market") != "CN"
        or identity.get("symbol") != origin["symbol"]
        or identity.get("cutoff") != cutoff
        or identity.get("numeric_storage") != "float64_with_exact_decimal_provenance"
        or type(count) is not int
        or not 1 <= count <= _MAX_FACTS
    ):
        raise ValueError("La edición contable no corresponde al activo y corte preparados")
    for name in (
        "source_sha256",
        "review_sha256",
        "document_sha256",
        "publication_sha256",
        "calendar_sha256",
    ):
        if not isinstance(identity.get(name), str) or not re.fullmatch(
            r"[0-9a-f]{64}", identity[name]
        ):
            raise ValueError("La edición contable necesita las huellas de sus fuentes")
    artifact = edition / "fundamentals.parquet"
    if _hash(artifact) != report.get("sha256"):
        raise ValueError("Ha cambiado la partición contable revisada")
    rows = _table(artifact, count=count, facts=True)
    _validate_rows(rows, identity, origin, clock, _day(cutoff))
    records = [index for row in rows for index in row["source_records"]]
    expected = dict(
        matched_records=len(records),
        unique_source_records=len(set(records)),
        duplicate_matches=len(records) - len(rows),
    )
    if any(type(report.get(k)) is not int or report[k] != v for k, v in expected.items()):
        raise ValueError("Los recuentos contables no concilian con los hechos revisados")
    return report, {path: signature, artifact: report["sha256"]}, rows


def _fact_editions(facts_edition, additional_facts):
    if not isinstance(additional_facts, (tuple, list)) or len(additional_facts) >= _MAX_EDITIONS:
        raise ValueError("Las ediciones contables superan el presupuesto o no forman una lista")
    editions = tuple(Path(path) for path in (facts_edition, *additional_facts))
    if len({path.resolve() for path in editions}) != len(editions):
        raise ValueError("Las ediciones contables contienen orígenes duplicados")
    for path in editions:
        safe_destination(path)
    return editions


def _fact_history(editions, origin, clock, cutoff):
    if len(editions) > 1:
        total_rows, total_bytes = 0, 0
        for edition in editions:
            path = edition / "fundamentals.parquet"
            _hash(path)
            with pq.ParquetFile(path) as file:
                total_rows += file.metadata.num_rows
                total_bytes += sum(
                    file.metadata.row_group(i).total_byte_size for i in range(file.num_row_groups)
                )
            if total_rows > _MAX_FACTS or total_bytes > _MAX_BYTES:
                raise ValueError("La historia contable supera el presupuesto de filas o memoria")
    parents, sources, rows, keys = [], {}, [], set()
    for edition in editions:
        report, guarded, facts = _facts(edition, origin, clock, cutoff)
        sources.update(guarded)
        parents.append(
            dict(
                path=str((edition / "report.json").resolve()),
                sha256=guarded[edition / "report.json"],
                identity=report["identity"],
                artifact_sha256=report["sha256"],
            )
        )
        if len(rows) + len(facts) > _MAX_FACTS:
            raise ValueError("La historia contable supera el presupuesto de filas")
        for row in facts:
            key = tuple(
                row[name]
                for name in ("concept", "period_start", "period_end", "filed", "accession")
            )
            if key in keys:
                raise ValueError("Las publicaciones contienen hechos duplicados o contradictorios")
            keys.add(key)
        rows.extend(facts)
    rows.sort(
        key=lambda row: (
            row["available_at"],
            row["period_end"],
            row["period_start"] or "",
            row["filed"],
            row["accession"],
            row["concept"],
        )
    )
    matched = sum(len(row["source_records"]) for row in rows)
    audit = dict(
        facts=len(rows),
        matched_records=matched,
        unique_source_records=len(
            {(row["source_file"], index) for row in rows for index in row["source_records"]}
        ),
        duplicate_matches=matched - len(rows),
    )
    return parents, sources, rows, audit


def _validate_rows(rows, identity, origin, clock, cutoff):
    concepts = {f"cn-reported:{concept}:CNY": kind for concept, kind in CONCEPTS.values()}
    keys, matches = set(), set()
    for row in rows:
        concept = row["concept"]
        if (
            concept not in concepts
            or row["unit"] != "CNY"
            or row["accounting_standard"] != "CAS"
            or row["statement_scope"] != "consolidated"
            or row["published_at"] is not None
            or row["availability_rule"] != "reviewed_cninfo_date_next_session_close"
            or row["document_sha256"] != identity["document_sha256"]
            or row["publication_evidence_sha256"] != identity["publication_sha256"]
            or origin["sources"].get(row["source_file"]) != identity["source_sha256"]
        ):
            raise ValueError("El hecho no conserva la unidad, perímetro y procedencia revisados")
        published, period = _day(row["filed"]), _day(row["period_end"])
        start = row["period_start"]
        if (
            period > published
            or published > cutoff
            or row["available_at"] != clock.date_available(row["filed"])
            or row["available_at"].date() > cutoff
            or concepts[concept] == "stock"
            and start is not None
            or concepts[concept] == "flow"
            and (start is None or _day(start) > period)
        ):
            raise ValueError("El hecho no conserva el periodo y su siguiente cierre CN")
        _provenance(
            {
                **row,
                "publication_date": row["filed"],
                "report_id": row["accession"],
            }
        )
        exact = row["value_exact"]
        try:
            valid = (
                isinstance(exact, str)
                and len(exact) <= 80
                and Decimal(exact).is_finite()
                and row["value"] is not None
                and math.isfinite(row["value"])
                and float(Decimal(exact)) == row["value"]
            )
        except (InvalidOperation, ValueError, OverflowError):
            valid = False
        if not valid:
            raise ValueError("El valor numérico no conserva su decimal revisado")
        key = (concept, start, row["period_end"], row["filed"], row["accession"])
        records = row["source_records"]
        if (
            key in keys
            or not isinstance(records, list)
            or not records
            or any(type(i) is not int or not 1 <= i <= _MAX_FACTS for i in records)
            or len(set(records)) != len(records)
            or any((i, concept) in matches for i in records)
            or len(matches) + len(records) > _MAX_FACTS
        ):
            raise ValueError("Los hechos duplican coincidencias o contienen ordinales no válidos")
        keys.add(key)
        matches.update((i, concept) for i in records)


def _verify(sources, code):
    if any(_hash(path) != digest for path, digest in sources.items()):
        raise ValueError("Una fuente o artefacto cambió durante la preparación")
    if any(sha256(Path(__file__).with_name(name)) != digest for name, digest in code.items()):
        raise ValueError("El código cambió durante la preparación")


def _derived_report(origin, parents, calendar, code, reviewed, facts_hash, cutoff):
    rule = _HISTORY_POLICY if parents.get("additional_facts") else _POLICY
    policy = dict(
        derivation=rule,
        cutoff=cutoff,
        calendar=calendar,
        code=code,
        pyarrow=pa.__version__,
        max_file_bytes=_MAX_BYTES,
        max_facts=_MAX_FACTS,
    )
    if parents.get("additional_facts"):
        policy["max_fact_editions"] = _MAX_EDITIONS
    artifacts = {**origin["artifacts"], "fundamentals.parquet": facts_hash}
    report = dict(
        schema_version=3,
        kind="derived_prepared_asset",
        market="CN",
        symbol=origin["symbol"],
        cohort_id="original_audited",
        news_content_policy=origin["news_content_policy"],
        policy=policy,
        parents=parents,
        sources=origin["sources"],
        artifacts=artifacts,
        counts={**origin["counts"], "fundamentals": reviewed["facts"]},
        fundamentals_audit=dict(
            policy=rule,
            accepted=reviewed["facts"],
            matched_records=reviewed["matched_records"],
            unique_source_records=reviewed["unique_source_records"],
            duplicate_matches=reviewed["duplicate_matches"],
        ),
        **{
            name: origin[name]
            for name in (
                "reserved_counts",
                "price_audit",
                "news_counts",
                "news_reasons",
                "chart_policy",
            )
        },
        training_ready=False,
        final_test_opened=False,
    )
    report["fingerprint"] = hashlib.sha256(
        json.dumps(report, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    return report


def derive_chinese_preparation(
    prepared_manifest, facts_edition, output, *, clock, cutoff="2023-12-31", additional_facts=()
):
    """Añadir publicaciones revisadas sin alterar sus fechas ni ediciones anteriores.

    El origen debe tener contabilidad vacía. Cada documento conserva CAS, CNY y
    su perímetro consolidado. La salida es un activo, no un corpus admitido.
    """
    prepared_manifest, facts_edition, output = map(Path, (prepared_manifest, facts_edition, output))
    editions = _fact_editions(facts_edition, additional_facts)
    if clock.market != "CN" or _day(cutoff) >= date(2024, 1, 1):
        raise ValueError("La preparación necesita calendario CN y reserva final cerrada")
    for path in (prepared_manifest, output):
        safe_destination(path)
    for source in (prepared_manifest.parent, *editions, Path("dataset")):
        outside_source(source, output)
        outside_source(output, source)
    code = {name: sha256(Path(__file__).with_name(name)) for name in _CODE}
    initial, parent_hash = read_manifest(prepared_manifest, maximum=4 * 1024**2)
    sources = {prepared_manifest: parent_hash}
    for name, digest in initial.get("artifacts", {}).items():
        if name not in {
            "prices.parquet",
            "fundamentals.parquet",
            "news/news.parquet",
            "news/excluded.parquet",
            "news/manifest.json",
        }:
            raise ValueError("La preparación declara un artefacto desconocido")
        path = prepared_manifest.parent / name
        if _hash(path) != digest:
            raise ValueError("Ha cambiado un artefacto preparado")
        sources[path] = digest
    origin, calendar, verified_hash = _prepared(prepared_manifest.parent, clock, "original_audited")
    counts = origin.get("counts", {})
    if (
        verified_hash != parent_hash
        or prepared_manifest.name != "manifest.json"
        or origin["policy"]["cutoff"] != cutoff
        or not isinstance(origin.get("symbol"), str)
        or not re.fullmatch(r"\d{6}\.(?:SZ|SS|SH)", origin["symbol"])
        or set(counts) != {"prices", "news", "fundamentals"}
        or any(type(n) is not int or n < 0 for n in counts.values())
        or counts["fundamentals"] != 0
        or not isinstance(origin.get("sources"), dict)
        or origin.get("chart_policy") != "regenerate_from_past_prices"
    ):
        raise ValueError("El origen no acredita un activo CN con contabilidad vacía")
    for name, count in (
        ("prices.parquet", counts["prices"]),
        ("fundamentals.parquet", 0),
        ("news/news.parquet", counts["news"]),
        ("news/excluded.parquet", None),
    ):
        _table(prepared_manifest.parent / name, count=count)
    fact_parents, reviewed_sources, rows, reviewed = _fact_history(editions, origin, clock, cutoff)
    sources.update(reviewed_sources)
    parents = dict(
        prepared=dict(
            path=str(prepared_manifest.resolve()),
            sha256=parent_hash,
            fingerprint=origin["fingerprint"],
        ),
        facts=fact_parents[0],
    )
    if len(editions) > 1:
        parents["additional_facts"] = fact_parents[1:]
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}.", dir=output.parent) as temporary:
        stage = Path(temporary) / "asset"
        (stage / "news").mkdir(parents=True)
        facts_hash = fact_parents[0]["artifact_sha256"]
        if len(editions) > 1:
            table = pa.Table.from_pylist(rows, schema=_SCHEMA)
            if table.nbytes > _MAX_BYTES:
                raise ValueError("La historia contable supera el presupuesto de memoria")
            atomic_parquet(stage / "fundamentals.parquet", table)
            facts_hash = _hash(stage / "fundamentals.parquet")
        report = _derived_report(origin, parents, calendar, code, reviewed, facts_hash, cutoff)
        artifacts = report["artifacts"]
        if output.exists():
            previous, signature = read_manifest(output / "manifest.json", maximum=4 * 1024**2)
            if previous != report:
                raise ValueError("La edición existente no conserva la identidad derivada")
            _verify(
                {
                    **sources,
                    output / "manifest.json": signature,
                    **{output / name: digest for name, digest in artifacts.items()},
                },
                code,
            )
            return {**report, "reused": True}
        for name in artifacts:
            if name == "fundamentals.parquet" and len(editions) > 1:
                continue
            source = facts_edition if name == "fundamentals.parquet" else prepared_manifest.parent
            shutil.copyfile(source / name, stage / name)
            with (stage / name).open("rb") as stream:
                os.fsync(stream.fileno())
        atomic_json(stage / "manifest.json", report)
        staged, signature = read_manifest(stage / "manifest.json", maximum=4 * 1024**2)
        if staged != report:
            raise ValueError("El manifiesto temporal no conserva la identidad derivada")
        _verify(
            {
                **sources,
                stage / "manifest.json": signature,
                **{stage / name: digest for name, digest in artifacts.items()},
            },
            code,
        )
        for directory in (stage / "news", stage):
            descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        _publish_directory(stage, output)
        descriptor = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    return {**report, "reused": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prepared-manifest", "facts-edition", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--calendar-start", required=True)
    parser.add_argument("--calendar-end", required=True)
    parser.add_argument("--cutoff", default="2023-12-31")
    parser.add_argument("--additional-facts", type=Path, action="append", default=[])
    args = parser.parse_args(argv)
    report = derive_chinese_preparation(
        args.prepared_manifest,
        args.facts_edition,
        args.output,
        clock=MarketClock("CN", args.calendar_start, args.calendar_end),
        cutoff=args.cutoff,
        additional_facts=args.additional_facts,
    )
    print(
        f"Activo {report['symbol']}: {report['counts']['fundamentals']} hechos revisados. "
        "Admisión multimodal pendiente."
    )


if __name__ == "__main__":
    main()
