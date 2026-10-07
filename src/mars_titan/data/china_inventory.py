"""Inventariar cierres de balances CN para su posterior reconciliación documental."""

import argparse
import hashlib
import json
import os
import re
import resource
import tempfile
import time
from collections import Counter
from datetime import date
from decimal import Decimal
from pathlib import Path

import ijson
import pyarrow as pa
from ijson.common import ObjectBuilder

from .cohort_files import read_manifest, safe_destination
from .corpus_catalog import _source_path
from .macro_coverage import _publish_directory
from .preparation import atomic_parquet
from .storage import atomic_json, outside_source, sha256

PERIODS = ("2021-12-31", "2022-12-31")
_MAX_ASSETS = 1024
_MAX_BYTES = 64 * 1024**2
_MAX_RECORDS = 100_000
_MAX_SELECTED = 4096
_MAX_FIELDS = 512
_PUBLICATION = {"ann_date", "f_ann_date", "filed", "publication_date", "published_at"}
_UNITS = {
    "currency",
    "currency_code",
    "curr_type",
    "unit",
    "unit_multiplier",
    "unit_scale",
    "monetary_unit",
}
_CONTEXT = {
    "report_type",
    "comp_type",
    "end_type",
    "statement_scope",
    "accounting_standard",
    "taxonomy",
    "namespace",
}
_METADATA = (
    _PUBLICATION
    | _UNITS
    | _CONTEXT
    | {"ts_code", "symbol", "end_date", "period_end", "date", "update_flag"}
)
_CODE = (
    "china_inventory.py",
    "cohort_files.py",
    "corpus_catalog.py",
    "macro_coverage.py",
    "preparation.py",
    "storage.py",
)
_FINGERPRINT = pa.struct(
    [
        ("array_record_1_based", pa.int64()),
        ("sha256", pa.string()),
        ("without_update_flag_sha256", pa.string()),
    ]
)
_QUEUE_SCHEMA = pa.schema(
    [
        (name, pa.string())
        for name in (
            "symbol",
            "market",
            "balance_file",
            "balance_sha256",
            "period_end",
            "source_locator",
            "declared_metadata_json",
        )
    ]
    + [
        (name, pa.int64())
        for name in (
            "records",
            "inspected_records",
            "exact_duplicate_records",
            "duplicate_records_without_update_flag",
            "distinct_payloads_without_update_flag",
        )
    ]
    + [
        (name, pa.list_(pa.string()))
        for name in (
            "numeric_fields",
            "nonnumeric_nonempty_fields",
            "differing_numeric_fields",
            "source_symbols",
            "pending_reasons",
        )
    ]
    + [
        ("original_record_ordinals", pa.list_(pa.int64())),
        ("record_fingerprints", pa.list_(_FINGERPRINT)),
    ]
    + [("temporal_admission_granted", pa.bool_()), ("original_context_recovered", pa.bool_())]
)


def _signature(path):
    safe_destination(path)
    info = path.stat()
    if not path.is_file() or info.st_size > _MAX_BYTES:
        raise ValueError("El archivo no es regular o supera el presupuesto de 64 MiB")
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def _read(path, sources):
    before = _signature(path)
    content, digest = read_manifest(path, maximum=8 * 1024**2)
    sources[path] = (digest, before)
    if _signature(path) != before:
        raise ValueError("El manifiesto cambió durante la lectura")
    if not isinstance(content, dict):
        raise ValueError("El manifiesto debe ser un objeto JSON")
    return content, digest


def _verify(sources, code):
    for path, (digest, signature) in sources.items():
        if _signature(path) != signature or sha256(path) != digest or _signature(path) != signature:
            raise ValueError("Una fuente cambió durante el inventario")
    if any(sha256(Path(__file__).with_name(name)) != digest for name, digest in code.items()):
        raise ValueError("El código cambió durante el inventario")


def _symbol(value):
    if not isinstance(value, str) or not re.fullmatch(r"\d{6}\.(?:SZ|SS|SH)", value):
        raise ValueError("El símbolo CN no es válido")
    # Comparar la identidad del emisor no modifica las dos grafías en la salida.
    return value[:-3] + ".SH" if value.endswith(".SS") else value


def _periods(values):
    if not isinstance(values, (tuple, list)) or not 1 <= len(values) <= 128:
        raise ValueError("La selección debe contener entre 1 y 128 fechas ISO")
    for value in values:
        if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
            raise ValueError("Los cierres deben ser fechas ISO")
        if not date(1990, 1, 1) <= date.fromisoformat(value) < date(2024, 1, 1):
            raise ValueError("Los cierres deben estar comprendidos entre 1990 y 2023")
    if list(values) != sorted(set(values)):
        raise ValueError("Los cierres deben ser únicos y estar ordenados")
    return tuple(values)


def _census(meta, periods):
    assets, config = meta.get("assets"), meta.get("configuration")
    if (
        type(meta.get("schema_version")) is not int
        or meta["schema_version"] != 1
        or meta.get("kind") != "prepared_cohort"
        or meta.get("status") not in ("completed", "completed_with_errors")
        or meta.get("cohort_id") != "original_audited"
        or not isinstance(assets, list)
        or not 1 <= len(assets) <= _MAX_ASSETS
        or type(meta.get("candidate_count")) is not int
        or meta["candidate_count"] != len(assets)
        or not isinstance(config, dict)
        or config.get("markets") != ["CN"]
    ):
        raise ValueError("El censo no acredita una preparación CN original_audited terminada")
    cutoff = config.get("cutoff")
    if not isinstance(cutoff, str) or not date.fromisoformat(periods[-1]) <= date.fromisoformat(
        cutoff
    ) < date(2024, 1, 1):
        raise ValueError("El corte no cubre los cierres solicitados o abre la reserva final")
    symbols, failures = set(), 0
    for item in assets:
        if not isinstance(item, dict) or item.get("market") != "CN":
            raise ValueError("El candidato no pertenece al mercado CN")
        symbol = _symbol(item.get("symbol"))
        if symbol in symbols or item.get("state") not in (
            "prepared",
            "failed",
            "missing_modalities",
        ):
            raise ValueError("El censo contiene candidatos duplicados o estados no válidos")
        symbols.add(symbol)
        failures += item["state"] == "failed"
        if item["state"] == "missing_modalities":
            missing = item.get("missing")
            if (
                not isinstance(missing, list)
                or not missing
                or any(not isinstance(v, str) for v in missing)
                or len(set(missing)) != len(missing)
                or not set(missing) <= {"prices", "news", "fundamentals", "charts"}
            ):
                raise ValueError("El candidato incompleto no identifica sus modalidades ausentes")
    if (
        type(meta.get("failed_assets")) is not int
        or meta["failed_assets"] != failures
        or (meta["status"] == "completed") != (failures == 0)
    ):
        raise ValueError("Los estados y recuentos del censo no concilian")
    roots = []
    for raw in (config.get("source_root"), meta.get("prepared_root")):
        if not isinstance(raw, str) or not raw or not Path(raw).is_absolute():
            raise ValueError("El censo debe declarar raíces absolutas")
        path = Path(raw)
        safe_destination(path)
        if not path.is_dir():
            raise ValueError("Falta un directorio declarado por el censo")
        roots.append(path.resolve())
    return assets, roots


def _source_day(value):
    if value is None or isinstance(value, str) and not value.strip():
        return None
    text = str(value).strip()
    if not re.fullmatch(r"(?:[0-9]{8}|[0-9]{4}-[0-9]{2}-[0-9]{2})", text):
        raise ValueError("La fecha del original no tiene formato ISO o compacto")
    return date.fromisoformat(text)


def _metadata(path, periods):
    """Comprobar el JSON completo conservando solo ordinales y fechas de selección."""
    selected, count, period, reserved, keys = {}, 0, None, set(), set()
    period_keys = set(periods)
    with path.open("rb") as stream:
        events = ijson.parse(stream, use_float=False)
        if next(events, None) != ("", "start_array", None):
            raise ValueError("El balance debe contener un array de registros planos")
        for prefix, event, value in events:
            if prefix == "item" and event == "start_map":
                count += 1
                if count > _MAX_RECORDS:
                    raise ValueError("El balance supera el presupuesto de registros")
                period, reserved, keys = None, set(), set()
            elif prefix == "item" and event == "map_key":
                if value in keys:
                    raise ValueError("El original contiene una clave duplicada")
                keys.add(value)
                if len(keys) > _MAX_FIELDS or len(value) > 256:
                    raise ValueError("El registro supera el presupuesto de campos")
            elif prefix == "item" and event == "end_map":
                if period is not None:
                    selected[count] = (period, tuple(sorted(reserved)))
                    if len(selected) > _MAX_SELECTED:
                        raise ValueError(
                            "El balance supera el presupuesto de cierres seleccionados"
                        )
            elif prefix.startswith("item.") and event in {"string", "number", "boolean", "null"}:
                if isinstance(value, str) and len(value) > 4096:
                    raise ValueError("El campo textual supera el presupuesto")
                field = prefix.removeprefix("item.")
                if field == "end_date":
                    day = _source_day(value)
                    period = day.isoformat() if day is not None else None
                    if period not in period_keys:
                        period = None
                elif field in _PUBLICATION and value is not None:
                    try:
                        published = _source_day(value)
                    except ValueError:
                        reserved.add("declared_publication_uninterpretable_not_materialized")
                    else:
                        if published is not None and published >= date(2024, 1, 1):
                            reserved.add("declared_publication_from_2024_not_materialized")
            elif (prefix, event) != ("", "end_array"):
                raise ValueError("El balance debe contener únicamente registros planos")
    return selected, count


def _selected_records(path, selected):
    """Omitir cierres con publicación declarada futura o no interpretable."""
    allowed = {i for i, (_, reserved) in selected.items() if not reserved}
    if not allowed:
        return
    ordinal, builder, last = 0, None, max(allowed)
    with path.open("rb") as stream:
        for prefix, event, value in ijson.parse(stream, use_float=False):
            if prefix == "item" and event == "start_map":
                ordinal += 1
                builder = ObjectBuilder() if ordinal in allowed else None
            if builder is not None:
                builder.event(event, value)
                if prefix == "item" and event == "end_map":
                    yield ordinal, builder.value
                    builder = None
                    if ordinal == last:
                        break


def _fingerprint(record):
    normalized = []
    for key, value in sorted(record.items()):
        if type(value) is int or isinstance(value, Decimal):
            number = Decimal(value)
            if not number.is_finite():
                raise ValueError("El balance contiene un número no finito")
            sign, digits, exponent = number.as_tuple()
            digits = list(digits)
            # Decimal.normalize depende de la precisión del contexto y puede redondear.
            while len(digits) > 1 and digits[-1] == 0:
                digits.pop()
                exponent += 1
            value = [
                "number",
                sign if any(digits) else 0,
                "".join(map(str, digits)),
                exponent if any(digits) else 0,
            ]
        else:
            value = [type(value).__name__, value]
        normalized.append([key, value])
    return hashlib.sha256(
        json.dumps(normalized, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


def _period_summary(period, selected, rows, symbol):
    ordinals = [i for i, (p, _) in selected.items() if p == period]
    values = [(i, row) for i, row in rows if selected[i][0] == period]
    numeric, other, fields, symbols, fingerprints = set(), set(), {}, set(), []
    declared = {key: {} for key in ("units", "publication", "context", "update_flag")}
    for index, row in values:
        fingerprints.append(
            dict(
                array_record_1_based=index,
                sha256=_fingerprint(row),
                without_update_flag_sha256=_fingerprint(
                    {k: v for k, v in row.items() if k != "update_flag"}
                ),
            )
        )
        for name in ("ts_code", "symbol"):
            if row.get(name) is not None:
                if _symbol(row[name]) != _symbol(symbol):
                    raise ValueError("El emisor del registro no coincide con el candidato")
                symbols.add(row[name])
        for field, value in row.items():
            if value is None or isinstance(value, str) and not value.strip():
                continue
            group = (
                "units"
                if field in _UNITS
                else "publication"
                if field in _PUBLICATION
                else "context"
                if field in _CONTEXT
                else "update_flag"
                if field == "update_flag"
                else None
            )
            if group:
                declared[group].setdefault(field, set()).add(str(value))
            if field in _METADATA:
                continue
            if type(value) is int or isinstance(value, Decimal):
                numeric.add(field)
                fields.setdefault(field, set()).add(Decimal(value))
            else:
                other.add(field)
    exact = {f["sha256"] for f in fingerprints}
    without_update = {f["without_update_flag_sha256"] for f in fingerprints}
    reasons = ["primary_publication_and_statement_context_not_verified_by_inventory"]
    if not ordinals:
        reasons.append("no_records_at_requested_close")
    reasons.extend(
        sorted({reason for p, excluded in selected.values() if p == period for reason in excluded})
    )
    if not declared["units"]:
        reasons.append("currency_and_scale_not_declared")
    if len(without_update) > 1:
        reasons.append("different_payloads_at_same_close_require_review")
    return dict(
        period_end=period,
        original_record_ordinals=ordinals,
        records=len(ordinals),
        inspected_records=len(values),
        exact_duplicate_records=len(values) - len(exact),
        duplicate_records_without_update_flag=len(values) - len(without_update),
        distinct_payloads_without_update_flag=len(without_update),
        record_fingerprints=fingerprints,
        numeric_fields=sorted(numeric),
        nonnumeric_nonempty_fields=sorted(other),
        differing_numeric_fields=sorted(k for k, v in fields.items() if len(v) > 1),
        source_symbols=sorted(symbols),
        declared_metadata={
            group: {key: sorted(v) for key, v in items.items()} for group, items in declared.items()
        },
        pending_reasons=reasons,
        temporal_admission_granted=False,
        original_context_recovered=False,
    )


def _asset(item, prepared, source, cutoff, periods, sources):
    path = _source_path(prepared, f"CN/{item['symbol']}/manifest.json")
    content, digest = _read(path, sources)
    if digest != item.get("manifest_sha256"):
        raise ValueError("La huella del manifiesto preparado no coincide con el censo")
    if (
        type(content.get("schema_version")) is not int
        or content["schema_version"] != 3
        or content.get("market") != "CN"
        or content.get("symbol") != item["symbol"]
        or content.get("cohort_id") != "original_audited"
        or not isinstance(content.get("policy"), dict)
        or content["policy"].get("cutoff") != cutoff
        or content.get("chart_policy") != "regenerate_from_past_prices"
        or not isinstance(content.get("sources"), dict)
    ):
        raise ValueError("La identidad preparada no corresponde al candidato")
    choices = [name for name in content["sources"] if Path(name).name == "balance_sheet.jsonl"]
    if len(choices) != 1:
        raise ValueError("El activo no declara una fuente de balance única")
    relative = choices[0]
    balance = _source_path(source, relative)
    before = _signature(balance)
    signature = sha256(balance)
    sources[balance] = (signature, before)
    if signature != content["sources"][relative]:
        raise ValueError("La huella del balance no coincide con la preparación")
    selected, count = _metadata(balance, periods)
    rows = list(_selected_records(balance, selected))
    return dict(
        status="inventoried",
        balance_file=relative,
        balance_sha256=signature,
        source_bytes=before[2],
        prepared_manifest=str(path),
        prepared_manifest_sha256=digest,
        metadata_records=count,
        four_source_basis="prepared_state_requires_prices_news_fundamentals_charts",
        periods=[_period_summary(period, selected, rows, item["symbol"]) for period in periods],
    )


def _sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def inventory_chinese_balances(manifest, output, *, periods=PERIODS):
    """Publicar un censo y una cola de revisión, sin inferir disponibilidad ni moneda.

    Los errores por activo permanecen visibles. Una alteración durante el recorrido
    cancela la publicación. Cada ejecución requiere un directorio nuevo.
    """
    started = time.perf_counter()
    periods = _periods(periods)
    manifest, output = Path(manifest).absolute(), Path(output).absolute()
    safe_destination(output)
    if output.exists():
        raise FileExistsError("La salida existe, usar un destino nuevo")
    code = {name: sha256(Path(__file__).with_name(name)) for name in _CODE}
    sources = {}
    meta, manifest_hash = _read(manifest, sources)
    assets, (source, prepared) = _census(meta, periods)
    for protected in (manifest.parent, source, prepared):
        outside_source(protected, output)
        outside_source(output, protected)
    configuration = dict(
        periods=list(periods),
        period_field="end_date",
        cutoff=meta["configuration"]["cutoff"],
        manifest=str(manifest),
        manifest_sha256=manifest_hash,
        source_root=str(source),
        prepared_root=str(prepared),
        code=code,
        ijson=ijson.__version__,
        pyarrow=pa.__version__,
        limits=dict(
            assets=_MAX_ASSETS,
            source_bytes=_MAX_BYTES,
            records_per_source=_MAX_RECORDS,
            selected_per_source=_MAX_SELECTED,
            selected_total=_MAX_RECORDS,
            fields_per_record=_MAX_FIELDS,
        ),
        duplicate_semantics=(
            "Igualdad Decimal exacta de todos los campos y, por separado, "
            "de todos salvo update_flag"
        ),
    )
    candidates, jobs, total_selected = [], [], 0
    for item in assets:
        row = dict(symbol=item["symbol"], market="CN", preparation_state=item["state"], periods=[])
        if item["state"] != "prepared":
            row.update(
                status="not_prepared", exclusion_reasons=item.get("missing", ["preparation_failed"])
            )
            if item["state"] == "failed":
                row["preparation_error"] = {key: item.get(key) for key in ("error_type", "detail")}
        else:
            try:
                row.update(
                    _asset(item, prepared, source, configuration["cutoff"], periods, sources)
                )
            except (OSError, ValueError, KeyError, ijson.JSONError) as error:
                row.update(
                    status="source_requires_review",
                    exclusion_reasons=[type(error).__name__],
                    detail=str(error),
                )
        summaries = row.pop("periods")
        total_selected += sum(p["records"] for p in summaries)
        if total_selected > _MAX_RECORDS:
            raise ValueError("El inventario supera el presupuesto total de registros seleccionados")
        start = len(jobs)
        for period in summaries:
            job = {
                **period,
                **{key: row[key] for key in ("symbol", "market", "balance_file", "balance_sha256")},
                "source_locator": "array_record_1_based",
            }
            job["declared_metadata_json"] = json.dumps(
                job.pop("declared_metadata"), ensure_ascii=False, sort_keys=True
            )
            jobs.append(job)
        row["queue_rows"] = dict(start=start, stop=len(jobs))
        if row["status"] == "inventoried":
            row["pending_reasons"] = sorted(
                {reason for period in summaries for reason in period["pending_reasons"]}
            )
        candidates.append(row)
    counts = dict(Counter(row["status"] for row in candidates))
    status = (
        "completed_with_errors"
        if counts.get("source_requires_review") or any(a["state"] == "failed" for a in assets)
        else "completed"
    )
    inventory = dict(
        schema_version=1,
        kind="cn_balance_close_inventory",
        status=status,
        configuration=configuration,
        candidates=candidates,
        queue=dict(
            path="reconciliation-queue.parquet", rows=len(jobs), indexing="zero_based_half_open"
        ),
        period_semantics=(
            "Registros con end_date al cierre solicitado, "
            "no certificación de informe anual auditado"
        ),
        temporal_admission_granted=False,
        original_context_recovered=False,
        training_ready=False,
        final_test_opened=False,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}.", dir=output.parent) as temporary:
        stage = Path(temporary) / "edition"
        stage.mkdir()
        table = pa.Table.from_pylist(jobs, schema=_QUEUE_SCHEMA)
        if table.nbytes > _MAX_BYTES:
            raise ValueError("La cola supera el presupuesto de memoria")
        atomic_parquet(stage / "reconciliation-queue.parquet", table)
        atomic_json(stage / "inventory.json", inventory)
        staged, artifacts = {}, {}
        for name in ("inventory.json", "reconciliation-queue.parquet"):
            path = stage / name
            signature = _signature(path)
            digest = sha256(path)
            staged[path] = (digest, signature)
            artifacts[name] = dict(sha256=digest, bytes=signature[2])
        report = dict(
            schema_version=1,
            kind="cn_balance_inventory_report",
            status=status,
            configuration=configuration,
            candidate_count=len(candidates),
            counts=counts,
            period_jobs=len(jobs),
            source_bytes=sum(row.get("source_bytes", 0) for row in candidates),
            periods={
                period: dict(
                    assets_with_records=sum(
                        j["records"] > 0 for j in jobs if j["period_end"] == period
                    ),
                    records=sum(j["records"] for j in jobs if j["period_end"] == period),
                    exact_duplicate_records=sum(
                        j["exact_duplicate_records"] for j in jobs if j["period_end"] == period
                    ),
                    duplicates_without_update_flag=sum(
                        j["duplicate_records_without_update_flag"]
                        for j in jobs
                        if j["period_end"] == period
                    ),
                )
                for period in periods
            },
            artifacts=artifacts,
            elapsed_seconds=time.perf_counter() - started,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            temporal_admission_granted=False,
            training_ready=False,
            final_test_opened=False,
        )
        if any(info["bytes"] > _MAX_BYTES for info in report["artifacts"].values()):
            raise ValueError("Los artefactos superan su presupuesto")
        atomic_json(stage / "report.json", report)
        published, _ = _read(stage / "report.json", staged)
        if json.dumps(published, sort_keys=True) != json.dumps(report, sort_keys=True):
            raise ValueError("El recibo temporal no conserva el contenido y tipos previstos")
        _verify({**sources, **staged}, code)
        _sync_directory(stage)
        _publish_directory(stage, output)
        _sync_directory(output.parent)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--period", action="append", help="Cierre ISO, repetible y en orden creciente"
    )
    args = parser.parse_args(argv)
    report = inventory_chinese_balances(args.manifest, args.output, periods=args.period or PERIODS)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
