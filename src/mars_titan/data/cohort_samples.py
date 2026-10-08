"""Vectores por ventanas temporales, con límites de memoria y cohorte identificada."""

import fcntl
import hashlib
import json
import math
import resource
import time
from collections import Counter
from datetime import datetime
from decimal import InvalidOperation
from itertools import chain
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .accounting_catalog import historical_accounting_context
from .batches import atomic_parquet_batches, read_bounded_table
from .charts import chart_png
from .china_sources import CONCEPTS as CHINESE_CONCEPTS
from .china_sources import _day, _number
from .cohort_contexts import FactCursor, NewsWindows
from .cohort_files import read_manifest, safe_destination
from .cohort_news import COHORT_POLICIES
from .company_factors import FACTOR_CONCEPTS, write_company_factors
from .input_policy import (
    MODALITIES,
    STRICT_INPUTS,
    masked_inputs,
    numeric_observations,
    policy_identity,
)
from .joint_projection import project_numeric_context
from .samples import FUNDAMENTAL_CONCEPTS, numeric_context
from .storage import atomic_json, outside_source, sha256
from .temporal import admission_errors, aware

_KINDS = ("article_candidate", "summary", "verified_full_article")
_CNY_CONCEPTS = {f"cn-reported:{concept}:CNY": kind for concept, kind in CHINESE_CONCEPTS.values()}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _schema(concepts, indicators, input_policy=STRICT_INPUTS):
    stamp = pa.timestamp("us", tz="UTC")
    schema = pa.schema(
        [(name, pa.string()) for name in ("cohort_id", "session", "chart_hash", "news_set_sha256")]
        + [
            (name, stamp)
            for name in (
                "prediction_at",
                "news_window_start",
                "news_available_at",
                "fundamentals_available_at",
                "macro_available_at",
            )
        ]
        + [("price_end_index", pa.int64()), ("news_count", pa.int64())]
        + [("fundamental_accessions", pa.list_(pa.string()))]
        + [("news_kind_counts", pa.struct([(kind, pa.int64()) for kind in _KINDS]))]
        + [
            (
                "input_availability",
                pa.struct(
                    [
                        (name, stamp)
                        for name in ("prices", "news", "fundamentals", "charts", "macro")
                    ]
                ),
            )
        ]
        + [
            (name, pa.list_(pa.float32(), width))
            for name, width in (
                ("news", 384),
                ("charts", 512),
                ("fundamentals", 3 * len(concepts)),
                ("macro", 3 * len(indicators)),
            )
        ]
    )
    if masked_inputs(input_policy):
        schema = schema.append(pa.field("presence", pa.list_(pa.bool_(), len(MODALITIES))))
        schema = schema.append(
            pa.field("missing_reasons", pa.struct([(name, pa.string()) for name in MODALITIES]))
        )
        schema = schema.append(
            pa.field("fundamental_missing_reasons", pa.list_(pa.string(), len(concepts)))
        )
        schema = schema.append(
            pa.field("macro_missing_reasons", pa.list_(pa.string(), len(indicators)))
        )
    return schema


def _artifact(path, root):
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Un artefacto no es regular o está fuera de su origen")
    return sha256(path)


def _prepared(source, clock, cohort, input_policy=STRICT_INPUTS):
    path = source / "manifest.json"
    manifest, manifest_hash = read_manifest(path)
    calendar = hashlib.sha256("|".join(t.isoformat() for t in clock.decisions).encode()).hexdigest()
    expected = {
        "prices.parquet",
        "fundamentals.parquet",
        "news/news.parquet",
        "news/excluded.parquet",
        "news/manifest.json",
    }
    if (
        cohort not in COHORT_POLICIES
        or manifest.get("schema_version") != (4 if masked_inputs(input_policy) else 3)
        or any(manifest.get(k) != v for k, v in policy_identity(input_policy).items())
        or manifest.get("cohort_id") != cohort
        or manifest.get("news_content_policy") != COHORT_POLICIES[cohort]
        or manifest.get("market") != clock.market
        or manifest.get("policy", {}).get("calendar") != calendar
        or manifest.get("policy", {}).get("cutoff", "9999") > "2023-12-31"
        or manifest.get("training_ready") is not False
        or set(manifest.get("artifacts", {})) != expected
    ):
        raise ValueError("La preparación no corresponde a la cohorte, calendario y reserva cerrada")
    for name, digest in manifest["artifacts"].items():
        if _artifact(source / name, source) != digest:
            raise ValueError(f"Ha cambiado la huella del artefacto preparado: {name}")
    if sha256(path) != manifest_hash:
        raise ValueError("El manifiesto preparado cambió durante su lectura")
    return manifest, calendar, manifest_hash


def _chinese_facts(facts, clock, cutoff):
    """Conservar el contexto CAS y la fecha revisada sin convertir monedas ni conceptos."""
    end = _day(cutoff)
    for row in facts:
        kind = _CNY_CONCEPTS.get(row.get("concept"))
        if (
            kind is None
            or row.get("unit") != "CNY"
            or row.get("accounting_standard") != "CAS"
            or row.get("statement_scope") != "consolidated"
            or row.get("published_at") is not None
            or row.get("availability_rule") != "reviewed_cninfo_date_next_session_close"
        ):
            raise ValueError("El hecho CNY no conserva su concepto y contexto contable revisados")
        value = row.get("value")
        try:
            valid = (
                type(value) in {int, float}
                and math.isfinite(value)
                and isinstance(row.get("value_exact"), str)
                and float(_number(row["value_exact"])) == value
            )
        except (InvalidOperation, ValueError, OverflowError):
            valid = False
        if not valid:
            raise ValueError("El valor CNY debe ser finito y conservar su decimal revisado")
        period, filed = _day(row.get("period_end")), _day(row.get("filed"))
        start, available = row.get("period_start"), row.get("available_at")
        if (
            not period <= filed <= end
            or not isinstance(available, datetime)
            or aware(available) != clock.date_available(row["filed"])
            or available.date() > end
            or kind == "stock"
            and start is not None
            or kind == "flow"
            and (start is None or _day(start) > period)
        ):
            raise ValueError("El hecho CNY no conserva su periodo y siguiente cierre CN")


def _vector(identity, width, encode, cache, hits, misses):
    kind = identity["kind"]
    vector = cache.get(identity)
    if vector is None:
        vector = np.asarray(encode(), dtype=np.float32)
        misses[kind] += 1
        if vector.shape != (width,) or not np.isfinite(vector).all():
            raise ValueError("El codificador produjo dimensiones o valores no válidos")
        cache.put(identity, vector)
    else:
        hits[kind] += 1
    if vector.shape != (width,) or not np.isfinite(vector).all():
        raise ValueError("La caché contiene dimensiones o valores no válidos")
    return vector


def _text_window(rows, cohort, encoder_hash, encoders, cache, hits, misses):
    total, count, latest = np.zeros(384, dtype=np.float64), 0, None
    kinds, digest = Counter(), hashlib.sha256()
    for row in rows:
        if row["cohort_id"] != cohort or row["content_kind"] not in _KINDS:
            raise ValueError("El texto no pertenece a la cohorte declarada")
        if cohort == "externally_verified" and row["content_kind"] != "verified_full_article":
            raise ValueError("La cohorte verificada contiene texto sin verificación completa")
        identity = dict(
            encoder=encoder_hash,
            kind="news",
            content=row["content_hash"],
            policy=row["availability_rule"],
        )
        total += _vector(
            identity, 384, lambda text=row["text"]: encoders.text(text), cache, hits, misses
        )
        count += 1
        kinds[row["content_kind"]] += 1
        latest = row["available_at"]
        digest.update(
            json.dumps(
                [row["event_id"], row["content_hash"], row["content_kind"], latest.isoformat()],
                separators=(",", ":"),
            ).encode()
            + b"\n"
        )
    if not count:
        return None
    return dict(
        news=(total / count).astype(np.float32).tolist(),
        news_count=count,
        news_kind_counts={kind: kinds[kind] for kind in _KINDS},
        news_available_at=latest,
        news_set_sha256=digest.hexdigest(),
    )


def _rows(
    prices,
    facts,
    clock,
    news,
    macros,
    encoders,
    cache,
    *,
    cohort,
    context,
    lookback,
    concepts,
    encoder_hash,
    excluded,
    hits,
    misses,
    admitted_decisions=None,
    input_policy=STRICT_INPUTS,
    missing_sources=(),
    fundamental_exclusions=(),
):
    masked = masked_inputs(input_policy)
    positions = {day.isoformat(): i for i, day in enumerate(clock.days)}
    sessions = prices["session"].tolist()
    try:
        indices = np.array([positions[day] for day in sessions], dtype=np.int64)
    except KeyError as error:
        raise ValueError("Los precios incluyen una sesión ajena al calendario") from error
    if np.any(np.diff(indices) <= 0) or any(day > "2023-12-31" for day in sessions):
        raise ValueError("Los precios no están ordenados o abren la reserva final")
    availability = [aware(t) for t in prices["available_at"]]
    if any(t > clock.decisions[p] for t, p in zip(availability, indices, strict=True)):
        raise ValueError("Los precios contienen información futura")
    # La diferencia entre extremos acredita continuidad porque las sesiones son únicas y ordenadas.
    ohlc = prices[["open", "high", "low", "close"]].to_numpy()
    unknown_publication = {r["concept"] for r in facts if r.get("available_at") is None}
    excluded_publications = {}
    if masked:
        for row in fundamental_exclusions:
            if row["reason"] != "period_after_filing":
                raise ValueError("La exclusión contable tiene un motivo desconocido")
            available = aware(datetime.fromisoformat(row["diagnostic_available_at"]))
            if available != clock.date_available(row["filed"]):
                raise ValueError(
                    "La disponibilidad de la exclusión no corresponde a su publicación"
                )
            concept = row["concept"]
            excluded_publications[concept] = min(
                available, excluded_publications.get(concept, available)
            )
    cursor = FactCursor(
        [r for r in facts if r.get("available_at") is not None] if masked else facts
    )
    for index, position in enumerate(indices):
        cutoff = clock.decisions[position]
        if masked and cutoff.date().isoformat() < "2000-01-01":
            excluded["before_history_start"] += 1
            continue
        if admitted_decisions is not None and cutoff not in admitted_decisions:
            excluded["outside_macro_admission"] += 1
            continue
        if index < context - 1 or position - indices[index - context + 1] != context - 1:
            excluded["incomplete_price_window"] += 1
            continue
        known = cursor.at(cutoff)
        selected = [known.get(concept) for concept in concepts]
        if not masked and not any(r is not None and r["value"] is not None for r in selected):
            excluded["missing_fundamentals"] += 1
            continue
        start = clock.decisions[max(0, position - lookback + 1)]
        articles = news.between(start, cutoff)
        first = next(articles, None)
        if first is None and not masked:
            excluded["missing_news"] += 1
            continue
        macro = macros.at(cutoff)
        if macro is None:
            if masked:
                raise ValueError("El contexto histórico no representa esta sesión macro")
            excluded["missing_macro"] += 1
            continue
        if first is None:
            text = dict(
                news=[0.0] * 384,
                news_count=0,
                news_kind_counts=dict.fromkeys(_KINDS, 0),
                news_available_at=None,
                news_set_sha256=hashlib.sha256(b"").hexdigest(),
            )
        else:
            text = _text_window(
                chain((first,), articles), cohort, encoder_hash, encoders, cache, hits, misses
            )
        if masked:
            values, ages, fact_time, fact_reasons = numeric_observations(
                [r if r is not None else {} for r in selected], cutoff
            )
            fact_reasons = [
                (
                    "source_missing"
                    if "fundamentals" in missing_sources
                    else "period_after_filing"
                    if concept in excluded_publications and excluded_publications[concept] <= cutoff
                    else "unknown_publication"
                    if concept in unknown_publication
                    else reason
                )
                if reason is not None
                else None
                for concept, reason in zip(concepts, fact_reasons, strict=True)
            ]
        else:
            values = [r["value"] if r is not None else None for r in selected]
            ages = [
                (cutoff - r["available_at"]).total_seconds() / 86400 if r is not None else 0
                for r in selected
            ]
            fact_time = max(r["available_at"] for r in selected if r is not None)
        available = dict(
            prices=availability[index],
            news=text["news_available_at"],
            fundamentals=fact_time,
            charts=cutoff,
            macro=macro[1],
        )
        if (
            any(aware(value) > cutoff for value in available.values() if value is not None)
            if masked
            else admission_errors(available, cutoff)
        ):
            raise ValueError("Las modalidades contienen información futura o ausente")
        png = chart_png(ohlc, end_index=index, context=context)
        chart_hash = hashlib.sha256(png).hexdigest()
        image = _vector(
            dict(encoder=encoder_hash, kind="chart", content=chart_hash),
            512,
            lambda png=png: encoders.images([png])[0],
            cache,
            hits,
            misses,
        )
        row = dict(
            **text,
            cohort_id=cohort,
            session=sessions[index],
            prediction_at=cutoff,
            price_end_index=index,
            news_window_start=start,
            fundamentals=numeric_context(values, ages),
            fundamentals_available_at=fact_time,
            fundamental_accessions=sorted({r["accession"] for r in selected if r is not None}),
            input_availability=available,
            macro=macro[0].tolist(),
            macro_available_at=macro[1],
            chart_hash=chart_hash,
            charts=image.tolist(),
        )
        if masked:
            present = [
                True,
                text["news_count"] > 0,
                True,
                any(v is not None for v in values),
                bool(np.any(macro[0][len(macros.indicators) : 2 * len(macros.indicators)])),
            ]
            row.update(
                presence=present,
                fundamental_missing_reasons=fact_reasons,
                macro_missing_reasons=macros.missing_at(cutoff),
                missing_reasons={
                    name: None
                    if observed
                    else "source_missing"
                    if name in missing_sources or (name == "macro" and macros.path is None)
                    else "no_admissible_value"
                    for name, observed in zip(MODALITIES, present, strict=True)
                },
            )
        yield row


def _project_fundamentals(table, source_concepts, target_concepts):
    projected = project_numeric_context(table["fundamentals"], source_concepts, target_concepts)
    table = table.set_column(
        table.schema.get_field_index("fundamentals"), "fundamentals", projected
    )
    positions = {name: index for index, name in enumerate(source_concepts)}
    reasons = [
        [
            row[positions[name]] if name in positions else "outside_source_accounting_catalog"
            for name in target_concepts
        ]
        for row in table["fundamental_missing_reasons"].to_pylist()
    ]
    return table.set_column(
        table.schema.get_field_index("fundamental_missing_reasons"),
        "fundamental_missing_reasons",
        pa.array(reasons, type=pa.list_(pa.string(), len(target_concepts))),
    )


def materialize_cohort_asset(
    source,
    destination,
    clock,
    macros,
    encoders,
    cache,
    *,
    cohort,
    context=64,
    news_lookback_sessions=5,
    company_factors=True,
    batch_rows=128,
    max_partition_bytes=64 * 1024**2,
    max_partition_rows=200_000,
    max_news_group_bytes=16 * 1024**2,
    fundamental_concepts=FUNDAMENTAL_CONCEPTS,
    source_unit="USD",
    admitted_decisions=None,
    input_policy=STRICT_INPUTS,
    target_fundamental_concepts=None,
):
    """Confirmar un activo completo. La caché persiste aunque se interrumpa su escritura."""
    masked = masked_inputs(input_policy)
    if masked and (context != 64 or admitted_decisions is not None):
        raise ValueError(
            "La política histórica requiere 64 sesiones y no filtra por completitud macro"
        )
    for value, low, high in (
        (context, 2, 512),
        (news_lookback_sessions, 1, 512),
        (batch_rows, 1, 1024),
    ):
        if type(value) is not int or not low <= value <= high:
            raise ValueError("El tamaño de contexto, ventana o lote no es válido")
    if type(company_factors) is not bool:
        raise ValueError("La selección de factores debe ser booleana")
    if not isinstance(source_unit, str) or source_unit not in {"USD", "CAD", "CNY"}:
        raise ValueError("La unidad contable debe ser USD, CAD o CNY")
    if admitted_decisions is not None:
        admitted_decisions = frozenset(aware(value) for value in admitted_decisions)
        if any(value.year >= 2024 or value not in clock.decisions for value in admitted_decisions):
            raise ValueError("La admisión debe pertenecer al calendario anterior a 2024")
    source, destination = Path(source), Path(destination)
    for root in (source, Path("dataset")):
        outside_source(root, destination)
    outside_source(destination, source)
    safe_destination(destination)
    macros.verify()
    if getattr(macros, "input_policy", STRICT_INPUTS) != input_policy:
        raise ValueError("El contexto macro no corresponde a la política de entradas")
    origin, calendar, origin_hash = _prepared(source, clock, cohort, input_policy)
    concepts = tuple(fundamental_concepts)
    if company_factors:
        concepts += tuple(name for name in FACTOR_CONCEPTS if name not in concepts)
    if not concepts or len(set(concepts)) != len(concepts):
        raise ValueError("El catálogo contable está vacío o duplicado")
    target = concepts
    if target_fundamental_concepts is not None:
        expected_context = historical_accounting_context(clock.market)
        target = tuple(target_fundamental_concepts)
        if (
            not masked
            or concepts != expected_context["fundamental_concepts"]
            or target != expected_context["target_fundamental_concepts"]
            or source_unit != expected_context["source_unit"]
            or company_factors != expected_context["company_factors"]
        ):
            raise ValueError("La proyección requiere el catálogo histórico y sus monedas acordadas")
    if source_unit == "CNY" or any(
        isinstance(name, str) and name.startswith("cn-reported:") for name in concepts
    ):
        if (
            source_unit != "CNY"
            or clock.market != "CN"
            or company_factors
            or not set(concepts) <= _CNY_CONCEPTS.keys()
        ):
            raise ValueError(
                "CNY requiere calendario CN, conceptos revisados y factores desactivados"
            )
    encoder_hash = _digest(encoders.spec)
    limits = dict(
        sample_batch_rows=batch_rows,
        partition_bytes=max_partition_bytes,
        partition_rows=max_partition_rows,
        news_group_bytes=max_news_group_bytes,
    )
    identity = dict(
        **policy_identity(input_policy),
        cohort_id=cohort,
        market=clock.market,
        symbol=origin["symbol"],
        prepared_manifest_sha256=origin_hash,
        calendar=calendar,
        macro_sha256=macros.sha256,
        macro_indicators=list(macros.indicators),
        macro_cutoff_year=getattr(macros, "cutoff_year", 2023),
        encoders_sha256=encoder_hash,
        context_sessions=context,
        news_lookback_sessions=news_lookback_sessions,
        fundamental_concepts=list(target),
        source_unit=source_unit,
        admitted_decisions_sha256=_digest(sorted(value.isoformat() for value in admitted_decisions))
        if admitted_decisions is not None
        else None,
        limits=limits,
        code={
            name: sha256(Path(__file__).with_name(name))
            for name in (
                "cohort_samples.py",
                "cohort_contexts.py",
                "cohort_files.py",
                "samples.py",
                "charts.py",
                "company_factors.py",
                "batches.py",
                "storage.py",
                "temporal.py",
                "input_policy.py",
            )
        },
    )
    if target_fundamental_concepts is not None:
        identity["accounting_projection"] = dict(
            source_concepts=list(concepts), target_concepts=list(target)
        )
        for name in ("accounting_catalog.py", "joint_projection.py"):
            identity["code"][name] = sha256(Path(__file__).with_name(name))
    if source_unit == "CNY" or target_fundamental_concepts is not None:
        identity["code"]["china_sources.py"] = sha256(Path(__file__).with_name("china_sources.py"))
    fingerprint = _digest(identity)
    for name in (
        ".asset.lock",
        "configuration.json",
        "manifest.json",
        "samples.parquet",
        "company-factors.parquet",
    ):
        if (destination / name).is_symlink():
            raise ValueError("Un destino de materialización es un enlace")
    destination.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with (destination / ".asset.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config, receipt = destination / "configuration.json", destination / "manifest.json"
        if config.exists():
            if json.loads(config.read_text()) != identity:
                raise ValueError("La configuración pertenece a otra edición")
        elif any(p.name != ".asset.lock" for p in destination.iterdir()):
            raise ValueError("El destino contiene una edición no identificada")
        else:
            atomic_json(config, identity)
        samples_path = destination / "samples.parquet"
        expected = dict(
            **policy_identity(input_policy),
            schema_version=4 if masked else 3,
            cohort_id=cohort,
            market=clock.market,
            symbol=origin["symbol"],
            news_content_policy=COHORT_POLICIES[cohort],
            training_ready=False,
            fingerprint=fingerprint,
            context_sessions=context,
            prepared_fingerprint=origin["fingerprint"],
            representation_code=identity["code"],
            fundamental_concepts=list(target),
            macro_indicators=macros.indicators,
            encoders=encoders.spec,
            news_lookback_sessions=news_lookback_sessions,
            text_aggregation="float64_sum_float32_mean_all_admitted_events",
            source_unit=source_unit,
            admitted_decisions_sha256=identity["admitted_decisions_sha256"],
        )
        if receipt.exists():
            old = json.loads(receipt.read_text())
            if (
                any(old.get(k) != v for k, v in expected.items())
                or type(old.get("samples")) is not int
            ):
                raise ValueError("El recibo no conserva la identidad declarada")
            if target_fundamental_concepts is not None:
                projection = old.get("accounting_projection")
                if (
                    not isinstance(projection, dict)
                    or _digest(
                        {key: projection.get(key) for key in identity["accounting_projection"]}
                    )
                    != _digest(identity["accounting_projection"])
                    or type(projection.get("outside_catalog_facts")) is not int
                    or not 0
                    <= projection["outside_catalog_facts"]
                    <= origin["counts"]["fundamentals"]
                ):
                    raise ValueError(
                        "El recibo no conserva los catálogos de la proyección contable"
                    )
            if _artifact(samples_path, destination) != old.get("samples_sha256"):
                raise ValueError("Ha cambiado el artefacto materializado")
            with pq.ParquetFile(samples_path) as table:
                if table.metadata.num_rows != old["samples"]:
                    raise ValueError("El recuento del recibo no coincide con las muestras")
            if company_factors and _artifact(
                destination / "company-factors.parquet", destination
            ) != old.get("company_factors_sha256"):
                raise ValueError("Ha cambiado el artefacto de factores empresariales")
            return {**old, "reused": True}
        prices = read_bounded_table(
            source / "prices.parquet", max_rows=max_partition_rows, max_bytes=max_partition_bytes
        ).to_pandas()
        facts = read_bounded_table(
            source / "fundamentals.parquet",
            max_rows=max_partition_rows,
            max_bytes=max_partition_bytes,
        ).to_pylist()
        if source_unit == "CNY":
            _chinese_facts(facts, clock, origin["policy"]["cutoff"])
        if target_fundamental_concepts is not None and any(
            row["concept"].startswith("us-gaap:")
            and row["concept"].rsplit(":", 1)[-1] != row.get("unit")
            for row in facts
        ):
            raise ValueError("El concepto contable no conserva su moneda o unidad")
        factor_audit = {}
        if company_factors:
            # Los hechos sin publicación conservan su diagnóstico, pero no forman ratios.
            factor_facts = [r for r in facts if r["available_at"] is not None] if masked else facts
            if target_fundamental_concepts is not None:
                factor_facts = [r for r in factor_facts if r["unit"] == source_unit]
            derived, factor_audit = write_company_factors(
                destination / "company-factors.parquet",
                factor_facts,
                batch_rows=batch_rows,
                max_facts=max_partition_rows,
                source_unit=source_unit,
            )
            facts.extend(derived)
        excluded, hits, misses, missing = Counter(), Counter(), Counter(), Counter()
        schema = _schema(concepts, macros.indicators, input_policy)
        with NewsWindows(
            source / "news/news.parquet", max_group_bytes=max_news_group_bytes
        ) as news:
            rows = _rows(
                prices,
                facts,
                clock,
                news,
                macros,
                encoders,
                cache,
                cohort=cohort,
                context=context,
                lookback=news_lookback_sessions,
                concepts=concepts,
                encoder_hash=encoder_hash,
                excluded=excluded,
                hits=hits,
                misses=misses,
                admitted_decisions=admitted_decisions,
                input_policy=input_policy,
                missing_sources=origin.get("missing_sources", ()),
                fundamental_exclusions=origin.get("fundamentals_audit", {}).get(
                    "temporal_exclusions", ()
                ),
            )

            def batches():
                pending = []

                def table(rows):
                    batch = pa.Table.from_pylist(rows, schema=schema)
                    return (
                        _project_fundamentals(batch, concepts, target)
                        if target_fundamental_concepts is not None
                        else batch
                    )

                for row in rows:
                    if masked:
                        missing.update(
                            f"{name}:{reason}"
                            for name, reason in row["missing_reasons"].items()
                            if reason is not None
                        )
                    pending.append(row)
                    if len(pending) == batch_rows:
                        yield table(pending)
                        pending = []
                yield table(pending)

            count = atomic_parquet_batches(samples_path, batches())
        if sha256(source / "manifest.json") != origin_hash or any(
            sha256(source / name) != digest for name, digest in origin["artifacts"].items()
        ):
            raise ValueError("Las entradas cambiaron durante la materialización")
        macros.verify()
        if count + sum(excluded.values()) != len(prices):
            raise ValueError("Las admisiones y exclusiones no concilian con los precios")
        result = dict(
            **expected,
            samples=count,
            samples_sha256=sha256(samples_path),
            macro_sha256=macros.sha256,
            company_factors_audit=factor_audit,
            company_factors_sha256=sha256(destination / "company-factors.parquet")
            if company_factors
            else None,
            limits=limits,
            excluded_reasons=dict(excluded),
            cache_hits=dict(hits),
            cache_misses=dict(misses),
            elapsed_seconds=time.perf_counter() - started,
            peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            pending_for_scientific_training=["targets", "frozen_scientific_cohort_and_splits"],
            reused=False,
        )
        if masked:
            result["missing_input_reasons"] = dict(missing)
        if target_fundamental_concepts is not None:
            result["accounting_projection"] = dict(
                identity["accounting_projection"],
                outside_catalog_facts=sum(r["concept"] not in concepts for r in facts),
            )
        atomic_json(receipt, result)
        return result
