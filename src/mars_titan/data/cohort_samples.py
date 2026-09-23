"""Vectores por ventanas temporales, con límites de memoria y cohorte identificada."""

import fcntl
import hashlib
import json
import resource
import time
from collections import Counter
from itertools import chain
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .batches import atomic_parquet_batches, read_bounded_table
from .charts import chart_png
from .cohort_contexts import FactCursor, NewsWindows
from .cohort_files import read_manifest, safe_destination
from .cohort_news import COHORT_POLICIES
from .company_factors import FACTOR_CONCEPTS, write_company_factors
from .samples import FUNDAMENTAL_CONCEPTS, numeric_context
from .storage import atomic_json, outside_source, sha256
from .temporal import admission_errors, aware

_KINDS = ("article_candidate", "summary", "verified_full_article")


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _schema(concepts, indicators):
    stamp = pa.timestamp("us", tz="UTC")
    return pa.schema(
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


def _artifact(path, root):
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Un artefacto no es regular o está fuera de su origen")
    return sha256(path)


def _prepared(source, clock, cohort):
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
        or manifest.get("schema_version") != 3
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
):
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
    cursor = FactCursor(facts)
    for index, position in enumerate(indices):
        if index < context - 1 or position - indices[index - context + 1] != context - 1:
            excluded["incomplete_price_window"] += 1
            continue
        cutoff = clock.decisions[position]
        known = cursor.at(cutoff)
        selected = [known.get(concept) for concept in concepts]
        if not any(r is not None and r["value"] is not None for r in selected):
            excluded["missing_fundamentals"] += 1
            continue
        start = clock.decisions[max(0, position - lookback + 1)]
        articles = news.between(start, cutoff)
        first = next(articles, None)
        if first is None:
            excluded["missing_news"] += 1
            continue
        macro = macros.at(cutoff)
        if macro is None:
            excluded["missing_macro"] += 1
            continue
        text = _text_window(
            chain((first,), articles), cohort, encoder_hash, encoders, cache, hits, misses
        )
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
        if admission_errors(available, cutoff):
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
        yield dict(
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
):
    """Confirmar un activo completo. La caché persiste aunque se interrumpa su escritura."""
    for value, low, high in (
        (context, 2, 512),
        (news_lookback_sessions, 1, 512),
        (batch_rows, 1, 1024),
    ):
        if type(value) is not int or not low <= value <= high:
            raise ValueError("El tamaño de contexto, ventana o lote no es válido")
    if type(company_factors) is not bool:
        raise ValueError("La selección de factores debe ser booleana")
    source, destination = Path(source), Path(destination)
    for root in (source, Path("dataset")):
        outside_source(root, destination)
    outside_source(destination, source)
    safe_destination(destination)
    macros.verify()
    origin, calendar, origin_hash = _prepared(source, clock, cohort)
    concepts = tuple(fundamental_concepts) + (FACTOR_CONCEPTS if company_factors else ())
    if not concepts or len(set(concepts)) != len(concepts):
        raise ValueError("El catálogo contable está vacío o duplicado")
    encoder_hash = _digest(encoders.spec)
    limits = dict(
        sample_batch_rows=batch_rows,
        partition_bytes=max_partition_bytes,
        partition_rows=max_partition_rows,
        news_group_bytes=max_news_group_bytes,
    )
    identity = dict(
        cohort_id=cohort,
        market=clock.market,
        symbol=origin["symbol"],
        prepared_manifest_sha256=origin_hash,
        calendar=calendar,
        macro_sha256=macros.sha256,
        encoders_sha256=encoder_hash,
        context_sessions=context,
        news_lookback_sessions=news_lookback_sessions,
        fundamental_concepts=list(concepts),
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
            )
        },
    )
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
            schema_version=3,
            cohort_id=cohort,
            market=clock.market,
            symbol=origin["symbol"],
            news_content_policy=COHORT_POLICIES[cohort],
            training_ready=False,
            fingerprint=fingerprint,
            context_sessions=context,
            prepared_fingerprint=origin["fingerprint"],
            representation_code=identity["code"],
            fundamental_concepts=list(concepts),
            macro_indicators=macros.indicators,
            encoders=encoders.spec,
            news_lookback_sessions=news_lookback_sessions,
            text_aggregation="float64_sum_float32_mean_all_admitted_events",
        )
        if receipt.exists():
            old = json.loads(receipt.read_text())
            if (
                any(old.get(k) != v for k, v in expected.items())
                or type(old.get("samples")) is not int
            ):
                raise ValueError("El recibo no conserva la identidad declarada")
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
        factor_audit = {}
        if company_factors:
            derived, factor_audit = write_company_factors(
                destination / "company-factors.parquet",
                facts,
                batch_rows=batch_rows,
                max_facts=max_partition_rows,
            )
            facts.extend(derived)
        excluded, hits, misses = Counter(), Counter(), Counter()
        schema = _schema(concepts, macros.indicators)
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
            )

            def batches():
                pending = []
                for row in rows:
                    pending.append(row)
                    if len(pending) == batch_rows:
                        yield pa.Table.from_pylist(pending, schema=schema)
                        pending = []
                yield pa.Table.from_pylist(pending, schema=schema)

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
        atomic_json(receipt, result)
        return result
