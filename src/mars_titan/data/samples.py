"""Intersección temporal de cuatro modalidades y contexto macro obligatorio."""

import hashlib
import json
import math
import resource
import time
from bisect import bisect_left, bisect_right
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa

from .batches import MacroContexts, atomic_parquet_batches, read_bounded_table
from .charts import chart_png
from .company_factors import FACTOR_CONCEPTS, FACTOR_DEFINITIONS, write_company_factors
from .fundamentals import snapshot
from .storage import atomic_json, outside_source, sha256
from .temporal import MarketClock, admission_errors, aware

FUNDAMENTAL_CONCEPTS = tuple(
    f"us-gaap:{tag}:USD"
    for tag in (
        "Assets",
        "AssetsCurrent",
        "Liabilities",
        "LiabilitiesCurrent",
        "StockholdersEquity",
        "CashAndCashEquivalentsAtCarryingValue",
        "AccountsPayableCurrent",
        "AccountsReceivableNetCurrent",
    )
)


def sample_table(rows: list[dict], *, fundamental_concepts=FUNDAMENTAL_CONCEPTS) -> pa.Table:
    if not rows:
        return pa.Table.from_pylist([])
    widths = {
        "news": 384,
        "charts": 512,
        "fundamentals": 3 * len(fundamental_concepts),
        "macro": len(rows[0]["macro"]),
    }
    schema = pa.Table.from_pylist(rows[:1]).schema
    schema = pa.schema(
        [
            pa.field(field.name, pa.list_(pa.float32(), widths[field.name]))
            if field.name in widths
            else field
            for field in schema
        ]
    )
    return pa.Table.from_pylist(rows, schema=schema)


def numeric_context(values: list[float | None], ages: list[float]) -> list[float]:
    """Transformación fija, sin ajustar estadísticas con otras muestras."""
    observed = [v is not None and math.isfinite(v) for v in values]
    levels = [
        math.copysign(math.log1p(abs(v)), v) if ok else 0.0
        for v, ok in zip(values, observed, strict=True)
    ]
    return (
        levels
        + [float(x) for x in observed]
        + [math.log1p(age) if ok else 0.0 for age, ok in zip(ages, observed, strict=True)]
    )


def macro_vector(rows: list[dict], cutoff: datetime) -> tuple[list[float], datetime]:
    cutoff = aware(cutoff)
    known = [r for r in rows if r["value"] is not None]
    if not known:
        raise ValueError("No hay contexto macro observado en esta decisión")
    if any(not math.isfinite(r["value"]) for r in known):
        raise ValueError("El contexto macro debe contener observaciones finitas")
    if any(r.get("available_at") is None for r in known):
        raise ValueError("Hay un valor macro observado sin fecha de disponibilidad")
    if any(aware(r["available_at"]) > cutoff for r in known):
        raise ValueError("El contexto macro contiene información futura")
    rows = sorted(rows, key=lambda r: r["indicator_id"])
    if len({r["indicator_id"] for r in rows}) != len(rows):
        raise ValueError("Hay un indicador macro duplicado en la decisión")
    ages = [
        (cutoff - aware(r["available_at"])).total_seconds() / 86400
        if r.get("available_at")
        else 0.0
        for r in rows
    ]
    return numeric_context([r["value"] for r in rows], ages), max(r["available_at"] for r in known)


def eligible_samples(
    prices: pd.DataFrame,
    news: list[dict],
    facts: list[dict],
    clock: MarketClock,
    *,
    context: int = 64,
    news_lookback_sessions: int = 5,
    fundamental_concepts=FUNDAMENTAL_CONCEPTS,
):
    if context < 2 or news_lookback_sessions < 1:
        raise ValueError("El contexto o la ventana retrospectiva de noticias no es válido")
    if not facts or not news:
        return
    news = sorted(news, key=lambda n: (n["available_at"], n["content_hash"]))
    times = [r["available_at"] for r in news]
    session_positions = {d.isoformat(): i for i, d in enumerate(clock.days)}
    sessions = list(prices["session"])
    for index in range(context - 1, len(prices)):
        day = sessions[index]
        position = session_positions[day]
        if position < context - 1:
            continue
        expected = [d.isoformat() for d in clock.days[position - context + 1 : position + 1]]
        if sessions[index - context + 1 : index + 1] != expected:
            continue
        cutoff = clock.decisions[position]
        beginning = clock.decisions[max(0, position - news_lookback_sessions + 1)]
        first, last = bisect_left(times, beginning), bisect_right(times, cutoff)
        if first == last:
            continue
        known = snapshot(facts, cutoff)
        selected = [known.get(concept) for concept in fundamental_concepts]
        if not any(r and r["value"] is not None for r in selected):
            continue
        available = max(r["available_at"] for r in selected if r)
        availability = {
            "prices": prices.iloc[index]["available_at"],
            "news": news[last - 1]["available_at"],
            "fundamentals": available,
            "charts": cutoff,
        }
        if any(aware(value) > cutoff for value in availability.values()):
            raise ValueError("Las modalidades candidatas contienen información futura")
        values = [r["value"] if r else None for r in selected]
        ages = [
            (cutoff - r["available_at"]).total_seconds() / 86400 if r else 0.0 for r in selected
        ]
        yield {
            "prediction_at": cutoff,
            "session": day,
            "price_end_index": index,
            "news_indices": list(range(first, last)),
            "fundamentals": numeric_context(values, ages),
            "fundamentals_available_at": available,
            "fundamental_accessions": sorted({r["accession"] for r in selected if r}),
            "news_available_at": news[last - 1]["available_at"],
            "input_availability": availability,
        }


def validate_sample_inputs(
    prepared: Path, destination: Path, panel: dict, clock: MarketClock
) -> tuple[str, dict[str, dict]]:
    """Comprueba destinos y calendarios antes de inicializar recursos o escribir."""
    outside_source(Path("dataset"), destination)
    if prepared.resolve() == destination.resolve():
        raise ValueError("La salida de muestras no puede sobrescribir los manifiestos preparados")
    calendar_fingerprint = hashlib.sha256(
        "|".join(value.isoformat() for value in clock.decisions).encode()
    ).hexdigest()
    manifests = {}
    for asset in panel["assets"]:
        symbol = asset["symbol"]
        target = destination / clock.market / symbol
        outside_source(Path("dataset"), target)
        for market in ("US", "CN"):
            outside_source(prepared / market, target)
        manifest = json.loads((prepared / clock.market / symbol / "manifest.json").read_text())
        if manifest.get("policy", {}).get("calendar") != calendar_fingerprint:
            raise ValueError(
                f"El calendario preparado difiere del calendario de las muestras: {symbol}"
            )
        manifests[symbol] = manifest
    return calendar_fingerprint, manifests


def materialize_samples(
    prepared: Path,
    macro_path: Path,
    destination: Path,
    panel: dict,
    clock: MarketClock,
    encoders,
    cache,
    *,
    batch_rows: int = 256,
    max_partition_bytes: int = 64 * 1024**2,
    max_partition_rows: int = 100_000,
    company_factors: bool = False,
) -> dict:
    """Guarda vectores por activo. Las ventanas de precios se obtienen bajo demanda."""
    if type(batch_rows) is not int or not 1 <= batch_rows <= 1024:
        raise ValueError("El tamaño del bloque de muestras no es válido")
    if type(company_factors) is not bool:
        raise ValueError("La selección de factores empresariales debe ser booleana")
    fundamental_concepts = FUNDAMENTAL_CONCEPTS + (FACTOR_CONCEPTS if company_factors else ())
    limits = {
        "sample_batch_rows": batch_rows,
        "partition_bytes": max_partition_bytes,
        "partition_rows": max_partition_rows,
    }
    calendar_fingerprint, manifests = validate_sample_inputs(prepared, destination, panel, clock)
    import torch

    with MacroContexts(macro_path) as macro_contexts:
        macro_schema = macro_contexts.indicators
        macro_hash = sha256(macro_path)
        encoder_fingerprint = hashlib.sha256(
            json.dumps(encoders.spec, sort_keys=True).encode()
        ).hexdigest()
        reports = []
        for asset in panel["assets"]:
            symbol = asset["symbol"]
            source = prepared / clock.market / symbol
            manifest = manifests[symbol]
            for name, digest in manifest["artifacts"].items():
                if sha256(source / name) != digest:
                    raise ValueError(f"Ha cambiado un artefacto preparado: {symbol}/{name}")
            target = destination / clock.market / symbol
            fingerprint = hashlib.sha256(
                json.dumps(
                    {
                        "prepared": manifest["fingerprint"],
                        "calendar": calendar_fingerprint,
                        "macro": macro_hash,
                        "encoders": encoder_fingerprint,
                        "sample_code": sha256(Path(__file__)),
                        "batch_code": sha256(Path(__file__).with_name("batches.py")),
                        "company_factor_code": sha256(
                            Path(__file__).with_name("company_factors.py")
                        ),
                        "fundamental_concepts": fundamental_concepts,
                        "limits": limits,
                        "renderer": sha256(Path(__file__).with_name("charts.py")),
                        "context": 64,
                        "news_lookback_sessions": 5,
                    },
                    sort_keys=True,
                ).encode()
            ).hexdigest()
            receipt_path = target / "manifest.json"
            if receipt_path.exists():
                old = json.loads(receipt_path.read_text())
                if old["fingerprint"] == fingerprint and (target / "samples.parquet").exists():
                    factor_path = target / "company-factors.parquet"
                    factors_valid = not company_factors or (
                        factor_path.is_file()
                        and sha256(factor_path) == old.get("company_factors_sha256")
                    )
                    if (
                        factors_valid
                        and sha256(target / "samples.parquet") == old["samples_sha256"]
                    ):
                        reports.append({**old, "reused": True})
                        continue
            started = time.perf_counter()
            torch.cuda.reset_peak_memory_stats()
            prices = read_bounded_table(
                source / "prices.parquet",
                max_rows=max_partition_rows,
                max_bytes=max_partition_bytes,
            ).to_pandas()
            news = sorted(
                read_bounded_table(
                    source / "news.parquet",
                    max_rows=max_partition_rows,
                    max_bytes=max_partition_bytes,
                ).to_pylist(),
                key=lambda r: (r["available_at"], r["content_hash"]),
            )
            facts = read_bounded_table(
                source / "fundamentals.parquet",
                max_rows=max_partition_rows,
                max_bytes=max_partition_bytes,
            ).to_pylist()
            factor_audit = {}
            if company_factors:
                derived, factor_audit = write_company_factors(
                    target / "company-factors.parquet",
                    facts,
                    batch_rows=batch_rows,
                    max_facts=max_partition_rows,
                )
                facts.extend(derived)
            ohlc = prices[["open", "high", "low", "close"]].to_numpy()
            no_macro = 0

            def encoded_batches(prices=prices, news=news, facts=facts, ohlc=ohlc):
                nonlocal no_macro
                samples = []
                for sample in eligible_samples(
                    prices, news, facts, clock, fundamental_concepts=fundamental_concepts
                ):
                    context = macro_contexts.at(sample["prediction_at"])
                    if (
                        not context
                        or sorted(r["indicator_id"] for r in context) != macro_schema
                        or not any(r["value"] is not None for r in context)
                    ):
                        no_macro += 1
                        continue
                    macro, macro_available = macro_vector(context, sample["prediction_at"])
                    errors = admission_errors(
                        {**sample["input_availability"], "macro": macro_available},
                        sample["prediction_at"],
                    )
                    if errors:
                        raise ValueError(f"La muestra no es admisible para entrenamiento: {errors}")
                    texts = []
                    for index in sample["news_indices"]:
                        article = news[index]
                        identity = {
                            "encoder": encoder_fingerprint,
                            "kind": "news",
                            "content": article["content_hash"],
                            "policy": article["availability_rule"],
                        }
                        vector = cache.get(identity)
                        if vector is None:
                            vector = encoders.text(article["text"])
                            cache.put(identity, vector)
                        texts.append(vector)
                    png = chart_png(ohlc, end_index=sample["price_end_index"])
                    chart_hash = hashlib.sha256(png).hexdigest()
                    identity = {
                        "encoder": encoder_fingerprint,
                        "kind": "chart",
                        "content": chart_hash,
                    }
                    image = cache.get(identity)
                    if image is None:
                        image = encoders.images([png])[0]
                        cache.put(identity, image)
                    text = np.mean(texts, axis=0)
                    if not np.isfinite(text).all() or not np.isfinite(image).all():
                        raise ValueError("La representación multimodal contiene valores no finitos")
                    samples.append(
                        {
                            **sample,
                            "news": text.tolist(),
                            "charts": image.tolist(),
                            "macro": macro,
                            "macro_available_at": macro_available,
                            "chart_hash": chart_hash,
                            "news_hashes": [
                                news[i]["content_hash"] for i in sample["news_indices"]
                            ],
                            "macro_units": [
                                r["unit"] for r in sorted(context, key=lambda r: r["indicator_id"])
                            ],
                        }
                    )
                    if len(samples) == batch_rows:
                        yield sample_table(samples, fundamental_concepts=fundamental_concepts)
                        samples = []

                if samples:
                    yield sample_table(samples, fundamental_concepts=fundamental_concepts)

            count = atomic_parquet_batches(target / "samples.parquet", encoded_batches())
            result = {
                "schema_version": 2 if company_factors else 1,
                "fundamental_concepts": fundamental_concepts,
                "company_factors_audit": factor_audit,
                "company_factor_derivation": {
                    "definitions": json.loads(json.dumps(FACTOR_DEFINITIONS)),
                    "code_sha256": sha256(Path(__file__).with_name("company_factors.py")),
                    "source_unit": "USD",
                    "positive_denominator_required": True,
                    "transform": "signed_log1p_then_presence_then_log1p_age_days",
                }
                if company_factors
                else None,
                "company_factors_sha256": sha256(target / "company-factors.parquet")
                if company_factors
                else None,
                "symbol": symbol,
                "market": clock.market,
                "fingerprint": fingerprint,
                "samples": count,
                "limits": limits,
                "excluded_no_macro": no_macro,
                "samples_sha256": sha256(target / "samples.parquet"),
                "macro_sha256": macro_hash,
                "macro_indicators": macro_schema,
                "encoders": encoders.spec,
                "prepared_fingerprint": manifest["fingerprint"],
                "calendar": calendar_fingerprint,
                "context_sessions": 64,
                "news_lookback_sessions": 5,
                "elapsed_seconds": time.perf_counter() - started,
                "cost_profile_ready": bool(count),
                "training_ready": False,
                "news_content_policy": manifest.get("news_content_policy", "not_reviewed"),
                "pending_for_scientific_training": [
                    "targets",
                    "frozen_scientific_cohort_and_splits",
                ]
                + (
                    []
                    if manifest.get("news_content_policy") == "verified_full_articles"
                    else ["unverified_news"]
                ),
                "macro_unit_policy": "native_historical_levels_and_unit_compatible_derived_values",
                "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                "peak_vram_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
                "peak_vram_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
                "purpose": "engineering_profile_only",
            }
            atomic_json(receipt_path, result)
            reports.append(result)
        return {
            "market": clock.market,
            "assets": reports,
            "samples": sum(r["samples"] for r in reports),
        }
