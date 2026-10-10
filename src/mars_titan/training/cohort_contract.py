"""Identidad de evidencia compartida por etiquetas, lotes y ejecuciones."""

import hashlib
import json

from mars_titan.data.cohort_news import COHORT_POLICIES
from mars_titan.data.input_policy import STRICT_INPUTS, masked_inputs, policy_identity
from mars_titan.data.price_windows import check_price_window_contract


def input_identity(meta, *, input_policy=STRICT_INPUTS):
    """Exigir adhesión y el contrato completo sin interpretar ceros como ausencias."""
    expected = policy_identity(input_policy)
    supplied = {key: meta[key] for key in ("input_policy", "mask_contract") if key in meta}
    if json.dumps(supplied, sort_keys=True, allow_nan=False) != json.dumps(
        expected, sort_keys=True, allow_nan=False
    ):
        raise ValueError(
            "La política de entradas de la cohorte necesita adhesión y contrato exactos"
        )
    return expected


def cohort_identity(meta, *, input_policy=STRICT_INPUTS):
    """Conservar la edición histórica y exigir procedencia explícita en la nueva."""
    input_identity(meta, input_policy=input_policy)
    masked = masked_inputs(input_policy)
    if masked and (
        meta.get("training_ready") is not False or meta.get("final_test_opened") is not False
    ):
        raise ValueError("La cohorte histórica no acredita admisión ni apertura de la reserva")
    if type(meta.get("schema_version")) is not int:
        raise ValueError("La versión de la cohorte debe ser un entero")
    if meta.get("schema_version") == 1:
        if masked or "cohort_id" in meta or "news_content_policy" in meta:
            raise ValueError("La cohorte explícita requiere la versión 2")
        return None
    cohort = meta.get("cohort_id")
    if (
        meta.get("schema_version") != (3 if masked else 2)
        or cohort not in COHORT_POLICIES
        or meta.get("news_content_policy") != COHORT_POLICIES[cohort]
        or any(a.get("cohort_id") != cohort for a in meta.get("assets", []))
    ):
        raise ValueError("La cohorte del manifiesto y sus activos no coincide")
    validate_population(meta, input_policy=input_policy)
    if meta.get("kind") == "corpus_supervision":
        signature = representation_hash(meta.get("representation", {}), input_policy=input_policy)
        if any(a.get("representation_sha256") != signature for a in meta["assets"]):
            raise ValueError("Las representaciones no conservan una identidad semántica común")
    return cohort


def validate_cohort_rows(table, cohort):
    if cohort is not None and (
        "cohort_id" not in table.column_names
        or any(value != cohort for value in table["cohort_id"].to_pylist())
    ):
        raise ValueError("Hay filas ausentes o de otra cohorte")


def validate_population(meta, *, input_policy=STRICT_INPUTS):
    missing = "missing_required_prices" if masked_inputs(input_policy) else "missing_modalities"
    coverage, expected = meta.get("coverage"), {}
    if (
        not isinstance(coverage, list)
        or type(meta.get("candidate_count")) is not int
        or meta["candidate_count"] != len(coverage)
    ):
        raise ValueError("La cobertura no reconcilia todos los candidatos")
    keys, total, failures = set(), 0, 0
    for row in coverage:
        key = row.get("market"), row.get("symbol")
        if key in keys:
            raise ValueError("La cobertura contiene candidatos duplicados")
        keys.add(key)
        state = row.get("state")
        if state == "encoded":
            count = row.get("samples")
            if type(count) is not int or count < 0:
                raise ValueError("La cobertura necesita recuentos enteros no negativos")
            total += count
            if count:
                expected[key] = count
        elif state == "failed":
            failures += 1
        elif state != missing:
            raise ValueError("La cobertura tiene un estado desconocido")
    observed = [(a["market"], a["symbol"]) for a in meta.get("assets", [])]
    if "markets" in meta and (
        not isinstance(meta["markets"], list)
        or len(meta["markets"]) != len(set(meta["markets"]))
        or set(meta["markets"]) != {market for market, _ in keys}
    ):
        raise ValueError("Los mercados declarados no coinciden con la cobertura de candidatos")
    if (
        len(observed) != len(set(observed))
        or set(observed) != set(expected)
        or meta.get("samples") != total
        or meta.get("failed_assets") != failures
        or (meta.get("cohort_complete") is True and failures)
        or any(
            "samples" in a and a["samples"] != expected[(a["market"], a["symbol"])]
            for a in meta.get("assets", [])
        )
    ):
        raise ValueError("La población, las muestras y la cobertura no concilian")


def representation_identity(receipt, *, input_policy=STRICT_INPUTS):
    identity = input_identity(receipt, input_policy=input_policy)
    fields = (
        "fundamental_concepts",
        "macro_indicators",
        "encoders",
        "representation_code",
        "text_aggregation",
        "context_sessions",
        "news_lookback_sessions",
    )
    if any(not receipt.get(field) for field in fields):
        raise ValueError("Falta la identidad semántica de la representación")
    for field in ("fundamental_concepts", "macro_indicators"):
        items = receipt[field]
        if (
            not isinstance(items, list)
            or any(not isinstance(v, str) or not v for v in items)
            or len(set(items)) != len(items)
        ):
            raise ValueError("Los conceptos e indicadores deben ser únicos y explícitos")
    result = {field: receipt[field] for field in fields} | identity
    if "price_window" in receipt:
        # Solo las ediciones desde la v3.1 lo declaran, así que las anteriores conservan su huella.
        result["price_window"] = check_price_window_contract(receipt["price_window"])
    return result


def representation_hash(receipt, *, input_policy=STRICT_INPUTS):
    return hashlib.sha256(
        json.dumps(
            representation_identity(receipt, input_policy=input_policy), sort_keys=True
        ).encode()
    ).hexdigest()
