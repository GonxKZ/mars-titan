"""Identidad y límites de disponibilidad de índices publicados por sus proveedores."""

import calendar

MONTH_ABBREVIATIONS = (
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
)

_MODELS = {
    "us_financial_conditions": ("NFCI", "Chicago Fed via FRED", "2011-05-25"),
    "us_financial_stress": ("STLFSI4", "St. Louis Fed via FRED", "2022-11-10"),
    "us_financial_stress_v3": ("STLFSI3", "St. Louis Fed via FRED", "2022-01-13"),
}
_SERIES = frozenset(model[0] for model in _MODELS.values())

STRESS_HISTORY_POLICY = "STLFSI_PUBLISHED_HISTORY_V1"


def stress_history_identity() -> dict:
    """Identificar la composición publicada de STLFSI3 y STLFSI4, sin equivalencia de niveles."""
    return {
        "id": "us_financial_stress",
        "kind": "raw",
        "series_id": "STLFSI3|STLFSI4",
        "provider": "St. Louis Fed via FRED",
        "frequency": "W_FRI",
        "source_url": "https://fred.stlouisfed.org/series/STLFSI4",
        "unit": "index",
        "formula": "",
        "input_ids": "",
        "vintage_policy": STRESS_HISTORY_POLICY,
        "availability_rule": "STLFSI3_20220113_STLFSI4_20221110_NEXT_SESSION",
        "verification_status": "verified_composed_model_vintages",
    }


def model_vintage_contract(entry: dict) -> tuple[str | None, str | None]:
    """Devuelve el primer límite temporal admisible y una posible exclusión.

    Cambiar la política de una serie registrada no elimina su contrato. Las
    fechas limitan el archivo verificado, no el periodo observado en cada fila.
    En GSCPI se usa el final del primer mes regular admitido, no un día exacto de publicación.
    Las fuentes y el alcance se describen en docs/data/macro-catalog.md.
    """
    policy = entry.get("vintage_policy")
    if policy == STRESS_HISTORY_POLICY or entry.get("series_id") == "STLFSI3|STLFSI4":
        if any(entry.get(field) != value for field, value in stress_history_identity().items()):
            return None, "unverified_stress_history"
        return "2022-01-13", None
    if (
        entry.get("id") == "global_supply_pressure"
        or entry.get("series_id") == "GSCPI"
        or policy == "NYFED_MONTHLY_VINTAGES"
    ):
        expected = {
            "id": "global_supply_pressure",
            "kind": "raw",
            "series_id": "GSCPI",
            "provider": "New York Fed",
            "frequency": "M",
            "source_url": "https://www.newyorkfed.org/research/policy/gscpi",
            "unit": "standard_deviations_of_provider_historical_mean",
            "vintage_policy": "NYFED_MONTHLY_VINTAGES",
            "availability_rule": "END_OF_VINTAGE_MONTH_THEN_NEXT_SESSION",
            "verification_status": "verified_metadata_not_ingested",
        }
        if any(entry.get(field) != value for field, value in expected.items()):
            return None, "unverified_monthly_model_vintages"
        return "2022-05-31", None
    model = _MODELS.get(entry.get("id"))
    if (
        model is None
        and entry.get("series_id") not in _SERIES
        and policy != "ALFRED_MODEL_VINTAGES"
    ):
        return None, None
    if model is None:
        return None, "unverified_model_vintages"
    series, provider, first = model
    expected = {
        "kind": "raw",
        "series_id": series,
        "provider": provider,
        "source_url": f"https://fred.stlouisfed.org/series/{series}",
        "frequency": "W_FRI",
        "unit": "index",
        "vintage_policy": "ALFRED_MODEL_VINTAGES",
        "verification_status": "verified_metadata_not_ingested",
    }
    if any(entry.get(field) != value for field, value in expected.items()):
        return None, "unverified_model_vintages"
    return first, None


def validate_monthly_bound(entry, row, original_start):
    """Impedir que una etiqueta mensual se convierta en una publicación diaria inventada."""
    if entry.get("vintage_policy") != "NYFED_MONTHLY_VINTAGES":
        return
    label = f"{MONTH_ABBREVIATIONS[original_start.month - 1]}-{original_start.year % 100:02d}"
    if (
        original_start.day != calendar.monthrange(original_start.year, original_start.month)[1]
        or row.get("availability_precision") != "month"
        or row.get("availability_policy") != "end_of_vintage_month_then_next_session"
        or row.get("publication_timestamp_verified") is not False
        or row.get("vintage_label") != label
    ):
        raise ValueError("La versión mensual necesita un límite conservador explícito")
