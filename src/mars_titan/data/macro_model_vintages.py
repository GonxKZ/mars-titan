"""Identidad y comienzo comprobados de los modelos publicados en ALFRED."""

_MODELS = {
    "us_financial_conditions": ("NFCI", "Chicago Fed via FRED", "2011-05-25"),
    "us_financial_stress": ("STLFSI4", "St. Louis Fed via FRED", "2022-11-10"),
}
_SERIES = frozenset(model[0] for model in _MODELS.values())


def model_vintage_contract(entry: dict) -> tuple[str | None, str | None]:
    """Devuelve primera versión y exclusión, sin autorizar modelos desconocidos.

    Cambiar la política de una serie registrada no elimina su contrato. Las
    fechas limitan el archivo verificado, no el periodo observado en cada fila.
    Las fuentes y el alcance se describen en docs/data/macro-catalog.md.
    """
    policy = entry.get("vintage_policy")
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
