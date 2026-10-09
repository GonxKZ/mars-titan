"""Estratos por presencia de noticias y fundamentales para la comparación walk-forward.

Es un análisis secundario y descriptivo, declarado el 9 de octubre de 2026 antes de
cualquier resultado. No sirve para elegir modelos ni cambia la conclusión principal.
En la edición desde 2000, precios, gráficos y macro están presentes en todas las
filas, así que los cuatro estratos solo distinguen noticias y fundamentales.

Este módulo fija y valida la declaración previa, sin abrir ningún dato.
"""

from datetime import date

STATUS = "secondary_descriptive"
USE = "description_only_not_for_model_selection_or_primary_conclusion"
PRESENCE_SOURCE = "view_evaluation_labels_and_samples_presence_v1"
ALWAYS_PRESENT = ("prices", "charts", "macro")
STRATA = {
    "news_and_fundamentals": dict(news=True, fundamentals=True),
    "news_only": dict(news=True, fundamentals=False),
    "fundamentals_only": dict(news=False, fundamentals=True),
    "neither": dict(news=False, fundamentals=False),
}
MULTIPLICITY = "bonferroni_over_strata_and_views_with_max_t_within_family"
FIELDS = {
    "status",
    "declared_at",
    "use",
    "presence_source",
    "always_present",
    "strata",
    "focus",
    "metrics",
    "recalibrate",
    "min_rows",
    "min_sessions",
    "multiplicity",
}
MAX_THRESHOLD = 10_000_000


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _iso_date(value):
    try:
        return isinstance(value, str) and date.fromisoformat(value).isoformat() == value
    except ValueError:
        return False


def declaration(section, series_metrics):
    """Validar la declaración previa de los estratos sin abrir ningún dato."""
    _require(
        isinstance(section, dict) and set(section) == FIELDS,
        "La declaración de estratos no tiene exactamente sus campos",
    )
    metrics = section["metrics"]
    _require(
        section["status"] == STATUS
        and section["use"] == USE
        and _iso_date(section["declared_at"])
        and section["presence_source"] == PRESENCE_SOURCE
        and section["always_present"] == list(ALWAYS_PRESENT)
        and section["multiplicity"] == MULTIPLICITY,
        "Los estratos deben declararse como análisis secundario y descriptivo con su fuente",
    )
    _require(
        section["strata"] == STRATA and section["focus"] in STRATA,
        "Los estratos son los cuatro patrones de noticias y fundamentales",
    )
    _require(
        isinstance(metrics, list)
        and metrics
        and metrics[0] == "mae"
        and len(set(metrics)) == len(metrics)
        and set(metrics) <= set(series_metrics),
        "Los contrastes por estrato deben empezar por el MAE y usar métricas por sesión",
    )
    _require(section["recalibrate"] is False, "Ningún estrato puede volver a calibrar")
    for name in ("min_rows", "min_sessions"):
        value = section[name]
        _require(
            type(value) is int and 1 <= value <= MAX_THRESHOLD,
            f"El umbral {name} debe ser un entero positivo fijado de antemano",
        )
    return section
