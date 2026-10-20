"""Formato acotado del feedback de cohortes, independiente de la publicación de sesión."""

import copy
import hashlib
import math

import numpy as np

from mars_titan.models.quantile_head import LEVELS

from .online_conformal import MAX_PENDING, OnlineConformal, _canonical, _instant, _require

KIND = "emitted_conformal_cohort_feedback"
POLICY = "complete_oldest_or_cancel_whole_excluded_cohort_v1"
MAX_ROWS = 32768
MAX_BYTES = 16 * 1024**2
FIELDS = {
    "schema_version",
    "kind",
    "policy",
    "max_rows",
    "max_bytes",
    "clock",
    "cancelled_cohorts",
    "calibrator",
    "cohorts",
    "state_sha256",
}
ROW_FIELDS = {"sample_id", "matures_at", "available_at", "value", "status", "evidence_sha256"}


def copy_cohorts(cohorts):
    """Copiar contenedores de filas ya validadas, cuyos campos son escalares inmutables."""
    return [dict(item, rows=[dict(row) for row in item["rows"]]) for item in cohorts]


def identifier(value):
    _require(
        type(value) is str
        and 0 < len(value) <= 128
        and value.isprintable()
        and len(value.encode()) <= 128,
        "La identidad de fila o mercado debe tener entre 1 y 128 bytes imprimibles",
    )
    return value


def limits(max_rows, max_bytes):
    _require(type(max_rows) is int and 1 <= max_rows <= MAX_ROWS, "El límite de filas no es válido")
    _require(
        type(max_bytes) is int and 1024 <= max_bytes <= MAX_BYTES,
        "El límite de bytes del estado no es válido",
    )


def real(value):
    _require(
        isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool),
        "La etiqueta debe ser un número real finito",
    )
    try:
        value = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError("La etiqueta no se puede representar como float64") from error
    _require(math.isfinite(value), "La etiqueta debe ser un número real finito")
    return value


def validate_feedback(row, prediction_at, clock):
    available = _instant(row["available_at"], "La disponibilidad del resultado")
    _require(prediction_at <= available <= clock, "El resultado no estaba disponible en el corte")
    if row["status"] == "label":
        _require(available >= row["matures_at"], "La etiqueta todavía no había madurado")
        _require(row["evidence_sha256"] is None, "Una etiqueta no es una exclusión")
        return real(row["value"])
    evidence = row["evidence_sha256"]
    _require(
        row["status"] in ("excluded", "finalized")
        and row["value"] is None
        and type(evidence) is str
        and len(evidence) == 64
        and all(character in "0123456789abcdef" for character in evidence),
        "La exclusión necesita estado identificado, evidencia y ausencia de etiqueta",
    )
    return None


def seal(payload):
    limits(payload["max_rows"], payload["max_bytes"])
    _require(
        type(payload["cancelled_cohorts"]) is int and 0 <= payload["cancelled_cohorts"] < 2**63,
        "El recuento de cancelaciones no es válido",
    )
    try:
        canonical = _canonical(payload)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("El estado necesita valores JSON finitos") from error
    digest = hashlib.sha256(canonical.encode()).hexdigest()
    result = dict(payload, state_sha256=digest)
    _require(
        len(_canonical(result).encode()) <= payload["max_bytes"],
        "El estado supera su presupuesto de bytes serializados",
    )
    return result


def packed_calibrator(calibrator):
    payload = calibrator.export()
    for market in payload["markets"].values():
        for cohort in market["pending"]:
            cohort["quantiles"] = cohort["quantiles"].tolist()
    return payload


def unpack_calibrator(payload, max_rows):
    _require(
        type(payload) is dict
        and set(payload) == {"header", "markets", "state_sha256"}
        and type(payload["header"]) is dict
        and payload["header"].get("levels") == list(LEVELS)
        and type(payload["markets"]) is dict
        and 1 <= len(payload["markets"]) <= 8,
        "El feedback v1 necesita los cinco cuantiles financieros y mercados acotados",
    )
    rows = 0
    for name, market in payload["markets"].items():
        identifier(name)
        _require(type(market) is dict and type(market.get("pending")) is list, "Cola incompatible")
        _require(len(market["pending"]) <= MAX_PENDING, "Demasiadas cohortes pendientes")
        for cohort in market["pending"]:
            _require(
                type(cohort) is dict and type(cohort.get("quantiles")) is list,
                "Cohorte incompatible",
            )
            quantiles = cohort["quantiles"]
            rows += len(quantiles)
            _require(
                0 < len(quantiles) and rows <= max_rows, "Se supera el límite de filas pendientes"
            )
            _require(
                all(type(row) is list and len(row) == len(LEVELS) for row in quantiles),
                "Los cuantiles pendientes no tienen cinco columnas",
            )
            for row in quantiles:
                for value in row:
                    real(value)
    restored = copy.deepcopy(payload)
    for market in restored["markets"].values():
        for cohort in market["pending"]:
            cohort["quantiles"] = np.asarray(cohort["quantiles"], dtype=np.float64)
    try:
        return OnlineConformal.restore(restored)
    except (KeyError, TypeError, IndexError, OverflowError) as error:
        raise ValueError("El estado del calibrador contiene campos incompatibles") from error


def validate_cohorts(cohorts, calibrator, clock, max_rows):
    _require(
        type(cohorts) is list and len(cohorts) <= 8 * MAX_PENDING, "Cola de feedback incompatible"
    )
    if clock is not None:
        clock = _instant(clock, "El reloj del feedback")
    pending = calibrator.export()["markets"]
    expected = {}
    for group, market in pending.items():
        for field in ("clock", "last_emitted"):
            if market[field] is not None:
                moment = _instant(market[field], "El reloj del mercado")
                _require(clock is not None and moment <= clock, "El reloj del feedback retrocede")
        for item in market["pending"]:
            at = _instant(item["prediction_at"], "La decisión pendiente")
            due = _instant(item["matures_at"], "La madurez pendiente")
            _require(due > at, "La madurez debe ser posterior a la decisión")
            expected[group, at] = (due, len(item["quantiles"]))
    previous, total, actual = None, 0, set()
    for item in cohorts:
        _require(
            type(item) is dict and set(item) == {"group", "prediction_at", "rows"},
            "Cohorte de feedback incompatible",
        )
        group, at = identifier(item["group"]), _instant(item["prediction_at"], "La decisión")
        key = group, at
        order = at, group
        _require(previous is None or order > previous, "Las cohortes no están en orden canónico")
        previous = order
        _require(
            key in expected and type(item["rows"]) is list,
            "El feedback no corresponde al calibrador",
        )
        rows = item["rows"]
        total += len(rows)
        _require(
            total <= max_rows and len(rows) == expected[key][1], "Filas pendientes incompatibles"
        )
        names, maturities = [], []
        for row in rows:
            _require(type(row) is dict and set(row) == ROW_FIELDS, "Fila de feedback incompatible")
            names.append(identifier(row["sample_id"]))
            due = _instant(row["matures_at"], "La madurez de fila")
            _require(due > at, "La etiqueta debe madurar después de la decisión")
            maturities.append(due)
            if row["available_at"] is None:
                _require(
                    all(row[field] is None for field in ("value", "status", "evidence_sha256")),
                    "Una fila pendiente contiene un resultado sin disponibilidad",
                )
            else:
                _require(clock is not None, "El feedback recibido necesita un reloj")
                validate_feedback(row, at, clock)
        _require(names == sorted(set(names)), "Las identidades no son únicas y canónicas")
        _require(max(maturities) == expected[key][0], "La madurez de cohorte no conserva sus filas")
        actual.add(key)
    _require(actual == set(expected), "Falta una cohorte pendiente del calibrador")


def restore_payload(payload):
    _require(
        type(payload) is dict and set(payload) == FIELDS,
        "El estado de feedback no conserva sus campos",
    )
    _require(
        payload["schema_version"] == 1
        and type(payload["schema_version"]) is int
        and payload["kind"] == KIND
        and payload["policy"] == POLICY,
        "El estado pertenece a otra versión o política de feedback",
    )
    limits(payload["max_rows"], payload["max_bytes"])
    calibrator = unpack_calibrator(payload["calibrator"], payload["max_rows"])
    validate_cohorts(payload["cohorts"], calibrator, payload["clock"], payload["max_rows"])
    sealed = seal({k: v for k, v in payload.items() if k != "state_sha256"})
    _require(
        sealed["state_sha256"] == payload["state_sha256"], "La huella del feedback no coincide"
    )
    return calibrator
