"""Calibración conformal en línea por mercado con etiquetas maduras (PT2).

Sigue el cuantil de la puntuación CQR con el seguimiento de cuantiles de Angelopoulos,
Candès y Tibshirani (2023), del que ACI (Gibbs y Candès, 2021) es un caso particular.
Para cada mercado g y cada intervalo central de cobertura nominal 1 − a, la corrección
vigente Q se aplica al emitir una cohorte, [q_bajo − Q, q_alto + Q], con la misma regla de
orden que la CQR estática, y queda guardada con esa cohorte. Cuando su etiqueta madura,

    E_i = max(q_bajo,i − y_i, y_i − q_alto,i),   err_i = 1{E_i > Q emitida},
    Q ← Q + γ (media de err_i en la cohorte − a).

La corrección inicial es la CQR estática ajustada en la partición de calibración y la tasa
es γ = κ · B̂, con B̂ la mayor puntuación absoluta de calibración del mercado y el intervalo.
El artículo escala la tasa con la mayor puntuación de una ventana móvil. Aquí la ventana es
la partición de calibración y queda fija, de modo que γ es constante y la cota de cobertura
a largo plazo de su Proposición 1 se aplica sin cambios. Con κ = 0 la corrección no se mueve
y cada emisión coincide bit a bit con la CQR estática.

Los errores se calculan con la corrección que se emitió para la cohorte, no con la vigente
al madurar. Con un retraso de hasta d cohortes pendientes y puntuaciones en [−b, b], la
media de (err − a) sobre M cohortes maduras queda acotada por (2b + γ(1 + d)) / (γM). Esa
adaptación a cohortes con retraso es una derivación propia y su demostración está en
`docs/engineering/online-conformal-calibration.md`. La garantía es marginal y de largo
plazo, no condicional por activo ni por sesión, y no sustituye la ausencia de
intercambiabilidad que ya declara la CQR estática.

El estado de cada mercado es independiente. Las operaciones de un mercado deben llegar
con un reloj que no retrocede, cada cohorte madura una sola vez y en el orden en que se
emitió, y una operación rechazada no cambia nada.
"""

import copy
import hashlib
import json
import math
from fractions import Fraction

import numpy as np

from mars_titan.calibration import conformal_quantiles as cqr
from mars_titan.environments.cohorts import FINAL_TEST_START_US

KIND = "online_conformal_quantile_calibration"
METHOD = "cqr_score_quantile_tracking_matured_cohorts_v1"
RATE_SCALE = "max_abs_calibration_score"
SCHEMA_VERSION = 1
# Con retraso de una sesión basta una cohorte pendiente. El límite deja margen para
# horizontes de varias semanas sin que la cola crezca sin control si nunca maduran.
MAX_PENDING = 64
_HEADER = (
    "schema_version",
    "kind",
    "method",
    "rate_scale",
    "levels",
    "nominals",
    "rate_fraction",
    "max_pending",
    "static_calibrator_sha256",
)
_INTERVAL = ("correction", "scale", "rate", "target_miscoverage", "error_sum")
_MARKET = ("intervals", "clock", "last_emitted", "matured", "pending")
_CLOCKS = ("clock", "last_emitted")
_PENDING = ("prediction_at", "matures_at", "corrections", "quantiles", "quantiles_sha256")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _instant(value, name):
    _require(
        isinstance(value, (int, np.integer)) and not isinstance(value, bool),
        f"{name} debe ser un entero de microsegundos UTC",
    )
    value = int(value)
    # El test final de 2024 queda cerrado también para la calibración en línea.
    _require(0 <= value < FINAL_TEST_START_US, f"{name} está fuera del periodo de desarrollo")
    return value


def _array_sha256(array):
    digest = hashlib.sha256(str(array.shape).encode())
    digest.update(np.ascontiguousarray(array, dtype="<f8").tobytes())
    return digest.hexdigest()


def _scores(target, quantiles, lower, upper):
    return np.maximum(quantiles[:, lower] - target, target - quantiles[:, upper])


def _target_miscoverage(nominal):
    # 1 − 0,8 en coma flotante da 0,19999999999999996. La fracción exacta evita ese sesgo.
    return float(1 - Fraction(str(nominal)))


class OnlineConformal:
    """Correcciones CQR que siguen en línea la cobertura de cada mercado.

    Se construye con :meth:`start` a partir de la partición de calibración o con
    :meth:`restore` desde un estado exportado. :meth:`emit` aplica la corrección vigente sin
    ver etiquetas y :meth:`mature` actualiza la corrección con una cohorte ya madura.
    """

    def __init__(self, header, markets):
        self._header = header
        self._markets = markets
        self._levels, self._pairs, _ = cqr.interval_pairs(header["levels"], header["nominals"])

    @classmethod
    def start(
        cls,
        target,
        quantiles,
        groups,
        *,
        levels,
        nominals,
        min_rows,
        rate_fraction,
        max_pending=MAX_PENDING,
    ):
        """Partir de la CQR estática de calibración con tasa γ = κ · B̂ por mercado e intervalo."""
        _require(
            type(rate_fraction) in (int, float)
            and math.isfinite(rate_fraction)
            and 0 <= rate_fraction <= 1,
            "La fracción de la tasa κ debe estar en [0, 1]",
        )
        _require(
            type(max_pending) is int and 1 <= max_pending <= MAX_PENDING,
            f"El máximo de cohortes pendientes debe estar entre 1 y {MAX_PENDING}",
        )
        record = cqr.fit_conformal_quantiles(
            target, quantiles, groups, levels=levels, nominals=nominals, min_rows=min_rows
        )
        missing = cqr.undefined_groups(record, groups)
        _require(not missing, f"La CQR estática no define una corrección inicial para {missing}")
        levels, pairs, _ = cqr.interval_pairs(levels, nominals)
        target = np.asarray(target, dtype=np.float64)
        quantiles = np.asarray(quantiles, dtype=np.float64)
        groups = np.asarray(groups)
        markets = {}
        for name, entry in record["groups"].items():
            selected = groups == name
            intervals = {}
            for nominal, lower, upper in pairs:
                key = f"{nominal:g}"
                scale = float(
                    np.abs(_scores(target[selected], quantiles[selected], lower, upper)).max()
                )
                _require(
                    rate_fraction == 0 or scale > 0,
                    f"Las puntuaciones de calibración de {name} son nulas y no fijan la escala",
                )
                intervals[key] = dict(
                    correction=entry["intervals"][key]["correction"],
                    scale=scale,
                    rate=float(rate_fraction) * scale,
                    target_miscoverage=_target_miscoverage(nominal),
                    error_sum=0.0,
                )
            markets[name] = dict(
                intervals=intervals, clock=None, last_emitted=None, matured=0, pending=[]
            )
        header = dict(
            schema_version=SCHEMA_VERSION,
            kind=KIND,
            method=METHOD,
            rate_scale=RATE_SCALE,
            levels=list(levels),
            nominals=[nominal for nominal, _, _ in pairs],
            rate_fraction=float(rate_fraction),
            max_pending=max_pending,
            static_calibrator_sha256=cqr.calibrator_sha256(record),
        )
        return cls(header, markets)

    def _market(self, group):
        _require(
            isinstance(group, str) and group in self._markets,
            f"El mercado {group!r} no tiene calibración en línea",
        )
        return self._markets[group]

    def _in_force(self, group):
        # Registro con el formato de la CQR estática y la corrección vigente de un solo
        # mercado. Así la emisión reutiliza la misma aplicación y la misma regla de orden.
        intervals = self._markets[group]["intervals"]
        return dict(
            schema_version=1,
            kind=cqr.KIND,
            method=cqr.METHOD,
            order_rule=cqr.ORDER_RULE,
            coverage_guaranteed=False,
            levels=self._header["levels"],
            nominals=self._header["nominals"],
            groups={
                group: dict(
                    intervals={
                        key: dict(correction=interval["correction"])
                        for key, interval in intervals.items()
                    }
                )
            },
        )

    def corrections(self, group):
        """Corrección vigente de cada intervalo del mercado, sin modificar el estado."""
        market = self._market(group)
        return {key: interval["correction"] for key, interval in market["intervals"].items()}

    def emit(self, group, prediction_at, quantiles, matures_at):
        """Emitir una cohorte con la corrección vigente y guardarla hasta su madurez.

        Devuelve los cuantiles calibrados y el registro de la emisión con la corrección
        aplicada a cada intervalo, que debe acompañar a la predicción emitida.
        """
        market = self._market(group)
        prediction_at = _instant(prediction_at, "La fecha de predicción")
        matures_at = _instant(matures_at, "La madurez de la etiqueta")
        _require(matures_at > prediction_at, "La etiqueta debe madurar después de la predicción")
        _require(
            market["last_emitted"] is None or prediction_at > market["last_emitted"],
            "Cada mercado emite una cohorte por instante y en orden creciente",
        )
        _require(
            market["clock"] is None or prediction_at >= market["clock"],
            "La emisión no puede ser anterior a una actualización ya aplicada",
        )
        _require(
            len(market["pending"]) < self._header["max_pending"],
            "Se ha alcanzado el máximo de cohortes pendientes de madurar",
        )
        quantiles = np.asarray(quantiles)
        rows = len(quantiles) if quantiles.ndim == 2 else 0
        calibrated, adjusted = cqr.apply_conformal_quantiles(
            self._in_force(group), quantiles, np.full(rows, group)
        )
        raw = np.array(quantiles, dtype=np.float64, copy=True)
        raw.setflags(write=False)
        corrections = self.corrections(group)
        market["pending"].append(
            dict(
                prediction_at=prediction_at,
                matures_at=matures_at,
                corrections=corrections,
                quantiles=raw,
                quantiles_sha256=_array_sha256(raw),
            )
        )
        market["last_emitted"] = market["clock"] = prediction_at
        emission = dict(
            group=group,
            prediction_at=prediction_at,
            matures_at=matures_at,
            corrections=dict(corrections),
            order_adjusted_rows=adjusted,
        )
        return calibrated, emission

    def mature(self, group, prediction_at, target, *, at):
        """Actualizar la corrección con la cohorte pendiente más antigua del mercado.

        Los errores usan los cuantiles base y la corrección guardados al emitir. Toda la
        comprobación ocurre antes de escribir, así que un rechazo deja el estado intacto.
        """
        market = self._market(group)
        prediction_at = _instant(prediction_at, "La fecha de predicción")
        at = _instant(at, "El instante de actualización")
        _require(market["pending"], f"{group} no tiene cohortes pendientes")
        oldest = market["pending"][0]
        _require(
            prediction_at >= oldest["prediction_at"],
            "La cohorte ya maduró o nunca se emitió en este mercado",
        )
        _require(
            prediction_at == oldest["prediction_at"],
            "La cohorte madura fuera de orden. Antes debe madurar la emitida en "
            f"{oldest['prediction_at']}",
        )
        _require(at >= oldest["matures_at"], "La etiqueta de la cohorte todavía no es madura")
        _require(at >= market["clock"], "El reloj del mercado no puede retroceder")
        raw = oldest["quantiles"]
        target = np.asarray(target)
        _require(
            target.shape == (len(raw),) and target.dtype.kind == "f",
            "Se necesita una etiqueta real por fila de la cohorte emitida",
        )
        target = target.astype(np.float64)
        _require(np.isfinite(target).all(), "Las etiquetas maduras contienen NaN o infinitos")
        updates = {}
        for nominal, lower, upper in self._pairs:
            key = f"{nominal:g}"
            interval = market["intervals"][key]
            emitted = oldest["corrections"][key]
            errors = _scores(target, raw, lower, upper) > emitted
            miscoverage = float(np.mean(errors))
            correction = interval["correction"] + interval["rate"] * (
                miscoverage - interval["target_miscoverage"]
            )
            _require(math.isfinite(correction), "La corrección actualizada no es finita")
            updates[key] = dict(
                emitted=emitted,
                miscoverage=miscoverage,
                previous=interval["correction"],
                correction=correction,
            )
        for key, update in updates.items():
            interval = market["intervals"][key]
            interval["correction"] = update["correction"]
            interval["error_sum"] += update["miscoverage"]
        market["pending"].pop(0)
        market["matured"] += 1
        market["clock"] = at
        return dict(group=group, prediction_at=prediction_at, at=at, intervals=updates)

    def pending(self, group):
        """Instantes de predicción y de madurez de las cohortes pendientes del mercado."""
        return [
            (cohort["prediction_at"], cohort["matures_at"])
            for cohort in self._market(group)["pending"]
        ]

    def coverage_gap(self, group):
        """Media de (err − a) sobre las cohortes maduras de cada intervalo, o None sin ninguna."""
        market = self._market(group)
        if market["matured"] == 0:
            return {key: None for key in market["intervals"]}
        return {
            key: interval["error_sum"] / market["matured"] - interval["target_miscoverage"]
            for key, interval in market["intervals"].items()
        }

    def export(self):
        """Estado completo con huella. Los cuantiles pendientes se copian como float64."""
        markets = copy.deepcopy(
            {
                name: {
                    **{field: market[field] for field in _MARKET if field != "pending"},
                    "pending": [
                        {field: cohort[field] for field in _PENDING if field != "quantiles"}
                        for cohort in market["pending"]
                    ],
                }
                for name, market in self._markets.items()
            }
        )
        arrays = {
            name: [np.array(cohort["quantiles"], copy=True) for cohort in market["pending"]]
            for name, market in self._markets.items()
        }
        for name, market in markets.items():
            for cohort, array in zip(market["pending"], arrays[name], strict=True):
                cohort["quantiles"] = array
        payload = dict(header=dict(self._header), markets=markets)
        payload["state_sha256"] = _state_sha256(payload)
        return payload

    def state_sha256(self):
        """Huella del estado completo, igual a la del estado exportado."""
        return self.export()["state_sha256"]

    @classmethod
    def restore(cls, payload):
        """Reconstruir el calibrador después de validar todos los campos y su huella."""
        _require(
            isinstance(payload, dict) and set(payload) == {"header", "markets", "state_sha256"},
            "El estado exportado no conserva sus campos",
        )
        header, markets = payload["header"], payload["markets"]
        _require(
            isinstance(header, dict)
            and set(header) == set(_HEADER)
            and header["schema_version"] == SCHEMA_VERSION
            and header["kind"] == KIND
            and header["method"] == METHOD
            and header["rate_scale"] == RATE_SCALE,
            "El estado pertenece a otro método o versión de calibración",
        )
        _, pairs, _ = cqr.interval_pairs(header["levels"], header["nominals"])
        keys = {f"{nominal:g}" for nominal, _, _ in pairs}
        _require(
            type(header["max_pending"]) is int and 1 <= header["max_pending"] <= MAX_PENDING,
            "El máximo de cohortes pendientes del estado no es válido",
        )
        _require(isinstance(markets, dict) and markets, "El estado no contiene mercados")
        for name, market in markets.items():
            _require(
                isinstance(name, str) and isinstance(market, dict) and set(market) == set(_MARKET),
                "Un mercado del estado no conserva sus campos",
            )
            _require(
                all(market[field] is None or type(market[field]) is int for field in _CLOCKS)
                and type(market["matured"]) is int
                and market["matured"] >= 0
                and len(market["pending"]) <= header["max_pending"]
                and (
                    not market["pending"]
                    or market["pending"][-1]["prediction_at"] == market["last_emitted"]
                ),
                "El reloj, el recuento o la cola de un mercado no son coherentes",
            )
            _require(
                set(market["intervals"]) == keys
                and all(
                    set(interval) == set(_INTERVAL)
                    and all(math.isfinite(interval[field]) for field in _INTERVAL)
                    for interval in market["intervals"].values()
                ),
                "Los intervalos de un mercado no son válidos",
            )
            previous = None
            for cohort in market["pending"]:
                _require(
                    set(cohort) == set(_PENDING)
                    and set(cohort["corrections"]) == keys
                    and isinstance(cohort["quantiles"], np.ndarray)
                    and cohort["quantiles"].dtype == np.float64
                    and cohort["quantiles"].ndim == 2
                    and _array_sha256(cohort["quantiles"]) == cohort["quantiles_sha256"]
                    and (previous is None or cohort["prediction_at"] > previous),
                    "Una cohorte pendiente no conserva sus cuantiles, su corrección o su orden",
                )
                previous = cohort["prediction_at"]
        _require(
            _state_sha256(payload) == payload["state_sha256"],
            "El estado exportado no coincide con su huella",
        )
        restored = copy.deepcopy({"header": header, "markets": markets})
        for market in restored["markets"].values():
            for cohort in market["pending"]:
                cohort["quantiles"].setflags(write=False)
        return cls(restored["header"], restored["markets"])


def _state_sha256(payload):
    scalars = {
        "header": payload["header"],
        "markets": {
            name: {
                **{field: market[field] for field in _MARKET if field != "pending"},
                "pending": [
                    {field: cohort[field] for field in _PENDING if field != "quantiles"}
                    for cohort in market["pending"]
                ],
            }
            for name, market in payload["markets"].items()
        },
    }
    # La huella de cada matriz pendiente ya está entre los escalares. Se recalcula aquí
    # para que un cambio en los cuantiles no pase inadvertido aunque nadie la actualice.
    digest = hashlib.sha256(_canonical(scalars).encode())
    for name in sorted(payload["markets"]):
        for cohort in payload["markets"][name]["pending"]:
            digest.update(_array_sha256(cohort["quantiles"]).encode())
    return digest.hexdigest()


def replay_online_conformal(calibrator, quantiles, groups, prediction_at, target, matures_at):
    """Recorrer un panel en orden temporal con la calibración en línea.

    Cada cohorte es el conjunto de filas de un mercado en un instante de predicción. Antes de
    emitirla se aplican, en el orden de emisión, las cohortes del mismo mercado cuyas
    etiquetas ya son maduras en ese instante. La madurez de una cohorte es la mayor de sus
    filas. Las etiquetas solo llegan al calibrador al madurar su cohorte. Devuelve los
    cuantiles emitidos en el orden original de las filas y los registros de emisión y de
    actualización. Modifica el calibrador recibido.
    """
    quantiles = np.asarray(quantiles)
    groups = np.asarray(groups)
    prediction_at = np.asarray(prediction_at)
    target = np.asarray(target)
    matures_at = np.asarray(matures_at)
    rows = len(groups)
    _require(
        rows >= 1
        and groups.shape == (rows,)
        and groups.dtype.kind == "U"
        and quantiles.ndim == 2
        and len(quantiles) == rows
        and all(array.shape == (rows,) for array in (prediction_at, target, matures_at))
        and prediction_at.dtype.kind in "iu"
        and matures_at.dtype.kind in "iu",
        "El panel necesita cuantiles, mercados, instantes, objetivos y madurez alineados",
    )
    emitted = np.empty(quantiles.shape, dtype=np.float64)
    keys = sorted(set(zip(prediction_at.tolist(), groups.tolist(), strict=True)))
    members = {}
    for index, key in enumerate(zip(prediction_at.tolist(), groups.tolist(), strict=True)):
        members.setdefault(key, []).append(index)
    emissions, updates = [], []
    for at, group in keys:
        for cohort_at, ready_at in calibrator.pending(group):
            if ready_at > at:
                break
            selected = members.get((cohort_at, group))
            _require(selected is not None, "Una cohorte pendiente no pertenece al panel")
            updates.append(calibrator.mature(group, cohort_at, target[selected], at=at))
        selected = members[(at, group)]
        calibrated, emission = calibrator.emit(
            group, at, quantiles[selected], int(matures_at[selected].max())
        )
        emitted[selected] = calibrated
        emissions.append(emission)
    return emitted, emissions, updates
