"""Memoria por activo del residuo madurado, etapa posterior sobre las emisiones (H1-b).

Es la memoria del laboratorio tabular del cuaderno de investigación (H1-a y H1-b) convertida
en una pieza con contrato causal, estado exportable y guardas. No entrena nada: lee la
predicción emitida por un modelo y las etiquetas que ya han madurado, y emite una puntuación
por sesión que solo ordena activos.

En cada sesión (mercado e instante de decisión) con predicciones p de sus activos:

    z_i = (p_i − media(p)) / desv(p),            0 si la desviación es cero,
    m_a = S_a / W_a,                              0 si W_a = 0 (activo sin historia),
    u_i = (m_a(i) − media(m)) / desv(m),          0 si la desviación es cero,
    emisión_i(κ) = z_i + κ · u_i.

Cuando madura la etiqueta y de una cohorte emitida, con g los rangos gaussianos centrados de
y en su sesión (rangos medios, Φ⁻¹((r − 1/2)/n)) y la z guardada al emitir:

    e = g − (z·g / n) z                           residuo de g fuera de la dirección de z,
    v = e (residual), g (label) o signo(e) (sign),
    S ← d·S, W ← d·W con d = 2^(−1/h) en todo el mercado, y después
    S_a += v_a, W_a += 1 para los activos de la cohorte.

h es la semivida en cohortes maduras del mercado. Con una etiqueta que madura en la decisión
siguiente, la emisión coincide con la del laboratorio (retraso de una sesión), que fijó la
variante del residuo con h = 21 y κ = 0,1 con el IC de su calibración.

Contrato causal: una cohorte se emite sin ver su etiqueta; madura una sola vez, en el orden
de emisión de su mercado y en un instante no anterior a su madurez; el reloj de cada mercado
no retrocede, y una operación rechazada no cambia el estado. Los mercados no comparten
memoria ni decaimiento. Una predicción o una etiqueta no finita se rechaza con su activo y su
instante, sin sustituirla. El test final de 2024 queda cerrado: ningún instante de decisión
ni de actualización puede llegar a él.

El placebo (`read_map`) lee en cada fila la memoria de otro activo fijado de antemano;
conserva la distribución de las lecturas y rompe su relación con el activo.
"""

import copy
import hashlib
import json
import math
from dataclasses import dataclass

import numpy as np
from scipy.special import ndtri

from mars_titan.environments.cohorts import FINAL_TEST_START_US

KIND = "asset_residual_memory"
METHOD = "asset_ewma_matured_rank_residual_v1"
SCHEMA_VERSION = 1
VALUES = ("residual", "label", "sign")
# Con retraso de una sesión basta una cohorte pendiente; el límite admite horizontes largos
# sin que la cola crezca sin control si una etiqueta no llega nunca.
MAX_PENDING = 64
MAX_ASSETS = 100_000


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _is_instant(value):
    """Un instante de desarrollo: entero (no booleano) de microsegundos UTC anterior a 2024."""
    return (
        isinstance(value, (int, np.integer))
        and not isinstance(value, bool)
        and 0 <= int(value) < FINAL_TEST_START_US
    )


def _instant(value, name):
    _require(
        isinstance(value, (int, np.integer)) and not isinstance(value, bool),
        f"{name} debe ser un entero de microsegundos UTC",
    )
    value = int(value)
    _require(0 <= value < FINAL_TEST_START_US, f"{name} está fuera del periodo de desarrollo")
    return value


@dataclass(frozen=True)
class AssetMemoryConfig:
    """Semivida en cohortes maduras del mercado y valor que se guarda."""

    half_life: float
    value: str = "residual"

    def __post_init__(self):
        _require(
            type(self.half_life) in (int, float)
            and math.isfinite(self.half_life)
            and self.half_life > 0,
            "La semivida de la memoria por activo es un número finito y positivo",
        )
        _require(self.value in VALUES, f"El valor guardado es uno de {VALUES}")

    @property
    def decay(self):
        return 0.5 ** (1.0 / float(self.half_life))

    def identity(self):
        return dict(half_life=float(self.half_life), value=self.value)


def zscore(values):
    """Puntuación típica de una sesión con la desviación poblacional; ceros si es constante."""
    spread = values.std()
    return (values - values.mean()) / spread if spread > 0 else values * 0.0


def average_ranks(values):
    """Rangos medios de 1 a n; los empates comparten su rango medio."""
    order = np.argsort(values, kind="mergesort")
    ordered = values[order]
    first = np.empty(len(values), dtype=bool)
    first[:1] = True
    first[1:] = ordered[1:] != ordered[:-1]
    starts = np.flatnonzero(first)
    ends = np.r_[starts[1:], len(values)]
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = ((starts + 1 + ends) / 2)[np.cumsum(first) - 1]
    return ranks


def gaussian_ranks(values):
    """Rangos gaussianos centrados de una sesión: Φ⁻¹((r − 1/2)/n) menos su media."""
    g = ndtri((average_ranks(values) - 0.5) / len(values))
    return g - g.mean()


def stored_value(z, target, value):
    """Valor que guarda la memoria para una cohorte madura, con la z emitida."""
    g = gaussian_ranks(target)
    if value == "label":
        return g
    e = g - (float(z @ g) / len(z)) * z
    return np.sign(e) if value == "sign" else e


def _finite(values, assets, at, name):
    bad = np.flatnonzero(~np.isfinite(values))
    _require(
        bad.size == 0,
        f"{name} no finita en {bad.size} filas; la primera es {assets[bad[0]] if bad.size else ''}"
        f" en el instante {at}",
    )


def _array_sha256(array):
    digest = hashlib.sha256(str(array.shape).encode())
    digest.update(np.ascontiguousarray(array, dtype="<f8").tobytes())
    return digest.hexdigest()


class _Market:
    __slots__ = ("index", "assets", "memory", "weight", "clock", "matured", "pending")

    def __init__(self):
        self.index, self.assets = {}, []
        self.memory = np.zeros(0, dtype=np.float64)
        self.weight = np.zeros(0, dtype=np.float64)
        self.clock, self.matured, self.pending = None, 0, []

    def positions(self, assets, *, grow):
        """Posición de cada activo; con grow los nuevos entran con memoria y peso nulos."""
        missing = [a for a in dict.fromkeys(assets.tolist()) if a not in self.index]
        if missing and grow:
            _require(
                len(self.assets) + len(missing) <= MAX_ASSETS,
                "La memoria por activo supera el máximo de activos por mercado",
            )
            for asset in missing:
                self.index[asset] = len(self.assets)
                self.assets.append(asset)
            self.memory = np.concatenate([self.memory, np.zeros(len(missing))])
            self.weight = np.concatenate([self.weight, np.zeros(len(missing))])
        return np.array([self.index.get(a, -1) for a in assets.tolist()], dtype=np.int64)


class AssetResidualMemory:
    """Memoria por activo de un modelo, con un estado independiente por mercado."""

    def __init__(self, config):
        _require(isinstance(config, AssetMemoryConfig), "La configuración es AssetMemoryConfig")
        self.config = config
        self._markets = {}

    def _market(self, market, *, create=False):
        _require(isinstance(market, str) and market, "El mercado es un texto no vacío")
        if market not in self._markets:
            _require(create, f"El mercado {market!r} no tiene memoria")
            self._markets[market] = _Market()
        return self._markets[market]

    def read(self, market, assets):
        """Media y peso de la memoria de cada activo; cero para un activo sin historia."""
        state = self._markets.get(market)
        assets = np.asarray(assets)
        if state is None:
            return np.zeros(len(assets)), np.zeros(len(assets))
        positions = state.positions(assets, grow=False)
        known = positions >= 0
        weight = np.where(known, state.weight[np.maximum(positions, 0)], 0.0)
        total = np.where(known, state.memory[np.maximum(positions, 0)], 0.0)
        # S_a/W_a exacta para todo peso positivo: un activo ausente muchas semividas conserva su
        # media (el divisor mínimo de 1e-12 la llevaba hacia cero; H03 de la auditoría).
        mean = np.divide(total, weight, out=np.zeros(len(assets)), where=weight > 0)
        return mean, weight

    def matured(self, market):
        state = self._markets.get(market)
        return 0 if state is None else state.matured

    def emit(self, market, prediction_at, assets, prediction, *, kappas, matures_at, read_map=None):
        """Puntuaciones de una cohorte para cada κ, sin ver su etiqueta.

        Devuelve {κ: emisión}. La cohorte queda pendiente con su z hasta que madure.
        """
        prediction_at = _instant(prediction_at, "La fecha de decisión")
        _require(
            isinstance(matures_at, (int, np.integer)) and not isinstance(matures_at, bool),
            "La madurez es un entero de microsegundos UTC",
        )
        _require(int(matures_at) > prediction_at, "Cada etiqueta madura después de su decisión")
        assets = np.asarray(assets)
        prediction = np.asarray(prediction)
        _require(
            assets.ndim == 1
            and assets.dtype.kind == "U"
            and len(assets) >= 1
            and prediction.shape == assets.shape
            and prediction.dtype.kind == "f",
            "Una cohorte necesita activos de texto y una predicción real por activo",
        )
        _require(
            len(set(assets.tolist())) == len(assets),
            f"Hay un activo repetido en la sesión {market} {prediction_at}",
        )
        _finite(prediction, assets, prediction_at, "La predicción")
        kappas = tuple(float(k) for k in kappas)
        _require(
            kappas
            and all(math.isfinite(k) and k >= 0 for k in kappas)
            and len(set(kappas)) == len(kappas),
            "Los pesos κ son números finitos, no negativos y distintos",
        )
        state = self._markets.get(market)
        if state is not None:
            _require(
                state.clock is None or prediction_at >= state.clock,
                "El reloj del mercado no puede retroceder",
            )
            _require(
                not state.pending or prediction_at > state.pending[-1]["prediction_at"],
                "Las cohortes de un mercado se emiten en orden y una sola vez",
            )
            _require(len(state.pending) < MAX_PENDING, "Demasiadas cohortes pendientes")
        if read_map is None:
            read_assets = assets
        else:
            _require(
                all(a in read_map for a in assets.tolist()),
                "El placebo necesita un activo de lectura para cada activo de la sesión",
            )
            read_assets = np.array([read_map[a] for a in assets.tolist()])
        mean, _ = self.read(market, read_assets)
        z = zscore(prediction.astype(np.float64))
        u = zscore(mean)
        emitted = {kappa: z + kappa * u for kappa in kappas}
        # El mercado nuevo se registra solo si su cohorte cabe: un rechazo no deja rastro.
        fresh = state is None
        if fresh:
            _require(isinstance(market, str) and market, "El mercado es un texto no vacío")
            state = _Market()
        state.positions(assets, grow=True)
        if fresh:
            self._markets[market] = state
        z = np.array(z, copy=True)
        z.setflags(write=False)
        state.pending.append(
            dict(
                prediction_at=prediction_at,
                matures_at=int(matures_at),
                assets=np.array(assets, copy=True),
                z=z,
            )
        )
        state.clock = prediction_at
        return emitted

    def pending(self, market):
        """Instantes de decisión y de madurez de las cohortes pendientes del mercado."""
        state = self._markets.get(market)
        if state is None:
            return []
        return [(c["prediction_at"], c["matures_at"]) for c in state.pending]

    def pending_assets(self, market, prediction_at):
        """Activos de una cohorte pendiente, en el orden en que se emitió."""
        state = self._market(market)
        for cohort in state.pending:
            if cohort["prediction_at"] == prediction_at:
                return np.array(cohort["assets"], copy=True)
        raise ValueError(f"No hay ninguna cohorte pendiente de {market} en {prediction_at}")

    def mature(self, market, prediction_at, target, *, at):
        """Escribir la cohorte pendiente más antigua del mercado con su etiqueta madura.

        `target` va en el orden de los activos de la cohorte emitida. Todo se comprueba
        antes de escribir, así que un rechazo deja el estado intacto.
        """
        state = self._market(market)
        prediction_at = _instant(prediction_at, "La fecha de decisión")
        at = _instant(at, "El instante de actualización")
        _require(state.pending, f"{market} no tiene cohortes pendientes")
        oldest = state.pending[0]
        _require(
            prediction_at == oldest["prediction_at"],
            "La cohorte madura fuera de orden: antes debe madurar la emitida en "
            f"{oldest['prediction_at']}",
        )
        _require(at >= oldest["matures_at"], "La etiqueta de la cohorte todavía no es madura")
        _require(
            state.clock is None or at >= state.clock, "El reloj del mercado no puede retroceder"
        )
        target = np.asarray(target)
        _require(
            target.shape == oldest["assets"].shape and target.dtype.kind == "f",
            "Se necesita una etiqueta real por activo de la cohorte emitida",
        )
        target = target.astype(np.float64)
        _finite(target, oldest["assets"], prediction_at, "La etiqueta madura")
        values = stored_value(oldest["z"], target, self.config.value)
        positions = state.positions(oldest["assets"], grow=False)
        _require(bool(np.all(positions >= 0)), "Un activo de la cohorte no tiene posición")
        decay = self.config.decay
        state.memory *= decay
        state.weight *= decay
        state.memory[positions] += values
        state.weight[positions] += 1.0
        state.pending.pop(0)
        state.matured += 1
        state.clock = at

    def export(self):
        """Estado completo con huella. Las matrices se copian en FP64."""
        markets = {}
        for name in sorted(self._markets):
            state = self._markets[name]
            markets[name] = dict(
                assets=list(state.assets),
                memory=np.array(state.memory, dtype=np.float64, copy=True),
                weight=np.array(state.weight, dtype=np.float64, copy=True),
                clock=state.clock,
                matured=state.matured,
                pending=[
                    dict(
                        prediction_at=c["prediction_at"],
                        matures_at=c["matures_at"],
                        assets=c["assets"].tolist(),
                        z=np.array(c["z"], dtype=np.float64, copy=True),
                    )
                    for c in state.pending
                ],
            )
        payload = dict(
            schema_version=SCHEMA_VERSION,
            kind=KIND,
            method=METHOD,
            config=self.config.identity(),
            markets=markets,
        )
        payload["state_sha256"] = _state_sha256(payload)
        return payload

    def state_sha256(self):
        return self.export()["state_sha256"]

    @classmethod
    def restore(cls, payload):
        """Reconstruir la memoria tras validar todos los campos y su huella."""
        _require(
            isinstance(payload, dict)
            and set(payload)
            == {"schema_version", "kind", "method", "config", "markets", "state_sha256"}
            and type(payload["schema_version"]) is int
            and payload["schema_version"] == SCHEMA_VERSION
            and payload["kind"] == KIND
            and payload["method"] == METHOD,
            "El estado pertenece a otra memoria o versión",
        )
        _require(
            _state_sha256(payload) == payload["state_sha256"],
            "El estado exportado no coincide con su huella",
        )
        memory = cls(AssetMemoryConfig(**payload["config"]))
        for name, saved in copy.deepcopy(payload["markets"]).items():
            state = memory._market(name, create=True)
            assets, clock, matured = saved["assets"], saved["clock"], saved["matured"]
            # La huella no basta: un estado firmado debe cumplir también los invariantes (H04).
            _require(
                len(set(assets)) == len(assets) <= MAX_ASSETS
                and all(isinstance(a, str) and a for a in assets)
                and saved["memory"].shape == saved["weight"].shape == (len(assets),)
                and np.isfinite(saved["memory"]).all()
                and np.isfinite(saved["weight"]).all()
                and np.all(saved["weight"] >= 0)
                and (clock is None or _is_instant(clock))
                and (clock is not None or not saved["pending"])
                and type(matured) is int
                and matured >= 0
                and len(saved["pending"]) <= MAX_PENDING,
                f"La memoria guardada de {name} no es coherente",
            )
            state.assets = list(assets)
            state.index = {asset: i for i, asset in enumerate(assets)}
            state.memory, state.weight = saved["memory"], saved["weight"]
            state.clock, state.matured = saved["clock"], saved["matured"]
            previous = None
            for cohort in saved["pending"]:
                z = np.array(cohort["z"], dtype=np.float64)
                decided, matures = cohort["prediction_at"], cohort["matures_at"]
                _require(
                    _is_instant(decided)
                    and isinstance(matures, (int, np.integer))
                    and not isinstance(matures, bool)
                    and matures > decided
                    and decided <= clock
                    and len(z) == len(cohort["assets"])
                    and np.isfinite(z).all()
                    and (previous is None or decided > previous)
                    and all(a in state.index for a in cohort["assets"]),
                    f"Una cohorte pendiente de {name} no es coherente",
                )
                z.setflags(write=False)
                state.pending.append(
                    dict(
                        prediction_at=cohort["prediction_at"],
                        matures_at=cohort["matures_at"],
                        assets=np.array(cohort["assets"]),
                        z=z,
                    )
                )
                previous = cohort["prediction_at"]
        return memory


def _state_sha256(payload):
    scalars = dict(
        schema_version=payload["schema_version"],
        kind=payload["kind"],
        method=payload["method"],
        config=payload["config"],
        markets={
            name: dict(
                assets=market["assets"],
                clock=market["clock"],
                matured=market["matured"],
                pending=[
                    dict(
                        prediction_at=c["prediction_at"],
                        matures_at=c["matures_at"],
                        assets=list(c["assets"]),
                    )
                    for c in market["pending"]
                ],
            )
            for name, market in payload["markets"].items()
        },
    )
    digest = hashlib.sha256(
        json.dumps(scalars, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    )
    for name in sorted(payload["markets"]):
        market = payload["markets"][name]
        digest.update(_array_sha256(np.asarray(market["memory"])).encode())
        digest.update(_array_sha256(np.asarray(market["weight"])).encode())
        for cohort in market["pending"]:
            digest.update(_array_sha256(np.asarray(cohort["z"])).encode())
    return digest.hexdigest()


def _cohorts(markets, prediction_at):
    """Filas de cada cohorte (mercado, instante) en orden de decisión y después de mercado."""
    order = np.lexsort((markets, prediction_at))
    keys = list(zip(prediction_at[order].tolist(), markets[order].tolist(), strict=True))
    bounds = [0] + [i for i in range(1, len(keys)) if keys[i] != keys[i - 1]] + [len(keys)]
    return [
        (keys[start][1], keys[start][0], order[start:end])
        for start, end in zip(bounds[:-1], bounds[1:], strict=True)
    ]


def _targets(sources, market, prediction_at, assets):
    """Etiquetas de una cohorte en el orden de sus activos emitidos, buscadas en las fuentes."""
    for source in sources:
        rows = source["lookup"].get((market, prediction_at))
        if rows is None:
            continue
        found = dict(zip(source["assets"][rows].tolist(), source["target"][rows], strict=True))
        _require(
            all(a in found for a in assets.tolist()),
            f"Falta la etiqueta de algún activo de la cohorte {market} {prediction_at}",
        )
        return np.array([found[a] for a in assets.tolist()], dtype=np.float64)
    raise ValueError(f"No se encuentran las etiquetas de la cohorte {market} {prediction_at}")


def _source(data):
    markets = np.asarray(data["markets"])
    prediction_at = np.asarray(data["prediction_at"])
    lookup = {(m, t): rows for m, t, rows in _cohorts(markets, prediction_at)}
    return dict(
        lookup=lookup, assets=np.asarray(data["assets"]), target=np.asarray(data["target"])
    )


def replay(
    memory,
    *,
    kappas,
    markets,
    assets,
    prediction_at,
    matures_at,
    prediction,
    target,
    read_map=None,
    flush=True,
    pending_targets=None,
):
    """Recorrer un panel en orden de decisión con la memoria por activo.

    Antes de emitir una cohorte se escriben, en orden de emisión, las cohortes de su mercado
    cuya madurez (la mayor de sus filas) no es posterior a la decisión. Las etiquetas solo
    llegan a la memoria al madurar. Con `flush` se escriben al final las cohortes que aún
    estén pendientes y maduren antes del test final; sin él quedan pendientes para una
    reanudación, cuyas etiquetas se buscan en `pending_targets`. Devuelve {κ: emisión} en
    el orden original de las filas. Modifica la memoria recibida.
    """
    markets, assets = np.asarray(markets), np.asarray(assets)
    prediction_at, matures_at = np.asarray(prediction_at), np.asarray(matures_at)
    prediction, target = np.asarray(prediction), np.asarray(target)
    rows = len(markets)
    _require(
        rows >= 1
        and all(a.shape == (rows,) for a in (assets, prediction_at, matures_at, prediction, target))
        and prediction_at.dtype.kind == "i"
        and matures_at.dtype.kind == "i",
        "El panel necesita columnas alineadas e instantes enteros",
    )
    _require(
        bool(np.all(matures_at > prediction_at)), "Cada etiqueta madura después de su decisión"
    )
    panel_columns = dict(markets=markets, assets=assets, prediction_at=prediction_at, target=target)
    sources = [_source(panel_columns)]
    if pending_targets is not None:
        sources.append(_source(pending_targets))
    kappas = tuple(float(k) for k in kappas)
    output = {kappa: np.full(rows, np.nan) for kappa in kappas}

    def settle(market, now):
        state = memory._markets.get(market)
        while state is not None and state.pending and state.pending[0]["matures_at"] <= now:
            cohort = state.pending[0]
            labels = _targets(sources, market, cohort["prediction_at"], cohort["assets"])
            memory.mature(market, cohort["prediction_at"], labels, at=now)

    for market, moment, chosen in _cohorts(markets, prediction_at):
        settle(market, moment)
        emitted = memory.emit(
            market,
            moment,
            assets[chosen],
            prediction[chosen],
            kappas=kappas,
            matures_at=int(matures_at[chosen].max()),
            read_map=read_map,
        )
        for kappa, values in emitted.items():
            output[kappa][chosen] = values
    if flush:
        for market in sorted(memory._markets):
            state = memory._markets[market]
            while state.pending and state.pending[0]["matures_at"] < FINAL_TEST_START_US:
                cohort = state.pending[0]
                settle(market, max(cohort["matures_at"], state.clock or 0))
    return output
