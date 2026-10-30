"""Puerta de régimen E4 `regime_evidence_gate_v1` sobre errores maduros (#750).

La puerta regula cuánto escribe la corrección madura de cada mercado. Para el mercado m y
una publicación con errores e_i = etiqueta − punto emitido, todos ya maduros:

    z_i = e_i / max(r_m, suelo)                    escala r_m anterior a la cohorte
    y   = media_i(z_i²) − 1                        exceso de varianza de la cohorte
    S   = min(h, max(0, S_anterior + y − k))       CUSUM unilateral de Page acotado en h
    g   = g_min + (1 − g_min) · min(1, S / h)      puerta de la publicación

Mientras el mercado no acumula `min_history` errores, y vale cero. La primera cohorte del
mercado fija r_m² con su media cuadrática. Después, todos los errores actualizan esa media
de forma exponencial, con semivida medida en publicaciones.
La puerta multiplica la tasa η de las escrituras proximales del mercado en esa misma
publicación. Con errores estables S se queda cerca de cero y la corrección escribe con
g_min. Un aumento sostenido del error abre la puerta y la corrección se adapta más deprisa.

El CUSUM de Page no tiene techo. Aquí S se acota en h porque la puerta ya está abierta del
todo en h. Sin ese techo, un cambio largo acumula tanto que la puerta tarda decenas de
publicaciones en cerrarse cuando el error vuelve a su nivel. Con el techo, S baja al menos
k − y por publicación desde h.

La motivación es la relación entre volatilidad percibida y tasa de aprendizaje de Behrens
et al. (2007, https://doi.org/10.1038/nn1954) y Nassar et al. (2010,
https://doi.org/10.1523/JNEUROSCI.0822-10.2010). El detector es el CUSUM de Page (1954,
https://doi.org/10.1093/biomet/41.1-2.100). Ninguno de esos trabajos propone esta puerta.

El control `constant` escribe siempre con la misma puerta, normalmente la media realizada
por la variante en la misma fase (`mean_gate`). Así separa el momento de abrir la puerta de
su efecto medio sobre la tasa. El estado se exporta y se recupera con huella, y cada
publicación devuelve otro estado sin modificar el anterior.
"""

import math
from dataclasses import asdict, dataclass, replace
from types import MappingProxyType

import torch

from .associative_memory import AssociativeMemory, MatureCorrection
from .salience_budget import MAX_COUNT, MAX_ROWS, MAX_TIME, _digest, _finite, _integer

KIND = "regime_evidence_gate_v1"
MODES = ("cusum", "constant")
MARKETS = ("US", "CN")


@dataclass(frozen=True)
class RegimeGateConfig:
    """Detector y puerta fijados antes de evaluar. `constant` solo existe en el control."""

    mode: str = "cusum"
    threshold: float = 4.0
    drift: float = 0.5
    floor_gate: float = 0.25
    half_life: float = 63.0
    min_history: int = 4
    scale_floor: float = 1e-8
    constant: float | None = None

    def __post_init__(self):
        if type(self.mode) is not str or self.mode not in MODES:
            raise ValueError("E4 admite los modos cusum y constant")
        _integer(self.min_history, 1, MAX_COUNT, "El historial mínimo de E4")
        names = ("threshold", "drift", "floor_gate", "half_life", "scale_floor")
        values = {
            name: _finite(getattr(self, name), f"El parámetro {name} de E4") for name in names
        }
        if (
            values["threshold"] <= 0
            or values["drift"] < 0
            or not 0 < values["floor_gate"] <= 1
            or values["half_life"] < 1
            or values["scale_floor"] <= 0
        ):
            raise ValueError("E4 necesita h > 0, k ≥ 0, 0 < g_min ≤ 1, semivida ≥ 1 y suelo > 0")
        if (self.mode == "constant") != (self.constant is not None):
            raise ValueError(
                "Solo el control constant declara su puerta fija, y siempre la necesita"
            )
        if self.constant is not None:
            values["constant"] = _finite(self.constant, "La puerta constante de E4")
            if not 0 < values["constant"] <= 1:
                raise ValueError("La puerta constante de E4 debe estar en (0, 1]")
        for name, value in values.items():
            object.__setattr__(self, name, value)

    @property
    def retention(self):
        """Peso de la media anterior tras una publicación: 2^(−1/semivida)."""
        return 2.0 ** (-1.0 / self.half_life)

    def gate(self, statistic):
        if self.mode == "constant":
            return self.constant
        return self.floor_gate + (1.0 - self.floor_gate) * min(1.0, statistic / self.threshold)

    def identity(self):
        return dict(
            asdict(self),
            kind=KIND,
            statistic="mean_squared_standardized_mature_error_minus_one",
            detector="one_sided_page_cusum_per_market_capped_at_the_threshold",
            scale="exponential_mean_square_of_issued_errors_before_the_cohort",
            target="rate_of_the_proximal_mature_writes_of_the_market",
        )


def check_correction(config, correction):
    """E4 pondera escrituras proximales de A, sin escalas, áreas, roles ni regla kalman.

    Con roles de B3 la memoria va envuelta y una escritura con otra tasa perdería esa
    envoltura, así que la combinación se rechaza igual que las demás.
    """
    if (
        type(config) is not RegimeGateConfig
        or type(correction) is not MatureCorrection
        or correction.memory.rule != "proximal"
        or correction.multiscale is not None
        or correction.areas is not None
        or correction.role_binding is not None
    ):
        raise ValueError("La puerta E4 necesita su configuración y un corrector proximal simple")
    return config


def gated_write(memory, feedback, gates, *, cutoff):
    """Escribir A con la tasa η·g de cada fila, sin cambiar su configuración.

    La regla proximal exige pesos que sumen uno. Con una sola puerta la tasa es η·g. Con
    varias es η·ḡ, con ḡ la media ponderada de las puertas, y los pesos pasan a w·g/ḡ, que por
    fila equivale a η·w·g. La retención y la tasa de las escrituras siguientes no cambian.
    """
    if type(memory) is not AssociativeMemory or len(gates) != len(feedback.values):
        raise ValueError("E4 escribe una memoria proximal simple con una puerta por fila")
    if len(set(gates)) == 1:
        scale, gated = gates[0], feedback
    else:
        weights = feedback.weights * torch.tensor(gates, dtype=feedback.weights.dtype)
        scale = float(weights.sum())
        gated = replace(feedback, weights=weights / scale)
    config = memory.config
    written = AssociativeMemory(
        replace(config, rate=config.rate * scale),
        memory.matrix,
        writes=memory.writes,
        cursor=memory.cursor,
    ).write(gated, cutoff=cutoff)
    return AssociativeMemory(config, written.matrix, writes=written.writes, cursor=written.cursor)


@dataclass(frozen=True)
class GateDecision:
    """Puerta de cada mercado de la publicación y su diagnóstico acotado."""

    gates: tuple
    statistics: tuple
    rows: tuple

    def by_market(self):
        return dict(self.gates)


class RegimeEvidenceGate:
    """Estado E4 por mercado: cuenta, media cuadrática, CUSUM y suma de puertas escritas."""

    def __init__(self, config, *, markets=None, cursor=None, publications=0):
        if type(config) is not RegimeGateConfig:
            raise ValueError("E4 necesita una configuración identificada")
        markets = {} if markets is None else markets
        if type(markets) is not dict or not set(markets) <= set(MARKETS):
            raise ValueError("El estado E4 solo conserva los mercados US y CN")
        _integer(publications, 0, MAX_COUNT, "Las publicaciones de E4")
        if cursor is not None:
            _integer(cursor, 0, MAX_TIME, "El corte de E4")
        checked = {}
        for market, values in markets.items():
            if not isinstance(values, (list, tuple)) or len(values) != 5:
                raise ValueError("Cada mercado E4 necesita cuenta, media, CUSUM, filas y puertas")
            count, square, statistic, rows, gated = values
            _integer(count, 1, MAX_COUNT, "La cuenta de E4")
            _integer(rows, 0, MAX_COUNT, "Las filas con puerta de E4")
            square = _finite(square, "La media cuadrática de E4")
            statistic = _finite(statistic, "El CUSUM de E4")
            gated = _finite(gated, "La suma de puertas de E4")
            if (
                square < 0
                or not 0 <= statistic <= config.threshold
                or not 0 <= gated <= rows
                or rows > count
            ):
                raise ValueError("El estado E4 de un mercado no es coherente")
            checked[market] = (count, square, statistic, rows, gated)
        if (cursor is None) != (publications == 0) or (publications == 0 and checked):
            raise ValueError("El corte y las publicaciones de E4 no son coherentes")
        self.config, self._markets = config, MappingProxyType(checked)
        self.cursor, self.publications = cursor, publications

    def _prior(self, market):
        count, square, statistic, _, _ = self._markets.get(market, (0, None, 0.0, 0, 0.0))
        return count, square, statistic

    def step(self, cohort, *, cutoff):
        """Puertas de una publicación completa y el estado siguiente.

        `cohort` son tuplas (flujo, decisión, disponibilidad, punto emitido, etiqueta). Se
        comprueban todas las disponibilidades antes de leer el primer valor financiero.
        Una publicación vacía también avanza el corte y no cambia las estadísticas.
        """
        _integer(cutoff, 0, MAX_TIME, "El corte de E4")
        if self.cursor is not None and cutoff < self.cursor:
            raise ValueError("Una publicación E4 no puede retroceder")
        if not isinstance(cohort, (list, tuple)) or len(cohort) > MAX_ROWS:
            raise ValueError("E4 necesita una secuencia de hasta 8.192 resultados")
        grouped, seen = {}, set()
        for item in cohort:
            if not isinstance(item, (list, tuple)) or len(item) != 5:
                raise ValueError("Cada resultado E4 necesita identidad, fechas, punto y etiqueta")
            flow, decision, available = item[:3]
            market = flow.split("/", 1)[0] if isinstance(flow, str) else None
            if market not in MARKETS:
                raise ValueError("El resultado E4 no identifica un activo US o CN")
            _integer(decision, 0, MAX_TIME, "La decisión de E4")
            _integer(available, decision + 1, cutoff, "La disponibilidad madura de E4")
            if (flow, decision) in seen:
                raise ValueError("E4 repite un resultado en la misma publicación")
            seen.add((flow, decision))
            grouped.setdefault(market, []).append(item)
        markets, gates, statistics, rows = dict(self._markets), [], [], []
        for market in sorted(grouped):
            items = grouped[market]
            errors = []
            for _, _, _, issued, target in items:
                error = _finite(target, "La etiqueta de E4") - _finite(issued, "El punto de E4")
                if not math.isfinite(error):
                    raise ValueError("El error madurado de E4 desborda FP64")
                errors.append(error)
            count, square, statistic = self._prior(market)
            if count >= self.config.min_history:
                scale = max(math.sqrt(square), self.config.scale_floor)
                # z·z da infinito al desbordar, mientras que z ** 2 lanzaría OverflowError.
                ratios = [error / scale for error in errors]
                excess = math.fsum(z * z for z in ratios) / len(errors) - 1.0
                if not math.isfinite(excess):
                    raise ValueError("El error estandarizado de E4 desborda FP64")
                statistic = min(
                    self.config.threshold, max(0.0, statistic + excess - self.config.drift)
                )
            gate = self.config.gate(statistic)
            mean_square = math.fsum(error * error for error in errors) / len(errors)
            retention = self.config.retention
            square = (
                mean_square
                if square is None
                else retention * square + (1.0 - retention) * mean_square
            )
            if not math.isfinite(square):
                raise ValueError("La media cuadrática de E4 desborda FP64")
            _, _, _, written, gated = markets.get(market, (0, 0.0, 0.0, 0, 0.0))
            markets[market] = (
                count + len(errors),
                square,
                statistic,
                written + len(errors),
                gated + gate * len(errors),
            )
            gates.append((market, gate))
            statistics.append((market, statistic))
            rows.append((market, len(errors)))
        following = RegimeEvidenceGate(
            self.config, markets=markets, cursor=cutoff, publications=self.publications + 1
        )
        return GateDecision(tuple(gates), tuple(statistics), tuple(rows)), following

    @property
    def observed(self):
        """Errores maduros acumulados en todos los mercados."""
        return sum(values[0] for values in self._markets.values())

    def mean_gate(self):
        """Puerta media por fila escrita, la que fija el control constante emparejado."""
        rows = sum(values[3] for values in self._markets.values())
        if rows == 0:
            return None
        return math.fsum(values[4] for values in self._markets.values()) / rows

    def export(self):
        payload = dict(
            schema_version=1,
            kind=KIND,
            configuration=self.config.identity(),
            cursor=self.cursor,
            publications=self.publications,
            markets={market: list(values) for market, values in sorted(self._markets.items())},
        )
        return dict(payload, sha256=_digest(payload))

    @classmethod
    def restore(cls, config, payload):
        fields = {"schema_version", "kind", "configuration", "cursor", "publications", "markets"}
        if (
            type(config) is not RegimeGateConfig
            or type(payload) is not dict
            or set(payload) != fields | {"sha256"}
            or type(payload["schema_version"]) is not int
            or payload["schema_version"] != 1
            or payload["kind"] != KIND
            or payload["configuration"] != config.identity()
            or type(payload["markets"]) is not dict
            or payload["sha256"] != _digest({k: v for k, v in payload.items() if k != "sha256"})
        ):
            raise ValueError("El estado E4 no conserva su contrato o su huella")
        markets = {}
        for market, values in payload["markets"].items():
            if not isinstance(values, list) or len(values) != 5:
                raise ValueError("Cada mercado E4 necesita cuenta, media, CUSUM, filas y puertas")
            count, square, statistic, rows, gated = values
            markets[market] = (count, square, statistic, rows, gated)
        restored = cls(
            config,
            markets=markets,
            cursor=payload["cursor"],
            publications=payload["publications"],
        )
        if restored.export() != payload:
            raise ValueError("Los valores exportados de E4 han cambiado")
        return restored
