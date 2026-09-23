"""Promedios por sesión y mercado sin conservar todas las predicciones en RAM."""

import math

import numpy as np


class SessionErrors:
    """Acumular conteo, error absoluto y error cuadrático por decisión de cada mercado."""

    def __init__(self, *, max_sessions=50_000):
        if type(max_sessions) is not int or not 1 <= max_sessions <= 100_000:
            raise ValueError("El presupuesto de sesiones debe ser un entero positivo y acotado")
        self.max_sessions, self.sessions = max_sessions, {}

    def update(self, markets, prediction_at, errors):
        """Confirmar un lote entero o mantener las estadísticas anteriores si no es válido."""
        if not hasattr(errors, "__len__") or not 1 <= len(errors) <= 4096:
            raise ValueError("El lote de errores debe contener entre 1 y 4096 observaciones")
        values, times, markets = np.asarray(errors), np.asarray(prediction_at), np.asarray(markets)
        if np.issubdtype(times.dtype, np.datetime64):
            converted = times.astype("datetime64[us]")
            if np.isnat(times).any() or not np.array_equal(converted.astype(times.dtype), times):
                raise ValueError("Las fechas están ausentes o pierden precisión en microsegundos")
            times = converted.astype(np.int64)
        if (
            values.ndim != 1
            or times.shape != values.shape
            or markets.shape != values.shape
            or np.iscomplexobj(values)
            or not np.issubdtype(times.dtype, np.integer)
            or not np.isin(markets, ("US", "CN")).all()
        ):
            raise ValueError("Los errores, mercados y fechas no forman un lote coherente")
        try:
            with np.errstate(over="raise", invalid="raise"):
                values = values.astype(np.float64)
                absolute, squared = np.abs(values), np.square(values)
        except (TypeError, ValueError, FloatingPointError) as error:
            raise ValueError("Los errores deben ser reales y representables en float64") from error
        if not np.isfinite(squared).all():
            raise ValueError("Los errores deben ser finitos")
        pending = {}
        for market, moment, ae, se in zip(markets, times, absolute, squared, strict=True):
            key = str(market), int(moment)
            entry = pending.setdefault(key, [0, 0.0, 0.0])
            entry[0] += 1
            entry[1] += float(ae)
            entry[2] += float(se)
        if (
            len(self.sessions) + sum(key not in self.sessions for key in pending)
            > self.max_sessions
        ):
            raise ValueError("El número de sesiones supera el presupuesto de evaluación")
        merged = {}
        for key, values in pending.items():
            previous = self.sessions.get(key, (0, 0.0, 0.0))
            merged[key] = [a + b for a, b in zip(previous, values, strict=True)]
            if not all(math.isfinite(v) for v in merged[key]):
                raise ValueError("La acumulación de errores supera el rango numérico")
        self.sessions.update(merged)

    @staticmethod
    def _means(rows):
        return dict(
            sessions=len(rows),
            samples=sum(row[0] for row in rows),
            mae=math.fsum(row[1] / row[0] / len(rows) for row in rows),
            mse=math.fsum(row[2] / row[0] / len(rows) for row in rows),
        )

    def summary(self):
        """Asignar el mismo peso a cada par de mercado e instante observado."""
        rows = list(self.sessions.values())
        try:
            means = self._means(rows) if rows else dict(sessions=0, samples=0, mae=None, mse=None)
            absolute = math.fsum(row[1] for row in rows)
            squared = math.fsum(row[2] for row in rows)
            by_market = {
                market: self._means([v for (m, _), v in self.sessions.items() if m == market])
                for market in sorted({m for m, _ in self.sessions})
            }
        except OverflowError as error:
            raise ValueError("Los totales de error superan el rango de float64") from error
        return dict(
            samples=means["samples"],
            absolute_error=absolute,
            squared_error=squared,
            session_count=means["sessions"],
            session_mae=means["mae"],
            session_mse=means["mse"],
            session_metrics_reason=None if rows else "No hay sesiones observadas",
            by_market_session=by_market,
        )
