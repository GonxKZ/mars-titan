"""Pruebas exhaustivas para el módulo ic_stability."""

from __future__ import annotations

import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/dgx_ic_stability.py"
SPEC = importlib.util.spec_from_file_location("dgx_ic_stability", SCRIPT)
ics = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ics)

# ---------------------------------------------------------------------------
# 1. month_of
# ---------------------------------------------------------------------------


class TestMonthOf:
    """Pruebas de conversión de instantes a mes UTC («YYYY-MM»)."""

    @pytest.mark.parametrize(
        ("instant", "expected"),
        [
            # ISO 8601 con «T» y zona horaria explícita UTC.
            ("2013-03-04T20:55:00+00:00", "2013-03"),
            # ISO 8601 con espacio en lugar de «T».
            ("2013-03-04 20:55:00+00:00", "2013-03"),
            # ISO 8601 con sufijo «Z».
            ("2013-03-04T20:55:00Z", "2013-03"),
            # Solo fecha (interpretada en UTC).
            ("2013-03-04", "2013-03"),
            # Fecha y hora sin desplazamiento horario (interpretadas en UTC).
            ("2013-03-04T20:55:00", "2013-03"),
            ("2013-03-04 20:55:00", "2013-03"),
            # ISO 8601 con fracción de segundos hasta nanosegundos.
            ("2013-03-04T20:55:00.123456789+00:00", "2013-03"),
            # Entero de nanosegundos desde la época Unix:
            # 2013-03-04T20:55:00+00:00 equivale a 1362430500 s = 1362430500000000000 ns.
            ("1362430500000000000", "2013-03"),
            # Época Unix exacta (1970-01-01T00:00:00Z).
            ("0", "1970-01"),
            # Un nanosegundo antes de la época (1969-12-31T23:59:59.999999999Z).
            ("-1", "1969-12"),
            # Cambio de mes al pasar a UTC (adelanta al mes siguiente):
            # 2023-03-31T23:30:00-02:00 == 2023-04-01T01:30:00+00:00 -> 2023-04.
            ("2023-03-31T23:30:00-02:00", "2023-04"),
            # Cambio de mes y año al pasar a UTC:
            # 2023-12-31T22:00:00-05:00 == 2024-01-01T03:00:00+00:00 -> 2024-01.
            ("2023-12-31T22:00:00-05:00", "2024-01"),
            # Cambio de mes al pasar a UTC (retrocede al mes anterior):
            # 2023-04-01T00:30:00+02:00 == 2023-03-31T22:30:00+00:00 -> 2023-03.
            ("2023-04-01T00:30:00+02:00", "2023-03"),
        ],
    )
    def test_formatos_validos_y_cambio_de_mes_utc(self, instant: str, expected: str) -> None:
        assert ics.month_of(instant) == expected

    @pytest.mark.parametrize(
        "bad_instant",
        ["", "   ", "no-es-una-fecha", "2023-13-01", "2023-02-30", "2023-03-04T25:00:00Z"],
    )
    def test_rechaza_formatos_invalidos(self, bad_instant: str) -> None:
        with pytest.raises(ValueError):
            ics.month_of(bad_instant)


# ---------------------------------------------------------------------------
# 2. max_drawdown
# ---------------------------------------------------------------------------


class TestMaxDrawdown:
    """Pruebas de caída máxima sobre la serie acumulada."""

    def test_serie_pequena_calculada_a_mano(self) -> None:
        # values = [0.2, 0.3, -0.4, -0.2, 0.5, -0.3]
        # Acumulada C:
        #   C_0 =  0.0
        #   C_1 =  0.2
        #   C_2 =  0.5  (máximo previo)
        #   C_3 =  0.1  (caída 0.5 - 0.1 = 0.4)
        #   C_4 = -0.1  (caída 0.5 - (-0.1) = 0.6 -> máxima)
        #   C_5 =  0.4  (caída 0.5 - 0.4 = 0.1)
        #   C_6 =  0.1  (caída 0.5 - 0.1 = 0.4)
        # Respuesta: depth = 0.6, start = 2, end = 4.
        result = ics.max_drawdown([0.2, 0.3, -0.4, -0.2, 0.5, -0.3])
        assert result["depth"] == pytest.approx(0.6)
        assert result["start"] == 2
        assert result["end"] == 4

    def test_caida_desde_antes_de_la_primera_sesion(self) -> None:
        # values = [-0.5, -0.25, 1.0, -0.5]
        # C = [0.0, -0.5, -0.75, 0.25, -0.25]
        # Máxima caída: C_0 - C_2 = 0.0 - (-0.75) = 0.75 (s = 0, t = 2).
        # Segunda caída: C_3 - C_4 = 0.25 - (-0.25) = 0.5 (menor).
        result = ics.max_drawdown([-0.5, -0.25, 1.0, -0.5])
        assert result == {"depth": 0.75, "start": 0, "end": 2}

    def test_casos_limite(self) -> None:
        # Serie vacía -> C = [0.0], depth = 0, start = end = 0.
        assert ics.max_drawdown([]) == {"depth": 0.0, "start": 0, "end": 0}
        # n = 1 positivo -> C = [0.0, 0.5], depth = 0, start = end = 0.
        assert ics.max_drawdown([0.5]) == {"depth": 0.0, "start": 0, "end": 0}
        # n = 1 cero -> C = [0.0, 0.0], depth = 0, start = end = 0.
        assert ics.max_drawdown([0.0]) == {"depth": 0.0, "start": 0, "end": 0}
        # n = 1 negativo -> C = [0.0, -0.25], depth = 0.25, start = 0, end = 1.
        assert ics.max_drawdown([-0.25]) == {"depth": 0.25, "start": 0, "end": 1}
        # Todos ceros -> depth = 0, start = end = 0.
        assert ics.max_drawdown([0.0, 0.0, 0.0, 0.0]) == {"depth": 0.0, "start": 0, "end": 0}
        # Todos positivos -> C estrictamente creciente -> depth = 0, start = end = 0.
        assert ics.max_drawdown([0.1, 0.2, 0.05, 0.4]) == {"depth": 0.0, "start": 0, "end": 0}
        # Todos negativos -> C = [0.0, -0.25, -0.75, -1.5] -> depth = 1.5, start = 0, end = 3.
        assert ics.max_drawdown([-0.25, -0.5, -0.75]) == {"depth": 1.5, "start": 0, "end": 3}

    def test_empates_devuelven_primera_ocurrencia(self) -> None:
        # Empate entre dos caídas distintas de profundidad 1.0:
        # values = [1.0, -1.0, 2.0, -1.0] -> C = [0, 1, 0, 2, 1]
        # (s=1, t=2) da 1.0 y (s=3, t=4) da 1.0 -> primera: (1, 2).
        assert ics.max_drawdown([1.0, -1.0, 2.0, -1.0]) == {"depth": 1.0, "start": 1, "end": 2}

        # Empate en el máximo previo:
        # values = [1.0, -0.5, 0.5, -2.0] -> C = [0.0, 1.0, 0.5, 1.0, -1.0]
        # Tanto s=1 como s=3 alcanzan 1.0 antes de t=4 (-1.0) -> primera: s=1, t=4.
        assert ics.max_drawdown([1.0, -0.5, 0.5, -2.0]) == {"depth": 2.0, "start": 1, "end": 4}

        # Empate en valle plano:
        # values = [-1.0, 0.0, 0.0] -> C = [0, -1, -1, -1] -> primera: s=0, t=1.
        assert ics.max_drawdown([-1.0, 0.0, 0.0]) == {"depth": 1.0, "start": 0, "end": 1}

        # Empate entre pico inicial C_0 = 0 y C_1 = 0:
        # values = [0.0, -1.0] -> C = [0, 0, -1] -> primera: s=0, t=2.
        assert ics.max_drawdown([0.0, -1.0]) == {"depth": 1.0, "start": 0, "end": 2}

    def test_rechaza_entradas_no_validas(self) -> None:
        for bad in ([float("nan")], [float("inf")], [[0.1, 0.2]], "123"):
            with pytest.raises(ValueError):
                ics.max_drawdown(bad)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 3. longest_negative_run
# ---------------------------------------------------------------------------


class TestLongestNegativeRun:
    """Pruebas de racha más larga de sesiones con valor <= 0."""

    def test_serie_pequena_calculada_a_mano(self) -> None:
        # [0.1, -0.2, 0.0, -0.1, 0.05, -0.3, -0.4]
        # Rachas <= 0: índices 1..3 (longitud 3, el cero cuenta) y 5..6 (longitud 2).
        assert ics.longest_negative_run([0.1, -0.2, 0.0, -0.1, 0.05, -0.3, -0.4]) == 3

    def test_casos_limite_y_empates(self) -> None:
        assert ics.longest_negative_run([]) == 0
        assert ics.longest_negative_run([0.01]) == 0
        assert ics.longest_negative_run([0.0]) == 1
        assert ics.longest_negative_run([-0.01]) == 1
        assert ics.longest_negative_run([0.0, 0.0, 0.0, 0.0]) == 4
        assert ics.longest_negative_run([0.1, 0.2, 0.3]) == 0
        assert ics.longest_negative_run([-0.1, -0.2, -0.3, -0.4, -0.5]) == 5
        # Empate entre dos rachas máximas de longitud 2.
        assert ics.longest_negative_run([-1.0, 0.0, 0.5, -0.1, -0.2]) == 2


# ---------------------------------------------------------------------------
# 4. worst_window
# ---------------------------------------------------------------------------


class TestWorstWindow:
    """Pruebas de peor ventana móvil consecutiva."""

    def test_serie_pequena_calculada_a_mano(self) -> None:
        # values = [0.5, -0.25, -0.75, 0.25, -0.5], window = 3
        # Ventana 0 ([0.5, -0.25, -0.75]): media = -0.5 / 3 = -1/6
        # Ventana 1 ([-0.25, -0.75, 0.25]): media = -0.75 / 3 = -0.25
        # Ventana 2 ([-0.75, 0.25, -0.5]): media = -1.0 / 3 = -1/3  (peor)
        result = ics.worst_window([0.5, -0.25, -0.75, 0.25, -0.5], window=3)
        assert result is not None
        assert result["start"] == 2
        assert result["mean"] == pytest.approx(-1.0 / 3.0)

    def test_casos_limite(self) -> None:
        # n < window -> None (incluida serie vacía).
        assert ics.worst_window([], window=1) is None
        assert ics.worst_window([0.1, 0.2], window=3) is None
        assert ics.worst_window([0.1], window=2) is None
        # n == window -> única ventana en start=0, media (0.25 + 0.75) / 2 = 0.5.
        assert ics.worst_window([0.25, 0.75], window=2) == {"mean": 0.5, "start": 0}
        # n == 1, window == 1.
        assert ics.worst_window([-0.5], window=1) == {"mean": -0.5, "start": 0}
        # Todos ceros -> media 0.0 en la primera posición (0).
        assert ics.worst_window([0.0, 0.0, 0.0], window=2) == {"mean": 0.0, "start": 0}

    def test_empates_devuelven_primera_ventana(self) -> None:
        # Incluye decimales que con sumas acumuladas podrían perder precisión y alterar empates.
        # Ventana 0 ([0.3, -0.1]): 0.1
        # Ventana 1 ([-0.1, 0.1]): 0.0  <- primera mínima
        # Ventana 2 ([0.1, 0.1]):  0.1
        # Ventana 3 ([0.1, -0.1]): 0.0
        # Ventana 4 ([-0.1, 0.1]): 0.0
        result = ics.worst_window([0.3, -0.1, 0.1, 0.1, -0.1, 0.1], window=2)
        assert result == {"mean": 0.0, "start": 1}

        # Empate con window = 1.
        assert ics.worst_window([0.2, -0.5, 0.1, -0.5], window=1) == {"mean": -0.5, "start": 1}

    @pytest.mark.parametrize("bad_window", [0, -1, -10, True, False, 2.5])
    def test_rechaza_ventana_invalida(self, bad_window: object) -> None:
        with pytest.raises(ValueError):
            ics.worst_window([0.1, 0.2, 0.3], window=bad_window)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 5. stability
# ---------------------------------------------------------------------------


class TestStability:
    """Pruebas del resumen completo de estabilidad."""

    def test_serie_pequena_sin_meses_calculada_a_mano(self) -> None:
        # values = [0.25, -0.25, 0.75, 0.25, -0.5], n = 5, window = 2
        # mean = 0.5 / 5 = 0.1
        # desviaciones: [0.15, -0.35, 0.65, 0.15, -0.60]
        # suma de cuadrados = 0.0225 + 0.1225 + 0.4225 + 0.0225 + 0.36 = 0.95
        # std (ddof=1) = sqrt(0.95 / 4) = sqrt(0.2375)
        # icir = 0.1 / sqrt(0.2375)
        # positive_fraction = 3 / 5 = 0.6
        # C = [0.0, 0.25, 0.0, 0.75, 1.0, 0.5] -> max_drawdown: depth=0.5, start=4, end=5
        # longest_negative_run = 1
        # worst_window(2): [0.0, 0.25, 0.5, -0.125] -> mean=-0.125, start=3
        # halves (n // 2 = 2):
        #   first = mean([0.25, -0.25]) = 0.0
        #   second = mean([0.75, 0.25, -0.5]) = 0.5 / 3 = 1/6
        #   difference = 1/6 - 0.0 = 1/6
        res = ics.stability([0.25, -0.25, 0.75, 0.25, -0.5], window=2)

        expected_std = math.sqrt(0.2375)
        assert res["sessions"] == 5
        assert res["mean"] == pytest.approx(0.1)
        assert res["std"] == pytest.approx(expected_std)
        assert res["icir"] == pytest.approx(0.1 / expected_std)
        assert res["positive_fraction"] == pytest.approx(0.6)
        assert res["max_drawdown"] == {"depth": 0.5, "start": 4, "end": 5}
        assert res["longest_negative_run"] == 1
        assert res["worst_window"] == {"mean": -0.125, "start": 3}
        assert res["halves"]["first"] == pytest.approx(0.0)
        assert res["halves"]["second"] == pytest.approx(1.0 / 6.0)
        assert res["halves"]["difference"] == pytest.approx(1.0 / 6.0)

        for key in ("monthly", "months_positive_fraction", "worst_month", "best_month"):
            assert key not in res

    def test_mitades_con_longitud_par_y_n2(self) -> None:
        # n = 4 -> mitades en 2: first = mean([1.0, 3.0]) = 2.0, second = mean([5.0, 7.0]) = 6.0.
        res4 = ics.stability([1.0, 3.0, 5.0, 7.0], window=2)
        assert res4["halves"] == {"first": 2.0, "second": 6.0, "difference": 4.0}

        # n = 2 -> mitades en 1: first = 1.0, second = -1.0, difference = -2.0.
        res2 = ics.stability([1.0, -1.0], window=2)
        assert res2["halves"] == {"first": 1.0, "second": -1.0, "difference": -2.0}

    def test_serie_con_meses_y_empates_cronologicos(self) -> None:
        # 2023-01: [0.2, 0.4] -> 2 sesiones, media 0.3   (empata como mejor mes)
        # 2023-02: [-0.2, 0.0] -> 2 sesiones, media -0.1 (empata como peor mes)
        # 2023-03: [-0.1] -> 1 sesión, media -0.1        (segundo en empate de peor mes)
        # 2023-04: [0.3] -> 1 sesión, media 0.3          (segundo en empate de mejor mes)
        # Meses positivos (> 0): 2023-01 y 2023-04 -> 2/4 = 0.5.
        values = [0.2, 0.4, -0.2, 0.0, -0.1, 0.3]
        months = ["2023-01", "2023-01", "2023-02", "2023-02", "2023-03", "2023-04"]

        res = ics.stability(values, months=months, window=3)

        assert len(res["monthly"]) == 4
        assert [m["month"] for m in res["monthly"]] == ["2023-01", "2023-02", "2023-03", "2023-04"]
        assert [m["sessions"] for m in res["monthly"]] == [2, 2, 1, 1]
        assert [m["mean"] for m in res["monthly"]] == pytest.approx([0.3, -0.1, -0.1, 0.3])
        assert res["months_positive_fraction"] == pytest.approx(0.5)
        assert res["worst_month"]["month"] == "2023-02"
        assert res["worst_month"]["mean"] == pytest.approx(-0.1)
        assert set(res["worst_month"].keys()) == {"month", "mean"}
        assert res["best_month"]["month"] == "2023-01"
        assert res["best_month"]["mean"] == pytest.approx(0.3)
        assert set(res["best_month"].keys()) == {"month", "mean"}

    def test_casos_limite_n1_ceros_constante_signos(self) -> None:
        # n = 1
        r1 = ics.stability([0.25], months=["2023-05"], window=21)
        assert r1["sessions"] == 1
        assert r1["mean"] == 0.25
        assert r1["std"] is None
        assert r1["icir"] is None
        assert r1["positive_fraction"] == 1.0
        assert r1["max_drawdown"] == {"depth": 0.0, "start": 0, "end": 0}
        assert r1["longest_negative_run"] == 0
        assert r1["worst_window"] is None
        assert r1["halves"] is None
        assert r1["months_positive_fraction"] == 1.0
        assert r1["worst_month"] == {"month": "2023-05", "mean": 0.25}
        assert r1["best_month"] == {"month": "2023-05", "mean": 0.25}

        # Todos ceros -> std = 0.0, icir = None.
        r_zero = ics.stability([0.0, 0.0, 0.0], window=2)
        assert r_zero["mean"] == 0.0
        assert r_zero["std"] == 0.0
        assert r_zero["icir"] is None
        assert r_zero["positive_fraction"] == 0.0
        assert r_zero["max_drawdown"] == {"depth": 0.0, "start": 0, "end": 0}
        assert r_zero["longest_negative_run"] == 3
        assert r_zero["worst_window"] == {"mean": 0.0, "start": 0}
        assert r_zero["halves"] == {"first": 0.0, "second": 0.0, "difference": 0.0}

        # Serie constante no exacta en binario (0.1 repetido) -> std debe ser 0.0 e icir None.
        r_const = ics.stability([0.1, 0.1, 0.1], window=2)
        assert r_const["std"] == 0.0
        assert r_const["icir"] is None

        # Todos positivos -> caída 0, racha negativa 0, fracción positiva 1.0.
        r_pos = ics.stability([0.1, 0.2, 0.3, 0.4], window=2)
        assert r_pos["positive_fraction"] == 1.0
        assert r_pos["max_drawdown"] == {"depth": 0.0, "start": 0, "end": 0}
        assert r_pos["longest_negative_run"] == 0

        # Todos negativos -> caída total 1.0 (0..4), racha negativa 4, fracción positiva 0.0.
        r_neg = ics.stability([-0.1, -0.2, -0.3, -0.4], window=2)
        assert r_neg["positive_fraction"] == 0.0
        assert r_neg["max_drawdown"]["depth"] == pytest.approx(1.0)
        assert r_neg["max_drawdown"]["start"] == 0
        assert r_neg["max_drawdown"]["end"] == 4
        assert r_neg["longest_negative_run"] == 4

    @pytest.mark.parametrize(
        ("values", "months", "window"),
        [
            ([], None, 21),
            (np.array([], dtype=np.float64), None, 21),
            (np.zeros((2, 2)), None, 21),
            (np.array(0.5), None, 21),
            ([0.1, float("nan")], None, 21),
            ([0.1, float("inf")], None, 21),
            ([0.1, float("-inf")], None, 21),
            ("0.1", None, 21),
            ([True, False], None, 21),
            ([0.1, 0.2], ["2023-01"], 21),
            ([0.1, 0.2], ["2023-01", "2023-13"], 21),
            ([0.1, 0.2], ["2023-01", "2023/02"], 21),
            ([0.1, 0.2], None, 0),
        ],
    )
    def test_rechaza_entradas_invalidas(self, values: object, months: object, window: int) -> None:
        with pytest.raises(ValueError):
            ics.stability(values, months=months, window=window)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 6. Propiedades con series aleatorias de semilla fija
# ---------------------------------------------------------------------------


def _naive_max_drawdown(values: np.ndarray) -> dict[str, float | int]:
    """Implementación ingenua O(n^2) de referencia para max_drawdown."""
    n = len(values)
    cum = np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))
    best_depth = 0.0
    best_start = 0
    best_end = 0
    for s in range(n + 1):
        for t in range(s, n + 1):
            drop = float(cum[s] - cum[t])
            if drop > best_depth:
                best_depth = drop
                best_start = s
                best_end = t
    if best_depth == 0.0:
        return {"depth": 0.0, "start": 0, "end": 0}
    return {"depth": best_depth, "start": best_start, "end": best_end}


class TestRandomProperties:
    """Propiedades invariantes comprobadas sobre series aleatorias con semilla fija."""

    def test_max_drawdown_no_negativo_y_coincide_con_referencia_o_n2(self) -> None:
        rng = np.random.default_rng(20240304)
        for length in (1, 5, 20, 80):
            for _ in range(10):
                series = rng.normal(loc=0.01, scale=0.1, size=length)
                got = ics.max_drawdown(series)
                ref = _naive_max_drawdown(series)
                assert got["depth"] >= 0.0
                assert got["depth"] == pytest.approx(ref["depth"], rel=1e-12, abs=1e-12)
                assert got["start"] == ref["start"]
                assert got["end"] == ref["end"]

    def test_positive_fraction_invariante_ante_escala_positiva(self) -> None:
        rng = np.random.default_rng(42)
        series = rng.normal(loc=0.0, scale=0.05, size=100)
        # Inyectamos algunos ceros exactos para comprobar el borde > 0.
        series[[0, 15, 50, 99]] = 0.0
        base = ics.stability(series)["positive_fraction"]
        for scale in (1e-6, 0.25, 3.7, 1e6):
            scaled = ics.stability(series * scale)["positive_fraction"]
            assert scaled == base

    def test_racha_negativa_de_serie_positiva_invertida_es_n(self) -> None:
        rng = np.random.default_rng(12345)
        for length in (1, 10, 64):
            positive = rng.uniform(low=1e-4, high=1.0, size=length)
            assert ics.longest_negative_run(positive) == 0
            assert ics.longest_negative_run(-positive) == length

    def test_worst_window_coincide_con_medias_moviles_ingenuas(self) -> None:
        rng = np.random.default_rng(98765)
        series = rng.normal(loc=0.02, scale=0.08, size=120)
        for window in (1, 5, 21, 120):
            got = ics.worst_window(series, window)
            naive_means = [
                float(np.mean(series[i : i + window])) for i in range(len(series) - window + 1)
            ]
            min_mean = min(naive_means)
            first_idx = naive_means.index(min_mean)
            assert got is not None
            assert got["mean"] == pytest.approx(min_mean, rel=1e-12, abs=1e-12)
            assert got["start"] == first_idx


# ---------------------------------------------------------------------------
# 7. read_sessions
# ---------------------------------------------------------------------------


class TestReadSessions:
    """Pruebas de lectura, validación y ordenación de archivos CSV."""

    def test_ordena_filas_desordenadas_con_dos_mercados_y_formatos_mixtos(
        self, tmp_path: Path
    ) -> None:
        # Instantes en UTC (orden cronológico real esperado):
        # 1) 2013-03-04 solo fecha -> 2013-03-04T00:00:00Z (mercado EU)
        # 2) 2013-03-04 solo fecha -> 2013-03-04T00:00:00Z (mercado US; desempate EU < US)
        # 3) 1362430500000000000 ns -> 2013-03-04T20:55:00.000000000Z (mercado US)
        # 4) 2013-03-04T20:55:00.000000001+00:00 -> 1 ns después (mercado US)
        # 5) 2013-03-31 23:30:00-02:00 -> 2013-04-01T01:30:00Z (mercado EU, mes UTC 2013-04)
        # 6) 2013-04-02T14:00:00+00:00 -> 2013-04-02T14:00:00Z (mercado US)
        csv_text = (
            "session,alfa,beta\n"
            "2013:US:2013-04-02T14:00:00+00:00,0.6,-0.6\n"
            "2013:US:2013-03-04,0.2,-0.2\n"
            "2013:EU:2013-03-31 23:30:00-02:00,0.5,-0.5\n"
            "2013:US:2013-03-04T20:55:00.000000001+00:00,0.4,-0.4\n"
            "2013:EU:2013-03-04,0.1,-0.1\n"
            "2013:US:1362430500000000000,0.3,-0.3\n"
        )
        csv_path = tmp_path / "sesiones.csv"
        csv_path.write_text(csv_text, encoding="utf-8")

        sessions, months, markets, names, matrix = ics.read_sessions(csv_path)

        assert names == ["alfa", "beta"]
        assert sessions == [
            "2013:EU:2013-03-04",
            "2013:US:2013-03-04",
            "2013:US:1362430500000000000",
            "2013:US:2013-03-04T20:55:00.000000001+00:00",
            "2013:EU:2013-03-31 23:30:00-02:00",
            "2013:US:2013-04-02T14:00:00+00:00",
        ]
        assert months == ["2013-03", "2013-03", "2013-03", "2013-03", "2013-04", "2013-04"]
        assert markets == ["EU", "US", "US", "US", "EU", "US"]
        assert isinstance(matrix, np.ndarray)
        assert matrix.shape == (6, 2)
        assert matrix.dtype == np.float64
        np.testing.assert_allclose(
            matrix,
            np.array(
                [
                    [0.1, -0.1],
                    [0.2, -0.2],
                    [0.3, -0.3],
                    [0.4, -0.4],
                    [0.5, -0.5],
                    [0.6, -0.6],
                ]
            ),
        )

    @pytest.mark.parametrize(
        ("case_name", "content"),
        [
            ("archivo_vacio", ""),
            ("sin_filas", "session,m1\n"),
            ("cabecera_sin_session", "fecha,m1\n2013:US:2013-03-04,0.1\n"),
            ("cabecera_sin_modelos", "session\n2013:US:2013-03-04\n"),
            ("nombre_modelo_vacio", "session,m1,\n2013:US:2013-03-04,0.1,0.2\n"),
            ("modelos_repetidos", "session,m1,m2,m1\n2013:US:2013-03-04,0.1,0.2,0.3\n"),
            ("fila_con_menos_campos", "session,m1,m2\n2013:US:2013-03-04,0.1\n"),
            ("fila_con_mas_campos", "session,m1\n2013:US:2013-03-04,0.1,0.2\n"),
            ("valor_nan", "session,m1\n2013:US:2013-03-04,nan\n"),
            ("valor_inf", "session,m1\n2013:US:2013-03-04,inf\n"),
            ("valor_menos_inf", "session,m1\n2013:US:2013-03-04,-inf\n"),
            ("valor_no_numerico", "session,m1\n2013:US:2013-03-04,abc\n"),
            (
                "sesion_repetida_identica",
                "session,m1\n2013:US:2013-03-04,0.1\n2013:US:2013-03-04,0.2\n",
            ),
            (
                "sesion_repetida_mismo_instante_y_mercado",
                (
                    "session,m1\n"
                    "2013:US:2013-03-04T20:55:00+00:00,0.1\n"
                    "2013:US:1362430500000000000,0.2\n"
                ),
            ),
            ("sesion_mal_formada", "session,m1\n2013-US-2013-03-04,0.1\n"),
            ("mercado_vacio", "session,m1\n2013::2013-03-04,0.1\n"),
            ("anio_invalido", "session,m1\nABCD:US:2013-03-04,0.1\n"),
            ("instante_invalido", "session,m1\n2013:US:fecha-rota,0.1\n"),
        ],
    )
    def test_rechaza_todos_los_errores(self, tmp_path: Path, case_name: str, content: str) -> None:
        csv_path = tmp_path / f"{case_name}.csv"
        csv_path.write_text(content, encoding="utf-8")
        with pytest.raises(ValueError):
            ics.read_sessions(csv_path)


# ---------------------------------------------------------------------------
# 8. stability_table
# ---------------------------------------------------------------------------


class TestStabilityTable:
    """Pruebas de stability_table con uno y dos mercados."""

    def test_un_solo_mercado_solo_ambito_all(self, tmp_path: Path) -> None:
        # Un único mercado (US), 3 sesiones en 2 meses:
        # Modelo m1: [0.1, -0.3, 0.5] -> media = 0.3 / 3 = 0.1,
        #   2023-01: (0.1 - 0.3)/2 = -0.1, 2023-02: 0.5 -> 50 % meses positivos.
        # Modelo m2: [0.2, 0.2, 0.2] -> media = 0.2, std = 0.0, icir = None.
        csv_path = tmp_path / "uno.csv"
        csv_path.write_text(
            "session,m1,m2\n"
            "2023:US:2023-02-01,0.5,0.2\n"
            "2023:US:2023-01-01,0.1,0.2\n"
            "2023:US:2023-01-15,-0.3,0.2\n",
            encoding="utf-8",
        )

        table = ics.stability_table(csv_path, window=2)

        assert list(table.keys()) == ["m1", "m2"]
        assert list(table["m1"].keys()) == ["all"]
        assert list(table["m2"].keys()) == ["all"]

        assert table["m1"]["all"]["sessions"] == 3
        assert table["m1"]["all"]["mean"] == pytest.approx(0.1)
        assert table["m1"]["all"]["months_positive_fraction"] == pytest.approx(0.5)
        assert table["m1"]["all"]["worst_month"] == {
            "month": "2023-01",
            "mean": pytest.approx(-0.1),
        }
        assert table["m1"]["all"]["worst_window"] == {"mean": pytest.approx(-0.1), "start": 0}

        assert table["m2"]["all"]["mean"] == pytest.approx(0.2)
        assert table["m2"]["all"]["std"] == 0.0
        assert table["m2"]["all"]["icir"] is None

    def test_dos_mercados_calcula_all_y_cada_mercado(self, tmp_path: Path) -> None:
        # Dos mercados (EU y US), 4 sesiones:
        # Ordenado:
        #   2023:EU:2023-01-02 -> m1 =  0.4
        #   2023:US:2023-01-03 -> m1 = -0.2
        #   2023:EU:2023-02-01 -> m1 =  0.2
        #   2023:US:2023-02-02 -> m1 = -0.4
        # Serie conjunta («all»): [0.4, -0.2, 0.2, -0.4] -> media = 0.0, positive_fraction = 0.5.
        # Mercado EU: [0.4, 0.2] -> media = 0.3, positive_fraction = 1.0, max_drawdown = 0.
        # Mercado US: [-0.2, -0.4] -> media = -0.3, positive_fraction = 0.0,
        #   max_drawdown = 0.6 (start=0, end=2), longest_negative_run = 2.
        csv_path = tmp_path / "dos.csv"
        csv_path.write_text(
            "session,m1\n"
            "2023:US:2023-02-02,-0.4\n"
            "2023:EU:2023-01-02,0.4\n"
            "2023:US:2023-01-03,-0.2\n"
            "2023:EU:2023-02-01,0.2\n",
            encoding="utf-8",
        )

        table = ics.stability_table(csv_path, window=2)

        assert list(table["m1"].keys()) == ["all", "EU", "US"]

        assert table["m1"]["all"]["sessions"] == 4
        assert table["m1"]["all"]["mean"] == pytest.approx(0.0)
        assert table["m1"]["all"]["positive_fraction"] == pytest.approx(0.5)

        assert table["m1"]["EU"]["sessions"] == 2
        assert table["m1"]["EU"]["mean"] == pytest.approx(0.3)
        assert table["m1"]["EU"]["positive_fraction"] == 1.0
        assert table["m1"]["EU"]["max_drawdown"] == {"depth": 0.0, "start": 0, "end": 0}

        assert table["m1"]["US"]["sessions"] == 2
        assert table["m1"]["US"]["mean"] == pytest.approx(-0.3)
        assert table["m1"]["US"]["positive_fraction"] == 0.0
        assert table["m1"]["US"]["max_drawdown"]["depth"] == pytest.approx(0.6)
        assert table["m1"]["US"]["max_drawdown"]["start"] == 0
        assert table["m1"]["US"]["max_drawdown"]["end"] == 2
        assert table["m1"]["US"]["longest_negative_run"] == 2


# ---------------------------------------------------------------------------
# 9. markdown y main
# ---------------------------------------------------------------------------


class TestMarkdownAndMain:
    """Pruebas de formato Markdown y de la interfaz de línea de comandos."""

    def test_markdown_columnas_formato_decimal_y_valores_ausentes(self, tmp_path: Path) -> None:
        # Dos mercados para comprobar varias filas; window=21 > n para probar peor ventana = «—».
        # EU tiene n=1 (std=None, icir=None) -> ICIR = «—».
        # US tiene [0.2, 0.2] (std=0, icir=None) -> ICIR = «—».
        # all tiene [-0.1, 0.2, 0.2] -> sesiones=3, mean=0.1,
        #   std = sqrt(((-0.2)^2 + 0.1^2 + 0.1^2)/2) = sqrt(0.03) = 0.17320508...
        #   icir = 0.1 / sqrt(0.03) = 0.57735027... -> «0,5774»
        #   % sesiones positivas = 2/3 = 66,7 %
        #   meses: 2023-01 ([-0.1, 0.2] -> 0.05), 2023-02 ([0.2] -> 0.2) -> 100,0 % meses positivos
        #   peor mes: 2023-01 (0,0500)
        #   caída máxima: C = [0, -0.1, 0.1, 0.3] -> depth = 0,1000
        #   racha negativa más larga: 1
        csv_path = tmp_path / "md.csv"
        csv_path.write_text(
            "session,modelo_A\n"
            "2023:EU:2023-01-02,-0.1\n"
            "2023:US:2023-01-03,0.2\n"
            "2023:US:2023-02-01,0.2\n",
            encoding="utf-8",
        )

        table = ics.stability_table(csv_path, window=21)
        md = ics.markdown(table)
        lines = md.strip().splitlines()

        # Cabecera + separador + 3 filas (all, EU, US).
        assert len(lines) == 5

        headers = [cell.strip() for cell in lines[0].strip("|").split("|")]
        assert len(headers) == 11
        assert headers == [
            "Modelo",
            "Ámbito",
            "Sesiones",
            "IC medio",
            "ICIR",
            "% sesiones positivas",
            "% meses positivos",
            "Peor mes",
            "Caída máxima del IC acumulado",
            "Racha negativa más larga",
            "Peor ventana",
        ]

        row_all = [cell.strip() for cell in lines[2].strip("|").split("|")]
        assert row_all[0] == "modelo_A"
        assert row_all[1] == "all"
        assert row_all[2] == "3"
        assert row_all[3] == "0,1000"
        assert row_all[4] == "0,5774"
        assert row_all[5] == "66,7 %"
        assert row_all[6] == "100,0 %"
        assert row_all[7] == "2023-01 (0,0500)"
        assert row_all[8] == "0,1000"
        assert row_all[9] == "1"
        assert row_all[10] == "—"

        row_eu = [cell.strip() for cell in lines[3].strip("|").split("|")]
        assert row_eu[1] == "EU"
        assert row_eu[2] == "1"
        assert row_eu[3] == "-0,1000"
        assert row_eu[4] == "—"
        assert row_eu[5] == "0,0 %"
        assert row_eu[6] == "0,0 %"
        assert row_eu[7] == "2023-01 (-0,1000)"
        assert row_eu[8] == "0,1000"
        assert row_eu[9] == "1"
        assert row_eu[10] == "—"

        assert "-0,0000" not in md

    def test_markdown_normaliza_cero_negativo_y_tolera_tabla_sin_meses(self) -> None:
        # Media = -1e-9 se redondea a «0,0000» (nunca «-0,0000») y la ausencia de meses da «—».
        table = {"m1": {"all": ics.stability([-1e-9, -1e-9], window=2)}}
        md = ics.markdown(table)
        row = [cell.strip() for cell in md.splitlines()[2].strip("|").split("|")]
        assert row[3] == "0,0000"
        assert row[6] == "—"
        assert row[7] == "—"
        assert row[10] == "0,0000"
        assert "-0,0000" not in md

    def test_main_imprime_markdown_y_escribe_json(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # Usamos nombres con tilde para comprobar ensure_ascii=False.
        csv_path = tmp_path / "entrada.csv"
        csv_path.write_text(
            "session,regresión,árbol\n"
            "2023:US:2023-01-02,0.1,-0.2\n"
            "2023:US:2023-01-03,0.3,0.0\n"
            "2023:US:2023-02-01,-0.1,0.4\n",
            encoding="utf-8",
        )
        json_path = tmp_path / "salida.json"

        code = ics.main([str(csv_path), "--window", "2", "--json", str(json_path)])

        assert code == 0
        captured = capsys.readouterr()
        expected_table = ics.stability_table(csv_path, window=2)
        assert captured.out.strip() == ics.markdown(expected_table).strip()
        assert captured.err == ""

        raw_json = json_path.read_text(encoding="utf-8")
        # Comprobamos ensure_ascii=False (tildes literales, sin escapes \uXXXX) e indent=1.
        assert '"regresión"' in raw_json
        assert '"árbol"' in raw_json
        assert "\\u00f3" not in raw_json
        assert '\n "regresión": {' in raw_json
        assert json.loads(raw_json) == expected_table

    def test_main_sin_json_no_crea_archivo_extra(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        csv_path = tmp_path / "solo_md.csv"
        csv_path.write_text(
            "session,m1\n2023:US:2023-01-02,0.25\n2023:US:2023-01-03,0.75\n",
            encoding="utf-8",
        )

        code = ics.main([str(csv_path)])

        assert code == 0
        captured = capsys.readouterr()
        assert "| Modelo |" in captured.out
        assert "0,5000" in captured.out
        assert list(tmp_path.iterdir()) == [csv_path]

    def test_main_devuelve_error_con_entrada_invalida(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        csv_path = tmp_path / "roto.csv"
        csv_path.write_text("session,m1\n", encoding="utf-8")

        code = ics.main([str(csv_path)])

        assert code != 0
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err.strip() != ""

    def test_ejecucion_como_script_cli(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "cli.csv"
        csv_path.write_text(
            "session,m1\n2023:US:2023-01-02,0.25\n2023:US:2023-01-03,0.75\n",
            encoding="utf-8",
        )
        json_path = tmp_path / "cli.json"
        script_path = Path(ics.__file__).resolve()

        proc = subprocess.run(
            [
                sys.executable,
                str(script_path),
                str(csv_path),
                "--window",
                "2",
                "--json",
                str(json_path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        assert proc.returncode == 0
        assert proc.stderr == ""
        assert "| m1 | all | 2 | 0,5000 |" in proc.stdout
        assert json.loads(json_path.read_text(encoding="utf-8"))["m1"]["all"]["mean"] == 0.5
