"""Inferencia entre modelos con IC por sesión: propiedades estadísticas, determinismo y órdenes."""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/dgx_ic_inference.py"
SPEC = importlib.util.spec_from_file_location("dgx_ic_inference", SCRIPT)
ic = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ic)


def _simular_ar1(
    n_obs: int,
    phi: float,
    media: float = 0.0,
    desv_innov: float = 1.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Genera una serie estacionaria AR(1) con media dada y periodo de calentamiento."""
    generador = np.random.default_rng(0) if rng is None else rng
    calentamiento = 200
    innov = generador.normal(0.0, desv_innov, size=n_obs + calentamiento)
    serie = np.empty(n_obs + calentamiento, dtype=np.float64)
    serie[0] = innov[0] / math.sqrt(max(1.0 - phi * phi, 1e-6))
    for t in range(1, n_obs + calentamiento):
        serie[t] = phi * serie[t - 1] + innov[t]
    return serie[calentamiento:] + media


def _t_simple(x: np.ndarray) -> float:
    """Estadístico t clásico de una muestra con ddof=1."""
    return float(np.mean(x) / (np.std(x, ddof=1) / math.sqrt(x.size)))


def test_newey_west_iid_y_ar1() -> None:
    """Newey-West coincide casi con el t simple en i.i.d. y se reduce mucho en AR(1) 0,8."""
    rng = np.random.default_rng(101)
    serie_iid = rng.normal(loc=0.15, scale=1.0, size=1000)
    t_simple_iid = _t_simple(serie_iid)
    t_nw_iid = ic.newey_west_t(serie_iid)

    assert math.isfinite(t_nw_iid)
    assert abs(t_nw_iid - t_simple_iid) / abs(t_simple_iid) < 0.15

    # Con lags=0 la relación con t simple es exactamente sqrt(T / (T - 1)).
    t_nw_cero = ic.newey_west_t(serie_iid, lags=0)
    esperado_cero = t_simple_iid * math.sqrt(serie_iid.size / (serie_iid.size - 1))
    assert math.isclose(t_nw_cero, esperado_cero, rel_tol=1e-12)

    # En una serie AR(1) persistente (phi=0,8), la varianza de largo plazo es mucho mayor.
    serie_ar1 = _simular_ar1(n_obs=1000, phi=0.8, media=0.4, desv_innov=0.6, rng=rng)
    t_simple_ar1 = _t_simple(serie_ar1)
    t_nw_ar1 = ic.newey_west_t(serie_ar1)

    assert abs(t_nw_ar1) < 0.65 * abs(t_simple_ar1)
    assert math.copysign(1.0, t_nw_ar1) == math.copysign(1.0, t_simple_ar1)


def test_optimal_block_length_crece_con_persistencia() -> None:
    """La longitud óptima de bloque crece notablemente de i.i.d. a AR(1) con phi=0,8."""
    rng = np.random.default_rng(202)
    bloques_iid = []
    bloques_ar1 = []
    for _ in range(10):
        x_iid = rng.standard_normal(600)
        x_ar1 = _simular_ar1(n_obs=600, phi=0.8, rng=rng)
        b_iid = ic.optimal_block_length(x_iid)
        b_ar1 = ic.optimal_block_length(x_ar1)
        assert b_iid >= 1.0
        assert b_ar1 >= 1.0
        bloques_iid.append(b_iid)
        bloques_ar1.append(b_ar1)

    assert float(np.median(bloques_iid)) < 4.0
    assert float(np.median(bloques_ar1)) > 10.0
    assert float(np.mean(bloques_ar1)) > 4.0 * float(np.mean(bloques_iid))


def test_stationary_bootstrap_indices_forma_rango_y_longitud_media() -> None:
    """Los índices tienen forma [B, T], están en [0, T-1] y reproducen la longitud media."""
    n_obs = 500
    n_boot = 800
    bloque = 8.0
    rng = np.random.default_rng(303)

    indices = ic.stationary_bootstrap_indices(T=n_obs, block=bloque, B=n_boot, rng=rng)
    assert indices.shape == (n_boot, n_obs)
    assert np.issubdtype(indices.dtype, np.integer)
    assert int(indices.min()) >= 0
    assert int(indices.max()) <= n_obs - 1

    # Un salto ocurre cuando idx[t] != (idx[t-1] + 1) % T.
    # La probabilidad teórica de salto visible es (1 / block) * (1 - 1 / T).
    saltos = indices[:, 1:] != ((indices[:, :-1] + 1) % n_obs)
    prob_salto_obs = float(np.mean(saltos))
    bloque_obs = (1.0 - 1.0 / n_obs) / prob_salto_obs
    assert abs(bloque_obs - bloque) / bloque < 0.08

    # También verificamos la longitud media directa de las rachas circulares observadas.
    num_bloques = float(np.sum(saltos)) + n_boot
    longitud_racha_media = (n_boot * n_obs) / num_bloques
    assert abs(longitud_racha_media - bloque) / bloque < 0.10


def test_cobertura_intervalo_bootstrap_media_iid() -> None:
    """La cobertura empírica del intervalo al 95 % en 200 réplicas i.i.d. cae en [0,88, 0,99]."""
    rng = np.random.default_rng(404)
    media_verdadera = 0.05
    n_sims = 200
    aciertos = 0

    for sim in range(n_sims):
        muestra = rng.normal(loc=media_verdadera, scale=0.2, size=120)
        res = ic.bootstrap_mean_interval(muestra, level=0.95, B=500, block=None, seed=sim)
        assert set(res.keys()) == {"mean", "low", "high", "block"}
        assert res["low"] < res["mean"] < res["high"]
        assert res["block"] >= 1.0
        if res["low"] <= media_verdadera <= res["high"]:
            aciertos += 1

    cobertura = aciertos / n_sims
    assert 0.88 <= cobertura <= 0.99


def test_diebold_mariano_nula_y_alternativa() -> None:
    """DM no rechaza en la mayoría bajo medias iguales y rechaza con diferencia clara."""
    rng = np.random.default_rng(505)
    no_rechazos = 0
    n_reps = 60
    for _ in range(n_reps):
        comun = rng.normal(0.02, 0.1, size=250)
        a = comun + rng.normal(0.0, 0.05, size=250)
        b = comun + rng.normal(0.0, 0.05, size=250)
        res = ic.diebold_mariano(a, b)
        assert set(res.keys()) == {"mean", "t", "p_value"}
        assert 0.0 <= res["p_value"] <= 1.0
        if res["p_value"] > 0.05:
            no_rechazos += 1

    assert no_rechazos / n_reps >= 0.85

    # Diferencia grande y sostenida entre modelos.
    a_mejor = rng.normal(0.08, 0.05, size=250)
    b_peor = rng.normal(0.02, 0.05, size=250)
    res_alt = ic.diebold_mariano(a_mejor, b_peor)
    assert res_alt["mean"] > 0.04
    assert res_alt["t"] > 5.0
    assert res_alt["p_value"] < 1e-6

    # Series idénticas no producen NaN y devuelven p_value = 1.
    res_igual = ic.diebold_mariano(a_mejor, a_mejor)
    assert res_igual["mean"] == 0.0
    assert res_igual["t"] == 0.0
    assert res_igual["p_value"] == 1.0


def test_spa_nula_y_alternativa() -> None:
    """SPA da p_consistent alto bajo H0 y < 0,05 cuando un candidato supera claramente."""
    rng = np.random.default_rng(606)
    n_obs = 300
    factor_comun = _simular_ar1(n_obs=n_obs, phi=0.4, media=0.03, desv_innov=0.04, rng=rng)
    benchmark = factor_comun + rng.normal(0.0, 0.02, size=n_obs)

    # Bajo H0: candidatos con media igual o inferior a la referencia.
    desplazamientos_h0 = np.array([0.0, -0.005, -0.015, -0.03, -0.05])
    cand_h0 = (
        factor_comun[:, np.newaxis]
        + desplazamientos_h0[np.newaxis, :]
        + rng.normal(0.0, 0.02, size=(n_obs, desplazamientos_h0.size))
    )
    res_h0 = ic.spa_test(cand_h0, benchmark, B=1000, block=None, seed=11)
    assert res_h0["p_lower"] <= res_h0["p_consistent"] <= res_h0["p_upper"]
    assert res_h0["p_consistent"] > 0.20

    # Bajo H1: el candidato 2 supera claramente a la referencia.
    desplazamientos_h1 = np.array([-0.02, 0.0, 0.025, -0.01, -0.03])
    cand_h1 = (
        factor_comun[:, np.newaxis]
        + desplazamientos_h1[np.newaxis, :]
        + rng.normal(0.0, 0.02, size=(n_obs, desplazamientos_h1.size))
    )
    res_h1 = ic.spa_test(cand_h1, benchmark, B=1000, block=None, seed=11)
    assert res_h1["p_lower"] <= res_h1["p_consistent"] <= res_h1["p_upper"]
    assert res_h1["p_consistent"] < 0.05
    assert res_h1["best_candidate"] == 2
    assert res_h1["statistic"] > 2.0


def test_mcs_conserva_mejor_elimina_peor_y_mantiene_iguales() -> None:
    """MCS conserva el mejor modelo, elimina uno peor y conserva casi todos si son iguales."""
    rng = np.random.default_rng(707)
    n_obs = 350
    factor = _simular_ar1(n_obs=n_obs, phi=0.3, media=0.02, desv_innov=0.05, rng=rng)

    # Escenario 1: modelo 0 es claramente el mejor y modelo 3 es claramente el peor.
    medias = np.array([0.045, 0.040, 0.038, -0.010, 0.015])
    scores = (
        factor[:, np.newaxis]
        + medias[np.newaxis, :]
        + rng.normal(0.0, 0.025, size=(n_obs, medias.size))
    )
    res = ic.model_confidence_set(scores, alpha=0.10, B=1000, block=None, seed=22)

    assert 0 in res["included"]
    assert 3 not in res["included"]
    assert res["order"][0] == 3
    assert len(res["order"]) == medias.size
    assert set(res["order"]) == set(range(medias.size))
    assert len(res["p_values"]) == medias.size
    assert res["p_values"][res["order"][-1]] == 1.0

    # Los p-valores MCS son monótonos no decrecientes en el orden de eliminación.
    p_ordenados = [res["p_values"][i] for i in res["order"]]
    assert all(p_ordenados[k] <= p_ordenados[k + 1] + 1e-12 for k in range(len(p_ordenados) - 1))
    assert res["included"] == [i for i, p in enumerate(res["p_values"]) if p >= 0.10]

    # Escenario 2: modelos con la misma esperanza.
    n_iguales = 6
    scores_iguales = factor[:, np.newaxis] + rng.normal(0.0, 0.03, size=(n_obs, n_iguales))
    res_iguales = ic.model_confidence_set(scores_iguales, alpha=0.10, B=1000, block=None, seed=22)
    assert len(res_iguales["included"]) >= n_iguales - 1


def test_determinismo_misma_semilla() -> None:
    """Con la misma semilla todas las funciones estocásticas dan resultados idénticos."""
    rng = np.random.default_rng(808)
    n_obs = 160
    bench = rng.normal(0.02, 0.05, size=n_obs)
    cands = rng.normal(0.02, 0.05, size=(n_obs, 4))

    idx_a = ic.stationary_bootstrap_indices(n_obs, 5.0, 200, np.random.default_rng(42))
    idx_b = ic.stationary_bootstrap_indices(n_obs, 5.0, 200, np.random.default_rng(42))
    assert np.array_equal(idx_a, idx_b)

    ci_a = ic.bootstrap_mean_interval(bench, level=0.95, B=400, block=None, seed=99)
    ci_b = ic.bootstrap_mean_interval(bench, level=0.95, B=400, block=None, seed=99)
    assert ci_a == ci_b

    spa_a = ic.spa_test(cands, bench, B=400, block=None, seed=99)
    spa_b = ic.spa_test(cands, bench, B=400, block=None, seed=99)
    assert spa_a == spa_b

    mcs_a = ic.model_confidence_set(cands, alpha=0.10, B=400, block=None, seed=99)
    mcs_b = ic.model_confidence_set(cands, alpha=0.10, B=400, block=None, seed=99)
    assert mcs_a == mcs_b


def test_validacion_entradas() -> None:
    """Las entradas mal formadas, no finitas o demasiado cortas lanzan ValueError."""
    with pytest.raises(ValueError):
        ic.newey_west_t([1.0])
    with pytest.raises(ValueError):
        ic.newey_west_t([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(ValueError):
        ic.newey_west_t([1.0, np.nan, 2.0])
    with pytest.raises(ValueError):
        ic.newey_west_t([1.0, 2.0, 3.0], lags=3)

    with pytest.raises(ValueError):
        ic.optimal_block_length([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        ic.optimal_block_length([1.0, 2.0, np.inf, 4.0])

    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        ic.stationary_bootstrap_indices(0, 2.0, 10, rng)
    with pytest.raises(ValueError):
        ic.stationary_bootstrap_indices(10, 0.5, 10, rng)
    with pytest.raises(ValueError):
        ic.stationary_bootstrap_indices(10, 2.0, 0, rng)

    with pytest.raises(ValueError):
        ic.bootstrap_mean_interval([1.0, 2.0, 3.0, 4.0], level=1.0, B=100)
    with pytest.raises(ValueError):
        ic.diebold_mariano([1.0, 2.0, 3.0], [1.0, 2.0])

    with pytest.raises(ValueError):
        ic.spa_test(np.ones((10, 2)), np.ones(9), B=100)
    with pytest.raises(ValueError):
        ic.spa_test(np.ones(10), np.ones(10), B=100)

    with pytest.raises(ValueError):
        ic.model_confidence_set(np.ones((10, 2)), alpha=0.0, B=100)
    with pytest.raises(ValueError):
        ic.model_confidence_set(np.ones((3, 2)), alpha=0.10, B=100)


def test_cli_genera_json_completo(tmp_path: Path) -> None:
    """La CLI procesa un CSV sintético y escribe un JSON con todas las claves esperadas."""
    rng = np.random.default_rng(909)
    n_obs = 120
    factor = rng.normal(0.01, 0.04, size=n_obs)
    datos = {
        "ref": factor + rng.normal(0.0, 0.02, size=n_obs),
        "modelo_bueno": factor + 0.03 + rng.normal(0.0, 0.02, size=n_obs),
        "modelo_malo": factor - 0.03 + rng.normal(0.0, 0.02, size=n_obs),
    }

    ruta_csv = tmp_path / "tabla.csv"
    with ruta_csv.open("w", encoding="utf-8", newline="") as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow(["session", "ref", "modelo_bueno", "modelo_malo"])
        for t in range(n_obs):
            escritor.writerow(
                [
                    f"2025-01-{t + 1:03d}",
                    f"{datos['ref'][t]:.8f}",
                    f"{datos['modelo_bueno'][t]:.8f}",
                    f"{datos['modelo_malo'][t]:.8f}",
                ]
            )

    ruta_json = tmp_path / "resultado.json"
    script = SCRIPT
    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            str(ruta_csv),
            "--benchmark",
            "ref",
            "--output",
            str(ruta_json),
            "--alpha",
            "0.10",
            "--bootstrap",
            "500",
            "--seed",
            "7",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert ruta_json.is_file()

    salida = json.loads(ruta_json.read_text(encoding="utf-8"))
    assert set(salida.keys()) == {"metadata", "models", "spa", "mcs"}
    assert set(salida["models"].keys()) == {"ref", "modelo_bueno", "modelo_malo"}

    for nombre in ("ref", "modelo_bueno", "modelo_malo"):
        info = salida["models"][nombre]
        assert {
            "mean",
            "t_simple",
            "t_newey_west",
            "bootstrap_ci_95",
            "diebold_mariano",
        }.issubset(info.keys())
        ci = info["bootstrap_ci_95"]
        assert set(ci.keys()) == {"mean", "low", "high", "block"}
        assert ci["low"] < ci["mean"] < ci["high"]
        dm = info["diebold_mariano"]
        assert set(dm.keys()) == {"mean", "t", "p_value"}

    # El bloque compartido es idéntico en todos los modelos (remuestreo simultáneo).
    bloques = {salida["models"][m]["bootstrap_ci_95"]["block"] for m in salida["models"]}
    assert len(bloques) == 1

    spa = salida["spa"]
    assert {
        "statistic",
        "p_lower",
        "p_consistent",
        "p_upper",
        "best_candidate",
    }.issubset(spa.keys())
    assert spa["best_candidate"] == "modelo_bueno"
    assert spa["p_consistent"] < 0.05

    mcs = salida["mcs"]
    assert {"included", "order", "p_values"}.issubset(mcs.keys())
    assert "modelo_bueno" in mcs["included"]
    assert "modelo_malo" not in mcs["included"]
    assert set(mcs["p_values"].keys()) == {"ref", "modelo_bueno", "modelo_malo"}


def test_preservacion_correlacion_contemporanea() -> None:
    """El remuestreo conjunto cancela ruido común grande y detecta diferencias pequeñas."""
    rng = np.random.default_rng(1010)
    n_obs = 250
    ruido_comun = rng.normal(0.0, 1.0, size=n_obs)
    benchmark = ruido_comun + rng.normal(0.0, 0.01, size=n_obs)
    cands = np.column_stack(
        [
            ruido_comun + 0.05 + rng.normal(0.0, 0.01, size=n_obs),
            ruido_comun - 0.05 + rng.normal(0.0, 0.01, size=n_obs),
        ]
    )

    res_spa = ic.spa_test(cands, benchmark, B=600, block=4.0, seed=5)
    assert res_spa["best_candidate"] == 0
    assert res_spa["p_consistent"] < 0.01

    scores = np.column_stack([benchmark, cands])
    res_mcs = ic.model_confidence_set(scores, alpha=0.10, B=600, block=4.0, seed=5)
    assert res_mcs["included"] == [1]
    assert res_mcs["order"][0] == 2


def test_casos_borde_y_errores_csv(tmp_path: Path) -> None:
    """Comprueba casos degenerados (constantes, M=1, todos peores) y errores de CSV."""
    constante_cero = np.zeros(20)
    assert ic.newey_west_t(constante_cero) == 0.0
    assert ic.optimal_block_length(constante_cero) == 1.0
    assert math.isinf(ic.newey_west_t(np.full(20, 0.5)))

    # SPA con todos los candidatos peores que la referencia: estadístico 0 y p-valores 1.
    rng = np.random.default_rng(1111)
    ref = rng.normal(0.10, 0.02, size=80)
    peores = rng.normal(-0.05, 0.02, size=(80, 3))
    res_spa = ic.spa_test(peores, ref, B=300, block=3.0, seed=0)
    assert res_spa["statistic"] == 0.0
    assert res_spa["p_lower"] == 1.0
    assert res_spa["p_consistent"] == 1.0
    assert res_spa["p_upper"] == 1.0

    # MCS con un solo modelo y con modelos idénticos.
    res_uno = ic.model_confidence_set(ref[:, np.newaxis], alpha=0.10, B=100, seed=0)
    assert res_uno == {"included": [0], "order": [0], "p_values": [1.0]}

    iguales = np.column_stack([ref, ref, ref])
    res_ig = ic.model_confidence_set(iguales, alpha=0.10, B=200, block=2.0, seed=0)
    assert res_ig["included"] == [0, 1, 2]
    assert all(p == 1.0 for p in res_ig["p_values"])

    # CSV con hueco vacío y CSV sin columna benchmark deben lanzar ValueError.
    csv_hueco = tmp_path / "hueco.csv"
    csv_hueco.write_text(
        "session,ref,m1\ns1,0.1,0.2\ns2,,0.3\ns3,0.1,0.2\ns4,0.2,0.3\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        ic.analizar_tabla_ic(csv_hueco, benchmark="ref", bootstrap=100, seed=0)

    csv_ok = tmp_path / "valido.csv"
    csv_ok.write_text(
        "session,ref,m1\ns1,0.1,0.2\ns2,0.2,0.3\ns3,0.1,0.2\ns4,0.2,0.3\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        ic.analizar_tabla_ic(csv_ok, benchmark="inexistente", bootstrap=100, seed=0)
