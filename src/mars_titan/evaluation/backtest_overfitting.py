"""Sobreajuste del backtest y Sharpe deflactado de la cartera larga y corta (#794).

Las fórmulas son las de Bailey y López de Prado (2014) y Bailey, Borwein, López de Prado y
Zhu (2017), sin cambios:

- PSR(SR*) = Z[(SR̂ − SR*)·√(T − 1) / √(1 − γ̂₃·SR̂ + (γ̂₄ − 1)/4·SR̂²)], la probabilidad de
  que el Sharpe verdadero supere SR* dada la longitud, la asimetría γ̂₃ y la curtosis γ̂₄ (sin
  restar 3) de la serie.
- SR₀ = √V[{SR̂ₙ}]·((1 − γ)·Z⁻¹[1 − 1/N] + γ·Z⁻¹[1 − 1/(N·e)]), el máximo esperado de N
  ensayos independientes con Sharpe verdadero nulo y varianza V entre ensayos, con γ la
  constante de Euler y Mascheroni.
- DSR = PSR(SR₀).
- PBO por validación cruzada combinatoria simétrica (CSCV): las T sesiones se parten en S
  bloques contiguos iguales y, en cada una de las C(S, S/2) mitades, se elige la estrategia
  de mayor Sharpe dentro y se sitúa su Sharpe fuera entre los de todas con el rango medio en
  los empates. Con ω = rango / (N + 1), λ = ln(ω / (1 − ω)) y PBO es la fracción de mitades
  con λ ≤ 0. También se publican la regresión del Sharpe fuera sobre el de dentro y la
  probabilidad de que el elegido pierda fuera.

Los Sharpe son por sesión, sin anualizar: media sobre desviación típica con ddof = 1. La
asimetría y la curtosis son los momentos de la muestra sin corrección, como en el código de
López de Prado. Las mitades de CSCV mezclan bloques de fechas distintas y no respetan el
orden temporal. Por eso estas medidas son un diagnóstico complementario del contraste por
bloques de la comparación y no prueban rentabilidad futura.

`analyze_long_short` aplica la declaración de `configs/evaluation/backtest-overfitting-v1.json`
a un informe de `long_short_comparison`: en cada mercado y coste, las estrategias son los
modelos de la comparación con las semillas promediadas sesión a sesión.
"""

import argparse
import itertools
import math
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
import scipy
from scipy import stats

from mars_titan.data.cohort_files import read_manifest, safe_destination
from mars_titan.data.storage import atomic_json, sha256

KIND = "backtest_overfitting_report"
DECLARATION_KIND = "backtest_overfitting_declaration"
SOURCE_KIND = "long_short_comparison_report"
EULER_MASCHERONI = 0.5772156649015329
# Con 16 bloques hay 12.870 mitades. Más bloques multiplican el coste sin cambiar el método.
MAX_BLOCKS = 16
MAX_STRATEGIES = 1024
# Mitades que se puntúan a la vez, para acotar la memoria con muchas estrategias.
CHUNK = 512
# Lo que el informe publica de CSCV. Los vectores por mitad no se guardan.
SUMMARY = (
    "blocks",
    "sessions_per_block",
    "combinations",
    "strategies",
    "pbo",
    "probability_of_loss",
    "degradation",
)
QUANTILES = ("p05", "p25", "p50", "p75", "p95")
DECLARED = dict(
    schema_version=1,
    kind=DECLARATION_KIND,
    identity="backtest_overfitting_v1",
    source=SOURCE_KIND,
    statistic="session_sharpe_ddof_1",
    seeds="mean_per_session",
    trim="earliest_sessions",
    constant_series="excluded_and_reported",
    final_test_access=False,
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _finite(value, name):
    _require(
        type(value) in (int, float) and math.isfinite(value), f"{name} debe ser un número finito"
    )
    return float(value)


def _count(value, low, name):
    _require(type(value) is int and value >= low, f"{name} debe ser un entero de al menos {low}")
    return value


def sharpe_statistics(returns):
    """Sharpe por sesión, asimetría, curtosis y longitud de una serie de rendimientos."""
    values = np.asarray(returns, dtype=np.float64)
    _require(
        values.ndim == 1 and len(values) >= 3 and bool(np.all(np.isfinite(values))),
        "La serie necesita al menos tres rendimientos finitos",
    )
    deviation = float(values.std(ddof=1))
    _require(deviation > 0, "La serie no varía y su Sharpe no está definido")
    return dict(
        sharpe=float(values.mean()) / deviation,
        skewness=float(stats.skew(values)),
        kurtosis=float(stats.kurtosis(values, fisher=False)),
        observations=len(values),
    )


def probabilistic_sharpe(sharpe, threshold, *, observations, skewness, kurtosis):
    """PSR: probabilidad de que el Sharpe verdadero supere `threshold`."""
    sharpe = _finite(sharpe, "El Sharpe")
    threshold = _finite(threshold, "El umbral")
    observations = _count(observations, 2, "La longitud de la serie")
    skewness = _finite(skewness, "La asimetría")
    kurtosis = _finite(kurtosis, "La curtosis")
    _require(kurtosis >= 1, "La curtosis sin restar 3 no puede ser menor que 1")
    spread = 1 - skewness * sharpe + (kurtosis - 1) / 4 * sharpe**2
    _require(spread > 0, "La asimetría y la curtosis no dan una varianza positiva del Sharpe")
    statistic = (sharpe - threshold) * math.sqrt(observations - 1) / math.sqrt(spread)
    return float(stats.norm.cdf(statistic))


def expected_maximum_sharpe(variance, trials):
    """SR₀: Sharpe máximo esperado de `trials` ensayos independientes con Sharpe nulo."""
    variance = _finite(variance, "La varianza entre ensayos")
    _require(variance >= 0, "La varianza entre ensayos no puede ser negativa")
    trials = _count(trials, 2, "El número de ensayos")
    gamma = EULER_MASCHERONI
    maximum = (1 - gamma) * stats.norm.ppf(1 - 1 / trials) + gamma * stats.norm.ppf(
        1 - 1 / (trials * math.e)
    )
    return math.sqrt(variance) * float(maximum)


def deflated_sharpe(sharpe, *, variance, trials, observations, skewness, kurtosis):
    """DSR: PSR con el umbral SR₀ de los ensayos realizados."""
    threshold = expected_maximum_sharpe(variance, trials)
    probability = probabilistic_sharpe(
        sharpe, threshold, observations=observations, skewness=skewness, kurtosis=kurtosis
    )
    return dict(threshold=threshold, probability=probability)


def halves(blocks):
    """Mitades de CSCV en orden lexicográfico: los bloques de dentro de cada una."""
    _require(
        type(blocks) is int and 2 <= blocks <= MAX_BLOCKS and blocks % 2 == 0,
        f"CSCV necesita un número par de bloques entre 2 y {MAX_BLOCKS}",
    )
    return np.array(list(itertools.combinations(range(blocks), blocks // 2)), dtype=np.int64)


def _block_sharpes(centered, grand, squares, size, inside):
    """Sharpe de cada estrategia en la unión de los bloques marcados de cada mitad.

    Combina las medias y los momentos centrados de cada bloque sin volver a recorrer las
    filas. Como los bloques son iguales, la media es la de las medias, y el momento de
    segundo orden suma el de cada bloque y la dispersión de sus medias,
    Σ (m_b − μ)² = Σ m_b² − k·μ² sobre los k bloques. Las medias llegan centradas en la de
    toda la serie para que esa resta no pierda precisión, y así todo son productos de
    matrices sin temporales de [mitades, bloques, estrategias].
    """
    weights = inside.astype(np.float64)
    count = weights.sum(axis=1, keepdims=True)
    mean = weights @ centered / count
    spread = weights @ centered**2 - count * mean**2
    variance = (weights @ squares + size * spread) / (count * size - 1)
    _require(bool(np.all(variance > 0)), "Una estrategia no varía en una mitad de CSCV")
    return (mean + grand) / np.sqrt(variance)


def pbo_cscv(returns, *, blocks):
    """Probabilidad de sobreajuste del backtest con CSCV sobre [sesiones, estrategias]."""
    values = np.asarray(returns, dtype=np.float64)
    combinations = halves(blocks)
    _require(
        values.ndim == 2
        and 2 <= values.shape[1] <= MAX_STRATEGIES
        and bool(np.all(np.isfinite(values))),
        f"CSCV necesita entre 2 y {MAX_STRATEGIES} estrategias con rendimientos finitos",
    )
    sessions, strategies = values.shape
    _require(
        sessions % blocks == 0 and sessions // blocks >= 2,
        "Las sesiones deben repartirse en bloques iguales de al menos dos sesiones",
    )
    size = sessions // blocks
    split = values.reshape(blocks, size, strategies)
    means = split.mean(axis=1)
    squares = ((split - means[:, None]) ** 2).sum(axis=1)
    grand = means.mean(axis=0)
    centered = means - grand
    inside = np.zeros((len(combinations), blocks), dtype=bool)
    inside[np.arange(len(combinations))[:, None], combinations] = True
    chosen, within, outside, logits = [], [], [], []
    for start in range(0, len(combinations), CHUNK):
        part = inside[start : start + CHUNK]
        train = _block_sharpes(centered, grand, squares, size, part)
        test = _block_sharpes(centered, grand, squares, size, ~part)
        # El primer máximo, para que el resultado no dependa de nada fuera de los datos.
        best = train.argmax(axis=1)
        rows = np.arange(len(part))
        score = test[rows, best]
        rank = (
            1 + (test < score[:, None]).sum(axis=1) + ((test == score[:, None]).sum(axis=1) - 1) / 2
        )
        omega = rank / (strategies + 1)
        chosen.append(best)
        within.append(train[rows, best])
        outside.append(score)
        logits.append(np.log(omega / (1 - omega)))
    chosen, within, outside, logits = (
        np.concatenate(items) for items in (chosen, within, outside, logits)
    )
    degradation = None
    if np.ptp(within) > 0:
        slope, intercept = np.polyfit(within, outside, 1)
        degradation = dict(slope=float(slope), intercept=float(intercept))
    return dict(
        blocks=blocks,
        sessions_per_block=size,
        combinations=len(combinations),
        strategies=strategies,
        pbo=float(np.mean(logits <= 0)),
        probability_of_loss=float(np.mean(outside < 0)),
        degradation=degradation,
        logits=logits,
        chosen=chosen,
        in_sample=within,
        out_of_sample=outside,
    )


def load_declaration(path):
    """Leer la declaración previa y exigir que describa lo que mide este módulo."""
    declaration, digest = read_manifest(Path(path), 1024**2)
    _require(isinstance(declaration, dict), "La declaración debe ser un objeto JSON")
    _require(
        {key: declaration.get(key) for key in DECLARED} == DECLARED,
        "La declaración no coincide con el análisis que se mide",
    )
    halves(declaration.get("blocks"))
    sensitivity = declaration.get("trials_sensitivity")
    _require(
        isinstance(sensitivity, list)
        and all(type(value) is int and value >= 2 for value in sensitivity)
        and sensitivity == sorted(set(sensitivity)),
        "La sensibilidad del número de ensayos es una lista creciente de enteros de al menos 2",
    )
    provenance = declaration.get("trials_provenance")
    _require(
        isinstance(provenance, dict)
        and set(provenance) == {"comparison", *map(str, sensitivity)}
        and all(isinstance(text, str) and text for text in provenance.values()),
        "Cada número de ensayos necesita su procedencia",
    )
    return declaration, digest


def _source(folder):
    """Informe de la cartera terminado, con sus sesiones verificadas por huella."""
    report, digest = read_manifest(folder / "long_short.json", 64 * 1024**2)
    _require(
        report.get("kind") == SOURCE_KIND and report.get("status") == "completed",
        "La fuente no es un informe terminado de la cartera larga y corta",
    )
    _require(report.get("final_test_opened") is False, "La fuente abre la reserva final")
    path = folder / "sessions.parquet"
    _require(
        report.get("artifacts", {}).get("sessions.parquet") == sha256(path),
        "Las sesiones no corresponden al informe de la cartera",
    )
    return report, digest, pq.read_table(path)


def _strategies(table, market, cost, arms):
    """Rendimientos netos [sesiones, modelos] de un mercado y un coste.

    Las semillas de cada modelo se promedian sesión a sesión en orden ascendente, que es el
    orden en que las declaran las comparaciones, y cada una tiene que definir una vez cada
    sesión del mercado. Un modelo determinista tiene una sola semilla nula. Las filas se
    ordenan por fecha dentro de cada semilla, así que el orden del archivo no cambia nada.
    """
    rows = table.filter(pc.equal(table["market"], market))
    times = rows["prediction_at"].to_numpy()
    sessions = np.unique(times)
    net = rows[f"net_return_{cost}bps"].to_numpy()
    groups = {}
    for index, key in enumerate(
        zip(rows["arm"].to_pylist(), rows["seed"].to_pylist(), strict=True)
    ):
        groups.setdefault(key, []).append(index)
    columns = []
    for arm in arms:
        seeds = sorted(seed for name, seed in groups if name == arm)
        series = []
        for seed in seeds:
            index = np.array(groups[arm, seed])
            index = index[np.argsort(times[index], kind="stable")]
            _require(
                np.array_equal(times[index], sessions),
                f"{arm} no define una vez todas las sesiones de {market}",
            )
            series.append(net[index])
        _require(bool(series), f"{arm} no tiene sesiones en {market}")
        columns.append(np.mean(series, axis=0))
    return sessions, np.stack(columns, axis=1)


def _microseconds(moment):
    return int(np.datetime64(moment, "us").astype(np.int64))


def _market(declaration, sessions, returns, arms):
    """CSCV y DSR de los modelos de un mercado y un coste."""
    blocks = declaration["blocks"]
    # Se descartan las sesiones más antiguas que no llenan un bloque.
    trimmed = len(sessions) % blocks
    # Un modelo cuya cartera no varía, como el control cero, que no abre posiciones, no
    # tiene Sharpe. Queda fuera del PBO y del DSR y el informe lo nombra.
    varies = np.ptp(returns[trimmed:], axis=0) > 0
    kept = [arm for arm, keep in zip(arms, varies, strict=True) if keep]
    _require(len(kept) >= 2, "Hacen falta al menos dos modelos cuya cartera varíe")
    values = returns[trimmed:, varies]
    overfitting = pbo_cscv(values, blocks=blocks)
    statistics = {arm: sharpe_statistics(values[:, i]) for i, arm in enumerate(kept)}
    sharpes = np.array([statistics[arm]["sharpe"] for arm in kept])
    variance = float(sharpes.var(ddof=1))
    trials = sorted({len(kept), *declaration["trials_sensitivity"]})
    strategies = {}
    for arm in kept:
        item = statistics[arm]
        strategies[arm] = dict(
            item,
            deflated={
                str(count): deflated_sharpe(
                    item["sharpe"],
                    variance=variance,
                    trials=count,
                    observations=item["observations"],
                    skewness=item["skewness"],
                    kurtosis=item["kurtosis"],
                )
                for count in trials
            },
        )
    chosen, counts = np.unique(overfitting["chosen"], return_counts=True)
    logits = np.quantile(overfitting["logits"], [0.05, 0.25, 0.5, 0.75, 0.95])
    cscv = {key: overfitting[key] for key in SUMMARY}
    return dict(
        sessions=len(values),
        trimmed_earliest_sessions=trimmed,
        first_session_us=_microseconds(sessions[trimmed]),
        last_session_us=_microseconds(sessions[-1]),
        excluded_without_variation=[arm for arm in arms if arm not in kept],
        variance_of_the_sharpes=variance,
        trials=trials,
        strategies=strategies,
        selected_in_full_sample=kept[int(np.argmax(sharpes))],
        cscv=dict(
            cscv,
            logit_quantiles=dict(zip(QUANTILES, logits.tolist(), strict=True)),
            chosen_in_sample={kept[i]: int(n) for i, n in zip(chosen, counts, strict=True)},
        ),
    )


def analyze_long_short(folder, declaration_path):
    """PBO y DSR de cada mercado y coste de un informe de la cartera, sin escribir nada."""
    folder = Path(folder)
    declaration, declaration_digest = load_declaration(declaration_path)
    source, source_digest, table = _source(folder)
    markets = source["markets"]
    arms = sorted(source["views"][markets[0]]["arms"])
    result = {}
    for market in markets:
        _require(
            sorted(source["views"][market]["arms"]) == arms,
            f"{market} no tiene los mismos modelos que los demás mercados",
        )
        result[market] = {
            str(cost): _market(declaration, *_strategies(table, market, cost, arms), arms)
            for cost in source["declaration"]["cost_bps_per_side"]
        }
    return dict(
        schema_version=1,
        kind=KIND,
        created_at_utc=datetime.now(UTC).isoformat(),
        final_test_opened=False,
        declaration=dict(path=str(declaration_path), sha256=declaration_digest),
        source=dict(
            long_short_sha256=source_digest, sessions_sha256=sha256(folder / "sessions.parquet")
        ),
        arms=arms,
        markets=result,
        code_sha256=sha256(Path(__file__)),
        versions=dict(numpy=np.__version__, scipy=scipy.__version__),
    )


def write_long_short(folder, declaration_path, output):
    """Publicar el análisis en un archivo nuevo fuera de la carpeta de la cartera."""
    output = Path(output)
    safe_destination(output)
    _require(not output.exists(), "La salida debe ser nueva")
    _require(
        not output.resolve().is_relative_to(Path(folder).resolve()),
        "La salida no puede quedar dentro de la fuente",
    )
    report = analyze_long_short(folder, declaration_path)
    atomic_json(output, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--long-short", type=Path, required=True)
    parser.add_argument(
        "--declaration", type=Path, default=Path("configs/evaluation/backtest-overfitting-v1.json")
    )
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    report = write_long_short(arguments.long_short, arguments.declaration, arguments.output)
    for market, costs in report["markets"].items():
        for cost, item in costs.items():
            print(market, cost, "PBO", item["cscv"]["pbo"])


if __name__ == "__main__":
    main()
