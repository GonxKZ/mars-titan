"""PBO por CSCV y Sharpe deflactado de #794, con el ejemplo publicado y casos construidos."""

import itertools
import json
import math
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from scipy import stats

from mars_titan.data.storage import atomic_json, sha256
from mars_titan.evaluation import backtest_overfitting as bo

DECLARATION = "configs/evaluation/backtest-overfitting-v1.json"
# Ejemplo numérico de Bailey y López de Prado (2014): Sharpe anual de 2,5 con 1.250 sesiones
# diarias de 250 por año, varianza anual de 1/2 entre 100 ensayos, asimetría −3 y curtosis 10.
SHARPE = 2.5 / math.sqrt(250)
EXAMPLE = dict(variance=0.5 / 250, trials=100, observations=1250, skewness=-3.0, kurtosis=10.0)


def test_the_deflated_sharpe_reproduces_the_published_example():
    result = bo.deflated_sharpe(SHARPE, **EXAMPLE)
    # El artículo da SR₀ ≈ 0,1132 sin anualizar y DSR ≈ 0,9004.
    assert round(result["threshold"], 4) == 0.1132
    assert round(result["probability"], 4) == 0.9004
    # Con 46 ensayos habría superado el 95 %: DSR = 0,9505.
    assert round(bo.deflated_sharpe(SHARPE, **dict(EXAMPLE, trials=46))["probability"], 4) == 0.9505
    # Con rendimientos normales el DSR vale 0,95 con 88 ensayos.
    normal = dict(EXAMPLE, trials=88, skewness=0.0, kurtosis=3.0)
    assert round(bo.deflated_sharpe(SHARPE, **normal)["probability"], 2) == 0.95


def test_the_expected_maximum_is_the_formula_of_the_published_code():
    # getExpMaxSR del apéndice con media nula y desviación √V.
    gamma = 0.5772156649
    for trials in (2, 10, 100, 1000):
        expected = (1 - gamma) * stats.norm.ppf(1 - 1 / trials) + gamma * stats.norm.ppf(
            1 - 1 / (trials * math.e)
        )
        assert bo.expected_maximum_sharpe(4.0, trials) == pytest.approx(2 * expected, rel=1e-9)
    # Crece con los ensayos y vale cero sin dispersión entre ellos.
    values = [bo.expected_maximum_sharpe(1.0, trials) for trials in (2, 10, 100, 1000)]
    assert values == sorted(values) and bo.expected_maximum_sharpe(0.0, 50) == 0.0


def test_the_probabilistic_sharpe_is_one_half_at_its_threshold_and_grows_with_the_sample():
    inputs = dict(skewness=-1.0, kurtosis=6.0)
    assert bo.probabilistic_sharpe(0.1, 0.1, observations=500, **inputs) == 0.5
    values = [
        bo.probabilistic_sharpe(0.1, 0.05, observations=count, **inputs)
        for count in (50, 500, 5000)
    ]
    assert values == sorted(values) and values[0] > 0.5
    # Sin asimetría ni exceso de curtosis es la normal de (SR − SR*)·√(T − 1)/√(1 + SR²/2).
    normal = bo.probabilistic_sharpe(0.1, 0.0, observations=101, skewness=0.0, kurtosis=3.0)
    assert normal == pytest.approx(stats.norm.cdf(0.1 * 10 / math.sqrt(1 + 0.01 / 2)), rel=1e-12)


def test_the_statistics_follow_the_declared_conventions():
    values = np.random.default_rng(1).standard_t(4, 300) * 0.01 + 0.001
    result = bo.sharpe_statistics(values)
    assert result["sharpe"] == values.mean() / values.std(ddof=1)
    assert result["skewness"] == stats.skew(values)
    assert result["kurtosis"] == stats.kurtosis(values, fisher=False)
    assert result["observations"] == 300


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda: bo.expected_maximum_sharpe(0.1, 1), "ensayos"),
        (lambda: bo.expected_maximum_sharpe(-0.1, 10), "negativa"),
        (lambda: bo.expected_maximum_sharpe(math.nan, 10), "finito"),
        (lambda: bo.expected_maximum_sharpe(0.1, 10.0), "entero"),
        (
            lambda: bo.probabilistic_sharpe(0.1, 0.0, observations=1, skewness=0.0, kurtosis=3.0),
            "longitud",
        ),
        (
            lambda: bo.probabilistic_sharpe(0.1, 0.0, observations=10, skewness=0.0, kurtosis=0.5),
            "curtosis",
        ),
        # 1 − γ₃·SR + (γ₄ − 1)/4·SR² = 1 − 5 + 0 < 0.
        (
            lambda: bo.probabilistic_sharpe(1.0, 0.0, observations=10, skewness=5.0, kurtosis=1.0),
            "varianza",
        ),
        (
            lambda: bo.probabilistic_sharpe(True, 0.0, observations=10, skewness=0.0, kurtosis=3.0),
            "finito",
        ),
        (lambda: bo.sharpe_statistics([0.01, 0.01, 0.01]), "no varía"),
        (lambda: bo.sharpe_statistics([0.01, math.inf, 0.0]), "finitos"),
        (lambda: bo.sharpe_statistics([0.01, 0.02]), "tres"),
    ],
    ids=[
        "one_trial",
        "negative_variance",
        "nan_variance",
        "float_trials",
        "short_series",
        "low_kurtosis",
        "negative_spread",
        "boolean",
        "constant",
        "infinite",
        "two_returns",
    ],
)
def test_invalid_inputs_are_refused(call, message):
    with pytest.raises(ValueError, match=message):
        call()


def brute_force(values, blocks):
    """CSCV recorriendo las filas de cada mitad, como en la definición del artículo."""
    size = len(values) // blocks
    rows = np.arange(len(values)).reshape(blocks, size)
    result = []
    for inside in itertools.combinations(range(blocks), blocks // 2):
        outside = [block for block in range(blocks) if block not in inside]
        train, test = values[rows[list(inside)].ravel()], values[rows[outside].ravel()]
        within = train.mean(0) / train.std(0, ddof=1)
        beyond = test.mean(0) / test.std(0, ddof=1)
        best = int(np.argmax(within))
        rank = stats.rankdata(beyond)[best]
        omega = rank / (values.shape[1] + 1)
        result.append((best, within[best], beyond[best], math.log(omega / (1 - omega))))
    return result


def test_the_cscv_matches_a_brute_force_over_every_half():
    values = np.random.default_rng(5).standard_normal((48, 5)) * 0.01
    # Dos estrategias iguales fuerzan empates en el rango de fuera.
    values[:, 4] = values[:, 3]
    result = bo.pbo_cscv(values, blocks=8)
    expected = brute_force(values, 8)
    assert result["combinations"] == len(expected) == math.comb(8, 4)
    assert result["chosen"].tolist() == [item[0] for item in expected]
    np.testing.assert_allclose(result["in_sample"], [item[1] for item in expected], rtol=1e-12)
    np.testing.assert_allclose(result["out_of_sample"], [item[2] for item in expected], rtol=1e-12)
    np.testing.assert_allclose(result["logits"], [item[3] for item in expected], rtol=1e-12)
    logits = np.array([item[3] for item in expected])
    outside = np.array([item[2] for item in expected])
    assert result["pbo"] == np.mean(logits <= 0)
    assert result["probability_of_loss"] == np.mean(outside < 0)
    slope, intercept = np.polyfit([item[1] for item in expected], outside, 1)
    assert result["degradation"] == pytest.approx(dict(slope=slope, intercept=intercept), rel=1e-10)
    assert (result["blocks"], result["sessions_per_block"], result["strategies"]) == (8, 6, 5)


def test_a_strategy_that_wins_every_block_does_not_overfit():
    rng = np.random.default_rng(2)
    values = rng.standard_normal((320, 12)) * 0.01
    values[:, 7] += 0.01
    result = bo.pbo_cscv(values, blocks=16)
    assert set(result["chosen"].tolist()) == {7}
    assert result["pbo"] == 0.0 and result["probability_of_loss"] == 0.0
    assert bool(np.all(result["logits"] > 0))


def test_a_choice_that_reverses_out_of_sample_always_overfits():
    # A gana en la primera mitad y pierde en la segunda, y B al revés.
    first = np.array([0.010, 0.012, -0.010, -0.012])
    values = np.stack([first, -first], axis=1)
    result = bo.pbo_cscv(values, blocks=2)
    assert result["chosen"].tolist() == [0, 1]
    assert result["pbo"] == 1.0 and result["probability_of_loss"] == 1.0
    np.testing.assert_allclose(result["logits"], math.log(0.5))
    # Las dos mitades eligen con el mismo Sharpe dentro y no hay recta que ajustar.
    assert result["degradation"] is None


def test_strategies_without_skill_overfit_about_half_the_time():
    # En una sola muestra el PBO del ruido varía mucho. Su media en doce muestras ronda 1/2.
    values = [
        bo.pbo_cscv(np.random.default_rng(seed).standard_normal((800, 10)) * 0.01, blocks=8)["pbo"]
        for seed in range(12)
    ]
    assert 0.4 < np.mean(values) < 0.65


@pytest.mark.parametrize(
    ("values", "blocks", "message"),
    [
        (np.zeros((40, 3)), 5, "par"),
        (np.zeros((40, 3)), 18, "par"),
        (np.ones((40, 1)), 4, "estrategias"),
        (np.full((40, 3), np.nan), 4, "finitos"),
        (np.random.default_rng(0).standard_normal((42, 3)), 4, "bloques iguales"),
        (np.random.default_rng(0).standard_normal((4, 3)), 4, "bloques iguales"),
        (np.ones((40, 3)), 4, "no varía"),
    ],
    ids=["odd", "too_many", "one_strategy", "nan", "uneven", "one_row", "constant"],
)
def test_cscv_refuses_inputs_outside_its_definition(values, blocks, message):
    with pytest.raises(ValueError, match=message):
        bo.pbo_cscv(values, blocks=blocks)


def long_short_folder(
    root, *, markets=("US",), sessions=64, arms=("a", "b", "c"), seeds=(1, 2), idle=()
):
    """Carpeta con el informe y las sesiones de una cartera larga y corta terminada.

    Los modelos de `idle` son deterministas, sin semilla, y no abren posiciones, como el
    control cero.
    """
    rng = np.random.default_rng(9)
    times = [1_546_380_000_000_000 + day * 86_400_000_000 for day in range(sessions)]
    tables, returns = [], {}
    series = [(arm, seed) for arm in arms for seed in seeds] + [(arm, None) for arm in idle]
    for market in markets:
        for arm, seed in series:
            scale = 0.0 if seed is None else 0.01
            values = {cost: rng.standard_normal(sessions) * scale for cost in (0, 10)}
            returns[market, arm, seed] = values
            tables.append(
                pa.table(
                    dict(
                        arm=[arm] * sessions,
                        seed=pa.array([seed] * sessions, pa.int64()),
                        market=[market] * sessions,
                        prediction_at=pa.array(times, pa.timestamp("us", tz="UTC")),
                        net_return_0bps=values[0],
                        net_return_10bps=values[10],
                    )
                )
            )
    root.mkdir(parents=True)
    pq.write_table(pa.concat_tables(tables), root / "sessions.parquet")
    report = dict(
        kind=bo.SOURCE_KIND,
        status="completed",
        final_test_opened=False,
        markets=list(markets),
        declaration=dict(cost_bps_per_side=[0, 10]),
        views={market: dict(arms={arm: {} for arm in (*arms, *idle)}) for market in markets},
        artifacts={"sessions.parquet": sha256(root / "sessions.parquet")},
    )
    atomic_json(root / "long_short.json", report)
    return returns


def test_the_analysis_measures_every_market_and_cost_of_the_portfolio(tmp_path):
    returns = long_short_folder(tmp_path / "ls", markets=("US", "CN"), sessions=70)
    report = bo.analyze_long_short(tmp_path / "ls", DECLARATION)
    assert report["arms"] == ["a", "b", "c"] and report["final_test_opened"] is False
    assert set(report["markets"]) == {"US", "CN"}
    for market in ("US", "CN"):
        for cost in (0, 10):
            item = report["markets"][market][str(cost)]
            # Las seis sesiones más antiguas no llenan un bloque de 16.
            assert item["sessions"] == 64 and item["trimmed_earliest_sessions"] == 6
            assert item["excluded_without_variation"] == []
            matrix = np.stack(
                [
                    np.mean([returns[market, arm, seed][cost] for seed in (1, 2)], axis=0)[6:]
                    for arm in ("a", "b", "c")
                ],
                axis=1,
            )
            direct = bo.pbo_cscv(matrix, blocks=16)
            assert item["cscv"]["pbo"] == direct["pbo"]
            assert sum(item["cscv"]["chosen_in_sample"].values()) == math.comb(16, 8)
            sharpes = [matrix[:, i].mean() / matrix[:, i].std(ddof=1) for i in range(3)]
            assert item["variance_of_the_sharpes"] == pytest.approx(np.var(sharpes, ddof=1))
            assert item["trials"] == [3, 28, 100]
            dsr = item["strategies"]["b"]["deflated"]
            assert [dsr[str(n)]["threshold"] for n in item["trials"]] == sorted(
                dsr[str(n)]["threshold"] for n in item["trials"]
            )
            assert item["selected_in_full_sample"] == "abc"[int(np.argmax(sharpes))]


def test_a_model_that_never_trades_is_left_out_and_named(tmp_path):
    # Dos controles que no abren posiciones, uno ordenado entre los modelos que sí operan.
    returns = long_short_folder(tmp_path / "ls", sessions=48, idle=("baseline", "zero"))
    report = bo.analyze_long_short(tmp_path / "ls", DECLARATION)
    assert report["arms"] == ["a", "b", "baseline", "c", "zero"]
    item = report["markets"]["US"]["10"]
    assert item["excluded_without_variation"] == ["baseline", "zero"]
    assert set(item["strategies"]) == {"a", "b", "c"} and item["trials"] == [3, 28, 100]
    assert set(item["cscv"]["chosen_in_sample"]) <= {"a", "b", "c"}
    matrix = np.stack(
        [np.mean([returns["US", arm, seed][10] for seed in (1, 2)], axis=0) for arm in "abc"],
        axis=1,
    )
    assert item["cscv"]["pbo"] == bo.pbo_cscv(matrix, blocks=16)["pbo"]


def test_the_analysis_needs_two_models_whose_portfolio_varies(tmp_path):
    long_short_folder(tmp_path / "ls", sessions=48, arms=("a",), idle=("zero",))
    with pytest.raises(ValueError, match="dos modelos cuya cartera varíe"):
        bo.analyze_long_short(tmp_path / "ls", DECLARATION)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda report: report.update(status="paused"), "terminado"),
        (lambda report: report.update(kind="walk_forward_comparison_report"), "terminado"),
        (lambda report: report.update(final_test_opened=True), "reserva"),
        (
            lambda report: report["artifacts"].update({"sessions.parquet": "0" * 64}),
            "no corresponden",
        ),
        (lambda report: report["views"]["US"]["arms"].pop("c"), "mismos modelos"),
    ],
    ids=["paused", "kind", "final_test", "sessions", "arms"],
)
def test_the_analysis_refuses_a_source_it_cannot_trust(tmp_path, change, message):
    long_short_folder(tmp_path / "ls", markets=("CN", "US"))
    report = json.loads((tmp_path / "ls" / "long_short.json").read_text())
    change(report)
    (tmp_path / "ls" / "long_short.json").unlink()
    atomic_json(tmp_path / "ls" / "long_short.json", report)
    with pytest.raises(ValueError, match=message):
        bo.analyze_long_short(tmp_path / "ls", DECLARATION)


def test_the_seeds_are_averaged_in_ascending_order_as_the_comparisons_declare_them(tmp_path):
    returns = long_short_folder(tmp_path / "ls", sessions=48, seeds=(44, 42, 43))
    strategies = bo.analyze_long_short(tmp_path / "ls", DECLARATION)["markets"]["US"]["0"]
    for arm in "abc":
        mean = np.mean([returns["US", arm, seed][0] for seed in (42, 43, 44)], axis=0)
        expected = bo.sharpe_statistics(mean)["sharpe"]
        assert strategies["strategies"][arm]["sharpe"] == expected


def rewrite_sessions(folder, table):
    """Sustituir las sesiones de la carpeta y actualizar su huella en el informe."""
    pq.write_table(table, folder / "sessions.parquet")
    report = json.loads((folder / "long_short.json").read_text())
    report["artifacts"]["sessions.parquet"] = sha256(folder / "sessions.parquet")
    (folder / "long_short.json").unlink()
    atomic_json(folder / "long_short.json", report)


def test_the_row_order_of_the_sessions_does_not_change_the_analysis(tmp_path):
    long_short_folder(tmp_path / "ls", sessions=48)
    expected = bo.analyze_long_short(tmp_path / "ls", DECLARATION)["markets"]
    table = pq.read_table(tmp_path / "ls" / "sessions.parquet")
    order = np.random.default_rng(3).permutation(table.num_rows)
    rewrite_sessions(tmp_path / "ls", table.take(order))
    assert bo.analyze_long_short(tmp_path / "ls", DECLARATION)["markets"] == expected


def test_the_analysis_needs_every_session_of_every_seed(tmp_path):
    long_short_folder(tmp_path / "ls")
    table = pq.read_table(tmp_path / "ls" / "sessions.parquet")
    rewrite_sessions(tmp_path / "ls", table.slice(1))
    with pytest.raises(ValueError, match="una vez todas las sesiones"):
        bo.analyze_long_short(tmp_path / "ls", DECLARATION)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        (dict(blocks=15), "par"),
        (dict(blocks=16.0), "par"),
        (dict(statistic="annual_sharpe"), "no coincide"),
        (dict(constant_series="sharpe_zero"), "no coincide"),
        (dict(final_test_access=True), "no coincide"),
        (dict(trials_sensitivity=[100, 26]), "creciente"),
        (dict(trials_sensitivity=[1]), "creciente"),
        (dict(trials_sensitivity=[28, 65, 100]), "procedencia"),
        (dict(trials_provenance={"28": "Criba", "100": "Ejemplo"}), "procedencia"),
        (
            dict(trials_provenance={"comparison": "", "28": "Criba", "100": "Ejemplo"}),
            "procedencia",
        ),
    ],
)
def test_the_declaration_must_describe_this_analysis(tmp_path, changes, message):
    document = json.loads(Path(DECLARATION).read_text(encoding="utf-8"))
    atomic_json(tmp_path / "declaration.json", document | changes)
    with pytest.raises(ValueError, match=message):
        bo.load_declaration(tmp_path / "declaration.json")


def test_the_command_line_writes_only_a_new_report_outside_the_source(tmp_path, capsys):
    long_short_folder(tmp_path / "ls")
    output = tmp_path / "out" / "overfitting.json"
    # Sin --declaration usa la declarada del repositorio.
    bo.main(["--long-short", str(tmp_path / "ls"), "--output", str(output)])
    written = json.loads(output.read_text())
    assert written["kind"] == bo.KIND and "PBO" in capsys.readouterr().out
    assert written["declaration"]["path"] == DECLARATION
    with pytest.raises(ValueError, match="nueva"):
        arguments = ["--long-short", str(tmp_path / "ls"), "--declaration", DECLARATION]
        bo.main([*arguments, "--output", str(output)])
    with pytest.raises(ValueError, match="dentro de la fuente"):
        bo.write_long_short(tmp_path / "ls", DECLARATION, tmp_path / "ls" / "overfitting.json")
