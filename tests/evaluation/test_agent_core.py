"""Núcleo agéntico sobre las emisiones: abstención, intervalos conformes, Shapley y costes."""

import json
import math
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mars_titan.evaluation import agent_core as ac
from mars_titan.evaluation import organ_aggregation as og
from mars_titan.evaluation import posttraining_pilots as pilots

ROOT = Path(__file__).resolve().parents[2]
DECLARATION = ROOT / "configs" / "experiments" / "posttraining-agent-core.json"
YEAR = 2018
DAY = 86_400_000_000
START = 1_514_851_200_000_000  # 2 de enero de 2018, 00:00 UTC
SESSIONS = dict(validation=30, calibration=40, evaluation=60)
ASSETS = 40


def declaration():
    return json.loads(DECLARATION.read_text(encoding="utf-8"))


def test_the_declaration_is_the_registered_one():
    document, digest = ac.checked_declaration(declaration())
    assert digest == ac.REGISTERED_SHA256
    assert document["abstention"]["coverages"][-1] == 1.0


@pytest.mark.parametrize(
    "change",
    [
        lambda d: d["abstention"].update(coverages=[0.9, 1.0]),
        lambda d: d["conformal"].update(levels=[0.9]),
        lambda d: d["costs"].update(base_cost_bps=1.0),
        lambda d: d.update(final_test_access=True),
    ],
    ids=["coberturas", "niveles", "coste-base", "test-final"],
)
def test_any_change_of_the_declared_rules_is_refused(change):
    document = declaration()
    change(document)
    with pytest.raises(ValueError, match="no es la registrada"):
        ac.checked_declaration(document)


def _game(players, weights, rng):
    """Juego aditivo con interacción por pares; el jugador «p» no aporta nada."""
    pairs = {
        frozenset(pair): rng.normal() for pair in __import__("itertools").combinations(players, 2)
    }
    values = {}
    for size in range(len(players) + 1):
        for coalition in __import__("itertools").combinations(players, size):
            members = frozenset(coalition) - {"p"}
            value = sum(weights[m] for m in members)
            value = value + sum(v for pair, v in pairs.items() if pair <= members)
            values[frozenset(coalition)] = np.array([value, 2 * value])
    return values


def test_exact_shapley_equals_the_mean_over_orderings_is_efficient_and_ignores_a_null_player():
    players = ("a", "b", "c", "p")
    values = _game(players, dict(a=1.0, b=0.5, c=-0.25), np.random.default_rng(3))
    phi = ac.shapley(players, values)
    check = ac.shapley_by_orderings(players, values)
    for player in players:
        np.testing.assert_allclose(phi[player], check[player], rtol=0, atol=1e-12)
    np.testing.assert_allclose(phi["p"], 0.0, atol=1e-12)
    np.testing.assert_allclose(sum(phi.values()), values[frozenset(players)], atol=1e-12)


def test_the_conformal_correction_is_the_finite_sample_quantile_of_the_scores():
    scores = np.arange(19, 0, -1, dtype=float)
    assert ac.conformal_correction(scores, 0.8) == 16.0
    assert ac.conformal_correction(scores, 0.95) == 19.0
    assert ac.conformal_correction(scores[:5], 0.95) == math.inf
    assert ac.conformal_correction(np.arange(1.0, 11.0), 0.8) == 9.0


def test_conformalized_intervals_reach_their_coverage_where_raw_quantiles_do_not():
    rng = np.random.default_rng(4)
    scale = rng.uniform(0.5, 2.0, size=40_000)
    target = scale * rng.standard_normal(40_000)
    lower, upper = -0.6 * scale, 0.6 * scale
    calibration, evaluation = slice(0, 20_000), slice(20_000, None)
    correction = ac.conformal_correction(
        ac.conformity(lower[calibration], upper[calibration], target[calibration]), 0.8
    )
    raw = ac.interval_summary(lower[evaluation], upper[evaluation], target[evaluation])
    fixed = ac.interval_summary(
        lower[evaluation] - correction, upper[evaluation] + correction, target[evaluation]
    )
    assert raw["coverage"] < 0.6
    assert abs(fixed["coverage"] - 0.8) < 0.01


def _panel(rng, sessions, assets=ASSETS):
    bounds = [(s * assets, (s + 1) * assets) for s in range(sessions)]
    keys = [("US", s) for s in range(sessions)]
    parent, tree = rng.standard_normal((2, sessions * assets))
    target = parent + tree + rng.standard_normal(sessions * assets)
    return bounds, keys, parent, tree, target


def test_the_coverage_is_chosen_with_calibration_alone():
    rng = np.random.default_rng(5)
    bounds, keys, parent, tree, target = _panel(rng, 30)
    score = ac.session_z(parent + tree, bounds)
    confidence = ac.asset_confidence(parent, tree, bounds)
    coverages = (0.5, 0.7, 0.9, 1.0)
    first = ac.choose_asset_coverage(
        confidence, score, target, bounds, keys, coverages, min_assets=10
    )
    assert first["coverage"] in coverages
    again = ac.choose_asset_coverage(
        confidence, score, target.copy(), bounds, keys, coverages, min_assets=10
    )
    assert first == again
    if first["coverage"] < 1:
        assert np.isclose((confidence >= first["threshold"]).mean(), first["coverage"], atol=0.01)


def test_controls_abstain_as_many_rows_as_the_rule_in_every_session():
    rng = np.random.default_rng(6)
    bounds, keys, parent, tree, target = _panel(rng, 12)
    score = ac.session_z(parent + tree, bounds)
    keep = rng.random(len(score)) < 0.7
    magnitude, random = ac.equal_count_controls(score, keep, bounds, draws=5, seed=1)
    for start, end in bounds:
        dropped = int((~keep[start:end]).sum())
        assert int((~magnitude[start:end]).sum()) == dropped
        assert all(int((~draw[start:end]).sum()) == dropped for draw in random)
        if dropped:
            largest_dropped = np.abs(score[start:end][~magnitude[start:end]]).max()
            assert largest_dropped <= np.abs(score[start:end][magnitude[start:end]]).min()
    again, again_random = ac.equal_count_controls(score, keep, bounds, draws=5, seed=1)
    assert np.array_equal(magnitude, again) and np.array_equal(random, again_random)


def test_groups_are_quantiles_within_each_session():
    values = np.r_[np.arange(9.0), np.arange(9.0)[::-1], [np.nan, 1.0, 2.0, 3.0]]
    groups = ac.within_session_groups(values, [(0, 9), (9, 18), (18, 22)], 3)
    assert groups[:9].tolist() == [0, 0, 0, 1, 1, 1, 2, 2, 2]
    assert groups[9:18].tolist() == [2, 2, 2, 1, 1, 1, 0, 0, 0]
    assert groups[18] == -1 and groups[19:].tolist() == [0, 1, 2]


def test_turnover_break_even_and_risk_on_a_hand_computed_book():
    assets = ["a", "b", "c", "d"] * 3
    score = np.array([1.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0, 3.0, 1.0, 4.0, 2.0])
    bounds = [(0, 4), (4, 8), (8, 12)]
    books = ac.tail_book(score, assets, bounds, 0.25)
    assert books[0] == {"a": -1.0, "d": 1.0} and books[1] == {"d": -1.0, "a": 1.0}
    assert books[2] == {"b": -1.0, "c": 1.0}
    returns = [
        dict(a=-0.01, b=0.0, c=0.0, d=0.02),
        dict(a=0.01, b=0.0, c=0.0, d=-0.01),
        dict(a=0.0, b=0.0, c=0.0, d=0.0),
    ]
    gross, turnover = ac.book_series(books, returns)
    np.testing.assert_allclose(gross, [0.03, 0.02, 0.0])
    assert math.isnan(turnover[0]) and turnover[1] == 4.0 and turnover[2] == 4.0
    risk = ac.risk_metrics(np.array([0.01, -0.02, 0.01, -0.01, 0.03]), annualization=252, cvar=0.2)
    assert risk["max_drawdown"] == pytest.approx(0.02)
    assert risk["cvar"] == pytest.approx(-0.02) and risk["hit_rate"] == pytest.approx(0.6)


def test_a_deflated_sharpe_falls_with_more_trials():
    daily = np.random.default_rng(7).normal(0.001, 0.01, 500)
    sharpes = [0.1, 0.05, 0.08, 0.02]
    one = ac.deflated_sharpe(daily, sharpes[:1], trials=1)
    four = ac.deflated_sharpe(daily, sharpes, trials=4)
    assert 0 < four < one < 1


def _write(folder, split, rows, prediction, quantiles=None):
    folder.mkdir(parents=True, exist_ok=True)
    columns = dict(
        sample_id=pa.array(rows["sample_id"]),
        asset_id=pa.array(rows["asset_id"]),
        market=pa.array(["US"] * len(rows["sample_id"])),
        prediction_at=pa.array(rows["prediction_at"], pa.timestamp("us", "UTC")),
        target=pa.array(rows["target"], pa.float32()),
        prediction=pa.array(prediction, pa.float32()),
    )
    for name, values in (quantiles or {}).items():
        columns[name] = pa.array(values, pa.float32())
    pq.write_table(pa.table(columns), folder / f"{split}-predictions.parquet")


def campaign(tmp_path, *, tree=True):
    """Padre con cuantiles estrechos, árboles y órgano sintéticos; la señal ve un tercer factor."""
    rng = np.random.default_rng(8)
    state, reference, organs = tmp_path / "estado", tmp_path / "referencia", tmp_path / "organos"
    parent = state / "runs" / "titans_mac_nogain" / str(YEAR) / "seed-42"
    trees = reference / "runs" / "lightgbm_technical" / str(YEAR) / "seed-42"
    combined = organs / "runs" / "titans_mac_nogain__combined" / str(YEAR) / "seed-42"
    known = dict(alpha={}, liquidity={}, volatility={}, matures={})
    day = 0
    for split, count in SESSIONS.items():
        rows = {name: [] for name in ("sample_id", "asset_id", "prediction_at", "target")}
        own, other, organ, alpha, quantiles = [], [], [], [], {q: [] for q in ac.QUANTILES}
        for session in range(count):
            instant = START + (day + session) * DAY
            z1, z2, z3, noise, n1, n2, n3 = rng.standard_normal((7, ASSETS))
            scale = rng.uniform(0.5, 1.5, ASSETS)
            rows["sample_id"] += [f"{split[0]}{session:03d}-{a:02d}" for a in range(ASSETS)]
            rows["asset_id"] += [f"US/A{a:02d}" for a in range(ASSETS)]
            rows["prediction_at"] += [instant] * ASSETS
            rows["target"] += list(0.3 * z1 + 0.2 * z2 + 0.3 * z3 + scale * noise)
            own += list(z1 + 0.7 * n1)
            other += list(z2 + 0.7 * n2)
            organ += list(z1 + 0.7 * n1 + 0.05 * (z3 + 0.5 * n3))
            alpha += list(0.002 * rng.standard_normal(ASSETS))
            for q, half in zip(ac.QUANTILES, (-0.9, -0.5, 0.0, 0.5, 0.9), strict=True):
                quantiles[q] += list(0.3 * (z1 + 0.7 * n1) + half * scale)
        day += count + 2
        _write(parent, split, rows, own, quantiles)
        if tree:
            _write(trees, split, rows, other)
        if split != "validation":
            _write(combined, split, rows, organ)
        size = count * ASSETS
        known["alpha"][split] = np.asarray(alpha)
        known["liquidity"][split] = rng.normal(15, 2, size)
        known["volatility"][split] = rng.normal(3, 0.5, size)
        known["matures"][split] = np.asarray(rows["prediction_at"], dtype=np.int64) + DAY
    (parent / "run.json").write_text(
        json.dumps(
            dict(
                status="completed",
                final_test_opened=False,
                run_id="padre-1",
                request=dict(view_sha256="f" * 64),
            )
        )
    )
    if tree:
        (trees / "run.json").write_text(
            json.dumps(dict(model="lightgbm_technical", seed=42, final_test_access=False))
        )
    (combined / "run.json").write_text(
        json.dumps(
            dict(
                status="completed",
                final_test_opened=False,
                model="titans_mac_nogain__combined",
                origin=YEAR,
                seed=42,
                parent=dict(model="titans_mac_nogain", state=str(state), run_id="padre-1"),
                arm="combined",
                inputs=dict(
                    emissions={
                        s: pilots._sha256(parent / f"{s}-predictions.parquet") for s in SESSIONS
                    }
                ),
                outputs={
                    s: pilots._sha256(combined / f"{s}-predictions.parquet") for s in og.AGGREGATED
                },
            )
        )
    )
    manifest = state / "cohorts" / str(YEAR) / "manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}")
    return dict(state=state, reference=reference, organs=organs, known=known)


def _arguments(paths, output, *extra):
    return [
        "medir",
        "--declaration",
        str(DECLARATION),
        "--state",
        str(paths["state"]),
        "--reference",
        str(paths["reference"]),
        "--organs",
        str(paths["organs"]),
        "--origin",
        str(YEAR),
        "--output",
        str(output),
        *extra,
    ]


def _known(paths):
    def known_rows(state, origin, tables, view):
        k = paths["known"]
        return (
            {s: k["alpha"][s] for s in tables},
            {s: k["liquidity"][s] for s in tables},
            {s: k["volatility"][s] for s in tables},
            {s: k["matures"][s] for s in tables},
        )

    return known_rows


def test_without_the_trees_it_waits_and_exits_with_code_75(tmp_path, capsys):
    paths = campaign(tmp_path, tree=False)
    assert ac.main(_arguments(paths, tmp_path / "nucleo", "--wait-hours", "0")) == og.MISSING_EXIT
    assert "lightgbm_technical" in capsys.readouterr().err


def test_the_four_measures_of_an_origin(tmp_path, monkeypatch, capsys):
    paths = campaign(tmp_path)
    monkeypatch.setattr(ac, "known_rows", _known(paths))
    output = tmp_path / "nucleo"
    assert ac.main(_arguments(paths, output)) == 0
    folder = output / str(YEAR)
    summary = json.loads((folder / "resumen.json").read_text(encoding="utf-8"))
    assert summary["status"] == "completed" and summary["players"] == list(ac.PLAYERS)
    for key in ("label/s1", "label/s2", "operable/s1", "operable/s2"):
        asset = summary["a1"][key]["asset"]
        assert asset["choice"]["coverage"] in declaration()["abstention"]["coverages"]
        assert len(asset["curve"]) == 6 and len(asset["liquidity_terciles"]) == 3
    level = summary["a2"]["levels"]["0.8"]["systems"]
    assert level["core_raw"]["coverage"] < 0.75
    assert abs(level["core_cqr"]["coverage"] - 0.8) < 0.05
    assert abs(level["combined_cqr"]["coverage"] - 0.8) < 0.05
    for key, game in summary["a3"].items():
        placebo = game["players"]["placebo"]
        if key.endswith("ridge_cov_active"):
            # Un combinador que aprende pesos puede ignorar al placebo: su Shapley sale en cero.
            assert abs(placebo["mean"]) < 0.01, key
        else:
            # En la media de rangos todo miembro pesa igual: el placebo diluye y resta.
            assert placebo["mean"] < 0, key
        total = sum(p["mean"] for p in game["players"].values())
        assert total == pytest.approx(game["whole"], abs=1e-9)
    books = summary["a4"]["books"]
    assert set(books) == {"s1", "s2", "parent", "tree"}
    assert books["s1"]["turnover"] > 0 and len(books["s1"]["deciles_bps"]) == 10
    for name in ("a1-sesiones.parquet", "a3-sesiones.parquet", "a4-dias.parquet", "resumen.md"):
        assert (folder / name).is_file()
    assert [p.name for p in output.iterdir()] == [str(YEAR)]
    capsys.readouterr()
    assert ac.main(_arguments(paths, output)) == 0
    assert "ya calculado" in capsys.readouterr().out
