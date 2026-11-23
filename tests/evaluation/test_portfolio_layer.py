"""P: la declaración, el libro con casos hechos a mano y la orden de un origen."""

import json

import numpy as np
import pytest
from scipy.stats import rankdata

from mars_titan.evaluation import organ_aggregation as og
from mars_titan.evaluation import portfolio_layer as pl
from tests.evaluation import test_agent_core as core

DECLARATION = core.ROOT / "configs" / "experiments" / "posttraining-portfolio-layer.json"


def declaration():
    return json.loads(DECLARATION.read_text(encoding="utf-8"))


def test_the_declaration_is_the_registered_one():
    document, digest = pl.checked_declaration(declaration())
    assert digest == pl.REGISTERED_SHA256
    assert document["rule"]["scope"] == "liquid" and document["costs_bps_per_side"] == [
        2.0,
        3.0,
        5.0,
    ]


@pytest.mark.parametrize(
    "change",
    [
        lambda d: d["rule"].update(beats_control_min_t=1.5),
        lambda d: d["smoothing"]["lambdas"].append(0.05),
        lambda d: d["choice"].update(cost_bps=2.0),
        lambda d: d.update(horizons=[1, 5]),
    ],
)
def test_any_change_of_the_protocol_is_refused(change):
    document = declaration()
    change(document)
    with pytest.raises(ValueError, match="no es la registrada para P"):
        pl.checked_declaration(document)


def _session(key, index, score, returns):
    return (
        key,
        np.asarray(index),
        np.asarray(score, dtype=float),
        np.asarray(returns, dtype=float),
    )


def test_the_book_enters_holds_within_the_band_and_counts_its_turnover():
    # Diez activos; el 9 es el más alto y el 0 el más bajo en la primera sesión.
    first = _session(("US", 1), range(10), range(10), [0.0] * 9 + [0.02])
    # En la segunda, el 9 baja al octavo puesto (percentil 0,75): sin banda sale; con banda de
    # salida 0,3 se queda.
    order = [0, 1, 2, 3, 4, 5, 6, 9, 7, 8]
    second = _session(("US", 2), order, range(10), [0.01] * 10)
    plain = pl.run_book([first, second], 10, smoothing=1.0, entry=0.1, exit_tail=0.1, min_assets=1)
    banded = pl.run_book([first, second], 10, smoothing=1.0, entry=0.1, exit_tail=0.3, min_assets=1)
    # Primera sesión: largo del 9 y corto del 0, desde cero; bruto 0,02 − 0.
    assert plain[0][1] == pytest.approx(0.02) and plain[0][2] == pytest.approx(2.0)
    # Sin banda se cambia el largo del 9 por el 8: rotación 2. Con banda el 9 se queda y el 8 entra
    # igual (el decil alto siempre entra): dos largos de 0,5, rotación 1.
    assert plain[1][2] == pytest.approx(2.0) and banded[1][2] == pytest.approx(1.0)
    assert banded[1][3] == 2 and banded[1][4] == 1


def test_smoothing_restarts_for_an_asset_absent_the_session_before():
    # El activo 2 falta en la segunda sesión y vuelve en la tercera: allí empieza en su z.
    sessions = [
        _session(("US", 1), [0, 1, 2], [0, 1, 2], [0, 0, 0]),
        _session(("US", 2), [0, 1], [1, 0], [0, 0]),
        _session(("US", 3), [0, 1, 2], [2, 1, 0], [0, 0, 0]),
    ]
    smooth = pl.run_book(sessions, 3, smoothing=0.5, entry=0.34, exit_tail=0.34, min_assets=1)
    assert [b[3] + b[4] for b in smooth] == [2, 2, 2]
    z = pl.session_z(np.array([2.0, 1.0, 0.0]))
    assert z.tolist() == pytest.approx(list((rankdata([2, 1, 0]) - 2) / np.std([1, 2, 3])))


def test_strong_smoothing_keeps_yesterday_s_book_when_the_ranking_flips():
    up = _session(("US", 1), range(10), range(10), [0.0] * 10)
    down = _session(("US", 2), range(10), range(9, -1, -1), [0.0] * 10)
    daily = pl.run_book([up, down], 10, smoothing=1.0, entry=0.1, exit_tail=0.1, min_assets=1)
    slow = pl.run_book([up, down], 10, smoothing=0.15, entry=0.1, exit_tail=0.1, min_assets=1)
    # Sin suavizar, el largo y el corto se cruzan (rotación 4); con λ = 0,15 el orden de ayer pesa
    # 0,85 y el libro no cambia.
    assert daily[1][2] == pytest.approx(4.0) and slow[1][2] == pytest.approx(0.0)


def test_thin_sessions_hold_nothing_and_pay_to_close():
    sessions = [
        _session(("US", 1), range(10), range(10), [0.0] * 10),
        _session(("US", 2), range(3), range(3), [0.0] * 3),
    ]
    book = pl.run_book(sessions, 10, smoothing=1.0, entry=0.1, exit_tail=0.1, min_assets=5)
    assert book[1][1] == 0.0 and book[1][2] == pytest.approx(2.0) and book[1][3] == 0


def test_net_costs_drawdown_and_the_simplest_choice_on_ties():
    book = [
        (("US", 1), 0.001, 2.0, 1, 1),
        (("US", 2), -0.002, 0.5, 1, 1),
        (("US", 3), 0.003, 0.0, 1, 1),
    ]
    assert pl.net_series(book, 5.0).tolist() == pytest.approx([0.0, -0.00225, 0.003])
    assert pl.drawdown(np.array([0.01, -0.03, 0.01, -0.01])) == pytest.approx(0.03)
    flat = [
        _session(("US", t), range(20), np.random.default_rng(t).random(20), [0.0] * 20)
        for t in range(5)
    ]
    chosen, table = pl.choose(flat, 20, declaration(), 1)
    # Todo da neto ≤ 0 por los costes: se informa la rejilla entera y gana el de menos rotación.
    assert len(table) == 15
    best = max(table, key=lambda r: r["mean_net_bps"])
    assert (chosen["smoothing"], chosen["exit_tail"]) == (best["smoothing"], best["exit_tail"])
    zero = [_session(("US", t), range(20), np.arange(20.0), [0.0] * 20) for t in range(5)]
    chosen, _ = pl.choose(zero, 20, declaration(), 1)
    # Con la misma puntuación cada día todos los libros rotan igual: empatan y gana lo más
    # sencillo, λ = 1 con salida 0,1.
    assert chosen == dict(smoothing=1.0, exit_tail=0.1)


def _scopes(versus_mean, versus_t, own_mean, own_t, chosen=(0.4, 0.2)):
    item = dict(
        chosen=dict(smoothing=chosen[0], exit_tail=chosen[1]),
        control_parameters=dict(smoothing=1.0, exit_tail=0.1),
        layer_vs_control={"3.0": dict(mean_bps=versus_mean, t_nw=versus_t)},
        layer=dict(costs={"5.0": dict(mean_bps=own_mean, t_nw=own_t)}),
    )
    return dict(liquid=item)


def test_the_verdict_reads_the_declared_rule():
    rule = declaration()["rule"]
    assert pl.verdict(_scopes(1.0, 2.5, 2.0, 3.0), rule)["enters"]
    assert len(pl.verdict(_scopes(1.0, 1.5, 2.0, 3.0), rule)["reasons"]) == 1
    assert len(pl.verdict(_scopes(1.0, 2.5, -0.1, 3.0), rule)["reasons"]) == 1
    assert (
        "propio control"
        in pl.verdict(_scopes(0.0, None, 2.0, 3.0, chosen=(1.0, 0.1)), rule)["reasons"][-1]
    )


def _arguments(paths, output, *extra):
    arguments = core._arguments(paths, output, *extra)
    arguments[arguments.index("--declaration") + 1] = str(DECLARATION)
    return arguments


def test_p_of_an_origin(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(core, "ASSETS", 96)
    paths = core.campaign(tmp_path)
    monkeypatch.setattr(pl.ac, "known_rows", core._known(paths))
    output = tmp_path / "cartera"
    assert pl.main(_arguments(paths, output)) == 0
    folder = output / str(core.YEAR)
    summary = json.loads((folder / "resumen.json").read_text(encoding="utf-8"))
    assert summary["status"] == "completed" and set(summary["scopes"]) == {"liquid", "all"}
    grid = declaration()
    for scope in summary["scopes"].values():
        assert scope["chosen"]["smoothing"] in grid["smoothing"]["lambdas"]
        assert scope["chosen"]["exit_tail"] in grid["band"]["exit_tails"]
        assert set(scope["layer"]["costs"]) == {"2.0", "3.0", "5.0"}
        assert set(scope["layer"]["costs"]["3.0"]["horizons"]) == {"1", "5", "21"}
    # El libro del alfa solo con todos los nombres, recalculado aparte.
    import pyarrow.parquet as pq

    table = pq.read_table(
        paths["state"]
        / "runs"
        / "titans_mac_nogain"
        / str(core.YEAR)
        / "seed-42"
        / "evaluation-predictions.parquet"
    ).sort_by(og.ORDER)
    ids = table.column("sample_id").to_pylist()
    alpha = np.array([paths["known"]["alpha"]["evaluation"][s] for s in ids])
    calibration = np.abs(np.array(list(paths["known"]["alpha"]["calibration"].values())))
    cap = float(np.quantile(calibration, 0.99))
    assert summary["alpha_cap"] == pytest.approx(cap)
    target = (table.column("target").to_numpy() + alpha.astype(np.float32)).astype(np.float32)
    instants = table.column("prediction_at").cast("int64").to_numpy()
    gross = []
    for instant in np.unique(instants):
        rows = (instants == instant) & (np.abs(alpha) <= cap)
        position = (rankdata(alpha[rows]) - 0.5) / rows.sum()
        r = target[rows].astype(np.float64)
        gross.append(r[position >= 0.9].mean() - r[position < 0.1].mean())
    assert summary["scopes"]["all"]["alpha"]["gross_bps"] == pytest.approx(
        1e4 * np.mean(gross), abs=1e-6
    )
    assert (folder / "sesiones.parquet").is_file()
    assert "P, la capa de cartera" in (folder / "resumen.md").read_text(encoding="utf-8")
    capsys.readouterr()
    assert pl.main(_arguments(paths, output)) == 0
    assert "ya calculado" in capsys.readouterr().out
