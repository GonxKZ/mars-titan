"""Calendario de regímenes por sesión sobre una edición preparada pequeña y sintética.

Los precios se generan en la prueba con semilla fija. La referencia construye cada ventana
con `price_windows.window_rows`, la función que usa la codificación, y llama a la regla de
`memory.regimes`, así que la comparación no repite el cálculo vectorizado del módulo. No hay
modelos ni pasos de optimizador.
"""

import json
from datetime import UTC, date, datetime, timedelta

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mars_titan.data.price_windows import CONTEXT_SESSIONS, window_rows
from mars_titan.evaluation import regime_calendar as rc
from mars_titan.memory.regimes import REGIME_RULE, UNCLASSIFIED, RegimeRule

RULE = RegimeRule(REGIME_RULE, min_assets=4)
HOUR = {"US": 21, "CN": 7}


def business_days(first, count):
    days, day = [], first
    while len(days) < count:
        if day.weekday() < 5:
            days.append(day.isoformat())
        day += timedelta(days=1)
    return days


def stamp(day, market):
    moment = datetime.fromisoformat(day).replace(hour=HOUR[market], tzinfo=UTC)
    return int(moment.timestamp() * 1_000_000)


def market_data(assets=6, sessions=160, seed=0, market="US", absent=(90,), gaps=None):
    """Cierres con tramos alcistas y bajistas, una ausencia de mercado y huecos de activo."""
    rng = np.random.default_rng(seed)
    days = business_days(date(2021, 1, 4), sessions)
    drift = np.repeat(rng.choice([-0.004, 0.004], size=sessions // 20 + 1), 20)[:sessions]
    scale = np.repeat(rng.choice([0.004, 0.02], size=sessions // 30 + 1), 30)[:sessions]
    returns = drift + scale * rng.standard_normal((assets, sessions))
    closes = 50.0 * np.exp(np.cumsum(returns, axis=1))
    mask = np.zeros(sessions, dtype=bool)
    mask[list(absent)] = True
    closes[:, mask] = np.nan
    for asset, index in gaps if gaps is not None else [(0, 120)]:
        closes[asset, index] = np.nan
    at = np.array([-1 if mask[i] else stamp(day, market) for i, day in enumerate(days)])
    return days, at, closes, mask


def reference_routes(market, days, at, closes, absent, rule):
    """Ruta de cada sesión con filas a partir de las ventanas de la codificación."""
    absent_positions = np.flatnonzero(absent)
    expected = []
    for end in np.flatnonzero(~absent):
        windows = []
        for row in closes:
            positions = np.flatnonzero(np.isfinite(row))
            if end not in positions:
                continue
            try:
                rows = window_rows(
                    positions, np.array([np.searchsorted(positions, end)]), 64, absent_positions
                )[0]
            except ValueError:
                continue
            present = rows >= 0
            logs = np.where(present, np.log(row[positions[np.maximum(rows, 0)]]), 0.0)
            window = np.zeros((CONTEXT_SESSIONS, 6))
            window[present, 3] = logs[present] - logs[np.argmax(present)]
            window[:, 5] = present
            windows.append(window)
        if windows:
            state = rule.state(market, np.stack(windows), int(at[end]))
            expected.append((days[end], state.route, state.assets, state.returns))
        else:
            first = max(0, end - CONTEXT_SESSIONS + 1)
            expected.append((days[end], UNCLASSIFIED, 0, int((~absent[first : end + 1]).sum()) - 1))
    return expected


def as_rows(calendar):
    return list(
        zip(
            calendar["sessions"],
            calendar["route"],
            calendar["assets"],
            calendar["returns"],
            strict=True,
        )
    )


def test_routes_match_the_rule_on_the_windows_of_the_encoding():
    days, at, closes, absent = market_data()
    calendar = rc.market_routes("US", days, at, closes, absent, RULE)
    expected = reference_routes("US", days, at, closes, absent, RULE)
    assert as_rows(calendar) == expected
    assert calendar["at"] == [int(value) for value in at[~absent]]
    routes = np.array(calendar["route"])
    # Las 63 primeras sesiones no tienen ventana completa y el resto se clasifica.
    assert set(routes[:63]) == {UNCLASSIFIED} and UNCLASSIFIED not in routes[64:]
    assert len(set(routes[64:])) >= 3
    # El activo con un hueco sale de las ventanas que lo contienen y la sesión ausente de
    # mercado no expulsa a nadie: solo resta un rendimiento.
    assets = dict(zip(calendar["sessions"], calendar["assets"], strict=True))
    returns = dict(zip(calendar["sessions"], calendar["returns"], strict=True))
    assert assets[days[119]] == 6 and assets[days[121]] == assets[days[159]] == 5
    assert returns[days[100]] == 62 and returns[days[160 - 1]] == 63


def test_a_session_only_depends_on_prices_up_to_it():
    days, at, closes, absent = market_data(seed=4)
    full = as_rows(rc.market_routes("US", days, at, closes, absent, RULE))
    for cut in (64, 91, 130):
        changed = closes.copy()
        changed[:, cut + 1 :] *= np.exp(
            np.random.default_rng(cut).normal(0, 0.3, changed[:, cut + 1 :].shape)
        )
        prefix = as_rows(rc.market_routes("US", days, at, changed, absent, RULE))
        kept = sum(1 for i in range(cut + 1) if not absent[i])
        assert prefix[:kept] == full[:kept]
        truncated = as_rows(
            rc.market_routes(
                "US", days[: cut + 1], at[: cut + 1], closes[:, : cut + 1], absent[: cut + 1], RULE
            )
        )
        assert truncated == full[:kept]


def test_cohorts_below_the_minimum_stay_unclassified():
    days, at, closes, absent = market_data(assets=3, gaps=[])
    calendar = rc.market_routes("CN", days, at, closes, absent, RULE)
    assert set(calendar["route"]) == {UNCLASSIFIED}
    assert calendar["assets"][-1] == 3


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda d, a, c, m: c.__setitem__((0, 90), 10.0), "ausente en todo el mercado"),
        (lambda d, a, c, m: a.__setitem__(5, a[4]), "no son crecientes"),
        (lambda d, a, c, m: a.__setitem__(90, 0), "Cada sesión con filas"),
        (lambda d, a, c, m: c.__setitem__((2, 10), 0.0), "positivos"),
    ],
)
def test_incoherent_markets_are_rejected(edit, message):
    days, at, closes, absent = market_data()
    edit(days, at, closes, absent)
    with pytest.raises(ValueError, match=message):
        rc.market_routes("US", days, at, closes, absent, RULE)


def write_prepared(root, markets=("US", "CN"), assets=5, sessions=150, edit=None):
    """Edición preparada mínima: manifiesto, ausencias de mercado y precios por activo."""
    absent_record = {}
    for market in markets:
        days, at, closes, absent = market_data(
            assets=assets, sessions=sessions, market=market, seed=len(market)
        )
        absent_record[market] = dict(
            sessions=[day for day, gone in zip(days, absent, strict=True) if gone],
            first_observed_session=days[0],
            last_observed_session=days[-1],
            last_considered_session="2023-12-31",
        )
        for index, row in enumerate(closes):
            kept = np.isfinite(row)
            table = dict(
                close=row[kept],
                session=[day for day, keep in zip(days, kept, strict=True) if keep],
                available_at=at[kept],
            )
            if edit is not None:
                edit(market, index, table)
            folder = root / "prepared" / market / f"{market}{index:03d}"
            folder.mkdir(parents=True)
            pq.write_table(
                pa.table(
                    dict(
                        close=pa.array(table["close"], pa.float64()),
                        session=pa.array(table["session"]).dictionary_encode(),
                        available_at=pa.array(table["available_at"], pa.timestamp("us", tz="UTC")),
                    )
                ),
                folder / "prices.parquet",
            )
    (root / "manifest.json").write_text(
        json.dumps(
            dict(
                kind="prepared_cohort",
                status="completed",
                cohort_id="fixture",
                input_policy="historical_masked_2000_v1",
            )
        )
    )
    (root / rc.ABSENT_FILE).write_text(json.dumps(absent_record))
    return root


def test_build_write_and_read_keep_the_calendar_and_its_identity(tmp_path):
    prepared = write_prepared(tmp_path / "prepared")
    calendar = rc.build(prepared, rule=RULE)
    days, at, closes, absent = market_data(assets=5, sessions=150, market="US", seed=2)
    assert as_rows(calendar["markets"]["US"]) == reference_routes(
        "US", days, at, closes, absent, RULE
    )
    assert calendar["rule"] == RULE.identity() and calendar["final_test_opened"] is False
    assert set(calendar["code_sha256"]) == set(rc._SOURCES)
    digest = rc.write(calendar, tmp_path / "out" / "calendar.json", sources=(prepared,))
    resolved, read_digest, rule = rc.read(tmp_path / "out" / "calendar.json")
    assert read_digest == digest and rule == RULE.identity()
    assert resolved["US"][0].tolist() == calendar["markets"]["US"]["at"]
    assert resolved["CN"][1].tolist() == calendar["markets"]["CN"]["route"]
    summary = rc.summary(calendar)
    assert summary["US"]["sessions"] == len(calendar["markets"]["US"]["sessions"])
    assert sum(sum(counts) for counts in summary["US"]["routes_by_year"].values()) == 149
    with pytest.raises(ValueError, match="archivo nuevo"):
        rc.write(calendar, tmp_path / "out" / "calendar.json")


def test_sessions_of_the_sealed_test_stop_the_build(tmp_path):
    def into_2024(market, index, table):
        if index == 1:
            table["session"][-1] = "2024-01-02"

    prepared = write_prepared(tmp_path, edit=into_2024)
    with pytest.raises(ValueError, match="test final"):
        rc.build(prepared, rule=RULE)


def test_an_edition_that_considers_sessions_of_the_sealed_test_is_rejected(tmp_path):
    prepared = write_prepared(tmp_path)
    record = json.loads((prepared / rc.ABSENT_FILE).read_text())
    record["US"]["last_considered_session"] = "2024-01-05"
    (prepared / rc.ABSENT_FILE).write_text(json.dumps(record))
    with pytest.raises(ValueError, match="considera sesiones del test final"):
        rc.build(prepared, rule=RULE)


def test_assets_that_disagree_on_the_instant_of_a_session_are_rejected(tmp_path):
    def shifted(market, index, table):
        if market == "US" and index == 2:
            table["available_at"] = table["available_at"].copy()
            table["available_at"][10] += 1

    prepared = write_prepared(tmp_path, edit=shifted)
    with pytest.raises(ValueError, match="instantes distintos"):
        rc.build(prepared, rule=RULE)


def test_an_incomplete_edition_is_rejected(tmp_path):
    prepared = write_prepared(tmp_path)
    manifest = json.loads((prepared / "manifest.json").read_text())
    (prepared / "manifest.json").write_text(json.dumps(dict(manifest, status="running")))
    with pytest.raises(ValueError, match="no está completa"):
        rc.build(prepared, rule=RULE)


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda c: c.update(kind="other"), "contrato"),
        (lambda c: c.update(final_test_opened=True), "contrato"),
        (lambda c: c["rule"].update(rule="calendar_month_v1"), "regla de régimen"),
        (lambda c: c["markets"]["US"].pop("assets"), "campos"),
        (lambda c: c["markets"]["US"]["route"].pop(), "misma longitud"),
        (lambda c: c["markets"]["US"]["route"].__setitem__(70, 5), "fuera de rango"),
        (lambda c: c["markets"]["US"]["at"].__setitem__(3, 0), "no son crecientes"),
        (
            lambda c: c["markets"]["US"]["at"].__setitem__(3, c["markets"]["US"]["at"][2]),
            "no son crecientes",
        ),
        (lambda c: c["markets"]["US"]["sessions"].__setitem__(3, "2024-01-03"), "test final"),
    ],
)
def test_a_calendar_that_breaks_its_contract_is_rejected(tmp_path, edit, message):
    calendar = rc.build(write_prepared(tmp_path), rule=RULE)
    edit(calendar)
    with pytest.raises(ValueError, match=message):
        rc.check(calendar)


def test_the_command_line_builds_with_the_declared_rule(tmp_path, capsys):
    prepared = write_prepared(tmp_path / "prepared", markets=("US",), assets=21)
    assert (
        rc.main(
            ["--prepared", str(prepared), "--output", str(tmp_path / "c.json"), "--market", "US"]
        )
        == 0
    )
    printed = json.loads(capsys.readouterr().out)
    calendar = json.loads((tmp_path / "c.json").read_text())
    assert calendar["rule"] == RegimeRule(REGIME_RULE).identity()
    assert list(calendar["markets"]) == ["US"] and printed["US"]["sessions"] == 149
    assert set(calendar["markets"]["US"]["route"][64:]) - {UNCLASSIFIED}
