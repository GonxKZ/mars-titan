"""Selección de desarrollo por cobertura pasada, sin usar resultados posteriores."""

import copy
import importlib
from collections import Counter

import pytest


def api():
    try:
        return importlib.import_module("benchmarks.combined_memory_cohort")
    except ModuleNotFoundError:
        pytest.fail("Falta la selección auditada de la cohorte combinada")


def population():
    assets, rows = [], []
    for index in range(8):
        for number in range(3):
            symbol = f"A{index}{number}"
            history = "long" if index & 4 else "short"
            count = 3000 if index & 4 else 100
            news, fund = bool(index & 1), bool(index & 2)
            assets.append(dict(market="US", symbol=symbol, counts=dict(train=count, evaluation=10)))
            rows.append(
                dict(
                    symbol=symbol,
                    folds={
                        "fold-018": dict(
                            stratum=f"n{int(news)}f{int(fund)}_{history}",
                            train_rows=count,
                            recent_input_rows=100,
                            presence_fraction=[1, float(news), 1, float(fund), 1],
                        )
                    },
                )
            )
    return dict(assets=assets), rows


def test_selection_balances_eight_past_input_strata_and_is_deterministic():
    source, rows = population()
    selected = api().choose_assets(source, rows, "fold-018", 16)
    assert len(selected) == len({r["symbol"] for r in selected}) == 16
    strata = {r["symbol"]: r["folds"]["fold-018"]["stratum"] for r in rows}
    assert set(Counter(strata[a["symbol"]] for a in selected).values()) == {2}
    assert selected == api().choose_assets(source, list(reversed(rows)), "fold-018", 16)


def test_future_counts_and_returns_do_not_choose_the_assets():
    source, rows = population()
    expected = [a["symbol"] for a in api().choose_assets(source, rows, "fold-018", 16)]
    changed = copy.deepcopy(source)
    for index, asset in enumerate(changed["assets"]):
        asset["counts"]["evaluation"] = index * 999
        asset["future_return"] = -index
    assert [a["symbol"] for a in api().choose_assets(changed, rows, "fold-018", 16)] == expected


def test_duplicate_or_inconsistent_audit_is_rejected():
    source, rows = population()
    with pytest.raises(ValueError, match="duplic"):
        api().choose_assets(source, rows + rows[:1], "fold-018", 16)
    rows[0]["folds"]["fold-018"]["stratum"] = "n1f1_long"
    with pytest.raises(ValueError, match="estrato"):
        api().choose_assets(source, rows, "fold-018", 16)


def test_empty_strata_are_reported_instead_of_inventing_assets():
    source, rows = population()
    source["assets"] = source["assets"][:3]
    selected = api().choose_assets(source, rows[:3], "fold-018", 2)
    assert len(selected) == 2
    with pytest.raises(ValueError, match="activos"):
        api().choose_assets(source, rows[:3], "fold-018", 4)


def test_projection_keeps_each_selected_asset_and_rejects_other_dates():
    source, _ = population()
    for asset in source["assets"]:
        asset.update(samples=4000, purged_by_boundary=dict(train=1, evaluation=0))
    source.update(
        input_policy="historical_masked_2000_v1",
        final_test_opened=False,
        counts=dict(train=100000, evaluation=240),
        coverage=[dict(market="US", symbol=a["symbol"]) for a in source["assets"]],
        temporal_view=dict(
            protocol=dict(market="US", selection=dict(max_epochs=30)),
            fold=dict(
                id="fold-018",
                train=["2000-01-01", "2022-04-01"],
                validation=["2022-04-01", "2022-10-01"],
                calibration=["2022-10-01", "2023-01-01"],
                evaluation=["2023-01-01", "2024-01-01"],
            ),
        ),
    )
    selected = source["assets"][:2]
    result = api().project_view(source, selected, "fold-018", 3)
    assert result["assets"] == selected
    assert result["counts"] == dict(train=200, evaluation=20)
    assert result["samples"] == 8000
    assert result["temporal_view"]["protocol"]["selection"]["max_epochs"] == 3
    assert source["temporal_view"]["protocol"]["selection"]["max_epochs"] == 30
    source["temporal_view"]["fold"]["evaluation"][1] = "2025-01-01"
    with pytest.raises(ValueError, match="ventana"):
        api().project_view(source, selected, "fold-018", 3)
