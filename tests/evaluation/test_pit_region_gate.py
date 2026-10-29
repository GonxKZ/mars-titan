"""Puerta de reescalado por regiones en fecha C4 y sus controles (#751).

Los historiales y paneles son sintéticos, con dos grupos de activos plantados. No hay
modelos ajustados ni pasos de optimizador.
"""

import json

import numpy as np
import pytest

from mars_titan.evaluation import pit_region_gate as gate
from mars_titan.evaluation import scale_diagnostics
from mars_titan.evaluation.forecast_panel import ForecastPanel
from mars_titan.evaluation.output_stages import OUTPUT_STAGES

DAY = 86_400_000_000
START = 1_672_617_600_000_000  # 2023-01-02, antes del test reservado
ASSETS = [f"A{i}" for i in range(6)] + [f"B{i}" for i in range(6)]
CONFIG = gate.DEFAULT | dict(clusters=2, window=80, min_labels=40)


def history(days=80, *, seed=3, end=START, extra=()):
    """Residuos de dos factores: los A cargan en el primero y los B en el segundo."""
    rng = np.random.default_rng(seed)
    factors = rng.normal(size=(2, days))
    rows = []
    for asset in ASSETS:
        loading = factors[0] if asset.startswith("A") else factors[1]
        values = loading + 0.3 * rng.normal(size=days)
        for day in range(days):
            rows.append((asset, values[day], end - (days - day) * DAY))
    rows.extend(extra)
    asset_id, target, matures_at = zip(*rows, strict=True)
    return dict(
        asset_id=np.array(asset_id),
        target=np.array(target),
        matures_at=np.array(matures_at, dtype=np.int64),
    )


def panel(assets, prediction, target, sessions, *, start=START, market="US"):
    """Panel en orden canónico con identidades `mercado/activo/instante`."""
    at = start + np.asarray(sessions, dtype=np.int64) * DAY
    order = np.lexsort((np.asarray(assets), at))
    assets = np.asarray(assets)[order]
    at = at[order]
    return ForecastPanel.from_columns(
        np.array([f"{market}/{a}/{t}" for a, t in zip(assets, at, strict=True)]),
        np.full(len(at), market),
        at,
        np.asarray(target, dtype=np.float64)[order],
        np.asarray(prediction, dtype=np.float64)[order],
    )


def calibration(seed=5, sessions=10):
    """Calibración con sesiones de tamaños distintos, para que los pesos 1/n importen."""
    rng = np.random.default_rng(seed)
    assets = [a for day in range(sessions) for a in ASSETS[: len(ASSETS) - 2 * (day % 3)]]
    days = [day for day in range(sessions) for _ in ASSETS[: len(ASSETS) - 2 * (day % 3)]]
    prediction = rng.normal(size=len(assets))
    scale = np.where(np.char.startswith(np.array(assets), "A"), 0.8, 0.1)
    target = scale * prediction + 0.05 * rng.normal(size=len(assets))
    return panel(assets, prediction, target, days)


def fitted(config=CONFIG, **options):
    cal = calibration()
    record = gate.fit(
        cal, cal.prediction_at + DAY, gate.panel_assets(cal), history(), config, **options
    )
    return record, cal


def test_planted_groups_are_recovered_and_numbered_by_their_first_asset():
    regions = gate.residual_regions(ASSETS, history(), START, clusters=2, window=80, min_labels=40)
    assert {a: regions[a] for a in ASSETS} == {
        **dict.fromkeys(ASSETS[:6], "1"),
        **dict.fromkeys(ASSETS[6:], "2"),
    }


def test_regions_follow_average_linkage_of_one_minus_correlation():
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    values = np.random.default_rng(21).normal(size=(len(ASSETS), 60))
    data = dict(
        asset_id=np.repeat(ASSETS, 60),
        target=values.ravel(),
        matures_at=np.tile(START - (60 - np.arange(60)) * DAY, len(ASSETS)),
    )
    regions = gate.residual_regions(ASSETS, data, START, clusters=4, window=60, min_labels=30)
    tree = {
        method: fcluster(
            linkage(squareform(1 - np.corrcoef(values), checks=False), method=method),
            t=4,
            criterion="maxclust",
        )
        for method in ("average", "single")
    }
    first = {}
    for asset, group in zip(ASSETS, tree["average"], strict=True):
        first.setdefault(group, asset)
    rank = {group: str(i + 1) for i, group in enumerate(sorted(first, key=first.get))}
    assert regions == {a: rank[g] for a, g in zip(ASSETS, tree["average"], strict=True)}
    # Con estos residuos el enlace simple reparte de otro modo, así que el método importa.
    assert len(set(zip(tree["average"], tree["single"], strict=True))) > 4


def test_regions_only_read_labels_matured_by_the_cutoff():
    extreme = [(asset, 50.0 * (-1) ** i, START) for i, asset in enumerate(ASSETS)]
    options = dict(clusters=2, window=80, min_labels=40)
    base = gate.residual_regions(ASSETS, history(), START, **options)
    assert gate.residual_regions(ASSETS, history(extra=extreme), START - 1, **options) == base
    # Una etiqueta que madura justo en el corte ya está disponible y cambia el reparto.
    moved = gate.residual_regions(ASSETS, history(extra=extreme), START, **options)
    assert moved == {a: str(1 + i % 2) for i, a in enumerate(ASSETS)} != base


def test_the_fit_cuts_the_history_at_the_first_calibration_emission():
    cal = calibration()
    extreme = [(asset, 50.0 * (-1) ** i, START + DAY) for i, asset in enumerate(ASSETS)]
    assets = gate.panel_assets(cal)
    record = gate.fit(cal, cal.prediction_at + DAY, assets, history(extra=extreme), CONFIG)
    assert record["regions"]["US"] == fitted()[0]["regions"]["US"]


def test_short_or_constant_histories_go_to_region_zero():
    data = history()
    keep = ~((data["asset_id"] == "A0") & (np.arange(len(data["target"])) % 2 == 0))
    data = {name: value[keep] for name, value in data.items()}
    data["target"] = np.where(data["asset_id"] == "B5", 1.0, data["target"])
    regions = gate.residual_regions(ASSETS, data, START, clusters=2, window=80, min_labels=41)
    assert regions["A0"] == regions["B5"] == gate.NO_HISTORY
    assert {regions[a] for a in ASSETS if a not in ("A0", "B5")} == {"1", "2"}
    # Con exactamente el mínimo de etiquetas, A0 ya tiene historia suficiente.
    regions = gate.residual_regions(ASSETS, data, START, clusters=2, window=80, min_labels=40)
    assert regions["A0"] == "1" and regions["B5"] == gate.NO_HISTORY


def test_two_labels_of_an_asset_at_the_same_instant_are_rejected():
    duplicated = history(extra=[("A1", 0.0, START - DAY)])
    with pytest.raises(ValueError, match="mismo instante"):
        gate.residual_regions(ASSETS, duplicated, START, clusters=2, window=80, min_labels=40)


def test_each_region_gets_the_a1_factor_of_its_own_rows():
    record, cal = fitted()
    assets = gate.panel_assets(cal)
    weights = scale_diagnostics.session_weights(cal)
    for region in ("1", "2"):
        rows = np.array([record["regions"]["US"][a] == region for a in assets])
        expected = scale_diagnostics.shrinkage_factor(
            cal.target[rows], cal.prediction[rows], weights[rows]
        )
        assert record["factors"]["US"][region] == expected
    assert record["factors"]["US"]["1"] > record["factors"]["US"]["2"]


def test_application_scales_each_row_and_unseen_assets_use_the_market_factor():
    record, cal = fitted()
    later = panel(["A2", "B3", "C9"], [1.0, -2.0, 4.0], [0.0, 0.0, 0.0], [20, 20, 20])
    result = gate.apply(record, later, gate.panel_assets(later))
    factors, market = record["factors"]["US"], record["market_factors"]["US"]
    assert result.tolist() == [factors["1"] * 1.0, factors["2"] * -2.0, market * 4.0]
    # Un activo nuevo no hereda la región 0 de los activos sin historia de la calibración.
    record["factors"]["US"][gate.NO_HISTORY] = 0.5
    assert gate.apply(record, later, gate.panel_assets(later))[2] == market * 4.0


def test_the_market_control_reproduces_a1_bit_for_bit():
    record, cal = fitted(CONFIG | dict(regions="market"))
    later = calibration(seed=9)
    later = panel(
        gate.panel_assets(later),
        later.prediction,
        later.target,
        (later.prediction_at - START) // DAY + 20,
    )
    a1 = scale_diagnostics.apply_shrinkage(
        later, later.prediction, scale_diagnostics.market_shrinkage(cal, cal.prediction)
    )
    assert np.array_equal(gate.apply(record, later, gate.panel_assets(later)), a1)


def test_the_permuted_control_keeps_sizes_and_changes_the_assignment():
    record, _ = fitted()
    control, _ = fitted(CONFIG | dict(regions="permuted"))
    other, _ = fitted(CONFIG | dict(regions="permuted", seed=7))
    variant, permuted = record["regions"]["US"], control["regions"]["US"]
    assert sorted(variant.values()) == sorted(permuted.values())
    assert variant != permuted
    assert control == fitted(CONFIG | dict(regions="permuted"))[0]
    assert other["regions"]["US"] != permuted
    # La región 0 no la decide el agrupamiento y no entra en la permutación.
    unseen = {"Z": gate.NO_HISTORY, **variant}
    assert all(gate.permuted_regions(unseen, [seed, 0])["Z"] == "0" for seed in range(8))


def test_the_current_sector_control_is_the_only_mode_that_reads_sectors():
    sectors = {a: ("Tecnología" if a < "B" else "Energía") for a in ASSETS[:-1]}
    record, _ = fitted(CONFIG | dict(regions="current_sector"), sectors=sectors)
    regions = record["regions"]["US"]
    assert regions["A0"] == "2" and regions["B0"] == "1" and regions["B5"] == gate.NO_HISTORY
    with pytest.raises(ValueError, match="current_sector"):
        fitted(sectors=sectors)
    with pytest.raises(ValueError, match="current_sector"):
        fitted(CONFIG | dict(regions="current_sector"))


def test_application_refuses_rows_emitted_before_the_calibration_matured():
    record, cal = fitted()
    with pytest.raises(ValueError, match="maduran después"):
        gate.apply(record, cal, gate.panel_assets(cal))
    with pytest.raises(ValueError, match="maduran después"):
        gate.apply(record, panel(["A2"], [1.0], [0.0], [5]), ["A2"])
    # Una emisión en el instante en que madura la última etiqueta de G ya se admite.
    at_maturity = panel(["A2"], [1.0], [0.0], [10])
    assert gate.apply(record, at_maturity, ["A2"]).tolist() == [record["factors"]["US"]["1"]]


def test_the_record_survives_json_and_the_stage_is_registered():
    record, cal = fitted()
    restored = json.loads(json.dumps(record))
    later = panel(["A2", "B3"], [1.0, -2.0], [0.0, 0.0], [20, 20])
    assets = gate.panel_assets(later)
    assert np.array_equal(gate.apply(restored, later, assets), gate.apply(record, later, assets))
    assert gate.stage_sha256(restored) == gate.stage_sha256(record)
    assert OUTPUT_STAGES[gate.IDENTITY] is gate.checked_config


@pytest.mark.parametrize(
    "change",
    [
        dict(regions="sector"),
        dict(clusters=1),
        dict(clusters=True),
        dict(window=10),
        dict(min_labels=200),
        dict(seed=-1),
    ],
)
def test_invalid_configurations_are_rejected(change):
    with pytest.raises(ValueError):
        gate.checked_config(gate.DEFAULT | change)


def test_row_identities_must_name_market_asset_and_instant():
    later = ForecastPanel.from_columns(
        np.array(["r1", "r2"]),
        np.array(["US", "US"]),
        np.array([START, START]),
        np.zeros(2),
        np.zeros(2),
    )
    with pytest.raises(ValueError, match="mercado/activo/instante"):
        gate.panel_assets(later)
