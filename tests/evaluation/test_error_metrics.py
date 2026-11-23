"""F7: la declaración, las piezas exactas de la calibración, la regla y la orden de un origen."""

import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mars_titan.evaluation import error_metrics as em
from mars_titan.evaluation import organ_aggregation as og
from mars_titan.evaluation import scale_diagnostics as sd
from tests.evaluation import test_agent_core as core

DECLARATION = core.ROOT / "configs" / "experiments" / "posttraining-error-metrics.json"


def declaration():
    return json.loads(DECLARATION.read_text(encoding="utf-8"))


def test_the_declaration_is_the_registered_one():
    document, digest = em.checked_declaration(declaration())
    assert digest == em.REGISTERED_SHA256
    assert document["rule"]["ladder"] == ["full", "point", "raw"]
    assert tuple(core.ac.QUANTILES) == tuple(em.QUANTILE_COLUMNS)


@pytest.mark.parametrize(
    "change",
    [
        lambda d: d["rule"].update(comparison="tolerance_0_001"),
        lambda d: d["metrics"]["intervals"].append(0.5),
        lambda d: d["predictors"]["seeds"]["seeds"].remove(44),
        lambda d: d["calibration"].update(split="evaluation"),
    ],
)
def test_any_change_of_the_protocol_is_refused(change):
    document = declaration()
    change(document)
    with pytest.raises(ValueError, match="no es la registrada para F7"):
        em.checked_declaration(document)


def _pinball(values, weights, level, center):
    u = values - center
    return float(np.sum(weights * np.maximum(level * u, (level - 1) * u)))


def test_the_weighted_quantile_is_the_lower_minimiser_of_the_weighted_pinball():
    assert em.weighted_quantile([1.0, 2.0, 3.0, 4.0], [1.0] * 4, 0.5) == 2.0
    assert em.weighted_quantile([1.0, 2.0, 3.0, 4.0], [1.0, 1.0, 1.0, 5.0], 0.5) == 4.0
    rng = np.random.default_rng(3)
    for level in em.LEVELS:
        values, weights = rng.standard_normal(200), rng.uniform(0.1, 2.0, 200)
        best = em.weighted_quantile(values, weights, level)
        losses = np.array([_pinball(values, weights, level, c) for c in values])
        assert _pinball(values, weights, level, best) <= losses.min() + 1e-12
        assert best == values[losses <= losses.min() + 1e-12].min()


def test_level_scales_are_the_exact_minimisers_of_the_pinball():
    rng = np.random.default_rng(5)
    rows = 300
    widths = rng.uniform(0.5, 1.5, rows)[:, None] * np.array([-1.6, -1.0, 0.0, 1.0, 1.6])
    residual = 0.7 * rng.standard_normal(rows)
    weights = rng.uniform(0.2, 1.0, rows)
    scales = em.level_scales(residual, widths, weights)
    assert scales[em.MEDIAN] == 1.0
    grid = np.linspace(0.0, 3.0, 3001)
    for j, level in enumerate(em.LEVELS):
        if j == em.MEDIAN:
            continue

        def loss(scale, j=j, level=level):
            u = residual - scale * widths[:, j]
            return float(np.sum(weights * np.maximum(level * u, (level - 1) * u)))

        assert loss(scales[j]) <= min(loss(s) for s in grid) + 1e-12
    with pytest.raises(ValueError, match="cruza la mediana"):
        em.level_scales(residual, -widths, weights)


def test_recalibrated_quantiles_never_cross_and_sorting_never_raises_the_pinball():
    center = np.zeros(4)
    widths = np.tile([-1.0, -0.9, 0.0, 0.9, 1.0], (4, 1))
    # Con estos factores el nivel 0,025 (−0,5) queda por encima del 0,1 (−0,9): se cruzan.
    scales = np.array([0.5, 1.0, 1.0, 1.0, 0.5])
    ordered = em.recalibrated(center, widths, scales)
    crossed = center[:, None] + widths * scales
    assert np.all(np.diff(ordered, axis=1) >= 0) and np.any(np.diff(crossed, axis=1) < 0)
    levels = np.array(em.LEVELS)
    target = np.array([-2.0, -0.7, 0.3, 1.5])

    def total(quantiles):
        u = target[:, None] - quantiles
        return np.maximum(levels * u, (levels - 1) * u).sum(axis=1)

    assert np.all(total(ordered) <= total(crossed) + 1e-15)


def _item(mae, rmse, pinball=None, coverage=(0.0, 0.0), ece=0.0):
    item = dict(mae=mae, rmse=rmse, rank_ic=0.1, pinball=None)
    if pinball is not None:
        item.update(
            pinball={str(level): pinball for level in em.LEVELS},
            mean_pinball=pinball,
            coverage_error={"0.8": coverage[0], "0.95": coverage[1]},
            sign_ece=ece,
        )
    return item


def test_the_ladder_delivers_the_most_complete_version_that_worsens_nothing():
    raw = _item(1.0, 2.0, 0.5, (0.05, 0.02), 0.10)
    point = _item(0.9, 1.9, 0.5, (0.05, 0.02), 0.10)
    full = _item(0.9, 1.9, 0.4, (0.01, 0.01), 0.02)
    assert em.ladder(dict(point=point, full=full), raw) == dict(
        delivered="full", worse=dict(full=[], point=[])
    )
    late = dict(point=point, full=_item(0.9, 1.9, 0.4, (0.01, 0.01), 0.11))
    assert em.ladder(late, raw)["delivered"] == "point"
    assert em.ladder(late, raw)["worse"]["full"] == ["sign_ece"]
    noisy = _item(0.9, 2.1, 0.5, (0.05, 0.02), 0.10)
    assert em.ladder(dict(point=noisy, full=noisy), raw)["delivered"] is None
    # Sin cuantiles se comparan solo el MAE y el RMSE; una métrica sin definir cuenta como peor.
    assert em.worse(_item(0.9, 1.9), raw) == []
    assert em.worse(dict(_item(0.9, 1.9), mae=None), raw) == ["mae"]


def test_relative_improvements_and_the_rmse_bound():
    control = _item(1.0, 2.0, 0.5, (0.01, 0.02), 0.03)
    result = em.relative(_item(0.99, 1.98, 0.45, (0.02, 0.01), 0.05), control)
    assert result["mae"] == pytest.approx(0.01) and result["rmse"] == pytest.approx(0.01)
    assert result["mean_pinball"] == pytest.approx(0.1)
    assert result["coverage_error"] == pytest.approx({"0.8": 0.01, "0.95": -0.01})
    assert result["sign_ece"] == pytest.approx(0.02)
    assert em.rmse_bound(0.14) == pytest.approx(1 - np.sqrt(1 - 0.0196))
    assert em.rmse_bound(None) is None


def _copy_run(paths, model, seed, *, shift, scale):
    """Otra ejecución con las filas del padre: punto y cuantiles escalados y movidos a la vez."""
    parent = paths["state"] / "runs" / "titans_mac_nogain" / str(core.YEAR) / "seed-42"
    folder = paths["state"] / "runs" / model / str(core.YEAR) / f"seed-{seed}"
    folder.mkdir(parents=True)
    rng = np.random.default_rng(seed + len(model))
    for split in core.SESSIONS:
        table = pq.read_table(parent / f"{split}-predictions.parquet")
        noise = shift * rng.standard_normal(table.num_rows)
        columns = {name: table.column(name) for name in table.column_names}
        for name in ("prediction", *core.ac.QUANTILES):
            values = scale * table.column(name).to_numpy().astype(np.float64) + noise
            columns[name] = pa.array(values, pa.float32())
        pq.write_table(pa.table(columns), folder / f"{split}-predictions.parquet")
    record = dict(
        status="completed",
        final_test_opened=False,
        run_id=f"{model}-{seed}",
        request=dict(view_sha256="f" * 64, seed=seed),
    )
    (folder / "run.json").write_text(json.dumps(record))


def _arguments(paths, output, *extra):
    arguments = core._arguments(paths, output, *extra)
    arguments[arguments.index("--declaration") + 1] = str(DECLARATION)
    return arguments


def _columns(paths, split):
    folder = paths["state"] / "runs" / "titans_mac_nogain" / str(core.YEAR) / "seed-42"
    table = pq.read_table(folder / f"{split}-predictions.parquet").sort_by(og.ORDER)
    ids = table.column("sample_id").to_pylist()
    target = table.column("target").to_numpy().astype(np.float64)
    prediction = table.column("prediction").to_numpy().astype(np.float64)
    return ids, target, prediction, table.column("prediction_at").cast("int64").to_numpy()


def test_f7_of_an_origin(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(core, "ASSETS", 96)
    paths = core.campaign(tmp_path)
    for seed in (43, 44):
        _copy_run(paths, "titans_mac_nogain", seed, shift=0.1, scale=1.0)
    _copy_run(paths, "titans_mac", 42, shift=0.2, scale=0.3)
    _, target, prediction, _ = _columns(paths, "calibration")
    monkeypatch.setattr(em.ac, "known_rows", core._known(paths))
    output = tmp_path / "errores"
    assert em.main(_arguments(paths, output)) == 0
    folder = output / str(core.YEAR)
    summary = json.loads((folder / "resumen.json").read_text(encoding="utf-8"))
    assert summary["references"]["included"] == ["core"]
    assert list(summary["references"]["absent"]) == ["noic"]
    label, operable = summary["modes"]["label"], summary["modes"]["operable"]
    # El agregador ya exige que toda etiqueta de calibración madure antes de la evaluación.
    assert label["rows"]["matured"] == label["rows"]["calibration"]
    assert set(label["predictors"]) == {"parent", "seeds", "core", "s1", "s2"}
    assert set(label["predictors"]["parent"]["versions"]) == {"raw", "point", "full"}
    assert set(label["predictors"]["s2"]["versions"]) == {"point", "full"}
    assert label["predictors"]["s2"]["verdict"]["reference"] == "zero"
    assert operable["predictors"]["s2"]["verdict"]["reference"] == "alpha"
    # El λ del padre y su MAE en evaluación, recalculados aparte.
    weights = np.full(len(target), 1.0 / core.ASSETS)
    factor = sd.shrinkage_factor(target, prediction, weights)
    item = label["predictors"]["parent"]
    assert item["parameters"]["lad"]["US"] == pytest.approx(factor)
    _, y, p, _ = _columns(paths, "evaluation")
    assert item["versions"]["point"]["mae"] == pytest.approx(np.mean(np.abs(y - factor * p)))
    assert item["versions"]["raw"]["mae"] == pytest.approx(np.mean(np.abs(y - p)))
    assert item["versions"]["raw"]["rank_ic"] == pytest.approx(item["versions"]["point"]["rank_ic"])
    assert label["controls"]["zero"]["mae"] == pytest.approx(np.mean(np.abs(y)))
    # El alfa solo en lo operable deja el error de la etiqueta: |y + α − α| = |y|.
    assert operable["controls"]["alpha"]["mae"] == pytest.approx(label["controls"]["zero"]["mae"])
    assert summary["operable_factors"]["parent"] in declaration()["operable_scales"]
    report = (folder / "resumen.md").read_text(encoding="utf-8")
    assert "Lo que cabe esperar" in report and "noic" in report
    assert pq.read_table(folder / "sesiones.parquet").num_rows > 0
    capsys.readouterr()
    assert em.main(_arguments(paths, output)) == 0
    assert "ya calculado" in capsys.readouterr().out
    # Con el noic emitido cambian las entradas: la raíz publicada no se mezcla.
    _copy_run(paths, "titans_mac_noic", 42, shift=0.05, scale=0.25)
    assert em.main(_arguments(paths, output)) == 1
    other = tmp_path / "errores-noic"
    assert em.main(_arguments(paths, other)) == 0
    summary = json.loads((other / str(core.YEAR) / "resumen.json").read_text(encoding="utf-8"))
    assert summary["references"]["included"] == ["core", "noic"]
    reading = summary["modes"]["label"]["predictors"]["parent"]["verdict"]["noic"]
    assert {"mae", "mean_pinball", "mae_versus", "rank_ic"} <= set(reading)
    report = other / str(core.YEAR) / "resumen.md"
    text = report.read_text(encoding="utf-8")
    assert "Nivel de referencia" in text
    report.unlink()
    assert em.main(["informe", "--summary", str(report.parent / "resumen.json")]) == 0
    assert report.read_text(encoding="utf-8") == text
