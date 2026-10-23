import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mars_titan.data.storage import sha256
from mars_titan.evaluation import scale_diagnostics as sd
from scripts import diagnose_prediction_scale as runner

DAY = 86_400_000_000
PHASES = dict(validation=(0, 30), calibration=(30, 20), evaluation=(50, 40))


def _table(phase, prediction, *, assets=8, seed=0):
    """Predicciones de un pilot sintético con los mismos activos y etiquetas en cada fase."""
    start, sessions = PHASES[phase]
    rng = np.random.default_rng(1000 + start)
    days = np.repeat(np.arange(start, start + sessions), assets)
    names = np.tile([f"US/A{i}" for i in range(assets)], sessions)
    target = rng.normal(scale=0.01, size=len(days))
    return pa.table(
        dict(
            sample_id=[f"{name}/{day}" for name, day in zip(names, days, strict=True)],
            asset_id=names,
            market=["US"] * len(days),
            prediction_at=pa.array(days * DAY, pa.timestamp("us", tz="UTC")),
            target=target,
            prediction=prediction(target, np.random.default_rng(seed)),
        )
    )


def _write_run(folder, prediction, *, seed=0, phases=PHASES):
    folder.mkdir(parents=True)
    for phase in phases:
        pq.write_table(
            _table(phase, prediction, seed=seed), folder / f"{phase}-predictions.parquet"
        )


def _declaration(tmp_path, sources):
    path = tmp_path / "declaration.json"
    declaration = json.loads(json.dumps(DECLARATION))
    declaration["sources"] = sources
    declaration["rule"]["primary_source"] = sources[0]["name"]
    declaration["bootstrap"]["replicates"] = 200
    declaration["bootstrap"]["block_length"] = 4
    declaration["bootstrap"]["sensitivity_block_lengths"] = [2, 8]
    path.write_text(json.dumps(declaration))
    return path


def _inventory(tmp_path, name, runs):
    items = []
    for run in runs:
        for phase in PHASES:
            relative = f"runs/{run}/{phase}-predictions.parquet"
            items.append(
                dict(
                    kind="sample_predictions",
                    relative_path=relative,
                    phase=phase,
                    sha256=sha256(tmp_path / "root" / name / relative),
                )
            )
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(dict(artifacts=items)))
    return path


DECLARATION = json.loads(
    (runner.REPOSITORY / "configs/evaluation/scale-diagnostics-phase0.json").read_text()
)


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "REPOSITORY", tmp_path)
    monkeypatch.setattr(runner, "_commit", lambda: dict(commit="test", clean=True))
    return tmp_path


def _signal(target, rng):
    return 5 * target + rng.normal(scale=0.002, size=len(target))


def _noise(target, rng):
    return rng.normal(scale=0.05, size=len(target))


def test_runner_deduplicates_signed_zero_and_records_skipped_folders(isolated):
    root = isolated / "root"
    _write_run(root / "pilot/runs/signal-42", _signal)
    _write_run(root / "pilot/runs/noise-43", _noise, seed=3)
    _write_run(root / "layout/positive-1", lambda t, r: np.zeros(len(t)))
    _write_run(root / "layout/negative-2", lambda t, r: -np.zeros(len(t)))
    _write_run(root / "layout/frozen-3", _noise, phases=["evaluation"])
    inventory = _inventory(isolated, "pilot", ["signal-42", "noise-43"])
    path = _declaration(
        isolated,
        [
            dict(name="pilot", inventory=inventory.name, directory="pilot"),
            dict(name="layout", layout="runs", directory="layout"),
        ],
    )
    summary = runner.run(path, root, isolated / "out")
    predictors = {p["id"]: p for cohort in summary["cohorts"] for p in cohort["predictors"]}
    # Las carpetas se recorren en orden alfabético: el cero negativo llega antes y el positivo
    # queda como alias suyo, no como otro predictor.
    assert "layout/positive-1" not in predictors
    assert predictors["layout/negative-2"]["aliases"] == ["layout/positive-1"]
    assert summary["skipped"] == [dict(id="layout/frozen-3", partitions=["evaluation"])]
    assert summary["inventories"] == {"pilot": sha256(inventory)}
    verdict = summary["verdict"]
    assert verdict["ic_distinguishable_positive"] == ["pilot/runs/signal-42"]
    assert set(verdict["family_sensitivity"]) == {"primary_source", "inventory_sources"}
    assert verdict["robust_to_family"] is True


def test_runner_rejects_a_changed_inventory_file(isolated):
    root = isolated / "root"
    _write_run(root / "pilot/runs/signal-42", _signal)
    inventory = _inventory(isolated, "pilot", ["signal-42"])
    path = _declaration(isolated, [dict(name="pilot", inventory=inventory.name, directory="pilot")])
    pq.write_table(
        _table("evaluation", _noise), root / "pilot/runs/signal-42/evaluation-predictions.parquet"
    )
    with pytest.raises(ValueError, match="huella"):
        runner.run(path, root, isolated / "out")


def test_past_median_uses_the_evaluation_rows_in_time_order():
    tables = [_table(phase, _signal).select(runner.KEYS) for phase in PHASES]
    result = runner._past_median_mae(
        tables, tables[-1], dict(lag_sessions=2, window=60, min_labels=20)
    )
    keys = pa.concat_tables(tables)
    sessions = keys["prediction_at"].cast(pa.int64()).to_numpy() // DAY
    expected = sd.past_median(
        np.asarray(keys["asset_id"].to_pylist()),
        sessions,
        keys["target"].to_numpy(),
        lag=2,
        window=60,
        min_labels=20,
    )[-len(tables[-1]) :]
    evaluation = tables[-1].set_column(
        tables[-1].schema.get_field_index("prediction"), "prediction", pa.array(expected)
    )
    panel = runner.ForecastPanel.from_arrow(evaluation)
    assert np.array_equal(result, sd.session_mae(panel, panel.prediction))
    with pytest.raises(ValueError, match="última partición"):
        runner._past_median_mae(tables, tables[0], dict(lag_sessions=2, window=60, min_labels=20))
    with pytest.raises(ValueError, match="orden temporal"):
        runner._past_median_mae(
            [tables[1], tables[0], tables[2]],
            tables[2],
            dict(lag_sessions=2, window=60, min_labels=20),
        )
