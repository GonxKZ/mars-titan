"""Ventanas E3 sobre pesos literales de fixture, sin entrenador ni pasos de optimizador."""

import json

import pyarrow.parquet as pq
import pytest
import torch

from mars_titan.memory.salience_budget import SalienceBudgetConfig
from mars_titan.training import mars_titan_correction as mc
from mars_titan.training.learning_hold import LearningHoldError
from tests.memory.spaced_selected_fixture import selected_fixture
from tests.memory.spaced_window_fixture import study_fixture


@pytest.fixture(scope="module", autouse=True)
def frozen_backend():
    previous = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    yield
    torch.backends.mha.set_fastpath_enabled(previous)


@pytest.fixture(autouse=True)
def only_manual_frozen_parameters(learning_doubles):
    """El padre lo guarda study_fixture con pesos literales y sin entrenador."""


@pytest.fixture(scope="module")
def window(tmp_path_factory):
    return study_fixture(tmp_path_factory.mktemp("salience-window"), assets=4, rich=True)


def run(window, output, salience=None, **options):
    return mc.run_correction_window(
        window["view"],
        window["anchor"].parent / "parent",
        "configs/titans/mature-correction-historical-masked.json",
        components={"associative_memory": {"rule": "proximal", "key": "codec"}},
        seed=42,
        output=output,
        search_case="eta25e-2",
        device="cpu",
        indices=window["indices"],
        salience=salience,
        **options,
    )


def test_window_persists_the_factor_and_preserves_all_rows_and_parent(window, tmp_path):
    baseline = run(window, tmp_path / "baseline")
    config = SalienceBudgetConfig(writes_per_market=1)
    active = run(window, tmp_path / "active", config)
    assert "salience" not in baseline["request"] and "salience" not in baseline["identity"]
    assert active["request"]["salience"] == active["identity"]["salience"] == config.identity()
    assert baseline["run_id"] != active["run_id"]
    assert baseline["identity"]["variant_sha256"] != active["identity"]["variant_sha256"]
    assert baseline["identity"]["parent"] == active["identity"]["parent"]
    selected = json.loads((tmp_path / "active/selected.json").read_text())
    assert selected["salience"] == config.identity()
    assert selected["variant_sha256"] == active["identity"]["variant_sha256"]
    for part in ("validation", "calibration", "evaluation"):
        before = pq.read_table(tmp_path / "baseline" / f"{part}-predictions.parquet")
        after = pq.read_table(tmp_path / "active" / f"{part}-predictions.parquet")
        for name in ("sample_id", "prediction_at", "target"):
            assert before[name].equals(after[name])
        scores = active["predictions"][part]["metrics"]["salience"]
        assert scores["observed"] == after.num_rows and scores["selected"] < scores["observed"]


def test_regeneration_keeps_the_chosen_factor_and_exact_predictions(window, tmp_path):
    config = SalienceBudgetConfig(writes_per_market=1, rule="uniform", seed=29)
    anchor = run(window, tmp_path / "anchor", config)
    report = mc.carry_correction(
        tmp_path / "anchor",
        window["view"],
        window["view"],
        tmp_path / "regenerated",
        device="cpu",
        regenerate=True,
    )
    assert report["salience"] == config.identity()
    for part in ("validation", "calibration", "evaluation"):
        actual = report["predictions"][part]["metrics"]
        assert "salience" in actual
        assert actual["salience"] == anchor["predictions"][part]["metrics"]["salience"]
        assert actual["associative_writes"] == actual["salience"]["selected"]
        assert pq.read_table(tmp_path / "anchor" / f"{part}-predictions.parquet").equals(
            pq.read_table(tmp_path / "regenerated" / f"{part}-predictions.parquet")
        )


def test_completed_phases_are_verified_and_a_changed_factor_is_rejected(
    window, tmp_path, monkeypatch
):
    config = SalienceBudgetConfig(writes_per_market=1)
    expected = run(window, tmp_path / "active", config)
    from mars_titan.training.salience_inference import SalienceCorrectionInference

    monkeypatch.setattr(
        SalienceCorrectionInference,
        "predict",
        lambda *_a, **_k: pytest.fail("Se repitió una fase ya confirmada"),
    )
    assert run(window, tmp_path / "active", config) == expected
    with pytest.raises(ValueError, match="petición"):
        run(window, tmp_path / "active", SalienceBudgetConfig(writes_per_market=2))


def test_guard_still_precedes_recipe_reads_and_output_creation(
    tmp_path, learning_hold, monkeypatch
):
    learning_hold(False)

    def forbidden(*_a, **_k):
        pytest.fail("Se leyó la receta antes de comprobar el bloqueo")

    monkeypatch.setattr(mc, "load_correction_recipe", forbidden)
    with pytest.raises(LearningHoldError):
        mc.run_correction_window(
            "absent",
            "absent",
            "absent",
            components={},
            seed=42,
            output=tmp_path / "blocked",
            search_case="eta25e-2",
            salience=SalienceBudgetConfig(),
        )
    assert not (tmp_path / "blocked").exists()


def test_carry_rejects_an_anchor_that_disagrees_about_the_factor(window, tmp_path, monkeypatch):
    run(window, tmp_path / "anchor", SalienceBudgetConfig(writes_per_market=1))
    path = tmp_path / "anchor/run.json"
    report = json.loads(path.read_text())
    report["identity"]["salience"]["writes_per_market"] = 2
    path.write_text(json.dumps(report))
    monkeypatch.setattr(
        mc,
        "_sources",
        lambda *_a, **_k: pytest.fail("Se abrieron datos con una identidad E3 contradictoria"),
    )
    with pytest.raises(ValueError, match="E3"):
        mc.carry_correction(
            tmp_path / "anchor",
            window["view"],
            window["view"],
            tmp_path / "invalid",
            device="cpu",
            regenerate=True,
        )


def test_pausing_after_a_phase_and_resuming_keeps_the_same_predictions(
    window, tmp_path, monkeypatch
):
    from mars_titan.training.salience_inference import SalienceCorrectionInference

    config = SalienceBudgetConfig(writes_per_market=1)
    output = tmp_path / "paused"

    class AfterValidation:
        @property
        def requested(self):
            return (output / "validation-predictions.parquet").is_file()

    paused = run(window, output, config, stop=AfterValidation())
    assert paused["status"] == "paused" and set(paused["predictions"]) == {"validation"}
    phases = []
    original = SalienceCorrectionInference.predict

    def observed(self, source, rows, **options):
        phases.append(source.phase.partition)
        return original(self, source, rows, **options)

    monkeypatch.setattr(SalienceCorrectionInference, "predict", observed)
    resumed = run(window, output, config)
    assert resumed["status"] == "completed" and phases == ["calibration", "evaluation"]
    run(window, tmp_path / "reference", config)
    for part in resumed["predictions"]:
        assert pq.read_table(output / f"{part}-predictions.parquet").equals(
            pq.read_table(tmp_path / "reference" / f"{part}-predictions.parquet")
        )


def test_quantile_parent_keeps_its_existing_columns_and_reproduces_regeneration(window, tmp_path):
    from mars_titan.models.quantile_head import QUANTILE_COLUMNS

    selected = selected_fixture(tmp_path / "quantile", window["source"], quantiles=True)
    context = dict(window, anchor=selected["anchor"])
    config = SalienceBudgetConfig(writes_per_market=1)
    active = run(context, tmp_path / "active", config)
    mc.carry_correction(
        tmp_path / "active",
        window["view"],
        window["view"],
        tmp_path / "repeat",
        device="cpu",
        regenerate=True,
    )
    for part in active["predictions"]:
        table = pq.read_table(tmp_path / "active" / f"{part}-predictions.parquet")
        assert set(QUANTILE_COLUMNS) <= set(table.column_names)
        assert table.equals(pq.read_table(tmp_path / "repeat" / f"{part}-predictions.parquet"))


def test_invalid_factor_is_rejected_before_loading_the_parent(window, tmp_path, monkeypatch):
    from mars_titan.training import mars_titan_walk_forward

    monkeypatch.setattr(
        mars_titan_walk_forward,
        "_parent",
        lambda *_a, **_k: pytest.fail("Se abrió el padre con una configuración E3 inválida"),
    )
    with pytest.raises(ValueError, match="E3"):
        run(window, tmp_path / "invalid", {})
    assert not (tmp_path / "invalid").exists()


def test_cost_harness_preserves_populations_and_the_manual_parent(window, tmp_path):
    from benchmarks.salience_window_cost_check import measure_window_cost

    measured = measure_window_cost(window, tmp_path / "cost", repeats=1)
    assert measured["parent_unchanged"]
    assert len(measured["cases"]) == 8
    assert {item["arm"] for item in measured["summaries"]} == {
        "baseline",
        "standardized_error",
        "absolute_error",
        "uniform",
    }
    assert all(
        item["samples"] == 1 and item["median_seconds"] > 0 for item in measured["summaries"]
    )
    saved = json.loads((tmp_path / "cost/cost-progress.json").read_text())
    assert saved["measurement_state"] == "completed" and len(saved["cases"]) == 8


def test_cost_harness_rejects_reused_output_before_timing_any_window(window, tmp_path, monkeypatch):
    from benchmarks import salience_window_cost_check as benchmark

    output = tmp_path / "used"
    output.mkdir()
    monkeypatch.setattr(
        benchmark,
        "run",
        lambda *_a, **_k: pytest.fail(
            "No se puede medir una ventana ya completada como inferencia nueva"
        ),
    )
    with pytest.raises(ValueError, match="salida nueva"):
        benchmark.measure_window_cost(window, output, repeats=1)


def test_cost_harness_keeps_confirmed_samples_when_a_later_window_fails(
    window, tmp_path, monkeypatch
):
    from benchmarks import salience_window_cost_check as benchmark

    original = benchmark.run
    calls = 0

    def fail_later(*args, **options):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("Fallo técnico después de una muestra confirmada")
        return original(*args, **options)

    monkeypatch.setattr(benchmark, "run", fail_later)
    with pytest.raises(RuntimeError, match="muestra confirmada"):
        benchmark.measure_window_cost(window, tmp_path / "cost", repeats=1)
    saved = json.loads((tmp_path / "cost/cost-progress.json").read_text())
    assert saved["measurement_state"] == "partial" and len(saved["cases"]) == 1
