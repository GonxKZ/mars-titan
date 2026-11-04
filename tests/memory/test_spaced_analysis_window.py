"""La declaración E2 se confirma antes de emitir sobre fuentes de fixture."""

import json
from types import SimpleNamespace

import pytest
from test_financial_session_controls import no_target_estimation as no_target_estimation
from test_frozen_financial import frozen_backend as frozen_backend
from test_native_episode_backend import native as native

from tests.memory.spaced_analysis_fixture import analysis_document
from tests.memory.spaced_window_fixture import study_fixture
from tests.memory.test_spaced_regional_selected import components


@pytest.fixture(scope="module", params=("standard", "regional"))
def window(request, tmp_path_factory):
    contract = request.param
    fixture = study_fixture(
        tmp_path_factory.mktemp(f"analysis-window-{contract}"),
        assets=2,
        rich=True,
        components=components("ring") if contract == "regional" else None,
        reader_contract=contract,
    )
    document = analysis_document(json.loads(fixture["study"].read_text()), contract)
    fixture["study"].write_text(json.dumps(document))
    return fixture


@pytest.fixture(autouse=True)
def technical_fixtures_only(monkeypatch, window):
    from mars_titan.training import spaced_inference, spaced_window

    assert window["dataset"].manifest["technical_fixture"] is True
    monkeypatch.setattr(spaced_window, "require_learning_allowed", lambda *_: None)
    monkeypatch.setattr(spaced_inference, "require_learning_allowed", lambda *_: None)


def run(window, output, *, arm="utility", **options):
    from mars_titan.training.spaced_window import run_spaced_window

    return run_spaced_window(
        window["view"],
        window["anchor"],
        window["source_manifest"],
        window["study"],
        arm=arm,
        output=output,
        device="cpu",
        indices=window["indices"],
        **options,
    )


def test_analysis_is_recorded_before_the_first_phase(window, native, tmp_path, monkeypatch):
    from mars_titan.training.spaced_inference import SpacedInference

    output = tmp_path / "run"
    expected = json.loads(window["study"].read_text())["analysis"]
    evaluate = SpacedInference.evaluate
    calls = []

    def observe(self, *args, **kwargs):
        record = json.loads((output / "run.json").read_text())
        assert record["request"]["study"]["analysis"] == expected
        assert record["identity"]["request"]["study"]["analysis"] == expected
        calls.append(record["status"])
        return evaluate(self, *args, **kwargs)

    monkeypatch.setattr(SpacedInference, "evaluate", observe)
    result = run(window, output)
    assert result["status"] == "completed"
    assert calls == ["running"] * 3
    assert set(result["predictions"]) == {"validation", "calibration", "evaluation"}


def test_resume_rejects_a_changed_margin_before_loading_weights(
    window, native, tmp_path, monkeypatch
):
    from mars_titan.training import spaced_window

    output = tmp_path / "paused"
    assert run(window, output, stop=SimpleNamespace(requested=True))["status"] == "paused"
    before = (output / "run.json").read_bytes()
    document = json.loads(window["study"].read_text())
    document["analysis"]["min_relevant_mae"] = 0.00002
    changed = tmp_path / "changed-study.json"
    changed.write_text(json.dumps(document))
    monkeypatch.setattr(
        spaced_window,
        "load_selected_reader",
        lambda *_a, **_k: pytest.fail("Cargó pesos después de cambiar la declaración"),
    )
    with pytest.raises(ValueError, match="petición|estado"):
        run(dict(window, study=changed), output)
    assert (output / "run.json").read_bytes() == before


def test_resume_with_the_same_analysis_keeps_the_confirmed_phases(
    window, native, tmp_path, monkeypatch
):
    from mars_titan.training.spaced_inference import SpacedInference

    output = tmp_path / "completed"
    first = run(window, output)
    monkeypatch.setattr(
        SpacedInference, "evaluate", lambda *_a, **_k: pytest.fail("Repitió las fases confirmadas")
    )
    assert run(window, output) == first


def test_real_runner_artifacts_feed_the_declared_comparator(window, native, tmp_path, monkeypatch):
    from mars_titan.evaluation import spaced_comparison
    from mars_titan.memory.spaced_study import load_study

    config, _ = load_study(window["study"])
    reports = {}
    for arm in config.arms:
        output = tmp_path / arm
        assert run(window, output, arm=arm)["status"] == "completed"
        reports[arm] = output / "run.json"
    monkeypatch.setattr(spaced_comparison, "require_learning_allowed", lambda *_: None)
    result = spaced_comparison.analyze_spaced_runs(
        reports, window["study"], partition="evaluation", output=tmp_path / "comparison"
    )
    assert result["status"] == "completed"
    assert result["reader_contract"] == config.reader_contract
    assert result["scores"]["baseline"] == result["scores"]["reservoir"]
    assert result["primary"]["assessment"] == "inconcluyente"
    assert result["primary"]["comparison"]["resampling"]["reason"]


def test_adding_an_analysis_does_not_change_the_emitted_predictions(window, native, tmp_path):
    import pyarrow.parquet as pq

    declared = tmp_path / "declared"
    run(window, declared)
    document = json.loads(window["study"].read_text())
    document.pop("analysis")
    regional = document["reader_contract"] == "regional"
    document["schema_version"] = 2 if regional else 1
    if not regional:
        document.pop("reader_contract")
    legacy_study = tmp_path / "legacy-study.json"
    legacy_study.write_text(json.dumps(document))
    legacy = tmp_path / "legacy"
    run(dict(window, study=legacy_study), legacy)
    for part in ("validation", "calibration", "evaluation"):
        assert pq.read_table(declared / f"{part}-predictions.parquet").equals(
            pq.read_table(legacy / f"{part}-predictions.parquet")
        )
