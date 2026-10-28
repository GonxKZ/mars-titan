"""B4 emite diagnósticos junto al pronóstico original, sin cambiar su trayectoria."""

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from test_financial_session_associative import four_flow_source as four_flow_source
from test_financial_session_associative import module_backend as module_backend
from test_financial_session_associative import no_target_estimation as no_target_estimation
from test_mars_titan_correction_parity import RECIPE, core, corrected_pass, emitted, shifted
from test_mars_titan_session_parity import PHASE, events
from test_multiscale_session import AREA, BRIDGE, ROLE

from mars_titan.memory.lesion_protocol import LesionProtocol
from mars_titan.memory.regimes import RegimeRule
from mars_titan.training.mars_titan_correction import CorrectionInference, prediction_rows

PROTOCOL = LesionProtocol(seeds=(0,), fractions=((1, 2),))
CASES = [
    replace(AREA, multiscale=None),
    replace(BRIDGE, multiscale=None),
    replace(ROLE, multiscale=replace(ROLE.multiscale, loss_scale=1.0)),
]


@pytest.mark.parametrize("correction", CASES, ids=["areas", "bridge", "e1_roles"])
def test_window_keeps_emissions_and_memory_and_writes_paired_columns(
    four_flow_source, correction, monkeypatch
):
    from test_regime_routing import daily_routes

    if correction.routing is not None:
        monkeypatch.setattr(RegimeRule, "routes", daily_routes)
    original, metrics = corrected_pass(four_flow_source, correction)
    inference = CorrectionInference(
        core(four_flow_source),
        RECIPE,
        correction,
        four_flow_source["codec"],
        audit=True,
        lesions=PROTOCOL,
    )
    rows = prediction_rows(inference)
    result = inference._pass(SimpleNamespace(phase=PHASE), events(four_flow_source), rows=rows)
    table = pa.concat_tables(rows.finish())
    assert emitted(inference) == emitted(original)
    assert inference.memory.export()["matrix_sha256"] == original.memory.export()["matrix_sha256"]
    if correction.multiscale is not None:
        assert inference.memory.export()["state_sha256"] == original.memory.export()["state_sha256"]
        assert not torch.equal(
            inference.memory.weights, torch.full((3,), 1 / 3, dtype=torch.float64)
        )
    assert result["lesions"]["cases"]["intact"]["session_mae"] == metrics["session_mae"]
    assert table["lesion_intact"].to_pylist() == table["prediction"].to_pylist()
    assert len(table) == result["labels"] == 16
    assert all("lesion_" + name in table.column_names for name in inference.lesion_plan.names)


def test_later_targets_do_not_rewrite_earlier_lesion_predictions(four_flow_source):
    def run(sequence):
        inference = CorrectionInference(
            core(four_flow_source), RECIPE, AREA, four_flow_source["codec"], lesions=PROTOCOL
        )
        rows = prediction_rows(inference)
        inference._pass(SimpleNamespace(phase=PHASE), sequence, rows=rows)
        return pa.concat_tables(rows.finish())

    original = run(events(four_flow_source))
    changed = run(shifted(four_flow_source, 128, label=1.0))
    for name in original.column_names:
        if name == "prediction" or name.startswith("lesion_"):
            assert original[name].to_pylist()[:12] == changed[name].to_pylist()[:12]


def test_emitted_columns_reach_the_session_curve_analysis(four_flow_source, tmp_path, monkeypatch):
    from mars_titan.data.storage import atomic_json, sha256
    from mars_titan.evaluation import lesion_curves

    inference = CorrectionInference(
        core(four_flow_source), RECIPE, CASES[1], four_flow_source["codec"], lesions=PROTOCOL
    )
    rows = prediction_rows(inference)
    metrics = inference._pass(SimpleNamespace(phase=PHASE), events(four_flow_source), rows=rows)
    table = pa.concat_tables(rows.finish())
    run = tmp_path / "fixture-run"
    run.mkdir()
    path = run / "evaluation-predictions.parquet"
    pq.write_table(table, path, row_group_size=3)
    report = run / "run.json"
    atomic_json(
        report,
        dict(
            schema_version=1,
            kind="mars_titan_correction_window",
            status="completed",
            final_test_opened=False,
            request=dict(lesions=PROTOCOL.identity(), seed=0),
            predictions=dict(
                evaluation=dict(
                    path=path.name,
                    sha256=sha256(path),
                    rows=len(table),
                    bytes=path.stat().st_size,
                    metrics=metrics,
                )
            ),
        ),
    )
    config = json.loads(
        (
            Path(__file__).resolve().parents[2] / "configs/experiments/lesion-protocol-v1.json"
        ).read_text()
    )
    config["protocol"] = PROTOCOL.identity()
    config["analysis"].update(replicates=10)
    study = tmp_path / "fixture-study.json"
    atomic_json(study, config)
    monkeypatch.setattr(lesion_curves, "require_learning_allowed", lambda *_: None)
    result = lesion_curves.analyze_lesion_run(
        report, study, partition="evaluation", output=tmp_path / "curves", batch_size=3
    )
    assert result["rows"] == metrics["labels"] == 16
    assert result["scores"]["intact"]["point"]["mae"] == pytest.approx(metrics["session_mae"])
    assert result["source"]["configuration"] == inference.lesion_plan.identity()


def test_protocol_identity_can_be_restored_without_accepting_changed_semantics():
    identity = PROTOCOL.identity()
    assert LesionProtocol.from_identity(identity).identity() == identity
    with pytest.raises(ValueError):
        LesionProtocol.from_identity(identity | dict(scope="future_labels"))


def test_carried_diagnostics_require_the_original_source_versions():
    from mars_titan.training.mars_titan_correction import _lesion_code, diagnostic_protocol

    request = dict(lesions=PROTOCOL.identity(), code=_lesion_code())
    assert diagnostic_protocol(request).identity() == PROTOCOL.identity()
    assert diagnostic_protocol({}) is None
    bad = request | dict(code={name: "0" * 64 for name in request["code"]})
    with pytest.raises(ValueError, match="fuentes"):
        diagnostic_protocol(bad)


def test_quantile_outputs_are_unchanged_when_diagnostics_are_enabled(four_flow_source):
    from mars_titan.models.quantile_head import QUANTILE_COLUMNS
    from mars_titan.models.titans.financial import FinancialPredictor

    recipe = replace(RECIPE, loss="pinball")
    config = replace(core(four_flow_source).config, head="quantile_head_v1")
    tables = []
    for protocol in (None, PROTOCOL):
        predictor = (
            FinancialPredictor(config, dtype=torch.float64, device="cpu")
            .eval()
            .requires_grad_(False)
        )
        inference = CorrectionInference(
            predictor, recipe, CASES[0], four_flow_source["codec"], lesions=protocol
        )
        rows = prediction_rows(inference)
        inference._pass(SimpleNamespace(phase=PHASE), events(four_flow_source), rows=rows)
        tables.append(pa.concat_tables(rows.finish()))
    for name in ("prediction", *QUANTILE_COLUMNS):
        assert tables[0][name].equals(tables[1][name])


def test_diagnostic_window_obeys_the_learning_hold_before_any_input_read(tmp_path, monkeypatch):
    import json

    from mars_titan.training.learning_hold import LearningHoldError
    from mars_titan.training.mars_titan_correction import run_correction_window

    hold = tmp_path / "hold.json"
    hold.write_text(json.dumps(dict(training_allowed=False, condition="Comprobación técnica")))
    monkeypatch.setenv("MARS_TITAN_TRAINING_HOLD", str(hold))
    with pytest.raises(LearningHoldError):
        run_correction_window(
            tmp_path / "missing-view",
            tmp_path / "missing-parent",
            tmp_path / "missing-recipe",
            components={},
            seed=42,
            output=tmp_path / "out",
            search_case="eta",
            lesions=PROTOCOL,
        )
    assert not (tmp_path / "out").exists()
