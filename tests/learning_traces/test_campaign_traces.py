"""Trazas de Ridge, XGBoost y de la declaración de ejecución de la campaña (#448).

Ridge y XGBoost no se ajustan aquí: se comprueban los eventos que escriben con un modelo y
un estado de selección fijados en la prueba, y la recuperación de su cursor.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mars_titan.learning_traces.hooks import open_recorder, unit_config
from mars_titan.learning_traces.recorder import read_traces
from mars_titan.training import campaign_resources, masked_campaign
from mars_titan.training.external_corpus import _trace_round
from mars_titan.training.tabular_corpus import _trace_ridge

EXECUTION = Path("configs/baselines/historical-masked-campaign-execution.json")
MIB = 1024**2


def test_ridge_writes_one_fit_event_with_its_coefficients(tmp_path):
    report = dict(model="ridge", alpha=1.0, manifest_sha256="a" * 64, code={})
    model = SimpleNamespace(coefficient=np.array([1.0, -3.0, 0.5]), intercept=0.25)
    predictions = dict(validation=dict(metrics=dict(session_mae=0.125)))
    _trace_ridge(tmp_path / "plain", None, dict(report), model, predictions)
    assert not (tmp_path / "plain" / "traces").exists()
    _trace_ridge(tmp_path, dict(every=50, max_bytes=MIB), report, model, predictions)
    manifest, table = read_traces(tmp_path / "traces")
    assert manifest["step_unit"] == "fit" and manifest["identity"]["family"] == "ridge"
    rows = {(r["metric"], r["stat"]): r for r in table.to_pylist()}
    assert {(r["step"], r["phase"]) for r in rows.values()} == {(0, "validation")}
    assert rows[("validation.score", "value")]["value"] == 0.125
    assert rows[("ridge.intercept", "value")]["value"] == 0.25
    assert rows[("ridge.coefficient", "abs_max")]["value"] == 3.0
    assert report["traces"] == dict(sequence=1, bytes=manifest_bytes(tmp_path), exhausted_at=None)


def manifest_bytes(folder):
    return sum(path.stat().st_size for path in (folder / "traces/parts").glob("part-*.parquet"))


def test_xgboost_rounds_follow_the_cadence_and_resume_from_the_confirmed_cursor(tmp_path):
    config = dict(every=2, max_bytes=MIB)
    recorder = open_recorder(
        tmp_path / "traces", config, dict(family="xgboost"), step_unit="boosting_round"
    )

    def state(count):
        return dict(
            last_session_mae=1 / count, best_session_mae=1 / count, rounds_without_improvement=0
        )

    _trace_round(None, 1, state(1))
    for count in (1, 2):
        _trace_round(recorder, count, state(count))
    recorder.flush()
    confirmed = recorder.state_dict()
    for count in (3, 4):
        _trace_round(recorder, count, state(count))
    recorder.flush()
    # Corte antes de confirmar la ronda 4: se repite desde el cursor de la ronda 2.
    resumed = open_recorder(
        tmp_path / "traces", config, dict(family="xgboost"), step_unit="boosting_round"
    )
    resumed.load_state_dict(confirmed)
    for count in (3, 4):
        _trace_round(resumed, count, state(count))
    resumed.flush()
    manifest, table = read_traces(tmp_path / "traces")
    rows = table.to_pylist()
    assert manifest["step_unit"] == "boosting_round"
    assert sorted({r["step"] for r in rows}) == [2, 4] and len(rows) == 6
    score = {r["step"]: r["value"] for r in rows if r["metric"] == "validation.score"}
    assert score == {2: 0.5, 4: 0.25}


def write_execution(path, traces):
    document = json.loads(EXECUTION.read_text())
    document.pop("learning_traces", None)
    if traces is not None:
        document["learning_traces"] = traces
    path.write_text(json.dumps(document))
    return path


def test_the_execution_declares_cadence_per_unit_and_budget(tmp_path):
    declared = dict(every=50, max_mib_per_job=8, every_by_unit=dict(wave=1, transition=1024))
    execution = campaign_resources.load_execution(write_execution(tmp_path / "e.json", declared))
    assert execution.traces == dict(
        every=50, max_bytes=8 * MIB, every_by_unit=dict(transition=1024, wave=1)
    )
    assert unit_config(execution.traces, "wave").every == 1
    assert unit_config(execution.traces, "write_event").every == 50
    assert campaign_resources.execution_traces(None) is None
    plain = campaign_resources.load_execution(write_execution(tmp_path / "p.json", None))
    assert plain.traces is None
    for wrong in (
        dict(declared, max_mib_per_job=0),
        dict(declared, every=0),
        dict(declared, every_by_unit=dict(optimizer_step=2)),
        dict(declared, every_by_unit=dict(epoch=2)),
        dict(declared, every_by_unit=dict(wave=0)),
        dict(declared, extra=1),
    ):
        with pytest.raises(ValueError, match="trazas"):
            campaign_resources.load_execution(write_execution(tmp_path / "w.json", wrong))


def test_the_disk_projection_adds_the_trace_bound_of_pending_jobs():
    jobs = [
        dict(id="a", kind="fit"),
        dict(id="b", kind="fit"),
        dict(id="c", kind="online"),
        dict(id="d", kind="predict"),
    ]
    projection = dict(peak_bytes=100)
    execution = SimpleNamespace(traces=dict(every=50, max_bytes=8 * MIB))
    traced = masked_campaign._with_traces(projection, jobs, {"a"}, execution)
    assert traced["peak_bytes"] == 100 + 2 * 8 * MIB
    assert traced["learning_traces"] == dict(jobs=2, bound_bytes=2 * 8 * MIB)
    plain = masked_campaign._with_traces(projection, jobs, set(), SimpleNamespace(traces=None))
    assert plain is projection
