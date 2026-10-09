"""Regeneración por inferencia de las predicciones de la campaña, en CPU y sin ajustes.

Los ajustes de las pruebas no cambian ningún peso: la referencia neuronal guarda como
estado elegido los pesos iniciales de su semilla y Titans-MAC usa un optimizador que solo
registra gradientes. Se comprueba que el traslado sobre la propia ventana repite bit a bit
las tablas que escribió el ajuste, también después de liberarlas, y que una tabla que no
se puede repetir se detecta en lugar de aceptarse.
"""

import json
import os

import numpy as np
import pyarrow.parquet as pq
import pytest

from mars_titan.data import prediction_files
from mars_titan.data.input_policy import HISTORICAL_MASKED
from mars_titan.data.storage import atomic_json, sha256
from mars_titan.training import campaign_plan as plan
from mars_titan.training import carried_predictions as carry
from mars_titan.training import masked_campaign as engine
from mars_titan.training import modality_ablation_stage as ablation
from mars_titan.training import prediction_regeneration as regeneration
from mars_titan.training import titans_walk_forward as wf
from mars_titan.training.corpus_inputs import CorpusDataset
from mars_titan.training.learning_hold import HOLD_ENV
from mars_titan.training.tabular_corpus import _predict
from tests.posttraining.campaign_fixture import CpuLease
from tests.training.test_carried_predictions import PresenceCount, tabular_anchor
from tests.training.test_carried_predictions import views as carried_views  # noqa: F401
from tests.training.test_modality_ablation_carry import titans  # noqa: F401
from tests.training.test_modality_ablation_native import episodic, readers  # noqa: F401
from tests.training.test_modality_ablation_stage import RUNNING, base, on_cpu  # noqa: F401
from tests.training.test_titans_walk_forward import unfused_attention

HELD_OUT = ("validation", "calibration", "evaluation")


@pytest.fixture(scope="module", autouse=True)
def strict_numerics():
    """Ajustes del fixture en FP32 estricto, como exige la campaña, y estado restaurado."""
    import torch

    before = (
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
        torch.get_float32_matmul_precision(),
    )
    regeneration.strict_fp32()
    yield
    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = before[:2]
    torch.set_float32_matmul_precision(before[2])


@pytest.fixture
def allowed(base, monkeypatch):  # noqa: F811
    on_cpu(monkeypatch)
    monkeypatch.setenv(HOLD_ENV, str(base.hold))
    return base


def jobs_of(base, **wanted):  # noqa: F811
    campaign = plan.load_campaign(base.campaign)
    return [
        job
        for job in plan.plan_campaign(campaign)
        if all(job[key] == value for key, value in wanted.items())
    ]


def test_a_neural_fit_regenerates_its_three_partitions_bit_for_bit(allowed, tmp_path):
    job = jobs_of(allowed, model="neural", kind=engine.FIT)[0]
    report = regeneration.regenerate_job(
        allowed.campaign, allowed.views, allowed.output, job["id"], tmp_path / "again"
    )
    assert report["identical"] is True and set(report["partitions"]) == set(HELD_OUT)
    assert all(item["same_file"] for item in report["partitions"].values())
    assert report["numerics"] == dict(
        matmul_allow_tf32=False, cudnn_allow_tf32=False, float32_matmul_precision="highest"
    )
    produced = json.loads((tmp_path / "again" / "run" / "carry.json").read_text())
    assert produced["regenerated"] is True and set(produced["predictions"]) == set(HELD_OUT)
    written = json.loads((tmp_path / "again" / "regeneration.json").read_text())
    assert written["identical"] is True and written["optimizer_steps"] == 0


def test_regeneration_still_matches_after_the_rows_are_released(allowed, tmp_path):
    job = jobs_of(allowed, model="neural", kind=engine.FIT)[-1]
    receipt = json.loads((allowed.output / "jobs" / job["id"] / "receipt.json").read_text())
    report_path = allowed.output / receipt["report"]["path"]
    report = json.loads(report_path.read_text())
    copies = tmp_path / "copies"
    copies.mkdir()
    released = []
    try:
        for record in report["predictions"].values():
            path = report_path.parent / record["path"]
            (copies / path.name).write_bytes(path.read_bytes())
            prediction_files.release(path, record["sha256"], job=job["id"])
            released.append((path, record["sha256"]))
        result = regeneration.regenerate_job(
            allowed.campaign, allowed.views, allowed.output, job["id"], tmp_path / "again"
        )
        assert result["identical"] is True
        assert {item["state"] for item in result["partitions"].values()} == {"released"}
    finally:
        # El fixture es compartido: se restauran los originales y su registro.
        for path, _ in released:
            path.write_bytes((copies / path.name).read_bytes())
        (report_path.parent / prediction_files.RETENTION_FILE).unlink(missing_ok=True)
    for path, digest in released:
        assert sha256(path) == digest


def flipped(regenerate):
    """Regenerador que cambia un bit de la evaluación, como una inferencia no determinista."""

    def run(job_run):
        produced = regenerate(job_run)
        path = job_run.folder / produced["predictions"]["evaluation"]["path"]
        table = pq.read_table(path)
        values = table["prediction"].to_numpy().copy()
        values[0] = np.nextafter(values[0], np.float32(np.inf))
        column = table.schema.get_field_index("prediction")
        pq.write_table(table.set_column(column, "prediction", [values]), path)
        return produced

    return run


def test_a_table_that_does_not_repeat_is_reported_as_different(allowed, tmp_path):
    job = jobs_of(allowed, model="neural", kind=engine.FIT)[0]
    available = engine.regenerators()
    report = regeneration.regenerate_job(
        allowed.campaign,
        allowed.views,
        allowed.output,
        job["id"],
        tmp_path / "again",
        regenerators={("neural", engine.FIT): flipped(available["neural", engine.FIT])},
    )
    assert report["identical"] is False
    assert report["partitions"]["evaluation"]["identical"] is False
    assert report["partitions"]["validation"]["identical"] is True


def test_regeneration_needs_a_new_destination_and_a_regenerator(allowed, tmp_path):
    job = jobs_of(allowed, model="neural", kind=engine.FIT)[0]
    (tmp_path / "used").mkdir()
    with pytest.raises(ValueError, match="destino nuevo"):
        regeneration.regenerate_job(
            allowed.campaign, allowed.views, allowed.output, job["id"], tmp_path / "used"
        )
    with pytest.raises(ValueError, match="no tiene regeneración"):
        regeneration.regenerate_job(
            allowed.campaign,
            allowed.views,
            allowed.output,
            job["id"],
            tmp_path / "none",
            regenerators={},
        )
    with pytest.raises(ValueError, match="no es un trabajo"):
        regeneration.regenerate_job(
            allowed.campaign, allowed.views, allowed.output, "US/fold-999/x", tmp_path / "x"
        )


def test_each_fit_model_has_a_regenerator_except_the_cm_v1_cores():
    available = engine.regenerators()
    fits = {model for model, kind in engine.EXECUTORS if kind == engine.FIT}
    assert {model for model, kind in available if kind == engine.FIT} == fits - {"cm_v1_core"}
    assert {key for key in available if key[1] == engine.CARRY} == {
        key for key in engine.EXECUTORS if key[1] == engine.CARRY
    }


def test_an_ablation_prediction_regenerates_bit_for_bit(allowed, tmp_path):
    output = tmp_path / "ablation"
    summary = ablation.run_stage(
        allowed.stage, allowed.views, allowed.output, output, lease=CpuLease, stop=RUNNING
    )
    assert summary["status"] == "completed"
    stage = ablation.load_stage(allowed.stage)
    job = ablation.plan_stage(stage)[0]
    report = regeneration.regenerate_ablation(
        allowed.stage, allowed.views, allowed.output, output, job["id"], tmp_path / "again"
    )
    assert report["identical"] is True and set(report["partitions"]) == {"evaluation"}
    assert report["variant"] == job["variant"]


def test_titans_regeneration_repeats_the_fitted_window(titans, tmp_path, monkeypatch):  # noqa: F811
    monkeypatch.setenv(HOLD_ENV, str(titans.hold))
    run = titans.original
    with unfused_attention():
        produced = wf.carry_titans(
            run.output, run.view, run.view, tmp_path / "again", device="cpu", regenerate=True
        )
    assert produced["regenerated"] is True and set(produced["predictions"]) == set(HELD_OUT)
    result = regeneration.compare(
        regeneration.originals(run.output, run.report), tmp_path / "again", produced
    )
    assert result["identical"] is True, result
    with pytest.raises(ValueError, match="sin ablación"):
        wf.carry_titans(
            run.output,
            run.view,
            run.view,
            tmp_path / "both",
            device="cpu",
            regenerate=True,
            modality_ablation="mask_news",
        )
    with pytest.raises(ValueError, match="propia ventana"):
        wf.carry_titans(
            run.output,
            run.view,
            titans.absent.view,
            tmp_path / "other",
            device="cpu",
            regenerate=True,
        )


def test_tabular_regeneration_repeats_the_writer_of_the_fit(
    carried_views,  # noqa: F811
    tmp_path,
    monkeypatch,
):
    """Ridge y XGBoost escriben con `_predict`, que el traslado regenerado repite."""
    model = PresenceCount()
    monkeypatch.setattr(carry, "_tabular_model", lambda *args: model)
    view = carried_views.view(0)
    anchor = tabular_anchor(tmp_path / "anchor", view)
    dataset = CorpusDataset(view, input_policy=HISTORICAL_MASKED)
    predictions = {}
    for partition in HELD_OUT:
        path = anchor / f"{partition}-predictions.parquet"
        _predict(model, None, dataset, partition, 3, path, presence=True)
        predictions[partition] = dict(path=path.name, sha256=sha256(path))
    report = json.loads((anchor / "run.json").read_text())
    atomic_json(anchor / "run.json", dict(report, predictions=predictions))
    produced = carry.carry_tabular(
        anchor,
        view,
        view,
        tmp_path / "again",
        kind="ridge",
        batch_size=3,
        input_policy=HISTORICAL_MASKED,
        regenerate=True,
    )
    assert produced["regenerated"] is True
    expected = regeneration.originals(anchor, dict(predictions=predictions))
    assert regeneration.compare(expected, tmp_path / "again", produced)["identical"] is True
    with pytest.raises(ValueError, match="propia ventana"):
        carry.carry_tabular(
            anchor,
            view,
            carried_views.view(1),
            tmp_path / "later",
            kind="ridge",
            batch_size=3,
            input_policy=HISTORICAL_MASKED,
            regenerate=True,
        )


@pytest.mark.parametrize(
    "numerics",
    [
        dict(
            float32_matmul_precision="highest", cuda_matmul_allow_tf32=False, cudnn_allow_tf32=True
        ),
        dict(matmul_precision="high", matmul_allow_tf32=False, cudnn_allow_tf32=False),
        dict(
            float32_matmul_precision="highest", cuda_matmul_allow_tf32=True, cudnn_allow_tf32=False
        ),
    ],
)
def test_a_fit_predicted_with_tf32_is_not_regenerated(numerics):
    report = dict(identity=dict(case={}, numerics=numerics), predictions={})
    with pytest.raises(ValueError, match="FP32 estricto"):
        regeneration.require_strict(report, "US/fold-000/x")


def test_strict_fits_and_tabular_reports_pass_the_numerics_check():
    strict = dict(float32_matmul_precision="highest", cuda_matmul_allow_tf32=False)
    regeneration.require_strict(
        dict(identity=dict(numerics=dict(strict, cudnn_allow_tf32=False))), "a"
    )
    regeneration.require_strict(
        dict(
            runs=[dict(matmul_precision="highest", matmul_allow_tf32=False, cudnn_allow_tf32=False)]
        ),
        "b",
    )
    regeneration.require_strict(dict(identity=dict(kind="ridge")), "c")
    assert regeneration.recorded_numerics(dict(identity=dict(kind="ridge"))) == []


native = pytest.mark.skipif(
    not os.environ.get("MARS_TITAN_EPISODIC_NATIVE"), reason="Falta el enlace nativo compilado"
)


@native
def test_the_episodic_gru_window_regenerates_its_rows(episodic, tmp_path, monkeypatch):  # noqa: F811
    from mars_titan.training import candidate_walk_forward as candidate

    monkeypatch.setenv(HOLD_ENV, str(episodic.hold))
    run = episodic.original
    produced = candidate.carry_window(
        run.output,
        run.view,
        run.view,
        tmp_path / "again",
        parent_id="US/fold-000/gru_episodic",
        device="cpu",
        regenerate=True,
    )
    assert produced["regenerated"] is True and produced["receipts"] == {}
    report = json.loads((run.output / "window.json").read_text())
    result = regeneration.compare(
        regeneration.originals(run.output, report), tmp_path / "again", produced
    )
    assert set(result["partitions"]) == set(HELD_OUT)
    assert result["identical"] is True, result


@native
def test_the_mars_titan_reader_regenerates_its_rows(readers, titans, tmp_path, monkeypatch):  # noqa: F811
    from mars_titan.training import mars_titan_walk_forward as mw

    monkeypatch.setenv(HOLD_ENV, str(titans.hold))
    run = readers.original
    with unfused_attention():
        produced = mw.carry_mars_titan(
            run.output, run.view, run.view, tmp_path / "again", device="cpu", regenerate=True
        )
    assert produced["regenerated"] is True
    report = json.loads((run.output / "run.json").read_text())
    result = regeneration.compare(
        regeneration.originals(run.output, report), tmp_path / "again", produced
    )
    assert set(result["partitions"]) == set(HELD_OUT)
    assert result["identical"] is True, result
