"""Etapa de ablación de modalidades sobre una campaña base reducida, en CPU y sin ajustes.

La campaña base recorta el protocolo US a dos ventanas con un brazo GRU de una semilla.
Su ejecutor sustituto guarda como estado elegido los pesos iniciales de la semilla, con la
identidad que exige el traslado de referencias, y escribe sus predicciones con la misma
evaluación que el traslado. La etapa usa después los ejecutores reales de la ablación en
CPU. No se aplica ningún paso de optimizador: la prueba del gancho usa un optimizador que
solo cuenta llamadas y no modifica pesos.
"""

import importlib
import json
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow.parquet as pq
import pytest
import torch

from mars_titan.data.input_policy import HISTORICAL_MASKED, MODALITIES, policy_identity
from mars_titan.data.modality_ablation import VARIANTS, ablation_identity
from mars_titan.data.storage import atomic_json, sha256
from mars_titan.evaluation import modality_ablation as analysis
from mars_titan.evaluation import walk_forward_comparison as comparison
from mars_titan.models.baselines.multimodal import PRESENCE_FUSION, MultimodalReference
from mars_titan.models.quantile_head import QUANTILE_COLUMNS, QUANTILE_HEAD
from mars_titan.training import campaign_plan as plan
from mars_titan.training import masked_campaign as engine
from mars_titan.training import modality_ablation_stage as ablation
from mars_titan.training.checkpoints import capture_rng, save_training_state
from mars_titan.training.corpus_inputs import CorpusDataset
from mars_titan.training.learning_hold import HOLD_ENV, LearningHoldError
from mars_titan.training.reference_run import _evaluate, scientific_identity
from tests.posttraining.campaign_fixture import SCORES, CpuLease, write_configs
from tests.training.historical_temporal_fixture import historical_temporal_fixture
from tests.training.test_masked_campaign import Recorder
from tests.training.test_modality_ablation_inputs import pattern
from tests.training.test_walk_forward_v2_views import sessions

STAGES = {
    variant: Path(path)
    for variant, path in plan.LATER_STAGES["modality_ablation"]["stages"].items()
}
RUNNING = SimpleNamespace(requested=False)


def on_cpu(patch):
    """Sustituir CUDA por CPU en el ancla y en el traslado, sin optimizadores."""
    import mars_titan.data.embeddings as embeddings

    patch.setattr(embeddings, "require_cuda", lambda **_: torch.device("cpu"))
    for name, value in dict(
        synchronize=None,
        reset_peak_memory_stats=None,
        max_memory_allocated=0,
        get_device_name="cpu-technical-fixture",
    ).items():
        patch.setattr(torch.cuda, name, lambda *_, value=value: value)
    patch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    patch.delenv("MARS_TITAN_INPUT_CACHE_MIB", raising=False)


def anchored_reference(run):
    """Estado elegido con los pesos iniciales y la identidad que comprueba el traslado."""
    case = run.case
    dataset = CorpusDataset(run.view, input_policy=HISTORICAL_MASKED)
    first = next(dataset.batches(partition="train", batch_size=1, epoch=0, seed=0))
    dimensions = {name: value.shape[-1] for name, value in first["inputs"].items()}
    torch.manual_seed(case["seed"])
    model = MultimodalReference(
        case["kind"],
        dimensions,
        context=dataset.context,
        mask_fusion=PRESENCE_FUSION,
        head=QUANTILE_HEAD,
        **case["architecture"],
    ).eval()
    identity = dict(
        **scientific_identity(
            kind=case["kind"], input_policy=HISTORICAL_MASKED, head=QUANTILE_HEAD
        ),
        manifest_sha256=sha256(run.view),
        case=case,
        dimensions=dimensions,
        context=dataset.context,
        **policy_identity(HISTORICAL_MASKED),
        mask_fusion=PRESENCE_FUSION,
    )
    score = SCORES[run.job["candidate"]] if run.job["stage"] == "search" else 0.015
    selection = dict(
        last_epoch=0,
        best_epoch=0,
        best_score=score,
        stale_epochs=0,
        should_stop=False,
        last_improved=True,
    )
    run.folder.mkdir(parents=True, exist_ok=True)
    state = dict(
        global_step=0,
        epoch=0,
        confirmed_cursor=None,
        model=model.state_dict(),
        optimizer={},
        rng=capture_rng("cpu"),
        statistics=dict(samples=0, squared_error=0.0, absolute_error=0.0, elapsed_seconds=0.0),
        history=[],
        selection=selection,
        initial_validation=None,
    )
    checkpoint = save_training_state(
        run.folder / "checkpoints", state, identity=identity, best=True
    )
    predictions = {}
    for partition in ("validation", "calibration", "evaluation"):
        path = run.folder / f"{partition}-predictions.parquet"
        metrics = _evaluate(
            model,
            dataset,
            run.batch_size,
            device="cpu",
            partition=partition,
            destination=path,
            quantiles=True,
        )
        predictions[partition] = dict(path=path.name, sha256=sha256(path), metrics=metrics)
    predictions["validation"]["metrics"] = dict(session_mae=score)
    report = dict(
        schema_version=1,
        status="completed",
        identity=identity,
        scope=dataset.manifest["scope"],
        samples=dataset.manifest["counts"],
        predictions=predictions,
        checkpoint=dict(path=str(checkpoint.relative_to(run.folder)), sha256=sha256(checkpoint)),
        selection=selection,
        final_test_opened=False,
    )
    atomic_json(run.folder / "run.json", report)
    return report


def campaign_executors():
    carry = Recorder()
    return {
        key: dict(entry, run=anchored_reference if key == ("neural", engine.FIT) else carry)
        for key, entry in engine.EXECUTORS.items()
    }


def write_stage(folder, **changes):
    stage = json.loads(STAGES["A"].read_text())
    stage.update(
        name="fixture-ablation-a",
        campaign="campaign.json",
        scopes=["US"],
        limits=dict(max_prediction_jobs=6),
    )
    stage.update(changes)
    path = folder / f"ablation-{len(list(folder.glob('ablation-*')))}.json"
    atomic_json(path, stage)
    return path


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    """Campaña A confirmada con dos ventanas y la etapa de ablación sin ejecutar."""
    root = tmp_path_factory.mktemp("ablation-stage")
    hold = root / "hold.json"
    hold.write_text(json.dumps({"training_allowed": True}), encoding="utf-8")
    with pytest.MonkeyPatch.context() as patch:
        on_cpu(patch)
        # La campaña base solo ejecuta dobles con pesos iniciales: ningún modelo se ajusta.
        patch.setenv(HOLD_ENV, str(hold))
        campaign, _ = write_configs(root / "config", "A")
        path = root / "config" / "comparison.json"
        declared = json.loads(path.read_text())
        declared["modality_ablation"].update(min_rows=1, min_sessions=1)
        atomic_json(path, declared)
        data = historical_temporal_fixture(
            root / "data", ("US",), {"US": sessions("US")}, assets=2, presence=pattern
        )
        engine.prepare_views(campaign, data.parent, root / "views")
        views = {"US": root / "views" / "US"}
        summary = engine.run_campaign(
            campaign,
            views,
            root / "campaign",
            executors=campaign_executors(),
            lease=CpuLease,
            stop=RUNNING,
        )
        assert summary["status"] == "completed"
        yield SimpleNamespace(
            root=root,
            campaign=campaign,
            comparison=path,
            stage=write_stage(root / "config"),
            views=views,
            output=root / "campaign",
            hold=hold,
        )


@pytest.fixture(scope="module")
def staged(base):
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv(HOLD_ENV, str(base.hold))
        summary = ablation.run_stage(
            base.stage,
            base.views,
            base.output,
            base.root / "ablation",
            lease=CpuLease,
            stop=RUNNING,
        )
    return SimpleNamespace(summary=summary, output=base.root / "ablation")


@pytest.fixture(scope="module")
def confirmed(base):
    """Recibos confirmados de la campaña base, leídos sin ejecutar ningún trabajo."""
    _, state = engine._confirmed_state(base.campaign, base.views, base.output)
    for job in plan.plan_campaign(plan.load_campaign(base.campaign)):
        case, _, sources = state.resolve(job)
        state.receipts[job["id"]] = state.confirmed(job, state.job_identity(job, case, sources))
    return state


def view_of(confirmed, window):
    return Path(confirmed.views["US"]["windows"][window]["path"])


def presence_of(view):
    dataset = CorpusDataset(view, input_policy=HISTORICAL_MASKED)
    ids, bits = [], []
    for batch in dataset.batches(partition="evaluation", batch_size=7, epoch=0, seed=0):
        ids += list(batch["sample_ids"])
        bits.append(batch["presence"])
    return dict(zip(ids, np.concatenate(bits), strict=True))


@pytest.mark.parametrize("variant", ["A", "B"])
def test_declared_stages_count_every_window_arm_seed_and_variant(variant):
    checked = ablation.check_stage(STAGES[variant])
    counts = checked["counts"]
    assert checked["status"] == "checked" and checked["variant"] == variant
    assert counts["training_jobs"] == 0 and counts["prediction_jobs"] == 4185
    assert counts["variants"] == dict.fromkeys(VARIANTS, 1395)
    assert {scope: value["windows"] for scope, value in counts["scopes"].items()} == {
        "US": 19,
        "CN": 13,
        "US+CN": 13,
    }
    seeds = sum(len(value) for value in checked["arms"].values())
    assert seeds == 31 and "zero" not in checked["arms"]
    assert (
        checked["declaration"]
        == comparison.load_config(
            Path("configs/evaluation/historical-masked-2000-comparison.json")
        )["modality_ablation"]
    )
    assert checked["optimizer_steps"] == 0 and checked["final_test_opened"] is False


def test_later_stage_entry_names_the_comparison_and_both_campaigns():
    declared = plan.LATER_STAGES["modality_ablation"]
    assert declared["issue"] == 414 and declared["pending"] == []
    assert Path(declared["config"]).is_file() and set(declared["stages"]) == set(plan.VARIANTS)
    module, _, function = declared["entry"].partition(":")
    assert getattr(importlib.import_module(module), function) is ablation.run_stage
    for variant, path in declared["stages"].items():
        stage = ablation.load_stage(path)
        assert stage["campaign"]["variant"] == variant
        assert stage["campaign"]["comparison_path"] == str(Path(declared["config"]).resolve())


@pytest.mark.parametrize(
    "changes,message",
    [
        (dict(limits=dict(max_prediction_jobs=5)), "supera el límite"),
        (dict(final_test_opened=True), "contrato"),
        (dict(status="running"), "contrato"),
        (dict(scopes=["CN"]), "ámbitos"),
        (dict(scopes=[]), "ámbitos"),
        (dict(limits=dict(max_prediction_jobs=6, max_training_jobs=0)), "límite"),
    ],
)
def test_stage_declaration_is_rejected_before_reading_data(base, changes, message):
    with pytest.raises(ValueError, match=message):
        ablation.check_stage(write_stage(base.root / "config", **changes))


def test_stage_is_held_before_creating_any_output(base, tmp_path, learning_hold):
    learning_hold(False)
    output = tmp_path / "held"
    with pytest.raises(LearningHoldError, match="evaluación científica"):
        ablation.run_stage(
            base.stage, base.views, base.output, output, lease=CpuLease, stop=RUNNING
        )
    assert not output.exists()


def test_each_job_predicts_the_evaluation_from_the_selected_state(base, staged, confirmed):
    summary = staged.summary
    assert summary["status"] == "completed" and summary["optimizer_steps"] == 0
    assert summary["planned"] == dict(training_jobs=0, prediction_jobs=6)
    assert summary["completed"] == dict(training_jobs=0, prediction_jobs=6)
    assert summary["variants"] == dict.fromkeys(VARIANTS, 2)
    for job in ablation.plan_stage(ablation.load_stage(base.stage)):
        receipt = json.loads((staged.output / "jobs" / job["id"] / "receipt.json").read_text())
        key, selected = confirmed.selected("US", job["window"], "gru", 42)
        identity = receipt["identity"]
        assert identity["modality_ablation"] == ablation_identity(job["variant"])
        assert identity["source"]["job"] == key and identity["parent"] == selected["parent"]
        assert receipt["optimizer_steps"] == 0 and receipt["final_test_opened"] is False
        record = selected["predictions"]["evaluation"]
        assert receipt["prediction"]["rows"] == record["rows"]
        assert receipt["prediction"]["rows_sha256"] == record["rows_sha256"]
        carry = json.loads((staged.output / receipt["report"]["path"]).read_text())
        assert set(carry["predictions"]) == {"evaluation"}
        assert carry["modality_ablation"] == ablation_identity(job["variant"])


def test_memoryless_reference_keeps_every_row_without_the_masked_modality(base, staged, confirmed):
    for job in ablation.plan_stage(ablation.load_stage(base.stage)):
        receipt = json.loads((staged.output / "jobs" / job["id"] / "receipt.json").read_text())
        _, selected = confirmed.selected("US", job["window"], "gru", 42)
        original = pq.read_table(base.output / selected["predictions"]["evaluation"]["path"])
        masked = pq.read_table(staged.output / receipt["prediction"]["path"])
        assert masked["sample_id"].equals(original["sample_id"])
        presence = presence_of(view_of(confirmed, job["window"]))
        columns = [MODALITIES.index(name) for name in VARIANTS[job["variant"]]]
        touched = np.array(
            [presence[key][columns].any() for key in original["sample_id"].to_pylist()]
        )
        assert touched.any() and (~touched).any()
        for name in ("prediction", *QUANTILE_COLUMNS):
            before, after = original[name].to_numpy(), masked[name].to_numpy()
            np.testing.assert_array_equal(after[~touched], before[~touched])
        assert (original["prediction"].to_numpy() != masked["prediction"].to_numpy())[touched].any()


def test_confirmed_jobs_are_reused_without_running_any_executor(base, staged):
    def refuse(run):
        raise AssertionError(f"Volvió a ejecutar {run.job['id']}")

    with pytest.MonkeyPatch.context() as patch:
        patch.setenv(HOLD_ENV, str(base.hold))
        again = ablation.run_stage(
            base.stage,
            base.views,
            base.output,
            staged.output,
            executors=dict.fromkeys(ablation.EXECUTORS, refuse),
            lease=CpuLease,
            stop=RUNNING,
        )
    assert again["status"] == "completed" and again["completed"]["prediction_jobs"] == 6


def test_sources_feed_the_comparison_with_the_original_calibrators(base, staged, confirmed):
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv(HOLD_ENV, str(base.hold))
        masked = ablation.write_sources(base.stage, base.views, base.output, staged.output, "US")
        primary = engine.write_sources(base.campaign, base.views, base.output, "US")
    manifest = json.loads(masked.read_text())
    assert manifest["kind"] == analysis.SOURCES_KIND and set(manifest["variants"]) == set(VARIANTS)
    assert set(manifest["variants"]["mask_news"]) == {"gru"}
    report, _ = comparison.evaluate_walk_forward(
        base.comparison, primary, "US", ablation_sources=masked
    )
    section = report["modality_ablation"]
    assert section["status"] == "computed" and section["recalibrated"] is False
    windows = [window for window in report["windows"]]
    for variant in VARIANTS:
        entry = section["variants"][variant]
        seeds = entry["arms"]["gru"]["42"]
        # Un modelo sin memoria no cambia ninguna fila sin la modalidad.
        assert seeds["unaffected_changed"] == dict.fromkeys(windows, 0)
        expected = 0
        for window in windows:
            presence = presence_of(view_of(confirmed, window))
            columns = [MODALITIES.index(name) for name in VARIANTS[variant]]
            count = sum(bool(bits[columns].any()) for bits in presence.values())
            assert entry["population"]["windows"][window]["rows"] == count
            expected += count
        assert entry["population"]["overall"]["US"]["rows"] == expected
        summary = seeds["overall"]["US"]
        assert summary["difference"] == pytest.approx(
            summary["masked_session_mae"] - summary["original_session_mae"], abs=1e-15
        )
        # Sin el mínimo de filas del calibrador común tampoco hay cobertura enmascarada.
        assert summary["calibrated_intervals"] is None
        assert summary["calibrated_reason"] == "Alguna ventana no tiene calibrador"


def test_one_window_sources_give_the_same_report_through_window_aggregates(base, staged, tmp_path):
    """Cada ventana se puntúa en cuanto termina y el informe final no abre ninguna fila."""
    from mars_titan.evaluation import window_aggregates

    volatile = {"created_at_utc", "resources", "sources_sha256"}
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv(HOLD_ENV, str(base.hold))
        config = comparison.load_config(base.comparison)
        windows = list(config["resolved_scopes"]["US"]["windows"])
        for window in windows:
            primary = engine.write_sources(
                base.campaign, base.views, base.output, "US", window=window
            )
            masked = ablation.write_sources(
                base.stage, base.views, base.output, staged.output, "US", window=window
            )
            assert primary.parent.name == window == masked.parent.name
            restricted = comparison.restrict_windows(config, "US", [window])
            sources = comparison.load_sources(primary, restricted, "US")
            assert list(sources["windows"]) == [window]
            ablated = comparison._ablation_sources(masked, restricted, sources)
            window_aggregates.write(tmp_path / "aggregates", restricted, sources, window, ablated)
        masked = ablation.write_sources(base.stage, base.views, base.output, staged.output, "US")
        primary = engine.write_sources(base.campaign, base.views, base.output, "US")
    expected, sessions = comparison.evaluate_walk_forward(
        base.comparison, primary, "US", ablation_sources=masked
    )
    report, from_aggregates = comparison.evaluate_walk_forward(
        base.comparison,
        primary,
        "US",
        ablation_sources=masked,
        aggregates=tmp_path / "aggregates",
    )
    assert {k: v for k, v in report.items() if k not in volatile} == {
        k: v for k, v in expected.items() if k not in volatile
    }
    assert from_aggregates.equals(sessions)


class Counting(torch.optim.Optimizer):
    """Optimizador que solo cuenta llamadas: nunca modifica los pesos."""

    def __init__(self, params):
        super().__init__(params, dict(lr=0.0))
        self.calls = 0

    def step(self, closure=None):
        self.calls += 1


@pytest.fixture
def only_stage_hooks(monkeypatch):
    """Retirar el gancho global de la sesión para observar solo el de la etapa."""
    optimizer = importlib.import_module("torch.optim.optimizer")
    monkeypatch.setattr(optimizer, "_global_optimizer_pre_hooks", OrderedDict())


def test_optimizer_steps_are_rejected_before_touching_weights(only_stage_hooks):
    weight = torch.nn.Parameter(torch.ones(3))
    weight.grad = torch.ones(3)
    optimizer = Counting([weight])
    with ablation.forbid_optimizer_steps():
        with pytest.raises(RuntimeError, match="no admite pasos de Counting"):
            optimizer.step()
    assert optimizer.calls == 0 and torch.equal(weight.detach(), torch.ones(3))
    # Fuera de la etapa el gancho desaparece y el optimizador falso vuelve a contar.
    optimizer.step()
    assert optimizer.calls == 1 and torch.equal(weight.detach(), torch.ones(3))


def test_a_job_that_tries_an_optimizer_step_fails_the_stage(base, tmp_path, only_stage_hooks):
    weight = torch.nn.Parameter(torch.ones(2))
    optimizer = Counting([weight])

    def stepping(run):
        optimizer.step()

    output = tmp_path / "stepping"
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv(HOLD_ENV, str(base.hold))
        with pytest.raises(RuntimeError, match="no admite pasos"):
            ablation.run_stage(
                base.stage,
                base.views,
                base.output,
                output,
                executors=dict.fromkeys(ablation.EXECUTORS, stepping),
                lease=CpuLease,
                stop=RUNNING,
            )
    summary = json.loads((output / "summary.json").read_text())
    assert summary["status"] == "failed" and summary["completed"]["prediction_jobs"] == 0
    assert optimizer.calls == 0 and torch.equal(weight.detach(), torch.ones(2))


def test_the_hold_is_checked_again_before_each_pending_job(base, tmp_path, learning_hold):
    hold = learning_hold(True)

    def then_block(run):
        report = ablation.EXECUTORS["neural"](run)
        # La protección vuelve a regir tras el primer trabajo confirmado.
        hold.write_text(json.dumps({"training_allowed": False}), encoding="utf-8")
        return report

    output = tmp_path / "blocked"
    with pytest.MonkeyPatch.context():
        executors = dict(ablation.EXECUTORS, neural=then_block)
        with pytest.raises(LearningHoldError):
            ablation.run_stage(
                base.stage,
                base.views,
                base.output,
                output,
                executors=executors,
                lease=CpuLease,
                stop=RUNNING,
            )
    summary = json.loads((output / "summary.json").read_text())
    assert summary["status"] == "blocked" and summary["completed"]["prediction_jobs"] == 1


def _without_ablation(report, folder):
    report.pop("modality_ablation")


def _other_anchor(report, folder):
    report["anchor"]["checkpoint_sha256"] = "0" * 64


def _with_calibration(report, folder):
    report["predictions"]["calibration"] = report["predictions"]["evaluation"]


def _opened(report, folder):
    report["final_test_opened"] = True


def _other_targets(report, folder):
    """Mismas filas y huella declarada coherente, pero otro objetivo en una fila."""
    record = report["predictions"]["evaluation"]
    path = folder / record["path"]
    table = pq.read_table(path)
    target = table["target"].to_numpy().copy()
    target[0] += 1.0
    table = table.set_column(table.schema.get_field_index("target"), "target", [target])
    pq.write_table(table, path)
    record["sha256"] = sha256(path)


@pytest.mark.parametrize(
    "edit,message",
    [
        (_without_ablation, "no declara la ablación pedida"),
        (_other_anchor, "no parte del estado elegido"),
        (_with_calibration, "solo debe predecir la evaluación"),
        (_opened, "abre la reserva final"),
        (_other_targets, "no evalúa las mismas filas ni objetivos"),
    ],
)
def test_a_report_that_does_not_match_its_job_is_not_confirmed(base, tmp_path, edit, message):
    def edited(run):
        report = ablation.EXECUTORS["neural"](run)
        edit(report, run.folder)
        return report

    output = tmp_path / "edited"
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv(HOLD_ENV, str(base.hold))
        with pytest.raises(ValueError, match=message):
            ablation.run_stage(
                base.stage,
                base.views,
                base.output,
                output,
                executors=dict(ablation.EXECUTORS, neural=edited),
                lease=CpuLease,
                stop=RUNNING,
            )
    assert not list((output / "jobs").glob("*/*/*/*/receipt.json"))


@pytest.mark.parametrize("variant", ["A", "B"])
def test_throughput_estimates_each_masked_prediction_with_the_measured_inference(variant):
    from mars_titan.training import campaign_throughput as throughput
    from tests.training.test_campaign_throughput import CAMPAIGNS, ROWS, chronological, uniform

    campaign = plan.load_campaign(CAMPAIGNS[variant])
    stage = ablation.load_stage(STAGES[variant])
    jobs = ablation.plan_stage(stage)
    counts, rates = uniform(campaign, ROWS)
    estimate = throughput.estimate_hours(campaign, counts, rates, ablation_stage=stage)
    masked = estimate[throughput.ABLATION_STAGE]
    neural = [job for job in jobs if job["family"] == plan.NEURAL]
    # Solo la evaluación, a 400 filas/s con la inferencia más lenta del brazo.
    assert (masked["training_jobs"], masked["prediction_jobs"]) == (0, len(neural))
    assert masked["hours"] == pytest.approx(len(neural) * ROWS["evaluation"] / 400 / 3600)
    assert masked["without_estimate"] == sorted(
        {job["arm"] for job in jobs if job["family"] != plan.NEURAL}
    )
    rates[plan.TITANS] = chronological(campaign[plan.TITANS]["arms"])
    masked = throughput.estimate_hours(campaign, counts, rates, ablation_stage=stage)[
        throughput.ABLATION_STAGE
    ]
    resolved = campaign["comparison_config"]["resolved_scopes"]
    titans = sum(
        throughput.titans_rows(ROWS, resolved[job["scope"]]["windows"][job["window"]], 12)[
            "evaluation"
        ]
        / 400
        for job in jobs
        if job["family"] == plan.TITANS
    )
    assert masked["hours"] == pytest.approx(
        (len(neural) * ROWS["evaluation"] / 400 + titans) / 3600
    )
    assert masked["without_estimate"] == ["ridge", "xgboost"]
    # La etapa de ablación no se suma a las horas de la campaña.
    assert throughput.ABLATION_STAGE not in estimate["families"]
    other = ablation.load_stage(STAGES["B" if variant == "A" else "A"])
    with pytest.raises(ValueError, match="no parte de esta campaña"):
        throughput.estimate_hours(campaign, counts, rates, ablation_stage=other)
