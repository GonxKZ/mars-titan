"""La medida de latencia solo consume selecciones y evaluaciones ya confirmadas."""

import copy
import json

import pytest

from mars_titan.data.storage import sha256


def api():
    from benchmarks import refinement_latency_study

    return refinement_latency_study


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def completed(tmp_path, monkeypatch):
    from benchmarks.refinement_study import jobs

    work = tmp_path / "study"
    job = next(row for row in jobs() if row["id"] == "linear_v1-42/evaluate-k1")
    prepared = dict(configuration=dict(cohort_size=64), jobs=[job], final_test_opened=False)
    put(work / "prepared.json", prepared)
    monkeypatch.setattr(api(), "verify_prepared", lambda _: copy.deepcopy(prepared))
    group = {"fixture": "both_readers_sealed"}
    monkeypatch.setattr(api(), "selection_group", lambda *_: group)
    selected = work / "runs" / job["parent"]
    policy = selected / "candidate" / "policy.json"
    put(policy, {"fixture": "fixed_policy"})
    seal = dict(selected=dict(id="candidate", policy_sha256=sha256(policy)))
    put(selected / "phase.json", seal)
    output = work / "runs" / job["id"]
    phases = {}
    for case in ("policy", "k1", "k4cc"):
        folder = (
            output / "evaluation" if case == "policy" else output / "controls" / case / "evaluation"
        )
        folder.mkdir(parents=True)
        artifact = folder / "predictions.parquet"
        artifact.write_bytes(b"Identidad de una fixture, sin filas financieras")
        record = dict(
            status="completed",
            identity={"origin": {"case": case}},
            artifacts={"predictions.parquet": sha256(artifact)},
            final_test_opened=False,
        )
        put(folder / "phase.json", record)
        phases[case] = record
    run = dict(
        status="completed",
        job=job,
        selection_group=group,
        selection_sha256=sha256(selected / "phase.json"),
        final_test_opened=False,
        phases={"evaluation": phases["policy"]},
        controls={case: {"evaluation": phases[case]} for case in ("k1", "k4cc")},
    )
    put(output / "run.json", run)
    return work, prepared, output, selected


@pytest.mark.parametrize("case", ["policy", "k1", "k4cc"])
def test_only_the_sealed_first_seed_and_reader_are_selected(completed, case):
    work, _, _, _ = completed
    result = api().load_case(work, case)
    assert result["job"]["seed"] == 42
    assert result["job"]["projection"] == "linear_v1"
    assert result["job"]["depth"] == 1
    assert result["phase"]["identity"]["origin"]["case"] == case


@pytest.mark.parametrize(
    "change", ["running", "final_test", "group", "selection", "predictions", "policy"]
)
def test_incomplete_or_changed_evidence_is_rejected(completed, change):
    work, _, output, selected = completed
    run_path = output / "run.json"
    run = json.loads(run_path.read_text())
    if change in ("running", "final_test", "group", "selection"):
        key, value = {
            "running": ("status", "running"),
            "final_test": ("final_test_opened", True),
            "group": ("selection_group", {}),
            "selection": ("selection_sha256", "0" * 64),
        }[change]
        run[key] = value
        put(run_path, run)
    elif change == "predictions":
        (output / "evaluation/predictions.parquet").write_bytes(b"changed")
    else:
        put(selected / "candidate/policy.json", {"changed": True})
    with pytest.raises(ValueError):
        api().load_case(work, "policy")


def test_the_catalog_cannot_choose_an_unregistered_case(completed):
    with pytest.raises(ValueError, match="caso"):
        api().load_case(completed[0], "best_seed")


@pytest.mark.parametrize("cohort", [4, 128, 512])
def test_other_cohorts_do_not_reuse_the_latency_protocol(completed, cohort):
    work, prepared, _, _ = completed
    prepared["configuration"]["cohort_size"] = cohort
    with pytest.raises(ValueError, match="cohorte"):
        api().load_case(work, "policy")


def test_learning_hold_precedes_sources_and_output(tmp_path, monkeypatch):
    from mars_titan.training.learning_hold import LearningHoldError

    hold = tmp_path / "hold.json"
    put(hold, {"training_allowed": False})
    monkeypatch.setenv("MARS_TITAN_TRAINING_HOLD", str(hold))
    monkeypatch.setattr(api(), "load_case", lambda *_: pytest.fail("No leer fuentes"))
    output = tmp_path / "output"
    with pytest.raises(LearningHoldError):
        api().run(tmp_path / "work", output, "policy", max_seconds=20)
    assert not output.exists()


def test_output_must_be_new_before_loading_any_data(tmp_path, monkeypatch):
    output = tmp_path / "already-present"
    output.mkdir()
    monkeypatch.setattr(api(), "require_learning_allowed", lambda _: None)
    monkeypatch.setattr(api(), "load_case", lambda *_: pytest.fail("No leer fuentes"))
    with pytest.raises(FileExistsError):
        api().run(tmp_path / "work", output, "policy", max_seconds=20)


@pytest.mark.parametrize("seconds", [0, -1, True, float("nan"), 1201])
def test_invalid_latency_budgets_fail_before_creating_output(tmp_path, monkeypatch, seconds):
    monkeypatch.setattr(api(), "require_learning_allowed", lambda _: None)
    with pytest.raises(ValueError, match="presupuesto"):
        api().run(tmp_path / "work", tmp_path / "output", "policy", max_seconds=seconds)
    assert not (tmp_path / "output").exists()


def test_busy_gpu_closes_the_failure_receipt_without_loading_models(
    completed, monkeypatch, tmp_path
):
    import os

    import torch

    from mars_titan.training import experiment_resources
    from mars_titan.training.gpu_supervisor import GpuSnapshot

    monkeypatch.setattr(api(), "require_learning_allowed", lambda _: None)
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path / "isolated-lock"))
    monkeypatch.setattr(
        experiment_resources, "read_gpu", lambda: GpuSnapshot(8192, 7000, (os.getpid() + 1,))
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *_: pytest.fail("No inicializar CUDA"))
    monkeypatch.setattr(api(), "_measure", lambda *_: pytest.fail("No cargar modelos"))
    output = tmp_path / "rejected"
    with pytest.raises(RuntimeError, match="otra carga"):
        api().run(completed[0], output, "policy", max_seconds=20)
    result = json.loads((output / "run.json").read_text())
    assert result["status"] == "failed" and result["optimizer_steps"] == 0
    assert result["final_test_opened"] is False
    assert "finished_at_utc" in result and result["process_seconds"] > 0


def test_deadline_counts_time_and_manual_interruption(monkeypatch):
    now = [10.0]
    monkeypatch.setattr(api().time, "monotonic", lambda: now[0])
    limit = api().Deadline(2)
    assert limit.requested is False
    now[0] = 12.0
    assert limit.requested is True
    limit = api().Deadline(2)
    assert limit.requested is False
    limit.request()
    assert limit.requested is True


@pytest.mark.parametrize(
    "case,change",
    [
        ("k1", None),
        ("k4cc", None),
        ("policy", None),
        ("k1", "origin"),
        ("k1", "prediction"),
        ("k1", "saved_prediction"),
    ],
)
def test_measurement_recreates_each_pass_and_checks_original_predictions(
    tmp_path, monkeypatch, case, change
):
    import os

    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch

    from benchmarks.refinement_latency import measure_pass
    from benchmarks.refinement_study_policy import save_policy
    from mars_titan.memory.native_backend import load_native
    from mars_titan.models.titans.refinement_policy import FEATURES, RoutePolicy
    from mars_titan.training.refinement_inference import RefinementInference
    from mars_titan.training.titans_walk_forward import checked_tables, unfused_attention
    from tests.training.test_financial_run import corpus
    from tests.training.test_refinement_inference import build

    dataset, streams = corpus(tmp_path / "data")
    # Esta fixture técnica utiliza su validación. La CLI solo admite la evaluación sellada.
    source = streams["validation"]
    native = load_native(os.environ["MARS_TITAN_EPISODIC_NATIVE"])

    def controller():
        count = len(FEATURES)
        return RoutePolicy(
            mean=torch.zeros(count, dtype=torch.float64),
            scale=torch.ones(count, dtype=torch.float64),
            coefficients=torch.zeros(count + 1, 5, dtype=torch.float64),
            support_min=torch.full((count,), -100.0, dtype=torch.float64),
            support_max=torch.full((count,), 100.0, dtype=torch.float64),
            costs=torch.zeros(5, dtype=torch.float64),
            min_gain=0.0,
        )

    def construct(policy=None):
        engine = build(streams, native, mode="fixed", route="k4cc" if case == "k4cc" else "k1")
        if policy is None:
            return engine
        return RefinementInference(
            engine.predictor,
            engine.readout,
            engine.recipe,
            admission=engine.admission,
            retention=engine.retention,
            native=native,
            codec=engine.codec,
            world=engine.world,
            fold=engine.fold,
            mode="policy",
            policy=policy,
        )

    with unfused_attention():
        reference = construct(controller() if case == "policy" else None)
        measured, rows = measure_pass(reference, source, instrument=False)
        tables = checked_tables(rows, measured["metrics"], dataset, source.phase.partition)
        predictions = tmp_path / "predictions.parquet"
        table = pa.concat_tables(tables)
        if change == "saved_prediction":
            # Reflejar el error conserva MAE y MSE, pero cambia la predicción emitida.
            column = table.schema.get_field_index("prediction")
            reflected = pa.array(
                2 * table["target"].to_numpy() - table["prediction"].to_numpy(),
                type=table["prediction"].type,
            )
            table = table.set_column(column, table.schema.field(column), reflected)
        pq.write_table(table, predictions)
        origin = {"fixture": "frozen_cpu_reader"}
        request = dict(
            prediction_path=predictions,
            job={"reader": "fixture"},
            phase={"identity": {"origin": origin}, "metrics": measured["metrics"]},
            policy_path=tmp_path / "policy.json",
        )
        save_policy(request["policy_path"], controller())
        monkeypatch.setattr(api(), "_source", lambda *_: (dataset, source))
        engines = []

        def reconstruct(*args, **kwargs):
            assert kwargs["device"] == "cpu"
            assert kwargs["mode"] == ("policy" if case == "policy" else "fixed")
            assert kwargs["route"] == (None if case == "policy" else case)
            engine = construct(kwargs["policy"])
            if change == "prediction":
                with torch.no_grad():
                    engine.predictor.head.bias.add_(0.5)
            engines.append(engine)
            return engine, {"changed": True} if change == "origin" else origin

        monkeypatch.setattr(api(), "reconstruct", reconstruct)
        output = tmp_path / "measurement"
        output.mkdir()
        if change is not None:
            with pytest.raises(ValueError):
                api()._measure(tmp_path, output, case, request, api().Deadline(60), device="cpu")
            assert not (output / "pass-0.json").exists()
            return
        result = api()._measure(tmp_path, output, case, request, api().Deadline(60), device="cpu")
    assert len(result["passes"]) == len({id(engine) for engine in engines}) == 4
    assert result["exact_original_predictions"] is True
    assert len(result["instrumentation_phase_time_ratios"]) == 3
    for index in range(4):
        record = json.loads((output / f"pass-{index}.json").read_text())
        assert record["exact_original_metrics"] is True
        assert record["instrumented"] is (index > 0)
