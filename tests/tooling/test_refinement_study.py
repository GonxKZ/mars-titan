"""Contratos del ensayo por etapas, sin ajuste ni acceso al test final."""

import copy
import importlib
import json

import pytest


def api():
    try:
        return importlib.import_module("benchmarks.refinement_study")
    except ModuleNotFoundError:
        pytest.fail("Falta el ejecutor del estudio por etapas")


def source(fold="fold-017"):
    year = 2022 if fold == "fold-017" else 2023
    assets = [
        dict(
            market="US",
            symbol=f"A{i}",
            samples=100,
            samples_sha256=str(i) * 64,
            counts=dict(train=50 if i else 0, validation=10, calibration=10, evaluation=10),
            purged_by_boundary=dict(train=1, validation=1, calibration=1, evaluation=1),
        )
        for i in range(8)
    ]
    return dict(
        final_test_opened=False,
        input_policy="historical_masked_2000_v1",
        temporal_view=dict(
            protocol=dict(market="US", selection=dict(max_epochs=30)),
            fold=dict(
                id=fold,
                train=["2000-01-01", f"{year - 1}-04-01"],
                validation=[f"{year - 1}-04-01", f"{year - 1}-10-01"],
                calibration=[f"{year - 1}-10-01", f"{year}-01-01"],
                evaluation=[f"{year}-01-01", f"{year + 1}-01-01"],
            ),
        ),
        assets=assets,
        counts=dict(train=350, validation=80, calibration=80, evaluation=80),
        coverage=[dict(market="US", symbol=a["symbol"]) for a in assets],
    )


def test_parent_history_selects_assets_independently_of_order_and_future_counts():
    before = source()
    selected = api().choose_assets(before, 4)
    assert len(selected) == len(set(selected)) == 4
    assert "A0" not in selected
    changed = copy.deepcopy(before)
    changed["assets"].reverse()
    for i, item in enumerate(changed["assets"]):
        item["counts"]["evaluation"] = 9000 * i
        item["future_target"] = -999.0 * i
    assert api().choose_assets(changed, 4) == selected
    assert before == source()


@pytest.mark.parametrize("count", [0, 8, 65, True])
def test_invalid_or_unavailable_cohort_size_is_rejected(count):
    with pytest.raises(ValueError, match="activos"):
        api().choose_assets(source(), count)


def test_duplicates_and_wrong_parent_window_fail():
    document = source()
    document["assets"].append(copy.deepcopy(document["assets"][0]))
    with pytest.raises(ValueError, match="duplic"):
        api().choose_assets(document, 4)
    with pytest.raises(ValueError, match="ventana"):
        api().choose_assets(source("fold-018"), 4)


def test_projection_preserves_rows_and_keeps_the_four_original_partitions():
    document = source("fold-018")
    selected = ("A1", "A3")
    result = api().project_view(document, selected, "fold-018", 2)
    assert [a["symbol"] for a in result["assets"]] == list(selected)
    assert result["assets"] == [document["assets"][1], document["assets"][3]]
    assert result["counts"] == dict(train=100, validation=20, calibration=20, evaluation=20)
    assert result["samples"] == 200
    assert result["purged_by_boundary"] == dict.fromkeys(document["counts"], 2)
    assert result["temporal_view"]["protocol"]["selection"]["max_epochs"] == 2
    assert result["training_ready"] is False and result["cohort_complete"] is False
    assert document == source("fold-018")


def test_projection_rejects_missing_assets_instead_of_changing_population():
    with pytest.raises(ValueError, match="activos"):
        api().project_view(source("fold-018"), ("A1", "MISSING"), "fold-018", 2)
    with pytest.raises(ValueError, match="ventana"):
        api().project_view(source("fold-018"), ("A1",), "fold-019", 2)


def test_catalog_preserves_parent_seed_and_does_not_train_a_gate_on_parent_rows():
    jobs = api().jobs()
    found = {}
    for job in jobs:
        assert job["id"] not in found
        if job.get("parent"):
            parent = found[job["parent"]]
            assert (parent["projection"], parent["seed"]) == (job["projection"], job["seed"])
        found[job["id"]] = job
    cores = [j for j in jobs if j["kind"] == "core"]
    assert {(j["projection"], j["seed"]) for j in cores} == {
        (p, s) for p in ("linear_v1", "mac_paper") for s in (42, 43, 44)
    }
    assert {j["depth"] for j in jobs if j["kind"] == "reader"} == {1, 4}
    assert all(j["fold"] == "fold-017" for j in jobs if j["kind"] in ("core", "reader"))
    assert all(j["fold"] == "fold-018" for j in jobs if j["kind"] == "collect")


def test_reader_components_use_valid_default_identity_for_k1():
    from mars_titan.training.mars_titan_walk_forward import _components

    one = traces_api().reader_components(1)
    four = traces_api().reader_components(4)
    assert _components(one) == {"episodic_bank": "m1"}
    assert _components(four) == {
        "episodic_bank": "m1",
        "refinements": 4,
        "refinement_episodes": "first_read",
    }


def test_learning_hold_is_checked_before_reading_study_or_loading_torch(tmp_path, monkeypatch):
    from mars_titan.training.learning_hold import LearningHoldError

    hold = tmp_path / "hold.json"
    hold.write_text(json.dumps(dict(training_allowed=False)))
    monkeypatch.setenv("MARS_TITAN_TRAINING_HOLD", str(hold))
    with pytest.raises(LearningHoldError):
        api().run_job(tmp_path / "absent", "absent")


def test_prepared_study_rejects_software_and_input_drift(tmp_path):
    from mars_titan.data.storage import atomic_json, sha256

    artifact, source = tmp_path / "artifact.json", tmp_path / "source.json"
    atomic_json(artifact, dict(value=1))
    atomic_json(source, dict(value=2))
    prepared = dict(
        configuration=api().configuration(),
        configuration_sha256=sha256(api().CONFIG),
        jobs=api().jobs(),
        code=api().code_identity(),
        environment=api().environment_identity(),
        artifacts={artifact.name: sha256(artifact)},
        source_manifests={"fixture": dict(path=str(source), sha256=sha256(source))},
    )
    path = tmp_path / "prepared.json"
    atomic_json(path, prepared)
    assert api().verify_prepared(tmp_path) == prepared
    changed = copy.deepcopy(prepared)
    changed["environment"]["packages"]["torch"] = "different-fixture-version"
    atomic_json(path, changed)
    with pytest.raises(ValueError):
        api().verify_prepared(tmp_path)
    atomic_json(path, prepared)
    atomic_json(source, dict(value=3))
    with pytest.raises(ValueError, match="origen"):
        api().verify_prepared(tmp_path)


def traces_api():
    try:
        return importlib.import_module("benchmarks.refinement_study_traces")
    except ModuleNotFoundError:
        pytest.fail("Falta el recorrido recuperable de las trazas")


def trace_record(number=1):
    return dict(
        flow_id=f"US/A{number}",
        decision_at=100,
        input_available_at=99,
        partition="train",
        snapshot_context="snapshot",
        features=[0.0] * 9,
        predictions=[0.1, 0.2, 0.0, 0.3, -0.1, 0.4],
        quantiles=None,
        maturity_at=200,
        target=0.05,
    )


def test_trace_sink_publishes_rows_only_after_confirmation_and_preserves_values(tmp_path):
    import pyarrow.parquet as pq

    path = tmp_path / "traces.parquet"
    with traces_api().TraceSink(path, partition="train", block_rows=2) as sink:
        sink.append(trace_record(1))
        sink.append(trace_record(2))
        assert not path.exists()
        sink.confirm()
    assert pq.read_table(path).to_pylist() == [trace_record(1), trace_record(2)]


def test_trace_verification_seals_rows_before_atomic_publication(tmp_path):
    import pyarrow.parquet as pq

    path = tmp_path / "traces.parquet"
    with traces_api().TraceSink(path, partition="train") as sink:
        sink.append(trace_record())
        assert len(sink.table()) == 1
        with pytest.raises(ValueError, match="cerrad"):
            sink.append(trace_record(2))
        sink.confirm()
    assert pq.read_table(path).to_pylist() == [trace_record()]


@pytest.mark.parametrize(
    "change",
    [
        {"maturity_at": 100},
        {"input_available_at": 101},
        {"partition": "evaluation"},
        {"features": [0.0] * 8},
        {"target": float("nan")},
        {"predictions": [0.1] * 5},
    ],
)
def test_invalid_trace_does_not_publish_a_partial_phase(tmp_path, change):
    path = tmp_path / "traces.parquet"
    with pytest.raises(ValueError):
        with traces_api().TraceSink(path, partition="train", block_rows=2) as sink:
            sink.append(trace_record())
            sink.append(trace_record(2) | change)
            sink.confirm()
    assert not path.exists()
    assert not tuple(tmp_path.iterdir())


def test_duplicate_trace_is_rejected_and_unconfirmed_sink_is_removed(tmp_path):
    path = tmp_path / "traces.parquet"
    with pytest.raises(ValueError, match="repetida"):
        with traces_api().TraceSink(path, partition="train") as sink:
            sink.append(trace_record())
            sink.append(trace_record())
    assert not tuple(tmp_path.iterdir())


def test_committed_phase_reuses_only_unchanged_identity_and_files(tmp_path):
    from mars_titan.data.storage import atomic_json, sha256

    path = tmp_path / "trace.parquet"
    path.write_bytes(b"confirmed")
    receipt = dict(
        status="completed", identity=dict(parent="fixed"), artifacts={path.name: sha256(path)}
    )
    atomic_json(tmp_path / "phase.json", receipt)
    assert traces_api().completed_phase(tmp_path, dict(parent="fixed")) == receipt
    with pytest.raises(ValueError, match="identidad"):
        traces_api().completed_phase(tmp_path, dict(parent="different"))
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="artefacto"):
        traces_api().completed_phase(tmp_path, dict(parent="fixed"))


def test_failed_attempt_time_reduces_remaining_budget(tmp_path):
    from mars_titan.data.storage import atomic_json

    atomic_json(
        tmp_path / "measurements/failed.json", dict(status="failed", process_seconds=21599.5)
    )
    assert traces_api().remaining_seconds(tmp_path) == 0.5
    atomic_json(tmp_path / "measurements/failed.json", dict(status="failed", process_seconds=-1))
    with pytest.raises(ValueError, match="coste"):
        traces_api().remaining_seconds(tmp_path)


@pytest.fixture(scope="module")
def technical_streams(tmp_path_factory):
    from tests.training.test_financial_run import corpus

    return corpus(tmp_path_factory.mktemp("refinement-study-streams"))


def test_real_route_collection_matches_exact_view_rows_without_parameter_updates(
    tmp_path, technical_streams, monkeypatch
):
    import os

    import torch

    from mars_titan.memory.native_backend import load_native
    from tests.training.test_refinement_inference import build

    dataset, sources = technical_streams
    source = sources["validation"]
    native = load_native(os.environ["MARS_TITAN_EPISODIC_NATIVE"])
    monkeypatch.setattr(torch.backends.mha, "get_fastpath_enabled", lambda: False)
    path = tmp_path / "traces.parquet"
    with traces_api().TraceSink(path, partition="validation", block_rows=2) as sink:
        engine = build(sources, native, traces=sink)
        before = {name: value.detach().clone() for name, value in engine.readout.named_parameters()}
        result = engine.evaluate(source)
        table = sink.table()
        expected = traces_api().expected_rows(dataset, source.phase)
        checked = traces_api().validate_trace_rows(table, expected, source.phase)
        assert checked == dict(missing=0, foreign=0, repeated=0, target=0)
        assert result["updates"] == 0 and result["labels"] == table.num_rows
        assert result["mac_updates"] == result["observations"]
        assert all(
            torch.equal(before[name], value) for name, value in engine.readout.named_parameters()
        )
        for wrong in (table.slice(1), table.append_column("extra", table["target"]).slice(1)):
            with pytest.raises(ValueError, match="filas"):
                traces_api().validate_trace_rows(wrong, expected, source.phase)
        sink.confirm()


def test_phase_verifier_rejects_changed_targets_and_temporal_boundary():
    import numpy as np
    import pyarrow as pa

    from mars_titan.memory.financial_session import FinancialPhase

    record = trace_record()
    phase = FinancialPhase("train", 1, 100, 300, 300)
    expected = dict(
        market=["US"], asset_id=["US/A1"], prediction_at=np.array([100]), target=np.array([0.05])
    )
    table = pa.Table.from_pylist([record])
    assert not any(traces_api().validate_trace_rows(table, expected, phase).values())
    for field, value in (("target", 0.9), ("decision_at", 300), ("maturity_at", 300)):
        with pytest.raises(ValueError):
            traces_api().validate_trace_rows(
                pa.Table.from_pylist([record | {field: value}]), expected, phase
            )


def policy_api():
    try:
        return importlib.import_module("benchmarks.refinement_study_policy")
    except ModuleNotFoundError:
        pytest.fail("Falta la selección sellada de la política")


def test_session_weights_do_not_count_many_assets_as_independent_sessions():
    import numpy as np

    weights = policy_api().session_weights(np.array([10, 10, 10, 20]))
    np.testing.assert_allclose(weights, [1 / 3, 1 / 3, 1 / 3, 1.0])
    with pytest.raises(ValueError):
        policy_api().session_weights(np.array([]))


def test_selection_uses_validation_error_then_work_then_declared_order():
    candidates = [
        dict(id="first", session_mae=0.1, mean_refinements=2.0),
        dict(id="second", session_mae=0.1, mean_refinements=1.0),
        dict(id="third", session_mae=0.1, mean_refinements=1.0),
    ]
    assert policy_api().best_candidate(candidates)["id"] == "second"
    candidates[0]["session_mae"] = 0.09
    assert policy_api().best_candidate(candidates)["id"] == "first"
    candidates[0]["session_mae"] = float("nan")
    with pytest.raises(ValueError):
        policy_api().best_candidate(candidates)


def test_policy_round_trip_preserves_choices_and_rejects_forged_identity(tmp_path):
    import torch

    from mars_titan.models.titans.refinement_policy import RoutePolicy

    zeros = torch.zeros(9, dtype=torch.float64)
    coefficients = torch.zeros((10, 5), dtype=torch.float64)
    coefficients[-1, 1] = 0.5
    model = RoutePolicy(
        mean=zeros,
        scale=zeros + 1,
        coefficients=coefficients,
        support_min=zeros - 1,
        support_max=zeros + 1,
        costs=torch.zeros(5, dtype=torch.float64),
        min_gain=0.0,
    )
    path = tmp_path / "policy.json"
    policy_api().save_policy(path, model)
    restored = policy_api().load_policy(path, device="cpu")
    assert restored(torch.zeros((2, 9), dtype=torch.float64)).tolist() == [2, 2]
    document = json.loads(path.read_text())
    document["identity"]["fallback"] = "ignore_outliers"
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError):
        policy_api().load_policy(path, device="cpu")


def test_threshold_round_trip_keeps_feature_cutoff_and_continuation(tmp_path):
    import torch

    from mars_titan.models.titans.refinement_controls import ThresholdRoutePolicy

    path = tmp_path / "threshold.json"
    model = ThresholdRoutePolicy(feature_name="first_interval_width", cutoff=0.3, route="k4cr")
    policy_api().save_policy(path, model)
    restored = policy_api().load_policy(path, device="cpu")
    features = torch.zeros((3, 9), dtype=torch.float64)
    features[:, 3] = torch.tensor([0.2, 0.3, 0.4], dtype=torch.float64)
    assert restored(features).tolist() == [0, 0, 4]


def test_trace_control_keeps_rows_targets_and_named_route_quantiles():
    import pyarrow as pa

    from mars_titan.models.quantile_head import QUANTILE_COLUMNS

    row = trace_record()
    row["quantiles"] = [[float(j) + shift for shift in (-2, -1, 0, 1, 2)] for j in range(6)]
    table = policy_api().route_table(pa.Table.from_pylist([row]), "k4cr")
    assert table["prediction"].to_pylist() == [-0.1]
    assert table["target"].to_pylist() == [0.05]
    assert table["sample_id"].to_pylist() == ["US/A1/100"]
    assert [table[column].to_pylist()[0] for column in QUANTILE_COLUMNS] == [
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
    ]


def test_selected_phase_cannot_start_without_the_learning_exception(tmp_path, monkeypatch):
    from mars_titan.training.learning_hold import LearningHoldError

    hold = tmp_path / "hold.json"
    hold.write_text(json.dumps(dict(training_allowed=False)))
    monkeypatch.setenv("MARS_TITAN_TRAINING_HOLD", str(hold))
    with pytest.raises(LearningHoldError):
        policy_api().evaluate_selected(tmp_path / "absent", {})


def test_ridge_fit_keeps_signed_losses_and_persists_the_actual_policy(tmp_path):
    import torch

    from mars_titan.training.refinement_policy_fit import fit_policy

    model, receipt = fit_policy(
        features=torch.zeros((4, 9), dtype=torch.float64),
        predictions=torch.tensor([[0.5, 0.0, 1.0, 0.5, 0.5, 0.5]] * 4, dtype=torch.float64),
        targets=torch.zeros(4, dtype=torch.float64),
        decision_at=torch.tensor([100, 100, 200, 200], dtype=torch.int64),
        maturity_at=torch.tensor([150, 150, 250, 250], dtype=torch.int64),
        session_weights=torch.ones(4, dtype=torch.float64) / 2,
        parent_selected_before=90,
        fit_start=100,
        fit_end=300,
        ridge=0.001,
        costs=torch.zeros(5, dtype=torch.float64),
        min_gain=0.0,
    )
    assert model.coefficients[-1, 0] > 0 and model.coefficients[-1, 1] < 0
    assert receipt["maximum_maturity_at"] == 250
    path = tmp_path / "policy.json"
    policy_api().save_policy(path, model)
    restored = policy_api().load_policy(path, device="cpu")
    assert restored(torch.zeros((4, 9), dtype=torch.float64)).tolist() == [1, 1, 1, 1]


def test_calibrator_is_frozen_before_evaluation_and_never_reads_its_targets(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from mars_titan.data.storage import sha256
    from mars_titan.models.quantile_head import QUANTILE_COLUMNS

    columns = dict(target=[0.0] * 256, prediction=[0.0] * 256, market=["US"] * 256)
    columns.update(
        {
            name: [value] * 256
            for name, value in zip(QUANTILE_COLUMNS, (-2.0, -1.0, 0.0, 1.0, 2.0), strict=True)
        }
    )
    table = pa.table(columns)
    calibration, evaluation = tmp_path / "calibration.parquet", tmp_path / "evaluation.parquet"
    pq.write_table(table, calibration)
    record = policy_api().fit_calibrator(tmp_path, calibration)
    before = sha256(tmp_path / "calibrator.json")
    assert not evaluation.exists()
    pq.write_table(table, evaluation)
    policy_api().apply_calibrator(tmp_path, record, evaluation)
    first = pq.read_table(tmp_path / "calibrated-evaluation.parquet")
    pq.write_table(
        table.set_column(table.schema.get_field_index("target"), "target", pa.array([999.0] * 256)),
        evaluation,
    )
    policy_api().apply_calibrator(tmp_path, record, evaluation)
    second = pq.read_table(tmp_path / "calibrated-evaluation.parquet")
    assert all(first[name].equals(second[name]) for name in QUANTILE_COLUMNS)
    assert sha256(tmp_path / "calibrator.json") == before
    assert first["prediction"].equals(table["prediction"])


def test_reconstruction_loads_selected_checkpoints_without_a_new_optimizer(
    tmp_path, learning_doubles
):
    import shutil

    from mars_titan.data.input_policy import HISTORICAL_MASKED
    from mars_titan.training.corpus_inputs import CorpusDataset
    from mars_titan.training.titans_walk_forward import _sources, unfused_attention, window_phases
    from tests.training.test_mars_titan_walk_forward import mars, readout_recipe
    from tests.training.test_titans_walk_forward import recipe, views, window

    work = tmp_path / "study"
    work.mkdir()
    view, protocol = views(tmp_path / "data")
    (work / "fold-017").mkdir()
    shutil.copyfile(view, work / "fold-017/view.json")
    with unfused_attention():
        parent = work / "runs/core"
        report, _ = window(
            view,
            protocol,
            recipe(tmp_path),
            parent,
            predictor_changes=dict(attention_layout="sequence_contiguous_v1"),
        )
        reader = work / "runs/reader"
        mars(
            view,
            parent,
            readout_recipe(tmp_path, refinement_initialization="zero_refinement"),
            reader,
        )
        dataset = CorpusDataset(view, input_policy=HISTORICAL_MASKED)
        fold = dataset.manifest["temporal_view"]["fold"]
        sources = _sources(dataset, window_phases(fold, 0), work / "indices")
        job = dict(parent="reader", seed=42)
        engine, origin = traces_api().reconstruct(
            work, job, sources["validation"], mode="fixed", route="k1", device="cpu"
        )
        result = engine.evaluate(sources["validation"])
    assert result["updates"] == 0 and result["labels"] == dataset.manifest["counts"]["validation"]
    assert origin["parent_checkpoint"] == report["checkpoint"]
    assert all(not p.requires_grad for p in engine.predictor.parameters())
    assert all(not p.requires_grad for p in engine.readout.parameters())


def test_staged_pipeline_collects_selects_seals_and_resumes_on_real_fixtures(
    tmp_path, monkeypatch, learning_doubles
):
    from datetime import date

    import pyarrow.parquet as pq

    from benchmarks.refinement_study_controls import frozen_reference
    from mars_titan.data.input_policy import HISTORICAL_MASKED
    from mars_titan.data.storage import sha256
    from mars_titan.data.temporal import MarketClock
    from mars_titan.training.mars_titan_walk_forward import run_mars_titan_window
    from mars_titan.training.temporal_corpus import prepare_temporal_corpus
    from mars_titan.training.titans_walk_forward import run_titans_window, unfused_attention
    from tests.training.historical_temporal_fixture import historical_temporal_fixture
    from tests.training.test_titans_walk_forward import Factory
    from tests.training.test_walk_forward_v2_views import sessions

    days = sessions("US")
    clock = MarketClock("US", "2022-10-01", "2022-12-31")
    days = sorted(set(days) | {day.isoformat() for day in clock.days if day < date(2022, 12, 1)})
    corpus = historical_temporal_fixture(
        tmp_path / "corpus", markets=("US",), days={"US": days}, assets=2
    )
    protocol = api().ROOT / "configs/evaluation/historical-masked-us-walk-forward-v2.json"
    sources = tmp_path / "sources"
    prepare_temporal_corpus(
        corpus.parent,
        protocol,
        None,
        None,
        sources,
        input_policy=HISTORICAL_MASKED,
        recover_annual_boundaries=True,
    )
    config = api().configuration()
    config["cohort_size"] = 2
    config["policy"].update(ridge=[0.1], min_gain=[0.0], cost_per_extra_refinement=[0.0])
    config["calibration"]["min_rows"] = 1
    monkeypatch.setattr(api(), "configuration", lambda: copy.deepcopy(config))
    monkeypatch.setattr(policy_api(), "configuration", lambda: copy.deepcopy(config))
    work = tmp_path / "work"
    prepared = api().prepare(sources, work)
    assert api().verify_prepared(work) == prepared
    chosen = {
        j["kind"]: j
        for j in prepared["jobs"]
        if j["projection"] == "linear_v1" and j["seed"] == 42 and j.get("depth", 1) == 1
    }
    source_hashes = {
        name: sha256(sources / name / "manifest.json") for name in ("fold-017", "fold-018")
    }
    with unfused_attention():
        run_titans_window(
            work / "fold-017/view.json",
            work / "fold-017/protocol.json",
            "fold-017",
            work / "recipes/linear_v1.json",
            variant="mac_online",
            seed=42,
            search_case="lr1e-3",
            output=work / "runs" / chosen["core"]["id"],
            device="cpu",
            optimizer_factory=Factory(retain_gradients=False),
        )
        run_mars_titan_window(
            work / "fold-017/view.json",
            work / "runs" / chosen["core"]["id"],
            work / "recipes/reader.json",
            components=traces_api().reader_components(1),
            seed=42,
            search_case="lr1e-3",
            output=work / "runs" / chosen["reader"]["id"],
            device="cpu",
            optimizer_factory=Factory(retain_gradients=False),
        )
        run_mars_titan_window(
            work / "fold-017/view.json",
            work / "runs" / chosen["core"]["id"],
            work / "recipes/reader.json",
            components={"episodic_bank": "m0_no_bank"},
            seed=42,
            search_case="lr1e-3",
            output=work / "runs" / chosen["m0_reader"]["id"],
            device="cpu",
            optimizer_factory=Factory(retain_gradients=False),
        )
        from mars_titan.training.financial_run import Paused

        reference_root = work / "runs" / chosen["reference"]["id"]

        class StopAfterValidation:
            @property
            def requested(self):
                path = reference_root / "validation/phase.json"
                return path.exists() and json.loads(path.read_text())["status"] == "completed"

        with pytest.raises(Paused):
            frozen_reference(work, chosen["reference"], device="cpu", stop=StopAfterValidation())
        assert not (reference_root / "frozen.json").exists()
        confirmed = sha256(reference_root / "validation/validation-predictions.parquet")
        frozen_reference(work, chosen["reference"], device="cpu")
        assert sha256(reference_root / "validation/validation-predictions.parquet") == confirmed
        frozen_reference(work, chosen["m0_reference"], device="cpu")
        collection = traces_api().collect(work, chosen["collect"], device="cpu")
        assert set(collection["phases"]) == {"train", "validation"}
        assert not (work / "runs" / chosen["collect"]["id"] / "evaluation").exists()
        selected = policy_api().select_policy(work, chosen["select"], device="cpu")
        assert selected["calibration_used_for_selection"] is False
        assert selected["evaluation_used_for_selection"] is False
        selection = work / "runs" / chosen["select"]["id"] / "phase.json"
        selection_hash = sha256(selection)
        evaluated = policy_api().evaluate_selected(work, chosen["evaluate"], device="cpu")
        assert evaluated["status"] == "completed"
        assert evaluated["comparison"]["missing_controls"] == []
        assert {
            "policy",
            "core",
            "m0",
            "scalar",
            "zero",
            "threshold",
            "hashed",
            "per_step_k4",
        } <= set(evaluated["comparison"]["scores"])
        assert sha256(selection) == selection_hash
        output = work / "runs" / chosen["evaluate"]["id"]
        original = sha256(output / "evaluation/predictions.parquet")
        again = policy_api().evaluate_selected(work, chosen["evaluate"], device="cpu")
        assert again["selection_sha256"] == selection_hash
        assert sha256(output / "evaluation/predictions.parquet") == original
        table = pq.read_table(output / "calibrated-evaluation.parquet")
        assert len(table) == prepared["counts"]["fold-018"]["evaluation"]
    assert all(
        sha256(sources / name / "manifest.json") == digest for name, digest in source_hashes.items()
    )
