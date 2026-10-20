"""Contratos del piloto, sin abrir corpus ni ejecutar aprendizaje."""

import copy
import importlib
import json

import pytest


def api():
    return importlib.import_module("benchmarks.neural_plasticity_study")


def view():
    return dict(
        input_policy="historical_masked_2000_v1",
        final_test_opened=False,
        assets=[dict(market="US", symbol=f"A{i:03}") for i in range(64)],
        counts=dict(train=123, validation=12, calibration=6, evaluation=25),
        temporal_view=dict(
            fold=dict(
                id="fold-018",
                train=["2000-01-01", "2022-04-01"],
                validation=["2022-04-01", "2022-10-01"],
                calibration=["2022-10-01", "2023-01-01"],
                evaluation=["2023-01-01", "2024-01-01"],
            ),
            protocol=dict(
                market="US",
                final_test_start="2024-01-01",
                primary_metric="session_mae",
                selection=dict(max_epochs=2),
            ),
        ),
    )


def prepared(tmp_path):
    folder = tmp_path / "source"
    folder.mkdir()
    source = folder / "view.json"
    source.write_text(json.dumps(view()))
    work = tmp_path / "study"
    return work, api().prepare(source, work)


def test_projection_keeps_every_asset_and_row_and_leaves_the_source_untouched():
    original = view()
    before = copy.deepcopy(original)
    actual = api().project_view(original, api().configuration())
    assert actual["counts"] == original["counts"]
    assert actual["assets"] == original["assets"]
    assert actual["temporal_view"]["fold"] == original["temporal_view"]["fold"]
    assert actual["temporal_view"]["protocol"]["selection"]["max_epochs"] == 1
    assert original == before


def test_a_future_training_boundary_cannot_hide_behind_the_same_fold_identifier():
    original = view()
    original["temporal_view"]["fold"]["train"][1] = "2025-01-01"
    with pytest.raises(ValueError):
        api().project_view(original, api().configuration())


@pytest.mark.parametrize("fault", ["protocol_market", "asset_market", "duplicate_asset"])
def test_a_wrong_market_or_repeated_asset_cannot_enter_the_declared_cohort(fault):
    original = view()
    if fault == "protocol_market":
        original["temporal_view"]["protocol"]["market"] = "CN"
    elif fault == "asset_market":
        original["assets"][0]["market"] = "CN"
    else:
        original["assets"][1] = copy.deepcopy(original["assets"][0])
    with pytest.raises(ValueError):
        api().project_view(original, api().configuration())


def test_prepared_catalogue_and_artifacts_are_sealed(tmp_path):
    work, receipt = prepared(tmp_path)
    assert api().verify_prepared(work) == receipt
    assert len(receipt["jobs"]) == 11
    assert len({job["id"] for job in receipt["jobs"]}) == 11
    recipe = json.loads((work / api().FILES["recipe"]).read_text())
    assert recipe["predictor"]["memory_projections"] == "titans_mac_paper_projections_v2"
    assert recipe["recipe"]["epochs"] == 1
    receipt["artifacts"].pop(api().FILES["recipe"])
    (work / "prepared.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        api().verify_prepared(work)


def test_scalar_control_study_is_separate_and_does_not_replace_the_historical_catalogue(tmp_path):
    module = api()
    source = tmp_path / "source" / "view.json"
    source.parent.mkdir()
    source.write_text(json.dumps(view()))
    work = tmp_path / "scalar"
    actual = module.prepare(source, work, version="v2")
    assert actual["study_version"] == "v2"
    assert module.verify_prepared(work) == actual
    legacy = dict(module.configuration())
    current = dict(actual["configuration"])
    current.pop("gradient_contract")
    current.pop("validity_control")
    assert current == legacy
    for job in actual["jobs"]:
        if "memory_plasticity" in job["changes"]:
            assert job["changes"]["memory_plasticity"]["gradient_contract"] == "scalar_controls_v2"
    historical = json.loads(module.VARIANTS.read_text())
    assert all(
        "gradient_contract" not in arm["predictor_changes"].get("memory_plasticity", {})
        for arm in historical["arms"].values()
    )
    actual["study_version"] = "v1"
    (work / "prepared.json").write_text(json.dumps(actual))
    with pytest.raises(ValueError):
        module.verify_prepared(work)
    with pytest.raises(ValueError):
        module.prepare(source, tmp_path / "unknown", version="v3")
    assert not (tmp_path / "unknown").exists()


def test_pending_reservation_blocks_a_second_attempt_until_its_real_close(tmp_path):
    work, receipt = prepared(tmp_path)
    job = receipt["jobs"][0]
    path, record = api()._reserve(work, receipt, job)
    with pytest.raises(RuntimeError):
        api()._reserve(work, receipt, job)
    record["status"] = "interrupted"
    api().settle_attempt(path, record, actual_seconds=12.5)
    _, following = api()._reserve(work, receipt, job)
    assert following["reserved_seconds"] == receipt["configuration"]["max_total_seconds"] - 12.5


def test_key_competition_reuses_the_sealed_pipeline_with_its_own_catalogue(tmp_path):
    source = tmp_path / "source" / "view.json"
    source.parent.mkdir()
    source.write_text(json.dumps(view()))
    work = tmp_path / "keys"
    receipt = api().prepare(source, work, version="key_competition_v1")
    assert api().verify_prepared(work) == receipt
    assert len(receipt["jobs"]) == 7
    assert {job["arm"] for job in receipt["jobs"]} == {
        "original",
        "all_components",
        "centered",
        "competitive",
        "centered_competitive",
        "fixed",
        "centered_fixed",
    }
    assert receipt["configuration"]["primary_candidate"] == "centered_competitive"
    assert receipt["configuration"]["validity_control"]["unit"] == "all_components"
    assert receipt["counts"] == view()["counts"]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1])
def test_corrupt_historical_cost_is_rejected(tmp_path, value):
    work, receipt = prepared(tmp_path)
    (work / "measurements").mkdir()
    (work / "measurements/corrupt.json").write_text(json.dumps(dict(process_seconds=value)))
    with pytest.raises(ValueError):
        api()._reserve(work, receipt, receipt["jobs"][0])


def test_learning_hold_precedes_any_view_or_recipe_read(tmp_path, learning_hold):
    from mars_titan.training.learning_hold import LearningHoldError

    learning_hold(False)
    with pytest.raises(LearningHoldError):
        api().run_job(tmp_path / "absent", "absent")
    assert not (tmp_path / "absent").exists()


@pytest.mark.parametrize("fails", [False, True])
@pytest.mark.parametrize("job_index", [0, 1])
def test_attempt_closes_its_cost_and_resources_when_the_executor_returns_or_fails(
    tmp_path, monkeypatch, fails, job_index
):
    import torch
    from torch.optim import optimizer

    from mars_titan.data.storage import outside_source
    from mars_titan.training import experiment_resources, titans_walk_forward
    from mars_titan.training.search_cases import checked_predictor_changes

    work, receipt = prepared(tmp_path)
    job = receipt["jobs"][job_index]
    calls = []

    class Lease:
        record = dict(device="cuda:0", fixture=True)

        def __init__(self, **kwargs):
            assert kwargs == dict(max_ram=6 * 1024**3, max_vram=6 * 1024**3)

        def __enter__(self):
            calls.append("enter")
            return self

        def check(self):
            calls.append("check")

        def __exit__(self):
            calls.append("exit")

    def executor(view, protocol, fold, recipe, **kwargs):
        assert view == work / api().FILES["view"] and recipe == work / api().FILES["recipe"]
        assert protocol == work / api().FILES["protocol"] and fold == "fold-018"
        assert kwargs["device"] == "cuda:0"
        outside_source(view.parent, kwargs["output"])
        accepted = checked_predictor_changes(kwargs["predictor_changes"])
        assert accepted == (job["changes"] or None)
        assert kwargs["seed"] == 42 and kwargs["search_case"] == "lr1e-3"
        assert kwargs["stop"].steps == 0
        calls.append("executor")
        if fails:
            raise RuntimeError("Fallo de ejecución del fixture")
        return dict(status="paused", run_id="fixture")

    monkeypatch.setattr(experiment_resources, "GpuLease", Lease)
    monkeypatch.setattr(titans_walk_forward, "run_titans_window", executor)
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: calls.append(device))
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setattr(torch, "get_num_interop_threads", lambda: 1)
    hooks = (
        len(optimizer._global_optimizer_pre_hooks),
        len(optimizer._global_optimizer_post_hooks),
    )
    if fails:
        with pytest.raises(RuntimeError, match="fixture"):
            api()._run(work, receipt, job)
    else:
        assert api()._run(work, receipt, job)["status"] == "paused"
    costs = list((work / "measurements").glob("*.json"))
    assert len(costs) == 1
    measured = json.loads(costs[0].read_text())
    assert measured["closure_confirmed"] is True
    assert measured["charged_seconds"] == measured["actual_seconds"] > 0
    assert measured["status"] == ("failed" if fails else "paused")
    assert measured["optimizer_steps"] == 0
    assert calls[0] == "enter" and calls[-1] == "exit"
    assert "cuda:0" in calls and "executor" in calls
    assert hooks == (
        len(optimizer._global_optimizer_pre_hooks),
        len(optimizer._global_optimizer_post_hooks),
    )


@pytest.mark.cuda
@pytest.mark.parametrize(
    "version,arm",
    [
        ("v1", name)
        for name in [
            "mac_original",
            "write_coherence",
            "cascade_original",
            "cascade_unity",
            "cascade_write",
            "cascade_capture",
            "cascade_combined",
            "cascade_fixed_write",
            "cascade_fixed_capture",
            "cascade_fixed_both",
            "fixed_write",
        ]
    ]
    + [
        ("key_competition_v1", name)
        for name in [
            "original",
            "all_components",
            "centered",
            "competitive",
            "centered_competitive",
            "fixed",
            "centered_fixed",
        ]
    ],
)
def test_real_trainer_accepts_prepared_inputs_and_pauses_without_optimizer_updates(
    tmp_path, monkeypatch, learning_doubles, version, arm
):
    from types import SimpleNamespace

    from mars_titan.training.titans_walk_forward import run_titans_window, unfused_attention
    from tests.training import test_titans_walk_forward as fixtures

    protocol = copy.deepcopy(fixtures.PROTOCOL)
    protocol["selection"]["patience"] = 5
    monkeypatch.setattr(fixtures, "PROTOCOL", protocol)
    source, _ = fixtures.views(tmp_path / "data")
    original = json.loads(source.read_text())
    config = api().configuration(version)
    config.update(
        cohort_size=3,
        fold=original["temporal_view"]["fold"]["id"],
        window={k: v for k, v in original["temporal_view"]["fold"].items() if k != "id"},
    )
    monkeypatch.setattr(api(), "configuration", lambda version="v1": copy.deepcopy(config))
    work = tmp_path / "study"
    receipt = api().prepare(source, work, version=version)
    artifacts = receipt["artifacts"]
    view_path = work / next(name for name in artifacts if name.endswith("view.json"))
    protocol_path = work / next(name for name in artifacts if name.endswith("protocol.json"))
    recipe_path = work / next(name for name in artifacts if name.endswith("recipe.json"))
    job = next(row for row in receipt["jobs"] if row["arm"] == arm)
    optimizer = fixtures.Factory(retain_gradients=False)
    with unfused_attention():
        report = run_titans_window(
            view_path,
            protocol_path,
            config["fold"],
            recipe_path,
            variant="mac_online",
            seed=42,
            device="cuda:0",
            output=work / "runs" / job["id"],
            indices=work / "indices",
            search_case=config["search_case"],
            predictor_changes=job["changes"] or None,
            optimizer_factory=optimizer,
            stop=SimpleNamespace(requested=True),
        )
    assert report["status"] == "paused"
    assert optimizer.instances and all(item.calls == 0 for item in optimizer.instances)
