"""Carga del estado elegido de E2 desde archivos de fixture, sin entrenar."""

import copy
import hashlib
import json
import os

import pytest
import torch
from test_financial_session_controls import four_flow_source as four_flow_source
from test_financial_session_controls import no_target_estimation as no_target_estimation
from test_frozen_financial import frozen_backend as frozen_backend
from test_native_episode_backend import native as native

from tests.memory.spaced_selected_fixture import (
    digest,
    replace_selected_reader,
    selected_fixture,
    write,
)


@pytest.fixture
def selected(tmp_path, four_flow_source):
    return selected_fixture(tmp_path, four_flow_source)


def load(selected, **options):
    from mars_titan.training.spaced_selected import load_selected_reader

    return load_selected_reader(
        selected["anchor"], selected["view"], selected["source"]["spec"], device="cpu", **options
    )


def test_selected_weights_are_exact_frozen_and_do_not_restore_rng_or_optimizer(selected):
    rng = torch.get_rng_state().clone()
    loaded = load(selected)
    for module, expected in (
        (loaded.consumer.predictor, selected["core"]),
        (loaded.consumer.readout, selected["reader"]),
    ):
        assert not module.training and all(not p.requires_grad for p in module.parameters())
        for key, item in module.state_dict().items():
            if isinstance(item, torch.Tensor):
                assert torch.equal(item, expected[key])
            else:
                assert item == expected[key]
    assert torch.equal(torch.get_rng_state(), rng)
    loaded.consumer.verify()
    assert loaded.recipe.neighbors == 4
    assert loaded.identity()["state_transfer"] == "shared_parameters_only"
    assert not torch.cuda.is_initialized()


def test_provenance_is_returned_as_an_independent_copy(selected):
    loaded = load(selected)
    original = loaded.identity()
    changed = loaded.identity()
    changed["anchor"]["run_sha256"] = "0" * 64
    assert loaded.identity() == original


def test_quantile_parent_preserves_every_selected_head_tensor(tmp_path, four_flow_source):
    fixture = selected_fixture(tmp_path, four_flow_source, quantiles=True)
    loaded = load(fixture)
    assert loaded.recipe.loss == "pinball"
    assert loaded.consumer.predictor.head.out_features == 5
    for name, item in loaded.consumer.predictor.state_dict().items():
        if isinstance(item, torch.Tensor):
            assert torch.equal(item, fixture["core"][name])


@pytest.mark.parametrize(
    "change",
    [
        lambda r: r.update(status="paused"),
        lambda r: r.update(final_test_opened=True),
        lambda r: r["request"].update(seed=True),
        lambda r: r["request"].update(view_sha256="0" * 64),
        lambda r: r["request"].update(components=dict(episodic_bank="m0_no_bank")),
        lambda r: r["request"].update(components=dict(episodic_bank="m1", refinements=2)),
        lambda r: r["checkpoint"].update(path="other.json"),
        lambda r: r["fit"].update(sha256="0" * 64),
    ],
)
def test_changed_selection_or_unsupported_variant_is_rejected(selected, change):
    report = copy.deepcopy(selected["report"])
    change(report)
    report["identity"]["request"] = copy.deepcopy(report["request"])
    report["run_id"] = digest(report["identity"])
    (selected["anchor"] / "run.json").write_text(json.dumps(report))
    with pytest.raises(ValueError):
        load(selected)


def test_corrupt_selected_descriptor_is_rejected(selected):
    path = selected["anchor"] / "selected.json"
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError):
        load(selected)


@pytest.mark.parametrize(
    "change",
    [
        lambda d: d.update(schema_version=True),
        lambda d: d.update(kind="another_selected_state"),
        lambda d: d.update(variant_sha256="0" * 64),
        lambda d: d["parent"].update(checkpoint_sha256="0" * 64),
    ],
)
def test_resealed_descriptor_still_needs_the_original_semantics(selected, change):
    descriptor = copy.deepcopy(selected["selected"])
    change(descriptor)
    current = copy.deepcopy(selected["report"])
    current["checkpoint"]["sha256"] = write(selected["anchor"] / "selected.json", descriptor)
    write(selected["anchor"] / "run.json", current)
    with pytest.raises(ValueError):
        load(selected)


def test_resealed_fit_rejects_a_boolean_schema_version(selected):
    path = selected["anchor"] / "fit/run.json"
    fit = json.loads(path.read_text())
    fit["schema_version"] = True
    report = copy.deepcopy(selected["report"])
    report["fit"]["sha256"] = write(path, fit)
    write(selected["anchor"] / "run.json", report)
    with pytest.raises(ValueError):
        load(selected)


@pytest.mark.parametrize("name", ["recipe", "parent"])
def test_malformed_embedded_objects_are_rejected_as_invalid_metadata(selected, name):
    report = copy.deepcopy(selected["report"])
    if name == "recipe":
        report["identity"]["recipe"] = None
    else:
        report["request"]["parent"] = None
        report["identity"]["request"] = copy.deepcopy(report["request"])
    report["run_id"] = digest(report["identity"])
    write(selected["anchor"] / "run.json", report)
    with pytest.raises(ValueError):
        load(selected)


def test_oversized_checkpoint_is_rejected_before_hashing_its_payload(selected, monkeypatch):
    from mars_titan.training import spaced_selected
    from mars_titan.training.checkpoints import MAX_STATE_BYTES

    path = selected["anchor"] / selected["selected"]["readout"]["path"]
    os.truncate(path, MAX_STATE_BYTES + 1)
    original = spaced_selected.sha256

    def bounded(filename):
        if filename == path:
            pytest.fail("Se intentó leer un checkpoint que supera el presupuesto")
        return original(filename)

    monkeypatch.setattr(spaced_selected, "sha256", bounded)
    with pytest.raises(ValueError):
        load(selected)


def test_descriptor_with_non_hexadecimal_checkpoint_hash_is_rejected(selected):
    fit_path = selected["anchor"] / "fit/run.json"
    fit = json.loads(fit_path.read_text())
    fit["best_checkpoint"].update(sha256="z" * 64, path="checkpoints/state-" + "z" * 64 + ".pt")
    descriptor = copy.deepcopy(selected["selected"])
    descriptor["readout"] = dict(
        fit["best_checkpoint"], path="fit/" + fit["best_checkpoint"]["path"]
    )
    report = copy.deepcopy(selected["report"])
    report["fit"]["best_checkpoint"] = descriptor["readout"]
    report["fit"]["sha256"] = write(fit_path, fit)
    report["checkpoint"]["sha256"] = write(selected["anchor"] / "selected.json", descriptor)
    write(selected["anchor"] / "run.json", report)
    with pytest.raises(ValueError):
        load(selected)


@pytest.mark.parametrize("kind", ["nan", "shape", "dtype", "digest", "missing_model"])
def test_resealed_weights_cannot_change_the_selected_tensor_contract(selected, kind):
    from mars_titan.training.checkpoints import load_training_state

    fit = json.loads((selected["anchor"] / "fit/run.json").read_text())
    state = load_training_state(
        selected["anchor"] / "fit/checkpoints", expected_identity=fit["identity"], selection="best"
    )
    if kind == "nan":
        state["model"]["step_logit"].fill_(float("nan"))
    elif kind == "shape":
        state["model"]["step_logit"] = torch.zeros(2, dtype=torch.float64)
    elif kind == "dtype":
        state["model"]["step_logit"] = state["model"]["step_logit"].to(torch.float32)
    elif kind == "missing_model":
        state["model"] = None
    if kind in ("digest", "missing_model"):
        state["readout_sha256"] = "0" * 64
    else:
        hashes = {
            name: hashlib.sha256(value.numpy().tobytes()).hexdigest()
            for name, value in state["model"].items()
            if isinstance(value, torch.Tensor)
        }
        state["readout_sha256"] = digest(hashes)
    replace_selected_reader(selected, state)
    with pytest.raises(ValueError):
        load(selected)


def test_source_change_during_construction_is_detected_before_return(selected, monkeypatch):
    from mars_titan.training import spaced_selected

    original = spaced_selected._carried_readout
    path, before = selected["view"], selected["view"].read_bytes()

    def changed(*args):
        original(*args)
        path.write_bytes(before + b" ")

    monkeypatch.setattr(spaced_selected, "_carried_readout", changed)
    try:
        with pytest.raises(ValueError, match="cambió"):
            load(selected)
    finally:
        path.write_bytes(before)


def test_loader_does_not_adopt_a_symlinked_selected_file(selected):
    path = selected["anchor"] / "selected.json"
    target = path.with_name("real-selected.json")
    path.rename(target)
    path.symlink_to(target)
    with pytest.raises(ValueError):
        load(selected)


def test_another_observation_index_is_not_an_implicit_parameter_transfer(selected):
    from mars_titan.models.titans.financial_inputs import FinancialInputSpec
    from mars_titan.training.spaced_selected import load_selected_reader

    original = selected["source"]["spec"]
    specification = FinancialInputSpec(
        source_sha256=original.source_sha256,
        view_sha256="0" * 64,
        representation=original.representation,
        dimensions=original.dimensions,
        input_policy=original.input_policy,
    )
    with pytest.raises(ValueError, match="índice"):
        load_selected_reader(selected["anchor"], selected["view"], specification, device="cpu")


@pytest.mark.parametrize("field", ["request", "indices", "recipe"])
def test_malformed_parent_report_is_rejected_as_invalid_metadata(selected, field):
    parent = selected["anchor"].parent / "parent/run.json"
    report = json.loads(parent.read_text())
    if field == "request":
        report["request"] = None
        report["identity"]["request"] = None
    elif field == "indices":
        report["identity"]["indices"] = None
    else:
        report["identity"]["recipe"] = None
    report["run_id"] = digest(report["identity"])
    anchor_report = copy.deepcopy(selected["report"])
    anchor_report["request"]["parent"]["run_sha256"] = write(parent, report)
    anchor_report["identity"]["request"] = copy.deepcopy(anchor_report["request"])
    descriptor = copy.deepcopy(selected["selected"])
    descriptor["parent"] = copy.deepcopy(anchor_report["request"]["parent"])
    anchor_report["checkpoint"]["sha256"] = write(selected["anchor"] / "selected.json", descriptor)
    anchor_report["run_id"] = digest(anchor_report["identity"])
    write(selected["anchor"] / "run.json", anchor_report)
    with pytest.raises(ValueError):
        load(selected)


def test_malformed_parent_identity_of_reader_fit_is_rejected(selected):
    path = selected["anchor"] / "fit/run.json"
    fit = json.loads(path.read_text())
    fit["identity"]["parent"] = None
    fit["run_id"] = digest(fit["identity"])
    report = copy.deepcopy(selected["report"])
    report["fit"].update(run_id=fit["run_id"], sha256=write(path, fit))
    report["identity"]["fit_run_id"] = fit["run_id"]
    report["run_id"] = digest(report["identity"])
    write(selected["anchor"] / "run.json", report)
    with pytest.raises(ValueError):
        load(selected)


@pytest.mark.native("MARS_TITAN_EPISODIC_NATIVE")
def test_selected_consumer_runs_the_spaced_phase_without_transferred_memory(selected, native):
    from test_spaced_inference import PHASE, sequence

    from mars_titan.memory.spaced_retention import SpacedRetentionConfig
    from mars_titan.training.spaced_inference import SpacedInference

    loaded = load(selected)
    engine = SpacedInference(
        loaded.consumer,
        loaded.recipe,
        prefixes=selected["source"]["prefixes"],
        retention=SpacedRetentionConfig(capacity=16, frontier=1),
        native=native,
        codec=loaded.codec,
        world="loaded_fixture",
        fold="0",
    )
    first = engine._pass(PHASE, sequence(selected["source"]))
    evidence = engine.spaced_state()
    assert first["predictions"] == first["spaced"]["issued"] == 20
    assert first["labels"] == first["spaced"]["resolved"] == 16
    assert engine._pass(PHASE, sequence(selected["source"])) == first
    assert engine.spaced_state() == evidence
    loaded.consumer.verify()
