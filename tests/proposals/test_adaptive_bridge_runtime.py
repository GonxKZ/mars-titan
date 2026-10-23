"""Materialización de parada residual y controles fijos con parámetros emparejados."""

import importlib
import random
import sys
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
import torch

from mars_titan.models.titans.episodic_readout import EpisodicReadout
from mars_titan.models.titans.financial import FinancialPredictor, _tensor_digest
from mars_titan.models.titans.mechanism_catalog import MechanismAction
from mars_titan.models.titans.mechanism_consumer import FrozenMechanismConsumer
from mars_titan.proposals.contracts import ProposalRegistry
from tests.proposals.test_candidate_runtime import materialize, tensors
from tests.proposals.test_candidate_runtime import runtime_source as runtime_source
from tests.proposals.test_mature_proposals import no_optimizer_steps as no_optimizer_steps

MODES = ("fixed", "adaptive_reference_full", "adaptive_compact")


def registered_source(source, parent_mode="fixed"):
    module = importlib.import_module("mars_titan.models.titans.bridge_recurrence")
    original = source["baseline"]
    bridge = original.readout.evidence_bridge
    fixed = replace(
        original,
        readout=replace(
            original.readout,
            evidence_bridge=replace(
                bridge, recurrence=replace(bridge.recurrence, steps=4, stopping=None)
            ),
        ),
    )
    entries = [fixed]
    for backend in MODES[1:]:
        geometry = replace(
            fixed.readout.evidence_bridge,
            recurrence=replace(
                fixed.readout.evidence_bridge.recurrence,
                stopping=module.BridgeStoppingConfig(1e308, 0.0, backend=backend),
            ),
        )
        entries.append(
            replace(
                fixed,
                readout=replace(fixed.readout, evidence_bridge=geometry),
                catalog=replace(fixed.catalog, reader_contract=geometry.identity()["kind"]),
            )
        )
    selected = entries[MODES.index(parent_mode)]
    reader = EpisodicReadout(selected.readout, device="cuda:0", dtype=torch.float64)
    payload = deepcopy(reader.state_dict())
    for name, value in source["parent"].readout.state_dict().items():
        if isinstance(value, torch.Tensor):
            payload[name] = value.detach().clone()
    reader.load_state_dict(payload)
    reader.eval().requires_grad_(False)
    parent = FrozenMechanismConsumer(
        source["parent"].predictor,
        readout=reader,
        action=selected.action,
        catalog=selected.catalog,
    )
    registry = ProposalRegistry(tuple(entries), selected.configuration_id)
    return source | dict(parent=parent, registry=registry, baseline=selected), tuple(entries)


@contextmanager
def forbidden_calls(*functions):
    """Rechazar forwards o construcción tardía sin alterar métodos sellados."""
    codes = {function.__code__ for function in functions}
    previous = sys.getprofile()

    def observe(frame, event, argument):
        if event == "call" and frame.f_code in codes:
            pytest.fail(f"La operación alcanzó {frame.f_code.co_name} antes de ser admisible")

    sys.setprofile(observe)
    try:
        yield
    finally:
        sys.setprofile(previous)


@pytest.mark.cuda
def test_adaptive_candidate_materialization_is_reachable_in_call(runtime_source):
    module = importlib.import_module("mars_titan.proposals.candidate_runtime")
    assert callable(getattr(module, "materialize_registered_candidate", None))
    source, entries = registered_source(runtime_source)
    result = materialize(source, entries[1])
    assert result.configuration_id == entries[1].configuration_id
    assert result.consumer.readout.config.evidence_bridge.recurrence.stopping.backend == MODES[1]


@pytest.mark.cuda
@pytest.mark.parametrize("parent_mode", MODES)
@pytest.mark.parametrize("target_mode", MODES)
def test_fixed_full_and_compact_transfer_exactly_the_same_tensors_without_state_or_forward(
    runtime_source, parent_mode, target_mode
):
    source, entries = registered_source(runtime_source, parent_mode)
    candidate = entries[MODES.index(target_mode)]
    parent = source["parent"]
    source_identity = parent.identity()
    registry_identity = source["registry"].identity()
    original_identity = runtime_source["parent"].identity()
    original_digests = {
        name: _tensor_digest(value) for name, value in tensors(runtime_source["parent"]).items()
    }
    cpu_rng, cuda_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state(0).clone()
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    with forbidden_calls(torch.nn.Module._call_impl):
        result = materialize(source, candidate)
        record = result.record()
    assert result.configuration_id == candidate.configuration_id
    assert record["registered_configuration"] == candidate.identity()
    assert record["effective_configuration"] == result.effective_configuration.identity()
    assert record["runtime_state_transferred"] is record["selection_performed"] is False
    assert record["parameter_aliases"] is record["codec_aliases"] is False
    assert record["rng_preserved"] is True
    assert record["mode"] == "paired_parent_weights_fresh_state_v1"
    actual, expected = tensors(result.consumer), tensors(parent)
    assert actual.keys() == expected.keys() == record["tensors"].keys()
    for name, value in actual.items():
        reference = expected[name]
        assert value.dtype == reference.dtype and value.shape == reference.shape
        assert torch.equal(value, reference), name
        if value.numel():
            assert value.untyped_storage().data_ptr() != reference.untyped_storage().data_ptr()
        assert record["tensors"][name]["source_sha256"] == _tensor_digest(reference)
        assert record["tensors"][name]["target_sha256"] == _tensor_digest(value)
    geometry = result.consumer.readout.config.evidence_bridge
    assert geometry.recurrence.steps == 4
    if target_mode == "fixed":
        assert geometry.recurrence.stopping is None
        assert result.consumer.catalog.reader_contract == "shared_evidence_bridge_nonlinear_v1"
    else:
        assert geometry.recurrence.stopping.backend == target_mode
        assert result.consumer.catalog.reader_contract == "shared_evidence_bridge_adaptive_v1"
    assert result.consumer.readout.config.refinements == 1
    assert (
        result.effective_configuration.predictor.inputs.identity()
        == source["input_spec"].identity()
    )
    assert (
        result.codec.identity()["projection_sha256"]
        == source["parent_codec"].identity()["projection_sha256"]
    )
    assert not np.shares_memory(result.codec._projection, source["parent_codec"]._projection)
    first = result.consumer.predictor.initial_state(("US/T0000",))
    second = result.consumer.predictor.initial_state(("US/T0000",))
    assert first.observed_steps.tolist() == second.observed_steps.tolist() == [0]
    assert first.mac.memory.steps.tolist() == second.mac.memory.steps.tolist() == [0]
    for left, right in zip(first.mac.memory.weights, second.mac.memory.weights, strict=True):
        assert torch.equal(left, right)
        assert left.untyped_storage().data_ptr() != right.untyped_storage().data_ptr()
    assert parent.identity() == source_identity
    assert source["registry"].identity() == registry_identity
    assert runtime_source["parent"].identity() == original_identity
    assert original_digests == {
        name: _tensor_digest(value) for name, value in tensors(runtime_source["parent"]).items()
    }
    assert torch.equal(cpu_rng, torch.get_rng_state())
    assert torch.equal(cuda_rng, torch.cuda.get_rng_state(0))
    assert python_rng == random.getstate()
    for before, after in zip(numpy_rng, np.random.get_state(), strict=True):
        assert np.array_equal(before, after)


def collateral(candidate, fault):
    reader, bridge = candidate.readout, candidate.readout.evidence_bridge
    if fault in ("topology", "seed", "gain"):
        changed = {
            "topology": dict(topology="complete"),
            "seed": dict(seed=bridge.seed + 1),
            "gain": dict(bridge_gain=0.375),
        }[fault]
        return replace(
            candidate, readout=replace(reader, evidence_bridge=replace(bridge, **changed))
        )
    if fault == "codec":
        return replace(
            candidate,
            readout=replace(
                reader, codec_id="e" * 64, evidence_bridge=replace(bridge, codec_id="e" * 64)
            ),
        )
    if fault == "action":
        return replace(candidate, action=MechanismAction(route="k2c"))
    if fault == "budget":
        return replace(
            candidate,
            catalog=replace(
                candidate.catalog, budget=replace(candidate.catalog.budget, max_reader_steps=2)
            ),
        )
    raise AssertionError(f"Cambio de prueba desconocido: {fault}")


@pytest.mark.cuda
@pytest.mark.parametrize("fault", ("topology", "seed", "gain", "codec", "action", "budget"))
def test_stopping_does_not_authorize_collateral_changes_before_model_construction(
    runtime_source, fault
):
    source, entries = registered_source(runtime_source)
    bad = collateral(entries[1], fault)
    registry = ProposalRegistry((*entries, bad), source["baseline"].configuration_id)
    source = source | dict(registry=registry)
    with forbidden_calls(
        FinancialPredictor.__init__, EpisodicReadout.__init__, torch.nn.Module._call_impl
    ):
        with pytest.raises(ValueError):
            materialize(source, bad)


@pytest.mark.cuda
def test_registry_keeps_k_separate_from_residual_stopping(runtime_source):
    _, entries = registered_source(runtime_source)
    with pytest.raises(ValueError):
        replace(entries[1], readout=replace(entries[1].readout, refinements=2))


@pytest.mark.cuda
@pytest.mark.parametrize("target_mode", MODES[1:])
def test_runtime_record_rejects_changed_regional_weights_instead_of_resealing(
    runtime_source, target_mode
):
    source, entries = registered_source(runtime_source)
    result = materialize(source, entries[MODES.index(target_mode)])
    parent_before = {
        name: _tensor_digest(value) for name, value in tensors(source["parent"]).items()
    }
    with torch.no_grad():
        result.consumer.readout.evidence_bridge.region_value[0, 0, 0].add_(0.125)
    with pytest.raises(ValueError):
        result.record()
    assert parent_before == {
        name: _tensor_digest(value) for name, value in tensors(source["parent"]).items()
    }
