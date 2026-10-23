"""Transferencia exacta de una candidata regional/J en CUDA FP32, sin ajuste."""

import copy
import random
from dataclasses import replace

import numpy as np
import pytest
import torch

from mars_titan.models.titans.episodic_readout import EpisodicReadout
from mars_titan.models.titans.financial import FinancialPredictor, _tensor_digest
from mars_titan.models.titans.mechanism_consumer import FrozenMechanismConsumer
from mars_titan.proposals.contracts import ProposalRegistry
from tests.proposals.test_candidate_runtime import materialize, tensors
from tests.proposals.test_candidate_runtime import runtime_source as runtime_source
from tests.proposals.test_mature_proposals import no_optimizer_steps as no_optimizer_steps


@pytest.mark.cuda
def test_combined_program_j2_transfers_fp32_tensors_without_aliases_or_rng_changes(runtime_source):
    original = runtime_source["parent"]
    original_identity = original.identity()
    original_tensors = tensors(original)
    original_digests = {name: _tensor_digest(value) for name, value in original_tensors.items()}
    cpu_rng, cuda_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state(0).clone()
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    entries = tuple(
        replace(entry, dtype="torch.float32") for entry in runtime_source["registry"].entries
    )
    registry = ProposalRegistry(entries, entries[0].configuration_id)
    with torch.random.fork_rng(devices=[0]):
        predictor = (
            FinancialPredictor(entries[0].predictor, device="cuda:0", dtype=torch.float32)
            .eval()
            .requires_grad_(False)
        )
        reader = (
            EpisodicReadout(entries[0].readout, device="cuda:0", dtype=torch.float32)
            .eval()
            .requires_grad_(False)
        )
        for source, destination in ((original.predictor, predictor), (original.readout, reader)):
            payload = copy.deepcopy(destination.state_dict())
            for name, value in source.state_dict().items():
                if isinstance(value, torch.Tensor):
                    payload[name] = value.detach().to(dtype=payload[name].dtype).clone()
            destination.load_state_dict(payload)
        predictor._seal_parameters()
        parent = FrozenMechanismConsumer(
            predictor, readout=reader, action=entries[0].action, catalog=entries[0].catalog
        )
    source = runtime_source | dict(parent=parent, registry=registry, baseline=entries[0])
    parent_identity = parent.identity()
    result = materialize(source, entries[-1])
    record = result.record()
    assert result.configuration_id == entries[-1].configuration_id == record["configuration_id"]
    assert result.configuration_id != runtime_source["registry"].entries[-1].configuration_id
    assert record["registry_id"] == registry.registry_id != runtime_source["registry"].registry_id
    assert record["registered_configuration"] == entries[-1].identity()
    assert record["registered_configuration"]["dtype"] == "torch.float32"
    assert record["effective_configuration"]["dtype"] == "torch.float32"
    assert result.effective_configuration.dtype == "torch.float32"
    assert record["effective_configuration_id"] == result.effective_configuration.configuration_id
    assert record["effective_configuration_id"] != result.configuration_id
    assert result.consumer.predictor.config.regional_execution.program == "summary_interleaved_v1"
    assert result.consumer.readout.config.evidence_bridge.recurrence.steps == 2
    assert record["rng_preserved"] is True
    assert record["parameter_aliases"] is record["runtime_state_transferred"] is False
    expected, actual = tensors(parent), tensors(result.consumer)
    assert actual.keys() == expected.keys() == original_tensors.keys() == record["tensors"].keys()
    for name, value in actual.items():
        before, old = expected[name], original_tensors[name]
        assert value.device == before.device == torch.device("cuda:0")
        assert value.dtype == before.dtype
        if value.is_floating_point():
            assert value.dtype == torch.float32 and old.dtype == torch.float64
        else:
            assert value.dtype == old.dtype
        assert torch.equal(before, old.to(dtype=before.dtype))
        assert torch.equal(value, before)
        assert value.untyped_storage().data_ptr() != before.untyped_storage().data_ptr()
        assert before.untyped_storage().data_ptr() != old.untyped_storage().data_ptr()
        assert record["tensors"][name]["dtype"] == str(value.dtype)
        assert record["tensors"][name]["bytes"] == value.numel() * value.element_size()
        assert record["tensors"][name]["source_sha256"] == _tensor_digest(before)
        assert record["tensors"][name]["target_sha256"] == _tensor_digest(value)
    parent.verify(strong=True)
    original.verify(strong=True)
    assert parent.identity() == parent_identity and original.identity() == original_identity
    assert {
        name: _tensor_digest(value) for name, value in original_tensors.items()
    } == original_digests
    assert torch.equal(cpu_rng, torch.get_rng_state())
    assert torch.equal(cuda_rng, torch.cuda.get_rng_state(0))
    assert python_rng == random.getstate()
    for before, after in zip(numpy_rng, np.random.get_state(), strict=True):
        assert np.array_equal(before, after)
