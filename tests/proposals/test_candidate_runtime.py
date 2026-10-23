"""Materialización identificada de candidatas, sin ajuste ni estado de una sesión."""

import importlib
import importlib.util
import random
import sys
from contextlib import ExitStack, contextmanager
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import numpy as np
import pytest
import torch

from mars_titan.memory.episodic_codec import FrozenEpisodeCodec
from mars_titan.models.titans.bridge_recurrence import BridgeRecurrenceConfig
from mars_titan.models.titans.episodic_readout import EpisodicReadout, EpisodicReadoutConfig
from mars_titan.models.titans.evidence_bridge import EvidenceBridgeConfig
from mars_titan.models.titans.financial import FinancialConfig, FinancialPredictor, _tensor_digest
from mars_titan.models.titans.financial_inputs import FinancialInputSpec
from mars_titan.models.titans.mechanism_catalog import (
    MechanismAction,
    MechanismBudget,
    MechanismCatalog,
)
from mars_titan.models.titans.mechanism_consumer import FrozenMechanismConsumer
from mars_titan.models.titans.regional_execution import RegionalExecutionConfig
from mars_titan.models.titans.regional_fusion import RegionalFusionConfig
from mars_titan.proposals.contracts import ProposalRegistry, RegisteredConfiguration
from tests.proposals.test_mature_proposals import no_optimizer_steps as no_optimizer_steps


def api():
    return importlib.import_module("mars_titan.proposals.candidate_runtime")


@pytest.fixture(scope="module")
def runtime_source(tmp_path_factory):
    """Pesos fijados en CUDA y contratos de vista declarados, sin abrir una fase S."""
    assert torch.cuda.is_available(), "La fixture requiere CUDA, sin sustitución por CPU"
    torch.cuda.set_device("cuda:0")
    fastpath = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        with ExitStack() as helpers:
            imports = helpers.enter_context(pytest.MonkeyPatch.context())
            imports.syspath_prepend(str(Path(__file__).resolve().parents[1] / "memory"))
            controls = importlib.import_module("test_financial_session_controls")
            helpers.enter_context(contextmanager(controls.no_target_estimation.__wrapped__)())
            source = controls.four_flow_source.__wrapped__(tmp_path_factory)
            spec = source["spec"]
            codec = FrozenEpisodeCodec(
                spec,
                seed=73,
                modality_weights=(1.0, 2.0, 3.0, 4.0, 5.0),
                presence_weight=2.0,
                key_constant=0.75,
                max_buffer_bytes=8 * 1024**2,
            )
            program = RegionalExecutionConfig(program="summary_at_end_v1")
            config = FinancialConfig(
                spec,
                variant="mac_online",
                hidden_size=32,
                head="quantile_head_v1",
                regional_fusion=RegionalFusionConfig(topology="self", routing="uniform"),
                regional_execution=program,
            )
            bridge = EvidenceBridgeConfig(
                codec.fingerprint(),
                hidden_size=32,
                cycles=0,
                recurrence=BridgeRecurrenceConfig(steps=1, max_node_updates_per_row=456),
            )
            readout = EpisodicReadoutConfig(
                codec.fingerprint(), hidden_size=32, evidence_bridge=bridge
            )
            baseline = RegisteredConfiguration(
                predictor=config,
                readout=readout,
                action=MechanismAction(),
                catalog=MechanismCatalog(
                    reader_contract=bridge.identity()["kind"], regional_execution=program
                ),
                dtype="torch.float64",
            )
            interleaved = replace(program, program="summary_interleaved_v1")
            program_candidate = replace(
                baseline,
                predictor=replace(config, regional_execution=interleaved),
                catalog=replace(baseline.catalog, regional_execution=interleaved),
            )
            j2 = replace(
                baseline,
                readout=replace(
                    readout,
                    evidence_bridge=replace(
                        bridge,
                        recurrence=BridgeRecurrenceConfig(steps=2, max_node_updates_per_row=912),
                    ),
                ),
            )
            entries = (
                baseline,
                program_candidate,
                j2,
                replace(program_candidate, readout=j2.readout),
            )
            registry = ProposalRegistry(entries, baseline.configuration_id)
            predictor = (
                FinancialPredictor(config, device="cuda:0", dtype=torch.float64)
                .eval()
                .requires_grad_(False)
            )
            reader = (
                EpisodicReadout(readout, device="cuda:0", dtype=torch.float64)
                .eval()
                .requires_grad_(False)
            )
            # Todos los pesos difieren de la inicialización para detectar una copia omitida.
            with torch.no_grad():
                for index, value in enumerate((*predictor.parameters(), *reader.parameters()), 1):
                    value.add_(index * 1e-4)
            predictor._seal_parameters()
            parent = FrozenMechanismConsumer(
                predictor, readout=reader, action=baseline.action, catalog=baseline.catalog
            )
            phase_spec = FinancialInputSpec(
                source_sha256=spec.source_sha256,
                view_sha256="d" * 64,
                representation=spec.representation,
                dimensions=spec.dimensions,
                input_policy=spec.input_policy,
            )
            yield dict(
                parent=parent,
                parent_codec=codec,
                registry=registry,
                input_spec=phase_spec,
                baseline=baseline,
            )
    finally:
        torch.backends.mha.set_fastpath_enabled(fastpath)


def materialize(source, configuration=None, **changes):
    arguments = {name: source[name] for name in ("parent", "parent_codec", "input_spec")}
    arguments.update(changes)
    configuration = source["baseline"] if configuration is None else configuration
    return api().materialize_registered_candidate(
        source["registry"], configuration.configuration_id, **arguments
    )


def tensors(consumer):
    result = {}
    for prefix, module in (("predictor", consumer.predictor), ("readout", consumer.readout)):
        if module is not None:
            result.update(
                {
                    f"{prefix}.{name}": value
                    for name, value in (*module.named_parameters(), *module.named_buffers())
                }
            )
    return result


def test_candidate_runtime_api_is_available_in_call():
    name = "mars_titan.proposals.candidate_runtime"
    try:
        found = importlib.util.find_spec(name)
    except ModuleNotFoundError:
        found = None
    assert found is not None, "Falta el módulo de materialización de candidatas registradas"
    module = importlib.import_module(name)
    assert callable(getattr(module, "materialize_registered_candidate", None))


@pytest.mark.cuda
@pytest.mark.parametrize("index", [0, 1, 2, 3], ids=["baseline", "program", "j2", "program-j2"])
def test_materialization_preserves_registered_id_and_transfers_every_tensor(runtime_source, index):
    source = runtime_source
    entry = source["registry"].entries[index]
    before = (
        source["registry"].identity(),
        source["parent"].identity(),
        source["parent_codec"].identity(),
    )
    result = materialize(source, entry)
    record = result.record()
    effective = result.effective_configuration
    assert result.configuration_id == entry.configuration_id == record["configuration_id"]
    assert (
        effective.configuration_id == record["effective_configuration_id"] != entry.configuration_id
    )
    assert effective.predictor.inputs.identity() == source["input_spec"].identity()
    assert record["registered_configuration"] == entry.identity()
    assert record["effective_configuration"] == effective.identity()
    assert result.consumer.predictor is not source["parent"].predictor
    assert result.consumer.readout is not source["parent"].readout
    assert record["mode"] == "paired_parent_weights_fresh_state_v1"
    assert record["runtime_state_transferred"] is record["selection_performed"] is False
    assert record["parameter_aliases"] is record["codec_aliases"] is False
    assert record["rng_preserved"] is True
    original, actual = tensors(source["parent"]), tensors(result.consumer)
    assert actual.keys() == original.keys() == record["tensors"].keys()
    for name, value in actual.items():
        expected = original[name]
        torch.testing.assert_close(value, expected, rtol=0, atol=0)
        assert value.untyped_storage().data_ptr() != expected.untyped_storage().data_ptr()
        assert record["tensors"][name]["shape"] == list(value.shape)
        assert record["tensors"][name]["dtype"] == str(value.dtype)
        assert record["tensors"][name]["source_sha256"] == _tensor_digest(expected)
        assert record["tensors"][name]["target_sha256"] == _tensor_digest(value)
    assert (
        effective.predictor.regional_execution.identity()
        == entry.predictor.regional_execution.identity()
    )
    assert (
        effective.readout.evidence_bridge.recurrence.identity()
        == entry.readout.evidence_bridge.recurrence.identity()
    )
    assert effective.catalog.identity() == entry.catalog.identity()
    assert (
        effective.readout.codec_id
        == effective.readout.evidence_bridge.codec_id
        == result.codec.fingerprint()
    )
    assert result.codec.fingerprint() != source["parent_codec"].fingerprint()
    assert (
        result.codec.identity()["projection_sha256"]
        == source["parent_codec"].identity()["projection_sha256"]
    )
    assert not np.shares_memory(result.codec._projection, source["parent_codec"]._projection)
    source["parent"].verify(strong=True)
    assert before == (
        source["registry"].identity(),
        source["parent"].identity(),
        source["parent_codec"].identity(),
    )


@pytest.mark.cuda
def test_codec_rebind_keeps_all_declared_nondefault_constructor_fields(runtime_source):
    source = runtime_source
    result = materialize(source)
    before, after = source["parent_codec"].identity(), result.codec.identity()
    for name in ("seed", "modality_weights", "presence_weight", "key_constant", "max_buffer_bytes"):
        assert after[name] == before[name]
    assert before["seed"] == 73 and before["key_constant"] == 0.75
    assert before["modality_weights"] == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert after["input_specification"] == source["input_spec"].identity()
    assert after["input_contract_id"] == source["input_spec"].fingerprint()
    excluded = {"input_specification", "input_contract_id"}
    assert {k: v for k, v in before.items() if k not in excluded} == {
        k: v for k, v in after.items() if k not in excluded
    }


@pytest.mark.cuda
def test_materialization_preserves_rng_and_does_not_run_a_forward(runtime_source):
    cpu, cuda = torch.get_rng_state().clone(), torch.cuda.get_rng_state(0).clone()
    numpy_rng, python_rng = np.random.get_state(), random.getstate()
    forbidden = torch.nn.Module._call_impl.__code__

    def no_forward(frame, event, argument):
        if event == "call" and frame.f_code is forbidden:
            pytest.fail("La materialización ejecutó un forward")

    previous = sys.getprofile()
    try:
        sys.setprofile(no_forward)
        result = materialize(runtime_source, runtime_source["registry"].entries[-1])
        assert result.record()["rng_preserved"] is True
    finally:
        sys.setprofile(previous)
    assert torch.equal(cpu, torch.get_rng_state()) and torch.equal(
        cuda, torch.cuda.get_rng_state(0)
    )
    assert python_rng == random.getstate()
    for before, after in zip(numpy_rng, np.random.get_state(), strict=True):
        assert np.array_equal(before, after)


@pytest.mark.cuda
@pytest.mark.parametrize("fault", ["source", "dimensions"])
def test_phase_rebind_rejects_changes_beyond_view_before_constructing(runtime_source, fault):
    original = runtime_source["input_spec"]
    dimensions = dict(original.dimensions)
    if fault == "dimensions":
        dimensions["news"] += 1
    changed = FinancialInputSpec(
        source_sha256="f" * 64 if fault == "source" else original.source_sha256,
        view_sha256=original.view_sha256,
        representation=original.representation,
        dimensions=dimensions,
        input_policy=original.input_policy,
    )
    with pytest.raises(ValueError):
        materialize(runtime_source, input_spec=changed)


@pytest.mark.cuda
@pytest.mark.parametrize(
    "fault", ["seed", "action", "catalog", "operator", "reader-seed", "bridge-topology"]
)
def test_unregistered_scope_changes_are_rejected(runtime_source, fault):
    source, baseline = runtime_source, runtime_source["baseline"]
    if fault == "seed":
        candidate = replace(baseline, predictor=replace(baseline.predictor, seed=43))
    elif fault == "action":
        candidate = replace(baseline, action=MechanismAction(route="k2c"))
    elif fault == "catalog":
        candidate = replace(
            baseline, catalog=replace(baseline.catalog, budget=MechanismBudget(max_reader_steps=2))
        )
    elif fault == "operator":
        candidate = replace(
            baseline,
            predictor=replace(
                baseline.predictor,
                regional_fusion=replace(baseline.predictor.regional_fusion, summary=False),
            ),
        )
    elif fault == "reader-seed":
        candidate = replace(baseline, readout=replace(baseline.readout, seed=43))
    else:
        candidate = replace(
            baseline,
            readout=replace(
                baseline.readout,
                evidence_bridge=replace(baseline.readout.evidence_bridge, topology="self"),
            ),
        )
    registry = ProposalRegistry((baseline, candidate), baseline.configuration_id)
    before = source["parent"].identity(), source["parent_codec"].identity()
    with pytest.raises(ValueError):
        api().materialize_registered_candidate(
            registry,
            candidate.configuration_id,
            parent=source["parent"],
            parent_codec=source["parent_codec"],
            input_spec=source["input_spec"],
        )
    assert before == (source["parent"].identity(), source["parent_codec"].identity())


@pytest.mark.cuda
def test_unknown_id_wrong_parent_and_wrong_codec_are_rejected(runtime_source):
    source = runtime_source
    with pytest.raises(ValueError):
        api().materialize_registered_candidate(
            source["registry"],
            "f" * 64,
            parent=source["parent"],
            parent_codec=source["parent_codec"],
            input_spec=source["input_spec"],
        )
    with pytest.raises(ValueError):
        materialize(source, parent=None)
    wrong = FrozenEpisodeCodec(source["baseline"].predictor.inputs, seed=1)
    with pytest.raises(ValueError):
        materialize(source, parent_codec=wrong)
    with pytest.raises(ValueError):
        materialize(source, device="cpu")


@pytest.mark.cuda
def test_record_is_frozen_returns_copies_and_rechecks_effective_objects(runtime_source):
    result = materialize(runtime_source)
    expected = result.record()
    changed = result.record()
    changed["codec"]["seed"] = 0
    changed["tensors"].clear()
    assert result.record() == expected
    with pytest.raises(FrozenInstanceError):
        result.configuration_id = "f" * 64
    object.__setattr__(result, "effective_configuration", runtime_source["baseline"])
    with pytest.raises(ValueError):
        result.record()


@pytest.mark.cuda
@pytest.mark.parametrize("component", ["predictor", "readout", "codec"])
def test_record_rejects_tensor_or_codec_drift_without_silent_resealing(runtime_source, component):
    result = materialize(runtime_source)
    parameter_id = result.consumer.predictor._parameter_id
    if component == "codec":
        object.__setattr__(result.codec, "_presence_weight", 7.0)
    else:
        module = getattr(result.consumer, component)
        with torch.no_grad():
            next(module.parameters()).add_(0.125)
    with pytest.raises(ValueError):
        result.record()
    assert result.consumer.predictor._parameter_id == parameter_id
    runtime_source["parent"].verify(strong=True)
