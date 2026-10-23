"""Registro finito de configuraciones reales, sin construir ni ejecutar modelos."""

import hashlib
import importlib
import importlib.util
import json
from dataclasses import FrozenInstanceError, replace

import pytest
from test_financial_adapter import specification

from mars_titan.models.titans.bridge_recurrence import BridgeRecurrenceConfig
from mars_titan.models.titans.episodic_readout import EpisodicReadoutConfig
from mars_titan.models.titans.evidence_bridge import EvidenceBridgeConfig
from mars_titan.models.titans.financial import FinancialConfig
from mars_titan.models.titans.mechanism_catalog import MechanismAction, MechanismCatalog
from mars_titan.models.titans.regional_execution import RegionalExecutionConfig
from mars_titan.models.titans.regional_fusion import RegionalFusionConfig


def api():
    try:
        found = importlib.util.find_spec("mars_titan.proposals.contracts")
    except ModuleNotFoundError:
        found = None
    assert found is not None, "Falta el contrato de registro de propuestas"
    return importlib.import_module("mars_titan.proposals.contracts")


def configuration(module, *, program="summary_at_end_v1", steps=1, seed=42):
    execution = RegionalExecutionConfig(program=program)
    predictor = FinancialConfig(
        specification(),
        variant="mac_online",
        hidden_size=32,
        seed=seed,
        head="quantile_head_v1",
        regional_fusion=RegionalFusionConfig(topology="self", routing="uniform"),
        regional_execution=execution,
    )
    bridge = EvidenceBridgeConfig(
        "c" * 64,
        hidden_size=32,
        cycles=0,
        recurrence=BridgeRecurrenceConfig(steps=steps, max_node_updates_per_row=456),
    )
    readout = EpisodicReadoutConfig("c" * 64, hidden_size=32, evidence_bridge=bridge)
    return module.RegisteredConfiguration(
        predictor=predictor,
        readout=readout,
        action=MechanismAction(),
        catalog=MechanismCatalog(
            reader_contract=bridge.identity()["kind"],
            regional_execution=execution,
        ),
        dtype="torch.float64",
    )


def test_contract_is_available_in_call():
    module = api()
    assert callable(getattr(module, "RegisteredConfiguration", None))
    assert callable(getattr(module, "ProposalRegistry", None))


def test_registry_distinguishes_real_programs_and_j_without_changing_k():
    module = api()
    baseline = configuration(module)
    interleaved = configuration(module, program="summary_interleaved_v1")
    j2 = configuration(module, steps=2)
    entries = (baseline, interleaved, j2)
    registry = module.ProposalRegistry(entries=entries, baseline_id=baseline.configuration_id)
    assert len({entry.configuration_id for entry in entries}) == 3
    for entry in entries:
        resolved = registry.resolve(entry.configuration_id)
        assert resolved.identity() == entry.identity()
        assert entry.predictor.refinements == entry.readout.refinements == 1
        assert len(entry.catalog.actions()) == 36
    assert interleaved.predictor.regional_execution.program == "summary_interleaved_v1"
    assert j2.readout.evidence_bridge.recurrence.steps == 2
    assert j2.readout.evidence_bridge.cycles == 0
    assert len(registry.registry_id) == 64


def test_identities_are_canonical_and_return_no_mutable_alias():
    module = api()
    entry = configuration(module)
    identity = entry.identity()
    expected = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    assert entry.configuration_id == expected
    saved = entry.identity()
    identity["predictor"]["seed"] = 99
    assert entry.identity() == saved
    with pytest.raises(FrozenInstanceError):
        entry.dtype = "torch.float32"


@pytest.mark.parametrize("fault", ("program", "reader", "k", "dtype"))
def test_incompatible_configurations_are_rejected_before_model_construction(fault):
    module = api()
    entry = configuration(module)
    changes = {
        "program": {"catalog": replace(entry.catalog, regional_execution=None)},
        "reader": {"catalog": replace(entry.catalog, reader_contract="legacy")},
        "k": {"readout": replace(entry.readout, refinements=2)},
        "dtype": {"dtype": "torch.float16"},
    }
    with pytest.raises(ValueError):
        replace(entry, **changes[fault])


def test_registry_rejects_duplicate_unknown_baseline_and_excess_entries():
    module = api()
    entries = tuple(configuration(module, seed=42 + index) for index in range(9))
    with pytest.raises(ValueError):
        module.ProposalRegistry(
            entries=(entries[0], entries[0]), baseline_id=entries[0].configuration_id
        )
    with pytest.raises(ValueError):
        module.ProposalRegistry(entries=(entries[0],), baseline_id="d" * 64)
    with pytest.raises(ValueError):
        module.ProposalRegistry(entries=entries, baseline_id=entries[0].configuration_id)
    registry = module.ProposalRegistry(entries=entries[:8], baseline_id=entries[0].configuration_id)
    assert registry.resolve(entries[7].configuration_id).identity() == entries[7].identity()
    with pytest.raises(ValueError):
        registry.resolve("d" * 64)


def test_registry_cannot_resolve_coherently_mutated_sealed_configs():
    module = api()
    entry = configuration(module)
    key = entry.configuration_id
    registry = module.ProposalRegistry(entries=(entry,), baseline_id=key)
    # El catálogo y el predictor comparten la configuración. Cambiar ambos no renueva el sello.
    object.__setattr__(entry.predictor.regional_execution, "program", "summary_interleaved_v1")
    with pytest.raises(ValueError):
        registry.resolve(key)


def test_calibration_is_not_an_unrecorded_component_of_the_recipe():
    module = api()
    entry = configuration(module)
    with pytest.raises(ValueError, match="calibrador"):
        replace(entry, action=MechanismAction(calibration="static"))


def test_plain_reference_without_reader_or_regional_program_remains_available():
    module = api()
    entry = module.RegisteredConfiguration(
        predictor=FinancialConfig(specification(), hidden_size=32),
        readout=None,
        action=MechanismAction(),
        catalog=MechanismCatalog(),
        dtype="torch.float32",
    )
    registry = module.ProposalRegistry((entry,), entry.configuration_id)
    assert registry.resolve(entry.configuration_id) is entry
    assert entry.identity()["readout"] is None
    saved = registry.identity()
    exposed = registry.identity()
    exposed["entries"].clear()
    assert registry.identity() == saved


def test_reader_and_predictor_dimensions_must_match_with_a_legacy_catalog():
    module = api()
    with pytest.raises(ValueError, match="dimensión"):
        module.RegisteredConfiguration(
            predictor=FinancialConfig(specification(), hidden_size=32),
            readout=EpisodicReadoutConfig("c" * 64, hidden_size=16),
            action=MechanismAction(),
            catalog=MechanismCatalog(),
            dtype="torch.float32",
        )


@pytest.mark.parametrize("field", ("predictor", "readout", "action", "catalog"))
def test_recipe_rejects_untyped_components(field):
    module = api()
    with pytest.raises(ValueError):
        replace(configuration(module), **{field: object()})


@pytest.mark.parametrize("entries", ((), [], (object(),)))
def test_registry_rejects_empty_mutable_or_untyped_entries(entries):
    with pytest.raises(ValueError):
        api().ProposalRegistry(entries, "d" * 64)


def test_registry_rejects_a_different_reference_after_sealing():
    module = api()
    first, second = configuration(module), configuration(module, steps=2)
    registry = module.ProposalRegistry((first, second), first.configuration_id)
    object.__setattr__(registry, "baseline_id", second.configuration_id)
    with pytest.raises(ValueError, match="sello"):
        registry.identity()


def test_registry_enforces_its_serialized_byte_budget(monkeypatch):
    module = api()
    entry = configuration(module)
    registry = module.ProposalRegistry((entry,), entry.configuration_id)
    size = len(json.dumps(registry.identity(), sort_keys=True, separators=(",", ":")).encode())
    monkeypatch.setattr(module, "MAX_REGISTRY_BYTES", size)
    assert (
        module.ProposalRegistry((entry,), entry.configuration_id).registry_id
        == registry.registry_id
    )
    monkeypatch.setattr(module, "MAX_REGISTRY_BYTES", size - 1)
    with pytest.raises(ValueError, match="MiB"):
        module.ProposalRegistry((entry,), entry.configuration_id)


@pytest.mark.parametrize("changed", ("regions", "entries"))
def test_equal_json_does_not_allow_invalid_mutable_component_types(changed):
    module = api()
    entry = configuration(module)
    key = entry.configuration_id
    registry = module.ProposalRegistry((entry,), key)
    if changed == "regions":
        bridge = entry.readout.evidence_bridge
        object.__setattr__(bridge, "regions", list(bridge.regions))
    else:
        object.__setattr__(registry, "entries", list(registry.entries))
    with pytest.raises(ValueError):
        registry.resolve(key)


@pytest.mark.parametrize("field", ("predictor", "readout", "action", "catalog"))
def test_replacing_a_typed_component_cannot_renew_its_seal(field):
    entry = configuration(api())
    sealed = entry._sealed
    object.__setattr__(entry, field, object())
    with pytest.raises(ValueError, match="cambiado"):
        entry.identity()
    assert entry._sealed == sealed
