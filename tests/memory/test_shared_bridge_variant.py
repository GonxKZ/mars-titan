"""Construcción identificada del puente común, sin otro banco ni ajustes de pesos."""

import copy
import json
from dataclasses import replace

import pytest
import torch
from test_financial_session_controls import four_flow_source as four_flow_source
from test_frozen_financial import frozen_backend as frozen_backend
from test_mars_titan_session_parity import SCALERS
from test_mars_titan_variant import assert_identical, options, session_run
from test_native_episode_backend import native as native

from mars_titan.memory import mars_titan_variant as api
from mars_titan.models.quantile_head import QUANTILE_HEAD
from mars_titan.models.titans.episodic_readout import EpisodicReadout, EpisodicReadoutConfig
from mars_titan.models.titans.evidence_bridge import EvidenceBridgeConfig
from mars_titan.models.titans.financial import FinancialConfig, FinancialPredictor
from mars_titan.models.titans.frozen_financial import FrozenFinancialConsumer

GEOMETRY = dict(
    regions=["view_1", "view_2", "view_3"], topology="ring", cycles=2, bridge_gain=0.25, seed=13
)
DEVICES = ("cpu", pytest.param("cuda:0", marks=pytest.mark.cuda))


@pytest.fixture(scope="module", params=DEVICES)
def predictor(request, four_flow_source):
    return (
        FinancialPredictor(
            FinancialConfig(
                four_flow_source["spec"], variant="mac_online", hidden_size=32, head=QUANTILE_HEAD
            ),
            device=request.param,
            dtype=torch.float64,
        )
        .eval()
        .requires_grad_(False)
    )


def selected(predictor, *, k=1, geometry=None, bank="m1", episodes="per_step"):
    components = dict(episodic_bank=bank, evidence_bridge=geometry or copy.deepcopy(GEOMETRY))
    if k != 1:
        components["refinements"] = k
    if episodes != "per_step":
        components["refinement_episodes"] = episodes
    return api.select_variant(
        api.load_declaration(),
        components,
        base=api.core_identity(predictor, "fixture_bridge_catalog"),
    )


def reader(config, predictor):
    return (
        EpisodicReadout(config, device=predictor.head.weight.device, dtype=torch.float64)
        .eval()
        .requires_grad_(False)
    )


@pytest.mark.parametrize("topology", ["self", "ring", "complete"])
@pytest.mark.parametrize("k", [1, 2, 4])
def test_catalog_builds_the_same_reader_and_consumer_as_explicit_configuration(
    predictor, four_flow_source, topology, k
):
    geometry = dict(GEOMETRY, topology=topology)
    variant = selected(predictor, geometry=geometry, k=k)
    common = dict(
        hidden_size=32, neighbors=2, temperature=0.5, max_batch=4, max_working_bytes=4 * 1024**2
    )
    codec = four_flow_source["codec"].fingerprint()
    built = variant.readout_config(codec, seed=43, **common)
    manual = EpisodicReadoutConfig(
        codec,
        refinements=k,
        seed=43,
        evidence_bridge=EvidenceBridgeConfig(
            codec, **dict(geometry, regions=tuple(geometry["regions"])), **common
        ),
        **common,
    )
    assert built.identity() == manual.identity()
    a, b = reader(built, predictor), reader(manual, predictor)
    assert (
        variant.consumer(predictor, a).identity()
        == FrozenFinancialConsumer(predictor, readout=b).identity()
    )
    assert variant.session_options(capacity=4, seed=73, frontier=1)["admission"] == "m1"


def test_bridge_identity_is_complete_normalized_and_detached_from_caller(predictor):
    geometry = dict(GEOMETRY, regions=["one"], cycles=0, bridge_gain=1)
    a = selected(predictor, geometry=geometry)
    geometry["regions"].append("two")
    b = selected(predictor, geometry=dict(GEOMETRY, regions=("one",), cycles=0, bridge_gain=1.0))
    assert a.fingerprint() == b.fingerprint()
    restored = api.select_variant(
        api.load_declaration(), json.loads(json.dumps(a.components)), base=a.base
    )
    assert restored.fingerprint() == a.fingerprint()
    for name, value in dict(
        regions=["two"], topology="complete", cycles=1, bridge_gain=0.5, seed=14
    ).items():
        assert (
            selected(
                predictor, geometry=dict(b.components["evidence_bridge"], **{name: value})
            ).fingerprint()
            != b.fingerprint()
        )
    legacy = api.select_variant(api.load_declaration(), dict(episodic_bank="m1"), base=a.base)
    assert legacy.identity() == dict(
        schema_version=1, variant=api.VARIANT, base=a.base, components=dict(episodic_bank="m1")
    )
    assert api.select_variant(api.load_declaration(), {}, base=a.base).identity() == a.base


@pytest.mark.parametrize(
    "changes",
    [
        dict(regions=[]),
        dict(regions=["x", "x"]),
        dict(regions=["x"] * 9),
        dict(regions="x"),
        dict(topology="unknown"),
        dict(cycles=True),
        dict(cycles=5),
        dict(bridge_gain=float("nan")),
        dict(bridge_gain=float("inf")),
        dict(bridge_gain=True),
        dict(bridge_gain=-0.1),
        dict(bridge_gain=1.1),
        dict(seed=True),
        dict(seed=-1),
        dict(seed=2**32),
        dict(hidden_size=32),
        dict(codec_id="a" * 64),
    ],
)
def test_invalid_bridge_geometry_is_rejected_at_selection(predictor, changes):
    with pytest.raises(ValueError):
        selected(predictor, geometry=dict(GEOMETRY, **changes))


def test_bridge_requires_the_episodic_component_and_cannot_be_smuggled_in(
    predictor, four_flow_source
):
    base = api.core_identity(predictor, "fixture_bridge_catalog")
    with pytest.raises(ValueError, match="necesita episodic_bank"):
        api.select_variant(api.load_declaration(), dict(evidence_bridge=GEOMETRY), base=base)
    declared = selected(predictor)
    legacy = api.select_variant(api.load_declaration(), dict(episodic_bank="m1"), base=base)
    plain = legacy.readout_config(four_flow_source["codec"].fingerprint(), hidden_size=32)
    bridged = declared.readout_config(four_flow_source["codec"].fingerprint(), hidden_size=32)
    with pytest.raises(ValueError, match="combinación"):
        legacy.consumer(predictor, reader(bridged, predictor))
    with pytest.raises(ValueError, match="combinación"):
        declared.consumer(predictor, reader(plain, predictor))
    for key, value in dict(
        regions=["other"], topology="self", cycles=1, bridge_gain=0.5, seed=14
    ).items():
        other = selected(predictor, geometry=dict(GEOMETRY, **{key: value}))
        with pytest.raises(ValueError, match="combinación"):
            declared.consumer(
                predictor,
                reader(
                    other.readout_config(four_flow_source["codec"].fingerprint(), hidden_size=32),
                    predictor,
                ),
            )


def test_bridge_checkpoint_and_no_bank_control_preserve_explicit_state(predictor, four_flow_source):
    variant = selected(predictor, bank="m0_no_bank", k=4, episodes="first_read")
    config = variant.readout_config(four_flow_source["codec"].fingerprint(), hidden_size=32)
    assert config.mode == "no_bank" and config.episode_selection == "first_read"
    before = torch.random.get_rng_state().clone()
    a, b = reader(config, predictor), reader(config, predictor)
    b.load_state_dict(a.state_dict())
    assert torch.equal(before, torch.random.get_rng_state())
    assert variant.consumer(predictor, a).identity() == variant.consumer(predictor, b).identity()
    assert variant.session_options(capacity=4, seed=73, frontier=1)["admission"] == "m0"


@pytest.mark.parametrize(
    "tamper",
    [
        "module_config",
        "module_absent",
        "metadata_type",
        "metadata_shape",
        "module_config_type",
        "metadata_shape_together",
    ],
)
def test_catalog_rejects_disagreement_between_reader_metadata_and_module(
    predictor, four_flow_source, tamper
):
    variant = selected(predictor)
    config = variant.readout_config(four_flow_source["codec"].fingerprint(), hidden_size=32)
    value = reader(config, predictor)
    if tamper == "module_config":
        value.evidence_bridge.config = replace(config.evidence_bridge, seed=99)
    elif tamper == "module_absent":
        value.evidence_bridge = None
    elif tamper == "metadata_type":
        object.__setattr__(value.config, "evidence_bridge", object())
    elif tamper == "module_config_type":
        value.evidence_bridge.config = object()
    else:
        object.__setattr__(
            value.config, "evidence_bridge", replace(config.evidence_bridge, hidden_size=64)
        )
        if tamper == "metadata_shape_together":
            value.evidence_bridge.config = value.config.evidence_bridge
    with pytest.raises(ValueError):
        variant.consumer(predictor, value)


def test_undeclared_physical_bridge_is_not_hidden_by_absent_metadata(predictor, four_flow_source):
    variant = selected(predictor)
    config = variant.readout_config(four_flow_source["codec"].fingerprint(), hidden_size=32)
    value = reader(config, predictor)
    object.__setattr__(value.config, "evidence_bridge", None)
    legacy = api.select_variant(api.load_declaration(), dict(episodic_bank="m1"), base=variant.base)
    with pytest.raises(ValueError):
        legacy.consumer(predictor, value)


def test_catalog_honours_a_declared_restriction_on_topology(predictor):
    declaration = api.load_declaration()
    declaration["components"]["evidence_bridge"]["allowed"] = ["self"]
    with pytest.raises(ValueError, match="topología"):
        api.select_variant(
            declaration,
            dict(episodic_bank="m1", evidence_bridge=GEOMETRY),
            base=api.core_identity(predictor, "fixture_bridge_catalog"),
        )


@pytest.mark.native("MARS_TITAN_EPISODIC_NATIVE")
@pytest.mark.parametrize("bank", ["m1", "m2", "m3"])
def test_catalog_native_session_matches_the_hand_built_bridge(
    native, predictor, four_flow_source, tmp_path, bank
):
    variant = selected(predictor, bank=bank, k=4, episodes="first_read")
    config = variant.readout_config(
        four_flow_source["codec"].fingerprint(), hidden_size=32, neighbors=4
    )
    session_settings = variant.session_options(
        capacity=8, seed=73, **(dict(scalers=SCALERS) if bank == "m3" else {})
    )
    built = options(
        native,
        four_flow_source,
        variant.consumer(predictor, reader(config, predictor)),
        **session_settings,
    )
    manual = options(
        native,
        four_flow_source,
        FrozenFinancialConsumer(predictor, readout=reader(config, predictor)),
        **session_settings,
    )
    left = session_run(native, four_flow_source, built, tmp_path / "variant")
    right = session_run(native, four_flow_source, manual, tmp_path / "manual")
    assert_identical(left, right)
    assert 0 < len(left["bank"][0]) <= 8
