"""Constructor de MARS-TITAN con ampliaciones: identidad, rechazos y paridad con el núcleo."""

import copy
import itertools
import json

import pytest
import torch
from test_financial_session import moment
from test_financial_session_associative import EVENTS, label, options
from test_financial_session_associative import four_flow_source as four_flow_source
from test_financial_session_associative import frozen_consumer as frozen_consumer
from test_financial_session_associative import module_backend as module_backend
from test_financial_session_associative import no_target_estimation as no_target_estimation
from test_financial_session_associative import shared_native as shared_native
from test_financial_session_controls import FLOWS, tensor_leaves

from mars_titan.memory import mars_titan_variant as api
from mars_titan.memory.associative_memory import MatureCorrection
from mars_titan.memory.financial_session import FinancialSession
from mars_titan.memory.retention_bank import RetentionConfig
from mars_titan.memory.write_policy import CompositeScoreConfig, MatureErrorConfig
from mars_titan.memory.write_scores import WriteScalers
from mars_titan.models.titans.episodic_readout import EpisodicReadout, EpisodicReadoutConfig
from mars_titan.models.titans.financial import FinancialConfig, FinancialPredictor
from mars_titan.models.titans.frozen_financial import FrozenFinancialConsumer
from mars_titan.models.titans.local_control import MACProjectionConfig

DELTA = dict(rule="delta", key="codec", rate=0.5, forgetting=0.25)
PROXIMAL = dict(rule="proximal", key="constant", rate=2.0, forgetting=0.1)


@pytest.fixture(scope="module")
def declaration():
    return api.load_declaration()


@pytest.fixture(scope="module")
def base(frozen_consumer):
    return api.core_identity(frozen_consumer.predictor, "fixture_chronological_recipe")


def valid_combinations():
    banks = [None, "m0_no_bank", "m1", "m2", "m3"]
    steps = [None, 2, 4]
    episodes = [None, "first_read"]
    corrections = [None, DELTA, PROXIMAL]
    for bank, k, episode, correction in itertools.product(banks, steps, episodes, corrections):
        if (k and not bank) or (episode and not k) or (correction and bank):
            continue
        yield {
            name: value
            for name, value in (
                ("episodic_bank", bank),
                ("refinements", k),
                ("refinement_episodes", episode),
                ("associative_memory", correction),
            )
            if value is not None
        }


def test_declaration_connects_exactly_the_components_of_the_builder(declaration, tmp_path):
    connected = [
        name for name, c in declaration["components"].items() if c["connection"] == "connected"
    ]
    assert tuple(connected) == api.CONNECTED
    for change in (
        lambda d: d["components"]["regime_context"].update(connection="connected"),
        lambda d: d["components"]["refinements"].update(connection="declared"),
        lambda d: d["components"]["episodic_bank"].update(enabled=True),
        lambda d: d.update(final_test_opened=True),
        # La definición declarada de M3 debe repetir la del código: pesos, escalas y huecos.
        lambda d: d["components"]["episodic_bank"]["m3"].update(weights=[0.5, 0.25, 0.25]),
        lambda d: d["components"]["episodic_bank"]["m3"].update(diversity="included"),
        lambda d: d["components"]["episodic_bank"].pop("m3"),
    ):
        document = copy.deepcopy(declaration)
        change(document)
        path = tmp_path / "declaration.json"
        path.write_text(json.dumps(document))
        with pytest.raises(ValueError):
            api.load_declaration(path)


def test_all_disabled_is_the_core_and_every_combination_has_its_own_identity(declaration, base):
    core = api.select_variant(declaration, {}, base=base)
    assert core.identity() == base
    assert (core.admission, core.readout_mode, core.refinements) == ("m0", None, 1)
    assert core.correction is None
    prints = {}
    for components in valid_combinations():
        variant = api.select_variant(declaration, components, base=base)
        prints.setdefault(variant.fingerprint(), []).append(components)
    assert len(prints) == len(list(valid_combinations())) == 23
    assert all(len(group) == 1 for group in prints.values())
    assert core.fingerprint() in prints


def test_identity_ignores_order_and_numeric_spelling(declaration, base):
    left = api.select_variant(
        declaration,
        dict(episodic_bank="m1", refinements=2, refinement_episodes="first_read"),
        base=base,
    )
    right = api.select_variant(
        declaration,
        dict(refinement_episodes="first_read", refinements=2, episodic_bank="m1"),
        base=base,
    )
    assert left.fingerprint() == right.fingerprint()
    one = api.select_variant(declaration, dict(associative_memory=dict(DELTA, rate=1)), base=base)
    other = api.select_variant(
        declaration, dict(associative_memory=dict(DELTA, rate=1.0)), base=base
    )
    assert one.fingerprint() == other.fingerprint()


def test_components_map_to_their_consumer_settings(declaration, base):
    m2 = api.select_variant(
        declaration,
        dict(episodic_bank="m2", refinements=4, refinement_episodes="first_read"),
        base=base,
    )
    assert (m2.admission, m2.readout_mode, m2.refinements, m2.episode_selection) == (
        "m2",
        "bank",
        4,
        "first_read",
    )
    settings = m2.session_options(capacity=8, seed=5)
    assert type(settings["retention"]) is MatureErrorConfig and "associative" not in settings
    config = m2.readout_config("a" * 64, hidden_size=32, neighbors=4, seed=9)
    assert (config.mode, config.refinements, config.episode_selection) == ("bank", 4, "first_read")
    m0 = api.select_variant(declaration, dict(episodic_bank="m0_no_bank"), base=base)
    assert (m0.admission, m0.readout_mode) == ("m0", "no_bank")
    assert m0.readout_config("a" * 64).mode == "no_bank"
    b6 = api.select_variant(declaration, dict(associative_memory=PROXIMAL), base=base)
    settings = b6.session_options(capacity=8, seed=73)
    assert type(settings["associative"]) is MatureCorrection
    assert (
        settings["associative"].key == "constant" and settings["associative"].memory.key_size == 1
    )
    assert type(settings["retention"]) is RetentionConfig and settings["admission"] == "m0"
    assert b6.readout_config("a" * 64) is None
    with pytest.raises(ValueError, match="combinación"):
        m2.readout_config("a" * 64, refinements=1)
    m3 = api.select_variant(declaration, dict(episodic_bank="m3"), base=base)
    assert (m3.admission, m3.readout_mode) == ("m3", "bank")
    scalers = WriteScalers(
        source_sha256="a" * 64,
        dataset_sha256="b" * 64,
        decision_start=1,
        decision_end=2,
        decisions=4,
        labels=4,
        filing_decisions=0,
        news_decisions=1,
        error_median=0.25,
        anomaly_median=1.0,
        filing_age_median=None,
        news_share=0.25,
    )
    settings = m3.session_options(capacity=8, seed=5, scalers=scalers)
    assert type(settings["retention"]) is CompositeScoreConfig
    assert settings["retention"].scalers is scalers and settings["admission"] == "m3"
    with pytest.raises(ValueError, match="escalas"):
        m3.session_options(capacity=8, seed=5)
    with pytest.raises(ValueError, match="M2 no usa escalas"):
        m2.session_options(capacity=8, seed=5, scalers=scalers)


@pytest.mark.parametrize(
    ("components", "message"),
    [
        (dict(episodic_bank="m4"), "no admite"),
        (dict(episodic_bank="readout_none_m0"), "omitiendo"),
        (dict(episodic_bank="m1", refinements=1), "omitiendo"),
        (dict(episodic_bank="m1", refinements=3), "no admite"),
        (dict(episodic_bank="m1", refinements=2.0), "no admite"),
        (dict(refinements=2), "necesita episodic_bank"),
        (dict(episodic_bank="m1", refinement_episodes="first_read"), "necesita refinements"),
        (dict(episodic_bank="m1", refinements=2, refinement_episodes="per_step"), "omitiendo"),
        (dict(episodic_bank="m1", associative_memory=DELTA), "sin banco"),
        (dict(associative_memory=dict(DELTA, key="label")), "codec o constant"),
        (dict(associative_memory=dict(DELTA, rule="ridge")), "permitidos"),
        (dict(associative_memory=dict(DELTA, rate=2.0)), "η ≤ 2 − λ"),
        (dict(cm_v1="bcm"), "no está declarado"),
    ],
)
def test_invalid_combinations_are_rejected_with_their_motive(
    declaration, base, components, message
):
    with pytest.raises(ValueError, match=message):
        api.select_variant(declaration, components, base=base)


@pytest.mark.parametrize(
    "name",
    [
        "document_selection",
        "regime_context",
        "information_view",
        "adaptive_refinements",
        "replay_schedule",
        "adapters",
    ],
)
def test_declared_components_are_rejected_with_their_pending_work(declaration, base, name):
    component = declaration["components"][name]
    value = component["allowed"][0] if component["allowed"] else "on"
    with pytest.raises(ValueError, match="declarado sin conexión") as raised:
        api.select_variant(declaration, {name: value}, base=base)
    assert component["pending"] in str(raised.value)


def test_base_must_be_the_selected_mac_online_core(declaration, base, four_flow_source):
    with pytest.raises(ValueError, match="mac_online"):
        api.select_variant(
            declaration,
            {},
            base=dict(base, configuration=dict(base["configuration"], variant="mac_frozen")),
        )
    with pytest.raises(ValueError, match="identidad de un núcleo"):
        api.select_variant(declaration, {}, base=dict(base, extra=1))
    variant = api.select_variant(declaration, {}, base=base)
    other = (
        FinancialPredictor(
            FinancialConfig(
                four_flow_source["spec"], variant="mac_online", hidden_size=32, seed=43
            ),
            dtype=torch.float64,
            device="cpu",
        )
        .eval()
        .requires_grad_(False)
    )
    with pytest.raises(ValueError, match="no es el núcleo"):
        variant.consumer(other)
    controlled = (
        FinancialPredictor(
            FinancialConfig(
                four_flow_source["spec"], variant="mac_online", hidden_size=32, seed=42
            ),
            local_control=MACProjectionConfig(mode="diagnostic", rank=1, grid_size=4, seed=91),
            dtype=torch.float64,
            device="cpu",
        )
        .eval()
        .requires_grad_(False)
    )
    controlled_base = api.core_identity(controlled, "fixture_chronological_recipe")
    with pytest.raises(ValueError, match="CM-v1"):
        api.select_variant(declaration, {}, base=controlled_base).consumer(controlled)


def readout(source, variant, seed=42):
    config = variant.readout_config(
        source["codec"].fingerprint(), hidden_size=32, neighbors=4, seed=seed
    )
    return EpisodicReadout(config, dtype=torch.float64, device="cpu").eval().requires_grad_(False)


def test_consumer_requires_the_reader_of_its_combination(
    declaration, base, frozen_consumer, four_flow_source
):
    predictor = frozen_consumer.predictor
    core = api.select_variant(declaration, {}, base=base)
    m1 = api.select_variant(declaration, dict(episodic_bank="m1"), base=base)
    fixed = api.select_variant(
        declaration,
        dict(episodic_bank="m1", refinements=2, refinement_episodes="first_read"),
        base=base,
    )
    with pytest.raises(ValueError, match="no lleva lector"):
        core.consumer(predictor, readout(four_flow_source, m1))
    with pytest.raises(ValueError, match="combinación"):
        m1.consumer(predictor)
    per_step = api.select_variant(declaration, dict(episodic_bank="m1", refinements=2), base=base)
    with pytest.raises(ValueError, match="combinación"):
        fixed.consumer(predictor, readout(four_flow_source, per_step))
    assert core.consumer(predictor).identity() == FrozenFinancialConsumer(predictor).identity()
    left = fixed.consumer(predictor, readout(four_flow_source, fixed)).identity()
    right = per_step.consumer(predictor, readout(four_flow_source, per_step)).identity()
    assert left["readout"] != right["readout"]


def session_run(native, source, settings, output):
    """Seis eventos con etiquetas fijas y el estado completo de la sesión al final."""
    rng = torch.random.get_rng_state().clone()
    session = FinancialSession(output, **settings)
    previous, emitted = [], []
    try:
        for index in EVENTS:
            at = moment(index)
            batches = [] if index == 129 else source["batches"][index]
            feedback = [native.Feedback(p.id, 0, at, label(p.asset, index)) for p in previous]
            kind = "settlement" if index == 129 else "decision"
            previous = session.step(batches, feedback, kind=kind, cutoff=at).predictions
            emitted.append([(p.id, p.asset, p.decision_at, p.value) for p in previous])
        bundle = session._bundle(session.snapshot()["state"])
        bank, episodes = session._bank(bundle["bank"])
        state = session._fast_store.gather(session.fast_state(), FLOWS)
        result = dict(
            identity=(session.contract_id, session.model_id),
            emitted=emitted,
            snapshot=session.snapshot(),
            bank=([r.id for r in bank.records()], episodes),
            pending=tensor_leaves(session._read(bundle["pending"], "pending")),
            fast=tensor_leaves(settings["consumer"].predictor.export_state_cpu(state)),
        )
    finally:
        session.close()
    assert torch.equal(rng, torch.random.get_rng_state())
    return result


def assert_identical(left, right):
    for name in ("identity", "emitted", "snapshot", "bank"):
        assert left[name] == right[name], name
    for name in ("pending", "fast"):
        assert left[name].keys() == right[name].keys()
        for key, value in left[name].items():
            assert torch.equal(value, right[name][key]), (name, key)


@pytest.mark.native_binding
def test_all_disabled_session_is_exactly_the_mac_online_session(
    declaration, base, shared_native, four_flow_source, frozen_consumer, tmp_path
):
    variant = api.select_variant(declaration, {}, base=base)
    built = options(
        shared_native,
        four_flow_source,
        variant.consumer(frozen_consumer.predictor),
        **variant.session_options(capacity=4, seed=73, frontier=1),
    )
    by_hand = options(
        shared_native, four_flow_source, FrozenFinancialConsumer(frozen_consumer.predictor)
    )
    assert set(built) == set(by_hand)
    left = session_run(shared_native, four_flow_source, built, tmp_path / "variant")
    right = session_run(shared_native, four_flow_source, by_hand, tmp_path / "core")
    assert_identical(left, right)
    assert left["bank"] == ([], {}) and len(left["emitted"][-1]) == 4


@pytest.mark.native_binding
def test_episodic_combination_matches_the_hand_built_consumer(
    declaration, base, shared_native, four_flow_source, frozen_consumer, tmp_path
):
    variant = api.select_variant(declaration, dict(episodic_bank="m1"), base=base)
    predictor = frozen_consumer.predictor
    built = options(
        shared_native,
        four_flow_source,
        variant.consumer(predictor, readout(four_flow_source, variant)),
        **variant.session_options(capacity=4, seed=73, frontier=1),
    )
    manual = (
        EpisodicReadout(
            EpisodicReadoutConfig(
                four_flow_source["codec"].fingerprint(), hidden_size=32, neighbors=4
            ),
            dtype=torch.float64,
            device="cpu",
        )
        .eval()
        .requires_grad_(False)
    )
    by_hand = options(
        shared_native,
        four_flow_source,
        FrozenFinancialConsumer(predictor, readout=manual),
        admission="m1",
    )
    left = session_run(shared_native, four_flow_source, built, tmp_path / "variant")
    right = session_run(shared_native, four_flow_source, by_hand, tmp_path / "manual")
    assert_identical(left, right)
    assert len(left["bank"][0]) == 4
