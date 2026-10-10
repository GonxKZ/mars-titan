"""B6 enrutada por régimen en el recorrido por ventanas y en `FinancialSession`.

Seis eventos de cinco flujos con ventanas de seis canales recorren los cuatro regímenes y
una cohorte sin clasificar. Cada ruta se deduce de la definición y el oráculo escribe cada
etiqueta en el compartimento del instante de su decisión, que aquí siempre difiere del de
su maduración. La sesión usa el fixture de cuatro flujos, cuyas ventanas son de cinco
canales, así que la regla se sustituye por una ruta fija por flujo y día que cambia entre
la decisión y la etiqueta. Ninguna prueba ajusta parámetros ni ejecuta pasos de optimizador.
"""

from datetime import UTC, datetime
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from test_financial_session_associative import four_flow_source as four_flow_source
from test_financial_session_associative import frozen_consumer as frozen_consumer
from test_financial_session_associative import module_backend as module_backend
from test_financial_session_associative import no_target_estimation as no_target_estimation
from test_financial_session_associative import options, trajectory
from test_financial_session_associative import shared_native as shared_native
from test_mars_titan_correction_parity import corrected_pass, emitted
from test_regimes import REGIME, SCENARIOS, scenario

from mars_titan.memory.associative_memory import (
    KEY_SIZES,
    AssociativeMemory,
    AssociativeMemoryConfig,
    MatureCorrection,
)
from mars_titan.memory.episodic_codec import FrozenEpisodeCodec
from mars_titan.memory.financial_observations import ObservationEvent
from mars_titan.memory.financial_session import FinancialPhase
from mars_titan.memory.regimes import CALENDAR_RULE, SLOTS, RegimeRule
from mars_titan.models.titans.financial import FinancialConfig, FinancialPredictor
from mars_titan.models.titans.financial_inputs import validated_cpu_batch
from mars_titan.training.financial_run import ChronologicalRecipe
from mars_titan.training.mars_titan_correction import CorrectionInference
from tests.models.test_price_presence_families import titans_specification
from tests.models.titans.test_financial_adapter import DIMENSIONS

FLOWS = tuple(f"US/A{index:03d}" for index in range(5))
ATS = tuple(int(datetime(2021, 3, day, 21, tzinfo=UTC).timestamp() * 1e6) for day in range(1, 7))
# Escenario de cada evento y ruta esperada. El quinto tiene 41 rendimientos y no se clasifica.
PLAN = ("calm_up", "turbulent_down", "calm_down", "turbulent_up", "unclassified", "calm_up")
ROUTES = (1, 4, 2, 3, 0, 1)
# Marzo de 2021 es el compartimento 3 del control de calendario.
CALENDAR_ROUTES = (3, 3, 3, 3, 0, 3)
SPEC = titans_specification(prices=6)
CALENDAR = RegimeRule(CALENDAR_RULE, min_assets=len(FLOWS))


def prices(name):
    if name == "unclassified":
        return scenario("calm_up", absent=tuple(range(22)))
    return scenario(name)


def raw(at, name, rows=slice(None), seed=0):
    generator = np.random.default_rng(seed)
    values = {
        modality: generator.normal(size=(len(FLOWS), width)).astype(np.float32)
        for modality, width in DIMENSIONS.items()
        if modality != "prices"
    }
    values["fundamentals"][:] = [0, 1, 0]
    values["macro"][:] = [0, 1, 0]
    values["prices"] = prices(name)
    flows = FLOWS[rows]
    values = {modality: value[rows] for modality, value in values.items()}
    return dict(
        inputs=values,
        presence=np.ones((len(flows), 5), dtype=np.bool_),
        sample_ids=[f"{flow}/{at}" for flow in flows],
        prediction_at=np.full(len(flows), at, dtype="datetime64[us]"),
        input_available_at=np.full(len(flows), at - 1, dtype="datetime64[us]"),
    )


def label(flow, index):
    return 0.125 * (int(flow[-1]) + 1) + 0.03125 * index


def events(plan=PLAN, *, split=False):
    result = []
    for index, (at, name) in enumerate(zip(ATS, plan, strict=True)):
        inputs = (
            (raw(at, name, slice(0, 2), index), raw(at, name, slice(2, None), index))
            if split
            else (raw(at, name, seed=index),)
        )
        labels = () if index == 0 else tuple((f, ATS[index - 1], label(f, index)) for f in FLOWS)
        result.append(ObservationEvent(at, inputs, labels, False))
    return result


@pytest.fixture(scope="module")
def parent():
    config = FinancialConfig(SPEC, variant="mac_online", hidden_size=32, layers=1, seed=42)
    predictor = FinancialPredictor(config, dtype=torch.float64, device="cpu")
    return predictor.eval().requires_grad_(False), FrozenEpisodeCodec(SPEC)


SOURCE = SimpleNamespace(
    phase=FinancialPhase("validation", ATS[0], ATS[0], ATS[-1] + 1, ATS[-1] + 1)
)


def routed(key, *, rate=0.25, routing=REGIME):
    config = AssociativeMemoryConfig(
        "proximal", key_size=KEY_SIZES[key], rate=rate, forgetting=0.01 if rate else 0.0
    )
    return MatureCorrection(config, key=key, routing=routing)


def run(parent, correction, sequence=None):
    predictor, codec = parent
    recipe = ChronologicalRecipe(block_rows=8)
    inference = CorrectionInference(predictor, recipe, correction, codec, audit=True)
    metrics = inference._pass(SOURCE, sequence or events())
    return inference, metrics


def oracle(parent, correction, inference, routes):
    """Emisiones y A esperadas con la ruta de cada decisión guardada hasta su etiqueta."""
    _, codec = parent
    core = {(f, at): value for kind, _, f, at, value in inference.audit if kind == "prediction"}
    memory, expected, pending, written = AssociativeMemory(correction.memory), {}, {}, 0
    for event, route in zip(events(), routes, strict=True):
        staged = sorted(
            (at, flow, value - core[flow, at], *pending.pop((flow, at)))
            for flow, at, value in event.labels
        )
        inputs = np.array(codec.encode(validated_cpu_batch(event.inputs[0], SPEC)).key_inputs)
        rows = np.full(len(FLOWS), route, dtype=np.int64)
        reads = memory.read(correction.keys(inputs, rows))[:, 0].tolist()
        for flow, key, read in zip(FLOWS, inputs, reads, strict=True):
            expected[flow, event.at] = core[flow, event.at] + read
            pending[flow, event.at] = (key, route)
        if staged:
            memory = memory.write(
                correction.feedback(
                    ids=list(range(written + 1, written + len(staged) + 1)),
                    decision_at=[item[0] for item in staged],
                    available_at=[event.at] * len(staged),
                    keys=correction.keys(
                        np.stack([item[3] for item in staged]),
                        np.array([item[4] for item in staged], dtype=np.int64),
                    ),
                    values=[item[2] for item in staged],
                ),
                cutoff=event.at,
            )
            written += len(staged)
    return expected, memory


@pytest.mark.parametrize(
    ("key", "routing", "routes"),
    [
        ("codec_by_regime", REGIME, ROUTES),
        ("regime", REGIME, ROUTES),
        ("codec_by_calendar", CALENDAR, CALENDAR_ROUTES),
    ],
)
def test_the_window_pass_writes_each_label_in_the_slot_of_its_decision(
    parent, key, routing, routes
):
    correction = routed(key, routing=routing)
    inference, metrics = run(parent, correction)
    expected, memory = oracle(parent, correction, inference, routes)
    assert {(f, at): value for f, at, value in emitted(inference)} == expected
    assert torch.equal(inference.memory.matrix, memory.matrix) and memory.writes == 25
    core = [entry[4] for entry in inference.audit if entry[0] == "prediction"]
    assert [value for *_, value in emitted(inference)][10:] != core[10:]
    summary = metrics["routing"]
    assert summary["rule"] == routing.name and summary["labels"] == list(routing.labels)
    assert [(a["at"], a["market"], a["route"]) for a in summary["assignments"]] == [
        (at, "US", route) for at, route in zip(ATS, routes, strict=True)
    ]
    reads = np.bincount(np.repeat(routes, len(FLOWS)), minlength=SLOTS).tolist()
    writes = np.bincount(np.repeat(routes[:-1], len(FLOWS)), minlength=SLOTS).tolist()
    assert summary["reads_by_route"] == reads and summary["writes_by_route"] == writes
    changes = sum(a != b for a, b in zip(routes, routes[1:], strict=False))
    assert summary["transitions_by_market"] == {"US": changes}


def test_the_regime_assignments_record_the_observable_state(parent):
    _, metrics = run(parent, routed("codec_by_regime"))
    assignments = metrics["routing"]["assignments"]
    for record, name in zip(assignments, PLAN, strict=True):
        state = REGIME.state("US", prices(name), record["at"])
        assert record == state.record(record["at"], REGIME.labels)
        assert record["label"] == name
    assert assignments[4]["returns"] == 41 and assignments[4]["trend"] is None
    assert set(PLAN) - {"unclassified"} == set(SCENARIOS)


def test_the_calendar_control_leaves_the_same_rows_unclassified(parent):
    _, regime = run(parent, routed("codec_by_regime"))
    _, calendar = run(parent, routed("codec_by_calendar", routing=CALENDAR))
    regime, calendar = regime["routing"]["assignments"], calendar["routing"]["assignments"]
    assert [a["route"] == 0 for a in regime] == [a["route"] == 0 for a in calendar]
    assert [a["route"] for a in calendar] == list(CALENDAR_ROUTES)


def test_routes_do_not_depend_on_how_the_event_is_split_in_blocks(parent):
    _, whole = run(parent, routed("codec_by_regime"))
    _, split = run(parent, routed("codec_by_regime"), events(split=True))
    assert split["routing"] == whole["routing"]


def test_later_windows_never_change_earlier_routes_or_emissions(parent):
    reference, base = run(parent, routed("codec_by_regime"))
    plan = (*PLAN[:-1], "turbulent_down")
    changed, later = run(parent, routed("codec_by_regime"), events(plan))
    assert later["routing"]["assignments"][:-1] == base["routing"]["assignments"][:-1]
    assert later["routing"]["assignments"][-1]["route"] == 4
    before = [entry for entry in emitted(reference) if entry[1] < ATS[-1]]
    assert [entry for entry in emitted(changed) if entry[1] < ATS[-1]] == before


def test_a_zero_rate_routed_correction_emits_the_core(parent):
    inference, metrics = run(parent, routed("codec_by_regime", rate=0.0))
    core = [entry[2:] for entry in inference.audit if entry[0] == "prediction"]
    assert emitted(inference) == core and len(core) == 30
    assert torch.count_nonzero(inference.memory.matrix) == 0
    assert metrics["routing"]["reads_by_route"] == [5, 10, 5, 5, 5]


def test_an_unrouted_correction_keeps_its_metrics(parent):
    plain = MatureCorrection(AssociativeMemoryConfig("proximal", rate=0.25, forgetting=0.01))
    _, metrics = run(parent, plain)
    assert "routing" not in metrics and metrics["associative_writes"] == 25


# Sesión frente a ventana con la ruta sustituida.


def daily_routes(self, prices, flow_ids, at):
    """Ruta por flujo y día: la etiqueta madura siempre en otro compartimento."""
    assert prices.shape == (len(flow_ids), 64, 5)
    day = at // 86_400_000_000
    return np.array([(day + int(f[-1])) % SLOTS for f in flow_ids], dtype=np.int64), []


@pytest.mark.native_binding
@pytest.mark.parametrize("key", ["codec_by_regime", "regime"])
def test_the_session_emits_what_the_window_emits_with_routes_and_recovery(
    shared_native, four_flow_source, frozen_consumer, tmp_path, monkeypatch, key
):
    monkeypatch.setattr(RegimeRule, "routes", daily_routes)
    correction = routed(key)
    session = trajectory(
        shared_native, four_flow_source, frozen_consumer, tmp_path / "plain", correction=correction
    )
    recovered = trajectory(
        shared_native,
        four_flow_source,
        frozen_consumer,
        tmp_path / "recovered",
        correction=correction,
        recover=True,
    )
    for left, right in zip(session["events"], recovered["events"], strict=True):
        assert left["predictions"] == right["predictions"]
        assert left["memory"]["matrix_sha256"] == right["memory"]["matrix_sha256"]
        assert torch.equal(left["core"], right["core"])
    assert session["snapshot"] == recovered["snapshot"]
    inference, _ = corrected_pass(four_flow_source, correction)
    expected = [(a, t, v) for event in session["events"] for _, a, t, v in event["predictions"]]
    assert emitted(inference) == expected and len(expected) == 20
    final = session["events"][-1]["memory"]
    assert torch.equal(inference.memory.matrix, final["matrix"])
    assert list(inference.memory.cursor) == final["cursor"] and final["writes"] == 16
    # Con la clave sin enrutar las emisiones serían otras desde que A tiene escrituras.
    plain = MatureCorrection(AssociativeMemoryConfig("proximal", rate=0.25, forgetting=0.01))
    unrouted, _ = corrected_pass(four_flow_source, plain)
    assert emitted(unrouted)[8:] != expected[8:]


@pytest.mark.native_binding
def test_the_session_rejects_pending_routes_that_do_not_match_its_key(
    shared_native, four_flow_source, frozen_consumer, tmp_path, monkeypatch
):
    from mars_titan.memory.financial_session import FinancialSession

    memory = AssociativeMemory(routed("codec_by_regime").memory).export()
    core, routes = torch.zeros(2, dtype=torch.float64), torch.tensor([1, 4])
    for index, (correction, value, message) in enumerate(
        [
            (routed("codec_by_regime"), dict(memory=memory, core=core), "rutas"),
            (
                routed("codec_by_regime"),
                dict(memory=memory, core=core, routes=routes.int()),
                "rutas",
            ),
            (routed("codec_by_regime"), dict(memory=memory, core=core, routes=routes + 1), "rutas"),
            (routed("codec_by_regime"), dict(memory=memory, core=core, routes=routes[:1]), "rutas"),
            (routed("codec_by_regime"), dict(memory=memory, core=core, routes=-routes), "rutas"),
            (
                MatureCorrection(AssociativeMemoryConfig("proximal")),
                dict(
                    memory=AssociativeMemory(AssociativeMemoryConfig("proximal")).export(),
                    core=core,
                )
                | dict(routes=routes),
                "rutas",
            ),
        ]
    ):
        settings = options(shared_native, four_flow_source, frozen_consumer, associative=correction)
        session = FinancialSession(tmp_path / str(index), **settings)
        try:
            monkeypatch.setattr(session, "_read", lambda reference, kind, value=value: value)
            with pytest.raises(ValueError, match=message):
                session._associative_state("staged")
        finally:
            session.close()
    session = FinancialSession(
        tmp_path / "valid",
        **options(
            shared_native, four_flow_source, frozen_consumer, associative=routed("codec_by_regime")
        ),
    )
    try:
        valid = dict(memory=memory, core=core, routes=routes)
        monkeypatch.setattr(session, "_read", lambda reference, kind: valid)
        assert torch.equal(session._associative_state("staged")[2], routes)
    finally:
        session.close()
