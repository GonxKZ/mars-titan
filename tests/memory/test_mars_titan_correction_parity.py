"""La corrección B6 del recorrido por ventanas emite lo mismo que `FinancialSession`.

Los seis eventos de cuatro flujos se entregan a `FinancialSession` con la corrección
asociativa y, como eventos de observación equivalentes, a `CorrectionInference`. Coincidir bit
a bit en las 20 emisiones y en la matriz A final acredita el mismo orden de lectura y
escritura, los mismos identificadores y el mismo valor escrito. Con η = 0 la corrección
reproduce la inferencia cronológica del núcleo. Alterar etiquetas o entradas posteriores no
cambia ninguna emisión anterior.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from test_financial_session import moment
from test_financial_session_associative import CONSTANT, DELTA, options
from test_financial_session_associative import four_flow_source as four_flow_source
from test_financial_session_associative import module_backend as module_backend
from test_financial_session_associative import no_target_estimation as no_target_estimation
from test_financial_session_associative import shared_native as shared_native
from test_mars_titan_session_parity import PHASE, events
from test_mars_titan_variant import session_run

from mars_titan.memory.associative_memory import AssociativeMemoryConfig, MatureCorrection
from mars_titan.memory.financial_observations import ObservationEvent
from mars_titan.models.titans.financial import FinancialConfig, FinancialPredictor
from mars_titan.models.titans.frozen_financial import FrozenFinancialConsumer
from mars_titan.training.financial_run import ChronologicalInference, ChronologicalRecipe, _Pass
from mars_titan.training.mars_titan_correction import CorrectionInference

PROXIMAL = MatureCorrection(AssociativeMemoryConfig("proximal", rate=0.25, forgetting=0.01))
ZERO = MatureCorrection(AssociativeMemoryConfig("proximal", rate=0.0, forgetting=0.0))
RECIPE = ChronologicalRecipe(block_rows=4)
SOURCE = SimpleNamespace(phase=PHASE)


def core(source):
    config = FinancialConfig(source["spec"], variant="mac_online", hidden_size=32, seed=42)
    predictor = FinancialPredictor(config, dtype=torch.float64, device="cpu")
    return predictor.eval().requires_grad_(False)


def emitted(inference):
    return [entry[2:] for entry in inference.audit if entry[0] == "emitted"]


def corrected_pass(source, correction, sequence=None):
    inference = CorrectionInference(core(source), RECIPE, correction, source["codec"], audit=True)
    metrics = inference._pass(SOURCE, sequence or events(source))
    return inference, metrics


@pytest.mark.parametrize("correction", [DELTA, PROXIMAL, CONSTANT], ids=["delta", "prox", "bias"])
def test_window_pass_emits_exactly_what_the_session_emits(
    shared_native, four_flow_source, tmp_path, correction
):
    settings = options(
        shared_native,
        four_flow_source,
        FrozenFinancialConsumer(core(four_flow_source)),
        associative=correction,
    )
    session = session_run(shared_native, four_flow_source, settings, tmp_path / "session")
    inference, metrics = corrected_pass(four_flow_source, correction)
    expected = [(a, t, v) for event in session["emitted"] for _, a, t, v in event]
    assert emitted(inference) == expected and len(expected) == 20
    # Las emisiones dependen de A: sin la corrección cambiarían a partir del tercer evento.
    plain = [entry[2:] for entry in inference.audit if entry[0] == "prediction"]
    assert plain[:8] == expected[:8] and plain[8:] != expected[8:]
    assert metrics["labels"] == metrics["associative_writes"] == 16
    assert inference.memory.writes == session["snapshot"]["applied"] == 16


def test_final_matrix_matches_the_session_generation(shared_native, four_flow_source, tmp_path):
    from mars_titan.memory.financial_session import FinancialSession

    settings = options(
        shared_native,
        four_flow_source,
        FrozenFinancialConsumer(core(four_flow_source)),
        associative=PROXIMAL,
    )
    session_run(shared_native, four_flow_source, settings, tmp_path / "session")
    with FinancialSession(tmp_path / "session", **settings, resume=True) as restored:
        bundle = restored._bundle(restored.snapshot()["state"])
        memory, _ = restored._associative_state(bundle["associative"])
    inference, metrics = corrected_pass(four_flow_source, PROXIMAL)
    assert torch.equal(inference.memory.matrix, memory.matrix)
    assert inference.memory.cursor == memory.cursor
    assert metrics["associative_matrix_sha256"] == memory.export()["matrix_sha256"]


def core_pass(inference, sequence, rows=None):
    """Recorrido manual de la inferencia del núcleo con el mismo orden de evento."""
    state = _Pass(rows=rows)
    with torch.no_grad():
        for event in sequence:
            inference._labels(state, event, train=False)
            if event.inputs:
                inference._observe(state, SOURCE, event, differentiable=False)
    return state


def test_zero_rate_reproduces_the_core_chronological_inference(four_flow_source):
    plain = ChronologicalInference(core(four_flow_source), RECIPE, audit=True)
    expected = []
    core_pass(plain, events(four_flow_source), expected)
    inference = CorrectionInference(
        core(four_flow_source), RECIPE, ZERO, four_flow_source["codec"], audit=True
    )
    rows = []
    metrics = inference._pass(SOURCE, events(four_flow_source), rows=rows)
    assert rows == expected and len(rows) == 16
    assert emitted(inference) == [e[2:] for e in plain.audit if e[0] == "prediction"]
    assert metrics["associative_writes"] == 16
    assert torch.count_nonzero(inference.memory.matrix) == 0


def shifted(source, index, *, label=0.0, inputs=0.0):
    """Eventos con las etiquetas o las entradas del instante `index` desplazadas."""
    sequence = []
    for event in events(source):
        if event.at != moment(index):
            sequence.append(event)
            continue
        labels = tuple((flow, at, value + label) for flow, at, value in event.labels)
        raws = []
        for raw in event.inputs:
            changed = dict(raw, inputs=dict(raw["inputs"]))
            changed["inputs"]["prices"] = np.asarray(raw["inputs"]["prices"]) + np.float32(inputs)
            raws.append(changed)
        sequence.append(ObservationEvent(event.at, tuple(raws), labels, event.close_phase))
    return sequence


def by_instant(values):
    result = {}
    for flow, at, value in values:
        result.setdefault(at, []).append((flow, value))
    return [result[at] for at in sorted(result)]


@pytest.mark.parametrize("correction", [DELTA, PROXIMAL], ids=["delta", "prox"])
def test_future_labels_and_inputs_never_change_earlier_emissions(four_flow_source, correction):
    reference, _ = corrected_pass(four_flow_source, correction)
    base = by_instant(emitted(reference))
    # Las etiquetas que maduran en el tercer instante solo afectan a las emisiones posteriores.
    labels, _ = corrected_pass(
        four_flow_source, correction, shifted(four_flow_source, 127, label=1)
    )
    changed = by_instant(emitted(labels))
    assert changed[:3] == base[:3] and changed[3:] != base[3:]
    # Las entradas del último instante no cambian ninguna emisión anterior.
    inputs, _ = corrected_pass(
        four_flow_source, correction, shifted(four_flow_source, 130, inputs=0.5)
    )
    later = by_instant(emitted(inputs))
    assert later[:-1] == base[:-1] and later[-1] != base[-1]


def test_a_cohort_of_two_decision_instants_is_written_in_canonical_order(
    four_flow_source, monkeypatch
):
    """Las etiquetas de dos instantes que maduran juntas se escriben por decisión y flujo.

    La regla delta escribe fila a fila, así que el orden cambia A. Las etiquetas del primer
    evento con resultados se retrasan al siguiente y llegan en orden inverso.
    """
    sequence = events(four_flow_source)
    first, second = [index for index, event in enumerate(sequence) if event.labels][:2]
    merged = tuple(reversed(sequence[first].labels + sequence[second].labels))
    sequence[first] = ObservationEvent(
        sequence[first].at, sequence[first].inputs, (), sequence[first].close_phase
    )
    sequence[second] = ObservationEvent(
        sequence[second].at, sequence[second].inputs, merged, sequence[second].close_phase
    )
    calls, original = [], MatureCorrection.feedback

    def spy(self, **values):
        calls.append(values)
        return original(self, **values)

    monkeypatch.setattr(MatureCorrection, "feedback", spy)
    inference, _ = corrected_pass(four_flow_source, DELTA, sequence)
    core = {entry[2:4]: entry[4] for entry in inference.audit if entry[0] == "prediction"}
    expected = sorted((at, flow, value - core[flow, at]) for flow, at, value in merged)
    (written,) = [call for call in calls if len(call["ids"]) == len(merged) == 8]
    assert written["decision_at"] == [at for at, _, _ in expected]
    assert written["values"] == [value for _, _, value in expected]
    assert written["ids"] == list(range(written["ids"][0], written["ids"][0] + 8))


def test_correction_never_reaches_the_parent_or_its_fast_state(four_flow_source):
    predictor = core(four_flow_source)
    tensors = lambda: dict((*predictor.named_parameters(), *predictor.named_buffers()))  # noqa: E731
    before = {name: value.clone() for name, value in tensors().items()}
    plain = ChronologicalInference(predictor, RECIPE, audit=True)
    inference = CorrectionInference(
        predictor, RECIPE, PROXIMAL, four_flow_source["codec"], audit=True
    )
    core_pass(plain, events(four_flow_source))
    inference._pass(SOURCE, events(four_flow_source))
    # La predicción del núcleo y el avance de su memoria rápida no ven la corrección.
    core_plain = [e[2:] for e in plain.audit if e[0] == "prediction"]
    core_corrected = [e[2:] for e in inference.audit if e[0] == "prediction"]
    assert core_plain == core_corrected
    assert emitted(inference) != core_corrected
    after = tensors()
    assert all(torch.equal(before[name], after[name]) for name in before)
    assert all(not p.requires_grad and p.grad is None for p in predictor.parameters())


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda p, c: (p.train(), c), "congelado"),
        (lambda p, c: (p.requires_grad_(True), c), "congelado"),
        (lambda p, c: (p, c.memory), "corrección"),
    ],
)
def test_inference_rejects_a_trainable_parent_or_an_unidentified_correction(
    four_flow_source, change, message
):
    predictor, correction = change(core(four_flow_source), PROXIMAL)
    with pytest.raises(ValueError, match=message):
        CorrectionInference(predictor, RECIPE, correction, four_flow_source["codec"])
