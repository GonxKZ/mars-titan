"""Memoria por áreas de B1 y su control de Nadaraya-Watson en el recorrido por ventanas.

Los seis eventos de cuatro flujos se recorren con `CorrectionInference`. Una referencia
independiente parte de las predicciones del núcleo y de las claves del codec y repite la
recurrencia de cada evento: leer con el estado anterior, emitir núcleo más lectura y escribir
después las etiquetas que maduran en él. Las comprobaciones de sesión frente a ventana con
`FinancialSession` necesitan el enlace episódico nativo y siguen el patrón de B6.
"""

import numpy as np
import pytest
import torch
from test_area_memory import proximal_step, reference_features
from test_financial_session_associative import four_flow_source as four_flow_source
from test_financial_session_associative import module_backend as module_backend
from test_financial_session_associative import no_target_estimation as no_target_estimation
from test_kernel_memory import reference as kernel_reference
from test_mars_titan_correction_parity import (
    RECIPE,
    SOURCE,
    by_instant,
    core,
    core_pass,
    corrected_pass,
    emitted,
    shifted,
)
from test_mars_titan_session_parity import events

from mars_titan.memory.area_memory import AreaLayout, AreaMemory
from mars_titan.memory.area_query_bridge import AreaQueryBridgeConfig
from mars_titan.memory.associative_memory import AssociativeMemoryConfig, MatureCorrection
from mars_titan.memory.kernel_memory import KernelMemory, KernelMemoryConfig
from mars_titan.models.titans.financial_inputs import validated_cpu_batch
from mars_titan.training.financial_run import ChronologicalInference
from mars_titan.training.mars_titan_correction import CorrectionInference

PROXIMAL = AssociativeMemoryConfig("proximal", rate=0.25, forgetting=0.01)
LAYOUT = AreaLayout(areas=3, shifts=1, seed=5)
AREAS = MatureCorrection(PROXIMAL, areas=LAYOUT)
SHUFFLED = MatureCorrection(
    PROXIMAL, areas=AreaLayout(areas=3, seed=5, labels="shuffled_within_cohort")
)
KERNEL = MatureCorrection(KernelMemoryConfig(bandwidth=0.3, forgetting=0.05, capacity=64))
BRIDGED = MatureCorrection(PROXIMAL, areas=LAYOUT, query_bridge=AreaQueryBridgeConfig())


def decisions(source, event):
    """Decisiones del evento y sus claves FP64 normalizadas, como las calcula la ventana."""
    rows, inputs = [], []
    for raw in event.inputs:
        cpu = validated_cpu_batch(raw, source["spec"])
        rows.extend(zip(cpu.flow_ids, cpu.prediction_at, strict=True))
        inputs.append(source["codec"].encode(cpu).key_inputs)
    return rows, AREAS.keys(np.concatenate(inputs))


def reference_emissions(source, read, write):
    """Recurrencia de la ventana escrita aparte: `read(keys)` y `write(keys, values)`."""
    plain = ChronologicalInference(core(source), RECIPE, audit=True)
    core_pass(plain, events(source))
    predicted = {(e[2], e[3]): e[4] for e in plain.audit if e[0] == "prediction"}
    pending, result = {}, []
    for event in events(source):
        staged = sorted(
            (at, flow, value - predicted[flow, at], pending.pop((flow, at)))
            for flow, at, value in event.labels
        )
        if event.inputs:
            rows, keys = decisions(source, event)
            corrections = read(keys)[:, 0].tolist()
            for (flow, at), key, correction in zip(rows, keys, corrections, strict=True):
                pending[flow, at] = key
                result.append((flow, at, predicted[flow, at] + correction))
        if staged:
            write(
                torch.stack([key for *_, key in staged]),
                torch.tensor([value for _, _, value, _ in staged], dtype=torch.float64),
            )
    return result


def area_reference(source, layout, config):
    matrices = [
        torch.zeros((layout.feature_size(64), 1), dtype=torch.float64) for _ in range(layout.areas)
    ]

    def read(keys):
        parts = reference_features(keys, layout)
        return torch.stack([part @ m for part, m in zip(parts, matrices, strict=True)]).mean(0)

    def write(keys, values):
        for index, part in enumerate(reference_features(keys, layout)):
            matrices[index] = proximal_step(matrices[index], part, values, config)

    return reference_emissions(source, read, write)


def kernel_reference_emissions(source, config):
    episodes, count = [], [0]

    def read(keys):
        if not episodes:
            return torch.zeros((keys.shape[0], 1), dtype=torch.float64)
        return kernel_reference(keys, episodes, count[0], config)

    def write(keys, values):
        count[0] += 1
        episodes.extend(
            (key, value, count[0])
            for key, value in zip(keys.tolist(), values.tolist(), strict=True)
        )

    return reference_emissions(source, read, write)


def test_area_emissions_follow_the_independent_recurrence(four_flow_source):
    inference, metrics = corrected_pass(four_flow_source, AREAS)
    actual, expected = emitted(inference), area_reference(four_flow_source, LAYOUT, PROXIMAL)
    assert len(actual) == len(expected) == 20
    assert [row[:2] for row in actual] == [row[:2] for row in expected]
    torch.testing.assert_close(
        torch.tensor([row[2] for row in actual], dtype=torch.float64),
        torch.tensor([row[2] for row in expected], dtype=torch.float64),
        rtol=0,
        atol=1e-13,
    )
    assert type(inference.memory) is AreaMemory
    assert metrics["labels"] == metrics["associative_writes"] == 16
    assert metrics["associative_matrix_sha256"] == inference.memory.export()["matrix_sha256"]
    # Las ocho primeras emisiones son anteriores a la primera escritura y valen el núcleo.
    plain = [entry[2:] for entry in inference.audit if entry[0] == "prediction"]
    assert plain[:8] == actual[:8] and plain[8:] != actual[8:]


def test_kernel_emissions_follow_the_nadaraya_watson_recurrence(four_flow_source):
    inference, metrics = corrected_pass(four_flow_source, KERNEL)
    actual = emitted(inference)
    expected = kernel_reference_emissions(four_flow_source, KERNEL.memory)
    assert [row[:2] for row in actual] == [row[:2] for row in expected]
    torch.testing.assert_close(
        torch.tensor([row[2] for row in actual], dtype=torch.float64),
        torch.tensor([row[2] for row in expected], dtype=torch.float64),
        rtol=1e-13,
        atol=1e-15,
    )
    assert type(inference.memory) is KernelMemory and inference.memory.size == 16
    assert metrics["associative_writes"] == 16 and inference.memory.cohorts == 4


def test_zero_rate_areas_reproduce_the_core_inference(four_flow_source):
    plain = ChronologicalInference(core(four_flow_source), RECIPE, audit=True)
    expected = []
    core_pass(plain, events(four_flow_source), expected)
    zero = MatureCorrection(AssociativeMemoryConfig("proximal", rate=0.0), areas=LAYOUT)
    inference = CorrectionInference(
        core(four_flow_source), RECIPE, zero, four_flow_source["codec"], audit=True
    )
    rows = []
    inference._pass(SOURCE, events(four_flow_source), rows=rows)
    assert rows == expected and len(rows) == 16
    assert emitted(inference) == [e[2:] for e in plain.audit if e[0] == "prediction"]
    assert torch.count_nonzero(inference.memory.matrix) == 0


def test_shuffled_labels_only_change_emissions_after_the_first_write(four_flow_source):
    mature, _ = corrected_pass(four_flow_source, AREAS)
    shuffled, metrics = corrected_pass(four_flow_source, SHUFFLED)
    base, control = by_instant(emitted(mature)), by_instant(emitted(shuffled))
    assert control[:2] == base[:2] and control[2:] != base[2:]
    assert metrics["associative_writes"] == 16


@pytest.mark.parametrize("correction", [AREAS, KERNEL, BRIDGED], ids=["areas", "kernel", "bridge"])
def test_future_labels_and_inputs_never_change_earlier_emissions(four_flow_source, correction):
    reference, _ = corrected_pass(four_flow_source, correction)
    base = by_instant(emitted(reference))
    labels, _ = corrected_pass(
        four_flow_source, correction, shifted(four_flow_source, 127, label=1)
    )
    changed = by_instant(emitted(labels))
    assert changed[:3] == base[:3] and changed[3:] != base[3:]
    inputs, _ = corrected_pass(
        four_flow_source, correction, shifted(four_flow_source, 130, inputs=0.5)
    )
    later = by_instant(emitted(inputs))
    assert later[:-1] == base[:-1] and later[-1] != base[-1]


@pytest.mark.parametrize("correction", [AREAS, KERNEL, BRIDGED], ids=["areas", "kernel", "bridge"])
def test_the_correction_never_reaches_the_parent(four_flow_source, correction):
    predictor = core(four_flow_source)
    tensors = lambda: dict((*predictor.named_parameters(), *predictor.named_buffers()))  # noqa: E731
    before = {name: value.clone() for name, value in tensors().items()}
    plain = ChronologicalInference(predictor, RECIPE, audit=True)
    core_pass(plain, events(four_flow_source))
    inference = CorrectionInference(
        predictor, RECIPE, correction, four_flow_source["codec"], audit=True
    )
    inference._pass(SOURCE, events(four_flow_source))
    core_plain = [e[2:] for e in plain.audit if e[0] == "prediction"]
    assert core_plain == [e[2:] for e in inference.audit if e[0] == "prediction"]
    after = tensors()
    assert set(after) == set(before)
    assert all(torch.equal(before[name], after[name]) for name in before)
    assert all(not p.requires_grad and p.grad is None for p in predictor.parameters())


def test_query_bridge_emissions_follow_independent_synchronous_maps(four_flow_source):
    width = LAYOUT.feature_size(64)
    values = [torch.zeros((width, 1), dtype=torch.float64) for _ in range(LAYOUT.areas)]
    maps = [torch.zeros((width, width), dtype=torch.float64) for _ in values]

    def read(keys):
        parts = reference_features(keys, LAYOUT)
        queries = torch.stack(parts)
        queries = torch.stack(
            [
                0.75 * queries[b]
                + 0.25 * queries[(b - 1) % len(values)] @ maps[(b - 1) % len(values)]
                for b in range(len(values))
            ]
        )
        norms = torch.linalg.vector_norm(queries, dim=-1, keepdim=True)
        queries = queries / torch.where(norms > 0, norms, 1)
        return torch.stack([q @ m for q, m in zip(queries, values, strict=True)]).mean(0)

    def write(keys, labels):
        parts = reference_features(keys, LAYOUT)
        rows = keys.shape[0]
        for a, part in enumerate(parts):
            values[a] = proximal_step(values[a], part, labels, PROXIMAL)
            gram = torch.eye(width, dtype=torch.float64) + PROXIMAL.rate * part.T @ part / rows
            rhs = (
                PROXIMAL.retention * maps[a]
                + PROXIMAL.rate * part.T @ parts[(a + 1) % len(values)] / rows
            )
            maps[a] = torch.linalg.solve(gram, rhs)

    expected = reference_emissions(four_flow_source, read, write)
    inference, metrics = corrected_pass(four_flow_source, BRIDGED)
    actual = emitted(inference)
    assert [row[:2] for row in actual] == [row[:2] for row in expected]
    torch.testing.assert_close(
        torch.tensor([row[2] for row in actual], dtype=torch.float64),
        torch.tensor([row[2] for row in expected], dtype=torch.float64),
        rtol=0,
        atol=1e-13,
    )
    assert metrics["associative_writes"] == 16
