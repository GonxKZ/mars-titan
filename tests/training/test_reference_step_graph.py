"""Paso de ajuste con CUDA Graphs frente a la ruta eager, bit a bit y sin optimizador.

Cada prueba recorre la misma secuencia de lotes con dos copias del mismo modelo: una con
la ruta eager de `reference_run` y otra con `ReferenceStepGraph`. Entre lotes los pesos de
ambas copias se sustituyen en su sitio por los de otra inicialización, sin optimizador,
para comprobar que el grafo lee los pesos vigentes. El dropout se activa para comprobar
que la captura consume el generador igual que la ruta eager.
"""

import numpy as np
import pytest
import torch

from mars_titan.models.baselines.multimodal import (
    MODALITIES,
    PRESENCE_FUSION,
    SCALAR_HEAD,
    STRICT_FUSION,
    MultimodalReference,
)
from mars_titan.models.quantile_head import QUANTILE_HEAD
from mars_titan.training import reference_run
from mars_titan.training.reference_step_graph import (
    GRAPH_KINDS,
    LOSS_MESSAGE,
    ReferenceStepGraph,
)

DIMENSIONS = dict(prices=5, news=8, charts=8, fundamentals=6, macro=7)
CONTEXT, BATCH = 16, 8
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="sin dispositivo CUDA")


@pytest.fixture
def deterministic(monkeypatch):
    previous = torch.are_deterministic_algorithms_enabled()
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    yield
    torch.use_deterministic_algorithms(previous)


def architecture(kind):
    value = dict(hidden_size=32, layers=2, dropout=0.1)
    if kind == "transformer":
        value["transformer"] = dict(heads=2, feedforward_multiplier=2)
    return value


def build(kind, *, seed, quantiles, masked):
    torch.manual_seed(seed)
    return MultimodalReference(
        kind,
        DIMENSIONS,
        context=CONTEXT,
        mask_fusion=PRESENCE_FUSION if masked else STRICT_FUSION,
        head=QUANTILE_HEAD if quantiles else SCALAR_HEAD,
        **architecture(kind),
    ).to("cuda:0")


def batch(rows, generator, *, masked):
    inputs = {
        name: generator.standard_normal(
            (rows, CONTEXT, size) if name == "prices" else (rows, size)
        ).astype(np.float32)
        for name, size in DIMENSIONS.items()
    }
    result = dict(inputs=inputs, target=generator.standard_normal(rows).astype(np.float32))
    if masked:
        presence = generator.random((rows, len(MODALITIES))) > 0.3
        presence[:, [0, 2]] = True
        result["presence"] = presence
    return result


def walk(model, batches, replacements, case, *, graphed, weighted):
    """Recorrer los lotes como el bucle de `reference_run`, sin recorte ni optimizador."""
    device = torch.device("cuda:0")
    loss = reference_run._training_loss(case, "head" in case)
    graph = ReferenceStepGraph(model, loss, BATCH) if graphed else None
    torch.cuda.manual_seed(7)
    records = []
    for item, replacement in zip(batches, replacements, strict=True):
        model.load_state_dict(replacement)
        target = torch.from_numpy(item["target"]).to(device, dtype=torch.float32)
        rows = None
        if weighted:
            rows = torch.linspace(0.5, 1.5, len(item["target"]), device=device)
        if graph is not None and graph.admits(item):
            prediction = graph.step(item, target, rows)
        else:
            model.zero_grad(set_to_none=True)
            emitted = reference_run._forward(model, item, device)
            prediction, value = loss(emitted, target, rows)
            if not torch.isfinite(value).item():
                raise ValueError(LOSS_MESSAGE)
            value.backward()
        records.append(
            (
                prediction.detach().clone(),
                [parameter.grad.detach().clone() for parameter in model.parameters()],
            )
        )
    return records, torch.cuda.get_rng_state(device), graph


def sequence(kind, *, quantiles, masked):
    generator = np.random.default_rng(3)
    sizes = (BATCH, BATCH, 5, BATCH, BATCH, 3)
    batches = [batch(rows, generator, masked=masked) for rows in sizes]
    replacements = [
        build(kind, seed=100 + index, quantiles=quantiles, masked=masked).state_dict()
        for index in range(len(sizes))
    ]
    return batches, replacements


def case_for(kind, *, quantiles):
    case = dict(kind=kind, loss="mse", huber_delta=0.01)
    if quantiles:
        case.update(loss="pinball", head=QUANTILE_HEAD)
    return case


@cuda
@pytest.mark.parametrize("kind", GRAPH_KINDS)
@pytest.mark.parametrize(
    ("quantiles", "masked", "weighted"), [(True, True, False), (False, False, True)]
)
def test_graph_steps_match_the_eager_steps_bit_for_bit(
    deterministic, kind, quantiles, masked, weighted
):
    batches, replacements = sequence(kind, quantiles=quantiles, masked=masked)
    case = case_for(kind, quantiles=quantiles)
    options = dict(graphed=False, weighted=weighted)
    eager, eager_rng, _ = walk(
        build(kind, seed=1, quantiles=quantiles, masked=masked),
        batches,
        replacements,
        case,
        **options,
    )
    options["graphed"] = True
    graphed, graph_rng, graph = walk(
        build(kind, seed=1, quantiles=quantiles, masked=masked),
        batches,
        replacements,
        case,
        **options,
    )
    # Cuatro lotes completos repiten el grafo y los dos parciales usan la ruta eager.
    assert graph.replays == 4
    assert torch.equal(eager_rng, graph_rng)
    for index, ((eager_out, eager_grads), (graph_out, graph_grads)) in enumerate(
        zip(eager, graphed, strict=True)
    ):
        assert torch.equal(eager_out, graph_out), index
        for left, right in zip(eager_grads, graph_grads, strict=True):
            assert torch.equal(left, right), index


@cuda
def test_dropout_changes_between_replays_like_eager(deterministic):
    kind = "dlinear"
    batches, _ = sequence(kind, quantiles=True, masked=True)
    batches = [batches[0]] * 3
    model = build(kind, seed=1, quantiles=True, masked=True)
    replacements = [model.state_dict()] * 3
    records, _, _ = walk(
        model, batches, replacements, case_for(kind, quantiles=True), graphed=True, weighted=False
    )
    assert not torch.equal(records[0][0], records[1][0])
    assert not torch.equal(records[1][0], records[2][0])


@cuda
@pytest.mark.parametrize(
    ("kind", "modality", "expected"),
    [
        ("transformer", "prices", "Los precios contienen valores no finitos"),
        ("transformer", "news", "Las modalidades contienen valores no finitos"),
        ("gru", "prices", LOSS_MESSAGE),
        ("dlinear", "macro", LOSS_MESSAGE),
    ],
)
def test_nonfinite_batches_raise_the_eager_message_before_any_update(
    deterministic, kind, modality, expected
):
    batches, replacements = sequence(kind, quantiles=True, masked=True)
    poisoned = dict(batches[1], inputs=dict(batches[1]["inputs"]))
    poisoned["inputs"][modality] = poisoned["inputs"][modality].copy()
    poisoned["inputs"][modality].reshape(-1)[0] = np.nan
    case = case_for(kind, quantiles=True)
    for graphed in (False, True):
        model = build(kind, seed=1, quantiles=True, masked=True)
        weights = [value.detach().clone() for value in model.parameters()]
        with pytest.raises(ValueError, match=expected):
            walk(
                model,
                [batches[0], poisoned],
                [model.state_dict()] * 2,
                case,
                graphed=graphed,
                weighted=False,
            )
        assert all(torch.equal(a, b) for a, b in zip(weights, model.parameters(), strict=True))


def test_only_full_batches_with_the_captured_shape_replay():
    graph = ReferenceStepGraph(torch.nn.Linear(1, 1), None, BATCH)
    generator = np.random.default_rng(0)
    full = batch(BATCH, generator, masked=True)
    assert graph.admits(full)
    assert not graph.admits(batch(BATCH - 1, generator, masked=True))
    graph.signature = (("other", None),)
    assert not graph.admits(full)
    with pytest.raises(ValueError, match="entero positivo"):
        ReferenceStepGraph(torch.nn.Linear(1, 1), None, 0)


@pytest.mark.parametrize(
    "change",
    [
        dict(cuda_graphs=False),
        dict(cuda_graphs="yes"),
        dict(cuda_graphs=True, architecture=None),
    ],
)
def test_the_graph_option_needs_true_and_a_scientific_architecture(monkeypatch, change):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    case = dict(
        kind="gru",
        loss="mse",
        learning_rate=1e-3,
        seed=42,
        epochs=1,
        huber_delta=0.01,
        architecture=architecture("gru"),
    )
    reference_run._options(dict(case, cuda_graphs=True), BATCH, 60, 0)
    changed = dict(case, **change)
    if changed.get("architecture") is None:
        changed.pop("architecture")
    with pytest.raises(ValueError, match="CUDA Graphs"):
        reference_run._options(changed, BATCH, 60, 0)


def test_the_graph_step_enters_the_code_identity_only_when_declared(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *_: "sin dispositivo")
    plain = reference_run.scientific_identity(kind="gru")["code"]
    graphed = reference_run.scientific_identity(kind="gru", graphs=True)["code"]
    assert "training/reference_step_graph.py" not in plain
    assert set(graphed) - set(plain) == {"training/reference_step_graph.py"}
