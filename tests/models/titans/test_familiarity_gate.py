"""R3 de #786: la puerta de familiaridad del lector episódico y su control constante."""

import hashlib
import json
from dataclasses import replace

import pytest
import torch
from test_episodic_readout import CODEC, CONTEXT, api, source

from mars_titan.models.titans.transition_jacobian import refinement_jacobians

# Identidades del lector calculadas con el código anterior a R3 (develop 1cbfcfde5).
LEGACY = {
    "default": ({}, "106e901ec25e279f9fcf5dbcb28f71cde266e219f69d01df6fc5da633d3b2c42"),
    "first_read": (
        dict(episode_selection="first_read", refinements=2),
        "c73711842a74ec982427bac6a645ba9916dad8aaa6e407930d73da5a5ebc699b",
    ),
    "zero_refinement": (
        dict(refinement_initialization="zero_refinement"),
        "eae6818c2560dff118890b4f5c96b0a31fd48add1362b57a3f9d10db994b91a6",
    ),
}
DEVICES = ["cpu", pytest.param("cuda:0", marks=pytest.mark.cuda)]


class Name(str):
    """Un texto que solo se parece al nombre de la puerta."""


def digest(identity):
    text = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


def build(
    gate=None, *, steps=1, selection="per_step", dtype=torch.float64, device="cpu", **options
):
    config = api().EpisodicReadoutConfig(
        CODEC,
        hidden_size=32,
        refinements=steps,
        episode_selection=selection,
        familiarity_gate=gate,
        **options,
    )
    return api().EpisodicReadout(config, dtype=dtype, device=device)


def unit_keys(size):
    generator = torch.Generator().manual_seed(17)
    keys = torch.randn(size, 64, generator=generator, dtype=torch.float64)
    return (keys / torch.linalg.vector_norm(keys, dim=1, keepdim=True)).float()


def episodes(size=12, *, dtype=torch.float64, device="cpu"):
    """Claves unitarias al azar, para que cada episodio se parezca distinto al presente."""
    data = source(size) | dict(keys=unit_keys(size))
    return api().EpisodeSnapshot.create(
        **data, cutoff=10, codec_id=CODEC, context_id=CONTEXT, device=device, dtype=dtype
    )


def states(dtype=torch.float64, device="cpu"):
    return torch.linspace(-0.8, 0.9, 96, dtype=dtype, device=device).reshape(3, 32)


def run(model, data, state, **options):
    return model(state, data, context_id=CONTEXT, cutoff=10, **options)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("gate", ["familiarity", "constant"])
@pytest.mark.parametrize(
    ("steps", "selection"), [(1, "per_step"), (2, "per_step"), (4, "first_read")]
)
def test_a_new_gate_starts_exactly_at_the_reference_reader(device, dtype, gate, steps, selection):
    options = dict(steps=steps, selection=selection, dtype=dtype, device=device)
    reference, gated = build(**options), build(gate, **options)
    data, state = episodes(dtype=dtype, device=device), states(dtype, device)
    left = run(reference, data, state, differentiable=True)
    right = run(gated, data, state, differentiable=True)
    assert torch.equal(left.state, right.state)
    weights = torch.linspace(-1, 1, 96, dtype=dtype, device=device).reshape(3, 32)
    (left.state * weights).sum().backward()
    (right.state * weights).sum().backward()
    # La puerta en uno multiplica de forma exacta, así que los gradientes compartidos coinciden
    # bit a bit y solo se añaden los de la puerta.
    shared = dict(gated.named_parameters())
    for name, value in reference.named_parameters():
        assert torch.equal(value.grad, shared[name].grad), name
    added = {name for name in shared if name not in dict(reference.named_parameters())}
    expected = {"familiarity_bias"} | ({"familiarity_weight"} if gate == "familiarity" else set())
    assert added == expected
    for name in added:
        assert torch.count_nonzero(shared[name].grad) == 1


def test_the_gate_follows_its_equation_in_every_row():
    # Con una temperatura distinta de uno, los cosenos y los logits de la atención difieren.
    model, control = build("familiarity", temperature=0.5), build("constant", temperature=0.5)
    with torch.no_grad():
        model.familiarity_weight.fill_(1.5)
        model.familiarity_bias.fill_(-0.25)
        control.familiarity_bias.fill_(-0.25)
    state = states()
    query = model.query_projection(state).detach()
    query = query / torch.linalg.vector_norm(query, dim=-1, keepdim=True)
    # Las dos primeras filas tienen un episodio que coincide con su consulta y la tercera no.
    keys = unit_keys(12)
    keys[:2] = query[:2].float() / torch.linalg.vector_norm(query[:2].float(), dim=-1, keepdim=True)
    data = api().EpisodeSnapshot.create(
        **(source(12) | dict(keys=keys)),
        cutoff=10,
        codec_id=CODEC,
        context_id=CONTEXT,
        device="cpu",
        dtype=torch.float64,
    )
    result = run(model, data, state, differentiable=True)
    read = result.reads[0]
    # La familiaridad se recalcula aparte desde las claves de la instantánea.
    scores = (query @ keys.double().T).topk(model.config.neighbors, dim=-1).values
    weights = (scores / model.config.temperature).softmax(-1)
    familiarity = (weights * scores).sum(-1, keepdim=True)
    torch.testing.assert_close(read.familiarity, familiarity, rtol=0, atol=1e-12)
    assert read.familiarity.grad_fn is None and not read.familiarity.requires_grad
    assert familiarity[:2].min() > familiarity[2].item() + 0.1
    joined = torch.cat((state, state, read.values, torch.ones(3, 1, dtype=torch.float64)), dim=-1)
    step = torch.sigmoid(model.step_logit)
    gate = 2 * torch.sigmoid(1.5 * familiarity - 0.25)
    expected = state + step * gate * torch.tanh(model.refinement(joined))
    torch.testing.assert_close(result.state, expected, rtol=0, atol=1e-12)
    # El control aplica 2σ(b) a todas las filas con la misma lectura y el mismo refinador.
    constant = 2 * torch.sigmoid(torch.tensor(-0.25, dtype=torch.float64))
    expected = state + step * constant * torch.tanh(control.refinement(joined))
    torch.testing.assert_close(run(control, data, state).state, expected, rtol=0, atol=1e-12)
    assert run(control, data, state).reads[0].familiarity is None


@pytest.mark.parametrize("gate", ["familiarity", "constant"])
def test_without_episodes_the_step_is_the_reference_one(gate):
    reference, gated = build(steps=2), build(gate, steps=2)
    with torch.no_grad():
        gated.familiarity_bias.fill_(-2.0)
        if gate == "familiarity":
            gated.familiarity_weight.fill_(3.0)
    empty = api().EpisodeSnapshot.create(
        **source(0),
        cutoff=10,
        codec_id=CODEC,
        context_id=CONTEXT,
        device="cpu",
        dtype=torch.float64,
    )
    state = states()
    left, right = run(reference, empty, state), run(gated, empty, state)
    assert torch.equal(left.state, right.state)
    if gate == "familiarity":
        assert torch.equal(right.reads[0].familiarity, torch.zeros(3, 1, dtype=torch.float64))


def test_fixed_episodes_and_cached_reads_keep_the_familiarity_of_each_step():
    model = build("familiarity", steps=2, selection="first_read")
    with torch.no_grad():
        model.familiarity_weight.fill_(2.0)
    data, state = episodes(), states()
    result = run(model, data, state)
    first, second = result.reads
    assert torch.equal(first.ids, second.ids)
    # La segunda lectura atiende los mismos episodios con la consulta del estado refinado.
    assert not torch.equal(first.familiarity, second.familiarity)
    _, _, cached = model._read_parts(result.states[0], data, CONTEXT, 10)
    again, _, _ = model._read_parts(result.states[0], data, CONTEXT, 10, cached=cached)
    fresh, _, _ = model._read_parts(result.states[0], data, CONTEXT, 10)
    torch.testing.assert_close(again.familiarity, fresh.familiarity, rtol=0, atol=1e-15)


@pytest.mark.parametrize(
    "options",
    [
        dict(familiarity_gate="mean"),
        dict(familiarity_gate=True),
        dict(familiarity_gate=Name("familiarity")),
        dict(familiarity_gate="familiarity", mode="no_bank"),
    ],
    ids=["unknown", "boolean", "subclass", "without_bank"],
)
def test_the_gate_needs_its_name_and_the_bank(options):
    with pytest.raises(ValueError, match="R3 declara"):
        api().EpisodicReadoutConfig(CODEC, hidden_size=32, **options)


def test_the_gate_is_not_combined_with_the_regional_bridge():
    from mars_titan.models.titans.evidence_bridge import EvidenceBridgeConfig

    bridge = EvidenceBridgeConfig(codec_id=CODEC, hidden_size=32)
    with pytest.raises(ValueError, match="R3 declara"):
        api().EpisodicReadoutConfig(
            CODEC, hidden_size=32, evidence_bridge=bridge, familiarity_gate="constant"
        )


@pytest.mark.parametrize("name", LEGACY)
def test_without_the_gate_the_identity_of_the_reader_does_not_change(name):
    options, expected = LEGACY[name]
    assert digest(api().EpisodicReadoutConfig(CODEC, hidden_size=32, **options).identity()) == (
        expected
    )


def test_the_identity_names_the_gate_and_its_control():
    plain = build().config.identity()
    gated, control = build("familiarity").config.identity(), build("constant").config.identity()
    added = {"familiarity_gate", "familiarity", "gate", "gate_initialization", "gated"}
    added |= {"without_episodes"}
    for identity, name in ((gated, "familiarity"), (control, "constant")):
        assert set(identity) - set(plain) == added
        assert {key: identity[key] for key in plain} == plain
        assert identity["familiarity_gate"] == name
    assert gated["gate"] == "two_sigmoid_of_weight_times_familiarity_plus_bias"
    assert control["gate"] == "two_sigmoid_of_bias"
    assert gated["familiarity"] == control["familiarity"]
    assert gated["without_episodes"] == "reference_step"


def test_a_read_without_its_familiarity_is_refused_by_the_gate():
    model, control = build("familiarity"), build("constant")
    data, state = episodes(), states()
    read = model.read(state, data, context_id=CONTEXT, cutoff=10)
    assert torch.equal(model.refine(state, state, read), run(model, data, state).state)
    bare = api().EpisodeRead(read.values, read.weights, read.ids, read.presence)
    for wrong in (
        bare,
        replace(read, familiarity=read.familiarity.float()),
        replace(read, familiarity=read.familiarity.reshape(-1)),
    ):
        with pytest.raises(ValueError, match="familiaridad de cada fila"):
            model.refine(state, state, wrong)
    with pytest.raises(ValueError, match="familiaridad de cada fila"):
        model._step(state, state, bare)
    with pytest.raises(ValueError, match="familiaridad"):
        model.refine(state, state, replace(read, familiarity=read.familiarity / 0))
    # El control no necesita la familiaridad.
    plain = control.read(state, data, context_id=CONTEXT, cutoff=10)
    assert torch.equal(control.refine(state, state, plain), run(control, data, state).state)


def test_the_gate_is_saved_with_the_reader_and_refused_by_another_contract():
    gated = build("familiarity")
    with torch.no_grad():
        gated.familiarity_weight.fill_(0.5)
        gated.familiarity_bias.fill_(-0.5)
    payload = gated.state_dict()
    restored = build("familiarity")
    restored.load_state_dict(payload)
    data, state = episodes(), states()
    assert torch.equal(run(restored, data, state).state, run(gated, data, state).state)
    for other in (build(), build("constant")):
        with pytest.raises(ValueError):
            other.load_state_dict(payload)
    assert set(build("constant").state_dict()) == set(build().state_dict()) | {"familiarity_bias"}
    before = torch.get_rng_state().clone()
    build("familiarity")
    assert torch.equal(torch.get_rng_state(), before)


def test_the_working_budget_counts_the_gate():
    reference, gated = build(steps=2), build("familiarity", steps=2)
    # Por paso y fila: cosenos y su producto por los pesos, el cambio escalado y cuatro
    # escalares, en FP64 y con ocho vecinos.
    per_step = 3 * (2 * 8 + 32 + 4) * 8
    assert gated.estimated_bytes(3, 12) - reference.estimated_bytes(3, 12) == 2 * per_step
    differentiable = gated.estimated_bytes(3, 12, differentiable=True)
    assert differentiable - reference.estimated_bytes(3, 12, differentiable=True) == 8 * per_step
    assert build("constant", steps=2).estimated_bytes(3, 12) == gated.estimated_bytes(3, 12)


def test_the_transition_jacobian_refuses_the_gate():
    data = episodes()
    state = states()[:1]
    reference = refinement_jacobians(build(), state, data, context_id=CONTEXT, cutoff=10)
    assert reference.shape == (1, 32, 32)
    with pytest.raises(ValueError, match="puerta de familiaridad"):
        refinement_jacobians(build("familiarity"), state, data, context_id=CONTEXT, cutoff=10)


def test_a_new_gate_keeps_the_prediction_of_the_frozen_parent_and_the_reader():
    from test_financial_adapter import raw_batch, setup

    from mars_titan.models.titans.financial import DecisionBatch, FinancialPredictor

    original, _ = setup()
    parent = (
        FinancialPredictor(replace(original.config, head="quantile_head_v1"), dtype=torch.float64)
        .eval()
        .requires_grad_(False)
    )
    batch = DecisionBatch.from_corpus(raw_batch(), parent.config.inputs, dtype=torch.float64)
    prepared = parent.prepare(batch, parent.initial_state(batch.flow_ids))
    cutoff = batch.prediction_at[0]
    data = api().EpisodeSnapshot.create(
        **(source(12) | dict(keys=unit_keys(12))),
        cutoff=cutoff,
        codec_id=CODEC,
        context_id=CONTEXT,
        device="cpu",
        dtype=torch.float64,
    )
    results = [
        api().apply_episodic_readout(
            prepared, parent.head, build(gate), data, context_id=CONTEXT, cutoff=cutoff
        )
        for gate in (None, "familiarity", "constant")
    ]
    for result in results[1:]:
        assert torch.equal(result.point_predictions, results[0].point_predictions)
        assert torch.equal(result.quantiles, results[0].quantiles)
