"""WP8 (a) y (b): vecinos hasta 256, temperatura fija/aprendida y vía directa de etiquetas."""

import copy
import hashlib
import json
import math
from dataclasses import replace

import pytest
import torch
from test_episodic_readout import CODEC, CONTEXT, analytic, api, source
from test_financial_adapter import raw_batch, setup

from mars_titan.models.quantile_head import QuantileHead, median, pinball_loss
from mars_titan.models.titans.financial import DecisionBatch, FinancialPredictor, PreparedDecisions
from mars_titan.models.titans.session_ic_objective import SessionICTerm

# Huellas SHA256 exactas del contrato anterior (idénticas a test_familiarity_gate.py).
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


class Name(str):
    """Subclase de str que no debe colarse como opción declarada."""


def digest(identity):
    text = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


def unit_keys(size, *, seed=17):
    generator = torch.Generator().manual_seed(seed)
    keys = torch.randn(size, 64, generator=generator, dtype=torch.float64)
    keys = (keys / torch.linalg.vector_norm(keys, dim=1, keepdim=True)).float()
    return (
        keys / torch.linalg.vector_norm(keys.double(), dim=1, keepdim=True).float()
    ).contiguous()


def episodes(size=12, *, dtype=torch.float64, labels=None, keys=None, cutoff=10, seed=17):
    data = dict(
        keys=unit_keys(size, seed=seed) if keys is None else keys,
        values=torch.zeros((size, 64), dtype=torch.float32),
        labels=(
            torch.linspace(-0.02, 0.02, size, dtype=torch.float64) if labels is None else labels
        ),
        ids=torch.arange(1, size + 1, dtype=torch.int64) * 10,
        decision_at=torch.ones(size, dtype=torch.int64),
        available_at=torch.ones(size, dtype=torch.int64),
        maturity_at=torch.full((size,), 2, dtype=torch.int64),
    )
    if size:
        gen = torch.Generator().manual_seed(seed + 1)
        data["values"] = torch.randn((size, 64), generator=gen, dtype=torch.float32)
    return api().EpisodeSnapshot.create(
        **data, cutoff=cutoff, codec_id=CODEC, context_id=CONTEXT, device="cpu", dtype=dtype
    )


def build(*, dtype=torch.float64, **options):
    config = api().EpisodicReadoutConfig(CODEC, hidden_size=32, **options)
    return api().EpisodicReadout(config, dtype=dtype)


def frozen_parent(*, head="quantile_head_v1", dtype=torch.float64, zero_head=False):
    original, _ = setup()
    parent = FinancialPredictor(replace(original.config, head=head), dtype=dtype)
    if zero_head:
        with torch.no_grad():
            parent.head.weight.zero_()
            parent.head.bias.zero_()
        parent._seal_parameters()
    parent = parent.eval().requires_grad_(False)
    batch = DecisionBatch.from_corpus(raw_batch(), parent.config.inputs, dtype=dtype)
    prepared = parent.prepare(batch, parent.initial_state(batch.flow_ids))
    return parent, batch, prepared


# ===========================================================================
# Contratos por defecto: cero cambios bit a bit
# ===========================================================================


@pytest.mark.parametrize("name", LEGACY)
def test_default_options_preserve_the_exact_legacy_identity_hash(name):
    options, expected = LEGACY[name]
    config = api().EpisodicReadoutConfig(CODEC, hidden_size=32, **options)
    assert config.learned_temperature is False
    assert config.label_path == "concatenated"
    assert config.label_scale == 1.0
    assert digest(config.identity()) == expected


def test_default_readout_leaves_mature_label_absent_and_parameters_unchanged():
    model = build()
    assert set(model.state_dict()) == {
        "query_projection.weight",
        "query_projection.bias",
        "value_projection.weight",
        "value_projection.bias",
        "refinement.weight",
        "refinement.bias",
        "step_logit",
        "_extra_state",
    }
    state = torch.linspace(-0.5, 0.5, 64, dtype=torch.float64).reshape(2, 32)
    result = model(state, episodes(12), context_id=CONTEXT, cutoff=10)
    assert result.reads[0].mature_label is None


# ===========================================================================
# (a) Vecinos hasta 256, temperatura fija < 1 y temperatura aprendida en [0.02, 1]
# ===========================================================================


def test_small_case_weights_match_independent_softmax_and_concentrate_at_low_temperature():
    # Cuatro episodios con cosenos exactos conocidos frente a la consulta [1, 0, 0, ...].
    keys = torch.zeros((4, 64), dtype=torch.float32)
    cosines = [1.0, 0.6, 0.0, -0.5]
    for row, cos_val in enumerate(cosines):
        keys[row, 0] = cos_val
        keys[row, 1] = math.sqrt(max(0.0, 1.0 - cos_val * cos_val))
    keys = (
        keys / torch.linalg.vector_norm(keys.double(), dim=1, keepdim=True).float()
    ).contiguous()
    data = episodes(4, keys=keys)
    z = torch.zeros((1, 32), dtype=torch.float64)
    z[0, 0] = 1.0

    warm = build(neighbors=4, temperature=1.0)
    sharp = build(neighbors=4, temperature=0.05)
    analytic(warm)
    analytic(sharp)

    res_warm = warm(z, data, context_id=CONTEXT, cutoff=10)
    res_sharp = sharp(z, data, context_id=CONTEXT, cutoff=10)

    # Cuenta independiente en aritmética estándar de Python.
    actual_cos = [float(keys[i, 0].double()) for i in range(4)]
    for res, temp in ((res_warm, 1.0), (res_sharp, 0.05)):
        scaled = [c / temp for c in actual_cos]
        peak = max(scaled)
        exps = [math.exp(v - peak) for v in scaled]
        expected = [v / sum(exps) for v in exps]
        torch.testing.assert_close(
            res.reads[0].weights[0],
            torch.tensor(expected, dtype=torch.float64),
            rtol=0,
            atol=1e-13,
        )

    assert res_sharp.reads[0].weights[0, 0].item() > 0.999
    assert res_warm.reads[0].weights[0, 0].item() < 0.5


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("neighbors", [64, 256])
def test_large_k_and_low_temperature_concentrate_weights_on_the_closest_neighbors(dtype, neighbors):
    data = episodes(300, dtype=dtype, seed=29)
    z = torch.linspace(-0.7, 0.8, 96, dtype=dtype).reshape(3, 32)
    warm = build(neighbors=neighbors, temperature=1.0, dtype=dtype)
    sharp = build(neighbors=neighbors, temperature=0.05, dtype=dtype)

    res_warm = warm(z, data, context_id=CONTEXT, cutoff=10)
    res_sharp = sharp(z, data, context_id=CONTEXT, cutoff=10)

    w_warm, w_sharp = res_warm.reads[0].weights, res_sharp.reads[0].weights
    assert w_warm.shape == w_sharp.shape == (3, neighbors)
    assert torch.equal(res_warm.reads[0].ids, res_sharp.reads[0].ids)
    # Con k grande y temperatura baja, los pesos se concentran en los vecinos más parecidos.
    assert bool((w_sharp[:, :4].sum(-1) > w_warm[:, :4].sum(-1) + 0.2).all())
    assert bool((w_sharp[:, :4].sum(-1) > 3.0 * w_warm[:, :4].sum(-1)).all())
    assert bool((w_sharp[:, 0] > w_warm[:, 0]).all())
    entropy_warm = -(w_warm * w_warm.clamp_min(1e-30).log()).sum(-1)
    entropy_sharp = -(w_sharp * w_sharp.clamp_min(1e-30).log()).sum(-1)
    assert bool((entropy_sharp < entropy_warm).all())
    # Determinismo exacto en repeticiones.
    assert torch.equal(res_sharp.state, sharp(z, data, context_id=CONTEXT, cutoff=10).state)


def test_neighbors_accepts_up_to_256_and_rejects_above_or_invalid():
    assert api().EpisodicReadoutConfig(CODEC, hidden_size=32, neighbors=256).neighbors == 256
    for bad in (0, -1, 257, True, 8.0):
        with pytest.raises(ValueError, match="vecinos"):
            api().EpisodicReadoutConfig(CODEC, hidden_size=32, neighbors=bad)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_learned_temperature_matches_fixed_at_init_is_bounded_and_learns(dtype):
    fixed = build(neighbors=16, temperature=0.2, dtype=dtype)
    learned = build(neighbors=16, temperature=0.2, learned_temperature=True, dtype=dtype)
    data = episodes(40, dtype=dtype)
    z = torch.linspace(-0.6, 0.7, 96, dtype=dtype).reshape(3, 32)

    out_fixed = fixed(z, data, context_id=CONTEXT, cutoff=10, differentiable=True)
    out_learned = learned(z, data, context_id=CONTEXT, cutoff=10, differentiable=True)
    tol = 1e-6 if dtype == torch.float32 else 1e-13
    torch.testing.assert_close(out_learned.state, out_fixed.state, rtol=tol, atol=tol)
    assert learned.effective_temperature().item() == pytest.approx(0.2, rel=1e-6)

    # Gradiente finito y no nulo hacia temperature_logit.
    loss = (out_learned.state * torch.linspace(-1, 1, 96, dtype=dtype).reshape(3, 32)).sum()
    loss.backward()
    assert learned.temperature_logit.grad is not None
    assert torch.isfinite(learned.temperature_logit.grad).all()
    assert learned.temperature_logit.grad.abs().item() > 0

    # Acotación estricta en [0.02, 1.0] incluso en extremos del logit (redondeo IEEE 754).
    with torch.no_grad():
        learned.temperature_logit.fill_(-100.0)
        low = learned.effective_temperature().item()
        learned.temperature_logit.fill_(100.0)
        high = learned.effective_temperature().item()
    lower_bound = torch.tensor(0.02, dtype=dtype).item()
    assert lower_bound <= low <= lower_bound + 1e-6
    assert 1.0 - 1e-6 <= high <= 1.0

    # Un logit no finito se rechaza explícitamente aunque la sigmoide sature.
    with torch.no_grad():
        learned.temperature_logit.fill_(float("inf"))
    with pytest.raises(ValueError, match="NaN|infinito"):
        learned(z, data, context_id=CONTEXT, cutoff=10)


def test_learned_temperature_passes_gradcheck_and_respects_contracts():
    model = build(neighbors=6, temperature=0.2, learned_temperature=True, dtype=torch.float64)
    data = episodes(12, dtype=torch.float64)
    z = torch.linspace(-0.5, 0.6, 64, dtype=torch.float64).reshape(2, 32)

    def forward_with_logit(logit):
        return torch.func.functional_call(
            model,
            {"temperature_logit": logit},
            (z, data),
            dict(context_id=CONTEXT, cutoff=10, differentiable=True),
        ).state

    probe = model.temperature_logit.detach().clone().requires_grad_()
    assert torch.autograd.gradcheck(forward_with_logit, (probe,), eps=1e-6, atol=1e-5, rtol=1e-4)

    # Identidad, state_dict y RNG global intacto.
    plain_id = build(neighbors=6, temperature=0.2).config.identity()
    learned_id = model.config.identity()
    assert learned_id["learned_temperature"] is True
    assert learned_id["temperature_bounds"] == [0.02, 1.0]
    assert {k: learned_id[k] for k in plain_id} == plain_id

    payload = copy.deepcopy(model.state_dict())
    replica = build(neighbors=6, temperature=0.2, learned_temperature=True, dtype=torch.float64)
    replica.load_state_dict(payload)
    assert torch.equal(
        replica(z, data, context_id=CONTEXT, cutoff=10).state,
        model(z, data, context_id=CONTEXT, cutoff=10).state,
    )
    with pytest.raises(ValueError):
        build(neighbors=6, temperature=0.2, dtype=torch.float64).load_state_dict(payload)

    before = torch.get_rng_state().clone()
    build(neighbors=6, temperature=0.2, learned_temperature=True)
    assert torch.equal(torch.get_rng_state(), before)


@pytest.mark.parametrize(
    "options",
    [
        dict(learned_temperature="yes"),
        dict(learned_temperature=1),
        dict(learned_temperature=True, temperature=1.0),
        dict(learned_temperature=True, temperature=0.02),
        dict(learned_temperature=True, temperature=0.01),
        dict(learned_temperature=True, temperature=2.0),
        dict(learned_temperature=True, temperature=0.2, mode="no_bank"),
    ],
)
def test_learned_temperature_rejects_invalid_declarations(options):
    with pytest.raises(ValueError):
        api().EpisodicReadoutConfig(CODEC, hidden_size=32, **options)


def test_learned_temperature_is_refused_with_evidence_bridge():
    from mars_titan.models.titans.evidence_bridge import EvidenceBridgeConfig

    bridge = EvidenceBridgeConfig(codec_id=CODEC, hidden_size=32, temperature=0.2)
    with pytest.raises(ValueError):
        api().EpisodicReadoutConfig(
            CODEC,
            hidden_size=32,
            temperature=0.2,
            evidence_bridge=bridge,
            learned_temperature=True,
        )


# ===========================================================================
# (b) Vía directa de etiquetas maduras: label_path="direct_v1"
# ===========================================================================


def test_direct_v1_computes_exact_scaled_weighted_mature_label_including_cached_reads():
    model = build(neighbors=4, temperature=0.1, label_path="direct_v1", label_scale=25.0)
    data = episodes(12, labels=torch.linspace(-0.03, 0.05, 12, dtype=torch.float64))
    z = torch.linspace(-0.8, 0.9, 96, dtype=torch.float64).reshape(3, 32)

    read, indices, cached = model._read_parts(z, data, CONTEXT, 10)
    raw_labels = data._values[2]
    selected_labels = raw_labels.index_select(0, indices.flatten()).reshape(3, 4)
    expected = (read.weights * selected_labels).sum(dim=-1, keepdim=True) * 25.0
    torch.testing.assert_close(read.mature_label, expected, rtol=0, atol=1e-14)

    # La ruta con caché devuelve exactamente la misma etiqueta madura.
    again, _, _ = model._read_parts(z, data, CONTEXT, 10, cached=cached)
    torch.testing.assert_close(again.mature_label, read.mature_label, rtol=0, atol=1e-14)

    # Instantánea vacía: presencia falsa y etiqueta madura exactamente cero.
    empty = episodes(0)
    empty_read = model.read(z, empty, context_id=CONTEXT, cutoff=10)
    assert empty_read.mature_label.shape == (3, 1)
    assert torch.equal(empty_read.mature_label, torch.zeros((3, 1), dtype=torch.float64))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("head", ["quantile_head_v1", "scalar"])
def test_direct_v1_starts_bitwise_identical_to_the_parent_with_zero_refinement(dtype, head):
    parent, batch, prepared = frozen_parent(head=head, dtype=dtype)
    cutoff = batch.prediction_at[0]
    data = episodes(16, dtype=dtype, cutoff=cutoff)
    reader = build(
        neighbors=8,
        temperature=0.1,
        refinement_initialization="zero_refinement",
        label_path="direct_v1",
        label_scale=50.0,
        dtype=dtype,
    )
    result = api().apply_episodic_readout(
        prepared, parent.head, reader, data, context_id=CONTEXT, cutoff=cutoff
    )
    assert torch.equal(result.point_predictions, prepared.point_predictions)
    if head == "quantile_head_v1":
        assert torch.equal(result.quantiles, prepared.quantiles)
    else:
        assert result.quantiles is None


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("head", ["quantile_head_v1", "scalar"])
@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_direct_v1_gain_moves_in_the_right_direction_in_one_step_without_zero_chain(
    dtype, head, sign
):
    # También probamos con cabeza en cero (predicción inicial constante) y zero_refinement:
    # aunque refinement arranca a cero, el gradiente de label_gain no pasa por ninguna capa a cero.
    rows = 4
    if head == "quantile_head_v1":
        head_module = QuantileHead(32, dtype=dtype)
    else:
        head_module = torch.nn.Linear(32, 1, dtype=dtype)
    with torch.no_grad():
        head_module.weight.zero_()
        head_module.bias.zero_()
    head_module = head_module.eval().requires_grad_(False)
    gen = torch.Generator().manual_seed(73)
    z = torch.randn((rows, 32), generator=gen, dtype=dtype)
    with torch.no_grad():
        quantiles = head_module(z) if head == "quantile_head_v1" else None
        points = median(quantiles) if quantiles is not None else head_module(z).squeeze(-1)
    prepared = PreparedDecisions(
        point_predictions=points,
        detached_tokens=None,
        next_state=None,
        representation_id="a" * 64,
        input_digest="b" * 64,
        working_state=z,
        local_control=None,
        quantiles=quantiles,
    )
    cutoff = 10
    reader = build(
        neighbors=4,
        temperature=0.05,
        refinement_initialization="zero_refinement",
        label_path="direct_v1",
        label_scale=10.0,
        dtype=dtype,
    )
    # Construimos un banco donde cada fila encuentra un episodio distinto y muy parecido cuya
    # etiqueta madura es informativa con signo conocido respecto al objetivo real.
    with torch.no_grad():
        q = reader.query_projection(prepared.working_state)
        q = q / torch.linalg.vector_norm(q, dim=-1, keepdim=True)
    keys = unit_keys(12, seed=41)
    for row in range(rows):
        keys[row] = q[row].float() / torch.linalg.vector_norm(q[row].double()).float()
    episode_labels = torch.zeros(12, dtype=torch.float64)
    signal = torch.tensor([-0.05, -0.02, 0.03, 0.06], dtype=torch.float64)
    episode_labels[:rows] = signal
    data = episodes(12, dtype=dtype, labels=episode_labels, keys=keys, cutoff=cutoff)

    result = api().apply_episodic_readout(
        prepared, head_module, reader, data, context_id=CONTEXT, cutoff=cutoff, differentiable=True
    )
    # Predicción inicial constante (cero, idéntica al padre).
    assert torch.equal(result.point_predictions, torch.zeros(rows, dtype=dtype))

    target = (sign * signal).to(dtype=dtype)
    if head == "quantile_head_v1":
        loss_before = pinball_loss(result.quantiles, target)
    else:
        loss_before = torch.nn.functional.l1_loss(result.point_predictions, target)
    loss_before.backward()

    grad = reader.label_gain.grad
    assert grad is not None and torch.isfinite(grad).all()
    assert grad.abs().item() > 0.0
    # Un paso de descenso por gradiente debe mover label_gain hacia el signo informativo.
    step = 0.05
    with torch.no_grad():
        reader.label_gain.sub_(step * grad)
    assert reader.label_gain.item() * sign > 0.0

    # Tras el paso, la pérdida baja y los cuantiles conservan su orden monótono.
    after = api().apply_episodic_readout(
        prepared, head_module, reader, data, context_id=CONTEXT, cutoff=cutoff
    )
    if head == "quantile_head_v1":
        assert bool((after.quantiles[:, 1:] >= after.quantiles[:, :-1]).all())
        assert torch.equal(after.point_predictions, median(after.quantiles))
        loss_after = pinball_loss(after.quantiles, target)
    else:
        loss_after = torch.nn.functional.l1_loss(after.point_predictions, target)
    assert loss_after.item() < loss_before.item()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_direct_v1_has_finite_gradients_under_session_ic_even_when_prediction_is_constant(dtype):
    # Regla 4: gradiente finito cuando la última capa arranca en cero (predicción constante).
    rows = 10
    head = QuantileHead(32, dtype=dtype)
    with torch.no_grad():
        head.weight.zero_()
        head.bias.zero_()
    reader = build(
        neighbors=4,
        temperature=0.1,
        refinement_initialization="zero_refinement",
        label_path="direct_v1",
        label_scale=10.0,
        dtype=dtype,
    )
    z = torch.linspace(-0.8, 0.8, rows * 32, dtype=dtype).reshape(rows, 32)
    data = episodes(20, dtype=dtype)
    with torch.no_grad():
        q = head(z)

    prepared = PreparedDecisions(
        point_predictions=median(q),
        detached_tokens=None,
        next_state=None,
        representation_id="a" * 64,
        input_digest="b" * 64,
        working_state=z,
        local_control=None,
        quantiles=q,
    )
    result = api().apply_episodic_readout(
        prepared, head, reader, data, context_id=CONTEXT, cutoff=10, differentiable=True
    )
    target = torch.linspace(-0.02, 0.03, rows, dtype=dtype)
    term = SessionICTerm(weight=3.0, minimum_rows=8)
    loss = pinball_loss(result.quantiles, target) + term.term(
        result.point_predictions, target, [("US", 10)] * rows
    )
    loss.backward()
    for name, parameter in reader.named_parameters():
        assert parameter.grad is None or torch.isfinite(parameter.grad).all(), name
    assert reader.label_gain.grad is not None and torch.isfinite(reader.label_gain.grad).all()
    assert reader.label_gain.grad.abs().item() > 0


def test_direct_v1_enforces_causality_immature_labels_never_enter():
    cutoff = 10
    # 1) Una instantánea que incluya una etiqueta no madura (maturity_at > cutoff) es rechazada.
    bad = source(3)
    bad["maturity_at"][2] = cutoff + 1
    with pytest.raises(ValueError, match="maduración"):
        api().EpisodeSnapshot.create(
            **bad, cutoff=cutoff, codec_id=CODEC, context_id=CONTEXT, dtype=torch.float64
        )

    # 2) El mismo banco maduro en t=10 produce la misma etiqueta directa y predicción aunque
    # existan episodios futuros fuera de la instantánea, y con banco vacío la contribución es cero.
    parent, batch, prepared = frozen_parent(head="quantile_head_v1", dtype=torch.float64)
    at = batch.prediction_at[0]
    mature_only = episodes(6, cutoff=at)
    reader = build(
        neighbors=6,
        temperature=0.1,
        refinement_initialization="zero_refinement",
        label_path="direct_v1",
        label_scale=20.0,
    )
    with torch.no_grad():
        reader.label_gain.fill_(0.5)
    res_mature = api().apply_episodic_readout(
        prepared, parent.head, reader, mature_only, context_id=CONTEXT, cutoff=at
    )
    # Comprobación de que solo episodios con maturity_at <= cutoff están en la instantánea.
    assert bool((mature_only._values[6] <= at).all())
    assert not torch.equal(res_mature.point_predictions, prepared.point_predictions)

    # Sin episodios maduros en el corte, la ganancia no altera la predicción ni recibe gradiente.
    empty = episodes(0, cutoff=at)
    res_empty = api().apply_episodic_readout(
        prepared, parent.head, reader, empty, context_id=CONTEXT, cutoff=at, differentiable=True
    )
    assert torch.equal(res_empty.point_predictions, prepared.point_predictions)
    res_empty.point_predictions.sum().backward()
    assert reader.label_gain.grad.item() == 0.0


def test_direct_v1_gradcheck_identity_state_dict_and_validation():
    parent, batch, prepared = frozen_parent(head="scalar", dtype=torch.float64)
    cutoff = batch.prediction_at[0]
    data = episodes(10, dtype=torch.float64, cutoff=cutoff)
    reader = build(neighbors=4, temperature=0.2, label_path="direct_v1", label_scale=5.0)
    with torch.no_grad():
        reader.label_gain.fill_(0.25)

    def forward_with_gain(gain):
        # Sustituimos label_gain funcionalmente para gradcheck exacto.
        original = reader.label_gain
        reader._parameters["label_gain"] = gain
        try:
            return (
                api()
                .apply_episodic_readout(
                    prepared,
                    parent.head,
                    reader,
                    data,
                    context_id=CONTEXT,
                    cutoff=cutoff,
                    differentiable=True,
                )
                .point_predictions
            )
        finally:
            reader._parameters["label_gain"] = original

    probe = torch.tensor(0.25, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(forward_with_gain, (probe,), eps=1e-6, atol=1e-6, rtol=1e-5)

    # Identidad y presupuesto.
    plain = build(neighbors=4, temperature=0.2)
    plain_id, direct_id = plain.config.identity(), reader.config.identity()
    assert direct_id["label_path"] == "direct_v1"
    assert direct_id["label_scale"] == 5.0
    assert {k: direct_id[k] for k in plain_id} == plain_id
    assert reader.estimated_bytes(3, 10) > plain.estimated_bytes(3, 10)

    # state_dict y rechazo cruzado.
    payload = copy.deepcopy(reader.state_dict())
    replica = build(neighbors=4, temperature=0.2, label_path="direct_v1", label_scale=5.0)
    replica.load_state_dict(payload)
    assert torch.equal(
        api()
        .apply_episodic_readout(
            prepared, parent.head, replica, data, context_id=CONTEXT, cutoff=cutoff
        )
        .point_predictions,
        api()
        .apply_episodic_readout(
            prepared, parent.head, reader, data, context_id=CONTEXT, cutoff=cutoff
        )
        .point_predictions,
    )
    with pytest.raises(ValueError):
        plain.load_state_dict(payload)
    with pytest.raises(ValueError):
        build(
            neighbors=4, temperature=0.2, label_path="direct_v1", label_scale=10.0
        ).load_state_dict(payload)

    # Validación de entradas inválidas.
    for bad in (
        dict(label_path="direct_v2"),
        dict(label_path=Name("direct_v1")),
        dict(label_path=True),
        dict(label_path="direct_v1", mode="no_bank"),
        dict(label_path="concatenated", label_scale=10.0),
        dict(label_path="direct_v1", label_scale=0.0),
        dict(label_path="direct_v1", label_scale=-1.0),
        dict(label_path="direct_v1", label_scale=float("nan")),
        dict(label_path="direct_v1", label_scale=True),
        dict(label_path="direct_v1", label_scale=1e5),
    ):
        with pytest.raises(ValueError):
            api().EpisodicReadoutConfig(CODEC, hidden_size=32, **bad)

    # Combinación independiente de (a) + (b).
    both = build(
        neighbors=32,
        temperature=0.1,
        learned_temperature=True,
        label_path="direct_v1",
        label_scale=10.0,
        refinement_initialization="zero_refinement",
    )
    out = api().apply_episodic_readout(
        prepared, parent.head, both, episodes(40, cutoff=cutoff), context_id=CONTEXT, cutoff=cutoff
    )
    assert torch.equal(out.point_predictions, prepared.point_predictions)


def test_direct_v1_is_refused_with_evidence_bridge_and_rejects_nonfinite_gain():
    from mars_titan.models.titans.evidence_bridge import EvidenceBridgeConfig

    bridge = EvidenceBridgeConfig(codec_id=CODEC, hidden_size=32, temperature=0.2)
    with pytest.raises(ValueError, match="direct_v1"):
        api().EpisodicReadoutConfig(
            CODEC,
            hidden_size=32,
            temperature=0.2,
            evidence_bridge=bridge,
            label_path="direct_v1",
        )

    parent, batch, prepared = frozen_parent(head="scalar", dtype=torch.float64)
    cutoff = batch.prediction_at[0]
    reader = build(neighbors=4, temperature=0.2, label_path="direct_v1", label_scale=5.0)
    with torch.no_grad():
        reader.label_gain.fill_(float("nan"))
    with pytest.raises(ValueError, match="NaN|infinito"):
        api().apply_episodic_readout(
            prepared,
            parent.head,
            reader,
            episodes(8, cutoff=cutoff),
            context_id=CONTEXT,
            cutoff=cutoff,
        )


def test_direct_v1_propagates_gradients_to_query_and_temperature_and_supports_supervised_k():
    from mars_titan.training.mars_titan_run import readout_loss

    rows = 4
    head = QuantileHead(32, dtype=torch.float64).eval().requires_grad_(False)
    gen = torch.Generator().manual_seed(91)
    z = torch.randn((rows, 32), generator=gen, dtype=torch.float64)
    with torch.no_grad():
        quantiles = head(z)
    prepared = PreparedDecisions(
        point_predictions=median(quantiles),
        detached_tokens=None,
        next_state=None,
        representation_id="a" * 64,
        input_digest="b" * 64,
        working_state=z,
        local_control=None,
        quantiles=quantiles,
    )
    reader = build(
        neighbors=6,
        temperature=0.2,
        learned_temperature=True,
        refinements=2,
        episode_selection="first_read",
        refinement_initialization="zero_refinement",
        label_path="direct_v1",
        label_scale=10.0,
        dtype=torch.float64,
    )
    data = episodes(16, dtype=torch.float64, cutoff=10, seed=53)
    target = torch.tensor([-0.03, -0.01, 0.02, 0.05], dtype=torch.float64)

    # Paso 0 con label_gain == 0: label_gain recibe gradiente directo (sin pasar por refinement=0).
    res0 = api().apply_episodic_readout(
        prepared, head, reader, data, context_id=CONTEXT, cutoff=10, differentiable=True
    )
    loss0 = readout_loss(
        res0, head, torch.arange(rows), target, loss="pinball", supervised=(1, 2), extension=reader
    )
    loss0.backward()
    assert reader.label_gain.grad is not None and reader.label_gain.grad.abs().item() > 0.0

    # Tras mover label_gain en el paso 0, en el paso 1 el gradiente ya llega a query_projection
    # y a temperature_logit a través de la vía directa aunque refinement siga en cero.
    with torch.no_grad():
        reader.label_gain.sub_(0.1 * reader.label_gain.grad)
    reader.zero_grad()
    res1 = api().apply_episodic_readout(
        prepared, head, reader, data, context_id=CONTEXT, cutoff=10, differentiable=True
    )
    loss1 = readout_loss(
        res1, head, torch.arange(rows), target, loss="pinball", supervised=(1, 2), extension=reader
    )
    # Comprobación cuantitativa: la pérdida supervisada incluye apply_direct_label en k=1 y k=2.
    with torch.no_grad():
        q1_raw = head(res1.readout.states[0])
        _, q1_direct = reader.apply_direct_label(median(q1_raw), q1_raw, res1.readout.reads[0])
        expected_loss1 = 0.5 * (
            pinball_loss(q1_direct, target) + pinball_loss(res1.quantiles, target)
        )
        without_k1_direct = 0.5 * (
            pinball_loss(q1_raw, target) + pinball_loss(res1.quantiles, target)
        )
    torch.testing.assert_close(loss1.detach(), expected_loss1, rtol=0, atol=1e-13)
    assert not torch.isclose(loss1.detach(), without_k1_direct, rtol=0, atol=1e-9)
    loss1.backward()
    for name in ("query_projection.weight", "temperature_logit", "label_gain"):
        grad = dict(reader.named_parameters())[name].grad
        assert grad is not None and torch.isfinite(grad).all(), name
        assert torch.count_nonzero(grad) > 0, name


def test_default_snapshot_roundtrip_and_estimated_bytes_unchanged():
    snap = episodes(12, dtype=torch.float64)
    restored = api().EpisodeSnapshot.restore(
        snap.export_cpu(),
        codec_id=CODEC,
        context_id=CONTEXT,
        cutoff=10,
        device="cpu",
        dtype=torch.float64,
    )
    assert restored._digest == snap._digest
    model = build(neighbors=8, temperature=1.0, dtype=torch.float64)
    z = torch.linspace(-0.4, 0.4, 64, dtype=torch.float64).reshape(2, 32)
    out_orig = model(z, snap, context_id=CONTEXT, cutoff=10)
    out_rest = model(z, restored, context_id=CONTEXT, cutoff=10)
    assert torch.equal(out_orig.state, out_rest.state)
    # Fórmula exacta de estimated_bytes sin opciones nuevas (idéntica al contrato previo).
    batch, ep, width, k, size = 2, 12, 32, 8, 8
    expected = (
        65536
        + batch * ep * 32
        + batch * (k * (64 * (8 + size) + (65 + width) * size + 32) + (12 * width + 128) * size)
    )
    assert model.estimated_bytes(batch, ep) == expected
