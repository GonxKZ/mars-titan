"""Atención diferencial de U1 (#786): paridad al empezar, fórmula, causalidad y presupuesto."""

import hashlib
import math

import pytest
import torch
from test_gate_bias import V1_MAC_FINGERPRINT
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from mars_titan.models.titans.config import MACConfig, MemoryConfig
from mars_titan.models.titans.differential_attention import (
    DIFFERENTIAL_ATTENTION,
    DifferentialMap,
    identity,
)
from mars_titan.models.titans.mac import TitansMAC

DIM, HEADS = 8, 2


@pytest.fixture(autouse=True)
def explicit_fastpath():
    previous = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    yield
    torch.backends.mha.set_fastpath_enabled(previous)


def mac(differential=None, *, mode="online", segment=1, budget=8 * 1024 * 1024):
    memory = MemoryConfig(dim=DIM, depth=2, max_batch=4, max_tokens=4, residual_layer_norm=True)
    config = MACConfig(
        memory=memory,
        heads=HEADS,
        persistent_tokens=2,
        max_segment=segment,
        memory_mode=mode,
        max_attention_elements=budget,
        differential_attention=differential,
    )
    return TitansMAC(config, dtype=torch.float64)


def walk(model, steps=3, rows=3, segment=1):
    generator = torch.Generator().manual_seed(3)
    state = model.initial_state(rows, differentiable=True)
    outputs = []
    with sdpa_kernel(SDPBackend.MATH):
        for _ in range(steps):
            token = torch.randn(rows, segment, DIM, generator=generator, dtype=torch.float64)
            output, state = model(token, state, differentiable=True)
            outputs.append(output)
    total = torch.stack(outputs)
    total.square().sum().backward()
    return total.detach(), state, {name: value.grad for name, value in model.named_parameters()}


@pytest.mark.parametrize("mode", ["online", "frozen", "disabled"])
def test_a_new_differential_map_starts_exactly_at_the_reference(mode):
    expected, last, control = walk(mac(mode=mode))
    model = mac(DIFFERENTIAL_ATTENTION, mode=mode)
    observed, final, gradients = walk(model)
    assert torch.equal(observed, expected)
    for left, right in zip(last.memory.weights, final.memory.weights, strict=True):
        assert torch.equal(left, right)
    added = {name: gradients.pop(name) for name in list(gradients) if name.startswith("differ")}
    assert set(added) == {"differential.query", "differential.key", "differential.weight"}
    assert gradients.keys() == control.keys()
    for name, value in control.items():
        assert (value is None) == (gradients[name] is None), name
        assert value is None or torch.equal(value, gradients[name]), name
    # λ recibe gradiente desde el primer paso. Q₂ y K₂ solo cuando λ deja de ser cero.
    assert bool((added["differential.weight"] != 0).all())
    assert not torch.count_nonzero(added["differential.query"])
    assert not torch.count_nonzero(added["differential.key"])


def explicit_map(module, packed, values_weight, output, mask, length):
    """El mismo segundo mapa con productos y softmax explícitos, cabeza a cabeza."""
    width = packed.shape[-1] // HEADS
    queries = packed[:, -length:] @ module.query.T
    keys = packed @ module.key.T
    values = packed @ values_weight.T
    heads = []
    for head in range(HEADS):
        part = slice(head * width, (head + 1) * width)
        scores = queries[..., part] @ keys[..., part].transpose(1, 2) / math.sqrt(width)
        scores = scores.masked_fill(mask[-length:], -math.inf)
        heads.append(module.weight[head] * (scores.softmax(-1) @ values[..., part]))
    return torch.cat(heads, dim=-1) @ output.weight.T


def test_the_second_map_matches_its_equation_head_by_head():
    module = DifferentialMap(DIM, HEADS, seed=5, dtype=torch.float64)
    with torch.no_grad():
        module.weight.copy_(torch.tensor([0.7, -0.3], dtype=torch.float64))
    generator = torch.Generator().manual_seed(9)
    packed = torch.randn(3, 7, DIM, generator=generator, dtype=torch.float64)
    values_weight = torch.randn(DIM, DIM, generator=generator, dtype=torch.float64)
    output = nn.Linear(DIM, DIM, bias=False, dtype=torch.float64)
    mask = torch.ones(7, 7, dtype=torch.bool).triu(1)
    with sdpa_kernel(SDPBackend.MATH):
        result = module(packed, values_weight, output, mask, 3)
    expected = explicit_map(module, packed, values_weight, output, mask, 3)
    assert result.shape == (3, 3, DIM)
    torch.testing.assert_close(result, expected, rtol=1e-12, atol=1e-14)


def test_each_segment_position_only_reads_earlier_positions():
    module = DifferentialMap(DIM, HEADS, seed=5, dtype=torch.float64)
    with torch.no_grad():
        module.weight.fill_(0.5)
    generator = torch.Generator().manual_seed(4)
    packed = torch.randn(2, 7, DIM, generator=generator, dtype=torch.float64)
    values_weight = torch.randn(DIM, DIM, generator=generator, dtype=torch.float64)
    output = nn.Linear(DIM, DIM, bias=False, dtype=torch.float64)
    mask = torch.ones(7, 7, dtype=torch.bool).triu(1)
    with sdpa_kernel(SDPBackend.MATH):
        base = module(packed, values_weight, output, mask, 3)
        changed = packed.clone()
        changed[:, 5] += 1.0
        moved = module(changed, values_weight, output, mask, 3)
    # Las consultas del segmento son las posiciones 4, 5 y 6. Cambiar la 5 no toca la 4.
    assert torch.equal(moved[:, 0], base[:, 0])
    assert not torch.equal(moved[:, 1], base[:, 1]) and not torch.equal(moved[:, 2], base[:, 2])


def test_mac_subtracts_the_second_map_from_the_attention_of_the_segment(monkeypatch):
    model = mac(DIFFERENTIAL_ATTENTION, segment=2)
    with torch.no_grad():
        model.differential.weight.copy_(torch.tensor([0.4, -0.6], dtype=torch.float64))
    seen = {}

    def record(module, inputs, outputs):
        seen.setdefault(id(module), []).append((inputs[0].detach(), outputs[0].detach()))

    model.attention.register_forward_hook(record)
    written = []
    update = model.memory.update

    def spy(observed, state, **options):
        written.append(observed.detach())
        return update(observed, state, **options)

    monkeypatch.setattr(model.memory, "update", spy)
    token = torch.randn(2, 2, DIM, generator=torch.Generator().manual_seed(8), dtype=torch.float64)
    with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
        model(token, model.initial_state(2))
        ((packed, attended),) = seen[id(model.attention)]
        mask = torch.ones(packed.shape[1], packed.shape[1], dtype=torch.bool).triu(1)
        second = model.differential(
            packed, model.attention.in_proj_weight[2 * DIM :], model.attention.out_proj, mask, 2
        )
    assert torch.equal(written[0], attended[:, -2:] - second)
    assert torch.count_nonzero(second) > 0


def test_the_projections_use_a_derived_seed_and_the_bound_of_the_in_proj():
    rng = torch.get_rng_state().clone()
    first = DifferentialMap(DIM, HEADS, seed=43, dtype=torch.float64)
    again = DifferentialMap(DIM, HEADS, seed=43, dtype=torch.float64)
    other = DifferentialMap(DIM, HEADS, seed=44, dtype=torch.float64)
    assert torch.equal(torch.get_rng_state(), rng)
    assert torch.equal(first.query, again.query) and torch.equal(first.key, again.key)
    assert not torch.equal(first.query, other.query)
    bound = math.sqrt(6 / (4 * DIM))
    for value in (first.query, first.key):
        assert value.shape == (DIM, DIM) and float(value.detach().abs().max()) <= bound
    # La semilla es la de MAC pasada por SHA-256, así que Q₂ y K₂ no repiten los primeros
    # valores que MAC sortea con su propia semilla.
    derived = hashlib.sha256(f"{DIFFERENTIAL_ATTENTION}/43".encode()).digest()
    generator = torch.Generator().manual_seed(int.from_bytes(derived[:7], "little"))
    drawn = torch.empty(2 * DIM, DIM, dtype=torch.float64).uniform_(
        -bound, bound, generator=generator
    )
    assert torch.equal(first.query, drawn[:DIM]) and torch.equal(first.key, drawn[DIM:])
    assert not torch.count_nonzero(first.weight)
    reference = dict(mac().named_parameters())
    declared = dict(mac(DIFFERENTIAL_ATTENTION).named_parameters())
    for name, value in reference.items():
        assert torch.equal(declared[name], value), name
    assert torch.equal(declared["differential.query"], first.query)


def test_the_heads_must_divide_the_dimension():
    with pytest.raises(ValueError, match="heads debe dividir dim"):
        DifferentialMap(DIM, 3, seed=43, dtype=torch.float64)


def test_the_attention_budget_counts_the_second_map():
    # Con 3 filas, 2 cabezas y la secuencia de 2 + 1 + 1 posiciones, MAC usa 3·2·16 = 96
    # elementos y el segundo mapa añade 3·2·1·4 = 24.
    token = torch.zeros(3, 1, DIM, dtype=torch.float64)
    with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
        plain = mac(budget=96)
        plain(token, plain.initial_state(3))
        declared = mac(DIFFERENTIAL_ATTENTION, budget=96)
        with pytest.raises(ValueError, match="presupuesto de elementos"):
            declared(token, declared.initial_state(3))
        enough = mac(DIFFERENTIAL_ATTENTION, budget=120)
        enough(token, enough.initial_state(3))


def test_the_identity_records_the_second_map_only_when_declared():
    memory = MemoryConfig(dim=4, depth=2, max_batch=2, max_tokens=3)
    plain = MACConfig(memory=memory, heads=2, persistent_tokens=2, max_segment=3)
    assert plain.fingerprint() == V1_MAC_FINGERPRINT
    assert "differential_attention" not in plain.identity()
    declared = MACConfig(
        memory=memory,
        heads=2,
        persistent_tokens=2,
        max_segment=3,
        differential_attention=DIFFERENTIAL_ATTENTION,
    )
    assert declared.identity()["differential_attention"] == identity()
    assert declared.fingerprint() != V1_MAC_FINGERPRINT
    with pytest.raises(ValueError, match="differential_attention admite"):
        MACConfig(
            memory=memory,
            heads=2,
            max_segment=3,
            differential_attention="differential_attention_v2",
        )


def test_lambda_away_from_zero_trains_the_second_projections():
    model = mac(DIFFERENTIAL_ATTENTION)
    with torch.no_grad():
        model.differential.weight.copy_(torch.tensor([0.5, 0.25], dtype=torch.float64))
    _, _, gradients = walk(model)
    for name in ("differential.query", "differential.key"):
        assert torch.isfinite(gradients[name]).all() and torch.count_nonzero(gradients[name])
