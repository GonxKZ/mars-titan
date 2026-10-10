"""Proyecciones de la sección 4.4 de Titans, sin optimizador ni datos de campaña."""

import copy
import hashlib
import importlib
from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from mars_titan.models.titans.causal_convolution import CausalDepthwiseConvolution
from mars_titan.models.titans.config import PAPER_CONVOLUTION_KERNEL, PAPER_PROJECTIONS
from mars_titan.models.titans.state import mac_tensors, map_mac_rows
from mars_titan.models.titans.transition_jacobian import (
    fast_state_dimension,
    fast_state_jacobian,
    fast_state_point,
    fast_state_transition,
)

# Huellas FP32 capturadas en develop 030e7b31, antes de introducir SiLU y convolución.
BASELINE = {
    "core/disabled": dict(
        memory="63ecfe659331f93facc449d0e6b0aaf7682be5ef09897810a2f4f862a91dbd1e",
        mac="19ea5f4e6191ea58f6e5469cd73b72a29e64b6a157bfac3d18631ed6c1998353",
        parameters="d46b7bafe74a539947f3ef704cf46734b8ec363276cfe9b22ad5e2dec65784aa",
        outputs="8b9889fcd41350d431c6e4057907bbd3fb778b95ad90ecadeba3b37b7691a063",
        state="ea5ad699d70dc57bf7a1ce3fe8b11c6fb1e72316ff88c2c4044c4835c3a2c163",
        gradients="ea0b1c4e0a38ac13a6afda08596d1eeb31485b440f2f698cb58b78d394aa041c",
        names="391103611bbd11bd78a424e914b4090713fbebb56a15b67ad21f2167d6c83228",
    ),
    "core/frozen": dict(
        memory="63ecfe659331f93facc449d0e6b0aaf7682be5ef09897810a2f4f862a91dbd1e",
        mac="6296fc32383c46f5416352f62987e5417cc1b6cb19f8386cdfd44696119108aa",
        parameters="d46b7bafe74a539947f3ef704cf46734b8ec363276cfe9b22ad5e2dec65784aa",
        outputs="ff18ca9a8a957bf176a45c7623e2e20bd8d65240c2cbd6e59c0724c132aaa27b",
        state="ea5ad699d70dc57bf7a1ce3fe8b11c6fb1e72316ff88c2c4044c4835c3a2c163",
        gradients="febe9a8f6e6a7fa95b1ebc3d048149a248933c0e3b3cca3156c1d6b57b593612",
        names="391103611bbd11bd78a424e914b4090713fbebb56a15b67ad21f2167d6c83228",
    ),
    "core/online": dict(
        memory="63ecfe659331f93facc449d0e6b0aaf7682be5ef09897810a2f4f862a91dbd1e",
        mac="82ac7c1dde0f67ea832ed71a64a986f4ecbfcb909ff3b429be4b6d9934744503",
        parameters="d46b7bafe74a539947f3ef704cf46734b8ec363276cfe9b22ad5e2dec65784aa",
        outputs="82f8329f1ff9d57dc463af392994e68bedaf0a27c0ff603c17d2f159b4fb1186",
        state="89b8bb781f24ee9d28216d81207043f8c01eac2d2f41081b15a04c287ede1bba",
        gradients="21a5f4ea37fb56c941298a8f46328bc3a549bd7f721664699ab7168780f751e2",
        names="391103611bbd11bd78a424e914b4090713fbebb56a15b67ad21f2167d6c83228",
    ),
    "core/online_residual_gate_bias": dict(
        memory="2c484856ad7f5ac81d47357e017512f7e0ab23f868452311a1fb62aeee574760",
        mac="2928af4c9b79587f0a2682d93d917b5ab6df86d3ca495c33cd6deb57790919bf",
        parameters="13a162fd92caf41104cd27f56fed48b33b5828e567aedd1909720338e80f4342",
        outputs="867c7cd137637f6d29e614a923ee8d2cd5577baf6c1d2bef452490c4ff95be46",
        state="b2ec965acb1cad2b9beefab987a3ba028978250447e8d26adde9af3ca9393960",
        gradients="b71bdcfac6e3c7c70145fbe76542ef2da2524a6eb94857730f9735de8027ed7e",
        names="db76e52b1ee7202f145ea8c71f7138b9796516e51f5979ef37c2a9e27154d039",
    ),
    "core/online_without_l2": dict(
        memory="5fc35333717c6a523e49464ee3be79564e205cc64965e5fd7d07651d82d56818",
        mac="99da682ce8b02894f0c1e816d3526b33007f51b6886feac057a50049389e32ea",
        parameters="d46b7bafe74a539947f3ef704cf46734b8ec363276cfe9b22ad5e2dec65784aa",
        outputs="59aafdd4a9b6f311a44c28f251fcb1a3b84d5c70a2aeff0a2c13e4cde156f9a3",
        state="d669281b903df585c10b2116914b21a82c632fefbc3c825bae01c44a5225d097",
        gradients="310e64242830254b8cef7585cb5e6c465357a00236baab5948ff702ccc630e13",
        names="391103611bbd11bd78a424e914b4090713fbebb56a15b67ad21f2167d6c83228",
    ),
    "financial/mac_disabled": dict(
        identity="688eab085bc6e94fe70ff4469213dd4bac24a5d55afb833ee7e8bf4bcbe1478b",
        parameter_id="93940d37840ab5746da39b18964d89283eb59f4e74d5fba2496f075a5f4e5056",
        predictions="a8f3df213e815ad2280bd1536e34ef02ec9c55ae2272d3af016ffc868c7f03c7",
        state="490b3e1e8d90134f4ebaf54d7659a02c376957654d0f728ce197df417393cf52",
        gradients="f591026a0a944e8a0f20c26aac3aee7ff1bb41e23807f42912ffe173daec11ae",
    ),
    "financial/mac_frozen": dict(
        identity="4cfac779ecc2ce675f2f4e42f09615cb1afe3eef8860352d2e1518cf53d778e1",
        parameter_id="93940d37840ab5746da39b18964d89283eb59f4e74d5fba2496f075a5f4e5056",
        predictions="2f4555d30f903581fbf89f4fb0070673b371ee848f73e1d51bbf5044cf359017",
        state="490b3e1e8d90134f4ebaf54d7659a02c376957654d0f728ce197df417393cf52",
        gradients="ba56473b55f1c1b8acfd2bc9af32786eefc5337c08da698cf001b9d4bc459bbb",
    ),
    "financial/mac_online": dict(
        identity="585457bae7eb8afa326f3991c3aed54c43ac4b315bb2329ed982ff80ffa0809f",
        parameter_id="93940d37840ab5746da39b18964d89283eb59f4e74d5fba2496f075a5f4e5056",
        predictions="f25360ed13b8ccb6f3a3a9f32abd7038f1a04e70645e4e5ba14bcd616d4ee7ea",
        state="6a754dc68a816e5e52ecbf078cf46fb00e06138b2689b90f379a1dae17defe16",
        gradients="7da18b3077582dd8d5001d0890389cadb0145491f05168bae4c322a08a9651eb",
    ),
    "financial/mac_online_campaign_memory": dict(
        identity="4cf875a3e9f765dca59d1879d92e7db6a151e7ee86cba547ddc9d1bb121e1b3a",
        parameter_id="3c5bdc8a97396dd4097a6b57ad28ed2d61e42dc166028a1b1f9e0321f7a3653a",
        predictions="0737d95f13d1a195bd33223799e6f8479d727ab27aa83519b803394b20341f39",
        state="367d67d5f0d603ad95b3d9b6125908aacf6ad0e62b52fdf6ba475ab29aaa4367",
        gradients="aea542c67cb1476315d183cbd0c681da1669d422504d779f974cb7935b35bad2",
    ),
}


# 4 de enero de 2021 en UTC y un día en microsegundos.
START, DAY = 1_609_718_400_000_000, 86_400_000_000


def api():
    return importlib.import_module("mars_titan.models.titans")


def digest(values):
    result = hashlib.sha256()
    for value in values:
        array = value.detach().cpu().contiguous()
        result.update(str((tuple(array.shape), str(array.dtype))).encode())
        result.update(array.numpy().tobytes())
    return result.hexdigest()


def gradient_digest(parameters, gradients):
    return digest(
        [
            torch.zeros_like(parameter) if gradient is None else gradient
            for parameter, gradient in zip(parameters, gradients, strict=True)
        ]
    )


def strict_fp32():
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def core_trace(mode="online", **options):
    """Cuatro segmentos FP32 con lectura, escritura y gradientes de medida sin optimizador."""
    strict_fp32()
    package = api()
    memory = package.MemoryConfig(dim=8, depth=2, max_batch=2, max_tokens=3, **options)
    config = package.MACConfig(
        memory=memory, heads=2, persistent_tokens=2, max_segment=3, memory_mode=mode
    )
    module = package.TitansMAC(config)
    generator = torch.Generator().manual_seed(11)
    segments = [0.5 * torch.randn(2, 3, 8, generator=generator) for _ in range(4)]
    state = module.initial_state(2)
    outputs = []
    with torch.no_grad():
        for segment in segments:
            output, state = module(segment, state)
            outputs.append(output)
    graph_state, loss = module.initial_state(2, differentiable=True), 0
    for segment in segments:
        output, graph_state = module(segment, graph_state, differentiable=True)
        loss = loss + output.square().sum()
    names, parameters = zip(*module.named_parameters(), strict=True)
    gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
    return dict(
        memory=memory.fingerprint(),
        mac=config.fingerprint(),
        parameters=digest(parameters),
        outputs=digest(outputs),
        state=digest([*state.memory.weights, *state.memory.momentum, state.memory.steps]),
        gradients=gradient_digest(parameters, gradients),
        names=hashlib.sha256(",".join(names).encode()).hexdigest(),
    )


def financial_trace(variant, **options):
    """Tres decisiones FP32 por flujo, estado exportado y gradiente de las predicciones."""
    from test_financial_adapter import raw_batch, specification

    strict_fp32()
    module = importlib.import_module("mars_titan.models.titans.financial")
    spec = specification()
    config = module.FinancialConfig(spec, variant=variant, hidden_size=32, **options)
    model = module.FinancialPredictor(config)
    batches = [
        module.DecisionBatch.from_corpus(raw_batch(at=START + day * DAY), spec, dtype=torch.float32)
        for day in range(3)
    ]
    state, predictions = model.initial_state(batches[0].flow_ids), []
    with torch.no_grad():
        for batch in batches:
            prepared = model.prepare(batch, state)
            predictions.append(prepared.point_predictions)
            state = prepared.next_state
    exported = model.export_state(state)
    tensors = [exported["observed_steps"]]
    if exported["mac"] is not None:
        memory = exported["mac"]["memory"]
        tensors.extend((*memory["weights"], *memory["momentum"], memory["steps"]))
    graph, loss = model.initial_state(batches[0].flow_ids, differentiable=True), 0
    for batch in batches:
        prepared = model.prepare(batch, graph, differentiable=True)
        loss = loss + prepared.point_predictions.square().sum()
        graph = prepared.next_state
    parameters = [p for p in model.parameters()]
    gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
    return dict(
        identity=hashlib.sha256(module.canonical(config.identity()).encode()).hexdigest(),
        parameter_id=model._parameter_id,
        predictions=digest(predictions),
        state=digest(tensors),
        gradients=gradient_digest(parameters, gradients),
    )


CORE_CASES = {
    "online": dict(),
    "online_without_l2": dict(normalize_qk=False),
    "online_residual_gate_bias": dict(residual_layer_norm=True, gate_bias="default"),
    "frozen": dict(mode="frozen"),
    "disabled": dict(mode="disabled"),
}
# La receta histórica de la campaña declara gate_bias y la memoria residual.
CAMPAIGN_MEMORY = dict(
    gate_bias=dict(alpha_half_life=256.0, eta=0.5, theta=0.05), memory_residual_layer_norm=True
)
FINANCIAL_CASES = {
    "mac_online": ("mac_online", {}),
    "mac_online_campaign_memory": ("mac_online", CAMPAIGN_MEMORY),
    "mac_frozen": ("mac_frozen", {}),
    "mac_disabled": ("mac_disabled", {}),
}


def core_case(name):
    options = dict(CORE_CASES[name])
    if options.get("gate_bias") == "default":
        options["gate_bias"] = api().GateBias()
    return core_trace(**options)


@pytest.mark.parametrize("name", sorted(CORE_CASES))
def test_disabled_projections_keep_the_develop_core_bit_for_bit(name):
    assert core_case(name) == BASELINE[f"core/{name}"]


@pytest.mark.parametrize("name", sorted(FINANCIAL_CASES))
def test_disabled_projections_keep_the_develop_financial_predictor_bit_for_bit(name):
    variant, options = FINANCIAL_CASES[name]
    assert financial_trace(variant, **options) == BASELINE[f"financial/{name}"]


# Núcleo nuevo: SiLU, convolución causal de núcleo 4 y L2 de q y k.
PAPER = dict(normalize_qk=True, qkv_silu=True, qkv_convolution=4)


def paper_memory(dim=8, *, batch=2, tokens=3, **options):
    return api().MemoryConfig(
        dim=dim, depth=2, max_batch=batch, max_tokens=tokens, **(PAPER | options)
    )


def paper_mac(dim=8, *, batch=2, mode="online", dtype=torch.float32, **options):
    package = api()
    memory = paper_memory(dim, batch=batch, **options)
    config = package.MACConfig(
        memory=memory, heads=2, persistent_tokens=2, max_segment=3, memory_mode=mode
    )
    return package.TitansMAC(config, dtype=dtype)


def reference_convolution(weight, sequence):
    """nn.Conv1d en profundidad con relleno K − 1 y recorte causal, como ShortConvolution."""
    kernel = weight.shape[-1]
    padded = F.pad(sequence.transpose(1, 2), (kernel - 1, 0))
    return F.conv1d(padded, weight, groups=weight.shape[0]).transpose(1, 2)


def state_tensors(state):
    return mac_tensors(state)


def assert_bits(left, right):
    left, right = list(left), list(right)
    assert len(left) == len(right)
    for first, second in zip(left, right, strict=True):
        assert first.dtype == second.dtype and torch.equal(first, second)


def segments(count, *, batch=2, dim=8, seed=5, dtype=torch.float32):
    generator = torch.Generator().manual_seed(seed)
    return [
        0.5 * torch.randn(batch, 3, dim, generator=generator, dtype=dtype) for _ in range(count)
    ]


def test_paper_identity_names_the_three_components_and_keeps_previous_identities():
    package = api()
    identity = package.MemoryConfig(dim=8, **PAPER).identity()
    assert identity["projections"] == PAPER_PROJECTIONS == "titans_mac_paper_projections_v2"
    assert identity["qkv_activation"]["function"] == "silu"
    assert identity["qkv_convolution"]["kernel"] == PAPER_CONVOLUTION_KERNEL == 4
    assert identity["qkv_convolution"]["bias"] is False
    assert identity["projection_order"] == "linear_convolution_silu_then_l2_query_key"
    assert identity["normalize_qk"] is True
    previous = package.MemoryConfig(dim=8).identity()
    for key in ("projections", "qkv_activation", "qkv_convolution", "qkv_silu"):
        assert key not in previous
    options = [
        {},
        dict(qkv_silu=True),
        dict(qkv_convolution=4),
        PAPER,
        PAPER | dict(normalize_qk=False),
        PAPER | dict(qkv_convolution=3),
    ]
    configs = [package.MemoryConfig(dim=8, **option) for option in options]
    assert len({config.fingerprint() for config in configs}) == len(options)
    named = [config for config in configs if "projections" in config.identity()]
    assert named == [package.MemoryConfig(dim=8, **PAPER)]


@pytest.mark.parametrize(
    "options",
    [
        dict(qkv_silu=1),
        dict(qkv_silu=None),
        dict(qkv_convolution=1),
        dict(qkv_convolution=9),
        dict(qkv_convolution=-1),
        dict(qkv_convolution=True),
        dict(qkv_convolution=False),
        dict(qkv_convolution=4.0),
        dict(qkv_convolution=0.0),
    ],
)
def test_projection_fields_reject_ambiguous_values(options):
    with pytest.raises(ValueError):
        api().MemoryConfig(dim=4, **options)


def test_paper_components_add_three_kernels_after_the_previous_draws():
    rng = torch.get_rng_state().clone()
    previous = paper_mac(qkv_silu=False, qkv_convolution=0).state_dict()
    current = paper_mac().state_dict()
    assert torch.equal(torch.get_rng_state(), rng)
    added = set(current) - set(previous)
    assert added == {
        "memory.key_convolution.weight",
        "memory.value_convolution.weight",
        "query_convolution.weight",
    }
    for name, value in previous.items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(current[name], value)
    kernels = [current[name] for name in sorted(added)]
    for kernel in kernels:
        assert kernel.shape == (8, 1, 4) and kernel.abs().max() <= 0.5
    assert not torch.equal(kernels[0], kernels[1]) and not torch.equal(kernels[1], kernels[2])


def test_convolution_matches_causal_conv1d_and_gives_the_same_bits_in_chunks():
    strict_fp32()
    generator = torch.Generator().manual_seed(3)
    module = CausalDepthwiseConvolution(6, 4, dtype=torch.float64)
    sequence = torch.randn(2, 11, 6, generator=generator, dtype=torch.float64)
    zeros = torch.zeros(2, 3, 6, dtype=torch.float64)
    whole, window = module(sequence, zeros)
    expected = reference_convolution(module.weight, sequence)
    torch.testing.assert_close(whole, expected, rtol=1e-13, atol=1e-14)
    assert torch.equal(window, sequence[:, -3:])
    module = module.float()
    sequence, zeros = sequence.float(), zeros.float()
    whole, window = module(sequence, zeros)
    for sizes in ([1] * 11, [2, 5, 4], [3, 8], [10, 1]):
        outputs, current = [], zeros
        for piece in sequence.split(sizes, dim=1):
            output, current = module(piece, current)
            outputs.append(output)
        assert torch.equal(torch.cat(outputs, dim=1), whole)
        assert torch.equal(current, window)
    perturbed = sequence.clone()
    perturbed[:, 7:] += 5
    future, _ = module(perturbed, zeros)
    assert torch.equal(future[:, :7], whole[:, :7])
    assert not torch.equal(future[:, 7:], whole[:, 7:])


def test_memory_update_writes_silu_of_the_causal_convolution_with_l2_keys():
    package = api()
    config = paper_memory(4, batch=1)
    memory = package.NeuralMemory(config, dtype=torch.float64)
    generator = torch.Generator().manual_seed(9)
    observed = 0.4 * torch.randn(1, 3, 4, generator=generator, dtype=torch.float64)
    state = memory.initial_state(1)
    actual = memory.update(observed, state)
    with torch.no_grad():
        projected_keys = memory.key_projection(observed)
        projected_values = memory.value_projection(observed)
        keys = F.normalize(
            F.silu(reference_convolution(memory.key_convolution.weight, projected_keys)),
            dim=-1,
            eps=1e-12,
        )
        values = F.silu(reference_convolution(memory.value_convolution.weight, projected_values))
    weights, momentum = state.weights, state.momentum
    for index in range(3):
        token = observed[:, index]
        with torch.no_grad():
            alpha = memory.alpha_projection(token).sigmoid().unsqueeze(-1)
            eta = memory.eta_projection(token).sigmoid().unsqueeze(-1)
            theta = config.theta_max * memory.theta_projection(token).sigmoid().unsqueeze(-1)
        local = tuple(weight.detach().clone().requires_grad_(True) for weight in weights)
        hidden = F.gelu(keys[:, index : index + 1] @ local[0].transpose(1, 2))
        residual = (hidden @ local[1].transpose(1, 2)).squeeze(1) - values[:, index]
        gradients = torch.autograd.grad(residual.square().sum(), local)
        momentum = tuple(
            eta * previous - theta * gradient
            for previous, gradient in zip(momentum, gradients, strict=True)
        )
        weights = tuple(
            (1 - alpha) * weight + surprise
            for weight, surprise in zip(weights, momentum, strict=True)
        )
    for left, right in zip(actual.weights + actual.momentum, weights + momentum, strict=True):
        torch.testing.assert_close(left, right, rtol=1e-12, atol=1e-13)
    # La ventana guarda las proyecciones sin convolución. El núcleo proyecta token a token.
    torch.testing.assert_close(actual.convolution[0], projected_keys, rtol=1e-14, atol=1e-15)
    torch.testing.assert_close(actual.convolution[1], projected_values, rtol=1e-14, atol=1e-15)


def test_mac_reads_with_the_l2_of_silu_of_the_convolved_query(monkeypatch):
    mac = paper_mac(4, batch=1, dtype=torch.float64)
    queries, original = [], mac.memory.read

    def spy(query, state):
        queries.append(query.detach().clone())
        return original(query, state)

    monkeypatch.setattr(mac.memory, "read", spy)
    pieces = segments(2, batch=1, dim=4, dtype=torch.float64)
    state = mac.initial_state(1)
    for piece in pieces:
        _, state = mac(piece, state)
    sequence = torch.cat(pieces, dim=1)
    with torch.no_grad():
        projected = mac.query_projection(sequence)
        expected = F.normalize(
            F.silu(reference_convolution(mac.query_convolution.weight, projected)),
            dim=-1,
            eps=1e-12,
        )
    # Cada segmento lee con q y después con la salida y al cierre.
    torch.testing.assert_close(torch.cat(queries[::2], dim=1), expected, rtol=1e-12, atol=1e-13)
    torch.testing.assert_close(state.convolution[0], projected[:, -3:], rtol=1e-14, atol=1e-15)


def test_memory_update_in_chunks_gives_the_same_bits_as_one_call():
    strict_fp32()
    memory = api().NeuralMemory(paper_memory(tokens=6))
    generator = torch.Generator().manual_seed(13)
    observed = 0.5 * torch.randn(2, 6, 8, generator=generator)
    whole = memory.update(observed, memory.initial_state(2))
    for sizes in ([1] * 6, [2, 4], [3, 1, 2], [5, 1]):
        state = memory.initial_state(2)
        for piece in observed.split(sizes, dim=1):
            state = memory.update(piece, state)
        assert_bits(
            (*state.weights, *state.momentum, state.steps, *state.convolution),
            (*whole.weights, *whole.momentum, whole.steps, *whole.convolution),
        )


def test_future_segments_do_not_change_past_outputs_or_windows():
    strict_fp32()
    mac = paper_mac()
    pieces = segments(4)
    state, outputs, states = mac.initial_state(2), [], []
    for piece in pieces:
        output, state = mac(piece, state)
        outputs.append(output)
        states.append(state)
    changed = [*pieces[:2], pieces[2] + 3, -pieces[3]]
    state = mac.initial_state(2)
    for index, piece in enumerate(changed):
        output, state = mac(piece, state)
        if index < 2:
            assert torch.equal(output, outputs[index])
            assert_bits(state_tensors(state), state_tensors(states[index]))
        else:
            assert not torch.equal(output, outputs[index])


def test_restoring_the_mac_state_mid_sequence_reproduces_the_same_bits():
    strict_fp32()
    mac = paper_mac()
    pieces = segments(4)
    state, outputs = mac.initial_state(2), []
    for piece in pieces:
        output, state = mac(piece, state)
        outputs.append(output)
    middle = mac.initial_state(2)
    for piece in pieces[:2]:
        _, middle = mac(piece, middle)
    payload = mac.export_state(middle)
    assert set(payload) == {"schema_version", "configuration", "memory", "convolution"}
    assert "convolution" in payload["memory"]
    restored_model = paper_mac()
    restored_model.load_state_dict(mac.state_dict())
    restored = restored_model.restore_state(copy.deepcopy(payload))
    resumed = []
    for piece in pieces[2:]:
        output, restored = restored_model(piece, restored)
        resumed.append(output)
    assert_bits(resumed, outputs[2:])
    assert_bits(state_tensors(restored), state_tensors(state))
    incomplete = dict(payload)
    incomplete.pop("convolution")
    with pytest.raises(ValueError):
        mac.restore_state(incomplete)


@pytest.mark.parametrize("mode", ["frozen", "disabled"])
def test_windows_only_advance_for_the_projections_each_mode_computes(mode):
    mac = paper_mac(4, batch=1, mode=mode, dtype=torch.float64)
    piece = segments(1, batch=1, dim=4, dtype=torch.float64)[0]
    initial = mac.initial_state(1)
    _, state = mac(piece, initial)
    for window in state.memory.convolution:
        assert torch.equal(window, torch.zeros_like(window))
    with torch.no_grad():
        expected = mac.query_projection(piece) if mode == "frozen" else initial.convolution[0]
    assert torch.equal(state.convolution[0], expected)
    assert state.convolution[0].data_ptr() != initial.convolution[0].data_ptr()


def test_windows_stay_isolated_per_flow():
    strict_fp32()
    mac = paper_mac()
    pieces = segments(3)
    changed = [piece.clone() for piece in pieces]
    for piece in changed:
        piece[1] = piece[1] * -2 + 1
    left, right = mac.initial_state(2), mac.initial_state(2)
    for original, other in zip(pieces, changed, strict=True):
        first, left = mac(original, left)
        second, right = mac(other, right)
        assert torch.equal(first[0], second[0]) and not torch.equal(first[1], second[1])
    for one, two in zip(state_tensors(left), state_tensors(right), strict=True):
        assert torch.equal(one[0], two[0])


def test_state_validation_rejects_missing_misshaped_shared_or_non_finite_windows():
    mac = paper_mac()
    piece = segments(1)[0]
    state = mac.initial_state(2)
    memory = state.memory
    key, value = memory.convolution
    query = state.convolution[0]
    invalid = [
        replace(state, convolution=()),
        replace(state, convolution=[query]),
        replace(state, memory=replace(memory, convolution=(key,))),
        replace(state, memory=replace(memory, convolution=(key, torch.zeros(2, 2, 8)))),
        replace(state, memory=replace(memory, convolution=(key, key))),
        replace(state, convolution=(key,)),
        replace(state, convolution=(torch.full_like(query, float("nan")),)),
        replace(state, convolution=(query.double(),)),
        replace(state, convolution=(torch.zeros(2, 8, 3).transpose(1, 2),)),
    ]
    for candidate in invalid:
        with pytest.raises(ValueError):
            mac(piece, candidate)
    package = api()
    previous = package.TitansMAC(
        package.MACConfig(
            memory=package.MemoryConfig(dim=8, max_batch=2, max_tokens=3),
            heads=2,
            persistent_tokens=2,
            max_segment=3,
        )
    )
    with pytest.raises(ValueError):
        previous(piece, replace(previous.initial_state(2), convolution=(query,)))


def test_state_budget_counts_the_three_windows():
    package = api()
    weights = 2 * (2 * 2 * 8 * 8 * 4 + 8)
    windows = 2 * 3 * 3 * 8 * 4
    exact = package.NeuralMemory(paper_memory(max_state_bytes=weights + windows))
    exact.initial_state(2)
    short = package.NeuralMemory(paper_memory(max_state_bytes=weights + windows - 1))
    with pytest.raises(ValueError):
        short.initial_state(2)


def paper_predictor(variant="mac_online", **options):
    from test_financial_adapter import specification

    module = importlib.import_module("mars_titan.models.titans.financial")
    spec = specification()
    config = module.FinancialConfig(
        spec, variant=variant, hidden_size=32, memory_projections=PAPER_PROJECTIONS, **options
    )
    return module, spec, module.FinancialPredictor(config)


def decisions(module, spec, count, *, scale=1.0):
    from test_financial_adapter import raw_batch

    result = []
    for day in range(count):
        raw = raw_batch(at=START + day * DAY)
        raw["inputs"] = {name: value * scale for name, value in raw["inputs"].items()}
        result.append(module.DecisionBatch.from_corpus(raw, spec, dtype=torch.float32))
    return result


def test_financial_switch_builds_the_paper_core_with_a_new_identity():
    module, spec, model = paper_predictor()
    memory = model.mac.config.memory
    assert (memory.qkv_silu, memory.qkv_convolution, memory.normalize_qk) == (True, 4, True)
    assert memory.identity()["projections"] == PAPER_PROJECTIONS
    assert model.config.identity()["memory_projections"] == PAPER_PROJECTIONS
    default = module.FinancialConfig(spec, variant="mac_online", hidden_size=32)
    assert "memory_projections" not in default.identity()
    for value in ("paper", None, 1, "linear_v2"):
        with pytest.raises(ValueError):
            module.FinancialConfig(spec, hidden_size=32, memory_projections=value)
    assert model._tensor_bytes_per_flow() == (4 * 32 + 3 * 3) * 32 * 4 + 16
    state = model.initial_state(("US/AAA", "US/BBB"))
    usage = model.state_usage(state)
    assert usage["tensor_bytes"] == 2 * model._tensor_bytes_per_flow()


@pytest.mark.parametrize("variant", ["mac_online", "mac_frozen", "mac_disabled"])
def test_financial_restore_mid_sequence_reproduces_the_same_bits(variant):
    strict_fp32()
    module, spec, model = paper_predictor(variant, **CAMPAIGN_MEMORY)
    batches = decisions(module, spec, 4)
    state, expected = model.initial_state(batches[0].flow_ids), []
    with torch.no_grad():
        for index, batch in enumerate(batches):
            prepared = model.prepare(batch, state)
            expected.append(prepared.point_predictions)
            state = prepared.next_state
            if index == 1:
                middle = state
    final = state
    for restore in (
        lambda value: model.restore_state(copy.deepcopy(model.export_state(value))),
        lambda value: model.restore_state(model.export_state_cpu(value), device="cpu"),
    ):
        state, resumed = restore(middle), []
        with torch.no_grad():
            for batch in batches[2:]:
                prepared = model.prepare(batch, state)
                resumed.append(prepared.point_predictions)
                state = prepared.next_state
        assert_bits(resumed, expected[2:])
        assert_bits(state_tensors(state.mac), state_tensors(final.mac))


def test_financial_selection_gathering_and_trainer_rows_carry_each_flow_window():
    from mars_titan.training.financial_run import FlowStates

    module, spec, model = paper_predictor()
    state = model.initial_state(("US/AAA", "US/BBB"))
    with torch.no_grad():
        for batch in decisions(module, spec, 2):
            state = model.prepare(batch, state).next_state
    windows = [*state.mac.memory.convolution, *state.mac.convolution]
    assert all(not torch.equal(window[0], window[1]) for window in windows)
    reverse = ("US/BBB", "US/AAA")
    selected = model.select_state(state, reverse)
    payload = model.export_state_cpu(state)
    gathered = model.gather_state(
        {flow: (payload, 1 - index) for index, flow in enumerate(reverse)}, reverse
    )
    # El entrenador guarda cada flujo en su fila del bloque y lo reúne en otro orden.
    flows = FlowStates()
    flows.put(state, detach=True)
    stacked = flows.gather(reverse)
    for candidate in (selected, gathered, stacked):
        assert candidate.flow_ids == reverse
        for original, value in zip(
            state_tensors(state.mac), state_tensors(candidate.mac), strict=True
        ):
            assert torch.equal(value, original.flip(0))


def test_dense_jacobian_with_windows_matches_differences_and_shifts_the_query_window():
    package = api()
    mac = package.TitansMAC(
        package.MACConfig(
            memory=package.MemoryConfig(dim=2, max_tokens=1, max_batch=4, **PAPER),
            persistent_tokens=2,
            max_segment=1,
        ),
        dtype=torch.float64,
    )
    tokens = torch.linspace(-0.4, 0.5, 8, dtype=torch.float64).reshape(1, 4, 2)
    state = mac.initial_state(1)
    with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
        for index in range(3):
            _, state = mac(tokens[:, index : index + 1], state)
    token = tokens[:, 3:]
    order = fast_state_dimension(mac.config)
    assert order == 2 * 2 * 4 + 3 * 3 * 2
    jacobian = fast_state_jacobian(mac, token, state)
    assert jacobian.shape == (order, order)
    point = fast_state_point(state).detach()
    transition = fast_state_transition(mac, token, state)
    generator = torch.Generator().manual_seed(5)
    for _ in range(3):
        direction = torch.randn(point.shape, generator=generator, dtype=torch.float64)
        step = 1e-6
        with torch.enable_grad(), sdpa_kernel(SDPBackend.MATH):
            numeric = (
                transition(point + step * direction) - transition(point - step * direction)
            ) / (2 * step)
        torch.testing.assert_close(jacobian @ direction, numeric, rtol=1e-6, atol=1e-8)
    # La ventana de q se desplaza y su entrada nueva W_Q x no depende de z.
    shift = torch.zeros(6, order, dtype=torch.float64)
    shift[0:4, order - 4 : order] = torch.eye(4, dtype=torch.float64)
    assert torch.equal(jacobian[-6:], shift)
    assert jacobian[16:28].abs().sum() > 0


def test_local_control_projects_the_jacobian_with_windows():
    package = api()
    control_api = importlib.import_module("mars_titan.models.titans.local_control")
    mac = package.TitansMAC(
        package.MACConfig(
            memory=package.MemoryConfig(dim=2, max_tokens=1, max_batch=4, **PAPER),
            persistent_tokens=2,
            max_segment=1,
        ),
        dtype=torch.float64,
    )
    control = control_api.MACProjectionControl(
        control_api.MACProjectionConfig(mode="diagnostic", rank=3, frequency=1, grid_size=13),
        mac.config,
        dtype=torch.float64,
    )
    assert control.state_dimension == fast_state_dimension(mac.config) == 34
    assert control.get_extra_state()["fast_state_layout"] == (
        "weights_momentum_then_key_value_query_windows"
    )
    token = torch.tensor([[[0.2, -0.1]], [[0.3, 0.4]]], dtype=torch.float64)
    state = mac.initial_state(2)
    with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
        _, state = mac(token, state)
    flows = ("flow-0", "flow-1")
    counters = state.memory.steps.detach().cpu()
    selection = control.select_flows(flows, counters, context_id="a" * 64)
    result = control(
        mac,
        token,
        state,
        flow_ids=flows,
        observed_steps=counters,
        selection=selection,
        context_id="a" * 64,
    )
    index = result.indices[0]
    single = map_mac_rows(state, lambda value: value[index : index + 1].clone())
    jacobian = fast_state_jacobian(mac, token[index : index + 1], single)
    torch.testing.assert_close(
        result.operators[0], control.basis.T @ jacobian @ control.basis, rtol=1e-11, atol=1e-12
    )


def test_windows_keep_a_graph_only_in_differentiable_calls():
    mac = paper_mac()
    piece = segments(1)[0]
    _, detached = mac(piece, mac.initial_state(2))
    assert all(
        value.grad_fn is None and not value.requires_grad for value in state_tensors(detached)
    )
    _, graph = mac(piece, mac.initial_state(2, differentiable=True), differentiable=True)
    windows = [*graph.memory.convolution, *graph.convolution]
    assert all(window.grad_fn is not None for window in windows)
    gradient = torch.autograd.grad(graph.convolution[0].sum(), mac.query_projection.weight)[0]
    assert gradient.abs().sum() > 0


def test_financial_state_checks_reject_a_foreign_query_window():
    _, _, model = paper_predictor()
    state = model.initial_state(("US/AAA", "US/BBB"))
    query = state.mac.convolution[0]
    for window in (torch.zeros(2, 2, 32), query[:1].clone(), query.double()):
        with pytest.raises(ValueError):
            model.state_usage(replace(state, mac=replace(state.mac, convolution=(window,))))


def test_trainer_checkpoint_round_trip_keeps_the_windows_of_every_flow():
    import io

    from mars_titan.training.financial_run import FlowStates

    module, spec, model = paper_predictor()
    state = model.initial_state(("US/AAA", "US/BBB"))
    with torch.no_grad():
        for batch in decisions(module, spec, 3):
            state = model.prepare(batch, state).next_state
    flows = FlowStates()
    flows.put(state, detach=True)
    # Mismo recorrido que el punto de control del entrenador, con torch.save y weights_only.
    stream = io.BytesIO()
    torch.save(model.export_state_cpu(flows.gather(state.flow_ids)), stream)
    stream.seek(0)
    payload = torch.load(stream, map_location="cpu", weights_only=True)
    restored = FlowStates()
    restored.put(model.restore_state(payload, device="cpu"))
    for flow in state.flow_ids:
        assert_bits(
            state_tensors(restored.gather([flow]).mac), state_tensors(flows.gather([flow]).mac)
        )


def test_frozen_consumer_admits_the_convolution_and_reproduces_the_predictor():
    frozen_api = importlib.import_module("mars_titan.models.titans.frozen_financial")
    module, spec, model = paper_predictor()
    model.eval().requires_grad_(False)
    previous = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        engine = frozen_api.FrozenFinancialConsumer(model)
        state = model.initial_state(("US/AAA", "US/BBB"))
        for batch in decisions(module, spec, 2):
            expected = model.prepare(batch, state)
            result = engine.prepare(batch, state, context_id="c" * 64)
            assert torch.equal(result.point_predictions, expected.point_predictions)
            assert_bits(
                state_tensors(result.next_state.mac), state_tensors(expected.next_state.mac)
            )
            state = result.next_state
    finally:
        torch.backends.mha.set_fastpath_enabled(previous)
    code = engine.identity()["implementation"]
    assert "mars_titan.models.titans.causal_convolution" in code
