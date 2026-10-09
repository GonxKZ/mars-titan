"""Proyecciones de la sección 4.4 de Titans, sin optimizador ni datos de campaña."""

import hashlib
import importlib

import pytest
import torch

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
