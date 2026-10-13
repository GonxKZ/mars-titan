"""PT4: memoria de mercado escrita una vez por sesión y su control sectorial, sin optimizadores.

Las pruebas calculan lecturas, escrituras y gradientes con tokens técnicos y con fichas de
sectores escritas a mano. Para ver el efecto del contexto se cargan valores fijos en la
ganancia de salida, sin ningún paso de ajuste.
"""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from test_financial_adapter import specification
from test_memory_stability import decision
from torch.nn import functional as F

from mars_titan.data.sector_map import SectorMap
from mars_titan.models.titans.config import MemoryConfig
from mars_titan.models.titans.financial import FinancialConfig, FinancialPredictor
from mars_titan.models.titans.local_control import MACProjectionConfig
from mars_titan.models.titans.neural_memory import NeuralMemory
from mars_titan.models.titans.session_context import (
    MARKET_MEMORY,
    SECTOR_MEAN,
    SessionContext,
)

ROOT = Path(__file__).resolve().parents[3]
DECLARATION = ROOT / "configs/evaluation/market-memory-comparison.json"
KINDS = (MARKET_MEMORY, SECTOR_MEAN)
DAY = 86_400_000_000
START = 1_609_459_200_000_000
SHARED = ("price_encoder", "encoders", "fusion", "head", "mac")


def sectors():
    groups = ("CN/unknown", "CN/银行", "US/Energy", "US/Technology", "US/unknown")
    flows = (("CN/600000.SS", 1), ("US/AAA", 3), ("US/BBB", 2), ("US/CCC", 3))
    return SectorMap(groups, flows, (("fixture.csv", "0" * 64),))


def context(kind=MARKET_MEMORY, *, hidden=8, seed=3, gain=None):
    module = SessionContext(
        kind,
        hidden,
        seed=seed,
        memory_options=dict(residual_layer_norm=True),
        sector_map=sectors() if kind == SECTOR_MEAN else None,
    ).double()
    if gain is not None:
        generator = torch.Generator().manual_seed(gain)
        with torch.no_grad():
            module.output.copy_(torch.randn(hidden, hidden, generator=generator).double())
    return module


def tokens(count, *, hidden=8, seed=11):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(count, hidden, generator=generator, dtype=torch.float64)


def shared_memory(*, dim=6, batch=1, **options):
    config = MemoryConfig(
        dim=dim, depth=2, max_batch=batch, max_tokens=4, residual_layer_norm=True, **options
    )
    return NeuralMemory(config, dtype=torch.float64)


def session(module, state, rows, flows, at, *, differentiable=False):
    """Leer una sesión en un solo bloque y confirmarla. Devuelve el contexto y el estado."""
    read, pending = module.read(rows, flows, (at,) * len(flows), state)
    return read, module.commit(pending, differentiable=differentiable)


def write_descriptions(root, us, cn):
    root.mkdir(parents=True, exist_ok=True)
    (root / "sp500stock_data_description.csv").write_text(
        "stock_name,Sector,Industry\n" + "".join(f"{s},{x},z\n" for s, x in us),
        encoding="utf-8-sig",
    )
    (root / "hs300stock_data_description.csv").write_bytes(
        ("证券代码,证券简称,申万一级行业\n" + "".join(f"{s},名,{x}\n" for s, x in cn)).encode(
            "gb18030"
        )
    )
    return root


def test_sector_map_reads_both_descriptions_and_never_invents_a_sector(tmp_path):
    root = write_descriptions(
        tmp_path / "descriptions",
        [("AAA", "Technology"), ("BBB", "N/A"), ("CCC", "Client Error"), ("DDD", "")],
        [("600000.SH", "银行"), ("000001.SZ", "")],
    )
    mapping = SectorMap.from_finmultitime(root)
    assert mapping.groups == ("CN/unknown", "CN/银行", "US/Technology", "US/unknown")
    group = {flow: mapping.groups[mapping.group(flow)] for flow in ("US/AAA", "CN/600000.SS")}
    assert group == {"US/AAA": "US/Technology", "CN/600000.SS": "CN/银行"}
    # Sin sector, con error de descarga o fuera de la ficha: el grupo desconocido del mercado.
    for flow in ("US/BBB", "US/CCC", "US/DDD", "US/ZZZ"):
        assert mapping.groups[mapping.group(flow)] == "US/unknown"
    assert mapping.groups[mapping.group("CN/000001.SZ")] == "CN/unknown"
    with pytest.raises(ValueError, match="US ni a CN"):
        mapping.group("EU/AAA")
    identity = mapping.identity()
    assert identity["point_in_time"] is False
    assert identity["assigned_flows"] == 6 and identity["groups"] == 4
    expected = hashlib.sha256((root / "sp500stock_data_description.csv").read_bytes()).hexdigest()
    assert identity["sources_sha256"]["sp500stock_data_description.csv"] == expected


def test_sector_map_rejects_conflicts_missing_columns_and_links(tmp_path):
    conflict = write_descriptions(tmp_path / "a", [("AAA", "Energy"), ("aaa", "Utilities")], [])
    with pytest.raises(ValueError, match="dos sectores"):
        SectorMap.from_finmultitime(conflict)
    columns = write_descriptions(tmp_path / "b", [("AAA", "Energy")], [])
    (columns / "sp500stock_data_description.csv").write_text("symbol,Sector\nAAA,Energy\n")
    with pytest.raises(ValueError, match="columnas"):
        SectorMap.from_finmultitime(columns)
    linked = write_descriptions(tmp_path / "c", [("AAA", "Energy")], [])
    target = linked / "sp500stock_data_description.csv"
    target.rename(linked / "real.csv")
    target.symlink_to(linked / "real.csv")
    with pytest.raises(ValueError, match="archivo regular"):
        SectorMap.from_finmultitime(linked)
    with pytest.raises(ValueError, match="coherente"):
        SectorMap(("US/unknown",), (("US/AAA", 1),), ())


def test_a_shared_write_of_one_row_is_the_ordinary_write_of_one_token():
    model = shared_memory()
    row = tokens(1, hidden=6)
    state = model.initial_state(1)
    expected = model.update(row.unsqueeze(1), state)
    actual = model.update_shared(row, state)
    for left, right in zip(
        (*actual.weights, *actual.momentum), (*expected.weights, *expected.momentum), strict=True
    ):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    assert actual.steps.tolist() == [1]
    query = F.normalize(tokens(4, hidden=6, seed=2), dim=-1)
    torch.testing.assert_close(
        model.read_shared(query, actual), model.read(query.unsqueeze(0), actual)[0], rtol=0, atol=0
    )


def test_a_shared_write_takes_the_mean_inner_gradient_and_the_gates_of_the_mean_logits():
    model = shared_memory()
    rows = tokens(5, hidden=6)
    state = model.initial_state(1)
    result = model.update_shared(rows, state)
    gradients = []
    for row in rows:
        local = tuple(w.detach().clone().requires_grad_(True) for w in state.weights)
        key = F.normalize(model.key_projection(row), dim=-1, eps=1e-12)
        value = model.value_projection(row)
        loss = (model._apply_memory(key.view(1, 1, -1), local).view(-1) - value).square().sum()
        gradients.append(torch.autograd.grad(loss, local))
    mean = [sum(parts) / len(rows) for parts in zip(*gradients, strict=True)]
    # Las proyecciones son lineales: los logits medios son los del token medio.
    average = rows.mean(dim=0, keepdim=True)
    alpha, eta, theta = model._gates(
        model.alpha_projection(average),
        model.eta_projection(average),
        model.theta_projection(average),
    )
    for index, gradient in enumerate(mean):
        momentum = eta * state.momentum[index] - theta * gradient
        torch.testing.assert_close(result.momentum[index], momentum)
        torch.testing.assert_close(
            result.weights[index], (1 - alpha) * state.weights[index] + momentum
        )
    shuffled = model.update_shared(rows[torch.tensor([3, 0, 4, 1, 2])], state)
    for left, right in zip(shuffled.weights, result.weights, strict=True):
        torch.testing.assert_close(left, right, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("case", ["window", "silu", "batch", "shape", "dtype", "empty"])
def test_the_shared_memory_rejects_per_flow_options_batches_and_other_shapes(case):
    options = dict(window=dict(qkv_convolution=4), silu=dict(qkv_silu=True)).get(case, {})
    model = shared_memory(batch=2 if case == "batch" else 1, **options)
    state = model.initial_state(2 if case == "batch" else 1)
    rows = tokens(3, hidden=6)
    rows = dict(shape=rows[:, :4], dtype=rows.float(), empty=rows[:0]).get(case, rows)
    with pytest.raises(ValueError):
        model.update_shared(rows, state)
    with pytest.raises(ValueError):
        model.read_shared(rows, state)


@pytest.mark.parametrize("kind", KINDS)
def test_a_session_reads_only_sessions_already_committed(kind):
    module = context(kind, gain=7)
    flows = ("CN/600000.SS", "US/AAA", "US/BBB", "US/CCC")
    state = module.initial_state("p")
    first, state = session(module, state, tokens(4), flows, START)
    changed = tokens(4)
    changed[2] += 4.0
    alone, after_change = session(module, module.initial_state("p"), changed, flows, START)
    # Otra fila de la misma sesión no cambia la lectura de una fila: solo su propia consulta.
    torch.testing.assert_close(alone[:2], first[:2], rtol=0, atol=0)
    if kind == SECTOR_MEAN:
        assert not first.any()
    second, _ = session(module, state, tokens(4, seed=5), flows, START + DAY)
    later, _ = session(module, after_change, tokens(4, seed=5), flows, START + DAY)
    # La sesión siguiente sí recibe la escritura de la fila cambiada.
    assert not torch.equal(second[1:], later[1:])
    torch.testing.assert_close(second[0], later[0], rtol=0, atol=0)
    assert state.committed_at == (START, START)


@pytest.mark.parametrize("kind", KINDS)
def test_the_block_partition_changes_neither_reads_nor_the_committed_state(kind):
    module = context(kind, gain=7)
    flows = ("US/AAA", "US/BBB", "US/CCC", "US/DDD")
    rows = tokens(4)
    state = module.initial_state("p")
    whole, committed = session(module, state, rows, flows, START)
    order = (2, 0, 3, 1)
    pending, parts = state, []
    for block in (order[:1], order[1:]):
        index = torch.tensor(block)
        read, pending = module.read(
            rows[index], tuple(flows[i] for i in block), (START,) * len(block), pending
        )
        parts.append((block, read))
    split = module.commit(pending)
    # Cada fila lee lo mismo salvo el redondeo de multiplicar por bloques de otro tamaño.
    for block, read in parts:
        torch.testing.assert_close(read, whole[torch.tensor(block)], rtol=1e-13, atol=1e-15)
    # La escritura recibe los mismos tokens en el mismo orden y coincide bit a bit.
    if kind == MARKET_MEMORY:
        for left, right in zip(split.memory[0].weights, committed.memory[0].weights, strict=True):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
    else:
        torch.testing.assert_close(split.means, committed.means, rtol=0, atol=0)


def test_reads_reject_mixed_sessions_repeated_flows_and_reopened_sessions():
    module = context()
    state = module.initial_state("p")
    rows = tokens(2)
    with pytest.raises(ValueError, match="una sesión"):
        module.read(rows, ("US/AAA", "US/BBB"), (START, START + DAY), state)
    # Dos mercados en el mismo bloque pueden tener instantes distintos.
    _, pending = module.read(rows, ("US/AAA", "CN/600000.SS"), (START, START + 1), state)
    with pytest.raises(ValueError, match="dos veces"):
        module.read(rows[:1], ("US/AAA",), (START,), pending)
    with pytest.raises(ValueError, match="pendiente"):
        module.read(rows[:1], ("US/BBB",), (START + DAY,), pending)
    with pytest.raises(ValueError, match="confirmadas"):
        module.detach(pending, "p")
    committed = module.commit(pending)
    with pytest.raises(ValueError, match="ya se confirmó"):
        module.read(rows[:1], ("US/BBB",), (START,), committed)
    with pytest.raises(ValueError, match="no corresponde"):
        module.validate(committed, "otra")


def test_the_sector_control_keeps_the_mean_token_of_each_sector_and_leaves_absent_ones():
    module = context(SECTOR_MEAN)
    rows = tokens(4)
    flows = ("US/AAA", "US/BBB", "US/CCC", "US/ZZZ")
    _, state = session(module, module.initial_state("p"), rows, flows, START)
    mapping = module.sector_map
    expected = torch.zeros(len(mapping.groups), 8, dtype=torch.float64)
    expected[mapping.group("US/AAA")] = (rows[0] + rows[2]) / 2
    expected[mapping.group("US/BBB")] = rows[1]
    expected[mapping.group("US/ZZZ")] = rows[3]
    torch.testing.assert_close(state.means, expected, rtol=0, atol=0)
    more = tokens(1, seed=4)
    _, later = session(module, state, more, ("US/BBB",), START + DAY)
    expected[mapping.group("US/BBB")] = more[0]
    torch.testing.assert_close(later.means, expected, rtol=0, atol=0)
    read, _ = module.read(more, ("US/CCC",), (START + 2 * DAY,), later)
    torch.testing.assert_close(read, expected[[mapping.group("US/CCC")]] @ module.output.T)


@pytest.mark.parametrize("kind", KINDS)
def test_export_and_restore_continue_bit_for_bit(kind):
    module = context(kind, gain=7)
    flows = ("CN/600000.SS", "US/AAA", "US/BBB")
    _, state = session(module, module.initial_state("p"), tokens(3), flows, START)
    payload = module.export_state(state)
    restored = module.restore_state(payload, "p", device=torch.device("cpu"))
    assert restored.committed_at == state.committed_at
    rows = tokens(3, seed=8)
    expected, continuous = session(module, state, rows, flows, START + DAY)
    actual, resumed = session(module, restored, rows, flows, START + DAY)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    if kind == MARKET_MEMORY:
        for left, right in zip(resumed.memory, continuous.memory, strict=True):
            pairs = zip(
                (*left.weights, *left.momentum), (*right.weights, *right.momentum), strict=True
            )
            for a, b in pairs:
                torch.testing.assert_close(a, b, rtol=0, atol=0)
    else:
        torch.testing.assert_close(resumed.means, continuous.means, rtol=0, atol=0)
    with pytest.raises(ValueError, match="no corresponde"):
        context(MARKET_MEMORY if kind == SECTOR_MEAN else SECTOR_MEAN).restore_state(
            payload, "p", device=torch.device("cpu")
        )


def test_a_later_read_differentiates_through_the_committed_write():
    module = context(gain=7)
    flows = ("US/AAA", "US/BBB")
    first = tokens(2).requires_grad_(True)
    _, state = session(module, module.initial_state("p"), first, flows, START, differentiable=True)
    read, _ = module.read(tokens(2, seed=6), flows, (START + DAY,) * 2, state)
    parameters = (first, module.query, module.output, module.memory.key_projection.weight)
    gradients = torch.autograd.grad(read.square().sum(), parameters)
    assert all(gradient.abs().sum() > 0 for gradient in gradients)
    cut = module.detach(state, "q")
    assert cut.parameter_id == "q"
    assert not any(weight.requires_grad for weight in cut.memory[0].weights)


@pytest.mark.parametrize("kind", KINDS)
def test_a_commit_without_gradient_keeps_no_graph_of_the_session_tokens(kind):
    module = context(kind, gain=7)
    rows = tokens(3).requires_grad_(True)
    _, state = session(
        module, module.initial_state("p"), rows, ("US/AAA", "US/BBB", "US/CCC"), START
    )
    kept = (
        [state.means]
        if kind == SECTOR_MEAN
        else [*state.memory[0].weights, *state.memory[0].momentum]
    )
    assert not any(value.requires_grad for value in kept)


def test_the_market_read_uses_the_own_query_normalized_and_the_output_gain():
    module = context(gain=7)
    flows = ("US/AAA", "US/BBB", "US/CCC")
    _, state = session(module, module.initial_state("p"), tokens(3), flows, START)
    rows = tokens(3, seed=4)
    read, _ = module.read(rows, flows, (START + DAY,) * 3, state)
    query = F.normalize(rows @ module.query.T, dim=-1, eps=1e-12)
    expected = module.memory.read_shared(query, state.memory[0]) @ module.output.T
    torch.testing.assert_close(read, expected, rtol=0, atol=0)


def pt4(kind=None, *, seed=42, **options):
    config = FinancialConfig(
        specification(),
        variant=options.pop("variant", "mac_online"),
        hidden_size=32,
        seed=seed,
        session_context=kind,
        sector_map=sectors() if kind == SECTOR_MEAN else None,
        **options,
    )
    return FinancialPredictor(config, dtype=torch.float64)


@pytest.mark.parametrize("kind", KINDS)
def test_the_predictor_starts_exactly_at_its_control_with_the_same_seed(kind):
    control, model = pt4(), pt4(kind)
    state = model.state_dict()
    for name, value in control.state_dict().items():
        if name.split(".", 1)[0] in SHARED:
            same = (
                torch.equal(state[name], value) if torch.is_tensor(value) else state[name] == value
            )
            assert same, name
    flows = ["US/AAA", "US/BBB", "US/CCC"]
    expected_state, actual_state = control.initial_state(flows), model.initial_state(flows)
    shared = model.initial_context()
    for event in range(3):
        batch = decision(model.config, flows, event)
        expected = control.prepare(batch, expected_state)
        actual = model.prepare(batch, actual_state, context=shared)
        torch.testing.assert_close(
            actual.point_predictions, expected.point_predictions, rtol=0, atol=0
        )
        torch.testing.assert_close(actual.detached_tokens, expected.detached_tokens, rtol=0, atol=0)
        expected_state, actual_state = expected.next_state, actual.next_state
        shared = model.commit_context(actual.next_context)
    assert shared.committed_at == (START + 2 * DAY, -1)
    assert "session_context" not in control.config.identity()
    assert model.config.identity()["session_context"]["kind"] == kind


def test_the_context_enters_the_token_before_the_memory_and_needs_its_state():
    model = pt4(MARKET_MEMORY)
    weights = model.state_dict()
    generator = torch.Generator().manual_seed(5)
    weights["session_context.output"] = 0.3 * torch.randn(32, 32, generator=generator).double()
    model.load_state_dict(weights)
    flows = ["US/AAA", "US/BBB"]
    shared = model.initial_context()
    for event in range(2):
        batch = decision(model.config, flows, event)
        prepared = model.prepare(batch, model.initial_state(flows), context=shared)
        shared = model.commit_context(prepared.next_context)
    # Con la misma semilla el control tiene los mismos pesos fuera del contexto.
    control = pt4()
    baseline = control.prepare(batch, control.initial_state(flows))
    assert not torch.equal(prepared.point_predictions, baseline.point_predictions)
    # El token separado sigue siendo el fusionado, sin el contexto de mercado.
    torch.testing.assert_close(prepared.detached_tokens, baseline.detached_tokens, rtol=0, atol=0)
    with pytest.raises(ValueError, match="contexto de sesión"):
        model.prepare(batch, model.initial_state(flows))
    with pytest.raises(ValueError, match="contexto de sesión"):
        control.prepare(batch, control.initial_state(flows), context=shared)


def test_declarations_fail_before_building(tmp_path):
    with pytest.raises(ValueError, match="mac_online"):
        pt4(MARKET_MEMORY, variant="mac_frozen")
    with pytest.raises(ValueError, match="session_context"):
        pt4("market_memory_v2")
    with pytest.raises(ValueError, match="sector_map"):
        FinancialConfig(specification(), variant="mac_online", session_context=SECTOR_MEAN)
    with pytest.raises(ValueError, match="sector_map"):
        FinancialConfig(
            specification(),
            variant="mac_online",
            session_context=MARKET_MEMORY,
            sector_map=sectors(),
        )
    with pytest.raises(ValueError, match="control local"):
        FinancialPredictor(pt4(MARKET_MEMORY).config, local_control=MACProjectionConfig())
    root = write_descriptions(tmp_path / "d", [("AAA", "Energy")], [("600000.SH", "银行")])
    config = FinancialConfig(
        specification(),
        variant="mac_online",
        session_context=SECTOR_MEAN,
        sector_map=dict(source=str(root)),
    )
    sectors_identity = config.identity()["session_context"]["sectors"]
    assert sectors_identity == config.sector_map.identity()
    with pytest.raises(ValueError, match="source"):
        FinancialConfig(
            specification(),
            variant="mac_online",
            session_context=SECTOR_MEAN,
            sector_map=dict(source=str(root), extra=1),
        )


def test_comparison_is_declared_with_both_controls_and_its_budget_but_not_executed():
    from mars_titan.training.financial_run import ChronologicalRecipe

    declaration = json.loads(DECLARATION.read_text())
    recipe = json.loads((ROOT / declaration["base_recipe"]).read_text())
    assert declaration["status"] == "declared_not_executed"
    assert declaration["final_test_opened"] is False and declaration["executions"] == 0
    base = {key: value for key, value in recipe["predictor"].items() if key != "dtype"}
    identities = {}
    for arm in declaration["arms"].values():
        options = base | arm["predictor_changes"]
        if "sector_map" in options:
            # La ficha real no forma parte del repositorio. Se usa el reparto de prueba.
            assert options["sector_map"] == {"source": "dataset"}
            options["sector_map"] = sectors()
        config = FinancialConfig(specification(), variant=declaration["variant"], **options)
        identities[arm["role"]] = config.identity()
    control = identities["control"]
    assert "session_context" not in control
    for role, kind in (("innovation", MARKET_MEMORY), ("trivial_alternative", SECTOR_MEAN)):
        changed = {key for key in identities[role] if identities[role][key] != control.get(key)}
        assert changed == {"session_context"}
        assert identities[role]["session_context"]["kind"] == kind
    # Los tres modelos comparten la receta sin acumulación, que PT4 necesita.
    common = recipe["recipe"] | declaration["common_recipe_changes"]
    assert ChronologicalRecipe(**common).accumulation_rows is None
    assert {"search_cases", "optimizer_updates", "stopping_rule", "seeds", "block_rows"} <= set(
        declaration["equal"]
    )
    budget = declaration["budget"]
    assert budget["consumed"] == dict(gpu_hours=0, executions=0, optimizer_updates=0)
    assert declaration["pilot"]["gpu_hours_max"] == budget["declared"]["gpu_hours_max"] == 8
    rule = declaration["decision_rule"]
    assert len(rule["keep_if"]) == 2 and all(term.endswith("< 0") for term in rule["keep_if"])


@pytest.fixture(scope="module")
def streams(tmp_path_factory):
    from tests.training.test_financial_run import corpus

    return corpus(tmp_path_factory.mktemp("pt4"))[1]


@pytest.fixture
def explicit_fastpath():
    previous = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    yield
    torch.backends.mha.set_fastpath_enabled(previous)


def corpus_predictor(streams, kind=None, *, gain=None):
    mapping = SectorMap(
        ("CN/unknown", "US/Energy", "US/unknown"),
        (("US/A0000", 1), ("US/A0001", 1)),
        (("fixture.csv", "0" * 64),),
    )
    config = FinancialConfig(
        streams["train"].specification(),
        variant="mac_online",
        hidden_size=32,
        seed=42,
        session_context=kind,
        sector_map=mapping if kind == SECTOR_MEAN else None,
    )
    model = FinancialPredictor(config, dtype=torch.float64)
    if gain is not None:
        weights = model.state_dict()
        generator = torch.Generator().manual_seed(gain)
        weights["session_context.output"] = 0.3 * torch.randn(32, 32, generator=generator).double()
        model.load_state_dict(weights)
    return model


def inference(model, *, block_rows=2, **options):
    from mars_titan.training.financial_run import ChronologicalInference, ChronologicalRecipe

    recipe = ChronologicalRecipe(
        truncation=3,
        epochs=1,
        block_rows=block_rows,
        selection=dict(metric="session_mae", patience=5, min_delta=0.0),
        **options,
    )
    return ChronologicalInference(model, recipe, audit=True)


def predictions(engine):
    return [entry for entry in engine.audit if entry[0] == "prediction"]


@pytest.mark.usefixtures("explicit_fastpath")
@pytest.mark.parametrize("kind", KINDS)
def test_the_validation_pass_commits_each_session_once_after_emitting_it(
    streams, kind, monkeypatch
):
    model = corpus_predictor(streams, kind)
    commits, original = [], model.session_context.commit

    def commit(state, **options):
        commits.append((state.pending_at, state.pending_flows))
        return original(state, **options)

    monkeypatch.setattr(model.session_context, "commit", commit)
    engine = inference(model)
    engine.evaluate(streams["validation"])
    moments = [pending[0] for pending, _ in commits]
    # Una escritura por instante, en orden, también en el calentamiento.
    assert moments == sorted(set(moments)) and len(moments) > 3
    assert all(pending[1] == -1 and not flows[1] for pending, flows in commits)
    committed = {pending[0]: sorted(flows[0]) for pending, flows in commits}
    emitted = {}
    for _, _, flow, at, _ in predictions(engine):
        emitted.setdefault(at, []).append(flow)
    # Cada escritura llega después de emitir todos los bloques de su instante, con sus flujos.
    assert emitted and all(committed[at] == sorted(flows) for at, flows in emitted.items())
    assert max(len(flows) for flows in emitted.values()) > 2
    # Con la ganancia en cero el recorrido emite exactamente lo mismo que su control.
    control = inference(corpus_predictor(streams))
    control.evaluate(streams["validation"])
    assert predictions(engine) == predictions(control)


@pytest.mark.usefixtures("explicit_fastpath")
@pytest.mark.parametrize("kind", KINDS)
def test_the_pass_does_not_depend_on_the_block_size(streams, kind, monkeypatch):
    emitted, written = [], []
    for rows in (1, 3):
        model = corpus_predictor(streams, kind, gain=5)
        states, original = [], model.session_context.commit

        def commit(state, *, original=original, states=states, **options):
            states.append(original(state, **options))
            return states[-1]

        monkeypatch.setattr(model.session_context, "commit", commit)
        engine = inference(model, block_rows=rows)
        engine.evaluate(streams["validation"])
        emitted.append(predictions(engine))
        written.append(states)
    one, three = emitted
    assert [entry[:4] for entry in one] == [entry[:4] for entry in three]
    torch.testing.assert_close(
        torch.tensor([entry[4] for entry in one]), torch.tensor([entry[4] for entry in three])
    )
    # El contexto confirmado tras cada sesión coincide salvo el redondeo de los tokens, que se
    # calculan por bloques de otro tamaño, y cambia de verdad a lo largo del recorrido.
    assert len(written[0]) == len(written[1])
    for left, right in zip(*written, strict=True):
        assert left.committed_at == right.committed_at
        for a, b in zip(context_tensors(left), context_tensors(right), strict=True):
            torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-12)
    first, last = context_tensors(written[0][0]), context_tensors(written[0][-1])
    assert any(not torch.equal(a, b) for a, b in zip(first, last, strict=True))


def context_tensors(state):
    if state.means is not None:
        return [state.means]
    return [value for memory in state.memory for value in (*memory.weights, *memory.momentum)]


@pytest.mark.usefixtures("explicit_fastpath")
def test_block_accumulation_is_rejected_with_pt4(streams):
    with pytest.raises(ValueError, match="acumulación"):
        inference(corpus_predictor(streams, MARKET_MEMORY), accumulation_rows=2)


@pytest.mark.usefixtures("explicit_fastpath")
@pytest.mark.parametrize("kind", KINDS)
def test_the_checkpoint_keeps_the_context_and_the_truncation_detaches_it(streams, kind, tmp_path):
    from mars_titan.training import financial_run
    from tests.training.test_financial_run import RecordingOptimizer

    def trainer(model, output):
        recipe = financial_run.ChronologicalRecipe(
            truncation=3,
            epochs=1,
            block_rows=2,
            selection=dict(metric="session_mae", patience=5, min_delta=0.0),
        )
        return financial_run.ChronologicalTrainer(
            model,
            recipe,
            train=streams["train"],
            validation=streams["validation"],
            output=output,
            optimizer_factory=RecordingOptimizer,
        )

    source = streams["validation"]
    events = list(source.batched_events(block_rows=2))
    middle = len(events) // 2
    engine = trainer(corpus_predictor(streams, kind, gain=5).eval(), tmp_path / "a")
    run = financial_run._Pass()
    with torch.no_grad():
        for event in events[:middle]:
            engine._labels(run, event, train=False)
            if event.inputs:
                engine._observe(run, source, event, differentiable=False)
        # Un paso cambiaría la identidad de los parámetros. Se simula con una identidad
        # anterior: sin etiquetas en la pérdida, la barrera solo corta y confirma la actual.
        run.context = replace(run.context, parameter_id="previous")
        engine._update(run, events[middle].at)
    assert engine.global_step == 0 and run.context.parameter_id == engine.predictor._parameter_id
    payload = engine._export(run)
    assert payload["context"]["committed_at"][0] == run.context.committed_at[0]
    resumed = trainer(corpus_predictor(streams, kind, gain=5).eval(), tmp_path / "b")
    restored = resumed._restore(payload)
    with torch.no_grad():
        for current, state in ((engine, run), (resumed, restored)):
            for event in events[middle:]:
                current._labels(state, event, train=False)
                if event.inputs:
                    current._observe(state, source, event, differentiable=False)
    assert run.errors.sessions == restored.errors.sessions
    for left, right in zip(
        context_tensors(run.context), context_tensors(restored.context), strict=True
    ):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    assert run.context.committed_at == restored.context.committed_at
    with pytest.raises(ValueError, match="contexto"):
        trainer(corpus_predictor(streams).eval(), tmp_path / "c")._restore(payload)
