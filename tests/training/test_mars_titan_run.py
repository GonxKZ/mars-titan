"""Lector episódico de MARS-TITAN sobre Titans-MAC congelado, sin pasos que ajusten pesos.

El optimizador inyectado registra gradientes y llamadas sin heredar de Optimizer. Las
pruebas recorren el bucle hasta el paso, comprueban el orden de cada evento, la causalidad,
el aislamiento del estado rápido del padre y la reanudación desde checkpoints coherentes.
"""

import copy
import json
import os
from pathlib import Path

import pytest
import torch

from mars_titan.memory.episodic_codec import FrozenEpisodeCodec
from mars_titan.memory.native_backend import load_native
from mars_titan.models.titans.episodic_readout import EpisodicReadout, EpisodicReadoutConfig
from mars_titan.models.titans.financial import FinancialConfig, FinancialPredictor
from mars_titan.models.titans.local_control import MACProjectionConfig
from mars_titan.training import mars_titan_run as mt
from mars_titan.training import titans_walk_forward
from mars_titan.training.checkpoints import load_training_state
from mars_titan.training.financial_run import ChronologicalInference, ChronologicalRecipe
from mars_titan.training.learning_hold import LearningHoldError
from mars_titan.training.selection import FIXED_BUDGET
from tests.training.chronological_fixture import decision
from tests.training.test_financial_run import RecordingOptimizer, StopAtStep, corpus, entries
from tests.training.test_financial_run import shared as shared

DECLARED = Path("configs/titans/episodic-readout-historical-masked.json")
TITANS = Path("configs/titans/chronological-training-historical-masked.json")
SELECTION = dict(metric="session_mae", patience=5, min_delta=0.0, stopping=FIXED_BUDGET)


@pytest.fixture(autouse=True)
def explicit_fastpath():
    previous = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    yield
    torch.backends.mha.set_fastpath_enabled(previous)


@pytest.fixture(scope="module")
def native():
    path = os.environ.get("MARS_TITAN_EPISODIC_NATIVE")
    if not path:
        pytest.skip("Falta el enlace nativo CPU compilado para esta comprobación")
    return load_native(path)


def parent(streams, *, head="quantile_head_v1", variant="mac_online", **options):
    config = FinancialConfig(
        streams["train"].specification(), variant=variant, hidden_size=32, seed=42, head=head
    )
    model = FinancialPredictor(config, dtype=torch.float64, **options)
    return model.eval().requires_grad_(False)


def recipe(**changes):
    options = dict(update_instants=3, epochs=1, block_rows=2, bank_capacity=8, selection=SELECTION)
    return mt.ReadoutRecipe(**(options | changes))


def reader(streams, admission, *, refinements=1, episodes="per_step", seed=42, **options):
    codec = FrozenEpisodeCodec(streams["train"].specification())
    config = EpisodicReadoutConfig(
        codec.fingerprint(),
        hidden_size=32,
        refinements=refinements,
        mode="no_bank" if admission == "m0" else "bank",
        max_working_bytes=128 * 1024**2,
        seed=seed,
        episode_selection=episodes,
        **options,
    )
    return EpisodicReadout(config, dtype=torch.float64), codec


def build(streams, output, native, *, admission="m1", model=None, plan=None, **options):
    plan = plan or recipe()
    factory = options.pop("factory", RecordingOptimizer)
    readout, codec = reader(streams, admission, **options)
    return mt.ReadoutTrainer(
        model or parent(streams),
        readout,
        plan,
        admission=admission,
        retention=mt.retention_config(plan, admission),
        native=native,
        codec=codec,
        train=streams["train"],
        validation=streams["validation"],
        output=output,
        world="fixture",
        fold="0",
        optimizer_factory=factory,
        audit=True,
    )


def gradients(engine):
    """Gradientes de cada paso por nombre del parámetro del lector."""
    names = {id(value): name for name, value in engine.readout.named_parameters()}
    return [
        {names[key]: value for key, value in record.items()} for record in engine.optimizer.records
    ]


def test_declared_recipe_searches_two_rates_with_the_titans_budget_and_segments():
    document = mt.load_recipe(DECLARED)
    titans = json.loads(TITANS.read_text())
    cases = document["walk_forward"]["search_cases"]
    assert document["status"] == "declared_not_executed"
    assert cases == titans["walk_forward"]["search_cases"]
    assert mt.SEARCHED == titans_walk_forward.SEARCHED
    for name, case in cases.items():
        plan = mt.case_recipe(document, name)
        assert plan.learning_rate == case["learning_rate"]
        assert plan.epochs == titans["recipe"]["epochs"] == 30
        assert plan.selection == titans["recipe"]["selection"]
        assert plan.update_instants == titans["recipe"]["truncation"]
        assert plan.loss == titans["recipe"]["loss"] == "pinball"
        assert plan.max_grad_norm == titans["recipe"]["max_grad_norm"]
    with pytest.raises(ValueError, match="casos de búsqueda"):
        mt.case_recipe(document, "lr1e-2")


@pytest.mark.parametrize(
    "change",
    [
        lambda d: d["recipe"].update(learning_rate=1e-3),
        lambda d: d["walk_forward"]["search_cases"].update(other={"learning_rate": 1e-4}),
        lambda d: d["walk_forward"]["search_cases"].update(wd={"weight_decay": 0.1}),
        lambda d: d["walk_forward"]["search_cases"].update({"Mal nombre": {"learning_rate": 1}}),
        lambda d: d["walk_forward"].update(warmup_months=12),
        lambda d: d.update(recipe_name="other"),
        lambda d: d["recipe"].update(bank_capacity=2048),
        lambda d: d["recipe"].update(loss="mse"),
    ],
)
def test_recipe_declarations_that_break_the_contract_are_rejected(tmp_path, change):
    document = json.loads(DECLARED.read_text())
    change(document)
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps(document))
    with pytest.raises((ValueError, TypeError)):
        mt.load_recipe(path)


@pytest.mark.parametrize(
    "options",
    [
        dict(update_instants=0),
        dict(neighbors=9),
        dict(bank_capacity=7),
        dict(learning_rate=0.0),
        dict(temperature=0.0),
        dict(max_grad_norm=-1.0),
        dict(checkpoint_seconds=0.0),
    ],
)
def test_recipe_rejects_invalid_values(options):
    with pytest.raises(ValueError):
        recipe(**options)


def test_m3_is_rejected_with_its_motive_and_m2_keeps_its_three_indices():
    plan = recipe()
    with pytest.raises(ValueError, match="M3"):
        mt.retention_config(plan, "m3")
    with pytest.raises(ValueError, match="tres índices"):
        mt.retention_config(plan, "m2", policy="anchored")
    assert mt.retention_config(plan, "m0") is None


@pytest.mark.parametrize("admission", ["m0", "m1"])
def test_roles_cover_the_readout_and_m0_leaves_the_episodic_read_inert(shared, admission):
    _, streams = shared
    readout, _ = reader(streams, admission)
    roles, inert = mt.parameter_roles(readout, admission)
    names = {name for name, _ in readout.named_parameters()}
    read = {name for name in names if name.startswith(("query_projection.", "value_projection."))}
    assert read and set(roles["refiner"]) == names - read
    if admission == "m0":
        assert set(inert) == read and "episodic_read" not in roles
    else:
        assert set(roles["episodic_read"]) == read and inert == []


def test_optimizer_must_cover_exactly_the_trainable_readout(shared, tmp_path, native):
    _, streams = shared

    def partial(groups):
        groups = [dict(group, params=list(group["params"])[1:]) for group in groups]
        return RecordingOptimizer(groups)

    with pytest.raises(ValueError, match="exactamente"):
        build(streams, tmp_path / "run", native, factory=partial)


@pytest.mark.parametrize(
    "case",
    [
        "frozen_variant",
        "trainable_parent",
        "training_mode",
        "local_control",
        "scalar_head_pinball",
        "m3",
        "m0_with_bank",
        "m1_without_bank",
        "other_neighbors",
        "other_codec",
    ],
)
def test_constructor_rejects_parents_readers_and_admissions_outside_the_contract(
    shared, tmp_path, native, case
):
    _, streams = shared
    admission, model, plan = "m1", parent(streams), recipe()
    readout, codec = reader(streams, "m1")
    retention = mt.retention_config(plan, "m1")
    if case == "frozen_variant":
        model = parent(streams, variant="mac_frozen")
    elif case == "trainable_parent":
        model.requires_grad_(True)
    elif case == "training_mode":
        model.train()
    elif case == "local_control":
        model = parent(
            streams, local_control=MACProjectionConfig(mode="diagnostic", rank=1, grid_size=4)
        )
    elif case == "scalar_head_pinball":
        model = parent(streams, head="scalar")
    elif case == "m3":
        admission = "m3"
    elif case == "m0_with_bank":
        admission, retention = "m0", None
    elif case == "m1_without_bank":
        retention = None
    elif case == "other_neighbors":
        readout, codec = reader(streams, "m1", neighbors=4)
    elif case == "other_codec":
        other = copy.copy(streams["train"].specification())
        object.__setattr__(other, "view_sha256", "c" * 64)
        codec = FrozenEpisodeCodec(other)
    with pytest.raises(ValueError):
        mt.MarsTitanInference(
            model,
            readout,
            plan,
            admission=admission,
            retention=retention,
            native=native,
            codec=codec,
            world="fixture",
            fold="0",
        )


@pytest.fixture(scope="module")
def runs(shared, tmp_path_factory, native):
    """Un recorrido completo por escritura con dos épocas y el optimizador registrador."""
    from mars_titan.training.learning_hold import HOLD_ENV

    _, streams = shared
    root = tmp_path_factory.mktemp("mars-titan-runs")
    hold = root / "hold.json"
    hold.write_text(json.dumps({"training_allowed": True}))
    result = {}
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv(HOLD_ENV, str(hold))
        previous = torch.backends.mha.get_fastpath_enabled()
        torch.backends.mha.set_fastpath_enabled(False)
        try:
            for admission in ("m0", "m1", "m2"):
                engine = build(
                    streams, root / admission, native, admission=admission, plan=recipe(epochs=2)
                )
                before = mt._parameters_digest(engine.readout)
                report = engine.run()
                result[admission] = (engine, report, before)
        finally:
            torch.backends.mha.set_fastpath_enabled(previous)
    return result


@pytest.mark.parametrize("admission", ["m0", "m1", "m2"])
def test_loop_reaches_the_step_without_changing_any_weight(runs, admission):
    engine, report, before = runs[admission]
    assert report["status"] == "completed" and report["final_test_opened"] is False
    assert engine.global_step == engine.optimizer.calls > 0
    assert mt._parameters_digest(engine.readout) == before
    engine.predictor.verify_parameter_identity()
    assert engine.predictor._parameter_id == engine.identity["parent"]["parameters_sha256"]
    assert all(p.grad is None for p in engine.predictor.parameters())
    train = report["history"][-1]["train"]
    assert train["labels_without_graph"] == 0 and train["labels_in_loss"] > 0
    assert train["mac_updates"] == train["observations"]
    used = set()
    for record in gradients(engine):
        used |= {name for name, value in record.items() if value is not None and value.any()}
    read = {name for name in used if name.startswith(("query_projection.", "value_projection."))}
    # M0 no tiene episodios: la lectura no está en el optimizador y no recibe gradiente.
    assert bool(read) == (admission != "m0")
    assert any(name.startswith("refinement.") for name in used)
    if admission == "m0":
        assert train["admitted"] == 0
    else:
        assert train["admitted"] == train["labels"]


def train_passes(audit):
    """Tramos contiguos del audit que pertenecen a un recorrido de ajuste."""
    passes, current = [], []
    for entry in audit:
        train = entry[0] == "update" or entry[1] == "train"
        if train:
            current.append(entry)
        elif current:
            passes.append(current)
            current = []
    return [*passes, current] if current else passes


def test_each_event_resolves_labels_updates_predicts_and_then_admits(runs):
    engine, _, _ = runs["m1"]
    passes = train_passes(engine.audit)
    assert len(passes) == 2
    for audit in passes:
        seen, position = 0, {}
        for index, entry in enumerate(audit):
            kind = entry[0]
            at = {"label": 4, "update": 1, "prediction": 3, "admit": 2}[kind]
            position.setdefault((kind, entry[at]), []).append(index)
            if kind == "prediction":
                # La predicción ve el banco confirmado antes de las etiquetas de su evento.
                assert entry[5] == seen
            elif kind == "admit":
                assert entry[3] == tuple(range(seen + 1, entry[4] + 1))
                seen = entry[4]
        updates = [key[1] for key in position if key[0] == "update"]
        assert updates and seen > 0
        for at in updates:
            update = position["update", at][0]
            assert all(index < update for index in position.get(("label", at), []))
            predictions = position.get(("prediction", at), [])
            assert all(update < index for index in predictions)
            if ("admit", at) in position:
                assert max(predictions, default=update) < position["admit", at][0]
        for at in {key[1] for key in position if key[0] == "admit"}:
            last = max(position.get(("prediction", at), [-1]))
            assert last < position["admit", at][0]
    for _, at, _, used in entries(engine.audit, "update"):
        assert all(decision_at < at for _, decision_at in used)


def leaves(value):
    if isinstance(value, torch.Tensor):
        return [value.detach().cpu().clone()]
    items = value.values() if isinstance(value, dict) else value
    if isinstance(value, (dict, list, tuple)):
        return [leaf for item in items for leaf in leaves(item)]
    return []


def test_parent_fast_state_does_not_depend_on_the_bank_or_the_readout(shared, tmp_path, native):
    _, streams = shared
    model = parent(streams)
    calls = []
    original = model.prepare

    def recording(batch, state, **options):
        prepared = original(batch, state, **options)
        calls.append(
            (
                tuple(batch.flow_ids),
                prepared.point_predictions.detach().cpu().clone(),
                leaves(model.export_state_cpu(prepared.next_state)),
            )
        )
        return prepared

    model.prepare = recording
    sequences = {}
    for admission in ("m0", "m1", "m2"):
        calls.clear()
        engine = build(streams, tmp_path / admission, native, admission=admission, model=model)
        engine.evaluate(streams["validation"])
        sequences[admission] = list(calls)
    calls.clear()
    plain = ChronologicalRecipe(
        truncation=3, loss="pinball", epochs=1, block_rows=2, selection=SELECTION
    )
    ChronologicalInference(model, plain).predict(streams["validation"], [])
    sequences["titans"] = list(calls)
    reference = sequences["titans"]
    assert reference
    for name, sequence in sequences.items():
        assert len(sequence) == len(reference), name
        for (flows, core, memory), (other_flows, other_core, other_memory) in zip(
            sequence, reference, strict=True
        ):
            assert flows == other_flows and torch.equal(core, other_core)
            assert len(memory) == len(other_memory) > 0
            assert all(torch.equal(a, b) for a, b in zip(memory, other_memory, strict=True))


def test_future_suffix_changes_neither_past_predictions_nor_past_gradients(
    tmp_path, native, learning_doubles
):
    cut = decision("2022-12-09")
    engines = []
    for name, perturb in (("base", None), ("future", cut)):
        _, streams = corpus(tmp_path / name, perturb_after=perturb)
        engine = build(streams, tmp_path / name / "run", native)
        engine.run()
        engines.append(engine)
    base, future = engines

    def past(engine):
        return [e for e in entries(engine.audit, "prediction", "train") if e[3] <= cut]

    assert past(base) and past(base) == past(future)
    assert entries(base.audit, "prediction", "train") != entries(
        future.audit, "prediction", "train"
    )
    steps = [u[2] for u in entries(base.audit, "update") if u[1] <= cut]
    assert steps == [u[2] for u in entries(future.audit, "update") if u[1] <= cut]
    for left, right in zip(gradients(base)[: len(steps)], gradients(future), strict=False):
        assert left.keys() == right.keys()
        for key, value in left.items():
            if value is None:
                assert right[key] is None
            else:
                torch.testing.assert_close(value, right[key], rtol=0, atol=0)


@pytest.mark.parametrize("admission", ["m1", "m2"])
def test_resume_after_a_pause_reproduces_the_continuous_run(
    shared, tmp_path, native, learning_doubles, admission
):
    _, streams = shared
    plan = recipe(epochs=2, checkpoint_updates=2)
    continuous = build(streams, tmp_path / "continuous", native, admission=admission, plan=plan)
    expected = continuous.run()
    first = build(streams, tmp_path / "paused", native, admission=admission, plan=plan)
    stop = StopAtStep(continuous.global_step // 2 + 1)
    stop.trainer = first
    assert first.run(stop=stop)["status"] == "paused"
    state = load_training_state(tmp_path / "paused/checkpoints", expected_identity=first.identity)
    assert state["cursor"]["phase"] == "train" and state["cursor"]["stage"] == "inputs"
    assert state["run"]["bank"] is not None and state["run"]["fast"]
    assert state["optimizer"]["calls"] == state["global_step"]
    second = build(streams, tmp_path / "paused", native, admission=admission, plan=plan)
    resumed = second.run(resume=True)
    assert resumed["status"] == "completed"
    assert first.audit + second.audit == continuous.audit
    joined = gradients(first) + gradients(second)
    assert len(joined) == len(gradients(continuous)) == continuous.global_step
    for left, right in zip(joined, gradients(continuous), strict=True):
        for key, value in left.items():
            if value is None:
                assert right[key] is None
            else:
                torch.testing.assert_close(value, right[key], rtol=0, atol=0)
    assert resumed["history"] == expected["history"]
    assert (
        resumed["best_checkpoint"]["readout_sha256"]
        == expected["best_checkpoint"]["readout_sha256"]
    )
    retained = list((tmp_path / "paused/checkpoints").glob("state-*.pt"))
    assert 1 <= len(retained) <= 3


def test_readout_fit_is_refused_while_the_learning_hold_blocks(
    shared, tmp_path, native, learning_hold
):
    _, streams = shared
    engine = build(streams, tmp_path / "run", native, factory=None)
    assert type(engine.optimizer) is torch.optim.AdamW
    learning_hold(False)
    with pytest.raises(LearningHoldError, match="Bloqueo de aprendizaje vigente"):
        engine.run()
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("refinements", [2, 4])
def test_refinements_count_per_prediction_and_fixed_episodes_have_their_own_identity(
    shared, tmp_path, native, refinements
):
    _, streams = shared
    per_step = build(streams, tmp_path / "per", native, refinements=refinements)
    fixed = build(
        streams, tmp_path / "fixed", native, refinements=refinements, episodes="first_read"
    )
    assert per_step.run_id != fixed.run_id
    for engine in (per_step, fixed):
        metrics = engine.evaluate(streams["validation"])
        assert metrics["refinements"] == refinements * metrics["predictions"]
        assert metrics["mac_updates"] == metrics["observations"]


def test_scalar_parent_fits_with_l1_and_writes_no_quantiles(
    shared, tmp_path, native, learning_doubles
):
    _, streams = shared
    engine = build(
        streams,
        tmp_path / "run",
        native,
        model=parent(streams, head="scalar"),
        plan=recipe(loss="mae"),
    )
    report = engine.run()
    assert report["status"] == "completed" and engine.optimizer.calls > 0
    rows = []
    engine.evaluate(streams["validation"], rows=rows)
    assert rows and all(levels is None for *_, levels in rows)
