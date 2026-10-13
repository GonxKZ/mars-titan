"""Entrenador cronológico comprobado hasta el paso del optimizador, sin modificar pesos."""

import copy
import json
from dataclasses import replace

import pytest
import torch

from mars_titan.data.input_policy import HISTORICAL_MASKED
from mars_titan.memory import financial_observations as api
from mars_titan.memory.financial_session import FinancialPhase
from mars_titan.models.titans.config import MAX_BLOCK_ROWS
from mars_titan.models.titans.financial import (
    VARIANTS,
    FinancialConfig,
    FinancialPredictor,
    copy_paired_parameters,
)
from mars_titan.models.titans.financial_inputs import FinancialInputSpec
from mars_titan.training import financial_run
from mars_titan.training.checkpoints import load_training_state
from mars_titan.training.corpus_inputs import CorpusDataset
from mars_titan.training.financial_run import (
    ChronologicalRecipe,
    ChronologicalTrainer,
    parameter_roles,
)
from mars_titan.training.selection import FIXED_BUDGET
from tests.training.chronological_fixture import (
    US_2023,
    chronological_corpus,
    decision,
    phases,
)

SELECTION = dict(metric="session_mae", patience=5, min_delta=0.0)


@pytest.fixture(autouse=True)
def explicit_fastpath():
    previous = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    yield
    torch.backends.mha.set_fastpath_enabled(previous)


class RecordingOptimizer:
    """Registrar gradientes y llamadas sin heredar de Optimizer ni modificar pesos."""

    def __init__(self, groups):
        self.param_groups = [dict(group, params=list(group["params"])) for group in groups]
        self.calls, self.records, self.zero_calls = 0, [], 0

    def step(self):
        self.calls += 1
        self.records.append(
            {
                id(p): None if p.grad is None else p.grad.detach().clone()
                for group in self.param_groups
                for p in group["params"]
            }
        )

    def zero_grad(self, set_to_none=True):
        assert set_to_none is True
        self.zero_calls += 1
        for group in self.param_groups:
            for value in group["params"]:
                value.grad = None

    def state_dict(self):
        return dict(calls=self.calls, roles=[group["role"] for group in self.param_groups])

    def load_state_dict(self, state):
        assert state["roles"] == [group["role"] for group in self.param_groups]
        self.calls = state["calls"]


class StopAtStep:
    def __init__(self, step):
        self.step, self.trainer = step, None

    @property
    def requested(self):
        return self.trainer is not None and self.trainer.global_step >= self.step


def corpus(root, *, perturb_after=None, train=None, validation=None):
    dataset = CorpusDataset(
        chronological_corpus(root / "corpus", perturb_after=perturb_after),
        input_policy=HISTORICAL_MASKED,
    )
    default = phases()
    result = {}
    for phase in (train or default[0], validation or default[1]):
        manifest = api.prepare_observation_index(
            dataset, root / f"index-{phase.partition}", phase=phase
        )
        result[phase.partition] = api.FinancialObservationSource(dataset, manifest)
    return dataset, result


@pytest.fixture(scope="module")
def shared(tmp_path_factory):
    return corpus(tmp_path_factory.mktemp("chronological"))


def predictor(streams, variant="mac_online", seed=42):
    specification = streams["train"].specification()
    config = FinancialConfig(specification, variant=variant, hidden_size=32, seed=seed)
    return FinancialPredictor(config, dtype=torch.float64)


def trainer(streams, output, *, model=None, variant="mac_online", pairing=None, **options):
    recipe = dict(truncation=3, epochs=1, block_rows=2, selection=SELECTION)
    recipe.update(options)
    if recipe.pop("fixed", False):
        recipe["selection"] = dict(recipe["selection"], stopping=FIXED_BUDGET)
    return ChronologicalTrainer(
        model or predictor(streams, variant),
        ChronologicalRecipe(**recipe),
        train=streams["train"],
        validation=streams["validation"],
        output=output,
        optimizer_factory=RecordingOptimizer,
        pairing=pairing,
        audit=True,
    )


def entries(audit, kind, phase=None):
    return [e for e in audit if e[0] == kind and (phase is None or e[1] == phase)]


def named_records(engine):
    names = {id(p): name for name, p in engine.predictor.named_parameters()}
    return [{names[key]: value for key, value in r.items()} for r in engine.optimizer.records]


@pytest.mark.parametrize(
    "options",
    [
        dict(truncation=0),
        dict(truncation=True),
        dict(loss="quantile"),
        dict(learning_rate=0.0),
        dict(weight_decay=-1.0),
        dict(max_grad_norm=float("inf")),
        dict(block_rows=MAX_BLOCK_ROWS + 1),
        dict(selection=dict(SELECTION, stopping="unbounded")),
        dict(selection=dict(SELECTION, minimum_epochs=1), epochs=1),
        dict(selection=dict(SELECTION, metric="mae")),
        dict(checkpoint_seconds=0.0),
    ],
)
def test_recipe_rejects_undeclared_or_invalid_declarations(options):
    with pytest.raises(ValueError):
        ChronologicalRecipe(**options)


@pytest.mark.parametrize("variant", VARIANTS)
def test_roles_cover_outer_parameters_and_exclude_fast_state(shared, tmp_path, variant):
    _, streams = shared
    engine = trainer(streams, tmp_path / "run", variant=variant)
    roles = parameter_roles(engine.predictor)
    names = [name for name, _ in engine.predictor.named_parameters()]
    assert sorted(sum(roles.values(), [])) == sorted(names)
    mac = variant != "transformer_direct"
    assert roles["persistent_memory"] == (["mac.persistent"] if mac else [])
    assert roles["initial_fast_weights"] == (
        ["mac.memory.initial_weights.0", "mac.memory.initial_weights.1"] if mac else []
    )
    assert {g["role"] for g in engine.optimizer.param_groups} == {k for k, v in roles.items() if v}
    state = engine.predictor.initial_state(("US/A0000",))
    outer = {id(p) for p in engine.predictor.parameters()}
    tensors = [state.observed_steps]
    if state.mac is not None:
        tensors += [*state.mac.memory.weights, *state.mac.memory.momentum]
    assert all(id(t) not in outer and not isinstance(t, torch.nn.Parameter) for t in tensors)
    assert engine.identity["parameter_roles"] == roles
    assert engine.identity["predictor"]["variant"] == variant


def test_optimizer_must_cover_exactly_the_outer_parameters(shared, tmp_path):
    _, streams = shared

    def partial(groups):
        return RecordingOptimizer(groups[:1])

    with pytest.raises(ValueError, match="ajuste externo"):
        ChronologicalTrainer(
            predictor(streams),
            ChronologicalRecipe(truncation=3, epochs=1, block_rows=2, selection=SELECTION),
            train=streams["train"],
            validation=streams["validation"],
            output=tmp_path / "run",
            optimizer_factory=partial,
        )


def test_loop_steps_only_at_segment_boundaries_with_matured_labels(shared, tmp_path, monkeypatch):
    _, streams = shared
    engine = trainer(streams, tmp_path / "run")
    initial = engine.predictor._parameter_id
    seals, seal = [], engine.predictor._seal_parameters
    monkeypatch.setattr(engine.predictor, "_seal_parameters", lambda: seals.append(1) or seal())
    report = engine.run()
    assert report["status"] == "completed"
    train = report["history"][1]["train"]
    updates = entries(engine.audit, "update")
    assert len(engine.optimizer.records) == len(updates) == train["updates"] > 3
    assert engine.optimizer.zero_calls == len(updates)
    assert len(seals) == len(updates) + 1
    assert engine.predictor._parameter_id == initial
    phase = streams["train"].phase
    instants = sorted({e[3] for e in entries(engine.audit, "prediction", "train")})
    assert instants[0] >= phase.decision_start
    boundaries = [u[1] for u in updates]
    for previous, current in zip([phase.decision_start - 1, *boundaries], boundaries, strict=False):
        inside = [at for at in instants if previous <= at < current]
        assert len(inside) == (3 if current != phase.close_at else len(inside))
        assert 1 <= len(inside) <= 3
    used = [label for update in updates for label in update[3]]
    assert all(decision_at < maturity <= u[1] for u in updates for _, decision_at, maturity in u[3])
    assert len(used) == len(set(used)) == train["labels"] == train["labels_in_loss"]
    assert train["labels_without_graph"] == 0
    resolved = {(e[2], e[3]) for e in entries(engine.audit, "label", "train")}
    assert resolved == {(flow, at) for flow, at, _ in used}
    count = len(engine.optimizer.records)
    engine.evaluate(streams["validation"])
    assert len(engine.optimizer.records) == count


def test_gradients_reach_only_the_components_each_control_uses(tmp_path):
    first = decision("2022-11-01")
    train = FinancialPhase("train", first, first, US_2023, US_2023)
    _, streams = corpus(tmp_path, train=train)
    for variant in VARIANTS:
        engine = trainer(streams, tmp_path / variant, variant=variant)
        engine.run()
        records = named_records(engine)

        def active(name, record):
            value = record[name]
            return value is not None and bool(value.abs().sum() > 0)

        for record in records:
            for name in ("head.weight", "fusion.0.weight"):
                assert active(name, record)
        memory = [n for n in records[0] if n.startswith("mac.memory.")]
        projections = [
            n
            for n in memory
            if n.split(".")[2] in {"key_projection", "value_projection", "alpha_projection"}
            or n.split(".")[2] in {"eta_projection", "theta_projection"}
        ]
        initial = ["mac.memory.initial_weights.0", "mac.memory.initial_weights.1"]
        if variant == "transformer_direct":
            assert not memory and "mac.persistent" not in records[0]
        elif variant == "mac_disabled":
            assert all(not active(n, r) for r in records for n in memory)
            assert all(not active("mac.persistent", r) for r in records)
            assert all(not active("mac.query_projection.weight", r) for r in records)
            assert all(active("mac.attention.in_proj_weight", r) for r in records)
        elif variant == "mac_frozen":
            assert all(not active(n, r) for r in records for n in projections)
            assert all(active(n, r) for r in records for n in initial)
            assert all(active("mac.persistent", r) for r in records)
        else:
            assert all(any(active(n, r) for r in records) for n in projections)
            assert all(active(n, records[0]) for n in initial)
            assert any(not active(initial[0], r) for r in records[1:])
            assert all(active("mac.persistent", r) for r in records)


def test_paired_controls_start_equal_and_keep_equal_updates(shared, tmp_path):
    _, streams = shared
    source = predictor(streams, "mac_online", seed=7)
    reports, identities = {}, set()
    for variant in VARIANTS:
        model = predictor(streams, variant, seed=11)
        receipt = copy_paired_parameters(source, model)
        engine = trainer(streams, tmp_path / variant, model=model, pairing=receipt, fixed=True)
        for name, value in model.named_parameters():
            torch.testing.assert_close(value, dict(source.named_parameters())[name], rtol=0, atol=0)
        identities.add(engine.run_id)
        reports[variant] = engine.run()
        assert engine.identity["pairing_sha256"] is not None
    assert len(identities) == len(VARIANTS)
    counts = {
        v: tuple(r["history"][1]["train"][k] for k in ("updates", "labels_in_loss", "segments"))
        for v, r in reports.items()
    }
    assert len(set(counts.values())) == 1
    other = predictor(streams, "mac_frozen")
    with pytest.raises(ValueError, match="emparejamiento"):
        trainer(streams, tmp_path / "wrong", model=other, pairing=receipt)


def test_future_suffix_does_not_change_past_predictions_or_gradients(tmp_path):
    cut = decision("2022-12-09")
    results = []
    for name, perturb in (("base", None), ("future", cut)):
        _, streams = corpus(tmp_path / name, perturb_after=perturb)
        engine = trainer(streams, tmp_path / name / "run", truncation=2)
        engine.run()
        results.append(engine)
    base, future = results

    def past(engine):
        return [e for e in entries(engine.audit, "prediction", "train") if e[3] <= cut]

    assert past(base) == past(future)
    assert entries(base.audit, "prediction", "train") != entries(
        future.audit, "prediction", "train"
    )
    steps = [u[2] for u in entries(base.audit, "update") if u[1] <= cut]
    assert steps and steps == [u[2] for u in entries(future.audit, "update") if u[1] <= cut]
    for left, right in zip(named_records(base)[: len(steps)], named_records(future), strict=False):
        assert left.keys() == right.keys()
        for key in left:
            torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)
    later = named_records(base)[len(steps)], named_records(future)[len(steps)]
    assert any(
        left is not None and not torch.equal(left, later[1][key]) for key, left in later[0].items()
    )


def test_resume_after_interruption_reproduces_the_continuous_run(shared, tmp_path):
    _, streams = shared
    options = dict(epochs=2, fixed=True, checkpoint_updates=2)
    continuous = trainer(streams, tmp_path / "continuous", **options)
    expected = continuous.run()
    stop = StopAtStep(15)
    first = trainer(streams, tmp_path / "interrupted", **options)
    stop.trainer = first
    paused = first.run(stop=stop)
    assert paused["status"] == "paused"
    state = load_training_state(
        tmp_path / "interrupted/checkpoints", expected_identity=first.identity
    )
    assert state["cursor"]["phase"] == "train" and state["cursor"]["stage"] == "inputs"
    assert state["cursor"]["epoch"] == 1 and state["global_step"] >= 15
    assert state["run"]["fast"] and state["run"]["pending"]["flows"] == ["US/A0000"]
    assert state["optimizer"]["calls"] == state["global_step"]
    assert len(state["run"]["fast"][0]["flow_ids"]) == 3
    second = trainer(streams, tmp_path / "interrupted", **options)
    resumed = second.run(resume=True)
    assert resumed["status"] == "completed"
    assert first.audit + second.audit == continuous.audit
    joined = named_records(first) + named_records(second)
    assert len(joined) == len(named_records(continuous)) == second.optimizer.calls
    for left, right in zip(joined, named_records(continuous), strict=True):
        for key in left:
            if left[key] is None:
                assert right[key] is None
            else:
                torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)
    assert resumed["history"] == expected["history"]
    assert resumed["best_epoch"] == expected["best_epoch"]
    retained = list((tmp_path / "interrupted/checkpoints").glob("state-*.pt"))
    assert 1 <= len(retained) <= 3


def test_completed_run_resumes_with_the_selected_state_not_the_latest(shared, tmp_path):
    from mars_titan.training.checkpoints import save_training_state

    _, streams = shared
    engine = trainer(streams, tmp_path / "run")
    report = engine.run()
    selected = {
        k: v.clone() for k, v in engine.predictor.state_dict().items() if torch.is_tensor(v)
    }
    checkpoints = tmp_path / "run/checkpoints"
    latest = load_training_state(checkpoints, expected_identity=engine.identity)
    # Estado de recuperación alterado a mano, sin optimizador, posterior al seleccionado.
    latest["model"]["head.bias"] = latest["model"]["head.bias"] + 1.0
    save_training_state(checkpoints, latest, identity=engine.identity)
    resumed = trainer(streams, tmp_path / "run")
    assert resumed.run(resume=True)["best_checkpoint"] == report["best_checkpoint"]
    for name, value in resumed.predictor.state_dict().items():
        if torch.is_tensor(value):
            torch.testing.assert_close(value, selected[name], rtol=0, atol=0)


def test_partition_inference_needs_a_later_partition_of_the_same_input_and_a_row_sink(
    shared, tmp_path
):
    _, streams = shared
    engine = trainer(streams, tmp_path / "run")
    rows = []
    with pytest.raises(ValueError, match="tramo"):
        engine.predict_partition(streams["train"], rows)
    early = copy.copy(streams["validation"])
    early.phase = replace(early.phase, decision_start=streams["train"].phase.decision_end - 1)
    with pytest.raises(ValueError, match="tramo"):
        engine.predict_partition(early, rows)
    with pytest.raises(ValueError, match="destino"):
        engine.predict_partition(streams["validation"], None)
    metrics = engine.predict_partition(streams["validation"], rows)
    assert len(rows) == metrics["labels"] == metrics["samples"] > 0
    flows = {flow for flow, *_ in rows}
    assert all(levels is None for *_, levels in rows) and flows <= {
        "US/A0000",
        "US/A0001",
        "US/A0002",
    }
    assert metrics == engine.evaluate(streams["validation"])


def test_resume_rejects_a_changed_recipe(shared, tmp_path):
    _, streams = shared
    engine = trainer(streams, tmp_path / "run", checkpoint_updates=1)
    stop = StopAtStep(2)
    stop.trainer = engine
    assert engine.run(stop=stop)["status"] == "paused"
    changed = trainer(streams, tmp_path / "run", truncation=4)
    with pytest.raises(ValueError, match="identidad"):
        changed.run(resume=True)
    with pytest.raises(ValueError):
        trainer(streams, tmp_path / "run").run()


def test_validation_matches_the_frozen_consumer_on_the_row_reader(shared, tmp_path):
    from mars_titan.models.titans.financial_inputs import DecisionBatch, validated_cpu_batch
    from mars_titan.models.titans.frozen_financial import FrozenFinancialConsumer

    _, streams = shared
    engine = trainer(streams, tmp_path / "run", block_rows=3)
    engine.evaluate(streams["validation"])
    issued = {(e[2], e[3]): e[4] for e in entries(engine.audit, "prediction", "validation")}
    model = engine.predictor
    model.requires_grad_(False)
    consumer = FrozenFinancialConsumer(model)
    stream, states, expected = streams["validation"], {}, {}
    for event in stream.events():
        for raw in event.inputs:
            batch = DecisionBatch.from_validated(
                validated_cpu_batch(raw, model.config.inputs), dtype=torch.float64
            )
            flow = batch.flow_ids[0]
            state = states.get(flow) or model.initial_state((flow,))
            warmup = event.at < stream.phase.decision_start
            result = consumer.prepare(batch, state, context_id="0" * 64, warmup=warmup)
            states[flow] = result.next_state
            if not warmup:
                expected[flow, event.at] = float(result.point_predictions[0])
    assert issued.keys() == expected.keys()
    for key, value in expected.items():
        assert issued[key] == pytest.approx(value, rel=1e-12, abs=1e-14)


@pytest.mark.parametrize("aligned", [True, False])
def test_scored_rows_correspond_to_the_corpus_partition(tmp_path, aligned):
    first = decision("2022-11-01")
    start = first if aligned else decision("2022-11-15")
    end = 1_704_067_200_000_000 if aligned else decision("2023-02-15")
    train = FinancialPhase("train", first, start, US_2023, US_2023)
    validation = FinancialPhase("validation", decision("2022-12-01"), US_2023, end, end)
    dataset, streams = corpus(tmp_path, train=train, validation=validation)
    engine = trainer(streams, tmp_path / "run")
    engine.run()
    for phase in (train, validation):
        scored = {(e[2], e[3]) for e in entries(engine.audit, "label", phase.partition)}
        reference = {}
        for batch in dataset.batches(partition=phase.partition, batch_size=64, epoch=0, seed=0):
            for index, sample in enumerate(batch["sample_ids"]):
                flow, at = sample.rsplit("/", 1)
                maturity = int(batch["target_available_at"][index].astype("int64"))
                reference[flow, int(at)] = maturity
        inside = {
            key
            for key, maturity in reference.items()
            if phase.decision_start <= key[1] < phase.decision_end and maturity < phase.decision_end
        }
        assert scored == inside
        assert (scored == set(reference)) == aligned
        if not aligned:
            assert set(reference) - scored
    assert json.loads((tmp_path / "run/run.json").read_text())["final_test_opened"] is False


def test_constructor_rejects_unsafe_or_mismatched_inputs(shared, tmp_path):
    from mars_titan.models.titans.local_control import MACProjectionConfig

    _, streams = shared
    recipe = ChronologicalRecipe(truncation=3, epochs=1, block_rows=2, selection=SELECTION)
    options = dict(output=tmp_path / "run", optimizer_factory=RecordingOptimizer)
    with pytest.raises(ValueError, match="fases"):
        ChronologicalTrainer(
            predictor(streams),
            recipe,
            train=streams["validation"],
            validation=streams["train"],
            **options,
        )
    torch.backends.mha.set_fastpath_enabled(True)
    with pytest.raises(ValueError, match="fastpath"):
        ChronologicalTrainer(
            predictor(streams),
            recipe,
            train=streams["train"],
            validation=streams["validation"],
            **options,
        )
    torch.backends.mha.set_fastpath_enabled(False)
    specification = streams["train"].specification()
    control = FinancialPredictor(
        FinancialConfig(specification, variant="mac_online", hidden_size=32),
        local_control=MACProjectionConfig(mode="diagnostic"),
        dtype=torch.float64,
    )
    # B y la penalización C se ajustan aquí. El diagnóstico pertenece a la sesión congelada.
    with pytest.raises(ValueError, match="diagnóstico"):
        ChronologicalTrainer(
            control, recipe, train=streams["train"], validation=streams["validation"], **options
        )
    view = FinancialInputSpec(
        source_sha256=specification.source_sha256,
        view_sha256="e" * 64,
        representation=specification.representation,
        dimensions=specification.dimensions,
        input_policy=specification.input_policy,
    )
    source = FinancialInputSpec(
        source_sha256="f" * 64,
        view_sha256=specification.view_sha256,
        representation=specification.representation,
        dimensions=specification.dimensions,
        input_policy=specification.input_policy,
    )
    assert financial_run._compatible(specification, view)
    assert not financial_run._compatible(specification, source)
    with pytest.raises(ValueError):
        ChronologicalTrainer(
            predictor(streams),
            recipe,
            train=streams["train"],
            validation=streams["validation"],
            output=streams["train"].path.parent / "inside",
            optimizer_factory=RecordingOptimizer,
        )


def scripted(values):
    scores = iter(values)

    def evaluate(source, *, stop=None):
        return dict(session_mae=next(scores), labels=1)

    return evaluate


def quick(engine):
    def train_pass(run, cursor, stop, save):
        engine.global_step += 1
        return dict(updates=1)

    return train_pass


def test_selection_follows_declared_patience_and_keeps_the_best_state(
    shared, tmp_path, monkeypatch
):
    _, streams = shared
    cases = (
        ("patience", dict(epochs=4, selection=dict(SELECTION, patience=2)), [0.5, 0.4, 0.45, 0.46]),
        (
            "fixed",
            dict(epochs=4, fixed=True, selection=dict(SELECTION, patience=2)),
            [0.5, 0.4, 0.45, 0.46, 0.3],
        ),
        ("parent", dict(epochs=2, fixed=True), [0.1, 0.4, 0.45]),
    )
    reports = {}
    for name, options, values in cases:
        engine = trainer(streams, tmp_path / name, **options)
        monkeypatch.setattr(engine, "evaluate", scripted(values))
        monkeypatch.setattr(engine, "_train_pass", quick(engine))
        initial = engine.predictor._parameter_id
        reports[name] = engine.run()
        assert [h["epoch"] for h in reports[name]["history"]] == list(range(len(values)))
        best = load_training_state(
            tmp_path / name / "checkpoints", expected_identity=engine.identity, selection="best"
        )
        assert best["global_step"] == reports[name]["best_epoch"]
        assert best["selection"]["best_epoch"] == reports[name]["best_epoch"]
        assert len(list((tmp_path / name / "checkpoints").glob("state-*.pt"))) <= 3
        assert engine.predictor._parameter_id == initial
    assert (reports["patience"]["best_epoch"], reports["patience"]["stopped_early"]) == (1, True)
    assert (reports["fixed"]["best_epoch"], reports["fixed"]["stopped_early"]) == (4, False)
    assert [h["global_step"] for h in reports["fixed"]["history"]] == [0, 1, 2, 3, 4]
    assert reports["fixed"]["plateau_epoch"] == 3 and reports["fixed"]["stopping"] == FIXED_BUDGET
    assert reports["patience"]["plateau_epoch"] is None
    assert (reports["parent"]["best_epoch"], reports["parent"]["best_score"]) == (0, 0.1)


def test_real_optimizer_is_refused_while_the_learning_hold_blocks(shared, tmp_path, monkeypatch):
    from mars_titan.training import financial_run
    from mars_titan.training.learning_hold import HOLD_ENV, LearningHoldError

    def past_the_guard():
        # Sin la protección, el recorrido llegaría al paso y el gancho global lo omitiría.
        raise AssertionError("El ajuste pasó de la protección del aprendizaje")

    monkeypatch.setattr(financial_run, "StopRequest", past_the_guard)
    _, streams = shared
    hold = tmp_path / "hold.json"
    hold.write_text(json.dumps(dict(training_allowed=False)))
    monkeypatch.setenv(HOLD_ENV, str(hold))
    engine = ChronologicalTrainer(
        predictor(streams),
        ChronologicalRecipe(truncation=3, epochs=1, block_rows=2, selection=SELECTION),
        train=streams["train"],
        validation=streams["validation"],
        output=tmp_path / "run",
    )
    assert type(engine.optimizer) is torch.optim.AdamW
    assert engine.identity["recipe"]["optimizer"] == "AdamW"
    with pytest.raises(LearningHoldError, match="Bloqueo de aprendizaje vigente"):
        engine.run()
    assert not (tmp_path / "run").exists()


def test_declared_recipe_builds_the_four_paired_controls(shared):
    from mars_titan.training.financial_run import load_recipe

    recipe, document = load_recipe("configs/titans/chronological-training.json")
    assert recipe.selection["stopping"] == FIXED_BUDGET
    assert document["status"] == "propuesta_sin_ejecutar"
    _, streams = shared
    specification = streams["train"].specification()
    options = {k: v for k, v in document["predictor"].items() if k != "dtype"}
    for variant in document["variants"]:
        FinancialConfig(specification, variant=variant, **options)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requiere cuda:0 explícito")
def test_cuda_pass_matches_cpu_without_optimizer_steps(shared, tmp_path):
    _, streams = shared
    engines = {}
    for device in ("cpu", "cuda:0"):
        model = predictor(streams).to(device)
        engines[device] = trainer(streams, tmp_path / device.replace(":", ""), model=model)
        assert engines[device].run()["status"] == "completed"
    cpu, cuda = engines.values()
    left = entries(cpu.audit, "prediction")
    right = entries(cuda.audit, "prediction")
    assert [e[:4] for e in left] == [e[:4] for e in right]
    for a, b in zip(left, right, strict=True):
        assert a[4] == pytest.approx(b[4], rel=1e-9, abs=1e-12)
    for a, b in zip(named_records(cpu), named_records(cuda), strict=True):
        for key, value in a.items():
            if value is None:
                assert b[key] is None
            else:
                torch.testing.assert_close(value, b[key].cpu(), rtol=1e-7, atol=1e-10)
