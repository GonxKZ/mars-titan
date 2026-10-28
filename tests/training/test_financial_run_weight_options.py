"""F2 y F3 (#749) en el recorrido cronológico, sin pasos de optimizador.

Los dobles registran gradientes sin modificar los pesos. Para que los pesos emitidos difieran
de los del ajuste, la sombra de F2 y el estado z de F3 se fijan a mano, sin ajuste.
"""

from collections import defaultdict

import pytest
import torch

from mars_titan.training import weight_averaging
from mars_titan.training.checkpoints import load_training_state
from mars_titan.training.financial_run import ChronologicalRecipe, ChronologicalTrainer
from tests.training.test_financial_run import (
    SELECTION,
    RecordingOptimizer,
    StopAtStep,
    corpus,
    named_records,
    predictor,
)

EMA = dict(weight_average=dict(decay=0.9))
FREE = dict(schedule_free=dict(warmup_steps=0, beta1=0.9))
SHRINK = dict(shrink_perturb=dict(shrink=0.6, noise=0.01, seed=7))


@pytest.fixture(autouse=True)
def explicit_fastpath():
    previous = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    yield
    torch.backends.mha.set_fastpath_enabled(previous)


@pytest.fixture(scope="module")
def shared(tmp_path_factory):
    return corpus(tmp_path_factory.mktemp("weight-options"))[1]


class ScheduleFreeDouble(RecordingOptimizer):
    """Doble con la forma de AdamWScheduleFree. Cada paso fija z a mano sin tocar y."""

    def __init__(self, groups):
        super().__init__([dict(group, betas=(0.9, 0.999), train_mode=True) for group in groups])
        self.state = defaultdict(dict)

    def _values(self):
        return [value for group in self.param_groups for value in group["params"]]

    def step(self):
        super().step()
        for value in self._values():
            self.state[value]["z"] = value.detach() + 1e-3 * self.calls

    def state_dict(self):
        z = [self.state[value].get("z") for value in self._values()]
        return dict(super().state_dict(), z=[None if v is None else v.clone() for v in z])

    def load_state_dict(self, state):
        super().load_state_dict(state)
        for value, z in zip(self._values(), state["z"], strict=True):
            if z is not None:
                self.state[value]["z"] = z.clone()


def trainer(streams, output, *, factory=RecordingOptimizer, model=None, **options):
    values = dict(truncation=3, epochs=1, block_rows=2, max_grad_norm=None, selection=SELECTION)
    return ChronologicalTrainer(
        model or predictor(streams),
        ChronologicalRecipe(**values | options),
        train=streams["train"],
        validation=streams["validation"],
        output=output,
        optimizer_factory=factory,
        audit=True,
    )


def test_without_options_the_recipe_identity_keeps_its_form():
    identity = ChronologicalRecipe().identity()
    assert not {"weight_average", "schedule_free", "shrink_perturb"} & set(identity)
    assert identity["optimizer"] == "AdamW"
    assert set(ChronologicalRecipe().weight_options().values()) == {None}


def test_each_declared_option_enters_the_recipe_identity():
    identities = [ChronologicalRecipe(**options).identity() for options in (EMA, FREE, SHRINK)]
    assert identities[0]["weight_average"]["name"] == "weight_ema_v1"
    assert identities[1]["schedule_free"]["name"] == "schedule_free_v1"
    assert identities[1]["optimizer"] == "AdamWScheduleFree"
    assert identities[2]["shrink_perturb"]["name"] == "shrink_perturb_v1"
    assert len({str(identity) for identity in identities}) == 3


@pytest.mark.parametrize(
    "options,message",
    [
        (dict(weight_average={}), "toda su configuración"),
        (dict(weight_average=dict(decay=0.9, extra=1)), "toda su configuración"),
        (dict(schedule_free=dict(warmup_steps=0)), "toda su configuración"),
        (dict(shrink_perturb="ash_adams"), "toda su configuración"),
        (dict(weight_average=dict(decay=1.0)), "0 < δ < 1"),
        (EMA | FREE, "promedian dos veces"),
        (FREE | dict(weight_decay_anchor="initial_parameters"), "anclado"),
        (SHRINK | dict(weight_decay_anchor="initial_parameters"), "anclado"),
    ],
)
def test_partial_or_combined_options_are_rejected(options, message):
    with pytest.raises(ValueError, match=message):
        ChronologicalRecipe(**options)


def test_the_trainer_rejects_what_has_no_declared_contrast(shared, tmp_path):
    with pytest.raises(ValueError, match="F2 y F3 no tienen"):
        ChronologicalTrainer(
            predictor(shared),
            ChronologicalRecipe(truncation=3, epochs=1, block_rows=2, selection=SELECTION, **EMA),
            train=shared["train"],
            validation=shared["validation"],
            output=tmp_path / "adapters",
            optimizer_factory=RecordingOptimizer,
            posttraining=dict(kind="adapters"),
        )
    same_seed = dict(shrink_perturb=dict(shrink=0.6, noise=0.01, seed=42))
    with pytest.raises(ValueError, match="otra semilla"):
        trainer(shared, tmp_path / "seed", **same_seed)
    with pytest.raises(ValueError, match="declara la receta para F3"):
        trainer(shared, tmp_path / "adamw", factory=torch.optim.AdamW, **FREE)


def own_weights(engine):
    return {name: value.detach().clone() for name, value in engine.trainable.items()}


@pytest.mark.parametrize("options", [EMA, FREE], ids=["weight_ema", "schedule_free"])
def test_validation_and_the_saved_model_use_the_emitted_weights(shared, tmp_path, options):
    factory = ScheduleFreeDouble if "schedule_free" in options else RecordingOptimizer
    engine = trainer(shared, tmp_path / "run", factory=factory, **options)
    engine._start_weights({})
    own, own_id = own_weights(engine), engine.predictor._parameter_id
    if engine.shadow is not None:
        for value in engine.shadow.shadow:
            value.add_(0.01)
    else:
        for value in engine.trainable.values():
            engine.optimizer.state[value]["z"] = value.detach() + 0.01
    emitted = engine._emission().emitted()
    states = engine._model_states()
    assert set(states) == {"model", "trained_model"} | (
        {"weight_average"} if "weight_average" in options else set()
    )
    for name, value in emitted.items():
        assert torch.equal(states["model"][name], value)
        assert torch.equal(states["trained_model"][name], own[name])
        assert not torch.equal(value, own[name])
    with engine._emitted_weights():
        assert engine.predictor._parameter_id != own_id
        metrics = engine.evaluate(engine.validation)
    assert engine.predictor._parameter_id == own_id
    assert all(torch.equal(value, own[name]) for name, value in engine.trainable.items())
    twin = predictor(shared)
    twin.load_state_dict(states["model"])
    reference = trainer(shared, tmp_path / "twin", model=twin)
    assert reference.evaluate(reference.validation) == metrics


def test_the_shadow_follows_every_update_of_the_run(shared, tmp_path, monkeypatch):
    calls, original = [], weight_averaging.ShadowWeights.update

    def spy(self, named):
        calls.append(sorted(named))
        return original(self, named)

    monkeypatch.setattr(weight_averaging.ShadowWeights, "update", spy)
    engine = trainer(shared, tmp_path / "ema", **EMA)
    report = engine.run()
    assert report["status"] == "completed"
    assert len(calls) == engine.optimizer.calls == engine.global_step > 2
    assert all(names == sorted(engine.trainable) for names in calls)
    best = load_training_state(
        tmp_path / "ema/checkpoints", expected_identity=engine.identity, selection="best"
    )
    assert best["weight_average"]["updates"] == best["global_step"]
    assert "weight_averaging" in str(engine.identity["implementation"])


def drifted_update(self, named):
    # Sombra distinta de los pesos sin ajustar nada: suma una constante en cada paso.
    for value in self.shadow:
        value.add_(1e-3)
    self.updates += 1


@pytest.mark.parametrize("options", [EMA, FREE, SHRINK], ids=["ema", "free", "shrink"])
def test_resume_restores_the_trained_weights_and_reproduces_the_run(
    shared, tmp_path, options, monkeypatch
):
    monkeypatch.setattr(weight_averaging.ShadowWeights, "update", drifted_update)
    factory = ScheduleFreeDouble if "schedule_free" in options else RecordingOptimizer
    values = dict(options, checkpoint_updates=2, factory=factory)
    continuous = trainer(shared, tmp_path / "continuous", **values)
    expected = continuous.run()
    first, stop = trainer(shared, tmp_path / "interrupted", **values), StopAtStep(5)
    stop.trainer = first
    assert first.run(stop=stop)["status"] == "paused"
    second = trainer(shared, tmp_path / "interrupted", **values)
    resumed = second.run(resume=True)
    assert resumed["status"] == "completed"
    assert first.audit + second.audit == continuous.audit
    joined = named_records(first) + named_records(second)
    assert len(joined) == len(named_records(continuous))
    for left, right in zip(joined, named_records(continuous), strict=True):
        for name, value in left.items():
            if value is None:
                assert right[name] is None
            else:
                torch.testing.assert_close(value, right[name], rtol=0, atol=0)
    assert resumed["history"] == expected["history"]
    assert resumed.get("shrink_perturb") == expected.get("shrink_perturb")
    left, right = (
        load_training_state(engine.output / "checkpoints", expected_identity=engine.identity)
        for engine in (second, continuous)
    )
    for key in set(left) & {"model", "trained_model"}:
        for name, value in left[key].items():
            if isinstance(value, torch.Tensor):
                assert torch.equal(value, right[key][name]), (key, name)


def test_shrink_and_perturb_applies_once_with_a_fresh_initialization(shared, tmp_path):
    engine = trainer(shared, tmp_path / "shrink", **SHRINK)
    parent, report = own_weights(engine), {}
    generator = torch.random.get_rng_state()
    engine._start_weights(report)
    # La inicialización nueva no consume el generador global del ajuste.
    assert torch.equal(torch.random.get_rng_state(), generator)
    fresh = dict(predictor(shared, seed=7).named_parameters())
    shrunk = own_weights(engine)
    for name, value in shrunk.items():
        expected = 0.6 * parent[name] + 0.01 * fresh[name].detach()
        torch.testing.assert_close(value, expected, rtol=1e-15, atol=1e-18)
    record = report["shrink_perturb"]
    assert record["parameters_before_sha256"] == engine.identity["initial_parameters_sha256"]
    assert record["parameters_after_sha256"] == engine.predictor._parameter_id
    assert record["fresh_parameters_sha256"] == predictor(shared, seed=7)._parameter_id
    assert record["configuration"]["shrink"] == 0.6
    # Con un estado recuperado no se vuelve a contraer.
    engine._start_weights(report, state={})
    assert all(torch.equal(value, shrunk[name]) for name, value in own_weights(engine).items())


@pytest.mark.parametrize(
    "path",
    [
        "configs/experiments/seed-ema-ensemble-v1.json",
        "configs/experiments/schedule-free-shrink-perturb-v1.json",
    ],
)
def test_declared_arms_build_their_recipes_and_pair_their_controls(path):
    import json
    from pathlib import Path

    declaration = json.loads(Path(path).read_text())
    base = json.loads(Path(declaration["base_recipe"]).read_text())
    assert declaration["status"] == "declared_not_trained"
    assert declaration["final_test_access"] is False and declaration["default_enabled"] is False
    for case in base["walk_forward"]["search_cases"].values():
        for name, arm in declaration["arms"].items():
            recipe = ChronologicalRecipe(**base["recipe"] | case | arm["recipe_changes"])
            options = recipe.weight_options()
            if arm["role"] == "variant":
                control = declaration["arms"][arm["control"]]
                assert control["role"] == "control" or "seeds" in control, name
                assert arm.get("applies_to") == control.get("applies_to"), name
            if options["shrink_perturb"] is not None:
                # θ₀ nunca es la inicialización de una semilla de ajuste de los protocolos.
                assert options["shrink_perturb"].seed not in (42, 43, 44)
            if arm.get("combine") == "seed_mean_v1":
                assert len(arm["seeds"]) >= 2
                assert declaration["arms"][arm["control"]]["updates_multiplier"] == len(
                    arm["seeds"]
                )
