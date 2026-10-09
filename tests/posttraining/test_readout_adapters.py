"""Brazos adaptados de MARS-TITAN y CM-v1 comprobados hasta el paso, sin cambiar pesos.

Cada brazo parte del mismo núcleo `mac_online` y del mismo lector. El optimizador registra
gradientes sin modificar parámetros, así que todos los pasos de un brazo se evalúan en el
mismo punto y se pueden comparar entre brazos y entre tamaños de bloque.
"""

import copy
import json
from pathlib import Path

import pytest
import torch

from mars_titan.models.predictive_adaptation import adapter_names, attach_adapters, base_digest
from mars_titan.posttraining import chronological_matrix as cm
from mars_titan.posttraining.adapter_matrix import read_matrix
from mars_titan.posttraining.readout_adapters import ReadoutAdapterTrainer
from mars_titan.training import mars_titan_run as mt
from mars_titan.training.checkpoints import load_training_state
from mars_titan.training.learning_hold import HOLD_ENV
from tests.training.test_financial_run import RecordingOptimizer, StopAtStep, entries
from tests.training.test_financial_run import shared as shared
from tests.training.test_financial_run_control import control
from tests.training.test_mars_titan_run import native as native
from tests.training.test_mars_titan_run import parent, reader, recipe

ROOT = Path(__file__).resolve().parents[2]
MATRIX, DIGEST = read_matrix(ROOT / "configs/posttraining/adapter-matrix-v3.json")
ARMS = ("full_continuation", "episodic_readout", "core", "core+episodic_readout")
REPLAY = "segment_replayed_by_flow_blocks_from_segment_start_bptt_v1"


@pytest.fixture(autouse=True)
def explicit_fastpath():
    previous = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    yield
    torch.backends.mha.set_fastpath_enabled(previous)


def spec(arm):
    """Puntos del núcleo y forma de la lectura episódica del brazo en la matriz."""
    found = {item["id"]: item for item in cm.arms(MATRIX, "mars_titan")}
    return found[arm]


def build(streams, output, native, arm, *, admission="m1", local_control=None, **options):
    plan = options.pop("plan", recipe())
    core_rows = options.pop("core_rows", None)
    readout, codec = reader(streams, admission)
    predictor = parent(streams, local_control=local_control)
    if arm != "full_continuation":
        readout.requires_grad_(False)
        declared = spec(arm)
        if declared["core"] is not None:
            attach_adapters(predictor, cm.titans_targets(MATRIX, declared["core"]), seed=7)
            predictor._seal_parameters()
        if declared["episodic_readout"] is not None:
            targets = cm.readout_targets(MATRIX, declared["episodic_readout"])
            attach_adapters(readout, targets, seed=8)
    return ReadoutAdapterTrainer(
        predictor,
        readout,
        plan,
        posttraining=dict(
            kind="technical_check",
            arm=arm,
            control="full_continuation" if arm == "full_continuation" else None,
        ),
        core_rows=core_rows,
        admission=admission,
        retention=mt.retention_config(plan, admission),
        native=native,
        codec=codec,
        train=streams["train"],
        validation=streams["validation"],
        output=output,
        world="fixture",
        fold="0",
        optimizer_factory=RecordingOptimizer,
        audit=True,
        **options,
    )


def named(engine):
    """Gradientes de cada paso con el nombre de su papel (`core.` para el núcleo)."""
    names = {id(value): name for name, value in engine.readout.named_parameters()}
    names |= {id(value): f"core.{name}" for name, value in engine.predictor.named_parameters()}
    return [
        {names[key]: value for key, value in record.items()} for record in engine.optimizer.records
    ]


def without_predictions(audit):
    return [entry for entry in audit if entry[0] != "prediction"]


@pytest.fixture(scope="module")
def arms(shared, tmp_path_factory, native):
    """Los cuatro brazos de M1 sobre el mismo padre y lector, con el registrador."""
    _, streams = shared
    root = tmp_path_factory.mktemp("readout-adapters")
    hold = root / "hold.json"
    hold.write_text(json.dumps({"training_allowed": True}))
    result = {}
    with pytest.MonkeyPatch.context() as patch:
        # Dobles del aprendizaje: el registrador nunca modifica pesos.
        patch.setenv(HOLD_ENV, str(hold))
        previous = torch.backends.mha.get_fastpath_enabled()
        torch.backends.mha.set_fastpath_enabled(False)
        try:
            for arm in ARMS:
                engine = build(streams, root / arm.replace("+", "_"), native, arm)
                core = base_digest(engine.predictor)
                readout = base_digest(engine.readout)
                result[arm] = (engine, engine.run(), core, readout)
        finally:
            torch.backends.mha.set_fastpath_enabled(previous)
    return result


def test_arms_share_events_updates_and_issued_predictions(arms):
    baseline, report, _, _ = arms["full_continuation"]
    expected = without_predictions(baseline.audit)
    issued = entries(baseline.audit, "prediction")
    assert entries(baseline.audit, "update") and entries(baseline.audit, "admit")
    for arm, (engine, result, _, _) in arms.items():
        assert result["status"] == "completed", arm
        assert without_predictions(engine.audit) == expected, arm
        # Ni el núcleo adaptado ni la lectura adaptada cambian un bit de lo emitido.
        assert entries(engine.audit, "prediction") == issued, arm
        assert engine.optimizer.calls == baseline.optimizer.calls > 0, arm
        keys = ("updates", "labels", "labels_in_loss", "segments", "admitted")
        assert {k: result["history"][-1]["train"][k] for k in keys} == {
            k: report["history"][-1]["train"][k] for k in keys
        }, arm


@pytest.mark.parametrize("arm", ARMS[1:])
def test_gradients_reach_only_the_declared_corrections(arms, arm):
    engine, _, core, readout = arms[arm]
    declared = spec(arm)
    expected = set()
    if declared["core"] is not None:
        expected |= {f"core.{name}" for name in adapter_names(engine.predictor)}
    if declared["episodic_readout"] is not None:
        expected |= set(adapter_names(engine.readout))
    roles = {name for names in engine.roles.values() for name in names}
    assert roles == expected
    records = named(engine)
    assert records and all(set(record) == expected for record in records)
    if declared["core"] is None:
        # Con el banco aún vacío la lectura no interviene y el paso no tiene gradiente.
        records = [r for r in records if any(v is not None for v in r.values())]
    for name, value in records[0].items():
        assert value is not None and torch.isfinite(value).all(), name
        if name.endswith(".down"):
            assert not value.abs().sum(), name
        else:
            assert value.abs().sum() > 0, name
    # Los tensores originales del núcleo y del lector no cambian.
    assert base_digest(engine.predictor) == core
    assert base_digest(engine.readout) == readout
    identity = engine.identity["posttraining"]
    assert identity["core_gradient"] == (None if declared["core"] is None else REPLAY)
    assert identity["control_penalty_on_adapted_operator"] is False


def test_episodic_gradients_follow_the_chain_rule_of_the_readout_continuation(
    shared, tmp_path, native
):
    """dL/dU = (α/r) dL/dW Vᵀ y dL/dV = 0 frente a la continuación del lector, sin recorte."""
    _, streams = shared
    plan = recipe(max_grad_norm=None)
    full = build(streams, tmp_path / "full", native, "full_continuation", plan=plan)
    full.run()
    arm = build(streams, tmp_path / "arm", native, "episodic_readout", plan=plan)
    arm.run()
    values = dict(arm.readout.named_parameters())
    reference, records = named(full), named(arm)
    assert len(records) == len(reference) > 3
    nonzero = 0
    for whole, record in zip(reference, records, strict=True):
        for name, gradient in record.items():
            module = name.split(".parametrizations.")[0]
            original = whole[f"{module}.weight"]
            if gradient is None:
                # Sin episodios la lectura no interviene y tampoco recibe gradiente en el padre.
                assert original is None or not original.abs().sum(), name
                continue
            if name.endswith(".up"):
                delta = arm.readout.get_submodule(module).parametrizations.weight[0]
                expected = delta.scaling * original @ values[name[: -len("up")] + "down"].T
                nonzero += bool(expected.abs().sum())
            else:
                expected = torch.zeros_like(gradient)
            torch.testing.assert_close(gradient, expected, rtol=1e-9, atol=1e-12)
    assert nonzero > 0


def test_full_continuation_fits_the_whole_readout_with_the_frozen_core(arms):
    engine, _, _, _ = arms["full_continuation"]
    assert set(engine.roles) == set(mt.parameter_roles(engine.readout, "m1")[0])
    assert all(not value.requires_grad for value in engine.predictor.parameters())
    assert engine.identity["parent"]["state"] == "selected_checkpoint_frozen"
    assert engine.identity["posttraining"]["core_base_parameters_sha256"] is None


def test_core_gradient_does_not_depend_on_the_flow_block_size(shared, tmp_path, native):
    _, streams = shared
    whole = build(streams, tmp_path / "whole", native, "core")
    whole.run()
    for rows in (1, 2):
        blocks = build(streams, tmp_path / f"rows-{rows}", native, "core", core_rows=rows)
        blocks.run()
        assert blocks.audit == whole.audit
        left, right = named(blocks), named(whole)
        assert len(left) == len(right) > 3
        for first, second in zip(left, right, strict=True):
            for key, value in first.items():
                torch.testing.assert_close(value, second[key], rtol=1e-9, atol=1e-13)
        assert blocks.run_id != whole.run_id


def test_core_without_bank_trains_only_the_core(shared, tmp_path):
    _, streams = shared
    engine = build(streams, tmp_path / "m0", None, "core", admission="m0")
    report = engine.run()
    assert report["history"][-1]["train"]["admitted"] == 0
    records = named(engine)
    assert records and all(name.startswith("core.") for name in records[0])
    assert any(records[0][name].abs().sum() > 0 for name in records[0])


def test_c_penalty_acts_on_the_adapted_operator_and_leaves_the_head_alone(shared, tmp_path, native):
    """g(2w) − g(w) = g(w) − g_B sobre los adaptadores del núcleo, sin recorte."""
    _, streams = shared
    plan = recipe(max_grad_norm=None)
    runs = {}
    for name, settings in (
        ("b", control("disabled")),
        ("one", control("penalty", 0.5)),
        ("two", control("penalty", 1.0)),
    ):
        engine = build(streams, tmp_path / name, native, "core", local_control=settings, plan=plan)
        runs[name] = (engine, engine.run())
    b, one, two = (runs[name][0] for name in ("b", "one", "two"))
    assert one.identity["posttraining"]["control_penalty_on_adapted_operator"] is True
    assert b.identity["posttraining"]["control_penalty_on_adapted_operator"] is False
    assert entries(b.audit, "update") == entries(one.audit, "update")
    changed = set()
    for g_b, g_1, g_2 in zip(named(b), named(one), named(two), strict=True):
        for key, value in g_b.items():
            torch.testing.assert_close(g_2[key] - g_1[key], g_1[key] - value, rtol=1e-7, atol=1e-12)
            if key.startswith("core.head."):
                torch.testing.assert_close(g_1[key], value, rtol=0, atol=0)
            elif not torch.equal(g_1[key], value):
                changed.add(key.split(".")[1])
    assert {"mac", "fusion"} <= changed
    metrics = runs["one"][1]["history"][-1]["train"]
    assert metrics["control_groups_in_objective"] > 0
    assert "control_groups_in_objective" not in runs["b"][1]["history"][-1]["train"]


def test_resume_reproduces_the_continuous_run_with_both_components(shared, tmp_path, native):
    _, streams = shared
    plan = recipe(epochs=2, checkpoint_updates=2)
    arm = "core+episodic_readout"
    continuous = build(streams, tmp_path / "continuous", native, arm, plan=plan)
    expected = continuous.run()
    first = build(streams, tmp_path / "paused", native, arm, plan=plan)
    stop = StopAtStep(continuous.global_step // 2 + 1)
    stop.trainer = first
    assert first.run(stop=stop)["status"] == "paused"
    state = load_training_state(tmp_path / "paused/checkpoints", expected_identity=first.identity)
    assert list(state["core_adapters"]) == adapter_names(first.predictor)
    assert state["core_sha256"] == first.predictor._parameter_id
    second = build(streams, tmp_path / "paused", native, arm, plan=plan)
    resumed = second.run(resume=True)
    assert resumed["status"] == "completed"
    assert first.audit + second.audit == continuous.audit
    joined = named(first) + named(second)
    assert len(joined) == len(named(continuous)) == continuous.global_step
    for left, right in zip(joined, named(continuous), strict=True):
        for key, value in left.items():
            torch.testing.assert_close(value, right[key], rtol=0, atol=0)
    assert resumed["history"] == expected["history"]


def test_checkpoint_rejects_core_adapters_that_do_not_match(shared, tmp_path, native):
    _, streams = shared
    engine = build(streams, tmp_path / "run", native, "core")
    state = engine._extra_state()
    engine._restore_extra(copy.deepcopy(state))
    broken = copy.deepcopy(state)
    first = next(iter(broken["core_adapters"]))
    broken["core_adapters"][first] = broken["core_adapters"][first] + 1
    with pytest.raises(ValueError, match="huella"):
        engine._restore_extra(broken)
    missing = copy.deepcopy(state)
    missing["core_adapters"].pop(first)
    with pytest.raises(ValueError, match="adaptadores"):
        engine._restore_extra(missing)


def test_constructor_rejects_cores_and_readers_outside_the_arm(shared, tmp_path, native):
    _, streams = shared
    readout, codec = reader(streams, "m1")
    plan = recipe()
    common = dict(
        admission="m1",
        retention=mt.retention_config(plan, "m1"),
        native=native,
        codec=codec,
        train=streams["train"],
        validation=streams["validation"],
        world="fixture",
        fold="0",
        optimizer_factory=RecordingOptimizer,
    )
    with pytest.raises(ValueError, match="declaración"):
        ReadoutAdapterTrainer(
            parent(streams), readout, plan, posttraining={}, output=tmp_path / "a", **common
        )
    # Sin adaptadores el caso solo puede ser la continuación del lector.
    with pytest.raises(ValueError, match="continuación"):
        ReadoutAdapterTrainer(
            parent(streams),
            readout,
            plan,
            posttraining=dict(kind="x", control=None),
            output=tmp_path / "b",
            **common,
        )
    # Un núcleo con adaptadores debe ser mac_online, en eval y sin otros tensores ajustables.
    for change in ("variant", "train", "unfrozen", "diagnostic"):
        options = {}
        if change == "variant":
            options["variant"] = "mac_frozen"
        if change == "diagnostic":
            options["local_control"] = control("diagnostic")
        model = parent(streams, **options)
        attach_adapters(model, cm.titans_targets(MATRIX, spec("core")["core"]), seed=7)
        if change == "train":
            model.train()
        if change == "unfrozen":
            model.fusion[0].bias.requires_grad_(True)
        model._seal_parameters()
        with pytest.raises(ValueError, match="núcleo"):
            ReadoutAdapterTrainer(
                model,
                reader(streams, "m1")[0].requires_grad_(False),
                plan,
                posttraining=dict(kind="x", control=None),
                output=tmp_path / f"c-{change}",
                **common,
            )
    # Con adaptadores en el lector, sus demás tensores no se ajustan.
    readout, _ = reader(streams, "m1")
    attach_adapters(
        readout, cm.readout_targets(MATRIX, spec("episodic_readout")["episodic_readout"]), seed=8
    )
    readout.refinement.requires_grad_(True)
    with pytest.raises(ValueError, match="correcciones"):
        ReadoutAdapterTrainer(
            parent(streams),
            readout,
            plan,
            posttraining=dict(kind="x", control=None),
            output=tmp_path / "d",
            **common,
        )
    with pytest.raises(ValueError, match="bloques"):
        ReadoutAdapterTrainer(
            parent(streams),
            reader(streams, "m1")[0],
            plan,
            posttraining=dict(kind="x", control="full_continuation"),
            core_rows=0,
            output=tmp_path / "e",
            **common,
        )
