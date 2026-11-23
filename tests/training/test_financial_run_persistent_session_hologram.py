"""Recorrido cronológico con campo holográfico persistente multiescala (MPSH)."""

import hashlib
import json
from dataclasses import asdict

import pytest
import torch

from mars_titan.models.titans.financial import FinancialConfig, FinancialPredictor
from mars_titan.models.titans.persistent_session_hologram import (
    INSTANT_ONLY,
    PERSISTENT,
    SELF_ONLY,
    SHUFFLED_HISTORY,
    PersistentSessionHologramConfig,
)
from mars_titan.training import mars_titan_walk_forward
from mars_titan.training.checkpoints import load_training_state
from mars_titan.training.financial_run import (
    ChronologicalRecipe,
    ChronologicalTrainer,
    clip_parameter_groups,
)
from tests.models.titans.test_financial_adapter import specification
from tests.training.chronological_fixture import decision
from tests.training.test_financial_run import SELECTION, StopAtStep, entries
from tests.training.test_financial_run_session_objective import (
    explicit_fastpath,  # noqa: F401
    records,
    streams,
    trainer,
)

MPSH = PersistentSessionHologramConfig(
    bands=4, phase_dim=8, half_lives=(1, 5, 10), mode=PERSISTENT, position="after_mac", seed=0
)
INSTANT = PersistentSessionHologramConfig(
    bands=4, phase_dim=8, half_lives=(1, 5, 10), mode=INSTANT_ONLY, position="after_mac", seed=0
)
SHUFFLED = PersistentSessionHologramConfig(
    bands=4,
    phase_dim=8,
    half_lives=(1, 5, 10),
    mode=SHUFFLED_HISTORY,
    position="after_mac",
    seed=0,
)
SELF = PersistentSessionHologramConfig(
    bands=4, phase_dim=8, half_lives=(1, 5, 10), mode=SELF_ONLY, position="after_mac", seed=0
)


@pytest.fixture(scope="module")
def shared(tmp_path_factory):
    return streams(tmp_path_factory.mktemp("mpsh-shared"))


def mpsh_predictor(sources, config=MPSH, *, gate_fill=None):
    cfg = FinancialConfig(
        sources["train"].specification(),
        hidden_size=32,
        variant="mac_online",
        memory_residual_layer_norm=True,
        gate_bias=dict(alpha_half_life=256.0, eta=0.5, theta=0.05),
        persistent_session_hologram=config,
    )
    model = FinancialPredictor(cfg, dtype=torch.float64)
    if gate_fill is not None and config is not None:
        with torch.no_grad():
            model.persistent_session_hologram.gate.fill_(gate_fill)
        model._seal_parameters()
    return model


@pytest.mark.parametrize("config", [MPSH, INSTANT, SHUFFLED, SELF])
def test_zero_init_matches_control_and_step_zero_reaches_mpsh_gate(shared, tmp_path, config):
    control = trainer(
        shared,
        tmp_path / f"control-{config.mode}",
        model=mpsh_predictor(shared, None),
        truncation=2,
        max_grad_norm=None,
    )
    control.run()
    variant = trainer(
        shared,
        tmp_path / f"variant-{config.mode}",
        model=mpsh_predictor(shared, config),
        truncation=2,
        max_grad_norm=None,
    )
    report = variant.run()
    assert report["status"] == "completed"
    assert variant.audit == control.audit
    left, right = records(variant), records(control)
    assert len(left) == len(right) > 2
    gate_grads = [step["persistent_session_hologram.gate"] for step in left]
    assert gate_grads[0] is not None and torch.isfinite(gate_grads[0]).all()
    assert gate_grads[0].abs().sum() > 0


def test_real_optimizer_steps_update_all_mpsh_parameters_including_multiscale_weights(
    shared, tmp_path
):
    model = mpsh_predictor(shared, MPSH, gate_fill=0.05)
    engine = ChronologicalTrainer(
        model,
        ChronologicalRecipe(truncation=2, epochs=1, block_rows=8, selection=SELECTION),
        train=shared["train"],
        validation=shared["validation"],
        output=tmp_path / "opt",
        audit=True,
    )
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    report = engine.run()
    assert report["status"] == "completed"
    for name in (
        "persistent_session_hologram.gate",
        "persistent_session_hologram.out_inst.weight",
        "persistent_session_hologram.out_mem.weight",
        "persistent_session_hologram.out_rot.weight",
        "persistent_session_hologram.scale_mem",
        "persistent_session_hologram.scale_rot",
        "persistent_session_hologram.proj.weight",
        "persistent_session_hologram.density.weight",
    ):
        assert not torch.equal(dict(model.named_parameters())[name], before[name]), name


def test_strict_causality_future_perturbation_and_cross_session_commit_boundary(tmp_path):
    cut = decision("2022-12-09")
    runs = {}
    for name, perturb, cfg, fill in (
        ("instant", None, INSTANT, 0.08),
        ("base", None, MPSH, 0.08),
        ("future", cut, MPSH, 0.08),
    ):
        src = streams(tmp_path / name, perturb_after=perturb)
        engine = trainer(
            src,
            tmp_path / name / "run",
            model=mpsh_predictor(src, cfg, gate_fill=fill),
            truncation=2,
        )
        engine.run()
        runs[name] = engine

    base_preds = entries(runs["base"].audit, "prediction", "train")
    future_preds = entries(runs["future"].audit, "prediction", "train")
    instant_preds = entries(runs["instant"].audit, "prediction", "train")
    # En la primera sesión tras calentamiento ya existe historial previo, así que MPSH difiere
    # de instant_only en train, y perturbar tras `cut` no altera nada <= cut.
    assert base_preds != instant_preds
    assert [e for e in base_preds if e[3] <= cut] == [e for e in future_preds if e[3] <= cut]
    assert base_preds != future_preds
    steps = [u[2] for u in entries(runs["base"].audit, "update") if u[1] <= cut]
    assert steps
    for left, right in zip(
        records(runs["base"])[: len(steps)], records(runs["future"]), strict=False
    ):
        assert left.keys() == right.keys()
        for key in left:
            torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)


def test_phase_boundaries_reset_persistent_hologram_between_train_validation_and_evaluation(
    shared, tmp_path
):
    """Cada fase (ajuste, validación, evaluación) arranca con el holograma persistente vacío."""
    model = mpsh_predictor(shared, MPSH, gate_fill=0.08)
    engine = trainer(shared, tmp_path / "phases", model=model, truncation=2)

    seen_updates = []
    original_prepare = model.prepare

    def spy(batch, state, **options):
        holo = options.get("persistent_hologram")
        updates = 0 if not holo.markets else sum(m.updates for m in holo.markets.values())
        seen_updates.append((model.training, updates))
        return original_prepare(batch, state, **options)

    model.prepare = spy
    report = engine.run()
    assert report["status"] == "completed"
    # El primer evento de validación (epoch 0), el primer evento de ajuste (epoch 0) y el
    # primer evento de validación (epoch 1) ven todos updates == 0 (reinicio exacto de fase).
    train_starts = [
        updates
        for idx, (is_train, updates) in enumerate(seen_updates)
        if is_train and (idx == 0 or not seen_updates[idx - 1][0])
    ]
    val_starts = [
        updates
        for idx, (is_train, updates) in enumerate(seen_updates)
        if not is_train and (idx == 0 or seen_updates[idx - 1][0])
    ]
    assert train_starts and all(u == 0 for u in train_starts)
    assert len(val_starts) >= 2 and all(u == 0 for u in val_starts)

    # Y una predicción posterior en evaluación/validación también reinicia en 0 y coincide
    # exactamente con una inferencia limpia sobre el mismo tramo.
    seen_updates.clear()
    rows_a = []
    engine.predict_partition(shared["validation"], rows_a)
    assert seen_updates[0] == (False, 0)
    assert seen_updates[-1][1] > 0

    rows_b = []
    engine.predict(shared["validation"], rows_b)
    assert rows_a == rows_b


@pytest.mark.parametrize("observe_without_graph", [False, True])
@pytest.mark.parametrize("config", [MPSH, SHUFFLED])
def test_replay_with_accumulation_matches_direct_backward_with_and_without_observe_graph(
    shared, tmp_path, config, observe_without_graph
):
    direct = trainer(
        shared,
        tmp_path / f"direct-{config.mode}-{observe_without_graph}",
        model=mpsh_predictor(shared, config, gate_fill=0.06),
        truncation=2,
    )
    direct.run()
    replayed = trainer(
        shared,
        tmp_path / f"replay-{config.mode}-{observe_without_graph}",
        model=mpsh_predictor(shared, config, gate_fill=0.06),
        truncation=2,
        accumulation_rows=8,
        observe_without_graph=observe_without_graph,
    )
    replayed.run()
    assert replayed.audit == direct.audit
    left, right = records(direct), records(replayed)
    assert len(left) == len(right) > 2
    for first, second in zip(left, right, strict=True):
        assert first.keys() == second.keys()
        for name in first:
            torch.testing.assert_close(second[name], first[name], rtol=1e-10, atol=1e-12)


def test_checkpoint_resume_after_interruption_reproduces_continuous_run_bitwise(shared, tmp_path):
    options = dict(truncation=2, epochs=2, checkpoint_updates=2, block_rows=8)

    def make(name):
        return trainer(
            shared,
            tmp_path / name,
            model=mpsh_predictor(shared, SHUFFLED, gate_fill=0.06),
            **options,
        )

    continuous = make("continuous")
    expected = continuous.run()
    first = make("interrupted")
    stop = StopAtStep(3)
    stop.trainer = first
    assert first.run(stop=stop)["status"] == "paused"
    saved = load_training_state(
        tmp_path / "interrupted/checkpoints", expected_identity=first.identity
    )
    assert "persistent_hologram" in saved["run"]
    assert saved["run"]["persistent_hologram"]["markets"]["US"]["updates"] > 0

    second = make("interrupted")
    resumed = second.run(resume=True)
    assert resumed["status"] == "completed"
    assert first.audit + second.audit == continuous.audit
    joined = records(first) + records(second)
    expected_grads = records(continuous)
    assert len(joined) == len(expected_grads)
    for a, b in zip(joined, expected_grads, strict=True):
        for name in a:
            torch.testing.assert_close(a[name], b[name], rtol=0, atol=0)
    assert resumed["history"] == expected["history"]


def test_clip_groups_split_session_rejection_and_frozen_parent_guard(shared, tmp_path):
    model = mpsh_predictor(shared, MPSH)
    groups = clip_parameter_groups(model)
    assert "persistent_session_hologram.gate" in groups["mac"]
    assert "persistent_session_hologram.out_mem.weight" in groups["mac"]
    assert "persistent_session_hologram.scale_rot" in groups["mac"]

    split = trainer(shared, tmp_path / "split", model=model, block_rows=2)
    with pytest.raises(ValueError, match="sesión completa"):
        split.run()

    view = tmp_path / "view.json"
    view.write_text("{}\n")
    report = dict(
        kind=mars_titan_walk_forward.TITANS_KIND,
        status="completed",
        final_test_opened=False,
        checkpoint={},
        request=dict(
            variant="mac_online",
            seed=42,
            view_sha256=hashlib.sha256(view.read_bytes()).hexdigest(),
            predictor_changes=dict(persistent_session_hologram=asdict(MPSH)),
        ),
        identity=dict(recipe=dict(predictor=dict(hidden_size=32, dtype="float64"))),
    )
    (tmp_path / "run.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="holograma persistente"):
        mars_titan_walk_forward._parent(tmp_path, view, 42)
    with pytest.raises(ValueError, match="holograma persistente"):
        mars_titan_walk_forward._frozen_parent(
            dict(predictor=dict(hidden_size=32, dtype="float64")),
            specification(),
            42,
            "cpu",
            tmp_path,
            report,
        )
