"""Recorrido cronológico con memoria neuronal supervisada por IC (`ic_titans_mlp`)."""

import hashlib
import json
from dataclasses import asdict, replace

import pytest
import torch

from mars_titan.models.titans.cross_sectional_rank_memory import CrossSectionalRankMemoryConfig
from mars_titan.models.titans.financial import FinancialConfig, FinancialPredictor
from mars_titan.models.titans.ic_titans_mlp import ICTitansMLPConfig
from mars_titan.models.titans.session_hologram import SessionHologramConfig
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

IC_MLP = ICTitansMLPConfig(
    dim=8,
    alpha_half_life=21.0,
    eta=0.5,
    theta=0.05,
    anchor=0.1,
    grad_clip=1.0,
    inner_steps=1,
    minimum_rows=3,
    mode="matured_rank",
    seed=0,
)
SHUFFLED = replace(IC_MLP, mode="shuffled_targets")
EVAL_FROZEN = replace(IC_MLP, mode="eval_frozen")


@pytest.fixture(scope="module")
def shared(tmp_path_factory):
    return streams(tmp_path_factory.mktemp("ic-mlp-shared"))


def ic_mlp_predictor(sources, config=IC_MLP, *, gate_fill=None, **extra):
    cfg = FinancialConfig(
        sources["train"].specification(),
        hidden_size=32,
        variant="mac_online",
        memory_residual_layer_norm=True,
        gate_bias=dict(alpha_half_life=256.0, eta=0.5, theta=0.05),
        ic_titans_mlp=config,
        **extra,
    )
    model = FinancialPredictor(cfg, dtype=torch.float64)
    if gate_fill is not None and config is not None:
        with torch.no_grad():
            model.ic_titans_mlp.gate.fill_(gate_fill)
        model._seal_parameters()
    return model


def test_zero_init_reproduces_control_and_step_one_reaches_gate(shared, tmp_path):
    """El tronco coincide bit a bit con el control y `gate` recibe gradiente ya en el paso 1."""
    control = trainer(
        shared,
        tmp_path / "control",
        model=ic_mlp_predictor(shared, None),
        truncation=2,
        max_grad_norm=None,
    )
    control.run()
    variant = trainer(
        shared,
        tmp_path / "variant",
        model=ic_mlp_predictor(shared, IC_MLP),
        truncation=2,
        max_grad_norm=None,
    )
    report = variant.run()
    assert report["status"] == "completed"
    assert variant.audit == control.audit
    left, right = records(variant), records(control)
    assert len(left) == len(right) > 2
    gate_grads = [step.pop("ic_titans_mlp.gate") for step in left]
    for step in left:
        for name in list(step):
            if name.startswith("ic_titans_mlp."):
                step.pop(name)
    for a, b in zip(left, right, strict=True):
        assert a.keys() == b.keys()
        for name in a:
            torch.testing.assert_close(a[name], b[name], rtol=0, atol=0)
    assert gate_grads[0] is not None and torch.count_nonzero(gate_grads[0]) > 0
    assert all(torch.isfinite(g).all() for g in gate_grads)


def test_real_optimization_updates_all_ic_titans_mlp_parameters(shared, tmp_path):
    model = ic_mlp_predictor(shared, IC_MLP, gate_fill=0.05)
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
        "ic_titans_mlp.gate",
        "ic_titans_mlp.out.weight",
        "ic_titans_mlp.readout_norm.weight",
        "ic_titans_mlp.initial_weights.0",
        "ic_titans_mlp.initial_weights.1",
        "ic_titans_mlp.w_read",
        "ic_titans_mlp.key_proj.weight",
        "ic_titans_mlp.norm.weight",
        "ic_titans_mlp.alpha_logits",
        "ic_titans_mlp.eta_logits",
        "ic_titans_mlp.theta_logits",
        "ic_titans_mlp.anchor_logits",
    ):
        assert not torch.equal(dict(model.named_parameters())[name], before[name]), name


def test_strict_causality_future_perturbation_and_no_memory_divergence_before_maturity(tmp_path):
    """Perturbar el futuro no cambia <= corte y antes de madurar coinciden ambos modos."""
    cut = decision("2022-12-09")
    runs = {}
    for name, perturb, cfg, fill in (
        ("shuffled", None, SHUFFLED, 0.08),
        ("base", None, IC_MLP, 0.08),
        ("future", cut, IC_MLP, 0.08),
    ):
        src = streams(tmp_path / name, perturb_after=perturb)
        engine = trainer(
            src,
            tmp_path / name / "run",
            model=ic_mlp_predictor(src, cfg, gate_fill=fill),
            truncation=2,
        )
        engine.run()
        runs[name] = engine

    shuf_preds = entries(runs["shuffled"].audit, "prediction", "train")
    base_preds = entries(runs["base"].audit, "prediction", "train")
    future_preds = entries(runs["future"].audit, "prediction", "train")
    first_maturity = min(e[4] for e in entries(runs["base"].audit, "label", "train"))
    # Antes de madurar ninguna etiqueta ambos leen exactamente M_0 y coinciden bit a bit.
    assert [e for e in base_preds if e[3] < first_maturity] == [
        e for e in shuf_preds if e[3] < first_maturity
    ]
    assert [e for e in base_preds if e[3] >= first_maturity] != [
        e for e in shuf_preds if e[3] >= first_maturity
    ]
    # Perturbar el futuro tras `cut` no altera ninguna predicción ni gradiente <= cut.
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


@pytest.mark.parametrize("observe_without_graph", [False, True])
@pytest.mark.parametrize("config", [IC_MLP, SHUFFLED])
def test_replay_matches_direct_backward_and_supports_observe_without_graph(
    shared, tmp_path, config, observe_without_graph
):
    direct = trainer(
        shared,
        tmp_path / f"direct-{config.mode}-{observe_without_graph}",
        model=ic_mlp_predictor(shared, config, gate_fill=0.06),
        truncation=2,
    )
    direct.run()
    replayed = trainer(
        shared,
        tmp_path / f"replay-{config.mode}-{observe_without_graph}",
        model=ic_mlp_predictor(shared, config, gate_fill=0.06),
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


def test_resume_after_interruption_reproduces_continuous_run_bit_for_bit(shared, tmp_path):
    options = dict(truncation=2, epochs=2, checkpoint_updates=2, block_rows=8)

    def make(name):
        return trainer(
            shared,
            tmp_path / name,
            model=ic_mlp_predictor(shared, SHUFFLED, gate_fill=0.06),
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
    assert "ic_mlp_memory" in saved["run"]
    assert saved["run"]["ic_mlp_memory"]["markets"]["US"]["updates"] > 0
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


def test_eval_frozen_matches_matured_rank_in_train_and_freezes_test_time_updates_in_validation(
    shared, tmp_path
):
    frozen = trainer(
        shared,
        tmp_path / "frozen",
        model=ic_mlp_predictor(shared, EVAL_FROZEN, gate_fill=0.08),
        truncation=2,
    )
    frozen.run()
    online = trainer(
        shared,
        tmp_path / "online",
        model=ic_mlp_predictor(shared, IC_MLP, gate_fill=0.08),
        truncation=2,
    )
    online.run()
    # En ajuste coinciden bit a bit.
    assert entries(frozen.audit, "prediction", "train") == entries(
        online.audit, "prediction", "train"
    )
    # En validación, antes de la primera maduración ambos parten de M_0 (reinicio de fase),
    # y a partir de la primera maduración `online` se adapta mientras `eval_frozen` mantiene M_0.
    val_frozen = entries(frozen.audit, "prediction", "validation")
    val_online = entries(online.audit, "prediction", "validation")
    first_val_maturity = min(e[4] for e in entries(online.audit, "label", "validation"))
    assert [e for e in val_online if e[3] < first_val_maturity] == [
        e for e in val_frozen if e[3] < first_val_maturity
    ]
    assert [e for e in val_online if e[3] >= first_val_maturity] != [
        e for e in val_frozen if e[3] >= first_val_maturity
    ]


def test_coexistence_in_chronological_trainer_with_crrm_and_hologram(shared, tmp_path):
    """IC-Titans-MLP convive con CRRM y holograma de sesión en el mismo recorrido con replay."""
    hologram = SessionHologramConfig(
        bands=2, phase_dim=4, mode="interferometric", position="after_mac"
    )
    crrm = CrossSectionalRankMemoryConfig(
        dim=8, time_constants=(5, 21), ridge=1.0, minimum_rows=3, mode="matured_rank", seed=0
    )

    def build_combo():
        m = ic_mlp_predictor(
            shared,
            IC_MLP,
            gate_fill=0.05,
            session_hologram=hologram,
            cross_sectional_rank_memory=crrm,
        )
        with torch.no_grad():
            m.session_hologram.gate.fill_(0.05)
            m.cross_sectional_rank_memory.out.weight.fill_(0.05)
        m._seal_parameters()
        return m

    direct = trainer(shared, tmp_path / "combo-direct", model=build_combo(), truncation=2)
    direct.run()
    replayed = trainer(
        shared,
        tmp_path / "combo-replay",
        model=build_combo(),
        truncation=2,
        accumulation_rows=8,
        observe_without_graph=True,
    )
    replayed.run()
    assert replayed.audit == direct.audit
    for a, b in zip(records(direct), records(replayed), strict=True):
        assert a.keys() == b.keys()
        for name in a:
            torch.testing.assert_close(b[name], a[name], rtol=1e-10, atol=1e-12)


def test_clip_groups_split_block_rejection_and_frozen_parent_guard(shared, tmp_path):
    model = ic_mlp_predictor(shared, IC_MLP)
    groups = clip_parameter_groups(model)
    assert "ic_titans_mlp.gate" in groups["mac"]
    assert "ic_titans_mlp.initial_weights.0" in groups["mac"]
    assert "ic_titans_mlp.w_read" in groups["mac"]

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
            predictor_changes=dict(ic_titans_mlp=asdict(IC_MLP)),
        ),
        identity=dict(recipe=dict(predictor=dict(hidden_size=32, dtype="float64"))),
    )
    (tmp_path / "run.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="IC-Titans-MLP"):
        mars_titan_walk_forward._parent(tmp_path, view, 42)
    with pytest.raises(ValueError, match="IC-Titans-MLP"):
        mars_titan_walk_forward._frozen_parent(
            dict(predictor=dict(hidden_size=32, dtype="float64")),
            specification(),
            42,
            "cpu",
            tmp_path,
            report,
        )
