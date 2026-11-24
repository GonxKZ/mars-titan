"""Recorrido cronológico con consolidación sináptica de tres factores de rango bajo (D3F-LR)."""

import hashlib
import json
from dataclasses import asdict

import pytest
import torch

from mars_titan.models.titans.cross_sectional_rank_memory import CrossSectionalRankMemoryConfig
from mars_titan.models.titans.financial import FinancialConfig, FinancialPredictor
from mars_titan.models.titans.session_hologram import SessionHologramConfig
from mars_titan.models.titans.three_factor_consolidation import (
    CROSS_ONLY,
    EVAL_FROZEN,
    LOCAL_ONLY,
    SHUFFLED_CREDIT,
    THREE_FACTOR,
    ThreeFactorConsolidationConfig,
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

D3F = ThreeFactorConsolidationConfig(
    dim=8,
    trace_tau=5.0,
    consolidation_tau=21.0,
    credit_scale=1.0,
    max_gain=0.25,
    minimum_rows=3,
    mode=THREE_FACTOR,
    seed=0,
)
SHUFFLED = ThreeFactorConsolidationConfig(
    dim=8,
    trace_tau=5.0,
    consolidation_tau=21.0,
    credit_scale=1.0,
    max_gain=0.25,
    minimum_rows=3,
    mode=SHUFFLED_CREDIT,
    seed=0,
)
LOCAL = ThreeFactorConsolidationConfig(
    dim=8,
    trace_tau=5.0,
    consolidation_tau=21.0,
    credit_scale=1.0,
    max_gain=0.25,
    minimum_rows=3,
    mode=LOCAL_ONLY,
    seed=0,
)
CROSS = ThreeFactorConsolidationConfig(
    dim=8,
    trace_tau=5.0,
    consolidation_tau=21.0,
    credit_scale=1.0,
    max_gain=0.25,
    minimum_rows=3,
    mode=CROSS_ONLY,
    seed=0,
)
EVAL_FROZEN_CFG = ThreeFactorConsolidationConfig(
    dim=8,
    trace_tau=5.0,
    consolidation_tau=21.0,
    credit_scale=1.0,
    max_gain=0.25,
    minimum_rows=3,
    mode=EVAL_FROZEN,
    seed=0,
)


@pytest.fixture(scope="module")
def shared(tmp_path_factory):
    return streams(tmp_path_factory.mktemp("d3f-shared"))


def d3f_predictor(sources, config=D3F, *, gate_fill=None, extra=None):
    cfg = FinancialConfig(
        sources["train"].specification(),
        hidden_size=32,
        variant="mac_online",
        memory_residual_layer_norm=True,
        gate_bias=dict(alpha_half_life=256.0, eta=0.5, theta=0.05),
        three_factor_consolidation=config,
        **(extra or {}),
    )
    model = FinancialPredictor(cfg, dtype=torch.float64)
    if gate_fill is not None and config is not None:
        with torch.no_grad():
            model.three_factor_consolidation.gate.fill_(gate_fill)
        model._seal_parameters()
    return model


def test_zero_init_reproduces_control_and_step_one_reaches_gate(shared, tmp_path):
    # Sin recorte global, los gradientes crudos del tronco coinciden bit a bit con el control
    # mientras `gate` (un solo cero sobre W_out no nula) recibe gradiente ya en el primer paso.
    control = trainer(
        shared,
        tmp_path / "control",
        model=d3f_predictor(shared, None),
        truncation=2,
        max_grad_norm=None,
    )
    control.run()
    variant = trainer(
        shared,
        tmp_path / "variant",
        model=d3f_predictor(shared, D3F),
        truncation=2,
        max_grad_norm=None,
    )
    report = variant.run()
    assert report["status"] == "completed"
    assert variant.audit == control.audit
    left, right = records(variant), records(control)
    assert len(left) == len(right) > 2
    gate_grads = [step.pop("three_factor_consolidation.gate") for step in left]
    for step in left:
        for key in list(step):
            if key.startswith("three_factor_consolidation."):
                step.pop(key)
    for a, b in zip(left, right, strict=True):
        assert a.keys() == b.keys()
        for name in a:
            torch.testing.assert_close(a[name], b[name], rtol=0, atol=0)
    assert gate_grads[0] is not None and torch.count_nonzero(gate_grads[0]) > 0
    assert all(torch.isfinite(g).all() for g in gate_grads if g is not None)


def test_real_optimizer_updates_all_three_factor_parameters_on_chronological_run(shared, tmp_path):
    model = d3f_predictor(shared, D3F)
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
    after = dict(model.named_parameters())
    # Arrancando desde gate=0, el paso 1 mueve gate y los pasos siguientes mueven todos los pesos.
    for name in (
        "three_factor_consolidation.gate",
        "three_factor_consolidation.out.weight",
        "three_factor_consolidation.cross_norm_weight",
        "three_factor_consolidation.local_norm_weight",
        "three_factor_consolidation.key_proj.weight",
        "three_factor_consolidation.val_proj.weight",
        "three_factor_consolidation.norm.weight",
        "three_factor_consolidation.trace_decay_logit",
        "three_factor_consolidation.cross_decay_logit",
        "three_factor_consolidation.local_decay_logit",
        "three_factor_consolidation.credit_scale_logit",
    ):
        assert not torch.equal(after[name], before[name]), name
        assert torch.isfinite(after[name]).all(), name


def test_strict_causality_future_perturbation_and_no_change_before_first_maturity(tmp_path):
    cut = decision("2022-12-09")
    runs = {}
    for name, perturb, cfg, fill in (
        ("control", None, None, None),
        ("base", None, D3F, 0.25),
        ("future", cut, D3F, 0.25),
    ):
        src = streams(tmp_path / name, perturb_after=perturb)
        engine = trainer(
            src,
            tmp_path / name / "run",
            model=d3f_predictor(src, cfg, gate_fill=fill),
            truncation=2,
        )
        engine.run()
        runs[name] = engine

    control_preds = entries(runs["control"].audit, "prediction", "train")
    base_preds = entries(runs["base"].audit, "prediction", "train")
    future_preds = entries(runs["future"].audit, "prediction", "train")
    first_maturity = min(e[4] for e in entries(runs["base"].audit, "label", "train"))
    # Antes de la primera maduración la memoria consolidada está vacía y emite igual que el control.
    assert [e for e in base_preds if e[3] < first_maturity] == [
        e for e in control_preds if e[3] < first_maturity
    ]
    assert [e for e in base_preds if e[3] >= first_maturity] != [
        e for e in control_preds if e[3] >= first_maturity
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
@pytest.mark.parametrize("config", [D3F, SHUFFLED, LOCAL, CROSS])
def test_replay_matches_direct_backward_and_supports_observe_without_graph(
    shared, tmp_path, config, observe_without_graph
):
    direct = trainer(
        shared,
        tmp_path / f"direct-{config.mode}-{observe_without_graph}",
        model=d3f_predictor(shared, config, gate_fill=0.2),
        truncation=2,
    )
    direct.run()
    replayed = trainer(
        shared,
        tmp_path / f"replay-{config.mode}-{observe_without_graph}",
        model=d3f_predictor(shared, config, gate_fill=0.2),
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


def test_resume_after_interruption_reproduces_the_continuous_run_bit_for_bit(shared, tmp_path):
    options = dict(truncation=2, epochs=2, checkpoint_updates=2, block_rows=8)

    def make(name):
        return trainer(
            shared,
            tmp_path / name,
            model=d3f_predictor(shared, SHUFFLED, gate_fill=0.2),
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
    assert "three_factor_state" in saved["run"]
    assert saved["run"]["three_factor_state"]["markets"]["US"]["updates"] > 0
    assert len(saved["run"]["three_factor_state"]["flows"]["flow_ids"]) > 0
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


def test_eval_frozen_matches_three_factor_in_train_and_is_neutral_in_validation(shared, tmp_path):
    control = trainer(shared, tmp_path / "control", model=d3f_predictor(shared, None), truncation=2)
    control.run()
    frozen = trainer(
        shared,
        tmp_path / "frozen",
        model=d3f_predictor(shared, EVAL_FROZEN_CFG, gate_fill=0.25),
        truncation=2,
    )
    frozen.run()
    online = trainer(
        shared,
        tmp_path / "online",
        model=d3f_predictor(shared, D3F, gate_fill=0.25),
        truncation=2,
    )
    online.run()
    # En ajuste coinciden; en validación eval_frozen iguala al control y online difiere.
    assert entries(frozen.audit, "prediction", "train") == entries(
        online.audit, "prediction", "train"
    )
    assert entries(frozen.audit, "prediction", "validation") == entries(
        control.audit, "prediction", "validation"
    )
    assert entries(online.audit, "prediction", "validation") != entries(
        control.audit, "prediction", "validation"
    )


def test_coexistence_with_crrm_and_hologram_in_chronological_run_with_replay(shared, tmp_path):
    """D3F-LR convive con CRRM y el interferómetro holográfico en el mismo recorrido."""
    extra = dict(
        session_hologram=SessionHologramConfig(
            bands=4, phase_dim=8, mode="interferometric", position="after_mac"
        ),
        cross_sectional_rank_memory=CrossSectionalRankMemoryConfig(
            dim=8, time_constants=(5,), ridge=1.0, minimum_rows=3, mode="matured_rank", seed=0
        ),
    )

    def make(name, **kwargs):
        mdl = d3f_predictor(shared, D3F, gate_fill=0.15, extra=extra)
        with torch.no_grad():
            mdl.session_hologram.gate.fill_(0.1)
            mdl.cross_sectional_rank_memory.out.weight.fill_(0.05)
        mdl._seal_parameters()
        return trainer(shared, tmp_path / name, model=mdl, truncation=2, **kwargs)

    direct = make("direct")
    direct.run()
    replayed = make("replayed", accumulation_rows=8, observe_without_graph=True)
    replayed.run()
    assert replayed.audit == direct.audit
    for a, b in zip(records(direct), records(replayed), strict=True):
        for name in a:
            torch.testing.assert_close(b[name], a[name], rtol=1e-10, atol=1e-12)


def test_clip_groups_split_block_rejection_and_frozen_parent_guard(shared, tmp_path):
    model = d3f_predictor(shared, D3F)
    groups = clip_parameter_groups(model)
    assert "three_factor_consolidation.gate" in groups["mac"]
    assert "three_factor_consolidation.key_proj.weight" in groups["mac"]
    assert "three_factor_consolidation.val_proj.weight" in groups["mac"]

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
            predictor_changes=dict(three_factor_consolidation=asdict(D3F)),
        ),
        identity=dict(recipe=dict(predictor=dict(hidden_size=32, dtype="float64"))),
    )
    (tmp_path / "run.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="tres factores"):
        mars_titan_walk_forward._parent(tmp_path, view, 42)
    with pytest.raises(ValueError, match="tres factores"):
        mars_titan_walk_forward._frozen_parent(
            dict(predictor=dict(hidden_size=32, dtype="float64")),
            specification(),
            42,
            "cpu",
            tmp_path,
            report,
        )
