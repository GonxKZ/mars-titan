"""Recorrido cronológico con memoria rápida de desfase entre grupos (LL-FM)."""

import hashlib
import json
from dataclasses import asdict

import pytest
import torch

from mars_titan.models.titans.cross_sectional_rank_memory import CrossSectionalRankMemoryConfig
from mars_titan.models.titans.financial import FinancialConfig, FinancialPredictor
from mars_titan.models.titans.lead_lag_memory import LeadLagMemoryConfig
from mars_titan.models.titans.session_hologram import SessionHologramConfig
from mars_titan.training import mars_titan_walk_forward
from mars_titan.training.checkpoints import load_training_state
from mars_titan.training.financial_run import (
    ChronologicalRecipe,
    ChronologicalTrainer,
    _Pass,
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

LLFM = LeadLagMemoryConfig(
    groups=6,
    lag_scales=(1, 5),
    decay=0.08,
    learning_rate=0.25,
    damping=1.0,
    minimum_rows=3,
    mode="lead_lag",
    seed=0,
)
SHUFFLED = LeadLagMemoryConfig(
    groups=6,
    lag_scales=(1, 5),
    decay=0.08,
    learning_rate=0.25,
    damping=1.0,
    minimum_rows=3,
    mode="shuffled_targets",
    seed=0,
)
STATIC_PRIOR = LeadLagMemoryConfig(
    groups=6,
    lag_scales=(1, 5),
    decay=0.08,
    learning_rate=0.25,
    damping=1.0,
    minimum_rows=3,
    mode="static_prior",
    seed=0,
)
EVAL_FROZEN = LeadLagMemoryConfig(
    groups=6,
    lag_scales=(1, 5),
    decay=0.08,
    learning_rate=0.25,
    damping=1.0,
    minimum_rows=3,
    mode="eval_frozen",
    seed=0,
)


@pytest.fixture(scope="module")
def shared(tmp_path_factory):
    return streams(tmp_path_factory.mktemp("llfm-shared"))


def llfm_predictor(sources, config=LLFM, *, gate_fill=None, **extra):
    cfg = FinancialConfig(
        sources["train"].specification(),
        hidden_size=32,
        variant="mac_online",
        memory_residual_layer_norm=True,
        gate_bias=dict(alpha_half_life=256.0, eta=0.5, theta=0.05),
        lead_lag_memory=config,
        **extra,
    )
    model = FinancialPredictor(cfg, dtype=torch.float64)
    if gate_fill is not None and config is not None:
        with torch.no_grad():
            model.lead_lag_memory.gate.fill_(gate_fill)
        model._seal_parameters()
    return model


def test_zero_init_reproduces_control_and_step_one_reaches_gate(shared, tmp_path):
    control = trainer(
        shared,
        tmp_path / "control",
        model=llfm_predictor(shared, None),
        truncation=2,
        max_grad_norm=None,
    )
    control.run()
    variant = trainer(
        shared,
        tmp_path / "variant",
        model=llfm_predictor(shared, LLFM),
        truncation=2,
        max_grad_norm=None,
    )
    report = variant.run()
    assert report["status"] == "completed"
    assert variant.audit == control.audit
    left, right = records(variant), records(control)
    assert len(left) == len(right) > 2
    gate_grads = [step.pop("lead_lag_memory.gate") for step in left]
    for step in left:
        for key in list(step):
            if key.startswith("lead_lag_memory."):
                step.pop(key, None)
    for a, b in zip(left, right, strict=True):
        assert a.keys() == b.keys()
        for name in a:
            torch.testing.assert_close(a[name], b[name], rtol=0, atol=0)
    # Con gate=0 sobre out.weight != 0 y prior_transition != 0, en el primer segmento
    # (truncation=2) la sesión 1 madura antes de la 2 y gate recibe gradiente ya en el paso 1.
    assert gate_grads[0] is not None and torch.count_nonzero(gate_grads[0]) > 0


def test_nonzero_gate_updates_all_lead_lag_parameters_including_delta_dynamics(shared, tmp_path):
    model = llfm_predictor(shared, LLFM, gate_fill=0.08)
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
        "lead_lag_memory.norm.weight",
        "lead_lag_memory.group_proj.weight",
        "lead_lag_memory.receptor_proj.weight",
        "lead_lag_memory.prior_transition",
        "lead_lag_memory.decay_logits",
        "lead_lag_memory.lr_logits",
        "lead_lag_memory.out.weight",
        "lead_lag_memory.gate",
    ):
        assert not torch.equal(dict(model.named_parameters())[name], before[name]), name


def test_strict_causality_future_perturbation_and_no_change_before_first_maturity(tmp_path):
    cut = decision("2022-12-09")
    runs = {}
    for name, perturb, cfg, fill in (
        ("control", None, None, None),
        ("base", None, LLFM, 0.08),
        ("future", cut, LLFM, 0.08),
    ):
        src = streams(tmp_path / name, perturb_after=perturb)
        engine = trainer(
            src,
            tmp_path / name / "run",
            model=llfm_predictor(src, cfg, gate_fill=fill),
            truncation=2,
        )
        engine.run()
        runs[name] = engine

    control_preds = entries(runs["control"].audit, "prediction", "train")
    base_preds = entries(runs["base"].audit, "prediction", "train")
    future_preds = entries(runs["future"].audit, "prediction", "train")
    first_maturity = min(e[4] for e in entries(runs["base"].audit, "label", "train"))
    # Antes de la primera maduración la memoria está vacía y emite igual que el control.
    assert [e for e in base_preds if e[3] < first_maturity] == [
        e for e in control_preds if e[3] < first_maturity
    ]
    assert [e for e in base_preds if e[3] >= first_maturity] != [
        e for e in control_preds if e[3] >= first_maturity
    ]
    # En validación también se reinicia el estado: antes de madurar la primera etiqueta de
    # validación, la emisión coincide bit a bit con el control.
    val_control = entries(runs["control"].audit, "prediction", "validation")
    val_base = entries(runs["base"].audit, "prediction", "validation")
    first_val_maturity = min(e[4] for e in entries(runs["base"].audit, "label", "validation"))
    assert [e for e in val_base if e[3] < first_val_maturity] == [
        e for e in val_control if e[3] < first_val_maturity
    ]
    assert [e for e in val_base if e[3] >= first_val_maturity] != [
        e for e in val_control if e[3] >= first_val_maturity
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
@pytest.mark.parametrize("config", [LLFM, SHUFFLED, STATIC_PRIOR])
def test_replay_matches_direct_backward_and_supports_observe_without_graph(
    shared, tmp_path, config, observe_without_graph
):
    direct = trainer(
        shared,
        tmp_path / f"direct-{config.mode}-{observe_without_graph}",
        model=llfm_predictor(shared, config, gate_fill=0.06),
        truncation=2,
    )
    direct.run()
    replayed = trainer(
        shared,
        tmp_path / f"replay-{config.mode}-{observe_without_graph}",
        model=llfm_predictor(shared, config, gate_fill=0.06),
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
            model=llfm_predictor(shared, LLFM, gate_fill=0.06),
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
    assert "lead_lag_memory" in saved["run"]
    assert saved["run"]["lead_lag_memory"]["markets"]["US"]["sessions"] > 1
    assert saved["run"]["lead_lag_memory"]["markets"]["US"]["updates"] > 0
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


def test_eval_frozen_and_static_prior_isolate_test_time_adaptation(shared, tmp_path):
    control = trainer(
        shared, tmp_path / "control", model=llfm_predictor(shared, None), truncation=2
    )
    control.run()
    frozen = trainer(
        shared,
        tmp_path / "frozen",
        model=llfm_predictor(shared, EVAL_FROZEN, gate_fill=0.08),
        truncation=2,
    )
    frozen.run()
    static = trainer(
        shared,
        tmp_path / "static",
        model=llfm_predictor(shared, STATIC_PRIOR, gate_fill=0.08),
        truncation=2,
    )
    static.run()
    online = trainer(
        shared,
        tmp_path / "online",
        model=llfm_predictor(shared, LLFM, gate_fill=0.08),
        truncation=2,
    )
    online.run()
    assert entries(frozen.audit, "prediction", "train") == entries(
        online.audit, "prediction", "train"
    )
    assert entries(frozen.audit, "prediction", "validation") == entries(
        control.audit, "prediction", "validation"
    )
    assert entries(static.audit, "prediction", "validation") != entries(
        control.audit, "prediction", "validation"
    )
    assert entries(online.audit, "prediction", "validation") != entries(
        static.audit, "prediction", "validation"
    )


def test_coexistence_with_crrm_and_hologram_in_trainer_and_checkpoint_resume(shared, tmp_path):
    crrm = CrossSectionalRankMemoryConfig(
        dim=8, time_constants=(5,), ridge=1.0, minimum_rows=3, mode="matured_rank", seed=0
    )
    holo = SessionHologramConfig(bands=2, phase_dim=4, mode="interferometric", position="after_mac")

    def build_combo():
        m = llfm_predictor(
            shared, LLFM, gate_fill=0.05, cross_sectional_rank_memory=crrm, session_hologram=holo
        )
        with torch.no_grad():
            m.cross_sectional_rank_memory.out.weight.fill_(0.05)
            m.session_hologram.gate.fill_(0.05)
        m._seal_parameters()
        return m

    continuous = trainer(
        shared,
        tmp_path / "combo-cont",
        model=build_combo(),
        truncation=2,
        epochs=1,
        checkpoint_updates=2,
        block_rows=8,
    )
    expected = continuous.run()
    first = trainer(
        shared,
        tmp_path / "combo-int",
        model=build_combo(),
        truncation=2,
        epochs=1,
        checkpoint_updates=2,
        block_rows=8,
    )
    stop = StopAtStep(2)
    stop.trainer = first
    assert first.run(stop=stop)["status"] == "paused"
    second = trainer(
        shared,
        tmp_path / "combo-int",
        model=build_combo(),
        truncation=2,
        epochs=1,
        checkpoint_updates=2,
        block_rows=8,
    )
    resumed = second.run(resume=True)
    assert resumed["status"] == "completed"
    assert first.audit + second.audit == continuous.audit
    assert resumed["history"] == expected["history"]


def test_close_clears_lead_lag_memory_state_and_replay_buffers(shared, tmp_path):
    engine = trainer(shared, tmp_path / "close-reset", model=llfm_predictor(shared, LLFM))
    run = _Pass()
    initial = engine.predictor.initial_lead_lag_memory()
    run.lead_lag_memory = initial
    run.lead_lag_memory_start = initial
    run.lead_lag_memory_labels.append((0, (("US/A", 1, 0.01),)))
    engine._close(run)
    assert run.lead_lag_memory is None
    assert run.lead_lag_memory_start is None
    assert run.lead_lag_memory_labels == []


def test_clip_groups_split_block_rejection_and_frozen_parent_guard(shared, tmp_path):
    model = llfm_predictor(shared, LLFM)
    groups = clip_parameter_groups(model)
    assert "lead_lag_memory.gate" in groups["mac"]
    assert "lead_lag_memory.group_proj.weight" in groups["mac"]
    assert "lead_lag_memory.prior_transition" in groups["mac"]
    # Partir un instante en bloques menores que la sesión se rechaza.
    split = trainer(shared, tmp_path / "split", model=model, block_rows=2)
    with pytest.raises(ValueError, match="sesión completa"):
        split.run()

    # mars_titan_walk_forward rechaza un padre con LL-FM antes de construir el consumidor congelado.
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
            predictor_changes=dict(lead_lag_memory=asdict(LLFM)),
        ),
        identity=dict(recipe=dict(predictor=dict(hidden_size=32, dtype="float64"))),
    )
    (tmp_path / "run.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="desfase"):
        mars_titan_walk_forward._parent(tmp_path, view, 42)
    with pytest.raises(ValueError, match="desfase"):
        mars_titan_walk_forward._frozen_parent(
            dict(predictor=dict(hidden_size=32, dtype="float64")),
            specification(),
            42,
            "cpu",
            tmp_path,
            report,
        )
