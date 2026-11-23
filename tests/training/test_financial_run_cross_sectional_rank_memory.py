"""Recorrido cronológico con memoria transversal de rango maduro (CRRM)."""

import hashlib
import json
from dataclasses import asdict

import pytest
import torch

from mars_titan.models.titans.cross_sectional_rank_memory import CrossSectionalRankMemoryConfig
from mars_titan.models.titans.financial import FinancialConfig, FinancialPredictor
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

CRRM = CrossSectionalRankMemoryConfig(
    dim=8, time_constants=(5, 21), ridge=1.0, minimum_rows=3, mode="matured_rank", seed=0
)
SHUFFLED = CrossSectionalRankMemoryConfig(
    dim=8, time_constants=(5, 21), ridge=1.0, minimum_rows=3, mode="shuffled_targets", seed=0
)
EVAL_FROZEN = CrossSectionalRankMemoryConfig(
    dim=8, time_constants=(5, 21), ridge=1.0, minimum_rows=3, mode="eval_frozen", seed=0
)


@pytest.fixture(scope="module")
def shared(tmp_path_factory):
    return streams(tmp_path_factory.mktemp("crmm-shared"))


def crmm_predictor(sources, config=CRRM, *, out_fill=None):
    cfg = FinancialConfig(
        sources["train"].specification(),
        hidden_size=32,
        variant="mac_online",
        memory_residual_layer_norm=True,
        gate_bias=dict(alpha_half_life=256.0, eta=0.5, theta=0.05),
        cross_sectional_rank_memory=config,
    )
    model = FinancialPredictor(cfg, dtype=torch.float64)
    if out_fill is not None and config is not None:
        with torch.no_grad():
            model.cross_sectional_rank_memory.out.weight.fill_(out_fill)
        model._seal_parameters()
    return model


def test_zero_init_reproduces_control_and_step_one_reaches_out_weight(shared, tmp_path):
    # Sin recorte global, los gradientes crudos del tronco coinciden bit a bit con el control
    # mientras out.weight (inicializada a cero) recibe gradiente ya en el primer paso.
    control = trainer(
        shared,
        tmp_path / "control",
        model=crmm_predictor(shared, None),
        truncation=2,
        max_grad_norm=None,
    )
    control.run()
    variant = trainer(
        shared,
        tmp_path / "variant",
        model=crmm_predictor(shared, CRRM),
        truncation=2,
        max_grad_norm=None,
    )
    report = variant.run()
    assert report["status"] == "completed"
    assert variant.audit == control.audit
    left, right = records(variant), records(control)
    assert len(left) == len(right) > 2
    out_grads = [step.pop("cross_sectional_rank_memory.out.weight") for step in left]
    for step in left:
        for extra in (
            "cross_sectional_rank_memory.norm.weight",
            "cross_sectional_rank_memory.norm.bias",
            "cross_sectional_rank_memory.key_proj.weight",
            "cross_sectional_rank_memory.ridge_logits",
        ):
            step.pop(extra, None)
    for a, b in zip(left, right, strict=True):
        assert a.keys() == b.keys()
        for name in a:
            torch.testing.assert_close(a[name], b[name], rtol=0, atol=0)
    # Con out.weight=0, en el primer segmento (truncation=2) la sesión 1 madura antes de la 2
    # y out.weight recibe gradiente no nulo ya en la primera actualización.
    assert any(g is not None and torch.count_nonzero(g) > 0 for g in out_grads)
    assert out_grads[0] is not None and torch.count_nonzero(out_grads[0]) > 0


def test_nonzero_out_weight_flows_meta_gradients_through_cholesky_on_real_step(shared, tmp_path):
    model = crmm_predictor(shared, CRRM, out_fill=0.05)
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
        "cross_sectional_rank_memory.out.weight",
        "cross_sectional_rank_memory.key_proj.weight",
        "cross_sectional_rank_memory.ridge_logits",
    ):
        assert not torch.equal(dict(model.named_parameters())[name], before[name]), name


def test_strict_causality_future_perturbation_and_no_change_before_first_maturity(tmp_path):
    cut = decision("2022-12-09")
    runs = {}
    for name, perturb, cfg, fill in (
        ("control", None, None, None),
        ("base", None, CRRM, 0.08),
        ("future", cut, CRRM, 0.08),
    ):
        src = streams(tmp_path / name, perturb_after=perturb)
        engine = trainer(
            src,
            tmp_path / name / "run",
            model=crmm_predictor(src, cfg, out_fill=fill),
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
@pytest.mark.parametrize("config", [CRRM, SHUFFLED])
def test_replay_matches_direct_backward_and_supports_observe_without_graph(
    shared, tmp_path, config, observe_without_graph
):
    direct = trainer(
        shared,
        tmp_path / f"direct-{config.mode}-{observe_without_graph}",
        model=crmm_predictor(shared, config, out_fill=0.06),
        truncation=2,
    )
    direct.run()
    replayed = trainer(
        shared,
        tmp_path / f"replay-{config.mode}-{observe_without_graph}",
        model=crmm_predictor(shared, config, out_fill=0.06),
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
            model=crmm_predictor(shared, SHUFFLED, out_fill=0.06),
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
    assert "rank_memory" in saved["run"]
    assert saved["run"]["rank_memory"]["markets"]["US"]["updates"] > 0
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


def test_eval_frozen_matches_matured_rank_in_train_but_stays_neutral_in_validation(
    shared, tmp_path
):
    control = trainer(
        shared, tmp_path / "control", model=crmm_predictor(shared, None), truncation=2
    )
    control.run()
    frozen = trainer(
        shared,
        tmp_path / "frozen",
        model=crmm_predictor(shared, EVAL_FROZEN, out_fill=0.08),
        truncation=2,
    )
    frozen.run()
    online = trainer(
        shared,
        tmp_path / "online",
        model=crmm_predictor(shared, CRRM, out_fill=0.08),
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


def test_clip_groups_split_block_rejection_and_frozen_parent_guard(shared, tmp_path):
    model = crmm_predictor(shared, CRRM)
    groups = clip_parameter_groups(model)
    assert "cross_sectional_rank_memory.out.weight" in groups["mac"]
    assert "cross_sectional_rank_memory.key_proj.weight" in groups["mac"]
    # Partir un instante en bloques menores que la sesión se rechaza.
    split = trainer(shared, tmp_path / "split", model=model, block_rows=2)
    with pytest.raises(ValueError, match="sesión completa"):
        split.run()

    # mars_titan_walk_forward rechaza un padre con CRRM antes de construir el consumidor congelado.
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
            predictor_changes=dict(cross_sectional_rank_memory=asdict(CRRM)),
        ),
        identity=dict(recipe=dict(predictor=dict(hidden_size=32, dtype="float64"))),
    )
    (tmp_path / "run.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="memoria transversal de rango"):
        mars_titan_walk_forward._parent(tmp_path, view, 42)
    with pytest.raises(ValueError, match="memoria transversal de rango"):
        mars_titan_walk_forward._frozen_parent(
            dict(predictor=dict(hidden_size=32, dtype="float64")),
            specification(),
            42,
            "cpu",
            tmp_path,
            report,
        )
