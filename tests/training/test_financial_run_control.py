"""Penalización C del factorial CM-v1 en el entrenador cronológico, sin pasos de optimizador.

B usa el control en modo disabled, con SDPA Math y la misma base que B+C. Las pruebas
recorren el bucle hasta el paso con el registrador de gradientes, que no modifica pesos.
"""

import pytest
import torch

from mars_titan.models.titans.financial import (
    FinancialConfig,
    FinancialPredictor,
    copy_paired_parameters,
)
from mars_titan.models.titans.local_control import MACProjectionConfig
from mars_titan.training.checkpoints import load_training_state
from mars_titan.training.financial_run import ChronologicalRecipe, ChronologicalTrainer
from tests.training.test_financial_run import (
    SELECTION,
    RecordingOptimizer,
    StopAtStep,
    entries,
    named_records,
)
from tests.training.test_financial_run import explicit_fastpath as explicit_fastpath
from tests.training.test_financial_run import shared as shared

# Todos los flujos son elegibles en cada observación y se mide uno por evento.
EVERY_STEP = dict(rank=2, frequency=1, seed=91, grid_size=16, threshold=0.0, max_flows=1)


def control(mode, weight=0.0, **options):
    return MACProjectionConfig(**(EVERY_STEP | dict(mode=mode, weight=weight) | options))


def model(streams, local_control=None, seed=42):
    specification = streams["train"].specification()
    config = FinancialConfig(specification, variant="mac_online", hidden_size=32, seed=seed)
    return FinancialPredictor(config, local_control=local_control, dtype=torch.float64)


def engine(streams, output, predictor, **options):
    recipe = dict(truncation=3, epochs=1, block_rows=2, selection=SELECTION)
    recipe.update(options)
    return ChronologicalTrainer(
        predictor,
        ChronologicalRecipe(**recipe),
        train=streams["train"],
        validation=streams["validation"],
        output=output,
        optimizer_factory=RecordingOptimizer,
        audit=True,
    )


def run(streams, output, local_control, **options):
    trainer = engine(streams, output, model(streams, local_control), **options)
    report = trainer.run()
    return trainer, report


def assert_records(left, right, *, exact=True, keys=None):
    assert len(left) == len(right) > 0
    for first, second in zip(left, right, strict=True):
        assert first.keys() == second.keys()
        for key in keys or first:
            if first[key] is None:
                assert second[key] is None
            elif exact:
                torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)
            else:
                torch.testing.assert_close(first[key], second[key], rtol=1e-9, atol=1e-12)


def train_metrics(report):
    return report["history"][1]["train"]


def test_paired_controls_share_parameters_and_basis_and_differ_only_in_identity(shared):
    _, streams = shared
    b, c = model(streams, control("disabled")), model(streams, control("penalty", 0.5))
    plain = model(streams)
    for name, value in b.named_parameters():
        torch.testing.assert_close(value, dict(c.named_parameters())[name], rtol=0, atol=0)
        torch.testing.assert_close(value, dict(plain.named_parameters())[name], rtol=0, atol=0)
    torch.testing.assert_close(b.local_control.basis, c.local_control.basis, rtol=0, atol=0)
    receipt = copy_paired_parameters(c, b)
    assert receipt["local_control"]["basis_transferred"] is False


def test_identity_marks_the_control_and_keeps_the_plain_form(shared, tmp_path):
    _, streams = shared
    plain = engine(streams, tmp_path / "plain", model(streams))
    assert "local_control" not in plain.identity
    b = engine(streams, tmp_path / "b", model(streams, control("disabled")))
    c = engine(streams, tmp_path / "c", model(streams, control("penalty", 0.5)))
    assert b.identity["local_control"]["objective"] == "task_only"
    assert c.identity["local_control"]["objective"].startswith("task_mean_plus")
    assert (
        b.identity["local_control"]["basis_sha256"] == c.identity["local_control"]["basis_sha256"]
    )
    assert len({plain.run_id, b.run_id, c.run_id}) == 3


def test_diagnostic_mode_and_penalty_with_accumulation_are_rejected(shared, tmp_path):
    _, streams = shared
    with pytest.raises(ValueError, match="diagnóstico"):
        engine(streams, tmp_path / "d", model(streams, control("diagnostic")))
    with pytest.raises(ValueError, match="acumulación"):
        engine(
            streams, tmp_path / "a", model(streams, control("penalty", 0.5)), accumulation_rows=1
        )
    # B admite la acumulación porque no mide nada.
    engine(streams, tmp_path / "b", model(streams, control("disabled")), accumulation_rows=1)


def test_penalty_without_measured_flows_is_exactly_b(shared, tmp_path):
    _, streams = shared
    never = control("penalty", 0.5, frequency=2**31 - 1)
    b, report_b = run(streams, tmp_path / "b", control("disabled"))
    c, report_c = run(streams, tmp_path / "c", never)
    assert entries(b.audit, "prediction") == entries(c.audit, "prediction")
    assert entries(b.audit, "update") == entries(c.audit, "update")
    assert_records(named_records(b), named_records(c))
    assert report_b["history"] == report_c["history"]


def test_penalty_gradient_is_linear_in_its_weight_and_leaves_the_head_alone(shared, tmp_path):
    """g(2w) − g(w) = g(w) − g_B sin recorte. La cabeza no está en el camino de C."""
    _, streams = shared
    options = dict(max_grad_norm=None)
    b, report_b = run(streams, tmp_path / "b", control("disabled"), **options)
    one, report_one = run(streams, tmp_path / "one", control("penalty", 0.5), **options)
    two, _ = run(streams, tmp_path / "two", control("penalty", 1.0), **options)
    # El término no cambia el cálculo emitido ni el número de pasos.
    assert entries(b.audit, "prediction") == entries(one.audit, "prediction")
    assert entries(b.audit, "update") == entries(one.audit, "update")
    base, first, second = named_records(b), named_records(one), named_records(two)
    assert len(base) == len(first) == len(second) > 3
    changed = set()
    for g_b, g_1, g_2 in zip(base, first, second, strict=True):
        for name in g_b:
            if g_b[name] is None:
                assert g_1[name] is None and g_2[name] is None
                continue
            torch.testing.assert_close(
                g_2[name] - g_1[name], g_1[name] - g_b[name], rtol=1e-7, atol=1e-12
            )
            if name.startswith("head."):
                torch.testing.assert_close(g_1[name], g_b[name], rtol=0, atol=0)
            elif not torch.equal(g_1[name], g_b[name]):
                changed.add(name.split(".")[0])
    assert {"mac", "fusion"} <= changed
    metrics = train_metrics(report_one)
    assert metrics["control_groups_kept"] == (
        metrics.get("control_groups_in_objective", 0) + metrics.get("control_groups_discarded", 0)
    )
    assert metrics["control_flows"] == metrics["control_groups"] > metrics["control_groups_kept"]
    assert metrics["control_penalty_sum"] > 0
    assert "control_groups" not in train_metrics(report_b)


def test_logical_group_is_the_event_and_not_the_physical_block(shared, tmp_path):
    _, streams = shared
    reports, records = {}, {}
    for rows in (1, 3):
        trainer, reports[rows] = run(
            streams, tmp_path / f"rows-{rows}", control("penalty", 0.5), block_rows=rows
        )
        records[rows] = named_records(trainer)
    one, three = train_metrics(reports[1]), train_metrics(reports[3])
    for key in ("control_groups", "control_flows", "control_groups_kept", "updates"):
        assert one[key] == three[key]
    assert one["control_flows"] == one["control_groups"]
    assert one["control_penalty_sum"] == pytest.approx(three["control_penalty_sum"], rel=1e-12)
    assert_records(records[1], records[3], exact=False)


def test_validation_with_the_penalty_emits_what_b_emits(shared, tmp_path):
    _, streams = shared
    b = engine(streams, tmp_path / "b", model(streams, control("disabled")))
    c = engine(streams, tmp_path / "c", model(streams, control("penalty", 0.5)))
    left, right = b.evaluate(streams["validation"]), c.evaluate(streams["validation"])
    assert entries(b.audit, "prediction") == entries(c.audit, "prediction")
    assert right["control_groups"] > 0 and right["control_groups_kept"] == 0
    assert {k: v for k, v in right.items() if not k.startswith("control_")} == left


def test_resume_with_the_penalty_reproduces_the_continuous_run(shared, tmp_path):
    _, streams = shared
    options = dict(epochs=2, selection=dict(SELECTION, stopping="fixed_budget"))
    options["checkpoint_updates"] = 2
    continuous, expected = run(streams, tmp_path / "continuous", control("penalty", 0.5), **options)
    stop = StopAtStep(15)
    first = engine(streams, tmp_path / "cut", model(streams, control("penalty", 0.5)), **options)
    stop.trainer = first
    assert first.run(stop=stop)["status"] == "paused"
    state = load_training_state(tmp_path / "cut/checkpoints", expected_identity=first.identity)
    assert state["run"]["counters"]["control_groups_kept"] > 0
    second = engine(streams, tmp_path / "cut", model(streams, control("penalty", 0.5)), **options)
    resumed = second.run(resume=True)
    assert first.audit + second.audit == continuous.audit
    assert_records(named_records(first) + named_records(second), named_records(continuous))
    assert resumed["history"] == expected["history"]
