"""Penalización C del factorial CM-v1 en el entrenador cronológico, sin pasos de optimizador.

B usa el control en modo disabled, con SDPA Math y la misma base que B+C. Las pruebas
recorren el bucle hasta el paso con el registrador de gradientes, que no modifica pesos.
Con `accumulation_rows`, el gradiente de la tarea más la media de los términos de C se
acumula por bloques de flujos y se compara en float64 con el del tramo completo.
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
from mars_titan.training.financial_run import ChronologicalRecipe, ChronologicalTrainer, _Pass
from mars_titan.training.graph_memory import saved_graph_bytes
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


def test_diagnostic_mode_is_rejected_and_the_penalty_admits_accumulation(shared, tmp_path):
    _, streams = shared
    # El diagnóstico sigue sin admitirse en el ajuste, con acumulación o sin ella.
    for rows in (None, 1):
        with pytest.raises(ValueError, match="diagnóstico"):
            engine(
                streams,
                tmp_path / f"d{rows}",
                model(streams, control("diagnostic")),
                accumulation_rows=rows,
            )
    full = engine(streams, tmp_path / "c", model(streams, control("penalty", 0.5)))
    blocks = engine(
        streams, tmp_path / "a", model(streams, control("penalty", 0.5)), accumulation_rows=1
    )
    b = engine(streams, tmp_path / "b", model(streams, control("disabled")), accumulation_rows=1)
    # Solo la combinación nueva declara cómo acumula C. Las demás identidades no cambian.
    assert "penalty_accumulation" in blocks.identity["local_control"]
    assert "penalty_accumulation" not in full.identity["local_control"]
    assert "penalty_accumulation" not in b.identity["local_control"]
    objective = full.identity["local_control"]["objective"]
    assert blocks.identity["local_control"]["objective"] == objective


@pytest.mark.parametrize("rows", [None, 2])
def test_penalty_without_measured_flows_is_exactly_b(shared, tmp_path, rows):
    _, streams = shared
    never = control("penalty", 0.5, frequency=2**31 - 1)
    b, report_b = run(streams, tmp_path / "b", control("disabled"), accumulation_rows=rows)
    c, report_c = run(streams, tmp_path / "c", never, accumulation_rows=rows)
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


def test_objective_adds_the_mean_of_the_group_penalties_of_each_segment(
    shared, tmp_path, monkeypatch
):
    """Objetivo = pérdida de la tarea + media de los términos de los grupos del tramo."""
    _, streams = shared
    trainer = engine(streams, tmp_path / "c", model(streams, control("penalty", 0.5)))
    inner, segments, objectives = trainer._backward, [], []
    backward = torch.Tensor.backward

    def spy(tensor, *args, **kwargs):
        objectives.append(float(tensor.detach()))
        return backward(tensor, *args, **kwargs)

    def recorded(run):
        penalties = [float(term.detach()) for term in run.penalties]
        loss = inner(run)
        segments.append((float(loss.detach()), penalties))
        return loss

    monkeypatch.setattr(torch.Tensor, "backward", spy)
    trainer._backward = recorded
    trainer.run()
    assert len(objectives) == len(segments) > 3
    assert any(len(penalties) > 1 for _, penalties in segments)
    for value, (loss, penalties) in zip(objectives, segments, strict=True):
        expected = loss + (sum(penalties) / len(penalties) if penalties else 0.0)
        assert value == pytest.approx(expected, rel=1e-12, abs=1e-15)


def test_frequency_follows_the_observation_counter_of_each_flow(shared, tmp_path):
    """Con frecuencia 2 se miden menos grupos que con 1, pero alguno."""
    _, streams = shared
    groups = {}
    for frequency in (1, 2):
        _, report = run(
            streams, tmp_path / f"f{frequency}", control("penalty", 0.5, frequency=frequency)
        )
        groups[frequency] = train_metrics(report)["control_groups"]
    assert 0 < groups[2] < groups[1]


def test_penalties_without_mature_labels_are_discarded_without_a_step(shared, tmp_path):
    """Un tramo sin etiquetas maduras no da paso, así que C no añade actualizaciones."""
    _, streams = shared
    b, report_b = run(streams, tmp_path / "b", control("disabled"), truncation=1)
    c, report_c = run(streams, tmp_path / "c", control("penalty", 0.5), truncation=1)
    metrics = train_metrics(report_c)
    assert metrics["control_groups_discarded"] > 0
    assert metrics["control_groups_kept"] == (
        metrics["control_groups_in_objective"] + metrics["control_groups_discarded"]
    )
    assert entries(b.audit, "update") == entries(c.audit, "update")
    assert metrics["updates"] == train_metrics(report_b)["updates"]


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


@pytest.mark.parametrize("rows", [None, 2])
def test_resume_with_the_penalty_reproduces_the_continuous_run(shared, tmp_path, rows):
    _, streams = shared
    options = dict(epochs=2, selection=dict(SELECTION, stopping="fixed_budget"))
    options.update(checkpoint_updates=2, accumulation_rows=rows)
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


def without_loss(metrics):
    return {key: value for key, value in metrics.items() if key != "mean_loss"}


# Con truncamiento 1 y un flujo medido por evento, un tramo mide A0000 en la decisión
# cuyo objetivo se rechaza por varianza nula: un flujo medido sin etiquetas en el tramo.
SEGMENTS = [dict(truncation=1, max_flows=1), dict(truncation=3, max_flows=3)]


@pytest.mark.parametrize("segment", SEGMENTS, ids=["measured_without_labels", "all_measured"])
@pytest.mark.parametrize("rows", [1, 2, 3])
def test_accumulated_penalty_gradient_matches_the_full_segment(
    shared, tmp_path, monkeypatch, segment, rows
):
    """Tarea más media de C por bloques de flujos, en float64, frente al tramo completo.

    Con tres flujos, 2 deja un bloque incompleto y 3 cubre el tramo con un solo bloque.
    """
    _, streams = shared
    measured = control("penalty", 0.5, max_flows=segment["max_flows"])
    truncation = segment["truncation"]
    full, expected = run(streams, tmp_path / "full", measured, truncation=truncation)
    blocks = engine(
        streams,
        tmp_path / "blocks",
        model(streams, measured),
        truncation=truncation,
        accumulation_rows=rows,
    )
    replay, prepare = blocks._replay, blocks.predictor.prepare
    sizes, inside, unlabelled = [], [], []

    def replay_spy(current):
        # Con acumulación el tramo solo guarda los flujos medidos de cada grupo.
        assert all(type(g) is tuple and all(type(f) is str for f in g) for g in current.penalties)
        labelled = {flow for flow, _, _ in current.used}
        unlabelled.append({flow for group in current.penalties for flow in group} - labelled)
        inside.append(True)
        try:
            return replay(current)
        finally:
            inside.pop()

    def prepare_spy(batch, state, **options):
        if inside:
            sizes.append(len(batch.flow_ids))
        return prepare(batch, state, **options)

    monkeypatch.setattr(blocks, "_replay", replay_spy)
    monkeypatch.setattr(blocks.predictor, "prepare", prepare_spy)
    report = blocks.run()
    assert blocks.audit == full.audit
    assert blocks.optimizer.calls == full.optimizer.calls == len(entries(full.audit, "update"))
    for left, right in zip(report["history"], expected["history"], strict=True):
        assert left["validation"] == right["validation"]
        if right["train"] is not None:
            # Grupos, flujos, términos emitidos y grupos del objetivo, bit a bit.
            assert without_loss(left["train"]) == without_loss(right["train"])
            assert left["train"]["mean_loss"] == pytest.approx(right["train"]["mean_loss"])
    metrics = train_metrics(report)
    assert metrics["control_groups_in_objective"] > 0
    # Solo cambian el orden de las sumas y la división por grupos dentro de cada bloque.
    for left, right in zip(named_records(blocks), named_records(full), strict=True):
        assert left.keys() == right.keys()
        for key, value in right.items():
            if value is None:
                assert left[key] is None, key
            else:
                torch.testing.assert_close(left[key], value, rtol=1e-10, atol=1e-13)
    assert sizes and max(sizes) <= rows
    if truncation == 1:
        assert any(unlabelled)


def test_emitted_penalty_keeps_no_graph_with_accumulation(shared, tmp_path):
    _, streams = shared
    trainer = engine(
        streams, tmp_path / "c", model(streams, control("penalty", 0.5)), accumulation_rows=1
    )
    inner, seen = trainer._penalty, []

    def spy(current, measured, *, keep, replay=False):
        seen.append((keep, replay, [term.requires_grad for term, _ in measured]))
        return inner(current, measured, keep=keep, replay=replay)

    trainer._penalty = spy
    trainer.run()
    assert any(replay for _, replay, _ in seen)
    assert all(not any(grads) for keep, replay, grads in seen if replay)
    assert all(keep == replay for keep, replay, _ in seen)


def peak_backward_graph(trainer, monkeypatch):
    """Mayor grafo guardado alcanzable desde cada backward de un recorrido de ajuste."""
    parameters, peaks = list(trainer.predictor.parameters()), []
    backward = torch.Tensor.backward

    def spy(tensor, *args, **kwargs):
        peaks.append(saved_graph_bytes([tensor], exclude=parameters))
        return backward(tensor, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "backward", spy)
        cursor = dict(epoch=0, phase="train", event=0, stage="start")
        trainer._train_pass(_Pass(), cursor, StopAtStep(10**9), None)
    return max(peaks)


def test_live_graph_with_the_penalty_shrinks_with_the_block(shared, tmp_path, monkeypatch):
    _, streams = shared
    measured = control("penalty", 0.5, max_flows=3)
    full = engine(streams, tmp_path / "full", model(streams, measured))
    blocks = engine(streams, tmp_path / "one", model(streams, measured), accumulation_rows=1)
    full_bytes = peak_backward_graph(full, monkeypatch)
    block_bytes = peak_backward_graph(blocks, monkeypatch)
    # Con tres flujos medidos en cada evento, un bloque de uno conserva cerca de un tercio.
    assert 0 < block_bytes < 0.5 * full_bytes
