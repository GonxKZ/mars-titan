"""Recorrido cronológico con las cabezas auxiliares de F5 (#749), sin pasos de optimizador."""

import bisect
import math
from collections import Counter, defaultdict

import pytest
import torch

from mars_titan.models.titans.auxiliary_horizons import AuxiliaryHorizons
from mars_titan.models.titans.financial import FinancialConfig, FinancialPredictor
from mars_titan.training import financial_run
from mars_titan.training.checkpoints import load_training_state
from mars_titan.training.financial_run import ChronologicalRecipe, ChronologicalTrainer
from tests.training.chronological_fixture import decision
from tests.training.test_financial_run import (
    SELECTION,
    RecordingOptimizer,
    StopAtStep,
    corpus,
    entries,
    named_records,
)

HORIZONS = AuxiliaryHorizons(horizons=(2, 5), weight=0.5, maturity_slack=2)
AUXILIARY = {"auxiliary_heads.weight", "auxiliary_heads.bias"}
# Parámetros fuera del token fusionado: el término auxiliar no puede cambiar su gradiente.
OUTSIDE_TOKEN = ("head.weight", "head.bias", "mac.persistent")


@pytest.fixture(autouse=True)
def explicit_fastpath():
    previous = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    yield
    torch.backends.mha.set_fastpath_enabled(previous)


@pytest.fixture(scope="module")
def shared(tmp_path_factory):
    return corpus(tmp_path_factory.mktemp("auxiliary-horizons"))


def predictor(streams, horizons=HORIZONS):
    config = FinancialConfig(
        streams["train"].specification(),
        variant="mac_online",
        hidden_size=32,
        memory_residual_layer_norm=True,
        gate_bias=dict(alpha_half_life=256.0, eta=0.5, theta=0.05),
        auxiliary_horizons=horizons,
    )
    return FinancialPredictor(config, dtype=torch.float64)


def trainer(streams, output, *, horizons=HORIZONS, **options):
    values = dict(truncation=3, epochs=1, block_rows=2, max_grad_norm=None, selection=SELECTION)
    return ChronologicalTrainer(
        predictor(streams, horizons),
        ChronologicalRecipe(**values | options),
        train=streams["train"],
        validation=streams["validation"],
        output=output,
        optimizer_factory=RecordingOptimizer,
        audit=True,
    )


def same(left, right):
    return left is right if left is None or right is None else torch.equal(left, right)


def train_counters(report):
    return report["history"][1]["train"]


def test_weight_zero_control_keeps_the_reference_gradients(shared, tmp_path):
    _, streams = shared
    reference = trainer(streams, tmp_path / "reference", horizons=None)
    control = trainer(streams, tmp_path / "control", horizons=AuxiliaryHorizons(weight=0.0))
    reports = reference.run(), control.run()
    counters = train_counters(reports[1])
    assert "auxiliary_labels" not in train_counters(reports[0])
    assert counters["auxiliary_labels"] > 0
    assert counters["auxiliary_labels"] == (
        counters["auxiliary_labels_in_loss"] + counters["auxiliary_labels_discarded"]
    )
    assert counters["auxiliary_loss_sum"] == 0
    assert control.optimizer.calls == reference.optimizer.calls > 2
    for left, right in zip(named_records(control), named_records(reference), strict=True):
        assert left.pop("auxiliary_heads.weight") is None
        assert left.pop("auxiliary_heads.bias") is None
        assert left.keys() == right.keys()
        assert all(same(value, right[name]) for name, value in left.items())
    assert reports[0]["history"][1]["validation"] == reports[1]["history"][1]["validation"]


def test_matured_horizons_train_heads_and_token_but_not_the_rest(shared, tmp_path):
    _, streams = shared
    reference = trainer(streams, tmp_path / "reference", horizons=None)
    engine = trainer(streams, tmp_path / "auxiliary")
    reference.run()
    report = engine.run()
    counters = train_counters(report)
    assert counters["auxiliary_labels_in_loss"] > 0 and counters["auxiliary_loss_sum"] > 0
    changed = 0
    for left, right in zip(named_records(engine), named_records(reference), strict=True):
        bias, weight = (left.pop(name) for name in sorted(AUXILIARY))
        for name in OUTSIDE_TOKEN:
            assert same(left[name], right[name]), name
        if weight is not None:
            # Los signos de pocas filas pueden anular el sesgo, no los pesos.
            changed += 1
            assert bias is not None and torch.count_nonzero(weight)
            assert not torch.equal(left["fusion.0.weight"], right["fusion.0.weight"])
    assert changed > 1


def reference_targets(dataset, engine, horizons):
    """Etiquetas auxiliares derivadas de forma independiente de las etiquetas del corpus."""
    values = {}
    for batch in dataset.batches(partition="train", batch_size=64, epoch=0, seed=0):
        for index, sample in enumerate(batch["sample_ids"]):
            flow, at = sample.rsplit("/", 1)
            values[flow, int(at)] = float(batch["target"][index])
    emitted = defaultdict(set)
    for _, _, flow, at, _ in entries(engine.audit, "prediction", "train"):
        emitted[flow].add(at)
    instants = sorted(set().union(*emitted.values()))
    maturity = {(e[2], e[3]): e[4] for e in entries(engine.audit, "label", "train")}
    result = []
    for flow, decisions in emitted.items():
        for start in range(len(instants)):
            for head, horizon in enumerate(horizons.horizons):
                window = instants[start : start + horizon]
                keys = [(flow, at) for at in window]
                if len(window) < horizon or any(key not in maturity for key in keys):
                    continue
                assert set(window) <= decisions
                # Ordinal del último instante con entradas anterior a la última maduración.
                now = bisect.bisect_left(instants, max(maturity[key] for key in keys)) - 1
                if start + horizon - 1 + horizons.maturity_slack >= now:
                    result.append((head, math.fsum(values[key] for key in keys)))
    return sorted(result)


def test_targets_are_sums_of_matured_labels_derived_independently(shared, tmp_path, monkeypatch):
    dataset, streams = shared
    calls, original = [], financial_run.auxiliary_errors

    def spy(outputs, targets, heads, horizons):
        calls.append(list(zip(heads.tolist(), targets.tolist(), strict=True)))
        return original(outputs, targets, heads, horizons)

    monkeypatch.setattr(financial_run, "auxiliary_errors", spy)
    engine = trainer(streams, tmp_path / "run")
    counters = train_counters(engine.run())
    expected = reference_targets(dataset, engine, HORIZONS)
    seen = sorted(pair for call in calls for pair in call)
    assert len(calls) > 2 and {head for head, _ in seen} == {0, 1}
    assert len(seen) == counters["auxiliary_labels_in_loss"]
    assert len(expected) == counters["auxiliary_labels"]
    assert len(expected) == len(seen) + counters["auxiliary_labels_discarded"]
    # Cada objetivo usado es la suma exacta de etiquetas maduras de una ventana de referencia.
    remaining = Counter(expected)
    remaining.subtract(Counter(seen))
    assert min(remaining.values()) >= 0


def test_replay_by_flow_blocks_keeps_the_auxiliary_gradient(shared, tmp_path):
    _, streams = shared
    records = []
    for name, rows in (("whole", None), ("blocks", 1)):
        engine = trainer(streams, tmp_path / name, accumulation_rows=rows)
        engine.run()
        records.append(named_records(engine))
    assert len(records[0]) == len(records[1]) > 2
    for left, right in zip(*records, strict=True):
        for name, value in left.items():
            if value is None or right[name] is None:
                assert value is right[name], name
            else:
                torch.testing.assert_close(value, right[name], rtol=1e-10, atol=1e-14)


def test_future_suffix_does_not_change_past_auxiliary_gradients(tmp_path):
    cut = decision("2022-12-09")
    engines = []
    for name, perturb in (("base", None), ("future", cut)):
        streams = corpus(tmp_path / name, perturb_after=perturb)[1]
        engine = trainer(streams, tmp_path / name / "run")
        engine.run()
        engines.append(engine)
    base, future = engines
    steps = [u[2] for u in entries(base.audit, "update") if u[1] <= cut]
    assert steps and steps == [u[2] for u in entries(future.audit, "update") if u[1] <= cut]
    past = named_records(base)[: len(steps)]
    assert any(record["auxiliary_heads.weight"] is not None for record in past)
    for left, right in zip(past, named_records(future), strict=False):
        assert all(same(value, right[name]) for name, value in left.items())
    later = zip(
        named_records(base)[len(steps) :], named_records(future)[len(steps) :], strict=False
    )
    assert any(
        left["auxiliary_heads.weight"] is not None
        and not same(left["auxiliary_heads.weight"], right["auxiliary_heads.weight"])
        for left, right in later
    )


def test_resume_restores_the_horizon_ledger(shared, tmp_path):
    _, streams = shared
    options = dict(epochs=2, selection=dict(SELECTION, stopping="fixed_budget"))
    options["checkpoint_updates"] = 2
    continuous = trainer(streams, tmp_path / "continuous", **options)
    expected = continuous.run()
    stop = StopAtStep(15)
    first = trainer(streams, tmp_path / "interrupted", **options)
    stop.trainer = first
    assert first.run(stop=stop)["status"] == "paused"
    state = load_training_state(
        tmp_path / "interrupted/checkpoints", expected_identity=first.identity
    )
    ledger = state["run"]["horizons"]
    stored = int(ledger["stored"].sum())
    assert ledger["flows"] and 0 < stored < len(ledger["flows"])
    assert ledger["inputs"]["prices"].shape[0] == stored
    second = trainer(streams, tmp_path / "interrupted", **options)
    resumed = second.run(resume=True)
    assert resumed["status"] == "completed"
    assert first.audit + second.audit == continuous.audit
    joined = named_records(first) + named_records(second)
    assert len(joined) == len(named_records(continuous))
    for left, right in zip(joined, named_records(continuous), strict=True):
        assert all(same(value, right[name]) for name, value in left.items())
    assert resumed["history"] == expected["history"]


def test_posttraining_is_refused_until_it_is_declared(shared, tmp_path):
    _, streams = shared
    with pytest.raises(ValueError, match="postentrenamiento"):
        ChronologicalTrainer(
            predictor(streams),
            ChronologicalRecipe(truncation=3, epochs=1, block_rows=2, selection=SELECTION),
            train=streams["train"],
            validation=streams["validation"],
            output=tmp_path / "run",
            optimizer_factory=RecordingOptimizer,
            posttraining=dict(kind="adapters"),
        )
