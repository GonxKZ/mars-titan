"""Paridad y límites de la instrumentación, con un recorrido CPU sin aprendizaje."""

from collections import Counter
from types import SimpleNamespace

import pytest
import torch


def api():
    from benchmarks import refinement_latency

    return refinement_latency


class Predictor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(2.0), requires_grad=False)
        self.eval()

    def get_extra_state(self):
        return {"fixture": "scalar_without_learning"}


class Engine:
    """Resolver etiquetas emitidas por una multiplicación, sin ajustar parámetros."""

    def __init__(self, *, fail=False, mutate=False):
        self.predictor, self.readout, self.policy = Predictor(), None, None
        self.device, self.quantiles = torch.device("cpu"), False
        self.fail, self.mutate = fail, mutate

    def _observe(self, run, phase, event, *, train):
        for flow, at, value in event.inputs:
            run.pending[flow, at] = float(self.predictor.weight * value)
        run.counters["observations"] += len(event.inputs)
        if self.fail and event.at == 11:
            raise ValueError("fallo de la fixture")

    def _labels(self, run, phase, event, *, train, rows):
        for flow, at, target in event.labels:
            issued = run.pending.pop((flow, at))
            rows.append((flow, at, issued, target, None))
            run.staged.append((flow, target))
        run.counters["labels"] += len(event.labels)

    def _admit(self, run, phase, at):
        run.counters["admitted"] += len(run.staged)
        run.staged.clear()

    def evaluate(self, source, *, stop=None, rows=None):
        run = SimpleNamespace(pending={}, staged=[], counters=Counter())
        for event in source.events:
            if stop is not None and stop.requested:
                raise RuntimeError("parada de la fixture")
            self._labels(run, source.phase, event, train=False, rows=rows)
            if event.inputs:
                self._observe(run, source.phase, event, train=False)
            self._admit(run, source.phase, event.at)
        if self.mutate:
            with torch.no_grad():
                self.predictor.weight.add_(1)
        return dict(
            labels=run.counters["labels"], updates=0, observations=run.counters["observations"]
        )


def source():
    values = [
        (1, [("US/A", 1, 0.5)], []),
        (10, [("US/A", 10, 1.5), ("US/B", 10, -1.0)], []),
        (11, [("US/B", 11, 2.0)], [("US/A", 10, 3.1)]),
        (12, [], [("US/B", 10, -2.1), ("US/B", 11, 4.1)]),
    ]
    return SimpleNamespace(
        phase=SimpleNamespace(decision_start=10),
        events=[
            SimpleNamespace(at=at, inputs=inputs, labels=labels) for at, inputs, labels in values
        ],
    )


def test_instrumentation_preserves_predictions_parameters_rng_and_methods():
    engine = Engine()
    before_rng = torch.random.get_rng_state().clone()
    original = engine._observe
    plain, plain_rows = api().measure_pass(engine, source(), instrument=False)
    measured, measured_rows = api().measure_pass(engine, source(), instrument=True)
    assert api().prediction_digest(plain_rows) == api().prediction_digest(measured_rows)
    assert measured["metrics"] == plain["metrics"]
    assert measured["shared_parameters_unchanged"] is True
    assert engine._observe == original and "_observe" not in engine.__dict__
    assert torch.equal(before_rng, torch.random.get_rng_state())
    assert {record["stage"] for record in measured["events"]} == {"observe", "labels", "admit"}
    observed = [row for row in measured["events"] if row["stage"] == "observe"]
    assert [(row["at"], row["rows"], row["historical_warmup"]) for row in observed] == [
        (1, 1, True),
        (10, 2, False),
        (11, 1, False),
    ]
    assert all(0 <= row["exclusive_seconds"] <= row["seconds"] for row in measured["events"])
    summary = api().summarize(measured["events"], warmup_decisions=1)
    assert summary["steady_decisions"]["count"] == 1
    assert summary["steady_decisions"]["rows"] == 1
    assert summary["technical_warmup_decisions"] == 1
    assert summary["historical_warmup_decisions"] == 1


def test_failed_measurement_restores_instance_overrides():
    engine = Engine(fail=True)
    original = engine._observe
    engine._observe = original
    with pytest.raises(ValueError, match="fixture"):
        api().measure_pass(engine, source(), instrument=True)
    assert engine.__dict__["_observe"] is original
    assert "_labels" not in engine.__dict__ and "_admit" not in engine.__dict__


def test_event_limit_rejects_instead_of_dropping_timings():
    engine = Engine()
    with pytest.raises(ValueError, match="límite"):
        api().measure_pass(engine, source(), instrument=True, max_records=1)
    assert "_observe" not in engine.__dict__


def test_parameter_mutation_is_not_published_as_a_valid_measurement():
    with pytest.raises(ValueError, match="parámetros"):
        api().measure_pass(Engine(mutate=True), source(), instrument=True)


def test_reported_updates_are_rejected_even_when_parameters_match():
    class Updated(Engine):
        def evaluate(self, *args, **kwargs):
            return super().evaluate(*args, **kwargs) | {"updates": 1}

    with pytest.raises(ValueError, match="actualizaciones"):
        api().measure_pass(Updated(), source(), instrument=True)


@pytest.mark.parametrize("limit", [0, -1, True, 1.5, 1000000])
def test_invalid_event_limits_are_rejected(limit):
    with pytest.raises(ValueError):
        api().measure_pass(Engine(), source(), instrument=True, max_records=limit)


def test_empty_steady_state_is_reported_without_fabricated_quantiles():
    record, _ = api().measure_pass(Engine(), source(), instrument=True)
    summary = api().summarize(record["events"], warmup_decisions=5)
    assert summary["steady_decisions"]["count"] == 0
    assert summary["steady_decisions"]["p50_seconds"] is None


def test_trainable_parameters_and_implicit_instrument_flags_are_rejected():
    engine = Engine()
    engine.predictor.weight.requires_grad_(True)
    with pytest.raises(ValueError, match="congelados"):
        api().measure_pass(engine, source(), instrument=True)
    with pytest.raises(ValueError, match="booleano"):
        api().measure_pass(Engine(), source(), instrument="yes")


def test_other_cuda_devices_do_not_fall_back_to_cpu():
    engine = Engine()
    engine.device = torch.device("cuda:1")
    with pytest.raises(ValueError, match="explícitos"):
        api().measure_pass(engine, source(), instrument=True)


def test_bad_warmup_timing_is_not_hidden_by_its_exclusion():
    record, _ = api().measure_pass(Engine(), source(), instrument=True)
    next(row for row in record["events"] if row["stage"] == "observe")["seconds"] = float("nan")
    with pytest.raises(ValueError, match="finitas"):
        api().summarize(record["events"])


def test_prediction_digest_detects_values_and_negative_zero_bits():
    from mars_titan.training.titans_walk_forward import PredictionRows

    digests = []
    for value in (0.0, -0.0, 1.0):
        rows = PredictionRows(False)
        rows.append(("US/A", 10, value, 0.0, None))
        digests.append(api().prediction_digest(rows))
    assert len(set(digests)) == 3


def test_amortized_cost_is_not_reported_as_an_individual_request_latency():
    records = [
        dict(
            stage="observe",
            at=10,
            rows=4,
            historical_warmup=False,
            seconds=0.8,
            exclusive_seconds=0.8,
            depth=0,
        )
    ]
    result = api().summarize(records, warmup_decisions=0)
    assert result["steady_decisions"]["p50_seconds"] == 0.8
    assert result["amortized_seconds_per_asset"]["p50_seconds"] == 0.2
    assert "total_exclusive_seconds" not in result["amortized_seconds_per_asset"]


def test_random_work_is_rejected_instead_of_being_presented_as_a_replay():
    class RandomEngine(Engine):
        def evaluate(self, *args, **kwargs):
            result = super().evaluate(*args, **kwargs)
            torch.rand(1)
            return result

    with torch.random.fork_rng(devices=[]), pytest.raises(ValueError, match="RNG"):
        api().measure_pass(RandomEngine(), source(), instrument=True)


@pytest.mark.cuda
def test_cuda_instrumentation_preserves_bits_and_records_allocator_peaks():
    if not torch.cuda.is_available():
        pytest.skip("CUDA no está disponible")
    engine = Engine()
    engine.device = torch.device("cuda:0")
    engine.predictor.to(engine.device)
    baseline, rows = api().measure_pass(engine, source(), instrument=False)
    measured, measured_rows = api().measure_pass(engine, source(), instrument=True)
    assert api().prediction_digest(rows) == api().prediction_digest(measured_rows)
    assert measured["metrics"] == baseline["metrics"]
    assert measured["cuda_peak_allocated_bytes"] > 0
    assert measured["cuda_peak_reserved_bytes"] >= measured["cuda_peak_allocated_bytes"]
    assert measured["cuda_timing"] == "synchronized_method_boundaries"


def test_nested_measurements_are_not_counted_twice():
    class Nested(Engine):
        def _labels(self, run, phase, event, *, train, rows):
            super()._labels(run, phase, event, train=train, rows=rows)
            self._admit(run, phase, event.at)

    record, _ = api().measure_pass(Nested(), source(), instrument=True)
    parents = [row for row in record["events"] if row["stage"] == "labels"]
    assert all(row["seconds"] > row["exclusive_seconds"] for row in parents)
    assert all(
        row["depth"] == 1 for row in record["events"] if row["stage"] == "admit" and row["depth"]
    )


def test_instrumentation_preserves_the_real_cpu_reader_and_mature_bank(tmp_path):
    import os

    from mars_titan.memory.native_backend import load_native
    from mars_titan.training.titans_walk_forward import unfused_attention
    from tests.training.test_financial_run import corpus
    from tests.training.test_refinement_inference import build

    _, streams = corpus(tmp_path / "data")
    native = load_native(os.environ["MARS_TITAN_EPISODIC_NATIVE"])
    with unfused_attention():
        engine = build(streams, native, mode="fixed", route="k4cc")
        original, rows = api().measure_pass(engine, streams["validation"], instrument=False)
        timed, timed_rows = api().measure_pass(engine, streams["validation"], instrument=True)
    assert timed["metrics"] == original["metrics"]
    assert api().prediction_digest(rows) == api().prediction_digest(timed_rows)
    assert (
        sum(row["rows"] for row in timed["events"] if row["stage"] == "observe")
        == timed["metrics"]["observations"]
    )
    assert (
        sum(row["rows"] for row in timed["events"] if row["stage"] == "admit")
        == timed["metrics"]["admitted"]
    )


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_real_cuda_reader_keeps_predictions_and_states_when_timed(tmp_path, dtype):
    import os

    from mars_titan.memory.native_backend import load_native
    from mars_titan.models.titans.episodic_readout import EpisodicReadout
    from mars_titan.models.titans.financial import FinancialPredictor
    from mars_titan.training.mars_titan_run import retention_config
    from mars_titan.training.refinement_inference import RefinementInference
    from mars_titan.training.titans_walk_forward import unfused_attention
    from tests.training.test_financial_run import corpus
    from tests.training.test_mars_titan_run import parent, reader, recipe

    if not torch.cuda.is_available():
        pytest.skip("CUDA no está disponible")
    _, streams = corpus(tmp_path / "data")
    native = load_native(os.environ["MARS_TITAN_EPISODIC_NATIVE"])
    with unfused_attention():
        predictor = FinancialPredictor(parent(streams).config, dtype=dtype, device="cuda:0")
        template, codec = reader(streams, "m1")
        readout = EpisodicReadout(template.config, dtype=dtype, device="cuda:0")
        plan = recipe()
        engine = RefinementInference(
            predictor.eval().requires_grad_(False),
            readout.eval().requires_grad_(False),
            plan,
            admission="m1",
            retention=retention_config(plan, "m1"),
            native=native,
            codec=codec,
            world="latency-fixture",
            fold="0",
            mode="fixed",
            route="k4cc",
        )
        reference, rows = api().measure_pass(engine, streams["validation"], instrument=False)
        actual, actual_rows = api().measure_pass(engine, streams["validation"], instrument=True)
    assert actual["metrics"] == reference["metrics"]
    assert api().prediction_digest(rows) == api().prediction_digest(actual_rows)
    assert actual["cuda_peak_allocated_bytes"] > 0
