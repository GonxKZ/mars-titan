"""Supresiones E2 sobre un lector real con fixtures explícitos, sin ajuste de parámetros."""

import copy
import importlib
import math
from dataclasses import replace

import pytest
import torch
from test_episodic_readout import CODEC, CONTEXT, analytic, source
from test_financial_adapter import raw_batch, specification

from mars_titan.memory.spaced_utility import SpacedUtilityLedger
from mars_titan.models.quantile_head import QuantileHead
from mars_titan.models.titans.episodic_readout import (
    EpisodicReadout,
    EpisodicReadoutConfig,
    apply_episodic_readout,
)
from mars_titan.models.titans.episodic_snapshot import EpisodeSnapshot
from mars_titan.models.titans.financial import (
    DecisionBatch,
    FinancialConfig,
    FinancialPredictor,
)

AT = 1_609_459_200_000_000


def snapshot(count=3, dtype=torch.float64, **changes):
    return EpisodeSnapshot.create(
        **(source(count) | changes),
        cutoff=AT,
        codec_id=CODEC,
        context_id=CONTEXT,
        dtype=dtype,
    )


def api():
    try:
        return importlib.import_module("mars_titan.memory.spaced_utility_readout")
    except ModuleNotFoundError:
        pytest.fail("Falta el productor de supresiones del soporte E2")


def fixture(*, dtype=torch.float64, **options):
    spec = specification()
    core = FinancialPredictor(
        FinancialConfig(spec, variant="mac_online", hidden_size=32, layers=1), dtype=dtype
    )
    core.eval()
    batch = DecisionBatch.from_corpus(raw_batch(at=AT), spec, dtype=dtype)
    prepared = core.prepare(batch, core.initial_state(batch.flow_ids))
    model = EpisodicReadout(EpisodicReadoutConfig(CODEC, hidden_size=32, **options), dtype=dtype)
    model.eval()
    analytic(model)
    with torch.device("cpu"), torch.random.fork_rng(devices=[]):
        head = torch.nn.Linear(32, 1, dtype=dtype).eval()
        with torch.no_grad():
            head.weight.zero_()
            head.weight[0, 0] = 1
            head.bias.zero_()
    state = torch.zeros((2, 32), dtype=dtype)
    state[:, 0] = 1
    return core, replace(prepared, working_state=state), head, model


def apply(prepared, head, model, data, **options):
    return api().apply_spaced_utility_readout(
        prepared, head, model, data, context_id=CONTEXT, cutoff=AT, enabled=True, **options
    )


def test_two_neighbor_deletions_match_the_complete_refinement_equation():
    _, prepared, head, model = fixture(neighbors=2)
    prediction, diagnostics = apply(prepared, head, model, snapshot(2))
    weight = math.e / (math.e + 1)
    expected = 1 + 0.1 * math.tanh(weight + (1 - weight) * 2)
    assert prediction.point_predictions.tolist() == pytest.approx([expected] * 2)
    assert diagnostics.episode_ids.tolist() == [[10, 20]] * 2
    torch.testing.assert_close(
        diagnostics.points,
        torch.tensor(
            [[expected, 1 + 0.1 * math.tanh(2), 1 + 0.1 * math.tanh(1)]] * 2,
            dtype=torch.float64,
        ),
        rtol=0,
        atol=1e-15,
    )


def test_deleting_selected_support_does_not_refill_or_repeat_mac_and_selection(monkeypatch):
    core, prepared, head, model = fixture(neighbors=2)
    data = snapshot(3)
    calls = dict(select=0, values=0)
    old_sort = torch.argsort

    def sort(*args, **kwargs):
        calls["select"] += 1
        return old_sort(*args, **kwargs)

    def count_values(*_):
        calls["values"] += 1

    monkeypatch.setattr(torch, "argsort", sort)
    handle = model.value_projection.register_forward_hook(count_values)
    monkeypatch.setattr(core, "prepare", lambda *a, **k: pytest.fail("MAC repetido"))
    try:
        _, diagnostics = apply(prepared, head, model, data)
    finally:
        handle.remove()
    assert calls == dict(select=1, values=1)
    assert diagnostics.episode_ids.tolist() == [[10, 20]] * 2
    assert diagnostics.points[:, 1].tolist() == pytest.approx([1 + 0.1 * math.tanh(2)] * 2)
    assert prepared.next_state.mac.memory.steps.tolist() == [1, 1]


@pytest.mark.parametrize("count", [0, 1])
def test_empty_support_and_single_deletion_keep_the_absent_read_refinement(count):
    _, prepared, head, model = fixture(neighbors=1)
    with torch.no_grad():
        model.refinement.bias[0] = 0.5
        model.refinement.weight[0, -1] = 0.3
    prediction, diagnostics = apply(prepared, head, model, snapshot(count))
    absent = model(prepared.working_state, snapshot(0), context_id=CONTEXT, cutoff=AT)
    expected = head(absent.state).squeeze(-1)
    assert diagnostics.points.shape == (2, count + 1)
    if count:
        assert torch.equal(diagnostics.points[:, 1], expected)
    else:
        assert torch.equal(prediction.point_predictions, expected)
    assert not torch.equal(expected, head(prepared.working_state).squeeze(-1))


def test_disabled_route_preserves_the_prepared_prediction_without_calling_a_head():
    _, prepared, _, _ = fixture()

    def forbidden(_):
        pytest.fail("La ruta apagada sin lector no necesita otra cabeza")

    result, diagnostics = api().apply_spaced_utility_readout(prepared, forbidden)
    assert result.point_predictions is prepared.point_predictions
    assert result.readout is None and diagnostics is None


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_saturated_original_attention_still_gives_finite_deletion_predictions(dtype):
    _, prepared, head, model = fixture(dtype=dtype, neighbors=3, temperature=1e-4)
    data = snapshot(3, dtype=dtype)
    prediction, diagnostics = apply(prepared, head, model, data)
    assert prediction.readout.reads[0].weights.tolist() == [[1, 0, 0]] * 2
    assert torch.isfinite(diagnostics.points).all()
    assert diagnostics.points[:, 1].tolist() == pytest.approx(
        [1 + 0.1 * math.tanh(2.5)] * 2, abs=2e-7 if dtype == torch.float32 else 1e-15
    )


def test_emitted_diagnostics_enter_the_ledger_without_reinferring_after_maturity(monkeypatch):
    _, prepared, head, model = fixture(neighbors=2)
    data = snapshot(3)
    prediction, diagnostics = apply(prepared, head, model, data)
    rows = diagnostics.ledger_rows(dict.fromkeys(prepared.next_state.flow_ids, AT + 1))
    assert [r["point"] for r in rows] == prediction.point_predictions.tolist()
    assert diagnostics.snapshot_sha256 == data.export_cpu()["sha256"]
    run = SpacedUtilityLedger(CODEC)
    run.begin(AT)
    run.issue(data, rows)
    assert run.publish() == []
    monkeypatch.setattr(head, "forward", lambda *a: pytest.fail("Cabeza después de madurar"))
    run.begin(AT + 1)
    run.receive([(r["flow_id"], AT, AT + 1, 0.0) for r in rows])
    receipt = run.publish()[0]
    expected = abs(diagnostics.points[0, 1].item()) - abs(diagnostics.points[0, 0].item())
    assert receipt["gains"][10]["mean"] == expected
    assert run.scores()[10]["sessions"] == 1


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("steps", [1, 2, 4])
@pytest.mark.parametrize("differentiable", [False, True])
def test_disabled_wrapper_keeps_bits_and_gradients_of_all_legacy_refinements(
    dtype, steps, differentiable
):
    _, prepared, head, model = fixture(dtype=dtype, refinements=steps)
    data = snapshot(3, dtype=dtype)
    options = dict(context_id=CONTEXT, cutoff=AT, differentiable=differentiable)
    reference = apply_episodic_readout(prepared, head, model, data, **options)
    result, diagnostics = api().apply_spaced_utility_readout(prepared, head, model, data, **options)
    assert diagnostics is None
    assert torch.equal(result.point_predictions, reference.point_predictions)
    assert torch.equal(result.readout.state, reference.readout.state)
    for a, b in zip(result.readout.reads, reference.readout.reads, strict=True):
        assert all(torch.equal(getattr(a, k), getattr(b, k)) for k in ("values", "weights", "ids"))
    if differentiable:
        parameters = (head.weight, model.refinement.weight, model.query_projection.weight)
        left = torch.autograd.grad(result.point_predictions.sum(), parameters)
        right = torch.autograd.grad(reference.point_predictions.sum(), parameters)
        assert all(torch.equal(a, b) for a, b in zip(left, right, strict=True))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_enabled_original_preserves_the_ordinary_prediction_and_rng(dtype):
    _, prepared, head, model = fixture(dtype=dtype)
    data = snapshot(3, dtype=dtype)
    before = torch.get_rng_state().clone()
    ordinary = apply_episodic_readout(prepared, head, model, data, context_id=CONTEXT, cutoff=AT)
    result, diagnostics = apply(prepared, head, model, data)
    assert torch.equal(ordinary.point_predictions, result.point_predictions)
    assert torch.equal(ordinary.readout.state, result.readout.state)
    assert torch.equal(before, torch.get_rng_state())
    assert not diagnostics.points.requires_grad and diagnostics.points.grad_fn is None
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("wrong", ["dtype", "device"])
def test_head_mismatch_fails_before_projection(monkeypatch, wrong):
    _, prepared, head, model = fixture()
    head = head.float() if wrong == "dtype" else head.to("meta")
    monkeypatch.setattr(
        model.query_projection, "forward", lambda *a: pytest.fail("Proyección antes del contrato")
    )
    with pytest.raises(ValueError):
        apply(prepared, head, model, snapshot())


def test_autocast_is_rejected_before_reading(monkeypatch):
    _, prepared, head, model = fixture(dtype=torch.float32)
    monkeypatch.setattr(
        model.query_projection, "forward", lambda *a: pytest.fail("Autocast sin declarar")
    )
    with torch.autocast("cpu", dtype=torch.bfloat16), pytest.raises(ValueError, match="autocast"):
        apply(prepared, head, model, snapshot(dtype=torch.float32))


@pytest.mark.parametrize("fault", ["steps", "mode", "training", "head", "gradient", "snapshot"])
def test_unsupported_contract_is_rejected_before_a_read(monkeypatch, fault):
    options = dict(refinements=2) if fault == "steps" else {}
    if fault == "mode":
        options["mode"] = "no_bank"
    _, prepared, head, model = fixture(**options)
    data, kwargs = snapshot(), {}
    if fault == "training":
        model.train()
    if fault == "head":
        head.train()
    if fault == "gradient":
        kwargs["differentiable"] = True
    if fault == "snapshot":
        data = None
    monkeypatch.setattr(model, "_read_parts", lambda *a, **k: pytest.fail("Lectura sin contrato"))
    with pytest.raises(ValueError):
        apply(prepared, head, model, data, **kwargs)


@pytest.mark.parametrize("flag", [None, 1, "true"])
def test_non_boolean_activation_is_rejected(flag):
    with pytest.raises(ValueError):
        api().apply_spaced_utility_readout(None, None, enabled=flag)


@pytest.mark.parametrize("fault", ["duplicate", "market", "time", "unknown"])
def test_unaligned_identity_is_rejected_before_a_read(monkeypatch, fault):
    _, prepared, head, model = fixture()
    changes = {
        "duplicate": dict(flow_ids=("US/AAA", "US/AAA")),
        "market": dict(flow_ids=("US/AAA", "CN/BBB")),
        "time": dict(last_prediction_at=(AT, AT + 1)),
        "unknown": dict(flow_ids=("US/AAA", "XX/BBB")),
    }
    prepared = replace(prepared, next_state=replace(prepared.next_state, **changes[fault]))
    monkeypatch.setattr(model, "_read_parts", lambda *a, **k: pytest.fail("Lectura desalineada"))
    with pytest.raises(ValueError):
        apply(prepared, head, model, snapshot())


def test_combined_budget_fails_before_projection(monkeypatch):
    _, _, _, reference = fixture()
    ordinary = reference.estimated_bytes(2, 3, refinements=1)
    combined = reference.estimated_bytes(2, 3, refinements=4)
    _, prepared, head, model = fixture(max_working_bytes=(ordinary + combined) // 2)
    monkeypatch.setattr(
        model.query_projection,
        "forward",
        lambda *a: pytest.fail("Proyección antes del presupuesto"),
    )
    with pytest.raises(ValueError, match="presupuesto"):
        apply(prepared, head, model, snapshot())


@pytest.mark.parametrize("fault", ["context", "codec", "version"])
def test_incompatible_snapshot_cannot_be_consumed(fault):
    _, prepared, head, model = fixture()
    data = snapshot()
    if fault == "context":
        object.__setattr__(data, "context_id", "c" * 64)
    elif fault == "codec":
        object.__setattr__(data, "codec_id", "c" * 64)
    else:
        data._values[1][0, 0] += 1
    with pytest.raises(ValueError):
        apply(prepared, head, model, data)


def test_existing_quantile_head_preserves_its_median_and_order():
    _, prepared, _, model = fixture()
    with torch.device("cpu"), torch.random.fork_rng(devices=[]):
        head = QuantileHead(32, dtype=torch.float64).eval()
    prepared = replace(prepared, quantiles=head(prepared.working_state).detach())
    ordinary = apply_episodic_readout(
        prepared, head, model, snapshot(), context_id=CONTEXT, cutoff=AT
    )
    result, diagnostics = apply(prepared, head, model, snapshot())
    assert torch.equal(ordinary.quantiles, result.quantiles)
    assert torch.equal(result.quantiles[:, 2], diagnostics.points[:, 0])
    assert (result.quantiles[:, 1:] >= result.quantiles[:, :-1]).all()


def test_readout_and_snapshot_recovery_reproduce_original_and_deletions():
    _, prepared, head, model = fixture()
    data = snapshot()
    reference, diagnostics = apply(prepared, head, model, data)
    replica = EpisodicReadout(model.config, dtype=torch.float64).eval()
    replica.load_state_dict(copy.deepcopy(model.state_dict()))
    restored = EpisodeSnapshot.restore(
        data.export_cpu(),
        codec_id=CODEC,
        context_id=CONTEXT,
        cutoff=AT,
        device="cpu",
        dtype=torch.float64,
    )
    following, trace = apply(prepared, head, replica, restored)
    assert torch.equal(reference.point_predictions, following.point_predictions)
    assert torch.equal(diagnostics.points, trace.points)
    assert torch.equal(diagnostics.episode_ids, trace.episode_ids)


@pytest.mark.parametrize("fault", ["missing", "extra", "early", "boolean"])
def test_ledger_rows_reject_missing_and_invalid_maturities(fault):
    _, prepared, head, model = fixture()
    _, diagnostics = apply(prepared, head, model, snapshot())
    times = dict.fromkeys(prepared.next_state.flow_ids, AT + 1)
    if fault == "missing":
        times.pop(prepared.next_state.flow_ids[0])
    elif fault == "extra":
        times["US/OTHER"] = AT + 1
    else:
        times[prepared.next_state.flow_ids[0]] = AT if fault == "early" else True
    with pytest.raises(ValueError):
        diagnostics.ledger_rows(times)


def test_asset_permutation_preserves_aligned_diagnostics():
    _, prepared, head, model = fixture()
    generator = torch.Generator().manual_seed(5)
    prepared = replace(
        prepared, working_state=torch.randn((2, 32), generator=generator, dtype=torch.float64)
    )
    _, original = apply(prepared, head, model, snapshot())
    other = replace(
        prepared,
        working_state=prepared.working_state.flip(0),
        next_state=replace(prepared.next_state, flow_ids=prepared.next_state.flow_ids[::-1]),
    )
    _, permuted = apply(other, head, model, snapshot())
    assert torch.equal(original.points, permuted.points.flip(0))
    assert torch.equal(original.episode_ids, permuted.episode_ids.flip(0))


def test_regional_readout_requires_its_own_deletion_producer(monkeypatch):
    from mars_titan.models.titans.evidence_bridge import EvidenceBridgeConfig

    _, prepared, head, _ = fixture()
    bridge = EvidenceBridgeConfig(CODEC, hidden_size=32)
    model = EpisodicReadout(
        EpisodicReadoutConfig(CODEC, hidden_size=32, evidence_bridge=bridge), dtype=torch.float64
    ).eval()
    monkeypatch.setattr(
        model, "_read_parts", lambda *a, **k: pytest.fail("Lectura regional implícita")
    )
    with pytest.raises(ValueError, match="estándar"):
        apply(prepared, head, model, snapshot())


@pytest.mark.parametrize("fault", ["shape", "dtype", "nan", "graph", "ids", "negative"])
def test_invalid_diagnostic_tensors_cannot_become_ledger_rows(fault):
    _, prepared, head, model = fixture()
    _, trace = apply(prepared, head, model, snapshot())
    if fault == "shape":
        trace = replace(trace, points=trace.points[:, :-1])
    elif fault == "dtype":
        trace = replace(trace, points=trace.points.half())
    elif fault == "nan":
        trace = replace(trace, points=torch.full_like(trace.points, torch.nan))
    elif fault == "graph":
        trace = replace(trace, points=trace.points.clone().requires_grad_())
    elif fault == "negative":
        trace = replace(trace, episode_ids=-trace.episode_ids)
    else:
        trace = replace(trace, episode_ids=torch.ones_like(trace.episode_ids))
    with pytest.raises(ValueError):
        trace.ledger_rows(dict.fromkeys(prepared.next_state.flow_ids, AT + 1))


def test_unknown_maturity_is_preserved_without_an_invented_label():
    _, prepared, head, model = fixture()
    _, trace = apply(prepared, head, model, snapshot())
    rows = trace.ledger_rows(dict.fromkeys(prepared.next_state.flow_ids, None))
    assert all(r["label_not_before"] is None and "target" not in r for r in rows)


def test_invalid_maturity_fails_before_copying_diagnostic_tensors(monkeypatch):
    _, prepared, head, model = fixture()
    _, trace = apply(prepared, head, model, snapshot())
    monkeypatch.setattr(torch.Tensor, "cpu", lambda *a: pytest.fail("Copia anterior a maduración"))
    with pytest.raises(ValueError):
        trace.ledger_rows(dict.fromkeys(prepared.next_state.flow_ids, AT))


@pytest.mark.parametrize("quantile", [False, True])
def test_malformed_head_output_cannot_be_published(quantile):
    _, prepared, head, model = fixture()
    if quantile:
        with torch.device("cpu"), torch.random.fork_rng(devices=[]):
            head = QuantileHead(32, dtype=torch.float64).eval()
        prepared = replace(prepared, quantiles=torch.zeros((2, 5), dtype=torch.float64))
    handle = head.register_forward_hook(lambda _, args, output: output.repeat(1, 2))
    try:
        with pytest.raises(ValueError, match="cabeza"):
            apply(prepared, head, model, snapshot())
    finally:
        handle.remove()


@pytest.mark.cuda
@pytest.mark.parametrize("count", [0, 3])
def test_cuda_decoder_preserves_prediction_and_emits_finite_detached_diagnostics(count):
    _, prepared, head, model = fixture(dtype=torch.float32)
    prepared = replace(prepared, working_state=prepared.working_state.to("cuda:0"))
    head, model = head.to("cuda:0"), model.to("cuda:0")
    data = EpisodeSnapshot.restore(
        snapshot(count, dtype=torch.float32).export_cpu(),
        codec_id=CODEC,
        context_id=CONTEXT,
        cutoff=AT,
        device="cuda:0",
        dtype=torch.float32,
    )
    ordinary = apply_episodic_readout(prepared, head, model, data, context_id=CONTEXT, cutoff=AT)
    prediction, trace = apply(prepared, head, model, data)
    assert torch.equal(ordinary.point_predictions, prediction.point_predictions)
    assert torch.isfinite(trace.points).all() and trace.points.grad_fn is None
    assert trace.points.device == torch.device("cuda:0")
    assert len(trace.ledger_rows(dict.fromkeys(prepared.next_state.flow_ids, AT + 1))) == 2
