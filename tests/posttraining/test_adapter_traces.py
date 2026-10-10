"""Ganchos de trazas de los adaptadores (#448), comprobados sin pasos de optimizador.

El gancho se llama tras cada validación completa. Estas pruebas recorren un brazo con el
gancho desactivado y activado y exigen los mismos gradientes, predicciones, recibos y RNG.
"""

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mars_titan.data.storage import sha256
from mars_titan.models.predictive_adaptation import AdapterTarget, adapted_copy, adapter_names
from mars_titan.models.quantile_head import QUANTILE_HEAD
from mars_titan.posttraining import adapter_matrix
from mars_titan.posttraining import chronological_matrix as cm
from mars_titan.posttraining.adapter_traces import prediction_change, update_statistics
from mars_titan.posttraining.run import run_case
from tests.posttraining.masked_fixture import masked_ordered
from tests.posttraining.test_quantile_adaptation import setup
from tests.posttraining.test_titans_adapters import adapted, posttrainer, predictor
from tests.training.test_candidate_walk_forward import allowed as allowed
from tests.training.test_candidate_walk_forward import views as views
from tests.training.test_financial_run import corpus, named_records
from tests.training.test_mars_titan_run import native as native

ROOT = Path(__file__).resolve().parents[2]
MATRIX_PATH = ROOT / "configs/posttraining/adapter-matrix-v3.json"
ARM = "head+readout+fusion"


@pytest.fixture(autouse=True)
def explicit_fastpath():
    previous = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    yield
    torch.backends.mha.set_fastpath_enabled(previous)


@pytest.fixture(scope="module")
def matrix():
    return adapter_matrix.read_matrix(MATRIX_PATH)


@pytest.fixture(scope="module")
def edition(tmp_path_factory):
    view, ordered, report = masked_ordered(tmp_path_factory.mktemp("traces") / "data")
    return SimpleNamespace(view=view, ordered=ordered, report=report)


class _Recorder:
    """Traza de prueba: estadísticas de la actualización y cambio frente al padre."""

    def __init__(self, parent, probe):
        self.parent, self.probe, self.events = parent, probe, []

    def __call__(self, event, modules):
        model = modules["model"]
        inputs, presence = self.probe
        with torch.inference_mode():
            change = prediction_change(model(inputs, presence), self.parent(inputs, presence))
        self.events.append((event, update_statistics(model), change))


def _probe(parent):
    """Filas de sondeo fijas, construidas antes del ajuste y fuera de su RNG."""
    generator = torch.Generator().manual_seed(0)
    inputs = {
        name: torch.randn(
            (3, parent.context, width) if name == "prices" else (3, width), generator=generator
        )
        for name, width in parent.dimensions.items()
    }
    return inputs, torch.ones((3, 5), dtype=torch.bool)


def test_the_trace_hook_does_not_change_a_run(tmp_path, recorder, edition, matrix):
    document, digest = matrix
    parent, data, grid, scale, handles = setup(edition, tmp_path, "transformer")
    case = next(
        item["case"]
        for item in adapter_matrix.cases(document, digest, "transformer", head=QUANTILE_HEAD)
        if item["id"] == f"seed-42/{ARM}"
    )
    runs = {}
    trace = _Recorder(parent.model, _probe(parent.model))
    for name, hook in (("plain", None), ("traced", trace)):
        report = run_case(
            data,
            tmp_path / name,
            case,
            grid,
            scale,
            parent=parent,
            batch_size=2,
            device="cpu",
            diagnostic=True,
            trace=hook,
        )
        runs[name] = (report, recorder.optimizers[-1], torch.random.get_rng_state())
    (plain, first, plain_rng), (traced, second, traced_rng) = runs["plain"], runs["traced"]
    assert len(first.calls) == len(second.calls) > 0
    for left, right in zip(first.calls, second.calls, strict=True):
        assert all(torch.equal(a, b) for a, b in zip(left, right, strict=True))
    assert torch.equal(plain_rng, traced_rng)
    for key in ("epochs", "selection", "baseline", "global_step", "identity"):
        assert plain[key] == traced[key], key
    assert sha256(tmp_path / "plain/validation-predictions.parquet") == sha256(
        tmp_path / "traced/validation-predictions.parquet"
    )
    events = [event for event, _, _ in trace.events]
    epochs = document["budget"]["epochs"]
    assert [event["epoch"] for event in events] == list(range(epochs + 1))
    assert [event["score"] for event in events] == [plain["baseline"]["session_mae"]] + [
        row["validation"]["session_mae"] for row in plain["epochs"]
    ]
    assert [event["global_step"] for event in events] == [
        epoch * len(first.calls) // epochs for epoch in range(epochs + 1)
    ]
    statistics, change = trace.events[-1][1], trace.events[-1][2]
    # Sin pasos, la actualización efectiva y el cambio frente al padre son exactamente cero.
    assert statistics and all(row["frobenius"] == 0.0 for row in statistics)
    matrices = [row for row in statistics if "singular_values" in row]
    assert matrices and all(set(row["singular_values"]) == {0.0} for row in matrices)
    assert change["quantile_distance"] == 0.0 and change["median_sign_flips"] == 0.0
    for handle in handles:
        handle.close()


def test_a_trace_hook_must_be_callable(tmp_path, recorder, edition, matrix):
    document, digest = matrix
    parent, data, grid, scale, handles = setup(edition, tmp_path, "gru")
    case = adapter_matrix.cases(document, digest, "gru", head=QUANTILE_HEAD)[1]["case"]
    with pytest.raises(ValueError, match="invocable"):
        run_case(
            data,
            tmp_path / "run",
            case,
            grid,
            scale,
            parent=parent,
            batch_size=2,
            device="cpu",
            diagnostic=True,
            trace="events.parquet",
        )
    assert not (tmp_path / "run").exists()
    for handle in handles:
        handle.close()


def test_the_chronological_trace_hook_does_not_change_a_run(matrix, tmp_path_factory):
    document, _ = matrix
    _, streams = corpus(tmp_path_factory.mktemp("traces-titans"))
    parent = predictor(streams, "mac_online")
    points = {
        arm["id"]: arm["points"] for arm in cm.arms(document, "titans_mac", variant="mac_online")
    }
    root, engines, events = tmp_path_factory.mktemp("traces-runs"), {}, []

    def trace(event, modules):
        events.append((event, update_statistics(modules["predictor"])))

    for name in ("plain", "traced"):
        engine = posttrainer(streams, root / name, adapted(parent, document, points[ARM]))
        if name == "traced":
            engine.trace = trace
        engine.run()
        engines[name] = engine
    plain, traced = engines["plain"], engines["traced"]
    assert plain.audit == traced.audit
    assert plain.optimizer.calls == traced.optimizer.calls > 0
    for left, right in zip(named_records(plain), named_records(traced), strict=True):
        assert left.keys() == right.keys() == set(adapter_names(plain.predictor))
        assert all(torch.equal(left[key], right[key]) for key in left)
    assert [event["epoch"] for event, _ in events] == list(range(len(events))) and events
    assert [event["global_step"] for event, _ in events][0] == 0
    assert all(row["frobenius"] == 0.0 for _, rows in events for row in rows)


def test_update_statistics_measure_the_effective_update():
    holder = torch.nn.Module()
    holder.layer = torch.nn.Linear(4, 3).double()
    model = adapted_copy(
        holder, [AdapterTarget("layer", "weight", "low_rank", rank=2, alpha=4.0)], seed=1
    )
    chain = model.layer.parametrizations.weight[0]
    with torch.no_grad():
        chain.up.copy_(torch.tensor([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]]))
    (row,) = update_statistics(model)
    delta = 2.0 * chain.up.detach() @ chain.down.detach()
    # El tercer valor singular es cero salvo redondeo: se compara con una tolerancia absoluta.
    np.testing.assert_allclose(
        row["singular_values"],
        np.linalg.svd(delta.numpy(), compute_uv=False),
        rtol=1e-12,
        atol=1e-14,
    )
    assert row["frobenius"] == pytest.approx(float(delta.norm()), rel=1e-12)
    reference = holder.layer.weight.detach().norm()
    assert row["relative"] == pytest.approx(float(delta.norm() / reference), rel=1e-12)
    assert row["max_abs"] == pytest.approx(float(delta.abs().max()), rel=1e-12)
    with pytest.raises(ValueError):
        update_statistics(model, top=0)


def test_prediction_change_compares_levels_and_signs():
    parent = torch.tensor([[-2.0, -1.0, -0.1, 1.0, 2.0], [-2.0, -1.0, 0.5, 1.0, 2.0]])
    arm = parent.clone()
    arm[0] += 0.2
    change = prediction_change(arm, parent)
    assert change["rows"] == 2
    assert change["quantile_distance"] == pytest.approx(0.1)
    assert change["mean_abs_change"] == pytest.approx([0.1] * 5)
    assert change["max_abs_change"] == pytest.approx(0.2)
    assert change["median_sign_flips"] == 0.5
    with pytest.raises(ValueError):
        prediction_change(arm[:, :4], parent[:, :4])


def _same_records(left, right):
    """Mismos gradientes paso a paso, en el orden de los grupos del optimizador."""
    assert left.calls == right.calls > 0
    for first, second in zip(left.records, right.records, strict=True):
        values = list(first.values()), list(second.values())
        assert len(values[0]) == len(values[1])
        for a, b in zip(*values, strict=True):
            assert (a is None and b is None) or torch.equal(a, b)


@pytest.mark.skipif(
    not os.environ.get("MARS_TITAN_EPISODIC_NATIVE"), reason="Falta el enlace nativo compilado"
)
def test_the_reader_trace_hook_does_not_change_a_run(tmp_path, native):
    from tests.posttraining.test_readout_adapters import build

    _, streams = corpus(tmp_path / "corpus")
    engines, events = {}, []
    for name in ("plain", "traced"):
        engine = build(streams, tmp_path / name, native, "full_continuation")
        if name == "traced":
            engine.trace = lambda event, modules: events.append((event, sorted(modules)))
        engine.run()
        engines[name] = engine
    plain, traced = engines["plain"], engines["traced"]
    assert plain.audit == traced.audit
    _same_records(plain.optimizer, traced.optimizer)
    assert events and all(names == ["predictor", "readout"] for _, names in events)
    assert [event["epoch"] for event, _ in events] == list(range(len(events)))


@pytest.mark.skipif(
    not os.environ.get("MARS_TITAN_EPISODIC_NATIVE"), reason="Falta el enlace nativo compilado"
)
def test_the_candidate_trace_hook_does_not_change_a_window(views, allowed, monkeypatch):
    from mars_titan.training.candidate_run import CandidateChronologicalTrainer
    from tests.training.test_candidate_walk_forward import fit
    from tests.training.test_financial_run import RecordingOptimizer

    view = views.windows["US"]["fold-000"]
    plain, plain_made = fit(view, views.root / "trace-plain", factory=RecordingOptimizer)
    events = []

    def trace(event, modules):
        events.append((event, sorted(modules)))

    monkeypatch.setattr(CandidateChronologicalTrainer, "trace", staticmethod(trace))
    traced, traced_made = fit(view, views.root / "trace-traced", factory=RecordingOptimizer)
    _same_records(plain_made[0], traced_made[0])
    # Las huellas de las predicciones de los tres tramos coinciden con y sin trazas.
    assert {name: record["sha256"] for name, record in plain["predictions"].items()} == {
        name: record["sha256"] for name, record in traced["predictions"].items()
    }
    assert events and all(names == ["model"] for _, names in events)
    assert [event["epoch"] for event, _ in events] == list(range(len(events)))
