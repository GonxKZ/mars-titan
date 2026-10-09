"""Estimación de horas por variante y medición de caudal sin pasos de optimizador.

La estimación se comprueba con caudales y recuentos fijados. La medición se recorre en
CPU sobre el corpus técnico y debe dejar los pesos intactos y no crear optimizadores.
No se usa la GPU ni se registran pérdidas.
"""

import copy
import json
from pathlib import Path

import pytest
import torch
from torch.optim import optimizer as optim_module

from mars_titan.models.baselines import multimodal
from mars_titan.training import campaign_throughput as throughput
from mars_titan.training.campaign_plan import load_campaign
from tests.training.test_walk_forward_v2_views import PROTOCOLS, fixture, prepare

CAMPAIGNS = {
    v: Path(f"configs/baselines/historical-masked-campaign-{v.lower()}.json") for v in "AB"
}


def uniform(campaign, rows):
    """Mismas filas por tramo en todas las ventanas y caudales iguales en todos los brazos."""
    counts = {
        scope: {window: dict(rows) for window in resolved["windows"]}
        for scope, resolved in campaign["comparison_config"]["resolved_scopes"].items()
        if scope in campaign["scopes"]
    }
    rates = {
        arm: {name: dict(train=100.0, inference=400.0) for name, _ in candidates}
        for arm, candidates in campaign["neural"]["candidates"].items()
    }
    return counts, rates


ROWS = dict(train=36_000, validation=4_000, calibration=2_000, evaluation=8_000)


def test_job_seconds_follow_epochs_validation_and_final_predictions():
    rates = {
        "gru-00": dict(train=100.0, inference=400.0),
        "gru-10": dict(train=50.0, inference=200.0),
    }
    fit = dict(kind="fit", candidate="gru-00")
    # 30 épocas de 360 s de ajuste y 10 s de validación, más 50.000 filas finales a 400/s.
    assert throughput.neural_job_seconds(fit, ROWS, rates, 30) == pytest.approx(30 * 370 + 125)
    finalist = dict(kind="fit", candidate=None)
    assert throughput.neural_job_seconds(finalist, ROWS, rates, 30) == pytest.approx(30 * 740 + 250)
    carried = dict(kind="carry", candidate=None)
    assert throughput.neural_job_seconds(carried, ROWS, rates, 30) == pytest.approx(10_000 / 200)


@pytest.mark.parametrize("variant", ["A", "B"])
def test_hours_add_every_planned_neural_job_and_leave_tabulars_unmeasured(variant):
    campaign = load_campaign(CAMPAIGNS[variant])
    counts, rates = uniform(campaign, ROWS)
    estimate = throughput.estimate_hours(campaign, counts, rates)
    fit, carry = (30 * 370 + 125) / 3600, 10_000 / 400 / 3600
    windows = dict(
        A={"US": (19, 0), "CN": (13, 0), "US+CN": (13, 0)},
        B={"US": (7, 12), "CN": (5, 8), "US+CN": (5, 8)},
    )
    for scope, (trained, carried) in windows[variant].items():
        expected = 5 * (4 * trained * fit + 3 * carried * carry)
        assert estimate["scopes"][scope]["hours"] == pytest.approx(expected)
        assert estimate["scopes"][scope]["arms"]["gru"] == pytest.approx(expected / 5)
    assert estimate["tabular"] == "not_measured" and estimate["variant"] == variant
    assert estimate["neural_hours"] == pytest.approx(
        sum(s["hours"] for s in estimate["scopes"].values())
    )


def test_variant_b_costs_less_than_a_with_the_same_rates():
    estimates = {}
    for variant in "AB":
        campaign = load_campaign(CAMPAIGNS[variant])
        estimates[variant] = throughput.estimate_hours(campaign, *uniform(campaign, ROWS))
    assert estimates["B"]["neural_hours"] < estimates["A"]["neural_hours"]


@pytest.fixture
def cpu(monkeypatch):
    import mars_titan.data.embeddings as embeddings

    monkeypatch.setattr(embeddings, "require_cuda", lambda **_: torch.device("cpu"))
    for name, value in dict(
        synchronize=None, reset_peak_memory_stats=None, max_memory_allocated=0
    ).items():
        monkeypatch.setattr(torch.cuda, name, lambda *_, value=value: value)
    monkeypatch.setattr(torch, "set_num_threads", lambda _: None)
    deterministic = torch.are_deterministic_algorithms_enabled()
    yield
    torch.use_deterministic_algorithms(deterministic)


def test_measurement_runs_forward_and_backward_without_changing_weights(tmp_path, cpu, monkeypatch):
    data = fixture(tmp_path / "data", ("US",))
    prepare(data, PROTOCOLS["US"], tmp_path / "views")
    campaign = load_campaign(CAMPAIGNS["A"])
    campaign["neural"] = dict(
        campaign["neural"],
        batch_size=1,
        candidates={"gru": campaign["neural"]["candidates"]["gru"][:1]},
    )
    built = []

    class Recorded(multimodal.MultimodalReference):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            built.append((self, copy.deepcopy(self.state_dict())))

    monkeypatch.setattr(multimodal, "MultimodalReference", Recorded)
    hooks = len(optim_module._global_optimizer_pre_hooks)
    rates = throughput.measure_rates(
        campaign, tmp_path / "views/fold-018/manifest.json", batches=1, warmup=0
    )
    assert len(optim_module._global_optimizer_pre_hooks) == hooks
    (record,) = rates["gru"].values()
    assert record["train"] > 0 and record["inference"] > 0
    assert record["measured_train_rows"] == record["measured_inference_rows"] == 1
    assert not {"loss", "mae", "session_mae"} & set(record)
    ((model, initial),) = built
    for name, value in model.state_dict().items():
        assert torch.equal(value, initial[name]), name
    assert all(parameter.grad is None for parameter in model.parameters())


def test_measurement_rejects_invalid_batch_counts():
    campaign = load_campaign(CAMPAIGNS["A"])
    for options in (dict(batches=0), dict(warmup=-1), dict(batches=1.5)):
        with pytest.raises(ValueError, match="enteros acotados"):
            throughput.measure_rates(campaign, Path("absent.json"), **options)


def test_report_shape_is_serializable():
    campaign = load_campaign(CAMPAIGNS["B"])
    estimate = throughput.estimate_hours(campaign, *uniform(campaign, ROWS))
    assert json.loads(json.dumps(estimate)) == estimate


def test_script_checks_a_campaign_without_reading_data(capsys):
    import runpy

    script = runpy.run_path("scripts/run_masked_campaign.py", run_name="script")
    assert script["main"](["check", "--campaign", str(CAMPAIGNS["B"])]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "checked" and report["counts"]["prediction_jobs"] == 868
