"""Medida del coste de las trazas sobre el corpus técnico en CPU (#448).

La orden real se ejecuta en `cuda:0` con las vistas de la campaña. Aquí se recorre en CPU,
con la GPU sustituida como en las pruebas de `campaign_throughput`, para comprobar que cada
condición escribe lo que declara, que ningún paso cambia pesos y que el informe lleva el
volumen de cada condición. Los tiempos no se comprueban.
"""

import pytest
import torch

from mars_titan.learning_traces import overhead
from mars_titan.learning_traces.recorder import read_traces
from mars_titan.posttraining import campaign_stage
from mars_titan.training import campaign_throughput as throughput
from mars_titan.training.campaign_plan import load_campaign
from tests.training.test_campaign_throughput import CAMPAIGNS, STAGES, titans_campaign

MIB = 1024**2
DECLARED = dict(every=2, max_bytes=MIB)


def test_conditions_rotate_so_each_one_goes_first_once():
    orders = overhead._orders(3)
    assert [order[0] for order in orders] == list(overhead.CONDITIONS)
    assert all(sorted(order) == sorted(overhead.CONDITIONS) for order in orders)
    assert overhead._config("off", DECLARED) is None
    assert overhead._config("every_step", DECLARED) == dict(every=1, max_bytes=MIB)
    assert overhead._config("declared", DECLARED) == DECLARED
    summary = overhead._summary([1.5, 1.2], [1.0, 1.0])
    assert summary == dict(
        median=pytest.approx(0.35),
        minimum=pytest.approx(0.2),
        maximum=pytest.approx(0.5),
        repeats=2,
    )


def guarded():
    return throughput._forbid_steps()


def test_neural_measure_writes_each_condition_without_changing_weights(views, cpu, tmp_path):
    campaign = load_campaign(CAMPAIGNS["A"])
    campaign["neural"] = dict(
        campaign["neural"],
        batch_size=2,
        candidates={"gru": campaign["neural"]["candidates"]["gru"][:1]},
    )
    settings = dict(warmup=1, steps=2, repeats=2)
    guard = guarded()
    try:
        result = overhead.measure_neural(
            campaign,
            views / "fold-018/manifest.json",
            tmp_path,
            DECLARED,
            settings,
            torch.device("cpu"),
        )
    finally:
        guard.remove()
    record = result["gru"]
    assert record["case"] == "gru-00" and record["rows_per_block"] == 4
    assert set(record["seconds_per_row"]) == set(overhead.CONDITIONS)
    assert {key: value["repeats"] for key, value in record["overhead"].items()} == dict(
        every_step=2, declared=2
    )
    assert record["volume"]["every_step"]["traced_steps"] == 3
    assert record["volume"]["declared"]["traced_steps"] == 1
    assert record["volume"]["every_step"]["bytes"] > record["volume"]["declared"]["bytes"] > 0
    folder = tmp_path / "neural" / "gru"
    assert not list(folder.glob("off-*"))
    for condition, steps in (("every_step", {1, 2, 3}), ("declared", {2})):
        for repeat in range(2):
            manifest, table = read_traces(folder / f"{condition}-{repeat}")
            rows = table.to_pylist()
            assert manifest["step_unit"] == "optimizer_step"
            assert manifest["identity"]["family"] == "neural_gru"
            assert {r["step"] for r in rows if r["phase"] == "train"} == steps
            assert {r["step"] for r in rows if r["phase"] == "validation"} == {3}


def test_posttraining_measure_traces_the_adapters_of_each_case(views, cpu, tmp_path):
    stage = campaign_stage.load_stage(STAGES["A"])
    budget = dict(stage["matrix"]["budget"], batch_size=1)
    stage.update(families={"gru": "gru"}, matrix=dict(stage["matrix"], budget=budget))
    guard = guarded()
    try:
        result = overhead.measure_posttraining(
            stage,
            views / "fold-018/manifest.json",
            tmp_path,
            DECLARED,
            dict(warmup=1, steps=1, repeats=1),
            torch.device("cpu"),
        )
    finally:
        guard.remove()
    cases = result["gru"]
    assert {"full_continuation", "head", "fusion_dora"} <= set(cases)
    for label, record in cases.items():
        assert record["volume"]["every_step"]["traced_steps"] == 2, label
    manifest, table = read_traces(tmp_path / "posttraining/gru/head/every_step-0")
    validation = {r["metric"] for r in table.to_pylist() if r["phase"] == "validation"}
    assert manifest["identity"]["family"] == "posttraining_gru"
    assert "adapter.delta_l2" in validation or "adapter.module_l2" in validation


def test_chronological_measure_attaches_the_traces_to_the_trainer(views, cpu, tmp_path):
    campaign = titans_campaign(tmp_path / "config")
    settings = dict(segments=1, segment_warmup=0, events=1, event_warmup=0)
    settings["chronological_repeats"] = 1
    result = overhead.measure_chronological(
        "titans_mac",
        throughput.measure_titans,
        campaign,
        views / "fold-000/manifest.json",
        tmp_path / "work",
        DECLARED,
        settings,
    )
    assert result and all(key.startswith("titans_mac_online/") for key in result)
    for key, record in result.items():
        assert set(record["rows_per_second"]) == set(overhead.CONDITIONS), key
        assert set(record["overhead"]) == {"every_step", "declared"}
        volume = record["volume"]["every_step"]
        assert volume["every"] == 1 and volume["bytes"] > 0 and volume["parts"] >= 1
        assert record["volume"]["declared"]["every"] == 2
    folders = sorted((tmp_path / "work/traces/titans_mac/every_step-0").iterdir())
    assert len(folders) == len(result)
    for folder in folders:
        manifest, table = read_traces(folder)
        assert manifest["identity"]["family"] == "titans_mac"
        assert "memory.alpha.mean" in {row["metric"] for row in table.to_pylist()}
    assert not (tmp_path / "work/traces/titans_mac/off-0").exists()


def test_events_take_the_width_of_the_view(views, tmp_path):
    campaign = load_campaign(CAMPAIGNS["A"])
    result = overhead.measure_events(
        views / "fold-018/manifest.json", campaign, tmp_path, DECLARED, repeats=4
    )
    assert result["ridge"]["coefficients"] > 0 and result["ridge"]["bytes"] > 0
    assert result["xgboost"]["rounds"] == 4 and result["xgboost"]["bytes_per_round"] > 0
    manifest, table = read_traces(tmp_path / "events/ridge")
    assert manifest["step_unit"] == "fit" and {r["step"] for r in table.to_pylist()} == {0}
    manifest, table = read_traces(tmp_path / "events/xgboost")
    assert manifest["step_unit"] == "boosting_round"
    assert {r["step"] for r in table.to_pylist()} == {1, 2, 3, 4}


def test_the_report_needs_the_deterministic_workspace(monkeypatch, tmp_path):
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    with pytest.raises(ValueError, match="CUBLAS_WORKSPACE_CONFIG"):
        overhead.measure(CAMPAIGNS["A"], tmp_path, tmp_path, declared=DECLARED, settings={})
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    with pytest.raises(ValueError, match="Familias"):
        overhead.measure(
            CAMPAIGNS["A"], tmp_path, tmp_path, declared=DECLARED, families=("x",), settings={}
        )
