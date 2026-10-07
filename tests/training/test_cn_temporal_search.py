"""Admisión de CN con calendario propio y recuperación sin ejecutar entrenamientos."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mars_titan.data.storage import atomic_json, sha256
from mars_titan.data.temporal import MarketClock
from mars_titan.evaluation.splits import FoldPartitioner
from mars_titan.training import temporal_search
from tests.training.test_temporal_search import metadata_views


def campaign(tmp_path, market="CN"):
    protocol = json.loads(Path("configs/evaluation/chinese-real-walk-forward.json").read_text())
    protocol["market"] = market
    views = metadata_views(tmp_path, protocol)
    plan = json.loads(Path("configs/baselines/convergence-temporal-search-us.json").read_text())
    plan["arms"] = [market]
    path = tmp_path / "configuration.json"
    atomic_json(path, plan)
    return path, views


def rewrite_views(views, change):
    report_path = views / "report.json"
    report = json.loads(report_path.read_text())
    for fold in report["folds"]:
        path = views / fold["id"] / "manifest.json"
        meta = json.loads(path.read_text())
        change(meta)
        atomic_json(path, meta)
        fold["manifest_sha256"] = sha256(path)
    atomic_json(report_path, report)


def test_cn_preflight_preserves_ten_windows_and_the_same_neural_budget(tmp_path):
    config, views = campaign(tmp_path)
    records, identity, per_fold = temporal_search._inputs(config, views)
    assert [r["id"] for r in records] == [f"fold-{i:03}" for i in range(10)]
    assert per_fold == 40
    assert per_fold * len(records) == 400
    assert records[0]["fold"]["train"] == ["2022-06-01", "2022-12-01"]
    assert records[-1]["fold"]["evaluation"] == ["2023-12-01", "2024-01-01"]
    assert identity["config_sha256"] == sha256(config)
    assert identity["manifests"]["fold-009"] == sha256(records[-1]["manifest"])


def test_cn_configuration_preserves_us_cases_selection_and_seed_budgets(tmp_path):
    from mars_titan.training.reference_search import _configuration

    us, us_cases, _ = _configuration(Path("configs/baselines/convergence-temporal-search-us.json"))
    path = Path("configs/baselines/convergence-temporal-search-cn.json")
    cn, cn_cases, digest = _configuration(path)
    assert cn["arms"] == ["CN"]
    assert cn | {"arms": ["US"]} == us
    assert cn_cases == us_cases
    _, views = campaign(tmp_path)
    records, identity, per_fold = temporal_search._inputs(path, views)
    assert identity["config_sha256"] == digest
    assert len(records) * per_fold == 400


@pytest.mark.parametrize("market", ["US", "CN"])
@pytest.mark.parametrize(
    "fault",
    [
        "protocol",
        "missing_protocol",
        "assets",
        "invalid_asset",
        "declared_markets",
        "empty_assets",
        "scope",
        "incomplete",
    ],
)
def test_single_market_preflight_rejects_mixed_or_misidentified_views(tmp_path, market, fault):
    config, views = campaign(tmp_path, market)
    other = "US" if market == "CN" else "CN"

    def change(meta):
        if fault == "protocol":
            meta["temporal_view"]["protocol"]["market"] = other
        elif fault == "missing_protocol":
            meta["temporal_view"]["protocol"] = None
        elif fault == "assets":
            meta["assets"].append(dict(market=other, symbol="other"))
        elif fault == "invalid_asset":
            meta["assets"].append(None)
        elif fault == "declared_markets":
            meta["markets"] = [market, other]
        elif fault == "empty_assets":
            meta["assets"] = []
        elif fault == "scope":
            meta["scope"] = "development_snapshot"
        else:
            meta["cohort_complete"] = False

    rewrite_views(views, change)
    with pytest.raises(ValueError):
        temporal_search._inputs(config, views)


@pytest.mark.parametrize("arms", [["US+CN"], ["US", "CN"]])
def test_pooled_arms_still_need_a_separate_temporal_contract(tmp_path, arms):
    config, views = campaign(tmp_path)
    plan = json.loads(config.read_text())
    plan["arms"] = arms
    atomic_json(config, plan)
    with pytest.raises(ValueError):
        temporal_search._inputs(config, views)


def test_single_market_configuration_cannot_claim_unused_balanced_weights(tmp_path):
    config, views = campaign(tmp_path, "US")
    plan = json.loads(config.read_text())
    plan["pooled_weightings"] = ["natural", "balanced_markets"]
    atomic_json(config, plan)
    with pytest.raises(ValueError):
        temporal_search._inputs(config, views)


def test_legacy_us_views_can_identify_the_market_from_their_assets(tmp_path):
    config, views = campaign(tmp_path, "US")
    rewrite_views(views, lambda meta: meta.pop("markets"))
    records, _, per_fold = temporal_search._inputs(config, views)
    assert len(records) == 10 and per_fold == 40


def test_cn_development_scope_stays_explicit_without_claiming_a_complete_cohort(tmp_path):
    config, views = campaign(tmp_path)
    plan = json.loads(config.read_text())
    plan["scope"] = "development_snapshot"
    atomic_json(config, plan)
    rewrite_views(
        views, lambda meta: meta.update(scope="development_snapshot", cohort_complete=False)
    )
    records, _, per_fold = temporal_search._inputs(config, views)
    assert len(records) == 10 and per_fold == 40


def test_cn_protocol_keeps_session_gap_mature_labels_and_available_inputs(tmp_path):
    config, views = campaign(tmp_path)
    records, _, _ = temporal_search._inputs(config, views)
    meta = json.loads(records[0]["manifest"].read_text())
    protocol = meta["temporal_view"]["protocol"]
    clock = MarketClock("CN", "2022-06-01", "2023-04-01")
    prediction = np.array(
        [
            int(clock.decision(day).timestamp() * 1_000_000)
            for day in ("2022-11-28", "2022-11-29", "2022-11-30", "2023-03-01")
        ]
    )
    maturity = np.array(
        [
            int(clock.decision(day).timestamp() * 1_000_000)
            for day in ("2022-11-29", "2022-12-01", "2022-12-01", "2023-03-02")
        ]
    )
    available = prediction.copy()
    available[-1] += 1
    result = FoldPartitioner(records[0]["fold"], clock, protocol).assign(
        prediction, available, maturity
    )
    assert result["partition"].tolist() == ["train", "excluded", "excluded", "excluded"]
    assert result["reason"].tolist() == [
        "accepted",
        "label_crosses_boundary",
        "session_gap",
        "future_inputs",
    ]


def test_cn_controller_recovers_a_pause_and_rejects_changed_campaign_identity(
    tmp_path, monkeypatch
):
    config, views = campaign(tmp_path)

    class Lease:
        def __enter__(self):
            return SimpleNamespace(record={"device": "prueba CPU"}, check=lambda: None)

        def __exit__(self, *args):
            pass

    calls = []

    def study(config, manifest, output, *, resume, progress):
        assert json.loads(manifest.read_text())["temporal_view"]["protocol"]["market"] == "CN"
        calls.append((output.name, resume))
        output.mkdir(exist_ok=True)
        result = dict(
            status="paused" if len(calls) == 1 else "completed",
            completed_runs=1 if len(calls) == 1 else 40,
            planned_runs=40,
        )
        progress(result)
        return result

    monkeypatch.setattr(temporal_search, "GpuLease", Lease)
    monkeypatch.setattr(temporal_search, "run_search", study)
    output = tmp_path / "campaign"
    first = temporal_search.run_temporal_search(config, views, output)
    assert first["status"] == "paused" and first["completed_runs"] == 1
    result = temporal_search.run_temporal_search(config, views, output, resume=True)
    assert result["status"] == "completed"
    assert result["completed_runs"] == result["planned_runs"] == 400
    assert calls[:2] == [("fold-000", False), ("fold-000", True)]
    assert calls[2:] == [(f"fold-{i:03}", False) for i in range(1, 10)]
    saved = (output / "summary.json").read_bytes()
    plan = json.loads(config.read_text())
    plan["finalist_seeds"] = [42, 43, 45]
    atomic_json(config, plan)
    with pytest.raises(ValueError, match="identidad"):
        temporal_search.run_temporal_search(config, views, output, resume=True)
    assert (output / "summary.json").read_bytes() == saved
    assert len(calls) == 11
