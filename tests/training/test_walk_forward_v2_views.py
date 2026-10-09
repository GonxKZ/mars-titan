"""Vistas anuales v2 sobre un corpus técnico con máscaras, sin modelos ni entrenamiento."""

import hashlib
import json
from collections import Counter
from datetime import UTC, date, datetime
from pathlib import Path
from types import SimpleNamespace

import pyarrow.parquet as pq
import pytest

from mars_titan.data.input_policy import HISTORICAL_MASKED
from mars_titan.data.storage import atomic_json
from mars_titan.data.temporal import MarketClock
from mars_titan.evaluation.splits import PARTITIONS, build_folds
from mars_titan.training import temporal_search
from mars_titan.training.joint_temporal_corpus import prepare_joint_temporal_corpus
from mars_titan.training.temporal_corpus import prepare_temporal_corpus
from tests.training.historical_temporal_fixture import historical_temporal_fixture

CONFIGS = Path("configs/evaluation")
PROTOCOLS = {
    "US": CONFIGS / "historical-masked-us-walk-forward-v2.json",
    "CN": CONFIGS / "historical-masked-cn-walk-forward-v2.json",
    "JOINT-US": CONFIGS / "historical-masked-joint-us-walk-forward-v2.json",
}
FIRST_ROWS = {"US": date(2000, 2, 1), "CN": date(2006, 7, 15)}
# Huellas de las vistas v1 calculadas con el código anterior a la versión 2.
V1_VIEWS = {
    "US": "88dc99e5c8958191b9408f1b693c33044d6b1d652820230cb819144c55285930",
    "JOINT": "52346f1085e364fa35f16353814f050fb38477557465eaf0f0c6fede4e8714b0",
}


def sessions(market):
    """Elegir decisiones dentro de cada tramo y la última sesión antes de cada frontera."""
    clock = MarketClock(market, "1999-01-01", "2024-01-05")
    chosen = []
    for year in range(2000, 2024):
        days = [day for day in clock.days if day.year == year and day >= FIRST_ROWS[market]]
        for month in (2, 5, 11):
            chosen += [day for day in days if day.month == month and day.day >= 15][:1]
        for month in (3, 9, 12):
            chosen += [day for day in days if day.month == month][-1:]
    return sorted(day.isoformat() for day in chosen)


def fixture(root, markets):
    days = {market: sessions(market) for market in markets}
    result = historical_temporal_fixture(root, markets=markets, days=days)
    return SimpleNamespace(**vars(result), days=days)


def micros(moment):
    return int(moment.timestamp() * 1_000_000)


def expected(market, days, protocol):
    """Derivar recuentos sin FoldPartitioner, con la etiqueta de la sesión siguiente."""
    clock = MarketClock(market, "1999-01-01", "2024-01-05")
    counts = Counter()
    purged = Counter()
    for day in days:
        if day == "2023-12-29":
            continue  # El padre ya la excluye porque madura después del corte de 2023.
        position = clock.days.index(date.fromisoformat(day))
        prediction = micros(clock.decisions[position])
        maturity = micros(clock.decisions[position + 1])
        for fold in build_folds(protocol):
            for name in PARTITIONS:
                start, end = (
                    micros(datetime.fromisoformat(v).replace(tzinfo=UTC)) for v in fold[name]
                )
                if start <= prediction < end:
                    target = counts if maturity < end else purged
                    target[(fold["id"], name, int(day[:4]))] += 1
    return counts, purged


def observed(output, fold, market):
    labels = output / fold / "labels" / market
    rows = Counter()
    for path in labels.glob("*/labels.parquet"):
        for row in pq.read_table(path).to_pylist():
            if row["partition"] is not None:
                rows[(fold, row["partition"], row["prediction_at"].year)] += 1
                assert row["prediction_at"].year < 2024
    return rows


def prepare(fixture, protocol, output):
    return prepare_temporal_corpus(
        fixture.parent,
        protocol,
        None,
        None,
        output,
        input_policy=HISTORICAL_MASKED,
        recover_annual_boundaries=True,
    )


def view_digest(output):
    digest = hashlib.sha256()
    for fold in sorted(p.name for p in output.iterdir() if p.name.startswith("fold-")):
        meta = json.loads((output / fold / "manifest.json").read_text())
        record = dict(
            counts=meta["counts"],
            recovered=meta.get("recovered_annual_labels"),
            assets=[
                {k: a[k] for k in ("market", "symbol", "counts", "excluded_reasons")}
                for a in meta["assets"]
            ],
            labels=[
                [str(v) for v in row.values()]
                for a in meta["assets"]
                for row in pq.read_table(
                    Path(meta["roots"]["labels"]) / a["market"] / a["symbol"] / "labels.parquet"
                ).to_pylist()
            ],
            fold=[v["fold"] for v in meta["temporal_views"].values()]
            if "temporal_views" in meta
            else meta["temporal_view"]["fold"],
        )
        digest.update(json.dumps(record, sort_keys=True).encode())
    return digest.hexdigest()


@pytest.fixture(scope="module")
def joint_views(tmp_path_factory):
    root = tmp_path_factory.mktemp("joint-v2")
    data = fixture(root, ("US", "CN"))
    output = root / "views"
    report = prepare_joint_temporal_corpus(
        data.parent,
        dict(US=dict(protocol=PROTOCOLS["JOINT-US"]), CN=dict(protocol=PROTOCOLS["CN"])),
        output,
        input_policy=HISTORICAL_MASKED,
        recover_annual_boundaries=True,
    )
    return SimpleNamespace(data=data, output=output, report=report)


def test_v1_historical_views_keep_exact_labels_counts_and_reasons(tmp_path):
    single = historical_temporal_fixture(tmp_path / "us")
    prepare(single, single.protocols["US"], tmp_path / "us-views")
    assert view_digest(tmp_path / "us-views") == V1_VIEWS["US"]
    joint = historical_temporal_fixture(tmp_path / "joint", markets=("US", "CN"))
    prepare_joint_temporal_corpus(
        joint.parent,
        {market: dict(protocol=path) for market, path in joint.protocols.items()},
        tmp_path / "joint-views",
        input_policy=HISTORICAL_MASKED,
        recover_annual_boundaries=True,
    )
    assert view_digest(tmp_path / "joint-views") == V1_VIEWS["JOINT"]
    report = json.loads((tmp_path / "joint-views/report.json").read_text())
    assert all("market_purged_by_boundary" not in row for row in report["folds"])


def test_us_v2_views_count_rows_by_year_and_record_each_purged_boundary(tmp_path):
    data = fixture(tmp_path, ("US",))
    report = prepare(data, PROTOCOLS["US"], tmp_path / "views")
    protocol = json.loads(PROTOCOLS["US"].read_text())
    counts, purged = expected("US", data.days["US"], protocol)
    assert [row["id"] for row in report["folds"]] == [f"fold-{i:03d}" for i in range(19)]
    for row in report["folds"]:
        fold = row["id"]
        assert observed(tmp_path / "views", fold, "US") == Counter(
            {key: n for key, n in counts.items() if key[0] == fold}
        )
        assert row["has_all_partitions"] is True
        assert row["purged_by_boundary"] == {
            name: sum(n for key, n in purged.items() if key[:2] == (fold, name))
            for name in PARTITIONS
        }
        meta = json.loads((tmp_path / "views" / fold / "manifest.json").read_text())
        assert meta["assets"][0]["purged_by_boundary"] == row["purged_by_boundary"]
        assert meta["assets"][0]["excluded_reasons"]["label_crosses_boundary"] == sum(
            row["purged_by_boundary"].values()
        )
        assert "session_gap" not in meta["assets"][0]["excluded_reasons"]
    last = report["folds"][-1]
    # 2022-03-31, 2022-09-30 y 2022-12-30 cruzan su frontera. 2023-12-29 ya llega excluida.
    assert last["purged_by_boundary"] == dict(train=1, validation=1, calibration=1, evaluation=0)


def test_joint_v2_views_start_in_2011_with_both_markets_in_every_partition(joint_views):
    data, output, report = joint_views.data, joint_views.output, joint_views.report
    assert len(report["folds"]) == 13
    for market, path in (("US", PROTOCOLS["JOINT-US"]), ("CN", PROTOCOLS["CN"])):
        counts, purged = expected(market, data.days[market], json.loads(path.read_text()))
        for row in report["folds"]:
            fold = row["id"]
            local = Counter({key: n for key, n in counts.items() if key[0] == fold})
            assert observed(output, fold, market) == local
            assert row["market_counts"][market] == {
                name: sum(n for key, n in local.items() if key[1] == name) for name in PARTITIONS
            }
            assert all(row["market_counts"][market].values())
            assert row["market_purged_by_boundary"][market] == {
                name: sum(n for key, n in purged.items() if key[:2] == (fold, name))
                for name in PARTITIONS
            }
    first = report["folds"][0]
    years = Counter(key[2] for key in observed(output, "fold-000", "CN").elements())
    assert min(years) == 2006 and first["has_all_partitions"] is True


def test_v2_rejects_a_joint_window_that_would_leave_a_market_empty(tmp_path):
    data = fixture(tmp_path, ("US", "CN"))
    early = {}
    for market in ("US", "CN"):
        value = json.loads(PROTOCOLS["US"].read_text()) | dict(market=market)
        early[market] = tmp_path / f"early-{market}.json"
        atomic_json(early[market], value)
    with pytest.raises(ValueError, match="sin filas en CN"):
        prepare_joint_temporal_corpus(
            data.parent,
            {market: dict(protocol=path) for market, path in early.items()},
            tmp_path / "joint",
            input_policy=HISTORICAL_MASKED,
            recover_annual_boundaries=True,
        )
    assert not (tmp_path / "joint").exists()


def plan(tmp_path, name="historical-masked-reference-search-us.json", **changes):
    value = json.loads((Path("configs/baselines") / name).read_text())
    value.update(arms=["US+CN"], models=["rnn"], finalist_seeds=[42])
    value.update(changes)
    path = tmp_path / "plan.json"
    atomic_json(path, value)
    return path


def test_search_check_accepts_masked_v2_views_without_reserving_the_gpu(
    joint_views, tmp_path, monkeypatch
):
    class Forbidden:
        def __enter__(self):
            raise AssertionError("La comprobación no debe reservar la GPU")

    monkeypatch.setattr(temporal_search, "GpuLease", Forbidden)
    result = temporal_search.check_temporal_search(plan(tmp_path), joint_views.output)
    assert result["status"] == "checked" and len(result["folds"]) == 13
    assert result["identity"]["input_policy"] == HISTORICAL_MASKED
    assert set(result["identity"]["protocols_sha256"]) == {"US", "CN"}
    assert result["folds"][0]["fold"]["evaluation"] == ["2011-01-01", "2012-01-01"]
    assert result["planned_runs"] == 13 * result["runs_per_fold"]
    assert result["scientific_training_started"] is False


@pytest.mark.parametrize(
    "change",
    [
        dict(patience=10),
        dict(max_epochs=20),
        dict(min_delta=0.0),
        dict(stopping="validation_plateau"),
    ],
)
def test_search_rejects_a_plan_that_does_not_apply_the_common_stopping_rule(
    joint_views, tmp_path, change
):
    with pytest.raises(ValueError, match="regla de parada"):
        temporal_search.check_temporal_search(plan(tmp_path, **change), joint_views.output)


@pytest.mark.parametrize(
    ("name", "changes"),
    [
        ("historical-masked-reference-search-us.json", dict(input_policy="strict_inputs_v1")),
        ("convergence-temporal-search-us.json", {}),
        ("strict-temporal-search-us.json", {}),
    ],
)
def test_search_reads_the_policy_from_the_plan_and_rejects_strict_plans_on_masked_views(
    joint_views, tmp_path, name, changes
):
    with pytest.raises(ValueError, match="política de entradas"):
        temporal_search.check_temporal_search(plan(tmp_path, name, **changes), joint_views.output)


def test_masked_campaign_runs_each_window_with_the_plan_that_declares_the_policy(
    joint_views, tmp_path, monkeypatch
):
    calls = []

    def study(config, manifest, folder, *, resume=False, progress=None):
        calls.append((json.loads(config.read_text())["input_policy"], manifest.parent.name))
        folder.mkdir()
        result = dict(status="paused", planned_runs=per_fold, completed_runs=0)
        progress(result)
        return result

    class Lease:
        def __enter__(self):
            return SimpleNamespace(record={"device": "prueba"}, check=lambda: None)

        def __exit__(self, *args):
            return False

    path = plan(tmp_path)
    per_fold = temporal_search.check_temporal_search(path, joint_views.output)["runs_per_fold"]
    monkeypatch.setattr(temporal_search, "run_search", study)
    monkeypatch.setattr(temporal_search, "GpuLease", Lease)
    summary = temporal_search.run_temporal_search(path, joint_views.output, tmp_path / "campaign")
    assert summary["status"] == "paused" and calls == [(HISTORICAL_MASKED, "fold-000")]
    assert summary["identity"]["input_policy"] == HISTORICAL_MASKED
    assert summary["planned_runs"] == 13 * per_fold


def test_strict_preflight_keeps_its_identity_fields(tmp_path):
    from tests.training.test_temporal_search import metadata_views

    records, identity, per_fold = temporal_search._inputs(
        Path("configs/baselines/strict-temporal-search-us.json"), metadata_views(tmp_path)
    )
    assert set(identity) == {
        "config_sha256",
        "views_report_sha256",
        "manifests",
        "code_sha256",
        "protocol_sha256",
    }
    assert len(records) == 4 and per_fold == 40


def test_plan_and_protocol_rules_compare_mode_minimum_and_budget():
    rule = json.loads(PROTOCOLS["US"].read_text())["selection"]
    historical = json.loads(
        Path("configs/baselines/historical-masked-reference-search-us.json").read_text()
    )
    assert temporal_search._stopping(historical) == temporal_search._stopping(rule)
    plateau = dict(rule, stopping="validation_plateau", minimum_epochs=3)
    convergence = dict(historical, schema_version=3, minimum_epochs=3)
    convergence.pop("stopping")
    assert temporal_search._stopping(convergence) == temporal_search._stopping(plateau)
    for change in (dict(minimum_epochs=4), dict(minimum_epochs=0), dict(patience=6)):
        assert temporal_search._stopping(convergence | change) != temporal_search._stopping(plateau)
