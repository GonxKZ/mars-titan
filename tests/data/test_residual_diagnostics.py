"""Diagnóstico de la etiqueta residual: reconciliación, causas, sensibilidades y límites."""

import json
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from scipy import stats

from mars_titan.data import residual_diagnostics as rd
from mars_titan.data.storage import sha256
from tests.data.residual_diagnostics_fixture import (
    DECLARATION,
    FACTOR_GAP,
    FACTOR_START,
    INVALID_OPEN,
    STOCK_GAP,
    build,
    clock,
    rewrite_labels,
)


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    return build(tmp_path_factory.mktemp("residual-diagnostics"))


@pytest.fixture(scope="module")
def report(corpus):
    declaration = rd.load_declaration(corpus["declaration"])
    return rd.diagnose(corpus["targets"], corpus["edition"], declaration)


def test_the_declared_file_keeps_the_protocol_window():
    declaration = rd.load_declaration(DECLARATION)
    assert declaration["main"] == {"history": 252, "minimum": 126}
    assert [s["name"] for s in declaration["sensitivities"]] == [
        "window_126",
        "window_504",
        "market_adjusted",
    ]
    assert declaration["sector"]["admitted"] is False
    assert len(declaration["sha256"]) == 64


@pytest.mark.parametrize(
    "change",
    [
        dict(main={"history": 126, "minimum": 63}),
        dict(cutoff="2024-12-31"),
        dict(partitions=["train", "validation", "final_test"]),
        dict(
            sensitivities=[{"name": "window_252", "kind": "window", "history": 252, "minimum": 126}]
        ),
        dict(sensitivities=[{"name": "w", "kind": "window", "history": 10, "minimum": 20}]),
        dict(sensitivities=[{"name": "same", "kind": "market_adjusted"}] * 2),
        dict(sensitivities=[{"name": "beta", "kind": "market_adjusted", "beta": 2}]),
        dict(return_thresholds=[0.1, 0.05]),
        dict(quantiles=[0.5, 1.0]),
        dict(beta_jump_thresholds=[]),
        dict(sector={"admitted": True, "reason": "CSV"}),
        dict(extremes=0),
        dict(minimum_assets_per_session=1),
        dict(extra=True),
    ],
)
def test_a_declaration_outside_its_contract_is_rejected(tmp_path, change):
    document = json.loads(DECLARATION.read_text())
    document.update(change)
    path = tmp_path / "declaration.json"
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="contrato"):
        rd.load_declaration(path)


def test_every_sample_lands_in_one_category_and_reconciles_with_the_manifest(corpus, report):
    manifest = json.loads(corpus["targets"].read_text())
    counts = report["coverage"]["samples"]["by_market"]["US"]
    assert sum(counts.values()) == manifest["samples"]
    assert counts["train"] == manifest["counts"]["train"]
    assert counts["validation"] == manifest["counts"]["validation"]
    years = report["coverage"]["samples"]["by_market_and_year"]["US"]
    for category, total in counts.items():
        assert sum(year.get(category, 0) for year in years.values()) == total
    # Las causas desdobladas reproducen los motivos guardados de la generación.
    stored = Counter()
    for receipt in manifest["assets"]:
        stored.update(receipt["excluded_reasons"])
    assert (
        counts["insufficient_stock_history"] + counts["missing_factor_history"]
        == stored["insufficient_history"]
    )
    split = ("next_stock_session_absent", "next_stock_price_invalid", "next_factor_return_missing")
    assert sum(counts[c] for c in split) == stored["missing_next_session"]
    assert counts["outside_label_calendar"] == 1
    assert report["stored_targets_recomputed"]["accepted"] == counts["train"] + counts["validation"]
    assets = report["coverage"]["assets"]
    assert assets["candidates"] == 3 and assets["labelled"] == 2
    assert assets["states"] == {"encoded": 2, "missing_required_prices": 1}
    assert assets["with_insufficient_history"] == 2


def test_causes_follow_the_series_that_fails(report):
    counts = report["coverage"]["samples"]["by_market"]["US"]
    # El factor empieza 150 sesiones después de AAA: AAA tiene historia propia sin pares.
    assert counts["missing_factor_history"] > 0
    # BBB empieza tarde y sus primeras muestras no tienen historia propia.
    assert counts["insufficient_stock_history"] > 0
    # Antes de que empiece el factor falta su retorno siguiente, y la generación lo registra
    # como sesión siguiente ausente: las muestras de AAA de la 64 a la 148. Además, el hueco
    # suelto del factor deja una muestra de cada activo.
    assert counts["next_factor_return_missing"] == (FACTOR_START - 1 - 64) + 2
    assert counts["next_stock_session_absent"] == 1
    assert counts["next_stock_price_invalid"] == 1


def test_exact_cause_rows(corpus):
    declaration = rd.load_declaration(corpus["declaration"])
    task = next(
        t
        for t in rd._tasks(
            json.loads(corpus["targets"].read_text()),
            json.loads(corpus["edition"].read_text()),
            declaration,
            samples=False,
        )
        if t["symbol"] == "AAA"
    )
    market = clock()
    prices = rd.read_prices(task["prices"], task["receipt"]["prices_sha256"], "2023-12-31")
    factor = rd.read_prices(task["factor"], task["factor_sha256"], "2023-12-31")
    frame, stock, stock_at = rd._calendar(prices, factor, market, declaration)
    labels = pd.read_parquet(task["labels"])
    joined, _ = rd.classify(labels, frame, stock, stock_at, declaration)
    by_slot = joined.dropna(subset=["slot"]).set_index("slot").category
    assert by_slot[FACTOR_GAP - 1] == "next_factor_return_missing"
    assert by_slot[STOCK_GAP - 1] == "next_stock_session_absent"
    assert by_slot[INVALID_OPEN - 1] == "next_stock_price_invalid"


def test_raw_and_residual_use_the_same_observations(report):
    for partition in rd.PARTITIONS:
        block = report["distributions"][f"US:{partition}"]
        assert block["raw"]["n"] == block["residual"]["n"] > 0
        assert 0 < block["variance_ratio"] < 1
    total = sum(report["by_year"]["US"][y]["n"] for y in report["by_year"]["US"])
    assert total == report["stored_targets_recomputed"]["accepted"]


def test_sensitivities_share_one_common_population(report):
    variants = report["sensitivities"]["variants"]
    for partition in rd.PARTITIONS:
        rows = {
            variants[f"US:{partition}:{n}"]["common"]["n"]
            for n in ("main", "window_126", "window_504", "market_adjusted")
        }
        assert len(rows) == 1
        main = variants[f"US:{partition}:main"]
        assert main["common"]["correlation_with_main"] == pytest.approx(1.0)
        assert main["common"]["mean_absolute_difference"] == 0
        assert main["coverage_change"] == 0
        adjusted = variants[f"US:{partition}:market_adjusted"]["beta_stability"]
        assert adjusted["max_absolute_change"] == 0
    assert variants["US:train:window_126"]["coverage_change"] > 0
    assert variants["US:train:window_504"]["coverage_change"] < 0
    population = report["sensitivities"]["population"]
    assert (
        population["rows"]
        == variants["US:train:main"]["common"]["n"] + variants["US:validation:main"]["common"]["n"]
    )
    assert len(population["sha256"]) == 64


def test_mask_patterns_cover_every_sample(corpus, report):
    assert report["mask_patterns"]["available"] is True
    patterns = report["coverage"]["by_pattern"]["US"]
    total = sum(sum(c.values()) for c in patterns.values())
    reserved = report["coverage"]["samples"]["by_market"]["US"].get("final_test_reserved", 0)
    assert total == json.loads(corpus["targets"].read_text())["samples"] - reserved
    # Precios, gráficos y macro siempre presentes, noticias y fundamentales alternos.
    assert set(patterns) == {"10101", "10111", "11101", "11111"}


def test_missing_samples_disable_the_pattern_breakdown(tmp_path):
    corpus = build(tmp_path)
    (tmp_path / "samples" / "US" / "BBB" / "samples.parquet").unlink()
    report = rd.diagnose(
        corpus["targets"], corpus["edition"], rd.load_declaration(corpus["declaration"])
    )
    assert report["mask_patterns"]["available"] is False
    assert report["mask_patterns"]["missing_samples"] == 1
    assert report["coverage"]["by_pattern"] == {}


def test_prices_after_the_cutoff_never_reach_the_diagnostic(tmp_path, report):
    late = build(tmp_path, late=3)
    path = tmp_path / "prepared" / "US" / "AAA" / "prices.parquet"
    manifest = json.loads(late["targets"].read_text())
    receipt = next(a for a in manifest["assets"] if a["symbol"] == "AAA")
    frame = rd.read_prices(path, receipt["prices_sha256"], "2023-12-31")
    assert frame.session.max() <= "2023-12-31" and frame.close.max() < 1e6
    other = rd.diagnose(late["targets"], late["edition"], rd.load_declaration(late["declaration"]))
    for section in ("coverage", "distributions", "by_year", "stability", "exposure"):
        assert other[section] == report[section]


def test_a_changed_stored_target_stops_the_diagnostic(tmp_path):
    corpus = build(tmp_path)

    def nudge(table):
        row = table.index[table.partition == "train"][0]
        table.loc[row, "target"] = np.nextafter(table.loc[row, "target"], 1.0)
        return table

    rewrite_labels(corpus, "AAA", nudge)
    with pytest.raises(ValueError, match="recalculado"):
        rd.diagnose(
            corpus["targets"], corpus["edition"], rd.load_declaration(corpus["declaration"])
        )


def test_a_changed_reason_stops_the_diagnostic(tmp_path):
    corpus = build(tmp_path)

    def relabel(table):
        row = table.index[table.reason == "insufficient_history"][0]
        table.loc[row, "reason"] = "target_after_cutoff"
        return table

    rewrite_labels(corpus, "BBB", relabel)
    with pytest.raises(ValueError, match="recalculado"):
        rd.diagnose(
            corpus["targets"], corpus["edition"], rd.load_declaration(corpus["declaration"])
        )


def test_a_final_test_label_with_a_target_stops_the_diagnostic(tmp_path):
    corpus = build(tmp_path)

    def reveal(table):
        extra = table.iloc[[0]].copy()
        extra["prediction_at"] = pd.Timestamp("2024-01-03 21:05", tz="UTC")
        extra["reason"], extra["partition"], extra["target"] = "final_test_reserved", None, 0.01
        return pd.concat([table, extra], ignore_index=True)

    rewrite_labels(corpus, "AAA", reveal)
    with pytest.raises(ValueError, match="prueba final"):
        rd.diagnose(
            corpus["targets"], corpus["edition"], rd.load_declaration(corpus["declaration"])
        )


def test_reserved_final_test_rows_are_only_counted(tmp_path):
    corpus = build(tmp_path)

    def reserve(table):
        extra = table.iloc[[0]].copy()
        extra["prediction_at"] = pd.Timestamp("2024-01-03 21:05", tz="UTC")
        extra["reason"], extra["partition"], extra["target"] = "final_test_reserved", None, None
        extra["target_available_at"] = pd.NaT
        return pd.concat([table, extra], ignore_index=True)

    rewrite_labels(corpus, "AAA", reserve)
    report = rd.diagnose(
        corpus["targets"],
        corpus["edition"],
        rd.load_declaration(corpus["declaration"]),
        patterns=False,
    )
    counts = report["coverage"]["samples"]["by_market"]["US"]
    assert counts["final_test_reserved"] == 1
    assert report["coverage"]["samples"]["by_market_and_year"]["US"]["2024"] == {
        "final_test_reserved": 1
    }
    assert "2024" not in report["by_year"]["US"]
    assert report["coverage"]["losses"]["excluded"] == sum(
        n for c, n in counts.items() if c not in ("train", "validation", "final_test_reserved")
    )


def test_changed_prices_stop_the_diagnostic(tmp_path):
    corpus = build(tmp_path)
    path = tmp_path / "prepared" / "US" / "BBB" / "prices.parquet"
    path.write_bytes(path.read_bytes() + b"\0")
    with pytest.raises(ValueError, match="huella"):
        rd.diagnose(
            corpus["targets"], corpus["edition"], rd.load_declaration(corpus["declaration"])
        )


def test_targets_of_another_edition_are_rejected(tmp_path):
    corpus = build(tmp_path)
    corpus["edition"].write_text(corpus["edition"].read_text() + " ")
    with pytest.raises(ValueError, match="edición"):
        rd.diagnose(
            corpus["targets"], corpus["edition"], rd.load_declaration(corpus["declaration"])
        )


def test_extremes_keep_their_source_price(report):
    rows = report["extremes"]["residual"]["US"]
    assert len(rows) == 3
    assert abs(rows[0]["target"]) >= abs(rows[-1]["target"])
    for row in rows:
        assert row["raw"] == pytest.approx(row["next_close"] / row["next_open"] - 1, rel=1e-12)
        assert row["target"] == pytest.approx(
            row["raw"] - row["alpha"] - row["beta"] * row["factor"], abs=1e-15
        )
        assert len(row["prices_sha256"]) == 64


def test_moments_match_scipy():
    rng = np.random.default_rng(3)
    x = rng.standard_t(5, 4000) * 0.01
    result = rd.moments(x, [0.01, 0.5, 0.99], [0.02, 0.05])
    assert result["skewness"] == pytest.approx(stats.skew(x))
    assert result["excess_kurtosis"] == pytest.approx(stats.kurtosis(x))
    assert result["std"] == pytest.approx(x.std(ddof=1))
    assert result["quantiles"]["0.5"] == pytest.approx(np.median(x))
    assert result["beyond"] == {
        "0.02": int((np.abs(x) > 0.02).sum()),
        "0.05": int((np.abs(x) > 0.05).sum()),
    }
    assert rd.moments([], [0.5], [0.1]) == {"n": 0}


def test_association_and_sums_agree_with_direct_formulas():
    rng = np.random.default_rng(4)
    f = rng.normal(0, 0.01, 500)
    y = 0.3 * f + rng.normal(0, 0.01, 500)
    main = y + rng.normal(0, 0.001, 500)
    direct = rd.association(y, f)
    assert direct["correlation"] == pytest.approx(np.corrcoef(y, f)[0, 1])
    assert direct["slope"] == pytest.approx(np.polyfit(f, y, 1)[0])
    summary = rd._from_sums(rd._sums(y, f, main))
    assert summary["correlation_with_factor"] == pytest.approx(direct["correlation"])
    assert summary["correlation_with_main"] == pytest.approx(np.corrcoef(y, main)[0, 1])
    assert summary["mean_absolute_difference"] == pytest.approx(np.abs(y - main).mean())
    assert summary["std"] == pytest.approx(y.std(ddof=1))


def test_beta_jumps_only_count_adjacent_sessions():
    beta = np.array([1.0, 1.2, 1.25, 3.0, 3.0])
    slot = np.array([10, 11, 12, 20, 21])
    n, total, largest, *above = rd._jumps(beta, slot, [0.1, 0.5])
    assert n == 3
    assert total == pytest.approx(0.25)
    assert largest == pytest.approx(0.2)
    assert above == [1, 0]


def test_the_command_writes_the_report_and_reproducible_figures(tmp_path):
    from scripts import diagnose_residual_targets as command

    corpus = build(tmp_path / "corpus")
    output = tmp_path / "report" / "diagnostics.json"
    command.main(
        [
            "run",
            "--targets",
            str(corpus["targets"]),
            "--edition",
            str(corpus["edition"]),
            "--declaration",
            str(corpus["declaration"]),
            "--output",
            str(output),
        ]
    )
    report = json.loads(output.read_text())
    assert report["kind"] == rd.KIND and report["final_test_opened"] is False
    assert report["training_executed"] is False and report["seconds"] > 0
    digests = []
    for name in ("first", "second"):
        command.main(["figures", "--report", str(output), "--output", str(tmp_path / name)])
        manifest = json.loads((tmp_path / name / "figures.json").read_text())
        assert manifest["report_sha256"] == sha256(output)
        assert set(manifest["figures"]) == {
            "coverage-by-year.svg",
            "beta-by-month.svg",
            "dispersion-by-year.svg",
            "exposure-by-year.svg",
        }
        digests.append(manifest["figures"])
    assert digests[0] == digests[1]


def test_the_report_is_not_written_inside_the_targets(tmp_path):
    from scripts import diagnose_residual_targets as command

    corpus = build(tmp_path)
    with pytest.raises(ValueError, match="dentro"):
        command.main(
            [
                "run",
                "--targets",
                str(corpus["targets"]),
                "--edition",
                str(corpus["edition"]),
                "--declaration",
                str(corpus["declaration"]),
                "--output",
                str(tmp_path / "labels" / "report.json"),
            ]
        )


def test_parallel_workers_give_the_same_report(corpus, report):
    declaration = rd.load_declaration(corpus["declaration"])
    parallel = rd.diagnose(corpus["targets"], corpus["edition"], declaration, workers=2)
    timing = {"started_at_utc", "finished_at_utc", "workers"}
    assert {k: v for k, v in parallel.items() if k not in timing} == {
        k: v for k, v in report.items() if k not in timing
    }


def _asset_results(corpus):
    declaration = rd.load_declaration(corpus["declaration"])
    tasks = rd._tasks(
        json.loads(corpus["targets"].read_text()),
        json.loads(corpus["edition"].read_text()),
        declaration,
        samples=False,
    )
    return [rd.diagnose_asset(task) for task in tasks]


def test_daily_exposure_only_uses_sessions_with_enough_assets(corpus, report):
    columns = rd._concatenate(_asset_results(corpus))
    for index, partition in enumerate(rd.PARTITIONS):
        slots = columns["slot"][columns["partition"] == index]
        per_session = np.bincount(slots)
        exposure = report["exposure"][f"US:{partition}"]
        # Con dos activos y un mínimo declarado de dos, solo cuentan las sesiones con ambos.
        assert exposure["sessions"] == int((per_session >= 2).sum()) > 0
        assert exposure["sessions_below_minimum"] == int((per_session == 1).sum())
        assert exposure["daily_mean_residual"]["n"] == exposure["sessions"]
        assert exposure["pooled_residual"]["n"] == len(slots)


def test_partial_windows_and_adjacent_changes_are_counted_per_asset(corpus, report):
    results = _asset_results(corpus)
    columns = rd._concatenate(results)
    for index, partition in enumerate(rd.PARTITIONS):
        keep = columns["partition"] == index
        block = report["stability"][f"US:{partition}"]
        assert block["partial_windows"] == int((columns["pairs"][keep] < 252).sum())
        adjacent = 0
        for result in results:
            own = result["columns"]
            slots = own["slot"][own["partition"] == index]
            adjacent += int((np.diff(slots) == 1).sum())
        assert block["adjacent_pairs"] == adjacent
    assert report["stability"]["US:train"]["minimum_pairs"] == 126


def _row(slot, prediction):
    return SimpleNamespace(slot=float(slot), prediction_at=prediction)


def test_insufficient_history_counts_only_known_stock_returns():
    moments = pd.date_range("2022-01-03 21:00", periods=10, freq="D", tz="UTC")
    stock = np.array([0.01] * 10)
    stock_at = pd.Series(moments)
    # Exactamente el mínimo de retornos propios conocidos: falta el factor, no la historia.
    assert rd._insufficient(_row(4, moments[4]), stock, stock_at, 5, 5) == (
        "missing_factor_history"
    )
    assert rd._insufficient(_row(3, moments[3]), stock, stock_at, 5, 5) == (
        "insufficient_stock_history"
    )
    # Un retorno que se conoce después de la decisión no cuenta como historia.
    late = stock_at.copy()
    late.iloc[4] = moments[4] + pd.Timedelta(minutes=10)
    assert rd._insufficient(_row(4, moments[4]), stock, late, 5, 5) == (
        "insufficient_stock_history"
    )
    stock[2] = np.nan
    assert rd._insufficient(_row(4, moments[4]), stock, stock_at, 5, 5) == (
        "insufficient_stock_history"
    )


def test_a_changed_maturity_stops_the_diagnostic(tmp_path):
    corpus = build(tmp_path)

    def delay(table):
        row = table.index[table.partition == "train"][0]
        table.loc[row, "target_available_at"] += pd.Timedelta(seconds=1)
        return table

    rewrite_labels(corpus, "AAA", delay)
    with pytest.raises(ValueError, match="recalculado"):
        rd.diagnose(
            corpus["targets"], corpus["edition"], rd.load_declaration(corpus["declaration"])
        )


def test_a_purged_label_must_cross_a_boundary(tmp_path):
    corpus = build(tmp_path)

    def purge(table):
        row = table.index[table.partition == "train"][0]
        table.loc[row, ["partition", "target"]] = None
        table.loc[row, "target_available_at"] = pd.NaT
        table.loc[row, "reason"] = "target_crosses_partition_boundary"
        return table

    rewrite_labels(corpus, "AAA", purge)
    with pytest.raises(ValueError, match="frontera"):
        rd.diagnose(
            corpus["targets"], corpus["edition"], rd.load_declaration(corpus["declaration"])
        )


def test_a_reserved_reason_before_2024_is_rejected(tmp_path):
    corpus = build(tmp_path)

    def reserve(table):
        row = table.index[table.reason == "insufficient_history"][0]
        table.loc[row, "reason"] = "final_test_reserved"
        return table

    rewrite_labels(corpus, "BBB", reserve)
    with pytest.raises(ValueError, match="prueba final"):
        rd.diagnose(
            corpus["targets"], corpus["edition"], rd.load_declaration(corpus["declaration"])
        )


def test_labels_that_differ_from_their_receipt_are_rejected(tmp_path):
    corpus = build(tmp_path)
    path = tmp_path / "labels" / "US" / "BBB" / "labels.parquet"
    table = pd.read_parquet(path)
    table.iloc[::-1].to_parquet(path, index=False)
    with pytest.raises(ValueError, match="han cambiado"):
        rd.diagnose(
            corpus["targets"], corpus["edition"], rd.load_declaration(corpus["declaration"])
        )


def test_changed_samples_disable_the_pattern_breakdown(tmp_path):
    corpus = build(tmp_path)
    path = tmp_path / "samples" / "US" / "BBB" / "samples.parquet"
    table = pq.read_table(path)
    flipped = [[True, True, True, True, True]] * table.num_rows
    table = table.set_column(1, "presence", pa.array(flipped, type=pa.list_(pa.bool_(), 5)))
    pq.write_table(table, path)
    report = rd.diagnose(
        corpus["targets"], corpus["edition"], rd.load_declaration(corpus["declaration"])
    )
    assert report["mask_patterns"]["available"] is False
    assert report["mask_patterns"]["states"] == {"verified": 1, "changed": 1}
    assert report["coverage"]["by_pattern"] == {}


def test_pattern_bits_follow_the_contract_order(corpus, report):
    manifest = json.loads(corpus["targets"].read_text())
    expected = Counter()
    for receipt in manifest["assets"]:
        for i in range(receipt["samples"]):
            expected[f"1{int(i % 3 == 0)}1{int(i % 5 == 0)}1"] += 1
    patterns = report["coverage"]["by_pattern"]["US"]
    assert {p: sum(c.values()) for p, c in patterns.items()} == dict(expected)


def test_a_maturity_after_the_cutoff_excludes_every_variant(corpus):
    declaration = rd.load_declaration(corpus["declaration"])
    manifest = json.loads(corpus["targets"].read_text())
    receipt = next(a for a in manifest["assets"] if a["symbol"] == "AAA")
    prices = rd.read_prices(
        corpus["root"] / "prepared" / "US" / "AAA" / "prices.parquet",
        receipt["prices_sha256"],
        "2023-12-31",
    )
    factor = rd.read_prices(
        manifest["market_factors"]["US"]["prices_path"],
        manifest["market_factors"]["US"]["prices_sha256"],
        "2023-12-31",
    )
    # El último factor de 2023 se conoce el 2 de enero: ninguna variante puede aceptarlo.
    factor.loc[factor.index[-1], "available_at"] = pd.Timestamp("2024-01-02 15:00", tz="UTC")
    frame, _, _ = rd._calendar(prices, factor, clock(), declaration)
    row = frame.iloc[-2]
    assert row.reason == "target_after_cutoff"
    for item in declaration["sensitivities"]:
        assert not row[f"{item['name']}:accepted"]


def test_stability_never_pairs_rows_of_two_assets():
    columns = dict(
        asset=np.array([0, 0, 1, 1]),
        slot=np.array([10, 11, 12, 13]),
        beta=np.array([1.0, 1.0, 3.0, 3.0]),
        alpha=np.zeros(4),
        pairs=np.full(4, 252),
        market=np.zeros(4, dtype=np.int8),
        partition=np.zeros(4, dtype=np.int8),
    )
    declaration = rd.load_declaration(DECLARATION)
    block = rd._stability(columns, {"US": 0}, declaration)["US:train"]
    assert block["adjacent_pairs"] == 2
    assert block["beta_change"]["max_absolute"] == 0
    assert block["zero_beta"] == 0


def test_flat_windows_are_counted_as_zero_betas():
    columns = dict(
        asset=np.zeros(4, dtype=np.int32),
        slot=np.arange(4),
        raw=np.array([0.0, 0.0, 0.01, -0.02]),
        residual=np.array([0.0, 0.0, 0.01, -0.02]),
        factor=np.array([0.01, -0.01, 0.0, 0.02]),
        beta=np.array([0.0, 0.0, 0.0, 1e-12]),
        alpha=np.zeros(4),
        pairs=np.full(4, 252),
        market=np.zeros(4, dtype=np.int8),
        partition=np.zeros(4, dtype=np.int8),
    )
    declaration = rd.load_declaration(DECLARATION)
    years = np.full(4, 2010, dtype=np.int16)
    row = rd._by_year(columns, {"US": 0}, years, declaration)["US"]["2010"]
    assert (row["zero_beta"], row["zero_raw"]) == (3, 2)
    assert rd._stability(columns, {"US": 0}, declaration)["US:train"]["zero_beta"] == 3
