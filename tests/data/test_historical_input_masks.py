"""Entradas históricas con ausencias declaradas, sin modelos ni aprendizaje."""

import json
from datetime import timedelta

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mars_titan.data.cohort_contexts import MacroVectors
from mars_titan.data.cohort_preparation import prepare_cohort_asset
from mars_titan.data.cohort_samples import materialize_cohort_asset
from mars_titan.data.embeddings import EmbeddingCache
from mars_titan.data.price_windows import calendar_digest, price_window_contract
from mars_titan.data.prices import read_prices
from mars_titan.data.samples import macro_vector
from mars_titan.data.storage import sha256
from mars_titan.data.temporal import MarketClock
from tests.data.test_cohort_samples import Encoders

HISTORICAL = "historical_masked_2000_v1"
CONCEPTS = ("us-gaap:Assets:USD",)


def prices_only(tmp_path, *, gap=False):
    raw = tmp_path / "raw"
    raw.mkdir()
    clock = MarketClock("US", "2023-01-01", "2025-01-01")
    days = list(clock.days[:68])
    # `gap` quita la sesión 10, o las posiciones indicadas.
    for position in sorted((10,) if gap is True else gap or (), reverse=True):
        days.pop(position)
    path = raw / "prices.csv"
    path.write_text(
        "Date,Open,High,Low,Close,Volume\n"
        + "".join(f"{day},10,12,9,11,100\n" for day in days)
        + "2024-01-03,10,12,9,11,100\n"
    )
    asset = dict(
        market="US",
        symbol="A",
        hashes={"prices.csv": sha256(path)},
        paths=dict(prices=["prices.csv"], news=[], fundamentals=[], charts=[]),
    )
    return raw, asset, clock


def prepare_missing(tmp_path, **kwargs):
    raw, asset, clock = prices_only(tmp_path)
    report = prepare_cohort_asset(
        raw,
        tmp_path / "prepared",
        asset,
        clock,
        cohort="original_audited",
        reviews={},
        input_policy=HISTORICAL,
        **kwargs,
    )
    return tmp_path / "prepared/US/A", clock, report


def encode_missing(source, output, clock, *, macro=None, price_window=None):
    macro = macro or MacroVectors(None, indicators=["a", "b"], input_policy=HISTORICAL)
    cache = EmbeddingCache(output.parent / "cache.sqlite")
    try:
        report = materialize_cohort_asset(
            source,
            output,
            clock,
            macro,
            Encoders(),
            cache,
            cohort="original_audited",
            context=64,
            company_factors=False,
            fundamental_concepts=CONCEPTS,
            input_policy=HISTORICAL,
            price_window=price_window,
        )
        return report, pq.read_table(output / "samples.parquet")
    finally:
        cache.close()


def test_price_only_preparation_retains_typed_absence_and_closed_cutoff(tmp_path):
    source, _, report = prepare_missing(tmp_path)
    assert report["schema_version"] == 4
    assert report["input_policy"] == HISTORICAL
    assert report["counts"] == dict(prices=68, news=0, fundamentals=0)
    assert report["reserved_counts"]["prices"] == 1
    assert report["missing_sources"] == ["news", "fundamentals", "charts"]
    assert pq.read_table(source / "news/news.parquet").num_rows == 0
    assert "available_at" in pq.read_schema(source / "fundamentals.parquet").names
    assert report["training_ready"] is False


def test_missing_optional_blocks_do_not_remove_price_windows(tmp_path):
    source, clock, _ = prepare_missing(tmp_path)
    report, table = encode_missing(source, tmp_path / "encoded", clock)
    assert report["samples"] == 5
    assert report["excluded_reasons"] == {"incomplete_price_window": 63}
    for row in table.to_pylist():
        assert row["presence"] == [True, False, True, False, False]
        assert row["missing_reasons"]["news"] == "source_missing"
        assert row["missing_reasons"]["macro"] == "source_missing"
        assert row["news_count"] == 0
        assert not any(row["news"] + row["fundamentals"] + row["macro"])
        assert row["input_availability"]["news"] is None
        assert row["input_availability"]["fundamentals"] is None
        assert row["input_availability"]["macro"] is None
        assert row["input_availability"]["prices"] <= row["prediction_at"]
        assert row["input_availability"]["charts"] == row["prediction_at"]
        assert row["session"] < "2024-01-01"
    again, repeated = encode_missing(source, tmp_path / "encoded", clock)
    assert again["reused"] is True
    assert repeated.equals(table)


def test_zero_is_observed_but_unknown_and_future_publications_are_missing(tmp_path):
    clock = MarketClock("US", "2023-01-01", "2024-01-01")
    cutoff = clock.decisions[0]
    rows = [
        dict(indicator_id="a", value=0.0, available_at=cutoff),
        dict(indicator_id="b", value=7.0, available_at=None),
        dict(indicator_id="c", value=9.0, available_at=cutoff + timedelta(days=1)),
    ]
    vector, available = macro_vector(rows, cutoff, input_policy=HISTORICAL)
    assert vector == [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert available == cutoff
    path = tmp_path / "macro.parquet"
    pq.write_table(pa.Table.from_pylist([dict(row, prediction_at=cutoff) for row in rows]), path)
    context = MacroVectors(path, input_policy=HISTORICAL)
    np.testing.assert_array_equal(context.at(cutoff)[0], vector)
    assert context.missing_at(cutoff) == [None, "unknown_publication", "not_yet_available"]
    with pytest.raises(ValueError, match="disponibilidad|futura"):
        macro_vector(rows, cutoff)


def test_all_missing_macro_session_remains_in_calendar(tmp_path):
    cutoff = MarketClock("US", "2023-01-01", "2024-01-01").decisions[0]
    schema = pa.schema(
        [
            ("prediction_at", pa.timestamp("us", tz="UTC")),
            ("indicator_id", pa.string()),
            ("value", pa.float64()),
            ("available_at", pa.timestamp("us", tz="UTC")),
        ]
    )
    path = tmp_path / "macro.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [dict(prediction_at=cutoff, indicator_id="a", value=None, available_at=None)],
            schema=schema,
        ),
        path,
    )
    context = MacroVectors(path, input_policy=HISTORICAL)
    np.testing.assert_array_equal(context.at(cutoff)[0], np.zeros(3))
    assert context.at(cutoff)[1] is None
    assert context.missing_at(cutoff) == ["missing_value"]
    assert MacroVectors(path).at(cutoff) is None


def test_price_gaps_are_not_padded_or_hidden_as_optional_missing(tmp_path):
    raw, asset, clock = prices_only(tmp_path, gap=True)
    prepare_cohort_asset(
        raw,
        tmp_path / "prepared",
        asset,
        clock,
        cohort="original_audited",
        reviews={},
        input_policy=HISTORICAL,
    )
    report, table = encode_missing(tmp_path / "prepared/US/A", tmp_path / "encoded", clock)
    assert table.num_rows == 0
    assert report["excluded_reasons"] == {"incomplete_price_window": 67}


def window_contract(clock, absent):
    calendar = dict(
        start=clock.days[0].isoformat(),
        end=clock.days[-1].isoformat(),
        decisions_sha256=calendar_digest(clock),
    )
    return price_window_contract({"US": calendar}, {"US": absent})


def prepared_with_gap(tmp_path, *, gap=True):
    raw, asset, clock = prices_only(tmp_path, gap=gap)
    prepare_cohort_asset(
        raw,
        tmp_path / "prepared",
        asset,
        clock,
        cohort="original_audited",
        reviews={},
        input_policy=HISTORICAL,
    )
    return tmp_path / "prepared/US/A", clock


def test_market_absent_session_is_admitted_with_its_slot_and_without_a_price(tmp_path):
    source, clock = prepared_with_gap(tmp_path)
    missing = clock.days[10].isoformat()
    contract = window_contract(clock, [missing])
    report, table = encode_missing(source, tmp_path / "encoded", clock, price_window=contract)
    assert report["samples"] == 5
    assert report["excluded_reasons"] == {"incomplete_price_window": 62}
    assert report["market_absent_windows"] == {"windows": 5, "absent_sessions": 5}
    assert report["price_window"] == contract
    rows = table.to_pylist()
    assert [row["price_end_index"] for row in rows] == [62, 63, 64, 65, 66]
    assert missing not in {row["session"] for row in rows}
    assert all(row["presence"][0] for row in rows)
    configuration = json.loads((tmp_path / "encoded/configuration.json").read_text())
    assert configuration["price_window"] == contract
    assert "price_windows.py" in configuration["code"]
    # La misma serie sin la ausencia declarada conserva la exclusión anterior.
    other, empty = encode_missing(
        source, tmp_path / "strict", clock, price_window=window_contract(clock, [])
    )
    assert empty.num_rows == 0 and other["excluded_reasons"] == {"incomplete_price_window": 67}
    # Un contrato con otro calendario no puede codificar este activo.
    wrong = window_contract(clock, [missing])
    wrong["calendars"]["US"]["decisions_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="otro calendario"):
        encode_missing(source, tmp_path / "wrong", clock, price_window=wrong)


def test_two_consecutive_market_absences_count_each_absent_session(tmp_path):
    source, clock = prepared_with_gap(tmp_path, gap=(10, 11))
    missing = [clock.days[10].isoformat(), clock.days[11].isoformat()]
    report, table = encode_missing(
        source, tmp_path / "encoded", clock, price_window=window_contract(clock, missing)
    )
    assert report["samples"] == 5
    assert report["excluded_reasons"] == {"incomplete_price_window": 61}
    assert report["market_absent_windows"] == {"windows": 5, "absent_sessions": 10}
    assert table.column("price_end_index").to_pylist() == [61, 62, 63, 64, 65]


def test_declared_market_absence_cannot_hide_an_existing_price(tmp_path):
    source, clock = prepared_with_gap(tmp_path, gap=False)
    contract = window_contract(clock, [clock.days[10].isoformat()])
    with pytest.raises(ValueError, match="ausente en todo el mercado tiene precios"):
        encode_missing(source, tmp_path / "encoded", clock, price_window=contract)


def test_audited_prices_are_reused_and_their_hash_is_checked(tmp_path, monkeypatch):
    raw, asset, clock = prices_only(tmp_path)
    prices, _ = read_prices(raw / "prices.csv", clock)
    path = tmp_path / "audited.parquet"
    pq.write_table(pa.Table.from_pandas(prices, preserve_index=False), path)
    audited = dict(
        path=str(path),
        sha256=sha256(path),
        rows=len(prices),
        source_sha256=asset["hashes"]["prices.csv"],
    )

    def do_not_parse_again(*args, **kwargs):
        raise AssertionError("La preparación no reutilizó los precios auditados")

    monkeypatch.setattr("mars_titan.data.cohort_preparation.read_prices", do_not_parse_again)
    report = prepare_cohort_asset(
        raw,
        tmp_path / "prepared",
        asset,
        clock,
        cohort="original_audited",
        reviews={},
        input_policy=HISTORICAL,
        audited_prices=audited,
    )
    assert report["counts"]["prices"] == 68
    assert report["price_audit"]["mode"] == "audited_parquet"
    assert report["policy"]["audited_prices"] == audited
    path.write_bytes(b"Archivo cambiado")
    with pytest.raises(ValueError, match="huella|auditad"):
        prepare_cohort_asset(
            raw,
            tmp_path / "other",
            asset,
            clock,
            cohort="original_audited",
            reviews={},
            input_policy=HISTORICAL,
            audited_prices=audited,
        )


def test_policy_and_cutoff_cannot_be_changed_in_existing_preparation(tmp_path):
    raw, asset, clock = prices_only(tmp_path)
    with pytest.raises(ValueError, match="reserva|2024"):
        prepare_cohort_asset(
            raw,
            tmp_path / "future",
            asset,
            clock,
            cohort="original_audited",
            reviews={},
            input_policy=HISTORICAL,
            cutoff="2024-01-01",
        )
    with pytest.raises(ValueError, match="política"):
        prepare_cohort_asset(
            raw,
            tmp_path / "bad",
            asset,
            clock,
            cohort="original_audited",
            reviews={},
            input_policy="invented",
        )
    with pytest.raises(ValueError, match="modalidad"):
        prepare_cohort_asset(
            raw, tmp_path / "strict", asset, clock, cohort="original_audited", reviews={}
        )


def test_historical_census_prepares_missing_optional_sources_and_keeps_failures(tmp_path):
    from mars_titan.data.corpus_preparation import prepare_cohort
    from tests.data.test_corpus_preparation import setup

    raw, inventory, reviews = setup(tmp_path)
    output = tmp_path / "historical"
    report = prepare_cohort(
        raw,
        inventory,
        reviews,
        output,
        cohort="original_audited",
        markets=("US",),
        input_policy=HISTORICAL,
    )
    assert report["schema_version"] == 2
    assert report["input_policy"] == HISTORICAL
    assert report["status"] == "completed"
    assert [row["state"] for row in report["assets"]] == ["prepared"] * 3
    assert report["assets"][-1]["counts"]["fundamentals"] == 0
    with pytest.raises(ValueError, match="configuración|edición"):
        prepare_cohort(raw, inventory, reviews, output, cohort="original_audited", markets=("US",))


def test_historical_encoding_requires_catalog_when_macro_file_is_absent(tmp_path):
    from mars_titan.data.corpus_encoding import encode_corpus
    from mars_titan.training.cohort_contract import cohort_identity

    source, clock, prepared = prepare_missing(tmp_path)
    parent = tmp_path / "preparation.json"
    parent.write_text(
        json.dumps(
            dict(
                schema_version=2,
                kind="prepared_cohort",
                status="completed",
                cohort_id="original_audited",
                input_policy=HISTORICAL,
                mask_contract=prepared["mask_contract"],
                candidate_count=1,
                failed_assets=0,
                prepared_root=str(source.parents[1]),
                assets=[
                    dict(
                        market="US",
                        symbol="A",
                        state="prepared",
                        manifest_sha256=sha256(source / "manifest.json"),
                    )
                ],
            )
        )
    )
    kwargs = dict(
        macros={},
        encoders=Encoders(),
        clocks={"US": clock},
        context=64,
        company_factors=False,
        fundamental_concepts=CONCEPTS,
        input_policy=HISTORICAL,
    )
    with pytest.raises(ValueError, match="catálogo"):
        encode_corpus(parent, tmp_path / "no-catalog", **kwargs)
    report = encode_corpus(parent, tmp_path / "encoded", macro_indicators=["a", "b"], **kwargs)
    assert report["schema_version"] == 3
    assert report["input_policy"] == HISTORICAL
    assert report["samples"] == 5
    assert report["cohort_complete"] is True
    assert report["training_ready"] is False
    with pytest.raises(ValueError, match="cohorte"):
        cohort_identity(report)


def test_catalog_and_masks_are_part_of_effective_recovery_identity(tmp_path):
    source, clock, _ = prepare_missing(tmp_path)
    cutoff = clock.decisions[65]
    path = tmp_path / "macro.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [dict(prediction_at=cutoff, indicator_id="a", value=0.0, available_at=cutoff)]
        ),
        path,
    )
    first = MacroVectors(path, indicators=["a", "b"], input_policy=HISTORICAL)
    second = MacroVectors(path, indicators=["a", "b", "c"], input_policy=HISTORICAL)
    report, table = encode_missing(source, tmp_path / "encoded", clock, macro=first)
    row = next(r for r in table.to_pylist() if r["prediction_at"] == cutoff)
    assert row["presence"][-1] is True
    assert row["macro"] == [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    assert report["input_policy"] == HISTORICAL
    with pytest.raises(ValueError, match="configuración|edición"):
        encode_missing(source, tmp_path / "encoded", clock, macro=second)


def test_optional_source_corruption_is_a_failure_not_an_absence(tmp_path):
    from mars_titan.data.corpus_preparation import prepare_cohort
    from tests.data.test_corpus_preparation import setup

    raw, inventory, reviews = setup(tmp_path)
    (raw / "text/sp500_news/AAA.jsonl").write_text("Archivo alterado")
    report = prepare_cohort(
        raw,
        inventory,
        reviews,
        tmp_path / "historical",
        cohort="original_audited",
        markets=("US",),
        input_policy=HISTORICAL,
    )
    assert report["status"] == "completed_with_errors"
    assert report["failed_assets"] == 1
    assert report["assets"][0]["state"] == "failed"
    assert report["assets"][-1]["state"] == "prepared"


def test_masked_receipt_cannot_change_mask_order(tmp_path):
    source, clock, report = prepare_missing(tmp_path)
    path = source / "manifest.json"
    report["mask_contract"]["order"].reverse()
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="preparación|política|cohorte"):
        encode_missing(source, tmp_path / "encoded", clock)


def test_corpus_uses_the_audited_price_catalog_without_reparsing_sources(tmp_path, monkeypatch):
    from mars_titan.data.corpus_catalog import corpus_candidates
    from mars_titan.data.corpus_preparation import prepare_cohort
    from tests.data.test_corpus_preparation import setup

    raw, inventory, reviews = setup(tmp_path)
    clock = MarketClock("US", "1990-01-01", "2026-01-01")
    audit_root, files = tmp_path / "price-audit", {}
    for asset in corpus_candidates(raw, inventory, "US"):
        relative = asset["paths"]["prices"][0]
        frame, _ = read_prices(raw / relative, clock)
        path = audit_root / "US" / asset["symbol"] / "prices.parquet"
        path.parent.mkdir(parents=True)
        pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), path)
        files[relative] = dict(
            market="US",
            symbol=asset["symbol"],
            source_sha256=asset["hashes"][relative],
            artifacts={"prices.parquet": dict(rows=len(frame), sha256=sha256(path))},
        )
    state = tmp_path / "price-state.json"
    state.write_text(json.dumps(dict(details_root=str(audit_root), files=files)))

    def no_reparse(*args, **kwargs):
        raise AssertionError("Se releen fuentes que ya tienen preparación auditada")

    monkeypatch.setattr("mars_titan.data.cohort_preparation.read_prices", no_reparse)
    result = prepare_cohort(
        raw,
        inventory,
        reviews,
        tmp_path / "history",
        cohort="original_audited",
        markets=("US",),
        input_policy=HISTORICAL,
        price_audit_state=state,
    )
    assert result["failed_assets"] == 0
    assert result["configuration"]["price_audit_sha256"] == sha256(state)
    for asset in result["assets"]:
        receipt = json.loads(
            (tmp_path / "history/prepared/US" / asset["symbol"] / "manifest.json").read_text()
        )
        assert receipt["price_audit"]["mode"] == "audited_parquet"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_numeric_values_are_errors_even_with_declared_absence(value):
    from mars_titan.data.input_policy import numeric_observations

    cutoff = MarketClock("US", "2023-01-01", "2024-01-01").decisions[0]
    with pytest.raises(ValueError, match="finito"):
        numeric_observations(
            [dict(value=value, available_at=cutoff, missing_reason="bad_source")], cutoff
        )


def test_declared_absence_retains_its_specific_diagnostic():
    from mars_titan.data.input_policy import numeric_observations

    cutoff = MarketClock("US", "2023-01-01", "2024-01-01").decisions[0]
    result = numeric_observations(
        [dict(value=None, available_at=None, missing_reason="missing_rate_inputs")], cutoff
    )
    assert result == ([None], [0.0], None, ["missing_rate_inputs"])


def test_audited_price_file_budget_is_checked_before_hashing(tmp_path, monkeypatch):
    from mars_titan.data import audited_prices

    path = tmp_path / "large.parquet"
    path.write_bytes(b"x" * 65)
    monkeypatch.setattr(audited_prices, "MAX_PRICE_BYTES", 64)

    def no_hash(*args):
        pytest.fail("Se calculó la huella antes de comprobar el presupuesto")

    monkeypatch.setattr(audited_prices, "sha256", no_hash)
    record = dict(path=str(path), sha256="a" * 64, rows=1, source_sha256="b" * 64)
    with pytest.raises(ValueError, match="presupuesto|MiB"):
        audited_prices.read_audited_prices(record, "b" * 64, None, "2023-12-31")


def test_audited_prices_reject_intermediate_symlink_before_hashing(tmp_path, monkeypatch):
    from mars_titan.data import audited_prices

    folder = tmp_path / "real"
    folder.mkdir()
    (folder / "prices.parquet").write_bytes(b"x")
    (tmp_path / "alias").symlink_to(folder, target_is_directory=True)

    def no_hash(*args):
        pytest.fail("Se siguió un enlace antes de validar la ruta")

    monkeypatch.setattr(audited_prices, "sha256", no_hash)
    record = dict(
        path=str(tmp_path / "alias/prices.parquet"), sha256="a" * 64, rows=1, source_sha256="b" * 64
    )
    with pytest.raises(ValueError, match="enlace"):
        audited_prices.read_audited_prices(record, "b" * 64, None, "2023-12-31")


def test_audited_session_dictionary_is_validated_before_expansion(tmp_path, monkeypatch):
    from mars_titan.data import audited_prices

    path = tmp_path / "dictionary.parquet"
    count = 64
    table = pa.table(
        {
            **{name: [1.0] * count for name in audited_prices.COLUMNS[:5]},
            "session": pa.DictionaryArray.from_arrays(
                pa.array([0] * count), pa.array(["x" * 10000])
            ),
            "available_at": pa.array([None] * count, type=pa.timestamp("us", tz="UTC")),
        }
    )
    pq.write_table(table, path)
    original = pq.ParquetFile

    class Guarded:
        def __init__(self, *args, **kwargs):
            assert kwargs.get("read_dictionary") == ["session"]
            self.inner = original(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(self.inner, name)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.inner.close()

        def read(self, *args, **kwargs):
            pytest.fail("Se materializó la columna de sesiones completa")

    monkeypatch.setattr(audited_prices.pq, "ParquetFile", Guarded)
    record = dict(path=str(path), sha256=sha256(path), rows=count, source_sha256="b" * 64)
    with pytest.raises(ValueError, match="longitud|sesión|sesiones"):
        audited_prices.read_audited_prices(record, "b" * 64, None, "2023-12-31")
