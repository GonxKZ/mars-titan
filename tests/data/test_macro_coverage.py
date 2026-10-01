"""Admisión estricta de decisiones macro con Parquet y catálogo reales."""

import csv
import importlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mars_titan.data.storage import sha256
from mars_titan.data.temporal import MarketClock

SCHEMA = pa.schema(
    [
        ("prediction_at", pa.timestamp("us", tz="UTC")),
        ("indicator_id", pa.string()),
        ("value", pa.float64()),
        ("available_at", pa.timestamp("us", tz="UTC")),
        ("period_start", pa.string()),
        ("missing_reason", pa.string()),
        ("source_hashes", pa.list_(pa.string())),
        ("unit", pa.string()),
        ("seasonal_adjustment", pa.string()),
    ]
)


def assess(source, catalog, output, **kwargs):
    assert importlib.util.find_spec("mars_titan.data.macro_coverage"), "Falta la puerta macro"
    return importlib.import_module("mars_titan.data.macro_coverage").assess_macro_completeness(
        source,
        catalog,
        output,
        **{"start": "2023-01-03", "end": "2023-01-05", "market": "US", **kwargs},
    )


@pytest.fixture
def inputs(tmp_path):
    folder = tmp_path / "source"
    folder.mkdir()
    catalog = folder / "catalog.csv"
    with Path("data/catalogs/macro-indicators.csv").open(encoding="utf-8") as stream:
        entries = [r for r in csv.DictReader(stream) if r["id"] in {"us_cpi", "us_cpi_mom"}]
    with catalog.open("w", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=entries[0])
        writer.writeheader()
        writer.writerows(entries)
    return folder / "macro.parquet", catalog, tmp_path / "admission"


def decision_rows(day, **overrides):
    moment = MarketClock("US", "2023-01-01", "2024-01-05").decision(day)
    rows = []
    for identifier in ("us_cpi", "us_cpi_mom"):
        rows.append(
            {
                "prediction_at": moment,
                "indicator_id": identifier,
                "value": 0.0 if identifier == "us_cpi_mom" else 100.0,
                "available_at": moment,
                "period_start": "2022-12-01",
                "missing_reason": None,
                "source_hashes": ["a" * 64],
                "unit": "percent_change" if identifier == "us_cpi_mom" else "Index 2000=100",
                "seasonal_adjustment": None
                if identifier == "us_cpi_mom"
                else "Seasonally Adjusted",
                **overrides,
            }
        )
    return rows


def write_panel(path, rows, **kwargs):
    pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), path, **kwargs)


def test_missing_value_with_present_mask_is_excluded_but_zero_is_admitted(inputs):
    source, catalog, output = inputs
    rows = decision_rows("2023-01-03") + decision_rows("2023-01-04")
    rows[-1].update(value=None, missing_reason="missing_period")
    table = pa.Table.from_pylist(rows, schema=SCHEMA).append_column("mask", pa.array([1] * 4))
    pq.write_table(table, source, row_group_size=3)
    before = sha256(source)

    report = assess(*inputs)

    assert report["total_decisions"] == 3
    assert report["complete_decisions"] == 1
    assert report["excluded_decisions"] == 2
    assert report["population_ready"] is True
    assert report["missing_by_indicator"] == {"us_cpi": 1, "us_cpi_mom": 2}
    assert report["zero_value_rows"] == 1
    assert report["source_sha256"] == before == sha256(source)
    assert report["catalog_sha256"] == sha256(catalog)
    assert report["complete_decisions_sha256"] == sha256(output / "complete-decisions.parquet")
    assert json.loads((output / "report.json").read_text()) == report
    assert pq.read_table(output / "complete-decisions.parquet").to_pylist() == [
        {
            "prediction_at": datetime(2023, 1, 3, 21, 5, tzinfo=UTC),
            "macro_available_at": datetime(2023, 1, 3, 21, 5, tzinfo=UTC),
            "indicator_count": 2,
        }
    ]


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"value": float("nan")}, "nonfinite_value"),
        ({"value": float("inf")}, "nonfinite_value"),
        ({"available_at": None}, "missing_availability"),
        ({"available_at": datetime(2023, 1, 4, tzinfo=UTC)}, "future_availability"),
        ({"missing_reason": "source_excluded:no_vintages_exclude"}, "declared_missing"),
        ({"source_hashes": []}, "invalid_provenance"),
        ({"source_hashes": ["bad"]}, "invalid_provenance"),
        ({"source_hashes": [None]}, "invalid_provenance"),
        ({"unit": " "}, "invalid_unit"),
        ({"seasonal_adjustment": None}, "missing_historical_adjustment"),
        ({"period_start": "2023-01-05"}, "invalid_period"),
        ({"period_start": "2022-15-01"}, "invalid_period"),
        ({"period_start": "0000-01-01"}, "invalid_period"),
    ],
)
def test_invalid_observation_excludes_the_whole_decision(inputs, changes, reason):
    source, _, output = inputs
    rows = decision_rows("2023-01-03")
    rows[0].update(changes)
    write_panel(source, rows)

    report = assess(*inputs, end="2023-01-03")

    assert report["complete_decisions"] == 0
    assert report["population_ready"] is False
    assert report["missing_by_indicator"] == {"us_cpi": 1, "us_cpi_mom": 0}
    assert report["invalid_rows_by_reason"][reason] == 1
    assert pq.read_table(output / "complete-decisions.parquet").num_rows == 0


def test_derived_unit_must_match_catalog_but_raw_unit_preserves_history(inputs):
    source, _, _ = inputs
    rows = decision_rows("2023-01-03")
    rows[1]["unit"] = "index_points"
    write_panel(source, rows)
    report = assess(*inputs, end="2023-01-03")
    assert report["complete_decisions"] == 0
    assert report["invalid_rows_by_reason"]["invalid_unit"] == 1
    assert report["missing_by_indicator"] == {"us_cpi": 0, "us_cpi_mom": 1}


def test_duplicate_across_row_groups_cannot_replace_missing_indicator(inputs):
    source, _, _ = inputs
    rows = decision_rows("2023-01-03")
    write_panel(source, rows + [rows[0]], row_group_size=2)
    report = assess(*inputs, end="2023-01-03")
    assert report["complete_decisions"] == 0
    assert report["duplicate_indicator_decisions"] == 1


def test_unknown_indicator_aborts_without_publishing_partial_report(inputs):
    source, _, output = inputs
    rows = decision_rows("2023-01-03") + decision_rows("2023-01-04")
    rows[-1]["indicator_id"] = "unknown"
    write_panel(source, rows, row_group_size=2)
    with pytest.raises(ValueError, match="catálogo"):
        assess(*inputs)
    assert not output.exists()


def test_non_session_decision_aborts(inputs):
    source, _, output = inputs
    rows = decision_rows("2023-01-03")
    rows[0]["prediction_at"] += timedelta(minutes=1)
    write_panel(source, rows)
    with pytest.raises(ValueError, match="sesión|reloj"):
        assess(*inputs)
    assert not output.exists()


@pytest.mark.parametrize("field", ["prediction_at", "available_at"])
def test_timezone_naive_schema_is_rejected(inputs, field):
    source, _, output = inputs
    schema = SCHEMA.set(SCHEMA.get_field_index(field), pa.field(field, pa.timestamp("us")))
    pq.write_table(pa.Table.from_pylist(decision_rows("2023-01-03"), schema=schema), source)
    with pytest.raises(ValueError, match="UTC"):
        assess(*inputs)
    assert not output.exists()


def test_output_cannot_replace_or_reside_among_sources(inputs):
    source, catalog, _ = inputs
    write_panel(source, decision_rows("2023-01-03"))
    before = sha256(source)
    with pytest.raises(ValueError, match="fuera"):
        assess(source, catalog, source.parent / "results")
    assert sha256(source) == before


def test_existing_output_is_never_overwritten(inputs):
    source, _, output = inputs
    write_panel(source, decision_rows("2023-01-03"))
    output.mkdir()
    (output / "report.json").write_text("previous")
    with pytest.raises(FileExistsError):
        assess(*inputs)
    assert (output / "report.json").read_text() == "previous"


def test_corrupt_parquet_never_publishes_a_report(inputs):
    source, _, output = inputs
    source.write_bytes(b"not a parquet file")
    with pytest.raises(ValueError):
        assess(*inputs)
    assert not output.exists()


def test_invalid_and_excessive_periods_are_rejected(inputs):
    source, _, output = inputs
    write_panel(source, decision_rows("2023-01-03"))
    for kwargs in (
        {"start": "2023-01-06"},
        {"start": "1900-01-01"},
        {"start": "2023-01-03T00:00:00"},
        {"market": "XX"},
    ):
        with pytest.raises(ValueError):
            assess(*inputs, **kwargs)
    assert not output.exists()


def test_shared_final_row_group_does_not_materialize_test_values(inputs, monkeypatch):
    source, _, output = inputs
    rows = decision_rows("2023-12-29") + decision_rows("2024-01-02", indicator_id="sealed")
    write_panel(source, rows)
    cutoff = datetime(2024, 1, 1, tzinfo=UTC)
    actual_batches = pq.ParquetFile.iter_batches
    decoded = []

    def guarded_batches(file, *args, **kwargs):
        for batch in actual_batches(file, *args, **kwargs):
            if "value" in batch.schema.names:
                assert all(moment < cutoff for moment in batch["prediction_at"].to_pylist())
                decoded.append(batch.num_rows)
            yield batch

    monkeypatch.setattr(pq.ParquetFile, "iter_batches", guarded_batches)
    report = assess(*inputs, start="2023-12-29", end="2023-12-31")
    assert report["complete_decisions"] == 1
    assert report["selected_rows"] == 2
    assert decoded == [2]
    assert pq.read_table(output / "complete-decisions.parquet").num_rows == 1


def test_interleaved_test_rows_abort_before_value_decoding(inputs, monkeypatch):
    source, _, output = inputs
    rows = decision_rows("2023-12-29")
    rows.insert(1, decision_rows("2024-01-02")[0])
    write_panel(source, rows)

    def forbidden_values(*args, **kwargs):
        pytest.fail("Se intentaron materializar valores que cruzan el corte")

    monkeypatch.setattr(pq.ParquetFile, "iter_batches", forbidden_values)
    with pytest.raises(ValueError, match="orden"):
        assess(*inputs, start="2023-12-29", end="2023-12-31")
    assert not output.exists()


def test_catalog_mutation_during_read_prevents_publication(inputs, monkeypatch):
    source, catalog, output = inputs
    write_panel(source, decision_rows("2023-01-03"))
    actual_batches = pq.ParquetFile.iter_batches

    def changed_catalog(*args, **kwargs):
        for batch in actual_batches(*args, **kwargs):
            with catalog.open("a") as stream:
                stream.write("\n")
            yield batch

    monkeypatch.setattr(pq.ParquetFile, "iter_batches", changed_catalog)
    with pytest.raises(ValueError, match="cambió"):
        assess(*inputs)
    assert not output.exists()


def test_group_budget_is_enforced_before_values_are_read(inputs, monkeypatch):
    source, _, output = inputs
    write_panel(source, decision_rows("2023-01-03") + decision_rows("2023-01-04"))
    from mars_titan.data import macro_coverage

    monkeypatch.setattr(macro_coverage, "_MAX_GROUP_ROWS", 3)
    with pytest.raises(ValueError, match="presupuesto"):
        assess(*inputs)
    assert not output.exists()


def test_empty_calendar_period_is_not_population_ready(inputs):
    source, _, output = inputs
    write_panel(source, decision_rows("2023-01-03"))
    report = assess(*inputs, start="2023-01-07", end="2023-01-08")
    assert report["total_decisions"] == 0
    assert report["population_ready"] is False
    assert pq.read_table(output / "complete-decisions.parquet").num_rows == 0


def test_pre_epoch_availability_is_not_replaced_with_epoch(inputs):
    source, _, output = inputs
    decision = MarketClock("US", "1960-01-01", "1960-01-07").decision("1960-01-04")
    rows = decision_rows("2023-01-03", prediction_at=decision, available_at=decision)
    for row in rows:
        row["period_start"] = "1959-12-01"
    write_panel(source, rows)
    report = assess(*inputs, start="1960-01-04", end="1960-01-04")
    assert report["complete_decisions"] == 1
    assert (
        pq.read_table(output / "complete-decisions.parquet")["macro_available_at"][0].as_py()
        == decision
    )


@pytest.mark.parametrize(
    "problem", ["empty", "duplicate", "missing_field", "empty_id", "policy", "formula"]
)
def test_malformed_catalog_aborts_before_publication(inputs, problem):
    source, catalog, output = inputs
    write_panel(source, decision_rows("2023-01-03"))
    with catalog.open() as stream:
        rows = list(csv.DictReader(stream))
    fields = list(rows[0])
    if problem == "empty":
        rows = []
    elif problem == "duplicate":
        rows.append(rows[0])
    elif problem == "missing_field":
        fields.remove("unit")
    elif problem == "empty_id":
        rows[0]["id"] = " "
    elif problem == "policy":
        rows[1]["vintage_policy"] = "NO_VINTAGES_EXCLUDE"
    elif problem == "formula":
        rows[1]["formula"] = "unknown[p]"
    with catalog.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError):
        assess(*inputs)
    assert not output.exists()


@pytest.mark.parametrize(
    "problem", ["missing_column", "string_value", "string_hashes", "integer_unit"]
)
def test_incompatible_panel_schema_aborts(inputs, problem):
    source, _, output = inputs
    table = pa.Table.from_pylist(decision_rows("2023-01-03"), schema=SCHEMA)
    if problem == "missing_column":
        table = table.drop(["missing_reason"])
    else:
        name, values = {
            "string_value": ("value", ["100", "0"]),
            "string_hashes": ("source_hashes", ["a" * 64] * 2),
            "integer_unit": ("unit", [1, 1]),
        }[problem]
        table = table.set_column(table.schema.get_field_index(name), name, pa.array(values))
    pq.write_table(table, source)
    with pytest.raises(ValueError):
        assess(*inputs)
    assert not output.exists()


def test_null_decision_is_not_silently_dropped(inputs):
    source, _, output = inputs
    rows = decision_rows("2023-01-03")
    rows[0]["prediction_at"] = None
    write_panel(source, rows)
    with pytest.raises(ValueError, match="nula"):
        assess(*inputs)
    assert not output.exists()


def test_decoded_dictionary_expansion_has_a_memory_budget(inputs, monkeypatch):
    source, _, output = inputs
    write_panel(source, decision_rows("2023-01-03") * 100)
    from mars_titan.data import macro_coverage

    with pq.ParquetFile(source) as file:
        stored_bytes = file.metadata.row_group(0).total_byte_size
    monkeypatch.setattr(macro_coverage, "_MAX_GROUP_BYTES", stored_bytes + 1)
    with pytest.raises(ValueError, match="presupuesto"):
        assess(*inputs)
    assert not output.exists()


def test_missing_statistics_still_select_dates_before_reading_values(inputs, monkeypatch):
    source, _, _ = inputs
    write_panel(source, decision_rows("2024-01-02"), write_statistics=False)

    def forbidden_values(*args, **kwargs):
        pytest.fail("Se intentaron leer valores fuera del periodo")

    monkeypatch.setattr(pq.ParquetFile, "iter_batches", forbidden_values)
    report = assess(*inputs)
    assert report["total_decisions"] == 3
    assert report["selected_rows"] == 0
    assert report["population_ready"] is False


def test_oversized_catalog_aborts(inputs):
    source, catalog, output = inputs
    write_panel(source, decision_rows("2023-01-03"))
    with catalog.open("a") as stream:
        stream.write("x" * (2 * 1024**2))
    with pytest.raises(ValueError, match="presupuesto"):
        assess(*inputs)
    assert not output.exists()


def test_symlinked_inputs_cannot_publish_inside_real_source(inputs, tmp_path):
    source, catalog, _ = inputs
    write_panel(source, decision_rows("2023-01-03"))
    links = tmp_path / "links"
    links.mkdir()
    linked_source, linked_catalog = links / "panel.parquet", links / "catalog.csv"
    linked_source.symlink_to(source)
    linked_catalog.symlink_to(catalog)
    output = source.parent / "admission"
    with pytest.raises(ValueError, match="fuera"):
        assess(linked_source, linked_catalog, output)
    assert not output.exists()


def test_concurrent_empty_output_is_not_replaced(inputs, monkeypatch):
    source, _, output = inputs
    write_panel(source, decision_rows("2023-01-03"))
    from mars_titan.data import macro_coverage

    actual_json = macro_coverage.atomic_json

    def competing_output(*args, **kwargs):
        actual_json(*args, **kwargs)
        output.mkdir()

    monkeypatch.setattr(macro_coverage, "atomic_json", competing_output)
    with pytest.raises(FileExistsError):
        assess(*inputs)
    assert output.is_dir()
    assert list(output.iterdir()) == []


def test_nanosecond_availability_is_preserved(inputs):
    source, _, output = inputs
    table = pa.Table.from_pylist(decision_rows("2023-01-03"), schema=SCHEMA)
    timestamp = int(datetime(2023, 1, 3, 21, 5, tzinfo=UTC).timestamp()) * 1_000_000_000 - 1
    availability = pa.array([timestamp, timestamp], type=pa.timestamp("ns", tz="UTC"))
    table = table.set_column(
        table.schema.get_field_index("available_at"), "available_at", availability
    )
    pq.write_table(table, source)
    report = assess(*inputs, end="2023-01-03")
    assert report["complete_decisions"] == 1
    actual = pq.read_table(output / "complete-decisions.parquet")["macro_available_at"]
    assert actual.cast(pa.int64()).to_pylist() == [timestamp]
    assert actual.type == pa.timestamp("ns", tz="UTC")


def test_extra_nested_column_does_not_shift_temporal_statistics(inputs):
    source, _, output = inputs
    table = pa.Table.from_pylist(decision_rows("2023-01-03"), schema=SCHEMA)
    table = table.add_column(0, "extra", pa.array([{"a": "x", "b": "y"}] * 2))
    pq.write_table(table, source)
    report = assess(*inputs, end="2023-01-03")
    assert report["complete_decisions"] == 1
    assert pq.read_table(output / "complete-decisions.parquet").num_rows == 1


@pytest.mark.parametrize("width", [1000, 20_000])
def test_dictionary_values_are_bounded_before_dense_expansion(inputs, monkeypatch, width):
    source, _, output = inputs
    rows = decision_rows("2023-01-03", missing_reason="x" * width) * 500
    write_panel(source, rows)
    from mars_titan.data import macro_coverage

    monkeypatch.setattr(macro_coverage, "_MAX_GROUP_BYTES", 65_536)
    actual_batches = pq.ParquetFile.iter_batches

    def bounded_batches(*args, **kwargs):
        for batch in actual_batches(*args, **kwargs):
            assert batch.nbytes <= 65_536
            yield batch

    monkeypatch.setattr(pq.ParquetFile, "iter_batches", bounded_batches)
    with pytest.raises(ValueError, match="presupuesto"):
        assess(*inputs)
    assert not output.exists()


def test_distinct_missing_reasons_cannot_grow_without_bound(inputs, monkeypatch):
    source, _, output = inputs
    rows = decision_rows("2023-01-03") + decision_rows("2023-01-04")
    for index, row in enumerate(rows):
        row.update(value=None, missing_reason=f"absent-{index}")
    write_panel(source, rows, row_group_size=2)
    from mars_titan.data import macro_coverage

    monkeypatch.setattr(macro_coverage, "_MAX_MISSING_REASONS", 3, raising=False)
    with pytest.raises(ValueError, match="motivos"):
        assess(*inputs)
    assert not output.exists()


def test_odd_temporal_prefix_does_not_create_one_arrow_batch_per_row(inputs, monkeypatch):
    source, _, _ = inputs
    rows = (decision_rows("2023-01-03") * 1024)[:-1] + decision_rows("2024-01-02")[:1]
    write_panel(source, rows)
    actual_batches = pq.ParquetFile.iter_batches
    decoded = []

    def counted_batches(*args, **kwargs):
        for batch in actual_batches(*args, **kwargs):
            decoded.append(batch.num_rows)
            yield batch

    monkeypatch.setattr(pq.ParquetFile, "iter_batches", counted_batches)
    report = assess(*inputs, end="2023-01-03")
    assert report["selected_rows"] == 2047
    assert report["decoded_row_groups"] == 1
    assert len(decoded) <= 4
