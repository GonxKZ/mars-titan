"""Composición por bloques temporales sin duplicar ni completar cifras ausentes."""

import csv
import importlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mars_titan.data.temporal import MarketClock


@pytest.fixture
def panels(tmp_path):
    root = tmp_path / "sources"
    root.mkdir()
    with Path("data/catalogs/macro-indicators.csv").open() as stream:
        entries = [r for r in csv.DictReader(stream) if r["id"] in {"us_cpi", "us_unemployment"}]
    assert len(entries) == 2
    catalog = root / "catalog.csv"
    with catalog.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=entries[0])
        writer.writeheader()
        writer.writerows(entries)
    clock = MarketClock("US", "2023-01-30", "2023-02-03")
    rows = []
    for moment in clock.decisions:
        for entry in entries:
            rows.append(
                dict(
                    prediction_at=moment,
                    indicator_id=entry["id"],
                    value=10.0,
                    available_at=moment,
                    period_start="2022-12-01",
                    missing_reason=None,
                    source_hashes=["a" * 64],
                    unit="test_published_unit",
                    seasonal_adjustment="published",
                )
            )
    table = pa.Table.from_pylist(rows)
    table = table.set_column(
        table.schema.get_field_index("missing_reason"),
        "missing_reason",
        pa.array([None] * len(rows), type=pa.string()),
    )
    base = root / "base.parquet"
    pq.write_table(table, base, row_group_size=3)
    replacement = root / "replacement.parquet"
    added = [
        dict(r, value=0.0, source_hashes=["b" * 64]) for r in rows if r["indicator_id"] == "us_cpi"
    ]
    pq.write_table(pa.Table.from_pylist(added, schema=table.schema), replacement, row_group_size=2)
    return base, [(replacement, ["us_cpi"])], catalog, rows


def compose(panels, destination):
    api = importlib.import_module("mars_titan.data.macro_edition")
    return api.compose_macro_edition(
        *panels[:3], destination, market="US", start="2023-01-30", end="2023-02-03"
    )


def test_replace_only_declared_concepts_with_zero_preserved_and_sorted_output(panels, tmp_path):
    report = compose(panels, tmp_path / "edition")
    rows = pq.read_table(tmp_path / "edition/macro.parquet").to_pylist()
    assert len(rows) == 10
    assert [(r["prediction_at"], r["indicator_id"]) for r in rows] == sorted(
        (r["prediction_at"], r["indicator_id"]) for r in rows
    )
    assert all(r["value"] == (0.0 if r["indicator_id"] == "us_cpi" else 10.0) for r in rows)
    assert report["replaced_indicators"] == ["us_cpi"]
    assert report["source_hashes"] != []
    assert pq.read_table(panels[0])["value"].to_pylist() == [10.0] * 10
    with pytest.raises(FileExistsError):
        compose(panels, tmp_path / "edition")


@pytest.mark.parametrize(
    "defect", ["duplicate", "missing", "wrong_indicator", "unsorted", "overlap"]
)
def test_invalid_replacement_never_publishes_a_partial_edition(panels, tmp_path, defect):
    base, additions, catalog, _ = panels
    path, names = additions[0]
    table = pq.read_table(path)
    rows = table.to_pylist()
    if defect == "duplicate":
        rows.insert(0, rows[0])
    elif defect == "missing":
        rows.pop(2)
    elif defect == "wrong_indicator":
        rows[0]["indicator_id"] = "us_unemployment"
    elif defect == "unsorted":
        rows.reverse()
    else:
        additions.append(additions[0])
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), path, row_group_size=2)
    with pytest.raises(
        ValueError, match={"duplicate": "duplicado", "overlap": "solapan"}.get(defect)
    ):
        compose((base, additions, catalog), tmp_path / "edition")
    assert not (tmp_path / "edition").exists()


def test_real_absence_stays_missing_and_final_year_is_not_materialized(panels, tmp_path):
    base, additions, catalog, _ = panels
    path, names = additions[0]
    table = pq.read_table(path)
    rows = table.to_pylist()
    rows[1].update(value=None, missing_reason="not_yet_available", available_at=None)
    rows.append(
        dict(
            rows[-1],
            prediction_at=MarketClock("US", "2024-01-01", "2024-01-03").decisions[0],
            value=999.0,
        )
    )
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), path, row_group_size=2)
    compose(panels, tmp_path / "edition")
    out = pq.read_table(tmp_path / "edition/macro.parquet").to_pylist()
    assert sum(r["value"] is None for r in out) == 1
    assert all(r["prediction_at"].year == 2023 for r in out)


def test_corrupt_parquet_footer_is_rejected_before_decoding(panels, tmp_path):
    path = panels[1][0][0]
    path.write_bytes(b"PAR1" + b"data" + (1024**3).to_bytes(4, "little") + b"PAR1")
    with pytest.raises(ValueError, match="metadatos"):
        compose(panels, tmp_path / "edition")
    assert not (tmp_path / "edition").exists()


def test_indicators_can_arrive_in_any_order_within_each_session(panels, tmp_path):
    base, additions, catalog, _ = panels
    table = pq.read_table(base)
    order = [i for start in range(0, len(table), 2) for i in (start + 1, start)]
    pq.write_table(table.take(pa.array(order)), base, row_group_size=3)
    compose(panels, tmp_path / "edition")
    out = pq.read_table(tmp_path / "edition/macro.parquet").to_pylist()
    assert [r["value"] for r in out] == [0.0, 10.0] * 5


def test_cli_preserves_explicit_source_ownership(panels, tmp_path, capsys):
    api = importlib.import_module("mars_titan.data.macro_edition")
    recipe = tmp_path / "replacements.json"
    recipe.write_text(
        json.dumps([{"path": str(panels[1][0][0]), "indicator_ids": panels[1][0][1]}])
    )
    assert (
        api.main(
            [
                "--base",
                str(panels[0]),
                "--catalog",
                str(panels[2]),
                "--replacements",
                str(recipe),
                "--output",
                str(tmp_path / "edition"),
                "--market",
                "US",
                "--start",
                "2023-01-30",
                "--end",
                "2023-02-03",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["admission_required"] is True
