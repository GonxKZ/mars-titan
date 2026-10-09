"""Edición sin ajustar sobre una población preparada mínima con huellas reales."""

import hashlib
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mars_titan.data import unadjusted_edition as edition
from mars_titan.data.unadjusted_edition import (
    _segments,
    build_edition,
    population_assets,
    reconstruct_asset,
    select_audit_sample,
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_asset(root, dataset, market, symbol, *, dividend_at=150, broken=False):
    """CSV del proveedor con precios en céntimos, un dividendo y un split, y su preparación."""
    rows = 260
    sessions = np.datetime64("2023-01-02") + np.arange(rows) * np.timedelta64(1, "D")
    rng = np.random.default_rng(len(symbol))
    raw = (2000 + np.cumsum(rng.integers(-20, 21, rows))) / 100.0
    dividends, splits = np.zeros(rows), np.zeros(rows)
    dividends[dividend_at] = 0.4
    splits[60] = 2.0
    later = np.ones(rows)
    for t in range(rows - 2, -1, -1):
        later[t] = later[t + 1] * (splits[t + 1] or 1.0)
    adjusted_split = raw / later
    published = dividends / later
    factor = np.ones(rows)
    for t in range(rows - 1, 0, -1):
        step = 1 - published[t] / adjusted_split[t - 1] if published[t] else 1.0
        factor[t - 1] = factor[t] * step
    adjusted = adjusted_split * factor
    if broken:
        adjusted[:100] *= 1.003  # Ajuste no publicado del proveedor antes de la sesión 100.
    folder = "S&P500_time_series" if market == "US" else "HS300_time_series"
    csv = dataset / "time_series" / folder / f"{symbol.lower()}.csv"
    csv.parent.mkdir(parents=True, exist_ok=True)
    lines = ["Date,Open,High,Low,Close,Volume,Dividends,Stock Splits"]
    for day, value, d, s in zip(sessions, adjusted, published, splits, strict=True):
        v, d, s = float(value), float(d), float(s)
        lines.append(f"{day} 00:00:00-05:00,{v!r},{v!r},{v!r},{v!r},1000,{d!r},{s!r}")
    csv.write_text("\n".join(lines) + "\n")
    prepared = root / "prepared" / market / symbol
    prepared.mkdir(parents=True)
    keep = np.arange(rows) != 7  # Una fila excluida en la preparación.
    table = pa.table(
        {
            "session": [str(day) for day in sessions[keep]],
            "open": adjusted[keep],
            "high": adjusted[keep],
            "low": adjusted[keep],
            "close": adjusted[keep],
            "volume": np.full(int(keep.sum()), 1000),
        }
    )
    pq.write_table(table, prepared / "prices.parquet")
    source = f"time_series/{folder}/{csv.name}"
    manifest = {
        "sources": {source: digest(csv)},
        "price_audit": {"source_sha256": digest(csv)},
        "artifacts": {"prices.parquet": digest(prepared / "prices.parquet")},
    }
    (prepared / "manifest.json").write_text(json.dumps(manifest))
    return {
        "market": market,
        "symbol": symbol,
        "state": "prepared",
        "counts": {"prices": int(keep.sum())},
    }


@pytest.fixture
def population(tmp_path):
    root, dataset = tmp_path / "prepared-root", tmp_path / "dataset"
    assets = [
        write_asset(root, dataset, "US", "AAA"),
        write_asset(root, dataset, "US", "BBB", broken=True),
        write_asset(root, dataset, "CN", "600000.SS"),
    ]
    assets.append({"market": "US", "symbol": "NOP", "state": "prepared", "counts": {"prices": 0}})
    (root / "manifest.json").write_text(
        json.dumps({"kind": "prepared_cohort", "status": "completed", "assets": assets})
    )
    return root, dataset, tmp_path / "edition"


def test_edition_reconstructs_verifies_and_resumes(population):
    root, dataset, output = population
    manifest = build_edition(root, dataset, output)
    assert manifest["assets"] == 3 and manifest["reused_assets"] == 0
    assert manifest["corporate_actions_complete"] is False
    receipt = json.loads((output / "assets/US/AAA/receipt.json").read_text())
    assert receipt["status"] == "verified" and receipt["rows"] == 259
    prices = pq.read_table(output / "assets/US/AAA/prices.parquet").to_pydict()
    cents = np.round(np.array(prices["close"]) * 100)
    np.testing.assert_allclose(np.array(prices["close"]) * 100, cents, atol=1e-6)
    assert all(prices["verified"])
    assert receipt["dividends_before_fit"] == 1 and receipt["affected"]["rate"] == 1.0
    controls = receipt["controls"]
    assert controls["no_dividends"]["fit"] and controls["no_dividends"]["affected"]["rate"] < 0.2
    for name in ("ex_date_plus_one", "ex_date_minus_one"):
        assert not controls[name]["fit"] or controls[name]["affected"]["rate"] < 0.2
    summary = manifest["summary"]["US"]["controls"]
    assert summary["method_affected_rate"] > 0.5 > summary["no_dividends"]["affected_rate"]
    events = pq.read_table(output / "assets/US/AAA/events.parquet").to_pydict()
    assert events["raw_dividend"] == pytest.approx([0.0, 0.4])
    broken = json.loads((output / "assets/US/BBB/receipt.json").read_text())
    assert broken["status"] == "partial"
    # El tramo entre el split (sesión 60) y el dividendo (150) mezcla filas buenas y malas.
    # La verificación es por tramo, así que empieza en la fecha ex del dividendo.
    assert broken["verified_since"] == "2023-06-01"
    assert broken["failed_segments"][0]["start"] == "2023-03-03"

    again = build_edition(root, dataset, output)
    assert again["reused_assets"] == 3 and again["edition_id"] == manifest["edition_id"]
    (output / "assets/US/AAA/events.parquet").write_bytes(b"altered")
    repaired = build_edition(root, dataset, output)
    assert repaired["reused_assets"] == 2 and repaired["edition_id"] == manifest["edition_id"]


def test_edition_refuses_changed_sources_and_outputs_inside_inputs(population):
    root, dataset, output = population
    with pytest.raises(ValueError, match="fuera"):
        build_edition(root, dataset, dataset / "edition")
    record = next(r for r in population_assets(root) if r["symbol"] == "AAA")
    csv = dataset / record["source"]
    csv.write_text(csv.read_text().replace(",1000,", ",1001,", 1))
    with pytest.raises(ValueError, match="no coincide"):
        reconstruct_asset(record, dataset)


def test_edition_refuses_prepared_prices_that_differ_from_the_original(population):
    root, dataset, _ = population
    record = next(r for r in population_assets(root) if r["symbol"] == "AAA")
    path = root / "prepared/US/AAA/prices.parquet"
    table = pq.read_table(path)
    close = table.column("close").to_numpy().copy()
    close[3] *= 1 + 1e-12
    pq.write_table(table.set_column(4, "close", pa.array(close)), path)
    record = {**record, "prices_sha256": digest(path)}
    with pytest.raises(ValueError, match="no es el original"):
        reconstruct_asset(record, dataset)


def test_segments_stop_at_the_latest_failing_stretch_and_skip_short_failures():
    sessions = np.datetime64("2023-01-02") + np.arange(30) * np.timedelta64(1, "D")
    events = sessions[[10, 20]]
    on_grid = np.ones(30, dtype=bool)
    on_grid[:10] = False
    summary, verified = _segments(sessions, events, on_grid)
    assert [s["rows"] for s in summary] == [10, 10, 10]
    assert verified.tolist() == [False] * 10 + [True] * 20
    on_grid[25] = on_grid[26] = False  # 80 % en el último tramo: falla todo.
    assert not _segments(sessions, events, on_grid)[1].any()
    # Un tramo de tres filas fuera de rejilla no detiene la verificación, pero no se verifica.
    short = np.ones(30, dtype=bool)
    short[20:23] = False
    _, verified = _segments(sessions, sessions[[20, 23]], short)
    assert verified.tolist() == [True] * 20 + [False] * 3 + [True] * 7


def test_audit_sample_is_deterministic_stratified_and_respects_exclusions():
    records = [{"market": "US", "symbol": f"S{i}"} for i in range(30)]
    first = {("US", f"S{i}"): ("2000-01-03" if i < 15 else "2019-05-01") for i in range(30)}
    per = {"US::start_2000": 4, "US::late": 3}
    chosen = select_audit_sample(records, first, seed="x", per_stratum=per, exclude=[("US", "S1")])
    assert len(chosen) == 7 and ("US", "S1") not in {(c["market"], c["symbol"]) for c in chosen}
    assert chosen == select_audit_sample(
        records, first, seed="x", per_stratum=per, exclude=[("US", "S1")]
    )
    assert chosen != select_audit_sample(records, first, seed="y", per_stratum=per)
    with pytest.raises(ValueError, match="estrato"):
        select_audit_sample(records, first, seed="x", per_stratum={"US::2010s": 1})


def test_population_requires_a_completed_preparation(population):
    root, _, _ = population
    (root / "manifest.json").write_text(
        json.dumps({"kind": "prepared_cohort", "status": "running"})
    )
    with pytest.raises(ValueError, match="completa"):
        population_assets(root)
    assert edition.SCHEMA_VERSION == 1
