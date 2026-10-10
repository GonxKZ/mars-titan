"""Revisión de precios sobre una preparación completa, con enlaces y sin tocar el padre."""

import json
import os

import numpy as np
import pyarrow.parquet as pq
import pytest

from mars_titan.data.audit import audit_prices
from mars_titan.data.cohort_samples import _prepared
from mars_titan.data.corpus_preparation import prepare_cohort
from mars_titan.data.input_policy import HISTORICAL_MASKED
from mars_titan.data.inventory import inventory as update_inventory
from mars_titan.data.price_revision import (
    link_verified,
    market_absences,
    observed_sessions,
    revise_prepared_prices,
)
from mars_titan.data.storage import sha256
from mars_titan.data.temporal import MarketClock
from tests.data.test_corpus_preparation import setup

ROUNDED = "3.976320878595475,3.9916150569915767,3.845233163135132,3.991615056991577"
CLOCK = MarketClock("US", "1990-01-01", "2026-01-01")


def editions(tmp_path, *, sessions=None):
    """Preparación estricta y auditoría tolerante. `sessions` alarga AAA sin huecos."""
    source, inventory, reviews = setup(tmp_path)
    if sessions is None:
        rows = ["2022-06-01,1,2,1,2,10", f"2022-06-02,{ROUNDED},10", "2022-06-06,1,2,1,2,10"]
    else:
        days = [d for d in CLOCK.days if d.isoformat() >= "2022-06-01"][:sessions]
        rows = [f"{day},1,2,1,2,10" for day in days]
        rows[-3] = f"{days[-3]},{ROUNDED},10"
    (source / "time_series/S&P500_time_series/aaa.csv").write_text(
        "Date,Open,High,Low,Close,Volume\n" + "".join(row + "\n" for row in rows)
    )
    update_inventory(source, inventory)
    strict = tmp_path / "strict.json"
    audit_prices(source, inventory, strict, details_root=tmp_path / "strict")
    parent = prepare_cohort(
        source,
        inventory,
        reviews,
        tmp_path / "parent",
        cohort="original_audited",
        markets=("US",),
        input_policy=HISTORICAL_MASKED,
        price_audit_state=strict,
    )
    assert parent["failed_assets"] == 0
    tolerant = tmp_path / "tolerant.json"
    audit_prices(
        source, inventory, tolerant, details_root=tmp_path / "tolerant", ordering_rtol=1e-9
    )
    return tmp_path / "parent/manifest.json", tolerant


def test_revision_rewrites_only_changed_prices_and_links_everything_else(tmp_path):
    parent, tolerant = editions(tmp_path)
    before = {p: sha256(p) for p in (tmp_path / "parent").rglob("*") if p.is_file()}
    output = tmp_path / "revised"
    result = revise_prepared_prices(parent, tolerant, output)
    assert result["status"] == "completed" and result["failed_assets"] == 0
    revision = result["price_revision"]
    assert revision["prices_rewritten"] == 1
    assert revision["prices_linked"] == len(result["assets"]) - 1
    assert revision["accepted_price_rows_delta"] == 1
    assert revision["ordering_rtol"] == 1e-9
    clock = MarketClock("US", "1990-01-01", "2026-01-01")
    old_root, new_root = tmp_path / "parent/prepared/US", output / "prepared/US"
    manifest, _, _ = _prepared(new_root / "AAA", clock, "original_audited", HISTORICAL_MASKED)
    assert manifest["counts"]["prices"] == 3
    assert manifest["price_audit"]["ordering_rtol"] == 1e-9
    assert manifest["price_revision_linked"] is False
    prices = pq.read_table(new_root / "AAA/prices.parquet").to_pandas()
    row = prices.loc[prices.session == "2022-06-02", ["open", "high", "low", "close"]]
    assert row.to_numpy()[0].tolist() == [float(v) for v in ROUNDED.split(",")]
    for name in ("fundamentals.parquet", "news/news.parquet", "news/manifest.json"):
        assert os.stat(new_root / "AAA" / name).st_ino == os.stat(old_root / "AAA" / name).st_ino
    assert (
        os.stat(new_root / "BBB/prices.parquet").st_ino
        == os.stat(old_root / "BBB/prices.parquet").st_ino
    )
    # El padre no cambia y la reanudación conserva el manifiesto.
    assert {p: sha256(p) for p in before} == before
    identity = sha256(output / "manifest.json")
    again = revise_prepared_prices(parent, tolerant, output)
    assert again["price_revision"] == revision
    assert sha256(output / "manifest.json") == identity


def test_revision_refuses_another_configuration_or_a_changed_parent_asset(tmp_path):
    parent, tolerant = editions(tmp_path)
    output = tmp_path / "revised"
    revise_prepared_prices(parent, tolerant, output)
    strict = tmp_path / "strict.json"
    with pytest.raises(ValueError, match="tolerancia|otra revisión"):
        revise_prepared_prices(parent, strict, output)
    manifest = tmp_path / "parent/prepared/US/AAA/manifest.json"
    original = manifest.read_bytes()
    manifest.write_bytes(original + b" ")
    with pytest.raises(ValueError, match="huella"):
        revise_prepared_prices(parent, tolerant, tmp_path / "tampered")
    manifest.write_bytes(original)
    news = tmp_path / "parent/prepared/US/BBB/news/manifest.json"
    news.write_text(news.read_text() + " ")
    with pytest.raises(ValueError):
        revise_prepared_prices(parent, tolerant, tmp_path / "other")


def test_market_absences_count_every_source_row_including_excluded_ones(tmp_path):
    _, tolerant = editions(tmp_path)
    observed = observed_sessions(tolerant)["US"]
    assert {"2022-06-01", "2022-06-02", "2022-06-06"} <= observed
    clocks = {"US": MarketClock("US", "1990-01-01", "2026-01-01")}
    receipt = market_absences(tolerant, clocks)["US"]
    assert receipt["sessions"] == ["2022-06-03"]
    assert receipt["last_considered_session"] == "2022-06-06"
    # Una fila excluida también acredita que la fuente tiene esa sesión.
    strict = market_absences(tmp_path / "strict.json", clocks)["US"]
    assert strict["sessions"] == ["2022-06-03"]


def test_links_check_their_signature_before_and_after(tmp_path):
    source = tmp_path / "a.bin"
    source.write_bytes(b"datos")
    target = tmp_path / "out/a.bin"
    with pytest.raises(ValueError, match="origen"):
        link_verified(source, target, "0" * 64)
    link_verified(source, target, sha256(source))
    assert os.stat(target).st_ino == os.stat(source).st_ino
    link_verified(source, target, sha256(source))
    with pytest.raises(ValueError, match="destino"):
        link_verified(source, target, "0" * 64)
    assert np.array_equal(
        np.frombuffer(target.read_bytes(), np.uint8), np.frombuffer(b"datos", np.uint8)
    )
    assert json.dumps(sorted(p.name for p in target.parent.iterdir())) == '["a.bin"]'


def test_encoding_reads_the_rounded_row_with_the_tolerance_of_its_audit(tmp_path):
    from tests.data.test_historical_input_masks import encode_missing

    parent, tolerant = editions(tmp_path, sessions=70)
    revise_prepared_prices(parent, tolerant, tmp_path / "revised")
    strict, strict_rows = encode_missing(
        tmp_path / "parent/prepared/US/AAA", tmp_path / "strict-samples/AAA", CLOCK
    )
    report, table = encode_missing(
        tmp_path / "revised/prepared/US/AAA", tmp_path / "samples/AAA", CLOCK
    )
    # El padre estricto pierde la fila y sus ventanas. La revisión las recupera sin cambiarla.
    assert strict["samples"] == 4 and report["samples"] == 7
    assert strict_rows.column("price_end_index").to_pylist() == list(range(63, 67))
    assert table.column("price_end_index").to_pylist() == list(range(63, 70))
    common = table.slice(0, 4).select(["session", "chart_hash", "charts"])
    assert common.equals(strict_rows.select(["session", "chart_hash", "charts"]))


def test_revision_refuses_an_asset_that_only_the_new_audit_gives_prices(tmp_path):
    parent, tolerant = editions(tmp_path)
    meta = json.loads(parent.read_text())
    row = next(r for r in meta["assets"] if r["symbol"] == "AAA")
    row["state"] = "missing_required_prices"
    parent.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="AAA necesita una preparación completa"):
        revise_prepared_prices(parent, tolerant, tmp_path / "revised")
