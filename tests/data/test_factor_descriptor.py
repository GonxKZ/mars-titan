"""Descriptor de factores de una preparación revisada, con la auditoría del descriptor v2."""

import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mars_titan.data.factor_descriptor import audit_factor_prices, describe_market_factors
from mars_titan.data.storage import sha256
from mars_titan.data.temporal import MarketClock

ROUNDED = (3.976320878595475, 3.9916150569915767, 3.845233163135132, 3.991615056991577)


def write_prices(path, market, *, first="2023-06-01", rows=6, edit=None):
    clock = MarketClock(market, "2000-01-01", "2023-12-31")
    pairs = [(d, t) for d, t in zip(clock.days, clock.decisions, strict=True)]
    pairs = [(d.isoformat(), t) for d, t in pairs if d.isoformat() >= first][:rows]
    table = [
        dict(session=d, open=10.0, high=11.0, low=9.0, close=10.5, available_at=t) for d, t in pairs
    ]
    if edit:
        edit(table)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist(
            table,
            schema=pa.schema(
                [("session", pa.string())]
                + [(name, pa.float64()) for name in ("open", "high", "low", "close")]
                + [("available_at", pa.timestamp("us", tz="UTC"))]
            ),
        ),
        path,
    )
    return sha256(path)


def sources(tmp_path, *, edit_us=None, edit_cn=None, ordering_rtol=1e-9):
    root = tmp_path / "prepared"
    spy = root / "US/SPY"
    signature = write_prices(spy / "prices.parquet", "US", edit=edit_us)
    audit = {"ordering_rtol": ordering_rtol} if ordering_rtol else {}
    (spy / "manifest.json").write_text(
        json.dumps(dict(artifacts={"prices.parquet": signature}, price_audit=audit))
    )
    preparation = tmp_path / "preparation.json"
    preparation.write_text(
        json.dumps(
            dict(
                prepared_root=str(root),
                assets=[
                    dict(market="US", symbol="SPY", manifest_sha256=sha256(spy / "manifest.json"))
                ],
                price_revision=dict(
                    number_parsing="python_float_correctly_rounded_v1", ordering_rtol=ordering_rtol
                ),
            )
        )
    )
    cn = tmp_path / "csi300"
    cn_signature = write_prices(cn / "prices.parquet", "CN", edit=edit_cn)
    (cn / "market-factors.json").write_text(
        json.dumps(
            {
                "CN": dict(
                    market="CN",
                    symbol="000300",
                    unit="index_points",
                    prices_path=str(cn / "prices.parquet"),
                    prices_sha256=cn_signature,
                    return_convention="close_over_open_minus_one",
                    point_in_time_verified=False,
                )
            }
        )
    )
    (cn / "report.json").write_text(
        json.dumps(dict(artifacts={"market-factors.json": sha256(cn / "market-factors.json")}))
    )
    previous = tmp_path / "descriptor-v2.json"
    previous.write_text("{}")
    return preparation, cn, previous


def round_last(table):
    table[-1].update(zip(("open", "high", "low", "close"), ROUNDED, strict=True))


def test_the_descriptor_points_to_the_revised_prices_with_a_new_identity(tmp_path):
    preparation, cn, previous = sources(tmp_path, edit_us=round_last)
    destination = tmp_path / "descriptor-v3-1.json"
    report = describe_market_factors(preparation, cn, destination, previous=previous)
    records = json.loads(destination.read_text())
    spy = tmp_path / "prepared/US/SPY"
    assert records["US"]["prices_path"] == str(spy / "prices.parquet")
    assert records["US"]["prices_sha256"] == sha256(spy / "prices.parquet")
    assert records["US"]["number_parsing"] == "python_float_correctly_rounded_v1"
    assert records["US"]["ordering_rtol"] == 1e-9
    assert records["US"]["base_preparation_sha256"] == sha256(preparation)
    assert records["CN"]["base_factor_report_sha256"] == sha256(cn / "report.json")
    assert report["descriptor_sha256"] == sha256(destination)
    assert report["previous_descriptor_sha256"] == sha256(previous)
    audit = json.loads((tmp_path / "descriptor-v3-1-audit.json").read_text())
    assert audit == report
    us = report["factor_audit"]["US"]
    assert us["rows"] == 6 and us["missing_within_span"] == [] and us["final_test_rows"] == 0
    # La fila de redondeo se admite con la tolerancia declarada y queda listada.
    assert us["rounded_ordering_sessions"] == [us["last_session"]]
    assert report["factor_audit"]["CN"]["rounded_ordering_sessions"] == []
    with pytest.raises(ValueError, match="ya existe"):
        describe_market_factors(preparation, cn, destination, previous=previous)


def test_a_rounding_row_is_rejected_without_a_declared_tolerance(tmp_path):
    preparation, cn, previous = sources(tmp_path, edit_us=round_last, ordering_rtol=0.0)
    with pytest.raises(ValueError, match="incoherente"):
        describe_market_factors(preparation, cn, tmp_path / "d.json", previous=previous)
    assert not (tmp_path / "d.json").exists()
    # La tolerancia de SPY no se extiende al CSI300, cuya fuente no la declara.
    preparation, cn, previous = sources(tmp_path / "cn", edit_cn=round_last)
    with pytest.raises(ValueError, match="incoherente"):
        describe_market_factors(preparation, cn, tmp_path / "d.json", previous=previous)


def shift(field, value):
    def edit(table):
        table[2][field] = value(table)

    return edit


def repeat(table):
    table[2].update(session=table[1]["session"], available_at=table[1]["available_at"])


@pytest.mark.parametrize(
    "edit,message",
    [
        (shift("close", lambda t: 0.0), "finitos y positivos"),
        (shift("close", lambda t: float("nan")), "finitos y positivos"),
        (shift("low", lambda t: 10.6), "incoherente"),
        (shift("session", lambda t: "2023-06-03"), "calendario"),
        (repeat, "crecientes"),
        (shift("available_at", lambda t: t[1]["available_at"]), "disponibilidad"),
    ],
)
def test_each_check_of_the_v2_audit_still_rejects_the_factor(tmp_path, edit, message):
    path = tmp_path / "prices.parquet"
    signature = write_prices(path, "US", edit=edit)
    with pytest.raises(ValueError, match=message):
        audit_factor_prices(path, signature, "US", ordering_rtol=1e-9)


def test_the_audit_never_reads_the_final_test_or_another_file(tmp_path):
    path = tmp_path / "prices.parquet"
    signature = write_prices(path, "US")
    with pytest.raises(ValueError, match="fuente cerrada"):
        audit_factor_prices(path, "0" * 64, "US")
    late = tmp_path / "late.parquet"

    def final_test(table):
        table[-1]["session"] = "2024-01-02"

    with pytest.raises(ValueError, match="calendario"):
        audit_factor_prices(late, write_prices(late, "US", edit=final_test), "US")
    assert audit_factor_prices(path, signature, "US")["rows"] == 6


def test_the_sources_must_be_the_prepared_and_reported_ones(tmp_path):
    preparation, cn, previous = sources(tmp_path)
    manifest = tmp_path / "prepared/US/SPY/manifest.json"
    manifest.write_text(manifest.read_text() + " ")
    with pytest.raises(ValueError, match="SPY"):
        describe_market_factors(preparation, cn, tmp_path / "d.json", previous=previous)
    preparation, cn, previous = sources(tmp_path / "other")
    (cn / "market-factors.json").write_text((cn / "market-factors.json").read_text() + " ")
    with pytest.raises(ValueError, match="CSI300"):
        describe_market_factors(preparation, cn, tmp_path / "d.json", previous=previous)
