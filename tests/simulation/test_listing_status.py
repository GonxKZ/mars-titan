"""Contrato de la tabla del estado de cotización y bandas que se derivan de ella."""

import copy
import hashlib
import json

import pytest

from mars_titan.simulation.listing_status import (
    COVERAGE_FROM,
    CUTOFF,
    TABLE_KIND,
    china_entry,
    read_listing_status,
    require_tape_status,
    tape_status,
)
from mars_titan.simulation.market_rules import (
    beijing_day,
    china_a_share_instrument,
    status_price_limits,
)

EMPTY = dict(
    listed_on="2000-01-04",
    limit_free_until=None,
    special_treatment=[],
    share_reform_pending=[],
    limit_free_days=[],
)


def entry(**changes):
    return {**copy.deepcopy(EMPTY), **changes}


def write(tmp_path, *, china=None, exits=None, **changes):
    table = dict(
        kind=TABLE_KIND,
        schema_version=1,
        cutoff=CUTOFF,
        sources={"fixture": dict(sha256=hashlib.sha256(b"fixture").hexdigest())},
        china=china if china is not None else {"CN/600000.SS": entry()},
        exits=exits or {},
    )
    table.update(changes)
    path = tmp_path / "listing-status.json"
    path.write_text(json.dumps(table))
    return path


def test_the_table_is_read_with_the_hash_of_its_bytes(tmp_path):
    path = write(tmp_path)
    table, digest = read_listing_status(path)
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    assert table["china"]["CN/600000.SS"] == EMPTY


@pytest.mark.parametrize(
    "changes",
    [
        dict(kind="other"),
        dict(schema_version=2),
        dict(cutoff="2024-12-31"),
        dict(sources={"fixture": dict(sha256="x")}),
    ],
)
def test_a_table_without_its_contract_is_rejected(tmp_path, changes):
    with pytest.raises(ValueError, match="contrato"):
        read_listing_status(write(tmp_path, **changes))


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        (dict(special_treatment=[["2009-12-31", "2011-01-04"]]), "cobertura"),
        (dict(share_reform_pending=[["2012-01-04", None], ["2013-01-04", None]]), "ISO"),
        (dict(special_treatment=[["2012-01-04", "2013-01-04"], ["2012-06-01", None]]), "solapes"),
        (dict(special_treatment=[["2012-01-04", "2024-01-02"]]), "corte"),
        (dict(limit_free_days=["2013-08-17"]), "sesión"),
        (dict(limit_free_days=["2013-08-20", "2013-08-20"]), "sesión"),
        (dict(limit_free_days=["2009-12-31"]), "sesión"),
        (dict(listed_on="2023-12-29", limit_free_until=None), "exención"),
    ],
)
def test_entries_need_ordered_spans_sessions_and_the_listing_rule(changes, message):
    asset = "CN/600000.SS"
    with pytest.raises(ValueError, match=message):
        china_entry(asset, entry(**changes))


def test_the_limit_free_window_is_recalculated_from_the_listing():
    # Cinco sesiones de XSHG desde la admisión, aunque crucen al año siguiente.
    assert china_entry("CN/688001.SS", entry(listed_on="2019-07-22", limit_free_until="2019-07-29"))
    late = entry(listed_on="2023-12-28", limit_free_until="2024-01-05")
    assert china_entry("CN/603001.SS", late)
    with pytest.raises(ValueError, match="exención"):
        china_entry("CN/688001.SS", entry(listed_on="2019-07-22", limit_free_until="2019-07-26"))


def test_exits_need_a_source_currency_and_a_later_payment(tmp_path):
    good = dict(
        last_session="2015-06-30", price=3.2, currency="CNY", paid_on="2015-07-15", source="fixture"
    )
    read_listing_status(write(tmp_path, exits={"CN/600000.SS": good}))
    for change in (
        dict(currency="USD"),
        dict(source="other"),
        dict(price=-1.0),
        dict(price=3),
        dict(paid_on="2015-06-30"),
    ):
        with pytest.raises(ValueError):
            read_listing_status(write(tmp_path, exits={"CN/600000.SS": {**good, **change}}))


def test_a_chinese_tape_needs_every_asset_and_the_covered_period(tmp_path):
    table, _ = read_listing_status(write(tmp_path))
    entries = tape_status(table, "CN", ["CN/600000.SS"])
    start = beijing_day(2011, 1, 4)
    require_tape_status("CN", ["CN/600000.SS"], entries, start)
    require_tape_status("US", ["US/AAA"], {}, beijing_day(2000, 1, 3))
    with pytest.raises(ValueError, match="cobertura"):
        require_tape_status("CN", ["CN/600000.SS"], entries, beijing_day(2009, 12, 31))
    with pytest.raises(ValueError, match="cubre"):
        require_tape_status("CN", ["CN/600000.SS", "CN/600010.SS"], entries, start)
    with pytest.raises(ValueError, match="Falta"):
        tape_status(table, "CN", ["CN/600010.SS"])
    assert COVERAGE_FROM == "2010-01-01"


def band_on(limits, year, month, day):
    at = beijing_day(year, month, day)
    period = next((p for p in limits if p.start <= at < p.end), None)
    return None if period is None else period.band


def test_warning_and_pending_reform_narrow_the_main_board_band():
    status = entry(
        special_treatment=[["2018-04-24", "2019-04-02"]],
        share_reform_pending=[["2010-01-01", "2013-08-20"]],
        limit_free_days=["2013-08-20"],
    )
    limits = status_price_limits("main", status)
    assert band_on(limits, 2012, 5, 2) == 0.05
    assert band_on(limits, 2013, 8, 19) == 0.05
    # El primer día tras la reforma no tiene banda y el siguiente vuelve al 10 %.
    assert band_on(limits, 2013, 8, 20) is None
    assert band_on(limits, 2013, 8, 21) == 0.10
    assert band_on(limits, 2018, 4, 24) == 0.05
    assert band_on(limits, 2019, 4, 2) == 0.10
    # Los tramos contiguos con la misma banda se unen.
    assert all(a.end != b.start or a.band != b.band for a, b in zip(limits, limits[1:]))


def test_other_boards_keep_their_band_under_warning_but_not_on_free_days():
    status = entry(special_treatment=[["2021-01-04", None]], limit_free_days=["2022-03-01"])
    limits = status_price_limits("chinext", status)
    assert band_on(limits, 2021, 6, 1) == 0.20
    assert band_on(limits, 2022, 3, 1) is None
    star = entry(listed_on="2019-07-22", limit_free_until="2019-07-29")
    limits = status_price_limits("star", star)
    assert band_on(limits, 2019, 7, 26) is None
    assert band_on(limits, 2019, 7, 29) == 0.20


def test_status_rules_are_identified_apart_from_board_rules():
    assert china_a_share_instrument("CN/600000.SS").rules == "cn_a_share_v1_main"
    assert china_a_share_instrument("CN/600000.SS", entry()).rules == "cn_a_share_v2_main"
