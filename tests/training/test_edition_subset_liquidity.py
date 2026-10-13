"""Universo de liquidez de los subconjuntos, decidido en el corte de cada año sin el futuro."""

import copy
import json
from collections import Counter

import numpy as np
import pyarrow.parquet as pq
import pytest

from mars_titan.data.storage import atomic_json
from mars_titan.evaluation import liquidity_strata as ls
from mars_titan.training.edition_catalog import read_catalog_manifest
from mars_titan.training.edition_subsets import (
    LIQUIDITY_CUT,
    liquidity_edition,
    materialize,
    read_subset,
    select,
    validate_definition,
    verify_subset,
    year_start,
)
from tests.simulation.unadjusted_edition_fixture import Asset, tape_days, write_edition
from tests.training.edition_catalog_fixture import built_catalog

MARKETS = ("CN", "US")


def low_in_2023(market, symbol):
    """Cierres por debajo de 1 en todas las sesiones de 2023 y normales antes."""
    rows = {index: dict(open=0.5, close=0.5) for index in range(len(tape_days(market)))}
    return Asset(symbol, overrides=rows)


def edition_assets(market):
    """A0000 líquido, A0001 por debajo de 1 desde el principio, A0002 solo en 2023 y A0003 fuera."""
    return [Asset("A0000"), Asset("A0001", base=0.5), low_in_2023(market, "A0002")]


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    root = tmp_path_factory.mktemp("liquidity")
    catalog = built_catalog(root, assets=4)
    edition = root / "unadjusted"
    manifest = write_edition(edition, {market: edition_assets(market) for market in MARKETS})
    return catalog, edition, manifest["edition_id"]


def section(edition_id, **changes):
    """Umbrales y ventana de la declaración de la campaña, con la edición de la fixture."""
    return (
        dict(
            declared_at="2026-10-13",
            source=ls.SOURCE,
            edition_id=edition_id,
            cut=LIQUIDITY_CUT,
            price_below=1.0,
            volume_below=1000,
            volume_sessions=20,
            keep=["liquid"],
        )
        | changes
    )


def definition(edition_id, **changes):
    liquidity = section(edition_id, **changes.pop("liquidity", {}))
    return (
        dict(
            schema_version=1,
            kind="edition_subset_definition",
            name="liquid-2022-2023",
            markets=list(MARKETS),
            years=dict(first=2022, last=2023),
            labels="all",
            liquidity=liquidity,
        )
        | changes
    )


def chosen(data, **changes):
    catalog, edition, edition_id = data
    declared = definition(edition_id, **changes)
    _, _, path = read_catalog_manifest(catalog.catalog)
    universe = {}
    table = select(
        path, declared, liquidity_edition(declared["liquidity"], edition), universe=universe
    )
    return table, universe


def keys(table):
    return Counter(
        (market, asset, moment.year)
        for market, asset, moment in zip(
            *(table[name].to_pylist() for name in ("market", "asset_id", "prediction_at")),
            strict=True,
        )
    )


def catalog_rows(data, years):
    _, _, path = read_catalog_manifest(data[0].catalog)
    rows = pq.read_table(path, columns=["market", "asset_id", "session"]).to_pylist()
    return Counter(
        (row["market"], row["asset_id"], row["session"].year)
        for row in rows
        if row["session"].year in years
    )


def prefix_stratum(edition, market, symbol, year):
    """Estrato del activo con las filas de la edición anteriores al año, por otro camino."""
    path = edition / "assets" / market / symbol / "prices.parquet"
    if not path.exists():
        return "unclassified"
    prices = pq.read_table(path).to_pylist()
    rows = [row for row in prices if row["session"] < f"{year}-01-01"]
    if not rows:
        return "unclassified"
    code, _ = ls.asset_codes(
        market,
        np.array([row["session"] for row in rows], dtype="datetime64[D]"),
        np.array([row["close"] for row in rows]),
        np.array([row["volume"] for row in rows]),
        np.array([row["verified"] for row in rows]),
        section("0" * 64),
    )
    return ls.NAMES[int(code[-1])]


def test_each_year_keeps_the_assets_liquid_at_its_cut(data):
    table, universe = chosen(data)
    rows = catalog_rows(data, {2023})
    expected = Counter({key: total for key, total in rows.items() if key[1] in ("A0000", "A0002")})
    # A0002 baja de 1 durante 2023, pero en el corte de 2023 solo cuentan las sesiones de 2022.
    assert keys(table) == expected and len(expected) == 4
    _, edition, _ = data
    for market in MARKETS:
        for year in ("2022", "2023"):
            found = {
                symbol: prefix_stratum(edition, market, symbol, int(year))
                for symbol in ("A0000", "A0001", "A0002", "A0003")
            }
            assert universe[market][year] == {
                name: sum(value == name for value in found.values()) for name in ls.NAMES
            }
    assert universe["US"]["2023"]["liquid"] == 2 and universe["US"]["2023"]["low_price"] == 1
    assert universe["US"]["2022"]["unclassified"] == 4


def test_the_cut_is_the_start_of_the_year_in_utc():
    assert year_start(2023) == 1_672_531_200_000_000
    assert year_start(2000) == 946_684_800_000_000


def test_later_sessions_do_not_change_the_universe(data, tmp_path):
    catalog, _, _ = data
    # La misma edición con A0000 por debajo de 1 en 2023: el corte de 2023 no lo ve.
    later = tmp_path / "later"
    assets = {
        market: [low_in_2023(market, "A0000"), *edition_assets(market)[1:]] for market in MARKETS
    }
    manifest = write_edition(later, assets)
    declared = definition(manifest["edition_id"])
    _, _, path = read_catalog_manifest(catalog.catalog)
    table = select(path, declared, liquidity_edition(declared["liquidity"], later))
    assert table.equals(chosen(data)[0])


def test_unclassified_rows_are_only_kept_when_declared(data):
    table, _ = chosen(data, liquidity=dict(keep=["unclassified"]))
    rows = catalog_rows(data, {2022, 2023})
    expected = Counter({k: v for k, v in rows.items() if k[2] == 2022 or k[1] == "A0003"})
    assert keys(table) == expected


def test_the_maximum_per_asset_applies_after_the_universe(data):
    table, _ = chosen(data, max_per_asset=2)
    counts = Counter((market, asset) for market, asset, _ in keys(table).elements())
    assert set(counts) == {(m, a) for m in MARKETS for a in ("A0000", "A0002")}
    assert set(counts.values()) == {2}
    assert all(year == 2023 for *_, year in keys(table))


def test_the_edition_must_be_the_declared_one(data, tmp_path):
    catalog, edition, edition_id = data
    _, _, path = read_catalog_manifest(catalog.catalog)
    other = definition("f" * 64)
    with pytest.raises(ValueError, match="no es la que declara"):
        liquidity_edition(other["liquidity"], edition)
    with pytest.raises(ValueError, match="necesita la edición"):
        liquidity_edition(definition(edition_id)["liquidity"], None)
    declared = definition(edition_id)
    changed = definition(edition_id, liquidity=dict(price_below=2.0))
    with pytest.raises(ValueError, match="no corresponde a la definición"):
        select(path, changed, liquidity_edition(declared["liquidity"], edition))
    with pytest.raises(ValueError, match="Solo una definición con liquidez"):
        select(path, declared)


def test_materialized_universe_is_verified_with_its_edition(data, tmp_path):
    catalog, edition, edition_id = data
    path = tmp_path / "definition.json"
    atomic_json(path, definition(edition_id))
    document = materialize(catalog.catalog, path, tmp_path / "subset", liquidity_root=edition)
    manifest = tmp_path / "subset" / "manifest.json"
    assert document["liquidity"]["edition_id"] == edition_id
    assert document["liquidity"]["universe"]["CN"]["2023"]["liquid"] == 2
    assert verify_subset(manifest)["ids_sha256"] == document["ids"]["sha256"]
    _, _, ids = read_subset(manifest)
    assert ids.equals(chosen(data)[0])
    # Un universo publicado distinto del que se repite no se acepta.
    changed = json.loads(manifest.read_text())
    changed["liquidity"]["universe"]["CN"]["2023"]["liquid"] = 3
    atomic_json(manifest, changed)
    with pytest.raises(ValueError, match="universo de liquidez"):
        verify_subset(manifest)
    plain = tmp_path / "plain.json"
    atomic_json(plain, {k: v for k, v in definition(edition_id).items() if k != "liquidity"})
    with pytest.raises(ValueError, match="no declara un universo"):
        materialize(catalog.catalog, plain, tmp_path / "plain", liquidity_root=edition)


@pytest.mark.parametrize(
    "changes",
    [
        dict(keep=["low_price", "liquid"]),
        dict(keep=["liquid", "liquid"]),
        dict(keep=[]),
        dict(keep=["penny"]),
        dict(cut="session"),
        dict(source="adjusted"),
        dict(declared_at="13/10/2026"),
        dict(edition_id="abc"),
        dict(price_below=0),
        dict(volume_below=1.5),
        dict(volume_sessions=0),
        dict(volume_sessions=ls.MAX_VOLUME_SESSIONS + 1),
        dict(target="value"),
    ],
)
def test_invalid_liquidity_is_rejected_before_reading(changes):
    declared = definition("0" * 64)
    declared["liquidity"] = copy.deepcopy(declared["liquidity"]) | changes
    with pytest.raises(ValueError):
        validate_definition(declared)
