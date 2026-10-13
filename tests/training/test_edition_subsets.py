"""Subconjuntos declarados: filtros, máximo por activo, muestreo con semilla y huellas."""

import copy
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from mars_titan.data.storage import atomic_json, sha256
from mars_titan.training.edition_catalog import build_catalog, read_catalog_manifest
from mars_titan.training.edition_subsets import (
    SubsetRows,
    materialize,
    read_subset,
    row_order,
    select,
    validate_definition,
    verify_subset,
)
from tests.training.edition_catalog_fixture import built_catalog

BASE = dict(
    schema_version=1,
    kind="edition_subset_definition",
    name="all-usable",
    markets=["CN", "US"],
    years=dict(first=2000, last=2023),
    labels="usable",
)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    return built_catalog(tmp_path_factory.mktemp("subsets"), assets=4)


def catalog_rows(built):
    _, _, catalog = read_catalog_manifest(built.catalog)
    return pq.read_table(catalog).to_pylist()


def keys(table):
    return list(
        zip(
            table["market"].to_pylist(),
            table["asset_id"].to_pylist(),
            table["prediction_at"].to_pylist(),
            strict=True,
        )
    )


def definition(**changes):
    return copy.deepcopy(BASE) | changes


def usable(row):
    return row["label_reason"] == "accepted" or bool(row["annual_boundary_label"])


def test_row_filters_match_a_direct_reading_of_the_catalog(built):
    chosen = definition(
        markets=["US"],
        years=dict(first=2009, last=2023),
        modalities=dict(required=["news", "macro"]),
    )
    expected = sorted(
        (row["market"], row["asset_id"], row["prediction_at"])
        for row in catalog_rows(built)
        if row["market"] == "US"
        and 2009 <= row["session"].year <= 2023
        and usable(row)
        and row["has_news"]
        and row["has_macro"]
    )
    _, _, catalog = read_catalog_manifest(built.catalog)
    assert expected and keys(select(catalog, chosen)) == expected
    everything = select(catalog, definition(labels="all"))
    assert everything.num_rows == len(catalog_rows(built))


def test_maximum_per_asset_keeps_the_rows_with_the_lowest_order_key(built):
    _, _, catalog = read_catalog_manifest(built.catalog)
    table = select(catalog, definition(max_per_asset=3))
    rows = [row for row in catalog_rows(built) if usable(row)]
    by_asset = {}
    for row in rows:
        by_asset.setdefault((row["market"], row["asset_id"]), []).append(row["prediction_at"])
    expected = []
    for (market, asset), moments in by_asset.items():
        digest = int.from_bytes(
            hashlib.blake2b(f"{market}/{asset}".encode(), digest_size=8).digest(), "little"
        )
        micros = np.array([int(m.timestamp() * 1_000_000) for m in moments], dtype=np.int64)
        order = row_order(0, np.full(len(micros), digest, dtype=np.uint64), micros)
        expected += [(market, asset, moments[i]) for i in np.argsort(order, kind="stable")[:3]]
    assert keys(table) == sorted(expected)
    assert max(Counter(k[:2] for k in keys(table)).values()) == 3


def test_stratified_sample_keeps_the_declared_share_of_each_stratum(built):
    _, _, catalog = read_catalog_manifest(built.catalog)
    sample = dict(fraction=0.4, strata=["market", "year"], seed=7)
    table = select(catalog, definition(sample=sample))
    population = Counter(
        (row["market"], row["session"].year) for row in catalog_rows(built) if usable(row)
    )
    chosen = Counter((market, moment.year) for market, _, moment in keys(table))
    for stratum, size in population.items():
        assert chosen[stratum] == math.floor(0.4 * size + 0.5)
    again = select(catalog, definition(sample=sample))
    assert again.equals(table)
    other = select(catalog, definition(sample=dict(sample, seed=8)))
    assert keys(other) != keys(table)


def test_selection_does_not_depend_on_the_row_groups_of_the_catalog(built, tmp_path):
    regrouped = tmp_path / "regrouped"
    build_catalog(built.parent, built.edition, regrouped, row_group_rows=5)
    chosen = definition(max_per_asset=4, sample=dict(fraction=0.5, strata=["market"], seed=3))
    _, _, catalog = read_catalog_manifest(built.catalog)
    _, _, other = read_catalog_manifest(regrouped / "manifest.json")
    assert select(other, chosen).equals(select(catalog, chosen))


def test_the_order_key_is_splitmix64_of_seed_asset_and_instant():
    mask = (1 << 64) - 1

    def mix(value):
        z = (value + 0x9E3779B97F4A7C15) & mask
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & mask
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & mask
        return z ^ (z >> 31)

    seed, asset = 42, 0x0123456789ABCDEF
    moments = np.array([946_944_300_000_000, 1_703_883_900_000_000, -1], dtype=np.int64)
    actual = row_order(seed, np.full(3, asset, dtype=np.uint64), moments)
    expected = [mix(mix((int(m) & mask) ^ mix(seed)) ^ asset) for m in moments]
    assert actual.tolist() == expected


def test_materialized_subset_is_reproduced_and_its_ids_are_protected(built, tmp_path):
    path = tmp_path / "definition.json"
    atomic_json(path, definition(name="us-recent", markets=["US"], max_per_asset=5))
    document = materialize(built.catalog, path, tmp_path / "subset")
    manifest = tmp_path / "subset" / "manifest.json"
    assert verify_subset(manifest)["ids_sha256"] == document["ids"]["sha256"]
    again = materialize(built.catalog, path, tmp_path / "again")
    assert again["ids"]["sha256"] == document["ids"]["sha256"]
    _, _, table = read_subset(manifest)
    assert document["counts"]["rows"] == table.num_rows
    assert document["counts"]["markets"] == {"US": table.num_rows}
    members = SubsetRows(table)
    market, asset, moment = keys(table)[0]
    micros = int(moment.timestamp() * 1_000_000)
    assert members.member(market, asset, np.array([micros, micros + 1])).tolist() == [True, False]
    assert not members.member("CN", asset, np.array([micros])).any()
    ids = tmp_path / "subset" / "ids.parquet"
    pq.write_table(table.slice(1), ids)
    with pytest.raises(ValueError, match="no corresponden"):
        read_subset(manifest)


def test_a_subset_is_tied_to_its_catalog(built, tmp_path):
    path = tmp_path / "definition.json"
    atomic_json(path, definition())
    materialize(built.catalog, path, tmp_path / "subset")
    manifest = tmp_path / "subset" / "manifest.json"
    document = json.loads(manifest.read_text())
    document["catalog"]["manifest_sha256"] = "0" * 64
    atomic_json(manifest, document)
    with pytest.raises(ValueError, match="catálogo declarado"):
        verify_subset(manifest)
    assert sha256(built.catalog) != "0" * 64


@pytest.mark.parametrize(
    "changes",
    [
        dict(markets=["US", "CN"]),
        dict(markets=["EU"]),
        dict(years=dict(first=2024, last=2024)),
        dict(years=dict(first=2010, last=2009)),
        dict(labels="accepted"),
        dict(modalities=dict(required=["macro", "news"])),
        dict(max_per_asset=0),
        dict(sample=dict(fraction=1.0, strata=[], seed=1)),
        dict(sample=dict(fraction=0.5, strata=["asset"], seed=1)),
        dict(sample=dict(fraction=0.5, strata=[], seed=-1)),
        dict(target="value"),
    ],
)
def test_invalid_definitions_are_rejected_before_reading(changes):
    with pytest.raises(ValueError):
        validate_definition(definition(**changes))


def test_declared_definitions_of_the_repository_are_valid():
    folder = Path(__file__).resolve().parents[2] / "configs/targets/edition-subsets"
    paths = sorted(folder.glob("*.json"))
    assert paths
    for path in paths:
        document = json.loads(path.read_text())
        assert validate_definition(document)["name"] == path.stem
