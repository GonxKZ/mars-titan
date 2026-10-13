"""Vistas, campaña y dimensionado con un subconjunto declarado del catálogo, sin ajustar."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow.parquet as pq
import pytest

from mars_titan.data.input_policy import HISTORICAL_MASKED
from mars_titan.data.storage import atomic_json, sha256
from mars_titan.training import campaign_budget as budget
from mars_titan.training import masked_campaign as engine
from mars_titan.training.campaign_plan import load_campaign
from mars_titan.training.corpus_inputs import CorpusDataset
from mars_titan.training.edition_catalog import read_catalog_manifest
from mars_titan.training.edition_subsets import SubsetRows, materialize, read_subset
from mars_titan.training.temporal_corpus import prepare_temporal_corpus
from tests.training.edition_catalog_fixture import built_catalog
from tests.training.test_campaign_a_joint import reduced

DEFINITION = dict(
    schema_version=1,
    kind="edition_subset_definition",
    name="from-2003",
    markets=["CN", "US"],
    years=dict(first=2003, last=2023),
    labels="usable",
)


def declare(campaign, subset):
    """Copia de la campaña reducida que declara el subconjunto."""
    value = json.loads(campaign.read_text())
    value["data_subset"] = dict(manifest=str(subset), sha256=sha256(subset))
    path = campaign.with_name("campaign-subset.json")
    atomic_json(path, value)
    return path


@pytest.fixture(scope="module")
def chosen(tmp_path_factory):
    root = tmp_path_factory.mktemp("subset-views")
    data = built_catalog(root, assets=2)
    atomic_json(root / "definition.json", DEFINITION)
    materialize(data.catalog, root / "definition.json", root / "subset")
    subset = root / "subset" / "manifest.json"
    full, _ = reduced(root / "config")
    restricted = declare(full, subset)
    views = {
        "full": engine.prepare_views(full, data.parent, root / "views-full"),
        "subset": engine.prepare_views(restricted, data.parent, root / "views-subset"),
    }
    manifest, _, catalog = read_catalog_manifest(data.catalog)
    return SimpleNamespace(
        **vars(data),
        root=root,
        subset=subset,
        campaigns=dict(full=full, subset=restricted),
        views=views,
        catalog_rows=(manifest, catalog),
    )


def view_counts(prepared):
    return {
        scope: {window: value["counts"] for window, value in record["windows"].items()}
        for scope, record in prepared.items()
    }


def test_catalog_counts_are_the_counts_of_the_published_views(chosen):
    campaign = load_campaign(chosen.campaigns["full"])
    counts, fresh = budget.catalog_window_counts(campaign, chosen.catalog_rows)
    assert counts == view_counts(chosen.views["full"])
    labels = chosen.metadata["roots"]["labels"]
    assert fresh == budget.campaign_posttraining_rows(campaign, labels)


def test_subset_views_keep_only_their_rows_and_the_counts_follow(chosen):
    campaign = load_campaign(chosen.campaigns["subset"])
    _, _, ids = read_subset(chosen.subset)
    members = SubsetRows(ids)
    counts, _ = budget.catalog_window_counts(campaign, chosen.catalog_rows, members)
    assert counts == view_counts(chosen.views["subset"])
    full = view_counts(chosen.views["full"])
    assert counts["US"]["fold-000"]["train"] < full["US"]["fold-000"]["train"]
    for name in ("validation", "calibration", "evaluation"):
        assert counts["US"]["fold-018"][name] == full["US"]["fold-018"][name]
    record = chosen.views["subset"]["US"]["windows"]["fold-000"]
    view = json.loads(Path(record["path"]).read_text())
    assert view["data_subset"] == campaign["data_subset"]
    labels = pq.read_table(Path(view["roots"]["labels"]) / "US/A0000/labels.parquet")
    for row in labels.to_pylist():
        moment = row["prediction_at"]
        if moment.year < 2003 and row["source_reason"] == "accepted":
            assert row["reason"] == "outside_subset" and row["partition"] is None
        if row["reason"] == "accepted":
            micros = np.array([int(moment.timestamp() * 1_000_000)])
            assert members.member("US", "A0000", micros)[0]


def test_a_subset_view_only_observes_the_rows_of_the_subset(chosen):
    record = chosen.views["subset"]["US"]["windows"]["fold-010"]
    dataset = CorpusDataset(Path(record["path"]), input_policy=HISTORICAL_MASKED)
    _, _, ids = read_subset(chosen.subset)
    members = SubsetRows(ids)
    seen = 0
    for batch in dataset.observation_batches(
        start=946_684_800_000_000, end=1_704_067_200_000_000, batch_size=16
    ):
        for key in batch["sample_ids"]:
            market, asset, micros = key.split("/")
            assert members.member(market, asset, np.array([int(micros)]))[0]
            seen += 1
    assert seen == sum(len(v) for k, v in members.rows.items() if k.startswith("US/"))


def test_views_and_campaign_must_declare_the_same_subset(chosen):
    campaign = load_campaign(chosen.campaigns["full"])
    with pytest.raises(ValueError, match="subconjunto que declara la campaña"):
        engine.scope_views(chosen.root / "views-subset" / "US", "US", campaign)
    restricted = load_campaign(chosen.campaigns["subset"])
    with pytest.raises(ValueError, match="subconjunto que declara la campaña"):
        engine.scope_views(chosen.root / "views-full" / "US", "US", restricted)


def test_a_changed_subset_stops_the_campaign(chosen, tmp_path):
    value = json.loads(chosen.campaigns["subset"].read_text())
    value["data_subset"]["sha256"] = "0" * 64
    path = chosen.campaigns["subset"].with_name("campaign-changed.json")
    atomic_json(path, value)
    with pytest.raises(ValueError, match="subconjunto declarado cambió"):
        load_campaign(path)


def test_a_subset_of_other_targets_cannot_build_a_view(chosen, tmp_path):
    folder = tmp_path / "subset"
    manifest = json.loads(chosen.subset.read_text())
    catalog = json.loads(chosen.catalog.read_text())
    catalog["assets"][0]["labels_sha256"] = "0" * 64
    atomic_json(tmp_path / "catalog.json", catalog)
    manifest["catalog"].update(
        manifest=str(tmp_path / "catalog.json"),
        manifest_sha256=sha256(tmp_path / "catalog.json"),
    )
    folder.mkdir()
    (folder / "ids.parquet").write_bytes((chosen.subset.parent / "ids.parquet").read_bytes())
    atomic_json(folder / "manifest.json", manifest)
    projection = chosen.root / "views-full" / "parents" / "US.json"
    with pytest.raises(ValueError, match="no se eligió sobre los objetivos"):
        prepare_temporal_corpus(
            projection,
            Path("configs/evaluation/historical-masked-us-walk-forward-v2.json"),
            None,
            None,
            tmp_path / "views",
            input_policy=HISTORICAL_MASKED,
            recover_annual_boundaries=True,
            subset=folder / "manifest.json",
        )


def test_the_budget_compares_the_subset_with_the_full_design(chosen, capsys):
    arguments = [
        "--campaign",
        str(chosen.campaigns["full"]),
        "--catalog",
        str(chosen.catalog),
        "--subset",
        str(chosen.subset),
        "--rate",
        "1000",
        "--epochs",
        "2",
    ]
    assert budget.main(arguments) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["subset"]["ids_sha256"] == read_subset(chosen.subset)[0]["ids"]["sha256"]
    assert report["rows"]["train"] < report["full_design"]["rows"]["train"]
    assert 0 < report["subset_over_full_neural_hours"] < 1
    with pytest.raises(SystemExit):
        budget.main(arguments[:2] + ["--subset", str(chosen.subset), "--rate", "1000"])
