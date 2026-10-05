"""La vista se resuelve después del contexto macro temporal y conserva la cohorte."""

import importlib
import json
import subprocess
import sys

import numpy as np
import pytest

from mars_titan.training.corpus_inputs import CorpusDataset
from tests.data.test_information_views import api, specification
from tests.training.test_corpus_inputs import corpus
from tests.training.test_temporal_corpus import inputs, prepare  # noqa: F401


def wrapper():
    try:
        return importlib.import_module("mars_titan.training.information_inputs").ViewDataset
    except ModuleNotFoundError:
        pytest.fail("Falta el lector con vista posterior a TemporalInputs")


def view_for(dataset, *, temporal=False):
    spec = specification()
    spec["cohort_sha256"] = dataset.identity
    spec["shapes"].update(fundamentals=[1], macro=[6] if temporal else [1])
    variables = [v for v in spec["variables"] if v["block"] not in {"fundamentals", "macro"}]
    for block in ("fundamentals", "macro"):
        width = spec["shapes"][block][0]
        variables.append(
            dict(
                name=block,
                source=block,
                block=block,
                value=list(range(width)),
                mask=[],
                age=[],
                dependencies=[],
                transformation="control",
                history="edición efectiva",
            )
        )
    spec["variables"] = variables
    spec["allowed_variables"] = [v["name"] for v in variables]
    return api().InformationView(spec)


def test_reduced_batches_keep_full_population_labels_and_confirmed_resume(tmp_path):
    source = CorpusDataset(corpus(tmp_path, assets=3, rows=9))
    view = view_for(source).without(sources=["news"])
    data = wrapper()(source, view)
    kwargs = dict(partition="train", batch_size=4, epoch=0, seed=42)
    original = list(source.batches(**kwargs))
    observed = list(data.batches(**kwargs))
    assert data.counts == source.manifest["counts"]
    for full, reduced in zip(original, observed, strict=True):
        assert full["sample_ids"] == reduced["sample_ids"]
        np.testing.assert_array_equal(full["target"], reduced["target"])
        assert not reduced["inputs"]["news"].any()
    tail = list(data.batches(**kwargs, cursor=observed[1]["confirmed_cursor"]))
    assert [b["sample_ids"] for b in tail] == [b["sample_ids"] for b in observed[2:]]
    wrong = wrapper()(source, view_for(source))
    with pytest.raises(ValueError, match="vista|cursor"):
        list(wrong.batches(**kwargs, cursor=observed[1]["confirmed_cursor"]))
    with pytest.raises(ValueError, match="cursor"):
        list(
            data.batches(**kwargs, cursor=observed[1]["confirmed_cursor"] | {"source_cursor": None})
        )


def test_macro_is_removed_after_temporal_replacement_in_every_partition(request, tmp_path):
    prepare(request.getfixturevalue("inputs"), tmp_path / "views")
    source = CorpusDataset(tmp_path / "views/fold-000/manifest.json")
    view = view_for(source, temporal=True)
    reduced = wrapper()(source, view.without(sources=["macro"]))
    full = wrapper()(source, view)
    for partition, count in source.manifest["counts"].items():
        kwargs = dict(partition=partition, batch_size=2, epoch=0, seed=42)
        base = list(full.batches(**kwargs))
        actual = list(reduced.batches(**kwargs))
        assert sum(len(b["target"]) for b in actual) == count
        for a, b in zip(base, actual, strict=True):
            assert a["inputs"]["macro"].shape[1] == 6
            assert a["inputs"]["macro"].any()
            assert not b["inputs"]["macro"].any()
            assert a["sample_ids"] == b["sample_ids"]


def test_source_identity_cannot_be_changed_to_reuse_a_view(tmp_path):
    path = corpus(tmp_path, assets=1)
    original = CorpusDataset(path)
    view = view_for(original)
    meta = json.loads(path.read_text())
    meta["scope"] = "full_corpus"
    meta["cohort_complete"] = True
    path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="cohorte|origen"):
        wrapper()(CorpusDataset(path), view)


def test_corpus_factory_tracks_macro_formulas_accounting_ratios_and_price_normalization(tmp_path):
    path = corpus(tmp_path, assets=1)
    meta = json.loads(path.read_text())
    meta["representation"] = dict(
        fundamental_concepts=["us-gaap:AssetsCurrent:USD", "company:current_ratio:ratio"],
        macro_indicators=["us_cpi", "us_cpi_yoy"],
        encoders={"kind": "synthetic_control"},
        representation_code={"control": "f" * 64},
        text_aggregation="mean",
        context_sessions=2,
        news_lookback_sessions=5,
    )
    path.write_text(json.dumps(meta))
    source = CorpusDataset(path)
    module = importlib.import_module("mars_titan.training.information_inputs")
    build = getattr(module, "corpus_view", None)
    assert callable(build), "Falta derivar la vista de las representaciones y catálogos existentes"
    full = build(source, macro_catalog="data/catalogs/macro-indicators.csv")
    reduced = full.without(
        variables=["macro/us_cpi", "fundamentals/us-gaap:AssetsCurrent:USD", "prices/close"]
    )
    allowed = reduced.manifest["allowed_variables"]
    assert "macro/us_cpi_yoy" not in allowed
    assert "fundamentals/company:current_ratio:ratio" not in allowed
    assert all(f"prices/{name}" not in allowed for name in ("open", "high", "low", "close"))
    assert "prices/volume" in allowed
    assert "charts" not in allowed
    spec = full.manifest
    dependency = next(v for v in spec["variables"] if v["name"] == "macro/us_cpi_yoy")
    assert "p-12" in dependency["history"]
    denominator = next(
        v for v in spec["variables"] if v["name"] == "fundamentals/us-gaap:LiabilitiesCurrent:USD"
    )
    assert denominator["block"] is None
    assert denominator["name"] in full.manifest["allowed_variables"]


def test_corpus_factory_does_not_guess_an_unidentified_representation(tmp_path):
    source = CorpusDataset(corpus(tmp_path, assets=1))
    module = importlib.import_module("mars_titan.training.information_inputs")
    build = getattr(module, "corpus_view", None)
    assert callable(build), "Falta la construcción verificada de vistas"
    with pytest.raises(ValueError, match="representación|semántica"):
        build(source, macro_catalog="data/catalogs/macro-indicators.csv")


def test_executable_view_example_keeps_population_and_zeroes_every_excluded_route(tmp_path):
    output = tmp_path / "example"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/example_information_views.py",
            "--output",
            str(output),
            "--assets",
            "2",
            "--rows",
            "8",
            "--repeats",
            "2",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["historical_evidence"] is False
    assert report["rows"] == 16
    assert report["same_population"] is True
    assert report["other_view_rejected"] is True
    assert set(report["consumer_checks"]) == {
        "parent",
        "memory",
        "hmm",
        "calibration",
        "router",
        "normalization",
    }
    assert all(value == 0.0 for value in report["consumer_checks"].values())
    assert report["measurement"]["peak_rss_bytes"] > 0
