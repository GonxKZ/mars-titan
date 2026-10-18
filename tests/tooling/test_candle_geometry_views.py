"""La sustitución visual conserva las etiquetas y su procedencia."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks import candle_geometry_views as views
from benchmarks.candle_geometry_views import project_samples, validate_source_samples
from mars_titan.data.input_policy import HISTORICAL_MASKED, policy_identity
from mars_titan.data.storage import atomic_json, sha256
from mars_titan.training.cohort_contract import representation_hash, representation_identity


def source():
    assets = [
        dict(
            market="US",
            symbol=name,
            samples=10,
            samples_sha256=name * 64,
            labels_sha256="d" * 64,
            prices_sha256="e" * 64,
            fingerprint="f" * 64,
            representation_sha256="a" * 64,
            counts=dict(train=6, validation=4),
        )
        for name in ("a", "b")
    ]
    return dict(
        input_policy="historical_masked_2000_v1",
        kind="corpus_supervision",
        final_test_opened=False,
        configuration=dict(source_manifest_sha256="1" * 64, backend="numpy"),
        roots=dict(samples="/old/samples", prepared="/prices", labels="/labels"),
        assets=assets,
        coverage=[dict(market="US", symbol=name) for name in ("a", "b")],
        counts=dict(train=12, validation=8),
        representation=dict(old=True),
        scope="full_corpus",
        cohort_complete=True,
        training_ready=False,
        candidate_count=2,
        samples=20,
    )


def replacement():
    return {("US", "b"): dict(samples=10, samples_sha256="9" * 64)}


def project(value=None, receipts=None):
    return project_samples(
        source() if value is None else value,
        original=Path("/old"),
        original_digest="1" * 64,
        geometry=Path("/new"),
        receipts=replacement() if receipts is None else receipts,
        representation=dict(new=True),
        representation_digest="2" * 64,
    )


def test_projection_keeps_label_bytes_prices_and_configuration():
    before = source()
    expected = copy.deepcopy(before)
    result = project(before)
    assert before == expected
    assert result["assets"][0] == {
        **before["assets"][1],
        "samples_sha256": "9" * 64,
        "representation_sha256": "2" * 64,
    }
    assert result["configuration"] == before["configuration"]
    assert result["roots"] == dict(samples="/new/samples", prepared="/prices", labels="/labels")
    assert result["counts"] == dict(train=6, validation=4)
    assert result["samples"] == 10 and result["candidate_count"] == 1
    assert result["scope"] == "development_snapshot"
    assert not result["cohort_complete"] and not result["training_ready"]
    assert result["coverage"] == [dict(market="US", symbol="b")]
    assert result["representation"] == dict(new=True)


@pytest.mark.parametrize(
    "fault",
    ["source", "root", "final", "policy", "kind", "duplicate", "rows", "unknown", "empty"],
)
def test_projection_rejects_unbound_sources_and_changed_population(fault):
    value, receipts = source(), replacement()
    if fault == "source":
        value["configuration"]["source_manifest_sha256"] = "3" * 64
    elif fault == "root":
        value["roots"]["samples"] = "/another/samples"
    elif fault == "final":
        value["final_test_opened"] = True
    elif fault == "policy":
        value["input_policy"] = "strict"
    elif fault == "kind":
        value["kind"] = "another"
    elif fault == "duplicate":
        value["assets"].append(copy.deepcopy(value["assets"][1]))
    elif fault == "rows":
        receipts["US", "b"]["samples"] = 9
    elif fault == "unknown":
        receipts["US", "c"] = receipts.pop(("US", "b"))
    else:
        receipts.clear()
    with pytest.raises(ValueError):
        project(value, receipts)


def test_temporal_counts_and_purge_are_projected_without_changing_cuts():
    value = source()
    value["temporal_view"] = dict(fold=dict(train=["2000-01-01", "2021-04-01"]))
    value["purged_by_boundary"] = dict(train=6, validation=2)
    value["assets"][0]["purged_by_boundary"] = dict(train=4, validation=1)
    value["assets"][1]["purged_by_boundary"] = dict(train=2, validation=1)
    result = project(value)
    assert result["temporal_view"] == value["temporal_view"]
    assert result["purged_by_boundary"] == dict(train=2, validation=1)


@pytest.mark.parametrize("fault", [None, "hash", "representation", "missing"])
def test_original_binding_is_verified_before_replacing_hashes(fault):
    value = source()
    original = {("US", "b"): dict(samples_sha256="b" * 64)}
    if fault == "hash":
        value["assets"][1]["samples_sha256"] = "8" * 64
    elif fault == "representation":
        value["representation"] = dict(changed=True)
    elif fault == "missing":
        value["assets"].pop()
    if fault is None:
        validate_source_samples(value, original, dict(old=True))
    else:
        with pytest.raises(ValueError, match="muestras originales"):
            validate_source_samples(value, original, dict(old=True))


@pytest.fixture
def paired(tmp_path, monkeypatch):
    original, geometry, source_views = (
        tmp_path / name for name in ("original", "geometry", "views")
    )
    output, parent_path = tmp_path / "output", tmp_path / "parent.json"
    roots = dict(
        samples=str(original / "samples"),
        prepared=str(tmp_path / "prices"),
        labels=str(tmp_path / "labels"),
    )
    for path in roots.values():
        Path(path).mkdir(parents=True)
    labels = Path(roots["labels"]) / "opaque-labels.bin"
    labels.write_bytes(b"Etiquetas originales de la prueba")
    old_receipt = policy_identity(HISTORICAL_MASKED) | dict(
        fundamental_concepts=["concept"],
        macro_indicators=["indicator"],
        encoders=dict(image_model="original_fixture"),
        representation_code=dict(samples="c" * 64),
        text_aggregation="fixture",
        context_sessions=64,
        news_lookback_sessions=5,
    )
    new_receipt = copy.deepcopy(old_receipt)
    new_receipt["encoders"]["image_model"] = "geometry_fixture"
    atomic_json(original / "manifest.json", dict(kind="materialized_corpus", assets=2))
    encoded = dict(
        assets=[dict(market="US", symbol="b")],
        chart_derivation=dict(
            source=dict(source=str(original), manifest_sha256=sha256(original / "manifest.json"))
        ),
    )
    atomic_json(geometry / "manifest.json", encoded)
    for variant, root, receipt in (
        ("baseline", original, old_receipt),
        ("geometry", geometry, new_receipt),
    ):
        for symbol in ("a", "b"):
            atomic_json(
                root / "samples" / "US" / symbol / "manifest.json",
                dict(
                    receipt,
                    samples=10,
                    samples_sha256=symbol * 64 if variant == "baseline" else "9" * 64,
                ),
            )
    parent = source()
    parent["roots"] = roots
    parent["configuration"]["source_manifest_sha256"] = sha256(original / "manifest.json")
    parent["representation"] = representation_identity(old_receipt, input_policy=HISTORICAL_MASKED)
    digest = representation_hash(parent["representation"], input_policy=HISTORICAL_MASKED)
    for asset in parent["assets"]:
        asset["representation_sha256"] = digest
        asset["labels_sha256"] = sha256(labels)
    atomic_json(parent_path, parent)
    originals = {}
    for fold, year in (("fold-017", "2022"), ("fold-018", "2023")):
        value = copy.deepcopy(parent)
        value["temporal_view"] = dict(
            parent_manifest=str(parent_path),
            parent_sha256=sha256(parent_path),
            protocol=dict(final_test_start="2024-01-01"),
            fold=dict(id=fold, evaluation=[year + "-01-01", str(int(year) + 1) + "-01-01"]),
        )
        atomic_json(source_views / fold / "manifest.json", value)
        originals[fold] = value
    calls, comparisons = [], []
    comparison = dict(
        assets=1, samples=10, same_inputs=True, different_columns_allowed=["charts", "chart_hash"]
    )

    def compare(previous, current):
        comparisons.append((previous, current))
        assert (previous, current) == (original, geometry)
        return copy.deepcopy(comparison)

    class Dataset:
        def __init__(self, path, *, cache_bytes, input_policy):
            assert cache_bytes == 0 and input_policy == HISTORICAL_MASKED
            calls.append(path)
            self.manifest = json.loads(path.read_text())
            self.identity = sha256(path)

    monkeypatch.setattr(views, "compare_representations", compare)
    monkeypatch.setattr(views, "CorpusDataset", Dataset)
    return SimpleNamespace(
        original=original,
        geometry=geometry,
        source_views=source_views,
        output=output,
        parent_path=parent_path,
        parent=parent,
        originals=originals,
        labels=labels,
        calls=calls,
        comparisons=comparisons,
        comparison=comparison,
        old_receipt=old_receipt,
        new_receipt=new_receipt,
    )


def prepare(paired):
    return views.prepare(paired.source_views, paired.geometry, paired.output)


def rewrite(path, change):
    value = json.loads(path.read_text())
    change(value)
    atomic_json(path, value)


def test_prepare_publishes_four_bound_views_with_the_original_label_provenance(paired):
    labels_before = paired.labels.read_bytes()
    report = prepare(paired)
    assert paired.comparisons == [(paired.original, paired.geometry)]
    assert len(paired.calls) == 6
    assert not report["labels_recalculated"] and not report["final_test_opened"]
    assert not report["training_ready"] and report["comparison"] == paired.comparison
    assert json.loads((paired.output / "prepared.json").read_text()) == report
    for variant, root, receipt in (
        ("baseline", paired.original, paired.old_receipt),
        ("geometry", paired.geometry, paired.new_receipt),
    ):
        parent_path = paired.output / variant / "parent.json"
        parent = json.loads(parent_path.read_text())
        assert parent["configuration"] == paired.parent["configuration"]
        assert parent["roots"] == paired.parent["roots"] | {"samples": str(root / "samples")}
        assert parent["assets"][0]["fingerprint"] == "f" * 64
        assert parent["assets"][0]["labels_sha256"] == sha256(paired.labels)
        assert parent["assets"][0]["prices_sha256"] == "e" * 64
        assert parent["representation"] == representation_identity(
            receipt, input_policy=HISTORICAL_MASKED
        )
        assert parent["representation_derivation"]["source_manifest_sha256"] == sha256(
            paired.parent_path
        )
        for fold in views.FOLDS:
            path = paired.output / variant / fold / "manifest.json"
            value = json.loads(path.read_text())
            original = paired.originals[fold]
            assert value["configuration"] == original["configuration"]
            assert value["temporal_view"] == original["temporal_view"] | dict(
                parent_manifest=str(parent_path), parent_sha256=sha256(parent_path)
            )
            assert value["representation_derivation"]["source_manifest_sha256"] == sha256(
                paired.source_views / fold / "manifest.json"
            )
            assert (
                value["representation_derivation"]["labels"]
                == "same_original_parquet_without_recalculation"
            )
            assert report["views"][variant][fold] == dict(
                path=str(path), sha256=sha256(path), counts=dict(train=6, validation=4)
            )
    assert paired.labels.read_bytes() == labels_before
    assert all(sha256(Path(path)) == digest for path, digest in report["source_manifests"].items())


@pytest.mark.parametrize(
    "fault",
    [
        "origin_hash",
        "fold",
        "final_start",
        "evaluation_end",
        "parent_path",
        "parent_hash",
        "changed_parent",
        "nested_parent",
        "samples_hash",
        "representation",
        "duplicate_asset",
        "mixed_representation",
    ],
)
def test_prepare_rejects_foreign_sources_before_publishing(paired, fault):
    view = paired.source_views / "fold-018" / "manifest.json"
    if fault == "origin_hash":
        rewrite(
            paired.geometry / "manifest.json",
            lambda value: value["chart_derivation"]["source"].update(manifest_sha256="0" * 64),
        )
    elif fault == "fold":
        rewrite(view, lambda value: value["temporal_view"]["fold"].update(id="fold-016"))
    elif fault == "final_start":
        rewrite(
            view,
            lambda value: value["temporal_view"]["protocol"].update(final_test_start="2025-01-01"),
        )
    elif fault == "evaluation_end":
        rewrite(
            view,
            lambda value: value["temporal_view"]["fold"].update(
                evaluation=["2023-01-01", "2024-01-02"]
            ),
        )
    elif fault == "parent_path":
        rewrite(
            view,
            lambda value: value["temporal_view"].update(
                parent_manifest=str(paired.parent_path.with_name("another.json"))
            ),
        )
    elif fault == "parent_hash":
        rewrite(view, lambda value: value["temporal_view"].update(parent_sha256="0" * 64))
    elif fault in ("changed_parent", "nested_parent"):
        rewrite(
            paired.parent_path,
            lambda value: value.update(temporal_view={} if fault == "nested_parent" else None),
        )
        if fault == "nested_parent":
            for fold in views.FOLDS:
                rewrite(
                    paired.source_views / fold / "manifest.json",
                    lambda value: value["temporal_view"].update(
                        parent_sha256=sha256(paired.parent_path)
                    ),
                )
    elif fault == "samples_hash":
        rewrite(view, lambda value: value["assets"][1].update(samples_sha256="0" * 64))
    elif fault == "representation":
        rewrite(
            view, lambda value: value["representation"]["encoders"].update(image_model="foreign")
        )
    elif fault == "duplicate_asset":
        rewrite(
            paired.geometry / "manifest.json",
            lambda value: value["assets"].append(copy.deepcopy(value["assets"][0])),
        )
    else:
        rewrite(
            paired.geometry / "manifest.json",
            lambda value: value["assets"].insert(0, dict(market="US", symbol="a")),
        )
        rewrite(
            paired.original / "samples/US/a/manifest.json",
            lambda value: value["encoders"].update(image_model="foreign"),
        )
    with pytest.raises(ValueError):
        prepare(paired)
    assert not (paired.output / "prepared.json").exists()
    assert not paired.calls


@pytest.mark.parametrize("source", ["views", "original", "geometry", "labels", "prices"])
def test_outputs_cannot_overwrite_sources_even_through_an_alias(paired, tmp_path, source):
    roots = dict(
        views=paired.source_views,
        original=paired.original,
        geometry=paired.geometry,
        labels=Path(paired.parent["roots"]["labels"]),
        prices=Path(paired.parent["roots"]["prepared"]),
    )
    alias = tmp_path / "alias"
    alias.symlink_to(roots[source], target_is_directory=True)
    with pytest.raises(ValueError, match="fuera"):
        views.prepare(paired.source_views, paired.geometry, alias / "new-views")
    assert not (roots[source] / "new-views").exists()
    assert not paired.comparisons


def test_existing_output_is_not_reused_or_modified(paired):
    paired.output.mkdir()
    marker = paired.output / "existing"
    marker.write_text("conservar")
    with pytest.raises(FileExistsError):
        prepare(paired)
    assert marker.read_text() == "conservar"
    assert not paired.comparisons


def test_read_detects_a_change_during_manifest_loading(paired, monkeypatch):
    original = views.read_manifest

    def changed(path, limit):
        result = original(path, limit)
        path.write_text(path.read_text() + "\n")
        return result

    monkeypatch.setattr(views, "read_manifest", changed)
    with pytest.raises(ValueError, match="mientras se leía"):
        prepare(paired)
    assert not paired.output.exists()


@pytest.mark.parametrize("restore_bytes", [False, True])
def test_late_source_mutation_prevents_the_completion_receipt(paired, monkeypatch, restore_bytes):
    compare = views.compare_representations
    path = paired.source_views / "fold-017" / "manifest.json"

    def changed(*args):
        result = compare(*args)
        before = path.read_bytes()
        path.write_bytes(before + b"\n")
        if restore_bytes:
            path.write_bytes(before)
        return result

    monkeypatch.setattr(views, "compare_representations", changed)
    with pytest.raises(ValueError, match="durante la preparación"):
        prepare(paired)
    assert not (paired.output / "prepared.json").exists()


def test_content_hash_still_rejects_a_change_with_stale_file_metadata(paired, monkeypatch):
    compare, stat = views.compare_representations, views._stat
    signatures = {}
    path = paired.source_views / "fold-017" / "manifest.json"

    def unchanged_signature(path):
        return signatures.setdefault(path, stat(path))

    def changed(*args):
        result = compare(*args)
        path.write_bytes(path.read_bytes() + b"\n")
        return result

    monkeypatch.setattr(views, "_stat", unchanged_signature)
    monkeypatch.setattr(views, "compare_representations", changed)
    with pytest.raises(ValueError, match="durante la preparación"):
        prepare(paired)
    assert not (paired.output / "prepared.json").exists()


@pytest.mark.parametrize("port", ["compare", "dataset"])
def test_failed_comparison_or_reader_validation_is_not_published(paired, monkeypatch, port):
    def failed(*args, **kwargs):
        raise ValueError("Contrato de datos rechazado")

    monkeypatch.setattr(
        views, "compare_representations" if port == "compare" else "CorpusDataset", failed
    )
    with pytest.raises(ValueError, match="Contrato"):
        prepare(paired)
    assert not (paired.output / "prepared.json").exists()


def test_cli_reports_the_four_views_and_closed_final_test(paired, capsys):
    assert (
        views.main(
            [
                "--source-views",
                str(paired.source_views),
                "--geometry",
                str(paired.geometry),
                "--output",
                str(paired.output),
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert set(result["views"]) == {"baseline", "geometry"}
    assert all(set(folds) == set(views.FOLDS) for folds in result["views"].values())
    assert result["labels_recalculated"] is False and result["final_test_opened"] is False
