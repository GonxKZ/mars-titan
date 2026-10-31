"""Cohortes por origen con manifiestos de prueba y sin abrir datos ni ajustar modelos."""

import copy
import hashlib
import json
import random
import sys
from pathlib import Path

import pytest

from mars_titan.data.input_policy import HISTORICAL_MASKED, policy_identity
from mars_titan.data.storage import atomic_json, sha256
from mars_titan.evaluation.splits import build_folds
from mars_titan.training import screening_cohorts as cohorts
from mars_titan.training.cohort_contract import cohort_identity, representation_hash

PROTOCOL = Path("configs/evaluation/historical-masked-us-walk-forward-v2.json")


def source(origin=2019):
    """Metadatos ficticios completos. Los archivos de muestras no existen deliberadamente."""
    policy = policy_identity(HISTORICAL_MASKED)
    representation = dict(
        policy,
        fundamental_concepts=["fixture"],
        macro_indicators=["fixture"],
        encoders={"fixture": "metadata_only"},
        representation_code={"fixture": "metadata_only"},
        text_aggregation="fixture",
        context_sessions=64,
        news_lookback_sessions=1,
    )
    signature = representation_hash(representation, input_policy=HISTORICAL_MASKED)
    protocol = json.loads(PROTOCOL.read_text())
    fold = next(f for f in build_folds(protocol) if f["evaluation"][0] == f"{origin}-01-01")
    assets = [
        dict(
            market="US",
            symbol=f"A{i:03d}",
            cohort_id="original_audited",
            representation_sha256=signature,
            samples=100,
            samples_sha256=hashlib.sha256(str(i).encode()).hexdigest(),
            labels_sha256=hashlib.sha256(f"labels/{i}".encode()).hexdigest(),
            counts=dict(train=50 if i else 0, validation=10, calibration=10, evaluation=10),
            purged_by_boundary=dict(train=1, validation=1, calibration=1, evaluation=1),
        )
        for i in range(301)
    ]
    document = dict(
        policy,
        schema_version=3,
        kind="corpus_supervision",
        scope="full_corpus",
        cohort_complete=True,
        cohort_id="original_audited",
        news_content_policy="source_audited_not_external",
        training_ready=False,
        final_test_opened=False,
        failed_assets=0,
        markets=["US"],
        roots={name: f"/fixture/{name}" for name in ("prepared", "samples", "labels")},
        representation=representation,
        assets=assets,
        coverage=[
            dict(market="US", symbol=a["symbol"], state="encoded", samples=a["samples"])
            for a in assets
        ],
        candidate_count=len(assets),
        samples=sum(a["samples"] for a in assets),
        temporal_view=dict(
            policy,
            schema_version=2,
            protocol=protocol,
            fold=fold,
            selection_partition="validation",
            recover_annual_boundaries=True,
            parent_manifest="/fixture/parent.json",
            parent_sha256="a" * 64,
        ),
    )
    return reconcile(document)


def reconcile(document):
    for field in ("counts", "purged_by_boundary"):
        document[field] = {
            phase: sum(a[field][phase] for a in document["assets"])
            for phase in ("train", "validation", "calibration", "evaluation")
        }
    return document


def test_each_origin_uses_its_own_past_and_keeps_all_selected_rows():
    for origin in (2019, 2020, 2021, 2022):
        original = source(origin)
        view, selection = cohorts.project_origin_cohort(original, origin=origin)
        selected = selection["assets"]
        assert len(selected) == len(set(selected)) == 256
        assert "A000" not in selected
        assert selection["information_before"] == f"{origin - 1}-04-01"
        assert selection["eligible_count"] == 300
        expected = [a for a in original["assets"] if a["symbol"] in selected]
        assert view["assets"] == expected
        assert view["samples"] == sum(a["samples"] for a in expected)
        assert view["temporal_view"] == original["temporal_view"]
        assert view["roots"] == original["roots"]
        assert view["training_ready"] is False
        assert view["final_test_opened"] is False
        assert view["cohort_complete"] is False
        assert cohort_identity(view, input_policy=HISTORICAL_MASKED) == "original_audited"
        assert original == source(origin)


def test_order_and_future_counts_cannot_choose_or_drop_an_asset():
    original = source()
    expected, identity = cohorts.project_origin_cohort(original, origin=2019)
    changed = copy.deepcopy(original)
    changed["assets"].reverse()
    changed["coverage"].reverse()
    for item in changed["assets"]:
        item["counts"].update(validation=0, calibration=0, evaluation=0)
        item["future_target"] = -1000
    actual, actual_identity = cohorts.project_origin_cohort(reconcile(changed), origin=2019)
    assert actual_identity == identity
    assert [a["symbol"] for a in actual["assets"]] == [a["symbol"] for a in expected["assets"]]
    assert actual["counts"]["evaluation"] == 0
    assert len(actual["assets"]) == 256


def test_positive_past_counts_and_cohort_size_have_no_future_threshold():
    original = source()
    _, full = cohorts.project_origin_cohort(original, origin=2019)
    _, small = cohorts.project_origin_cohort(original, origin=2019, size=64)
    assert set(small["assets"]) < set(full["assets"])
    for item in original["assets"]:
        if item["counts"]["train"]:
            item["counts"]["train"] = 1
    _, changed = cohorts.project_origin_cohort(reconcile(original), origin=2019)
    assert changed == full


@pytest.mark.parametrize("origin", [2018, 2023, 2024, True, 2019.0, "2019"])
def test_rejects_undeclared_or_final_origins(origin):
    with pytest.raises(ValueError, match="origen"):
        cohorts.project_origin_cohort(source(), origin=origin)


@pytest.mark.parametrize("size", [0, 257, True, 1.0, "256"])
def test_rejects_invalid_cohort_sizes(size):
    with pytest.raises(ValueError, match="activos"):
        cohorts.project_origin_cohort(source(), origin=2019, size=size)


def test_rejects_a_later_origin_or_no_eligible_history():
    with pytest.raises(ValueError, match="origen"):
        cohorts.project_origin_cohort(source(2021), origin=2019)
    original = source()
    for asset in original["assets"]:
        asset["counts"]["train"] = 0
    with pytest.raises(ValueError, match="suficientes"):
        cohorts.project_origin_cohort(reconcile(original), origin=2019)


@pytest.mark.parametrize("bad", [-1, True, 1.5, None])
def test_invalid_declared_counts_fail_before_selection(bad):
    original = source()
    original["assets"][0]["counts"]["train"] = bad
    with pytest.raises(ValueError, match="recuentos"):
        cohorts.project_origin_cohort(original, origin=2019)


@pytest.mark.parametrize(
    "change",
    [
        dict(final_test_opened=True),
        dict(training_ready=True),
        dict(scope="development_snapshot"),
        dict(cohort_complete=False),
        dict(markets=["CN"]),
        dict(data_subset={"future": "filter"}),
    ],
)
def test_rejects_unsealed_or_partial_parent(change):
    with pytest.raises(ValueError):
        cohorts.project_origin_cohort(source() | change, origin=2019)


def inputs(tmp_path):
    root = tmp_path / "sources"
    for origin, fold in cohorts.ORIGIN_FOLDS.items():
        atomic_json(root / fold / "manifest.json", source(origin))
    return root


def test_preparation_and_verification_only_read_manifests(tmp_path):
    sources, work = inputs(tmp_path), tmp_path / "prepared"
    paths = [sources / fold / "manifest.json" for fold in cohorts.ORIGIN_FOLDS.values()]
    before = {p: sha256(p) for p in paths}
    result = cohorts.prepare(sources, work)
    assert result["kind"] == "screening_origin_cohorts_v1"
    assert result["optimizer_steps"] == 0
    assert result["row_files_opened"] is False
    assert result["final_test_opened"] is False
    assert result["verification_scope"] == "manifests_only"
    assert cohorts.verify(work) == result
    for origin in (2019, 2020, 2021, 2022):
        view = json.loads((work / str(origin) / "manifest.json").read_text())
        assert len(view["assets"]) == 256
    assert {p: sha256(p) for p in before} == before
    with pytest.raises(FileExistsError):
        cohorts.prepare(sources, work)
    assert cohorts.verify(work) == result


@pytest.mark.parametrize("change_source", [False, True])
def test_verification_rejects_changed_sources_and_outputs(tmp_path, change_source):
    sources, work = inputs(tmp_path), tmp_path / "prepared"
    cohorts.prepare(sources, work)
    path = sources / "fold-014/manifest.json" if change_source else work / "2019/manifest.json"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="huella"):
        cohorts.verify(work)


def test_failed_preparation_does_not_publish_a_partial_cohort(tmp_path):
    sources, work = inputs(tmp_path), tmp_path / "prepared"
    (sources / "fold-016/manifest.json").write_text("{}")
    with pytest.raises(ValueError):
        cohorts.prepare(sources, work)
    assert not work.exists()


def test_preparation_cannot_write_inside_the_source(tmp_path):
    sources = inputs(tmp_path)
    with pytest.raises(ValueError, match="fuera"):
        cohorts.prepare(sources, sources / "projected")


def test_selection_does_not_change_global_random_state():
    before = random.getstate()
    cohorts.project_origin_cohort(source(), origin=2019)
    assert random.getstate() == before


@pytest.mark.parametrize("symbol", ["..", "../OTHER", "", "ñ", "a" * 65])
def test_rejects_invalid_asset_identities(symbol):
    document = source()
    document["assets"][0]["symbol"] = symbol
    document["coverage"][0]["symbol"] = symbol
    with pytest.raises(ValueError, match="identidad"):
        cohorts.project_origin_cohort(document, origin=2019)


@pytest.mark.parametrize("field", ["counts", "purged_by_boundary"])
def test_rejects_inconsistent_or_null_counts(field):
    document = source()
    document[field]["train"] += 1
    with pytest.raises(ValueError, match="recuentos"):
        cohorts.project_origin_cohort(document, origin=2019)
    document = source()
    document["assets"][0][field] = None
    with pytest.raises(ValueError, match="recuentos"):
        cohorts.project_origin_cohort(document, origin=2019)


def test_rejects_relative_data_roots():
    document = source()
    document["roots"]["samples"] = "samples"
    with pytest.raises(ValueError, match="raíces"):
        cohorts.project_origin_cohort(document, origin=2019)


def test_rejects_duplicate_candidates():
    document = source()
    document["assets"][0]["symbol"] = document["assets"][1]["symbol"]
    with pytest.raises(ValueError, match="concilian"):
        cohorts.project_origin_cohort(document, origin=2019)


def test_verification_rejects_a_tampered_selection_even_with_the_same_view_hash(tmp_path):
    sources, work = inputs(tmp_path), tmp_path / "prepared"
    receipt = cohorts.prepare(sources, work)
    receipt["origins"]["2019"]["selection"]["assets"].reverse()
    atomic_json(work / "prepared.json", receipt)
    with pytest.raises(ValueError, match="selección"):
        cohorts.verify(work)


def test_rejects_a_changed_receipt_contract(tmp_path):
    sources, work = inputs(tmp_path), tmp_path / "prepared"
    receipt = cohorts.prepare(sources, work)
    receipt["row_files_opened"] = True
    atomic_json(work / "prepared.json", receipt)
    with pytest.raises(ValueError, match="contrato"):
        cohorts.verify(work)


def test_detects_source_changes_before_publication(tmp_path, monkeypatch):
    sources, work = inputs(tmp_path), tmp_path / "prepared"
    original = cohorts.atomic_json

    def change_source(path, value):
        original(path, value)
        parent = sources / "fold-014/manifest.json"
        parent.write_text(parent.read_text() + "\n")

    monkeypatch.setattr(cohorts, "atomic_json", change_source)
    with pytest.raises(ValueError, match="huella"):
        cohorts.prepare(sources, work)
    assert not work.exists()


def test_publication_does_not_overwrite_a_concurrent_destination(tmp_path, monkeypatch):
    sources, work = inputs(tmp_path), tmp_path / "prepared"
    publish = cohorts._publish_directory

    def race(staged, output):
        output.mkdir()
        (output / "existing").write_text("original")
        publish(staged, output)

    monkeypatch.setattr(cohorts, "_publish_directory", race)
    with pytest.raises(FileExistsError):
        cohorts.prepare(sources, work)
    assert (work / "existing").read_text() == "original"
    assert not (work / "prepared.json").exists()


def test_rejects_symlinked_output(tmp_path):
    sources = inputs(tmp_path)
    (tmp_path / "link").symlink_to(sources, target_is_directory=True)
    with pytest.raises(ValueError, match="enlace"):
        cohorts.prepare(sources, tmp_path / "link" / "prepared")


def test_cli_prepares_and_verifies_without_learning(tmp_path, monkeypatch, capsys):
    sources, work = inputs(tmp_path), tmp_path / "prepared"
    for arguments in (
        ["prepare", "--sources", str(sources), "--output", str(work)],
        ["verify", "--work", str(work)],
    ):
        monkeypatch.setattr(sys, "argv", ["screening_cohorts", *arguments])
        cohorts.main()
        assert json.loads(capsys.readouterr().out) == dict(
            kind="screening_origin_cohorts_v1", size=256, verification_scope="manifests_only"
        )
