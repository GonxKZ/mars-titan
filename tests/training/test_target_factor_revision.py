"""Revisión de factores residuales con las mismas entradas y sin modelos."""

import copy
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mars_titan.data.input_policy import HISTORICAL_MASKED
from mars_titan.data.storage import sha256
from mars_titan.data.temporal import MarketClock
from mars_titan.posttraining.preparation import encoder_contract
from mars_titan.training.corpus_inputs import CorpusDataset
from mars_titan.training.corpus_targets import prepare_corpus_targets
from tests.training.test_corpus_targets import audited_edition
from tests.training.test_historical_corpus_inputs import historical_edition


def dump(path, value):
    path.write_text(json.dumps(value, sort_keys=True))


@pytest.fixture
def revision(tmp_path):
    return make_revision(tmp_path)


def make_revision(tmp_path, *, historical=True):
    manifest, prepared = historical_edition(tmp_path) if historical else audited_edition(tmp_path)
    meta = json.loads(manifest.read_text())
    old = tmp_path / "market.parquet"
    replacement = tmp_path / "expanded-market.parquet"
    shutil.copyfile(old, replacement)
    pq.write_table(pq.read_table(old).slice(130), old)
    meta["market_factors"]["US"]["prices_sha256"] = sha256(old)
    meta["market_factors"]["US"]["point_in_time_verified"] = False
    sample_receipt = tmp_path / "samples/US/AAA/manifest.json"
    receipt = json.loads(sample_receipt.read_text())
    receipt.update(samples=meta["samples"], fingerprint="a" * 64)
    dump(sample_receipt, receipt)
    meta["coverage"][0]["fingerprint"] = receipt["fingerprint"]
    meta.update(scope="full_corpus", cohort_complete=True, final_test_opened=False)
    meta["configuration"] = dict(
        encoders=receipt["encoders"],
        prepared_root=str(prepared),
        market_factors=copy.deepcopy(meta["market_factors"]),
    )
    dump(manifest, meta)
    factors = copy.deepcopy(meta["market_factors"])
    factors["US"].update(prices_path=str(replacement), prices_sha256=sha256(replacement))
    descriptor = tmp_path / "target-factors.json"
    dump(descriptor, factors)
    return manifest, prepared, descriptor


def prepare(revision, output, **kwargs):
    manifest, prepared, descriptor = revision
    return prepare_corpus_targets(
        manifest,
        prepared,
        output,
        target_factors=descriptor,
        input_policy=HISTORICAL_MASKED,
        **kwargs,
    )


def test_factor_extension_recovers_labels_without_changing_inputs(revision, tmp_path):
    manifest, prepared, descriptor = revision
    protected = [manifest, descriptor, *(tmp_path / "samples").rglob("*.*")]
    before = {path: (sha256(path), path.stat().st_mtime_ns) for path in protected}
    legacy = prepare_corpus_targets(
        manifest, prepared, tmp_path / "old-labels", input_policy=HISTORICAL_MASKED
    )
    assert legacy["counts"] == {"train": 0, "validation": 1}

    result = prepare(revision, tmp_path / "new-labels")

    assert result["counts"] == {"train": 1, "validation": 1}
    assert result["roots"]["samples"] == legacy["roots"]["samples"]
    assert result["representation"] == legacy["representation"]
    assert result["coverage"] == legacy["coverage"]
    assert result["configuration"]["source_manifest_sha256"] == sha256(manifest)
    labels = pq.read_table(tmp_path / "new-labels/labels/US/AAA/labels.parquet")
    assert labels["sample_row"].to_pylist() == [0, 1, 2, 3]
    assert labels["target"][0].as_py() == pytest.approx(0.02, abs=1e-14)
    assert labels["reason"].to_pylist() == [
        "accepted",
        "target_crosses_partition_boundary",
        "accepted",
        "target_after_cutoff",
    ]
    assert encoder_contract(tmp_path / "new-labels/manifest.json", manifest)[
        "encoded_sha256"
    ] == sha256(manifest)
    reader = CorpusDataset(tmp_path / "new-labels/manifest.json", input_policy=HISTORICAL_MASKED)
    batch = next(reader.batches(partition="train", batch_size=1, epoch=0, seed=42))
    assert batch["presence"].tolist() == [[True, False, True, False, False]]
    assert {path: (sha256(path), path.stat().st_mtime_ns) for path in protected} == before


@pytest.mark.parametrize("change", [{"cohort_complete": False}, {"failed_assets": False}])
def test_factor_revision_rejects_partial_or_untyped_parent(revision, tmp_path, change):
    manifest, _, _ = revision
    meta = json.loads(manifest.read_text())
    meta.update(change)
    dump(manifest, meta)
    with pytest.raises(ValueError, match="complet|censo|población|contrato"):
        prepare(revision, tmp_path / "rejected")
    assert not (tmp_path / "rejected").exists()


def test_setting_complete_flags_does_not_hide_missing_candidate(revision, tmp_path):
    manifest, _, _ = revision
    meta = json.loads(manifest.read_text())
    meta["coverage"].pop()
    meta.update(scope="full_corpus", cohort_complete=True, failed_assets=0)
    dump(manifest, meta)
    with pytest.raises(ValueError, match="cobertura|censo|población"):
        prepare(revision, tmp_path / "rejected")
    assert not (tmp_path / "rejected").exists()


@pytest.mark.parametrize(
    "field,value", [("market", "CN"), ("symbol", "IVV"), ("point_in_time_verified", 0)]
)
def test_factor_revision_preserves_economic_semantics(revision, tmp_path, field, value):
    descriptor = revision[2]
    meta = json.loads(descriptor.read_text())
    meta["US"][field] = value
    dump(descriptor, meta)
    with pytest.raises(ValueError, match="factor|mercado|semántica"):
        prepare(revision, tmp_path / "rejected")
    assert not (tmp_path / "rejected").exists()


@pytest.mark.parametrize("artifact", ["target-factors.json", "expanded-market.parquet"])
def test_reader_and_recovery_reject_changed_factor_sources(revision, tmp_path, artifact):
    output = tmp_path / "labels"
    prepare(revision, output)
    digest = sha256(output / "manifest.json")
    path = tmp_path / artifact
    with path.open("ab") as stream:
        stream.write(b"\n")
    with pytest.raises(ValueError, match="factor|fuente|descriptor"):
        CorpusDataset(output / "manifest.json", input_policy=HISTORICAL_MASKED)
    with pytest.raises(ValueError, match="factor|fuente|configuración"):
        prepare(revision, output)
    assert sha256(output / "manifest.json") == digest


def test_factor_revision_recovery_preserves_artifacts_and_cursor(revision, tmp_path):
    output = tmp_path / "labels"
    first = prepare(revision, output)
    reader = CorpusDataset(output / "manifest.json", input_policy=HISTORICAL_MASKED)
    options = dict(partition="train", batch_size=1, epoch=0, seed=42)
    cursor = next(reader.batches(**options))["confirmed_cursor"]
    paths = [output / "manifest.json", output / "labels/US/AAA/labels.parquet"]
    before = [(sha256(p), p.stat().st_mtime_ns) for p in paths]

    second = prepare(revision, output)

    assert second["reused_assets"] == 1
    assert second["configuration"] == first["configuration"]
    assert [(sha256(p), p.stat().st_mtime_ns) for p in paths] == before
    assert list(reader.batches(**options, cursor=cursor)) == []


def test_complete_flags_do_not_hide_false_sample_counts(revision, tmp_path):
    manifest = revision[0]
    meta = json.loads(manifest.read_text())
    meta["coverage"][0]["samples"] = 5
    meta["samples"] = 5
    dump(manifest, meta)
    with pytest.raises(ValueError, match="muestra|población|cobertura"):
        prepare(revision, tmp_path / "rejected")
    assert not (tmp_path / "rejected").exists()


def test_confirmed_revision_does_not_rebuild_corrupt_labels(revision, tmp_path):
    output = tmp_path / "labels"
    prepare(revision, output)
    manifest_hash = sha256(output / "manifest.json")
    labels = output / "labels/US/AAA/labels.parquet"
    with labels.open("ab") as stream:
        stream.write(b"changed")
    altered = labels.read_bytes()
    with pytest.raises(ValueError, match="etiqueta|confirmad|recibo"):
        prepare(revision, output)
    assert labels.read_bytes() == altered
    assert sha256(output / "manifest.json") == manifest_hash


@pytest.mark.parametrize("corruption", ["reasons", "count_type"])
def test_partial_recovery_reconciles_receipt_counts(revision, tmp_path, monkeypatch, corruption):
    import mars_titan.training.corpus_targets as module

    output = tmp_path / "labels"
    write = module.atomic_json

    def interrupt(path, value):
        if path == output / "manifest.json":
            raise InterruptedError("Corte antes de confirmar el corpus")
        write(path, value)

    monkeypatch.setattr(module, "atomic_json", interrupt)
    with pytest.raises(InterruptedError):
        prepare(revision, output)
    assert not (output / "manifest.json").exists()
    receipt = output / "labels/US/AAA/receipt.json"
    meta = json.loads(receipt.read_text())
    if corruption == "reasons":
        meta["excluded_reasons"] = {"insufficient_history": 2}
    else:
        meta["counts"]["train"] = True
    dump(receipt, meta)
    monkeypatch.setattr(module, "atomic_json", write)
    with pytest.raises(ValueError, match="exclusi|recuento|recibo"):
        prepare(revision, output)
    assert not (output / "manifest.json").exists()


def test_final_confirmation_detects_a_changed_descriptor(revision, tmp_path, monkeypatch):
    import mars_titan.training.corpus_targets as module

    original = module.residual_targets_array

    def changed(*args, **kwargs):
        result = original(*args, **kwargs)
        with revision[2].open("ab") as stream:
            stream.write(b"\n")
        return result

    monkeypatch.setattr(module, "residual_targets_array", changed)
    with pytest.raises(ValueError, match="fuente|factor|descriptor"):
        prepare(revision, tmp_path / "labels")
    assert not (tmp_path / "labels/manifest.json").exists()


def test_revision_cannot_replace_the_prepared_root(revision, tmp_path):
    manifest, prepared, descriptor = revision
    other = tmp_path / "other-prepared"
    shutil.copytree(prepared, other)
    with pytest.raises(ValueError, match="prepar|entrada|origen"):
        prepare_corpus_targets(
            manifest,
            other,
            tmp_path / "labels",
            target_factors=descriptor,
            input_policy=HISTORICAL_MASKED,
        )
    assert not (tmp_path / "labels").exists()


def test_confirmed_revision_manifest_is_not_silently_repaired(revision, tmp_path):
    output = tmp_path / "labels"
    prepare(revision, output)
    path = output / "manifest.json"
    meta = json.loads(path.read_text())
    meta["counts"]["train"] = True
    dump(path, meta)
    before = path.read_bytes()
    with pytest.raises(ValueError, match="confirmad|manifiesto|recibo"):
        prepare(revision, output)
    assert path.read_bytes() == before


def test_target_implementation_change_prevents_confirmation(revision, tmp_path, monkeypatch):
    import mars_titan.training.corpus_targets as module

    calculate = module.residual_targets_array
    original_root = Path(module.__file__).parents[1]
    snapshot = tmp_path / "code/mars_titan"
    for relative in (
        "training/corpus_targets.py",
        "training/target_factors.py",
        "training/cohort_contract.py",
        "data/budget_targets.py",
        "data/residual_arrays.py",
        "data/temporal.py",
        "data/cohort_files.py",
    ):
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original_root / relative, target)
    monkeypatch.setattr(module, "__file__", str(snapshot / "training/corpus_targets.py"))

    def changed(*args, **kwargs):
        value = calculate(*args, **kwargs)
        with (snapshot / "data/residual_arrays.py").open("a") as stream:
            stream.write("\n# Cambio posterior a la identidad fijada.\n")
        return value

    monkeypatch.setattr(module, "residual_targets_array", changed)
    with pytest.raises(ValueError, match="código|implementación|fuente"):
        prepare(revision, tmp_path / "labels")
    assert not (tmp_path / "labels/manifest.json").exists()


def test_reference_backend_and_revision_keep_identical_targets(revision, tmp_path):
    prepare(revision, tmp_path / "numpy")
    prepare(revision, tmp_path / "reference", backend="reference")
    assert sha256(tmp_path / "numpy/labels/US/AAA/labels.parquet") == sha256(
        tmp_path / "reference/labels/US/AAA/labels.parquet"
    )


def test_strict_revision_keeps_2024_reserved(tmp_path):
    manifest, prepared, descriptor = make_revision(tmp_path, historical=False)
    result = prepare_corpus_targets(
        manifest, prepared, tmp_path / "labels", target_factors=descriptor
    )
    labels = pq.read_table(tmp_path / "labels/labels/US/AAA/labels.parquet").to_pylist()
    assert labels[-1]["prediction_at"].year == 2024
    assert labels[-1]["reason"] == "final_test_reserved"
    assert labels[-1]["target"] is None
    assert labels[-1]["target_available_at"] is None
    assert result["final_test_opened"] is False


def test_temporal_and_market_views_preserve_the_factor_binding(revision, tmp_path):
    from mars_titan.training.reference_campaign import campaign_views
    from mars_titan.training.temporal_corpus import prepare_temporal_corpus

    output = tmp_path / "labels"
    result = prepare(revision, output)
    projected = campaign_views(output / "manifest.json", ["US"], input_policy=HISTORICAL_MASKED)
    parent = tmp_path / "us-view.json"
    dump(parent, projected["US"])
    views = tmp_path / "views"
    prepare_temporal_corpus(
        parent,
        Path("configs/evaluation/historical-masked-us-walk-forward.json"),
        None,
        None,
        views,
        input_policy=HISTORICAL_MASKED,
        recover_annual_boundaries=True,
    )
    for index in (0, 9):
        path = views / f"fold-{index:03}/manifest.json"
        reader = CorpusDataset(path, input_policy=HISTORICAL_MASKED)
        assert reader.manifest["configuration"] == result["configuration"]
        assert reader.manifest["market_factors"] == result["market_factors"]
        assert reader.roots["samples"] == tmp_path / "samples"
        assert encoder_contract(path, revision[0])["encoded_sha256"] == sha256(revision[0])
    with revision[2].open("ab") as stream:
        stream.write(b"\n")
    with pytest.raises(ValueError, match="factor|descriptor"):
        CorpusDataset(views / "fold-009/manifest.json", input_policy=HISTORICAL_MASKED)


@pytest.mark.parametrize("reuse", [False, True])
def test_last_identity_check_precedes_publication_or_reuse(revision, tmp_path, monkeypatch, reuse):
    import mars_titan.training.corpus_targets as module

    output = tmp_path / "labels"
    if reuse:
        prepare(revision, output)
    previous = (output / "manifest.json").read_bytes() if reuse else None
    original = module.cohort_identity

    def changed(meta, **kwargs):
        result = original(meta, **kwargs)
        if meta.get("kind") == "corpus_supervision":
            with revision[2].open("ab") as stream:
                stream.write(b"\n")
        return result

    monkeypatch.setattr(module, "cohort_identity", changed)
    with pytest.raises(ValueError, match="factor|fuente|descriptor"):
        prepare(revision, output)
    if reuse:
        assert (output / "manifest.json").read_bytes() == previous
    else:
        assert not (output / "manifest.json").exists()


def test_reader_public_preparation_api_accepts_the_explicit_revision(revision, tmp_path):
    from mars_titan.training.corpus_inputs import prepare_corpus_targets as public_prepare

    manifest, prepared, descriptor = revision
    result = public_prepare(
        manifest,
        prepared,
        tmp_path / "labels",
        target_factors=descriptor,
        input_policy=HISTORICAL_MASKED,
    )
    assert result["counts"] == {"train": 1, "validation": 1}
    assert result["configuration"]["source_manifest_sha256"] == sha256(manifest)


def test_chinese_factor_revision_uses_its_own_calendar(revision, tmp_path):
    manifest, prepared, descriptor = revision
    clock = MarketClock("CN", "2021-01-01", "2023-12-31")
    factor_returns = np.resize([-0.02, 0.01, 0.03, -0.01], len(clock.days))
    common = dict(
        session=[day.isoformat() for day in clock.days],
        open=100.0,
        high=110.0,
        low=90.0,
        volume=100.0,
        available_at=clock.decisions,
    )
    factor = pd.DataFrame({**common, "close": 100 * (1 + factor_returns)})
    stock = pd.DataFrame({**common, "close": 100 * (1 + 0.003 + 1.5 * factor_returns)})
    stock.loc[131, "close"] += 2
    symbol = "000001.SZ"
    for root in (prepared, tmp_path / "samples"):
        (root / "CN").mkdir()
        (root / "US/AAA").rename(root / "CN" / symbol)
    prices = prepared / "CN" / symbol / "prices.parquet"
    pq.write_table(pa.Table.from_pandas(stock, preserve_index=False), prices)
    for path, frame in (
        (tmp_path / "market.parquet", factor.iloc[130:]),
        (tmp_path / "expanded-market.parquet", factor),
    ):
        pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), path)
    sample_path = tmp_path / "samples/CN" / symbol / "samples.parquet"
    table = pq.read_table(sample_path)
    rows = table.to_pylist()
    dates = [
        clock.decisions[130],
        clock.decision("2022-12-30"),
        clock.decision("2023-01-03"),
        clock.decisions[-1],
    ]
    for row, moment in zip(rows, dates, strict=True):
        row["prediction_at"] = moment
        row["price_end_index"] = clock.decisions.index(moment)
        row["input_availability"].update(prices=moment, charts=moment)
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), sample_path)
    for path in (prices.parent / "manifest.json", sample_path.parent / "manifest.json"):
        meta = json.loads(path.read_text())
        meta.update(market="CN", symbol=symbol)
        if "artifacts" in meta:
            meta["artifacts"]["prices.parquet"] = sha256(prices)
        else:
            meta["samples_sha256"] = sha256(sample_path)
        dump(path, meta)
    meta = json.loads(manifest.read_text())
    meta["assets"][0].update(market="CN", symbol=symbol)
    meta["coverage"][0].update(market="CN", symbol=symbol)
    meta["coverage"][1]["market"] = "CN"
    meta["calendar_start"] = {"CN": "2021-01-01"}
    original = dict(
        market="CN",
        symbol="000300",
        unit="index_points",
        return_convention="close_over_open_minus_one",
        point_in_time_verified=False,
        prices_path=str(tmp_path / "market.parquet"),
        prices_sha256=sha256(tmp_path / "market.parquet"),
    )
    meta["market_factors"] = {"CN": original}
    meta["configuration"]["market_factors"] = copy.deepcopy(meta["market_factors"])
    dump(manifest, meta)
    expanded = dict(
        original,
        prices_path=str(tmp_path / "expanded-market.parquet"),
        prices_sha256=sha256(tmp_path / "expanded-market.parquet"),
    )
    dump(descriptor, {"CN": expanded})

    result = prepare(revision, tmp_path / "labels")

    assert result["counts"] == {"train": 1, "validation": 1}
    labels = pq.read_table(tmp_path / "labels/labels/CN" / symbol / "labels.parquet")
    assert labels["target"][0].as_py() == pytest.approx(0.02, abs=1e-14)
    assert labels["prediction_at"][0].as_py() == dates[0]
    reader = CorpusDataset(tmp_path / "labels/manifest.json", input_policy=HISTORICAL_MASKED)
    batch = next(reader.batches(partition="train", batch_size=1, epoch=0, seed=42))
    assert batch["sample_ids"][0].startswith("CN/000001.SZ/")


def test_parent_factor_must_match_its_frozen_encoding_configuration(revision, tmp_path):
    manifest = revision[0]
    meta = json.loads(manifest.read_text())
    meta["configuration"]["market_factors"]["US"]["symbol"] = "IVV"
    dump(manifest, meta)
    with pytest.raises(ValueError, match="factor|configuración|identidad"):
        prepare(revision, tmp_path / "labels")
    assert not (tmp_path / "labels").exists()


def test_completed_revision_does_not_recreate_a_missing_receipt(revision, tmp_path):
    output = tmp_path / "labels"
    prepare(revision, output)
    receipt = output / "labels/US/AAA/receipt.json"
    receipt.unlink()
    before = sha256(output / "manifest.json")
    with pytest.raises(ValueError, match="recibo|confirmad"):
        prepare(revision, output)
    assert not receipt.exists()
    assert sha256(output / "manifest.json") == before


def test_reader_rechecks_factor_before_using_an_existing_cursor(revision, tmp_path):
    output = tmp_path / "labels"
    prepare(revision, output)
    reader = CorpusDataset(output / "manifest.json", input_policy=HISTORICAL_MASKED)
    options = dict(partition="train", batch_size=1, epoch=0, seed=42)
    cursor = next(reader.batches(**options))["confirmed_cursor"]
    with revision[2].open("ab") as stream:
        stream.write(b"\n")
    with pytest.raises(ValueError, match="factor|fuente|descriptor"):
        list(reader.batches(**options, cursor=cursor))


def test_recovery_destination_cannot_overwrite_a_previous_factor(revision, tmp_path):
    manifest, _, _ = revision
    output = tmp_path / "labels"
    previous_factor = output / "labels/US/AAA/labels.parquet"
    previous_factor.parent.mkdir(parents=True)
    shutil.copyfile(tmp_path / "market.parquet", previous_factor)
    meta = json.loads(manifest.read_text())
    meta["market_factors"]["US"]["prices_path"] = str(previous_factor)
    meta["configuration"]["market_factors"] = copy.deepcopy(meta["market_factors"])
    dump(manifest, meta)
    prepare(revision, tmp_path / "reference-labels")
    shutil.copyfile(tmp_path / "reference-labels/configuration.json", output / "configuration.json")
    before = previous_factor.read_bytes()

    with pytest.raises(ValueError):
        prepare(revision, output)

    assert previous_factor.read_bytes() == before
    assert not (output / "manifest.json").exists()


@pytest.mark.parametrize("linked", ["labels", "labels/US", "labels/US/AAA"])
def test_recovery_rejects_nested_links_before_writing_a_factor(revision, tmp_path, linked):
    protected_root = tmp_path / "protected-factor"
    relative = Path("labels/US/AAA/labels.parquet").relative_to(linked)
    protected = protected_root / relative
    protected.parent.mkdir(parents=True)
    shutil.copyfile(tmp_path / "expanded-market.parquet", protected)
    description = json.loads(revision[2].read_text())
    description["US"].update(prices_path=str(protected), prices_sha256=sha256(protected))
    dump(revision[2], description)
    reference = tmp_path / "reference-labels"
    prepare(revision, reference)
    output = tmp_path / "revision-labels"
    output.mkdir()
    shutil.copyfile(reference / "configuration.json", output / "configuration.json")
    link = output / linked
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(protected_root, target_is_directory=True)
    before = protected.read_bytes(), protected.stat().st_mtime_ns

    with pytest.raises(ValueError):
        prepare(revision, output)

    assert (protected.read_bytes(), protected.stat().st_mtime_ns) == before
    assert not (protected.parent / "receipt.json").exists()
    assert not (output / "manifest.json").exists()


def test_reader_rejects_an_unconfirmed_factor_contract_hash(revision, tmp_path):
    output = tmp_path / "labels"
    prepare(revision, output)
    manifest = output / "manifest.json"
    meta = json.loads(manifest.read_text())
    meta["configuration"]["target_factor_revision"]["contract_sha256"] = "0" * 64
    dump(manifest, meta)

    with pytest.raises(ValueError):
        CorpusDataset(manifest, input_policy=HISTORICAL_MASKED)


@pytest.mark.parametrize("completed", [False, True])
def test_empty_receipt_is_rejected_without_repair(revision, tmp_path, completed):
    output = tmp_path / "labels"
    prepare(revision, output)
    manifest = output / "manifest.json"
    if not completed:
        manifest.unlink()
    receipt = output / "labels/US/AAA/receipt.json"
    labels = receipt.with_name("labels.parquet")
    dump(receipt, {})
    before = receipt.read_bytes(), labels.read_bytes(), labels.stat().st_mtime_ns

    with pytest.raises(ValueError):
        prepare(revision, output)

    assert (receipt.read_bytes(), labels.read_bytes(), labels.stat().st_mtime_ns) == before
    assert manifest.exists() is completed


@pytest.mark.parametrize("artifact", ["prices", "samples"])
@pytest.mark.parametrize("reuse", [False, True])
def test_final_confirmation_rechecks_asset_parquets(
    revision, tmp_path, monkeypatch, artifact, reuse
):
    import mars_titan.training.corpus_targets as module

    output = tmp_path / "labels"
    if reuse:
        prepare(revision, output)
    manifest = output / "manifest.json"
    previous = manifest.read_bytes() if reuse else None
    source = (
        revision[1] / "US/AAA/prices.parquet"
        if artifact == "prices"
        else tmp_path / "samples/US/AAA/samples.parquet"
    )
    original = module.cohort_identity

    def changed(meta, **kwargs):
        result = original(meta, **kwargs)
        if meta.get("kind") == "corpus_supervision":
            with source.open("ab") as stream:
                stream.write(b"changed-before-final-confirmation")
        return result

    monkeypatch.setattr(module, "cohort_identity", changed)
    with pytest.raises(ValueError):
        prepare(revision, output)
    if reuse:
        assert manifest.read_bytes() == previous
    else:
        assert not manifest.exists()
