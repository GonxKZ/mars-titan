"""Recorrido de todos los activos preparados y confirmación de la edición codificada."""

import importlib
import json

import pytest

from mars_titan.data.storage import sha256
from tests.data.test_cohort_samples import Encoders, fixture


def encode(*args, **kwargs):
    try:
        module = importlib.import_module("mars_titan.data.corpus_encoding")
    except ModuleNotFoundError:
        pytest.fail("Falta el recorrido de codificación del corpus")
    return module.encode_corpus(*args, **kwargs)


def prepared_edition(tmp_path, articles=None):
    source, clock, macro = fixture(tmp_path, articles)
    manifest = tmp_path / "prepared-cohort.json"
    manifest.write_text(
        json.dumps(
            dict(
                schema_version=1,
                kind="prepared_cohort",
                status="completed",
                cohort_id="original_audited",
                candidate_count=2,
                failed_assets=0,
                prepared_root=str(source.parents[1]),
                configuration=dict(markets=["US"]),
                assets=[
                    dict(
                        market="US",
                        symbol="A",
                        state="prepared",
                        manifest_sha256=sha256(source / "manifest.json"),
                    ),
                    dict(market="US", symbol="B", state="missing_modalities", missing=["news"]),
                ],
            )
        )
    )
    return manifest, clock, macro.path


def test_full_candidate_coverage_retains_exclusions_and_recovers_without_encoding(tmp_path):
    manifest, clock, macro = prepared_edition(tmp_path)
    output = tmp_path / "encoded"
    kwargs = dict(macros={"US": macro}, encoders=Encoders(), clocks={"US": clock}, context=2)
    first = encode(manifest, output, **kwargs)
    assert first["schema_version"] == 2
    assert first["cohort_id"] == "original_audited"
    assert first["scope"] == "full_corpus"
    assert first["cohort_complete"] is True
    assert first["candidate_count"] == 2
    assert len(first["coverage"]) == 2
    assert first["coverage"][1]["state"] == "missing_modalities"
    assert len(first["assets"]) == 1
    assert first["assets"][0]["cohort_id"] == "original_audited"
    assert first["samples"] == 5
    identity = sha256(output / "manifest.json")
    before = kwargs["encoders"].calls
    second = encode(manifest, output, **kwargs)
    assert second["reused_assets"] == 1
    assert kwargs["encoders"].calls == before
    assert sha256(output / "manifest.json") == identity


@pytest.mark.parametrize(
    "change", [dict(status="running"), dict(candidate_count=3), dict(failed_assets=1)]
)
def test_partial_or_inconsistent_preparation_is_not_a_full_corpus(tmp_path, change):
    manifest, clock, macro = prepared_edition(tmp_path)
    manifest.write_text(json.dumps({**json.loads(manifest.read_text()), **change}))
    output = tmp_path / "encoded"
    with pytest.raises(ValueError, match="preparación|candidatos|completa"):
        encode(
            manifest,
            output,
            macros={"US": macro},
            encoders=Encoders(),
            clocks={"US": clock},
            context=2,
        )
    assert not (output / "manifest.json").exists()


def test_failed_source_is_recorded_and_cannot_become_a_training_edition(tmp_path):
    manifest, clock, macro = prepared_edition(tmp_path)
    (tmp_path / "prepared/US/A/prices.parquet").write_bytes(b"Cambio de entrada")
    output = tmp_path / "encoded"
    result = encode(
        manifest, output, macros={"US": macro}, encoders=Encoders(), clocks={"US": clock}, context=2
    )
    assert result["failed_assets"] == 1
    assert result["cohort_complete"] is False
    assert result["coverage"][0]["state"] == "failed"
    assert not (output / "manifest.json").exists()
    assert (output / "progress.json").exists()


def test_missing_market_context_does_not_substitute_another_market(tmp_path):
    manifest, clock, _ = prepared_edition(tmp_path)
    with pytest.raises(ValueError, match="macro|mercado"):
        encode(manifest, tmp_path / "encoded", macros={}, encoders=Encoders(), clocks={"US": clock})


def test_intermediate_output_symlink_never_writes_outside_edition(tmp_path, monkeypatch):
    manifest, clock, macro = prepared_edition(tmp_path)
    output = tmp_path / "encoded"
    module = importlib.import_module("mars_titan.data.corpus_encoding")
    original = module.materialize_cohort_asset

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    kwargs = dict(macros={"US": macro}, encoders=Encoders(), clocks={"US": clock}, context=2)
    monkeypatch.setattr(module, "materialize_cohort_asset", interrupt)
    with pytest.raises(KeyboardInterrupt):
        encode(manifest, output, **kwargs)
    monkeypatch.setattr(module, "materialize_cohort_asset", original)
    external = tmp_path / "external"
    external.mkdir()
    (output / "samples").mkdir(exist_ok=True)
    (output / "samples/US").symlink_to(external, target_is_directory=True)
    result = encode(manifest, output, **kwargs)
    assert result["cohort_complete"] is False
    assert result["failed_assets"] == 1
    assert not list(external.iterdir())


def test_manifest_hash_identifies_the_population_actually_read(tmp_path, monkeypatch):
    manifest, clock, macro = prepared_edition(tmp_path)
    module = importlib.import_module("mars_titan.data.corpus_encoding")
    original = module._read
    changed = False

    def replace_after_read(path):
        nonlocal changed
        data = original(path)
        if path == manifest and not changed:
            replacement = json.loads(manifest.read_text())
            replacement["assets"].append(
                dict(market="US", symbol="C", state="missing_modalities", missing=["news"])
            )
            replacement["candidate_count"] = 3
            manifest.write_text(json.dumps(replacement))
            changed = True
        return data

    monkeypatch.setattr(module, "_read", replace_after_read)
    with pytest.raises(ValueError, match="cambió|cambiado"):
        encode(
            manifest,
            tmp_path / "encoded",
            macros={"US": macro},
            encoders=Encoders(),
            clocks={"US": clock},
            context=2,
        )
    assert not (tmp_path / "encoded/manifest.json").exists()


def test_reviewed_subset_keeps_its_scope_and_requested_sample_contract(tmp_path):
    from mars_titan.data.samples import FUNDAMENTAL_CONCEPTS

    manifest, clock, macro = prepared_edition(tmp_path)
    content = json.loads(manifest.read_text())
    content.update(scope="reviewed_asset_subset", parent_preparation={"candidate_count": 892})
    manifest.write_text(json.dumps(content))
    result = encode(
        manifest,
        tmp_path / "subset",
        macros={"US": macro},
        encoders=Encoders(),
        clocks={"US": clock},
        context=2,
        fundamental_concepts=FUNDAMENTAL_CONCEPTS,
        company_factors=False,
        source_unit="USD",
        admitted_decisions={"US": {clock.decision("2023-07-06")}},
    )
    assert result["scope"] == "development_snapshot"
    assert result["cohort_complete"] is False
    assert result["preparation_scope"] == "reviewed_asset_subset"
    assert result["parent_preparation"]["candidate_count"] == 892
    assert result["samples"] == 1
    assert (tmp_path / "subset/manifest.json").is_file()
    receipt = json.loads((tmp_path / "subset/samples/US/A/manifest.json").read_text())
    assert receipt["fundamental_concepts"] == list(FUNDAMENTAL_CONCEPTS)
    assert receipt["source_unit"] == "USD"
    assert receipt["admitted_decisions_sha256"] is not None


def test_unknown_preparation_scope_is_rejected(tmp_path):
    manifest, clock, macro = prepared_edition(tmp_path)
    content = json.loads(manifest.read_text())
    content["scope"] = "unidentified_selection"
    manifest.write_text(json.dumps(content))
    with pytest.raises(ValueError, match="preparación|alcance"):
        encode(
            manifest,
            tmp_path / "out",
            macros={"US": macro},
            encoders=Encoders(),
            clocks={"US": clock},
        )


def test_reserved_admission_fails_before_loading_a_model(tmp_path, monkeypatch):
    manifest, clock, macro = prepared_edition(tmp_path)
    module = importlib.import_module("mars_titan.data.corpus_encoding")
    monkeypatch.setattr(module, "FrozenEncoders", lambda: pytest.fail("Se cargó un modelo"))
    with pytest.raises(ValueError, match="2024"):
        encode(
            manifest,
            tmp_path / "out",
            macros={"US": macro},
            clocks={"US": clock},
            admitted_decisions={"US": {clock.decision("2024-01-03")}},
        )
