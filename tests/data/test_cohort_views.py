"""Vistas de mercado sin ocultar errores ni reducir la población seleccionada."""

import importlib
import json

import pytest

from mars_titan.data.storage import sha256
from tests.data.test_corpus_encoding import prepared_edition


def project(*args, **kwargs):
    try:
        module = importlib.import_module("mars_titan.data.cohort_views")
    except ModuleNotFoundError:
        pytest.fail("Falta la selección identificada de mercados")
    return module.project_prepared_cohort(*args, **kwargs)


def mixed_source(tmp_path):
    manifest, clock, macro = prepared_edition(tmp_path)
    meta = json.loads(manifest.read_text())
    meta.update(status="completed_with_errors", candidate_count=3, failed_assets=1)
    meta["configuration"]["markets"] = ["US", "CN"]
    meta["assets"].append(
        dict(market="CN", symbol="000001.SZ", state="failed", detail="Error de fuente")
    )
    manifest.write_text(json.dumps(meta))
    return manifest, clock, macro


def test_market_projection_keeps_every_selected_candidate_and_parent_failures(tmp_path):
    source, _, _ = mixed_source(tmp_path)
    original_hash = sha256(source)
    output = tmp_path / "view/manifest.json"
    result = project(source, output, markets=("US",))
    assert result["status"] == "completed"
    assert result["scope"] == "market_projection"
    assert result["candidate_count"] == 2
    assert [(a["symbol"], a["state"]) for a in result["assets"]] == [
        ("A", "prepared"),
        ("B", "missing_modalities"),
    ]
    assert result["parent_preparation"]["sha256"] == original_hash
    assert result["parent_preparation"]["candidate_count"] == 3
    assert result["parent_preparation"]["failed_assets"] == 1
    assert result["parent_preparation"]["excluded_markets"] == ["CN"]
    assert result["configuration"]["markets"] == ["US"]
    assert sha256(source) == original_hash
    before = output.stat().st_mtime_ns
    assert project(source, output, markets=("US",)) == result
    assert output.stat().st_mtime_ns == before


def test_selected_market_with_failed_candidate_cannot_be_marked_complete(tmp_path):
    source, _, _ = mixed_source(tmp_path)
    output = tmp_path / "view/manifest.json"
    with pytest.raises(ValueError, match="fallos|fallido"):
        project(source, output, markets=("CN",))
    assert not output.exists()


@pytest.mark.parametrize(
    "change", [dict(status="running"), dict(candidate_count=4), dict(failed_assets=0)]
)
def test_projection_rejects_unfinished_or_unreconciled_parent(tmp_path, change):
    source, _, _ = mixed_source(tmp_path)
    source.write_text(json.dumps({**json.loads(source.read_text()), **change}))
    with pytest.raises(ValueError, match="origen|preparación|candidatos|recuento"):
        project(source, tmp_path / "view/manifest.json", markets=("US",))


def test_projection_detects_changed_prepared_manifest(tmp_path):
    source, _, _ = mixed_source(tmp_path)
    path = tmp_path / "prepared/US/A/manifest.json"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="huella|cambiado"):
        project(source, tmp_path / "view/manifest.json", markets=("US",))


def test_frozen_view_rejects_a_new_parent_instead_of_overwriting(tmp_path):
    source, _, _ = mixed_source(tmp_path)
    output = tmp_path / "view/manifest.json"
    project(source, output, markets=("US",))
    before = sha256(output)
    source.write_text(source.read_text() + "\n")
    with pytest.raises(ValueError, match="edición|identidad"):
        project(source, output, markets=("US",))
    assert sha256(output) == before


def test_projection_flows_into_encoding_with_declared_market_scope(tmp_path):
    from mars_titan.data.corpus_encoding import encode_corpus
    from tests.data.test_cohort_samples import Encoders

    source, clock, macro = mixed_source(tmp_path)
    view = tmp_path / "view/manifest.json"
    project(source, view, markets=("US",))
    result = encode_corpus(
        view,
        tmp_path / "encoded",
        macros={"US": macro},
        encoders=Encoders(),
        clocks={"US": clock},
        context=2,
    )
    assert result["cohort_complete"] is True
    assert result["markets"] == ["US"]
    assert result["preparation_scope"] == "market_projection"
    assert result["parent_preparation"]["failed_assets"] == 1
    assert result["candidate_count"] == 2
    assert result["samples"] == 5


def test_nested_projection_cannot_hide_the_original_parent_errors(tmp_path):
    source, _, _ = mixed_source(tmp_path)
    view = tmp_path / "view/manifest.json"
    project(source, view, markets=("US",))
    with pytest.raises(ValueError, match="origen|vista"):
        project(view, tmp_path / "nested/manifest.json", markets=("US",))
    assert not (tmp_path / "nested/manifest.json").exists()
