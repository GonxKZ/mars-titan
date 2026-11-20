"""Compatibilidad explícita de los ZIP de trazas de PC1 con la cola de PC2."""

from types import SimpleNamespace

import pytest

from mars_titan.evaluation import comparison_campaign_analysis as analysis
from mars_titan.evaluation.comparison_bundle import ResultsBundle
from tests.evaluation.test_comparison_campaign_analysis import packages

FIXTURE_RELEASE = "7" * 40


def archived(commit, machine, **change):
    provenance = dict(
        training_commit=commit,
        code_changes_sha256=None,
        authorized_declaration_sha256=analysis.GPU_DECLARATION,
        declaration_sha256=analysis.PACKED_DECLARATION,
    )
    return SimpleNamespace(
        manifest=dict(provenance=dict(provenance, **change), machine=machine, cohorts={}, jobs={})
    )


@pytest.mark.parametrize(
    "pc2", [analysis.GPU_RELEASE, analysis.WARMUP_RELEASE, analysis.PACKED_RELEASE]
)
def test_only_declared_traces_pair_can_join_pc2(monkeypatch, pc2):
    monkeypatch.setattr(analysis, "TRACE_RELEASE", FIXTURE_RELEASE, raising=False)
    pair = [archived(FIXTURE_RELEASE, "references"), archived(pc2, "mars_titan")]
    with pytest.raises(ValueError, match="código|commit"):
        analysis._merge(pair, verified_packed_inputs=True)
    assert analysis._merge(pair, verified_learning_traces=True) == {}
    assert analysis._merge(pair[::-1], verified_learning_traces=True) == {}
    bad = archived(pc2, "mars_titan", code_changes_sha256="1" * 64)
    with pytest.raises(ValueError):
        analysis._merge([pair[0], bad], verified_learning_traces=True)
    with pytest.raises(ValueError):
        analysis._merge(
            [archived(FIXTURE_RELEASE, "mars_titan"), archived(pc2, "references")],
            verified_learning_traces=True,
        )
    with pytest.raises(ValueError):
        analysis._merge([pair[0], archived("c" * 40, "mars_titan")], verified_learning_traces=True)


def test_traces_zip_keeps_its_runtime_and_exact_comparison_scope(tmp_path, monkeypatch):
    monkeypatch.setattr(analysis, "TRACE_RELEASE", FIXTURE_RELEASE, raising=False)
    first, second = packages(tmp_path, commits=(FIXTURE_RELEASE, analysis.PACKED_RELEASE))
    with ResultsBundle(first) as archive:
        provenance = archive.manifest["provenance"]
    monkeypatch.setattr(analysis, "GPU_DECLARATION", provenance["authorized_declaration_sha256"])
    monkeypatch.setattr(analysis, "PACKED_DECLARATION", provenance["declaration_sha256"])
    result = analysis.analyze(
        [first, second], tmp_path / "result", plots=False, verified_learning_traces=True
    )
    data = result["provenance"]
    assert data["training_commit"] is None and data["verified_learning_traces"] is True
    assert data["verified_packed_inputs"] is False
    assert {row["machine"]: row["training_commit"] for row in data["archives"]} == dict(
        references=FIXTURE_RELEASE, mars_titan=analysis.PACKED_RELEASE
    )
    assert "comparison_trace_bundle.py" in result["analysis_implementation"]["files"]
