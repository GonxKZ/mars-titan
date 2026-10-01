"""Integración pequeña de la búsqueda con Ridge y XGBoost reales en CUDA."""

import importlib.util
import json

import pytest

from tests.training.test_reference_run import training_corpus

pytestmark = pytest.mark.skipif(
    any(importlib.util.find_spec(name) is None for name in ("cupy", "xgboost")),
    reason="Requiere los paquetes de boosting y CUDA",
)


def test_real_backends_share_the_population_and_reuse_finished_models(tmp_path):
    from mars_titan.training.tabular_search import run_tabular_search

    manifest = training_corpus(tmp_path / "data")
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            dict(
                schema_version=1,
                ridge_alphas=[1.0],
                depths=[2],
                bins=[16],
                rates=[0.1],
                rounds=2,
                search_seed=42,
                finalist_seeds=[42, 43],
                batch_size=5,
                max_batch_bytes=64 * 1024**2,
                max_host_cache_bytes=64 * 1024**2,
                on_host=False,
                checkpoint_interval=1,
                final_test_opened=False,
            )
        )
    )
    output = tmp_path / "study"
    report = run_tabular_search(config, manifest, output)
    assert report["status"] == "completed" and report["completed_runs"] == 3
    assert report["counts"] == {"train": 12, "validation": 6}
    assert report["scope"] == "development_snapshot" and not report["final_test_opened"]
    hashes = [row["report_sha256"] for row in report["runs"]]
    resumed = run_tabular_search(config, manifest, output, resume=True)
    assert [row["report_sha256"] for row in resumed["runs"]] == hashes
