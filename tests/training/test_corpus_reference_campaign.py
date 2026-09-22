import importlib
import json
import os
import signal
from pathlib import Path

import pytest

from mars_titan.data.storage import sha256
from tests.training.test_reference_run import training_corpus


def setup(tmp_path, *, scope="development_snapshot", arms=("US",)):
    manifest = training_corpus(tmp_path / "data")
    recipe = json.loads(Path("configs/baselines/expanded-reference-variants.json").read_text())
    recipe.update(models=["rnn"], seeds=[42], losses=["mse"], learning_rates=[0.001], epochs=2)
    (tmp_path / "recipe.json").write_text(json.dumps(recipe))
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            dict(
                schema_version=2,
                scope=scope,
                arms=list(arms),
                recipe="recipe.json",
                batch_size=5,
                checkpoint_seconds=900,
                pooled_weightings=["natural"],
                final_test_opened=False,
            )
        )
    )
    return config, manifest


def module():
    return importlib.import_module("mars_titan.training.reference_campaign")


def test_completed_case_is_not_retrained_after_campaign_pause(tmp_path, monkeypatch):
    config, manifest = setup(tmp_path)
    engine = module()
    runner = engine.run_reference_case
    calls = []

    def interrupt_after_first(*args, **kwargs):
        report = runner(*args, **kwargs)
        calls.append(report["identity"]["case"])
        os.kill(os.getpid(), signal.SIGTERM)
        return report

    monkeypatch.setattr(engine, "run_reference_case", interrupt_after_first)
    output = tmp_path / "campaign"
    first = engine.run_reference_campaign(config, manifest, output)
    assert first["status"] == "paused" and first["completed_runs"] == 1
    initial = output / first["runs"][0]["path"] / "run.json"
    digest, modified = sha256(initial), initial.stat().st_mtime_ns
    monkeypatch.setattr(engine, "run_reference_case", runner)
    final = engine.run_reference_campaign(config, manifest, output, resume=True)
    assert final["status"] == "completed" and final["completed_runs"] == final["planned_runs"] == 3
    assert sha256(initial) == digest and initial.stat().st_mtime_ns == modified
    assert len(calls) == 1


def test_partial_corpus_cannot_be_declared_full_campaign(tmp_path):
    config, manifest = setup(tmp_path, scope="full_corpus")
    with pytest.raises(ValueError, match="complet|edición|alcance"):
        module().run_reference_campaign(config, manifest, tmp_path / "campaign")
    assert not (tmp_path / "campaign").exists()


def test_missing_market_is_not_replaced_with_another_arm(tmp_path):
    config, manifest = setup(tmp_path, arms=("US", "CN", "US+CN"))
    with pytest.raises(ValueError, match="mercado|brazo|población"):
        module().run_reference_campaign(config, manifest, tmp_path / "campaign")
    assert not (tmp_path / "campaign").exists()


def test_pooled_view_preserves_exact_union_and_market_weights(tmp_path):
    config, _ = setup(tmp_path)
    manifest = training_corpus(tmp_path / "pooled", markets=("US", "CN"))
    engine = module()
    views = engine.campaign_views(manifest, ["US", "CN", "US+CN"])

    def keys(view):
        return {(a["market"], a["symbol"]) for a in view["assets"]}

    assert keys(views["US+CN"]) == keys(views["US"]) | keys(views["CN"])
    assert views["US+CN"]["counts"] == {"train": 24, "validation": 12}
    from mars_titan.training.reference_run import training_weights

    weights = training_weights(
        [{"market": "US", "counts": {"train": 12}}, {"market": "CN", "counts": {"train": 4}}],
        "balanced_markets",
    )
    assert weights == {"US": pytest.approx(2 / 3), "CN": 2.0}
    assert 12 * weights["US"] == 4 * weights["CN"] == 8.0


@pytest.mark.parametrize("change", ["source", "environment"])
def test_campaign_cannot_mix_completed_cases_with_a_new_model_environment(
    tmp_path, monkeypatch, change
):
    config, manifest = setup(tmp_path)
    recipe_path = tmp_path / "recipe.json"
    recipe = json.loads(recipe_path.read_text())
    recipe["losses"] = ["mse", "mae"]
    recipe_path.write_text(json.dumps(recipe))
    engine = module()
    runner = engine.run_reference_case

    def pause(*args, **kwargs):
        report = runner(*args, **kwargs)
        os.kill(os.getpid(), signal.SIGTERM)
        return report

    monkeypatch.setattr(engine, "run_reference_case", pause)
    output = tmp_path / "campaign"
    paused = engine.run_reference_campaign(config, manifest, output)
    assert paused["completed_runs"] == 1
    monkeypatch.setattr(engine, "run_reference_case", runner)
    neural = importlib.import_module("mars_titan.training.reference_run")
    if change == "source":
        digest = neural.sha256
        monkeypatch.setattr(
            neural, "sha256", lambda p: "0" * 64 if p.name == "profiling.py" else digest(p)
        )
    else:
        monkeypatch.setattr(neural.torch, "__version__", "changed-environment")
    with pytest.raises(ValueError):
        engine.run_reference_campaign(config, manifest, output, resume=True)
    assert not (output / paused["runs"][1]["path"]).exists()
