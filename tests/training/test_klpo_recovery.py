"""Interrupciones antes del primer recibo y cambios al terminar el último ajuste."""

import json

import pytest

from mars_titan.data.storage import sha256
from mars_titan.environments import corpus_source
from mars_titan.training import klpo_queue, predictive_parents, predictive_run, predictive_study
from mars_titan.training.checkpoints import StopRequest
from tests.training.test_klpo_study import config, klpo_case
from tests.training.test_predictive_parents import setup


def inputs(tmp_path, monkeypatch):
    ordered, parent, _ = setup(tmp_path / "sources")
    original = tmp_path / "sources/data/manifest.json"
    meta = json.loads(ordered.read_text())
    proof = dict(
        manifest=str(original),
        manifest_sha256=sha256(original),
        counts=meta["counts"],
        parents={"ridge": dict(report=str(parent), sha256=sha256(parent))},
    )
    monkeypatch.setattr(klpo_queue, "selected_parents", lambda *args, **kwargs: proof)
    design = tmp_path / "design.json"
    design.write_text(json.dumps(config() | dict(seeds=[42], betas=[0.1], epochs=1, batch_size=5)))
    return design, ordered, parent


@pytest.mark.parametrize("level", ["queue", "study", "ordered", "case", "cache"])
def test_initial_receipt_failure_can_resume_without_discarding_other_files(
    tmp_path, monkeypatch, level
):
    design, ordered, parent = inputs(tmp_path, monkeypatch)
    output = tmp_path / "output"
    stop = StopRequest()
    stop.request_stop()
    if level == "queue":
        module, receipt = klpo_queue, "summary.json"

        def action(resume):
            return klpo_queue.run_queue(design, parent, parent, output, stop=stop)
    elif level == "study":
        module, receipt = predictive_study, "summary.json"

        def action(resume):
            return predictive_study.run_predictive_study(
                design, ordered, parent, output, resume=resume, stop=stop
            )
    elif level == "ordered":
        module, receipt = corpus_source, "progress.json"
        original = tmp_path / "sources/data/manifest.json"

        def action(resume):
            return corpus_source.prepare_causal_corpus(original, output, resume=resume, stop=stop)
    elif level == "cache":
        module, receipt = predictive_parents, "progress.json"

        def action(resume):
            return predictive_parents.prepare_parent_cache(ordered, parent, output, resume=resume)
    else:
        from mars_titan.training.predictive_parents import prepare_parent_cache

        cache = tmp_path / "cache"
        prepare_parent_cache(ordered, parent, cache)
        from mars_titan.training import run_receipts

        module, receipt = run_receipts, "initialization.json"

        def action(resume):
            return predictive_run.run_predictive_case(
                ordered, cache / "manifest.json", output, klpo_case(), resume=resume, stop=stop
            )

    original_write = module.atomic_json

    def fail_once(path, data):
        if path == output / receipt:
            raise OSError("Fallo antes de confirmar el recibo inicial")
        return original_write(path, data)

    monkeypatch.setattr(module, "atomic_json", fail_once)
    with pytest.raises(OSError, match="recibo inicial"):
        action(False)
    monkeypatch.setattr(module, "atomic_json", original_write)
    assert action(True)["status"] == ("completed" if level == "cache" else "paused")


@pytest.mark.parametrize("level", ["queue", "study"])
def test_changed_configuration_after_last_child_cannot_be_completed(tmp_path, monkeypatch, level):
    design, ordered, parent = inputs(tmp_path, monkeypatch)
    module = klpo_queue if level == "queue" else predictive_study
    name = "run_predictive_study" if level == "queue" else "run_predictive_case"
    original_run = getattr(module, name)
    calls = 0

    def change_after_last(*args, **kwargs):
        nonlocal calls
        result = original_run(*args, **kwargs)
        calls += 1
        if calls == (1 if level == "queue" else 6):
            design.write_text(design.read_text() + "\n")
        return result

    monkeypatch.setattr(module, name, change_after_last)
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="cambiado"):
        if level == "queue":
            module.run_queue(design, parent, parent, output)
        else:
            module.run_predictive_study(design, ordered, parent, output)
    assert json.loads((output / "summary.json").read_text())["status"] == "failed"
