"""Continuación en otro estado sin reescribir resultados ni perder el padre de un lector."""

import copy
import json
from pathlib import Path

import pytest

from mars_titan.data.storage import atomic_json, sha256
from mars_titan.training import comparison_campaign as campaign
from mars_titan.training.comparison_inheritance import inherit
from tests.evaluation.comparison_results_fixtures import campaign_state


def source(tmp_path, *, paused=False):
    declaration, digest, layout = campaign_state(tmp_path / "original", reader=True)
    root = Path(campaign.ROOT)
    for machine in campaign.MACHINES:
        for job in campaign.plan(declaration, machine):
            path = layout.output(job.key) / "run.json"
            report = json.loads(path.read_text())
            report["request"].update(
                recipe_sha256=sha256(
                    root / declaration["recipes"][declaration["models"][job.model]["recipe"]]
                ),
                code={
                    "mars_titan.models.baselines.momentum_reversal": sha256(
                        root / "src/mars_titan/models/baselines/momentum_reversal.py"
                    )
                },
            )
            if job.needs:
                report["request"]["parent"]["run_sha256"] = sha256(
                    layout.output(job.needs) / "run.json"
                )
                report["request"]["parent"]["path"] = str(layout.output(job.needs).resolve())
            atomic_json(path, report)
    if paused:
        path = layout.record("mars_titan_m3/2019/seed-42")
        record = json.loads(path.read_text())
        record["status"] = "paused"
        atomic_json(path, record)
        path = layout.output("mars_titan_m3/2019/seed-42") / "run.json"
        report = json.loads(path.read_text())
        report["status"] = "paused"
        atomic_json(path, report)
    return declaration, digest, layout


def perform(tmp_path, old, digest, original, new=None):
    target = campaign.Layout(tmp_path / "continued", original.views)
    result = inherit(
        old, new or old, digest, "b" * 64, original, target, commit="c" * 40, root=campaign.ROOT
    )
    return result, target


def test_completed_parents_keep_their_resolved_paths_and_paused_readers_are_copied(tmp_path):
    old, digest, original = source(tmp_path, paused=True)
    reader = "mars_titan_m3/2019/seed-42"
    previous = {"code_commit": "d" * 40, "source_record_sha256": "e" * 64}
    record = campaign.read_record(original, reader)
    atomic_json(original.record(reader), dict(record, source_execution=previous))
    before = (original.output(reader) / "run.json").read_bytes()
    receipt, target = perform(tmp_path, old, digest, original)
    assert len(receipt["inherited"]) == 6
    assert target.output("titans_mac/2019/seed-42").resolve() == original.output(
        "titans_mac/2019/seed-42"
    )
    assert not target.output(reader).is_symlink()
    assert (target.output(reader) / "run.json").read_bytes() == before
    (target.output(reader) / "run.json").write_text("copia modificada")
    assert (original.output(reader) / "run.json").read_bytes() == before
    record = campaign.read_record(target, reader)
    assert record["source_execution"]["code_commit"] == "a" * 40
    assert record["source_execution"]["source_record_sha256"] == sha256(original.record(reader))
    assert record["source_execution"]["previous"] == previous
    assert json.loads(original.authorization.read_text())["training_allowed"] is False
    assert not target.authorization.exists()


def test_changed_training_budget_is_not_reused(tmp_path):
    old, digest, original = source(tmp_path)
    new = copy.deepcopy(old)
    new["stopping"]["fit"]["max_epochs"] = 2
    with pytest.raises(ValueError, match="presupuesto|parada|contrato"):
        perform(tmp_path, old, digest, original, new)
    assert json.loads(original.authorization.read_text())["training_allowed"] is True


def test_an_active_record_prevents_inheritance(tmp_path):
    old, digest, original = source(tmp_path)
    path = original.record("transformer/2019/seed-42")
    record = json.loads(path.read_text())
    record["status"] = "running"
    atomic_json(path, record)
    with pytest.raises(ValueError, match="activo|paus"):
        perform(tmp_path, old, digest, original)
    assert not (tmp_path / "continued").exists()


def test_modified_recipe_or_numerical_code_is_not_accepted(tmp_path):
    old, digest, original = source(tmp_path)
    path = original.output("transformer/2019/seed-42") / "run.json"
    report = json.loads(path.read_text())
    report["request"]["code"]["mars_titan.models.baselines.momentum_reversal"] = "0" * 64
    atomic_json(path, report)
    with pytest.raises(ValueError, match="código"):
        perform(tmp_path, old, digest, original)


@pytest.mark.parametrize(
    ("scope", "status", "module_hash", "allowed"),
    [
        (True, "completed", "original", True),
        (True, "paused", "original", False),
        (False, "completed", "original", False),
        (True, "completed", "other", False),
    ],
)
def test_only_a_completed_pinned_readout_can_cross_the_warmup_revision(
    tmp_path, scope, status, module_hash, allowed
):
    from mars_titan.training import comparison_inheritance as migration

    declaration, digest, layout = source(tmp_path)
    job = next(j for j in campaign.plan(declaration, "mars_titan") if j.runner == "reader")
    folder = layout.output(job.key)
    report = json.loads((folder / "run.json").read_text())
    report.update(status=status, kind="mars_titan_walk_forward_window")
    report["request"]["code"][migration.READOUT_MODULE] = (
        migration.PREVIOUS_READOUT if module_hash == "original" else "0" * 64
    )
    args = (
        folder,
        report,
        job,
        declaration["models"][job.model],
        declaration,
        campaign.ROOT,
        sha256(layout.cohort(job.year) / "manifest.json"),
    )
    if allowed:
        migration._compatible_report(*args, warmup_release=scope)
    else:
        with pytest.raises(ValueError, match="código"):
            migration._compatible_report(*args, warmup_release=scope)


def test_an_existing_destination_is_preserved(tmp_path):
    old, digest, original = source(tmp_path)
    (tmp_path / "continued").mkdir()
    (tmp_path / "continued/keep.txt").write_text("conservar")
    with pytest.raises(ValueError, match="existe"):
        perform(tmp_path, old, digest, original)
    assert (tmp_path / "continued/keep.txt").read_text() == "conservar"


def test_inherited_indices_are_accepted_by_the_real_reader_and_do_not_alias_the_source(
    tmp_path, monkeypatch
):
    from mars_titan.training.window_indices import SHARED_INDICES_ENV, shared_root

    old, digest, original = source(tmp_path)
    original.indices.mkdir()
    cached = original.indices / "cache.json"
    cached.write_text("índice original")
    _, target = perform(tmp_path, old, digest, original)
    monkeypatch.setenv(SHARED_INDICES_ENV, str(target.indices))
    assert shared_root() == target.indices
    assert not target.indices.is_symlink()
    (target.indices / "cache.json").write_text("índice de la continuación")
    assert cached.read_text() == "índice original"


def test_incomplete_inheritance_cannot_be_authorized(tmp_path):
    target = campaign.Layout(tmp_path / "continued")
    atomic_json(target.state / "inheritance-status.json", dict(status="failed"))
    with pytest.raises(ValueError, match="herencia"):
        campaign.authorize(target, "b" * 64, commit="c" * 40, note="Continuación de prueba")
    assert not target.authorization.exists()


@pytest.mark.parametrize("matches", [True, False])
def test_imported_data_receipt_follows_the_verified_cohorts(tmp_path, matches):
    old, digest, original = source(tmp_path)
    receipt = dict(
        cohorts=campaign.cohort_identity(old, original) if matches else {},
        declaration_sha256=digest,
        code_commit="a" * 40,
    )
    atomic_json(original.bundle, receipt)
    if matches:
        _, target = perform(tmp_path, old, digest, original)
        assert target.bundle.read_bytes() == original.bundle.read_bytes()
    else:
        with pytest.raises(ValueError, match="cohortes"):
            perform(tmp_path, old, digest, original)
        assert json.loads(original.authorization.read_text())["training_allowed"] is True


def test_inherited_state_cannot_be_authorized_with_another_configuration(tmp_path):
    old, digest, original = source(tmp_path)
    _, target = perform(tmp_path, old, digest, original)
    with pytest.raises(ValueError, match="herencia|identidad"):
        campaign.authorize(target, "d" * 64, commit="c" * 40, note="Otra declaración")
    campaign.authorize(target, "b" * 64, commit="c" * 40, note="Continuación de prueba")
    assert campaign.authorized(target, "b" * 64, commit="c" * 40, changes=None)


def test_inherited_and_fresh_records_can_be_analyzed_without_losing_execution_origin(tmp_path):
    from mars_titan.evaluation.comparison_bundle import export_results
    from mars_titan.evaluation.comparison_campaign_analysis import analyze

    old, digest, original = source(tmp_path)
    _, target = perform(tmp_path, old, digest, original)
    campaign.authorize(target, "b" * 64, commit="c" * 40, note="Continuación de prueba")
    for job in campaign.plan(old, "mars_titan"):
        record = campaign.read_record(target, job.key)
        record.pop("source_execution")
        for attempt in record["attempts"]:
            attempt["execution"] = dict(code_commit="c" * 40, declaration_sha256="b" * 64)
        atomic_json(target.record(job.key), record)
    paths = []
    for machine in campaign.MACHINES:
        path = tmp_path / (machine + ".zip")
        export_results(old, "b" * 64, target, machine, path)
        paths.append(path)
    result = analyze(paths, tmp_path / "analysis", plots=False)
    assert result["complete"] is True
    first = next(row for row in result["resources"] if row["model"] == "transformer")
    second = next(row for row in result["resources"] if row["model"] == "titans_mac")
    assert first["execution_lineage"][0]["code_commit"] == "a" * 40
    assert second["execution_lineage"] == []
    assert second["attempt_executions"][0]["code_commit"] == "c" * 40
