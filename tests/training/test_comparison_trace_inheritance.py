"""Herencia de PC1 con trazas sobre una copia comprobada y un ajuste ya iniciado."""

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from mars_titan.data.storage import atomic_json, sha256
from mars_titan.training import comparison_campaign as campaign
from mars_titan.training import comparison_inheritance as migration
from mars_titan.training import comparison_input_migration as packed
from mars_titan.training.checkpoints import load_training_state, save_training_state
from tests.training.test_comparison_inheritance import source

FIXTURE_RELEASE = "7" * 40
PAUSED = "transformer/2019/seed-42"
PARENT = "titans_mac/2019/seed-42"
READER = "mars_titan_m3/2019/seed-42"


def test_pinned_release_keeps_the_numerical_training_and_window_modules():
    assert migration.TRACE_RELEASE == "d5f27f50073d12706c08130a155ea958d822b28b"
    for name in (
        "financial_run",
        "mars_titan_run",
        "titans_walk_forward",
        "mars_titan_walk_forward",
    ):
        module = "mars_titan.training." + name
        packed_code = packed.git_digest(campaign.ROOT, packed.PACKED_RELEASE, module)
        assert packed_code is not None
        assert packed.git_digest(campaign.ROOT, migration.TRACE_RELEASE, module) == packed_code


def test_the_qualified_batches_module_changes_only_prediction_writer_options():
    file = "src/mars_titan/data/batches.py"

    def at(commit):
        return subprocess.check_output(
            ["git", "show", f"{commit}:{file}"], cwd=campaign.ROOT, text=True
        )

    old = at(packed.PACKED_RELEASE)
    new = at(migration.TRACE_RELEASE)
    assert (
        packed.git_digest(campaign.ROOT, migration.TRACE_RELEASE, migration.traced.BATCHES_MODULE)
        == migration.traced.QUALIFIED_WRITER_SHA256
    )
    expected = old.replace(
        "import pyarrow.parquet as pq",
        "import pyarrow.parquet as pq\nfrom .comparison_prediction_storage import writer_options",
    ).replace(
        'writer = pq.ParquetWriter(temporary, table.schema, compression="zstd")',
        "options = writer_options(path, table.schema)\n"
        "                writer = pq.ParquetWriter(temporary, table.schema, "
        'compression="zstd", **options)',
    )
    assert ast.dump(ast.parse(expected)) == ast.dump(ast.parse(new))


def read(path):
    return json.loads(path.read_text())


def history(tmp_path, monkeypatch):
    doc, digest, old = source(tmp_path)
    doc["execution"]["neural_blocks"] = {f"2019/seed-{seed}": "references" for seed in doc["seeds"]}
    monkeypatch.setattr(migration, "TRACE_RELEASE", FIXTURE_RELEASE, raising=False)
    monkeypatch.setattr(migration.traced, "TRACE_RELEASE", FIXTURE_RELEASE, raising=False)
    monkeypatch.setattr(migration, "code_commit", lambda root: FIXTURE_RELEASE, raising=False)
    monkeypatch.setattr(migration, "code_changes", lambda root: None, raising=False)
    monkeypatch.setattr(migration, "GPU_DECLARATION", digest)
    atomic_json(old.authorization, dict(read(old.authorization), code_commit=packed.PACKED_RELEASE))
    for record in old.state.glob("queue/*.json"):
        original = read(record)
        original["attempts"][-1]["execution"] = dict(
            code_commit=packed.PACKED_RELEASE, code_changes_sha256=None, declaration_sha256=digest
        )
        atomic_json(record, original)
    ancestor = campaign.Layout(tmp_path / "prior", old.views)
    old_reader = old.output(READER)
    prior_reader = ancestor.output(READER)
    prior_reader.mkdir(parents=True)
    reader_report = read(old_reader / "run.json")
    module = "mars_titan.training.mars_titan_run"
    reader_report["request"]["code"][module] = packed.git_digest(
        campaign.ROOT, packed.WARMUP_RELEASE, module
    )
    atomic_json(old_reader / "run.json", reader_report)
    (prior_reader / "run.json").write_bytes((old_reader / "run.json").read_bytes())
    record = read(old.record(READER))
    execution = dict(
        code_commit=packed.WARMUP_RELEASE, declaration_sha256=digest, code_changes_sha256=None
    )
    record["attempts"][-1]["execution"] = execution
    atomic_json(ancestor.record(READER), record)
    origin = dict(
        code_commit=packed.WARMUP_RELEASE,
        declaration_sha256=digest,
        source_state=str(ancestor.state),
        source_record_sha256=sha256(ancestor.record(READER)),
    )
    atomic_json(old.record(READER), dict(record, source_execution=origin))
    atomic_json(old.state / "inheritance-status.json", dict(status="completed"))
    atomic_json(
        old.state / "inheritance.json",
        dict(
            kind="comparison_campaign_inheritance_v1",
            schema_version=1,
            code_commit=packed.PACKED_RELEASE,
            source_state=str(ancestor.state),
            source_declaration_sha256=digest,
            declaration_sha256=digest,
            final_test_opened=False,
            inherited={READER: dict(origin, status="completed")},
            omitted={},
        ),
    )
    folder = old.output(PAUSED)
    outer = read(folder / "run.json")
    request = outer["request"]
    idx = {}
    for part in ("train", "validation", "calibration", "evaluation"):
        index = old.indices / request["view_sha256"][:16] / (part + "-fixture")
        index.mkdir(parents=True)
        events = index / "events.parquet"
        events.write_bytes(part.encode())
        checksum = sha256(events)
        filepath = index / f"events-{checksum}.parquet"
        events.rename(filepath)
        atomic_json(
            index / "manifest.json",
            dict(
                kind="financial_observation_index",
                final_test_opened=False,
                identity=dict(phase=dict(partition=part), source_sha256=request["view_sha256"]),
                bytes=filepath.stat().st_size,
                events_path=filepath.name,
                events_sha256=checksum,
            ),
        )
        idx[part] = sha256(index / "manifest.json")
    identity = dict(
        implementation=request["code"],
        dataset_sha256=request["view_sha256"],
        final_test_opened=False,
        sources={part: dict(index_sha256=idx[part]) for part in ("train", "validation")},
    )
    cursor = dict(epoch=3, phase="train", event=123, stage="inputs")
    state = dict(
        global_step=42,
        cursor=cursor,
        model=dict(weight=torch.tensor([-0.0, 2.0])),
        optimizer=dict(state={0: dict(exp_avg=torch.tensor([-0.0, 0.25]))}),
        rng={"torch": torch.Generator().manual_seed(6).get_state()},
        run=dict(pending=["a"]),
        history=[],
        selection=None,
    )
    (folder / "fit/checkpoints/model.pt").unlink()
    best = save_training_state(
        folder / "fit/checkpoints", dict(state, global_step=0), identity=identity, best=True
    )
    ck = save_training_state(folder / "fit/checkpoints", state, identity=identity)
    fit = dict(
        kind="titans_financial_chronological_run",
        status="paused",
        final_test_opened=False,
        identity=identity,
        run_id=packed.run_id(identity),
        global_step=42,
        cursor=cursor,
        history=[],
        selection=None,
        best_checkpoint=dict(path="checkpoints/" + best.name, sha256=sha256(best)),
        recovery_checkpoint=dict(path="checkpoints/" + ck.name, sha256=sha256(ck)),
    )
    atomic_json(folder / "fit/run.json", fit)
    win = dict(request=request, indices=idx, fit_run_id=fit["run_id"])
    atomic_json(
        folder / "run.json",
        dict(
            outer,
            identity=win,
            run_id=packed.run_id(win),
            status="paused",
            fit=None,
            checkpoint=None,
            predictions={},
        ),
    )
    record = read(old.record(PAUSED))
    atomic_json(old.record(PAUSED), dict(record, status="paused"))
    return doc, digest, old, campaign.Layout(tmp_path / "traces", old.views), state


def migrate(doc, digest, old, target):
    return migration.inherit(
        doc, doc, digest, digest, old, target, commit=FIXTURE_RELEASE, root=campaign.ROOT
    )


@pytest.mark.parametrize("problem", ["source_commit", "target_changes", "declaration", "pc2"])
def test_only_the_clean_pc1_release_can_inherit_the_packed_campaign(tmp_path, monkeypatch, problem):
    doc, digest, old, target, _ = history(tmp_path, monkeypatch)
    if problem == "source_commit":
        atomic_json(
            old.authorization, dict(read(old.authorization), code_commit=packed.WARMUP_RELEASE)
        )
    elif problem == "target_changes":
        monkeypatch.setattr(migration, "code_changes", lambda _: "0" * 64, raising=False)
    elif problem == "declaration":
        digest = "0" * 64
    else:
        doc["execution"]["neural_blocks"]["2019/seed-42"] = "mars_titan"
    with pytest.raises(ValueError):
        migrate(doc, digest, old, target)
    assert read(old.authorization)["training_allowed"] and not target.authorization.exists()


def test_resume_keeps_the_checkpoint_and_its_indices_bit_for_bit(tmp_path, monkeypatch):
    doc, digest, old, target, state = history(tmp_path, monkeypatch)
    receipt = migrate(doc, digest, old, target)
    assert (
        not target.output(PARENT).is_symlink()
        and (target.output(PARENT) / "fit/checkpoints/model.pt").is_file()
    )
    assert target.output(READER).is_symlink()
    assert (
        read(target.record(READER))["source_execution"]["previous"]
        == read(old.record(READER))["source_execution"]
    )
    fit = read(target.output(PAUSED) / "fit/run.json")
    original = read(old.output(PAUSED) / "fit/run.json")
    assert fit == original and not target.output(PAUSED).is_symlink()
    ck = fit["recovery_checkpoint"]["sha256"]
    assert packed.same_state(
        state,
        load_training_state(
            target.output(PAUSED) / "fit/checkpoints",
            expected_identity=fit["identity"],
            expected_sha256=ck,
        ),
    )
    assert receipt["inherited"][PAUSED]["recovery_checkpoint"]["sha256"] == ck
    assert (target.indices).is_dir() and not target.indices.is_symlink()
    assert migrate(doc, digest, old, target) == receipt
    assert not read(old.authorization)["training_allowed"] and not target.authorization.exists()


def test_paused_reader_keeps_its_original_parent_and_permission(tmp_path, monkeypatch):
    doc, digest, old, target, _ = history(tmp_path, monkeypatch)
    record = read(old.record(READER))
    record["status"] = "paused"
    record.pop("source_execution")
    record["attempts"][-1].update(
        status="paused",
        execution=dict(
            code_commit=packed.PACKED_RELEASE, code_changes_sha256=None, declaration_sha256=digest
        ),
    )
    atomic_json(old.record(READER), record)
    folder = old.output(READER)
    atomic_json(
        folder / "run.json",
        dict(read(folder / "run.json"), status="paused", fit=None, checkpoint=None, predictions={}),
    )
    receipt = read(old.state / "inheritance.json")
    del receipt["inherited"][READER]
    atomic_json(old.state / "inheritance.json", receipt)
    with pytest.raises(ValueError, match="lector pausado"):
        migrate(doc, digest, old, target)
    assert read(old.authorization)["training_allowed"] and not target.state.exists()


@pytest.mark.parametrize(
    "error",
    [
        "unproven_history",
        "new_code",
        "damaged_checkpoint",
        "damaged_best",
        "wrong_cursor",
        "wrong_index",
    ],
)
def test_incompatible_attempt_never_closes_the_source(tmp_path, monkeypatch, error):
    doc, digest, old, target, _ = history(tmp_path, monkeypatch)
    folder = old.output(PAUSED)
    fit = read(folder / "fit/run.json")
    if error == "unproven_history":
        entry = read(old.state / "inheritance.json")
        del entry["inherited"][READER]
        atomic_json(old.state / "inheritance.json", entry)
    elif error == "new_code":
        path = folder / "run.json"
        doc0 = read(path)
        doc0["request"]["code"]["mars_titan.training.financial_run"] = packed.git_digest(
            campaign.ROOT, packed.WARMUP_RELEASE, "mars_titan.training.financial_run"
        )
        atomic_json(path, doc0)
    elif error == "damaged_checkpoint":
        (folder / "fit" / fit["recovery_checkpoint"]["path"]).write_bytes(b"estado incompleto")
    elif error == "damaged_best":
        best = folder / "fit" / fit["best_checkpoint"]["path"]
        best.write_bytes(b"x" * best.stat().st_size)
    elif error == "wrong_cursor":
        fit["cursor"]["event"] += 1
        atomic_json(folder / "fit/run.json", fit)
    else:
        events = next(old.indices.rglob("events-*.parquet"))
        events.write_bytes(b"x" * events.stat().st_size)
    with pytest.raises(ValueError):
        migrate(doc, digest, old, target)
    assert read(old.authorization)["training_allowed"] is True and not target.authorization.exists()


@pytest.mark.parametrize("cut", ["close", "complete"])
def test_retry_finishes_an_interrupted_publication_without_copying_again(
    tmp_path, monkeypatch, cut
):
    doc, digest, old, target, _ = history(tmp_path, monkeypatch)
    write = migration.atomic_json

    def interruption(path, value):
        if (
            cut == "close"
            and path == old.authorization
            and value.get("status") == "continued_in_another_state"
        ) or (
            cut == "complete"
            and path == target.state / "inheritance-status.json"
            and value.get("status") == "completed"
        ):
            raise OSError("interrupción de prueba")
        return write(path, value)

    with monkeypatch.context() as moment:
        moment.setattr(migration, "atomic_json", interruption)
        with pytest.raises(OSError, match="interrupción"):
            migrate(doc, digest, old, target)
    assert read(old.authorization)["training_allowed"] == (cut == "close")
    assert not target.authorization.exists()
    old_time = (target.output(PAUSED) / "fit/run.json").stat().st_mtime_ns
    receipt = migrate(doc, digest, old, target)
    assert (target.output(PAUSED) / "fit/run.json").stat().st_mtime_ns == old_time
    assert receipt == read(target.state / "inheritance.json")
    assert not read(old.authorization)["training_allowed"]


@pytest.mark.parametrize("what", ["lineage", "checkpoint", "bundle", "log"])
def test_a_damaged_prepared_copy_cannot_close_the_source(tmp_path, monkeypatch, what):
    doc, digest, old, target, _ = history(tmp_path, monkeypatch)
    atomic_json(old.bundle, dict(cohorts=campaign.cohort_identity(doc, old)))
    old.log(PAUSED).parent.mkdir(exist_ok=True)
    old.log(PAUSED).write_text("Intento pausado\n")
    previous = migration.atomic_json

    def stop(path, value):
        if path == old.authorization:
            raise OSError("interrumpido")
        return previous(path, value)

    with monkeypatch.context() as changed:
        changed.setattr(migration, "atomic_json", stop)
        with pytest.raises(OSError, match="interrumpido"):
            migrate(doc, digest, old, target)
    if what == "lineage":
        copied = read(target.record(PAUSED))
        copied["source_execution"]["code_commit"] = "a" * 40
        atomic_json(target.record(PAUSED), copied)
    elif what == "checkpoint":
        path = target.output(PAUSED) / "fit"
        (path / read(path / "run.json")["recovery_checkpoint"]["path"]).write_bytes(
            b"copia incompleta"
        )
    elif what == "bundle":
        target.bundle.write_text("{}")
    else:
        target.log(PAUSED).write_text("registro incompleto")
    with pytest.raises(ValueError):
        migrate(doc, digest, old, target)
    assert read(old.authorization)["training_allowed"] and not target.authorization.exists()


def interrupt_copy_worker():
    doc, digest, old_path, out_path, views = json.load(sys.stdin)
    migration.GPU_DECLARATION = digest
    migration.TRACE_RELEASE = FIXTURE_RELEASE
    migration.code_commit = lambda _: FIXTURE_RELEASE
    migration.code_changes = lambda _: None
    previous = migration._copy_completed_parent

    def stop_after_copy(source, destination):
        previous(source, destination)
        os._exit(37)

    migration._copy_completed_parent = stop_after_copy
    migrate(
        doc, digest, campaign.Layout(Path(old_path), views), campaign.Layout(Path(out_path), views)
    )


def test_a_killed_copy_reclaims_its_unpublished_files_on_retry(tmp_path, monkeypatch):
    doc, digest, old, target, _ = history(tmp_path, monkeypatch)
    code = "from tests.training.test_comparison_trace_inheritance import interrupt_copy_worker"
    result = subprocess.run(
        [sys.executable, "-c", code + "\ninterrupt_copy_worker()"],
        input=json.dumps([doc, digest, str(old.state), str(target.state), str(old.views)]),
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 37
    assert not target.state.exists() and read(old.authorization)["training_allowed"]
    prepared = set(target.state.parent.glob(f".{target.state.name}.*"))
    assert prepared
    migrate(doc, digest, old, target)
    assert not any(path.exists() for path in prepared)


@pytest.mark.parametrize(
    ("origin", "problem"),
    [
        ("warmup", None),
        ("gpu", None),
        ("fresh", None),
        ("warmup", "other_module"),
        ("fresh", "other_module"),
        ("gpu", "previous_digest"),
        ("fresh", "previous_digest"),
        ("warmup", "unqualified_writer"),
        ("fresh", "unqualified_writer"),
        ("fresh", "unapproved_commit_blob"),
        ("fresh", "wrong_qualified_runtime"),
        ("fit", None),
        ("paused", None),
    ],
)
def test_storage_equivalence_is_specific_to_completed_readers(
    tmp_path, monkeypatch, origin, problem
):
    doc, digest, old, target, _ = history(tmp_path, monkeypatch)
    name = "mars_titan.data.batches"
    original_digest = packed.git_digest(campaign.ROOT, packed.PACKED_RELEASE, name)
    runtime = tmp_path / "qualified-runtime"
    modules = runtime / "src/mars_titan"
    (modules / "data").mkdir(parents=True)
    for folder in ("training", "models"):
        (modules / folder).symlink_to(
            campaign.ROOT / "src/mars_titan" / folder, target_is_directory=True
        )
    (runtime / "configs").symlink_to(campaign.ROOT / "configs", target_is_directory=True)
    writer = modules / "data/batches.py"
    writer.write_bytes(b"Referencia temporal de un escritor cualificado")
    qualified_digest = sha256(writer)
    monkeypatch.setattr(
        migration.traced, "QUALIFIED_WRITER_SHA256", qualified_digest, raising=False
    )
    if problem == "unqualified_writer":
        writer.write_bytes(b"Otra implementacion")
    if problem == "unapproved_commit_blob":
        monkeypatch.setattr(migration.traced, "QUALIFIED_WRITER_SHA256", "d" * 64)
    if problem == "wrong_qualified_runtime":
        monkeypatch.setattr(migration.traced, "TRACE_RELEASE", "d" * 40, raising=False)
    git_digest = packed.git_digest

    def runtime_digest(_, commit, module):
        if commit == FIXTURE_RELEASE and module == name:
            return qualified_digest
        return git_digest(campaign.ROOT, commit, module)

    monkeypatch.setattr(packed, "git_digest", runtime_digest)
    key = {
        "gpu": READER,
        "warmup": READER,
        "fresh": "mars_titan_m3/2019/seed-43",
        "fit": PARENT,
        "paused": PAUSED,
    }[origin]
    path = old.output(key) / "run.json"
    report = read(path)
    if origin in ("gpu", "warmup", "fresh"):
        report["kind"] = "mars_titan_walk_forward_window"
    code = report["request"]["code"]
    code[name] = qualified_digest if problem == "previous_digest" else original_digest
    if problem == "other_module":
        code["mars_titan.models.baselines.momentum_reversal"] = "f" * 64
    if origin in ("warmup", "gpu"):
        commit = packed.WARMUP_RELEASE if origin == "warmup" else packed.GPU_RELEASE
        module = "mars_titan.training.mars_titan_run"
        code[module] = git_digest(campaign.ROOT, commit, module)
        prior_doc = read(old.state / "inheritance.json")
        previous = campaign.Layout(Path(prior_doc["source_state"]), old.views)
        rec = read(previous.record(key))
        rec["attempts"][-1]["execution"]["code_commit"] = commit
        atomic_json(previous.record(key), rec)
        provenance = dict(
            read(old.record(key))["source_execution"],
            code_commit=commit,
            source_record_sha256=sha256(previous.record(key)),
        )
        atomic_json(old.record(key), dict(rec, source_execution=provenance))
        prior_doc["inherited"][key] = dict(provenance, status="completed")
        atomic_json(old.state / "inheritance.json", prior_doc)
        atomic_json(previous.output(key) / "run.json", report)
    atomic_json(path, report)
    if problem or origin in ("fit", "paused"):
        with pytest.raises(ValueError, match="código|escritor"):
            migration.inherit(
                doc, doc, digest, digest, old, target, commit=FIXTURE_RELEASE, root=runtime
            )
        assert read(old.authorization)["training_allowed"]
    else:
        result = migration.inherit(
            doc, doc, digest, digest, old, target, commit=FIXTURE_RELEASE, root=runtime
        )
        assert key in result["inherited"] and target.output(key).is_symlink()
        assert result["inherited"][key]["prediction_writer"] == dict(
            module=name, source_sha256=original_digest, runtime_sha256=qualified_digest
        )
