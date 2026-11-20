"""ZIP de resultados con límites, huellas y exclusión de los pesos."""

import hashlib
import json
import shutil
import struct
import zipfile

import pytest

from mars_titan.evaluation import comparison_bundle as bundle
from mars_titan.training import comparison_campaign as campaign
from tests.evaluation.comparison_results_fixtures import COMMIT, campaign_state


def test_results_zip_retains_predictions_curves_and_identity_without_models(tmp_path):
    declaration, digest, layout = campaign_state(tmp_path / "source")
    archive = tmp_path / "results.zip"

    result = bundle.export_results(declaration, digest, layout, "references", archive)

    assert result["completed_jobs"] == 2 and result["complete"] is True
    with zipfile.ZipFile(archive) as zipped:
        names = zipped.namelist()
        assert not any(name.endswith((".pt", ".pth", ".npz", "model.json")) for name in names)
        assert len([name for name in names if name.endswith(".parquet")]) == 6
    shutil.rmtree(layout.state)
    shutil.rmtree(layout.views)
    with bundle.ResultsBundle(archive) as opened:
        assert opened.manifest["provenance"]["training_commit"] == COMMIT
        job = opened.manifest["jobs"]["transformer/2019/seed-42"]
        history = opened.read_json(job["fit_report"])["history"]
        assert history[-1]["epoch"] == 1
        table = opened.read_table(job["tables"]["42"]["evaluation"])
        assert table.num_rows == 120 and "target" in table.column_names
        assert "quantile_0025" in table.column_names


def test_results_zip_does_not_overwrite_an_existing_file(tmp_path):
    declaration, digest, layout = campaign_state(tmp_path / "source")
    archive = tmp_path / "results.zip"
    archive.write_bytes(b"archivo anterior")
    with pytest.raises(ValueError, match="existe"):
        bundle.export_results(declaration, digest, layout, "references", archive)
    assert archive.read_bytes() == b"archivo anterior"


def test_zip_accepts_an_asset_with_the_longest_symbol_in_its_cohort(tmp_path):
    from mars_titan.evaluation.comparison_metrics import panel_from_table

    declaration, digest, layout = campaign_state(tmp_path / "source", symbol_width=64)
    archive = tmp_path / "results.zip"
    bundle.export_results(declaration, digest, layout, "references", archive)
    with bundle.ResultsBundle(archive) as opened:
        job = opened.manifest["jobs"]["transformer/2019/seed-42"]
        table = opened.read_table(job["tables"]["42"]["evaluation"])
        assert len(table["asset_id"][0].as_py()) == len("US/") + 64
        assert panel_from_table(table).rows == table.num_rows


def test_results_cannot_be_written_inside_the_training_state(tmp_path):
    declaration, digest, layout = campaign_state(tmp_path / "source")
    with pytest.raises(ValueError, match="origen|proteg|estado"):
        bundle.export_results(declaration, digest, layout, "references", layout.state / "out.zip")


def test_historical_zip_can_recover_the_asset_from_its_sample_id(tmp_path):
    import pyarrow.parquet as pq

    from mars_titan.data.storage import sha256
    from mars_titan.evaluation.comparison_metrics import panel_from_table

    declaration, digest, layout = campaign_state(tmp_path / "source")
    key = "transformer/2019/seed-42"
    root = layout.output(key)
    report = json.loads((root / "run.json").read_text())
    entry = report["predictions"]["evaluation"]
    path = root / entry["path"]
    pq.write_table(pq.read_table(path).drop(["asset_id"]), path)
    entry.update(sha256=sha256(path), bytes=path.stat().st_size)
    (root / "run.json").write_text(json.dumps(report))
    archive = tmp_path / "historical.zip"
    bundle.export_results(declaration, digest, layout, "references", archive)
    with bundle.ResultsBundle(archive) as opened:
        name = opened.manifest["jobs"][key]["tables"]["42"]["evaluation"]
        table = opened.read_table(name)
        assert (
            "asset_id" not in table.column_names and panel_from_table(table).rows == table.num_rows
        )
    assert not (layout.state / "out.zip").exists()


def test_changed_predictions_are_not_packaged_as_verified_results(tmp_path):
    declaration, digest, layout = campaign_state(tmp_path / "source")
    folder = layout.output("transformer/2019/seed-42")
    (folder / "evaluation-predictions.parquet").write_bytes(b"alterado")
    archive = tmp_path / "results.zip"
    with pytest.raises(ValueError, match="huella"):
        bundle.export_results(declaration, digest, layout, "references", archive)
    assert not archive.exists()


def test_symlinked_predictions_are_rejected(tmp_path):
    declaration, digest, layout = campaign_state(tmp_path / "source")
    folder = layout.output("transformer/2019/seed-42")
    path = folder / "evaluation-predictions.parquet"
    moved = tmp_path / "outside.parquet"
    path.rename(moved)
    path.symlink_to(moved)
    with pytest.raises(ValueError, match="regular|ruta"):
        bundle.export_results(declaration, digest, layout, "references", tmp_path / "out.zip")


def test_a_directory_link_cannot_refer_to_external_predictions(tmp_path):
    declaration, digest, layout = campaign_state(tmp_path / "source")
    folder = layout.output("transformer/2019/seed-42")
    outside = tmp_path / "external"
    outside.mkdir()
    shutil.copyfile(folder / "evaluation-predictions.parquet", outside / "evaluation.parquet")
    (folder / "external").symlink_to(outside, target_is_directory=True)
    report = json.loads((folder / "run.json").read_text())
    report["predictions"]["evaluation"]["path"] = "external/evaluation.parquet"
    (folder / "run.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="ruta"):
        bundle.export_results(declaration, digest, layout, "references", tmp_path / "out.zip")


def test_a_completed_neural_fit_needs_its_history(tmp_path):
    declaration, digest, layout = campaign_state(tmp_path / "source")
    folder = layout.output("transformer/2019/seed-42")
    report = json.loads((folder / "run.json").read_text())
    report.pop("fit")
    (folder / "run.json").write_text(json.dumps(report))
    (folder / "fit/run.json").unlink()
    with pytest.raises(ValueError, match="historial"):
        bundle.export_results(declaration, digest, layout, "references", tmp_path / "out.zip")


@pytest.mark.parametrize(
    "name", ["../external.parquet", "/external.parquet", "x\\external.parquet"]
)
def test_report_paths_cannot_escape_the_job_directory(tmp_path, name):
    declaration, digest, layout = campaign_state(tmp_path / "source")
    report_path = layout.output("transformer/2019/seed-42") / "run.json"
    report = json.loads(report_path.read_text())
    report["predictions"]["evaluation"]["path"] = name
    report_path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="ruta"):
        bundle.export_results(declaration, digest, layout, "references", tmp_path / "results.zip")


def test_partial_archive_retains_failed_and_pending_jobs(tmp_path):
    declaration, digest, layout = campaign_state(tmp_path / "source")
    key = "transformer/2019/seed-43"
    path = layout.record(key)
    record = json.loads(path.read_text())
    record["status"] = "failed"
    record["attempts"].append(dict(status="failed", error="fallo de prueba", seconds=1.5))
    path.write_text(json.dumps(record))
    log = layout.log(key)
    log.parent.mkdir(parents=True)
    log.write_text("Traza completa del fallo de prueba\n")

    result = bundle.export_results(
        declaration, digest, layout, "references", tmp_path / "partial.zip"
    )

    assert result["complete"] is False and result["completed_jobs"] == 1
    with bundle.ResultsBundle(tmp_path / "partial.zip") as opened:
        failed = opened.manifest["jobs"][key]
        assert failed["status"] == "failed" and failed["tables"] == {}
        assert opened.read_json(failed["record"])["attempts"][-1]["error"] == "fallo de prueba"
        assert opened.archive.read(failed["log"]).decode() == "Traza completa del fallo de prueba\n"


def test_a_different_declaration_cannot_claim_existing_results(tmp_path):
    declaration, _, layout = campaign_state(tmp_path / "source")
    with pytest.raises(ValueError, match="autorizada"):
        bundle.export_results(declaration, "b" * 64, layout, "references", tmp_path / "out.zip")


@pytest.mark.parametrize("extra", ["../outside", "manifest.json"])
def test_archive_rejects_unexpected_or_duplicate_members(tmp_path, extra):
    declaration, digest, layout = campaign_state(tmp_path / "source")
    archive = tmp_path / "results.zip"
    bundle.export_results(declaration, digest, layout, "references", archive)
    with zipfile.ZipFile(archive, "a") as zipped:
        if extra == "manifest.json":
            with pytest.warns(UserWarning, match="Duplicate"):
                zipped.writestr(extra, b"{}")
        else:
            zipped.writestr(extra, b"no permitido")
    with pytest.raises(ValueError, match="ruta|repetid|declarad"):
        with bundle.ResultsBundle(archive):
            pass


def test_archive_reader_rejects_modified_payloads(tmp_path):
    declaration, digest, layout = campaign_state(tmp_path / "source")
    original, changed = tmp_path / "original.zip", tmp_path / "changed.zip"
    bundle.export_results(declaration, digest, layout, "references", original)
    with zipfile.ZipFile(original) as source, zipfile.ZipFile(changed, "w") as target:
        for name in source.namelist():
            data = source.read(name)
            if name.endswith("fit.json"):
                data = data.replace(b'"epoch":1', b'"epoch":9')
                data = data.replace(b'"epoch": 1', b'"epoch": 9')
            target.writestr(name, data)
    with pytest.raises(ValueError, match="huella"):
        with bundle.ResultsBundle(changed):
            pass


def test_archive_reader_obeys_an_uncompressed_size_limit(tmp_path, monkeypatch):
    declaration, digest, layout = campaign_state(tmp_path / "source")
    archive = tmp_path / "results.zip"
    bundle.export_results(declaration, digest, layout, "references", archive)
    monkeypatch.setattr(bundle, "MAX_TOTAL_BYTES", 100)
    with pytest.raises(ValueError, match="límite"):
        with bundle.ResultsBundle(archive):
            pass


def test_zip64_payloads_remain_readable(tmp_path, monkeypatch):
    declaration, digest, layout = campaign_state(tmp_path / "source")
    monkeypatch.setattr(zipfile, "ZIP64_LIMIT", 1024)
    archive = tmp_path / "zip64.zip"
    bundle.export_results(declaration, digest, layout, "references", archive)
    with bundle.ResultsBundle(archive) as opened:
        assert opened.manifest["complete"] is True


def test_a_huge_directory_is_refused_before_zipfile_allocates_it(tmp_path, monkeypatch):
    declaration, digest, layout = campaign_state(tmp_path / "source")
    archive = tmp_path / "results.zip"
    bundle.export_results(declaration, digest, layout, "references", archive)
    data = bytearray(archive.read_bytes())
    footer = list(struct.unpack("<4s4H2LH", data[-22:]))
    footer[5] = 1024**3
    data[-22:] = struct.pack("<4s4H2LH", *footer)
    archive.write_bytes(data)

    def forbidden(*_, **__):
        raise AssertionError("Se intentó cargar el directorio sin comprobar su tamaño")

    monkeypatch.setattr(bundle.zipfile, "ZipFile", forbidden)
    with pytest.raises(ValueError, match="límite"):
        with bundle.ResultsBundle(archive):
            pass


def test_parquet_row_limit_is_checked_before_materializing_the_table(tmp_path, monkeypatch):
    declaration, digest, layout = campaign_state(tmp_path / "source")
    archive = tmp_path / "results.zip"
    bundle.export_results(declaration, digest, layout, "references", archive)
    with bundle.ResultsBundle(archive) as opened:
        table = opened.manifest["jobs"]["transformer/2019/seed-42"]["tables"]["42"]["evaluation"]
        monkeypatch.setattr(bundle, "MAX_TABLE_ROWS", 10)
        with pytest.raises(ValueError, match="límite"):
            opened.read_table(table)


def test_result_export_is_a_read_only_cli_command(tmp_path):
    declaration, _, layout = campaign_state(tmp_path / "source")
    destination = tmp_path / "cli.zip"
    result = campaign.main(
        [
            "export-results",
            "--declaration",
            str(tmp_path / "source/declaration.json"),
            "--state",
            str(layout.state),
            "--views",
            str(layout.views),
            "--machine",
            "references",
            "--destination",
            str(destination),
        ]
    )
    assert result == 0 and destination.is_file()
    assert declaration["final_test_access"] is False


def rewrite_json(source, destination, name, change):
    with zipfile.ZipFile(source) as archive:
        payloads = {key: archive.read(key) for key in archive.namelist()}
    manifest = json.loads(payloads["manifest.json"])
    value = json.loads(payloads[name])
    change(value)
    payloads[name] = json.dumps(value).encode()
    manifest["payload_bytes"] += len(payloads[name]) - manifest["files"][name]["bytes"]
    manifest["files"][name] = dict(
        bytes=len(payloads[name]), sha256=hashlib.sha256(payloads[name]).hexdigest()
    )
    payloads["manifest.json"] = json.dumps(manifest).encode()
    with zipfile.ZipFile(destination, "w") as archive:
        for key, data in payloads.items():
            archive.writestr(key, data)


def test_reader_cannot_claim_a_different_parent_checkpoint(tmp_path):
    declaration, digest, layout = campaign_state(tmp_path / "source", reader=True)
    original, changed = tmp_path / "original.zip", tmp_path / "changed.zip"
    bundle.export_results(declaration, digest, layout, "mars_titan", original)
    with bundle.ResultsBundle(original):
        pass
    rewrite_json(
        original,
        changed,
        "reports/mars_titan_m3/2019/seed-42/run.json",
        lambda report: report["request"]["parent"].update(checkpoint_sha256="0" * 64),
    )
    with pytest.raises(ValueError, match="padre|checkpoint"):
        with bundle.ResultsBundle(changed):
            pass


def test_report_seed_is_checked_again_after_transfer(tmp_path):
    declaration, digest, layout = campaign_state(tmp_path / "source")
    original, changed = tmp_path / "original.zip", tmp_path / "changed.zip"
    bundle.export_results(declaration, digest, layout, "references", original)
    rewrite_json(
        original,
        changed,
        "reports/transformer/2019/seed-42/run.json",
        lambda report: report["request"].update(seed=99),
    )
    with pytest.raises(ValueError, match="semilla"):
        with bundle.ResultsBundle(changed):
            pass


def traced_job(layout, status="completed", *, model="transformer", family=None):
    from mars_titan.data.storage import atomic_json, sha256
    from mars_titan.learning_traces.hooks import trace_identity
    from mars_titan.learning_traces.recorder import TraceConfig, TraceRecorder

    key = f"{model}/2019/seed-42"
    root = layout.output(key)
    fit_path = root / "fit/run.json"
    fit = json.loads(fit_path.read_text())
    fit.update(identity=dict(fold="fold-014", seed=42), global_step=8)
    fit["recovery_checkpoint"] = fit["best_checkpoint"]
    atomic_json(fit_path, fit)
    report = json.loads((root / "run.json").read_text())
    report["fit"].update(sha256=sha256(fit_path), run_id=fit["run_id"])
    report["identity"]["fit_run_id"] = fit["run_id"]
    atomic_json(root / "run.json", report)
    policy = dict(
        every=2,
        max_bytes=65536,
        start_step=4,
        resume_step=4,
        policy_sha256="b" * 64,
        qualification_sha256="c" * 64,
    )
    record = json.loads(layout.record(key).read_text())
    record.update(status=status)
    record["attempts"][-1].update(
        status=status, learning_traces=policy, trace_checkpoint_confirmed=True
    )
    atomic_json(layout.record(key), record)
    recorder = TraceRecorder(
        root / "traces",
        TraceConfig(policy["every"], policy["max_bytes"]),
        trace_identity(family or model, fit["identity"]),
    )
    for step in (6, 8):
        recorder.scalar("loss", 1.0 / step)
        recorder.flush(step, "train")
    receipt = dict(
        schema_version=1,
        kind="comparison_learning_trace_checkpoint",
        fit_run_id=fit["run_id"],
        checkpoint_sha256=fit["recovery_checkpoint"]["sha256"],
        start_step=4,
        global_step=8,
        cursor=recorder.state_dict(),
        policy_sha256=policy["policy_sha256"],
        qualification_sha256=policy["qualification_sha256"],
    )
    atomic_json(root / "trace-receipt.json", receipt)
    return key, root, recorder


@pytest.mark.parametrize(
    ("model", "family", "wrong", "machine"),
    [
        ("transformer", "transformer", "titans_mac", "references"),
        ("mars_titan_m3", "mars_titan_readout", "mars_titan_m3", "mars_titan"),
    ],
)
def test_trace_family_matches_its_job_on_export_and_read(tmp_path, model, family, wrong, machine):
    from mars_titan.data.storage import atomic_json

    declaration, digest, layout = campaign_state(tmp_path / "source", reader=True)
    key, root, _ = traced_job(layout, model=model, family=family)
    archive = tmp_path / "valid.zip"
    bundle.export_results(declaration, digest, layout, machine, archive)
    with bundle.ResultsBundle(archive) as opened:
        name = opened.manifest["jobs"][key]["learning_traces"]["manifest"]
        assert opened.read_json(name)["identity"]["family"] == family

    transferred = tmp_path / "changed.zip"
    rewrite_json(archive, transferred, name, lambda trace: trace["identity"].update(family=wrong))
    with pytest.raises(ValueError, match="familia"), bundle.ResultsBundle(transferred):
        pass

    path = root / "traces/manifest.json"
    manifest = json.loads(path.read_text())
    manifest["identity"]["family"] = wrong
    atomic_json(path, manifest)
    with pytest.raises(ValueError, match="familia"):
        bundle.export_results(declaration, digest, layout, machine, tmp_path / "invalid.zip")


@pytest.mark.parametrize("status", ["completed", "paused"])
def test_zip_transports_only_the_confirmed_trace_parts(tmp_path, status):
    declaration, digest, layout = campaign_state(tmp_path / "source")
    key, root, recorder = traced_job(layout, status)
    recorder.scalar("loss", 999.0)
    recorder.flush(10, "train")
    with (root / "traces/parts/.part-00000004.parquet.tmp").open("wb") as stream:
        stream.write(b"parcial")
    archive = tmp_path / "traces.zip"
    bundle.export_results(declaration, digest, layout, "references", archive)
    shutil.rmtree(layout.state)
    with bundle.ResultsBundle(archive) as opened:
        trace = opened.manifest["jobs"][key]["learning_traces"]
        assert trace["status"] == "confirmed"
        receipt = opened.read_json(trace["receipt"])
        assert (receipt["start_step"], receipt["global_step"]) == (4, 8)
        assert opened.read_json(trace["manifest"])["config"]["every"] == 2
        assert len(trace["parts"]) == receipt["cursor"]["sequence"] == 2
        assert (
            sum(opened.manifest["files"][p]["bytes"] for p in trace["parts"])
            == receipt["cursor"]["bytes"]
        )
        assert [opened.read_table(p)["step"].to_pylist() for p in trace["parts"]] == [[6], [8]]
        assert not any(
            name.endswith((".pt", ".tmp", "part-00000003.parquet"))
            for name in opened.archive.namelist()
        )


def test_zip_leaves_the_open_attempt_unconfirmed(tmp_path):
    declaration, digest, layout = campaign_state(tmp_path / "source")
    key, _, _ = traced_job(layout, "running")
    archive = tmp_path / "open.zip"
    bundle.export_results(declaration, digest, layout, "references", archive)
    with bundle.ResultsBundle(archive) as opened:
        assert opened.manifest["jobs"][key]["learning_traces"] == {"status": "unconfirmed"}
        assert not any(name.startswith("traces/") for name in opened.manifest["files"])


@pytest.mark.parametrize(
    "fault", ["missing", "bytes", "identity", "checkpoint", "start", "duplicate"]
)
def test_zip_rejects_an_inconsistent_confirmed_trace(tmp_path, fault):
    import pyarrow.parquet as pq

    from mars_titan.data.storage import atomic_json

    declaration, digest, layout = campaign_state(tmp_path / "source")
    _, root, recorder = traced_job(layout)
    path = root / "trace-receipt.json"
    receipt = json.loads(path.read_text())
    first = recorder.parts / "part-00000001.parquet"
    if fault == "missing":
        first.unlink()
    elif fault == "bytes":
        receipt["cursor"]["bytes"] += 1
    elif fault == "identity":
        manifest = json.loads((recorder.folder / "manifest.json").read_text())
        manifest["identity"]["run_identity_sha256"] = "d" * 64
        atomic_json(recorder.folder / "manifest.json", manifest)
    elif fault == "checkpoint":
        receipt["checkpoint_sha256"] = "d" * 64
    elif fault == "start":
        receipt["start_step"] = 7
    else:
        second = recorder.parts / "part-00000002.parquet"
        table = pq.read_table(first)
        pq.write_table(table, second)
        receipt["cursor"]["bytes"] = first.stat().st_size + second.stat().st_size
    atomic_json(path, receipt)
    archive = tmp_path / "invalid.zip"
    with pytest.raises(ValueError):
        bundle.export_results(declaration, digest, layout, "references", archive)
    assert not archive.exists()


def test_archive_revalidates_a_trace_receipt_even_with_matching_file_hashes(tmp_path):
    declaration, digest, layout = campaign_state(tmp_path / "source")
    key, _, _ = traced_job(layout)
    archive, changed = tmp_path / "valid.zip", tmp_path / "invalid.zip"
    exported = bundle.export_results(declaration, digest, layout, "references", archive)
    trace = exported["jobs"][key]["learning_traces"]
    rewrite_json(archive, changed, trace["receipt"], lambda r: r["cursor"].update(sequence=1))
    with pytest.raises(ValueError):
        with bundle.ResultsBundle(changed):
            pass


@pytest.mark.parametrize("fault", ["window", "resume", "budget", "exhausted"])
def test_trace_coverage_is_bound_to_window_and_attempt(tmp_path, fault):
    from mars_titan.data.storage import atomic_json

    declaration, digest, layout = campaign_state(tmp_path / "source")
    key, root, recorder = traced_job(layout)
    if fault == "window":
        path = root / "run.json"
        document = json.loads(path.read_text())
        document["identity"]["fit_run_id"] = "0" * 64
    elif fault == "exhausted":
        path = root / "trace-receipt.json"
        document = json.loads(path.read_text())
        document["cursor"]["exhausted_at"] = 5
        manifest = json.loads((recorder.folder / "manifest.json").read_text())
        manifest["exhausted_at_step"] = 5
        atomic_json(recorder.folder / "manifest.json", manifest)
    else:
        path = layout.record(key)
        document = json.loads(path.read_text())
        declaration_of_attempt = document["attempts"][0]["learning_traces"]
        if fault == "resume":
            declaration_of_attempt["resume_step"] = 2
        else:
            declaration_of_attempt["max_bytes"] = 1 << 50
            manifest = json.loads((recorder.folder / "manifest.json").read_text())
            manifest["config"]["max_bytes"] = 1 << 50
            atomic_json(recorder.folder / "manifest.json", manifest)
    atomic_json(path, document)
    with pytest.raises(ValueError):
        bundle.export_results(declaration, digest, layout, "references", tmp_path / "invalid.zip")


def test_confirmed_empty_trace_declares_no_observed_step(tmp_path):
    from mars_titan.data.storage import atomic_json

    declaration, digest, layout = campaign_state(tmp_path / "source")
    key, root, _ = traced_job(layout, "paused")
    path = root / "trace-receipt.json"
    receipt = json.loads(path.read_text())
    receipt["cursor"] = dict(sequence=0, bytes=0, exhausted_at=None)
    atomic_json(path, receipt)
    destination = tmp_path / "empty.zip"
    bundle.export_results(declaration, digest, layout, "references", destination)
    with bundle.ResultsBundle(destination) as opened:
        descriptor = opened.manifest["jobs"][key]["learning_traces"]
        assert descriptor["parts"] == [] and descriptor["rows"] == 0
        assert descriptor["first_observed_step"] is descriptor["last_observed_step"] is None


@pytest.mark.parametrize("status", ["completed", "paused"])
def test_attempt_without_trace_cursor_leaves_previous_files_out(tmp_path, status):
    from mars_titan.data.storage import atomic_json

    declaration, digest, layout = campaign_state(tmp_path / "source")
    key, root, _ = traced_job(layout, status)
    path = layout.record(key)
    record = json.loads(path.read_text())
    record["attempts"][-1]["trace_checkpoint_confirmed"] = False
    atomic_json(path, record)
    if status == "completed":
        (root / "trace-receipt.json").unlink()
    archive = tmp_path / "empty.zip"
    bundle.export_results(declaration, digest, layout, "references", archive)
    with bundle.ResultsBundle(archive) as opened:
        assert opened.manifest["jobs"][key]["learning_traces"] == {"status": "unconfirmed"}
        assert not any(name.startswith("traces/") for name in opened.manifest["files"])


def test_a_confirmed_pause_needs_its_receipt(tmp_path):
    declaration, digest, layout = campaign_state(tmp_path / "source")
    _, root, _ = traced_job(layout, "paused")
    (root / "trace-receipt.json").unlink()
    with pytest.raises(ValueError):
        bundle.export_results(declaration, digest, layout, "references", tmp_path / "invalid.zip")


def test_paused_trace_cannot_use_another_seeds_window(tmp_path):
    from mars_titan.data.storage import atomic_json

    declaration, digest, layout = campaign_state(tmp_path / "source")
    _, root, _ = traced_job(layout, "paused")
    path = root / "run.json"
    report = json.loads(path.read_text())
    report["request"]["seed"] = 43
    atomic_json(path, report)
    with pytest.raises(ValueError, match="semilla"):
        bundle.export_results(declaration, digest, layout, "references", tmp_path / "invalid.zip")


def test_trace_reader_checks_dictionary_name_lengths(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from mars_titan.evaluation import comparison_trace_bundle
    from mars_titan.learning_traces.schema import SCHEMA

    table = pa.Table.from_pylist(
        [dict(step=8, phase="train", metric="a" * 97, group="all", stat="value", value=0.125)],
        schema=SCHEMA,
    )
    path = tmp_path / "part.parquet"
    pq.write_table(table, path, use_dictionary=["metric"])
    receipt = dict(start_step=4, global_step=8, cursor=dict(exhausted_at=None))
    with pytest.raises(ValueError, match="longitud"):
        comparison_trace_bundle.check_parts([path], receipt, 2, max_rows=100, max_bytes=65536)
