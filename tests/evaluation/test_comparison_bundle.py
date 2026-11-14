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


def test_results_cannot_be_written_inside_the_training_state(tmp_path):
    declaration, digest, layout = campaign_state(tmp_path / "source")
    with pytest.raises(ValueError, match="origen|proteg|estado"):
        bundle.export_results(declaration, digest, layout, "references", layout.state / "out.zip")
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
