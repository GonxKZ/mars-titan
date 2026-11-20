"""Análisis de dos ZIP sin conservar los pesos ni los datos originales."""

import csv
import json
import shutil
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mars_titan.data.storage import sha256
from mars_titan.evaluation import comparison_campaign_analysis as analysis
from mars_titan.evaluation.comparison_bundle import export_results
from tests.evaluation.comparison_results_fixtures import campaign_state


def packages(tmp_path, *, failed=False, changed_column=None, changed_commit=False, commits=None):
    declaration, digest, layout = campaign_state(tmp_path / "source")

    def set_commit(commit):
        permission = json.loads(layout.authorization.read_text())
        permission["code_commit"] = commit
        layout.authorization.write_text(json.dumps(permission))

    if commits:
        set_commit(commits[0])
    first = tmp_path / "references.zip"
    export_results(declaration, digest, layout, "references", first)
    key = "titans_mac/2019/seed-43"
    if failed:
        path = layout.record(key)
        record = json.loads(path.read_text())
        record["status"] = "failed"
        record["attempts"].append(dict(status="failed", error="fallo de prueba", seconds=2))
        path.write_text(json.dumps(record))
    if changed_column:
        directory = layout.output(key)
        path = directory / "evaluation-predictions.parquet"
        table = pq.read_table(path)
        values = table[changed_column].to_pylist()
        if changed_column == "asset_id":
            values[0], values[1] = values[1], values[0]
        else:
            values[0] += 0.01
        table = table.set_column(
            table.schema.get_field_index(changed_column), changed_column, pa.array(values)
        )
        pq.write_table(table, path)
        report = json.loads((directory / "run.json").read_text())
        report["predictions"]["evaluation"].update(sha256=sha256(path), bytes=path.stat().st_size)
        (directory / "run.json").write_text(json.dumps(report))
    if changed_commit or commits:
        set_commit(commits[1] if commits else "b" * 40)
    second = tmp_path / "mars-titan.zip"
    export_results(declaration, digest, layout, "mars_titan", second)
    return first, second


def test_all_metrics_and_figures_are_rebuilt_after_the_sources_are_removed(tmp_path):
    first, second = packages(tmp_path)
    shutil.rmtree(tmp_path / "source")

    report = analysis.analyze([first, second], tmp_path / "analysis")

    assert report["complete"] is True and report["missing_jobs"] == {}
    assert report["scope"]["expected_jobs"] == 4
    assert report["models"]["transformer"]["pooled"]["mae_raw"] > 0
    assert report["models"]["transformer"]["pooled"]["mae_calibrated"] == pytest.approx(
        report["models"]["transformer"]["pooled"]["zero_mae"]
    )
    assert report["models"]["transformer"]["by_origin"]["2019"]["seed_count"] == 2
    assert report["per_seed"][0]["calibration_factors"]["US"] is not None
    assert report["comparisons"] and report["protocol"]["bootstrap"]["stratification"] == "origin"
    for file in (
        "report.json",
        "metrics.csv",
        "resources.csv",
        "learning-curves.csv",
        "README.md",
        "plots/mae.png",
        "plots/information-coefficient.svg",
    ):
        assert (tmp_path / "analysis" / file).is_file(), file
    saved = json.loads((tmp_path / "analysis/report.json").read_text())
    assert saved == report


def test_archive_order_does_not_change_the_measured_results(tmp_path):
    first, second = packages(tmp_path)
    one = analysis.analyze([first, second], tmp_path / "one", plots=False)
    two = analysis.analyze([second, first], tmp_path / "two", plots=False)
    assert one["models"] == two["models"]
    assert one["comparisons"] == two["comparisons"]


def test_repeated_jobs_are_never_counted_twice(tmp_path):
    first, _ = packages(tmp_path)
    with pytest.raises(ValueError, match="repetid"):
        analysis.analyze([first, first], tmp_path / "analysis", plots=False)


def test_different_training_commits_are_not_mixed(tmp_path):
    first, second = packages(tmp_path, changed_commit=True)
    with pytest.raises(ValueError, match="código|commit"):
        analysis.analyze([first, second], tmp_path / "analysis", plots=False)


@pytest.mark.parametrize("older", sorted(analysis.WARMUP_COMMITS))
def test_packed_inputs_require_the_exact_release_and_declared_option(older):
    from mars_titan.training.comparison_input_migration import PACKED_RELEASE

    base = dict(
        training_commit=older,
        code_changes_sha256=None,
        authorized_declaration_sha256=analysis.GPU_DECLARATION,
        declaration_sha256=analysis.PACKED_DECLARATION,
    )
    other = dict(base, training_commit=PACKED_RELEASE)

    def bundle(p):
        return SimpleNamespace(manifest=dict(provenance=p, cohorts={"a": "b"}, jobs={}))

    with pytest.raises(ValueError, match="código|commit"):
        analysis._merge([bundle(base), bundle(other)], verified_warmup=True)
    assert analysis._merge([bundle(base), bundle(other)], verified_packed_inputs=True) == {}
    with pytest.raises(ValueError, match="código|commit"):
        analysis._merge(
            [bundle(dict(base, training_commit="2" * 40)), bundle(other)],
            verified_packed_inputs=True,
        )
    for key, val in (
        ("training_commit", "2" * 40),
        ("code_changes_sha256", "1" * 64),
        ("declaration_sha256", "4" * 64),
    ):
        with pytest.raises(ValueError, match="código|declaración"):
            analysis._merge(
                [bundle(base), bundle(dict(other, **{key: val}))], verified_packed_inputs=True
            )


def test_mixed_zip_report_names_each_runtime(tmp_path, monkeypatch):
    from mars_titan.evaluation.comparison_bundle import ResultsBundle

    first, second = packages(tmp_path, changed_commit=True)
    with ResultsBundle(first) as bundle:
        provenance = bundle.manifest["provenance"]
    monkeypatch.setattr(analysis, "WARMUP_COMMITS", {"a" * 40, "b" * 40})
    monkeypatch.setattr(analysis, "GPU_DECLARATION", provenance["authorized_declaration_sha256"])
    monkeypatch.setattr(analysis, "PACKED_DECLARATION", provenance["declaration_sha256"])
    report = analysis.analyze(
        [first, second], tmp_path / "mixed", plots=False, verified_warmup=True
    )
    origin = report["provenance"]
    assert origin["training_commit"] is None
    assert {row["training_commit"] for row in origin["archives"]} == analysis.WARMUP_COMMITS
    assert origin["verified_warmup"] is True
    assert report["complete"] and report["comparisons"]


@pytest.mark.parametrize("previous", sorted(analysis.WARMUP_COMMITS))
def test_packed_release_keeps_each_zip_origin_and_its_metrics(tmp_path, monkeypatch, previous):
    from mars_titan.evaluation.comparison_bundle import ResultsBundle

    raw, packed = packages(tmp_path, commits=(previous, analysis.PACKED_RELEASE))
    with ResultsBundle(raw) as bundle:
        provenance = bundle.manifest["provenance"]
    monkeypatch.setattr(analysis, "GPU_DECLARATION", provenance["authorized_declaration_sha256"])
    monkeypatch.setattr(analysis, "PACKED_DECLARATION", provenance["declaration_sha256"])
    report = analysis.analyze(
        [raw, packed], tmp_path / "mixed", plots=False, verified_packed_inputs=True
    )
    origin = report["provenance"]
    assert origin["training_commit"] is None and origin["verified_packed_inputs"]
    assert not origin["verified_warmup"]
    assert {row["training_commit"] for row in origin["archives"]} == {
        previous,
        analysis.PACKED_RELEASE,
    }
    assert report["complete"] and report["comparisons"]
    assert report["models"]["transformer"]["pooled"]["mae_raw"] > 0


@pytest.mark.parametrize(
    "change", ["commit", "dirty", "declaration", "packed", "cohorts", "missing"]
)
def test_warmup_option_still_rejects_unverified_pairs(change):
    older, newer = sorted(analysis.WARMUP_COMMITS)
    first = dict(
        training_commit=older,
        code_changes_sha256=None,
        authorized_declaration_sha256=analysis.GPU_DECLARATION,
        declaration_sha256=analysis.PACKED_DECLARATION,
    )
    second = dict(first, training_commit=newer)
    cohorts = {"2019": "misma cohorte"}
    if change == "commit":
        second["training_commit"] = "c" * 40
    if change == "missing":
        del first["code_changes_sha256"], second["code_changes_sha256"]
    if change in {"dirty", "declaration", "packed"}:
        key = {
            "dirty": "code_changes_sha256",
            "declaration": "authorized_declaration_sha256",
            "packed": "declaration_sha256",
        }[change]
        first[key] = second[key] = "c" * 64
    bundles = [
        SimpleNamespace(manifest=dict(provenance=first, cohorts=cohorts, jobs={})),
        SimpleNamespace(
            manifest=dict(
                provenance=second, cohorts={} if change == "cohorts" else cohorts, jobs={}
            )
        ),
    ]
    with pytest.raises(ValueError, match="código|cohortes|declaración"):
        analysis._merge(bundles, verified_warmup=True)


def test_missing_seeds_prevent_a_complete_comparison(tmp_path):
    first, second = packages(tmp_path, failed=True)
    with pytest.raises(ValueError, match="trabajos|incomplet"):
        analysis.analyze([first, second], tmp_path / "strict", plots=False)
    report = analysis.analyze(
        [first, second], tmp_path / "partial", allow_partial=True, plots=False
    )
    assert report["complete"] is False
    assert "titans_mac/2019/seed-43" in report["missing_jobs"]
    assert report["comparisons"] == {}
    assert report["inference_reason"]


@pytest.mark.parametrize("size", [4096, 8193])
def test_resource_csv_carries_last_readout_warmup_and_leaves_legacy_cells_empty(tmp_path, size):
    declaration, digest, layout = campaign_state(tmp_path / "source", reader=True)
    key = "mars_titan_m3/2019/seed-42"
    root = layout.output(key)
    fit = root / "fit/run.json"
    data = json.loads(fit.read_text())
    data["validation_warmup"] = dict(
        hits=3, events=4, tensor_bytes=size, limit_bytes=8192, allocation_fallbacks=1
    )
    fit.write_text(json.dumps(data))
    run = root / "run.json"
    descriptor = json.loads(run.read_text())
    descriptor["fit"]["sha256"] = sha256(fit)
    run.write_text(json.dumps(descriptor))
    record_path = layout.record(key)
    record = json.loads(record_path.read_text())
    record["attempts"][-1]["training_input_archive"] = dict(
        packed_passes=5,
        builds=1,
        build_seconds=4.5,
        built_bytes=1234,
        verified_rows=91,
        input_sha256="a" * 64,
        archive_sha256="b" * 64,
        index_sha256="d" * 64,
    )
    record_path.write_text(json.dumps(record))
    paths = []
    for machine in ("references", "mars_titan"):
        paths.append(tmp_path / (machine + ".zip"))
        export_results(declaration, digest, layout, machine, paths[-1])
    output = tmp_path / "analysis"
    if size > 8192:
        with pytest.raises(ValueError, match="recibo de calentamiento"):
            analysis.analyze(paths, output, plots=False)
        return
    analysis.analyze(paths, output, plots=False)
    with (output / "resources.csv").open() as stream:
        rows = {r["job"]: r for r in csv.DictReader(stream)}
    assert rows[key]["last_fit_warmup_hits"] == "3"
    assert rows[key]["archive_packed_passes"] == "5" and rows[key]["archive_build_seconds"] == "4.5"
    assert rows["titans_mac/2019/seed-42"]["archive_packed_passes"] == ""
    assert rows[key]["last_fit_warmup_tensor_bytes"] == "4096"
    assert rows[key]["last_fit_warmup_allocation_fallbacks"] == "1"
    assert rows[key]["bundle_training_commit"] == "a" * 40
    assert rows["titans_mac/2019/seed-42"]["last_fit_warmup_hits"] == ""


def test_different_targets_are_rejected_even_with_valid_file_hashes(tmp_path):
    first, second = packages(tmp_path, changed_column="target")
    with pytest.raises(ValueError, match="filas|objetivos|población"):
        analysis.analyze([first, second], tmp_path / "analysis", plots=False)


def test_swapped_assets_in_one_archived_model_are_rejected(tmp_path):
    first, second = packages(tmp_path, changed_column="asset_id")
    with pytest.raises(ValueError, match="activo"):
        analysis.analyze([first, second], tmp_path / "analysis", plots=False)


def test_memory_peaks_distinguish_measured_attempts_in_json_and_csv(tmp_path):
    declaration, digest, layout = campaign_state(tmp_path / "source")
    key = "transformer/2019/seed-42"
    path = layout.record(key)
    record = json.loads(path.read_text())
    record["attempts"].insert(0, dict(status="failed", error_type="KernelOOMKill", seconds=2.0))
    record["attempts"][1]["resources"] = dict(peak_rss_bytes=4096, cuda_reserved_peak_bytes=0)
    path.write_text(json.dumps(record))
    complete = "transformer/2019/seed-43"
    path = layout.record(complete)
    record = json.loads(path.read_text())
    record["attempts"][0]["resources"] = dict(peak_rss_bytes=3072)
    record["attempts"].insert(0, dict(status="paused", resources=dict(peak_rss_bytes=6144)))
    path.write_text(json.dumps(record))
    archive = tmp_path / "references.zip"
    export_results(declaration, digest, layout, "references", archive)
    output = tmp_path / "analysis"
    report = analysis.analyze([archive], output, allow_partial=True, plots=False)
    rows = {row["job"]: row for row in report["resources"]}
    partial = rows[key]
    assert partial["peak_rss_bytes"] is None
    assert partial["observed_peak_rss_bytes"] == 4096 and partial["peak_rss_bytes_attempts"] == 1
    assert partial["cuda_reserved_peak_bytes"] is None
    assert partial["observed_cuda_reserved_peak_bytes"] == 0
    assert partial["cuda_reserved_peak_bytes_attempts"] == 1
    assert partial["observed_gpu_memory_peak_sampled_bytes"] is None
    assert partial["gpu_memory_peak_sampled_bytes_attempts"] == 0 and partial["attempts"] == 2
    assert rows[complete]["peak_rss_bytes"] == rows[complete]["observed_peak_rss_bytes"] == 6144
    assert rows[complete]["peak_rss_bytes_attempts"] == 2
    with (output / "resources.csv").open() as stream:
        csv_rows = {r["job"]: r for r in csv.DictReader(stream)}
    assert csv_rows[key]["peak_rss_bytes"] == ""
    assert csv_rows[key]["observed_peak_rss_bytes"] == "4096"
    assert csv_rows[key]["peak_rss_bytes_attempts"] == "1"


@pytest.mark.parametrize("value", [-1, False, 1.5, "4096"])
def test_memory_peak_requires_an_integer_byte_count(value):
    with pytest.raises(ValueError, match="peak_rss_bytes"):
        analysis._memory_peaks([{"peak_rss_bytes": value}], 1)


def test_memory_peaks_without_attempts_remain_unknown():
    peaks = analysis._memory_peaks([], 0)
    assert peaks["peak_rss_bytes"] is None and peaks["observed_peak_rss_bytes"] is None
    assert peaks["peak_rss_bytes_attempts"] == 0


def test_analysis_never_overwrites_previous_outputs(tmp_path):
    first, second = packages(tmp_path)
    destination = tmp_path / "analysis"
    destination.mkdir()
    (destination / "previous.txt").write_text("resultado anterior")
    with pytest.raises(ValueError, match="existe"):
        analysis.analyze([first, second], destination, plots=False)
    assert (destination / "previous.txt").read_text() == "resultado anterior"


def test_offline_analysis_has_a_reproducible_cli(tmp_path):
    first, second = packages(tmp_path)
    assert (
        analysis.main(
            [
                "--bundle",
                str(first),
                "--bundle",
                str(second),
                "--output",
                str(tmp_path / "cli"),
                "--no-plots",
            ]
        )
        == 0
    )
    assert (tmp_path / "cli/report.json").is_file()


def test_neural_seed_blocks_are_joined_without_ensembling_predictions(tmp_path):
    declaration, digest, layout = campaign_state(tmp_path / "source", blocks=True, reader=True)
    files = []
    for machine in ("references", "mars_titan"):
        path = tmp_path / (machine + ".zip")
        export_results(declaration, digest, layout, machine, path)
        files.append(path)
    report = analysis.analyze(files, tmp_path / "analysis", plots=False)
    assert report["complete"] and report["scope"]["expected_jobs"] == 6
    assert report["models"]["mars_titan_m3"]["by_origin"]["2019"]["seed_count"] == 2
    rows = [r for r in report["per_seed"] if r["model"] == "transformer"]
    observed = [r["partitions"]["evaluation"]["raw"]["point"]["mae"] for r in rows]
    assert report["models"]["transformer"]["pooled"]["mae_raw"] == pytest.approx(sum(observed) / 2)
    assert report["analysis_implementation"]["files"]["forecast_scores.py"]
    reader = next(r for r in report["resources"] if r["model"] == "mars_titan_m3")
    assert reader["wall_seconds"] == 5
    assert reader["pipeline_wall_seconds"] == 10
    assert reader["pipeline_cost_jobs"] == [reader["parent_job"], reader["job"]]


def test_single_machine_archive_stays_descriptive_until_both_are_joined(tmp_path):
    first, _ = packages(tmp_path)
    report = analysis.analyze([first], tmp_path / "one-machine", allow_partial=True, plots=False)
    assert report["complete"] is False and len(report["missing_jobs"]) == 2
    assert report["models"]["titans_mac"]["pooled"] == {}
    assert report["comparisons"] == {}


def test_failed_analysis_leaves_a_status_without_a_success_report(tmp_path, monkeypatch):
    first, second = packages(tmp_path)

    def failure(*_, **__):
        raise ValueError("tabla ilegible de prueba")

    monkeypatch.setattr(analysis.ResultsBundle, "read_table", failure)
    output = tmp_path / "failed"
    with pytest.raises(ValueError, match="ilegible"):
        analysis.analyze([first, second], output, plots=False)
    status = json.loads((output / "analysis-status.json").read_text())
    assert status["status"] == "failed" and "ilegible" in status["error"]
    assert not (output / "report.json").exists()
