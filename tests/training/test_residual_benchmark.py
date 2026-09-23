import json
import subprocess
import sys
from pathlib import Path

from tests.training.test_corpus_targets import materialized


def test_benchmark_compares_both_backends_without_reusing_previous_labels(tmp_path):
    manifest, prepared = materialized(tmp_path)
    output = tmp_path / "benchmark"
    command = [
        sys.executable,
        "reports/analysis/residual_preparation_benchmark.py",
        str(manifest),
        str(prepared),
        str(output),
        "--repetitions",
        "1",
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    receipt = json.loads((output / "measurement.json").read_text())
    assert receipt["exact_labels_equal"] is True
    assert receipt["assets"] == 1
    assert len(receipt["measurements"]) == 2
    assert {item["backend"] for item in receipt["measurements"]} == {"reference", "numpy"}
    for item in receipt["measurements"]:
        assert item["compute_seconds"] > 0
        assert item["process_seconds"] >= item["compute_seconds"]
        assert item["process_peak_rss_bytes"] > 0
    before = (output / "measurement.json").read_bytes()
    repeated = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert repeated.returncode != 0
    assert (output / "measurement.json").read_bytes() == before
    assert Path(manifest).is_file()
