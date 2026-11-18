"""La medida de arranque usa la misma primera entrada en procesos independientes."""

import json
import os
import subprocess
import sys

import pytest

from benchmarks.comparison_file_digests import measure
from mars_titan.training.corpus_inputs import DIGEST_CACHE_ENV
from tests.training.test_financial_run import corpus


def test_cold_and_warm_processes_verify_the_same_batch(tmp_path):
    data, sources = corpus(tmp_path)
    command = [
        sys.executable,
        "-m",
        "benchmarks.comparison_file_digests",
        str(data.path),
        str(sources["train"].path),
    ]
    env = dict(os.environ, **{DIGEST_CACHE_ENV: str(tmp_path / "shared.json")})
    cold, warm = [
        json.loads(subprocess.check_output(command, env=env, text=True)) for _ in range(2)
    ]
    assert cold["parquet_files_hashed"] > 0 and cold["parquet_bytes_hashed"] > 0
    assert warm["parquet_files_hashed"] == warm["parquet_bytes_hashed"] == 0
    for key in ("cohort_sha256", "index_sha256", "first_at", "input_sha256"):
        assert cold[key] == warm[key]
    assert cold["input_sha256"]
    assert measure(data.path, sources["train"].path, 512)["input_sha256"] == warm["input_sha256"]
    with pytest.raises(ValueError, match="índice de ajuste"):
        measure(data.path, sources["validation"].path, 512)
