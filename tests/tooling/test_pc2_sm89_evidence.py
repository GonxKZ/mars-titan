"""Una medición debe identificar los módulos y distinguir llamadas de lanzamientos."""

import hashlib
from collections import Counter
from types import SimpleNamespace

import pytest

from benchmarks.comparison_efficiency_20261118 import pc2_sm89 as evidence


def test_runtime_must_supply_the_pinned_source_and_corresponding_kernel(tmp_path, monkeypatch):
    path = "src/mars_titan/memory/fixture.py"
    content = b"original\n"
    blob = hashlib.sha1(b"blob 9\0" + content, usedforsecurity=False).hexdigest()
    target = tmp_path / path
    target.parent.mkdir(parents=True)
    target.write_bytes(content)

    def git(argv, **_):
        if "ls-tree" in argv:
            return f"100644 blob {blob}\t{path}\0".encode()
        return "a" * 40 + "\n"

    monkeypatch.setattr(evidence.subprocess, "check_output", git)
    assert evidence.source_tree(tmp_path, "reference")["python_modules"] == 1
    target.write_bytes(b"alterado\n")
    with pytest.raises(ValueError, match="fixture.py"):
        evidence.source_tree(tmp_path, "reference")
    target.write_bytes(content)
    other = tmp_path / "src/mars_titan/models/titans/cuda_kernels.py"
    other.parent.mkdir(parents=True)
    other.touch()
    with pytest.raises(ValueError, match="modo"):
        evidence.source_tree(tmp_path, "reference")


def test_launch_counter_forwards_the_grid_and_counts_only_effective_calls():
    observed = []

    class Program:
        def __getitem__(self, grid):
            def launch(tensor, *, num_warps):
                observed.append((grid, tensor.shape, num_warps))
                return 23

            return launch

    counts = Counter()
    call = evidence.LaunchCounts(Program(), counts)[(2,)]
    assert not counts
    assert call(SimpleNamespace(shape=(8, 64, 64)), num_warps=4) == 23
    assert counts == {(8, 64, 64): 1}
    assert observed == [((2,), (8, 64, 64), 4)]
