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


def test_kernel_control_requires_the_two_original_preparation_modules(tmp_path, monkeypatch):
    paths = ("src/mars_titan/models/learning_scale.py", "src/mars_titan/models/titans/financial.py")
    kernel = "src/mars_titan/models/titans/cuda_kernels.py"

    def blob(value):
        return hashlib.sha1(
            f"blob {len(value)}\0".encode() + value, usedforsecurity=False
        ).hexdigest()

    for name in (*paths, kernel):
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"kernel" if name == kernel else b"origen")

    def git(argv, **_):
        if "ls-tree" in argv:
            entries = {name: blob(b"candidato") for name in paths}
            entries[kernel] = blob(b"kernel")
            return b"".join(
                f"100644 blob {entries[name]}\t{name}\0".encode() for name in (*paths, kernel)
            )
        if "rev-parse" in argv and argv[-1].partition(":")[2] in paths:
            return blob(b"origen") + "\n"
        return "a" * 40 + "\n"

    monkeypatch.setattr(evidence.subprocess, "check_output", git)
    receipt = evidence.source_tree(tmp_path, "write", original_preparation=True)
    assert receipt["restored_from_reference"] == list(paths)
    with pytest.raises(ValueError, match="kernel en prueba"):
        evidence.source_tree(tmp_path, "portable", original_preparation=True)
    (tmp_path / paths[1]).write_bytes(b"candidato")
    with pytest.raises(ValueError, match="financial.py"):
        evidence.source_tree(tmp_path, "write", original_preparation=True)
