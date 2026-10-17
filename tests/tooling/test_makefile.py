"""El punto de entrada de órdenes no puede ejecutar pasos de optimizador nativos."""

import json
import re
import shlex
import subprocess
import sys
from pathlib import Path

from tests.suite_support import NATIVE_VARIABLES

MAKEFILE = Path(__file__).resolve().parents[2] / "Makefile"
CHECK_DEVICES = tuple(
    f"MARS_TITAN_{name}_CHECK_DEVICE"
    for name in (
        "GRU",
        "M3",
        "REGENERATION",
        "MARS_RUN",
        "TITANS_ADAPTERS",
        "CM_CONTROL",
        "CANDIDATE_RUN",
        "READOUT_ADAPTERS",
        "CANDIDATE_ADAPTERS",
    )
)


def recorded_commands(tmp_path, target, *options):
    """Ejecutar las recetas de Make sin lanzar otra suite dentro de esta prueba."""
    output = tmp_path / "commands.jsonl"
    runner = tmp_path / "capture.py"
    runner.write_text(
        "import json, os, sys\n"
        "keys = ('CUDA_VISIBLE_DEVICES', 'MARS_TITAN_REQUIRE_CUDA', "
        "'MARS_TITAN_CANDIDATE_MEMORY_CHECK_DEVICE', "
        f"'MARS_TITAN_CANDIDATE_MEMORY_CHECK_ASSETS') + {CHECK_DEVICES!r}"
        f" + {tuple(NATIVE_VARIABLES)!r}\n"
        f"with open({str(output)!r}, 'a') as stream:\n"
        "    stream.write(json.dumps(dict(argv=sys.argv[1:], "
        "environment={key: os.environ.get(key) for key in keys})) + '\\n')\n"
    )
    result = subprocess.run(
        [
            "make",
            "--no-print-directory",
            "-s",
            target,
            f"RUN={shlex.join([sys.executable, str(runner)])}",
            f"CUDA_CHECK_DIR={tmp_path / 'receipts with spaces'}",
            *options,
        ],
        cwd=MAKEFILE.parent,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return [json.loads(line) for line in output.read_text().splitlines()]


def test_cpu_target_covers_every_collected_file_including_root_and_hardware(tmp_path):
    root = MAKEFILE.parent
    expected = set(root.glob("tests/**/test_*.py")) | set(root.glob("tests/**/*_test.py"))
    covered = set()
    for record in recorded_commands(tmp_path, "test"):
        command = record["argv"]
        assert record["environment"]["CUDA_VISIBLE_DEVICES"] == "-1"
        assert command[:2] == ["pytest", "-q"]
        for argument in command:
            path = root / argument
            if argument.startswith("tests/"):
                if path.is_dir():
                    covered.update(path.rglob("test_*.py"))
                    covered.update(path.rglob("*_test.py"))
                else:
                    covered.add(path)
    assert expected <= covered, sorted(str(path.relative_to(root)) for path in expected - covered)


def test_explicit_cuda_target_runs_every_check_in_its_own_process(tmp_path, monkeypatch):
    for name in CHECK_DEVICES:
        monkeypatch.setenv(name, "cpu")
    records = recorded_commands(tmp_path, "test-cuda-checks")
    commands = [record["argv"] for record in records]
    expected = {
        str(path.relative_to(MAKEFILE.parent))
        for path in MAKEFILE.parent.glob("tests/**/cuda_*_check.py")
    }
    selected = [
        argument
        for command in commands
        for argument in command
        if argument.startswith("tests/") and argument.endswith("_check.py")
    ]
    assert set(selected) == expected
    assert len(selected) == len(expected)
    assert len(commands) == len(expected)
    assert all("-m" not in command for command in commands)
    assert all(record["environment"]["CUDA_VISIBLE_DEVICES"] == "0" for record in records)
    assert all(record["environment"]["MARS_TITAN_REQUIRE_CUDA"] == "1" for record in records)
    assert all(
        record["environment"][name] == "cuda:0" for record in records for name in CHECK_DEVICES
    )


def test_cuda_memory_target_selects_cuda_and_the_full_capacity_comparison(tmp_path):
    records = recorded_commands(tmp_path, "test-cuda-memory")
    assert len(records) == 1
    assert "tests/training/candidate_memory_check.py" in records[0]["argv"]
    env = records[0]["environment"]
    assert env["MARS_TITAN_CANDIDATE_MEMORY_CHECK_DEVICE"] == "cuda:0"
    assert env["MARS_TITAN_CANDIDATE_MEMORY_CHECK_ASSETS"] == "128,256,512"


def test_native_artifacts_are_declared_when_the_build_path_contains_spaces(tmp_path):
    build = tmp_path / "native build"
    artifacts = {
        "MARS_TITAN_EPISODIC_NATIVE": "native-candidate-cuda/_episodic_native.test.so",
        "MARS_TITAN_NATIVE_LIBRARY": "native-release/libmars_titan_simulation.so",
        "MARS_TITAN_SIM_EXECUTABLE": "native-release/mars-titan-sim",
        "MARS_TITAN_PPO_EXECUTABLE": "native-ppo-release/mars-titan-ppo",
        "MARS_TITAN_KLPO_EXECUTABLE": "native-ppo-release/mars-titan-klpo",
    }
    for name in artifacts.values():
        path = build / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    records = recorded_commands(tmp_path, "test", f"BUILD={build}")
    for record in records:
        assert {name: record["environment"][name] for name in artifacts} == {
            name: str(build / relative) for name, relative in artifacts.items()
        }


def test_every_ctest_call_excludes_optimizer_steps_and_cuda():
    text = MAKEFILE.read_text(encoding="utf-8")
    value = re.search(r"^CTEST_EXCLUDED := (.+)$", text, re.M).group(1).replace("$$", "$")
    for label in ("optimizer-steps", "cuda"):
        assert re.search(value, label)
    for label in ("unit", "integration", "ppo", "optimizer"):
        assert not re.search(value, label)
    calls = [line for line in text.splitlines() if "ctest " in line]
    assert calls
    for line in calls:
        # La exclusión puede ir en la línea siguiente de la misma orden.
        command = text[text.index(line) :].split("done", 1)[0]
        assert "-LE '$(CTEST_EXCLUDED)'" in command
