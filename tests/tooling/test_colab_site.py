"""La ficha del equipo anota lo que decide los bits y no lleva datos personales."""

import importlib.util
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("colab_site", ROOT / "scripts" / "colab_site.py")
site = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(site)


def smi(output):
    def run(command, **options):
        assert command[0] == "nvidia-smi" and options["timeout"] == 5 and options["check"]
        return SimpleNamespace(stdout=output)

    return run


def missing(*_, **__):
    raise FileNotFoundError("nvidia-smi")


def test_the_sheet_reads_the_gpu_and_keeps_no_personal_data(tmp_path, monkeypatch):
    monkeypatch.setenv("MARS_TITAN_SITE", "google-colab-rtx-pro-6000")
    run = smi("NVIDIA RTX PRO 6000 Blackwell Server Edition, 97887, 590.48.01\n")
    document = site.collect(disk=tmp_path, run=run, with_torch=False)
    assert document["site"] == "google-colab-rtx-pro-6000"
    assert document["gpu"] == dict(
        name="NVIDIA RTX PRO 6000 Blackwell Server Edition",
        memory_mib=97887,
        driver="590.48.01",
        capability=None,
    )
    assert document["disk"]["free_bytes"] > 0 and document["packages"]["numpy"]
    text = json.dumps(document)
    assert str(Path.home()) not in text and "hostname" not in text and "user" not in text


def test_a_gpu_with_unified_memory_keeps_its_name_and_driver():
    document = site.collect(run=smi("NVIDIA GB10, [N/A], 590.48.01\n"), with_torch=False)
    assert document["gpu"] == dict(
        name="NVIDIA GB10", memory_mib=None, driver="590.48.01", capability=None
    )


def test_a_machine_without_gpu_or_with_strange_output_says_none():
    assert site.collect(run=missing, with_torch=False)["gpu"] is None
    assert site.collect(run=smi("una sola columna\n"), with_torch=False)["gpu"] is None
    assert site.collect(run=smi(", 100, 1.0\n"), with_torch=False)["gpu"] is None

    def slow(*_, **__):
        raise subprocess.TimeoutExpired("nvidia-smi", 5)

    assert site.collect(run=slow, with_torch=False)["gpu"] is None


def sheet(**changes):
    document = dict(
        platform=dict(system="Linux", machine="aarch64"),
        gpu=dict(name="NVIDIA GB10", capability=[12, 1], driver="590.48.01"),
        torch=dict(cuda_runtime="13.0", cudnn=92400),
        packages=dict(torch="2.14.0+cu130", triton="3.8.0", numpy="2.5.3"),
        python="3.12.3",
        cpu=dict(model="uno", logical=20),
    )
    for path, value in changes.items():
        target = document
        *parents, leaf = path.split(".")
        for key in parents:
            target = target[key]
        target[leaf] = value
    return document


def test_two_equal_machines_have_no_differences_and_the_cpu_model_is_not_one():
    assert site.differences(sheet(), sheet(**{"cpu.model": "otro"})) == []


def test_the_differences_name_each_field_that_decides_the_bits():
    other = sheet(
        **{
            "platform.machine": "x86_64",
            "gpu.name": "NVIDIA RTX PRO 6000 Blackwell",
            "gpu.capability": [12, 0],
            "packages.torch": "2.14.0+cu130",
        }
    )
    fields = [row["field"] for row in site.differences(sheet(), other)]
    assert fields == ["platform.machine", "gpu.name", "gpu.capability"]
    assert site.differences(sheet(), dict(platform=None))[0]["second"] is None


def test_the_command_line_compares_two_sheets(tmp_path, capsys):
    first, second = tmp_path / "a.json", tmp_path / "b.json"
    first.write_text(json.dumps(sheet()))
    second.write_text(json.dumps(sheet(**{"gpu.driver": "600.1"})))
    assert site.main(["--compare", str(first), str(second)]) == 0
    output = capsys.readouterr().out
    assert "gpu.driver: '590.48.01' frente a '600.1'" in output and "1 diferencias" in output
    assert site.main(["--compare", str(first), str(first)]) == 0
    assert "coinciden" in capsys.readouterr().out
