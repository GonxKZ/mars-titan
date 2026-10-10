"""Perfiles de hardware: contrato, comprobación contra la plataforma, límites y mezcla."""

import json

import pytest

from mars_titan.hardware import hardware_profiles as profiles
from tests.hardware.platform_doubles import GIB, gb10, laptop

MIB = 1024**2


@pytest.fixture(scope="module")
def rtx():
    return profiles.load_profile("rtx4070-laptop")


@pytest.fixture(scope="module")
def spark():
    return profiles.load_profile("dgx-gb10")


def test_declared_profiles_load_with_their_digest(rtx, spark):
    assert rtx["memory"] == dict(model="dedicated", host_mib=24576, device_mib=7680)
    assert spark["memory"] == dict(model="unified", total_mib=65536)
    assert spark["gpu"] == dict(name="NVIDIA GB10", compute_capability=[12, 1])
    assert len(rtx["sha256"]) == 64 and rtx["sha256"] != spark["sha256"]


def write(folder, name, **changes):
    document = json.loads((profiles.PROFILES / "rtx4070-laptop.json").read_text())
    document.update(name=name, **changes)
    (folder / f"{name}.json").write_text(json.dumps(document))


@pytest.mark.parametrize(
    "changes",
    [
        dict(schema_version=2),
        dict(kind="otro"),
        dict(machine="riscv64"),
        dict(gpu=dict(name="NVIDIA X")),
        dict(gpu=dict(name="NVIDIA X", compute_capability=[8])),
        dict(gpu=dict(name="NVIDIA X", compute_capability=[8, -1])),
        dict(memory=dict(model="dedicated", host_mib=1024)),
        dict(memory=dict(model="unified", total_mib=0)),
        dict(memory=dict(model="swap", total_mib=1024)),
        dict(cpu_threads=0),
        dict(extra=True),
    ],
)
def test_a_profile_outside_its_contract_is_rejected(tmp_path, changes):
    write(tmp_path, "candidate", **changes)
    with pytest.raises(ValueError):
        profiles.load_profile("candidate", folder=tmp_path)


def test_profile_names_are_checked_against_the_file(tmp_path):
    write(tmp_path, "other")
    (tmp_path / "renamed.json").write_text((tmp_path / "other.json").read_text())
    with pytest.raises(ValueError, match="contrato"):
        profiles.load_profile("renamed", folder=tmp_path)
    for name in ("../rtx4070-laptop", "RTX", "missing"):
        with pytest.raises(ValueError):
            profiles.load_profile(name, folder=tmp_path)


def test_each_machine_matches_only_its_own_profile(rtx, spark):
    assert profiles.hardware_mismatches(rtx, laptop()) == []
    assert profiles.hardware_mismatches(spark, gb10()) == []
    assert len(profiles.hardware_mismatches(spark, laptop())) >= 3
    assert len(profiles.hardware_mismatches(rtx, gb10())) >= 3


def gpu(**changes):
    return [dict(laptop()["gpus"][0], **changes)]


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        (dict(emulated=True), "emulada"),
        (dict(machine="aarch64"), "arquitectura"),
        (dict(gpus=[]), "ninguna GPU"),
        (dict(gpus=gpu(name="NVIDIA GeForce RTX 4060 Laptop GPU")), "GPU NVIDIA GeForce RTX 4060"),
        (dict(gpus=gpu(compute_capability=[8, 6])), "capacidad"),
        (dict(gpus=gpu(memory_total_mib=7679)), "7680 MiB propios"),
        (dict(gpus=gpu(memory_total_mib=None)), "7680 MiB propios"),
        (dict(memory_total_bytes=24576 * MIB - 1), "menos de 24576 MiB"),
        (dict(cpu=dict(laptop()["cpu"], logical=15)), "menos de 16 hilos"),
    ],
)
def test_each_difference_is_reported_once(rtx, changes, message):
    found = profiles.hardware_mismatches(rtx, laptop(**changes))
    assert len(found) == 1 and message in found[0]


def test_limits_at_the_boundary_still_match(rtx, spark):
    exact = laptop(gpus=gpu(memory_total_mib=7680), memory_total_bytes=24576 * MIB)
    assert profiles.hardware_mismatches(rtx, exact) == []
    assert profiles.hardware_mismatches(spark, gb10(memory_total_bytes=65536 * MIB)) == []
    found = profiles.hardware_mismatches(spark, gb10(memory_total_bytes=65536 * MIB - 1))
    assert len(found) == 1 and "65536" in found[0]


def test_unified_memory_needs_a_cgroup_within_the_limit(spark):
    with pytest.raises(ValueError, match="cgroup"):
        profiles.check_profile(spark, gb10(), memory_limit=lambda: None)
    with pytest.raises(ValueError, match="cgroup"):
        profiles.check_profile(spark, gb10(), memory_limit=lambda: 65536 * MIB + 1)
    profiles.check_profile(spark, gb10(), memory_limit=lambda: 65536 * MIB)


def test_dedicated_memory_does_not_read_the_cgroup(rtx):
    def unexpected():
        raise AssertionError("El perfil dedicado no consulta el cgroup")

    profiles.check_profile(rtx, laptop(), memory_limit=unexpected)
    with pytest.raises(ValueError, match="rtx4070-laptop: GPU"):
        profiles.check_profile(rtx, laptop(gpus=gpu(name="NVIDIA GB10")), memory_limit=unexpected)


def cgroup(tmp_path, limits, line="0::/user.slice/run-1.scope"):
    tmp_path.mkdir(parents=True, exist_ok=True)
    proc = tmp_path / "cgroup"
    proc.write_text(f"{line}\n")
    root = tmp_path / "fs"
    for relative, value in limits.items():
        folder = root / relative
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "memory.max").write_text(f"{value}\n")
    return proc, root


def test_cgroup_limit_is_the_tightest_of_the_process_and_its_parents(tmp_path):
    proc, root = cgroup(
        tmp_path, {"": "max", "user.slice": str(8 * GIB), "user.slice/run-1.scope": str(16 * GIB)}
    )
    assert profiles.cgroup_memory_limit(proc, root) == 8 * GIB
    proc, root = cgroup(tmp_path / "free", {"user.slice/run-1.scope": "max"})
    assert profiles.cgroup_memory_limit(proc, root) is None
    proc, root = cgroup(tmp_path / "v1", {}, line="1:memory:/user.slice")
    assert profiles.cgroup_memory_limit(proc, root) is None


def test_execution_budgets_must_fit_the_profile(rtx, spark):
    profiles.check_budgets(rtx, vram_bytes=7680 * MIB, host_bytes=24576 * MIB, threads=16)
    for changes in (
        dict(vram_bytes=7680 * MIB + 1),
        dict(host_bytes=24576 * MIB + 1),
        dict(threads=17),
    ):
        arguments = dict(vram_bytes=0, host_bytes=0, threads=1) | changes
        with pytest.raises(ValueError, match="rtx4070-laptop"):
            profiles.check_budgets(rtx, **arguments)
    # En memoria unificada VRAM y RAM comparten un único límite.
    profiles.check_budgets(spark, vram_bytes=32 * GIB, host_bytes=32 * GIB, threads=8)
    with pytest.raises(ValueError, match="juntas"):
        profiles.check_budgets(spark, vram_bytes=32 * GIB, host_bytes=32 * GIB + 1, threads=8)
    with pytest.raises(ValueError, match="8"):
        profiles.check_budgets(spark, vram_bytes=None, host_bytes=None, threads=9)


def test_a_comparison_keeps_one_platform(rtx, spark):
    same = profiles.same_platform(dict(a=laptop(), b=laptop(kernel="7.1.0")))
    assert same == dict(
        platform_sha256=laptop()["platform_sha256"],
        recorded=2,
        unrecorded=0,
        unrecorded_profile=None,
    )
    with pytest.raises(ValueError, match="mezcla plataformas"):
        profiles.same_platform(dict(a=laptop(), b=gb10()))
    with pytest.raises(ValueError, match="no corresponde a su contenido"):
        profiles.same_platform(dict(a=dict(laptop(), machine="aarch64")))
    with pytest.raises(ValueError, match="No hay"):
        profiles.same_platform({})


def test_unrecorded_runs_need_a_declared_profile_that_the_others_match(rtx, spark):
    records = dict(old=None, new=laptop())
    with pytest.raises(ValueError, match="no registran su plataforma"):
        profiles.same_platform(records)
    result = profiles.same_platform(records, unrecorded=rtx)
    assert result["unrecorded"] == 1 and result["unrecorded_profile"] == "rtx4070-laptop"
    with pytest.raises(ValueError, match="perfil dgx-gb10 atribuido"):
        profiles.same_platform(records, unrecorded=spark)
    only_old = profiles.same_platform(dict(old=None), unrecorded=spark)
    assert only_old == dict(
        platform_sha256=None, recorded=0, unrecorded=1, unrecorded_profile="dgx-gb10"
    )


def test_commands_without_an_execution_declaration_name_their_profile():
    with pytest.raises(ValueError, match="MARS_TITAN_HARDWARE_PROFILE"):
        profiles.declared_profile({})
    assert profiles.declared_profile({profiles.PROFILE_ENV: "dgx-gb10"})["name"] == "dgx-gb10"
