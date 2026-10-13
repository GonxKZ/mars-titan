"""Perfiles de hardware: contrato, comprobación contra la plataforma, límites y mezcla."""

import json

import pytest

from mars_titan.hardware import hardware_profiles as profiles
from tests.hardware.platform_doubles import desktop, laptop

MIB = 1024**2


@pytest.fixture(scope="module")
def rtx():
    return profiles.load_profile("rtx4070-laptop")


def write(folder, name, **changes):
    document = json.loads((profiles.PROFILES / "rtx4070-laptop.json").read_text())
    document.update(name=name, **changes)
    (folder / f"{name}.json").write_text(json.dumps(document))


@pytest.fixture(scope="module")
def workstation(tmp_path_factory):
    """Perfil de otra máquina x86-64, escrito solo para las pruebas."""
    folder = tmp_path_factory.mktemp("profiles")
    write(
        folder,
        "rtx4090-desktop",
        gpu=dict(name="NVIDIA GeForce RTX 4090", compute_capability=[8, 9]),
        memory=dict(model="dedicated", host_mib=57344, device_mib=23552),
        cpu_threads=32,
    )
    return profiles.load_profile("rtx4090-desktop", folder=folder)


def test_the_laptop_is_the_only_declared_profile(rtx):
    assert sorted(path.stem for path in profiles.PROFILES.glob("*.json")) == ["rtx4070-laptop"]
    assert rtx["machine"] == "x86_64"
    assert rtx["memory"] == dict(model="dedicated", host_mib=24576, device_mib=7680)
    assert len(rtx["sha256"]) == 64


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
        dict(memory=dict(model="unified", total_mib=65536)),
        dict(memory=dict(model="swap", host_mib=1024, device_mib=1024)),
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


def test_each_machine_matches_only_its_own_profile(rtx, workstation):
    assert profiles.hardware_mismatches(rtx, laptop()) == []
    assert profiles.hardware_mismatches(workstation, desktop()) == []
    assert len(profiles.hardware_mismatches(workstation, laptop())) >= 3
    assert len(profiles.hardware_mismatches(rtx, desktop())) >= 1


def gpu(**changes):
    return [dict(laptop()["gpus"][0], **changes)]


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        (dict(emulated=True), "emulada"),
        (dict(machine="riscv64"), "arquitectura"),
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


def test_every_gpu_of_the_machine_must_match_the_profile(rtx):
    # nvidia-smi lista todas las GPU NVIDIA, también las que CUDA_VISIBLE_DEVICES oculta.
    other = gpu(name="NVIDIA GeForce RTX 4060 Laptop GPU", uuid="GPU-other")[0]
    found = profiles.hardware_mismatches(rtx, laptop(gpus=[*gpu(), other]))
    assert len(found) == 1 and "RTX 4060" in found[0]


def test_limits_at_the_boundary_still_match(rtx):
    exact = laptop(gpus=gpu(memory_total_mib=7680), memory_total_bytes=24576 * MIB)
    assert profiles.hardware_mismatches(rtx, exact) == []


def test_check_profile_stops_on_the_first_report(rtx):
    profiles.check_profile(rtx, laptop())
    with pytest.raises(ValueError, match="rtx4070-laptop: GPU"):
        profiles.check_profile(rtx, laptop(gpus=gpu(name="NVIDIA GeForce RTX 4090")))


def test_execution_budgets_must_fit_the_profile(rtx):
    profiles.check_budgets(rtx, vram_bytes=7680 * MIB, host_bytes=24576 * MIB, threads=16)
    for changes in (
        dict(vram_bytes=7680 * MIB + 1),
        dict(host_bytes=24576 * MIB + 1),
        dict(threads=17),
    ):
        arguments = dict(vram_bytes=0, host_bytes=0, threads=1) | changes
        with pytest.raises(ValueError, match="rtx4070-laptop"):
            profiles.check_budgets(rtx, **arguments)
    # Sin presupuesto declarado, la VRAM y la RAM cuentan como cero.
    profiles.check_budgets(rtx, vram_bytes=None, host_bytes=None, threads=16)


def test_a_comparison_keeps_one_platform(rtx):
    same = profiles.same_platform(dict(a=laptop(), b=laptop(kernel="7.1.0")))
    assert same == dict(
        platform_sha256=laptop()["platform_sha256"],
        recorded=2,
        unrecorded=0,
        unrecorded_profile=None,
    )
    with pytest.raises(ValueError, match="mezcla plataformas"):
        profiles.same_platform(dict(a=laptop(), b=desktop()))
    # Otro PyTorch en el mismo portátil también es otra plataforma.
    other_torch = laptop(torch=dict(version="2.15.0+cu130", cuda="13.0", cudnn=92400))
    with pytest.raises(ValueError, match="mezcla plataformas"):
        profiles.same_platform(dict(a=laptop(), b=other_torch))
    with pytest.raises(ValueError, match="no corresponde a su contenido"):
        profiles.same_platform(dict(a=dict(laptop(), emulated=True)))
    with pytest.raises(ValueError, match="No hay"):
        profiles.same_platform({})


def test_unrecorded_runs_need_a_declared_profile_that_the_others_match(rtx, workstation):
    records = dict(old=None, new=laptop())
    with pytest.raises(ValueError, match="no registran su plataforma"):
        profiles.same_platform(records)
    result = profiles.same_platform(records, unrecorded=rtx)
    assert result["unrecorded"] == 1 and result["unrecorded_profile"] == "rtx4070-laptop"
    # Sin ejecuciones anteriores al registro, el perfil declarado no se atribuye a ninguna.
    complete = profiles.same_platform(dict(new=laptop()), unrecorded=rtx)
    assert complete["unrecorded"] == 0 and complete["unrecorded_profile"] is None
    with pytest.raises(ValueError, match="perfil rtx4090-desktop atribuido"):
        profiles.same_platform(records, unrecorded=workstation)
    only_old = profiles.same_platform(dict(old=None), unrecorded=workstation)
    assert only_old == dict(
        platform_sha256=None, recorded=0, unrecorded=1, unrecorded_profile="rtx4090-desktop"
    )


def recorded(**changes):
    """Registro `platform` de unas fuentes de la base, como lo escribe `same_platform`."""
    record = dict(
        platform_sha256=laptop()["platform_sha256"],
        recorded=3,
        unrecorded=0,
        unrecorded_profile=None,
    )
    return record | changes


def test_a_stage_inherits_the_platform_recorded_by_its_base(rtx):
    stage = dict(a=laptop(), b=laptop(kernel="7.1.0"))
    assert profiles.inherited_platform(stage, recorded()) == profiles.same_platform(stage)
    with pytest.raises(ValueError, match="otra plataforma que la base"):
        profiles.inherited_platform(dict(a=desktop()), recorded())
    # El manifiesto de la base solo guarda la huella: sin la de la etapa no hay comparación.
    with pytest.raises(ValueError, match="los recibos de la etapa no"):
        profiles.inherited_platform(dict(a=None), recorded(), unrecorded=rtx)
    # Una etapa que mezcla plataformas se rechaza antes de mirar la base.
    with pytest.raises(ValueError, match="mezcla plataformas"):
        profiles.inherited_platform(dict(a=laptop(), b=desktop()), recorded())


def test_a_base_without_platform_needs_the_declared_profile(rtx, workstation):
    stage = dict(a=laptop())
    unrecorded_base = recorded(
        platform_sha256=None, recorded=0, unrecorded_profile="rtx4070-laptop"
    )
    for base in (None, unrecorded_base):
        with pytest.raises(ValueError, match="no registran su plataforma"):
            profiles.inherited_platform(stage, base)
        assert profiles.inherited_platform(stage, base, unrecorded=rtx)["recorded"] == 1
        # La etapa registrada debe corresponder al perfil al que se atribuye la base.
        with pytest.raises(ValueError, match="atribuido a las fuentes de la base"):
            profiles.inherited_platform(stage, base, unrecorded=workstation)
    # La base ya atribuyó sus recibos a un perfil: la etapa no puede atribuirlos a otro.
    with pytest.raises(ValueError, match="perfiles distintos"):
        profiles.inherited_platform(
            dict(a=None), dict(unrecorded_base, unrecorded_profile="otro"), unrecorded=rtx
        )
    with pytest.raises(ValueError, match="perfiles distintos"):
        profiles.inherited_platform(
            dict(a=laptop(), b=None),
            recorded(unrecorded=1, unrecorded_profile="otro"),
            unrecorded=rtx,
        )
    mixed = profiles.inherited_platform(
        dict(a=laptop(), b=None),
        recorded(unrecorded=1, unrecorded_profile="rtx4070-laptop"),
        unrecorded=rtx,
    )
    assert mixed["unrecorded_profile"] == "rtx4070-laptop"


def test_only_a_declared_name_attributes_unrecorded_runs_to_a_profile():
    assert profiles.attributed_profile(None) is None
    assert profiles.attributed_profile("rtx4070-laptop")["name"] == "rtx4070-laptop"
    with pytest.raises(ValueError):
        profiles.attributed_profile("otra-maquina")


def test_commands_without_an_execution_declaration_name_their_profile():
    with pytest.raises(ValueError, match="MARS_TITAN_HARDWARE_PROFILE"):
        profiles.declared_profile({})
    declared = profiles.declared_profile({profiles.PROFILE_ENV: "rtx4070-laptop"})
    assert declared["name"] == "rtx4070-laptop"
