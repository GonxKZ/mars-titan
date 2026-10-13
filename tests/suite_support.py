"""Apoyo común de la suite que no depende de ningún dominio del proyecto.

La única excepción es la carga del enlace episódico en el modo estricto, que se importa
dentro de su función para que la suite general no la necesite.
"""

import importlib
import os
import sys
import tomllib
from importlib import metadata
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# Binarios nativos que la comprobación local completa debe declarar con una ruta existente.
NATIVE_VARIABLES = (
    "MARS_TITAN_EPISODIC_NATIVE",
    "MARS_TITAN_NATIVE_LIBRARY",
    "MARS_TITAN_SIM_EXECUTABLE",
    "MARS_TITAN_PPO_EXECUTABLE",
    "MARS_TITAN_KLPO_EXECUTABLE",
)
REQUIRE_NATIVE, REQUIRE_CUDA = "MARS_TITAN_REQUIRE_NATIVE", "MARS_TITAN_REQUIRE_CUDA"
REQUIRE_REFERENCE = "MARS_TITAN_REQUIRE_REFERENCE"
SWITCHES = (REQUIRE_NATIVE, REQUIRE_CUDA, REQUIRE_REFERENCE)
# Marca de las pruebas que necesitan un binario nativo. En modo estricto su omisión cuenta
# como fallo.
NATIVE_MARKER = "native_binding"
# Marca de las paridades con bibliotecas externas del grupo de dependencias `reference`.
REFERENCE_MARKER = "external_reference"
REFERENCE_GROUP = "reference"
# Variable que convierte en fallo la omisión de cada marca y cómo se nombra la prueba.
STRICT_MARKERS = {
    NATIVE_MARKER: (REQUIRE_NATIVE, "del enlace nativo"),
    REFERENCE_MARKER: (REQUIRE_REFERENCE, "de referencia externa"),
}


def cuda_available():
    try:
        import torch
    except ModuleNotFoundError:
        return False
    return torch.cuda.is_available()


# Sin CUDA visible la prueba se omite con su motivo y nunca pasa a CPU. MARS_TITAN_REQUIRE_CUDA=1
# convierte esa ausencia en un error de la sesión completa.
requires_cuda = pytest.mark.skipif(
    not cuda_available(), reason="Requiere cuda:0, que no está visible en esta ejecución"
)


def skip_without_episodic_native():
    """Omitir con su motivo una preparación que carga el enlace episódico nativo sin declararlo."""
    if not os.environ.get("MARS_TITAN_EPISODIC_NATIVE"):
        pytest.skip("Falta el enlace episódico nativo compilado (MARS_TITAN_EPISODIC_NATIVE)")


def episodic_load_problem(path):
    """Motivo por el que el enlace episódico declarado no carga, o None si carga.

    Una ruta existente no basta, porque un enlace compilado con otro PyTorch u otro contrato
    se rechaza al cargarlo y sus pruebas se omitirían después una a una.
    """
    from mars_titan.memory.native_backend import load_native

    try:
        load_native(path)
    except Exception as error:
        return f"MARS_TITAN_EPISODIC_NATIVE no carga como enlace válido: {error}"
    return None


def native_required(environment):
    """Indica si la sesión exige los binarios nativos. El valor ya se validó al empezar."""
    return environment.get(REQUIRE_NATIVE) == "1"


def strict_skip(markers, environment):
    """Texto del fallo de una prueba omitida con estas marcas, o None si la omisión vale.

    La omisión solo falla cuando todo lo que la prueba necesita está exigido. Una prueba que
    compara el núcleo nativo con una biblioteca externa puede omitirse por falta del binario
    si solo se exige el grupo `reference`, y al revés.
    """
    present = [STRICT_MARKERS[marker] for marker in STRICT_MARKERS if marker in markers]
    if not present or any(environment.get(switch) != "1" for switch, _ in present):
        return None
    switches = ", ".join(f"{switch}=1" for switch, _ in present)
    descriptions = " y ".join(description for _, description in present)
    return f"{switches} y la prueba {descriptions} se omitió"


def reference_pins(pyproject=ROOT / "pyproject.toml"):
    """Distribución y versión exacta de cada biblioteca del grupo `reference`.

    Se leen de `pyproject.toml` para que el modo estricto compruebe lo mismo que fija el
    lock. Un requisito sin versión exacta se rechaza, porque las tolerancias de las paridades
    se declararon para esas versiones.
    """
    document = tomllib.loads(Path(pyproject).read_text(encoding="utf-8"))
    pins = {}
    for requirement in document["dependency-groups"][REFERENCE_GROUP]:
        name, separator, version = requirement.partition("==")
        if not separator or not name or not version or any(c.isspace() for c in requirement):
            raise ValueError(
                f"El grupo {REFERENCE_GROUP} debe fijar versiones exactas: {requirement}"
            )
        pins[name] = version
    return pins


def reference_problems(pins=None, installed=metadata.version):
    """Bibliotecas del grupo `reference` ausentes o con otra versión que la fijada."""
    problems = []
    for name, version in (reference_pins() if pins is None else pins).items():
        try:
            found = installed(name)
        except metadata.PackageNotFoundError:
            problems.append(f"falta {name}=={version} del grupo {REFERENCE_GROUP}")
            continue
        if found != version:
            problems.append(f"{name} {found} instalado, el grupo {REFERENCE_GROUP} fija {version}")
    return problems


def reference_module(name):
    """Importar una biblioteca del grupo `reference` u omitir la prueba con su motivo.

    Solo se omite cuando falta el propio paquete. Si el paquete está pero falla una de sus
    dependencias, la instalación está rota y el error sigue apareciendo como tal.
    """
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as error:
        if error.name != name.partition(".")[0]:
            raise
        pytest.skip(
            f"Falta {error.name} del grupo de dependencias {REFERENCE_GROUP} "
            f"(uv sync --group {REFERENCE_GROUP})"
        )


def strict_problems(
    environment, cuda=cuda_available, episodic=episodic_load_problem, reference=reference_problems
):
    """Requisitos incumplidos de la comprobación local completa, vacío si no hay ninguno.

    En la suite general las pruebas que dependen de un binario nativo, de CUDA o del grupo
    `reference` se omiten con su motivo. `MARS_TITAN_REQUIRE_NATIVE=1` exige declarar cada
    binario con una ruta existente y además cargar el enlace episódico.
    `MARS_TITAN_REQUIRE_CUDA=1` exige CUDA visible, así que ninguna prueba se omite por esa
    causa. `MARS_TITAN_REQUIRE_REFERENCE=1` exige el grupo `reference` con las versiones
    exactas de `pyproject.toml`.
    """
    problems = [
        f"{switch} debe valer 0 o 1"
        for switch in SWITCHES
        if environment.get(switch, "0") not in ("0", "1")
    ]
    if native_required(environment):
        for variable in NATIVE_VARIABLES:
            value = environment.get(variable)
            if not value:
                problems.append(f"falta declarar {variable}")
            elif not Path(value).is_file():
                problems.append(f"{variable} no apunta a un archivo: {value}")
            elif variable == "MARS_TITAN_EPISODIC_NATIVE" and (problem := episodic(value)):
                problems.append(problem)
    if environment.get("MARS_TITAN_REQUIRE_CUDA") == "1" and not cuda():
        problems.append("CUDA no está disponible")
    if environment.get(REQUIRE_REFERENCE) == "1":
        problems.extend(reference())
    return problems


def python_shebang(directory, executable=None):
    """Línea #! que arranca el intérprete y el entorno actuales desde una ruta sin espacios.

    Linux corta la línea #! en el primer espacio. Con el entorno en una ruta con espacios,
    como el repositorio principal, un ejecutable falso no arranca y la búsqueda en PATH pasa
    al siguiente con el mismo nombre, por ejemplo el nvidia-smi real. El enlace apunta al
    directorio del entorno, así que Python sigue encontrando `pyvenv.cfg` junto al ejecutable.
    """
    executable = Path(executable or sys.executable)
    link = Path(directory) / "environment"
    link.symlink_to(executable.parent.parent, target_is_directory=True)
    line = f"#!{link / executable.parent.name / executable.name}"
    if any(character.isspace() for character in line):
        raise ValueError(f"La ruta del enlace al intérprete contiene espacios: {line}")
    return line + "\n"
