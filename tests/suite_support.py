"""Apoyo común de la suite que no depende de ningún dominio del proyecto.

La única excepción es la carga del enlace episódico en el modo estricto, que se importa
dentro de su función para que la suite general no la necesite.
"""

import importlib
import os
import sys
import tomllib
from contextlib import contextmanager
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


# Hilos de PyTorch en CPU con los que se capturaron las huellas FP32 bit a bit. El reparto
# de las reducciones y los productos de matrices entre hilos cambia el orden de las sumas,
# así que con otro número de hilos cambian los bits (#572).
REFERENCE_CPU_THREADS = 2


@contextmanager
def cpu_threads(count=REFERENCE_CPU_THREADS):
    """Fija los hilos de PyTorch en CPU durante el bloque y restaura los que había."""
    import torch

    previous = torch.get_num_threads()
    torch.set_num_threads(count)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def cuda_available():
    try:
        import torch
    except ModuleNotFoundError:
        return False
    return torch.cuda.is_available()


# Marcas de selección. `cuda` y `native` declaran lo que necesita la prueba, y la sesión omite
# con un motivo común las que no se pueden ejecutar. `slow` se aplica desde SLOW_TESTS y
# `unguarded_fit` señala las pruebas que ajustan sin pasar por un punto de entrada protegido.
CUDA_MARKER, NATIVE_SELECTION, SLOW_MARKER = "cuda", "native", "slow"
UNGUARDED_FIT_MARKER = "unguarded_fit"
SLOW_TESTS = ROOT / "tests" / "slow_tests.txt"
BINARIES_GUIDE = "tests/README.md, sección Binarios nativos"

CUDA_REASON = (
    "Requiere cuda:0 visible, con una GPU NVIDIA, su controlador y el PyTorch +cu130 que "
    "instala `make setup`. Nunca se sustituye por CPU"
)
# Artefacto que nombra cada variable, preset que lo compila con `make native-build` y archivo
# que se usa sin la variable, como hacen sus consumidores. El enlace episódico no tiene ruta
# por defecto: siempre se declara.
NATIVE_ARTIFACTS = {
    "MARS_TITAN_EPISODIC_NATIVE": ("el enlace episódico nativo", "native-candidate-cuda", None),
    "MARS_TITAN_NATIVE_LIBRARY": (
        "la biblioteca de simulación",
        "native-release",
        "libmars_titan_simulation.so",
    ),
    "MARS_TITAN_SIM_EXECUTABLE": (
        "el ejecutable mars-titan-sim",
        "native-release",
        "mars-titan-sim",
    ),
    "MARS_TITAN_PPO_EXECUTABLE": (
        "el ejecutable mars-titan-ppo",
        "native-ppo-release",
        "mars-titan-ppo",
    ),
    "MARS_TITAN_KLPO_EXECUTABLE": (
        "el ejecutable mars-titan-klpo",
        "native-ppo-release",
        "mars-titan-klpo",
    ),
}

# Sin CUDA visible la prueba se omite con CUDA_REASON y nunca pasa a CPU.
# MARS_TITAN_REQUIRE_CUDA=1 convierte esa ausencia en un error de la sesión completa.
requires_cuda = pytest.mark.cuda

# Integraciones CUDA que aplican pasos de optimizador. Además de la marca `cuda` necesitan
# una activación explícita, que solo procede fuera del bloqueo de aprendizaje.
CUDA_INTEGRATION = "MARS_TITAN_CUDA_INTEGRATION"
requires_cuda_integration = pytest.mark.skipif(
    os.environ.get(CUDA_INTEGRATION) != "1",
    reason=(
        f"Aplica pasos de optimizador en cuda:0. Se activa con {CUDA_INTEGRATION}=1 en una "
        "ventana GPU exclusiva y solo cuando el bloqueo de aprendizaje lo permite"
    ),
)


def skip_without_cuda():
    """Omitir con CUDA_REASON desde una preparación que necesita cuda:0.

    Las pruebas que la llegan a usar llevan también la marca `cuda`, que las omite antes y
    permite seleccionarlas.
    """
    if not cuda_available():
        pytest.skip(CUDA_REASON)


def missing_native(variable):
    """Motivo de omisión que dice qué artefacto nativo falta y cómo conseguirlo."""
    artifact, preset, default = NATIVE_ARTIFACTS[variable]
    where = f"se declara su ruta en {variable}"
    if default is not None:
        where = f"se deja en build/native/{preset}/{default} o {where}"
    return (
        f"Falta {artifact}. Se compila con `make native-build` (preset {preset}) y {where} "
        f"({BINARIES_GUIDE})"
    )


def native_available(variable, environment, root=ROOT):
    """Indica si el artefacto está declarado o, si admite ruta por defecto, compilado allí."""
    if environment.get(variable):
        return True
    _, preset, default = NATIVE_ARTIFACTS[variable]
    return default is not None and (root / "build" / "native" / preset / default).is_file()


def selection_skip(markers, environment, cuda=cuda_available, root=ROOT):
    """Motivo para omitir una prueba por lo que declaran sus marcas, o None si puede correr.

    `markers` asocia cada nombre de marca con los argumentos de sus apariciones. `native`
    sin argumentos solo sirve para seleccionar. Con nombres de variables, la prueba se omite
    si falta alguno de sus artefactos. Una ruta declarada que no existe no se omite y la
    prueba falla al cargarla, como en el resto de la suite.
    """
    for arguments in markers.get(NATIVE_SELECTION, ()):
        for variable in arguments:
            if variable not in NATIVE_ARTIFACTS:
                raise ValueError(f"Variable nativa desconocida en la marca native: {variable}")
            if not native_available(variable, environment, root):
                return missing_native(variable)
    if CUDA_MARKER in markers and not cuda():
        return CUDA_REASON
    return None


# Variables con las que la sesión exterior pasa opciones o complementos, como `--junitxml`, a
# pytest. Una sesión lanzada desde una prueba no debe heredarlas.
OUTER_SESSION = ("PYTEST_ADDOPTS", "PYTEST_PLUGINS")


def inner_environment(removed=()):
    """Entorno para una sesión de pytest lanzada desde una prueba, sin las opciones exteriores."""
    skipped = {*OUTER_SESSION, *removed}
    return {key: value for key, value in os.environ.items() if key not in skipped}


def read_slow_tests(path=SLOW_TESTS):
    """Identificadores de las funciones lentas, sin parámetros ni comentarios."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return {line.strip() for line in lines if line.strip() and not line.startswith("#")}


def function_id(nodeid):
    """Identificador de una prueba sin la parte de sus parámetros."""
    return nodeid.split("[", 1)[0]


# El enlace de native-release no incluye el candidato GRU ni el backend CUDA. Estas pruebas
# necesitan el de native-candidate-cuda.
CANDIDATE_BINDING_REASON = (
    "El enlace episódico declarado no incluye el candidato GRU. Hace falta el del preset "
    f"native-candidate-cuda, que compila `make native-build` ({BINARIES_GUIDE})"
)
CUDA_BINDING_REASON = (
    "El enlace episódico declarado se compiló sin backend CUDA. Hace falta el del preset "
    f"native-candidate-cuda, que compila `make native-build` ({BINARIES_GUIDE})"
)


# Extra de `pyproject.toml` que instala cada biblioteca opcional que comprueban las pruebas.
MODULE_EXTRAS = {
    "torch": "cuda",
    "duckdb": "data",
    "gymnasium": "reinforcement",
    "cupy": "boosting",
    "xgboost": "boosting",
    "sklearn": "research",
}


def missing_extra(*modules):
    """Motivo de omisión por bibliotecas de extras sin instalar y la orden que las instala."""
    names = " o ".join(f"{module} (extra {MODULE_EXTRAS[module]})" for module in modules)
    return (
        f"Falta {names}. Se instala con `make setup`, que ejecuta `uv sync --locked --all-extras`"
    )


UNADJUSTED_EDITION_REASON = (
    "Faltan MARS_TITAN_UNADJUSTED_EDITION y MARS_TITAN_LISTING_STATUS, con las rutas de una "
    "edición de precios sin ajustar y de su listing-status.json (docs/data/unadjusted-prices.md). "
    "La prueba solo las lee"
)
HMMLEARN_REASON = (
    "Falta hmmlearn 0.3.3, que no está en uv.lock. Se añade con "
    "`uv run --with hmmlearn==0.3.3`, y como ajusta el HMM espera a que el bloqueo de "
    "aprendizaje lo permita"
)


def skip_without_episodic_native():
    """Omitir con su motivo una preparación que carga el enlace episódico nativo sin declararlo."""
    if not os.environ.get("MARS_TITAN_EPISODIC_NATIVE"):
        pytest.skip(missing_native("MARS_TITAN_EPISODIC_NATIVE"))


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
            f"Falta {error.name} del grupo de dependencias {REFERENCE_GROUP}. Se instala con "
            f"`uv sync --locked --all-extras --group {REFERENCE_GROUP}`"
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
