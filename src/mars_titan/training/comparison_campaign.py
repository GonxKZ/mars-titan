"""Campaña de comparación en dos equipos (#836): declaración, trabajos y cola reanudable.

La declaración (`configs/experiments/comparison-campaign-v1.json`) fija antes de ejecutar la
población, los orígenes, las semillas, el presupuesto de épocas, las recetas y los modelos de
cada equipo. Este módulo la convierte en una lista ordenada de trabajos y los ejecuta de uno
en uno o con varios trabajadores sobre la misma GPU. No ajusta nada por sí mismo: cada trabajo
es una ventana de los ejecutores ya probados (`titans_walk_forward.run_titans_window`,
`mars_titan_walk_forward.run_mars_titan_window`, los controles comunes de CPU y Ridge), que
comprueban la protección del aprendizaje, guardan checkpoints coherentes y se reanudan con la
misma llamada.

Reparto: cada bloque de año y semilla conserva todos sus modelos neuronales en un equipo,
incluidos padre y lectores. Los controles por origen permanecen en `references`. Los dos
equipos leen la misma edición, las mismas cohortes y las mismas recetas.

Orden: etapa, nivel de prioridad, semilla, origen y modelo. Así la primera semilla de todos
los orígenes y modelos de un nivel termina antes de empezar la segunda, y un plazo que se
agote deja comparaciones completas en lugar de modelos a medias. Los orígenes de papel
`history` forman una segunda etapa que solo empieza al terminar la primera.

Cada trabajo se ejecuta en un proceso propio. Un fallo queda registrado con su traza, se
reintenta hasta el máximo declarado y no detiene la cola. SIGINT o SIGTERM pausan cada
trabajo en su siguiente barrera confirmada, y la misma orden continúa después.
"""

import argparse
import fcntl
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from mars_titan.data.cohort_files import read_manifest, safe_destination
from mars_titan.data.storage import atomic_json

from .comparison_cohorts import write_cohort
from .input_pipeline import PipelineOptions
from .learning_hold import HOLD_ENV
from .selection import CHRONOLOGICAL_MODES, validate_selection
from .window_indices import SHARED_INDICES_ENV

ROOT = Path(__file__).resolve().parents[3]
DECLARATION = "configs/experiments/comparison-campaign-v1.json"
KIND = "comparison_campaign_v1"
MACHINES = ("references", "mars_titan")
# Los orígenes `history` son años anteriores que amplían la evidencia. Forman una segunda
# etapa, que empieza cuando la primera ha recorrido todos sus niveles.
HISTORY = "history"
ROLES = ("development", "confirmation", HISTORY)
RUNNERS = ("fit", "reader", "rule", "tabular", "ridge")
GPU_RUNNERS = ("fit", "reader")
# Estos controles ajustan todas las semillas en un trabajo o no dependen de ninguna.
ORIGIN_RUNNERS = ("rule", "tabular", "ridge")
# Ridge acumula sus ecuaciones normales en la GPU, como los ajustes y los lectores.
CUDA_RUNNERS = (*GPU_RUNNERS, "ridge")
# Ejecutores cuya predicción es una magnitud del residuo. La regla y el control tabular
# predicen un orden, y su MAE sin calibrar no es comparable.
MAGNITUDE_RUNNERS = ("fit", "reader", "ridge")
# Carpetas del repositorio que deciden qué ejecuta la campaña.
CODE_PATHS = ("src", "configs", "native", "pyproject.toml", "uv.lock", ".python-version")
STATE_ENV = "MARS_TITAN_COMPARISON_STATE"
VIEWS_ENV = "MARS_TITAN_COMPARISON_VIEWS"
DEFAULT_STATE = Path.home() / ".local/state/mars-titan/comparison-campaign-v1"
DEFAULT_VIEWS = Path.home() / ".local/state/mars-titan/campaign-a-v2-views-v31-20261013/views/US"
NATIVE_ENV = "MARS_TITAN_EPISODIC_NATIVE"
# Enlace episódico compilado que usan M1, M2 y M3. El paquete de datos lo lleva a esta ruta.
DEFAULT_NATIVE = (
    Path.home()
    / ".local/share/mars-titan/worktrees/native-develop-20261028/build/native"
    / "native-candidate-cuda/_episodic_native.cpython-312-x86_64-linux-gnu.so"
)
# Memoria residente medida por trabajo con 1.024 activos (#836), con margen.
JOB_MEMORY_GIB = 6.0
NO_BANK = "m0_no_bank"
MAX_ATTEMPTS = 3
# Disco libre por debajo del cual la cola deja de coger trabajos. Un disco lleno haría
# fallar a cada trabajo hasta agotar sus intentos, y así la cola se detiene sin perderlos.
MIN_FREE_GIB = 1.0
DONE, FAILED, PAUSED, RUNNING, PENDING, BLOCKED = (
    "completed",
    "failed",
    "paused",
    "running",
    "pending",
    "blocked",
)
_MODEL_FIELDS = {
    "fit": {"variant"},
    "reader": {"parent", "components", "search_case"},
    "rule": {"config"},
    "tabular": {"config", "fit_months", "radius", "min_assets"},
    "ridge": {"config"},
}
_OPTIONAL = {
    "fit": {"predictor_changes", "epochs"},
    "reader": {"epochs"},
    "rule": set(),
    "tabular": set(),
    "ridge": set(),
}
_MODEL_NAME = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}")
MAX_MODELS = 128
MAX_SEEDS = 16


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _checked_rule(rule, runner):
    """Regla de parada de un ejecutor. Solo el ajuste desde cero puede medir su estado inicial
    sin emitirlo: el lector conserva a su padre como candidato."""
    _require(
        isinstance(rule, dict)
        and rule.get("metric") == "session_mae"
        and type(rule.get("max_epochs")) is int
        and 1 <= rule["max_epochs"] <= 1000,
        "Cada regla de parada declara MAE por sesión y su máximo de épocas",
    )
    validate_selection(
        {key: value for key, value in rule.items() if key != "max_epochs"},
        epochs=rule["max_epochs"],
        modes=CHRONOLOGICAL_MODES,
        initial_reported=runner == "fit",
    )
    return rule


def job_rule(declaration, model):
    """Regla de parada de un modelo: la de su ejecutor con sus épocas propias si las declara."""
    rule = dict(declaration["stopping"][model["runner"]])
    if "epochs" in model:
        rule["max_epochs"] = model["epochs"]
    return _checked_rule(rule, model["runner"])


def _git(root, *arguments):
    """Salida de una orden de git en el repositorio, o None si no se puede ejecutar."""
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *arguments],
            capture_output=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout if result.returncode == 0 else None


def code_commit(root=ROOT):
    """Commit del repositorio que ejecuta la campaña, o None si no se puede leer."""
    output = _git(root, "rev-parse", "HEAD")
    commit = "" if output is None else output.decode("utf-8", "replace").strip()
    return commit if len(commit) == 40 else None


def code_changes(root=ROOT):
    """Huella de los cambios sin confirmar del código y la configuración, o None si no hay.

    Incluye las diferencias respecto a HEAD y los archivos nuevos de `CODE_PATHS`. Un error
    al consultar Git impide identificar el código y nunca se interpreta como un árbol limpio.
    """
    output = _git(root, "diff", "--binary", "--no-ext-diff", "HEAD", "--", *CODE_PATHS)
    added = _git(root, "ls-files", "--others", "--exclude-standard", "-z", "--", *CODE_PATHS)
    _require(output is not None and added is not None, "No se pudo leer la identidad del código")
    if not output and not added:
        return None
    digest = hashlib.sha256(output)
    for name in sorted(filter(None, added.split(b"\0"))):
        path = Path(root) / os.fsdecode(name)
        digest.update(b"\0" + name + b"\0")
        with path.open("rb") as stream:
            digest.update(hashlib.file_digest(stream, "sha256").digest())
    return digest.hexdigest()


def load_declaration(path=None, root=ROOT):
    """Leer y comprobar la declaración. Las rutas de sus recetas son relativas al repositorio."""
    path = Path(root) / DECLARATION if path is None else Path(path)
    document, digest = read_manifest(path, 1024**2)
    return checked_declaration(document, root=root), digest


def _deadline(value):
    """Fecha límite con zona explícita, convertida a UTC."""
    try:
        parsed = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        raise ValueError("La fecha límite necesita una fecha ISO con zona horaria") from None
    _require(parsed.utcoffset() is not None, "La fecha límite necesita una zona horaria")
    return parsed.astimezone(UTC)


def _checked_budget(budget):
    _require(
        isinstance(budget, dict)
        and set(budget) <= {"disk_gib", "memory_gib", "min_free_gib", "deadline_utc"}
        and isinstance(budget.get("disk_gib", {}), dict)
        and set(budget.get("disk_gib", {})) <= set(MACHINES)
        and isinstance(budget.get("memory_gib", {}), dict)
        and set(budget.get("memory_gib", {})) <= set(MACHINES)
        and all(
            type(value) in (int, float) and value >= 0
            for value in (
                *budget.get("disk_gib", {}).values(),
                *budget.get("memory_gib", {}).values(),
                budget.get("min_free_gib", 0),
            )
        ),
        "El presupuesto declara el disco estimado, la memoria de cada equipo y el margen libre",
    )
    if "deadline_utc" in budget:
        _deadline(budget["deadline_utc"])


def _checked_blocks(execution, origins, seeds):
    """Un bloque anual de una semilla conserva juntos todos sus modelos neuronales."""
    if "neural_blocks" not in execution:
        return
    blocks = execution["neural_blocks"]
    expected = {f"{origin['year']}/seed-{seed}" for origin in origins for seed in seeds}
    _require(
        isinstance(blocks, dict)
        and set(blocks) == expected
        and all(machine in MACHINES for machine in blocks.values()),
        "Los bloques neuronales deben asignar cada año y semilla a uno de los dos equipos",
    )


def checked_declaration(document, *, root=ROOT, check_recipe_files=True):
    """Comprobar la forma de la declaración, sus modelos y las dependencias entre ellos."""
    _require(
        isinstance(document, dict)
        and document.get("schema_version") == 1
        and document.get("kind") == KIND
        and document.get("final_test_access") is False
        and document.get("market") == "US",
        "La declaración no es la campaña de comparación de US con el test final cerrado",
    )
    origins = document.get("origins")
    _require(
        isinstance(origins, list)
        and origins
        and all(
            isinstance(origin, dict)
            and set(origin) == {"year", "fold", "role"}
            and type(origin["year"]) is int
            and 2005 <= origin["year"] <= 2023
            and isinstance(origin["fold"], str)
            and origin["role"] in ROLES
            for origin in origins
        )
        and len({origin["year"] for origin in origins}) == len(origins)
        and len({origin["fold"] for origin in origins}) == len(origins)
        and any(origin["role"] != HISTORY for origin in origins),
        "Los orígenes declaran año hasta 2023, ventana y papel, sin repetirse",
    )
    seeds = document.get("seeds")
    _require(
        isinstance(seeds, list)
        and seeds
        and len(seeds) <= MAX_SEEDS
        and all(type(seed) is int and 0 <= seed < 2**31 for seed in seeds)
        and len(set(seeds)) == len(seeds),
        "Las semillas deben ser enteros distintos",
    )
    cohort = document.get("cohort")
    _require(
        isinstance(cohort, dict) and set(cohort) == {"size"} and type(cohort["size"]) is int,
        "La cohorte declara su tamaño",
    )
    stopping = document.get("stopping")
    _require(
        isinstance(stopping, dict) and set(stopping) == set(GPU_RUNNERS),
        "La parada declara la regla de los ajustes y la de los lectores",
    )
    for runner, rule in stopping.items():
        _checked_rule(rule, runner)
    execution = document.get("execution")
    _require(
        isinstance(execution, dict)
        and {"pipeline", "workers"} <= set(execution) <= {"pipeline", "workers", "neural_blocks"}
        and isinstance(execution["workers"], dict)
        and set(execution["workers"]) == set(MACHINES)
        and all(type(count) is int and 1 <= count <= 8 for count in execution["workers"].values()),
        "La ejecución declara la tubería de lectura y los trabajadores de cada equipo",
    )
    PipelineOptions(**execution["pipeline"])
    _checked_blocks(execution, origins, seeds)
    _checked_budget(document.get("budget", {}))
    tiers, recipes, models = (document.get(key) for key in ("tiers", "recipes", "models"))
    _require(
        isinstance(tiers, list)
        and tiers
        and all(isinstance(tier, str) for tier in tiers)
        and len(set(tiers)) == len(tiers),
        "Los niveles de prioridad son nombres distintos y ordenados",
    )
    _require(
        isinstance(recipes, dict)
        and recipes
        and all(
            isinstance(path, str) and (not check_recipe_files or (Path(root) / path).is_file())
            for path in recipes.values()
        ),
        "Cada receta declarada debe existir en el repositorio",
    )
    _require(
        isinstance(models, dict)
        and 1 <= len(models) <= MAX_MODELS
        and all(isinstance(name, str) and _MODEL_NAME.fullmatch(name) for name in models),
        "La declaración necesita modelos con identificadores válidos y dentro del límite",
    )
    if "analysis" in document:
        from mars_titan.evaluation.comparison_protocol import checked

        checked(document["analysis"], models)
    for name, model in models.items():
        runner = model.get("runner") if isinstance(model, dict) else None
        _require(runner in RUNNERS, f"El modelo {name} no declara un ejecutor admitido")
        required = {"machine", "tier", "runner", *_MODEL_FIELDS[runner]}
        if runner in GPU_RUNNERS:
            required.add("recipe")
        _require(
            required <= set(model) <= required | _OPTIONAL[runner] | {"role", "extended_of"}
            and model["machine"] in MACHINES
            and model["tier"] in tiers
            and (runner not in GPU_RUNNERS or model["recipe"] in recipes),
            f"El modelo {name} no declara sus campos, su equipo, su nivel o su receta",
        )
        if runner in GPU_RUNNERS:
            job_rule(document, model)
    for name, model in models.items():
        if model["runner"] != "reader":
            continue
        parent = models.get(model["parent"])
        _require(
            parent is not None
            and parent["runner"] == "fit"
            and parent["variant"] == "mac_online"
            and parent["machine"] == model["machine"]
            and tiers.index(parent["tier"]) <= tiers.index(model["tier"]),
            f"El lector {name} necesita un padre mac_online de su equipo y de un nivel anterior",
        )
    return document


@dataclass(frozen=True)
class Job:
    """Un trabajo de la cola: un modelo en un origen y, si procede, una semilla."""

    model: str
    year: int
    fold: str
    seed: int | None
    runner: str
    tier: str
    needs: str | None = None
    history: bool = False

    @property
    def key(self):
        suffix = "all-seeds" if self.seed is None else f"seed-{self.seed}"
        return f"{self.model}/{self.year}/{suffix}"

    @property
    def stage(self):
        """Nivel del trabajo con su etapa, para la tabla de estado."""
        return f"{self.tier}/{HISTORY}" if self.history else self.tier


def plan(declaration, machine):
    """Trabajos del equipo en su orden de ejecución: etapa, nivel, semilla, origen y modelo.

    La primera etapa recorre los orígenes de desarrollo y de confirmación. La segunda repite
    los mismos niveles con los orígenes `history`, de modo que un plazo que se agote recorta
    primero los años añadidos y, dentro de ellos, los niveles menos prioritarios.
    """
    _require(machine in MACHINES, f"El equipo debe ser uno de {', '.join(MACHINES)}")
    tiers, models = declaration["tiers"], declaration["models"]
    jobs = []
    for history in (False, True):
        origins = [
            origin for origin in declaration["origins"] if (origin["role"] == HISTORY) is history
        ]
        for tier in tiers:
            names = [name for name, model in models.items() if model["tier"] == tier]
            # Dentro de un nivel, un padre va antes que sus lectores.
            names.sort(key=lambda name: models[name]["runner"] == "reader")
            for position, seed in enumerate(declaration["seeds"]):
                for origin in origins:
                    for name in names:
                        runner = models[name]["runner"]
                        if runner in ORIGIN_RUNNERS and position > 0:
                            continue
                        owner = models[name]["machine"]
                        blocks = declaration["execution"].get("neural_blocks")
                        if blocks is not None and runner in GPU_RUNNERS:
                            owner = blocks[f"{origin['year']}/seed-{seed}"]
                        if owner != machine:
                            continue
                        needs = None
                        if runner == "reader":
                            needs = f"{models[name]['parent']}/{origin['year']}/seed-{seed}"
                        jobs.append(
                            Job(
                                name,
                                origin["year"],
                                origin["fold"],
                                None if runner in ORIGIN_RUNNERS else seed,
                                runner,
                                tier,
                                needs,
                                history,
                            )
                        )
    return jobs


class Layout:
    """Rutas del estado de la campaña en un equipo. Nada de esto vive en el repositorio."""

    def __init__(self, state=None, views=None):
        self.state = Path(state or os.environ.get(STATE_ENV) or DEFAULT_STATE)
        self.views = Path(views or os.environ.get(VIEWS_ENV) or DEFAULT_VIEWS)
        self.native = Path(os.environ.get(NATIVE_ENV) or DEFAULT_NATIVE)
        _require(
            self.state.is_absolute() and self.views.is_absolute(),
            "El estado y las vistas de la campaña necesitan rutas absolutas",
        )
        safe_destination(self.state)

    @property
    def authorization(self):
        return self.state / "authorization.json"

    @property
    def indices(self):
        return self.state / "indices"

    @property
    def bundle(self):
        """Identidad del paquete de datos importado en este equipo, si llegó así."""
        return self.state / "data-bundle.json"

    def cohort(self, job_or_year):
        year = job_or_year.year if isinstance(job_or_year, Job) else job_or_year
        return self.state / "cohorts" / str(year)

    def source(self, fold):
        return self.views / fold / "manifest.json"

    def output(self, key):
        return self.state / "runs" / key

    def record(self, key):
        return self.state / "queue" / f"{key.replace('/', '--')}.json"

    def lock(self, key):
        return self.state / "queue" / f"{key.replace('/', '--')}.lock"

    def log(self, key):
        return self.state / "logs" / f"{key.replace('/', '--')}.log"


def authorize(layout, declaration_sha256, *, note, commit=None, changes=None):
    """Registrar el permiso expreso de quien lanza la campaña en este equipo.

    El permiso es un archivo propio de la campaña. La protección local del aprendizaje
    (`training-hold-2000.json`) no se toca: los trabajos leen este archivo en su lugar. Vale
    para esta declaración y para el código con el que se concede: su commit y, si los hay,
    sus cambios sin confirmar tal como están en ese momento.
    """
    _require(isinstance(note, str) and note.strip(), "El permiso necesita una nota de quien lo da")
    layout.state.mkdir(parents=True, exist_ok=True)
    document = dict(
        schema_version=1,
        kind="comparison_campaign_authorization",
        training_allowed=True,
        scope="comparison_campaign_v1_only",
        declaration_sha256=declaration_sha256,
        code_commit=commit,
        code_changes_sha256=changes,
        note=note.strip(),
        authorized_at_utc=datetime.now(UTC).isoformat(),
        final_test_access=False,
    )
    if layout.authorization.is_file():
        previous = json.loads(layout.authorization.read_text(encoding="utf-8"))
        identity = ("declaration_sha256", "code_commit", "code_changes_sha256")
        if any(previous.get(name) != document.get(name) for name in identity):
            started = any(
                json.loads(path.read_text(encoding="utf-8")).get("attempts")
                for path in (layout.state / "queue").glob("*.json")
            )
            _require(
                not started,
                "Este estado ya tiene trabajos de otra declaración o código. Usa otro estado "
                "con --state para conservarlos sin mezclar ejecuciones",
            )
    atomic_json(layout.authorization, document)
    return document


def authorized(layout, declaration_sha256, *, commit=None, changes=None):
    """El permiso existe, permite el aprendizaje y corresponde a esta declaración y código.

    Un cambio de la declaración, del commit o de los cambios sin confirmar pide un permiso
    nuevo, para que la campaña no continúe con otro código sin que quien la lanza lo decida.
    """
    if not layout.authorization.is_file():
        return False
    document = json.loads(layout.authorization.read_text(encoding="utf-8"))
    return (
        document.get("training_allowed") is True
        and document.get("declaration_sha256") == declaration_sha256
        and document.get("code_commit") == commit
        and document.get("code_changes_sha256") == changes
    )


def prepare(declaration, layout):
    """Escribir la cohorte de cada origen. Solo lee y escribe manifiestos."""
    receipts = {}
    for origin in declaration["origins"]:
        source = layout.source(origin["fold"])
        _require(source.is_file(), f"Falta la vista del origen {origin['year']} en {source}")
        _, receipt = write_cohort(
            source, layout.cohort(origin["year"]), size=declaration["cohort"]["size"]
        )
        selection = receipt["selection"]
        # La carpeta, la ventana y el año declarado deben ser el mismo origen.
        _require(
            selection["fold"] == origin["fold"]
            and selection["evaluation"][0] == f"{origin['year']}-01-01",
            f"La vista de {origin['year']} no es la ventana {origin['fold']} de ese año",
        )
        receipts[str(origin["year"])] = selection
    return receipts


def read_record(layout, key):
    path = layout.record(key)
    if not path.is_file():
        return dict(status=PENDING, attempts=[])
    return json.loads(path.read_text(encoding="utf-8"))


def _write_record(layout, key, record):
    path = layout.record(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(path, record)


def job_state(layout, job, records=None):
    """Estado de un trabajo: su registro, o bloqueado si su padre no ha terminado."""
    record = read_record(layout, job.key) if records is None else records[job.key]
    status = record["status"]
    if status in (DONE, FAILED):
        return status
    if job.needs is not None:
        parent = (read_record(layout, job.needs) if records is None else records[job.needs])[
            "status"
        ]
        if parent != DONE:
            return BLOCKED
    return PENDING if status == RUNNING else status


def execute_job(declaration, layout, job, *, device="cuda:0", root=ROOT, preflight=False):
    """Ejecutar un trabajo en este proceso y devolver el informe de su ventana.

    Importa PyTorch y los ejecutores aquí, de modo que planificar y consultar el estado no
    los necesitan. La protección del aprendizaje la comprueba cada ejecutor. Con `preflight`
    la parada se pide antes de empezar: el trabajo abre sus datos, construye su modelo y se
    detiene en su primera barrera, de donde continúa después la cola.
    """
    from .checkpoints import StopRequest

    model = declaration["models"][job.model]
    view = layout.cohort(job) / "manifest.json"
    _require(view.is_file(), f"Falta la cohorte de {job.year}. Ejecuta antes la preparación")
    output = layout.output(job.key)
    if job.runner in ORIGIN_RUNNERS:
        with StopRequest() as stop:
            if preflight:
                stop.request_stop()
            if job.runner == "ridge":
                from .comparison_ridge import run_ridge_window

                return run_ridge_window(view, output=output, config=model["config"], stop=stop)
            from . import common_control_windows as controls
            from .titans_walk_forward import view_protocol

            protocol = view_protocol(view)
            if job.runner == "rule":
                return controls.run_rule_window(
                    view, protocol, job.fold, output=output, config=model["config"], stop=stop
                )
            return controls.run_tabular_window(
                view,
                protocol,
                job.fold,
                seeds=list(declaration["seeds"]),
                fit_months=model["fit_months"],
                max_new_fits=1,
                output=output,
                config=model["config"],
                radius=model["radius"],
                min_assets=model["min_assets"],
                stop=stop,
            )
    from .titans_walk_forward import run_titans_window, unfused_attention, view_protocol

    recipe = Path(root) / declaration["recipes"][model["recipe"]]
    rule = job_rule(declaration, model)
    with StopRequest() as stop, unfused_attention():
        if preflight:
            stop.request_stop()
        if job.runner == "fit":
            return run_titans_window(
                view,
                view_protocol(view),
                job.fold,
                recipe,
                variant=model["variant"],
                seed=job.seed,
                output=output,
                device=device,
                stop=stop,
                stopping=rule,
                predictor_changes=model.get("predictor_changes"),
            )
        from .mars_titan_walk_forward import run_mars_titan_window

        return run_mars_titan_window(
            view,
            layout.output(job.needs),
            recipe,
            components=dict(model["components"]),
            seed=job.seed,
            output=output,
            search_case=model["search_case"],
            device=device,
            stop=stop,
            stopping=rule,
        )


def run_job(declaration, layout, job, *, device="cuda:0", root=ROOT, preflight=False):
    """Ejecutar un trabajo y dejar su resultado en el registro de la cola.

    Devuelve el estado final. Un informe pausado conserva el trabajo como pausado, para que
    la siguiente orden lo continúe desde su último checkpoint.
    """
    from .comparison_telemetry import AttemptMetrics

    record = read_record(layout, job.key)
    attempt = dict(started_at_utc=datetime.now(UTC).isoformat(), pid=os.getpid())
    if preflight:
        attempt["preflight"] = True
    record.update(status=RUNNING, model=job.model, year=job.year, seed=job.seed, tier=job.tier)
    record.setdefault("attempts", []).append(attempt)
    _write_record(layout, job.key, record)
    measurement = AttemptMetrics()
    started = measurement.started
    try:
        report = execute_job(
            declaration, layout, job, device=device, root=root, preflight=preflight
        )
        status = DONE if report["status"] == "completed" else PAUSED
        attempt.update(status=report["status"])
        if status == DONE:
            record["result"] = dict(_result(report), **_scores(layout.output(job.key), job, report))
    except BaseException as error:
        status = PAUSED if isinstance(error, KeyboardInterrupt) else FAILED
        attempt.update(
            status="interrupted" if status == PAUSED else "failed",
            error_type=type(error).__name__,
            error=str(error)[:2000],
            traceback=traceback.format_exc()[-6000:],
        )
    attempt.update(seconds=time.perf_counter() - started, finished_at_utc=_now())
    try:
        attempt["resources"] = measurement.finish(device if job.runner in CUDA_RUNNERS else "cpu")
    except (OSError, RuntimeError, ValueError) as error:
        attempt["resources_error"] = f"{type(error).__name__}: {str(error)[:300]}"
    failures = sum(item.get("status") == "failed" for item in record["attempts"])
    if status == FAILED and failures < MAX_ATTEMPTS:
        # Vuelve a la cola. Los ejecutores continúan desde su último checkpoint coherente.
        status = PENDING
    record["status"] = status
    _write_record(layout, job.key, record)
    return status


def _now():
    return datetime.now(UTC).isoformat()


def _result(report):
    """Resumen del informe de una ventana para la cola y la tabla de estado."""
    predictions = report.get("predictions", {})
    result = dict(run_id=report.get("run_id"), kind=report.get("kind"))
    fit = report.get("fit")
    if isinstance(fit, dict):
        result.update(
            best_epoch=fit.get("best_epoch"),
            best_score=fit.get("best_score"),
            global_step=fit.get("global_step"),
        )
    result["session_mae"] = {
        name: record["metrics"].get("session_mae")
        for name, record in predictions.items()
        if isinstance(record, dict) and isinstance(record.get("metrics"), dict)
    }
    return result


def _scores(output, job, report):
    """Medidas de seguimiento de un trabajo terminado, leídas de sus tablas de predicción.

    Son un resumen para la tabla de resultados. Si no se pueden calcular, el trabajo sigue
    terminado y el motivo queda anotado en su registro.
    """
    try:
        from .comparison_scores import job_scores

        return dict(scores=job_scores(output, report, magnitude=job.runner in MAGNITUDE_RUNNERS))
    except Exception as error:  # noqa: BLE001 - el resumen no decide el estado del trabajo.
        return dict(scores_error=f"{type(error).__name__}: {str(error)[:300]}")


def _last_error(record):
    """Primera línea del último error anotado en un registro, o None."""
    errors = [item["error"] for item in record.get("attempts", []) if item.get("error")]
    return errors[-1].splitlines()[0][:200] if errors else None


def summary(declaration, layout, machine):
    """Recuento de trabajos por estado y nivel, con los segundos gastados y los pendientes.

    Incluye los trabajos fallidos con su último error y los que tienen un intento abierto,
    para ver de un vistazo qué necesita atención y qué está en marcha.
    """
    jobs = plan(declaration, machine)
    records = {job.key: read_record(layout, job.key) for job in jobs}
    counts, spent, durations = {}, 0.0, {}
    for job in jobs:
        state = job_state(layout, job, records)
        counts.setdefault(job.stage, {}).setdefault(state, 0)
        counts[job.stage][state] += 1
        seconds = sum(item.get("seconds", 0.0) for item in records[job.key]["attempts"])
        spent += seconds
        if state == DONE:
            durations.setdefault(job.model, []).append(seconds)
    mean = {name: sum(values) / len(values) for name, values in durations.items()}
    known = [value for values in durations.values() for value in values]
    fallback = sum(known) / len(known) if known else None
    remaining = None
    if fallback is not None:
        remaining = sum(
            mean.get(job.model, fallback)
            for job in jobs
            if job_state(layout, job, records) not in (DONE, FAILED)
        )
    return dict(
        machine=machine,
        jobs=len(jobs),
        by_tier=counts,
        seconds_spent=spent,
        mean_seconds_by_model=mean,
        estimated_seconds_remaining=remaining,
        failed={
            job.key: _last_error(records[job.key])
            for job in jobs
            if records[job.key]["status"] == FAILED
        },
        open_attempts={
            job.key: records[job.key]["attempts"][-1].get("started_at_utc")
            for job in jobs
            if records[job.key]["status"] == RUNNING and records[job.key]["attempts"]
        },
    )


RESULT_FIELDS = ("raw", "zero", "calibrated", "pearson_ic", "rank_ic")


def results(declaration, layout, machine):
    """Medidas por modelo y etapa con los trabajos terminados de un equipo.

    Cada fila promedia, con igual peso por origen y semilla, las medidas que cada trabajo
    dejó en su registro al terminar: el MAE por sesión del año evaluado, el de la predicción
    cero con las mismas filas, el MAE tras escalar la predicción con el factor elegido en la
    calibración y el IC por sesión. Los controles que predicen un orden no tienen MAE sin
    calibrar. Es un resumen para seguir la campaña: el análisis con incertidumbre se hace al
    cerrarla.
    """
    jobs = plan(declaration, machine)
    records = {job.key: read_record(layout, job.key) for job in jobs}
    groups = {}
    for job in jobs:
        groups.setdefault((job.history, job.model), []).append(job)
    rows = []
    for (history, model), members in groups.items():
        done = [job for job in members if records[job.key]["status"] == DONE]
        scored = [
            records[job.key]["result"]["scores"]
            for job in done
            if isinstance(records[job.key].get("result", {}).get("scores"), dict)
        ]
        row = dict(
            model=model, history=history, done=len(done), jobs=len(members), scored=len(scored)
        )
        for name in RESULT_FIELDS:
            values = [item[name] for item in scored]
            known = bool(values) and all(value is not None for value in values)
            row[name] = sum(values) / len(values) if known else None
        for name, source in (("difference_percent", "raw"), ("calibrated_percent", "calibrated")):
            row[name] = (
                None
                if row[source] is None or not row["zero"]
                else 100.0 * (row[source] - row["zero"]) / row["zero"]
            )
        rows.append(row)
    return rows


def _print_results(machine, rows):
    def number(value, width, pattern):
        return f"{'-':>{width}}" if value is None else f"{format(value, pattern):>{width}}"

    columns = (
        ("raw", 11, ".6f"),
        ("zero", 11, ".6f"),
        ("difference_percent", 8, "+.3f"),
        ("calibrated_percent", 9, "+.3f"),
        ("pearson_ic", 8, "+.4f"),
        ("rank_ic", 8, "+.4f"),
    )
    for history in (False, True):
        stage = [row for row in rows if row["history"] is history]
        if not stage:
            continue
        print(f"equipo {machine}, {'años añadidos' if history else 'primera etapa'}")
        print(
            f"  {'modelo':32s} {'hechos':>7s} {'evaluación':>11s} {'cero':>11s} {'dif. %':>8s} "
            f"{'calibr. %':>9s} {'IC':>8s} {'IC rang.':>8s}"
        )
        for row in stage:
            print(
                f"  {row['model']:32s} {row['done']:3d}/{row['jobs']:<3d} "
                + " ".join(number(row[name], width, pattern) for name, width, pattern in columns)
            )
    print(
        "evaluación y cero: MAE por sesión del año evaluado. dif. %: evaluación frente a cero. "
        "calibr. %: lo mismo tras escalar la predicción con el factor elegido en la calibración. "
        "IC: correlación por sesión entre predicción y residuo."
    )


def _claim(layout, jobs, taken):
    """Siguiente trabajo ejecutable que nadie tiene, con su cerrojo ya cogido."""
    for job in jobs:
        if job.key in taken or job_state(layout, job) not in (PENDING, PAUSED):
            continue
        path = layout.lock(job.key)
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = path.open("a")
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            handle.close()
            continue
        # Otro trabajador pudo terminarlo entre la lectura y el cerrojo.
        if job_state(layout, job) in (PENDING, PAUSED):
            return job, handle
        handle.close()
    return None, None


def _environment(layout, declaration):
    environment = dict(os.environ)
    # La tubería de lectura solo cambia el caudal: el conjunto y el orden de las filas no.
    environment.update(PipelineOptions(**declaration["execution"]["pipeline"]).environment())
    environment[HOLD_ENV] = str(layout.authorization)
    environment[SHARED_INDICES_ENV] = str(layout.indices)
    environment[STATE_ENV], environment[VIEWS_ENV] = str(layout.state), str(layout.views)
    environment[NATIVE_ENV] = str(layout.native)
    # cuBLAS determinista, como exige el recorrido cronológico en cuda:0.
    environment.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    environment.setdefault("PYTHONUNBUFFERED", "1")
    return environment


class Queue:
    """Cola de un equipo: reparte los trabajos entre trabajadores y propaga la parada."""

    def __init__(
        self, declaration, layout, machine, *, workers=1, tiers=None, declaration_path=None
    ):
        _require(type(workers) is int and 1 <= workers <= 8, "Entre 1 y 8 trabajadores")
        jobs = plan(declaration, machine)
        if tiers is not None:
            _require(set(tiers) <= set(declaration["tiers"]), "Nivel de prioridad desconocido")
            jobs = [job for job in jobs if job.tier in tiers]
        self.declaration, self.layout, self.machine = declaration, layout, machine
        self.jobs, self.workers, self.declaration_path = jobs, workers, declaration_path
        self.stop = threading.Event()
        self.guard = threading.Lock()
        self.taken, self.children = set(), {}
        self.min_free_gib = declaration.get("budget", {}).get("min_free_gib", MIN_FREE_GIB)
        deadline = declaration.get("budget", {}).get("deadline_utc")
        self.deadline = None if deadline is None else _deadline(deadline)
        # Espera entre consultas cuando solo quedan lectores cuyo padre se está ajustando.
        self.poll_seconds = 15.0

    def request_stop(self, *_):
        """Pedir a cada trabajo que pare en su siguiente barrera confirmada."""
        self.stop.set()
        with self.guard:
            for child in self.children.values():
                if child.poll() is None:
                    child.send_signal(signal.SIGTERM)

    def _command(self, job, *, preflight=False):
        command = [sys.executable, "-m", "mars_titan.training.comparison_campaign", "job"]
        if self.declaration_path is not None:
            command += ["--declaration", str(self.declaration_path)]
        command += ["--machine", self.machine, "--key", job.key]
        return [*command, "--preflight"] if preflight else command

    def _launch(self, job, *, preflight=False):
        """Ejecutar un trabajo en su proceso y devolver su código de salida."""
        previous_attempts = len(read_record(self.layout, job.key)["attempts"])
        started_at, started = _now(), time.perf_counter()
        log = self.layout.log(job.key)
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a") as stream:
            mark = " (comprobación inicial)" if preflight else ""
            stream.write(f"\n=== {_now()} {job.key}{mark}\n")
            stream.flush()
            child = subprocess.Popen(
                self._command(job, preflight=preflight),
                stdout=stream,
                stderr=subprocess.STDOUT,
                env=_environment(self.layout, self.declaration),
                cwd=str(ROOT),
            )
            with self.guard:
                self.children[job.key] = child
                # La parada puede llegar después de Popen y antes de registrar al hijo.
                if self.stop.is_set() and child.poll() is None:
                    child.send_signal(signal.SIGTERM)
            try:
                code = child.wait()
            finally:
                with self.guard:
                    self.children.pop(job.key, None)
        if code not in (0, 3):
            self._abnormal_exit(
                job,
                code,
                previous_attempts=previous_attempts,
                started_at=started_at,
                seconds=time.perf_counter() - started,
            )
        record = read_record(self.layout, job.key)
        if len(record["attempts"]) > previous_attempts:
            # Incluye arranque del intérprete, importaciones y cierre del proceso hijo.
            record["attempts"][-1].update(
                process_wall_seconds=time.perf_counter() - started,
                process_exit_code=code,
            )
            _write_record(self.layout, job.key, record)
        return code

    def preflight(self):
        """Arrancar una vez cada modelo hasta su primera barrera, antes de vaciar la cola.

        Un error de configuración de un modelo de un nivel tardío aparece así en los primeros
        minutos y no días después. Los lectores necesitan a su padre ajustado y no se
        comprueban aquí. Devuelve los trabajos que no llegaron a su barrera.
        """
        first = {}
        for job in self.jobs:
            if job.runner != "reader":
                first.setdefault(job.model, job)
        failed = []
        for job in first.values():
            if self.stop.is_set() or self._deadline_reached() or self._disk_exhausted():
                break
            # Un trabajo ya arrancado en otra ocasión no necesita repetir la comprobación.
            if read_record(self.layout, job.key)["attempts"]:
                continue
            claimed, handle = _claim(self.layout, [job], set())
            if claimed is None:
                continue
            try:
                code = self._launch(job, preflight=True)
            finally:
                handle.close()
            state = "bien" if code in (0, 3) else "FALLA"
            print(f"comprobación inicial {state}: {job.key}", flush=True)
            if code not in (0, 3):
                failed.append(job.key)
        return failed

    def _deadline_reached(self):
        """Pedir una parada recuperable al alcanzar la fecha declarada."""
        if self.deadline is None or datetime.fromisoformat(_now()) < self.deadline:
            return False
        if not self.stop.is_set():
            print(
                "Se ha alcanzado la fecha límite. Los trabajos guardan su próxima barrera",
                flush=True,
            )
            self.request_stop()
        return True

    def _disk_exhausted(self):
        """Detener la cola si el disco del estado se queda sin el margen mínimo."""
        free = _free_gib(self.layout.state)
        if free >= self.min_free_gib:
            return False
        if not self.stop.is_set():
            print(
                f"Quedan {free:.1f} GiB libres y la cola necesita {self.min_free_gib:.1f}. "
                "Se detiene sin perder trabajos: libera disco y repite la orden run",
                flush=True,
            )
        self.request_stop()
        return True

    def _work(self):
        while not self.stop.is_set():
            if self._deadline_reached() or self._disk_exhausted():
                return
            with self.guard:
                job, handle = _claim(self.layout, self.jobs, self.taken)
                if job is not None:
                    self.taken.add(job.key)
            if job is None:
                with self.guard:
                    busy = bool(self.taken)
                if not busy or self._finished():
                    return
                # Queda algún trabajo bloqueado por un padre que otro trabajador ajusta.
                self.stop.wait(self.poll_seconds)
                continue
            try:
                self._launch(job)
            finally:
                with self.guard:
                    self.taken.discard(job.key)
                handle.close()

    def _abnormal_exit(self, job, code, *, previous_attempts=None, started_at=None, seconds=None):
        """Un proceso que muere sin escribir su estado (OOM, señal) cuenta como intento fallido."""
        record = read_record(self.layout, job.key)
        if record["status"] in (DONE, FAILED):
            return
        attempts = record.setdefault("attempts", [])
        if previous_attempts is not None and len(attempts) == previous_attempts:
            # También se acotan fallos anteriores a run_job, como un permiso invalidado.
            attempts.append(dict(started_at_utc=started_at))
        elif record["status"] != RUNNING:
            return
        # Una parada pedida que llega antes de la primera barrera no es un fallo del trabajo.
        outcome = "interrupted" if self.stop.is_set() else "failed"
        if attempts and "status" not in attempts[-1]:
            attempts[-1].update(
                status=outcome,
                error=f"El proceso terminó con el código {code}",
                finished_at_utc=_now(),
            )
            if seconds is not None:
                attempts[-1]["seconds"] = seconds
        failures = sum(item.get("status") == "failed" for item in attempts)
        record["status"] = FAILED if failures >= MAX_ATTEMPTS else PENDING
        _write_record(self.layout, job.key, record)

    def _finished(self):
        return all(job_state(self.layout, job) in (DONE, FAILED, BLOCKED) for job in self.jobs)

    def run(self, *, preflight=True):
        """Ejecutar hasta vaciar la cola o recibir una parada. Devuelve el resumen final.

        Antes arranca cada modelo una vez hasta su primera barrera, salvo que se pida no
        hacerlo. Un modelo que falla ahí queda anotado y la cola sigue con los demás.
        """
        if preflight:
            failed = self.preflight()
            if failed:
                print(f"{len(failed)} modelos no arrancan. Revisa sus registros", flush=True)
        threads = [threading.Thread(target=self._work, daemon=True) for _ in range(self.workers)]
        for thread in threads:
            thread.start()
        while any(thread.is_alive() for thread in threads):
            self._deadline_reached()
            for thread in threads:
                thread.join(timeout=1.0)
        return summary(self.declaration, self.layout, self.machine)


def retry_failed(declaration, layout, machine):
    """Devolver a la cola los trabajos fallidos, con sus intentos anteriores conservados."""
    count = 0
    for job in plan(declaration, machine):
        record = read_record(layout, job.key)
        if record["status"] == FAILED:
            for attempt in record["attempts"]:
                if attempt.get("status") == "failed":
                    attempt["status"] = "failed_before_retry"
            record["status"] = PENDING
            _write_record(layout, job.key, record)
            count += 1
    return count


def _data_checks(declaration, layout):
    """Vistas, cohortes y archivos de cada cohorte. Solo consulta que existen."""
    checks = []
    for origin in declaration["origins"]:
        year, source = origin["year"], layout.source(origin["fold"])
        checks.append((f"vista {year}", source.is_file(), str(source)))
        path = layout.cohort(year) / "manifest.json"
        if not path.is_file():
            checks.append((f"cohorte {year}", False, "Falta. Ejecuta antes la orden prepare"))
            continue
        cohort = json.loads(path.read_text(encoding="utf-8"))
        roots = {name: Path(root) for name, root in cohort["roots"].items()}
        present = sum(
            (
                roots["prepared" if kind == "prices" else kind]
                / asset["market"]
                / asset["symbol"]
                / f"{kind}.parquet"
            ).is_file()
            for asset in cohort["assets"]
            for kind in ("prices", "samples", "labels")
        )
        total = 3 * len(cohort["assets"])
        checks.append(
            (f"datos de la cohorte {year}", present == total, f"{present} de {total} archivos")
        )
    return checks


def cohort_identity(declaration, layout):
    """Huella de los activos de cada cohorte preparada, por año de origen."""
    result = {}
    for origin in declaration["origins"]:
        receipt = layout.cohort(origin["year"]) / "cohort.json"
        if receipt.is_file():
            selection = json.loads(receipt.read_text(encoding="utf-8"))["selection"]
            result[str(origin["year"])] = selection["assets_sha256"]
    return result


def _bundle_check(declaration, layout, digest):
    """Con datos importados, sus cohortes deben ser las del equipo que exportó el paquete.

    Las cohortes deciden qué archivos viajan, así que deben coincidir. La declaración y el
    commit pueden cambiar después de exportar sin afectar a los datos. En ese caso se avisa,
    y la línea del código permite comparar los dos equipos antes de lanzar.
    """
    if not layout.bundle.is_file():
        return []
    identity = json.loads(layout.bundle.read_text(encoding="utf-8"))
    same = identity.get("cohorts") == cohort_identity(declaration, layout)
    detail = "coinciden" if same else "no coinciden con las del paquete importado"
    checks = [("mismas cohortes que el equipo de origen", same, detail)]
    exported = identity.get("declaration_sha256"), identity.get("code_commit")
    if exported != (digest, code_commit()):
        checks.append(
            (
                "paquete exportado con otra declaración u otro commit",
                None,
                f"declaración {str(exported[0])[:12]}, commit {str(exported[1])[:12]}. "
                "Los datos sirven si las cohortes coinciden",
            )
        )
    return checks


def _code_check(digest):
    """Identidad del código y de la declaración, para comparar los dos equipos."""
    commit = code_commit()
    detail = f"commit {commit[:12] if commit else 'desconocido'}"
    if digest is not None:
        detail += f", declaración {digest[:12]}"
    if commit is None:
        return "código y declaración", False, detail
    try:
        changes = code_changes()
    except (OSError, ValueError) as error:
        return "código y declaración", False, f"{detail}, {error}"
    if changes is None:
        return "código y declaración", True, f"{detail}, sin cambios locales"
    return (
        "código y declaración",
        None,
        f"{detail}, con cambios sin confirmar en {', '.join(CODE_PATHS)} que el otro equipo "
        "no tendrá",
    )


def _gpu_check():
    try:
        import torch

        if not torch.cuda.is_available():
            return False, "torch.cuda.is_available() es falso. No se cambia a CPU"
        free = torch.cuda.mem_get_info(0)[0] / 2**30
        name = torch.cuda.get_device_name(0)
        return True, f"{name}, {free:.1f} GiB libres, torch {torch.__version__}"
    except Exception as error:  # noqa: BLE001 - el diagnóstico informa de cualquier causa.
        return False, f"{type(error).__name__}: {str(error)[:300]}"


def _native_check(layout):
    try:
        from mars_titan.memory.native_backend import load_native

        load_native(layout.native)
        return True, str(layout.native)
    except Exception as error:  # noqa: BLE001 - el diagnóstico informa de cualquier causa.
        return False, f"{type(error).__name__}: {str(error)[:300]}"


def _free_gib(path):
    """GiB libres en el disco de `path` o de su primera carpeta existente."""
    import shutil

    path = Path(path)
    while not path.exists():
        path = path.parent
    return shutil.disk_usage(path).free / 2**30


def _available_memory_gib():
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 2**20
    return None


def doctor(declaration, layout, machine, *, digest=None, workers=None):
    """Comprobaciones previas de un equipo: datos, cohortes, GPU, enlace nativo, memoria y disco.

    Devuelve una lista de `(nombre, estado, detalle)`. El estado es verdadero, falso o `None`
    para un aviso que no impide lanzar. No ajusta nada ni abre el test final.
    """
    checks = _data_checks(declaration, layout) + _bundle_check(declaration, layout, digest)
    checks.append(_code_check(digest))
    jobs = plan(declaration, machine)
    if any(job.runner in CUDA_RUNNERS for job in jobs):
        checks.append(("cuda:0", *_gpu_check()))
    if any(
        declaration["models"][job.model]["components"].get("episodic_bank") != NO_BANK
        for job in jobs
        if job.runner == "reader"
    ):
        checks.append(("enlace episódico nativo (M1, M2 y M3)", *_native_check(layout)))
    workers = workers or declaration["execution"]["workers"][machine]
    per_job = declaration.get("budget", {}).get("memory_gib", {}).get(machine, JOB_MEMORY_GIB)
    available, needed = _available_memory_gib(), workers * per_job
    if available is not None:
        checks.append(
            (
                "memoria disponible",
                available >= needed,
                f"{available:.1f} GiB, {needed:.0f} GiB para {workers} trabajos a la vez",
            )
        )
    free = _free_gib(layout.state)
    needed = declaration.get("budget", {}).get("disk_gib", {}).get(machine)
    checks.append(
        (
            "disco libre",
            needed is None or free >= needed,
            f"{free:.1f} GiB libres" + ("" if needed is None else f", estimados {needed} GiB"),
        )
    )
    if digest is not None:
        granted = authorized(layout, digest, commit=code_commit(), changes=code_changes())
        detail = str(layout.authorization) if granted else "Pendiente. Se registra con authorize"
        checks.append(("permiso expreso de esta declaración", True if granted else None, detail))
    return checks


def _print_summary(result):
    print(f"equipo {result['machine']}: {result['jobs']} trabajos")
    for tier, counts in result["by_tier"].items():
        line = ", ".join(f"{count} {state}" for state, count in sorted(counts.items()))
        print(f"  nivel {tier}: {line}")
    print(f"  horas gastadas: {result['seconds_spent'] / 3600:.1f}")
    if result["estimated_seconds_remaining"] is not None:
        print(f"  horas estimadas restantes: {result['estimated_seconds_remaining'] / 3600:.1f}")
    for key, started in result["open_attempts"].items():
        print(f"  con un intento abierto: {key}, desde {started}")
    for key, error in result["failed"].items():
        print(f"  fallido: {key}: {error or 'sin mensaje'}")


_MACHINE_COMMANDS = ("plan", "doctor", "run", "status", "results", "retry", "job", "export-results")


def _export(declaration, digest, layout, args):
    from . import comparison_data

    _require(args.destination is not None, "export necesita --destination")
    args.destination.mkdir(parents=True, exist_ok=True)
    needs_native = any(
        model["runner"] == "reader" and model["components"].get("episodic_bank") != NO_BANK
        for model in declaration["models"].values()
    )
    files = comparison_data.campaign_files(
        declaration, layout, native=layout.native if needs_native else None
    )
    total = sum(path.stat().st_size for path in files) / 2**30
    print(f"{len(files)} archivos, {total:.1f} GiB. Copiando a {args.destination}")
    manifest = comparison_data.export_bundle(
        files,
        args.destination,
        part_bytes=None if args.part_gib is None else int(args.part_gib * 2**30),
        identity=dict(
            declaration_sha256=digest,
            cohorts=cohort_identity(declaration, layout),
            code_commit=code_commit(),
        ),
    )
    print(f"paquete escrito en {len(manifest['parts'])} partes con su lista de huellas")
    return 0


def _import(layout, args):
    from . import comparison_data

    _require(args.source is not None, "import necesita --source")
    result = comparison_data.import_bundle(args.source)
    layout.state.mkdir(parents=True, exist_ok=True)
    atomic_json(layout.bundle, result["identity"])
    print(
        f"{result['restored']} archivos restaurados y {result['kept']} ya presentes, "
        f"{result['total_bytes'] / 2**30:.1f} GiB con sus huellas comprobadas"
    )
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "command",
        choices=(
            "plan",
            "prepare",
            "export",
            "import",
            "doctor",
            "authorize",
            "run",
            "status",
            "results",
            "export-results",
            "retry",
            "job",
        ),
    )
    parser.add_argument("--machine", choices=MACHINES)
    parser.add_argument("--declaration", type=Path)
    parser.add_argument("--state", type=Path, help=f"Estado de la campaña (o {STATE_ENV})")
    parser.add_argument("--views", type=Path, help=f"Vistas por ventana (o {VIEWS_ENV})")
    parser.add_argument("--workers", type=int, help="Por defecto, los de la declaración")
    parser.add_argument("--tier", action="append", help="Limitar a estos niveles de prioridad")
    parser.add_argument("--key", help="Trabajo que ejecuta la orden interna job")
    parser.add_argument("--note", help="Nota del permiso expreso, obligatoria en authorize")
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    parser.add_argument(
        "--destination", type=Path, help="Carpeta para export o ZIP nuevo para export-results"
    )
    parser.add_argument("--source", type=Path, help="Carpeta del paquete que lee import")
    parser.add_argument("--part-gib", type=float, help="Tamaño máximo de cada parte de export")
    parser.add_argument("--preflight", action="store_true", help="Uso interno de la orden job")
    parser.add_argument(
        "--no-preflight", action="store_true", help="No arrancar antes cada modelo una vez"
    )
    args = parser.parse_args(argv)
    _require(
        args.machine is not None or args.command not in _MACHINE_COMMANDS,
        f"La orden {args.command} necesita --machine",
    )
    declaration, digest = load_declaration(args.declaration)
    layout = Layout(args.state, args.views)
    if args.command == "plan":
        jobs = plan(declaration, args.machine)
        for job in jobs:
            print(f"{job.stage:20s} {job.key}" + (f"  <- {job.needs}" if job.needs else ""))
        print(f"{len(jobs)} trabajos en {args.machine}")
        return 0
    if args.command == "prepare":
        receipts = prepare(declaration, layout)
        for year, selection in receipts.items():
            print(f"{year}: {selection['size']} activos de {selection['eligible_count']} elegibles")
        return 0
    if args.command == "export":
        return _export(declaration, digest, layout, args)
    if args.command == "import":
        return _import(layout, args)
    if args.command == "export-results":
        from mars_titan.evaluation.comparison_bundle import export_results

        _require(args.destination is not None, "export-results necesita --destination")
        _require(args.part_gib is None, "export-results crea un único ZIP, sin --part-gib")
        exported = export_results(declaration, digest, layout, args.machine, args.destination)
        print(
            f"Resultados guardados en {args.destination}: {exported['completed_jobs']} "
            f"de {len(exported['jobs'])} trabajos terminados, sin modelos"
        )
        print(f"  {args.destination.stat().st_size / 2**30:.2f} GiB de archivo ZIP")
        return 0
    if args.command == "doctor":
        checks = doctor(declaration, layout, args.machine, digest=digest, workers=args.workers)
        labels = {True: "bien ", False: "FALTA", None: "aviso"}
        for name, passed, detail in checks:
            print(f"{labels[passed]} {name}: {detail}")
        return 2 if any(passed is False for _, passed, _ in checks) else 0
    commit, changes = code_commit(), code_changes()
    if args.command == "authorize":
        document = authorize(layout, digest, note=args.note or "", commit=commit, changes=changes)
        print(f"permiso registrado en {layout.authorization} ({document['authorized_at_utc']})")
        print(f"  {_code_check(digest)[2]}")
        return 0
    if args.command == "status":
        _print_summary(summary(declaration, layout, args.machine))
        return 0
    if args.command == "results":
        _print_results(args.machine, results(declaration, layout, args.machine))
        return 0
    if args.command == "retry":
        print(f"{retry_failed(declaration, layout, args.machine)} trabajos devueltos a la cola")
        return 0
    _require(
        authorized(layout, digest, commit=commit, changes=changes),
        f"Falta el permiso expreso de esta declaración y de este código en "
        f"{layout.authorization}. Se registra con la orden authorize",
    )
    if args.command == "job":
        jobs = {job.key: job for job in plan(declaration, args.machine)}
        _require(args.key in jobs, "El trabajo no pertenece a la cola de este equipo")
        status = run_job(
            declaration, layout, jobs[args.key], device=args.device, preflight=args.preflight
        )
        return {DONE: 0, PAUSED: 3, PENDING: 4}.get(status, 1)
    queue = Queue(
        declaration,
        layout,
        args.machine,
        workers=args.workers or declaration["execution"]["workers"][args.machine],
        tiers=args.tier,
        declaration_path=args.declaration,
    )
    for name in (signal.SIGINT, signal.SIGTERM):
        signal.signal(name, queue.request_stop)
    result = queue.run(preflight=not args.no_preflight)
    _print_summary(result)
    pending = sum(job_state(layout, job) != DONE for job in queue.jobs)
    return 0 if pending == 0 else 3


if __name__ == "__main__":
    raise SystemExit(main())
