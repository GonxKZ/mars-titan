"""Opciones de la campaña A v2 desde su declaración hasta el proceso que entrena.

Una copia reducida de la campaña v2 conserva la sección neuronal declarada (lote, precisión y
CUDA Graphs), las recetas de Titans-MAC, la GRU episódica y CM-v1 y sus opciones de memoria
con el recibo de la medida. Se lanza la ventana fold-006 de cada ámbito con los ejecutores
reales hasta la entrada de su entrenador (`launch_doubles`), primero en ranuras con un
proceso por trabajo. Se comprueba qué recibe cada entrenador, que una opción no declarada se
rechaza antes de abrir datos, que cada opción cambia la identidad y que una campaña cortada
se reanuda con las mismas opciones. Ningún entrenador da un paso de ajuste y la GPU no se usa.
"""

import json
import os
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from mars_titan.data.storage import atomic_json
from mars_titan.training import campaign_numerics as numerics
from mars_titan.training import campaign_plan as plan
from mars_titan.training import campaign_schedule as order
from mars_titan.training import masked_campaign as engine
from mars_titan.training.campaign_resources import Execution
from mars_titan.training.input_pipeline import PipelineOptions
from mars_titan.training.kernel_policy import FP32_STRICT, RETIRED_NATIVE_OVERRIDES
from tests.training import launch_doubles
from tests.training.test_campaign_a_joint import CAMPAIGN, reduced
from tests.training.test_walk_forward_v2_views import fixture

WINDOW = "fold-006"
ARMS = ("gru", "ridge", "titans_mac_online", "gru_episodic", *plan.CM_ARMS)
TITANS_RECIPE = Path("configs/titans/chronological-training-historical-masked.json")
EPISODIC_RECIPE = Path("configs/candidate/chronological-training.json")
CM_DECLARATION = Path("configs/titans/cm-v1-factorial.json")
# Entrenador que recibe la receta de cada familia con opciones de memoria. Los núcleos de
# CM-v1 pasan por el recorrido de Titans-MAC con su propia receta.
TRAINERS = {
    plan.TITANS: "titans_walk_forward",
    plan.CM: "titans_walk_forward",
    plan.EPISODIC: "candidate_walk_forward",
}


def declared_config(folder, change=None):
    """Campaña v2 reducida a los brazos que reciben opciones, con sus recetas declaradas."""
    path, _ = reduced(folder, arms=ARMS)
    value = json.loads(path.read_text())
    declared = json.loads(CAMPAIGN.read_text())
    value["neural"]["arms"] = {"gru": declared["neural"]["arms"]["gru"]}
    for section, key in (
        (plan.TITANS, "recipe"),
        (plan.EPISODIC, "recipe"),
        (plan.CM, "declaration"),
    ):
        value[section] = dict(
            declared[section], **{key: str((CAMPAIGN.parent / declared[section][key]).resolve())}
        )
    value[plan.TITANS]["arms"] = {"titans_mac_online": "mac_online"}
    memory = declared["memory_options"]
    value["memory_options"] = dict(
        memory, receipt=str((CAMPAIGN.parent / memory["receipt"]).resolve())
    )
    if change is not None:
        change(value, folder)
    atomic_json(path, value)
    return path


def recipe_copy(folder, source, name, **options):
    """Copia de una receta con otras opciones de memoria y su protocolo con ruta absoluta."""
    document = json.loads(source.read_text())
    document["recipe"].update(options)
    if "protocol" in document:
        document["protocol"] = str((source.parent / document["protocol"]).resolve())
    atomic_json(folder / name, document)
    return folder / name


def measured(value, folder, family, **options):
    """Fijar opciones de memoria con una copia del recibo que contiene su medida."""
    path = folder / "memory-receipt.json"
    if not path.exists():
        atomic_json(path, json.loads(Path(value["memory_options"]["receipt"]).read_text()))
    receipt = json.loads(path.read_text())
    model = plan.RECEIPT_MODELS[family]
    receipt["entries"].append(dict(model=model, measured=True, recipe_options=options))
    atomic_json(path, receipt)
    value["memory_options"].update({"receipt": str(path), family: options})


def titans_rows(value, folder):
    value[plan.TITANS]["recipe"] = str(
        recipe_copy(folder, TITANS_RECIPE, "titans.json", accumulation_rows=2048)
    )
    measured(value, folder, plan.TITANS, accumulation_rows=2048)


def episodic_options(value, folder):
    value[plan.EPISODIC]["recipe"] = str(
        recipe_copy(
            folder, EPISODIC_RECIPE, "candidate.json", accumulation_rows=256, recompute=True
        )
    )
    measured(value, folder, plan.EPISODIC, accumulation_rows=256, recompute=True)


def cm_rows(value, folder):
    core = recipe_copy(folder, TITANS_RECIPE, "cm-core.json", accumulation_rows=2048)
    document = json.loads(CM_DECLARATION.read_text())
    document["base"].update(
        core_recipe=str(core.resolve()),
        readout_recipe=str((CM_DECLARATION.parent / document["base"]["readout_recipe"]).resolve()),
    )
    atomic_json(folder / "cm.json", document)
    value[plan.CM]["declaration"] = str((folder / "cm.json").resolve())
    measured(value, folder, plan.CM, accumulation_rows=2048)


# Cada opción declarada y las familias cuyos trabajos cambian de caso con ella. El lote no
# está en el caso: cambia la huella de la campaña, que entra en la identidad de cada trabajo.
CHANGES = {
    "batch_512": (lambda value, _: value["neural"].update(batch_size=512), ()),
    "graphs_off": (lambda value, _: value["neural"].update(cuda_graphs=False), (plan.NEURAL,)),
    "precision_absent": (lambda value, _: value["neural"].pop("precision"), (plan.NEURAL,)),
    "titans_rows": (titans_rows, (plan.TITANS,)),
    "episodic_options": (episodic_options, (plan.EPISODIC,)),
    "cm_rows": (cm_rows, (plan.CM,)),
}


def strict_policy(recorded):
    """La política que registra el entrenador es FP32 estricto con las sustituciones retiradas.

    La lista de sustituciones vigentes depende del proceso, así que solo se exige que no
    quede ninguna de las retiradas.
    """
    fixed = {key: value for key, value in recorded.items() if key != "native_overrides"}
    return fixed == dict(
        schema_version=1,
        precision=FP32_STRICT,
        matmul_allow_tf32=False,
        cudnn_allow_tf32=False,
        float32_matmul_precision="highest",
        retired_native_overrides=list(RETIRED_NATIVE_OVERRIDES),
    ) and not any(
        item.split("/", 1)[0] in RETIRED_NATIVE_OVERRIDES for item in recorded["native_overrides"]
    )


def run_window(campaign, views, output, *, execution=None):
    executors = {
        key: dict(entry, run=launch_doubles.launch_job) for key, entry in engine.EXECUTORS.items()
    }
    return engine.run_campaign(
        campaign,
        views,
        output,
        executors=executors,
        lease=nullcontext,
        stop=SimpleNamespace(requested=False),
        execution=execution,
        window=WINDOW,
    )


def received(path):
    """Lo que recibió cada entrenador por trabajo, sin el proceso que lo ejecutó."""
    result = {}
    for entry in launch_doubles.received(path):
        result.setdefault(entry.pop("id"), []).append(
            {k: v for k, v in entry.items() if k != "pid"}
        )
    return result


def differing(first, second):
    """Recibos y campos de identidad distintos entre dos salidas."""
    assert set(first) == set(second)
    return sorted(
        (path, key)
        for path in first
        for key in set(first[path]) | set(second[path])
        if first[path].get(key) != second[path].get(key)
    )


def receipts(output):
    """Identidad de cada recibo. Las huellas de los recibos de origen incluyen su fecha de
    confirmación, así que de las fuentes se conservan los trabajos que nombran."""
    result = {}
    for path in sorted((output / "jobs").rglob("receipt.json")):
        identity = json.loads(path.read_text())["identity"]
        identity["sources"] = {
            key: value for key, value in identity["sources"].items() if not key.endswith("_sha256")
        }
        result[path.relative_to(output).as_posix()] = identity
    return result


@pytest.fixture(scope="module")
def permitted(tmp_path_factory):
    from mars_titan.training.learning_hold import HOLD_ENV

    path = tmp_path_factory.mktemp("hold") / "training-hold.json"
    path.write_text(json.dumps({"training_allowed": True}), encoding="utf-8")
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv(HOLD_ENV, str(path))
        yield path


@pytest.fixture(autouse=True)
def _restore_numerics():
    """La campaña fija la precisión del proceso. Las demás pruebas conservan la suya."""
    before = numerics.current()
    yield
    numerics.apply(before)


@pytest.fixture(scope="module")
def launched(tmp_path_factory, permitted):
    """La ventana lanzada en dos ranuras, con un proceso por trabajo."""
    root = tmp_path_factory.mktemp("launch-options")
    data = fixture(root / "data", ("US", "CN"))
    campaign = declared_config(root / "config")
    checked = engine.prepare_views(campaign, data.parent, root / "views")
    views = {scope: root / "views" / scope for scope in checked}
    log = root / "launch.jsonl"
    slots = Execution(
        gpu_slots=2,
        cpu_workers=1,
        threads_per_job=2,
        pipeline=PipelineOptions(decode_workers=2, prefetch_batches=2),
    )
    before = numerics.current()
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("LAUNCH_LOG", str(log))
        summary = run_window(campaign, views, root / "out", execution=slots)
    numerics.apply(before)
    return SimpleNamespace(
        root=root,
        campaign=campaign,
        loaded=plan.load_campaign(campaign),
        views=views,
        output=root / "out",
        summary=summary,
        entries=launch_doubles.received(log),
        received=received(log),
    )


def window_jobs(loaded):
    return order.window_jobs(loaded, plan.plan_campaign(loaded), WINDOW)


def reached_jobs(jobs):
    return [job for job in jobs if (job["model"], job["kind"]) in launch_doubles.REACHED]


def test_every_trainer_of_the_window_receives_the_declared_options_in_its_slot(launched):
    assert launched.summary["status"] == "completed"
    jobs = reached_jobs(window_jobs(launched.loaded))
    assert {job["family"] for job in jobs} == {plan.NEURAL, plan.TITANS, plan.EPISODIC, plan.CM}
    # Cada trabajo llega una sola vez a su entrenador, en un proceso distinto de la campaña.
    assert sorted(launched.received) == sorted(job["id"] for job in jobs)
    assert all(len(entries) == 1 for entries in launched.received.values())
    assert os.getpid() not in {entry["pid"] for entry in launched.entries}
    neural = launched.loaded["neural"]
    memory = launched.loaded["memory_options"]
    assert memory == {
        plan.TITANS: dict(accumulation_rows=1024),
        plan.EPISODIC: dict(accumulation_rows=128, recompute=False),
        plan.CM: dict(accumulation_rows=1024),
    }
    identities = {identity["id"]: identity for identity in receipts(launched.output).values()}
    for job in jobs:
        (entry,) = launched.received[job["id"]]
        # Todos los entrenadores trabajan con FP32 estricto, sin TF32.
        assert entry["numerics"] == numerics.STRICT_FP32
        if job["family"] == plan.NEURAL:
            # El caso que confirma el recibo, con la precisión y el paso con grafo declarados.
            assert entry["trainer"] == "reference_run"
            assert entry["case"] == identities[job["id"]]["case"]
            assert (entry["case"]["precision"], entry["case"]["cuda_graphs"]) == (
                FP32_STRICT,
                True,
            )
            assert entry["batch_size"] == neural["batch_size"] == 256
            assert strict_policy(entry["kernel_policy"])
            assert entry["graph_step"] is True
        else:
            # La receta que recibe el entrenador lleva las opciones medidas y su precisión.
            assert entry["trainer"] == TRAINERS[job["family"]]
            options = memory[job["family"]]
            assert {key: entry["recipe"][key] for key in options} == options
            assert entry["recipe"]["precision"] == FP32_STRICT


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda v: v["neural"].update(precision="tf32"), "precisión neuronal"),
        (lambda v: v["neural"].update(cuda_graphs="true"), "booleano"),
        (lambda v: v["neural"].update(batch_size=4097), "Lote"),
        (lambda v: v["neural"].update(compile=True), "sección neuronal no cumple"),
        (
            lambda v: v["memory_options"][plan.TITANS].update(accumulation_rows=512),
            "coincidir con su receta",
        ),
        (
            lambda v: v["memory_options"][plan.TITANS].update(accumulation_rows=1024.0),
            "coincidir con su receta",
        ),
        (
            lambda v: v["memory_options"][plan.EPISODIC].update(recompute=0),
            "coincidir con su receta",
        ),
        (lambda v: v["memory_options"].pop("receipt"), "recibo de su medida"),
        (
            lambda v: v["memory_options"][plan.CM].update(accumulation_rows="pending"),
            "no se puede lanzar",
        ),
    ],
    ids=[
        "tf32",
        "graphs_text",
        "batch_over_limit",
        "undeclared_option",
        "rows_other_than_recipe",
        "rows_as_float",
        "recompute_as_integer",
        "without_receipt",
        "pending_rows",
    ],
)
def test_an_undeclared_value_is_rejected_before_opening_views_or_output(
    tmp_path, permitted, change, message
):
    campaign = declared_config(tmp_path / "config", lambda value, _: change(value))
    views = {scope: tmp_path / "missing" / scope for scope in ("US+CN", "US", "CN")}
    with pytest.raises(ValueError, match=message):
        run_window(campaign, views, tmp_path / "out")
    assert not (tmp_path / "missing").exists() and not (tmp_path / "out").exists()


@pytest.mark.parametrize("name", CHANGES)
def test_each_option_changes_the_identity_of_the_campaign_and_only_the_cases_it_reaches(
    tmp_path, name
):
    change, families = CHANGES[name]
    base = plan.load_campaign(declared_config(tmp_path / "base"))
    changed = plan.load_campaign(declared_config(tmp_path / name, change))
    # La huella de la campaña entra en `campaign_identity_sha256` de cada trabajo.
    assert changed["sha256"] != base["sha256"]
    jobs = {job["id"]: job for job in plan.plan_campaign(base)}
    other = {job["id"]: job for job in plan.plan_campaign(changed)}
    assert set(other) == set(jobs)
    moved = {jobs[key]["family"] for key in jobs if jobs[key] != other[key]}
    assert moved == set(families)


def test_a_cut_campaign_resumes_every_job_with_the_same_options(launched, tmp_path, monkeypatch):
    log = tmp_path / "launch.jsonl"
    monkeypatch.setenv("LAUNCH_LOG", str(log))
    jobs = reached_jobs(window_jobs(launched.loaded))
    # El corte llega a mitad de la ventana, en una búsqueda neuronal cuyo entrenador ya
    # recibió las opciones. Los trabajos anteriores quedan confirmados.
    cut = next(job["id"] for job in jobs if job["family"] == plan.NEURAL and job["scope"] == "US")
    monkeypatch.setenv("LAUNCH_PAUSE", cut)
    output = tmp_path / "out"
    assert run_window(launched.campaign, launched.views, output)["status"] == "paused"
    monkeypatch.delenv("LAUNCH_PAUSE")
    first = received(log)
    assert cut in first
    # Con una opción distinta la salida cortada pertenece a otra campaña y no se lanza nada.
    for name, (change, _) in CHANGES.items():
        other = declared_config(tmp_path / name, change)
        with pytest.raises(ValueError, match="otra campaña"):
            run_window(other, launched.views, output)
    assert received(log) == first
    assert run_window(launched.campaign, launched.views, output)["status"] == "completed"
    resumed = received(log)
    # El trabajo cortado vuelve a su entrenador con las mismas opciones. Los demás llegan
    # una vez y con lo mismo que en la ejecución sin corte, en ranuras.
    assert resumed[cut] == first[cut] * 2
    assert {key: entries[-1:] for key, entries in resumed.items()} == launched.received
    assert differing(receipts(output), receipts(launched.output)) == []


def test_changed_options_reach_the_trainers_that_use_them(launched, tmp_path, monkeypatch):
    def everything(value, folder):
        for name in ("batch_512", "graphs_off", "titans_rows", "episodic_options", "cm_rows"):
            CHANGES[name][0](value, folder)

    log = tmp_path / "launch.jsonl"
    monkeypatch.setenv("LAUNCH_LOG", str(log))
    campaign = declared_config(tmp_path / "config", everything)
    assert run_window(campaign, launched.views, tmp_path / "out")["status"] == "completed"
    jobs = reached_jobs(window_jobs(plan.load_campaign(campaign)))
    got = received(log)
    assert sorted(got) == sorted(job["id"] for job in jobs)
    for job in jobs:
        (entry,) = got[job["id"]]
        if job["family"] == plan.NEURAL:
            assert entry["batch_size"] == 512 and "cuda_graphs" not in entry["case"]
            assert entry["case"]["precision"] == FP32_STRICT and entry["graph_step"] is False
            assert strict_policy(entry["kernel_policy"])
        elif job["family"] == plan.EPISODIC:
            assert (entry["recipe"]["accumulation_rows"], entry["recipe"]["recompute"]) == (
                256,
                True,
            )
        else:
            assert entry["recipe"]["accumulation_rows"] == 2048
    before, after = receipts(launched.output), receipts(tmp_path / "out")
    assert set(before) == set(after)
    assert all(
        before[key]["campaign_identity_sha256"] != after[key]["campaign_identity_sha256"]
        for key in before
    )
