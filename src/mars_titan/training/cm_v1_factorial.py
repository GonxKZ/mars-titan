"""Factorial CM-v1 sobre Titans-MAC: B, B+C, B+M y B+C+M con identidades propias.

B fija la referencia del factorial. Es un núcleo Titans-MAC `mac_online` con el control
local C en modo disabled, que impone SDPA Math y la misma base que B+C, ajustado con la
receta del núcleo de la campaña, más un lector episódico M1 con K = 1 sobre ese núcleo
congelado. C solo cambia el ajuste del núcleo, con la penalización de `RᵀJR`. M solo cambia
la retención del banco del lector, que pasa a centros fijos con episodios reales. Por eso B
y B+M comparten el núcleo `cm_v1_core_b`, y B+C y B+C+M el núcleo `cm_v1_core_c`.

Los brazos usan las mismas semillas de parámetros, los mismos casos de búsqueda y el mismo
presupuesto de épocas. C no añade pasos: sin etiquetas maduras su término se descarta. La
base de C tiene su propio generador y la retención con centros fijos no usa el RNG del
reservorio, así que ningún factor consume el RNG de otro.

La declaración está en `configs/titans/cm-v1-factorial.json`. Sus recetas se resuelven
junto a ella. Cada entrada de ventana comprueba la protección del aprendizaje antes de leer
fuentes.
"""

import hashlib
from dataclasses import asdict
from pathlib import Path

from mars_titan.data.cohort_files import read_manifest
from mars_titan.data.input_policy import HISTORICAL_MASKED
from mars_titan.data.storage import sha256
from mars_titan.memory.mars_titan_variant import core_identity
from mars_titan.models.titans.config import canonical
from mars_titan.models.titans.episodic_readout import EpisodicReadoutConfig
from mars_titan.models.titans.local_control import MACProjectionConfig

from .financial_run import Paused
from .learning_hold import require_learning_allowed
from .mars_titan_run import retention_config
from .mars_titan_walk_forward import ReadoutFamily, carry_readout, run_readout_window
from .titans_walk_forward import _require, run_titans_window, unfused_attention, view_protocol

DECLARATION = Path(__file__).resolve().parents[3] / "configs/titans/cm-v1-factorial.json"
NAME = "mars_titan_cm_v1_factorial"
KIND = "cm_v1_walk_forward_window"
CARRY_KIND = "cm_v1_carried_predictions"
WORLD = "cm_v1_walk_forward"
# Brazo: (C activo, M activo). Núcleo: C activo.
ARMS = {
    "cm_v1_b": (False, False),
    "cm_v1_bc": (True, False),
    "cm_v1_bm": (False, True),
    "cm_v1_bcm": (True, True),
}
CORES = {"cm_v1_core_b": False, "cm_v1_core_c": True}
BASE = dict(
    variant="mac_online",
    admission="m1",
    refinements=1,
    episode_selection="per_step",
    retention="reservoir",
    control_mode="disabled",
)
_FIELDS = {"schema_version", "name", "status", "base", "control", "consolidation"}
_FIELDS |= {"cores", "arms", "pending"}
_CONTROL = {"rank", "frequency", "seed", "grid_size", "threshold", "weight", "max_flows"}
_CONTROL |= {"max_estimated_bytes"}
_CONSOLIDATION = {"policy", "frontier", "new_candidates", "max_swaps", "max_distance_pairs"}
_CONSOLIDATION |= {"max_working_bytes"}
# Campos del caso que la campaña declara para núcleos y brazos.
CORE_FIELDS = {"declaration", "declaration_sha256", "recipe", "recipe_sha256", "core"}
CORE_FIELDS |= {"seed", "search_case"}
ARM_FIELDS = {"declaration", "declaration_sha256", "recipe", "recipe_sha256", "arm"}
ARM_FIELDS |= {"seed", "search_case", "parent_arm"}


def load_declaration(path=DECLARATION):
    """Leer y validar la declaración. Devuelve el documento con su huella y sus recetas."""
    path = Path(path)
    document, digest = read_manifest(path, 64 * 1024)
    _require(
        isinstance(document, dict)
        and set(document) == _FIELDS
        and document["schema_version"] == 1
        and document["name"] == NAME
        and document["status"] == "declared_not_executed"
        and isinstance(document["base"], dict)
        and {key: document["base"].get(key) for key in BASE} == BASE
        and set(document["base"]) == set(BASE) | {"core_recipe", "readout_recipe"}
        and document["arms"]
        == {name: dict(control=c, consolidation=m) for name, (c, m) in ARMS.items()}
        and document["cores"] == CORES
        and isinstance(document["pending"], list)
        and all(isinstance(note, str) and note for note in document["pending"]),
        "La declaración de CM-v1 no conserva su esquema, su B y sus cuatro brazos",
    )
    control, consolidation = document["control"], document["consolidation"]
    _require(
        isinstance(control, dict)
        and set(control) == _CONTROL
        and isinstance(consolidation, dict)
        and set(consolidation) == _CONSOLIDATION
        and consolidation["policy"] == "anchored",
        "C declara los campos de su control y M la retención con centros fijos",
    )
    MACProjectionConfig(mode="penalty", **control)
    recipes = {
        name: (path.parent / document["base"][name]).resolve()
        for name in ("core_recipe", "readout_recipe")
    }
    return dict(
        document,
        path=str(path.resolve()),
        sha256=digest,
        recipes={name: str(value) for name, value in recipes.items()},
    )


def control_contract(declaration, enabled):
    """Contrato normalizado de C: penalty en B+C, disabled con la misma base en B."""
    fields = dict(declaration["control"])
    if not enabled:
        fields["weight"] = 0.0
    return asdict(MACProjectionConfig(mode="penalty" if enabled else "disabled", **fields))


def bank_retention(declaration, recipe, consolidation):
    """Banco M1 de B o de M. Comparten capacidad, semilla y límites. Solo cambia la política."""
    options = {k: v for k, v in declaration["consolidation"].items() if k != "policy"}
    policy = "anchored" if consolidation else "reservoir"
    return retention_config(recipe, "m1", policy=policy, **options)


class FactorialArm:
    """Brazo del factorial con la interfaz de variante que usa la ventana del lector."""

    admission = "m1"
    readout_mode = "bank"
    refinements = 1
    episode_selection = "per_step"

    def __init__(self, declaration, name, base):
        _require(name in ARMS, "El brazo no pertenece al factorial CM-v1")
        self.name, self.base = name, base
        self.control, self.consolidation = ARMS[name]
        self.declaration_sha256 = declaration["sha256"]

    def identity(self):
        return dict(
            schema_version=1,
            factorial=NAME,
            arm=self.name,
            control=self.control,
            consolidation=self.consolidation,
            readout=dict(
                admission=self.admission,
                refinements=self.refinements,
                episode_selection=self.episode_selection,
            ),
            base=self.base,
            declaration_sha256=self.declaration_sha256,
        )

    def fingerprint(self):
        return hashlib.sha256(canonical(self.identity()).encode()).hexdigest()

    def readout_config(self, codec_id, **options):
        _require(
            not {"mode", "refinements", "episode_selection", "codec_id"} & set(options),
            "Modo, K y episodios pertenecen al brazo del factorial",
        )
        return EpisodicReadoutConfig(codec_id, mode="bank", refinements=1, **options)


def readout_family(declaration, arm):
    """Familia de lector del brazo: control C esperado en el padre y retención de su banco."""
    _require(arm in ARMS, "El brazo no pertenece al factorial CM-v1")
    control, consolidation = ARMS[arm]

    def variant(predictor, report):
        request = report["request"]
        base = core_identity(
            predictor,
            dict(
                recipe_sha256=request["recipe_sha256"],
                search_case=request.get("search_case"),
                window=request["window"],
                local_control=request["local_control"],
            ),
        )
        return FactorialArm(declaration, arm, base)

    return ReadoutFamily(
        label="CM-v1",
        kind=KIND,
        carry_kind=CARRY_KIND,
        selected_kind="cm_v1_selected_state",
        world=WORLD,
        request=dict(
            arm=arm, declaration=declaration["path"], declaration_sha256=declaration["sha256"]
        ),
        control=control_contract(declaration, control),
        variant=variant,
        retention=lambda recipe, _: bank_retention(declaration, recipe, consolidation),
        code=("mars_titan.training.cm_v1_factorial", "mars_titan.models.titans.local_control"),
    )


def run_cm_v1_core_window(
    view,
    window,
    *,
    core,
    seed,
    output,
    search_case,
    declaration=DECLARATION,
    device="cuda:0",
    indices=None,
    stop=None,
    optimizer_factory=None,
):
    """Ajustar el núcleo de B (C disabled) o de B+C (C penalty) con la receta de la campaña."""
    require_learning_allowed("run_cm_v1_core_window de CM-v1")
    _require(core in CORES, "El núcleo no pertenece al factorial CM-v1")
    document = load_declaration(declaration)
    return run_titans_window(
        view,
        view_protocol(view),
        window,
        document["recipes"]["core_recipe"],
        variant="mac_online",
        seed=seed,
        output=output,
        device=device,
        indices=indices,
        stop=stop,
        optimizer_factory=optimizer_factory,
        search_case=search_case,
        local_control=control_contract(document, CORES[core]),
    )


def run_cm_v1_window(
    view,
    parent,
    *,
    arm,
    seed,
    output,
    search_case,
    declaration=DECLARATION,
    device="cuda:0",
    indices=None,
    stop=None,
    optimizer_factory=None,
):
    """Ajustar el lector M1 con K = 1 del brazo sobre su núcleo elegido en la ventana."""
    require_learning_allowed("run_cm_v1_window de CM-v1")
    document = load_declaration(declaration)
    return run_readout_window(
        readout_family(document, arm),
        view,
        parent,
        document["recipes"]["readout_recipe"],
        seed=seed,
        output=output,
        search_case=search_case,
        device=device,
        indices=indices,
        stop=stop,
        optimizer_factory=optimizer_factory,
    )


def carry_cm_v1(anchor, anchor_view, view, output, *, device="cuda:0", stop=None):
    """Predecir una ventana posterior con el núcleo y el lector elegidos en el ancla.

    La declaración es la que registra la petición del ancla, con la misma huella.
    """
    require_learning_allowed("la predicción trasladada de CM-v1")

    def family_of(report):
        request = report["request"]
        _require(
            isinstance(request.get("declaration"), str) and request.get("arm") in ARMS,
            "El ancla no es una ventana de un brazo de CM-v1",
        )
        document = load_declaration(request["declaration"])
        _require(
            request.get("declaration_sha256") == document["sha256"],
            "La declaración de CM-v1 cambió después de ajustar el ancla",
        )
        return readout_family(document, request["arm"])

    return carry_readout(family_of, anchor, anchor_view, view, output, device=device, stop=stop)


def _case(run, fields):
    case = run.case
    _require(
        isinstance(case, dict)
        and set(case) == fields
        and case["seed"] == run.job["seed"]
        and run.policy == HISTORICAL_MASKED,
        "El trabajo no declara un caso de CM-v1 de la campaña con máscaras",
    )
    _require(
        sha256(Path(case["declaration"])) == case["declaration_sha256"]
        and sha256(Path(case["recipe"])) == case["recipe_sha256"],
        "La declaración o la receta de CM-v1 cambiaron después de planificar la campaña",
    )
    return case


def cm_v1_core_fit(run, *, device="cuda:0", optimizer_factory=None):
    """Ejecutor de ajuste del núcleo para `training.masked_campaign`."""
    from .masked_campaign import Paused as CampaignPaused

    case = _case(run, CORE_FIELDS)
    _require(
        load_declaration(case["declaration"])["recipes"]["core_recipe"] == case["recipe"],
        "La receta del núcleo no es la de la declaración",
    )
    # El recorrido cronológico exige fastpath=False solo mientras dura el trabajo.
    with unfused_attention():
        report = run_cm_v1_core_window(
            run.view,
            run.job["window"],
            core=case["core"],
            seed=case["seed"],
            output=run.folder,
            search_case=case["search_case"],
            declaration=case["declaration"],
            device=device,
            stop=run.stop,
            optimizer_factory=optimizer_factory,
        )
    if report["status"] == "paused":
        raise CampaignPaused
    _require(
        report["status"] == "completed" and report["request"]["view_sha256"] == run.view_sha256,
        "La ventana del núcleo de CM-v1 no confirma la vista del trabajo",
    )
    return report


def cm_v1_fit(run, *, device="cuda:0", optimizer_factory=None):
    """Ejecutor de ajuste del lector del brazo, con el núcleo que resuelve la campaña."""
    from .masked_campaign import Paused as CampaignPaused

    case = _case(run, ARM_FIELDS)
    _require(isinstance(run.parent, dict), "El brazo de CM-v1 necesita su núcleo")
    _require(
        load_declaration(case["declaration"])["recipes"]["readout_recipe"] == case["recipe"],
        "La receta del lector no es la de la declaración",
    )
    with unfused_attention():
        report = run_cm_v1_window(
            run.view,
            run.parent["folder"],
            arm=case["arm"],
            seed=case["seed"],
            output=run.folder,
            search_case=case["search_case"],
            declaration=case["declaration"],
            device=device,
            stop=run.stop,
            optimizer_factory=optimizer_factory,
        )
    if report["status"] == "paused":
        raise CampaignPaused
    _require(
        report["status"] == "completed"
        and report["request"]["view_sha256"] == run.view_sha256
        and report["request"]["parent"]["checkpoint_sha256"] == run.parent["checkpoint_sha256"],
        "La ventana de CM-v1 no confirma la vista ni el núcleo del trabajo",
    )
    return report


def cm_v1_carry(run, *, device="cuda:0"):
    """Ejecutor de predicción trasladada del brazo (variante B de la campaña)."""
    from .masked_campaign import Paused as CampaignPaused

    try:
        with unfused_attention():
            return carry_cm_v1(
                run.anchor["folder"],
                run.anchor["view"],
                run.view,
                run.folder,
                device=device,
                stop=run.stop,
            )
    except Paused as error:
        raise CampaignPaused from error
