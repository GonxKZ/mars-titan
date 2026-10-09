"""Matriz finita de adaptadores sobre padres congelados, declarada antes de ajustar.

Cada brazo combina uno, dos o tres puntos de inserción con la misma población, las
mismas semillas, el mismo número de actualizaciones y la misma validación temporal.
El padre congelado, la corrección lineal residual y la continuación completa son
controles de todos los brazos. La matriz no añade combinaciones después de leerse.

La versión 1 solo declara objetivos para padres de salida escalar. La versión 2 los
declara por cabeza del padre: los escalares conservan los de la versión 1 y los de
`quantile_head_v1` optimizan la pinball de sus cinco niveles. Un control que no tiene
objetivo para una cabeza se excluye con su motivo, no se reinterpreta. La versión 3
conserva todo lo anterior y añade los destinos de las familias con entrenador cronológico
(`chronological_matrix`).
"""

import itertools
import math
from pathlib import Path

from mars_titan.data.cohort_files import read_manifest
from mars_titan.data.input_policy import INPUT_POLICIES
from mars_titan.models.predictive_adaptation import ADAPTER_FORMS, AdapterTarget
from mars_titan.models.quantile_head import QUANTILE_HEAD

from .inputs import fingerprint
from .selection import selection_policy

POINTS = ("head", "readout", "fusion")
CONTROLS = ("frozen_parent", "linear_residual", "full_continuation")
# Familias de MultimodalReference que el postentrenamiento puede cargar como padre.
FAMILIES = ("rnn", "lstm", "gru", "dlinear", "transformer")
# Solo el Transformer compacto tiene una lectura con consulta, claves y salida.
READOUT_FAMILIES = ("transformer",)
STATES = (
    "cached_parent_predictions",
    "attention_outputs",
    "fused_representations",
    "episodic_bank_keys",
)
BUDGET = {
    "seeds",
    "epochs",
    "batch_size",
    "learning_rate",
    "weight_decay",
    "clip_norm",
    "beta",
    "behavior_epsilon",
    "auxiliary_samples",
}
MAX_ARMS = 12
KIND = "posttraining_adapter_matrix"
SCALAR = "scalar"
HEADS = (SCALAR, QUANTILE_HEAD)
# Objetivos admitidos por cabeza. Una pinball de cinco niveles no tiene corrección
# lineal definida sobre la predicción escalar que guarda la caché del padre.
ADAPTER_OBJECTIVES = {SCALAR: ("neural_mae", "neural_mse"), QUANTILE_HEAD: ("neural_pinball",)}
LINEAR_OBJECTIVES = {SCALAR: ("mae", "expected"), QUANTILE_HEAD: ()}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _point(spec):
    _require(
        isinstance(spec, dict)
        and spec.get("form") in ADAPTER_FORMS
        and set(spec)
        == {"form", "invalidates"} | ({"rank", "alpha"} if spec["form"] == "low_rank" else set()),
        "Cada punto declara forma, estados invalidados y, si es de bajo rango, rango y escala",
    )
    _require(
        isinstance(spec["invalidates"], list)
        and spec["invalidates"]
        and len(set(spec["invalidates"])) == len(spec["invalidates"])
        and set(spec["invalidates"]) <= set(STATES),
        "Los estados invalidados no pertenecen al contrato",
    )
    if spec["form"] == "low_rank":
        _require(
            type(spec["rank"]) is int
            and 1 <= spec["rank"] <= 64
            and type(spec["alpha"]) in (int, float)
            and math.isfinite(spec["alpha"])
            and spec["alpha"] > 0,
            "El rango o la escala del adaptador no son válidos",
        )
    return dict(spec)


def _arm_id(points):
    return "+".join(points)


def validate_matrix(matrix):
    """Rechazar matrices incompletas, abiertas o con presupuestos distintos por brazo."""
    keys = {
        "schema_version",
        "kind",
        "input_policy",
        "final_test_opened",
        "points",
        "arms",
        "controls",
        "objectives",
        "budget",
        "selection",
        "architectures",
    }
    _require(
        isinstance(matrix, dict)
        and set(matrix) == keys
        and matrix["schema_version"] in (1, 2, 3)
        and type(matrix["schema_version"]) is int
        and matrix["kind"] == KIND
        and matrix["input_policy"] in INPUT_POLICIES
        and matrix["final_test_opened"] is False,
        "La matriz de adaptadores no cumple su contrato",
    )
    _require(
        isinstance(matrix["points"], dict) and tuple(matrix["points"]) == POINTS,
        "La matriz declara exactamente cabeza, lectura y fusión",
    )
    points = {name: _point(spec) for name, spec in matrix["points"].items()}
    _require(points["head"]["form"] == "residual", "La cabeza usa una corrección completa")
    _require(matrix["controls"] == list(CONTROLS), "Faltan los controles comunes")
    version_one = matrix["schema_version"] == 1
    declared = {SCALAR: matrix["objectives"]} if version_one else matrix["objectives"]
    _require(
        isinstance(declared, dict) and set(declared) == ({SCALAR} if version_one else set(HEADS)),
        "La versión 1 declara objetivos escalares y las demás uno por cada cabeza del padre",
    )
    for head, value in declared.items():
        _objectives_for(head, value)
    budget = matrix["budget"]
    _require(
        isinstance(budget, dict)
        and set(budget) == BUDGET
        and isinstance(budget["seeds"], list)
        and 1 <= len(budget["seeds"]) <= 10
        and len(set(budget["seeds"])) == len(budget["seeds"])
        and all(type(seed) is int and 0 <= seed < 2**32 for seed in budget["seeds"])
        and type(budget["batch_size"]) is int
        and 1 <= budget["batch_size"] <= 4096,
        "El presupuesto común no es válido",
    )
    selection = matrix["selection"]
    _require(
        isinstance(selection, dict)
        and selection.get("version") == 2
        and selection.get("patience") is None,
        "La matriz conserva todas las actualizaciones y elige el mejor estado, padre incluido",
    )
    seen, combinations = set(), set()
    _require(
        isinstance(matrix["arms"], list) and 1 <= len(matrix["arms"]) <= MAX_ARMS,
        "La matriz debe ser finita",
    )
    for arm in matrix["arms"]:
        _require(
            isinstance(arm, dict)
            and set(arm) <= {"id", "points", "overrides"}
            and {"id", "points"} <= set(arm)
            and isinstance(arm["points"], list)
            and 1 <= len(arm["points"]) <= 3
            and arm["points"] == [name for name in POINTS if name in arm["points"]]
            and len(set(arm["points"])) == len(arm["points"])
            and arm["id"] not in seen
            and arm["id"] not in CONTROLS,
            "Cada brazo combina de uno a tres puntos distintos en orden canónico",
        )
        seen.add(arm["id"])
        overrides = arm.get("overrides")
        if overrides is None:
            _require(arm["id"] == _arm_id(arm["points"]), "El brazo se nombra por sus puntos")
            combinations.add(tuple(arm["points"]))
        else:
            _require(
                isinstance(overrides, dict)
                and overrides
                and set(overrides) <= set(arm["points"])
                and "head" not in overrides,
                "Un control de rango solo cambia la forma de sus puntos",
            )
            for name, spec in overrides.items():
                _require(_point(spec) != points[name], "El control de rango cambia la forma")
    expected = {
        combination for size in (1, 2, 3) for combination in itertools.combinations(POINTS, size)
    }
    _require(
        combinations == expected, "La matriz contiene todas las combinaciones de 1, 2 y 3 puntos"
    )
    _architectures(matrix["architectures"], matrix["schema_version"])
    from .run import validate_case

    # Los casos derivados deben pertenecer al diseño emparejado del ajuste.
    for head in declared:
        for item in cases(matrix, "0" * 64, READOUT_FAMILIES[0], head=head):
            validate_case(item["case"])
    return matrix


def _objectives_for(head, value):
    """Exigir el mismo objetivo en adaptadores y continuación y explicar cada exclusión."""
    excluded = value.get("linear_residual") if isinstance(value, dict) else None
    _require(
        isinstance(value, dict)
        and set(value) == {"adapters", "full_continuation", "linear_residual"}
        and value["adapters"] in ADAPTER_OBJECTIVES[head]
        and value["full_continuation"] == value["adapters"]
        and (
            value["linear_residual"] in LINEAR_OBJECTIVES[head]
            or (
                not LINEAR_OBJECTIVES[head]
                and isinstance(excluded, dict)
                and set(excluded) == {"excluded"}
                and isinstance(excluded["excluded"], str)
                and len(excluded["excluded"].strip()) >= 20
            )
        ),
        "Los adaptadores y la continuación comparten el objetivo de la cabeza del padre",
    )
    return value


def objectives(matrix, head):
    """Objetivos declarados para la cabeza del padre."""
    _require(head in HEADS, "La cabeza del padre no pertenece al contrato")
    if matrix["schema_version"] == 1:
        _require(head == SCALAR, "La matriz versión 1 solo declara objetivos de salida escalar")
        return matrix["objectives"]
    return matrix["objectives"][head]


def excluded_controls(matrix, head):
    """Controles sin objetivo para la cabeza, con el motivo declarado antes de ajustar."""
    linear = objectives(matrix, head)["linear_residual"]
    return {"linear_residual": linear["excluded"]} if isinstance(linear, dict) else {}


def model_head(model):
    """Cabeza de un padre neuronal: cinco cuantiles o un centro escalar."""
    return QUANTILE_HEAD if getattr(model, "emits_quantiles", False) else SCALAR


def _architectures(declared, version=2):
    groups = {"executable", "pending"} | ({"chronological"} if version >= 3 else set())
    _require(
        isinstance(declared, dict) and set(declared) == groups,
        "Las arquitecturas se separan en ejecutables, cronológicas desde la versión 3 y pendientes",
    )
    if version >= 3:
        from .chronological_matrix import validate

        validate(declared["chronological"], POINTS)
    executable = declared["executable"]
    _require(
        isinstance(executable, dict)
        and set(executable) == set(FAMILIES)
        and all(
            value == [name for name in POINTS if name != "readout" or family in READOUT_FAMILIES]
            for family, value in executable.items()
        ),
        "Cada familia declara los puntos que existen en su red",
    )
    for name, value in declared["pending"].items():
        _require(
            isinstance(name, str)
            and isinstance(value, dict)
            and set(value) == {"reason", "targets", "frozen"}
            and isinstance(value["reason"], str)
            and value["reason"]
            and isinstance(value["targets"], dict)
            and set(value["targets"]) <= set(POINTS)
            and isinstance(value["frozen"], list)
            and all(
                isinstance(path, str)
                and not any(path == item or path.startswith(item + ".") for item in value["frozen"])
                for paths in value["targets"].values()
                for path in paths
            ),
            "Una arquitectura pendiente declara destinos fuera de sus estados congelados",
        )


def read_matrix(path):
    matrix, digest = read_manifest(Path(path), 1024**2)
    return validate_matrix(matrix), digest


def _family(family):
    _require(family in FAMILIES, "La matriz de adaptadores solo se aplica a padres neuronales")
    return family


def arms(matrix, family):
    """Brazos aplicables a la familia, con la forma resuelta de cada punto."""
    _family(family)
    result = []
    for arm in matrix["arms"]:
        if "readout" in arm["points"] and family not in READOUT_FAMILIES:
            continue
        points = {
            name: dict(arm.get("overrides", {}).get(name, matrix["points"][name]))
            for name in arm["points"]
        }
        result.append(dict(id=arm["id"], points=points))
    return result


def _case(matrix, seed, mode, adapter=None):
    budget = matrix["budget"]
    case = {key: budget[key] for key in sorted(BUDGET - {"seeds", "batch_size"})} | dict(
        mode=mode, condition="real", seed=seed, selection=dict(matrix["selection"])
    )
    if adapter is not None:
        case["adapter"] = adapter
    selection_policy(case)
    return case


def cases(matrix, digest, family, *, head=SCALAR):
    """Casos del ajuste, en orden fijo. El padre congelado se evalúa sin actualizaciones.

    Los objetivos dependen de la cabeza del padre. Un control excluido para esa cabeza
    no genera caso.
    """
    _family(family)
    declared = objectives(matrix, head)
    result = []
    for seed in matrix["budget"]["seeds"]:
        if not isinstance(declared["linear_residual"], dict):
            result.append(
                dict(
                    id=f"seed-{seed}/linear_residual",
                    control="linear_residual",
                    case=_case(matrix, seed, declared["linear_residual"]),
                )
            )
        result.append(
            dict(
                id=f"seed-{seed}/full_continuation",
                control="full_continuation",
                case=_case(matrix, seed, declared["full_continuation"]),
            )
        )
        for arm in arms(matrix, family):
            adapter = dict(
                matrix_sha256=digest,
                input_policy=matrix["input_policy"],
                arm=arm["id"],
                points=arm["points"],
            )
            result.append(
                dict(
                    id=f"seed-{seed}/{arm['id']}",
                    control=None,
                    case=_case(matrix, seed, declared["adapters"], adapter),
                )
            )
    return result


def validate_adapter(adapter):
    _require(
        isinstance(adapter, dict)
        and set(adapter) == {"matrix_sha256", "input_policy", "arm", "points"}
        and isinstance(adapter["matrix_sha256"], str)
        and len(adapter["matrix_sha256"]) == 64
        and adapter["input_policy"] in INPUT_POLICIES
        and isinstance(adapter["arm"], str)
        and isinstance(adapter["points"], dict)
        and 1 <= len(adapter["points"]) <= 3
        and list(adapter["points"]) == [name for name in POINTS if name in adapter["points"]],
        "El adaptador del caso no pertenece a una matriz declarada",
    )
    for spec in adapter["points"].values():
        _point(spec)
    return adapter


def targets(adapter, model):
    """Traducir los puntos del brazo a tensores del padre, sin tocar sus pesos."""
    validate_adapter(adapter)
    family = _family(getattr(model, "kind", None))
    hidden = model.architecture["hidden_size"]
    result = []
    for name, spec in adapter["points"].items():
        options = (
            {}
            if spec["form"] == "residual"
            else dict(rank=spec["rank"], alpha=float(spec["alpha"]))
        )
        if name == "head":
            result.append(AdapterTarget("head", "weight", spec["form"], **options))
            if spec["form"] == "residual":
                result.append(AdapterTarget("head", "bias", "residual"))
        elif name == "fusion":
            result.append(AdapterTarget("fusion.0", "weight", spec["form"], **options))
        else:
            _require(family in READOUT_FAMILIES, "La familia no tiene una lectura con consulta")
            for layer in range(len(model.price_encoder.blocks)):
                prefix = f"price_encoder.blocks.{layer}.self_attn"
                # Solo el bloque de consulta de la proyección empaquetada. Claves y valores
                # quedan congelados, igual que su salida original antes del adaptador.
                result.append(
                    AdapterTarget(
                        prefix, "in_proj_weight", spec["form"], rows=(0, hidden), **options
                    )
                )
                result.append(
                    AdapterTarget(f"{prefix}.out_proj", "weight", spec["form"], **options)
                )
    return result


def adapter_seed(case):
    """Semilla de la inicialización de V, separada del RNG global del ajuste."""
    return int(
        fingerprint([case["seed"], case["adapter"]["arm"], case["adapter"]["matrix_sha256"]])[:15],
        16,
    )


def describe(adapter, model):
    """Identidad de los tensores modificados y recuento exacto de parámetros entrenables."""
    resolved = targets(adapter, model)
    rows = [
        target.identity(tuple(getattr(model.get_submodule(target.module), target.tensor).shape))
        for target in resolved
    ]
    return dict(targets=rows, trainable_parameters=sum(row["trainable_parameters"] for row in rows))


def _states(points):
    return sorted({state for spec in points.values() for state in spec["invalidates"]})


def plan(matrix, digest, family, model, *, updates_per_epoch, linear_features):
    """Registrar antes del primer ajuste brazos, controles, parámetros y actualizaciones.

    Los bytes de estado cuentan parámetros entrenables y los dos momentos de AdamW en
    la precisión de los pesos. No incluyen activaciones, que se miden al ejecutar.
    """
    _require(
        type(updates_per_epoch) is int
        and updates_per_epoch >= 1
        and type(linear_features) is int
        and linear_features >= 1,
        "Las actualizaciones por época o la anchura lineal no son válidas",
    )
    head = model_head(model)
    itemsize = next(model.parameters()).element_size()
    updates = matrix["budget"]["epochs"] * updates_per_epoch
    # La continuación completa invalida todo lo que invalida cualquier punto aplicable.
    every = sorted({state for arm in arms(matrix, family) for state in _states(arm["points"])})
    rows = [
        dict(
            id="frozen_parent",
            control="frozen_parent",
            trainable_parameters=0,
            state_bytes=0,
            updates=0,
            invalidates=[],
            case=None,
        )
    ]
    for item in cases(matrix, digest, family, head=head):
        case = item["case"]
        if item["control"] == "linear_residual":
            # Una capa lineal float32 sobre modalidades, bits y predicción del padre.
            count, size, invalidates = linear_features + 1, 4, []
        elif item["control"] == "full_continuation":
            count = sum(value.numel() for value in model.parameters())
            size, invalidates = itemsize, every
        else:
            count = describe(case["adapter"], model)["trainable_parameters"]
            size, invalidates = itemsize, _states(case["adapter"]["points"])
        rows.append(
            dict(
                id=item["id"],
                control=item["control"],
                trainable_parameters=count,
                state_bytes=3 * size * count,
                updates=updates,
                invalidates=invalidates,
                case=case,
            )
        )
    _require(len({row["updates"] for row in rows[1:]}) == 1, "Los brazos no comparten presupuesto")
    return rows
