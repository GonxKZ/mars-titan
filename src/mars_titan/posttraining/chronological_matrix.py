"""Brazos de la matriz de adaptadores para las familias con entrenador cronológico.

La versión 3 de la matriz añade Titans-MAC, los lectores episódicos de MARS-TITAN y CM-v1
y la GRU candidata nativa. Comparten puntos, formas, controles, objetivos, presupuesto y
selección con las referencias. Solo cambian los tensores de cada punto:

- Titans-MAC: cabeza, lectura de MAC (proyección de consulta de la memoria y `out_proj`
  de su atención) y primera capa de la fusión. La memoria neuronal y la persistente no se
  adaptan. La lectura solo existe en las variantes que leen la memoria.
- Lectores episódicos: el núcleo con los mismos destinos que `mac_online`, la lectura
  episódica (`query_projection` y `value_projection` del lector) o ambos, como brazos
  separados. Sin banco (M0) la lectura no recibe gradiente y solo existe el núcleo.
- GRU candidata: solo la cabeza, calculada en Python sobre el estado nativo final. Los
  demás puntos viven dentro del módulo LibTorch y se excluyen con su motivo.

La sección opcional `variety` (`adapter_variety`) añade brazos de un solo punto a las
variantes de Titans-MAC donde existe su punto y, si los nombra en `readers`, al núcleo de
los lectores. La GRU candidata no recibe ninguno: sus otros parámetros viven en LibTorch.

Los casos no incluyen la corrección lineal, excluida para padres de cuantiles, y el padre
congelado no se ajusta. Las actualizaciones las fija el recorrido cronológico de cada
ventana: dependen de los datos y del tramo, no de los parámetros, así que coinciden entre
los brazos de un mismo padre.
"""

import itertools

from mars_titan.models.predictive_adaptation import AdapterTarget, target_shape
from mars_titan.models.quantile_head import QUANTILE_HEAD

from . import adapter_variety
from .inputs import fingerprint
from .selection import selection_policy

TITANS, READOUT, CANDIDATE = "titans_mac", "episodic_readout", "episodic_gru"
DESIGNS = (TITANS, READOUT, CANDIDATE)
# Familias de la campaña que usan cada diseño de la matriz.
CAMPAIGN_DESIGNS = {
    "titans_mac": TITANS,
    "mars_titan": READOUT,
    "cm_v1": READOUT,
    "episodic_gru": CANDIDATE,
}
TITANS_VARIANTS = ("transformer_direct", "mac_disabled", "mac_frozen", "mac_online")
# La memoria neuronal (pesos rápidos y sus proyecciones) y la persistente nunca se adaptan.
TITANS_FROZEN = ["mac.memory", "mac.persistent"]
COMPONENTS = ("core", "episodic_readout")
CANDIDATE_HEAD = ("head_weight", "head_bias")
CONTROL = "full_continuation"
OBJECTIVE = "neural_pinball"
# Pérdida de las recetas cronológicas que corresponde al objetivo de la matriz.
RECIPE_LOSS = {OBJECTIVE: "pinball"}
CASE_FIELDS = {"objective", "seed", "epochs", "learning_rate", "weight_decay", "clip_norm"}
CASE_FIELDS |= {"selection", "control", "adapter"}
_MIN_REASON = 20


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _reason(value):
    return isinstance(value, str) and len(value.strip()) >= _MIN_REASON


def _paths(values):
    return (
        isinstance(values, list)
        and values
        and len(set(values)) == len(values)
        and all(isinstance(value, str) and value for value in values)
    )


def _outside(paths, frozen):
    return not any(path == item or path.startswith(item + ".") for path in paths for item in frozen)


def validate(section, points):
    """Comprobar el diseño cronológico de la matriz frente a sus puntos declarados."""
    _require(
        isinstance(section, dict) and set(section) == set(DESIGNS),
        "La versión 3 declara Titans-MAC, los lectores episódicos y la GRU candidata",
    )
    titans = section[TITANS]
    _require(
        isinstance(titans, dict)
        and set(titans) == {"targets", "frozen", "variants", "inapplicable"}
        and isinstance(titans["targets"], dict)
        and tuple(titans["targets"]) == tuple(points)
        and all(_paths(paths) for paths in titans["targets"].values())
        and titans["frozen"] == TITANS_FROZEN
        and all(_outside(paths, titans["frozen"]) for paths in titans["targets"].values())
        and isinstance(titans["variants"], dict)
        and tuple(titans["variants"]) == TITANS_VARIANTS
        and all(
            isinstance(value, list) and value == [p for p in points if p in value] and value
            for value in titans["variants"].values()
        )
        and titans["variants"]["mac_online"] == list(points),
        "Titans-MAC declara destinos fuera de la memoria y los puntos de cada variante",
    )
    missing = {p for value in titans["variants"].values() for p in points if p not in value}
    _require(
        isinstance(titans["inapplicable"], dict)
        and set(titans["inapplicable"]) == missing
        and all(_reason(value) for value in titans["inapplicable"].values()),
        "Cada punto que falta en una variante de Titans-MAC necesita su motivo",
    )
    readout = section[READOUT]
    components = readout.get("components") if isinstance(readout, dict) else None
    _require(
        isinstance(readout, dict)
        and set(readout) == {"families", "components", "arms", "frozen", "without_bank"}
        and readout["families"] == ["mars_titan", "cm_v1"]
        and isinstance(components, dict)
        and tuple(components) == COMPONENTS
        and components["core"] == dict(family=TITANS, variant="mac_online")
        and isinstance(components["episodic_readout"], dict)
        and set(components["episodic_readout"]) == {"point", "targets"}
        and components["episodic_readout"]["point"] in points
        and _paths(components["episodic_readout"]["targets"])
        and _paths(readout["frozen"])
        and _outside(components["episodic_readout"]["targets"], readout["frozen"])
        and _reason(readout["without_bank"]),
        "Los lectores declaran el núcleo, la lectura episódica y su motivo sin banco",
    )
    expected = [
        list(combination)
        for size in (1, 2)
        for combination in itertools.combinations(COMPONENTS, size)
    ]
    _require(
        isinstance(readout["arms"], list)
        and sorted(map(tuple, readout["arms"])) == sorted(map(tuple, expected))
        and all(arm == [c for c in COMPONENTS if c in arm] for arm in readout["arms"]),
        "Los lectores adaptan el núcleo, la lectura o ambos como brazos separados",
    )
    candidate = section[CANDIDATE]
    _require(
        isinstance(candidate, dict)
        and set(candidate) == {"points", "targets", "excluded"}
        and candidate["points"] == ["head"]
        and candidate["targets"] == {"head": list(CANDIDATE_HEAD)}
        and isinstance(candidate["excluded"], dict)
        and set(candidate["excluded"]) == set(points) - {"head"}
        and all(_reason(value) for value in candidate["excluded"].values()),
        "La GRU candidata adapta su cabeza y explica cada punto nativo excluido",
    )
    return section


def design(matrix, family):
    """Diseño de la matriz que corresponde a una familia de la campaña."""
    _require(
        matrix["schema_version"] >= 3 and family in CAMPAIGN_DESIGNS,
        "La familia no tiene un diseño cronológico en esta matriz",
    )
    return CAMPAIGN_DESIGNS[family], matrix["architectures"]["chronological"]


def _spec(matrix, arm, name):
    return dict(arm.get("overrides", {}).get(name, matrix["points"][name]))


def arms(matrix, family, *, variant=None, bank=True, reserve=False):
    """Brazos aplicables, en el orden de la matriz, con la forma resuelta de cada punto.

    `variant` es la de Titans-MAC y `bank` dice si el lector consulta episodios. Los brazos
    de la variedad siguen a los de la matriz. `reserve` añade los que no se proponen para
    la campaña.
    """
    kind, section = design(matrix, family)
    if kind == TITANS:
        _require(variant in TITANS_VARIANTS, "La variante no pertenece a Titans-MAC")
        allowed = section[TITANS]["variants"][variant]
        return [
            dict(id=arm["id"], points={name: _spec(matrix, arm, name) for name in arm["points"]})
            for arm in matrix["arms"]
            if set(arm["points"]) <= set(allowed)
        ] + adapter_variety.titans_arms(matrix, variant, reserve=reserve)
    if kind == CANDIDATE:
        _require(variant is None, "La GRU candidata no tiene variantes de Titans-MAC")
        return [
            dict(id=arm["id"], points={"head": _spec(matrix, arm, "head")})
            for arm in matrix["arms"]
            if arm["points"] == section[CANDIDATE]["points"] and "overrides" not in arm
        ]
    _require(variant is None and type(bank) is bool, "El lector declara solo si tiene banco")
    readout = section[READOUT]
    core = {
        name: dict(matrix["points"][name]) for name in section[TITANS]["variants"]["mac_online"]
    }
    episodic = dict(matrix["points"][readout["components"]["episodic_readout"]["point"]])
    result = []
    for components in readout["arms"]:
        if "episodic_readout" in components and not bank:
            continue
        result.append(
            dict(
                id="+".join(components),
                core=core if "core" in components else None,
                episodic_readout=episodic if "episodic_readout" in components else None,
            )
        )
    return result + adapter_variety.reader_arms(matrix, reserve=reserve)


def _case(matrix, seed, objective, *, control=None, adapter=None):
    budget = matrix["budget"]
    case = dict(
        objective=objective,
        seed=seed,
        epochs=budget["epochs"],
        learning_rate=budget["learning_rate"],
        weight_decay=budget["weight_decay"],
        clip_norm=budget["clip_norm"],
        selection=dict(matrix["selection"]),
        control=control,
        adapter=adapter,
    )
    validate_case(case)
    return case


def cases(matrix, digest, family, *, variant=None, bank=True, reserve=False):
    """Casos por semilla: continuación completa y brazos, sin corrección lineal ni padre."""
    from . import adapter_matrix

    kind, _ = design(matrix, family)
    declared = adapter_matrix.objectives(matrix, QUANTILE_HEAD)
    _require(
        declared["adapters"] == OBJECTIVE
        and declared["full_continuation"] == OBJECTIVE
        and isinstance(declared["linear_residual"], dict),
        "Las familias cronológicas emiten cuantiles y se ajustan con pinball",
    )
    result = []
    for seed in matrix["budget"]["seeds"]:
        result.append(
            dict(
                id=f"seed-{seed}/{CONTROL}",
                control=CONTROL,
                case=_case(matrix, seed, declared[CONTROL], control=CONTROL),
            )
        )
        for arm in arms(matrix, family, variant=variant, bank=bank, reserve=reserve):
            adapter = dict(
                matrix_sha256=digest,
                input_policy=matrix["input_policy"],
                design=kind,
                **arm,
                arm=arm["id"],
            )
            adapter.pop("id")
            result.append(
                dict(
                    id=f"seed-{seed}/{arm['id']}",
                    control=None,
                    case=_case(matrix, seed, declared["adapters"], adapter=adapter),
                )
            )
    return result


def validate_case(case):
    """Caso cronológico de la matriz: presupuesto común y un control o un adaptador."""
    _require(
        isinstance(case, dict)
        and set(case) == CASE_FIELDS
        and case["objective"] in RECIPE_LOSS
        and type(case["seed"]) is int
        and 0 <= case["seed"] < 2**32
        and type(case["epochs"]) is int
        and 1 <= case["epochs"] <= 1000
        and (case["control"] is None) != (case["adapter"] is None)
        and case["control"] in (None, CONTROL),
        "El caso cronológico no pertenece a la matriz declarada",
    )
    selection_policy(dict(case, condition="real"))
    if case["adapter"] is not None:
        adapter = case["adapter"]
        keys = {"matrix_sha256", "input_policy", "design", "arm"}
        _require(
            isinstance(adapter, dict)
            and adapter.get("design") in DESIGNS
            and set(adapter)
            == keys | ({"core", "episodic_readout"} if adapter["design"] == READOUT else {"points"})
            and isinstance(adapter["matrix_sha256"], str)
            and len(adapter["matrix_sha256"]) == 64,
            "El adaptador del caso no pertenece a un diseño cronológico",
        )
    return case


def recipe_options(case):
    """Optimizador, presupuesto y selección del caso para una receta cronológica."""
    from .selection import _options

    return dict(
        loss=RECIPE_LOSS[case["objective"]],
        learning_rate=case["learning_rate"],
        weight_decay=case["weight_decay"],
        max_grad_norm=case["clip_norm"],
        epochs=case["epochs"],
        selection=_options(case["selection"], case["epochs"]),
    )


def component_seed(case, component):
    """Semilla de V por componente, separada del RNG global del ajuste."""
    adapter = case["adapter"]
    return int(
        fingerprint([case["seed"], adapter["arm"], adapter["matrix_sha256"], component])[:15], 16
    )


def _options(spec):
    if spec["form"] == "residual":
        return {}
    return dict(rank=spec["rank"], alpha=float(spec["alpha"]))


def variant_arm(matrix, variant, adapter):
    """Exigir que el adaptador del caso sea un brazo, de la matriz o de reserva, de la variante."""
    _require(
        any(
            arm["id"] == adapter["arm"] and arm["points"] == adapter["points"]
            for arm in arms(matrix, "titans_mac", variant=variant, reserve=True)
        ),
        "El brazo no pertenece a los puntos de esta variante de Titans-MAC",
    )


def titans_targets(matrix, points, model=None):
    """Tensores de Titans-MAC para los puntos del brazo, con su forma resuelta.

    Los subconjuntos de la variedad (sesgos y normalizaciones) se enumeran sobre `model`.
    """
    targets = matrix["architectures"]["chronological"][TITANS]["targets"]
    result = []
    for name, spec in points.items():
        if adapter_variety.is_variety(name, spec):
            result += adapter_variety.titans_targets(matrix, name, spec, model)
            continue
        for module in targets[name]:
            result.append(AdapterTarget(module, "weight", spec["form"], **_options(spec)))
            if name == "head" and spec["form"] == "residual":
                result.append(AdapterTarget(module, "bias", "residual"))
    return result


def readout_targets(matrix, spec):
    """Proyecciones de consulta y valor del lector episódico, sin sus sesgos."""
    section = matrix["architectures"]["chronological"][READOUT]["components"]["episodic_readout"]
    return [
        AdapterTarget(module, "weight", spec["form"], **_options(spec))
        for module in section["targets"]
    ]


def candidate_targets(points):
    """Cabeza de la GRU candidata como corrección completa de su peso y su sesgo."""
    _require(set(points) == {"head"}, "La GRU candidata solo adapta su cabeza")
    _require(points["head"]["form"] == "residual", "La cabeza usa una corrección completa")
    return [AdapterTarget("", "weight", "residual"), AdapterTarget("", "bias", "residual")]


def describe(targets, model):
    """Identidad de los tensores modificados y recuento exacto de parámetros entrenables."""
    rows = [target.identity(target_shape(model, target)) for target in targets]
    return dict(targets=rows, trainable_parameters=sum(row["trainable_parameters"] for row in rows))
