"""Variedad de adaptadores de la matriz de versión 3 (#444), declarada antes de ajustar.

La sección opcional `variety` de la matriz añade brazos de un solo punto sobre la misma
estructura de #364, con el mismo presupuesto, los mismos objetivos y la misma selección:

- formas nuevas en la lectura y en la fusión: DoRA, (IA)³ y adaptadores en cuello de
  botella en paralelo y en serie;
- subconjuntos de parámetros del padre: todos los sesgos (BitFit), las normalizaciones y
  la memoria persistente de Titans-MAC, que es su prefijo aprendido.

Cada brazo se aplica a las familias donde existe su punto. Un punto que falta en alguna
familia necesita un motivo escrito en la matriz. `campaign` enumera los ámbitos donde el
brazo se propone para la campaña: las referencias neuronales, cada variante de Titans-MAC
o los núcleos de los lectores. En la vista conjunta, un caso de Titans-MAC cuesta entre 13 y
104 veces uno de una referencia, así que un brazo barato en las referencias puede quedar de
reserva en Titans-MAC.
Fuera de sus ámbitos el brazo sigue implementado y comprobado como reserva. La selección
forma parte de la matriz, de su huella y de la identidad de cada ajuste.
`docs/engineering/adapter-variety.md` relaciona cada forma con su ecuación y su fuente.
"""

import math

from torch import nn

from mars_titan.models.predictive_adaptation import MODULE_ROOT, AdapterTarget

SELECTIVE_POINTS = ("bias", "norm", "persistent")
# Formas admitidas en cada punto de inserción de #364. La lectura solo admite formas por
# tensor, porque un gancho dentro de un bloque de atención apagaría la ruta rápida del
# Transformer y cambiaría el redondeo frente al padre.
INSERTION_FORMS = {
    "readout": ("dora", "ia3"),
    "fusion": ("dora", "ia3", "parallel_adapter", "serial_adapter"),
}
FORMS = ("dora", "ia3", "parallel_adapter", "serial_adapter")
RANKED = ("dora", "parallel_adapter", "serial_adapter")
SECTION = {"points", "arms", "readers", "inapplicable"}
_RANK = ("rank", "alpha")
_ARM = ("id", "points", "campaign")
MAX_ARMS = 12
# Familias neuronales con capas de normalización. RNN, LSTM, GRU y DLinear no tienen.
NORM_FAMILIES = ("transformer",)
# Variantes de Titans-MAC cuya atención lee la memoria persistente.
PERSISTENT_VARIANTS = ("mac_frozen", "mac_online")
# Puntos que faltan en alguna familia o variante y exigen un motivo en la matriz.
PARTIAL = ("norm", "persistent")
TITANS_FROZEN = ("mac.memory", "mac.persistent")
FUSION_BLOCK = "fusion"
_MIN_REASON = 20
# Ámbitos de la propuesta. Los de Titans-MAC llevan la variante tras el prefijo.
REFERENCES = "references"
READERS = "readers"
TITANS_SCOPE = "titans_mac:"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def is_variety(name, spec):
    """Punto de la sección: un subconjunto del padre o una forma nueva en la lectura o fusión."""
    return name in SELECTIVE_POINTS or (isinstance(spec, dict) and spec.get("form") in FORMS)


def _states(values):
    from .adapter_matrix import STATES

    return (
        isinstance(values, list)
        and values
        and len(set(values)) == len(values)
        and set(values) <= set(STATES)
    )


def check_spec(name, spec):
    """Validar la forma resuelta de un punto de la sección."""
    if name in SELECTIVE_POINTS:
        _require(
            isinstance(spec, dict)
            and set(spec) == {"form", "invalidates"}
            and spec["form"] == "residual"
            and _states(spec["invalidates"]),
            "Un subconjunto del padre se adapta con una corrección completa nula",
        )
        return spec
    _require(
        name in INSERTION_FORMS
        and isinstance(spec, dict)
        and spec.get("form") in INSERTION_FORMS[name]
        and set(spec) == {"form", "invalidates"} | (set(_RANK) if spec["form"] in RANKED else set())
        and _states(spec["invalidates"]),
        "La forma del punto no pertenece a la variedad declarada",
    )
    if spec["form"] in RANKED:
        _require(
            type(spec["rank"]) is int
            and 1 <= spec["rank"] <= 64
            and type(spec["alpha"]) in (int, float)
            and math.isfinite(spec["alpha"])
            and spec["alpha"] > 0,
            "El rango o la escala de la forma no son válidos",
        )
    return spec


def validate(section, matrix):
    """Comprobar la sección `variety` frente a los brazos y controles de la matriz."""
    from .adapter_matrix import CONTROLS

    _require(
        isinstance(section, dict) and set(section) == SECTION,
        "La variedad declara puntos, brazos, lectores y motivos de exclusión",
    )
    points = section["points"]
    _require(
        isinstance(points, dict) and tuple(points) == SELECTIVE_POINTS,
        "La variedad declara los sesgos, las normalizaciones y la memoria persistente",
    )
    for name, spec in points.items():
        check_spec(name, spec)
    base = {arm["id"] for arm in matrix["arms"]}
    arms = section["arms"]
    _require(isinstance(arms, list) and 1 <= len(arms) <= MAX_ARMS, "La variedad debe ser finita")
    seen = set()
    for arm in arms:
        _require(
            isinstance(arm, dict)
            and {"id", "points", "campaign"} <= set(arm) <= {*_ARM, "overrides"}
            and isinstance(arm["campaign"], list)
            and all(isinstance(scope, str) for scope in arm["campaign"])
            and len(set(arm["campaign"])) == len(arm["campaign"])
            and isinstance(arm["points"], list)
            and len(arm["points"]) == 1
            and arm["points"][0] in (*INSERTION_FORMS, *SELECTIVE_POINTS)
            and arm["id"] not in seen | base | set(CONTROLS),
            "Cada brazo de la variedad tiene un punto, una identidad nueva y su propuesta",
        )
        seen.add(arm["id"])
        (point,) = arm["points"]
        if point in SELECTIVE_POINTS:
            _require(
                "overrides" not in arm and arm["id"] == point,
                "Un subconjunto del padre se nombra por su punto y no cambia su forma",
            )
        else:
            overrides = arm.get("overrides")
            _require(
                isinstance(overrides, dict)
                and set(overrides) == {point}
                and arm["id"] == f"{point}_{overrides[point].get('form')}",
                "Un brazo de la variedad declara la forma de su punto y se nombra por ella",
            )
            check_spec(point, overrides[point])
    declared = {arm["points"][0] for arm in arms}
    reasons = section["inapplicable"]
    _require(
        isinstance(reasons, dict)
        and set(reasons) == {point for point in PARTIAL if point in declared}
        and all(
            isinstance(value, str) and len(value.strip()) >= _MIN_REASON
            for value in reasons.values()
        ),
        "Cada punto que falta en alguna familia necesita su motivo",
    )
    titans = {arm["id"] for arm in titans_arms(matrix, "mac_online", reserve=True, section=section)}
    readers = section["readers"]
    _require(
        isinstance(readers, list) and len(set(readers)) == len(readers) and set(readers) <= titans,
        "Los brazos del núcleo de los lectores deben existir en mac_online",
    )
    for arm in arms:
        for scope in arm["campaign"]:
            _require(
                _applies(matrix, arm, scope, readers),
                "Cada ámbito de la propuesta debe existir y contener el punto del brazo",
            )
    return section


def _applies(matrix, arm, scope, readers):
    """El ámbito existe y el punto del brazo existe en él."""
    (point,) = arm["points"]
    if scope == REFERENCES:
        return point != "persistent"
    if scope == READERS:
        return arm["id"] in readers
    variants = matrix["architectures"]["chronological"]["titans_mac"]["variants"]
    variant = scope.removeprefix(TITANS_SCOPE)
    return (
        scope.startswith(TITANS_SCOPE)
        and variant in variants
        and _in_variant(point, variant, variants[variant])
    )


def _in_variant(point, variant, allowed):
    """El punto existe en la variante: la lectura y la fusión según la matriz y la memoria
    persistente solo donde la atención de MAC la lee."""
    if point in INSERTION_FORMS:
        return point in allowed
    return point != "persistent" or variant in PERSISTENT_VARIANTS


def _section(matrix):
    return matrix.get("variety") if matrix.get("schema_version", 0) >= 3 else None


def _resolved(section, arm):
    (point,) = arm["points"]
    spec = arm["overrides"][point] if "overrides" in arm else section["points"][point]
    return {point: dict(spec)}


def _selected(section, reserve, scope):
    """Brazos propuestos en el ámbito o, con `reserve`, todos los implementados."""
    if section is None:
        return []
    return [arm for arm in section["arms"] if reserve or scope in arm["campaign"]]


def neural_arms(matrix, family, *, reserve=False):
    """Brazos de la variedad aplicables a una referencia neuronal, en el orden de la matriz."""
    from .adapter_matrix import READOUT_FAMILIES

    section = _section(matrix)
    result = []
    for arm in _selected(section, reserve, REFERENCES):
        (point,) = arm["points"]
        if (
            (point == "readout" and family not in READOUT_FAMILIES)
            or (point == "norm" and family not in NORM_FAMILIES)
            or point == "persistent"
        ):
            continue
        result.append(dict(id=arm["id"], points=_resolved(section, arm)))
    return result


def titans_arms(matrix, variant, *, reserve=False, section=None):
    """Brazos de la variedad aplicables a una variante de Titans-MAC."""
    section = section if section is not None else _section(matrix)
    allowed = matrix["architectures"]["chronological"]["titans_mac"]["variants"][variant]
    return [
        dict(id=arm["id"], points=_resolved(section, arm))
        for arm in _selected(section, reserve, TITANS_SCOPE + variant)
        if _in_variant(arm["points"][0], variant, allowed)
    ]


def reader_arms(matrix, *, reserve=False):
    """Brazos del núcleo `mac_online` de MARS-TITAN y CM-v1 que declara la variedad."""
    section = _section(matrix)
    if section is None:
        return []
    chosen = {arm["id"]: arm for arm in _selected(section, reserve, READERS)}
    return [
        dict(id=f"core_{name}", core=_resolved(section, chosen[name]), episodic_readout=None)
        for name in section["readers"]
        if name in chosen
    ]


def _excluded(path, frozen):
    return any(path == item or path.startswith(item + ".") for item in frozen)


def _owned(model, frozen):
    """Módulos propios del padre, sin los congelados, las parametrizaciones ni los adaptadores."""
    for name, module in model.named_modules():
        parts = name.split(".")
        if _excluded(name, frozen) or "parametrizations" in parts or parts[0] == MODULE_ROOT:
            continue
        yield name, module


def _is_bias(name):
    return name == "bias" or name.endswith("_bias") or name.startswith("bias_")


def bias_targets(model, frozen=()):
    """Todos los sesgos del padre fuera de los módulos congelados (BitFit)."""
    result = [
        AdapterTarget(module_name, tensor, "residual")
        for module_name, module in _owned(model, frozen)
        for tensor, value in module.named_parameters(recurse=False)
        if _is_bias(tensor) and value.ndim == 1
    ]
    _require(result, "El padre no tiene sesgos que adaptar")
    return result


def norm_targets(model, frozen=()):
    """Ganancia y sesgo de cada LayerNorm del padre fuera de los módulos congelados."""
    result = []
    for module_name, module in _owned(model, frozen):
        if isinstance(module, nn.LayerNorm) and module.elementwise_affine:
            for tensor in ("weight", "bias"):
                if getattr(module, tensor, None) is not None:
                    result.append(AdapterTarget(module_name, tensor, "residual"))
    _require(result, "El padre no tiene normalizaciones que adaptar")
    return result


def _ranked(spec):
    return dict(rank=spec["rank"], alpha=float(spec["alpha"]))


def _fusion(spec, modules):
    form = spec["form"]
    if form in ("parallel_adapter", "serial_adapter"):
        return [AdapterTarget(FUSION_BLOCK, "output", form, **_ranked(spec))]
    if form == "dora":
        return [AdapterTarget(module, "weight", "dora", **_ranked(spec)) for module in modules]
    # (IA)³: la ganancia por columnas reescala la activación de cada codificador antes de la
    # capa lineal de la fusión, como l_ff antes de W_2 en el artículo.
    return [AdapterTarget(module, "weight", "gain_columns") for module in modules]


def neural_targets(model, name, spec):
    """Tensores de una referencia neuronal (`MultimodalReference`) para un punto de la variedad."""
    check_spec(name, spec)
    if name == "bias":
        return bias_targets(model)
    if name == "norm":
        return norm_targets(model)
    _require(name != "persistent", "Las referencias neuronales no tienen memoria persistente")
    if name == "fusion":
        return _fusion(spec, ["fusion.0"])
    hidden = model.architecture["hidden_size"]
    result = []
    for layer in range(len(model.price_encoder.blocks)):
        prefix = f"price_encoder.blocks.{layer}.self_attn"
        attention = model.get_submodule(prefix)
        if spec["form"] == "dora":
            result += [
                AdapterTarget(prefix, "in_proj_weight", "dora", rows=(0, hidden), **_ranked(spec)),
                AdapterTarget(f"{prefix}.out_proj", "weight", "dora", **_ranked(spec)),
            ]
            continue
        # (IA)³: la ganancia de la consulta (filas y sesgo) equivale a la de las claves en
        # el producto escalar, y la de las columnas de out_proj a la de los valores.
        result.append(AdapterTarget(prefix, "in_proj_weight", "gain_rows", rows=(0, hidden)))
        if attention.in_proj_bias is not None:
            result.append(
                AdapterTarget(
                    prefix, "in_proj_bias", "gain_rows", rows=(0, hidden), shares="in_proj_weight"
                )
            )
        result.append(AdapterTarget(f"{prefix}.out_proj", "weight", "gain_columns"))
    return result


def titans_targets(matrix, name, spec, model):
    """Tensores de Titans-MAC para un punto de la variedad.

    La memoria neuronal (pesos rápidos y sus proyecciones) nunca se adapta. La memoria
    persistente solo en su propio brazo, donde es el único destino.
    """
    check_spec(name, spec)
    if name == "persistent":
        return [AdapterTarget("mac", "persistent", "residual")]
    if name in ("bias", "norm"):
        _require(model is not None, "Los subconjuntos del padre se enumeran sobre el modelo")
        select = bias_targets if name == "bias" else norm_targets
        return select(model, TITANS_FROZEN)
    modules = matrix["architectures"]["chronological"]["titans_mac"]["targets"][name]
    if name == "fusion":
        return _fusion(spec, modules)
    if spec["form"] == "dora":
        return [AdapterTarget(module, "weight", "dora", **_ranked(spec)) for module in modules]
    # Consulta de la memoria por filas y `out_proj` de la atención por columnas. La atención
    # de MAC no tiene sesgos.
    return [
        AdapterTarget(
            module, "weight", "gain_columns" if module.endswith("out_proj") else "gain_rows"
        )
        for module in modules
    ]
