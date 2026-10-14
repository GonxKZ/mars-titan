"""Control de continuación completa con el decaimiento anclado al padre (#444).

Con el mismo λ, AdamW lleva la corrección de un adaptador hacia cero, es decir, el peso
efectivo hacia el padre, pero contrae hacia cero los pesos de la continuación completa.
Con el presupuesto de la matriz v3 (η = 1e-4, λ = 0,01) son entre un 0,33 % y un 0,60 % del
padre a lo largo del ajuste. La comparación entre adaptadores y continuación mezcla así el
número de parámetros ajustados con el destino del decaimiento.

La sección opcional `anchored_continuation` de la matriz de versión 3 declara un control
con su propia identidad: la continuación completa con el mismo objetivo, presupuesto,
λ y selección, cuyo decaimiento lleva cada peso a su valor en el padre
(`training/anchored_decay.py`, forma desacoplada de L2-SP). `campaign` enumera los ámbitos
donde se propone para la campaña, con el vocabulario de la variedad de adaptadores (las
referencias neuronales, cada variante de Titans-MAC y los lectores) más la GRU candidata.
Fuera de ellos queda implementado como reserva.

Una etapa solo ejecuta el control si lo nombra en `additional_controls`. Así una matriz
compartida por varias etapas no cambia el plan de las que todavía no lo han declarado.
"""

from mars_titan.training.anchored_decay import INITIAL

from . import adapter_variety

CONTROL = "anchored_continuation"
# Clave del caso de las referencias que declara el ancla del decaimiento.
ANCHOR = "weight_decay_anchor"
SECTION = {"anchor", "campaign"}
CANDIDATE = "episodic_gru"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _scopes(matrix):
    """Ámbitos posibles de la propuesta en la matriz."""
    variants = matrix["architectures"]["chronological"]["titans_mac"]["variants"]
    return {
        adapter_variety.REFERENCES,
        adapter_variety.READERS,
        CANDIDATE,
        *(adapter_variety.TITANS_SCOPE + variant for variant in variants),
    }


def validate(section, matrix):
    """Comprobar la sección frente a la matriz de versión 3 y su presupuesto."""
    _require(
        isinstance(section, dict)
        and set(section) == SECTION
        and isinstance(section["anchor"], str)
        and section["anchor"] == INITIAL
        and isinstance(section["campaign"], list)
        and all(isinstance(scope, str) for scope in section["campaign"])
        and len(set(section["campaign"])) == len(section["campaign"])
        and set(section["campaign"]) <= _scopes(matrix),
        "La continuación anclada declara el ancla en el padre y ámbitos existentes",
    )
    decay = matrix["budget"]["weight_decay"]
    _require(
        type(decay) in (int, float) and decay > 0,
        "Con λ = 0 la continuación anclada coincide con la continuación completa",
    )
    return section


def declared(matrix):
    """Sección del control en la matriz, o None si no la declara."""
    return matrix.get(CONTROL) if matrix["schema_version"] >= 3 else None


def proposed(matrix, scope, *, reserve=False):
    """Si la matriz declara el control y lo propone en el ámbito, o en todos con `reserve`."""
    section = declared(matrix)
    return section is not None and (reserve or scope in section["campaign"])


def chronological_scope(kind, variant):
    """Ámbito de un diseño cronológico de la matriz: variante de Titans-MAC, lector o GRU."""
    from .chronological_matrix import CANDIDATE as CANDIDATE_DESIGN
    from .chronological_matrix import READOUT, TITANS, TITANS_VARIANTS

    if kind == TITANS:
        _require(variant in TITANS_VARIANTS, "La variante no pertenece a Titans-MAC")
        return adapter_variety.TITANS_SCOPE + variant
    # La variante de un lector o de la GRU la rechaza después `chronological_matrix.arms`.
    return {READOUT: adapter_variety.READERS, CANDIDATE_DESIGN: CANDIDATE}[kind]


def stage_controls(value, matrix):
    """Controles adicionales que ejecuta una etapa. Deben existir en su matriz."""
    _require(
        isinstance(value, list)
        and value
        and len(set(value)) == len(value)
        and all(control == CONTROL and declared(matrix) is not None for control in value),
        "La etapa solo añade controles que su matriz declara",
    )
    return value
