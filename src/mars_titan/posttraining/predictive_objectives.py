"""Objetivos de adaptación predictiva de #128 como controles de la etapa de adaptadores.

#128 compara, con la misma capacidad, población, ponderación e inicialización, cómo se
ajusta una corrección lineal residual sobre la predicción de un padre congelado: REINFORCE
con baseline de información completa, la pérdida esperada exacta sobre las 21 acciones, el
MAE del centro continuo y las tres variantes de KLPO (`docs/research/predictive-adaptation.md`).
La sección opcional `predictive_objectives` de la matriz de versión 3 los declara como
controles de la etapa, uno por objetivo, con el presupuesto, la selección y las
actualizaciones de la continuación completa. Una etapa solo los ejecuta si nombra la sección
en `additional_controls`.

Los padres de la campaña emiten cinco cuantiles, pero la corrección lineal corrige un único
valor, la mediana guardada en la caché del padre, y su salida es la mediana de la política,
un pronóstico puntual. Por eso estos controles son solo contrastes: no compiten en la cadena,
que da a las políticas los cinco niveles de su predictor, y la comparación los evalúa como
salida puntual con el MAE por sesión. Es el motivo por el que la matriz excluye la corrección
lineal como control de los padres de cuantiles, y no cambia.

Quedan fuera, con su motivo y la condición que reabriría la decisión en `excluded`:

- `neural_mae` y `neural_mse`. Un padre de cuantiles se continúa con su pinball
  (`full_continuation`), y `run_case` rechaza las dos desde #397.
- Los padres cronológicos (Titans-MAC, los lectores de MARS-TITAN y CM-v1 y la GRU
  candidata). La corrección lee la caché de `MatrixParent`, que solo carga las cinco redes
  de referencia.

Un objetivo que ya ejecuta la corrección lineal residual de la cabeza (el MAE de un padre
escalar) no se repite con otro nombre.
"""

from mars_titan.models.klpo import MODES as KLPO

from . import adapter_variety

SECTION_NAME = "predictive_objectives"
# Los seis objetivos lineales de los diseños de #128, en su orden.
MODES = ("reinforce", "expected", "mae", *KLPO)
PREFIX = "objective_"
CONTROLS = tuple(PREFIX + mode for mode in MODES)
SECTION = {"modes", "campaign", "excluded"}
EXCLUDED = ("neural_mae", "neural_mse", "chronological_parents")
REASON = {"reason", "revisit_if"}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _text(value, minimum):
    return isinstance(value, str) and len(value.strip()) >= minimum


def validate(section):
    """Comprobar la sección de la matriz de versión 3."""
    _require(
        isinstance(section, dict)
        and set(section) == SECTION
        and section["modes"] == list(MODES)
        and isinstance(section["campaign"], list)
        and len(set(section["campaign"])) == len(section["campaign"])
        and set(section["campaign"]) <= {adapter_variety.REFERENCES},
        "Los objetivos de #128 declaran los seis objetivos lineales y solo se proponen para "
        "las referencias",
    )
    excluded = section["excluded"]
    _require(
        isinstance(excluded, dict)
        and tuple(excluded) == EXCLUDED
        and all(
            isinstance(item, dict)
            and set(item) == REASON
            and _text(item["reason"], 40)
            and _text(item["revisit_if"], 20)
            for item in excluded.values()
        ),
        "Cada objetivo o padre que queda fuera declara su motivo y cuándo se revisaría",
    )
    return section


def declared(matrix):
    """Sección de los objetivos en la matriz, o None si no la declara."""
    return matrix.get(SECTION_NAME) if matrix["schema_version"] >= 3 else None


def proposed(matrix, scope, *, reserve=False):
    """Si la matriz declara los objetivos y los propone en el ámbito, o en todos con `reserve`."""
    section = declared(matrix)
    return section is not None and (reserve or scope in section["campaign"])


def control(mode):
    """Control de la matriz que ajusta la corrección lineal con el objetivo `mode`."""
    _require(mode in MODES, "El objetivo no pertenece a los diseños de #128")
    return PREFIX + mode


def modes(matrix, linear_residual):
    """Objetivos que añade la sección para una cabeza, sin repetir su corrección lineal.

    `linear_residual` es el objetivo de la corrección lineal de la cabeza, o None si la
    matriz la excluye, como en los padres de cuantiles.
    """
    section = declared(matrix)
    return [mode for mode in section["modes"] if mode != linear_residual]
