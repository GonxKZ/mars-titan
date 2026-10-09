"""Política de datos de la campaña: solo la edición real verificada, sin datos sintéticos.

Por orden del autor del 9 de octubre de 2026, nada sintético en ninguna etapa, tampoco en el
entorno de RL ni en el postentrenamiento: todo aprende con datos reales del dataset y con la
división walk-forward de toda la serie. La campaña v2 lo declara con
`data_policy="real_edition_only"` y `plan_campaign` lo comprueba antes de enumerar trabajos:

- la campaña lee las vistas de la política histórica con máscaras, y cada documento declarado
  que fija una política de entradas usa esa misma,
- ningún documento declarado por la campaña ni por sus etapas registradas (configuración,
  configuración tabular, recetas, declaración de CM-v1, matriz de adaptadores, ablación y
  políticas de RL) contiene una clave o un valor de condiciones sintéticas, remuestreadas o
  de aumento, ni una fuente distinta de las vistas o las cintas reconstruidas de la edición,
- XGBoost no remuestrea filas ni columnas.

La comparación queda fuera del examen porque no entrena: su remuestreo por bloques de días
solo mide la incertidumbre de las métricas. Las garantías durante la ejecución de las etapas
de adaptadores y de políticas están en sus propios módulos. Las pruebas con datos de juguete
no entrenan modelos de la campaña y no se ven afectadas.
"""

import json
import re
from pathlib import Path

from mars_titan.data.input_policy import HISTORICAL_MASKED

REAL_EDITION_ONLY = "real_edition_only"
# Condiciones sintéticas, remuestreadas o de aumento, en inglés y en español.
FORBIDDEN = re.compile(
    r"synthetic|sint[eé]tic|augment|aumento de datos|resampl|remuestre|bootstrap|mixup|"
    r"jitter|noise|ruido|hmm|markov|scenario|escenario|fixture|toy_|generated_data",
    re.IGNORECASE,
)
# Claves que nombran una fuente de datos y los únicos valores admitidos.
SOURCE_KEYS = {"source", "sources", "data_source", "dataset", "corpus", "tapes"}
REAL_SOURCES = {"edition_views", "reconstructed_tapes"}
# Textos que explican una decisión o lo que falta. Se examinan sus claves, no su redacción.
DESCRIPTIVE = {"pending", "pending_arms", "description", "notes", "rationale", "excluded"}
REPOSITORY = Path(__file__).resolve().parents[3]


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def declared(value):
    """Validar la política declarada en una campaña v2."""
    _require(
        value == REAL_EDITION_ONLY,
        f"La campaña declara data_policy={REAL_EDITION_ONLY}: solo datos reales de la edición",
    )
    return value


def findings(document, policy, label):
    """Claves, valores y políticas de entradas de un documento que la política no admite."""
    found = []

    def walk(value, path, prose=False):
        if isinstance(value, dict):
            for key, item in value.items():
                where = f"{path}.{key}"
                if FORBIDDEN.search(str(key)):
                    found.append(f"{label}: clave {where}")
                if key == "input_policy" and item != policy:
                    found.append(f"{label}: {where}={item!r} no es la política de la campaña")
                if key in SOURCE_KEYS and isinstance(item, str) and item not in REAL_SOURCES:
                    found.append(f"{label}: fuente {where}={item!r}")
                walk(item, where, prose or key in DESCRIPTIVE)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                walk(item, f"{path}[{index}]", prose)
        elif isinstance(value, str) and not prose and FORBIDDEN.search(value):
            found.append(f"{label}: valor {path}={value[:80]!r}")

    walk(document, "")
    return found


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def documents(campaign, later_stages):
    """Documentos que la campaña y sus etapas registradas declaran, con su ruta."""
    paths = [Path(campaign["path"]), Path(campaign["tabular"]["path"])]
    for family in ("titans_mac", "episodic_gru", "mars_titan"):
        section = campaign.get(family)
        if section:
            paths.append(Path(section["path"]))
    if campaign.get("cm_v1"):
        section = campaign["cm_v1"]
        paths += [Path(section["path"]), *map(Path, section["recipes"].values())]
    for entry in later_stages.values():
        for relative in (*entry["stages"].values(), entry.get("joint_stage")):
            if relative is None:
                continue
            stage_path = REPOSITORY / relative
            stage = _read(stage_path)
            target = (stage_path.parent / stage["campaign"]).resolve()
            if target != Path(campaign["path"]).resolve():
                continue
            paths.append(stage_path)
            for key in ("matrix", "policies"):
                if key in stage:
                    paths.append(stage_path.parent / stage[key])
    unique = []
    for path in paths:
        if path.resolve() not in {p.resolve() for p in unique}:
            unique.append(path)
    return unique


def check(campaign, later_stages):
    """Exigir la política de datos declarada en la campaña y en sus etapas registradas."""
    from mars_titan.models.baselines.external_boosting import ROW_SAMPLING

    declared(campaign["data_policy"])
    policy = campaign["input_policy"]
    _require(
        policy == HISTORICAL_MASKED and campaign["comparison_config"]["input_policy"] == policy,
        "La campaña solo lee las vistas de la edición histórica con máscaras",
    )
    found = []
    for path in documents(campaign, later_stages):
        found += findings(_read(path), policy, path.name)
    if ROW_SAMPLING != dict(subsample=1.0, colsample_bytree=1.0):
        found.append(f"XGBoost remuestrea filas o columnas: {ROW_SAMPLING}")
    _require(
        not found,
        "La política real_edition_only no admite condiciones sintéticas, remuestreadas o de "
        "aumento ni otras fuentes: " + "; ".join(found),
    )
