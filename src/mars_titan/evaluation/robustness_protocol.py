"""Declaración de robustez de la campaña desde 2000, comprobada contra el código (#37).

La declaración (`configs/evaluation/historical-masked-2000-robustness.json`) se fija antes de
evaluar ninguna variante y reúne tres cosas:

- las sensibilidades que ya tienen consumidor (costes, ausencia de modalidades, periodos y
  abstención), que solo se citan con sus archivos y campos;
- las perturbaciones de entradas de `data.input_perturbations`, con su semilla, intensidades
  y generador;
- las guardas del lector: lo que se rechaza antes de llegar al modelo, con la prueba que lo
  demuestra, y lo que la lectura no puede detectar.

`load_declaration` exige que todo eso coincida con el repositorio. Una intensidad, una semilla
o una prueba cambiadas en el código sin cambiar la declaración, o al revés, la invalidan.
"""

from pathlib import Path

from mars_titan.data.cohort_files import read_manifest
from mars_titan.data.input_perturbations import PERTURBATIONS, RANDOM, SEED

KIND = "historical_masked_robustness"
SENSITIVITIES = ("costs", "modality_absence", "periods", "abstention")
ROOT = Path(__file__).resolve().parents[3]


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _field(document, dotted):
    for part in dotted.split("."):
        if not isinstance(document, dict) or part not in document:
            return None, False
        document = document[part]
    return document, True


def _registered(name, entry, root):
    _require(
        isinstance(entry, dict)
        and isinstance(entry.get("sources"), list)
        and isinstance(entry.get("fields"), list),
        f"La sensibilidad {name} necesita fuentes y campos",
    )
    if not entry["sources"]:
        _require(
            not entry["fields"] and isinstance(entry.get("pending"), str) and entry["pending"],
            f"La sensibilidad {name} sin fuente debe decir qué falta",
        )
        return
    documents = []
    for source in entry["sources"]:
        path = root / source
        _require(path.is_file(), f"La fuente {source} de {name} no existe")
        documents.append(read_manifest(path, 8 * 1024**2)[0])
    for dotted in entry["fields"]:
        found = [value for value, present in (_field(d, dotted) for d in documents) if present]
        _require(found, f"El campo {dotted} de {name} no está en sus fuentes")
        if "values_bps" in entry:
            _require(
                all(value == entry["values_bps"] for value in found),
                f"Los costes de {dotted} no son los declarados",
            )


def _perturbations(section):
    declared = {
        name: {key: list(value) if key == "modalities" else value for key, value in spec.items()}
        for name, spec in PERTURBATIONS.items()
    }
    _require(
        isinstance(section, dict)
        and section.get("seed") == SEED
        and section.get("random") == RANDOM
        and section.get("variants") == declared,
        "Las perturbaciones declaradas no son las del código",
    )


def _guards(section, root):
    _require(isinstance(section, dict) and section, "Faltan las guardas del lector")
    for name, guard in section.items():
        _require(isinstance(guard, dict), f"La guarda {name} no es un objeto")
        if guard.get("expected") == "not_detectable_by_the_reader":
            _require(
                isinstance(guard.get("reason"), str) and guard["reason"],
                f"La guarda {name} debe explicar por qué no se detecta",
            )
            continue
        reference = guard.get("test")
        _require(isinstance(reference, str) and "::" in reference, f"{name} no cita su prueba")
        path, test = reference.split("::", 1)
        _require(
            (root / path).is_file() and f"def {test}(" in (root / path).read_text(),
            f"La prueba de la guarda {name} no existe",
        )


def load_declaration(path, root=ROOT):
    """Leer la declaración y exigir que coincida con el repositorio de `root`."""
    declaration, digest = read_manifest(Path(path), 1024**2)
    _require(
        isinstance(declaration, dict)
        and declaration.get("schema_version") == 1
        and declaration.get("kind") == KIND
        and declaration.get("final_test_opened") is False
        and declaration.get("executions") == 0
        and declaration.get("partition") == "evaluation",
        "La declaración de robustez no es válida o abre la reserva",
    )
    registered = declaration.get("registered")
    _require(
        isinstance(registered, dict) and tuple(registered) == SENSITIVITIES,
        "La declaración debe citar costes, ausencia de modalidades, periodos y abstención",
    )
    for name, entry in registered.items():
        _registered(name, entry, Path(root))
    _perturbations(declaration.get("input_perturbations"))
    _guards(declaration.get("reader_guards"), Path(root))
    return declaration, digest
