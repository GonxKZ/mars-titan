"""Configuración guardable del diagnóstico B4, separada del modelo y sus recetas."""

from dataclasses import dataclass
from pathlib import Path

from mars_titan.data.cohort_files import read_manifest, safe_destination
from mars_titan.data.storage import atomic_json, sha256

from .lesion_protocol import LesionProtocol

CONFIG_KIND = "lesion_study_v1"


def _integer(value, lower, upper):
    return type(value) is int and lower <= value <= upper


def _analysis(value):
    if not isinstance(value, dict) or set(value) != {
        "block_length",
        "replicates",
        "seed",
        "confidence",
        "market_weighting",
        "sensitivity_block_lengths",
        "rank_ic_min_assets",
    }:
        raise ValueError("El análisis B4 necesita sus siete opciones declaradas")
    for key, lower, upper in (
        ("block_length", 1, 100_000),
        ("replicates", 10, 100_000),
        ("seed", 0, 2**63 - 1),
        ("rank_ic_min_assets", 3, 8192),
    ):
        if not _integer(value[key], lower, upper):
            raise ValueError(f"El parámetro B4 {key} no es válido")
    lengths = value["sensitivity_block_lengths"]
    if (
        type(value["confidence"]) is not float
        or not 0.5 <= value["confidence"] < 1
        or value["market_weighting"] not in ("session", "market")
        or not isinstance(lengths, list)
        or len(lengths) > 16
        or any(not _integer(length, 1, 100_000) for length in lengths)
        or len(set(lengths)) != len(lengths)
        or value["block_length"] in lengths
    ):
        raise ValueError("La confianza, ponderación o sensibilidad B4 no son válidas")
    return dict(value, sensitivity_block_lengths=list(lengths))


@dataclass(frozen=True)
class LesionStudy:
    """Protocolo y análisis. Sus opciones no eligen máscaras a partir de errores."""

    protocol: LesionProtocol
    analysis: dict


def load_study(path):
    """Leer hasta 64 KiB y rechazar opciones desconocidas o activación del test final."""
    document, digest = read_manifest(Path(path), maximum=64 * 1024)
    if (
        not isinstance(document, dict)
        or set(document)
        != {
            "schema_version",
            "kind",
            "variant_id",
            "status",
            "default_enabled",
            "final_test_access",
            "protocol",
            "analysis",
        }
        or type(document["schema_version"]) is not int
        or document["schema_version"] != 1
        or document["kind"] != CONFIG_KIND
        or document["variant_id"] != "lesion_protocol_v1"
        or document["status"] != "declared_not_executed"
        or document["default_enabled"] is not False
        or document["final_test_access"] is not False
    ):
        raise ValueError("La configuración B4 debe seguir independiente y desactivada")
    return LesionStudy(
        LesionProtocol.from_identity(document["protocol"]), _analysis(document["analysis"])
    ), digest


def select_arm(path, arm):
    """Reutilizar una variante de corrección existente. La ventana valida sus componentes."""
    document, digest = read_manifest(Path(path), maximum=256 * 1024)
    if (
        not isinstance(document, dict)
        or document.get("kind") != "mature_correction_variants"
        or document.get("default_enabled") is not False
        or document.get("final_test_access") is not False
        or not isinstance(document.get("arms"), dict)
        or arm not in document["arms"]
    ):
        raise ValueError("El estudio no contiene esa variante independiente de corrección")
    value = document["arms"][arm]
    if (
        not isinstance(value, dict)
        or set(value) != {"recipe", "components"}
        or not isinstance(value["recipe"], str)
        or not value["recipe"]
        or not isinstance(value["components"], dict)
    ):
        raise ValueError("La variante necesita una receta y sus componentes")
    return Path(value["recipe"]), value["components"], digest


def freeze_study(output, *, study, study_sha256, arms_sha256, arm, seed, search_case, recipe):
    """Fijar el análisis antes de ejecutar. Una reanudación no cambia esta declaración."""
    output = Path(output)
    safe_destination(output)
    if not _integer(seed, 0, 2**32 - 1):
        raise ValueError("La semilla de la ventana B4 no es válida")
    declaration = dict(
        schema_version=1,
        kind=CONFIG_KIND,
        study_sha256=study_sha256,
        protocol=study.protocol.identity(),
        analysis=study.analysis,
        arms_sha256=arms_sha256,
        arm=arm,
        seed=seed,
        search_case=search_case,
        recipe_sha256=sha256(Path(recipe)),
    )
    path = output / "lesion-study.json"
    if path.exists():
        previous, _ = read_manifest(path, maximum=64 * 1024)
        if previous != declaration:
            raise ValueError("La declaración B4 de esta ventana ha cambiado")
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError("No se añade una declaración retrospectiva a otra ventana")
        output.mkdir(parents=True, exist_ok=True)
        atomic_json(path, declaration)
