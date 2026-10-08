"""Catálogo identificado de escenarios existentes, sin ajuste ni evaluación de modelos."""

import hashlib
import json
import platform
import re
from importlib.metadata import version
from pathlib import Path

from mars_titan.data.storage import atomic_json, sha256

from . import adaptation_scenarios

MAX_PRICE_ROWS = 1_048_576


def _fields(value, expected, name):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError(f"Campos no admitidos en {name}")


def _identifier(value):
    if not isinstance(value, str) or re.fullmatch(r"[a-z][a-z0-9_]{0,63}", value) is None:
        raise ValueError("La identidad debe ser breve y no puede contener rutas")


def _profile_settings(settings, profile):
    return dict(
        schema_version=1,
        families=[item["family"] for item in settings["benchmarks"]],
        **settings["profiles"][profile],
        final_test_opened=False,
    )


def _benchmark_entries(entries):
    if not isinstance(entries, list) or not 1 <= len(entries) <= len(adaptation_scenarios.FAMILIES):
        raise ValueError("El catálogo necesita entre uno y ocho benchmarks")
    unique = {key: set() for key in ("id", "name", "family")}
    for item in entries:
        _fields(item, unique, "el benchmark")
        _identifier(item["id"])
        name = item["name"]
        if (
            not isinstance(name, str)
            or not 1 <= len(name) <= 80
            or not name.isprintable()
            or name != name.strip()
        ):
            raise ValueError("El nombre del benchmark debe ser un texto breve sin controles")
        if (
            not isinstance(item["family"], str)
            or item["family"] not in adaptation_scenarios.FAMILIES
        ):
            raise ValueError("Familia de escenarios desconocida")
        for key in unique:
            if item[key] in unique[key]:
                raise ValueError("El catálogo repite identidades, nombres o familias")
            unique[key].add(item[key])
    return {item["family"]: item for item in entries}


def _planned_cases(settings, by_family):
    profiles = settings["profiles"]
    if not isinstance(profiles, dict) or not 1 <= len(profiles) <= 4:
        raise ValueError("El catálogo admite entre uno y cuatro perfiles")
    planned, seeds, price_rows = [], set(), 0
    for profile in profiles:
        _identifier(profile)
    for profile in sorted(profiles):
        config = profiles[profile]
        _fields(config, ("generator", "worlds", "seed_base"), "el perfil")
        _fields(
            config["generator"],
            ("assets", "sessions", "warmup_sessions", "regime_sessions"),
            "el generador",
        )
        _fields(config["worlds"], adaptation_scenarios.SPLITS, "las particiones")
        cases = adaptation_scenarios.cases(_profile_settings(settings, profile))
        generator = config["generator"]
        if generator["sessions"] < generator["warmup_sessions"] + 3 * generator["regime_sessions"]:
            raise ValueError(
                "Cada perfil debe contener al menos tres periodos tras el calentamiento"
            )
        for case in cases:
            if case["seed"] in seeds:
                raise ValueError("Las semillas se solapan entre perfiles o particiones")
            seeds.add(case["seed"])
            price_rows += case["assets"] * case["sessions"]
            if price_rows > MAX_PRICE_ROWS:
                raise ValueError(
                    "El volumen del catálogo supera el presupuesto de filas de precios"
                )
            entry = by_family[case["family"]]
            planned.append(
                dict(
                    case,
                    benchmark_id=entry["id"],
                    name=entry["name"],
                    profile=profile,
                    evaluator_only=case["split"] == "audit",
                )
            )
    return planned, price_rows


def benchmark_catalog(settings):
    """Validar el catálogo completo antes de generar un perfil o crear su destino."""
    _fields(
        settings,
        ("schema_version", "suite_id", "benchmarks", "profiles", "final_test_opened"),
        "el catálogo",
    )
    if type(settings["schema_version"]) is not int or settings["schema_version"] != 1:
        raise ValueError("Versión de catálogo no admitida")
    if settings["final_test_opened"] is not False:
        raise ValueError("El test final debe permanecer cerrado")
    _identifier(settings["suite_id"])
    by_family = _benchmark_entries(settings["benchmarks"])
    planned, price_rows = _planned_cases(settings, by_family)
    root = Path(__file__).parents[1]
    sources = {
        name: sha256(root / name)
        for name in (
            "simulation/benchmark_catalog.py",
            "simulation/adaptation_scenarios.py",
            "simulation/market.py",
            "simulation/portfolio.py",
            "simulation/storage.py",
            "data/storage.py",
            "data/cohort_files.py",
        )
    }
    identity = dict(
        settings=settings,
        sources=sources,
        runtime={
            "python": platform.python_version(),
            "numpy": version("numpy"),
            "pyarrow": version("pyarrow"),
        },
    )
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return dict(
        schema_version=1,
        suite_id=settings["suite_id"],
        identity_sha256=hashlib.sha256(encoded).hexdigest(),
        identity=json.loads(encoded),
        total_price_rows=price_rows,
        cases=planned,
        scientific_evaluation_executed=False,
        difficulty_empirically_measured=False,
        final_test_opened=False,
    )


def prepare_benchmark_profile(settings, profile, output):
    """Preparar un perfil con el generador original y enlazar su índice sin modificarlo."""
    catalog = benchmark_catalog(settings)
    _identifier(profile)
    if profile not in settings["profiles"]:
        raise ValueError("El perfil no pertenece al catálogo")
    # Usar la copia validada impide cambios posteriores del diccionario original.
    settings = catalog["identity"]["settings"]
    generated = adaptation_scenarios.prepare_adaptation_scenarios(
        _profile_settings(settings, profile), output, fit_markov=False
    )
    by_family = {item["family"]: item for item in settings["benchmarks"]}
    records = [
        dict(
            record,
            benchmark_id=by_family[record["family"]]["id"],
            benchmark_name=by_family[record["family"]]["name"],
            profile=profile,
        )
        for record in generated["records"]
    ]
    report = dict(
        schema_version=1,
        suite_id=settings["suite_id"],
        profile=profile,
        status="prepared",
        identity_sha256=catalog["identity_sha256"],
        identity=catalog["identity"],
        source_index="index.json",
        source_index_sha256=sha256(Path(output) / "index.json"),
        records=records,
        scientific_evaluation_executed=False,
        difficulty_empirically_measured=False,
        final_test_opened=False,
    )
    atomic_json(Path(output) / "benchmarks.json", report)
    return report
