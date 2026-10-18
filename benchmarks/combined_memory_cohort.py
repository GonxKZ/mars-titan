"""Cohortes de desarrollo por cobertura de entradas anterior a cada corte de ajuste."""

import copy
import hashlib
import math
from collections import defaultdict

SALT = "mars-titan-combined-20261018"
YEARS = {"fold-015": 2020, "fold-018": 2023}


def stratum(record):
    fractions, count = record["presence_fraction"], record["train_rows"]
    if (
        not isinstance(fractions, list)
        or len(fractions) != 5
        or any(
            type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1
            for v in fractions
        )
        or type(count) is not int
        or count <= 0
    ):
        raise ValueError("La auditoría no conserva sus fracciones y filas de ajuste")
    history = "long" if count >= 2520 else "short"
    return f"n{int(fractions[1] >= 0.05)}f{int(fractions[3] >= 0.05)}_{history}"


def choose_assets(source, audit_rows, window, count):
    """Equilibrar los estratos por rondas y ordenar cada uno por una huella fija."""
    if window not in YEARS or type(count) is not int or not 1 <= count <= 64:
        raise ValueError("La ventana y el número de activos no están admitidos")
    assets = [a for a in source["assets"] if a["market"] == "US" and a["counts"]["train"] > 0]
    by_symbol = {a["symbol"]: a for a in assets}
    if len(by_symbol) != len(assets):
        raise ValueError("Hay activos duplicados en el manifiesto")
    symbols = [r["symbol"] for r in audit_rows]
    if len(set(symbols)) != len(symbols):
        raise ValueError("Hay activos duplicados en la auditoría")
    groups = defaultdict(list)
    for row in audit_rows:
        symbol = row["symbol"]
        if window not in row["folds"] or symbol not in by_symbol:
            continue
        record = row["folds"][window]
        name = stratum(record)
        if (
            name != record["stratum"]
            or record["train_rows"] != by_symbol[symbol]["counts"]["train"]
        ):
            raise ValueError("El estrato o las filas no coinciden con la auditoría")
        groups[name].append(symbol)
    if sum(map(len, groups.values())) != len(assets):
        raise ValueError("La auditoría no cubre todos los activos elegibles")
    if count > len(assets):
        raise ValueError("No hay suficientes activos en la cohorte")
    for values in groups.values():
        values.sort(
            key=lambda symbol: hashlib.sha256(f"{SALT}/{window}/{symbol}".encode()).digest()
        )
    selected, position = [], 0
    while len(selected) < count:
        for name in sorted(groups):
            if position < len(groups[name]) and len(selected) < count:
                selected.append(groups[name][position])
        position += 1
    return [by_symbol[symbol] for symbol in sorted(selected)]


def project_view(source, assets, window, epochs):
    """Conservar las filas y etiquetas de los activos elegidos, sin rehacer sus objetivos."""
    if window not in YEARS or type(epochs) is not int or not 1 <= epochs <= 10:
        raise ValueError("Ventana o épocas fuera del estudio de desarrollo")
    year = YEARS[window]
    expected = dict(
        id=window,
        train=["2000-01-01", f"{year - 1}-04-01"],
        validation=[f"{year - 1}-04-01", f"{year - 1}-10-01"],
        calibration=[f"{year - 1}-10-01", f"{year}-01-01"],
        evaluation=[f"{year}-01-01", f"{year + 1}-01-01"],
    )
    temporal = source.get("temporal_view", {})
    if (
        source.get("final_test_opened") is not False
        or source.get("input_policy") != "historical_masked_2000_v1"
        or temporal.get("fold") != expected
        or temporal.get("protocol", {}).get("market") != "US"
    ):
        raise ValueError("La fuente no conserva la ventana histórica declarada")
    original = {a["symbol"]: a for a in source["assets"]}
    selected = {a["symbol"] for a in assets}
    if (
        not selected
        or len(selected) != len(assets)
        or any(original.get(a["symbol"]) != a for a in assets)
    ):
        raise ValueError("La cohorte no pertenece íntegramente a la fuente")
    result = copy.deepcopy(source)
    result["assets"] = copy.deepcopy(assets)
    result["coverage"] = [
        r for r in result["coverage"] if r["market"] == "US" and r["symbol"] in selected
    ]
    result.update(
        scope="development_snapshot",
        cohort_complete=False,
        candidate_count=len(assets),
        samples=sum(a["samples"] for a in assets),
        training_ready=False,
    )
    result["counts"] = {
        phase: sum(a["counts"][phase] for a in assets) for phase in source["counts"]
    }
    result["purged_by_boundary"] = {
        phase: sum(a["purged_by_boundary"][phase] for a in assets) for phase in source["counts"]
    }
    result["temporal_view"]["protocol"]["selection"]["max_epochs"] = epochs
    return result
