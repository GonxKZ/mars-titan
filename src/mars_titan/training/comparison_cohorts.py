"""Cohortes de la campaña de comparación, elegidas con la historia de ajuste de cada origen (#836).

Una vista de la campaña con máscaras contiene todos los activos de EE. UU. de una ventana.
La comparación ajusta cada modelo con una cohorte de tamaño declarado, la misma para todas
las familias y semillas de ese origen. La regla es la de la criba de variantes: se ordenan
por una huella fija los activos que tienen al menos una etiqueta admitida en el tramo de
ajuste y se toman los primeros. Ni el objetivo, ni el tamaño del activo, ni su cobertura
posterior al ajuste intervienen. La sal es propia de esta campaña, así que sus cohortes no
dependen de las de la criba.

La proyección solo lee y escribe manifiestos. No abre muestras, etiquetas ni modelos.
"""

import copy
import hashlib
import json
import re
from pathlib import Path

from mars_titan.data.cohort_files import read_manifest, safe_destination
from mars_titan.data.input_policy import HISTORICAL_MASKED
from mars_titan.data.storage import atomic_json, outside_source, sha256
from mars_titan.evaluation.splits import PARTITIONS

from .cohort_contract import cohort_identity
from .temporal_contract import temporal_contracts

COHORT_SALT = "mars-titan-comparison-campaign-v1"
RULE = "sha256_among_origin_train_positive"
MAX_SIZE = 4096
KIND = "comparison_campaign_cohort_v1"
_SYMBOL = re.compile(r"[A-Z0-9.^_=\-]{1,64}")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _fold(source):
    """Ventana de una vista completa de EE. UU., tras comprobar que no trae otros filtros."""
    _require(
        isinstance(source, dict)
        and source.get("kind") == "corpus_supervision"
        and source.get("scope") == "full_corpus"
        and source.get("cohort_complete") is True
        and source.get("failed_assets") == 0
        and source.get("markets") == ["US"]
        and source.get("final_test_opened") is False
        and "data_subset" not in source,
        "El origen necesita la vista histórica completa de US, sin filtros adicionales",
    )
    cohort_identity(source, input_policy=HISTORICAL_MASKED)
    contracts = temporal_contracts(source, input_policy=HISTORICAL_MASKED)
    _require(set(contracts) == {"US"}, "El origen necesita un único calendario US")
    roots = source.get("roots", {})
    _require(
        isinstance(roots, dict)
        and set(roots) == {"prepared", "samples", "labels"}
        and all(isinstance(path, str) and Path(path).is_absolute() for path in roots.values()),
        "Las raíces de los datos deben ser rutas absolutas",
    )
    for asset in source["assets"]:
        symbol = asset.get("symbol")
        _require(
            asset.get("market") == "US"
            and isinstance(symbol, str)
            and symbol not in {".", ".."}
            and _SYMBOL.fullmatch(symbol) is not None,
            "El activo no tiene una identidad US válida",
        )
        counts = asset.get("counts", {})
        _require(
            isinstance(counts, dict)
            and set(counts) == set(PARTITIONS)
            and all(type(count) is int and count >= 0 for count in counts.values()),
            "Los recuentos por activo deben ser enteros no negativos",
        )
    _require(
        source.get("counts")
        == {name: sum(asset["counts"][name] for asset in source["assets"]) for name in PARTITIONS},
        "Los recuentos del origen no concilian con sus activos",
    )
    return contracts["US"]["fold"]


def _rank(symbol):
    return hashlib.sha256(f"{COHORT_SALT}/US/{symbol}".encode()).digest(), symbol


def project_cohort(source, *, size):
    """Vista con `size` activos de la ventana y el registro de su selección.

    Cada activo elegido conserva todas sus filas, máscaras y referencias. Un activo sin
    etiquetas en validación o evaluación sigue en la cohorte: su elección no mira esos tramos.
    """
    fold = _fold(source)
    _require(
        type(size) is int and 1 <= size <= MAX_SIZE,
        f"El número de activos debe estar entre 1 y {MAX_SIZE}",
    )
    candidates = [asset for asset in source["assets"] if asset["counts"]["train"] > 0]
    _require(len(candidates) >= size, "No hay suficientes activos con historia de ajuste")
    ranked = sorted(candidates, key=lambda asset: _rank(asset["symbol"]))[:size]
    symbols = sorted(asset["symbol"] for asset in ranked)
    selected = set(symbols)
    view = copy.deepcopy(source)
    for field in ("assets", "coverage"):
        view[field] = sorted(
            (item for item in view[field] if item["symbol"] in selected),
            key=lambda item: item["symbol"],
        )
    view.update(
        scope="development_snapshot",
        cohort_complete=False,
        candidate_count=size,
        samples=sum(asset["samples"] for asset in view["assets"]),
    )
    for field in ("counts", "purged_by_boundary"):
        view[field] = {
            name: sum(asset[field][name] for asset in view["assets"]) for name in PARTITIONS
        }
    selection = dict(
        fold=fold["id"],
        rule=RULE,
        salt=COHORT_SALT,
        information_before=fold["train"][1],
        evaluation=list(fold["evaluation"]),
        eligible_count=len(candidates),
        size=size,
        assets_sha256=hashlib.sha256(
            json.dumps([["US", symbol] for symbol in symbols], separators=(",", ":")).encode()
        ).hexdigest(),
        counts=view["counts"],
        targets_used=False,
        future_coverage_used=False,
    )
    return view, selection


def write_cohort(source_path, output, *, size):
    """Escribir la vista de la cohorte en `output/manifest.json` junto a su recibo.

    Si ya existe se comprueba que corresponde a la misma vista de origen, regla y tamaño, y
    se reutiliza. Devuelve la ruta del manifiesto y el recibo.
    """
    source_path, output = Path(source_path), Path(output)
    safe_destination(output)
    source, digest = read_manifest(source_path, 64 * 1024**2)
    view, selection = project_cohort(source, size=size)
    for root in source["roots"].values():
        outside_source(Path(root), output)
    manifest, receipt_path = output / "manifest.json", output / "cohort.json"
    receipt = dict(
        schema_version=1,
        kind=KIND,
        source=dict(path=str(source_path.resolve()), sha256=digest),
        selection=selection,
        verification_scope="manifests_only",
        row_files_opened=False,
        optimizer_steps=0,
        final_test_opened=False,
    )
    if manifest.exists() or receipt_path.exists():
        previous, _ = read_manifest(receipt_path, 1024**2)
        current, _ = read_manifest(manifest, 64 * 1024**2)
        _require(
            {key: value for key, value in previous.items() if key != "manifest_sha256"} == receipt
            and previous.get("manifest_sha256") == sha256(manifest)
            and current == view,
            "La cohorte guardada no corresponde a esta vista, regla y tamaño",
        )
        return manifest, previous
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(manifest, view)
    receipt["manifest_sha256"] = sha256(manifest)
    atomic_json(receipt_path, receipt)
    return manifest, receipt
