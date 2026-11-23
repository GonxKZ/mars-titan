"""Cohortes de la campaña de comparación, elegidas con la historia de ajuste de cada origen (#836).

Una vista de la campaña con máscaras contiene todos los activos de EE. UU. de una ventana.
La comparación ajusta cada modelo con una cohorte de tamaño declarado, la misma para todas
las familias y semillas de ese origen. La regla es la de la criba de variantes: se ordenan
por una huella fija los activos que tienen al menos una etiqueta admitida en el tramo de
ajuste y se toman los primeros. Ni el objetivo, ni el tamaño del activo, ni su cobertura
posterior al ajuste intervienen. La sal es propia de esta campaña, así que sus cohortes no
dependen de las de la criba.

La proyección solo lee y escribe manifiestos. No abre muestras, etiquetas ni modelos.

Una campaña con todos los activos no elige ninguno: su cohorte es la vista completa del origen,
byte a byte (`write_full_view`). Vale para la vista de un mercado y para la conjunta de US y
CN, donde cada mercado conserva su calendario y los dos comparten ventana.
"""

import copy
import hashlib
import json
import re
from pathlib import Path

from mars_titan.data.cohort_files import read_manifest, safe_destination
from mars_titan.data.input_policy import HISTORICAL_MASKED
from mars_titan.data.panel_signals import checked_panel_spec
from mars_titan.data.storage import atomic_json, outside_source, sha256
from mars_titan.data.technical_features import VERSIONS as TECHNICAL
from mars_titan.evaluation.splits import PARTITIONS

from .cohort_contract import cohort_identity
from .temporal_contract import temporal_contracts

COHORT_SALT = "mars-titan-comparison-campaign-v1"
RULE = "sha256_among_origin_train_positive"
MAX_SIZE = 4096
KIND = "comparison_campaign_cohort_v1"
FULL_VIEW_RULE = "entire_source_view"
FULL_VIEW_KIND = "comparison_campaign_full_view_v1"
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


def checked_extensions(extensions, *, markets=None):
    """Extensión de entradas que una campaña declara para sus cohortes, o None.

    Admite rasgos técnicos de horizonte largo (`technical`), en una de sus versiones, que
    `CorpusDataset` añade al bloque del gráfico al leer, junto a `market_model: required`
    opcional y un panel precalculado (`panel`) opcional cuyos mercados coincidan exactamente con
    los de la vista. La declara el manifiesto de la cohorte, así que su huella, sus índices y sus
    copias empaquetadas son distintos de los de la misma vista sin extensión o con otra versión o
    panel. Con `market_model: required` la lectura se detiene si falta en el equipo el factor de
    mercado de algún mercado, en vez de dejar a cero sus rasgos del modelo de mercado.
    """
    if extensions is None:
        return None
    _require(
        isinstance(extensions, dict)
        and {"technical"} <= set(extensions) <= {"technical", "market_model", "panel"}
        and isinstance(extensions["technical"], str)
        and extensions["technical"] in TECHNICAL
        and extensions.get("market_model", "required") == "required",
        f"La única extensión de entradas es technical ({' o '.join(TECHNICAL)}), con "
        "market_model required si el factor de mercado no puede faltar y panel opcional",
    )
    if "panel" in extensions:
        checked_panel_spec(extensions["panel"], markets=markets)
    return extensions


def write_cohort(source_path, output, *, size, extensions=None):
    """Escribir la vista de la cohorte en `output/manifest.json` junto a su recibo.

    Si ya existe se comprueba que corresponde a la misma vista de origen, regla, tamaño y
    extensión de entradas, y se reutiliza. Devuelve la ruta del manifiesto y el recibo.
    """
    source_path, output = Path(source_path), Path(output)
    safe_destination(output)
    source, digest = read_manifest(source_path, 64 * 1024**2)
    view, selection = project_cohort(source, size=size)
    if checked_extensions(extensions, markets=("US",)) is not None:
        view = dict(view, input_extensions=extensions)
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
    if extensions is not None:
        receipt["input_extensions"] = extensions
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


def _asset_counts(asset):
    counts = asset.get("counts", {})
    _require(
        isinstance(counts, dict)
        and set(counts) == set(PARTITIONS)
        and all(type(count) is int and count >= 0 for count in counts.values()),
        "Los recuentos por activo deben ser enteros no negativos",
    )
    return counts


def full_view(source):
    """Mercados, ventana común y activos de una vista histórica completa, sin elegir ninguno."""
    _require(
        isinstance(source, dict)
        and source.get("kind") == "corpus_supervision"
        and source.get("scope") == "full_corpus"
        and source.get("cohort_complete") is True
        and source.get("failed_assets") == 0
        and source.get("final_test_opened") is False
        and "data_subset" not in source,
        "El origen necesita la vista histórica completa, sin filtros adicionales",
    )
    cohort_identity(source, input_policy=HISTORICAL_MASKED)
    contracts = temporal_contracts(source, input_policy=HISTORICAL_MASKED)
    markets = sorted(contracts)
    declared = source.get("markets")
    _require(
        bool(markets) and isinstance(declared, list) and sorted(declared) == markets,
        "Los mercados de la vista no corresponden a sus calendarios",
    )
    folds = [contracts[market]["fold"] for market in markets]
    _require(all(fold == folds[0] for fold in folds), "Los mercados no comparten la ventana")
    roots = source.get("roots", {})
    _require(
        isinstance(roots, dict)
        and set(roots) == {"prepared", "samples", "labels"}
        and all(isinstance(path, str) and Path(path).is_absolute() for path in roots.values()),
        "Las raíces de los datos deben ser rutas absolutas",
    )
    assets = source.get("assets")
    _require(isinstance(assets, list) and assets, "La vista no tiene activos")
    identities = []
    for asset in assets:
        symbol = asset.get("symbol") if isinstance(asset, dict) else None
        _require(
            isinstance(asset, dict)
            and asset.get("market") in markets
            and isinstance(symbol, str)
            and symbol not in {".", ".."}
            and _SYMBOL.fullmatch(symbol) is not None,
            "Un activo no tiene una identidad válida de su mercado",
        )
        _asset_counts(asset)
        identities.append([asset["market"], symbol])
    _require(
        len({tuple(identity) for identity in identities}) == len(identities),
        "La vista repite un activo",
    )
    _require(
        {market for market, _ in identities} == set(markets),
        "Algún mercado de la vista no tiene activos",
    )
    _require(
        source.get("counts")
        == {name: sum(asset["counts"][name] for asset in assets) for name in PARTITIONS},
        "Los recuentos del origen no concilian con sus activos",
    )
    return markets, folds[0], sorted(identities)


def write_full_view(source_path, output, *, size, extensions=None):
    """Escribir como cohorte la vista completa del origen, byte a byte, junto a su recibo.

    Con una extensión de entradas el manifiesto es el de la vista más su declaración
    (`input_extensions`): los mismos activos, raíces y contratos, con otra huella.

    `size` es el número de activos que la declaración espera: la vista debe tener exactamente
    esos. No se elige ningún activo, así que ni los objetivos ni la cobertura intervienen. Si la
    cohorte ya existe se comprueba que es la misma vista y se reutiliza. Devuelve la ruta del
    manifiesto y el recibo.
    """
    source_path, output = Path(source_path), Path(output)
    safe_destination(output)
    source, digest = read_manifest(source_path, 64 * 1024**2)
    raw = source_path.read_bytes()
    _require(hashlib.sha256(raw).hexdigest() == digest, "La vista cambió mientras se leía")
    markets, fold, identities = full_view(source)
    _require(
        type(size) is int and len(identities) == size,
        f"La vista tiene {len(identities)} activos y la declaración espera {size}",
    )
    for root in source["roots"].values():
        outside_source(Path(root), output)
    manifest, receipt_path = output / "manifest.json", output / "cohort.json"
    receipt = dict(
        schema_version=1,
        kind=FULL_VIEW_KIND,
        source=dict(path=str(source_path.resolve()), sha256=digest),
        manifest_sha256=digest,
        selection=dict(
            fold=fold["id"],
            rule=FULL_VIEW_RULE,
            markets=markets,
            evaluation=list(fold["evaluation"]),
            size=size,
            assets_sha256=hashlib.sha256(
                json.dumps(identities, separators=(",", ":")).encode()
            ).hexdigest(),
            counts=source["counts"],
            targets_used=False,
            future_coverage_used=False,
        ),
        verification_scope="manifests_only",
        row_files_opened=False,
        optimizer_steps=0,
        final_test_opened=False,
    )
    document = None
    if checked_extensions(extensions, markets=markets) is not None:
        document = dict(source, input_extensions=extensions)
        receipt["input_extensions"] = extensions
        del receipt["manifest_sha256"]
    if manifest.exists() or receipt_path.exists():
        previous, _ = read_manifest(receipt_path, 1024**2)
        if document is None:
            same = previous == receipt and manifest.is_file() and sha256(manifest) == digest
        else:
            same = (
                {key: value for key, value in previous.items() if key != "manifest_sha256"}
                == receipt
                and manifest.is_file()
                and previous.get("manifest_sha256") == sha256(manifest)
                and read_manifest(manifest, 64 * 1024**2)[0] == document
            )
        _require(same, "La cohorte guardada no es esta vista completa")
        return manifest, previous
    output.mkdir(parents=True, exist_ok=True)
    if document is None:
        temporary = output / "manifest.json.tmp"
        temporary.write_bytes(raw)
        temporary.replace(manifest)
    else:
        atomic_json(manifest, document)
        receipt["manifest_sha256"] = sha256(manifest)
    atomic_json(receipt_path, receipt)
    return manifest, receipt
