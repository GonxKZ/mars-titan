"""Edición técnica con objetivos y manifiesto de la edición para construir el catálogo."""

import json
from pathlib import Path
from types import SimpleNamespace

from mars_titan.data.input_policy import HISTORICAL_MASKED
from mars_titan.data.storage import atomic_json, sha256
from mars_titan.training.edition_catalog import build_catalog
from tests.training.historical_temporal_fixture import historical_temporal_fixture
from tests.training.test_walk_forward_v2_views import PROTOCOLS, sessions


def presence(market, symbol, row):
    """Noticias, fundamentales y macro variados por activo y fila."""
    seed = row + len(symbol) + (3 if market == "CN" else 0) + int(symbol[-1])
    return seed % 2 == 0, seed % 3 != 0, seed % 4 != 1


def edition_fixture(root, markets=("US", "CN"), *, assets=3, days=None):
    """Objetivos técnicos y un manifiesto de edición que los declara como su origen.

    Sin `days`, cada mercado tiene filas en todos los tramos de las ventanas v2.
    """
    root = Path(root)
    chosen = {market: days[market] if days else sessions(market) for market in markets}
    data = historical_temporal_fixture(
        root, markets=markets, days=chosen, assets=assets, presence=presence
    )
    meta = json.loads(data.parent.read_text())
    edition = root / "edition" / "manifest.json"
    atomic_json(
        edition,
        dict(
            schema_version=3,
            kind="materialized_corpus",
            input_policy=HISTORICAL_MASKED,
            cohort_complete=True,
            failed_assets=0,
            final_test_opened=False,
            samples_root=meta["roots"]["samples"],
            samples=meta["samples"],
        ),
    )
    meta.setdefault("configuration", {})["source_manifest_sha256"] = sha256(edition)
    atomic_json(data.parent, meta)
    return SimpleNamespace(
        **{**vars(data), "metadata": meta}, edition=edition, protocols_v2=PROTOCOLS
    )


def built_catalog(root, markets=("US", "CN"), **kwargs):
    data = edition_fixture(Path(root) / "data", markets, **kwargs)
    build_catalog(data.parent, data.edition, Path(root) / "catalog")
    return SimpleNamespace(**vars(data), catalog=Path(root) / "catalog" / "manifest.json")
