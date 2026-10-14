"""Declaraciones mínimas del paso final sobre una campaña reducida de las pruebas.

La matriz compara GRU con el control cero, con un linaje de un componente, y puede añadir la
vista de la cartera. La comparación postentrenada es la de A sobre la etapa reducida. Todas
las rutas son absolutas, así que las declaraciones pueden vivir fuera de la configuración.
"""

import json
from pathlib import Path
from types import SimpleNamespace

from mars_titan.data.storage import atomic_json
from mars_titan.evaluation import comparison_matrix
from mars_titan.evaluation import walk_forward_comparison as walk
from mars_titan.training import campaign_publication as publication

ROOT = Path(__file__).resolve().parents[2]
POSTTRAINING = ROOT / "configs/posttraining/historical-masked-adapter-comparison-a.json"


def declare(folder, campaign, *, stage=None, portfolio=False):
    """Matriz, comparación postentrenada (con `stage`) y publicación de la campaña."""
    folder, campaign = Path(folder), Path(campaign).resolve()
    views = dict(
        forecast=dict(
            source="walk_forward_comparison_report",
            quantiles="raw",
            metrics=list(walk.SERIES_METRICS),
        )
    )
    if portfolio:
        views["portfolio"] = dict(
            source="long_short_comparison_report", statistics=["mean_net_return", "sharpe"]
        )
    matrix = folder / "matrix" / "matrix.json"
    matrix.parent.mkdir(parents=True)
    atomic_json(
        matrix,
        dict(
            schema_version=1,
            kind="comparison_matrix",
            name="fixture-matrix",
            status="declared_before_results",
            declared_at="2026-10-10",
            use=comparison_matrix.USE,
            comparison=str(campaign.parent / "comparison.json"),
            conditional_arms={},
            derived_arms={},
            groups=dict(models=["gru"]),
            questions=dict(
                versus_zero=dict(
                    kind="pairs", pairs=[["zero", "gru"]], question="¿GRU mejora al control cero?"
                )
            ),
            lineages=dict(
                network=dict(
                    question="¿Cuánto aporta la red frente al control cero?",
                    components=dict(network=dict(label="Red", requires=[])),
                    arms=dict(zero=[], gru=["network"]),
                    ladder=["network"],
                    full=["network"],
                    interactions=[],
                    games=[],
                )
            ),
            candidates={},
            views=views,
        ),
    )
    posttraining = None
    if stage is not None:
        posttraining = folder / "posttraining.json"
        declared = json.loads(POSTTRAINING.read_text())
        atomic_json(posttraining, dict(declared, stage=str(Path(stage).resolve())))
    path = folder / "publication.json"
    atomic_json(
        path,
        dict(
            schema_version=1,
            kind=publication.KIND,
            status=publication.DECLARED,
            name="fixture-publication",
            campaign=str(campaign),
            comparison_matrix=str(matrix),
            posttraining_comparison=None if posttraining is None else str(posttraining),
            final_test_opened=False,
        ),
    )
    return SimpleNamespace(matrix=matrix, stage=posttraining, publication=path)
