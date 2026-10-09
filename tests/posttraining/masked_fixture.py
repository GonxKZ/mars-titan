"""Corpus ordenado con máscaras y padres técnicos, sin ajustar ningún modelo."""

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from mars_titan.data.input_policy import HISTORICAL_MASKED, policy_identity
from mars_titan.data.storage import atomic_json, sha256
from mars_titan.environments.corpus_source import ParquetCohortSource, prepare_causal_corpus
from mars_titan.models.baselines.multimodal import PRESENCE_FUSION, MultimodalReference
from mars_titan.models.quantile_head import CONTRACT
from mars_titan.training.checkpoints import save_training_state
from mars_titan.training.reference_run import HELDOUT_FULL_TRAIN_SESSIONS
from tests.training.historical_temporal_fixture import historical_temporal_fixture
from tests.training.test_historical_temporal import prepare

ROOT = Path("src/mars_titan")


def masked_view(root, mutate=None):
    """Preparar la primera ventana. `mutate` cambia las muestras antes de crear vistas."""
    fixture = historical_temporal_fixture(root / "source")
    if mutate is not None:
        meta = json.loads(fixture.parent.read_text())
        for asset in meta["assets"]:
            for kind in ("samples", "labels"):
                path = Path(meta["roots"][kind]) / asset["market"] / asset["symbol"]
                path = path / f"{kind}.parquet"
                table = pq.read_table(path)
                rows = mutate(kind, table.to_pylist())
                pq.write_table(
                    pa.Table.from_pylist(rows, schema=table.schema), path, row_group_size=3
                )
                asset[kind + "_sha256"] = sha256(path)
        atomic_json(fixture.parent, meta)
    prepare(fixture, root / "views")
    return root / "views/fold-000/manifest.json"


def masked_ordered(root, mutate=None):
    view = masked_view(root, mutate)
    report = prepare_causal_corpus(
        view, root / "ordered", batch_size=2, input_policy=HISTORICAL_MASKED
    )
    return view, root / "ordered/manifest.json", report


def sources(ordered):
    return (
        ParquetCohortSource(ordered, partition="train", input_policy=HISTORICAL_MASKED),
        ParquetCohortSource(ordered, partition="validation", input_policy=HISTORICAL_MASKED),
    )


def architecture(kind):
    value = dict(hidden_size=32, layers=1, dropout=0.0)
    if kind == "transformer":
        value["transformer"] = dict(heads=2, feedforward_multiplier=2)
    return value


def masked_parent(
    ordered, folder, kind="gru", *, seed=3, policy=True, fusion=PRESENCE_FUSION, head=None
):
    """Escribir un recibo completo de referencia con fusión de presencia y pesos aleatorios."""
    source = json.loads(Path(ordered).read_text())
    shapes = source["shapes"]
    dimensions = {name: shape[-1] for name, shape in shapes.items()}
    case = dict(kind=kind, architecture=architecture(kind), epochs=1)
    if head is not None:
        case["head"] = head
    names = [
        "training/corpus_inputs.py",
        "training/temporal_corpus.py",
        "evaluation/splits.py",
        "evaluation/split_readiness.py",
        "models/baselines/dlinear.py",
        "models/baselines/multimodal.py",
        "models/baselines/transformer.py",
        "data/input_policy.py",
        *(["models/quantile_head.py"] if head is not None else []),
    ]
    identity = dict(
        manifest_sha256=source["source_sha256"],
        case=case,
        dimensions=dimensions,
        context=shapes["prices"][0],
        model_family="scientific_multimodal_reference",
        weighting="natural",
        market_weights={"US": 1.0},
        code={name: sha256(ROOT / name) for name in names},
        prediction_retention=HELDOUT_FULL_TRAIN_SESSIONS,
    )
    if policy:
        identity.update(policy_identity(HISTORICAL_MASKED), mask_fusion=fusion)
    if head is not None:
        identity["output_head"] = dict(CONTRACT)
    torch.manual_seed(seed)
    model = MultimodalReference(
        kind,
        dimensions,
        context=shapes["prices"][0],
        mask_fusion=fusion,
        **({"head": head} if head is not None else {}),
        **case["architecture"],
    )
    folder = Path(folder)
    folder.mkdir(parents=True)
    checkpoint = save_training_state(
        folder / "checkpoints",
        dict(
            global_step=1,
            epoch=1,
            model=model.state_dict(),
            confirmed_cursor=None,
            statistics={"samples": 0},
        ),
        identity=identity,
    )
    report = dict(
        status="completed",
        final_test_opened=False,
        identity=identity,
        scope=source["scope"],
        cohort_complete=source["cohort_complete"],
        samples=source["counts"],
        predictions={name: {} for name in ("validation", "calibration", "evaluation")},
        train_summary={},
        checkpoint=dict(path=str(checkpoint.relative_to(folder)), sha256=sha256(checkpoint)),
    )
    atomic_json(folder / "run.json", report)
    return folder / "run.json", model


def change_after_training(train_moments):
    """Alterar entradas y objetivos de todas las filas que no pertenecen al ajuste."""

    def mutate(kind, rows):
        for row in rows:
            moment = int(np.datetime64(row["prediction_at"].replace(tzinfo=None), "us").astype(int))
            if moment in train_moments:
                continue
            if kind == "samples":
                row["charts"] = [0.75] * len(row["charts"])
                if row["presence"][4]:
                    # Solo cambia el valor de un indicador observado, no su máscara.
                    row["macro"] = [5.0, *row["macro"][1:]]
            elif row["target"] is not None:
                row["target"] += 1000.0
        return rows

    return mutate
