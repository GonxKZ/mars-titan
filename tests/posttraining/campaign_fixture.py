"""Campaña base reducida con padres de cuantiles reales y pesos iniciales, sin ajustes.

El protocolo US se recorta a dos ventanas (evaluación de 2022 y 2023) y la comparación a
un único brazo GRU con la semilla 42. El ejecutor neuronal sustituto construye la
referencia de la campaña con la cabeza `quantile_head_v1` y pesos iniciales, guarda su
estado elegido y escribe validación, calibración y evaluación con sus predicciones
reales en CPU. Así la etapa de postentrenamiento carga padres verdaderos sin que la
campaña base aplique ningún paso de optimizador. Los traslados de la variante B usan el
doble de las pruebas de la campaña, que escribe predicciones nulas.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from mars_titan.data.input_policy import HISTORICAL_MASKED, policy_identity
from mars_titan.data.storage import atomic_json, sha256
from mars_titan.models.baselines.multimodal import PRESENCE_FUSION, MultimodalReference
from mars_titan.models.quantile_head import CONTRACT, QUANTILE_COLUMNS, QUANTILE_HEAD, median
from mars_titan.training import masked_campaign as engine
from mars_titan.training.checkpoints import save_training_state
from mars_titan.training.corpus_inputs import CorpusDataset
from mars_titan.training.learning_hold import HOLD_ENV
from mars_titan.training.reference_run import HELDOUT_FULL_TRAIN_SESSIONS
from tests.training.test_carried_predictions import PresenceCount
from tests.training.test_masked_campaign import Recorder
from tests.training.test_walk_forward_v2_views import fixture

CONFIGS = Path("configs").resolve()
ROOT = Path("src/mars_titan").resolve()
# Lote común de la campaña y la matriz. Con un activo por sesión, cada lote de ajuste
# tiene una fila y la evaluación agrupa sesiones igual que el doble de la referencia.
BATCH = 64
# Huellas del contrato de inferencia que exige `posttraining.parents` a una referencia.
INFERENCE = (
    "training/corpus_inputs.py",
    "training/temporal_corpus.py",
    "evaluation/splits.py",
    "evaluation/split_readiness.py",
    "models/baselines/dlinear.py",
    "models/baselines/multimodal.py",
    "models/baselines/transformer.py",
    "models/quantile_head.py",
    "data/input_policy.py",
)
SCORES = {"gru-00": 0.02, "gru-10": 0.01}


def write_configs(folder, variant, *, tabular=False):
    """Comparación, protocolo, campaña, matriz y etapa reducidos para una variante.

    Con `tabular` la comparación, la campaña y la etapa declaran además Ridge, cuya cadena
    solo tiene el padre congelado.
    """
    folder.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(
        (CONFIGS / "evaluation/historical-masked-us-walk-forward-v2.json").read_text()
    )
    protocol["first_validation_start"] = "2021-04-01"
    atomic_json(folder / "us-protocol.json", protocol)
    declared = json.loads(
        (CONFIGS / "evaluation/historical-masked-2000-comparison.json").read_text()
    )
    for scope in declared["scopes"].values():
        scope["protocols"] = {
            market: str((CONFIGS / "evaluation" / name).resolve())
            for market, name in scope["protocols"].items()
        }
    declared["scopes"]["US"]["protocols"]["US"] = str((folder / "us-protocol.json").resolve())
    names = ("zero", "gru", "ridge") if tabular else ("zero", "gru")
    declared["arms"] = {key: declared["arms"][key] for key in names}
    # Los pares de retención nombran brazos que esta comparación reducida no conserva.
    declared.pop("retention_interference")
    declared["arms"]["gru"]["seeds"] = [42]
    declared["comparison"].update(
        replicates=20,
        families=dict(references_vs_zero=dict(kind="delta", base="zero", variants=["gru"])),
    )
    atomic_json(folder / "comparison.json", declared)
    search = json.loads((CONFIGS / "baselines/tabular-historical-masked.json").read_text())
    search.update(ridge_alphas=[1.0], depths=[3], bins=[64], rates=[0.1])
    atomic_json(folder / "tabular.json", search)
    campaign = json.loads(
        (CONFIGS / f"baselines/historical-masked-campaign-{variant.lower()}.json").read_text()
    )
    campaign.update(
        comparison="comparison.json",
        scopes=["US"],
        retrain_every_months=12 if variant == "A" else 24,
        limits=dict(max_training_jobs=100, max_prediction_jobs=100),
    )
    campaign["neural"].update(arms={"gru": "gru"}, batch_size=BATCH)
    campaign["tabular"].update(config="tabular.json", arms={"ridge": "ridge"} if tabular else {})
    # La comparación reducida no declara los brazos de Titans-MAC.
    campaign.pop("titans_mac")
    atomic_json(folder / "campaign.json", campaign)
    matrix = json.loads((CONFIGS / "posttraining/adapter-matrix-v2.json").read_text())
    matrix["budget"].update(seeds=[42], epochs=1, batch_size=BATCH)
    atomic_json(folder / "matrix.json", matrix)
    stage = json.loads(
        (
            CONFIGS / f"posttraining/historical-masked-adapter-stage-{variant.lower()}.json"
        ).read_text()
    )
    stage.update(
        campaign="campaign.json",
        matrix="matrix.json",
        scopes=["US"],
        arms=["gru", "ridge"] if tabular else ["gru"],
        limits=dict(max_training_jobs=100, max_prediction_jobs=100),
    )
    atomic_json(folder / "stage.json", stage)
    return folder / "campaign.json", folder / "stage.json"


def _rows(model, dataset, partition):
    """Predicciones reales del padre por fila, con el esquema de la referencia."""
    parts = []
    with torch.inference_mode():
        for batch in dataset.batches(partition=partition, batch_size=BATCH, epoch=0, seed=0):
            inputs = {name: torch.from_numpy(value) for name, value in batch["inputs"].items()}
            levels = model(inputs, torch.from_numpy(batch["presence"])).numpy()
            columns = dict(
                sample_id=batch["sample_ids"],
                asset_id=["/".join(key.split("/")[:2]) for key in batch["sample_ids"]],
                market=batch["market"],
                prediction_at=pa.array(batch["prediction_at"], type=pa.timestamp("us", tz="UTC")),
                target=batch["target"],
                prediction=median(torch.from_numpy(levels)).numpy(),
                zero=np.zeros(len(levels), dtype=np.float64),
            )
            columns.update(zip(QUANTILE_COLUMNS, levels.T, strict=True))
            parts.append(pa.table(columns))
    return pa.concat_tables(parts)


def quantile_reference(run):
    """Ejecutor neuronal sustituto: estado elegido con pesos iniciales y recibo completo."""
    case = run.case
    dataset = CorpusDataset(run.view, input_policy=HISTORICAL_MASKED)
    first = next(dataset.batches(partition="train", batch_size=1, epoch=0, seed=0))
    dimensions = {name: value.shape[-1] for name, value in first["inputs"].items()}
    torch.manual_seed(case["seed"])
    model = MultimodalReference(
        case["kind"],
        dimensions,
        context=dataset.context,
        mask_fusion=PRESENCE_FUSION,
        head=QUANTILE_HEAD,
        **case["architecture"],
    ).eval()
    identity = dict(
        manifest_sha256=dataset.identity,
        case=case,
        model_family="scientific_multimodal_reference",
        batch_size=run.batch_size,
        dimensions=dimensions,
        context=dataset.context,
        weighting="natural",
        code={name: sha256(ROOT / name) for name in INFERENCE},
        prediction_retention=HELDOUT_FULL_TRAIN_SESSIONS,
        mask_fusion=PRESENCE_FUSION,
        output_head=dict(CONTRACT),
        **policy_identity(HISTORICAL_MASKED),
    )
    score = SCORES[run.job["candidate"]] if run.job["stage"] == "search" else 0.015
    run.folder.mkdir(parents=True, exist_ok=True)
    checkpoint = save_training_state(
        run.folder / "checkpoints",
        dict(
            global_step=0,
            epoch=0,
            confirmed_cursor=None,
            model=model.state_dict(),
            statistics={"samples": 0},
            selection=dict(best_score=score, best_epoch=0),
        ),
        identity=identity,
        best=True,
    )
    predictions = {}
    for partition in ("validation", "calibration", "evaluation"):
        path = run.folder / f"{partition}-predictions.parquet"
        pq.write_table(_rows(model, dataset, partition), path)
        predictions[partition] = dict(path=path.name, sha256=sha256(path))
    predictions["validation"]["metrics"] = dict(session_mae=score)
    report = dict(
        schema_version=1,
        status="completed",
        identity=identity,
        scope=dataset.manifest["scope"],
        cohort_complete=dataset.manifest["cohort_complete"],
        samples=dataset.manifest["counts"],
        predictions=predictions,
        train_summary={},
        checkpoint=dict(path=str(checkpoint.relative_to(run.folder)), sha256=sha256(checkpoint)),
        selection=dict(best_epoch=0, best_score=score, last_epoch=0),
        final_test_opened=False,
    )
    atomic_json(run.folder / "run.json", report)
    return report


def fixed_tabular(run):
    """Ejecutor Ridge sustituto: estado ficticio y el recibo que acepta un traslado tabular.

    Predice con la suma fija de las presencias, sin ajustar nada. Como no depende de la
    ventana, el padre congelado de k-1 debe repetir sobre k las predicciones de la base en k.
    """
    from mars_titan.data.input_policy import masked_inputs
    from mars_titan.training.tabular_corpus import _predict, feature_order

    dataset = CorpusDataset(run.view, input_policy=HISTORICAL_MASKED)
    run.folder.mkdir(parents=True, exist_ok=True)
    # El estado no tiene parámetros: su huella solo identifica al padre en los recibos.
    (run.folder / "model.npz").write_text(run.job["id"])
    predictions = {}
    for partition in ("validation", "calibration", "evaluation"):
        path = run.folder / f"{partition}-predictions.parquet"
        metrics = _predict(
            PresenceCount(),
            None,
            dataset,
            partition,
            BATCH,
            path,
            presence=masked_inputs(HISTORICAL_MASKED),
        )
        predictions[partition] = dict(path=path.name, sha256=sha256(path), metrics=metrics)
    report = dict(
        status="completed",
        model="ridge",
        final_test_opened=False,
        manifest_sha256=sha256(run.view),
        feature_order=feature_order(HISTORICAL_MASKED),
        checkpoint=dict(path="model.npz", sha256=sha256(run.folder / "model.npz")),
        samples=dataset.manifest["counts"],
        predictions=predictions,
        **policy_identity(HISTORICAL_MASKED),
    )
    atomic_json(run.folder / "run.json", report)
    return report


def executors():
    carry = Recorder()
    fits = {("neural", engine.FIT): quantile_reference, ("ridge", engine.FIT): fixed_tabular}
    return {key: dict(entry, run=fits.get(key, carry)) for key, entry in engine.EXECUTORS.items()}


class CpuLease:
    """Sustituto de la reserva de GPU para recorrer la etapa en CPU."""

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def check(self):
        pass


def base_campaign(root, variant, *, tabular=False):
    """Preparar vistas y ejecutar la campaña base reducida con los dobles."""
    campaign, stage = write_configs(root / "config", variant, tabular=tabular)
    data = fixture(root / "data", ("US",))
    hold = root / "hold.json"
    hold.write_text(json.dumps({"training_allowed": True}), encoding="utf-8")
    with pytest.MonkeyPatch.context() as patch:
        # La campaña base solo ejecuta dobles: ningún modelo se ajusta.
        patch.setenv(HOLD_ENV, str(hold))
        engine.prepare_views(campaign, data.parent, root / "views")
        views = {"US": root / "views" / "US"}
        summary = engine.run_campaign(
            campaign,
            views,
            root / "campaign",
            executors=executors(),
            lease=CpuLease,
            stop=SimpleNamespace(requested=False),
        )
    assert summary["status"] == "completed"
    return SimpleNamespace(
        campaign=campaign, stage=stage, views=views, output=root / "campaign", root=root
    )
