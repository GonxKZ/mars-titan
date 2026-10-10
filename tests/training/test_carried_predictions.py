"""Predicciones trasladadas de la variante B, en CPU y sin ajustar ningún modelo.

El ancla neuronal se construye a mano con pesos iniciales y un recibo seleccionado,
y el ancla tabular con una función fija de las entradas. Así se comprueban la carga
del estado, las filas de la ventana posterior y los rechazos sin pasos de optimizador.
"""

import json
from types import SimpleNamespace

import numpy as np
import pyarrow.parquet as pq
import pytest
import torch

from mars_titan.data.input_policy import HISTORICAL_MASKED, STRICT_INPUTS, policy_identity
from mars_titan.data.storage import atomic_json, sha256
from mars_titan.models.baselines.multimodal import (
    PRESENCE_FUSION,
    MultimodalReference,
    transformer_batch_options,
)
from mars_titan.models.quantile_head import MEDIAN_INDEX, QUANTILE_COLUMNS, QUANTILE_HEAD
from mars_titan.training import carried_predictions as carry
from mars_titan.training.checkpoints import capture_rng, save_training_state
from mars_titan.training.corpus_inputs import CorpusDataset
from mars_titan.training.kernel_policy import declared_policy
from mars_titan.training.reference_run import scientific_identity
from mars_titan.training.tabular_corpus import feature_order
from tests.training.test_masked_reference_run import masked_case
from tests.training.test_walk_forward_v2_views import PROTOCOLS, fixture, prepare


@pytest.fixture(scope="module")
def views(tmp_path_factory):
    root = tmp_path_factory.mktemp("carried")
    data = fixture(root / "data", ("US",))
    prepare(data, PROTOCOLS["US"], root / "views")
    return SimpleNamespace(
        root=root, view=lambda index: root / f"views/fold-{index:03d}/manifest.json"
    )


def manifest(path):
    return json.loads(path.read_text())


def test_carried_window_accepts_only_later_windows_of_the_same_edition(views, tmp_path):
    anchor, target = manifest(views.view(0)), manifest(views.view(2))
    first, later, months = carry.carried_window(anchor, target, input_policy=HISTORICAL_MASKED)
    assert first["evaluation"] == ["2005-01-01", "2006-01-01"]
    assert later["evaluation"] == ["2007-01-01", "2008-01-01"]
    # El ancla deja de aprender el 1 de octubre de 2004 y se evalúa desde enero de 2007.
    assert months == 27
    for earlier in (views.view(0), views.view(1)):
        with pytest.raises(ValueError, match="no es posterior"):
            carry.carried_window(
                manifest(views.view(1)), manifest(earlier), input_policy=HISTORICAL_MASKED
            )
    other = json.loads(json.dumps(target))
    other["temporal_view"]["parent_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="otra edición"):
        carry.carried_window(anchor, other, input_policy=HISTORICAL_MASKED)


class PresenceCount:
    """Función fija de las entradas, sin parámetros ajustados."""

    def predict(self, values):
        return values[:, -5:].sum(axis=1)


def reader(view, partition):
    dataset = CorpusDataset(view, input_policy=HISTORICAL_MASKED)
    ids, presence, targets, batches = [], [], [], []
    for batch in dataset.batches(partition=partition, batch_size=4, epoch=0, seed=0):
        ids += list(batch["sample_ids"])
        presence.append(batch["presence"])
        targets.append(batch["target"])
        batches.append(batch)
    return ids, np.concatenate(presence), np.concatenate(targets), batches


def tabular_anchor(folder, view, **changes):
    folder.mkdir(parents=True)
    report = dict(
        status="completed",
        model="ridge",
        final_test_opened=False,
        manifest_sha256=sha256(view),
        feature_order=feature_order(HISTORICAL_MASKED),
        checkpoint=dict(path="model.npz", sha256="0" * 64),
        samples=manifest(view)["counts"],
        **policy_identity(HISTORICAL_MASKED),
    )
    atomic_json(folder / "run.json", report | changes)
    return folder


def test_tabular_carry_predicts_the_later_window_rows_with_the_anchor_model(
    views, tmp_path, monkeypatch
):
    loads = []
    monkeypatch.setattr(
        carry, "_tabular_model", lambda *args: loads.append(args) or PresenceCount()
    )
    anchor = tabular_anchor(tmp_path / "anchor", views.view(0))
    output = tmp_path / "carry"
    receipt = carry.carry_tabular(
        anchor,
        views.view(0),
        views.view(1),
        output,
        kind="ridge",
        batch_size=3,
        input_policy=HISTORICAL_MASKED,
    )
    assert len(loads) == 1 and loads[0][2] == "ridge"
    assert receipt["status"] == "completed" and receipt["final_test_opened"] is False
    assert receipt["manifest_sha256"] == sha256(views.view(1))
    assert receipt["anchor"]["manifest_sha256"] == sha256(views.view(0))
    assert receipt["months_since_anchor_information"] == 15
    assert set(receipt["predictions"]) == {"calibration", "evaluation"}
    for partition in carry.CARRIED_PARTITIONS:
        ids, presence, targets, _ = reader(views.view(1), partition)
        rows = pq.read_table(output / receipt["predictions"][partition]["path"]).to_pylist()
        assert [row["sample_id"] for row in rows] == ids and ids
        np.testing.assert_array_equal([row["target"] for row in rows], targets)
        np.testing.assert_array_equal([row["prediction"] for row in rows], presence.sum(axis=1))
    assert json.loads((output / "carry.json").read_text()) == receipt
    with pytest.raises(ValueError, match="directorio nuevo"):
        carry.carry_tabular(
            anchor,
            views.view(0),
            views.view(1),
            output,
            kind="ridge",
            batch_size=3,
            input_policy=HISTORICAL_MASKED,
        )


@pytest.mark.parametrize(
    "changes",
    [
        dict(status="running"),
        dict(model="xgboost_external_cuda"),
        dict(final_test_opened=True),
        dict(feature_order=feature_order(STRICT_INPUTS)),
        dict(input_policy=STRICT_INPUTS),
        dict(manifest_sha256="0" * 64),
        dict(features=3),
    ],
)
def test_tabular_carry_rejects_anchors_that_do_not_match(views, tmp_path, monkeypatch, changes):
    monkeypatch.setattr(carry, "_tabular_model", lambda *args: PresenceCount())
    anchor = tabular_anchor(tmp_path / "anchor", views.view(0), **changes)
    with pytest.raises(ValueError):
        carry.carry_tabular(
            anchor,
            views.view(0),
            views.view(1),
            tmp_path / "carry",
            kind="ridge",
            batch_size=3,
            input_policy=HISTORICAL_MASKED,
        )
    assert not (tmp_path / "carry").exists()


def test_the_frozen_tabular_parent_also_predicts_the_validation_of_the_next_window(
    views, tmp_path, monkeypatch
):
    monkeypatch.setattr(carry, "_tabular_model", lambda *args: PresenceCount())
    anchor = tabular_anchor(tmp_path / "anchor", views.view(0))
    options = dict(kind="ridge", batch_size=3, input_policy=HISTORICAL_MASKED)
    output = tmp_path / "frozen"
    receipt = carry.carry_tabular(
        anchor, views.view(0), views.view(1), output, frozen_parent=True, **options
    )
    assert receipt["frozen_parent"] is True
    assert set(receipt["predictions"]) == set(carry.FROZEN_PARENT_PARTITIONS)
    # La validación de k la lee la cadena para elegir. Las filas son las de la vista de k.
    for partition in carry.FROZEN_PARENT_PARTITIONS:
        ids, presence, _, _ = reader(views.view(1), partition)
        rows = pq.read_table(output / receipt["predictions"][partition]["path"]).to_pylist()
        assert [row["sample_id"] for row in rows] == ids and ids
        np.testing.assert_array_equal([row["prediction"] for row in rows], presence.sum(axis=1))
    # Un traslado normal no lleva la marca ni predice la validación.
    normal = carry.carry_tabular(
        anchor, views.view(0), views.view(1), tmp_path / "carry", **options
    )
    assert "frozen_parent" not in normal and "validation" not in normal["predictions"]
    for changes in (dict(regenerate=True), dict(modality_ablation="mask_news")):
        with pytest.raises(ValueError, match="sin ablación ni regeneración"):
            carry.carry_tabular(
                anchor,
                views.view(0),
                views.view(1),
                tmp_path / "rejected",
                frozen_parent=True,
                **options,
                **changes,
            )
    assert not (tmp_path / "rejected").exists()


@pytest.fixture
def cpu(monkeypatch):
    """Sustituir CUDA por CPU solo en la prueba, sin optimizadores ni pasos."""
    import mars_titan.data.embeddings as embeddings

    monkeypatch.setattr(embeddings, "require_cuda", lambda **_: torch.device("cpu"))
    for name, value in dict(
        synchronize=None,
        reset_peak_memory_stats=None,
        max_memory_allocated=0,
        get_device_name="cpu-technical-fixture",
    ).items():
        monkeypatch.setattr(torch.cuda, name, lambda *_, value=value: value)
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.delenv("MARS_TITAN_INPUT_CACHE_MIB", raising=False)


def neural_anchor(folder, view, kind="gru", *, precision=None, batch_size=2):
    """Recibo seleccionado con los pesos iniciales de la semilla, como en la época 0."""
    case = masked_case(kind, epochs=2) | dict(head=QUANTILE_HEAD, loss="pinball")
    if precision is not None:
        case["precision"] = precision
    kernel_policy = declared_policy(precision)
    dataset = CorpusDataset(view, input_policy=HISTORICAL_MASKED)
    first = next(dataset.batches(partition="train", batch_size=1, epoch=0, seed=0))
    dimensions = {name: value.shape[-1] for name, value in first["inputs"].items()}
    torch.manual_seed(7)
    model = MultimodalReference(
        kind,
        dimensions,
        context=64,
        mask_fusion=PRESENCE_FUSION,
        head=QUANTILE_HEAD,
        **case["architecture"],
        **transformer_batch_options(kind, batch_size),
    )
    identity = dict(
        **scientific_identity(kind=kind, input_policy=HISTORICAL_MASKED, head=QUANTILE_HEAD),
        manifest_sha256=sha256(view),
        case=case,
        batch_size=batch_size,
        dimensions=dimensions,
        context=64,
        **policy_identity(HISTORICAL_MASKED),
        mask_fusion=PRESENCE_FUSION,
    )
    if kernel_policy is not None:
        identity["kernel_policy"] = kernel_policy
    selection = dict(
        last_epoch=1,
        best_epoch=1,
        best_score=0.25,
        stale_epochs=0,
        should_stop=False,
        last_improved=True,
    )
    state = dict(
        global_step=1,
        epoch=1,
        confirmed_cursor=None,
        model=model.state_dict(),
        optimizer={},
        rng=capture_rng("cpu"),
        statistics=dict(samples=0, squared_error=0.0, absolute_error=0.0, elapsed_seconds=0.0),
        history=[],
        selection=selection,
        initial_validation=None,
    )
    path = save_training_state(folder / "checkpoints", state, identity=identity, best=True)
    report = dict(
        status="completed",
        final_test_opened=False,
        identity=identity,
        scope=manifest(view)["scope"],
        selection=selection,
        checkpoint=dict(path=str(path.relative_to(folder)), sha256=sha256(path)),
    )
    atomic_json(folder / "run.json", report)
    return model


def test_neural_carry_loads_the_selected_state_and_predicts_the_later_window(views, tmp_path, cpu):
    model = neural_anchor(tmp_path / "anchor", views.view(0))
    output = tmp_path / "carry"
    receipt = carry.carry_reference(
        tmp_path / "anchor",
        views.view(0),
        views.view(2),
        output,
        batch_size=2,
        input_policy=HISTORICAL_MASKED,
    )
    assert receipt["months_since_anchor_information"] == 27
    assert (
        receipt["anchor"]["checkpoint_sha256"]
        == json.loads((tmp_path / "anchor/run.json").read_text())["checkpoint"]["sha256"]
    )
    model.eval()
    for partition in carry.CARRIED_PARTITIONS:
        ids, presence, targets, batches = reader(views.view(2), partition)
        table = pq.read_table(output / receipt["predictions"][partition]["path"])
        assert table["sample_id"].to_pylist() == ids and ids
        with torch.inference_mode():
            expected = np.concatenate(
                [
                    model(
                        {k: torch.from_numpy(v) for k, v in batch["inputs"].items()},
                        torch.from_numpy(batch["presence"]),
                    ).numpy()
                    for batch in batches
                ]
            )
        levels = np.column_stack([table[name].to_numpy() for name in QUANTILE_COLUMNS])
        np.testing.assert_allclose(levels, expected, rtol=1e-6, atol=1e-7)
        assert np.array_equal(table["prediction"].to_numpy(), levels[:, MEDIAN_INDEX])


def test_neural_carry_rejects_another_environment_and_an_earlier_window(
    views, tmp_path, cpu, monkeypatch
):
    neural_anchor(tmp_path / "anchor", views.view(1))
    with pytest.raises(ValueError, match="no es posterior"):
        carry.carry_reference(
            tmp_path / "anchor",
            views.view(1),
            views.view(0),
            tmp_path / "a",
            batch_size=2,
            input_policy=HISTORICAL_MASKED,
        )
    with pytest.raises(ValueError, match="misma política"):
        carry.carry_reference(
            tmp_path / "anchor",
            views.view(0),
            views.view(2),
            tmp_path / "b",
            batch_size=2,
            input_policy=HISTORICAL_MASKED,
        )
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *_: "otra-gpu")
    with pytest.raises(ValueError, match="entorno o el código"):
        carry.carry_reference(
            tmp_path / "anchor",
            views.view(1),
            views.view(2),
            tmp_path / "c",
            batch_size=2,
            input_policy=HISTORICAL_MASKED,
        )
    assert not any((tmp_path / name).exists() for name in "abc")
