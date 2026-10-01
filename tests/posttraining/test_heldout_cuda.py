"""Paridad real de inferencia CUDA, habilitada solo en una ventana de GPU exclusiva."""

import copy
import os

import numpy as np
import pyarrow.parquet as pq
import pytest
import torch

from mars_titan.environments.actions import ActionGrid
from mars_titan.models.baselines.multimodal import MultimodalReference
from mars_titan.models.predictive_adaptation import LinearResidualPolicy
from mars_titan.posttraining.heldout import evaluate_partition
from mars_titan.posttraining.parents import FrozenParent
from mars_titan.training.checkpoints import StopRequest
from mars_titan.training.corpus_inputs import CorpusDataset
from mars_titan.training.experiment_resources import GpuLease
from tests.training.test_temporal_corpus import inputs as inputs
from tests.training.test_temporal_corpus import prepare


@pytest.mark.skipif(
    os.environ.get("MARS_TITAN_CUDA_INTEGRATION") != "1",
    reason="Requiere una ventana CUDA exclusiva y activación explícita",
)
def test_selected_inference_preserves_cpu_reference_on_cuda(inputs, tmp_path):
    prepare(inputs, tmp_path / "views")
    dataset = CorpusDataset(tmp_path / "views/fold-000/manifest.json")
    batch = next(dataset.batches(partition="train", batch_size=2, epoch=0, seed=0))
    shapes = {name: values.shape[1:] for name, values in batch["inputs"].items()}
    torch.manual_seed(42)
    model = MultimodalReference(
        "gru",
        {name: shape[-1] for name, shape in shapes.items()},
        context=dataset.context,
        hidden_size=32,
        layers=1,
        dropout=0,
    )
    identity = dict(model="gru", checkpoint_sha256="a" * 64)
    cpu = FrozenParent(model, identity, shapes, "cpu")
    grid = ActionGrid.fit(batch["target"], source_sha256=dataset.identity, partition="train")
    size = 1 + sum(np.prod(shape) for shape in shapes.values())
    adapter = LinearResidualPolicy(np.zeros(size), np.ones(size), target_scale=grid.scale)
    for discrete in (False, True):
        kwargs = dict(model=adapter if discrete else None, grid=grid if discrete else None)
        evaluate_partition(
            dataset,
            cpu,
            "evaluation",
            tmp_path / f"cpu-{discrete}.parquet",
            stop=StopRequest(),
            device="cpu",
            **kwargs,
        )
    with GpuLease() as lease:
        gpu = FrozenParent(copy.deepcopy(cpu.model), identity, shapes, "cuda:0")
        for discrete in (False, True):
            kwargs = dict(
                model=copy.deepcopy(adapter).to("cuda:0") if discrete else None,
                grid=grid if discrete else None,
            )
            evaluate_partition(
                dataset,
                gpu,
                "evaluation",
                tmp_path / f"cuda-{discrete}.parquet",
                stop=StopRequest(),
                device="cuda:0",
                **kwargs,
            )
            reference = pq.read_table(tmp_path / f"cpu-{discrete}.parquet").to_pydict()
            actual = pq.read_table(tmp_path / f"cuda-{discrete}.parquet").to_pydict()
            assert actual["sample_id"] == reference["sample_id"]
            for name in ("prediction", "parent"):
                np.testing.assert_allclose(actual[name], reference[name], rtol=1e-5, atol=1e-7)
        lease.check()
