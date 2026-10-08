"""Inputs históricos completos, separados de la admisión de sus etiquetas."""

import numpy as np
import pytest

from mars_titan.data.input_policy import HISTORICAL_MASKED
from mars_titan.training.corpus_inputs import CorpusDataset
from tests.training.test_historical_corpus_inputs import supervised

START = 946_684_800_000_000
END = 1_704_067_200_000_000


def test_every_input_row_is_read_without_calling_label_selection(tmp_path, monkeypatch):
    dataset = CorpusDataset(supervised(tmp_path), input_policy=HISTORICAL_MASKED)
    monkeypatch.setattr(dataset, "_labels", lambda *args: pytest.fail("Consultó labels"))
    batches = list(dataset.observation_batches(start=START, end=END, batch_size=1))
    assert len(batches) == 4
    assert all(not {"target", "target_available_at", "reason", "weight"} & set(b) for b in batches)
    assert all(b["presence"].tolist() == [[True, False, True, False, False]] for b in batches)
    assert all(np.all(b["inputs"]["macro"] == 0) for b in batches)
    assert len({key for b in batches for key in b["sample_ids"]}) == 4


def test_shared_rows_preserve_exact_supervised_inputs(tmp_path):
    dataset = CorpusDataset(supervised(tmp_path), input_policy=HISTORICAL_MASKED)
    labeled = list(dataset.batches(partition="train", batch_size=1, epoch=0, seed=42))[0]
    observed = list(dataset.observation_batches(start=START, end=END, batch_size=1))
    paired = next(b for b in observed if b["sample_ids"] == labeled["sample_ids"])
    for name in labeled["inputs"]:
        np.testing.assert_array_equal(paired["inputs"][name], labeled["inputs"][name])
    for name in ("presence", "prediction_at", "input_available_at"):
        np.testing.assert_array_equal(paired[name], labeled[name])


@pytest.mark.parametrize(
    "start,end,size",
    [(True, END, 1), (START, END + 1, 1), (END, START, 1), (START, END, 0), (START, END, 257)],
)
def test_observation_bounds_and_buffer_are_explicit(tmp_path, start, end, size):
    dataset = CorpusDataset(supervised(tmp_path), input_policy=HISTORICAL_MASKED)
    with pytest.raises(ValueError):
        list(dataset.observation_batches(start=start, end=end, batch_size=size))
