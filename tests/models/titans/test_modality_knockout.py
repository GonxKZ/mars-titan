"""Knockout de modalidades en evaluación: misma ruta que una ausencia real y sin efecto apagado."""

import pytest
import torch

from mars_titan.models.baselines.multimodal import MODALITIES
from tests.models.titans.test_asset_groups import model, priced_batch


def _emitted(predictor, batch):
    return predictor.prepare(batch, predictor.initial_state(batch.flow_ids)).point_predictions


def _without(batch, names):
    keep = torch.ones_like(batch.presence)
    keep[:, [MODALITIES.index(name) for name in names]] = False
    return batch.with_absences(keep, "corpus_absence")


@pytest.mark.parametrize(
    "names", [("news",), ("charts", "macro"), ("news", "charts", "fundamentals", "macro")]
)
def test_knockout_equals_the_same_rows_arriving_without_those_modalities(names):
    batch = priced_batch(rows=64)
    predictor = model()
    predictor.eval()
    with predictor.modality_knockout(names):
        knocked = _emitted(predictor, batch)
    torch.testing.assert_close(knocked, _emitted(predictor, _without(batch, names)), rtol=0, atol=0)


def test_knockout_is_scoped_restores_the_previous_mode_and_changes_nothing_when_off():
    batch = priced_batch(rows=64)
    predictor = model()
    predictor.eval()
    plain = _emitted(predictor, batch)
    with predictor.modality_knockout(("news",)):
        with predictor.modality_knockout(("macro", "news")):
            inner = _emitted(predictor, batch)
        outer = _emitted(predictor, batch)
    torch.testing.assert_close(
        inner, _emitted(predictor, _without(batch, ("news", "macro"))), rtol=0, atol=0
    )
    torch.testing.assert_close(
        outer, _emitted(predictor, _without(batch, ("news",))), rtol=0, atol=0
    )
    torch.testing.assert_close(_emitted(predictor, batch), plain, rtol=0, atol=0)
    assert predictor._knockout is None


@pytest.mark.parametrize("names", [(), ("prices",), ("news", "news"), ("sentiment",)])
def test_knockout_rejects_prices_repeats_unknown_and_empty_sets(names):
    predictor = model()
    predictor.eval()
    with pytest.raises(ValueError, match="knockout"):
        with predictor.modality_knockout(names):
            pass


def test_knockout_is_an_evaluation_mode_only():
    predictor = model()
    predictor.train()
    with pytest.raises(ValueError, match="knockout"):
        with predictor.modality_knockout(("news",)):
            pass
    predictor.eval()
    batch = priced_batch(rows=16)
    with predictor.modality_knockout(("news",)):
        predictor.train()
        with pytest.raises(ValueError, match="evaluación"):
            _emitted(predictor, batch)
