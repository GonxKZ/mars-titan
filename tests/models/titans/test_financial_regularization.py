"""Dropout declarado del predictor financiero: fusión multimodal y retirada de modalidades.

Sin declararlos, la identidad y el cálculo no cambian. Declarados, solo actúan en ajuste: en
evaluación el predictor emite exactamente lo mismo que sin ellos, y una modalidad retirada
sigue la misma ruta que una ausencia real.
"""

import pytest
import torch

from mars_titan.models.baselines.multimodal import MODALITIES
from mars_titan.models.titans.financial import FUSION_DROPOUTS, MAX_DROPOUT, MODALITY_DROPOUT
from tests.models.titans.test_asset_groups import model, priced_batch

PRICE = MODALITIES.index("prices")


def _emitted(predictor, batch):
    return predictor.prepare(batch, predictor.initial_state(batch.flow_ids)).point_predictions


def test_undeclared_options_keep_identity_and_declared_ones_enter_it():
    base, explicit = model(), model(fusion_dropout=None, modality_dropout=None)
    assert explicit.config.identity() == base.config.identity()
    assert base.config.identity()["dropout"] == 0.0
    assert "modality_dropout" not in base.config.identity()
    fused = model(fusion_dropout=0.1).config.identity()
    assert fused["dropout"] == 0.1 and "modality_dropout" not in fused
    dropped = model(modality_dropout=0.2).config.identity()
    assert dropped["dropout"] == 0.0
    assert dropped["modality_dropout"]["probability"] == 0.2
    assert dropped["modality_dropout"]["phase"] == "fit_only"


@pytest.mark.parametrize("value", [0, 0.0, -0.1, MAX_DROPOUT + 0.01, True, float("nan"), "0.1"])
@pytest.mark.parametrize("name", ["fusion_dropout", "modality_dropout"])
def test_probabilities_outside_the_declared_interval_are_rejected(name, value):
    with pytest.raises(ValueError, match=name):
        model(**{name: value})


@pytest.mark.parametrize("value", [0.05, 0.15, 0.3])
def test_fusion_dropout_keeps_the_experimental_grid_of_the_references(value):
    assert value not in FUSION_DROPOUTS
    with pytest.raises(ValueError, match="fusion_dropout"):
        model(fusion_dropout=value)
    model(modality_dropout=value)


def test_declared_dropout_keeps_initial_weights_and_evaluation_outputs_exactly():
    batch = priced_batch()
    base = model()
    for changes in (dict(fusion_dropout=0.2), dict(modality_dropout=0.4)):
        declared = model(**changes)
        for (name, a), (_, b) in zip(
            base.named_parameters(), declared.named_parameters(), strict=True
        ):
            assert torch.equal(a, b), name
        base.eval(), declared.eval()
        torch.manual_seed(0)
        expected = _emitted(base, batch)
        torch.manual_seed(1)
        actual = _emitted(declared, batch)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_fit_masks_depend_on_the_fit_epoch_and_never_on_the_global_generator():
    batch = priced_batch(rows=256)
    for changes in (dict(fusion_dropout=0.2), dict(modality_dropout=0.4)):
        predictor = model(**changes)
        predictor.train()
        predictor.set_fit_epoch(1)
        torch.manual_seed(7)
        before = torch.get_rng_state()
        first = _emitted(predictor, batch)
        # Las máscaras tienen su propio generador: el flujo global queda intacto.
        assert torch.equal(torch.get_rng_state(), before)
        torch.manual_seed(8)
        repeated = _emitted(predictor, batch)
        torch.testing.assert_close(repeated, first, rtol=0, atol=0)
        predictor.set_fit_epoch(2)
        other = _emitted(predictor, batch)
        assert not torch.equal(other, first)
        predictor.eval()
        torch.testing.assert_close(
            _emitted(predictor, batch), _emitted(predictor, batch), rtol=0, atol=0
        )
    with pytest.raises(ValueError, match="época"):
        model().set_fit_epoch(-1)


def test_added_absences_follow_the_route_of_real_ones_and_never_remove_prices():
    batch = priced_batch(rows=16)
    keep = torch.ones_like(batch.presence)
    keep[::2, MODALITIES.index("news")] = False
    keep[1::3, MODALITIES.index("macro")] = False
    derived = batch.with_absences(keep, MODALITY_DROPOUT)
    assert torch.equal(derived.presence, batch.presence & keep)
    assert derived.input_digest != batch.input_digest
    assert torch.equal(derived.inputs["prices"], batch.inputs["prices"])
    for column, name in enumerate(MODALITIES):
        if name == "prices":
            continue
        dropped = ~keep[:, column]
        assert torch.count_nonzero(derived.inputs[name][dropped]) == 0
        torch.testing.assert_close(
            derived.inputs[name][~dropped], batch.inputs[name][~dropped], rtol=0, atol=0
        )
    # Una fila con la modalidad retirada emite lo mismo que si llegara ausente desde el corpus.
    predictor = model()
    predictor.eval()
    torch.testing.assert_close(
        _emitted(predictor, derived),
        _emitted(predictor, batch.with_absences(keep, "corpus_absence")),
        rtol=0,
        atol=0,
    )
    keep[0, PRICE] = False
    with pytest.raises(ValueError, match="no retiran el precio"):
        batch.with_absences(keep, MODALITY_DROPOUT)


def test_modality_dropout_retires_present_modalities_at_its_rate_in_fit_mode_only():
    batch = priced_batch(rows=256)
    predictor = model(modality_dropout=0.25)
    seen = []
    original = type(batch).with_absences

    def spy(self, keep, transform):
        seen.append((keep.clone(), transform))
        return original(self, keep, transform)

    type(batch).with_absences = spy
    try:
        predictor.train()
        torch.manual_seed(3)
        _emitted(predictor, batch)
        predictor.eval()
        _emitted(predictor, batch)
    finally:
        type(batch).with_absences = original
    assert len(seen) == 1 and seen[0][1] == MODALITY_DROPOUT
    keep = seen[0][0]
    assert bool(keep[:, PRICE].all())
    others = [column for column, name in enumerate(MODALITIES) if name != "prices"]
    rate = 1 - keep[:, others].float().mean().item()
    # 1.024 sorteos con semilla fija: ±0,05 son casi cuatro desviaciones típicas.
    assert abs(rate - 0.25) < 0.05


@pytest.fixture(scope="module")
def chronological(tmp_path_factory):
    from tests.training.test_financial_run import corpus

    return corpus(tmp_path_factory.mktemp("regularized"))


# Con dropout, la repetición por bloques necesita el instante entero en un bloque: las máscaras
# se sortean con la huella del lote y un bloque parcial tendría otras. El corpus tiene tres flujos.
@pytest.mark.parametrize("accumulation", [None, 8], ids=["segment_graph", "flow_block_replay"])
def test_resume_with_dropout_reproduces_the_continuous_run(
    chronological, tmp_path, monkeypatch, accumulation
):
    from mars_titan.models.titans.financial import FinancialConfig, FinancialPredictor
    from tests.training.test_financial_run import StopAtStep, named_records, trainer

    # El recorrido exige el fastpath de atención declarado y desactivado.
    previous = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    monkeypatch.setattr(torch.backends.mha, "get_fastpath_enabled", lambda: False)
    _, streams = chronological

    def regularized():
        config = FinancialConfig(
            streams["train"].specification(),
            variant="mac_online",
            hidden_size=32,
            seed=42,
            fusion_dropout=0.1,
            modality_dropout=0.3,
        )
        return FinancialPredictor(config, dtype=torch.float64)

    options = dict(epochs=2, fixed=True, checkpoint_updates=2)
    if accumulation is not None:
        # La acumulación repite cada bloque con grafo desde el inicio del tramo: las máscaras
        # de esa repetición deben ser las de la pasada sin grafo.
        options["accumulation_rows"] = accumulation
    continuous = trainer(streams, tmp_path / "continuous", model=regularized(), **options)
    expected = continuous.run()
    plain = trainer(streams, tmp_path / "plain", **options).run()
    # El dropout actúa en ajuste: la historia de validación cambia respecto al modelo sin él.
    assert expected["history"] != plain["history"]
    stop = StopAtStep(15)
    first = trainer(streams, tmp_path / "interrupted", model=regularized(), **options)
    stop.trainer = first
    assert first.run(stop=stop)["status"] == "paused"
    second = trainer(streams, tmp_path / "interrupted", model=regularized(), **options)
    resumed = second.run(resume=True)
    assert resumed["status"] == "completed"
    joined, reference = named_records(first) + named_records(second), named_records(continuous)
    assert len(joined) == len(reference) == second.optimizer.calls
    for left, right in zip(joined, reference, strict=True):
        for key in left:
            if left[key] is None:
                assert right[key] is None
            else:
                torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)
    assert resumed["history"] == expected["history"]
    torch.backends.mha.set_fastpath_enabled(previous)


def _regularized_trainer(streams, output, monkeypatch, **options):
    from mars_titan.models.titans.financial import FinancialConfig, FinancialPredictor
    from tests.training.test_financial_run import trainer

    monkeypatch.setattr(torch.backends.mha, "get_fastpath_enabled", lambda: False)
    config = FinancialConfig(
        streams["train"].specification(),
        variant="mac_online",
        hidden_size=32,
        seed=42,
        fusion_dropout=0.1,
        modality_dropout=0.3,
    )
    model = FinancialPredictor(config, dtype=torch.float64)
    return trainer(streams, output, model=model, epochs=2, fixed=True, **options)


def test_the_replay_with_dropout_uses_the_masks_of_the_emission(
    chronological, tmp_path, monkeypatch
):
    from tests.training.test_financial_run import named_records

    previous = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    _, streams = chronological
    full = _regularized_trainer(streams, tmp_path / "full", monkeypatch)
    expected = full.run()
    blocks = _regularized_trainer(streams, tmp_path / "blocks", monkeypatch, accumulation_rows=8)
    report = blocks.run()
    # Mismas predicciones y, con las mismas máscaras, el mismo gradiente que el tramo entero.
    assert blocks.audit == full.audit
    assert [h["validation"] for h in report["history"]] == [
        h["validation"] for h in expected["history"]
    ]
    for left, right in zip(named_records(blocks), named_records(full), strict=True):
        for key, value in right.items():
            if value is None:
                assert left[key] is None, key
            else:
                torch.testing.assert_close(left[key], value, rtol=1e-10, atol=1e-13)
    torch.backends.mha.set_fastpath_enabled(previous)


def test_a_partial_block_with_dropout_is_rejected(chronological, tmp_path, monkeypatch):
    previous = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    _, streams = chronological
    engine = _regularized_trainer(streams, tmp_path / "partial", monkeypatch, accumulation_rows=1)
    with pytest.raises(ValueError, match="dropout por lote"):
        engine.run()
    torch.backends.mha.set_fastpath_enabled(previous)
