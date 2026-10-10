"""Matriz cuantizada compartida y rutas de predicción de XGBoost en CUDA, sin ajustar árboles.

Los bosques se escriben a mano en el formato JSON de XGBoost con umbrales en los cortes de
la propia matriz, como los que produce `hist`. Ninguna prueba llama a `xgb.train`. Se
comprueba que la matriz construida dos veces es idéntica, que predecir sobre ella equivale
bit a bit a predecir sobre los float32 y que la ubicación o el reparto de los bloques de
validación no cambia ninguna predicción.
"""

import hashlib
import importlib

import numpy as np
import pytest

from mars_titan.models.baselines import external_boosting as external
from mars_titan.models.baselines.boosting_selection import ResidentValidation, session_validation
from tests.models.hand_forest import hand_forest

pytestmark = pytest.mark.skipif(
    any(importlib.util.find_spec(name) is None for name in ("cupy", "xgboost")),
    reason="La comprobación CUDA requiere el extra boosting",
)

ROWS, FEATURES, MAX_BIN, BATCH = 6000, 24, 32, 1024


def cuda():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA no disponible, sin alternativa CPU para la ruta externa")


def values(seed=20261009):
    """Valores repetidos, ceros de modalidad ausente y bits, como en la edición con máscaras."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(ROWS, FEATURES)).astype(np.float32)
    x[rng.random(x.shape) < 0.3] = 0.0
    x[:, :4] = np.round(x[:, :4], 1)
    x[:, -3:] = rng.integers(0, 2, size=(ROWS, 3))
    return x, rng.normal(size=ROWS)


def factory(x, y, size=100):
    def blocks():
        for start in range(0, len(x), size):
            yield x[start : start + size], y[start : start + size]

    return blocks


def build(tmp_path, name="pages", **changes):
    x, y = values()
    options = dict(
        expected_rows=ROWS,
        max_bin=MAX_BIN,
        max_batch_bytes=64 * 1024,
        on_host=False,
        max_disk_cache_bytes=1024**3,
    )
    return external.build_external_matrix(factory(x, y), tmp_path / name, **(options | changes))


def forest(cuts, **options):
    return hand_forest(cuts, FEATURES, **options)


def wrapped(booster, matrix):
    audit = dict(rows=matrix.rows, rounds=booster.num_boosted_rounds())
    return external.ExternalBoostingModel(booster, matrix.rows, matrix.features, audit)


def pages_digest(matrix):
    digest = hashlib.sha256()
    for path in sorted(matrix.directory.rglob("*")):
        if path.is_file():
            digest.update(path.read_bytes())
    return digest.hexdigest()


def test_two_constructions_produce_the_same_cuts_pages_and_audit(tmp_path):
    cuda()
    first, second = build(tmp_path, "first"), build(tmp_path, "second")
    try:
        for a, b in zip(first.data.get_quantile_cut(), second.data.get_quantile_cut(), strict=True):
            np.testing.assert_array_equal(a, b)
        assert pages_digest(first) == pages_digest(second)
        assert first.audit == second.audit and first.construction == second.construction
        assert first.audit["completed_pass_rows"] == [ROWS, ROWS]
        assert first.disk_bytes() > 0
    finally:
        first.close()
        second.close()
    assert not first.directory.exists() and not second.directory.exists()


def test_batch_bytes_change_the_cuts_so_they_belong_to_the_construction(tmp_path):
    """El reparto en lotes altera el bosquejo: no es un parámetro solo de rendimiento."""
    cuda()
    base, smaller = build(tmp_path, "base"), build(tmp_path, "small", max_batch_bytes=16 * 1024)
    try:
        assert base.construction != smaller.construction
        changed = any(
            not np.array_equal(a, b)
            for a, b in zip(
                base.data.get_quantile_cut(), smaller.data.get_quantile_cut(), strict=True
            )
        )
        assert changed
    finally:
        base.close()
        smaller.close()


def test_quantized_training_rows_predict_like_their_float32_values(tmp_path):
    cuda()
    import cupy as cp

    x, _ = values()
    matrix = build(tmp_path)
    try:
        booster = forest(matrix.data.get_quantile_cut())
        model = wrapped(booster, matrix)
        passes = list(matrix.iterator.completed)
        quantized = model.predict_matrix(matrix)
        raw = np.concatenate(
            [model.predict(x[start : start + BATCH]) for start in range(0, ROWS, BATCH)]
        )
        np.testing.assert_array_equal(quantized, raw)
        assert matrix.iterator.completed == passes
        # Un solo bloque o bloques de 1024 filas y la copia previa en la GPU dan los mismos bits.
        whole = model.predict_device(cp.asarray(x))
        np.testing.assert_array_equal(whole, raw)
        # Con umbrales entre cortes la equivalencia ya no está garantizada: la prueba la detecta.
        shifted = wrapped(forest(matrix.data.get_quantile_cut(), on_cuts=False), matrix)
        assert not np.array_equal(
            shifted.predict_matrix(matrix),
            np.concatenate(
                [shifted.predict(x[start : start + BATCH]) for start in range(0, ROWS, BATCH)]
            ),
        )
    finally:
        matrix.close()


def test_resident_validation_matches_the_block_metric_at_any_placement(tmp_path):
    cuda()
    x, y = values(seed=5)
    rng = np.random.default_rng(9)
    markets = np.where(rng.random(ROWS) < 0.3, "CN", "US")
    moments = (1_300_000_000_000_000 + (np.arange(ROWS) // 40) * 86_400_000_000).astype(
        "datetime64[us]"
    )

    def blocks():
        for start in range(0, ROWS, BATCH):
            stop = start + BATCH
            yield x[start:stop], y[start:stop], list(markets[start:stop]), moments[start:stop]

    matrix = build(tmp_path)
    try:
        model = wrapped(forest(matrix.data.get_quantile_cut(), trees=25), matrix)
        reference = session_validation(model, blocks, expected_rows=ROWS)
        resident = ResidentValidation(blocks, expected_rows=ROWS, max_bytes=1024**3)
        assert session_validation(model, resident, expected_rows=ROWS) == reference
        for budget in (BATCH * FEATURES * 4 * 2, 1024**3):
            placed = resident.place(budget)
            assert 0 < placed <= budget
            assert session_validation(model, resident, expected_rows=ROWS) == reference
        resident.release_device()
        assert resident.device_bytes == 0 and not resident.device
        with pytest.raises(ValueError, match="población"):
            session_validation(model, resident, expected_rows=ROWS + 1)
    finally:
        matrix.close()


def test_resident_validation_rejects_its_budget_and_nonfinite_values():
    x, y = values()

    def blocks(values_=x):
        yield values_[:BATCH], y[:BATCH], ["US"] * BATCH, np.arange(BATCH)

    with pytest.raises(ValueError, match="presupuesto"):
        ResidentValidation(blocks, expected_rows=BATCH, max_bytes=1024)
    broken = x.copy()
    broken[0, 0] = np.inf
    with pytest.raises(ValueError, match="presupuesto|dimensiones"):
        ResidentValidation(lambda: blocks(broken), expected_rows=BATCH, max_bytes=1024**3)
    with pytest.raises(ValueError, match="población"):
        ResidentValidation(blocks, expected_rows=BATCH + 1, max_bytes=1024**3)


def test_train_metrics_from_the_matrix_equal_those_of_the_reader_batches(tmp_path):
    """Mismos errores, sesiones y sumas que `_predict` sobre los float32 del lector."""
    cuda()
    from mars_titan.training.external_corpus import _matrix_metrics, _TrainingMatrix, _TrainRows
    from mars_titan.training.tabular_corpus import _Errors

    x, y = values()
    rng = np.random.default_rng(21)
    markets = np.where(rng.random(ROWS) < 0.3, "CN", "US")
    moments = 1_300_000_000_000_000 + (np.arange(ROWS) // 30) * 86_400_000_000
    rows = _TrainRows(ROWS, china=markets == "CN", moments=moments, target=y)
    matrix = build(tmp_path)
    try:
        model = wrapped(forest(matrix.data.get_quantile_cut(), trees=30), matrix)
        expected = _Errors()
        for start in range(0, ROWS, BATCH):
            part = slice(start, start + BATCH)
            expected.add(
                model.predict(x[part]),
                y[part],
                list(markets[part]),
                moments[part].astype("datetime64[us]"),
            )
        assert _matrix_metrics(model, _TrainingMatrix(matrix, rows), BATCH) == expected.summary()
    finally:
        matrix.close()


def growing(forest_, rounds):
    """Un mismo booster que gana un árbol por ronda, como el de `xgb.train`, sin ajustar."""
    import xgboost as xgb

    booster = xgb.Booster()
    for count in range(1, rounds + 1):
        booster.load_model(forest_[:count].save_raw("json"))
        booster.set_param(dict(device="cuda:0", nthread=4))
        yield count, booster


def validation_blocks(seed=5):
    x, y = values(seed=seed)
    rng = np.random.default_rng(9)
    markets = np.where(rng.random(ROWS) < 0.3, "CN", "US")
    moments = (1_300_000_000_000_000 + (np.arange(ROWS) // 40) * 86_400_000_000).astype(
        "datetime64[us]"
    )

    def blocks():
        for start in range(0, ROWS, BATCH):
            stop = start + BATCH
            yield x[start:stop], y[start:stop], list(markets[start:stop]), moments[start:stop]

    return blocks


@pytest.mark.parametrize("placement", [0, 2 * BATCH * FEATURES * 4, 1024**3])
def test_incremental_rounds_match_the_full_prediction_at_every_round(tmp_path, placement):
    cuda()
    blocks = validation_blocks()
    matrix = build(tmp_path)
    try:
        trees = forest(matrix.data.get_quantile_cut(), trees=60, depth=5)
    finally:
        matrix.close()
    resident = ResidentValidation(blocks, expected_rows=ROWS, max_bytes=1024**3)
    resident.place(placement)
    for count, booster in growing(trees, 60):
        model = external.ExternalBoostingModel(booster, ROWS, FEATURES, dict(rounds=count))
        reference = session_validation(model, blocks, expected_rows=ROWS)
        assert session_validation(model, resident, expected_rows=ROWS) == reference
        assert resident.incremental and resident.sums["count"] == count
    # Otro booster o una ronda saltada reinician el estado con la predicción completa.
    other = external.ExternalBoostingModel(trees[:30], ROWS, FEATURES, dict(rounds=30))
    assert session_validation(other, resident, expected_rows=ROWS) == session_validation(
        other, blocks, expected_rows=ROWS
    )
    assert resident.sums["booster"] is other.booster and resident.sums["count"] == 30


def test_incremental_state_falls_back_or_fails_when_it_cannot_reproduce(tmp_path, monkeypatch):
    cuda()
    from mars_titan.models.baselines import boosting_selection

    blocks = validation_blocks(seed=8)
    matrix = build(tmp_path)
    try:
        trees = forest(matrix.data.get_quantile_cut(), trees=6, depth=4)
    finally:
        matrix.close()
    resident = ResidentValidation(blocks, expected_rows=ROWS, max_bytes=1024**3)
    resident.place(1024**3)
    rounds = growing(trees, 6)
    count, booster = next(rounds)
    model = external.ExternalBoostingModel(booster, ROWS, FEATURES, dict(rounds=count))
    session_validation(model, resident, expected_rows=ROWS)
    # Un estado alterado se detecta en la ronda siguiente con el primer bloque.
    resident.sums["values"][0][0] += 1.0
    count, booster = next(rounds)
    with pytest.raises(ValueError, match="incremental"):
        session_validation(model, resident, expected_rows=ROWS)
    # Si la suma no reproduce la predicción al reiniciar, se usa siempre la completa.
    monkeypatch.setattr(boosting_selection, "_base_score", lambda _: np.float32(0.5))
    for _ in rounds:
        expected = session_validation(model, blocks, expected_rows=ROWS)
        assert session_validation(model, resident, expected_rows=ROWS) == expected
    assert resident.incremental is False and resident.sums is None
