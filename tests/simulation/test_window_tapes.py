"""Cintas de la etapa de políticas y trampas sobre su procedencia, sin aprender.

Usa la edición sintética con el formato real, identificada como fixture, y recibos de las
ventanas US que evalúan 2021, 2022 y 2023. Las puntuaciones son sintéticas y no proceden
de ningún modelo. Las políticas son guionizadas y ningún paso ajusta parámetros.
"""

import dataclasses
import hashlib
from pathlib import Path

import numpy as np
import pytest

from mars_titan.environments.walk_forward_receipt import WalkForwardWindow
from mars_titan.evaluation.splits import build_folds
from mars_titan.simulation import campaign_stage, window_tapes
from mars_titan.simulation.environment import FinancialEnv
from tests.environments import walk_forward_fixture as receipts
from tests.simulation import unadjusted_edition_fixture as editions
from tests.simulation.unadjusted_edition_fixture import Asset

FOLDS = {fold["id"]: fold for fold in build_folds(receipts.protocol("US"))}
YEARS = {"fold-016": 2021, "fold-017": 2022, "fold-018": 2023}
JOB = dict(
    scope="US",
    market="US",
    predictor="parent",
    anchor="fold-018",
    window="fold-018",
    train=["fold-016"],
    validation="fold-017",
)
POLICIES = dict(
    environment=dict(dividend_payment_lag_sessions=0),
    predictor=dict(arms=["parent"]),
    universe=dict(max_assets=3),
)
STAGE = dict(
    policies=dict(
        environment=dict(
            capital=10000,
            cost_bps=10,
            participation=0.01,
            score_scale=0.01,
            ruin_penalty=-20,
            dividend_payment_lag_sessions=0,
        ),
        evaluation_costs_bps=[0, 10, 25],
        policies={},
        references=["cash"],
    )
)
# Liquidez decreciente por precio. DDD no tiene predicciones en la validación, EEE empieza
# a cotizar en 2023 y CCC tiene una fila sin verificar en la evaluación de 2023.
LIQUID = [Asset("DDD", base=60), Asset("AAA", base=50), Asset("BBB", base=40)]
LIQUID += [Asset("FFF", base=5), Asset("EEE", base=10, start=100)]


def edition(root, assets):
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(editions, "HISTORY_START", "2020-09-01")
        editions.write_edition(root, {"US": assets})
    return root


def symbols(window, assets):
    names = [a.symbol for a in assets if a.start is None]
    return [name for name in names if not (window == "fold-017" and name == "DDD")]


def source_for(assets, score=None):
    """Recibo y puntuaciones de cada ventana, con la huella de su tramo de evaluación."""
    cache = {}

    def source(scope, market, window, predictor):
        if window not in cache:
            year = YEARS[window]
            values = editions.predictions(
                market,
                symbols(window, assets),
                score=score,
                start=f"{year}-01-01",
                end=f"{year}-12-31",
            )
            index = list(FOLDS).index(window)
            cache[window] = receipts.window(market, index=index, values=values), values
        return cache[window]

    return source


def tapes(root, assets, *, source=None, output="stage"):
    root.mkdir(parents=True, exist_ok=True)
    path = (
        edition(root / "edition", assets) if not (root / "edition").exists() else root / "edition"
    )
    return campaign_stage._Tapes(
        POLICIES, source or source_for(assets), path, "edition", root / output
    )


def test_policy_windows_use_earlier_evaluations_and_anchors_end_before_their_carries():
    folds = list(FOLDS.values())
    rows = window_tapes.policy_windows(folds, 3)
    assert rows[0] == dict(
        window="fold-004", train=["fold-000", "fold-001", "fold-002"], validation="fold-003"
    )
    assert [row["window"] for row in rows] == [f"fold-{i:03d}" for i in range(4, 19)]
    scheduled = window_tapes.policy_schedule(rows, 3, FOLDS)
    assert [row["anchor"] for row in scheduled[:4]] == ["fold-004"] * 3 + ["fold-007"]
    assert [row["trained"] for row in scheduled[:4]] == [True, False, False, True]
    for bad in (0, 13, True):
        with pytest.raises(ValueError, match="ventanas de ajuste"):
            window_tapes.policy_windows(folds, bad)
    with pytest.raises(ValueError, match="ventanas suficientes"):
        window_tapes.policy_windows(folds[:4], 3)
    # Una validación posterior a la evaluación, o un ajuste solapado, se rechaza.
    swapped = dict(rows[0], validation="fold-004", window="fold-003")
    with pytest.raises(ValueError, match="posterior a su evaluación"):
        window_tapes.check_order(swapped, FOLDS)
    repeated = dict(rows[0], train=["fold-000", "fold-000", "fold-002"])
    with pytest.raises(ValueError, match="posterior a su evaluación"):
        window_tapes.check_order(repeated, FOLDS)
    with pytest.raises(ValueError, match="periodo"):
        window_tapes.policy_schedule(rows, 0, FOLDS)


def admission(**assets):
    return {
        name: dict(traded_value=value, predicted=predicted)
        for name, (value, predicted) in assets.items()
    }


def test_universe_rule_uses_only_training_and_validation_admission():
    train = [admission(A=(1, True), B=(1, True), C=(1, True), D=(1, False))]
    validation = admission(A=(5, True), B=(7, True), C=(7, True), D=(9, False), E=(99, True))
    # E no estaba admitido en el ajuste y D no tiene predicción en la validación.
    assert window_tapes.select_universe(train, validation, 4) == ["A", "B", "C"]
    # Empate de efectivo: decide el identificador.
    assert window_tapes.select_universe(train, validation, 2) == ["B", "C"]
    assert window_tapes.select_universe(train, validation, 1) == ["B"]
    with pytest.raises(ValueError, match="Ningún activo"):
        window_tapes.select_universe([admission(Z=(1, True))], validation, 3)
    for bad in (0, 4097, 1.0):
        with pytest.raises(ValueError, match="universo admite"):
            window_tapes.select_universe(train, validation, bad)


def test_tapes_share_one_universe_and_record_an_excluded_asset_as_failed_episodes(tmp_path):
    assets = [*LIQUID, Asset("CCC", base=30, unverified=(10,))]
    store = tapes(tmp_path, assets)
    opened = store.open(JOB)
    # DDD sin predicción en la validación, EEE sin cotizar en el ajuste y FFF por liquidez.
    assert opened.universe == ("US/AAA", "US/BBB", "US/CCC")
    assert [tape.assets for tape in opened.train] == [list(opened.universe)]
    assert opened.validation.assets == list(opened.universe)
    assert opened.evaluation is None and opened.paths["evaluation"] is None
    assert opened.failure == dict(
        reason="universe_assets_excluded", excluded={"US/CCC": "unverified_rows_in_tape"}
    )
    # Todos los brazos y referencias de la ventana reciben las mismas cintas.
    assert store.open(dict(JOB)).identity == opened.identity
    records = campaign_stage.evaluate_policy(opened, None, STAGE, "US", backend="python")
    assert [r["status"] for r in records] == ["failed"] * 3
    assert {r["reason"] for r in records} == {"universe_assets_excluded"}
    summary = campaign_stage.summarize(
        STAGE, {"job": dict(identity=dict(arm="cash"), evaluation=records)}
    )
    assert summary["cash"]["10"] == dict(
        episodes=1,
        completed=0,
        ruined=0,
        failed=1,
        failure_reasons={"universe_assets_excluded": 1},
        mean_liquidated_log_growth=None,
        denominator="completed",
    )
    # Al reanudar se leen el universo, las cintas y el fallo confirmados.
    resumed = tapes(tmp_path, assets).open(JOB)
    assert resumed.identity == opened.identity and resumed.universe == opened.universe


def test_confirmed_universe_and_tapes_reject_another_receipt(tmp_path):
    assets = LIQUID[1:3]
    tapes(tmp_path, assets).open(JOB)
    shifted = source_for(assets, score=lambda k, i: 0.02 * ((i + k) % 3 - 1))
    with pytest.raises(ValueError, match="universo"):
        tapes(tmp_path, assets, source=shifted).open(JOB)


def later(source, mapping):
    """Fuente adversaria que entrega el recibo y las puntuaciones de otra ventana."""

    def replaced(scope, market, window, predictor):
        return source(scope, market, mapping.get(window, window), predictor)

    return replaced


def test_a_policy_cannot_train_or_validate_with_predictions_of_a_later_fit(tmp_path):
    assets = LIQUID[1:3]
    honest = source_for(assets)
    # El tramo de validación de 2022 se sustituye por el de 2023, ajustado más tarde.
    for mapping in ({"fold-017": "fold-018"}, {"fold-016": "fold-018"}):
        with pytest.raises(ValueError, match="otra ventana"):
            tapes(tmp_path, assets, source=later(honest, mapping), output=str(mapping)).open(JOB)
    # Las puntuaciones de 2022 presentadas con el recibo de 2023 no cumplen su huella.
    receipt, _ = honest("US", "US", "fold-018", "parent")
    _, values = honest("US", "US", "fold-017", "parent")
    with pytest.raises(ValueError):
        window_tapes.build_segment_tape(
            tmp_path / "edition", receipt, values, market="US", role="train", lag=0
        )
    # Un recibo construido a mano, sin `read_window_receipt`, que declara un ajuste con
    # etiquetas del propio tramo, se rechaza antes de leer la edición.
    _, values = honest("US", "US", "fold-018", "parent")
    start = receipt.segment("evaluation")[0]
    forged = dataclasses.replace(receipt, labels_used_until=start)
    assert isinstance(forged, WalkForwardWindow)
    with pytest.raises(ValueError, match="etiquetas posteriores"):
        window_tapes.build_segment_tape(
            tmp_path / "nowhere", forged, values, market="US", role="evaluation", lag=0
        )


def test_tapes_from_another_market_are_rejected(tmp_path):
    assets = LIQUID[1:3]

    def chinese(scope, market, window, predictor):
        receipt, values = source_for(assets)(scope, market, window, predictor)
        return dataclasses.replace(receipt, market="CN"), values

    with pytest.raises(ValueError, match="otro mercado"):
        tapes(tmp_path, assets, source=chinese).open(JOB)


def run_policy(tape, policy):
    env = FinancialEnv(tape)
    observation, _ = env.reset(seed=0)
    steps = []
    while not env.done:
        action = policy(observation, len(steps))
        following, reward, _, _, info = env.step(action)
        steps.append(dict(observation=observation, action=action, reward=reward, info=info))
        observation = following
    return steps


def peeking(observation, step):
    """Guion que convierte toda la observación en una acción: vería cualquier fuga."""
    return hashlib.sha256(observation.tobytes()).digest()[0] % 6


def test_a_scripted_policy_cannot_see_the_next_price_or_prediction(tmp_path):
    cut = 120
    assets = LIQUID[1:3]
    changed = [
        dataclasses.replace(
            asset,
            overrides={
                i: dict(open=asset.base * 1.5, close=asset.base * 1.6) for i in range(cut + 1, 250)
            },
        )
        for asset in assets
    ]
    one = tapes(tmp_path / "one", assets).open(JOB).evaluation
    future = source_for(changed, score=lambda k, i: 0.05 if k > cut else 0.01 * ((i + k) % 5 - 1))
    two = tapes(tmp_path / "two", changed, source=future).open(JOB).evaluation
    assert one.sha256 != two.sha256
    first, second = run_policy(one, peeking), run_policy(two, peeking)
    for t in range(cut + 1):
        np.testing.assert_array_equal(first[t]["observation"], second[t]["observation"])
        assert first[t]["action"] == second[t]["action"]
        if t < cut:
            assert first[t]["reward"] == second[t]["reward"]
    # El cambio es visible en cuanto llega: la observación siguiente ya difiere.
    assert not np.array_equal(first[cut + 1]["observation"], second[cut + 1]["observation"])
    # La orden de la decisión `cut` se dimensiona con su cierre y se ejecuta después.
    quantities = [
        [(x["asset"], x["quantity"]) for x in s[cut]["info"]["trades"]] for s in (first, second)
    ]
    assert quantities[0] == quantities[1]


def test_maximum_exposure_only_trades_universe_assets_at_executable_opens(tmp_path):
    assets = [
        Asset("AAA", base=50, zero_volume=(20, 21), missing=(30,), off_grid_open=(40,)),
        Asset("BBB", base=40),
        Asset("GGG", base=45, unverified=(5,)),
    ]
    opened = tapes(tmp_path, assets).open(dict(JOB))
    # GGG queda fuera por su fila sin verificar en 2023 y la evaluación falla entera.
    assert opened.evaluation is None
    clean = tapes(tmp_path / "clean", assets[:2]).open(JOB).evaluation
    steps = run_policy(clean, lambda observation, step: 5 if step % 3 else 1)
    traded = 0
    for t, step in enumerate(steps):
        for trade in step["info"]["trades"]:
            i = clean.assets.index(trade["asset"])
            assert trade["asset"] in clean.assets and np.isfinite(clean.prices[t + 1, i, 0])
            assert trade["price"] == clean.prices[t + 1, i, 0] and clean.prices[t, i, 4] > 0
            traded += 1
    assert traded > 0
    # Las sesiones sin volumen, ausentes o con apertura fuera de rejilla no ejecutan.
    i = clean.assets.index("US/AAA")
    blocked = {20, 21, 30, 40}
    assert all(np.isnan(clean.prices[t, i, 0]) for t in blocked)


def test_universe_is_saved_with_its_identity(tmp_path):
    assets = LIQUID[1:3]
    tapes(tmp_path, assets).open(JOB)
    path = Path(tmp_path / "stage/universes/US/US/fold-018.json")
    record = campaign_stage.read_manifest(path, 1024**2)[0]
    assert record["assets"] == ["US/AAA", "US/BBB"]
    assert record["identity"]["rule"] == window_tapes.UNIVERSE_RULE
    assert list(record["identity"]["segments"]) == ["fold-016", "fold-017"]
