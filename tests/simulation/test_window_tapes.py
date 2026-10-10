"""Cintas de la etapa de políticas y trampas sobre su procedencia, sin aprender.

Usa la edición sintética con el formato real, identificada como fixture, y recibos de las
ventanas US que evalúan 2021, 2022 y 2023. Las puntuaciones son sintéticas y no proceden
de ningún modelo. Las políticas son guionizadas y ningún paso ajusta parámetros.
"""

import copy
import dataclasses
import hashlib
from pathlib import Path

import numpy as np
import pytest

from mars_titan.data import prediction_files
from mars_titan.environments.walk_forward_receipt import WalkForwardWindow
from mars_titan.evaluation.splits import build_folds
from mars_titan.simulation import campaign_stage, window_tapes
from mars_titan.simulation.environment import FinancialEnv
from mars_titan.simulation.evaluation import fixed_policy
from mars_titan.simulation.market import MarketTape
from mars_titan.simulation.reconstructed_tape import read_edition
from tests.environments import walk_forward_fixture as receipts
from tests.simulation import unadjusted_edition_fixture as editions
from tests.simulation.unadjusted_edition_fixture import Asset

FOLDS = {fold["id"]: fold for fold in build_folds(receipts.protocol("US"))}
YEARS = {"fold-016": 2021, "fold-017": 2022, "fold-018": 2023}
JOB = dict(
    scope="US",
    market="US",
    predictor="parent",
    arm="cash",
    anchor="fold-018",
    window="fold-018",
    train=["fold-016"],
    validation="fold-017",
)
POLICIES = dict(
    environment=dict(dividend_payment_lag_sessions=0),
    universe=dict(max_assets=3),
)
STAGE = dict(
    predictors=["parent"],
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
    ),
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
    """Recibo y lector de las puntuaciones de cada ventana, con la huella de su evaluación."""
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
        receipt, values = cache[window]
        return receipt, lambda: values

    return source


def tapes(root, assets, *, source=None, output="stage"):
    root.mkdir(parents=True, exist_ok=True)
    path = (
        edition(root / "edition", assets) if not (root / "edition").exists() else root / "edition"
    )
    edition_id = read_edition(path)["edition_id"]
    return campaign_stage._Tapes(
        POLICIES, source or source_for(assets), path, edition_id, root / output, "parent"
    )


EXPANDING = dict(rule=window_tapes.EXPANDING, minimum=3, maximum=17)
FIXED = dict(rule=window_tapes.FIXED, minimum=3, maximum=3)


def test_policy_windows_use_earlier_evaluations_and_anchors_end_before_their_carries():
    folds = list(FOLDS.values())
    rows = window_tapes.policy_windows(folds, EXPANDING)
    first = dict(
        window="fold-004", train=["fold-000", "fold-001", "fold-002"], validation="fold-003"
    )
    # La primera ventana no cambia: tiene exactamente el mínimo de evaluaciones previas.
    assert rows[0] == window_tapes.policy_windows(folds, FIXED)[0] == first
    assert [row["window"] for row in rows] == [f"fold-{i:03d}" for i in range(4, 19)]
    # Cada ventana se ajusta con todas las evaluaciones anteriores a su validación.
    for index, row in enumerate(rows, start=4):
        assert row["train"] == [f"fold-{i:03d}" for i in range(index - 1)]
        assert row["validation"] == f"fold-{index - 1:03d}"
    assert len(rows[-1]["train"]) == 17
    fixed = window_tapes.policy_windows(folds, FIXED)
    assert all(
        row["train"] == expanded["train"][-3:] for row, expanded in zip(fixed, rows, strict=True)
    )
    scheduled = window_tapes.policy_schedule(rows, 3, FOLDS)
    assert [row["anchor"] for row in scheduled[:4]] == ["fold-004"] * 3 + ["fold-007"]
    assert [row["trained"] for row in scheduled[:4]] == [True, False, False, True]
    # Con un máximo, las anclas tardías se ajustan con las evaluaciones más recientes.
    capped = window_tapes.policy_windows(folds, dict(EXPANDING, maximum=5))
    for row, expanded in zip(capped, rows, strict=True):
        assert row == dict(expanded, train=expanded["train"][-5:])
    assert [len(row["train"]) for row in capped] == [3, 4, 5, *[5] * 12]
    bad_rules = [3, dict(EXPANDING, minimum=0), dict(EXPANDING, minimum=13)]
    bad_rules += [dict(EXPANDING, minimum=True), dict(EXPANDING, rule="all_windows")]
    bad_rules += [dict(EXPANDING, extra=1), dict(rule=window_tapes.EXPANDING, minimum=3)]
    bad_rules += [dict(EXPANDING, maximum=2), dict(EXPANDING, maximum=16.0)]
    # La regla fija no admite otro máximo y la expansión necesita margen sobre el mínimo.
    bad_rules += [dict(FIXED, maximum=4), dict(EXPANDING, maximum=3)]
    for bad in bad_rules:
        with pytest.raises(ValueError, match="ventanas de ajuste"):
            window_tapes.policy_windows(folds, bad)
    with pytest.raises(ValueError, match="ventanas suficientes"):
        window_tapes.policy_windows(folds[:4], EXPANDING)
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
        STAGE, {"job": dict(identity=dict(predictor="parent", arm="cash"), evaluation=records)}
    )
    assert summary["parent"]["cash"]["10"] == dict(
        episodes=1,
        completed=0,
        ruined=0,
        failed=1,
        failure_reasons={"universe_assets_excluded": 1},
        mean_liquidated_log_growth=None,
        denominator="completed",
    )
    # Un informe que presenta un episodio completo sobre la cinta fallida se rechaza.
    completed = dict(records[0], status="completed", reason=None, net_return=0.0)
    completed.update(liquidated_net_return=0.0, max_drawdown=0.0)
    report = dict(status="completed", transitions=0, updates=0, selection=None, policy=None)
    report["evaluation"] = [completed, *records[1:]]
    with pytest.raises(ValueError, match="cinta de evaluación fallida"):
        campaign_stage.check_report(STAGE, dict(id="job", kind="reference"), report, opened)
    # Al reanudar se leen el universo, las cintas y el fallo confirmados.
    resumed = tapes(tmp_path, assets).open(JOB)
    assert resumed.identity == opened.identity and resumed.universe == opened.universe


def test_only_the_evaluation_tape_may_lose_a_universe_asset(tmp_path, monkeypatch):
    assets = LIQUID[1:3]
    build = window_tapes.build_segment_tape

    def losing(edition, receipt, values, *, market, role, lag, symbols=None):
        # La cinta de ajuste pierde un activo que el universo admitió.
        if symbols is not None and role == "train":
            symbols = symbols[1:]
        return build(edition, receipt, values, market=market, role=role, lag=lag, symbols=symbols)

    monkeypatch.setattr(window_tapes, "build_segment_tape", losing)
    with pytest.raises(ValueError, match="no es admisible en train"):
        tapes(tmp_path, assets).open(JOB)


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
    values = honest("US", "US", "fold-017", "parent")[1]()
    with pytest.raises(ValueError):
        window_tapes.build_segment_tape(
            tmp_path / "edition", receipt, values, market="US", role="train", lag=0
        )
    # Un recibo construido a mano, sin `read_window_receipt`, que declara un ajuste con
    # etiquetas del propio tramo, se rechaza antes de leer la edición.
    values = honest("US", "US", "fold-018", "parent")[1]()
    start = receipt.segment("evaluation")[0]
    forged = dataclasses.replace(receipt, labels_used_until=start)
    assert isinstance(forged, WalkForwardWindow)
    with pytest.raises(ValueError, match="etiquetas posteriores"):
        window_tapes.build_segment_tape(
            tmp_path / "nowhere", forged, values, market="US", role="evaluation", lag=0
        )


def test_a_later_receipt_renamed_as_an_earlier_window_is_rejected_by_its_segments(tmp_path):
    assets = LIQUID[1:3]
    honest = source_for(assets)

    def renamed(scope, market, window, predictor):
        # El recibo de 2023 se presenta con el nombre de la ventana de ajuste de 2021.
        if window != "fold-016":
            return honest(scope, market, window, predictor)
        receipt, values = honest(scope, market, "fold-018", predictor)
        return dataclasses.replace(receipt, fold=window), values

    with pytest.raises(ValueError, match="tramos posteriores a su evaluación"):
        tapes(tmp_path, assets, source=renamed).open(JOB)


def without(assets, missing):
    """Fuente en la que algunos predictores no emitieron filas del mercado en un tramo."""
    honest = source_for(assets)

    def source(scope, market, window, predictor):
        if (predictor, window) in missing:
            return receipts.window(market, index=list(FOLDS).index(window)), None
        return honest(scope, market, window, predictor)

    return source


def test_a_predictor_without_evaluation_predictions_fails_its_episodes(tmp_path):
    assets = LIQUID[1:3]
    other = dict(JOB, predictor="other")
    store = tapes(tmp_path, assets, source=without(assets, {("other", "fold-018")}))
    opened = store.open(other)
    reason = window_tapes.NO_PREDICTIONS
    assert opened.unfit is None and opened.evaluation is None
    assert opened.failure == dict(reason=reason, window="fold-018")
    records = campaign_stage.evaluate_policy(opened, None, STAGE, "US", backend="python")
    assert [(r["status"], r["reason"]) for r in records] == [("failed", reason)] * 3
    # El predictor del universo conserva su evaluación y el universo es el mismo.
    assert store.open(JOB).evaluation is not None
    assert store.open(JOB).universe == opened.universe


def test_a_predictor_without_training_predictions_has_no_policy_to_fit(tmp_path):
    assets = LIQUID[1:3]
    other = dict(JOB, predictor="other")
    opened = tapes(tmp_path, assets, source=without(assets, {("other", "fold-016")})).open(other)
    reason = window_tapes.NO_PREDICTIONS
    assert opened.unfit == dict(reason=reason, window="fold-016") and opened.failure is None
    assert opened.train == (None,) and opened.evaluation is not None
    fit = dict(id="fit", kind="fit")
    report = campaign_stage.unfit_report(STAGE, opened)
    assert [(r["status"], r["reason"]) for r in report["evaluation"]] == [("failed", reason)] * 3
    campaign_stage.check_report(STAGE, fit, report, opened)
    # Un ejecutor que presentara una política ajustada sin datos de ajuste se rechaza.
    forged = dict(report, policy=dict(id="fit", sha256="a" * 64), transitions=64)
    with pytest.raises(ValueError, match="no hay política"):
        campaign_stage.check_report(STAGE, fit, forged, opened)
    # Las referencias solo necesitan la evaluación y la conservan.
    cash = fixed_policy("cash")
    records = campaign_stage.evaluate_policy(opened, cash, STAGE, "US", backend="python")
    assert all(r["status"] != "failed" for r in records)


def test_the_universe_predictor_must_have_training_and_validation_predictions(tmp_path):
    assets = LIQUID[1:3]
    store = tapes(tmp_path, assets, source=without(assets, {("parent", "fold-017")}))
    with pytest.raises(ValueError, match="predictor parent del universo"):
        store.open(dict(JOB, predictor="other"))


class Base:
    """Campaña base mínima con el trabajo elegido de una ventana."""

    def __init__(self, markets, shift=0):
        self.record = dict(path="p.parquet", sha256="c" * 64, markets=markets)
        self.shift = shift

    def selected(self, scope, window, predictor, seed):
        parent = dict(receipts.PARENT)
        return "job", dict(parent=parent, predictions=dict(evaluation=self.record))

    def labels_used_until(self, scope, window, selected):
        start = FOLDS[window]["evaluation"][0]
        return receipts.microseconds(start) - 1 + self.shift


def test_campaign_source_reports_a_market_without_rows_as_no_predictions(tmp_path):
    from mars_titan.data.storage import atomic_json

    folder = tmp_path / "windows/US/fold-016/other/seed-42"
    index = list(FOLDS).index("fold-016")
    atomic_json(folder / "US.json", receipts.receipt("US", index=index))
    receipt, load = campaign_stage.campaign_source(Base({}), tmp_path, 42)(
        "US", "US", "fold-016", "other"
    )
    assert load is None and receipt.fold == "fold-016"
    # Si la campaña declara filas del mercado, el recibo sin predicciones no le corresponde.
    declared = Base({"US": dict(rows=10, sha256="d" * 64)})
    with pytest.raises(ValueError, match="no corresponde al predictor elegido"):
        campaign_stage.campaign_source(declared, tmp_path, 42)("US", "US", "fold-016", "other")
    # La maduración que recalcula la campaña debe coincidir con la que declara el recibo.
    later = Base({}, shift=1)
    with pytest.raises(ValueError, match="maduración real"):
        campaign_stage.campaign_source(later, tmp_path, 42)("US", "US", "fold-016", "other")


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


def test_tapes_and_universes_on_disk_do_not_read_the_predictions_again(tmp_path):
    assets = LIQUID[1:3]
    first = tapes(tmp_path, assets).open(JOB)
    honest = source_for(assets)
    read = []

    def released(scope, market, window, predictor):
        # Los recibos siguen disponibles, pero las filas se liberaron tras montar las cintas.
        receipt, _ = honest(scope, market, window, predictor)

        def load():
            read.append(window)
            raise prediction_files.PredictionsReleased(f"{window} se liberaron")

        return receipt, load

    again = tapes(tmp_path, assets, source=released).open(JOB)
    assert again.identity == first.identity and read == []
    # Una cinta que todavía no está en disco sí necesita las filas.
    with pytest.raises(prediction_files.PredictionsReleased):
        tapes(tmp_path, assets, source=released, output="other").open(JOB)
    assert read


def test_universe_is_saved_with_its_identity(tmp_path):
    assets = LIQUID[1:3]
    tapes(tmp_path, assets).open(JOB)
    path = Path(tmp_path / "stage/universes/US/US/fold-018.json")
    record = campaign_stage.read_manifest(path, 1024**2)[0]
    assert record["assets"] == ["US/AAA", "US/BBB"]
    assert record["identity"]["rule"] == window_tapes.UNIVERSE_RULE
    assert list(record["identity"]["segments"]) == ["fold-016", "fold-017"]


def test_policy_tapes_must_be_real_tapes_of_the_declared_edition(tmp_path):
    stage = tapes(tmp_path, LIQUID)
    receipt, load = stage.source(JOB, "fold-017", "parent")
    values = load()
    tape, _ = window_tapes.build_segment_tape(
        stage.edition, receipt, values, market="US", role="validation", lag=0
    )
    assert window_tapes.require_real_tape(tape, stage.edition_id, "validation") is tape
    with pytest.raises(ValueError, match="no es una cinta real"):
        window_tapes.require_real_tape(tape, "0" * 64, "validation")
    synthetic = MarketTape(
        tape.prices,
        tape.close_times,
        tape.assets,
        tape.scores,
        domain="synthetic",
        currency=tape.currency,
        partition=tape.partition,
        open_times=tape.open_times,
    )
    with pytest.raises(ValueError, match="no es una cinta real"):
        window_tapes.require_real_tape(synthetic, stage.edition_id, "validation")
    # Con la identidad de la edición, el dominio sintético basta para rechazarla.
    relabeled = copy.copy(tape)
    relabeled.domain = "synthetic"
    with pytest.raises(ValueError, match="no es una cinta real"):
        window_tapes.require_real_tape(relabeled, stage.edition_id, "validation")
    # Una cinta real que declarase otro origen o precios ajustados tampoco se admite.
    for key, field, value in (("source", "kind", "episode_world"), ("audit", "price_basis", "x")):
        forged = copy.copy(tape)
        forged.identity = dict(tape.identity, **{key: dict(tape.identity[key], **{field: value})})
        with pytest.raises(ValueError, match="no es una cinta real"):
            window_tapes.require_real_tape(forged, stage.edition_id, "validation")


@pytest.mark.parametrize("name", ["neural", "ridge", "titans"])
def test_segment_predictions_read_compacted_files_and_stop_on_released_ones(tmp_path, name):
    """La retención v2 compacta las predicciones por fila de las que salen las cintas.

    Las cintas deben recibir exactamente las mismas puntuaciones con el archivo original y con
    el compactado, bit a bit, y un archivo liberado debe detener la etapa con su motivo en
    lugar de dejar la cinta sin filas.
    """
    from tests.data.test_prediction_files import writer, written

    path, digest = written(tmp_path / "attempt", writer(name))
    before = {
        market: window_tapes.segment_predictions(path, digest, market) for market in "US CN".split()
    }
    prediction_files.compact(path, digest, tmp_path / "rows")
    assert not path.exists()
    for market, values in before.items():
        after = window_tapes.segment_predictions(path, digest, market)
        assert len(values["score"]) > 0
        assert np.array_equal(after["prediction_at"], values["prediction_at"])
        assert np.array_equal(after["asset_id"], values["asset_id"])
        assert after["score"].dtype == values["score"].dtype
        assert after["score"].tobytes() == values["score"].tobytes()
    prediction_files.release(path, digest, stage="fixture")
    with pytest.raises(prediction_files.PredictionsReleased, match="se liberaron"):
        window_tapes.segment_predictions(path, digest, "US")
