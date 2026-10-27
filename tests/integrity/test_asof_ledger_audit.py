"""Auditoría as-of del registro de etiquetas G3 (#748, #442).

Los registros son sintéticos con semilla fija o escritos a mano. Las etapas auditadas son las
que ya existen en `develop` y leen etiquetas maduras: la volatilidad pasada de A2, PT2 en
línea y su recalibración periódica. Cada auditoría se acompaña de una versión con una fuga
deliberada que la auditoría debe detectar. No hay modelos ajustados ni pasos de optimizador.
"""

import numpy as np
import pytest

from mars_titan.calibration import online_conformal, periodic_conformal
from mars_titan.calibration.conformal_quantiles import fit_conformal_quantiles
from mars_titan.environments.cohorts import FINAL_TEST_START_US
from mars_titan.evaluation import neutral_abstention as na
from mars_titan.integrity import asof_ledger_audit as audit
from mars_titan.models.quantile_head import LEVELS

NOMINALS = (0.8, 0.95)
DAY = 86_400_000_000
START = 1_577_836_800_000_000  # 2020-01-01


def ledger(sessions=30, assets=6, horizon=2, seed=4):
    """Etiquetas diarias de un mercado con cuantiles fijos y madurez a `horizon` días."""
    rng = np.random.default_rng(seed)
    session = np.repeat(np.arange(sessions), assets)
    issued = START + session * DAY
    return dict(
        asset_id=np.tile([f"US/A{i}" for i in range(assets)], sessions),
        group=np.full(sessions * assets, "US"),
        issued_at=issued,
        available_at=issued + horizon * DAY,
        target=rng.normal(0, 1 + (session >= sessions // 2), sessions * assets),
        quantiles=np.tile([-2.0, -1.3, 0.0, 1.3, 2.0], (sessions * assets, 1)),
    )


def cutoffs(records):
    return np.unique(records["issued_at"])


def volatility(records, *, shift=0):
    """Volatilidad pasada de A2 en cada emisión. `shift` adelanta la madurez y crea una fuga."""
    history = dict(
        asset_id=records["asset_id"],
        target=records["target"],
        matures_at=np.asarray(records["available_at"]) - shift,
    )
    values = na.past_volatility(
        records["asset_id"], records["issued_at"], history, window=5, min_labels=2
    )
    return np.nan_to_num(values, nan=-1.0), records["issued_at"]


def online(records, *, shift=0):
    calibration = ledger(sessions=10, seed=9)
    calibrator = online_conformal.OnlineConformal.start(
        calibration["target"],
        calibration["quantiles"],
        calibration["group"],
        levels=LEVELS,
        nominals=NOMINALS,
        min_rows=1,
        rate_fraction=0.25,
    )
    emitted, _, _ = online_conformal.replay_online_conformal(
        calibrator,
        records["quantiles"],
        records["group"],
        records["issued_at"],
        records["target"],
        np.asarray(records["available_at"]) - shift,
    )
    return emitted, records["issued_at"]


def periodic(records, *, shift=0):
    calibration = ledger(sessions=10, seed=9)
    record = fit_conformal_quantiles(
        calibration["target"],
        calibration["quantiles"],
        calibration["group"],
        levels=LEVELS,
        nominals=NOMINALS,
        min_rows=1,
    )
    part = dict(
        target=records["target"],
        quantiles=records["quantiles"],
        groups=records["group"],
        prediction_at=records["issued_at"],
        matures_at=np.asarray(records["available_at"]) - shift,
    )
    earlier = {
        key: values - 40 * DAY if key in ("prediction_at", "matures_at") else values
        for key, values in dict(
            target=calibration["target"],
            quantiles=calibration["quantiles"],
            groups=calibration["group"],
            prediction_at=calibration["issued_at"],
            matures_at=calibration["available_at"],
        ).items()
    }
    emitted, _ = periodic_conformal.replay_periodic_refit(
        record, earlier, part, every_sessions=3, lookback_days=10
    )
    return emitted, records["issued_at"]


STAGES = {"past_volatility": volatility, "online_conformal": online, "periodic_refit": periodic}


@pytest.mark.parametrize("name", STAGES)
def test_each_stage_reconstructs_its_past_emissions_at_every_cutoff(name):
    records = ledger()
    receipt = audit.audit(STAGES[name], records, columns=("target",), cutoffs=cutoffs(records))
    assert receipt["passed"], receipt["failures"]
    assert len(receipt["cutoffs"]) == 30
    assert all(entry["future_events"] > 0 for entry in receipt["cutoffs"])


@pytest.mark.parametrize("name", STAGES)
def test_a_stage_reading_labels_one_day_early_is_detected(name):
    records = ledger()
    leaky = lambda values: STAGES[name](values, shift=DAY)  # noqa: E731
    receipt = audit.audit(leaky, records, columns=("target",), cutoffs=cutoffs(records))
    assert not receipt["passed"]
    assert {failure["check"] for failure in receipt["failures"]} == {"reconstruction"}


@pytest.mark.parametrize("name", ["online_conformal", "periodic_refit"])
def test_stages_that_receive_both_instants_reject_impossible_records(name):
    assert audit.rejection(STAGES[name], ledger()) == []


def toy(records, *, leak=False):
    """Media de las etiquetas disponibles en cada emisión, o de todas si `leak`."""
    issued = np.asarray(records["issued_at"])
    available = np.asarray(records["available_at"])
    target = np.asarray(records["target"])
    seen = np.ones((len(issued), len(issued)), dtype=bool)
    if not leak:
        seen = available[None, :] <= issued[:, None]
    sums = (seen * target[None, :]).sum(axis=1)
    return sums / np.maximum(seen.sum(axis=1), 1), issued


def test_a_correct_stage_passes_and_its_receipt_is_reproducible():
    records = ledger(sessions=8, assets=2)
    first = audit.audit(toy, records, columns=("target",), cutoffs=cutoffs(records))
    second = audit.audit(toy, records, columns=("target",), cutoffs=cutoffs(records))
    assert first["passed"] and first == second
    assert [entry["emitted_rows"] for entry in first["cutoffs"]] == list(range(2, 17, 2))
    # Con madurez a dos días, en el corte de la sesión k aún no han llegado las etiquetas de
    # las sesiones k − 1 y posteriores.
    assert [entry["future_events"] for entry in first["cutoffs"]] == [16, 16, 14, 12, 10, 8, 6, 4]


def test_reading_every_label_is_detected_at_each_cutoff_with_future_events():
    records = ledger(sessions=8, assets=2)
    receipt = audit.audit(
        lambda values: toy(values, leak=True),
        records,
        columns=("target",),
        cutoffs=cutoffs(records),
    )
    failed = {(item["cutoff"], item["replacement"]) for item in receipt["failures"]}
    assert failed == {(int(t), how) for t in cutoffs(records) for how in ("zero", "extreme")}


def test_a_label_read_exactly_when_it_becomes_available_is_allowed():
    records = ledger(sessions=4, assets=1)
    # La primera etiqueta llega un microsegundo después de la segunda emisión.
    records["available_at"][0] = records["issued_at"][1] + 1
    receipt = audit.audit(toy, records, columns=("target",), cutoffs=cutoffs(records))
    assert receipt["passed"]
    early = lambda values: toy(  # noqa: E731
        values | dict(available_at=np.asarray(values["available_at"]) - 1)
    )
    leaky = audit.audit(early, records, columns=("target",), cutoffs=cutoffs(records))
    assert {item["cutoff"] for item in leaky["failures"]} == {int(records["issued_at"][1])}


def test_a_leak_through_a_one_sided_transform_is_still_detected():
    # Con etiquetas positivas, min(y, 0) vale cero con el valor original, con cero y con un
    # extremo positivo. Solo el extremo de signo contrario hace visible la fuga.
    records = ledger(sessions=6, assets=2)
    records["target"] = np.abs(records["target"]) + 0.1

    def one_sided(values):
        emitted = np.full(len(values["issued_at"]), np.minimum(values["target"], 0).sum())
        return emitted, values["issued_at"]

    receipt = audit.audit(one_sided, records, columns=("target",), cutoffs=cutoffs(records))
    assert {item["replacement"] for item in receipt["failures"]} == {"extreme"}


def test_repetition_read_only_and_stored_emissions_are_checked():
    records = ledger(sessions=6, assets=2)
    generator = np.random.default_rng(1)

    def noisy(values):
        emitted, at = toy(values)
        return emitted + generator.normal(0, 1e-12, len(emitted)), at

    def writer(values):
        values["target"][0] += 1.0
        return toy(values)

    checks = {
        "repeat": audit.audit(noisy, records, columns=("target",), cutoffs=[START]),
        "read_only": audit.audit(
            writer, ledger(sessions=6, assets=2), columns=("target",), cutoffs=[START]
        ),
        "stored": audit.audit(
            toy, records, columns=("target",), cutoffs=[START], stored=toy(records)[0] + 1
        ),
    }
    for name, receipt in checks.items():
        assert name in {item["check"] for item in receipt["failures"]}, name
    stored = audit.audit(toy, records, columns=("target",), cutoffs=[START], stored=toy(records)[0])
    assert stored["passed"]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda r: r | dict(available_at=r["issued_at"]), "su emisión"),
        (lambda r: r | dict(available_at=r["available_at"] - 3 * DAY), "su emisión"),
        (
            lambda r: r | dict(available_at=np.r_[r["available_at"][:-1], FINAL_TEST_START_US]),
            "2024",
        ),
        (lambda r: r | dict(target=r["target"][:-1]), "alineada"),
        (lambda r: r | dict(target=np.arange(len(r["target"]))), "real"),
        (lambda r: {k: v for k, v in r.items() if k != "issued_at"}, "issued_at"),
    ],
)
def test_impossible_or_malformed_ledgers_are_rejected_before_auditing(change, message):
    with pytest.raises(ValueError, match=message):
        audit.audit(toy, change(ledger(sessions=4, assets=2)), columns=("target",), cutoffs=[START])
