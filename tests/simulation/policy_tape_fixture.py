"""Cintas de política por ventana mensual sobre una edición sintética con el formato real.

Las ventanas proceden de un protocolo walk-forward v2 con evaluaciones mensuales de 2023,
válido para el lector de recibos y distinto de los protocolos versionados. Cada cinta se
monta con ``window_tapes.build_segment_tape``, el mismo camino que usa la etapa. Las
puntuaciones no proceden de ningún modelo y los precios son una fixture identificada.
"""

import numpy as np

from mars_titan.environments.walk_forward_receipt import RECEIPT_KIND, read_window_receipt
from mars_titan.evaluation.splits import build_folds
from mars_titan.simulation import window_tapes
from mars_titan.simulation.market_rules import china_a_share_instrument
from mars_titan.simulation.storage import write_tape
from tests.environments.walk_forward_fixture import PARENT, microseconds, protocol, record
from tests.simulation.unadjusted_edition_fixture import Asset, predictions, write_edition

LAG = 0
EDITION = {
    "US": [
        Asset("AAA", base=20.0, zero_volume=(170,), events=((180, 0.25, 0), (200, 0, 2.0))),
        Asset("BBB", base=45.0, missing=(190,), events=((215, 0.3, 0),)),
        Asset("CCC", base=150.0),
    ],
    "CN": [
        Asset("600000.SS", base=10.0, zero_volume=(175,), events=((185, 0.2, 0),)),
        Asset("300750.SZ", base=50.0),
        Asset("688981.SS", base=40.0, events=((205, 0.1, 1.5),)),
    ],
}
# Ajuste, validación y evaluación consecutivos: septiembre a diciembre de 2023.
ROLES = (("train", -4), ("train", -3), ("validation", -2), ("evaluation", -1))


def monthly_protocol(market):
    return dict(
        protocol(market),
        train_start="2019-01-01",
        first_validation_start="2022-07-01",
        validation_months=3,
        calibration_months=1,
        minimum_train_months=42,
        evaluation_months=1,
        step_months=1,
    )


def monthly_window(market, index, symbols, *, parent=None, score=None):
    """Recibo de la ventana mensual ``index`` con sus puntuaciones de evaluación."""
    rules = monthly_protocol(market)
    fold = build_folds(rules)[index]
    start, end = fold["evaluation"]
    # El tramo excluye su final y el reloj incluye su último día.
    last = str(np.datetime64(end, "D") - np.timedelta64(1, "D"))
    values = predictions(market, symbols, start=start, end=last, score=score)
    receipt = dict(
        kind=RECEIPT_KIND,
        schema_version=1,
        protocol=rules,
        fold=fold,
        parent=dict(PARENT if parent is None else parent),
        labels_used_until=microseconds(start) - 1,
        predictions={"evaluation": record(values)},
    )
    return read_window_receipt(receipt), values


def instruments(tape, market):
    if market != "CN":
        return None
    return {asset: china_a_share_instrument(asset) for asset in tape.assets}


def write_policy_tapes(root, market, *, lag=LAG):
    """Escribir edición y cintas de ajuste, validación y evaluación de un mercado."""
    edition = root / "edition"
    if not edition.exists():
        write_edition(edition, EDITION)
    symbols = [asset.symbol for asset in EDITION[market]]
    result = {"train": [], "validation": [], "evaluation": []}
    for number, (role, index) in enumerate(ROLES):
        window, values = monthly_window(market, index, symbols)
        tape, _ = window_tapes.build_segment_tape(
            edition, window, values, market=market, role=role, lag=lag
        )
        folder = root / f"{market}-lag{lag}" / f"{number}-{role}"
        write_tape(tape, folder, instruments=instruments(tape, market))
        result[role].append((folder, tape))
    # La primera ventana de ajuste con el papel de evaluación: su manifiesto es distinto del
    # de la cinta de ajuste, pero termina antes de la selección.
    window, values = monthly_window(market, ROLES[0][1], symbols)
    tape, _ = window_tapes.build_segment_tape(
        edition, window, values, market=market, role="evaluation", lag=lag
    )
    folder = root / f"{market}-lag{lag}" / "early-evaluation"
    write_tape(tape, folder, instruments=instruments(tape, market))
    result["early_evaluation"] = [(folder, tape)]
    return result
