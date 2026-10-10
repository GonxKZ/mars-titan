"""Cintas reales de una ventana de política para medir la etapa RL nativa sin aprendizaje.

Monta, con el mismo camino que la etapa (`window_tapes`), las cintas de ajuste, validación
y evaluación de la última ventana de política de cada mercado sobre la edición sin ajustar.
El universo sigue la regla `median_traded_value_in_validation_v1` sobre un conjunto de
candidatos acotado por orden alfabético, para que la preparación sea corta. Las
puntuaciones son sintéticas y no proceden de ningún modelo: las cintas solo sirven para
medir el coste del entorno y de la red, nunca para evaluar políticas.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from mars_titan.data.storage import atomic_json
from mars_titan.evaluation.splits import build_folds
from mars_titan.simulation import window_tapes
from mars_titan.simulation.market_rules import china_a_share_instrument
from mars_titan.simulation.storage import write_tape
from tests.environments.walk_forward_fixture import protocol, window
from tests.simulation.unadjusted_edition_fixture import predictions

TRAIN_WINDOWS = 3


def _segment_values(market, fold, symbols):
    start, end = fold["evaluation"]
    # El tramo excluye su final y el reloj incluye su último día.
    last = str(np.datetime64(end, "D") - np.timedelta64(1, "D"))
    return predictions(market, symbols, start=start, end=last)


def _tape(edition, market, folds, name, role, symbols, lag):
    index = [fold["id"] for fold in folds].index(name)
    values = _segment_values(market, folds[index], symbols)
    receipt = window(market, values=values, index=index)
    tape, report = window_tapes.build_segment_tape(
        edition, receipt, values, market=market, role=role, lag=lag, symbols=symbols
    )
    return tape, report


def prepare(edition, output, market, *, candidates, max_assets, lag=0):
    """Escribir las cuatro cintas de la última ventana de política y su descripción."""
    folds = build_folds(protocol(market))
    row = window_tapes.policy_windows(folds, TRAIN_WINDOWS)[-1]
    pool = sorted(path.name for path in (Path(edition) / "assets" / market).iterdir())
    pool = pool[:candidates]
    admitted = {}
    for name in (*row["train"], row["validation"]):
        role = "validation" if name == row["validation"] else "train"
        tape, _ = _tape(edition, market, folds, name, role, pool, lag)
        admitted[name] = window_tapes.admission(tape)
    selected = window_tapes.select_universe(
        [admitted[name] for name in row["train"]], admitted[row["validation"]], max_assets
    )
    # La admisión identifica los activos con su mercado y la edición los pide sin él.
    universe = [asset.split("/", 1)[1] for asset in selected]
    plan = [*((name, "train") for name in row["train"])]
    plan += [(row["validation"], "validation"), (row["window"], "evaluation")]
    written = []
    for number, (name, role) in enumerate(plan):
        tape, report = _tape(edition, market, folds, name, role, universe, lag)
        folder = Path(output) / market / f"{number}-{role}-{name}"
        rules = {a: china_a_share_instrument(a) for a in tape.assets} if market == "CN" else None
        write_tape(tape, folder, instruments=rules)
        written.append(
            dict(
                role=role,
                window=name,
                folder=str(folder),
                assets=len(tape.assets),
                sessions=int(tape.prices.shape[0]),
                excluded=len(report.get("excluded", {})),
            )
        )
    return dict(
        market=market,
        policy_window=row["window"],
        candidates=candidates,
        universe=len(universe),
        tapes=written,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--edition", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--markets", nargs="+", default=["US", "CN"])
    parser.add_argument("--candidates", type=int, default=400)
    parser.add_argument("--max-assets", type=int, default=128)
    args = parser.parse_args()
    summary = dict(
        kind="rl_stage_benchmark_tapes",
        scores="synthetic_not_from_any_model",
        edition=str(args.edition),
        markets=[
            prepare(
                args.edition,
                args.output,
                market,
                candidates=args.candidates,
                max_assets=args.max_assets,
            )
            for market in args.markets
        ],
    )
    atomic_json(args.output / "tapes.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
