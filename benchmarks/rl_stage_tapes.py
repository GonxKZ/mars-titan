"""Cintas reales de una ventana de política para medir la etapa RL nativa sin aprendizaje.

Monta, con el mismo camino que la etapa (`window_tapes`), las cintas de ajuste, validación
y evaluación de la última ventana de política de cada mercado sobre la edición sin ajustar.
El universo de cada tramo sigue la regla `point_in_time_median_traded_value_v2` sobre un
conjunto de candidatos acotado por orden alfabético, para que la preparación sea corta, y
las cuatro cintas comparten el diseño que une esos universos. Las
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
from mars_titan.simulation.listing_status import read_listing_status
from mars_titan.simulation.market_rules import tape_instruments
from mars_titan.simulation.reconstructed_tape import census
from mars_titan.simulation.storage import write_tape
from tests.environments.walk_forward_fixture import protocol, window
from tests.simulation.unadjusted_edition_fixture import predictions

# Es la regla principal de la etapa, con las tres evaluaciones anteriores a la validación,
# y reproduce las ventanas de ajuste que recoge el informe de la medición.
TRAIN_WINDOWS = dict(rule=window_tapes.FIXED, minimum=3, maximum=3)
RANKING_SESSIONS = 252


def _segment_values(market, fold, symbols):
    start, end = fold["evaluation"]
    # El tramo excluye su final y el reloj incluye su último día.
    last = str(np.datetime64(end, "D") - np.timedelta64(1, "D"))
    return predictions(market, symbols, start=start, end=last)


def _bounds(fold):
    return tuple(int(np.datetime64(day, "us").astype(np.int64)) for day in fold["evaluation"])


def prepare(edition, output, market, *, candidates, max_assets, status, lag=0):
    """Escribir las cuatro cintas de la última ventana de política y su descripción."""
    folds = build_folds(protocol(market))
    ids = [fold["id"] for fold in folds]
    row = window_tapes.policy_windows(folds, TRAIN_WINDOWS)[-1]
    pool = sorted(path.name for path in (Path(edition) / "assets" / market).iterdir())
    pool = pool[:candidates]
    # Las puntuaciones sintéticas cubren todo el conjunto, así que cubren cada tramo.
    keys = {f"{market}/{symbol}" for symbol in pool}
    plan = [*((name, "train") for name in row["train"])]
    plan += [(row["validation"], "validation"), (row["window"], "evaluation")]
    universes = {}
    for name, role in plan:
        rows = census(
            edition,
            _bounds(folds[ids.index(name)]),
            market=market,
            ranking_sessions=RANKING_SESSIONS,
            symbols=pool,
        )
        universes[name] = window_tapes.select_universe(
            rows, keys, max_assets, evaluation=role == "evaluation"
        )
    layout = sorted(set().union(*universes.values()))
    written = []
    for number, (name, role) in enumerate(plan):
        index = ids.index(name)
        values = _segment_values(market, folds[index], pool)
        receipt = window(market, values=values, index=index)
        tape, report = window_tapes.build_segment_tape(
            edition,
            receipt,
            values,
            market=market,
            role=role,
            lag=lag,
            listing_status=status,
            # El diseño identifica los activos con su mercado y la edición los pide sin él.
            symbols=[asset.split("/", 1)[1] for asset in layout],
            universe=[asset.split("/", 1)[1] for asset in universes[name]],
        )
        folder = Path(output) / market / f"{number}-{role}-{name}"
        write_tape(tape, folder, instruments=tape_instruments(tape))
        written.append(
            dict(
                role=role,
                window=name,
                folder=str(folder),
                assets=len(tape.assets),
                universe=len(universes[name]),
                sessions=int(tape.prices.shape[0]),
                excluded=len(report.get("excluded", {})),
            )
        )
    return dict(
        market=market,
        policy_window=row["window"],
        candidates=candidates,
        layout=len(layout),
        tapes=written,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--edition", required=True, type=Path)
    parser.add_argument("--listing-status", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--markets", nargs="+", default=["US", "CN"])
    parser.add_argument("--candidates", type=int, default=400)
    parser.add_argument("--max-assets", type=int, default=128)
    args = parser.parse_args()
    status = read_listing_status(args.listing_status)
    summary = dict(
        kind="rl_stage_benchmark_tapes",
        scores="synthetic_not_from_any_model",
        edition=str(args.edition),
        listing_status_sha256=status[1],
        markets=[
            prepare(
                args.edition,
                args.output,
                market,
                candidates=args.candidates,
                max_assets=args.max_assets,
                status=status,
            )
            for market in args.markets
        ],
    )
    atomic_json(args.output / "tapes.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
