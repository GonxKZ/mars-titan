"""Lanzador local de la medición supervisada breve, sin tocar el test final."""

import argparse
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from mars_titan.budget_training import train_budget_grid  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, default=Path("data/processed/phase1"))
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--panels", type=int, nargs="+", choices=[4, 11, 22], default=[4, 11, 22])
    parser.add_argument("--kinds", nargs="+", choices=["mlp", "gru"], default=["mlp", "gru"])
    args = parser.parse_args()
    train_budget_grid(
        args.prepared,
        args.report,
        args.output,
        epochs=args.epochs,
        panel_sizes=tuple(args.panels),
        kinds=tuple(args.kinds),
    )
