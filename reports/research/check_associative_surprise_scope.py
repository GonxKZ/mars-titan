"""Comprobar que una pérdida mayor no implica un gradiente mayor en memoria no lineal."""

import argparse
import json
from fractions import Fraction
from pathlib import Path

import torch

from mars_titan.data.storage import atomic_json, sha256


def check():
    """Dos casos racionales de M(w)=w², contrastados con autograd FP64 en cuda:0."""
    if not torch.cuda.is_available():
        raise RuntimeError("La comprobación necesita cuda:0")
    rational = []
    for weight, target in ((Fraction(0), Fraction(1)), (Fraction(1), Fraction(3, 2))):
        loss = (weight**2 - target) ** 2
        derivative = 4 * weight * (weight**2 - target)
        rational.append(
            dict(
                weight=str(weight),
                target=str(target),
                loss=str(loss),
                derivative=str(derivative),
                gradient_squared=str(derivative**2),
            )
        )
    weights = torch.tensor([0.0, 1.0], device="cuda:0", dtype=torch.float64, requires_grad=True)
    targets = torch.tensor([1.0, 1.5], device="cuda:0", dtype=torch.float64)
    losses = (weights.square() - targets).square()
    (gradients,) = torch.autograd.grad(losses.sum(), weights)
    expected_loss = torch.tensor([1.0, 0.25], device="cuda:0", dtype=torch.float64)
    expected_gradient = torch.tensor([0.0, -2.0], device="cuda:0", dtype=torch.float64)
    if not torch.equal(losses, expected_loss) or not torch.equal(gradients, expected_gradient):
        raise AssertionError("Autograd no coincide con la derivada racional")
    if not (
        Fraction(rational[0]["loss"]) > Fraction(rational[1]["loss"])
        and Fraction(rational[0]["gradient_squared"]) < Fraction(rational[1]["gradient_squared"])
    ):
        raise AssertionError("Los casos no refutan la monotonía general")
    return dict(
        schema_version=1,
        kind="associative_surprise_scope_counterexample",
        derivation="loss=(w^2-v)^2, derivative=4*w*(w^2-v)",
        rational=rational,
        cuda_losses=losses.detach().cpu().tolist(),
        cuda_gradients=gradients.cpu().tolist(),
        exact_agreement_for_these_binary_representable_values=True,
        torch=torch.__version__,
        device="cuda:0",
        dtype="float64",
        optimizer_steps=0,
        source_sha256=sha256(Path(__file__)),
        scope="Contraejemplo elemental para una memoria no lineal. "
        "No es una ejecución del predictor ni una prueba de novedad matemática.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("La comprobación necesita un destino nuevo")
    result = check()
    atomic_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
