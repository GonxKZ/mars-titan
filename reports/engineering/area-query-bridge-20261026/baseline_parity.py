"""Repetir lecturas y escrituras técnicas de B1 con la implementación importada.

Se ejecuta en procesos separados sobre una revisión anterior y la actual. Los hashes de
salida se comparan bit a bit. No ajusta un predictor ni estima calidad financiera.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from mars_titan.memory import area_memory, associative_memory
from mars_titan.memory.area_memory import AreaLayout, AreaMemory
from mars_titan.memory.associative_memory import (
    AssociativeMemory,
    AssociativeMemoryConfig,
    MatureFeedback,
)


def digest(value):
    return hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()


def keys(rows, generator):
    values = torch.randn((rows, 64), dtype=torch.float64, generator=generator)
    return values / torch.linalg.vector_norm(values, dim=-1, keepdim=True)


def measure():
    torch.set_num_threads(2)
    result = {}
    for rule in ("proximal", "delta"):
        for name in ("b6", "areas", "shuffled_areas"):
            generator = torch.Generator().manual_seed(20261026)
            config = AssociativeMemoryConfig(rule, rate=0.25, forgetting=0.01)
            memory = (
                AssociativeMemory(config)
                if name == "b6"
                else AreaMemory(
                    config,
                    AreaLayout(
                        areas=4,
                        seed=7,
                        labels="mature" if name == "areas" else "shuffled_within_cohort",
                    ),
                )
            )
            reads = []
            for index in range(2):
                start, at = index * 128 + 1, 10 * (index + 1)
                feedback = MatureFeedback(
                    ids=torch.arange(start, start + 128),
                    decision_at=torch.full((128,), at - 5),
                    available_at=torch.full((128,), at),
                    keys=keys(128, generator),
                    values=torch.randn((128, 1), dtype=torch.float64, generator=generator),
                    weights=torch.full((128,), 1 / 128, dtype=torch.float64)
                    if rule == "proximal"
                    else None,
                )
                query = keys(256, generator)
                reads.append(digest(memory.read(query)))
                memory = memory.write(feedback, cutoff=at)
                reads.append(digest(memory.read(query)))
            result[f"{rule}/{name}"] = dict(
                reads_sha256=reads,
                matrix_sha256=memory.export()["matrix_sha256"],
                writes=memory.writes,
                cursor=memory.cursor,
                configuration=memory.identity() if name != "b6" else config.identity(),
            )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("La comprobación necesita un archivo nuevo")
    payload = dict(
        cases=measure(),
        sources_sha256={
            "src/" + module.__name__.replace(".", "/") + ".py": hashlib.sha256(
                Path(module.__file__).read_bytes()
            ).hexdigest()
            for module in (area_memory, associative_memory)
        },
        runtime=dict(torch=torch.__version__, numpy=np.__version__, threads=2),
    )
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
