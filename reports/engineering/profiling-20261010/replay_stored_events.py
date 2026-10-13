"""Ejecuta `benchmarks/chronological_replay.py` sobre eventos guardados sin leer la vista.

Es el envoltorio de lectura bloqueada que usa la medida de medoids de la rama de núcleos.
Solo sirve para medir y nunca para ajustar.

- La vista no se vuelve a comprobar archivo a archivo (`_verify_files`) y cualquier intento
  de leer muestras o etiquetas falla, así que el recorrido solo usa los eventos de
  `--events` y los índices guardados en `MEDOID_INDICES`.
- La identidad de código de las observaciones se toma del índice guardado. Así develop y
  una rama con otro código pueden repetir los mismos eventos, que es lo que necesita la
  comparación bit a bit de los rangos NVTX.
- La especificación de entradas usa las anchuras de `MEDOID_WIDTHS`, las del índice
  guardado, porque sin leer muestras no se pueden deducir.

Uso: replay_stored_events.py benchmarks/chronological_replay.py VISTA TRABAJO [opciones]
"""

import json
import os
import runpy
import sys
from pathlib import Path

from mars_titan.memory import financial_observations
from mars_titan.training import corpus_inputs

corpus_inputs.CorpusDataset._verify_files = lambda self: None
_STORED = json.loads(next(Path(os.environ["MEDOID_INDICES"]).glob("*/identity.json")).read_text())[
    "code"
]
_identity = financial_observations._identity
financial_observations._identity = lambda dataset, phase: dict(
    _identity(dataset, phase), code=_STORED
)
corpus_inputs.CorpusDataset._file = lambda self, asset, kind: self._path(asset, kind)


def _no_reads(*_args, **_kwargs):
    raise RuntimeError("La medida intentó leer muestras de la vista")


financial_observations.FinancialObservationSource._input = _no_reads
financial_observations.FinancialObservationSource._label_arrays = _no_reads
WIDTHS = json.loads(Path(os.environ["MEDOID_WIDTHS"]).read_text())


def _specification(self):
    return financial_observations.FinancialInputSpec(
        source_sha256=self.dataset.identity,
        view_sha256=self.identity,
        representation=self.dataset.manifest["representation"],
        dimensions=WIDTHS,
        input_policy=self.dataset.manifest["input_policy"],
    )


financial_observations.FinancialObservationSource.specification = _specification
sys.argv = [sys.argv[1], *sys.argv[2:]]
runpy.run_path(sys.argv[0], run_name="__main__")
