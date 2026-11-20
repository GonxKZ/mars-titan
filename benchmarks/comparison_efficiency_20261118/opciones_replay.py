"""Variantes locales para medir dos copias y la máscara de la atención financiera."""

import hashlib
import inspect
from contextlib import ExitStack, contextmanager
from functools import cache
from textwrap import dedent
from unittest.mock import patch

import torch

from mars_titan.models.titans.config import canonical
from mars_titan.models.titans.financial_inputs import DecisionBatch
from mars_titan.models.titans.mac import TitansMAC
from mars_titan.training.financial_run import ChronologicalTrainer

OPCIONES = ("mascara", "seleccion", "perdidas")


def _aplicar(self, fn, recurse=True):
    resultado = super(TitansMAC, self)._apply(fn, recurse=recurse)
    self._mascara_ensayo = None
    if self.config.max_segment == 1:
        n = 1 if self.config.memory_mode == "disabled" else self.config.persistent_tokens + 2
        # El identificador del predictor incluye incluso sus buffers no persistentes.
        with torch.inference_mode(False), torch.no_grad():
            mascara = torch.ones(n, n, dtype=torch.bool, device="cpu").triu(1)
            self._mascara_ensayo = mascara.to(self.persistent.device)
    return resultado


def seleccionar(lote, filas):
    """El replay entrega índices Python únicos en orden de aparición con enumerate."""
    if len(filas) != len(lote.flow_ids) or filas != list(range(len(filas))):
        return lote.select(filas)
    if not all(t.is_contiguous() for t in (*lote.inputs.values(), lote.presence)):
        return lote.select(filas)
    lote.verify()
    huella = hashlib.sha256(canonical(["selection_v1", lote.input_digest, filas]).encode())
    return DecisionBatch._create(
        {k: v.clone() for k, v in lote.inputs.items()},
        lote.presence.clone(),
        lote.flow_ids,
        lote.sample_ids,
        lote.prediction_at,
        lote.input_available_at,
        lote.input_contract_id,
        huella.hexdigest(),
    )


@cache
def _funcion(nombre):
    """Insertar solo el cambio que se cronometra. Un cambio de la fuente invalida el ensayo."""
    if nombre == "mascara":
        original = TitansMAC.forward
        cambios = [
            (
                """mask = torch.ones(
            packed_length, packed_length, dtype=torch.bool, device=segment.device
        ).triu(1)""",
                """mask = self._mascara_ensayo
        if mask is None or mask.device != segment.device:
            mask = torch.ones(packed_length, packed_length, dtype=torch.bool,
                              device=segment.device).triu(1)""",
            )
        ]
    elif nombre == "seleccion":
        original = ChronologicalTrainer._replay
        cambios = [("batch = batch.select(rows)", "batch = seleccionar(batch, rows)")]
    elif nombre == "perdidas":
        original = ChronologicalTrainer._replay
        cambios = [
            ("total, replayed = 0.0, 0", "losses, replayed = [], 0"),
            ("total += float(loss.detach())", "losses.append(loss.detach())"),
            (
                "return torch.tensor(total)",
                """total = 0.0
    if losses:
        for value in torch.stack(losses).tolist():
            total += value
    return torch.tensor(total)""",
            ),
        ]
    else:
        raise ValueError("Opción de ensayo desconocida")
    fuente = dedent(inspect.getsource(original))
    for referencia, candidata in cambios:
        if fuente.count(referencia) != 1:
            raise ValueError(f"La fuente de {original.__qualname__} difiere de la referencia")
        fuente = fuente.replace(referencia, candidata)
    nombres = {**original.__globals__, "seleccionar": seleccionar}
    exec(compile(fuente, inspect.getfile(original), "exec"), nombres)
    return nombres[original.__name__]


@contextmanager
def variante(nombre):
    with ExitStack() as pila:
        if nombre == "mascara":
            pila.enter_context(patch.object(TitansMAC, "forward", _funcion(nombre)))
            pila.enter_context(patch.object(TitansMAC, "_apply", _aplicar))
        elif nombre != "base":
            pila.enter_context(patch.object(ChronologicalTrainer, "_replay", _funcion(nombre)))
        yield
