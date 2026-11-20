"""Tiempo aislado de máscara y selección con las entradas de un instante del entrenamiento."""

from collections import Counter
from time import perf_counter_ns

import torch
from opciones_replay import seleccionar
from torch.profiler import ProfilerActivity, profile
from torch.utils._python_dispatch import TorchDispatchMode

from mars_titan.models.titans.state import deferred_checks


class OperadoresCPU(TorchDispatchMode):
    """Contar ATen sin inicializar el perfilador CUDA de una instalación mixta."""

    def __init__(self):
        super().__init__()
        self.cuentas = Counter()

    def __torch_dispatch__(self, op, types, args=(), kwargs=None):
        self.cuentas[str(op)] += 1
        return op(*args, **(kwargs or {}))


def medir(operacion, llamadas, repeticiones, dispositivo):
    cuda = dispositivo.type == "cuda"
    muestras_host, muestras_eventos = [], []
    with deferred_checks():
        operacion()
    for _ in range(repeticiones):
        if cuda:
            torch.cuda.synchronize(dispositivo)
            inicio, fin = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        reloj = perf_counter_ns()
        if cuda:
            inicio.record()
        with deferred_checks():
            for _ in range(llamadas):
                operacion()
        if cuda:
            fin.record()
            fin.synchronize()
        muestras_host.append((perf_counter_ns() - reloj) / 1000 / llamadas)
        if cuda:
            muestras_eventos.append(inicio.elapsed_time(fin) * 1000 / llamadas)
    actividad = {}
    if cuda:
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as perfil:
            with deferred_checks():
                operacion()
            torch.cuda.synchronize(dispositivo)
        aten = {e.key: e.count for e in perfil.key_averages() if e.key.startswith("aten::")}
        actividad = dict(
            Counter(
                e.name for e in perfil.events() if e.device_type == torch.autograd.DeviceType.CUDA
            )
        )
    else:
        with OperadoresCPU() as contadas, deferred_checks():
            operacion()
        aten = dict(contadas.cuentas)
    return dict(
        llamadas=llamadas,
        host_us=muestras_host,
        eventos_cuda_us=muestras_eventos,
        aten=aten,
        actividad_cuda=actividad,
    )


def medir_primitivas(lote, persistentes):
    dispositivo = lote.presence.device
    n = persistentes + 2
    fija = torch.ones(n, n, device="cpu", dtype=torch.bool).triu(1).to(dispositivo)
    filas = list(range(len(lote.flow_ids)))
    cuda = dispositivo.type == "cuda"
    pruebas = (
        (
            "mascara_base",
            lambda: torch.ones(n, n, device=dispositivo, dtype=torch.bool).triu(1),
            100 if cuda else 500,
        ),
        ("mascara_fija", lambda: fija, 100 if cuda else 500),
        ("seleccion_base", lambda: lote.select(filas), 8 if cuda else 10),
        ("seleccion_clon", lambda: seleccionar(lote, filas), 8 if cuda else 10),
    )
    return {
        "filas": len(filas),
        "huella_lote": lote.input_digest,
        "bytes": sum(x.nbytes for x in (*lote.inputs.values(), lote.presence)),
        **{
            nombre: medir(op, nllamadas, 5 if cuda else 9, dispositivo)
            for nombre, op, nllamadas in pruebas
        },
    }
