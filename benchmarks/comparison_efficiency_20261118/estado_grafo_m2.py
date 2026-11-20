"""Control de estado de la emisión real del padre congelado, sin entrenamiento del lector."""

import argparse
import json
import os
import subprocess
import time
from dataclasses import fields, is_dataclass, replace
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import torch

from mars_titan.memory import event_archive as archivo
from mars_titan.models.titans import state as checks
from mars_titan.models.titans.financial import DecisionBatch, FinancialInputSpec
from mars_titan.models.titans.state import _PendingChecks, mac_tensors, map_mac_rows
from mars_titan.training import mars_titan_walk_forward as rw
from mars_titan.training import titans_walk_forward as tw
from mars_titan.training.checkpoints import capture_rng
from mars_titan.training.comparison_input_migration import same_state
from mars_titan.training.financial_run import FlowStates
from mars_titan.training.kernel_policy import apply_kernel_policy


def plano(x):
    if is_dataclass(x):
        return {k.name: plano(getattr(x, k.name)) for k in fields(x)}
    if isinstance(x, tuple):
        return tuple(plano(v) for v in x)
    return x


def materiales(ruta, carpeta, num, dispositivo):
    estado = ruta / "runs/mars_titan_m2/2023/seed-43"
    lector = json.loads((estado / "run.json").read_text())
    raiz = ruta / "runs/titans_mac/2023/seed-43"
    padre = json.loads((raiz / "run.json").read_text())
    ajuste = json.loads((raiz / "fit/run.json").read_text())
    entradas = dict(ajuste["identity"]["predictor"]["inputs"])
    del entradas["schema_version"], entradas["context"]
    espec = FinancialInputSpec(**entradas)
    sello = json.loads((carpeta / "small.json").read_text())
    sha = lector["request"]["parent"]["checkpoint_sha256"]
    assert (
        lector["status"] == padre["status"] == "completed"
        and not padre["final_test_opened"]
        and not lector["final_test_opened"]
    )
    assert sha == padre["fit"]["best_checkpoint"]["sha256"]
    assert (
        sello["index"]
        == lector["identity"]["indices"]["train"]
        == padre["identity"]["indices"]["train"]
    )
    assert (
        sello["spec"] == espec.fingerprint()
        and sello["verified"]["digest"]
        and sello["block"] == 512
    )
    origen = carpeta / "small.arrow"
    firma = archivo.verified_signature(origen, sello)
    # _parent verifica la identidad y el checkpoint del padre que usa el lector.
    pr, _ = rw._parent(raiz, ruta / "cohorts/2023/manifest.json", 43)
    p = rw._frozen_parent(
        pr["identity"]["recipe"], espec, 43, dispositivo, raiz, pr, allow_context=True
    )
    inicio = int(0.8 * sello["events"])
    lotes = []
    with pa.memory_map(str(origen), "r") as stream:
        reader = pa.ipc.open_file(stream, options=pa.ipc.IpcReadOptions(use_threads=False))
        for i in range(inicio, min(inicio + 80, reader.num_record_batches)):
            registro = reader.get_batch(i)
            evento = archivo._decode(registro, espec, 512, registro.column(0)[0].as_py(), 512)
            if not evento.inputs:
                continue
            lotes.append(
                [DecisionBatch.from_corpus(b, espec, device=dispositivo) for b in evento.inputs]
            )
            if len(lotes) == num:
                break
    if archivo._signature(origen) != firma:
        raise archivo.InvalidArchive("El archivo verificado cambió durante la lectura del corte")
    assert len(lotes) == num
    return (
        p,
        lotes,
        dict(
            origen=2023,
            semilla=43,
            checkpoint_padre=sha,
            checkpoint_lector=lector["fit"]["best_checkpoint"]["sha256"],
            indice=sello["index"],
            archivo=sello["sha256"],
            inicio=inicio,
        ),
    )


class Bloque:
    def __init__(self, mac, por_grafo):
        self.mac = mac
        self.fn = mac.forward
        self.grafico = por_grafo
        self.estatico = None
        self.captura_s = None
        self.llamadas = 0
        self.recursos = {}

    def devolver(self, x, st):
        return x.detach().clone(), map_mac_rows(st, lambda t: t.detach().clone())

    def cargar(self, x, st, copiar):
        self.fijo.copy_(x)
        if copiar:
            for dst, src in zip(mac_tensors(self.estatico), mac_tensors(st), strict=True):
                dst.copy_(src)

    def iniciar(self, x, st):
        self.fijo = x.detach().clone()
        self.estatico = map_mac_rows(st, lambda t: t.detach().clone())
        if self.grafico:
            torch.cuda.synchronize()
            self.recursos["asignados_antes"] = torch.cuda.memory_allocated()
            self.recursos["reservados_antes"] = torch.cuda.memory_reserved()
            inicio = time.perf_counter()
            etiqueta = checks._PENDING.set(None)
            try:
                lateral = torch.cuda.Stream()
                lateral.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(lateral):
                    for _ in range(3):
                        with checks.deferred_checks():
                            self.fn(self.fijo, self.estatico)
                torch.cuda.current_stream().wait_stream(lateral)
                self.grafo = torch.cuda.CUDAGraph()
                self.indicadores = _PendingChecks()
                aux = checks._PENDING.set(self.indicadores)
                try:
                    with torch.cuda.graph(self.grafo):
                        self.salida, self.siguiente = self.fn(self.fijo, self.estatico)
                finally:
                    checks._PENDING.reset(aux)
            finally:
                checks._PENDING.reset(etiqueta)
            torch.cuda.synchronize()
            self.captura_s = time.perf_counter() - inicio
            self.recursos["asignados_despues"] = torch.cuda.memory_allocated()
            self.recursos["reservados_despues"] = torch.cuda.memory_reserved()
            self.recursos["indicadores"] = len(self.indicadores.flags)

    def pasar(self, x, st, *, differentiable=False, copiar=True, **opciones):
        if len(x) != 512 or differentiable or opciones or self.mac.memory.observer is not None:
            return self.fn(x, st, differentiable=differentiable, **opciones)
        # Revisar los tensores reales antes de trasladarlos a direcciones fijas.
        self.mac._validate_state(st)
        self.mac.memory.validate_input(x, st.memory)
        if self.estatico is None:
            self.iniciar(x, st)
        self.cargar(x, st, copiar)
        self.llamadas += 1
        if not self.grafico:
            return self.devolver(*self.fn(self.fijo, self.estatico))
        self.grafo.replay()
        externo = checks._PENDING.get()
        if externo is None:
            mensaje = self.indicadores.first_failure()
            if mensaje is not None:
                raise ValueError(mensaje)
        else:
            for indicador, mensaje in zip(
                self.indicadores.flags, self.indicadores.messages, strict=True
            ):
                externo.add(indicador.clone(), mensaje)
        return self.devolver(self.salida, self.siguiente)


def control(p, lotes, dispositivo):
    estados = FlowStates()
    estados_copia = FlowStates()
    dinamico = Bloque(p.mac, dispositivo == "cuda:0")
    obsoleto = Bloque(p.mac, False)
    filas = []
    for indice, bloques in enumerate(lotes):
        for lote in bloques:
            nuevos = tuple(f for f in lote.flow_ids if f not in estados)
            if nuevos:
                estados.put(p.initial_state(nuevos))
                estados_copia.put(p.initial_state(nuevos))
            st = estados.gather(lote.flow_ids)
            real = p.prepare(lote, st)
            entrada_copia = estados_copia.gather(lote.flow_ids)
            if not same_state(plano(st), plano(entrada_copia)):
                raise ValueError("Los flujos del paso anterior perdieron el estado")
            with patch.object(p.mac, "forward", dinamico.pasar):
                copia = p.prepare(lote, entrada_copia)
            if not same_state(plano(real), plano(copia)):
                raise ValueError("Difiere una salida o un campo del estado financiero")
            with patch.object(
                p.mac, "forward", lambda x, s, **kw: obsoleto.pasar(x, s, copiar=False, **kw)
            ):
                fijo = p.prepare(lote, st)
            filas.append(
                dict(
                    instante=indice,
                    filas=len(lote.flow_ids),
                    huella_entrada=lote.input_digest,
                    iguales_todos_los_campos=True,
                    error_estado_fijo_cuantiles=float(
                        (real.quantiles - fijo.quantiles).abs().max()
                    ),
                    error_estado_fijo_peso=float(
                        (
                            real.next_state.mac.memory.weights[0]
                            - fijo.next_state.mac.memory.weights[0]
                        )
                        .abs()
                        .max()
                    ),
                    error_estado_fijo_contador=int(
                        (real.next_state.mac.memory.steps - fijo.next_state.mac.memory.steps)
                        .abs()
                        .max()
                    ),
                )
            )
            estados.put(real.next_state)
            estados_copia.put(copia.next_state)
    assert dinamico.llamadas >= 2 and any(x["error_estado_fijo_contador"] for x in filas)
    return filas, dinamico


def validar_error(bloque):
    entrada = bloque.fijo.detach().clone()
    estado = map_mac_rows(bloque.estatico, lambda x: x.detach().clone())
    malo = replace(
        estado, memory=replace(estado.memory, steps=torch.full_like(estado.memory.steps, 2**63 - 1))
    )
    mensajes = []
    for funcion in (bloque.fn, bloque.pasar):
        try:
            with checks.deferred_checks():
                funcion(entrada, malo)
                funcion(entrada, estado)
        except ValueError as exc:
            mensajes.append(str(exc))
        else:
            raise AssertionError("No se informó del desbordamiento")
    if mensajes != ["El contador de pasos desbordaría int64"] * 2:
        raise AssertionError(mensajes)
    return mensajes[0]


def principal():
    if not __debug__:
        raise RuntimeError("Ejecute el control sin -O para conservar las comprobaciones")
    ap = argparse.ArgumentParser()
    ap.add_argument("--estado", type=Path, required=True)
    ap.add_argument("--archivo", type=Path, required=True)
    ap.add_argument("--salida", type=Path, required=True)
    ap.add_argument("--dispositivo", choices=["cpu", "cuda:0"], default="cpu")
    ap.add_argument("--referencia-cpu", type=Path)
    a = ap.parse_args()
    if a.salida.exists():
        raise ValueError("El recibo ya existe")
    if a.dispositivo == "cuda:0":
        if (
            os.environ.get("CUDA_VISIBLE_DEVICES") != "0"
            or os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8"
        ):
            raise ValueError("Se necesita cuda:0 explícita y FP32 reproducible")
        escritorio = {"/usr/bin/nautilus": 1, "/usr/bin/gnome-text-editor": 1, "/usr/bin/papers": 2}
        ocupada = subprocess.check_output(
            [
                "nvidia-smi",
                "--id=0",
                "--query-compute-apps=pid,process_name,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        for fila in ocupada.splitlines():
            pid, nombre, mb = (v.strip() for v in fila.split(","))
            if nombre not in escritorio or not escritorio[nombre]:
                raise ValueError(f"Otra carga ocupa la GPU: {nombre}, {pid}")
            escritorio[nombre] -= 1
        rw._cuda("cuda:0")
        if not torch.cuda.is_available() or "4070 Laptop" not in torch.cuda.get_device_name(0):
            raise ValueError("El control espera la RTX 4070 Laptop de PC1")
        torch.cuda.set_device("cuda:0")
        apply_kernel_policy("fp32_strict")
        if not a.referencia_cpu:
            raise ValueError("Faltan las huellas de las filas del control CPU")
    torch.set_num_threads(1 if a.dispositivo == "cpu" else 2)
    torch.set_num_interop_threads(1)
    with tw.unfused_attention(), torch.no_grad():
        p, lotes, recibo = materiales(a.estado, a.archivo, 3, a.dispositivo)
        if a.referencia_cpu:
            respaldo = json.loads(a.referencia_cpu.read_text())
            if respaldo["checkpoint_padre"] != recibo["checkpoint_padre"] or [
                b.input_digest for ev in lotes for b in ev
            ] != [b["huella_entrada"] for b in respaldo["filas"]]:
                raise ValueError("La muestra de ajuste o el checkpoint difieren del control CPU")
        rng = capture_rng(a.dispositivo)
        inicio = time.perf_counter()
        filas, bloque = control(p, lotes, a.dispositivo)
        mensaje = validar_error(bloque)
        if a.dispositivo == "cuda:0":
            torch.cuda.synchronize()
        if not same_state(rng, capture_rng(a.dispositivo)):
            raise ValueError("Cambió un generador durante las comparaciones")
        error = torch.tensor(False)
        tipos = {}
        for conserva in (False, True):
            error.fill_(False)
            anteriores = _PendingChecks()
            anteriores.add(error.clone() if conserva else error, "primera emisión")
            error.fill_(True)
            anteriores.add(error, "segunda emisión")
            tipos[str(conserva)] = anteriores.first_failure()
        assert tipos == {"False": None, "True": "primera emisión"}
    recibo.update(
        filas=filas,
        bloques_elegibles=sum(b["filas"] == 512 for b in filas),
        dispositivo=a.dispositivo,
        indicador_reutilizado=tipos,
        primera_excepcion=mensaje,
        rng_identico=True,
        tiempo_de_captura_s=bloque.captura_s,
        memoria_grafo=bloque.recursos,
        control_total_s=time.perf_counter() - inicio,
        ajustes=0,
    )
    a.salida.write_text(json.dumps(recibo, indent=2) + "\n")
    print(json.dumps(recibo))


if __name__ == "__main__":
    principal()
