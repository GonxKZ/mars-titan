"""Repetir una comparación sobre la partición de ajuste con el mismo checkpoint."""

import argparse
import os
import random
import statistics
from contextlib import ExitStack, closing
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from opciones_replay import OPCIONES, variante

from mars_titan.data.storage import atomic_json
from mars_titan.memory import event_archive, training_event_cache
from mars_titan.models.titans import host_finite
from mars_titan.training.checkpoints import capture_rng
from mars_titan.training.comparison_input_migration import same_state
from mars_titan.training.input_pipeline import background


def pasada(recorrido, motor, espec, corte, etapas, comprobar):
    exportados = {}
    ejecutar = motor._train_pass

    def guardar(run, cursor, stop, save):
        def captura(posicion, actual):
            exportados.update(cursor=posicion, run=actual)

        return ejecutar(run, cursor, stop, captura)

    random.seed(42)
    np.random.seed(42)
    with ExitStack() as pila:
        if comprobar:
            pila.enter_context(patch.object(motor, "_train_pass", guardar))
        resultado = recorrido.profile_slice(
            motor,
            espec,
            "original",
            comprobar,
            warm=2,
            until=etapas,
            source_range=corte,
            rng_seed=42,
        )
    estado = None
    if comprobar:
        estado = (exportados["cursor"], motor._export(exportados["run"]), capture_rng("cuda:0"))
    return resultado, estado


def comparar(args):
    import recorrido
    from entradas_campana import sources

    if args.salida.exists():
        raise ValueError("Ya existe la salida")
    if not 3 <= args.etapas <= 16 or not 1 <= args.pares <= 9:
        raise ValueError("Se admiten de 3 a 16 etapas y de 1 a 9 pares")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0" or not torch.cuda.is_available():
        raise ValueError("La comparación requiere cuda:0 explícita")
    torch.cuda.set_device("cuda:0")
    recorrido.rw._cuda("cuda:0")
    recorrido.apply_kernel_policy("fp32_strict")
    fuentes = sources(args.origen, "titans_mac", args.semilla)
    fuente = fuentes["train"]
    carpeta = Path(os.environ[training_event_cache.ENV])
    fuentes["train"] = fuente = training_event_cache.PackedTrainingSource(
        fuente.dataset, fuente.path, carpeta
    )
    espec = fuente.specification()
    with ExitStack() as pila:
        pila.callback(recorrido.perf._forbid_steps().remove)
        pila.enter_context(recorrido.tw.unfused_attention())
        pila.enter_context(host_finite.on_host())
        with variante(args.candidato):
            motor, informe, _ = recorrido.build(
                "titans_mac", fuentes, espec, args.semilla, year=args.origen
            )
        if not isinstance(motor.optimizer, recorrido.FixedWeights):
            raise ValueError("El ensayo requiere pesos fijos")
        pesos = recorrido.parameter_digest(motor)
        n = len(fuente.metadata["groups"])
        cantidad = motor.recipe.truncation * (args.etapas + 1) + 16
        inicio = min(int(0.8 * n), n - cantidad - 1)
        if inicio < 0 or fuente.phase.partition != "train":
            raise ValueError("La fuente no contiene el corte de ajuste")
        ancho = motor.recipe.block_rows
        ruta = carpeta / ("wide.arrow" if ancho >= 1024 else "small.arrow")
        sello = recorrido.read(ruta.with_suffix(".json"))
        if not sello["verified"]["digest"] or sello["block"] != ancho:
            raise ValueError("La copia Arrow no corresponde al tamaño de bloque")
        firma = event_archive.verified_signature(ruta, sello)

        def eventos(*, start_cursor=0, block_rows):
            flujo = event_archive.replay(
                fuente, espec, ruta, sello, cursor=start_cursor, block=block_rows, signature=firma
            )
            prefijo = fuente.dataset.pipeline.prefetch_batches
            yield from background(flujo, prefijo) if prefijo else flujo

        pila.enter_context(patch.object(fuente, "batched_events", eventos))
        modos = ("base", args.candidato)
        evidencias, tiempos = {}, []
        for modo in modos:
            with variante(modo):
                valores, estado = pasada(
                    recorrido, motor, espec, (inicio, cantidad), args.etapas, True
                )
            if recorrido.parameter_digest(motor) != pesos:
                raise ValueError("Los pesos cambiaron")
            evidencias[modo] = valores["parity"]
            if modo == "base":
                estado_base = estado
            elif evidencias[modo] != evidencias["base"] or not same_state(estado, estado_base):
                raise ValueError("Difieren las emisiones, los gradientes, el estado o el RNG")
        del estado, estado_base
        for i in range(args.pares):
            for modo in modos if i % 2 == 0 else reversed(modos):
                with variante(modo):
                    valores, _ = pasada(
                        recorrido, motor, espec, (inicio, cantidad), args.etapas, False
                    )
                tiempos.append(dict(modo=modo, par=i, **valores))
        if recorrido.parameter_digest(motor) != pesos:
            raise ValueError("Los pesos cambiaron durante las medidas")
        primitivas = None
        if args.primitivas:
            from primitivas_replay import medir_primitivas

            from mars_titan.models.titans.financial_inputs import DecisionBatch

            with closing(eventos(start_cursor=inicio, block_rows=ancho)) as lista:
                bloque = next(ev.inputs[0] for ev in lista if ev.inputs)
            lote = DecisionBatch.from_corpus(bloque, espec, device="cuda:0")
            primitivas = medir_primitivas(lote, motor.predictor.config.persistent_tokens)
    atomic_json(
        args.salida,
        dict(
            candidato=args.candidato,
            origen=args.origen,
            semilla=args.semilla,
            checkpoint=informe["fit"]["best_checkpoint"]["sha256"],
            archivo=sello["sha256"],
            indice=fuente.identity,
            corte=[inicio, cantidad],
            torch=str(torch.__version__),
            gpu=torch.cuda.get_device_name(0),
            hilos=torch.get_num_threads(),
            paridad=evidencias,
            estado_y_rng_identicos=True,
            pasos_de_optimizador=0,
            tiempos=tiempos,
            primitivas=primitivas,
            mediana_estable_s={
                modo: statistics.median(f["seconds"] for f in tiempos if f["modo"] == modo)
                for modo in modos
            },
        ),
    )


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("candidato", choices=OPCIONES)
    cli.add_argument("salida", type=Path)
    cli.add_argument("--origen", type=int, default=2021)
    cli.add_argument("--semilla", type=int, default=44)
    cli.add_argument("--etapas", type=int, default=4)
    cli.add_argument("--pares", type=int, default=3)
    cli.add_argument("--primitivas", action="store_true")
    comparar(cli.parse_args())
