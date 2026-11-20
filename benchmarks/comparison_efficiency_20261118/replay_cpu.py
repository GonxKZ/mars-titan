"""Emisión y replay del mismo corte de ajuste con menos filas y un solo hilo CPU."""

import argparse
import json
import os
import resource
from pathlib import Path
from time import perf_counter_ns
from types import SimpleNamespace

import pyarrow as pa
import torch
from opciones_replay import OPCIONES, variante

from mars_titan.data.storage import atomic_json
from mars_titan.memory import event_archive
from mars_titan.models.titans.financial import DecisionBatch, FinancialInputSpec
from mars_titan.models.titans.state import mac_tensors
from mars_titan.training import financial_run as fr
from mars_titan.training import titans_walk_forward as ventana
from mars_titan.training.checkpoints import load_training_state
from mars_titan.training.comparison_input_migration import same_state


def lotes_ajuste(ruta, sello, espec, filas, instantes):
    firma = event_archive.verified_signature(ruta, sello)
    inicio = int(sello["events"] * 0.8)
    lotes, metas = [], {}
    with pa.memory_map(str(ruta), "r") as stream:
        lector = pa.ipc.open_file(stream, options=pa.ipc.IpcReadOptions(use_threads=False))
        if lector.schema != event_archive.SCHEMA or lector.num_record_batches != sello["events"]:
            raise ValueError("El archivo no coincide con la declaración de eventos")
        for indice in range(inicio, min(inicio + 100, lector.num_record_batches)):
            datos = lector.get_batch(indice)
            instante = datos.column(0)[0].as_py()
            evento = event_archive._decode(datos, espec, sello["block"], instante, sello["block"])
            metas.update(((flujo, fecha), valor) for flujo, fecha, valor in evento.labels)
            if evento.inputs:
                bruto = evento.inputs[0]
                parte = {k: v[:filas] for k, v in bruto.items() if k != "inputs"}
                parte["inputs"] = {k: v[:filas] for k, v in bruto["inputs"].items()}
                lotes.append(DecisionBatch.from_corpus(parte, espec))
            if len(lotes) == instantes:
                break
    if event_archive._signature(ruta) != firma:
        raise event_archive.InvalidArchive(
            "El archivo verificado cambió durante la lectura del corte"
        )
    claves = [
        (f, t)
        for b in lotes
        for f, t in zip(b.flow_ids, b.prediction_at, strict=True)
        if (f, t) in metas
    ]
    if len(lotes) != instantes or not claves:
        raise ValueError("El corte carece de emisiones o de etiquetas maduras")
    return lotes, claves, metas


def emitir_repetir(motor, lotes, claves, metas, comprobar):
    modelo = motor.predictor
    todos = tuple(dict.fromkeys(f for b in lotes for f in b.flow_ids))
    run = SimpleNamespace(
        used=[(f, t, t) for f, t in claves],
        targets=[metas[k] for k in claves],
        penalties=[],
        starts=dict.fromkeys(todos),
        segment=[(b, {}, False) for b in lotes],
    )
    modelo.zero_grad(set_to_none=True)
    estados, salidas = fr.FlowStates(), []
    inicio = perf_counter_ns()
    with torch.no_grad():
        for lote in lotes:
            nuevos = tuple(f for f in lote.flow_ids if f not in estados)
            if nuevos:
                estados.put(modelo.initial_state(nuevos))
            preparacion = modelo.prepare(lote, estados.gather(lote.flow_ids))
            estados.put(preparacion.next_state)
            if comprobar:
                salidas.append(preparacion.quantiles)
    corte = perf_counter_ns()
    perdida = motor._replay(run)
    fin = perf_counter_ns()
    prueba = None
    if comprobar:
        estado = estados.gather(todos)
        prueba = (
            salidas,
            mac_tensors(estado.mac),
            estado.observed_steps,
            estado.flow_ids,
            estado.last_sample_ids,
            estado.last_prediction_at,
            estado.parameter_id,
            perdida,
            [p.grad for p in modelo.parameters()],
            torch.get_rng_state(),
        )
    return dict(
        emision_ms=(corte - inicio) / 1e6,
        replay_ms=(fin - corte) / 1e6,
        ambos_ms=(fin - inicio) / 1e6,
    ), prueba


def comparar(args):
    if args.salida.exists() or not (
        1 <= args.filas <= 1024 and 2 <= args.instantes <= 16 and 1 <= args.pares <= 25
    ):
        raise ValueError("Se necesita una salida nueva y tamaños dentro de los límites del corte")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    carpeta = Path(os.environ["MARS_TITAN_COMPARISON_STATE"]) / "runs/titans_mac/2021/seed-44"
    informe = json.loads((carpeta / "run.json").read_text())
    ajuste = json.loads((carpeta / "fit/run.json").read_text())
    ruta = Path(os.environ["MARS_TITAN_TRAIN_ARCHIVE"]) / "wide.arrow"
    sello = json.loads(ruta.with_suffix(".json").read_text())
    parametros = dict(ajuste["identity"]["predictor"]["inputs"])
    del parametros["schema_version"], parametros["context"]
    espec = FinancialInputSpec(**parametros)
    if (
        informe["status"] != "completed"
        or informe["final_test_opened"]
        or sello["index"] != informe["identity"]["indices"]["train"]
        or sello["spec"] != espec.fingerprint()
        or not sello["verified"]["digest"]
        or ruta.stat().st_size != sello["bytes"]
    ):
        raise ValueError("El archivo o el checkpoint no corresponde al ajuste")
    lotes, claves, metas = lotes_ajuste(ruta, sello, espec, args.filas, args.instantes)
    if args.solo_primitivas:
        from primitivas_replay import medir_primitivas

        atomic_json(
            args.salida,
            medir_primitivas(lotes[0], ajuste["identity"]["predictor"]["persistent_tokens"]),
        )
        return
    sha = ajuste["best_checkpoint"]["sha256"]
    pesos = load_training_state(
        carpeta / "fit/checkpoints",
        expected_identity=ajuste["identity"],
        selection="best",
        expected_sha256=sha,
    )["model"]
    doc = informe["identity"]["recipe"]
    receta = ventana.case_recipe(doc, informe["request"].get("search_case"))
    modos = ("base", args.candidato)
    motores, tiempos = {}, []
    with ventana.unfused_attention():
        for modo in modos:
            with variante(modo):
                predictor, _ = ventana._predictor(doc, espec, "mac_online", 44, "cpu")
                predictor.load_state_dict(pesos)
                # El corte usa la emisión y el replay con pesos fijos.
                motor = fr.ChronologicalTrainer.__new__(fr.ChronologicalTrainer)
                fr.ChronologicalInference.__init__(motor, predictor, receta)
                motor.teacher = None
                motores[modo] = motor
                _, evidencia = emitir_repetir(motor, lotes, claves, metas, True)
            if modo == "base":
                base = evidencia
            elif not same_state(base, evidencia):
                raise ValueError("Difieren las salidas, gradientes, estado o RNG")
        del base, evidencia
        for i in range(args.pares):
            for modo in modos if i % 2 == 0 else reversed(modos):
                with variante(modo):
                    muestra, _ = emitir_repetir(motores[modo], lotes, claves, metas, False)
                tiempos.append(dict(modo=modo, par=i, **muestra))
    atomic_json(
        args.salida,
        dict(
            candidato=args.candidato,
            checkpoint=sha,
            indice=sello["index"],
            lotes=[b.input_digest for b in lotes],
            etiquetas=len(claves),
            paridad=True,
            llamadas_optimizador=0,
            tiempos=tiempos,
            rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        ),
    )


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("candidato", choices=OPCIONES)
    cli.add_argument("salida", type=Path)
    cli.add_argument("--filas", type=int, default=32)
    cli.add_argument("--instantes", type=int, default=8)
    cli.add_argument("--pares", type=int, default=6)
    cli.add_argument("--solo-primitivas", action="store_true")
    comparar(cli.parse_args())
