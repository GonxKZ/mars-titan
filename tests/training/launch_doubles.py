"""Ejecutores reales de la campaña hasta la entrada de su entrenador.

`launch_job` es importable desde el proceso de una ranura. Sustituye el arranque de cada
entrenador por una parada que anota lo que recibe, ejecuta el ejecutor real del trabajo y lo
termina con `slot_doubles.record_job`, que escribe filas nulas. Así se comprueba que las
opciones de la campaña llegan al proceso que entrenaría, también cuando ese proceso es una
ranura. Ningún entrenador da un paso de ajuste y la GPU no se usa: la referencia neuronal
corre en CPU hasta su primer informe y las recetas de Titans-MAC, CM-v1 y la GRU episódica
se detienen al pasar al entrenador, antes de abrir su vista.

Cada llegada se anota en líneas JSON en `LAUNCH_LOG`. Con `LAUNCH_PAUSE`, el trabajo con ese
identificador pide la pausa de la campaña tras llegar a su entrenador, sin escribir nada más.
"""

import json
import os

import pytest

from tests.training import slot_doubles


class Reached(Exception):
    """El entrenador recibió su configuración y el doble detiene ahí el trabajo."""


def _log(entry):
    line = json.dumps(entry, sort_keys=True) + "\n"
    descriptor = os.open(os.environ["LAUNCH_LOG"], os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    try:
        os.write(descriptor, line.encode())
    finally:
        os.close(descriptor)


def _neural_report(path, report):
    """Primer informe de `run_reference_case`, con la identidad ya fijada y sin ningún paso."""
    identity = report["identity"]
    raise Reached(
        dict(
            trainer="reference_run",
            case=identity["case"],
            batch_size=identity["batch_size"],
            kernel_policy=identity.get("kernel_policy"),
            graph_step="training/reference_step_graph.py" in identity["code"],
            numerics={key: identity["numerics"][key] for key in _NUMERICS},
        )
    )


_NUMERICS = ("float32_matmul_precision", "cuda_matmul_allow_tf32", "cudnn_allow_tf32")


def _seams(patch):
    """Paradas en la entrada de cada entrenador. Solo viven mientras corre el trabajo."""
    import torch

    from mars_titan.training import campaign_numerics, reference_run
    from mars_titan.training import candidate_walk_forward as candidate
    from mars_titan.training import titans_walk_forward as titans

    real_recipe = titans._recipe

    def titans_recipe(*args, **kwargs):
        # La receta que `run_titans_window` entrega a `ChronologicalTrainer`.
        chronological, document, options = real_recipe(*args, **kwargs)
        raise Reached(
            dict(
                trainer="titans_walk_forward",
                recipe=chronological.identity(),
                numerics=campaign_numerics.current(),
            )
        )

    def candidate_window(view, folder, recipe, **_):
        raise Reached(
            dict(
                trainer="candidate_walk_forward",
                recipe=recipe.identity(),
                numerics=campaign_numerics.current(),
            )
        )

    def no_step(*_, **__):
        raise RuntimeError("Un paso del optimizador durante una prueba de lanzamiento")

    patch.setattr(reference_run, "require_cuda", lambda: torch.device("cpu"))
    patch.setattr(reference_run, "atomic_json", _neural_report)
    patch.setattr(torch.cuda, "get_device_name", lambda *_: "sin dispositivo")
    patch.setattr(titans, "_recipe", titans_recipe)
    patch.setattr(candidate, "fit_window", candidate_window)
    patch.setattr(torch.optim.AdamW, "step", no_step)


# Ejecutores que se recorren hasta su entrenador: los que reciben lote, precisión y CUDA
# Graphs de la sección neuronal o una receta con opciones de memoria. Los lectores de CM-v1
# necesitan un núcleo ajustado de verdad y los tabulares no tienen esas opciones.
REACHED = {
    ("neural", "fit"),
    ("titans_mac", "fit"),
    ("episodic_gru", "fit"),
    ("cm_v1_core", "fit"),
}


def launch_job(run):
    """Ejecutor real hasta su entrenador y doble que termina el trabajo con filas nulas."""
    from mars_titan.training import masked_campaign as engine

    key = (run.job["model"], run.job["kind"])
    if key in REACHED:
        with pytest.MonkeyPatch.context() as patch:
            _seams(patch)
            try:
                engine.EXECUTORS[key]["run"](run)
            except Reached as reached:
                received = reached.args[0]
            else:
                raise AssertionError(f"{run.job['id']} no llegó a su entrenador")
        _log(dict(id=run.job["id"], pid=os.getpid(), **received))
    if os.environ.get("LAUNCH_PAUSE") == run.job["id"]:
        raise engine.Paused
    return slot_doubles.record_job(
        run,
        quantiles=run.job["family"] != "tabular_reference",
        report_name=engine.EXECUTORS[key]["report"],
    )


def received(path):
    """Lo que recibió cada entrenador, por trabajo, en el orden en que llegó."""
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]
