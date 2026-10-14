"""Medir en `cuda:0` el coste de inferencia del profesor y del alumno de #57.

La destilación solo compensa si el profesor cuesta claramente más al predecir que el alumno.
Aquí se recorren, sobre la misma vista y los mismos eventos, el alumno `titans_mac_online` y
los lectores `mars_titan_m1` (el profesor declarado) y `mars_titan_m1_k4` (la alternativa
K = 4 de MT-055) sobre ese mismo padre, con las funciones de la orden de caudal
(`measure_titans` y `measure_mars_titan`). Ningún peso cambia: el gancho global prohíbe pasos
y el optimizador de la medida no actualiza. No se registran pérdidas ni errores.

La vista es la de 64 activos de EE. UU. que prepara `benchmarks/memory_pilot_seed_cost.py`
(fase `prepare`) en `--work`, con sus índices ya construidos, de modo que la GPU no espera a
lecturas de disco. Se repite `--repeats` veces porque la GPU es compartida.
"""

import argparse
import json
import os
import platform
import statistics
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

CAMPAIGN = Path("configs/baselines/historical-masked-campaign-a.json")
EXTENSIONS = Path("configs/baselines/historical-masked-campaign-extensions.json")
DECLARATION = Path("configs/evaluation/temporal-distillation.json")
STUDENT, TEACHER, ALTERNATIVE = "titans_mac_online", "mars_titan_m1", "mars_titan_m1_k4"
SETTINGS = dict(segments=160, segment_warmup=10, events=512, event_warmup=32)


def _commit():
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()


def _gpu():
    return subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.used,driver_version,clocks.sm,temperature.gpu",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()


def _campaign():
    """Campaña A con la sección preparada de MARS-TITAN, limitada a los tres modelos."""
    from mars_titan.training.campaign_extensions import extended_campaign, load_extensions
    from mars_titan.training.campaign_plan import load_campaign
    from mars_titan.training.campaign_throughput import MARS, TITANS

    campaign = extended_campaign(load_extensions(EXTENSIONS), load_campaign(CAMPAIGN))
    titans = dict(campaign[TITANS], candidates={STUDENT: campaign[TITANS]["candidates"][STUDENT]})
    readers = {arm: campaign[MARS]["candidates"][arm] for arm in (TEACHER, ALTERNATIVE)}
    return dict(campaign, **{TITANS: titans, MARS: dict(campaign[MARS], candidates=readers)})


def measure(view, work, output, repeats):
    import torch

    from mars_titan.data.storage import atomic_json, sha256
    from mars_titan.training import campaign_throughput as ct

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA no está disponible. No se cambia a CPU")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("Falta CUBLAS_WORKSPACE_CONFIG=:4096:8")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.set_num_threads(2)
    if type(repeats) is not int or not 1 <= repeats <= 9:
        raise ValueError("Las repeticiones van de una a nueve")
    campaign = _campaign()
    before, rounds = _gpu(), []
    for _ in range(repeats):
        began = time.perf_counter()
        core = ct.measure_titans(campaign, view, work, **SETTINGS)[STUDENT]
        readers = ct.measure_mars_titan(campaign, view, work, **SETTINGS)
        rounds.append(
            dict(
                seconds=time.perf_counter() - began,
                gpu_after=_gpu(),
                student=core,
                teacher=readers[TEACHER],
                alternative=readers[ALTERNATIVE],
            )
        )

    def inference(role):
        return [entry[role]["inference"] for entry in rounds]

    student, teacher, alternative = (inference(r) for r in ("student", "teacher", "alternative"))
    # Coste de inferencia del profesor frente al alumno: cociente de caudales por ronda, con
    # los dos medidos sobre los mismos eventos y en la misma ronda.
    ratios = [s / t for s, t in zip(student, teacher, strict=True)]
    k4 = [s / a for s, a in zip(student, alternative, strict=True)]
    receipt = dict(
        schema_version=1,
        kind="temporal_distillation_cost",
        recorded_at_utc=datetime.now(UTC).isoformat(),
        commit=_commit(),
        scope=(
            "Recorrido de ajuste hasta el paso con _NoStepOptimizer y el gancho que prohíbe "
            "pasos, e inferencia congelada. Padre y lectores con pesos iniciales. Sin pérdidas "
            "ni errores registrados."
        ),
        device="cuda:0",
        gpu_before=before,
        versions=dict(
            python=platform.python_version(),
            torch=torch.__version__,
            cuda=torch.version.cuda,
            cudnn=torch.backends.cudnn.version(),
        ),
        precision=dict(
            dtype="float32",
            tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
            tf32_cudnn=torch.backends.cudnn.allow_tf32,
            matmul_precision=torch.get_float32_matmul_precision(),
            cublas_workspace=os.environ["CUBLAS_WORKSPACE_CONFIG"],
        ),
        threads=torch.get_num_threads(),
        campaign=dict(path=str(CAMPAIGN), sha256=sha256(CAMPAIGN)),
        extensions=dict(path=str(EXTENSIONS), sha256=sha256(EXTENSIONS)),
        declaration=dict(path=str(DECLARATION), sha256=sha256(DECLARATION)),
        view=dict(path=str(view), sha256=sha256(Path(view))),
        models=dict(student=STUDENT, teacher=TEACHER, alternative=ALTERNATIVE),
        settings=SETTINGS,
        repeats=repeats,
        rounds=rounds,
        inference_rows_per_second=dict(student=student, teacher=teacher, alternative=alternative),
        teacher_over_student_inference_cost=dict(
            per_round=ratios, median=statistics.median(ratios), lowest=min(ratios)
        ),
        k4_over_student_inference_cost=dict(
            per_round=k4, median=statistics.median(k4), lowest=min(k4)
        ),
        optimizer_steps=0,
        training_runs=0,
        evaluation=False,
        final_test_opened=False,
    )
    atomic_json(Path(output), receipt)
    print(json.dumps(dict(teacher_over_student=ratios, k4_over_student=k4), indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--view", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    measure(args.view, args.work, args.output, args.repeats)


if __name__ == "__main__":
    main()
