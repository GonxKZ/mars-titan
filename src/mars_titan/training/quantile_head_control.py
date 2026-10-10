"""Ejecutar el control de la cabeza de cuantiles de #22 y aplicar su regla de retroceso.

El plan `configs/baselines/quantile-head-control-us.json` declara doce casos del
Transformer compacto (`reference_design.head_control_cases`): índices de diseño 0 y 10,
semillas 42, 43 y 44, y en cada par una salida escalar L1 y `quantile_head_v1` con la
misma arquitectura, tasa, semilla, presupuesto fijo y selección. Cada trabajo es una
ventana, una semilla, un índice y un brazo. Se ajusta con `run_reference_case`, que
reanuda desde su carpeta, y confirma un recibo con huellas, filas de validación y MAE por
sesión. Un trabajo confirmado con la misma identidad no se repite. Todos los trabajos de
una ventana deben evaluar las mismas filas y objetivos de validación.

El plan fija el protocolo US v2 pero no qué ventanas recorre. Este ejecutor
declara por defecto solo la primera (`fold-000`), la opción más barata. La potencia del
contraste depende sobre todo del número de sesiones de validación, que es parecido en
cualquier ventana, mientras que el coste crece con las filas de ajuste. La elección es del
ejecutor y queda pendiente de revisión. `windows` permite declarar otras ventanas antes de
ejecutar y forma parte de la identidad, así que una salida no se reutiliza con otras.

Para cada índice de diseño se compara
`delta = MAE(quantile_head_v1) − MAE(scalar_l1)` del MAE por sesión de la mediana en la
validación, con las semillas promediadas sesión a sesión y las ventanas unidas, mediante
`paired_comparisons.compare_series`. La longitud de bloque, las réplicas, la semilla, la
confianza y la ponderación son las de la comparación principal declarada. La familia tiene
un contraste por índice y sus intervalos simultáneos usan el máximo estudentizado. Si algún
intervalo simultáneo excluye el cero en contra de la cabeza (límite inferior positivo), la
regla declarada recomienda la opción A. Si ninguno lo hace y todos están definidos, se
mantiene la opción B. Sin intervalos suficientes la decisión queda indeterminada.

Este módulo no abre la reserva final. `run_control` comprueba el bloqueo de aprendizaje
antes de abrir fuentes o crear salidas y antes de cada trabajo.
"""

import argparse
import fcntl
import hashlib
import json
import math
import os
import re
from contextlib import nullcontext
from pathlib import Path

from mars_titan.data.cohort_files import read_manifest, safe_destination
from mars_titan.data.storage import atomic_json, outside_source, sha256
from mars_titan.evaluation import walk_forward_comparison as comparison
from mars_titan.evaluation.forecast_panel import SessionSeries
from mars_titan.evaluation.forecast_scores import SessionScores, score_sessions
from mars_titan.evaluation.paired_comparisons import compare_series, delta

from .learning_hold import require_learning_allowed
from .reference_design import HEAD_CONTROL_ARMS, QUANTILE_HEAD, SCALAR_L1, head_control_cases

KIND = "quantile_head_control_run"
RECEIPT_KIND = "quantile_head_control_job"
DECISION_KIND = "quantile_head_control_decision"
SCOPE = "US"
PARTITION = "validation"
DEFAULT_COMPARISON = "configs/evaluation/historical-masked-2000-comparison.json"
DEFAULT_WINDOWS = ("fold-000",)
WINDOW_RULE = "first_protocol_window_cheapest_option_declared_by_the_executor_pending_review"
# Se usa el mismo intervalo de checkpoints que en las referencias neuronales de la campaña.
CHECKPOINT_SECONDS = 300
OPTION_A, OPTION_B, UNDETERMINED = "option_a_scalar_l1", "option_b_quantile_head", "undetermined"
CONSEQUENCES = {
    OPTION_A: (
        "La comparación principal pasa a la salida escalar L1 común (pérdida mae sobre la "
        "mediana, variante m1_k1_l1_median de la GRU candidata y cabeza escalar en las demás "
        "familias neuronales). Los cuantiles quedan como experimento secundario"
    ),
    OPTION_B: "La comparación principal conserva quantile_head_v1 como factor fijado",
    UNDETERMINED: (
        "Algún intervalo simultáneo no está definido y ninguno excluye el cero en contra de "
        "la cabeza. La regla no se puede aplicar con estas ventanas"
    ),
}
_PAIR = re.compile(r"transformer-(\d{2})-s(\d+)")


class Paused(Exception):
    """Señala una parada solicitada en una barrera confirmada de un trabajo."""


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _index(item, plan):
    """Obtiene el índice de diseño de un caso a partir de su par y lo comprueba contra el plan."""
    match = _PAIR.fullmatch(item["pair"])
    _require(match is not None, f"El par {item['pair']} no es un caso del Transformer compacto")
    index, seed = int(match.group(1)), int(match.group(2))
    _require(
        index in plan["case_indices"] and seed == item["case"]["seed"],
        f"El par {item['pair']} no corresponde a los índices y semillas del plan",
    )
    return index


def load_control(plan_path, *, windows=None, comparison_path=None):
    """Validar plan, protocolo, comparación y ventanas, y enumerar los trabajos sin leer datos.

    Las rutas relativas del plan y la comparación por defecto parten de la raíz del
    repositorio que contiene el plan (`configs/baselines/...`).
    """
    plan_path = Path(plan_path)
    plan, plan_sha256 = read_manifest(plan_path, 64 * 1024)
    cases = head_control_cases(plan)
    root = plan_path.resolve().parents[2]
    protocol_path = Path(plan["walk_forward"])
    protocol_path = protocol_path if protocol_path.is_absolute() else root / protocol_path
    _, protocol_sha256 = read_manifest(protocol_path, 1024**2)
    comparison_path = Path(comparison_path) if comparison_path else root / DEFAULT_COMPARISON
    declared = comparison.load_config(comparison_path)
    scope = declared["resolved_scopes"].get(SCOPE)
    _require(
        declared["input_policy"] == plan["input_policy"]
        and scope is not None
        and scope["protocol_sha256"] == {SCOPE: protocol_sha256},
        "La comparación no declara el ámbito US con el protocolo y la política del control",
    )
    selected = list(DEFAULT_WINDOWS if windows is None else windows)
    _require(
        selected
        and all(isinstance(window, str) for window in selected)
        and len(set(selected)) == len(selected)
        and selected == sorted(selected)
        and set(selected) <= set(scope["windows"]),
        "Las ventanas del control deben ser ventanas distintas y ordenadas del protocolo US",
    )
    settings = declared["comparison"]
    bootstrap = dict(
        block_length=settings["block_length"],
        replicates=settings["replicates"],
        seed=settings["seed"],
        confidence=settings["confidence"],
        market_weighting=declared["metrics"]["market_weighting"],
        sensitivity_block_lengths=tuple(settings["sensitivity_block_lengths"]),
    )
    jobs = [
        dict(
            id=f"{window}/{item['id']}",
            window=window,
            pair=item["pair"],
            arm=item["arm"],
            seed=item["case"]["seed"],
            index=_index(item, plan),
            case=item["case"],
        )
        for window in selected
        for item in cases
    ]
    _require(len({job["id"] for job in jobs}) == len(jobs), "El control repite trabajos")
    return dict(
        plan=plan,
        plan_sha256=plan_sha256,
        protocol_sha256=protocol_sha256,
        comparison_path=str(comparison_path.resolve()),
        comparison_sha256=declared["sha256"],
        comparison_config=declared,
        scope=scope,
        windows=selected,
        window_rule=(
            WINDOW_RULE if selected == list(DEFAULT_WINDOWS) else "declared_before_execution"
        ),
        bootstrap=bootstrap,
        rank_ic_min_assets=declared["metrics"]["rank_ic_min_assets"],
        jobs=jobs,
    )


def check_control(plan_path, *, windows=None, comparison_path=None):
    """Validar y contar los trabajos del control sin leer datos ni reservar la GPU."""
    control = load_control(plan_path, windows=windows, comparison_path=comparison_path)
    plan = control["plan"]
    return dict(
        status="checked",
        plan_sha256=control["plan_sha256"],
        protocol_sha256=control["protocol_sha256"],
        comparison_sha256=control["comparison_sha256"],
        windows=control["windows"],
        window_rule=control["window_rule"],
        jobs=len(control["jobs"]),
        jobs_per_window=len(control["jobs"]) // len(control["windows"]),
        arms=list(HEAD_CONTROL_ARMS),
        case_indices=plan["case_indices"],
        seeds=plan["seeds"],
        contrasts=[_contrast_name(index) for index in plan["case_indices"]],
        bootstrap=dict(
            control["bootstrap"],
            sensitivity_block_lengths=list(control["bootstrap"]["sensitivity_block_lengths"]),
        ),
        decision_rule=plan["comparison"]["fallback"],
        scientific_training_started=False,
        final_test_opened=False,
    )


def _views(control, directory):
    """Carga las vistas US que prepara la campaña y las valida contra la comparación del control."""
    from .masked_campaign import scope_views

    campaign = dict(
        input_policy=control["plan"]["input_policy"],
        comparison_config=control["comparison_config"],
    )
    return scope_views(Path(directory), SCOPE, campaign)


def _code():
    root = Path(__file__).parents[1]
    names = ("training/quantile_head_control.py", "training/reference_design.py")
    return {name: sha256(root / name) for name in names}


def _identity(control, views):
    plan = control["plan"]
    return dict(
        schema_version=1,
        kind=KIND,
        plan_sha256=control["plan_sha256"],
        protocol_sha256=control["protocol_sha256"],
        comparison_sha256=control["comparison_sha256"],
        input_policy=plan["input_policy"],
        windows=control["windows"],
        window_rule=control["window_rule"],
        views=dict(
            report_sha256=views["report_sha256"],
            windows={w: views["windows"][w]["sha256"] for w in control["windows"]},
        ),
        batch_size=plan["batch_size"],
        checkpoint_seconds=CHECKPOINT_SECONDS,
        prediction_retention=plan["prediction_retention"],
        code=_code(),
        final_test_opened=False,
    )


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _job_identity(identity_sha256, job, views):
    fields = ("id", "window", "pair", "arm", "seed", "index", "case")
    return dict(
        control_identity_sha256=identity_sha256,
        **{key: job[key] for key in fields},
        view_sha256=views["windows"][job["window"]]["sha256"],
    )


def _columns(job):
    extra = comparison.QUANTILE_COLUMNS if job["arm"] == QUANTILE_HEAD else ()
    return comparison.COLUMNS + extra


class _Rows:
    """Calcula la huella de filas y objetivos de validación por ventana, común a sus trabajos."""

    def __init__(self):
        self.seen = {}

    def check(self, job, digest):
        expected = self.seen.setdefault(job["window"], (job["id"], digest))
        _require(
            expected[1] == digest,
            f"{job['id']} no evalúa las mismas filas ni objetivos de validación que {expected[0]}",
        )


def _confirmed(output, job, identity, rows):
    """Leer un recibo existente y comprobar identidad, artefactos y filas."""
    path = output / "jobs" / job["id"] / "receipt.json"
    if not path.is_file():
        return None
    receipt, digest = read_manifest(path, 1024**2)
    _require(
        receipt.get("identity") == identity,
        f"El trabajo confirmado {job['id']} cambió de identidad",
    )
    for record in (receipt["report"], receipt["validation"]):
        _require(
            sha256(output / record["path"]) == record["sha256"],
            f"Un artefacto confirmado de {job['id']} ha cambiado",
        )
    rows.check(job, receipt["validation"]["rows_sha256"])
    return dict(receipt, sha256=digest)


def _confirm(output, job, identity, folder, report, view, fold, rows):
    """Comprobar el informe y las predicciones de validación y escribir el recibo."""
    from .masked_campaign import _rows_digest

    _require(
        report.get("status") == "completed"
        and report.get("final_test_opened") is False
        and report["identity"]["manifest_sha256"] == view["sha256"]
        and report["identity"]["case"] == job["case"],
        f"{job['id']} no confirma su vista, su caso o la reserva cerrada",
    )
    record = report["predictions"][PARTITION]
    path = folder / record["path"]
    safe_destination(path)
    table = comparison._read_predictions(dict(path=path, sha256=record["sha256"]), _columns(job))
    label = f"{job['id']} ({PARTITION})"
    comparison._check_segment(table, fold, PARTITION, [SCOPE], label)
    _require(
        table.num_rows == view["counts"][PARTITION],
        f"{label}: {table.num_rows} filas frente a {view['counts'][PARTITION]} de la vista",
    )
    score = record["metrics"]["session_mae"]
    _require(
        type(score) in (int, float) and math.isfinite(score) and score >= 0,
        f"{job['id']} no tiene un MAE de validación finito",
    )
    digest = _rows_digest(table)
    rows.check(job, digest)
    report_path = folder / "run.json"
    receipt = dict(
        schema_version=1,
        kind=RECEIPT_KIND,
        status="completed",
        identity=identity,
        report=dict(path=str(report_path.relative_to(output)), sha256=sha256(report_path)),
        validation=dict(
            path=str(path.relative_to(output)),
            sha256=record["sha256"],
            rows=table.num_rows,
            rows_sha256=digest,
            session_mae=score,
        ),
        final_test_opened=False,
    )
    destination = output / "jobs" / job["id"] / "receipt.json"
    atomic_json(destination, receipt)
    return dict(receipt, sha256=sha256(destination))


def _fit(view, folder, case, **options):
    from .reference_run import run_reference_case

    return run_reference_case(view, folder, case, **options)


def _gpu_lease():
    from .experiment_resources import GpuLease

    return GpuLease()


def _summary(output, identity, control, receipts, status, **extra):
    summary = dict(
        schema_version=1,
        kind=KIND,
        status=status,
        identity_sha256=_digest(identity),
        planned=len(control["jobs"]),
        completed=len(receipts),
        jobs={job["id"]: job["id"] in receipts for job in control["jobs"]},
        final_test_opened=False,
        **extra,
    )
    atomic_json(output / "summary.json", summary)
    return summary


def _open_output(output, views_directory):
    """Crear la salida del control fuera de las vistas y del dataset."""
    safe_destination(output)
    for protected in (views_directory, Path("dataset")):
        outside_source(protected, output)
        outside_source(output, protected)
    output.mkdir(parents=True, exist_ok=True)


def _mark(output, identity):
    """Fijar la identidad de una salida nueva o exigir la misma en una existente."""
    marker = output / "control.json"
    if marker.exists():
        _require(
            read_manifest(marker, 1024**2)[0] == identity,
            "La salida pertenece a otro control, otras ventanas, vistas o código",
        )
    else:
        _require(
            all(p.name in {".lock", "summary.json"} for p in output.iterdir()),
            "La salida sin identidad contiene artefactos ajenos",
        )
        atomic_json(marker, identity)


def run_control(
    plan_path,
    views,
    output,
    *,
    windows=None,
    comparison_path=None,
    executor=None,
    lease=None,
    stop=None,
):
    """Ejecutar o reanudar los trabajos del control. El ejecutor y la reserva se pueden sustituir.

    `views` es la carpeta de vistas US que prepara `run_masked_campaign.py prepare`. Cada
    trabajo se ajusta en `jobs/<ventana>/<caso>/run` y se reanuda si la carpeta existe.
    """
    from .checkpoints import StopRequest

    require_learning_allowed("el control de la cabeza de cuantiles")
    control = load_control(plan_path, windows=windows, comparison_path=comparison_path)
    views_directory, output = Path(views), Path(output)
    checked = _views(control, views_directory)
    identity = _identity(control, checked)
    identity_sha256 = _digest(identity)
    _open_output(output, views_directory)
    executor = executor or _fit
    plan = control["plan"]
    receipts, rows = {}, _Rows()
    descriptor = os.open(output / ".lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _mark(output, identity)
        _summary(output, identity, control, receipts, "running")
        signals = StopRequest() if stop is None else nullcontext(stop)
        try:
            with signals as request, (lease or _gpu_lease)():
                for job in control["jobs"]:
                    job_identity = _job_identity(identity_sha256, job, checked)
                    receipt = _confirmed(output, job, job_identity, rows)
                    if receipt is None:
                        if request.requested:
                            raise Paused
                        require_learning_allowed(f"el trabajo {job['id']}")
                        folder = output / "jobs" / job["id"] / "run"
                        view = checked["windows"][job["window"]]
                        report = executor(
                            Path(view["path"]),
                            folder,
                            job["case"],
                            batch_size=plan["batch_size"],
                            checkpoint_seconds=CHECKPOINT_SECONDS,
                            resume=folder.exists(),
                            stop=request,
                            input_policy=plan["input_policy"],
                            prediction_retention=plan["prediction_retention"],
                        )
                        if report["status"] == "paused":
                            raise Paused
                        fold = control["scope"]["windows"][job["window"]]
                        receipt = _confirm(
                            output, job, job_identity, folder, report, view, fold, rows
                        )
                    receipts[job["id"]] = receipt
            status = "completed"
        except Paused:
            status = "paused"
        except BaseException as error:
            _summary(output, identity, control, receipts, "failed", error=str(error))
            raise
        return _summary(output, identity, control, receipts, status)
    finally:
        os.close(descriptor)


def _contrast_name(index):
    return f"index_{index:02d}"


def _model(arm, index):
    return f"{arm}@{index:02d}"


def contrast_scores(scores, plan, bootstrap):
    """Contrasta el delta de cada índice con las semillas promediadas sobre puntuaciones unidas.

    `scores` asigna a cada (brazo, índice, semilla) su `SessionScores` de validación con
    todas las ventanas del control. El resultado es el de `compare_series`.
    """
    series = {
        _model(arm, index): SessionSeries.average(
            [scores[arm, index, seed].series("mae") for seed in plan["seeds"]]
        )
        for index in plan["case_indices"]
        for arm in HEAD_CONTROL_ARMS
    }
    contrasts = {
        _contrast_name(index): delta(_model(SCALAR_L1, index), _model(QUANTILE_HEAD, index))
        for index in plan["case_indices"]
    }
    return compare_series(series, contrasts, **bootstrap)


def apply_fallback(result):
    """Aplica la regla declarada, que elige A si algún intervalo excluye el cero en contra.

    `delta` es MAE de la cabeza de cuantiles menos MAE escalar, así que un límite inferior
    positivo indica que la cabeza empeora el MAE dentro de la familia.
    """
    rows = result["contrasts"]
    against = [
        row["name"]
        for row in rows
        if row["simultaneous_interval"] is not None and row["simultaneous_interval"][0] > 0
    ]
    undefined = [row["name"] for row in rows if row["simultaneous_interval"] is None]
    option = OPTION_A if against else UNDETERMINED if undefined else OPTION_B
    return dict(
        option=option,
        against_quantile_head=against,
        undefined_intervals=undefined,
        consequence=CONSEQUENCES[option],
    )


def decide_control(plan_path, views, output, *, windows=None, comparison_path=None):
    """Calcular el contraste con los trabajos confirmados y escribir la decisión.

    Lee solo predicciones de validación ya confirmadas. La decisión no lleva marcas de
    tiempo, así que repetirla con los mismos recibos produce los mismos bytes.
    """
    control = load_control(plan_path, windows=windows, comparison_path=comparison_path)
    checked = _views(control, Path(views))
    identity = _identity(control, checked)
    identity_sha256 = _digest(identity)
    output = Path(output)
    marker = output / "control.json"
    _require(
        marker.is_file() and read_manifest(marker, 1024**2)[0] == identity,
        "La salida no corresponde a este control, sus ventanas, sus vistas o su código",
    )
    # Los recibos ya exigen las mismas filas y objetivos en cada ventana, y compare_series
    # rechaza series de poblaciones distintas.
    plan, rows = control["plan"], _Rows()
    parts, receipts = {}, {}
    for job in control["jobs"]:
        receipt = _confirmed(output, job, _job_identity(identity_sha256, job, checked), rows)
        _require(receipt is not None, f"Falta confirmar {job['id']} antes de decidir")
        receipts[job["id"]] = receipt["sha256"]
        record = receipt["validation"]
        table = comparison._read_predictions(
            dict(path=output / record["path"], sha256=record["sha256"]), _columns(job)
        )
        label = f"{job['id']} ({PARTITION})"
        head = QUANTILE_HEAD if job["arm"] == QUANTILE_HEAD else comparison.POINT
        panel = comparison._panel(
            table,
            control["scope"]["windows"][job["window"]],
            PARTITION,
            dict(markets=[SCOPE]),
            label,
            output=head,
        )
        key = job["arm"], job["index"], job["seed"]
        parts.setdefault(key, []).append(
            score_sessions(panel, rank_ic_min_assets=control["rank_ic_min_assets"])
        )
    joined = {key: SessionScores.concatenate(items) for key, items in parts.items()}
    result = contrast_scores(joined, plan, control["bootstrap"])
    decision = dict(
        schema_version=1,
        kind=DECISION_KIND,
        control_identity_sha256=identity_sha256,
        plan_sha256=control["plan_sha256"],
        comparison_sha256=control["comparison_sha256"],
        rule=plan["comparison"],
        windows=control["windows"],
        window_rule=control["window_rule"],
        contrast="delta = session_mae(quantile_head_v1) - session_mae(scalar_l1)",
        seeds="averaged_session_by_session",
        bootstrap=dict(
            control["bootstrap"],
            sensitivity_block_lengths=list(control["bootstrap"]["sensitivity_block_lengths"]),
        ),
        receipts=receipts,
        contrasts=result,
        decision=apply_fallback(result),
        final_test_opened=False,
    )
    destination = output / "decision.json"
    if destination.exists():
        _require(
            read_manifest(destination, 8 * 1024**2)[0] == decision,
            "Ya existe una decisión distinta para este control",
        )
    else:
        atomic_json(destination, decision)
    return decision


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check", help="Validar y contar los trabajos sin leer datos")
    execute = commands.add_parser("run", help="Ejecutar o reanudar los trabajos del control")
    decide = commands.add_parser("decide", help="Calcular el contraste y aplicar la regla")
    for command in (check, execute, decide):
        command.add_argument("--plan", type=Path, required=True)
        command.add_argument("--comparison", type=Path)
        command.add_argument("--windows", nargs="+")
    for command in (execute, decide):
        command.add_argument("--views", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    options = dict(windows=args.windows, comparison_path=args.comparison)
    if args.command == "check":
        result = check_control(args.plan, **options)
    elif args.command == "run":
        result = run_control(args.plan, args.views, args.output, **options)
        result.pop("jobs")
    else:
        result = decide_control(args.plan, args.views, args.output, **options)
        result = dict(status="decided", decision=result["decision"])
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] in {"checked", "completed", "decided"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
