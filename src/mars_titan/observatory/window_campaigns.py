"""Estado de las campañas por ventanas para el observatorio, leído sin tomar sus bloqueos.

La campaña base, la etapa de adaptadores, la ablación de modalidades y la etapa de
políticas escriben un `summary.json` con el mapa de trabajos confirmados. Aquí se convierte
en celdas de ámbito, ventana, brazo y nombre, igual para el servidor en directo y para el
recolector que publica Pages. Solo salen estados, fechas de confirmación y las curvas por
época de los intentos abiertos. Las métricas que lleve el resumen, como las financieras de
la etapa de políticas, no se copian.
"""

import math
import os
import re
import stat
from datetime import UTC, datetime
from pathlib import Path

from mars_titan.training.campaign_chain import CHAIN_SUFFIX, SELECTION, chain_folder

from .bounded_files import read_json, signature, utc_text

JOB_PART = re.compile(r"[A-Za-z0-9][\w.+-]{0,95}")
KIND = re.compile(r"[a-z][a-z0-9_]{0,95}")
COUNT_KEY = re.compile(r"[a-z][a-z_]{0,39}")
SELECT = re.compile(r"select-s(\d+)")
# Tipo de resumen de cada etapa, con su nombre en el observatorio y las partes de sus
# identificadores. Las políticas añaden el mercado y el predictor:
# `ámbito/mercado/ventana/predictor/brazo/nombre`.
STAGES = {
    "historical_masked_campaign_run": ("base", 4),
    "historical_masked_posttraining_stage_run": ("adapters", 4),
    "historical_masked_modality_ablation_stage_run": ("ablation", 4),
    "historical_masked_rl_stage_run": ("policies", 6),
}
# Brazos con el mismo nombre que su modelo del catálogo.
EXACT_MODELS = frozenset(
    {
        "rnn",
        "lstm",
        "gru",
        "dlinear",
        "ridge",
        "xgboost",
        "double_dqn",
        "cash",
        "hold_initial",
        "rebalance_50",
        "equal_weight_monthly",
        "market_index",
    }
)
# Familias que agrupan varios brazos: el nombre exacto o seguido de `_`. Así
# `titans_mac_online` y `titans_transformer_direct` son Titans-MAC, `mars_titan_m1_k4` es
# MARS-TITAN y `transformer_compact_online` es el control en línea del Transformer compacto.
FAMILY_MODELS = (
    ("transformer_compact", "transformer_compact"),
    ("titans", "titans_mac"),
    ("gru_episodic", "gru_episodic"),
    ("mars_titan", "mars_titan"),
    ("cm_v1", "cm_v1"),
    ("klpo", "klpo"),
    ("ppo", "ppo"),
)


def arm_model(arm):
    """Modelo del catálogo de un brazo, o `unknown` si no se reconoce.

    Los adaptadores (`base__punto`), el padre congelado y la cadena (`base__chain`) son el
    modelo de su brazo base. En la etapa de políticas el brazo es `predictor/política` y
    cuenta la política.
    """
    base = arm.rsplit("/", 1)[-1].split("__", 1)[0]
    if base in EXACT_MODELS:
        return base
    for family, model in FAMILY_MODELS:
        if base == family or base.startswith(family + "_"):
            return model
    return "unknown"


def _cell(parts):
    """Ámbito, ventana, brazo y nombre de un identificador de cuatro o seis partes."""
    if len(parts) == 4:
        return parts
    scope, market, window, predictor, arm, name = parts
    return f"{scope}/{market}", window, f"{predictor}/{arm}", name


def _confirmation(folder, jobs, job_id, scope, window, arm, name):
    """Archivo cuya fecha aproxima la confirmación del trabajo.

    La selección de la cadena no tiene recibo en `jobs/`: la confirma su `selection.json`,
    que la etapa de adaptadores escribe la última en la carpeta de la cadena. Las rutas
    son texto porque con decenas de miles de trabajos `pathlib` costaba más que `stat`.
    """
    select = SELECT.fullmatch(name)
    if select and arm.endswith(CHAIN_SUFFIX):
        base = arm.removesuffix(CHAIN_SUFFIX)
        return chain_folder(folder, scope, window, base, int(select[1])) / SELECTION
    return f"{jobs}/{job_id}/receipt.json"


def _directory(path):
    return path.is_dir() and not path.is_symlink()


def _is_folder(path):
    try:
        return stat.S_ISDIR(os.lstat(path).st_mode)
    except OSError:
        return False


def _open_attempt(jobs, job_id, confirmed):
    """Carpeta del intento abierto: el último `attempt-*` o `run`, que usan las etapas.

    La continuación de un ajuste con parada conjunta no crea carpeta hasta su recibo:
    reanuda en la de su meseta (`plateau-<nombre>`), cuyo intento pasa a ser el suyo en
    cuanto `confirmed` da la meseta por confirmada.
    """
    found = job_id
    if not _is_folder(f"{jobs}/{job_id}"):
        head, _, name = job_id.rpartition("/")
        found = f"{head}/plateau-{name}"
        if confirmed.get(found) is not True or not _is_folder(f"{jobs}/{found}"):
            return None
    job_folder = Path(jobs) / found
    attempts = sorted(p for p in job_folder.glob("attempt-*") if _directory(p))
    if attempts:
        return attempts[-1]
    run = job_folder / "run"
    return run if _directory(run) else None


def _utc(value):
    try:
        parsed = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _counts(value):
    """Recuentos por tipo de trabajo tal como los declara el resumen, o None."""
    if not isinstance(value, dict) or len(value) > 16:
        return None
    if not all(
        isinstance(key, str) and COUNT_KEY.fullmatch(key) and type(count) is int and count >= 0
        for key, count in value.items()
    ):
        return None
    return dict(value)


def _finite(value):
    return value if type(value) in (int, float) and math.isfinite(value) and value >= 0 else None


def epoch_point(epoch):
    """Medidas observadas de una época en curso. No se calcula ninguna medida nueva."""
    train = epoch.get("train") if isinstance(epoch.get("train"), dict) else {}
    validation = epoch.get("validation") if isinstance(epoch.get("validation"), dict) else {}
    return dict(
        epoch=epoch.get("epoch") if type(epoch.get("epoch")) is int else None,
        train_mae=_finite(train.get("mae")),
        mae=_finite(validation.get("mae")),
        session_mae=_finite(validation.get("session_mae")),
        train_samples_per_second=_finite(train.get("samples_per_second")),
        train_seconds=_finite(train.get("elapsed_seconds")),
    )


def _active(folder, job_id, attempt, max_bytes):
    relative = attempt.relative_to(folder) / "run.json"
    try:
        report, info = read_json(folder, relative, max_bytes)
        if not isinstance(report, dict):
            raise ValueError("El informe del intento no es un objeto")
    except (OSError, ValueError):
        return dict(job=job_id, attempt=attempt.name, updated_at=None, global_step=None, epochs=[])
    epochs = report.get("epochs") if isinstance(report.get("epochs"), list) else []
    return dict(
        job=job_id,
        attempt=attempt.name,
        updated_at=utc_text(info.st_mtime_ns / 1e9),
        global_step=report.get("global_step") if type(report.get("global_step")) is int else None,
        epochs=[epoch_point(epoch) for epoch in epochs[:2000] if isinstance(epoch, dict)],
    )


def campaign_state(label, folder, *, max_bytes=8 * 1024**2, max_jobs=20_000, max_active=8):
    """Normalizar el resumen de una campaña por ventanas sin tomar su bloqueo.

    `summary.json` declara qué trabajos están confirmados. Un trabajo sin confirmar con una
    carpeta de intento se marca como «intento sin confirmar», que no equivale a que el
    proceso siga vivo. La fecha de modificación de cada recibo aproxima su confirmación
    y sirve al navegador para estimar el ritmo, siempre rotulado como estimación.
    """
    folder = Path(folder)
    summary, info = read_json(folder, "summary.json", max_bytes)
    if not isinstance(summary, dict):
        raise ValueError("El resumen de la campaña no es un objeto")
    jobs = summary.get("jobs")
    if not isinstance(jobs, dict) or len(jobs) > max_jobs:
        raise ValueError("El resumen de la campaña no declara sus trabajos dentro del límite")
    kind = summary.get("kind") if isinstance(summary.get("kind"), str) else None
    stage, parts_expected = STAGES.get(kind, (None, None))
    vocabulary = {key: {} for key in ("scopes", "windows", "arms", "names")}

    def code(key, value):
        return vocabulary[key].setdefault(value, len(vocabulary[key]))

    cells, open_attempts = [], []
    jobs_folder = os.fspath(folder / "jobs")
    for job_id, done in jobs.items():
        parts = job_id.split("/")
        # Un tipo sin etapa conocida fija la forma con su primer trabajo.
        parts_expected = parts_expected or len(parts)
        if (
            type(done) is not bool
            or len(parts) != parts_expected
            or len(parts) not in (4, 6)
            or not all(JOB_PART.fullmatch(part) for part in parts)
        ):
            raise ValueError("Identificador de trabajo fuera del contrato de la campaña")
        scope, window, arm, name = _cell(parts)
        confirmed, state = None, "pending"
        if done:
            state = "done"
            receipt = signature(
                _confirmation(folder, jobs_folder, job_id, scope, window, arm, name)
            )
            confirmed = utc_text(receipt[2] / 1e9) if receipt else None
        else:
            attempt = _open_attempt(jobs_folder, job_id, jobs)
            if attempt is not None:
                state = "attempt"
                open_attempts.append((job_id, attempt))
        cells.append(
            [
                code("scopes", scope),
                code("windows", window),
                code("arms", arm),
                code("names", name),
                state,
                confirmed,
            ]
        )
    # Los intentos con el informe modificado más recientemente van primero.
    open_attempts.sort(key=lambda item: -(signature(item[1] / "run.json") or (0, 0, 0))[2])
    active = [
        _active(folder, job_id, attempt, max_bytes)
        for job_id, attempt in open_attempts[:max_active]
    ]
    final = summary.get("final_test_opened")
    return dict(
        id=label,
        kind=kind if kind and KIND.fullmatch(kind) else None,
        stage=stage,
        status=summary["status"]
        if isinstance(summary.get("status"), str) and KIND.fullmatch(summary["status"])
        else None,
        updated_at=_utc(summary.get("updated_at_utc")),
        summary_modified_at=utc_text(info.st_mtime_ns / 1e9),
        planned=_counts(summary.get("planned")),
        completed=_counts(summary.get("completed")),
        final_test_opened=final if isinstance(final, bool) else None,
        vocabulary={key: list(values) for key, values in vocabulary.items()},
        models={arm: arm_model(arm) for arm in vocabulary["arms"]},
        cells=cells,
        active=active,
    )
