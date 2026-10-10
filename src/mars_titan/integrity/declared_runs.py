"""Todas las ejecuciones declaradas aparecen en la publicación, también las que fallaron.

Un resultado deja de ser creíble si se ejecutan más casos de los declarados y se publica el
mejor, o si se quita de la comparación una semilla o una ventana que salió mal. Esta
comprobación vuelve a planificar la campaña y su etapa de adaptadores desde sus
declaraciones y lo coteja con lo que hay en disco y con lo que publican las comparaciones:

1. cada trabajo planificado tiene su recibo confirmado (en la cadena, su selección) y se
   cuentan todos sus intentos, también los que fallaron o se interrumpieron;
2. no hay recibos de trabajos que el plan no declara;
3. cada comparación publicada contiene exactamente las semillas y las ventanas declaradas de
   cada modelo.

El informe lista cada trabajo con su estado, sus intentos y, si su informe lo dice, si paró
antes de agotar sus épocas y por qué. No elige ni descarta ninguno.
"""

from pathlib import Path

from mars_titan.data.cohort_files import read_manifest

KIND = "declared_runs"
# Campos de los informes de ajuste que dicen si un ajuste paró antes de su presupuesto.
STOP_FIELDS = ("stopped_early", "stop_reason", "plateau_epoch")


def _receipt(output, job_id):
    path = Path(output) / "jobs" / job_id / "receipt.json"
    return read_manifest(path, 8 * 1024**2)[0] if path.is_file() else None


def _stop(output, receipt):
    """Parada del ajuste según el informe que confirma el recibo, si declara alguna."""
    record = receipt.get("report") or receipt.get("run")
    if not isinstance(record, dict) or "path" not in record:
        return {}
    path = Path(output) / record["path"]
    if not path.is_file():
        return dict(report_missing=True)
    report, _ = read_manifest(path, 256 * 1024**2)
    return {name: report[name] for name in STOP_FIELDS if name in report}


def _job(output, job):
    folder = Path(output) / "jobs" / job["id"]
    attempts = sorted(p.name for p in folder.glob("attempt-*")) if folder.is_dir() else []
    receipt = _receipt(output, job["id"])
    confirmed = receipt is not None and receipt.get("status") in ("completed", "plateau_confirmed")
    entry = dict(
        id=job["id"],
        kind=job["kind"],
        stage=job.get("stage"),
        status="missing" if receipt is None else receipt.get("status"),
        attempts=len(attempts),
        # El intento confirmado es uno de ellos. Los demás fallaron o se interrumpieron.
        unconfirmed_attempts=len(attempts) - (1 if confirmed and attempts else 0),
    )
    if confirmed:
        entry.update(_stop(output, receipt))
    return entry, confirmed


def _extra(output, planned):
    """Recibos en disco de trabajos que el plan no declara."""
    root = Path(output) / "jobs"
    found = []
    for path in sorted(root.glob("**/receipt.json")):
        job_id = path.parent.relative_to(root)
        # Los intentos guardan sus propios artefactos. Solo cuenta el recibo del trabajo.
        if any(part.startswith("attempt-") for part in job_id.parts):
            continue
        if str(job_id) not in planned:
            found.append(str(job_id))
    return found


def check_output(jobs, output, *, chains=(), selected=None):
    """Cotejar el plan `jobs` con los recibos de `output`.

    `chains` son los trabajos de selección de la cadena y `selected(job)` dice si su
    selección está confirmada.
    """
    entries, missing = [], []
    for job in jobs:
        entry, confirmed = _job(output, job)
        entries.append(entry)
        if not confirmed:
            missing.append(job["id"])
    for job in chains:
        confirmed = selected(job)
        entries.append(
            dict(id=job["id"], kind=job["kind"], status="selected" if confirmed else "missing")
        )
        if not confirmed:
            missing.append(job["id"])
    extra = _extra(output, {job["id"] for job in jobs})
    return dict(
        planned=len(jobs) + len(chains),
        reported=len(entries),
        confirmed=len(entries) - len(missing),
        unconfirmed_attempts=sum(entry.get("unconfirmed_attempts", 0) for entry in entries),
        stopped_early=sum(entry.get("stopped_early") is True for entry in entries),
        missing=missing,
        undeclared=extra,
        jobs=entries,
        passed=not missing and not extra and len(entries) == len(jobs) + len(chains),
    )


def check_published(report, sources):
    """Semillas y ventanas de cada modelo en una comparación publicada frente a su declaración."""
    differing = []
    windows = set(sources["windows"])
    for arm, declared in sources["arms"].items():
        published = report["arms"].get(arm)
        seeds = {str(seed) for seed in declared["seeds"]} or {"deterministic"}
        if published is None or set(published["seeds"]) != seeds:
            differing.append(f"{arm}: semillas publicadas distintas de las declaradas")
            continue
        for seed, entry in published["seeds"].items():
            if set(entry["windows"]) != windows:
                differing.append(f"{arm} semilla {seed}: ventanas distintas de las declaradas")
    extra = sorted(set(report["arms"]) - set(sources["arms"]))
    differing += [f"{arm}: modelo publicado sin declarar" for arm in extra]
    return dict(arms=len(sources["arms"]), differing=differing, passed=not differing)
