"""Informe de integridad que bloquea la publicación de la campaña (#442).

`training.campaign_publication.run_publication` lo calcula en la carpeta provisional, después
de escribir comparaciones, matriz y comparación postentrenada y antes de dar a esa carpeta el
nombre del destino. Si alguna comprobación falla, el informe se queda en la carpeta provisional
para revisarlo y la publicación termina con un error. Así ninguna cifra se publica sin un
informe superado. Reúne:

- `comparisons`: mismas filas en todos los modelos, filas quitadas por el diseño, métricas
  recalculadas por otro camino y abstención con su cobertura en cada comparación
  walk-forward y en cada comparación postentrenada (`comparison_integrity`), y las marcas de
  abstención de la matriz que lee cada comparación walk-forward (`abstention_marks`);
- `runs`: todos los trabajos declarados de la campaña y de su etapa de adaptadores con sus
  intentos, sin recibos de más, y las semillas y ventanas de cada comparación publicada
  (`declared_runs`);
- `selection`: elección con la validación anterior a cada ventana, recibos de ventana, orden
  de los tramos y, con el diseño por etapas, la disjunción de #437 (`window_selection`);
- `budget`: presupuesto de cada modelo frente al de los modelos propios, con sus diferencias
  declaradas, y presupuesto consumido por cada ajuste confirmado (`fit_budget`).

Nada de este módulo ajusta modelos ni abre la reserva de 2024.
"""

import json
from datetime import UTC, datetime
from pathlib import Path

from mars_titan.data.storage import atomic_json, sha256
from mars_titan.evaluation import walk_forward_comparison as walk

from . import (
    abstention_marks,
    comparison_integrity,
    declared_runs,
    fit_budget,
    window_selection,
)

KIND = "campaign_publication_integrity"
REPORT = "integrity.json"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _comparison(entry):
    """Integridad de una comparación publicada y sus semillas y ventanas declaradas."""
    folder = Path(entry["folder"])
    checked = comparison_integrity.published(
        entry["config"],
        entry["sources"],
        entry["scope"],
        folder,
        aggregates=entry.get("aggregates"),
        ablation_sources=entry.get("ablation"),
        edition=entry.get("edition"),
    )
    report = json.loads((folder / "comparison.json").read_text(encoding="utf-8"))
    config = walk.resolve_config(entry["config"])
    sources = walk.load_sources(entry["sources"], config, entry["scope"])
    seeds = declared_runs.check_published(report, sources)
    passed = checked["passed"] and seeds["passed"]
    if entry.get("matrix") is not None:
        # La matriz lee las mismas sesiones, así que hereda la cobertura recalculada.
        matrix = json.loads(Path(entry["matrix"]).read_text(encoding="utf-8"))
        marks = abstention_marks.check_matrix(matrix, checked["abstention"]["coverage"])
        checked = dict(checked, matrix=marks)
        passed = passed and marks["passed"]
    return dict(checked, declared=seeds, passed=passed)


def _staged(posttraining):
    from mars_titan.posttraining import campaign_stage

    return posttraining is not None and posttraining["stage"]["design"] == campaign_stage.STAGED


def _runs(campaign, output, posttraining, adapter_output):
    """Trabajos declarados de la campaña y, si se publica, de su etapa de adaptadores."""
    from mars_titan.training.campaign_plan import plan_campaign

    jobs = plan_campaign(campaign)
    result = dict(campaign=declared_runs.check_output(jobs, output))
    if posttraining is not None:
        from mars_titan.posttraining import campaign_stage
        from mars_titan.training import campaign_chain

        stage = posttraining["stage"]
        planned = campaign_stage.plan_stage(stage)
        chains = campaign_stage.plan_chain(stage, planned) if _staged(posttraining) else []

        def selected(job):
            found = campaign_chain.read_selection(
                adapter_output, job["scope"], job["window"], job["base_arm"], job["seed"]
            )
            return found is not None

        result["posttraining"] = declared_runs.check_output(
            planned, adapter_output, chains=chains, selected=selected
        )
    return jobs, result


def build(loaded, views, output, comparisons, *, adapter_output=None):
    """Calcular el informe de integridad de una publicación sin escribir nada.

    `loaded` es la declaración validada de la publicación, `views` y `output` los de la
    campaña confirmada y `comparisons` la lista de comparaciones publicadas, cada una con
    `label`, `config`, `sources`, `scope`, `folder`, `edition` (la de los estratos de
    liquidez, si se declaran) y, si se calcularon desde la retención v2, `aggregates` y
    `ablation`.
    """
    campaign, posttraining = loaded["campaign_config"], loaded["posttraining"]
    checked = {entry["label"]: _comparison(entry) for entry in comparisons}
    jobs, runs = _runs(campaign, output, posttraining, adapter_output)
    selection = dict(
        base=window_selection.check_selection(campaign, jobs, output),
        partitions=window_selection.check_partitions(
            campaign["comparison_config"], campaign["scopes"]
        ),
    )
    if campaign.get("walk_forward_stages") or _staged(posttraining):
        selection["chain"] = window_selection.check_chain(
            campaign, views, output, adapter_output if posttraining is not None else None
        )
    budget = dict(campaign=fit_budget.check(campaign, jobs, output, loaded["budget_differences"]))
    sections = dict(comparisons=checked, runs=runs, selection=selection, budget=budget)
    failing = sorted(
        f"{name}/{key}"
        for name, entries in sections.items()
        for key, entry in entries.items()
        if not entry["passed"]
    )
    return dict(
        schema_version=1,
        kind=KIND,
        declaration=dict(name=loaded["name"], sha256=loaded["sha256"]),
        campaign=dict(name=campaign["name"], sha256=campaign["sha256"]),
        **sections,
        failing=failing,
        passed=not failing,
        created_at_utc=datetime.now(UTC).isoformat(),
        training_executed=False,
        final_test_opened=False,
    )


def write(folder, report):
    """Guardar el informe y detener la publicación si no está superado."""
    path = Path(folder) / REPORT
    atomic_json(path, report)
    _require(
        report["passed"],
        f"La integridad de la publicación falla en {', '.join(report['failing'])}: revisa {path}",
    )
    return dict(path=path, sha256=sha256(path))
