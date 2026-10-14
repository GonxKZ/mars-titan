"""Presupuesto declarado y consumido por cada modelo de la campaña (#442).

Una referencia con menos búsqueda, menos épocas o una parada distinta que los modelos propios
los favorece sin que se note en las métricas. Este módulo lo comprueba en dos niveles.

**Declarado.** Para cada modelo ajustado de la campaña reúne los casos de búsqueda, las
semillas y la forma de iterar (unidad, entrenador, máximo, regla de parada y paciencia), tal
como los resuelve el plan. Los modelos propios (MARS-TITAN, Titans-MAC y CM-v1) fijan el
presupuesto de referencia, el del primer modelo propio declarado. Cada campo en el que otro
modelo difiere debe figurar con su motivo en la declaración de la publicación
(`budget_differences`). Una diferencia sin declarar invalida la comparación, y una declarada
que ya no existe también hace fallar la comprobación, porque describiría otra campaña.

**Consumido.** Para cada ajuste confirmado lee su informe y, en los cronológicos, el registro
de épocas de su entrenador:

1. repite la regla de selección sobre su historial de validación y exige que el ajuste
   terminara justo donde la regla lo manda (todas las épocas con presupuesto fijo, la primera
   meseta con parada individual y la época común con parada conjunta) y que entregue la época
   que la regla elige;
2. un ajuste sin padre que entrega su estado inicial (época 0) no está ajustado. Una
   referencia sin ajustar no cuenta como control y un modelo propio sin ajustar tampoco
   cuenta como modelo;
3. con presupuesto fijo, los ajustes del mismo entrenador en el mismo ámbito y ventana aplican
   las mismas actualizaciones, y los de un grupo de parada conjunta terminan en la misma época;
4. XGBoost no supera sus rondas y elige una ronda ajustada, y el control en línea no usa más
   etiquetas que las escrituras del banco que lo limita;
5. la corrección B6 no tiene épocas ni parámetros: cada trabajo predice todos los tramos medidos
   sobre el padre congelado y escribe su memoria con etiquetas maduras en cada uno. La elección
   entre sus casos la hace la campaña con la validación, como en los demás modelos.

Solo lee declaraciones, recibos e informes. No ajusta nada ni abre la reserva de 2024.
"""

from pathlib import Path

from mars_titan.data.cohort_files import read_manifest
from mars_titan.data.storage import sha256
from mars_titan.training import campaign_plan as plan
from mars_titan.training.selection import (
    FINISH,
    FIXED_BUDGET,
    JOINT_PLATEAU,
    advance_selection,
    epoch_decision,
    individual_stop,
    initial_selection,
)

KIND = "fit_budget"
# Familias de los modelos propios, en el orden en que fijan el presupuesto de referencia.
OWN = (plan.MARS, plan.TITANS, plan.CM)
FIELDS = ("search_cases", "seeds", "iterations")
CHRONOLOGICAL = (plan.EPISODIC, plan.TITANS, plan.MARS, plan.CM)
CONFIRMED = ("completed",)
# B6 itera por escrituras maduras de su memoria en cada tramo medido, sin regla de parada.
MATURE_WRITES = dict(unit="mature_writes", trainer="mature_correction")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _epochs(rule, trainer):
    """Iteración por épocas con la regla de parada declarada para el modelo."""
    return dict(unit="epoch", trainer=trainer, **rule)


def _tabular(campaign, arm, candidates):
    kind = campaign["tabular"]["arms"][arm]
    if kind == "ridge":
        return dict(unit="closed_form")
    # Sin selección declarada, XGBoost agota sus rondas.
    shared = {
        (case["rounds"], repr(sorted((case.get("selection") or {}).items())))
        for _, case in candidates
    }
    _require(len(shared) == 1, f"Los casos de {arm} no comparten rondas ni selección")
    case = candidates[0][1]
    return dict(
        unit="round", trainer="xgboost", max_rounds=case["rounds"], **(case.get("selection") or {})
    )


def declared(campaign):
    """Presupuesto declarado de cada modelo ajustado de la campaña, sin leer resultados."""
    rules = plan._arm_rules(campaign["protocol_rule"], campaign["early_stop"])
    arms = campaign["comparison_config"]["arms"]
    budgets = {}
    for spec in plan._all_arm_specs(campaign):
        if spec["helper"]:
            continue
        arm, family = spec["arm"], spec["family"]
        rule = rules(arm) or campaign["protocol_rule"]
        if family == plan.MARS and plan._correction_arm(spec["candidates"][0][1]["components"]):
            iterations = dict(MATURE_WRITES)
        elif family == plan.NEURAL:
            iterations = _epochs(rule, "row_batches")
            iterations["batch_rows"] = campaign["neural"]["batch_size"]
        elif family in CHRONOLOGICAL:
            iterations = _epochs(rule, "chronological")
        else:
            iterations = _tabular(campaign, arm, spec["candidates"])
        budgets[arm] = dict(
            family=family,
            search_cases=len(spec["candidates"]),
            seeds=sorted(spec["seeds"]),
            iterations=iterations,
        )
    for arm, section in (campaign.get("online_controls") or {}).get("arms", {}).items():
        budgets[arm] = dict(
            family=plan.ONLINE_CONTROL,
            search_cases=len(section["search_cases"]),
            seeds=sorted(arms[arm]["seeds"]),
            iterations=dict(unit="update", trainer="online", **section["rule"]),
        )
    return budgets


def differences(budgets):
    """Modelo propio de referencia y campos en los que cada modelo difiere de él."""
    own = [arm for family in OWN for arm, item in budgets.items() if item["family"] == family]
    if not own:
        return None, {}
    reference = budgets[own[0]]
    found = {}
    for arm, item in budgets.items():
        fields = [field for field in FIELDS if item[field] != reference[field]]
        if fields:
            found[arm] = fields
    return own[0], found


def checked_declarations(declarations):
    """Validar la forma de `budget_differences`: modelo, campo y un motivo escrito."""
    _require(
        isinstance(declarations, dict)
        and all(
            isinstance(arm, str)
            and isinstance(fields, dict)
            and fields
            and all(
                field in FIELDS and isinstance(reason, str) and reason.strip()
                for field, reason in fields.items()
            )
            for arm, fields in declarations.items()
        ),
        "Las diferencias de presupuesto declaran por modelo cada campo distinto con su motivo",
    )
    return declarations


def check_declared(campaign, declarations):
    """Cotejar las diferencias reales de presupuesto con las declaradas en la publicación."""
    declarations = checked_declarations(declarations)
    budgets = declared(campaign)
    reference, found = differences(budgets)
    actual = {(arm, field) for arm, fields in found.items() for field in fields}
    stated = {(arm, field) for arm, fields in declarations.items() for field in fields}
    undeclared = sorted(f"{arm}/{field}" for arm, field in actual - stated)
    stale = sorted(f"{arm}/{field}" for arm, field in stated - actual)
    return dict(
        reference=reference,
        budgets=budgets,
        differences={arm: sorted(fields) for arm, fields in found.items()},
        undeclared=undeclared,
        stale=stale,
        passed=not undeclared and not stale,
    )


def _document(path):
    return read_manifest(path, 256 * 1024**2)[0]


def _fit_documents(output, receipt):
    """Informe confirmado del ajuste y documento con sus épocas.

    Las ventanas cronológicas guardan el registro de su entrenador en `fit` (Titans-MAC,
    MARS-TITAN y CM-v1) o en `run` (GRU episódica), con su huella. La referencia neuronal y
    los tabulares lo llevan en el propio informe.
    """
    path = Path(output) / receipt["report"]["path"]
    _require(
        path.is_file() and sha256(path) == receipt["report"].get("sha256"),
        f"{path}: el informe no es el que confirma su recibo",
    )
    report = _document(path)
    for key in ("fit", "run"):
        record = report.get(key)
        if isinstance(record, dict) and isinstance(record.get("path"), str):
            nested = path.parent / record["path"]
            _require(
                nested.is_file() and sha256(nested) == record.get("sha256"),
                f"{path}: el registro del entrenador no coincide con su huella",
            )
            return report, _document(nested)
    return report, report


def _scores(run):
    """Validación de cada época evaluada, con la época 0 si el ajuste la evaluó."""
    if isinstance(run.get("history"), list):
        entries = run["history"]
    elif isinstance(run.get("epochs"), list):
        entries = list(run["epochs"])
        if run.get("initial_validation") is not None:
            entries = [dict(epoch=0, validation=run["initial_validation"]), *entries]
    else:
        return None
    return [(entry["epoch"], entry["validation"]["session_mae"]) for entry in entries]


def replay(scores, rule, joint_epoch=None):
    """Repetir la regla de selección sobre un historial y devolver su selección y su final.

    El final es la posición en la que la regla manda terminar, o None si no llega a mandarlo.
    Con parada conjunta, la época común solo cuenta desde la primera meseta del ajuste, como
    en el trabajo final que continúa tras ella.
    """
    options = {key: value for key, value in rule.items() if key != "max_epochs"}
    maximum, selection = rule["max_epochs"], None
    for index, (epoch, score) in enumerate(scores):
        if epoch == 0:
            _require(index == 0, "La época 0 solo puede ser la primera evaluación")
            selection = initial_selection(score, options)
        else:
            selection = advance_selection(selection, score, epoch, options)
        common = joint_epoch if individual_stop(selection, maximum) is not None else None
        if epoch_decision(selection, options, maximum, common) == FINISH:
            return selection, index
    return selection, None


def check_epochs(job, rule, run, report):
    """Comprobar un ajuste por épocas frente a su regla declarada.

    `run` es el documento con las épocas evaluadas y la selección, y `report` el informe que
    confirma el recibo. Son el mismo en la referencia neuronal.
    """
    scores = _scores(run)
    if not scores:
        return dict(failures=["el informe no declara sus épocas evaluadas"])
    joint_epoch = run.get("joint_stop_epoch", report.get("joint_stop_epoch"))
    joint_epoch = joint_epoch if rule["stopping"] == JOINT_PLATEAU else None
    failures = []
    try:
        selection, finished = replay(scores, rule, joint_epoch)
    except ValueError as error:
        return dict(failures=[f"historial que la regla no admite: {error}"])
    last = scores[-1][0]
    if finished is None:
        failures.append(f"termina en la época {last} antes de lo que manda su regla")
    elif finished != len(scores) - 1:
        failures.append(f"sigue tras la época {scores[finished][0]} en la que su regla termina")
    reported = run.get("selection") or {}
    if reported.get("best_epoch") != selection["best_epoch"]:
        failures.append(
            f"entrega la época {reported.get('best_epoch')} y la regla elige la "
            f"{selection['best_epoch']}"
        )
    steps = run.get("global_step", (report.get("fit") or {}).get("global_step"))
    if type(steps) is not int or steps < 1:
        failures.append(f"declara {steps} actualizaciones")
    fitted = job.get("parent") is not None or selection["best_epoch"] > 0
    if not fitted:
        failures.append("entrega su estado inicial sin ajustar")
    return dict(
        evaluated_epochs=last,
        best_epoch=selection["best_epoch"],
        global_step=steps,
        joint_stop_epoch=joint_epoch,
        fitted=fitted,
        failures=failures,
    )


def _xgboost_fit(budget, report):
    rounds, selected = report.get("completed_rounds"), report.get("selected_round")
    failures = []
    if type(rounds) is not int or not 1 <= rounds <= budget["max_rounds"]:
        failures.append(f"rondas completadas {rounds} fuera de 1 a {budget['max_rounds']}")
    elif type(selected) is not int or not 1 <= selected <= rounds:
        failures.append(f"ronda elegida {selected} sin ajustar o posterior a las completadas")
    elif rounds < budget["max_rounds"] and not report.get("stop_reason"):
        failures.append("para antes de sus rondas sin declarar por qué")
    return dict(completed_rounds=rounds, selected_round=selected, failures=failures)


def _online_fit(report):
    """El control en línea no usa más etiquetas que las escrituras del banco que lo limita."""
    failures, used = [], {}
    admitted = report.get("bank", {}).get("admitted", {})
    for name, record in report.get("predictions", {}).items():
        metrics = record.get("metrics", {})
        labels, cap = metrics.get("labels_used"), admitted.get(name)
        used[name] = labels
        if type(labels) is not int or type(cap) is not int or labels > cap:
            failures.append(f"{name}: usa {labels} etiquetas con {cap} escrituras del banco")
        elif metrics.get("cap") != cap:
            failures.append(f"{name}: el tope aplicado no es el del banco")
    if not used:
        failures.append("el informe no declara las etiquetas usadas por tramo")
    return dict(labels_used=used, failures=failures)


def _correction_fit(report):
    """Cada tramo medido de B6 está predicho y su memoria escribió etiquetas maduras."""
    expected = report.get("identity", {}).get("expected_rows")
    if not isinstance(expected, dict) or not expected:
        return dict(writes={}, failures=["el informe no declara sus tramos medidos"])
    failures, writes = [], {}
    for name in expected:
        record = report.get("predictions", {}).get(name)
        count = (record or {}).get("metrics", {}).get("associative_writes")
        writes[name] = count
        if record is None:
            failures.append(f"{name}: tramo sin predecir")
        elif type(count) is not int or count < 1:
            failures.append(f"{name}: la memoria no escribe ninguna etiqueta madura")
    return dict(writes=writes, fitted=not failures, failures=failures)


def _receipt(output, job_id):
    path = Path(output) / "jobs" / job_id / "receipt.json"
    return read_manifest(path, 8 * 1024**2)[0] if path.is_file() else None


def _rule(campaign, arm):
    rules = plan._arm_rules(campaign["protocol_rule"], campaign["early_stop"])
    return rules(arm) or campaign["protocol_rule"]


def _consumed(campaign, budget, job, output, receipt):
    """Presupuesto consumido por un ajuste confirmado según la forma en que itera."""
    report, run = _fit_documents(output, receipt)
    if job["kind"] == plan.ONLINE:
        return _online_fit(report)
    # Los auxiliares, como los núcleos de CM-v1, iteran por épocas con la regla común.
    unit = "epoch" if budget is None else budget["iterations"]["unit"]
    if unit == "epoch":
        return check_epochs(job, _rule(campaign, job["arm"]), run, report)
    if unit == "round":
        return _xgboost_fit(budget["iterations"], report)
    if unit == MATURE_WRITES["unit"]:
        return _correction_fit(report)
    # Ridge tiene solución cerrada: su recibo confirmado basta.
    return dict(failures=[])


def check_consumed(campaign, jobs, output):
    """Presupuesto consumido por cada ajuste confirmado frente al declarado."""
    budgets = declared(campaign)
    entries, groups, joint = [], {}, {}
    for job in jobs:
        if job["kind"] not in (plan.FIT, plan.ONLINE) or job.get("phase") == plan.PLATEAU:
            continue
        receipt = _receipt(output, job["id"])
        if receipt is None or receipt.get("status") not in CONFIRMED:
            # Los trabajos sin confirmar ya los cuenta `declared_runs`.
            continue
        try:
            entry = _consumed(campaign, budgets.get(job["arm"]), job, output, receipt)
        except ValueError as error:
            entry = dict(failures=[str(error)])
        entry = dict(id=job["id"], arm=job["arm"], **entry)
        entries.append(entry)
        fixed = _rule(campaign, job["arm"])["stopping"] == FIXED_BUDGET
        if fixed and "global_step" in entry:
            key = (job["model"], job["scope"], job["window"])
            groups.setdefault(key, {})[job["id"]] = entry["global_step"]
        if job.get("joint_group") and "joint_stop_epoch" in entry:
            joint.setdefault(tuple(job["joint_group"]), {})[job["id"]] = entry["evaluated_epochs"]
    unequal = sorted(
        f"{'/'.join(key)}: actualizaciones {sorted(set(steps.values()))}"
        for key, steps in groups.items()
        if len(set(steps.values())) > 1
    )
    unequal += sorted(
        f"grupo de {group[0]}: épocas {sorted(set(epochs.values()))}"
        for group, epochs in joint.items()
        if len(set(epochs.values())) > 1
    )
    failing = [entry["id"] for entry in entries if entry["failures"]]
    return dict(
        fits=len(entries),
        unfitted=sorted(entry["id"] for entry in entries if entry.get("fitted") is False),
        failing=failing,
        unequal_updates=unequal,
        jobs=entries,
        passed=not failing and not unequal,
    )


def check(campaign, jobs, output, declarations):
    """Sección `budget` del informe de integridad de la publicación."""
    stated = check_declared(campaign, declarations)
    consumed = check_consumed(campaign, jobs, output)
    return dict(
        kind=KIND,
        declared=stated,
        consumed=consumed,
        passed=stated["passed"] and consumed["passed"],
    )
