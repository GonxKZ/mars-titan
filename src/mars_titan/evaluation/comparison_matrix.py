"""Matriz de comparaciones y atribución por componentes, declaradas antes de los resultados.

La declaración (``comparison_matrix``) fija qué se compara y con qué pregunta: entre
familias, cada variante de MARS-TITAN frente a cada referencia, el núcleo frente a la
implementación de referencia, la cadena por etapas y el control en línea. Los linajes de
``component_attribution`` añaden la escalera acumulada, dejar uno fuera, los efectos
condicionados, las interacciones y Shapley. Cada pregunta se compila en familias de
contrastes lineales. La corrección por máximo estudentizado se aplica dentro de cada
familia, igual que en la comparación walk-forward, y no entre familias.

La evaluación lee las tablas por sesión que ya publican otras comparaciones mediante
``session_table_contrasts``, sin volver a puntuar predicciones. Los contrastes cuyos brazos
todavía no existen quedan pendientes con lo que les falta, y el informe de brazos que faltan
estima su coste con un documento de horas por brazo, medido o proyectado, que se declara
como tal.

Nada de este módulo ajusta modelos, lee la edición ni abre la reserva de 2024.
"""

import argparse
import json
import math
import resource
import time
from datetime import UTC, date, datetime
from importlib.metadata import version
from pathlib import Path

from mars_titan.data.cohort_files import read_manifest, safe_destination
from mars_titan.data.storage import atomic_json, outside_source, sha256

from . import component_attribution as attribution
from . import long_short
from . import session_table_contrasts as tables
from . import walk_forward_comparison as walk
from .paired_comparisons import MAX_CONTRASTS, delta

KIND = "comparison_matrix"
REPORT_KIND = "comparison_matrix_report"
MISSING_KIND = "comparison_matrix_missing_arms"
DECLARED = "declared_before_results"
USE = "declared_contrasts_and_component_attribution_not_for_selecting_the_architecture"
_FIELDS = {
    "schema_version",
    "kind",
    "name",
    "status",
    "declared_at",
    "use",
    "comparison",
    "conditional_arms",
    "derived_arms",
    "groups",
    "questions",
    "lineages",
    "candidates",
    "views",
}
_QUESTION_FIELDS = {
    "pairwise": {"kind", "arms", "question"},
    "against": {"kind", "each", "references", "question"},
    "pairs": {"kind", "pairs", "question"},
}
_CANDIDATE_FIELDS = {"lineage", "producer", "code", "change", "cost_like", "purpose"}
PRODUCERS = ("reader_fit", "core_fit", "prediction_only")
CODE = ("existing", "requires_change")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _issue(value):
    return type(value) is int and value > 0


# Declaración


def _conditions(value, label):
    _require(isinstance(value, dict), f"{label} deben ser un diccionario")
    for name, entry in value.items():
        _require(
            tables.is_arm_name(name)
            and isinstance(entry, dict)
            and set(entry) == {"condition", "issue"}
            and tables.is_text(entry["condition"])
            and _issue(entry["issue"]),
            f"{label}: {name} declara condición e issue",
        )
    return value


class Arms:
    """Clasifica cada brazo que nombra la matriz según quién lo produce.

    Un brazo puede ser de la campaña declarada, condicionado a otra tarea, derivado de uno
    declarado con un sufijo conocido o candidato. La clase decide si un contraste se puede
    estimar con la campaña o qué le falta, sin consultar ningún dato.
    """

    def __init__(self, declared, conditional, suffixes, candidates):
        self.declared, self.conditional = set(declared), conditional
        self.suffixes, self.candidates = suffixes, candidates

    def kind(self, arm):
        """Devuelve la clase del brazo, o None si la matriz no lo conoce.

        Un derivado solo se reconoce si su base es un brazo declarado, para que un sufijo
        no convierta en válido un nombre inventado.
        """
        if arm in self.declared:
            return "declared"
        if arm in self.conditional:
            return "conditional"
        if arm in self.candidates:
            return "candidate"
        base, separator, suffix = arm.rpartition("__")
        if separator and base in self.declared and suffix in self.suffixes:
            return "derived"
        return None

    def require(self, arm, label):
        """Rechaza un brazo sin clase, porque daría un contraste que nunca podría estimarse."""
        _require(
            tables.is_arm_name(arm) and self.kind(arm) is not None,
            f"{label} usa el brazo {arm} sin declarar",
        )
        return arm


def _group(groups, name, label):
    _require(isinstance(name, str) and name in groups, f"{label} nombra un grupo no declarado")
    return groups[name]


def _groups(value, arms):
    _require(isinstance(value, dict) and value, "Faltan los grupos de brazos")
    for name, members in value.items():
        _require(
            walk._name(name)
            and isinstance(members, list)
            and members
            and len(set(members)) == len(members),
            f"El grupo {name} necesita brazos distintos",
        )
        for arm in members:
            arms.require(arm, f"El grupo {name}")
    return value


def _family(name, question, origin, contrasts):
    _require(walk._name(name), f"La familia {name} necesita un nombre válido")
    _require(
        1 <= len(contrasts) <= MAX_CONTRASTS,
        f"La familia {name} tiene entre 1 y {MAX_CONTRASTS} contrastes",
    )
    return dict(question=question, origin=origin, contrasts=contrasts)


def _pair_contrasts(pairs, label):
    contrasts = {}
    for base, variant in pairs:
        _require(base != variant, f"{label} compara {base} consigo mismo")
        name = f"{variant}-{base}"
        _require(name not in contrasts, f"{label} repite el contraste {name}")
        contrasts[name] = dict(coefficients=delta(base, variant), unnamed=[])
    return contrasts


def _question(name, question, groups, arms):
    """Compila las familias de una pregunta declarada.

    Si la pregunta tiene ``each``, cada miembro forma su propia familia. Así la corrección
    múltiple se aplica a las comparaciones de un mismo brazo y no mezcla variantes que
    responden a preguntas distintas.
    """
    kind = question.get("kind") if isinstance(question, dict) else None
    fields = _QUESTION_FIELDS.get(kind)
    _require(
        walk._name(name)
        and fields is not None
        and (set(question) == fields or (kind == "pairs" and set(question) == fields | {"each"}))
        and tables.is_text(question["question"]),
        f"La pregunta {name} no está bien declarada",
    )
    origin, text = f"question:{name}", question["question"]
    if kind == "pairwise":
        members = _group(groups, question["arms"], name)
        pairs = [(a, b) for i, a in enumerate(members) for b in members[i + 1 :]]
        return {name: _family(name, text, origin, _pair_contrasts(pairs, name))}
    if kind == "against":
        references = _group(groups, question["references"], name)
        return {
            f"{name}__{member}": _family(
                f"{name}__{member}",
                text,
                origin,
                _pair_contrasts([(ref, member) for ref in references if ref != member], name),
            )
            for member in _group(groups, question["each"], name)
        }
    declared = question["pairs"]
    _require(
        isinstance(declared, list)
        and declared
        and all(
            isinstance(pair, list) and len(pair) == 2 and all(isinstance(a, str) for a in pair)
            for pair in declared
        ),
        f"La pregunta {name} declara pares [base, variante]",
    )
    templated = any("{arm}" in arm for pair in declared for arm in pair)
    _require(
        templated == ("each" in question),
        f"La pregunta {name} usa {{arm}} solo con each, y each solo con {{arm}}",
    )
    members = _group(groups, question["each"], name) if templated else [None]
    families = {}
    for member in members:
        family = name if member is None else f"{name}__{member}"
        pairs = [[arm.replace("{arm}", member or "") for arm in pair] for pair in declared]
        for arm in (arm for pair in pairs for arm in pair):
            arms.require(arm, family)
        families[family] = _family(family, text, origin, _pair_contrasts(pairs, family))
    return families


def _candidates(value, lineages, arms):
    _require(isinstance(value, dict), "Los candidatos deben ser un diccionario")
    for name, entry in value.items():
        _require(
            tables.is_arm_name(name)
            and isinstance(entry, dict)
            and set(entry) == _CANDIDATE_FIELDS
            and entry["lineage"] in lineages
            and name in lineages[entry["lineage"]].arms
            and entry["producer"] in PRODUCERS
            and entry["code"] in CODE
            and all(tables.is_text(entry[key]) for key in ("change", "purpose")),
            f"El candidato {name} no está bien declarado",
        )
        _require(
            arms.kind(name) == "candidate",
            f"El candidato {name} ya está declarado como brazo de la campaña o condicionado",
        )
        _require(
            arms.kind(entry["cost_like"]) in ("declared", "conditional"),
            f"El candidato {name} debe estimar su coste con un brazo declarado",
        )
    return value


def _views(value):
    _require(isinstance(value, dict) and value, "Faltan las vistas de métrica")
    for name, view in value.items():
        source = view.get("source") if isinstance(view, dict) else None
        _require(walk._name(name), f"La vista {name} necesita un nombre válido")
        if source == tables.WALK_SOURCE:
            metrics = view.get("metrics")
            _require(
                set(view) == {"source", "quantiles", "metrics"}
                and view["quantiles"] in ("raw", "calibrated")
                and isinstance(metrics, list)
                and metrics
                and len(set(metrics)) == len(metrics)
                and set(metrics) <= set(tables.WALK_METRICS)
                and (
                    view["quantiles"] == "raw"
                    or all(tables.quantile_metric(metric) for metric in metrics)
                ),
                f"La vista {name} declara métricas por sesión de la comparación",
            )
        elif source == tables.PORTFOLIO_SOURCE:
            statistics = view.get("statistics")
            _require(
                set(view) == {"source", "statistics"}
                and isinstance(statistics, list)
                and statistics
                and len(set(statistics)) == len(statistics)
                and set(statistics) <= set(long_short.STATISTICS),
                f"La vista {name} declara estadísticos de la cartera",
            )
        else:
            metrics = view.get("metrics") if isinstance(view, dict) else None
            _require(
                source == tables.TABLE_SOURCE and _issue(view.get("issue")),
                f"La vista {name} no tiene una fuente admitida",
            )
            if metrics is None:
                _require(
                    set(view) == {"source", "metrics", "pending", "issue"}
                    and tables.is_text(view["pending"]),
                    f"La vista {name} sin métricas declara por qué está pendiente",
                )
            else:
                _require(
                    set(view) == {"source", "metrics", "issue"}
                    and isinstance(metrics, dict)
                    and metrics
                    and all(
                        walk._name(metric) and kind in tables.ORIENTATIONS
                        for metric, kind in metrics.items()
                    ),
                    f"La vista {name} declara cada métrica como pérdida o ganancia",
                )
    return value


def _lineage_families(lineage, arms):
    for arm in lineage.arms:
        arms.require(arm, f"El linaje {lineage.name}")
    compiled, limitations = attribution.families(lineage)
    families = {}
    for kind, terms in compiled.items():
        contrasts = {}
        for name, term in terms.items():
            coefficients, unnamed = attribution.resolve(lineage, term)
            entry = dict(coefficients=coefficients, unnamed=unnamed)
            if kind == "leave_one_out":
                entry["removes"] = attribution.removed_with(lineage, name[1:])
            contrasts[name] = entry
        family = f"{lineage.name}__{kind}"
        families[family] = _family(
            family, lineage.question, f"lineage:{lineage.name}/{kind}", contrasts
        )
    return families, [dict(lineage=lineage.name, **item) for item in limitations]


def load_matrix(path):
    """Valida la declaración, carga su comparación y compila todas las familias.

    La matriz se compila completa antes de leer ningún informe, de modo que el número de
    familias y contrastes queda fijado por la declaración y no por los resultados.
    """
    path = Path(path)
    declaration, digest = read_manifest(path, 4 * 1024**2)
    _require(
        isinstance(declaration, dict)
        and set(declaration) == _FIELDS
        and declaration["schema_version"] == 1
        and declaration["kind"] == KIND
        and declaration["status"] == DECLARED
        and declaration["use"] == USE
        and isinstance(declaration["name"], str)
        and walk._name(declaration["name"].replace("-", "_")),
        "La matriz de comparaciones no cumple su contrato",
    )
    try:
        date.fromisoformat(declaration["declared_at"])
    except (TypeError, ValueError) as error:
        raise ValueError("La matriz necesita su fecha de declaración") from error
    comparison = walk.load_config(path.parent / declaration["comparison"])
    conditional = _conditions(declaration["conditional_arms"], "Los brazos condicionados")
    suffixes = _conditions(declaration["derived_arms"], "Los brazos derivados")
    lineages = {
        name: attribution.load_lineage(name, entry)
        for name, entry in _items(declaration["lineages"], "Los linajes")
    }
    arms = Arms(comparison["arms"], conditional, suffixes, declaration["candidates"])
    _candidates(declaration["candidates"], lineages, arms)
    groups = _groups(declaration["groups"], arms)
    families, limitations = {}, []
    for name, question in _items(declaration["questions"], "Las preguntas"):
        compiled = _question(name, question, groups, arms)
        _require(not set(compiled) & set(families), f"La pregunta {name} repite una familia")
        families.update(compiled)
    for lineage in lineages.values():
        compiled, limits = _lineage_families(lineage, arms)
        _require(not set(compiled) & set(families), f"El linaje {lineage.name} repite familias")
        families.update(compiled)
        limitations += limits
    _views(declaration["views"])
    return dict(
        declaration,
        sha256=digest,
        comparison_config=comparison,
        arms=arms,
        families=families,
        limitations=limitations,
        resolved_lineages=lineages,
    )


def _items(value, label):
    _require(isinstance(value, dict) and value, f"{label} deben ser un diccionario no vacío")
    return value.items()


def split(family, available):
    """Separa los contrastes estimables con los brazos disponibles de los pendientes.

    Un contraste al que le falta algún brazo no se estima con los términos que sí existen,
    porque cambiaría su significado. Queda pendiente con la lista de lo que falta.
    """
    ready, pending = {}, []
    for name, contrast in family["contrasts"].items():
        missing = sorted(arm for arm in contrast["coefficients"] if arm not in available)
        if missing or contrast["unnamed"]:
            pending.append(dict(name=name, missing=missing, unnamed=contrast["unnamed"]))
        else:
            ready[name] = contrast["coefficients"]
    return ready, pending


def evaluate(matrix_path, sources_path, scope):
    """Calcula el informe de la matriz para un ámbito sin escribir nada.

    Las familias se reducen a sus contrastes estimables antes de calcular las vistas, y cada
    familia conserva su corrección por máximo estudentizado con esos contrastes. El informe
    guarda las huellas de la matriz, las fuentes y el código para poder reproducirlo.
    """
    started = time.perf_counter()
    matrix = load_matrix(matrix_path)
    config = matrix["comparison_config"]
    sources = tables.load_sources(sources_path, matrix["views"], config, scope)
    markets, hours = list(sources["markets"]), sources["hours"]
    forecast = tables.forecast_arms(sources)
    available = set(forecast["raw"]) | set(forecast["calibrated"])
    published = {}
    for name, view in matrix["views"].items():
        if view["source"] == tables.TABLE_SOURCE and view["metrics"] is not None:
            published[name] = tables.table_arms(sources, name, view["metrics"])
            available |= {arm for _, arm in published[name]}
    portfolios = [i for i in sources["reports"] if i["kind"] == tables.PORTFOLIO_SOURCE]
    declaration = portfolios[0]["report"]["declaration"] if portfolios else None
    portfolio = None
    if declaration is not None:
        portfolio = tables.portfolio_arms(sources, declaration["cost_bps_per_side"])
        available |= {arm for arms in portfolio.values() for arm in arms}
    estimable, families = {}, {}
    for name, family in matrix["families"].items():
        ready, pending = split(family, available)
        if ready:
            estimable[name] = ready
        families[name] = dict(
            question=family["question"],
            origin=family["origin"],
            estimable=list(ready),
            pending=pending,
        )
    views = {}
    for name, view in matrix["views"].items():
        if view["source"] == tables.WALK_SOURCE:
            views[name] = tables.forecast_view(
                config, estimable, forecast[view["quantiles"]], view["metrics"], markets, hours
            )
        elif view["source"] == tables.PORTFOLIO_SOURCE:
            views[name] = (
                dict(reason="No hay informes de cartera en las fuentes")
                if portfolio is None
                else tables.portfolio_view(
                    config, estimable, portfolio, declaration, view["statistics"], hours
                )
            )
        elif view["metrics"] is None:
            views[name] = dict(reason=view["pending"], issue=view["issue"])
        else:
            views[name] = tables.table_view(
                config, estimable, published[name], view["metrics"], markets, hours
            )
    comparison = config["comparison"]
    return dict(
        schema_version=1,
        kind=REPORT_KIND,
        status="completed",
        created_at_utc=datetime.now(UTC).isoformat(),
        final_test_opened=False,
        scope=scope,
        markets=markets,
        matrix=dict(name=matrix["name"], sha256=matrix["sha256"]),
        comparison=dict(name=config["name"], sha256=config["sha256"]),
        sources_sha256=sources["sha256"],
        reports=[
            dict(kind=item["kind"], path=str(item["path"]), sha256=item["sha256"])
            for item in sources["reports"]
        ],
        windows=sources["windows"],
        resampling=dict(
            method="circular_block_bootstrap",
            unit="utc_calendar_day_with_all_sessions_and_assets",
            portfolio_unit="market_session_in_calendar_order",
            market_weighting=config["metrics"]["market_weighting"],
            multiplicity="max_absolute_studentized_bootstrap_within_each_family",
            block_length=comparison["block_length"],
            sensitivity_block_lengths=comparison["sensitivity_block_lengths"],
            replicates=comparison["replicates"],
            seed=comparison["seed"],
            confidence=comparison["confidence"],
        ),
        hours=None
        if hours is None
        else dict(basis=hours["basis"], source=hours["source"], sha256=hours["sha256"]),
        families=families,
        limitations=matrix["limitations"],
        views=views,
        versions={name: version(name) for name in ("numpy", "pyarrow")},
        analysis_source_sha256={
            name: sha256(Path(__file__).parents[1] / name)
            for name in (
                "evaluation/comparison_matrix.py",
                "evaluation/component_attribution.py",
                "evaluation/session_table_contrasts.py",
                "evaluation/paired_comparisons.py",
                "evaluation/long_short.py",
            )
        },
        resources=dict(
            process_lifetime_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * 1024,
            gpu_used=False,
            elapsed_seconds=time.perf_counter() - started,
        ),
    )


def write(matrix_path, sources_path, scope, output):
    """Publica el informe en un directorio nuevo y separado de las fuentes.

    La salida no puede existir ni solaparse con la declaración o los informes leídos, para
    que una evaluación no sobrescriba nunca lo que consume.
    """
    output = Path(output)
    safe_destination(output)
    _require(not output.exists(), "La salida debe ser nueva")
    for source in (Path(matrix_path).parent, Path(sources_path).parent):
        outside_source(source, output)
        outside_source(output, source)
    report = evaluate(matrix_path, sources_path, scope)
    json.dumps(report, allow_nan=False)
    output.mkdir(parents=True)
    atomic_json(output / "matrix.json", report)
    return report


# Brazos que faltan


def missing_arms(matrix, hours=None):
    """Calcula qué brazos faltan para cada contraste que la campaña declarada no produce.

    Los brazos condicionados y derivados tienen su propio plan en otra tarea y se suponen
    disponibles al calcular lo que desbloquea cada candidato. Un candidato desbloquea solo
    un contraste si es el único brazo que le falta. Los demás contrastes necesitan varios
    candidatos y aparecen como lotes con su coste conjunto. El informe sirve para decidir
    qué brazos añadir y no declara ninguno en la campaña.
    """
    arms = matrix["arms"]
    conditional = {
        arm: dict(entry, contrasts=0)
        for arm, entry in matrix["conditional_arms"].items()
        if arms.kind(arm) == "conditional"
    }
    derived = {suffix: dict(entry, contrasts=0) for suffix, entry in matrix["derived_arms"].items()}
    needs, unidentified = {}, []
    totals = dict(contrasts=0, declared=0, planned_elsewhere=0, candidates=0, unidentified=0)
    for family, entry in matrix["families"].items():
        for name, contrast in entry["contrasts"].items():
            totals["contrasts"] += 1
            if contrast["unnamed"]:
                totals["unidentified"] += 1
                unidentified.append(
                    dict(family=family, contrast=name, missing_sets=contrast["unnamed"])
                )
                continue
            kinds = {arm: arms.kind(arm) for arm in contrast["coefficients"]}
            for arm, kind in kinds.items():
                if kind == "conditional":
                    conditional[arm]["contrasts"] += 1
                elif kind == "derived":
                    derived[arm.rpartition("__")[2]]["contrasts"] += 1
            candidates = frozenset(arm for arm, kind in kinds.items() if kind == "candidate")
            if candidates:
                totals["candidates"] += 1
                needs.setdefault(candidates, []).append(f"{family}/{name}")
            elif all(kind == "declared" for kind in kinds.values()):
                totals["declared"] += 1
            else:
                totals["planned_elsewhere"] += 1

    def own(arm):
        # Las horas de un candidato son las propias del brazo declarado al que se parece,
        # porque sus padres ya existen en la campaña y no se vuelven a contar.
        if hours is None:
            return None
        return hours["own"].get(matrix["candidates"][arm]["cost_like"])

    rows = []
    for arm, entry in matrix["candidates"].items():
        alone = needs.get(frozenset([arm]), [])
        within = [c for bundle, items in needs.items() if arm in bundle for c in items]
        cost = own(arm)
        tier = 3
        if alone:
            tier = 1 if entry["code"] == "existing" else 2
        rows.append(
            dict(
                arm=arm,
                **entry,
                components=matrix["resolved_lineages"][entry["lineage"]].order(
                    matrix["resolved_lineages"][entry["lineage"]].arms[arm]
                ),
                gpu_hours=cost,
                contrasts_needing=len(within),
                unlocked_alone=alone,
                unlocked_per_100_gpu_hours=None if not cost else 100 * len(alone) / cost,
                tier=tier,
            )
        )
    rows.sort(
        key=lambda row: (
            row["tier"],
            -(row["unlocked_per_100_gpu_hours"] or 0.0),
            -len(row["unlocked_alone"]),
            row["arm"],
        )
    )
    bundles = []
    for bundle, items in sorted(needs.items(), key=lambda item: (len(item[0]), sorted(item[0]))):
        if len(bundle) < 2:
            continue
        costs = [own(arm) for arm in sorted(bundle)]
        bundles.append(
            dict(
                arms=sorted(bundle),
                contrasts=items,
                gpu_hours=None if None in costs else math.fsum(costs),
            )
        )
    return dict(
        schema_version=1,
        kind=MISSING_KIND,
        matrix=dict(name=matrix["name"], sha256=matrix["sha256"]),
        hours=None
        if hours is None
        else dict(basis=hours["basis"], source=hours["source"], sha256=hours["sha256"]),
        rule=dict(
            planned_elsewhere="Los brazos condicionados y derivados se suponen disponibles",
            unlocked_alone="El candidato es el único brazo que le falta al contraste",
            tiers={
                "1": "Código existente y desbloquea contrastes por sí solo",
                "2": "Requiere un cambio de código y desbloquea contrastes por sí solo",
                "3": "Solo desbloquea contrastes junto con otros candidatos",
            },
            order="Nivel y después contrastes desbloqueados por cada 100 horas GPU",
            gpu_hours="Horas propias del brazo de referencia (cost_like). Sus padres ya existen",
        ),
        totals=totals,
        conditional_arms=conditional,
        derived_arms=derived,
        candidates=rows,
        bundles=bundles,
        unidentified=unidentified,
        limitations=matrix["limitations"],
        scientific_training_started=False,
    )


def summary(matrix):
    """Resume cuántas familias y contrastes aporta cada origen de la declaración.

    Permite revisar el tamaño de la matriz y sus limitaciones antes de tener resultados.
    """
    origins = {}
    for entry in matrix["families"].values():
        kind = entry["origin"].split(":")[0]
        record = origins.setdefault(kind, dict(families=0, contrasts=0))
        record["families"] += 1
        record["contrasts"] += len(entry["contrasts"])
    return dict(
        matrix=dict(name=matrix["name"], sha256=matrix["sha256"]),
        comparison=dict(
            name=matrix["comparison_config"]["name"],
            sha256=matrix["comparison_config"]["sha256"],
        ),
        families=len(matrix["families"]),
        contrasts=sum(len(f["contrasts"]) for f in matrix["families"].values()),
        by_origin=origins,
        limitations=matrix["limitations"],
    )


def main(argv=None):
    """Ofrece las órdenes ``check``, ``missing`` y ``evaluate`` desde la línea de órdenes."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check", help="Validar la matriz y contar sus contrastes")
    check.add_argument("--matrix", type=Path, required=True)
    missing = commands.add_parser("missing", help="Brazos que faltan, con coste y prioridad")
    missing.add_argument("--matrix", type=Path, required=True)
    missing.add_argument("--hours", type=Path)
    missing.add_argument("--scope", default="US+CN")
    missing.add_argument("--output", type=Path)
    run = commands.add_parser("evaluate", help="Evaluar la matriz con informes publicados")
    run.add_argument("--matrix", type=Path, required=True)
    run.add_argument("--sources", type=Path, required=True)
    run.add_argument("--scope", choices=tuple(walk.SCOPES), required=True)
    run.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "evaluate":
        report = write(args.matrix, args.sources, args.scope, args.output)
        print(json.dumps(dict(output=str(args.output), families=len(report["families"]))))
        return 0
    matrix = load_matrix(args.matrix)
    if args.command == "check":
        print(json.dumps(summary(matrix), ensure_ascii=False, indent=2))
        return 0
    hours = None if args.hours is None else tables.load_hours(args.hours, args.scope)
    report = missing_arms(matrix, hours)
    text = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output is None:
        print(text)
    else:
        safe_destination(args.output)
        _require(not args.output.exists(), "La salida debe ser nueva")
        atomic_json(args.output, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
