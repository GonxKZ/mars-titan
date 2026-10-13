"""Control de varianza entre semillas de los pilotos de memoria (PT7, #459).

Con una señal tan débil, cambiar solo la semilla del control puede mover el MAE por sesión
tanto como una propuesta. La declaración (``seed_variance_control``) fija antes de ver
resultados cuántas semillas del control se ajustan, qué diferencia relativa es la mínima
interpretable y cómo se combina con el bootstrap circular por bloques de cada regla.

- **Banda entre semillas.** El control se ajusta con al menos tres semillas sobre las
  mismas filas, ventanas y presupuesto, y cada semilla repite la búsqueda y la selección
  con validación. En cada réplica del bootstrap se calcula la mayor diferencia relativa
  del MAE por sesión entre dos de esas semillas, con la de menor MAE como base para que
  no dependa del orden de las semillas. La diferencia mínima
  interpretable (MID) es el cuantil declarado de ese máximo. Las réplicas remuestrean los
  mismos días que el contraste de la propuesta y todas las series cubren las mismas
  sesiones, así que la banda recoge la dispersión entre semillas y la de las sesiones.
- **Contraste.** La variante y su base se emparejan por semilla y se promedian sobre las
  semillas de la regla. La diferencia relativa es (variante − base) / base con su
  intervalo de percentiles, el mismo que da ``compare_series`` para la mejora relativa.
- **Regla.** Una mejora solo se afirma si el intervalo queda por debajo de −MID y un
  empeoramiento si queda por encima de MID. La igualdad con una alternativa trivial exige
  el intervalo dentro de [−MID, MID]. Un margen de no inferioridad que no supera la MID no
  se puede interpretar y la regla lo dice en lugar de decidir. Nunca se ensancha un margen
  declarado.

La banda supone que la variante y su base tienen una dispersión entre semillas parecida a
la del control. Con tres semillas es una estimación pobre de la cola de esa dispersión,
así que la regla es una condición mínima y no una garantía. La comparación se hace sobre la
validación de los pilotos, que decide qué propuestas siguen y no es la evaluación final.
Nada de este módulo ajusta modelos ni abre la reserva de 2024.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from mars_titan.data.storage import sha256

from .forecast_panel import SessionSeries
from .paired_comparisons import delta, relative_difference_draws

KIND = "seed_variance_control"
SCHEMA_VERSION = 1
NON_INFERIORITY, SUPERIORITY, EQUIVALENCE = "non_inferiority", "superiority", "equivalence"
KINDS = (NON_INFERIORITY, SUPERIORITY, EQUIVALENCE)
ROLES = ("control", "innovation", "trivial_alternative")
ROWS = ("all", "missing_modality")
MIN_CONTROL_SEEDS = 3
BAND_STATISTIC = "max_symmetric_relative_difference_between_control_seeds"
PRIMARY = "relative_difference_of_session_mae_variant_minus_base"
PROPOSAL_KIND = "innovation_arm_declaration"
# PT7 se descarta en los pilotos siguientes si ninguna regla evaluada cambia de decisión
# al quitar la banda: entonces las semillas añadidas no aportan nada.
DISCARD = "same_decision_with_and_without_band_in_every_evaluated_rule"
# Las semillas añadidas repiten el procedimiento completo de la semilla emparejada: los
# mismos casos de búsqueda, la selección con validación y el presupuesto fijo. Así la banda
# incluye también la variación que introduce la selección.
EXTRA_SEED_PROCEDURE = "same_search_cases_validation_selection_and_budget_as_paired_seed"
ROOT = Path(__file__).resolve().parents[3]


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _seeds(value, label, minimum=1):
    _require(
        isinstance(value, list)
        and len(value) >= minimum
        and all(type(seed) is int and seed >= 0 for seed in value)
        and len(set(value)) == len(value),
        f"{label} deben ser al menos {minimum} enteros distintos no negativos",
    )
    return value


def _fraction(value, label):
    _require(
        isinstance(value, float) and math.isfinite(value) and 0 < value < 1,
        f"{label} debe ser una fracción entre 0 y 1",
    )
    return value


def _rule(name, rule, control_seeds):
    _require(isinstance(rule, dict), f"La regla {name} debe ser un objeto")
    allowed = {"declaration", "kind", "variant", "base", "seeds", "rows"}
    if rule.get("kind") == NON_INFERIORITY:
        allowed |= {"keep_margin", "abandon_margin"}
    _require(set(rule) == allowed, f"La regla {name} no declara exactamente {sorted(allowed)}")
    _require(rule["kind"] in KINDS, f"La regla {name} usa un tipo desconocido")
    _require(
        rule["variant"] == "innovation" and rule["base"] in ("control", "trivial_alternative"),
        f"La regla {name} debe contrastar la innovación con el control o la alternativa trivial",
    )
    seeds = _seeds(rule["seeds"], f"Las semillas de la regla {name}")
    _require(
        set(seeds) < set(control_seeds),
        f"Las semillas de la regla {name} deben ser del control y dejarle semillas propias",
    )
    _require(rule["rows"] in ROWS, f"La regla {name} declara unas filas desconocidas")
    declaration = rule["declaration"]
    _require(
        declaration is None or (isinstance(declaration, str) and declaration.endswith(".json")),
        f"La regla {name} debe enlazar la declaración de su propuesta o dejarla pendiente",
    )
    if rule["kind"] == NON_INFERIORITY:
        keep = _fraction(rule["keep_margin"], f"El margen de la regla {name}")
        abandon = _fraction(rule["abandon_margin"], f"El margen de abandono de la regla {name}")
        _require(keep < abandon, f"La regla {name} debe abandonar por encima de su margen")


def validate_declaration(value):
    """Comprobar la declaración del control de varianza antes de ver resultados."""
    _require(isinstance(value, dict), "La declaración debe ser un objeto")
    _require(
        value.get("schema_version") == SCHEMA_VERSION and value.get("kind") == KIND,
        "La declaración no es un control de varianza entre semillas de la versión 1",
    )
    _require(
        value.get("final_test_opened") is False,
        "El control de varianza no puede abrir la reserva de 2024",
    )
    _require(
        value.get("status") in ("declared_not_executed", "executed")
        and type(value.get("executions")) is int
        and (value["executions"] == 0) == (value["status"] == "declared_not_executed"),
        "El estado y las ejecuciones de la declaración no concuerdan",
    )
    control = value.get("control")
    _require(
        isinstance(control, dict) and set(control) == {"model", "recipe", "variant", "seeds"},
        "El control debe declarar modelo, receta, variante y semillas",
    )
    seeds = _seeds(control["seeds"], "Las semillas del control", MIN_CONTROL_SEEDS)
    metric = value.get("metric")
    _require(isinstance(metric, dict), "Falta la métrica")
    _require(
        metric.get("primary") == PRIMARY
        and metric.get("partition") == "validation"
        and metric.get("uncertainty") == "circular_block_bootstrap"
        and metric.get("market_weighting") in ("session", "market"),
        "La métrica debe ser la diferencia relativa del MAE por sesión en validación",
    )
    for name in ("block_length", "replicates", "bootstrap_seed"):
        _require(
            type(metric.get(name)) is int and metric[name] > 0,
            f"El parámetro {name} del bootstrap debe ser un entero positivo",
        )
    _fraction(metric.get("confidence"), "La confianza")
    band = value.get("band")
    _require(
        isinstance(band, dict) and band.get("statistic") == BAND_STATISTIC,
        "La banda debe ser el máximo de las diferencias entre semillas del control",
    )
    _fraction(band.get("quantile"), "El cuantil de la banda")
    _require(value.get("discard_if") == DISCARD, "Falta el criterio que descarta el control")
    rules = value.get("rules")
    _require(isinstance(rules, dict) and rules, "Faltan las reglas")
    for name, rule in rules.items():
        _rule(name, rule, seeds)
    pilot = value.get("pilot")
    _require(isinstance(pilot, dict), "Falta el piloto")
    extra = _seeds(pilot.get("extra_control_seeds"), "Las semillas añadidas del control")
    paired = {seed for rule in rules.values() for seed in rule["seeds"]}
    _require(
        set(extra) == set(seeds) - paired,
        "Las semillas añadidas deben ser las del control que no empareja ninguna regla",
    )
    _require(
        isinstance(pilot.get("gpu_hours_reserved"), (int, float))
        and pilot["gpu_hours_reserved"] > 0,
        "El piloto debe reservar horas de GPU para las semillas añadidas",
    )
    _require(
        pilot.get("extra_seed_procedure") == EXTRA_SEED_PROCEDURE,
        "Las semillas añadidas deben repetir la búsqueda y la selección de la emparejada",
    )
    _require(
        isinstance(pilot.get("shared_by"), list) and set(pilot["shared_by"]) <= set(rules),
        "El piloto solo puede compartirse entre reglas declaradas",
    )
    return value


def load_declaration(path):
    path = Path(path)
    value = validate_declaration(json.loads(path.read_text()))
    return value, sha256(path)


def proposal_models(declaration, name, root=ROOT):
    """Modelos de cada papel de una regla, leídos de la declaración de su propuesta.

    El control de la propuesta debe ser el del control de varianza, con la misma receta y
    variante, para que la banda mida la dispersión del mismo modelo.
    """
    rule = declaration["rules"][name]
    _require(rule["declaration"] is not None, f"La propuesta de {name} aún no está declarada")
    proposal = json.loads((Path(root) / rule["declaration"]).read_text())
    _require(
        proposal.get("kind") == PROPOSAL_KIND and proposal.get("final_test_opened") is False,
        f"La declaración de {name} no es una propuesta cerrada antes de ver la reserva",
    )
    roles = {}
    for model, arm in proposal["arms"].items():
        _require(arm["role"] not in roles, f"La propuesta de {name} repite el papel {arm['role']}")
        roles[arm["role"]] = model
    control = declaration["control"]
    _require(
        roles.get("control") == control["model"]
        and proposal.get("base_recipe") == control["recipe"]
        and proposal.get("variant") == control["variant"],
        f"El control de {name} no es el del control de varianza",
    )
    _require(
        rule["base"] in roles and "innovation" in roles,
        f"La propuesta de {name} no declara los papeles de su regla",
    )
    return dict(variant=roles["innovation"], base=roles[rule["base"]], control=control["model"])


def decide(kind, interval, band, *, keep_margin=None, abandon_margin=None):
    """Decisión de una regla con el intervalo de la diferencia relativa y la banda.

    Devuelve la decisión y su motivo. La banda nunca ensancha un margen declarado: si no
    queda por debajo de él, el margen no se puede interpretar con esta dispersión.
    """
    lower, upper = interval
    _require(lower <= upper and band >= 0, "El intervalo o la banda no son válidos")
    if kind == NON_INFERIORITY:
        if upper > abandon_margin:
            if band < abandon_margin:
                return "abandon", "El límite superior supera el margen de abandono"
            return "not_interpretable", "La banda entre semillas alcanza el margen de abandono"
        if upper <= keep_margin:
            if band < keep_margin:
                return "keep", "El límite superior no supera el margen"
            return "not_interpretable", "La banda entre semillas alcanza el margen"
        return "undecided", "El límite superior queda entre el margen y el de abandono"
    if kind == SUPERIORITY:
        if upper < -band:
            return "keep", "El intervalo queda por debajo de la banda entre semillas"
        return "abandon", "La mejora no supera la banda entre semillas"
    if kind == EQUIVALENCE:
        if -band <= lower and upper <= band:
            return "equivalent", "El intervalo queda dentro de la banda entre semillas"
        return "not_equivalent", "El intervalo sale de la banda entre semillas"
    raise ValueError(f"Tipo de regla desconocido: {kind}")


def _name(model, seed):
    return f"{model}@{seed}"


def _symmetric(relative):
    """Diferencia relativa entre dos semillas con la de menor MAE como base.

    Con r = b / a − 1, es max(b / a, a / b) − 1. No depende de qué semilla se tome como base
    del contraste y nunca es menor que |r|.
    """
    ratio = 1.0 + np.asarray(relative, dtype=np.float64)
    return np.maximum(ratio, 1.0 / ratio) - 1.0


def evaluate_rule(declaration, name, series, models):
    """Aplicar una regla declarada a las series por sesión de cada modelo y semilla.

    ``series`` asigna a cada modelo un diccionario de semilla a ``SessionSeries`` del MAE
    por sesión, con las filas que declara la regla. ``models`` da el modelo de la variante,
    el de la base y el del control, por ejemplo con ``proposal_models``.
    """
    rule = declaration["rules"][name]
    metric = declaration["metric"]
    control_seeds = declaration["control"]["seeds"]
    seeds = rule["seeds"]
    variant, base, control = models["variant"], models["base"], models["control"]
    _require(variant not in (base, control), "La variante no puede ser la base ni el control")
    for model, wanted in ((control, control_seeds), (variant, seeds), (base, seeds)):
        _require(
            isinstance(series.get(model), dict) and set(wanted) <= set(series[model]),
            f"Faltan semillas de {model}",
        )
        _require(
            all(isinstance(series[model][seed], SessionSeries) for seed in wanted),
            f"Las series de {model} deben ser SessionSeries",
        )
    first = series[control][control_seeds[0]]
    _require(first.metric == "mae" and first.loss, "La regla se aplica al MAE por sesión")
    # La banda y el contraste deben ver las mismas sesiones: todas las series comparten
    # sesiones y máscara, así que ninguna semilla puede dejar fuera días o filas.
    used = [series[control][seed] for seed in control_seeds]
    used += [series[model][seed] for model in (variant, base) for seed in seeds]
    _require(
        all(
            np.array_equal(item.session_market, first.session_market)
            and np.array_equal(item.session_period, first.session_period)
            and np.array_equal(item.defined, first.defined)
            for item in used
        ),
        "Todas las semillas y modelos deben cubrir las mismas sesiones",
    )
    models_series = {
        "variant": SessionSeries.average([series[variant][seed] for seed in seeds]),
        "base": SessionSeries.average([series[base][seed] for seed in seeds]),
    }
    contrasts = {"proposal": delta("base", "variant")}
    for position, first_seed in enumerate(control_seeds):
        models_series[_name(control, first_seed)] = series[control][first_seed]
        for second_seed in control_seeds[position + 1 :]:
            contrasts[f"{_name(control, second_seed)}-{_name(control, first_seed)}"] = delta(
                _name(control, first_seed), _name(control, second_seed)
            )
    estimates, draws = relative_difference_draws(
        models_series,
        contrasts,
        block_length=metric["block_length"],
        replicates=metric["replicates"],
        seed=metric["bootstrap_seed"],
        market_weighting=metric["market_weighting"],
    )
    tail = (1 - metric["confidence"]) / 2
    lower, upper = np.quantile(draws[:, 0], [tail, 1 - tail], method="linear")
    spread = _symmetric(draws[:, 1:]).max(axis=1)
    band = float(np.quantile(spread, declaration["band"]["quantile"], method="linear"))
    margins = {key: rule[key] for key in ("keep_margin", "abandon_margin") if key in rule}
    interval = (float(lower), float(upper))
    decision, reason = decide(rule["kind"], interval, band, **margins)
    # La misma regla sin banda, para el criterio que descarta el control.
    without_band, _ = decide(rule["kind"], interval, 0.0, **margins)
    pairs = [
        dict(name=pair, relative_difference=float(value))
        for pair, value in zip(list(contrasts)[1:], estimates[1:], strict=True)
    ]
    return dict(
        schema_version=SCHEMA_VERSION,
        kind="seed_variance_control_rule",
        rule=name,
        rule_kind=rule["kind"],
        rows=rule["rows"],
        variant=variant,
        base=base,
        control=control,
        paired_seeds=list(seeds),
        control_seeds=list(control_seeds),
        sessions=int(np.sum(models_series["variant"].defined & models_series["base"].defined)),
        relative_difference=float(estimates[0]),
        interval=list(interval),
        control_pairs=pairs,
        observed_seed_spread=float(np.max(_symmetric(estimates[1:]))),
        minimum_interpretable_difference=band,
        margins=margins,
        decision=decision,
        reason=reason,
        decision_without_band=without_band,
        bootstrap=dict(
            method="circular_block_bootstrap",
            block_length=metric["block_length"],
            replicates=metric["replicates"],
            seed=metric["bootstrap_seed"],
            confidence=metric["confidence"],
            band_quantile=declaration["band"]["quantile"],
            market_weighting=metric["market_weighting"],
        ),
        final_test_opened=False,
    )


def discard_verdict(declaration, reports):
    """Criterio que descarta el control tras un piloto, con los informes de sus reglas.

    Si ninguna regla evaluada cambia de decisión al quitar la banda, las semillas añadidas
    no han aportado nada y el control se descarta en los pilotos siguientes.
    """
    names = [report["rule"] for report in reports]
    _require(names, "El criterio necesita al menos una regla evaluada")
    _require(
        len(set(names)) == len(names) and set(names) <= set(declaration["rules"]),
        "Cada regla evaluada debe estar declarada y aparecer una vez",
    )
    changed = sorted(
        report["rule"]
        for report in reports
        if report["decision"] != report["decision_without_band"]
    )
    return dict(
        criterion=DISCARD, evaluated=sorted(names), changed_by_band=changed, discard=not changed
    )


def check_declaration(path, root=ROOT):
    """Resumen de la declaración y de las propuestas enlazadas, sin leer predicciones."""
    declaration, digest = load_declaration(path)
    rules = {}
    for name, rule in declaration["rules"].items():
        models = None if rule["declaration"] is None else proposal_models(declaration, name, root)
        rules[name] = dict(kind=rule["kind"], rows=rule["rows"], seeds=rule["seeds"], models=models)
    return dict(
        declaration=str(path),
        sha256=digest,
        control=declaration["control"],
        extra_control_seeds=declaration["pilot"]["extra_control_seeds"],
        gpu_hours_reserved=declaration["pilot"]["gpu_hours_reserved"],
        rules=rules,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check", help="Comprobar la declaración sin leer predicciones")
    check.add_argument("--declaration", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(check_declaration(args.declaration), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
