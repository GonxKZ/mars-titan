"""Control en línea de la campaña A: el Transformer compacto que sigue aprendiendo.

Decisión del autor del 9 de octubre de 2026. `transformer_compact_online` parte del mismo
estado elegido que `transformer_compact` en cada ámbito, ventana y semilla. Durante la
calibración y la evaluación recibe las mismas etiquetas maduras que el banco episódico de
`mars_titan_m1`, en el instante en que maduran, y actualiza sus pesos con una regla
declarada antes de ejecutar. Sirve para descartar que la mejora de MARS-TITAN venga solo
de seguir aprendiendo. Se compara emparejado con `transformer_compact` congelado y con
MARS-TITAN, con las mismas filas.

La sección `online_controls` de la campaña declara el brazo, su padre, el brazo que fija
el tope, la regla, la rejilla de tasas y el límite de trabajos. La regla iguala al banco en
información y cadencia (decisión del 10 de octubre):

- `update_cap = episodic_bank_writes`: en cada tramo, las etiquetas usadas en pasos no
  superan las escrituras del banco de `cap_arm` en el mismo ámbito, ventana y semilla.
- `update_every = 1`: una actualización en cada instante de maduración, con todas las
  etiquetas del instante, igual que el banco las escribe todas en ese instante.

La tasa se elige como en las demás familias: un caso de búsqueda por tasa con la semilla de
búsqueda del padre, la elección por el MAE por sesión de la validación y el caso elegido
repetido con las demás semillas. La rejilla incluye la tasa cero, que reproduce el
Transformer congelado, para que el control nunca quede por debajo de él en validación.
Solo hay trabajos en los ámbitos cuya comparación evalúa el brazo, que en A v2 es el
conjunto. El ejecutor lo registra el motor como trabajo `online`. Este módulo solo declara
y planifica los trabajos.
"""

import math

ARM = "transformer_compact_online"
PARENT_ARM = "transformer_compact"
CAP_ARM = "mars_titan_m1"
# Los de `online_reference.PARTITIONS`, sin importar PyTorch. Una prueba lo comprueba.
PARTITIONS = ["validation", "calibration", "evaluation"]
CAP = "episodic_bank_writes"
# Una actualización por instante de maduración, la cadencia de las escrituras del banco.
BANK_CADENCE = 1
_SECTION = {"arms", "limits"}
_ARM = {"parent_arm", "cap_arm", "partitions", "rule", "search_cases"}
_RULE = {"optimizer", "accumulation_rows", "update_every", "max_grad_norm", "update_cap"}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _cases(cases):
    """Comprueba la rejilla de tasas: de 2 a 8 tasas distintas en [0, 1], con el cero."""
    rates = [
        case.get("learning_rate") if isinstance(case, dict) else None for case in cases.values()
    ]
    return (
        2 <= len(cases) <= 8
        and all(isinstance(name, str) and name for name in cases)
        and all(
            isinstance(case, dict) and set(case) == {"learning_rate"} for case in cases.values()
        )
        and all(type(rate) is float and 0.0 <= rate <= 1.0 for rate in rates)
        and len(set(rates)) == len(rates)
        and 0.0 in rates
    )


def declared(section, campaign):
    """Valida la sección `online_controls` frente a los brazos ya resueltos de cada ámbito.

    La comparación debe evaluar el brazo en algún ámbito, y en cada uno de ellos deben
    ajustarse el padre y el brazo del tope, porque sin ellos el control no tendría estado de
    partida ni tope.
    """
    from .campaign_plan import scope_arms

    _require(
        isinstance(section, dict)
        and set(section) == _SECTION
        and isinstance(section["arms"], dict)
        and set(section["arms"]) == {ARM}
        and isinstance(section["limits"], dict)
        and set(section["limits"]) == {"max_online_jobs"}
        and type(section["limits"]["max_online_jobs"]) is int
        and 0 <= section["limits"]["max_online_jobs"] <= 100_000,
        f"online_controls declara solo {ARM} y su límite max_online_jobs",
    )
    arm = section["arms"][ARM]
    rule = arm.get("rule") if isinstance(arm, dict) else None
    _require(
        isinstance(arm, dict)
        and set(arm) == _ARM
        and arm["parent_arm"] == PARENT_ARM
        and arm["cap_arm"] == CAP_ARM
        and arm["partitions"] == PARTITIONS
        and isinstance(rule, dict)
        and set(rule) == _RULE
        and rule["optimizer"] == "sgd"
        and rule["update_cap"] == CAP
        and rule["update_every"] == BANK_CADENCE
        and type(rule["update_every"]) is int
        and type(rule["accumulation_rows"]) is int
        and 1 <= rule["accumulation_rows"] <= 4096
        and type(rule["max_grad_norm"]) is float
        and math.isfinite(rule["max_grad_norm"])
        and rule["max_grad_norm"] > 0
        and isinstance(arm["search_cases"], dict)
        and _cases(arm["search_cases"]),
        f"{ARM} parte de {PARENT_ARM}, recibe las etiquetas del banco de {CAP_ARM} con su "
        "tope y su cadencia, predice validación, calibración y evaluación, declara su regla "
        "con SGD y elige la tasa en una rejilla con el cero",
    )
    _require(scopes(campaign), f"La comparación no evalúa {ARM} en ningún ámbito")
    for scope in scopes(campaign):
        arms = scope_arms(campaign, scope)
        _require(
            PARENT_ARM in arms and CAP_ARM in arms,
            f"En {scope} el control en línea necesita {PARENT_ARM} y {CAP_ARM}",
        )
    return section


def scopes(campaign):
    """Ámbitos cuya comparación evalúa el control en línea con predicciones propias."""
    from .campaign_plan import scope_arms

    return [scope for scope in campaign["scopes"] if ARM in scope_arms(campaign, scope)]


def search_cases(section):
    """Casos de búsqueda en el orden declarado, cada uno con la regla completa."""
    arm = section["arms"][ARM]
    return [
        (name, dict(rule=dict(arm["rule"], learning_rate=case["learning_rate"])))
        for name, case in arm["search_cases"].items()
    ]


def plan_online(campaign, base_jobs):
    """Planifica la búsqueda de la tasa y el caso elegido en cada ámbito evaluado y ventana.

    Hay un trabajo de búsqueda por tasa con la semilla de búsqueda del padre y un finalista
    por cada otra semilla del padre. Cada trabajo depende de los que eligen el estado de
    `transformer_compact` y de `mars_titan_m1` en su ventana y semilla, porque parte del
    primero y su tope son las escrituras del banco del segundo. Un finalista depende además
    de todas las búsquedas de su ventana, de las que sale la tasa elegida. Los ámbitos que
    no evalúan el brazo no tienen trabajos, porque sus predicciones no entrarían en ninguna
    comparación.
    """
    from .campaign_chain import parent_jobs
    from .campaign_plan import NEURAL, ONLINE

    section = campaign.get("online_controls")
    if not section:
        return []
    cases = search_cases(section)
    jobs = []
    for scope in scopes(campaign):
        for window in campaign["comparison_config"]["resolved_scopes"][scope]["windows"]:
            parent = [
                job
                for job in base_jobs
                if (job["scope"], job["window"], job["arm"]) == (scope, window, PARENT_ARM)
            ]
            searched = {job["seed"] for job in parent if job["stage"] == "search"}
            _require(
                len(searched) == 1,
                f"{scope}/{window} no busca {PARENT_ARM} con una sola semilla",
            )
            (search_seed,) = searched
            prefix = f"{scope}/{window}/{ARM}"
            searches = [f"{prefix}/search-{name}" for name, _ in cases]

            def states(seed, scope=scope, window=window):
                return [
                    *parent_jobs(base_jobs, scope, window, PARENT_ARM, seed),
                    *parent_jobs(base_jobs, scope, window, CAP_ARM, seed),
                ]

            common = dict(
                scope=scope,
                window=window,
                arm=ARM,
                family=NEURAL,
                model="neural",
                kind=ONLINE,
                anchor=window,
                # Sus predicciones salen de pasos en línea, así que la retención no
                # puede regenerarlas repitiendo la inferencia del estado elegido.
                regenerable=False,
            )
            for name, case in cases:
                jobs.append(
                    dict(
                        id=f"{prefix}/search-{name}",
                        stage="search",
                        seed=search_seed,
                        candidate=name,
                        case=case,
                        depends=states(search_seed),
                        **common,
                    )
                )
            for seed in sorted({job["seed"] for job in parent} - {search_seed}):
                jobs.append(
                    dict(
                        id=f"{prefix}/finalist-s{seed}",
                        stage="finalist",
                        seed=seed,
                        candidate=None,
                        case=None,
                        depends=[*searches, *states(seed)],
                        **common,
                    )
                )
    limit = section["limits"]["max_online_jobs"]
    _require(
        len(jobs) <= limit,
        f"El control en línea prevé {len(jobs)} trabajos y supera max_online_jobs={limit}",
    )
    return jobs
