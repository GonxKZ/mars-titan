"""Control en línea de la campaña A: el Transformer compacto que sigue aprendiendo.

Decisión del autor del 9 de octubre de 2026. `transformer_compact_online` parte del mismo
estado elegido que `transformer_compact` en cada ámbito, ventana y semilla. Durante la
calibración y la evaluación recibe las mismas etiquetas maduras que el banco episódico de
`mars_titan_m1`, en el instante en que maduran, y actualiza sus pesos con una regla
declarada antes de ejecutar. Sirve para descartar que la mejora de MARS-TITAN venga solo
de seguir aprendiendo. Se compara emparejado con `transformer_compact` congelado y con
MARS-TITAN, con las mismas filas.

La sección `online_controls` de la campaña declara el brazo, su padre, el brazo que fija
el tope, la regla y el límite de trabajos. Un valor `pending` de la regla bloquea el
lanzamiento hasta fijarlo. El tope `episodic_bank_writes` significa que, en cada tramo, las
etiquetas usadas en pasos no superan las escrituras del banco de `cap_arm` en el mismo
ámbito, ventana y semilla. Es igualdad de información, no de pasos. El ejecutor vive en la
campaña base (`masked_campaign`, clase `online`). Este módulo solo declara y planifica.
"""

ARM = "transformer_compact_online"
PARENT_ARM = "transformer_compact"
CAP_ARM = "mars_titan_m1"
ONLINE = "online"
PENDING = "pending"
PARTITIONS = ["calibration", "evaluation"]
CAP = "episodic_bank_writes"
_SECTION = {"arms", "limits"}
_ARM = {"parent_arm", "cap_arm", "partitions", "rule"}
_RULE = {"optimizer", "learning_rate", "block_rows", "update_every", "max_grad_norm", "update_cap"}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _positive(value, kind):
    return value == PENDING or (type(value) is kind and value > 0)


def declared(section, campaign):
    """Validar la sección de la campaña con sus brazos ya resueltos en cada ámbito."""
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
        and _positive(rule["learning_rate"], float)
        and _positive(rule["max_grad_norm"], float)
        and _positive(rule["block_rows"], int)
        and _positive(rule["update_every"], int),
        f"{ARM} parte de {PARENT_ARM}, se limita con las escrituras del banco de {CAP_ARM} "
        "en calibración y evaluación y declara su regla con SGD",
    )
    for scope in campaign["scopes"]:
        arms = scope_arms(campaign, scope)
        _require(
            PARENT_ARM in arms and CAP_ARM in arms,
            f"En {scope} el control en línea necesita {PARENT_ARM} y {CAP_ARM}",
        )
    return section


def blockers(campaign):
    """Valores de la regla que siguen pendientes y bloquean el lanzamiento."""
    section = campaign.get("online_controls")
    if not section:
        return []
    rule = section["arms"][ARM]["rule"]
    return [
        f"{ARM}.rule.{name} sigue pendiente" for name, value in rule.items() if value == PENDING
    ]


def plan_online(campaign, base_jobs):
    """Un trabajo por ámbito, ventana y semilla del padre, tras elegir padre y tope."""
    from .campaign_chain import parent_jobs
    from .campaign_plan import NEURAL

    section = campaign.get("online_controls")
    if not section:
        return []
    jobs = []
    for scope in campaign["scopes"]:
        for window in campaign["comparison_config"]["resolved_scopes"][scope]["windows"]:
            seeds = sorted(
                {
                    job["seed"]
                    for job in base_jobs
                    if (job["scope"], job["window"], job["arm"]) == (scope, window, PARENT_ARM)
                }
            )
            _require(seeds, f"{scope}/{window} no ajusta {PARENT_ARM}")
            for seed in seeds:
                depends = parent_jobs(base_jobs, scope, window, PARENT_ARM, seed)
                depends += parent_jobs(base_jobs, scope, window, CAP_ARM, seed)
                jobs.append(
                    dict(
                        id=f"{scope}/{window}/{ARM}/{ONLINE}-s{seed}",
                        scope=scope,
                        window=window,
                        arm=ARM,
                        family=NEURAL,
                        model="neural",
                        stage=ONLINE,
                        kind=ONLINE,
                        seed=seed,
                        candidate=None,
                        case=dict(rule=section["arms"][ARM]["rule"]),
                        anchor=window,
                        depends=depends,
                        # Sus predicciones salen de pasos en línea: no se regeneran por inferencia.
                        regenerable=False,
                    )
                )
    limit = section["limits"]["max_online_jobs"]
    _require(
        len(jobs) <= limit,
        f"El control en línea prevé {len(jobs)} trabajos y supera max_online_jobs={limit}",
    )
    return jobs
