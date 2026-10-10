"""Declaración preparada de la GRU candidata, MARS-TITAN y CM-v1 en las campañas A y B.

Las campañas A y B no declaran todavía las secciones `episodic_gru`, `mars_titan` y
`cm_v1`, porque antes hay que medir su memoria y su caudal en `cuda:0`. El archivo
`configs/baselines/historical-masked-campaign-extensions.json` las prepara sin activarlas,
junto a los límites que tendrían cada campaña y su etapa de políticas. Vive en la carpeta de
las campañas, así que sus rutas relativas significan lo mismo copiadas a cada archivo.

`extended_campaign` aplica las secciones a una campaña cargada con las reglas de
`campaign_plan`. La medición de caudal lo usa para medir y estimar las familias como si
estuvieran declaradas. `check_extensions` exige que los límites preparados coincidan con
los recuentos exactos del plan, de la etapa de adaptadores y de la etapa de políticas.
Activar la declaración consiste en copiar las secciones y los límites a las campañas y los
límites de políticas a sus etapas. La etapa de adaptadores no cambia, porque solo parte de
las referencias neuronales. Nada aquí lee vistas, reserva la GPU ni ajusta modelos.
"""

import argparse
import json
from pathlib import Path

from mars_titan.data.cohort_files import read_manifest

from .campaign_plan import (
    OPTIONAL,
    VARIANTS,
    _require,
    count_jobs,
    extend_campaign,
    load_campaign,
    pending_families,
)

KIND = "historical_masked_campaign_extensions"
STATUS = "prepared_not_declared"
DEFAULT = (
    Path(__file__).resolve().parents[3]
    / "configs/baselines/historical-masked-campaign-extensions.json"
)
_FIELDS = {"schema_version", "kind", "status", "sections", "variants", "pending"}
_FIELDS |= {"final_test_opened"}
_VARIANT = {"campaign", "limits", "adapter_stage", "rl_stage"}
_STAGE = {"path", "limits"}


def load_extensions(path=DEFAULT):
    """Validar la forma del archivo. Las secciones se validan al aplicarlas a cada campaña."""
    path = Path(path)
    document, digest = read_manifest(path, 256 * 1024)
    _require(
        isinstance(document, dict)
        and set(document) == _FIELDS
        and document["schema_version"] == 1
        and document["kind"] == KIND
        and document["status"] == STATUS
        and document["final_test_opened"] is False
        and isinstance(document["sections"], dict)
        and document["sections"]
        and set(document["sections"]) <= set(OPTIONAL)
        and isinstance(document["pending"], list)
        and all(isinstance(note, str) and note for note in document["pending"]),
        "La declaración preparada no conserva su esquema",
    )
    variants, base = document["variants"], path.resolve().parent
    _require(
        isinstance(variants, dict)
        and variants
        and set(variants) <= set(VARIANTS)
        and all(
            isinstance(entry, dict)
            and set(entry) == _VARIANT
            and all(isinstance(entry[key], str) for key in ("campaign", "adapter_stage"))
            and isinstance(entry["rl_stage"], dict)
            and set(entry["rl_stage"]) == _STAGE
            and isinstance(entry["rl_stage"]["path"], str)
            for entry in variants.values()
        ),
        "Cada variante preparada nombra su campaña, sus límites y sus dos etapas",
    )
    resolved = {}
    for name, entry in variants.items():
        campaign = (base / entry["campaign"]).resolve()
        # Copiadas al archivo de la campaña, las rutas de las secciones no deben cambiar.
        _require(
            campaign.parent == base and campaign.is_file(),
            "La declaración preparada debe estar en la carpeta de las campañas que amplía",
        )
        resolved[name] = dict(
            entry,
            campaign=str(campaign),
            adapter_stage=str((base / entry["adapter_stage"]).resolve()),
            rl_stage=dict(
                entry["rl_stage"], path=str((base / entry["rl_stage"]["path"]).resolve())
            ),
        )
    return dict(document, variants=resolved, path=str(path.resolve()), sha256=digest)


def _variant(extensions, campaign):
    entry = extensions["variants"].get(campaign["variant"])
    _require(
        entry is not None and entry["campaign"] == campaign["path"],
        "La declaración preparada no amplía esta campaña",
    )
    return entry


def extended_campaign(extensions, campaign):
    """La campaña cargada con las secciones y los límites preparados para su variante."""
    entry = _variant(extensions, campaign)
    return extend_campaign(campaign, extensions["sections"], limits=entry["limits"])


def extended_policy_stage(stage, campaign, limits=None):
    """Etapa de políticas que resuelve sus predictores con la campaña ampliada.

    Repite la resolución final de `policy_plan.load_stage`: los niveles y el predictor del
    universo salen de los productores de la campaña, que ahora incluyen las secciones
    añadidas. Sin `limits`, la etapa conserva los suyos.
    """
    from mars_titan.simulation import policy_plan

    _require(
        stage["campaign"]["path"] == campaign["path"],
        "La etapa de políticas no parte de esta campaña",
    )
    levels = policy_plan.resolve_levels(campaign, stage["policies"])
    result = dict(
        stage,
        campaign=campaign,
        levels=levels,
        predictors=levels[policy_plan.ALL_PREDICTORS]["predictors"],
        universe_predictor=levels[policy_plan.ALGORITHMS]["predictors"][0],
    )
    if limits is not None:
        _require(
            isinstance(limits, dict)
            and set(limits) == set(stage["limits"])
            and all(type(v) is int and 0 <= v <= 100_000 for v in limits.values()),
            "Los límites preparados de la etapa de políticas deben ser enteros declarados",
        )
        result["limits"] = limits
    return result


def extended_policies(extensions, stage, campaign):
    """La etapa de políticas preparada para la variante, resuelta con la campaña ampliada."""
    entry = _variant(extensions, campaign)
    _require(
        stage["path"] == entry["rl_stage"]["path"],
        "La etapa de políticas no es la preparada para esta variante",
    )
    return extended_policy_stage(stage, campaign, entry["rl_stage"]["limits"])


def _exact(counts, limits, pairs, label):
    """Los límites preparados son el recuento exacto, como en las configuraciones declaradas."""
    for kind, limit in pairs:
        _require(
            counts[kind] == limits[limit],
            f"{label} prevé {counts[kind]} trabajos ({kind}) y la declaración preparada "
            f"fija {limit}={limits[limit]}",
        )


def check_variant(extensions, variant):
    """Recuentos de la campaña, la etapa de adaptadores y la de políticas, antes y después."""
    from mars_titan.posttraining import campaign_stage as adapters
    from mars_titan.simulation import policy_plan

    entry = extensions["variants"][variant]
    campaign = load_campaign(entry["campaign"])
    extended = extended_campaign(extensions, campaign)
    base, counts = count_jobs(campaign), count_jobs(extended)
    _exact(
        counts,
        extended["limits"],
        (("training_jobs", "max_training_jobs"), ("prediction_jobs", "max_prediction_jobs")),
        "La campaña ampliada",
    )
    stage = adapters.load_stage(entry["adapter_stage"])
    _require(
        stage["campaign"]["path"] == campaign["path"],
        "La etapa de adaptadores no parte de esta campaña",
    )
    before = adapters.count_stage(stage)
    after = adapters.count_stage(dict(stage, campaign=extended))
    _require(
        (before["training_jobs"], before["prediction_jobs"])
        == (after["training_jobs"], after["prediction_jobs"]),
        "La etapa de adaptadores no declara las familias ampliadas y no debe cambiar",
    )
    policies = policy_plan.load_stage(entry["rl_stage"]["path"])
    rl = extended_policies(extensions, policies, extended)
    rl_counts = policy_plan.count_stage(rl)
    _exact(
        rl_counts,
        rl["limits"],
        (("training_jobs", "max_training_jobs"), ("evaluation_jobs", "max_evaluation_jobs")),
        "La etapa de políticas ampliada",
    )

    def totals(value, *kinds):
        return {kind: value[kind] for kind in kinds}

    rl_kinds = ("training_jobs", "carried_jobs", "reference_jobs", "evaluation_jobs")
    return dict(
        variant=variant,
        campaign=dict(
            path=campaign["path"],
            declared=totals(base, "training_jobs", "prediction_jobs"),
            extended=totals(counts, "training_jobs", "prediction_jobs"),
            limits=extended["limits"],
            added_sections=sorted(extensions["sections"]),
            pending_families=pending_families(extended),
            scopes={
                scope: totals(value, "training_jobs", "prediction_jobs")
                for scope, value in counts["scopes"].items()
            },
        ),
        adapter_stage=dict(
            path=stage["path"],
            counts=totals(after, "training_jobs", "prediction_jobs"),
            limits=stage["limits"],
        ),
        rl_stage=dict(
            path=policies["path"],
            predictors=dict(declared=len(policies["predictors"]), extended=len(rl["predictors"])),
            declared=totals(policy_plan.count_stage(policies), *rl_kinds),
            extended=totals(rl_counts, *rl_kinds),
            declared_limits=policies["limits"],
            limits=rl["limits"],
        ),
    )


def check_extensions(path=DEFAULT):
    """Validar la declaración preparada en cada variante sin leer datos ni activar nada."""
    extensions = load_extensions(path)
    return dict(
        status="checked",
        kind=KIND,
        path=extensions["path"],
        sha256=extensions["sha256"],
        sections=sorted(extensions["sections"]),
        variants={name: check_variant(extensions, name) for name in extensions["variants"]},
        pending=extensions["pending"],
        scientific_training_started=False,
        final_test_opened=False,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--extensions", type=Path, default=DEFAULT)
    args = parser.parse_args(argv)
    print(json.dumps(check_extensions(args.extensions), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
