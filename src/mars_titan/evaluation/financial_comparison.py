"""Comparar auditorías financieras sintéticas con referencias sobre los mismos mundos."""

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

from mars_titan.data.cohort_files import read_manifest, safe_destination
from mars_titan.data.storage import atomic_json, outside_source, sha256
from mars_titan.simulation.campaign_receipts import read_adaptive_receipt

POLICIES = ("cash", "hold_initial", "rebalance_25", "rebalance_50", "rebalance_75", "rebalance_100")
COSTS = (0, 10, 25)
FIELDS = ("net_return", "log_growth", "max_drawdown", "costs", "turnover", "ruined")
MAX_WORLDS = 512
MAX_JSON = 32 * 1024**2


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _json(path, expected=None):
    safe_destination(path)
    _require(path.suffix == ".json", "Solo se admiten recibos JSON")
    value, signature = read_manifest(path, MAX_JSON)
    _require(isinstance(value, dict), "El recibo debe ser un objeto JSON")
    _require(expected is None or expected == signature, "La huella del recibo no coincide")
    return value, signature


def _inside(root, value):
    _require(isinstance(value, str), "La ruta del recibo no es texto")
    relative = Path(value)
    _require(
        not relative.is_absolute() and ".." not in relative.parts, "La ruta sale de la campaña"
    )
    path = root / relative
    safe_destination(path)
    _require(path.resolve().is_relative_to(root.resolve()), "La ruta sale de la campaña")
    return path


def _closed(report):
    _require(
        report.get("status") == "completed"
        and report.get("final_test_opened") is False
        and report.get("domain") == "synthetic",
        "Falta un resultado sintético completo con test cerrado",
    )


def _population(identity, freeze, controls):
    expected = [row for row in identity["source_manifests"] if row["split"] == "audit"]
    _require(freeze["audit_sources"] == expected, "La congelación cambió la población de auditoría")
    frozen = {row["manifest_sha256"]: row for row in expected}
    sources = controls["sources"]
    _require(
        1 <= len(sources) == len(frozen) == len(expected) <= MAX_WORLDS,
        "La población contiene mundos ausentes, duplicados o excesivos",
    )
    worlds, seeds = {}, set()
    for row in sources:
        signature = row["manifest_sha256"]
        _require(
            signature in frozen and signature not in worlds and row["seed"] not in seeds,
            "La población contiene mundos o semillas duplicados",
        )
        _require(
            all(
                row[key] == frozen[signature][key]
                for key in ("name", "manifest_sha256", "context_sha256", "split")
            )
            and row["split"] == "audit"
            and row["evaluator_only"] is True
            and row["warmup_sessions"] == row["decision_start"] == controls["decision_start"] == 64,
            "El mundo no conserva su cohorte y calentamiento",
        )
        _require(
            isinstance(row["family"], str)
            and re.fullmatch(r"[a-z_]{1,64}", row["family"])
            and row["name"] == f"{row['family']}-audit-{row['seed']}",
            "La familia no corresponde al mundo congelado",
        )
        worlds[signature] = row
        seeds.add(row["seed"])
    order = identity["source_order"]
    _require(
        set(order["audit"]) == set(worlds) and len(order["audit"]) == len(worlds),
        "El orden de auditoría no corresponde a su población",
    )
    for split in ("train", "validation"):
        _require(
            not set(order[split]).intersection(worlds), "La auditoría reutiliza mundos de selección"
        )
    _require(
        len({row["family"] for row in worlds.values()}) <= 8, "Hay más familias de las previstas"
    )
    return worlds


def _episodes(rows, worlds, environment, *, policy=None):
    _require(
        isinstance(rows, list) and len(rows) == len(worlds) * len(COSTS),
        "Faltan episodios de la población prevista",
    )
    admitted = {}
    for row in rows:
        world, cost = row["manifest_sha256"], row["cost_bps"]
        _require(
            world in worlds and type(cost) in (int, float) and cost in COSTS,
            "El episodio corresponde a otro mundo o coste",
        )
        key = world, cost
        _require(key not in admitted, "Hay episodios duplicados")
        values = {
            field: row[field] for field in ("net_return", "max_drawdown", "costs", "turnover")
        }
        _require(
            all(type(value) in (int, float) and math.isfinite(value) for value in values.values())
            and values["net_return"] >= -1
            and 0 <= values["max_drawdown"] <= 1
            and values["costs"] >= 0
            and values["turnover"] >= 0,
            "Las métricas contienen valores inválidos o no finitos",
        )
        ruined = values["net_return"] == -1
        _require(
            row["completed"] is True
            and type(row["steps"]) is int
            and 1 <= row["steps"] <= 255
            and (row["steps"] == 255 or ruined)
            and row["invalid_reason"] == ("ruined" if ruined else ""),
            "El episodio no conserva el horizonte completo o una ruina confirmada",
        )
        expected_cost = values["turnover"] * environment["capital"] * cost / 10000
        _require(
            math.isclose(values["costs"], expected_cost, rel_tol=1e-10, abs_tol=1e-10),
            "Los costes no concilian con la rotación y el capital",
        )
        if policy is not None:
            _require(
                row["family"] == worlds[world]["family"]
                and row["generator_seed"] == worlds[world]["seed"],
                "El control cambió su cohorte",
            )
            if policy == "cash":
                _require(
                    all(value == 0 for value in values.values()),
                    "El efectivo no es un control cero",
                )
        values.update(
            log_growth=environment["ruin_penalty"] if ruined else math.log1p(values["net_return"]),
            ruined=int(ruined),
        )
        admitted[key] = values
    _require(
        set(admitted) == {(world, cost) for world in worlds for cost in COSTS},
        "La población no conserva todos los pares de mundo y coste",
    )
    return admitted


def _admit(campaign, controls_path, controls_hash):
    _require(
        isinstance(controls_hash, str) and re.fullmatch(r"[a-f0-9]{64}", controls_hash),
        "Se necesita la huella revisada de los controles",
    )
    controls, _ = _json(controls_path, controls_hash)
    _closed(controls)
    _require(
        controls.get("kind") == "frozen_audit_financial_controls"
        and controls.get("split") == "audit"
        and controls.get("learning") is False,
        "Los controles no pertenecen a la auditoría sin aprendizaje",
    )
    identity, identity_hash = _json(campaign / "identity.json", controls["identity_file_sha256"])
    journal, campaign_hash = _json(campaign / "campaign.json", controls["campaign_file_sha256"])
    freeze, freeze_hash = _json(campaign / "freeze.json", controls["freeze_file_sha256"])
    receipt = read_adaptive_receipt(campaign / "registry.json")
    _require(receipt["status"] == "completed", "La campaña todavía no está cerrada")
    state = journal["payload"]
    _require(
        journal["sha256"] == _digest(state)
        and controls["campaign_identity_sha256"] == freeze["identity_sha256"] == _digest(identity)
        and state["freeze_sha256"] == freeze_hash
        and freeze["final_test_opened"] is False
        and freeze["selection_uses"] == "validation_only"
        and controls["catalog_file_sha256"] == identity["index_sha256"],
        "La identidad o la selección congelada no coincide",
    )
    _require(
        controls["cost_bps"] == identity["settings"]["audit_cost_bps"] == list(COSTS),
        "La evaluación cambió sus niveles de coste",
    )
    environment = identity["base_configuration"]["environment"]
    _require(
        type(environment["capital"]) in (int, float)
        and math.isfinite(environment["capital"])
        and environment["capital"] > 0
        and math.isfinite(environment["ruin_penalty"]),
        "El entorno financiero contiene parámetros no finitos o inválidos",
    )
    _require(
        all(
            controls[field] == environment[field]
            for field in ("capital", "participation", "score_scale", "ruin_penalty")
        ),
        "Los controles cambiaron los parámetros financieros",
    )
    worlds = _population(identity, freeze, controls)
    selected = {(row["variant"], row["seed"]): row for row in freeze["selections"]}
    training_cases = [row for row in state["cases"] if row["stage"] in {"main", "auxiliary"}]
    _require(
        len(selected) == len(freeze["selections"]) == len(training_cases)
        and set(selected) == {(row["variant"], row["seed"]) for row in training_cases},
        "Las selecciones están duplicadas o no cubren los entrenamientos",
    )
    learned, training = {}, []
    for case in state["cases"]:
        audit = case["stage"] == "audit"
        path = _inside(campaign, case["output"]) / ("audit.json" if audit else "run.json")
        report, _ = _json(path, case["receipt_sha256"])
        _closed(report)
        config, _ = _json(_inside(campaign, case["config"]), case["config_sha256"])
        _require(
            config["final_test_opened"] is False and config["environment"] == environment,
            "La configuración cambió el entorno financiero o abrió el test",
        )
        key = case["variant"], case["seed"]
        if not audit:
            _require(
                report["agent_variant"] == key[0]
                and report["seed"] == key[1]
                and report["transitions"] == case["confirmed_transitions"]
                and report.get("best") == case.get("best"),
                "El recibo de aprendizaje cambió",
            )
            training.append(
                dict(
                    id=case["id"],
                    stage=case["stage"],
                    variant=key[0],
                    seed=key[1],
                    **{
                        field: report[field]
                        for field in (
                            "transitions",
                            "optimizer_steps",
                            "auxiliary_steps",
                            "auxiliary_samples",
                            "stopping_reason",
                            "selection",
                        )
                        if field in report
                    },
                    selected_transitions=(report.get("best") or {}).get("transitions"),
                )
            )
            if case["stage"] in {"main", "auxiliary"}:
                chosen = selected[key]
                _require(
                    chosen["output"] == case["output"]
                    and chosen["case_id"] == case["id"]
                    and chosen["config_sha256"] == case["config_sha256"]
                    and chosen["receipt_sha256"] == case["receipt_sha256"]
                    and chosen["training_identity_sha256"] == report["identity_sha256"],
                    "La selección no corresponde al entrenamiento confirmado",
                )
            continue
        chosen, audit_identity = selected[key], report["identity"]
        _require(
            report["kind"] == "native_ppo_audit"
            and report["partition"] == "audit"
            and report["model"] == key[0]
            and report["seed"] == key[1]
            and audit_identity["final_test_opened"] is False
            and audit_identity["configuration"] == config
            and audit_identity["cost_bps"] == list(COSTS)
            and audit_identity["policy_sha256"] == chosen["policy_sha256"]
            and audit_identity["selected_identity_sha256"] == chosen["training_identity_sha256"]
            and case["training_output"] == chosen["output"],
            "La auditoría cambió la política seleccionada",
        )
        sources = audit_identity["sources"]
        _require(
            [row["manifest_sha256"] for row in sources] == identity["source_order"]["audit"]
            and all(
                row["context_sha256"] == worlds[row["manifest_sha256"]]["context_sha256"]
                and row["generator_seed"] == worlds[row["manifest_sha256"]]["seed"]
                for row in sources
            ),
            "La auditoría y los controles no comparten la misma cohorte",
        )
        metrics = _episodes(report["metrics"], worlds, environment)
        _require(
            report["confirmed_episodes"] == report["total_episodes"] == len(metrics)
            and report["confirmed_decisions"] == sum(row["steps"] for row in report["metrics"])
            and key not in learned,
            "La auditoría no concilia sus recuentos",
        )
        learned[key] = metrics
    _require(set(learned) == set(selected), "Falta una auditoría de la selección congelada")
    metrics = controls["metrics"]
    expected_count = len(worlds) * len(COSTS) * len(POLICIES)
    _require(
        len(metrics)
        == controls["completed_episodes"]
        == controls["expected_episodes"]
        == expected_count
        and controls["invalid_episodes"] == 0
        and {row["policy"] for row in metrics} == set(POLICIES),
        "Los controles no concilian su población completa",
    )
    control_rows = {
        policy: _episodes(
            [r for r in metrics if r["policy"] == policy], worlds, environment, policy=policy
        )
        for policy in POLICIES
    }
    return (
        learned,
        control_rows,
        worlds,
        training,
        dict(
            campaign_sha256=campaign_hash,
            identity_sha256=identity_hash,
            freeze_sha256=freeze_hash,
            controls_sha256=controls_hash,
            catalog_sha256=identity["index_sha256"],
            campaign_snapshot_sha256=journal["sha256"],
        ),
    )


def _mean(rows, field):
    return math.fsum(row[field] for row in rows) / len(rows)


def _aggregates(learned, controls, worlds):
    rows = []
    for kind, cases in (
        ("learned", learned),
        ("control", {(policy, None): values for policy, values in controls.items()}),
    ):
        for (variant, seed), values in sorted(cases.items()):
            for cost in COSTS:
                for family in ("all", *sorted({row["family"] for row in worlds.values()})):
                    selected = [
                        row
                        for (world, rate), row in values.items()
                        if rate == cost and (family == "all" or worlds[world]["family"] == family)
                    ]
                    rows.append(
                        dict(
                            kind=kind,
                            variant=variant,
                            seed=seed,
                            family=family,
                            cost_bps=cost,
                            worlds=len(selected),
                            **{f"mean_{field}": _mean(selected, field) for field in FIELDS},
                        )
                    )
    return rows


def _paired_worlds(learned, controls, worlds):
    variants = sorted({variant for variant, _ in learned})
    for variant in variants:
        seeds = sorted(seed for name, seed in learned if name == variant)
        for cost in COSTS:
            for world, source in sorted(worlds.items()):
                sampled = [learned[variant, seed][world, cost] for seed in seeds]
                averaged = {field: _mean(sampled, field) for field in FIELDS}
                for policy in POLICIES:
                    control = controls[policy][world, cost]
                    yield dict(
                        variant=variant,
                        control=policy,
                        family=source["family"],
                        cost_bps=cost,
                        manifest_sha256=world,
                        policy_seeds=len(seeds),
                        **{f"delta_{field}": averaged[field] - control[field] for field in FIELDS},
                    )


def _csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        iterator = iter(rows)
        first = next(iterator)
        writer = csv.DictWriter(stream, fieldnames=list(first))
        writer.writeheader()
        writer.writerow(first)
        writer.writerows(iterator)


def compare_financial_campaign(campaign, controls, output, *, controls_sha256):
    """Guardar agregados y diferencias por mundo, con media previa de las semillas.

    Los pares no son trayectorias. Cada coste se conserva separado y los
    intervalos se dejan al análisis posterior, sin multiplicar n por semillas.
    """
    campaign, controls, output = map(Path, (campaign, controls, output))
    for path in (campaign, controls, output):
        safe_destination(path)
    for source in (campaign, controls.parent):
        outside_source(source, output)
        outside_source(output, source)
    if output.exists():
        raise FileExistsError("La comparación necesita una salida nueva")
    try:
        learned, references, worlds, training, provenance = _admit(
            campaign, controls, controls_sha256
        )
        aggregates = _aggregates(learned, references, worlds)
        groups = defaultdict(list)
        pair_count = 0
        for row in _paired_worlds(learned, references, worlds):
            for family in (row["family"], "all"):
                groups[row["variant"], row["control"], family, row["cost_bps"]].append(row)
            pair_count += 1
        paired = [
            dict(
                variant=key[0],
                control=key[1],
                family=key[2],
                cost_bps=key[3],
                worlds=len(rows),
                policy_seeds=rows[0]["policy_seeds"],
                **{f"delta_{field}": _mean(rows, f"delta_{field}") for field in FIELDS},
            )
            for key, rows in sorted(groups.items())
        ]
        report = dict(
            schema_version=1,
            status="completed",
            domain="synthetic",
            analysis_domain="technical",
            split="audit",
            final_test_opened=False,
            provenance=provenance,
            population=dict(
                worlds=len(worlds),
                policy_seeds=sorted({seed for _, seed in learned}),
                worlds_by_family=dict(
                    sorted(Counter(row["family"] for row in worlds.values()).items())
                ),
                cost_bps=list(COSTS),
                decision_start=64,
            ),
            uncertainty=dict(
                unit="world",
                seed_reduction="arithmetic_mean_within_world",
                strata="family",
                intervals=None,
                reason="Se exportan pares exactos, sin remuestreo ni pruebas de hipótesis.",
            ),
            delta_convention="Media de semillas menos referencia, con el mismo mundo y coste.",
            log_growth_convention=(
                "log1p(retorno), o penalización declarada cuando el retorno es -1."
            ),
            counts=dict(
                audit_episodes=sum(len(rows) for rows in learned.values()),
                control_episodes=sum(len(rows) for rows in references.values()),
                paired_world_rows=pair_count,
            ),
            training=training,
            aggregates=aggregates,
            paired=paired,
        )
        output.mkdir(parents=True, exist_ok=False)
        _csv(output / "aggregates.csv", aggregates)
        _csv(output / "paired_summary.csv", paired)
        _csv(output / "paired_worlds.csv", _paired_worlds(learned, references, worlds))
        report["artifacts"] = {
            name: dict(path=name, sha256=sha256(output / name))
            for name in ("aggregates.csv", "paired_summary.csv", "paired_worlds.csv")
        }
        atomic_json(output / "comparison.json", report)
        return report
    except (KeyError, TypeError, IndexError, AttributeError) as error:
        raise ValueError("Los recibos no conservan los campos requeridos") from error


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("campaign", "controls", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--controls-sha256", required=True)
    args = parser.parse_args(argv)
    report = compare_financial_campaign(
        args.campaign, args.controls, args.output, controls_sha256=args.controls_sha256
    )
    print(json.dumps(report["counts"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
