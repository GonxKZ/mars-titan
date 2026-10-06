"""Emparejar ajustes con finalistas de la misma configuración y semilla."""

from pathlib import Path

from mars_titan.data.cohort_files import read_manifest
from mars_titan.data.storage import sha256

MATCHING_FIELDS = frozenset({"schema_version", "parent_seed_policy", "parents_by_seed"})
NEURAL = ("rnn", "lstm", "gru", "dlinear")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def matching_seeds(proof):
    """Admitir exclusivamente el contrato completo nuevo o la prueba antigua sin ampliaciones."""
    if not MATCHING_FIELDS.intersection(proof):
        return None
    _require(
        type(proof.get("schema_version")) is int
        and proof["schema_version"] == 2
        and proof.get("parent_seed_policy") == "matching"
        and isinstance(proof.get("parents_by_seed"), dict)
        and set(proof["parents_by_seed"]) == set(proof["parents"]),
        "La prueba no conserva el contrato de emparejamiento por semilla",
    )
    seeds = None
    for family, records in proof["parents_by_seed"].items():
        _require(
            isinstance(records, dict)
            and 1 <= len(records) <= 10
            and all(
                isinstance(key, str)
                and key.isascii()
                and key.isdecimal()
                and str(int(key)) == key
                and 0 <= int(key) < 2**32
                for key in records
            ),
            "Las semillas de los padres no son válidas",
        )
        current = sorted(int(key) for key in records)
        _require(seeds is None or seeds == current, "Faltan semillas de una familia de padres")
        seeds = current
        for seed, record in records.items():
            _require(
                isinstance(record, dict)
                and record.get("shared_deterministic") is (family == "ridge")
                and (
                    record.get("parent_seed") is None
                    if family == "ridge"
                    else type(record.get("parent_seed")) is int
                    and record["parent_seed"] == int(seed)
                ),
                "El padre no corresponde a la semilla del ajuste",
            )
        if family == "ridge":
            first = next(iter(records.values()))
            _require(
                all(record == first for record in records.values()),
                "Ridge debe compartir un único padre determinista",
            )
    _require(seeds is not None, "Faltan padres emparejados")
    return seeds


def parent_for_seed(proof, family, seed):
    """Resolver el padre declarado sin convertir una prueba incompleta en una entrada antigua."""
    seeds = matching_seeds(proof)
    if seeds is None:
        return proof["parents"][family]
    _require(type(seed) is int and seed in seeds, "Falta el padre de la semilla del ajuste")
    return proof["parents_by_seed"][family][str(seed)]


def same_configuration(selected, candidate, seed):
    """Permitir únicamente la semilla como diferencia de la configuración seleccionada."""
    return (
        isinstance(selected, dict)
        and isinstance(candidate, dict)
        and type(candidate.get("seed")) is int
        and candidate["seed"] == seed
        and {key: value for key, value in selected.items() if key != "seed"}
        == {key: value for key, value in candidate.items() if key != "seed"}
    )


def _record(root, row, family):
    folder = row["path"] if family in NEURAL else row["attempts"][-1]["path"]
    path = root / folder / "run.json"
    report, signature = read_manifest(path, 8 * 1024**2)
    _require(signature == row["report_sha256"], "El recibo del padre ha cambiado")
    if family != "ridge":
        field = "case" if family in NEURAL else "options"
        configuration = row["case"] if family in NEURAL else row["parameters"]
        _require(
            report["identity"][field] == configuration
            and report["predictions"]["validation"]["metrics"]["session_mae"] == row["session_mae"],
            "El padre no conserva la configuración o selección confirmadas",
        )
        if family in NEURAL and "selection" in configuration:
            selection = report.get("selection") or {}
            _require(
                selection.get("best_score") == row["session_mae"]
                and type(selection.get("best_epoch")) is int
                and type(selection.get("last_epoch")) is int
                and 0
                <= selection["best_epoch"]
                <= selection["last_epoch"]
                <= configuration["epochs"],
                "La época y puntuación seleccionadas del padre no concuerdan",
            )
            index, _ = read_manifest(path.parent / "checkpoints/latest.json", 8 * 1024**2)
            chosen = index.get("best") or {}
            _require(
                report["checkpoint"]
                == dict(path=f"checkpoints/{chosen.get('name')}", sha256=chosen.get("sha256")),
                "El checkpoint no corresponde a la selección del padre",
            )
    return dict(
        report=str(path.resolve()),
        sha256=signature,
        run_id=row["id"],
        parent_seed=None if family == "ridge" else configuration["seed"],
        shared_deterministic=family == "ridge",
    ), report


def matching_parents(reference, tabular, arm="US", *, seeds):
    """Verificar los ganadores y sus finalistas antes de abrir la cola de ajustes."""
    from mars_titan.training.klpo_queue import selected_parents

    _require(
        isinstance(seeds, list)
        and 1 <= len(seeds) <= 10
        and all(type(seed) is int and 0 <= seed < 2**32 for seed in seeds)
        and len(set(seeds)) == len(seeds),
        "Las semillas de emparejamiento no son válidas",
    )
    reference, tabular = Path(reference), Path(tabular)
    proof = selected_parents(reference, tabular, arm)
    neural, neural_hash = read_manifest(reference, 8 * 1024**2)
    tables, table_hash = read_manifest(tabular, 8 * 1024**2)
    _require(
        neural_hash == proof["reference_summary_sha256"]
        and table_hash == proof["tabular_summary_sha256"],
        "Las campañas cambiaron tras admitir la selección",
    )
    paired = {}
    for family, winner in proof["parents"].items():
        summary, root = (neural, reference.parent) if family in NEURAL else (tables, tabular.parent)
        selected = next(row for row in summary["runs"] if row["id"] == winner["run_id"])
        initial, report = _record(root, selected, family)
        field = "case" if family in NEURAL else "parameters"
        paired[family] = {}
        for seed in seeds:
            if family == "ridge" or initial["parent_seed"] == seed:
                record = initial
            else:
                candidates = [
                    row
                    for row in summary["runs"]
                    if row.get("stage") == "finalist"
                    and (row.get("case", {}).get("kind") if family in NEURAL else row.get("kind"))
                    == family
                    and (
                        family not in NEURAL
                        or (row.get("arm") == arm and row.get("weighting") == "natural")
                    )
                    and same_configuration(selected[field], row.get(field), seed)
                ]
                _require(
                    len(candidates) == 1,
                    "Falta un finalista único de la misma configuración y semilla",
                )
                record, replica = _record(root, candidates[0], family)
                parameter = "case" if family in NEURAL else "options"
                _require(
                    {k: v for k, v in report["identity"].items() if k != parameter}
                    == {k: v for k, v in replica["identity"].items() if k != parameter},
                    "El finalista cambió la identidad de configuración del padre",
                )
            if family in NEURAL:
                expected = dict(
                    arm=arm, weighting="natural", kind=family, seed=seed, run_id=record["run_id"]
                )
                _require(
                    summary.get("finalists", []).count(expected) == 1,
                    "El padre no es el finalista declarado",
                )
            paired[family][str(seed)] = record
    _require(
        sha256(reference) == neural_hash
        and sha256(tabular) == table_hash
        and sha256(Path(proof["manifest"])) == proof["manifest_sha256"],
        "Las fuentes cambiaron durante el emparejamiento",
    )
    result = proof | dict(schema_version=2, parent_seed_policy="matching", parents_by_seed=paired)
    matching_seeds(result)
    return result
