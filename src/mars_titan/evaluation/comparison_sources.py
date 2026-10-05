"""Admitir predicciones congeladas sin abrir pesos, datos brutos ni el test final."""

import re
from collections import Counter
from pathlib import Path

from mars_titan.data.cohort_files import read_manifest, safe_destination
from mars_titan.evaluation.splits import PARTITIONS, build_folds

_HELDOUT = ("calibration", "evaluation")
_STAGES = ("reference", "tabular", "posttraining")
_MAX_RUNS = 512
_MAX_JSON = 8 * 1024**2


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _digest(value):
    _require(
        isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value),
        "Falta una huella SHA256 válida",
    )
    return value


def _family(name):
    return "xgboost" if name == "xgboost_external_cuda" else name


def _closed(value, *, count=None):
    _require(
        value.get("status") == "completed" and value.get("final_test_opened") is False,
        "El recibo debe estar completo y conservar el test final cerrado",
    )
    if count is not None:
        _require(
            type(value.get("planned_runs")) is int
            and type(value.get("completed_runs")) is int
            and type(count) is int
            and value["planned_runs"] == value.get("completed_runs") == count,
            "El cierre no concilia con el número de ejecuciones",
        )


class _Reader:
    def __init__(self, reference, completion):
        self.roots = {"reference": reference, "completion": completion}
        self.cache = {}

    def path(self, base, value, *, relative=False):
        _require(isinstance(value, (str, Path)) and bool(str(value)), "Falta una ruta")
        part = Path(value)
        _require(
            ".." not in part.parts and not (relative and part.is_absolute()),
            "La ruta sale de la campaña",
        )
        path = part if part.is_absolute() else base / part
        safe_destination(path)
        path = path.resolve()
        _require(
            any(path.is_relative_to(root) for root in self.roots.values()),
            "La ruta sale de las dos campañas admitidas",
        )
        return path

    def json(self, path, expected=None, *, external=False):
        _require(path.suffix == ".json", "Solo se pueden abrir metadatos JSON en esta admisión")
        if external:
            safe_destination(path)
            path = path.resolve()
        else:
            path = self.path(path.parent, path)
        if path not in self.cache:
            value, digest = read_manifest(path, _MAX_JSON)
            _require(isinstance(value, dict), "El recibo JSON debe ser un objeto")
            self.cache[path] = value, digest
        value, digest = self.cache[path]
        _require(
            expected is None or digest == _digest(expected), "La huella del recibo no coincide"
        )
        return value, digest

    def label(self, path):
        for name, root in self.roots.items():
            if path.is_relative_to(root):
                return f"{name}/{path.relative_to(root).as_posix()}"
        raise ValueError("La ruta no tiene una procedencia publicable")


def _runs(summary, stage):
    records = summary["runs"]
    _require(
        isinstance(records, dict if stage == "posttraining" else list)
        and 1 <= len(records) <= _MAX_RUNS,
        "Faltan ejecuciones o exceden su límite",
    )
    _closed(summary, count=len(records))
    result = {}
    iterable = records.items() if isinstance(records, dict) else ((r["id"], r) for r in records)
    for name, record in iterable:
        _require(
            isinstance(name, str)
            and 0 < len(name) <= 256
            and name not in result
            and record.get("status") == "completed",
            "Hay ejecuciones inválidas o duplicadas",
        )
        result[name] = record
    return result


def _jobs(reader, summaries, folders):
    jobs, records, originals = {}, {}, {}
    for stage in _STAGES:
        entries = _runs(summaries[stage], stage)
        for name, entry in entries.items():
            if stage == "posttraining":
                relative, signature, phase = entry["path"], entry["sha256"], "posttraining"
            else:
                folder = entry["path"] if stage == "reference" else entry["attempts"][-1]["path"]
                relative, signature, phase = (
                    f"{folder}/run.json",
                    entry["report_sha256"],
                    entry["stage"],
                )
            _require(phase in {"search", "finalist", "posttraining"}, "La fase no está admitida")
            path = reader.path(folders[stage], relative, relative=True)
            _require(path.name == "run.json", "La ejecución no apunta a su recibo JSON")
            original, _ = reader.json(path, signature)
            _closed(original)
            key = f"{stage}/{name}"
            records[key], originals[key] = entry, original
            jobs[key] = dict(
                id=key,
                stage=stage,
                report=str(path),
                sha256=signature,
                comparator=None,
                phase=phase,
            )
        if stage == "reference":
            for name, entry in entries.items():
                parent = entry.get("parent")
                if parent is None:
                    continue
                key, parent_key = f"reference/{name}", f"reference/{parent}"
                _require(parent_key in jobs, "Falta el padre de una continuación")
                checkpoint = originals[parent_key]["checkpoint"]["sha256"]
                _require(
                    originals[key]["identity"]["initialization"]["parent_checkpoint_sha256"]
                    == checkpoint,
                    "La continuación no corresponde a su padre",
                )
                jobs[key]["comparator"] = dict(
                    path=jobs[parent_key]["report"], sha256=jobs[parent_key]["sha256"]
                )
    _require(
        len(jobs) <= _MAX_RUNS and len({j["report"] for j in jobs.values()}) == len(jobs),
        "Las ejecuciones duplican recibos o exceden su límite",
    )
    return jobs, records, originals


def _temporal(reader, proof, reference_proof, fold_id, base_hash):
    _require(
        {k: v for k, v in proof.items() if k not in {"parents", "tabular_summary_sha256"}}
        == reference_proof
        and proof.get("final_test_opened") is False,
        "La prueba de población no corresponde a la referencia congelada",
    )
    # Solo este manifiesto acreditado puede ser externo. Sus enlaces no se siguen.
    manifest, signature = reader.json(
        Path(proof["manifest"]), proof["manifest_sha256"], external=True
    )
    _require(
        manifest.get("kind") == "corpus_supervision"
        and manifest.get("cohort_complete") is True
        and manifest.get("final_test_opened") is False
        and manifest.get("counts") == proof["counts"]
        and manifest.get("source_manifest_sha256", signature) == base_hash,
        "El manifiesto no conserva la cohorte y su origen",
    )
    _require(
        isinstance(manifest.get("cohort_id"), str)
        and re.fullmatch(r"[a-zA-Z0-9_-]{1,128}", manifest["cohort_id"]),
        "La cohorte necesita un identificador publicable",
    )
    counts = manifest["counts"]
    _require(
        set(counts) == set(PARTITIONS) and all(type(n) is int and n > 0 for n in counts.values()),
        "La cohorte necesita las cuatro poblaciones temporales",
    )
    view = manifest["temporal_view"]
    protocol, fold = view["protocol"], view["fold"]
    _require(
        view["schema_version"] == 1
        and fold["id"] == fold_id
        and protocol["final_test_start"] == "2024-01-01"
        and fold in build_folds(protocol),
        "La ventana temporal no conserva el protocolo y el test reservado",
    )
    return manifest, signature, fold


def _parents(reader, proof, summaries, jobs, originals):
    selected = set()
    for stage in ("reference", "tabular"):
        values = summaries[stage]["selected"]
        _require(isinstance(values, dict) and values, "Falta la selección de la búsqueda")
        for value in values.values():
            key = f"{stage}/{value}"
            _require(
                key in jobs and jobs[key]["phase"] == "search",
                "La búsqueda elegida no está confirmada",
            )
            selected.add(key)
    parents = {}
    for family, record in proof["parents"].items():
        path = reader.path(reader.roots["reference"], record["report"])
        matches = [key for key in selected if jobs[key]["report"] == str(path)]
        _require(len(matches) == 1, "El padre no pertenece a la búsqueda seleccionada")
        key = matches[0]
        _require(
            key.split("/", 1)[1] == record["run_id"] and jobs[key]["sha256"] == record["sha256"],
            "La identidad del padre no coincide con su selección",
        )
        _digest(originals[key]["checkpoint"]["sha256"])
        original = originals[key]
        model = original.get("identity", {}).get("case", {}).get("kind", original.get("model"))
        _require(_family(model) == family, "El padre pertenece a una familia incompatible")
        parents[family] = key
    _require(set(parents.values()) == selected, "Falta un padre de la selección congelada")
    return selected, parents


def _cohort(original, stage, manifest, signature, ordered_hash, parent):
    identity = original.get("identity", {})
    if stage != "posttraining":
        _require(
            identity.get("manifest_sha256", original.get("manifest_sha256")) == signature
            and original.get("cohort_complete") is True
            and original.get("scope") == "full_corpus"
            and original.get("samples") == manifest["counts"],
            "La ejecución no conserva la identidad y población de la cohorte",
        )
        return
    inherited, dataset = identity["parent"], identity["dataset"]
    _require(
        original.get("domain") == "real"
        and inherited.get("final_test_opened") is False
        and inherited["source_sha256"] == signature
        and inherited["counts"] == manifest["counts"]
        and inherited["cohort_id"] == manifest["cohort_id"]
        and inherited["ordered_manifest_sha256"] == ordered_hash
        and inherited["parent_report_sha256"] == parent["sha256"]
        and inherited["checkpoint_sha256"] == parent["checkpoint_sha256"]
        and _family(inherited["model"]) == parent["family"]
        and dataset["parent_sha256"] == parent["checkpoint_sha256"]
        and dataset["train_sha256"] == dataset["validation_sha256"] == ordered_hash
        and dataset.get("synthetic") is None,
        "El ajuste no conserva el padre o la cohorte real congelados",
    )
    _require(
        all(original["samples"].get(p) == manifest["counts"][p] for p in ("train", "validation")),
        "Las muestras del ajuste no corresponden a la cohorte",
    )


def _metadata(original, result):
    metadata = {"case": result["case"], "selection": result.get("selection")}
    for field in ("grid",):
        if field in original.get("identity", {}):
            metadata[field] = original["identity"][field]
    for field in (
        "baseline",
        "stop_reason",
        "stopped_early",
        "budget",
        "global_step",
        "total_steps",
        "completed_rounds",
        "selected_round",
        "parameters",
        "fit_seconds",
        "total_seconds",
        "process_lifetime_peak_rss_bytes",
        "peak_vram_allocated_bytes",
    ):
        if field in original:
            metadata[field] = original[field]
    if "epochs" in original:
        metadata["completed_epochs"] = len(original["epochs"])
    if "attempts" in original:
        keys = {
            "seconds",
            "total_seconds",
            "updates",
            "process_lifetime_peak_rss_bytes",
            "peak_vram_allocated_bytes",
            "peak_vram_reserved_bytes",
            "fit_seconds",
        }
        metadata["attempts"] = [
            {k: v for k, v in row.items() if k in keys} for row in original["attempts"]
        ]
    return metadata


def _fold_sources(reader, fold_id, summaries, folders, evaluation, manifest, signature, proof):
    _require(
        evaluation.get("kind") == "frozen_temporal_evaluation",
        "La evaluación no acredita el trabajo temporal congelado",
    )
    jobs, records, originals = _jobs(reader, summaries, folders)
    identity = evaluation["identity"]
    _digest(identity["ordered_sha256"])
    _require(
        identity["manifest_sha256"] == signature and identity["partitions"] == list(_HELDOUT),
        "La evaluación cambió la cohorte o abrió otra partición",
    )
    saved_jobs = identity["jobs"]
    _require(
        isinstance(saved_jobs, list)
        and len(saved_jobs) == len(jobs)
        and len({j["id"] for j in saved_jobs}) == len(jobs),
        "El trabajo congelado contiene duplicados",
    )
    for job in saved_jobs:
        _require(
            job == jobs.get(job["id"]), "El trabajo de evaluación no conserva su copia congelada"
        )
    results = evaluation["runs"]
    _require(
        isinstance(results, dict) and set(results) == set(jobs), "Faltan evaluaciones confirmadas"
    )
    _closed(evaluation, count=len(jobs))
    selected, parents = _parents(reader, proof, summaries, jobs, originals)
    rows = []
    for key, job in jobs.items():
        original = originals[key]
        saved = results[key]
        path = reader.path(folders["evaluation"], saved["path"], relative=True)
        _require(path.name == "run.json", "La evaluación no apunta a un recibo JSON")
        result, _ = reader.json(path, saved["sha256"])
        _closed(result)
        case = (
            original.get("identity", {}).get("case")
            or original.get("identity", {}).get("options")
            or {"alpha": original.get("alpha")}
        )
        _require(
            result["job"] == job
            and result["checkpoint"] == original["checkpoint"]
            and result.get("selection") == original.get("selection")
            and result["case"] == case,
            "La evaluación no conserva el recibo original congelado",
        )
        family = _family(result["family"])
        stage = job["stage"]
        _require(family in parents, "La familia no tiene un padre acreditado")
        parent_id = None
        if stage == "posttraining":
            parent_id = parents[family]
            method = case["mode"]
        elif job["comparator"] is not None:
            parent_id = f"reference/{records[key]['parent']}"
            method = f"reference_{case['loss']}"
        else:
            method = "reference" if stage == "reference" else family
        parent = (
            None
            if parent_id is None
            else dict(
                jobs[parent_id],
                checkpoint_sha256=originals[parent_id]["checkpoint"]["sha256"],
                family=family,
            )
        )
        _cohort(original, stage, manifest, signature, identity["ordered_sha256"], parent)
        expected_family = (
            case.get("kind")
            if stage == "reference"
            else (records[key].get("kind") if stage == "tabular" else original.get("model"))
        )
        _require(
            family == _family(expected_family),
            "La familia de la evaluación no coincide con su modelo",
        )
        _require(
            result["primary"] == ("median" if stage == "posttraining" else "continuous"),
            "La salida principal no conserva su definición",
        )
        _require(
            set(result["predictions"]) == set(_HELDOUT),
            "Solo se admiten las dos particiones de predicción",
        )
        for partition in _HELDOUT:
            prediction = result["predictions"][partition]
            _require(
                prediction["path"] == f"{partition}.parquet",
                "La partición apunta a otra predicción",
            )
            metrics = prediction["metrics"]
            _require(
                isinstance(metrics, dict)
                and metrics
                and all(
                    isinstance(v, dict) and v.get("samples") == manifest["counts"][partition]
                    for v in metrics.values()
                ),
                "La predicción no conserva la población declarada",
            )
            rows.append(
                dict(
                    fold=fold_id,
                    id=key,
                    stage=stage,
                    phase=job["phase"],
                    family=family,
                    method=method,
                    seed=case.get("seed"),
                    included=job["phase"] != "search" or key in selected,
                    primary=result["primary"],
                    partition=partition,
                    path=reader.path(path.parent, prediction["path"], relative=True),
                    sha256=_digest(prediction["sha256"]),
                    declared_metrics=metrics,
                    bounds=manifest["temporal_view"]["fold"][partition],
                    parent_id=parent_id,
                    metadata=_metadata(original, result),
                )
            )
    _require(
        len({r["path"] for r in rows}) == len(rows), "Las predicciones contienen rutas duplicadas"
    )
    return rows


def predictive_sources(reference: Path, completion: Path) -> tuple[list[dict], dict]:
    """Admitir recibos y devolver rutas con sus hashes, sin abrir los Parquet.

    Cada fila identifica modelo y partición. ``bounds`` incluye el inicio y
    excluye el final. ``included`` conserva la búsqueda ganadora y los finalistas.
    ``provenance`` solo contiene rutas relativas a las dos campañas.
    """
    reference, completion = Path(reference), Path(completion)
    for path in (reference, completion):
        safe_destination(path)
    reference, completion = reference.resolve(), completion.resolve()
    _require(
        reference != completion, "Las campañas de referencia y continuación deben ser distintas"
    )
    reader = _Reader(reference, completion)
    try:
        ref, ref_hash = reader.json(reference / "summary.json")
        end, end_hash = reader.json(completion / "summary.json")
        _require(
            ref["kind"] == "temporal_reference_search"
            and end["kind"] == "temporal_posttraining_completion"
            and end["identity"]["reference_sha256"] == ref_hash,
            "La continuación no conserva la referencia original",
        )
        folds, stages = ref["folds"], end["stages"]
        _require(
            isinstance(folds, list)
            and 1 <= len(folds) <= 128
            and [r["id"] for r in folds] == [f"fold-{i:03d}" for i in range(len(folds))],
            "Las ventanas están duplicadas o fuera de secuencia",
        )
        _closed(ref, count=sum(r["planned_runs"] for r in folds))
        _closed(end, count=sum(r["planned_runs"] for r in stages))
        stages_by_key = {(r["fold"], r["stage"]): r for r in stages}
        expected_stages = {
            (f["id"], stage) for f in folds for stage in ("tabular", "posttraining", "evaluation")
        }
        _require(
            len(stages) == len(stages_by_key) and set(stages_by_key) == expected_stages,
            "Las etapas están duplicadas o no corresponden a las ventanas",
        )
        provenance = dict(
            schema_version=1,
            reference_sha256=ref_hash,
            completion_sha256=end_hash,
            final_test_opened=False,
            domain="real",
            folds=[],
        )
        sources = []
        for fold in folds:
            fold_id = fold["id"]
            _require(
                fold["status"] == "completed" and fold["completed_runs"] == fold["planned_runs"],
                "Falta una ventana neuronal completa",
            )
            folders = {
                stage: (
                    reference / fold_id if stage == "reference" else completion / fold_id / stage
                )
                for stage in (*_STAGES, "evaluation")
            }
            reference_proof = end["identity"]["references"][fold_id]
            summaries, signatures = {}, {}
            for stage in (*_STAGES, "evaluation"):
                receipt = None if stage == "reference" else stages_by_key[fold_id, stage]
                if receipt is not None:
                    _require(
                        receipt["status"] == "completed"
                        and receipt["completed_runs"] == receipt["planned_runs"],
                        "Falta una etapa terminada",
                    )
                expected = (
                    reference_proof["reference_summary_sha256"]
                    if receipt is None
                    else receipt["sha256"]
                )
                summaries[stage], signatures[stage] = reader.json(
                    folders[stage] / "summary.json", expected
                )
                expected_count = (
                    fold["planned_runs"] if receipt is None else receipt["planned_runs"]
                )
                _closed(summaries[stage], count=expected_count)
            proof = summaries["posttraining"]["identity"]["proof"]
            _require(
                proof["tabular_summary_sha256"] == signatures["tabular"],
                "La prueba tabular no conserva su huella",
            )
            manifest, signature, window = _temporal(
                reader, proof, reference_proof, fold_id, ref["identity"]["manifests"][fold_id]
            )
            _require(
                summaries["reference"]["identity"]["manifest_sha256"]
                == ref["identity"]["manifests"][fold_id]
                and summaries["tabular"]["identity"]["manifest_sha256"] == signature,
                "Los resúmenes no comparten la cohorte temporal",
            )
            for stage, record in summaries["evaluation"]["identity"]["sources"].items():
                _require(
                    stage in _STAGES
                    and reader.path(folders[stage], record["path"])
                    == folders[stage] / "summary.json"
                    and record["sha256"] == signatures[stage],
                    "La evaluación cambió sus resúmenes de origen",
                )
            _require(
                set(summaries["evaluation"]["identity"]["sources"]) == set(_STAGES),
                "Faltan fuentes de la evaluación congelada",
            )
            rows = _fold_sources(
                reader,
                fold_id,
                summaries,
                folders,
                summaries["evaluation"],
                manifest,
                signature,
                proof,
            )
            sources.extend(rows)
            provenance["folds"].append(
                dict(
                    id=fold_id,
                    counts=manifest["counts"],
                    cohort_id=manifest["cohort_id"],
                    manifest_sha256=signature,
                    windows={name: window[name] for name in _HELDOUT},
                    sources={
                        stage: dict(
                            path=reader.label(folders[stage] / "summary.json"),
                            sha256=signatures[stage],
                        )
                        for stage in (*_STAGES, "evaluation")
                    },
                    prediction_files=len(rows),
                    included_files=sum(r["included"] for r in rows),
                )
            )
            reader.cache.clear()
        provenance["counts"] = dict(
            prediction_files=len(sources),
            included_files=sum(r["included"] for r in sources),
            models=len(sources) // 2,
            by_stage=dict(Counter(r["stage"] for r in sources)),
        )
        return sources, provenance
    except (KeyError, TypeError, IndexError, AttributeError) as error:
        raise ValueError("Los recibos no conservan los campos y tipos requeridos") from error
