"""Reconstruir parámetros elegidos para E2, sin trasladar estados de aprendizaje o memoria."""

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

from mars_titan.data.cohort_files import read_manifest, safe_destination
from mars_titan.data.storage import sha256
from mars_titan.memory.episodic_codec import FrozenEpisodeCodec
from mars_titan.models.titans.config import canonical
from mars_titan.models.titans.financial_inputs import FinancialInputSpec
from mars_titan.models.titans.frozen_financial import FrozenFinancialConsumer

from .checkpoints import MAX_STATE_BYTES, load_training_state
from .kernel_policy import declared_policy
from .mars_titan_run import ReadoutRecipe, _parameters_digest, case_recipe
from .mars_titan_walk_forward import (
    KIND,
    _carried_readout,
    _components,
    _cuda,
    _frozen_parent,
    _parent,
    _readout,
    _variant,
)
from .titans_walk_forward import KIND as CORE_KIND
from .titans_walk_forward import _require, _verify


def _digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def _report(path, kind):
    safe_destination(path)
    value, digest = read_manifest(path, 16 * 1024**2)
    _require(
        type(value) is dict
        and type(value.get("schema_version")) is int
        and value["schema_version"] == 1
        and value.get("kind") == kind
        and value.get("status") == "completed"
        and value.get("final_test_opened") is False
        and type(value.get("identity")) is dict
        and type(value.get("request")) is dict
        and type(value.get("predictions")) is dict
        and type(value.get("checkpoint")) is dict
        and value["identity"].get("request") == value["request"]
        and value.get("run_id") == _digest(value["identity"]),
        "El informe elegido no conserva su estado completado, identidad y test cerrado",
    )
    return value, digest


def _fit(folder, report, reference):
    link = report.get("fit")
    _require(type(link) is dict and link.get("path") == "fit/run.json", "Falta el ajuste elegido")
    path = folder / "fit/run.json"
    safe_destination(path)
    fit, digest = read_manifest(path, 16 * 1024**2)
    _require(
        digest == link.get("sha256")
        and type(fit) is dict
        and type(fit.get("schema_version")) is int
        and fit["schema_version"] == 1
        and fit.get("status") == "completed"
        and fit.get("final_test_opened") is False
        and type(fit.get("identity")) is dict
        and fit.get("run_id") == _digest(fit["identity"]) == link.get("run_id")
        and type(fit.get("best_checkpoint")) is dict,
        "El ajuste elegido no concuerda con su informe y huella",
    )
    best = fit["best_checkpoint"]
    _require(
        type(best.get("sha256")) is str
        and re.fullmatch(r"[a-f0-9]{64}", best["sha256"]) is not None
        and best.get("path") == f"checkpoints/state-{best['sha256']}.pt"
        and reference == dict(best, path="fit/" + best["path"]),
        "La referencia no corresponde al mejor checkpoint del ajuste",
    )
    checkpoint = folder / "fit" / best["path"]
    safe_destination(checkpoint)
    _require(
        checkpoint.is_file() and 0 < checkpoint.stat().st_size <= MAX_STATE_BYTES,
        "El mejor checkpoint no es regular o supera su presupuesto",
    )
    _require(sha256(checkpoint) == best["sha256"], "El mejor checkpoint ha cambiado")
    return fit, [(path, digest), (checkpoint, best["sha256"])]


@dataclass(frozen=True)
class SelectedSpacedReader:
    """Parámetros compartidos y procedencia. La identidad se devuelve sin referencias mutables."""

    consumer: FrozenFinancialConsumer
    codec: FrozenEpisodeCodec
    recipe: ReadoutRecipe
    _identity_json: str

    def identity(self):
        return json.loads(self._identity_json)


def load_selected_reader(
    anchor, anchor_view, specification, *, device="cuda:0", reader_contract="standard"
):
    """Cargar el estado elegido en su vista e índice originales, sin ejecutar predicciones.

    El controlador puede reconstruir la especificación del índice de entrenamiento desde
    el informe del padre. Trasladar parámetros a otra vista exige su contrato temporal
    independiente. Los checksums detectan cambios, pero no autentican un entrenamiento.
    """
    _require(type(specification) is FinancialInputSpec, "Falta la especificación original")
    _require(
        type(reader_contract) is str and reader_contract in ("standard", "regional"),
        "La carga E2 necesita un contrato de lector explícito",
    )
    anchor, view = Path(anchor), Path(anchor_view)
    safe_destination(anchor)
    safe_destination(view)
    report, report_sha = _report(anchor / "run.json", KIND)
    request, identity = report["request"], report["identity"]
    _, view_sha = read_manifest(view, 64 * 1024**2)
    _require(
        request.get("view_sha256") == view_sha == specification.source_sha256
        and type(request.get("seed")) is int
        and 0 <= request["seed"] < 2**32,
        "La selección no corresponde a la vista y semilla originales",
    )
    components = _components(request.get("components"))
    _require(
        components.get("episodic_bank") == "m1"
        and type(components.get("refinements", 1)) is int
        and components.get("refinements", 1) == 1
        and (components.get("evidence_bridge") is not None) == (reader_contract == "regional"),
        "El estado E2 necesita M1 K1 y el contrato estándar o regional correspondiente",
    )
    link = request.get("parent")
    _require(
        type(link) is dict and type(link.get("path")) is str and Path(link["path"]).is_absolute(),
        "La selección no identifica la carpeta original del padre",
    )
    parent = Path(link["path"])
    safe_destination(parent)
    checked_parent, checked_sha = _report(parent / "run.json", CORE_KIND)
    _require(
        type(checked_parent["identity"].get("indices")) is dict,
        "El padre no declara sus índices de observaciones",
    )
    parent_report, parent_sha = _parent(parent, view, request["seed"])
    _require(
        parent_sha == checked_sha == link.get("run_sha256")
        and parent_report["checkpoint"]["sha256"] == link.get("checkpoint_sha256")
        and parent_report["identity"]["indices"].get("train") == specification.view_sha256
        and parent_report["request"].get("window") == request.get("window"),
        "El padre no conserva su informe, checkpoint, ventana o índice de entrenamiento",
    )
    selected, selected_sha = read_manifest(anchor / "selected.json", 256 * 1024)
    _require(
        report.get("checkpoint") == dict(path="selected.json", sha256=selected_sha)
        and type(selected) is dict
        and type(selected.get("schema_version")) is int
        and selected.get("schema_version") == 1
        and selected.get("kind") == "mars_titan_selected_state"
        and selected.get("parent") == link
        and selected.get("variant_sha256") == identity.get("variant_sha256")
        and type(report.get("fit")) is dict
        and selected.get("readout") == report["fit"].get("best_checkpoint"),
        "El descriptor compuesto no concuerda con la selección del padre y lector",
    )
    _, parent_files = _fit(parent, parent_report, parent_report["checkpoint"])
    fit, fit_files = _fit(anchor, report, selected["readout"])
    _require(
        identity.get("fit_run_id") == fit["run_id"]
        and fit["identity"].get("dataset_sha256") == view_sha
        and fit["identity"].get("admission") == "m1"
        and type(fit["identity"].get("parent")) is dict
        and type(fit["identity"].get("readout")) is dict,
        "El ajuste del lector no conserva su identidad, datos o admisión",
    )
    _verify(anchor, report)
    try:
        recipe = case_recipe(identity["recipe"], request.get("search_case"))
    except (KeyError, TypeError) as error:
        raise ValueError("La receta guardada del lector no conserva su estructura") from error
    _cuda(device)
    declared_policy(recipe.precision)
    try:
        predictor = _frozen_parent(
            parent_report["identity"]["recipe"],
            specification,
            request["seed"],
            device,
            parent,
            parent_report,
        )
    except (KeyError, TypeError) as error:
        raise ValueError("La receta guardada del padre no conserva su estructura") from error
    variant = _variant(predictor, parent_report, components)
    _require(
        variant.fingerprint() == identity.get("variant_sha256")
        and canonical(variant.identity()) == canonical(identity.get("variant"))
        and predictor._parameter_id == fit["identity"].get("parent", {}).get("parameters_sha256"),
        "La variante o los parámetros del padre no corresponden al ajuste del lector",
    )
    codec = FrozenEpisodeCodec(specification)
    reader = _readout(variant, codec, predictor, recipe, request["seed"])
    state = load_training_state(
        anchor / "fit/checkpoints",
        expected_identity=fit["identity"],
        selection="best",
        expected_sha256=selected["readout"]["sha256"],
    )
    _require(
        state.get("readout_sha256") == selected["readout"].get("readout_sha256")
        and type(state.get("model")) is dict
        and type(state["model"].get("_extra_state")) is dict
        and canonical(state["model"]["_extra_state"]) == canonical(fit["identity"]["readout"]),
        "Los pesos y el contrato del lector no corresponden al checkpoint elegido",
    )
    _carried_readout(state["model"], reader)
    _require(
        _parameters_digest(reader) == state["readout_sha256"],
        "La carga no reproduce los parámetros elegidos del lector",
    )
    readout_sha = state["readout_sha256"]
    del state
    reader.eval().requires_grad_(False)
    consumer = FrozenFinancialConsumer(predictor, readout=reader)
    pins = [
        (view, view_sha),
        (anchor / "run.json", report_sha),
        (anchor / "selected.json", selected_sha),
        (parent / "run.json", parent_sha),
        *parent_files,
        *fit_files,
    ]
    _require(all(sha256(path) == value for path, value in pins), "La selección cambió al cargar")
    provenance = dict(
        schema_version=1,
        kind="spaced_selected_regional_reader_v1"
        if reader_contract == "regional"
        else "spaced_selected_reader_v1",
        state_transfer="shared_parameters_only",
        anchor=dict(run_id=report["run_id"], run_sha256=report_sha, selected_sha256=selected_sha),
        view_sha256=view_sha,
        seed=request["seed"],
        variant_sha256=variant.fingerprint(),
        parent=dict(
            run_id=parent_report["run_id"],
            run_sha256=parent_sha,
            parameters_sha256=predictor._parameter_id,
        ),
        readout_sha256=readout_sha,
        consumer_id=consumer.model_id,
    )
    if reader_contract == "regional":
        provenance["reader_contract"] = reader_contract
    return SelectedSpacedReader(consumer, codec, recipe, canonical(provenance))
