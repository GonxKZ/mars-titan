"""Postentrenamiento de la GRU candidata nativa: cabeza adaptada o continuación completa.

La GRU, la fusión, la lectura episódica y el refinador viven dentro del módulo LibTorch
`Candidate`, así que la matriz solo adapta la cabeza (`head_weight` y `head_bias`). La
cabeza adaptada se calcula en Python sobre el estado final nativo con la misma
parametrización ordenada (`ordered_quantiles`). Con la corrección nula reproduce la cabeza
nativa, y el módulo nativo queda congelado y fuera del grafo. La continuación completa
ajusta todos los parámetros nativos desde el estado elegido, con el presupuesto de la matriz.

El banco, su admisión y su instantánea por evento son los del ajuste base. Las fases (con
su calentamiento), la admisión, K, los bloques y la acumulación salen de la identidad del
padre. Solo cambian el optimizador, el presupuesto y la selección.
"""

import hashlib
import json
import time
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path

import pyarrow.parquet as pq
import torch
from torch import nn
from torch.nn import functional

from mars_titan.data.cohort_files import read_manifest, safe_destination
from mars_titan.data.input_policy import HISTORICAL_MASKED, policy_identity
from mars_titan.data.storage import atomic_json, outside_source, sha256
from mars_titan.memory.financial_observations import (
    FinancialObservationSource,
    prepare_observation_index,
)
from mars_titan.memory.financial_session import FinancialPhase
from mars_titan.models.predictive_adaptation import adapter_names, attach_adapters
from mars_titan.models.quantile_head import ordered_quantiles
from mars_titan.models.titans.config import canonical
from mars_titan.training.candidate_run import (
    HELDOUT,
    CandidateChronologicalPredictor,
    CandidateChronologicalTrainer,
    _Pause,
    restore_selected,
)
from mars_titan.training.candidate_walk_forward import (
    WORLD,
    _anchor,
    _prepare_output,
    _window,
    anchor_adapter,
    check_view_rows,
    window_sources,
)
from mars_titan.training.carried_predictions import _destination, _receipt, carried_window
from mars_titan.training.checkpoints import load_training_state
from mars_titan.training.corpus_inputs import CorpusDataset
from mars_titan.training.learning_hold import require_learning_allowed

from . import chronological_matrix as cm

KIND = "candidate_posttraining_window"
CARRIED = ("calibration", "evaluation")
PARTITIONS = ("train", "validation", *HELDOUT)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


class CandidateHead(nn.Module):
    """Cabeza de cuantiles de la candidata, copiada del módulo nativo y congelada."""

    def __init__(self, model):
        super().__init__()
        values = model.named_parameters()
        self.weight = nn.Parameter(values["head_weight"].detach().clone(), requires_grad=False)
        self.bias = nn.Parameter(values["head_bias"].detach().clone(), requires_grad=False)

    def forward(self, state):
        return ordered_quantiles(functional.linear(state, self.weight, self.bias))


def adapted_head(model, points, seed):
    """Cabeza con la corrección completa e inicialmente nula del brazo `head`."""
    return attach_adapters(CandidateHead(model), cm.candidate_targets(points), seed=seed)


class _HeadForward:
    """Con cabeza adaptada, el módulo nativo solo da su estado final, sin grafo."""

    head = None

    def _forward(self, batch, memory, *, grad):
        if self.head is None:
            return super()._forward(batch, memory, grad=grad)
        inputs = self._inputs(batch)
        with torch.no_grad():
            state = self.model.forward(inputs, memory, self.recipe.refinements).state
        with torch.set_grad_enabled(grad):
            return self.head(state)


class CandidateHeadPredictor(_HeadForward, CandidateChronologicalPredictor):
    """Recorrido congelado de la candidata con la cabeza adaptada elegida."""

    def __init__(self, adapter, recipe, *, head, **options):
        self.head = head
        super().__init__(adapter, recipe, **options)


class CandidatePosttrainer(_HeadForward, CandidateChronologicalTrainer):
    """Ajustar la cabeza adaptada o continuar todos los parámetros nativos.

    Con `head` solo se ajustan sus correcciones y los parámetros nativos quedan congelados
    y comprobados por su huella. Sin `head` es la continuación completa del ajuste base.
    """

    def __init__(self, adapter, recipe, *, posttraining, head=None, **options):
        _require(isinstance(posttraining, dict) and posttraining, "Falta la declaración del caso")
        self.head = head
        native = adapter.model.named_parameters()
        if head is not None:
            _require(
                adapter_names(head)
                and all(
                    value.requires_grad == (name in adapter_names(head))
                    for name, value in head.named_parameters()
                ),
                "La cabeza adaptada solo ajusta sus correcciones",
            )
            for value in native.values():
                value.requires_grad_(False)
        self.native_sha256 = adapter.model.parameter_fingerprint()
        super().__init__(adapter, recipe, **options)
        self.identity["posttraining"] = dict(
            posttraining,
            native_parameters_sha256=self.native_sha256 if head is not None else None,
            head="python_ordered_quantiles_over_native_final_state" if head else None,
        )
        self.identity = json.loads(canonical(self.identity))
        self.run_id = hashlib.sha256(canonical(self.identity).encode()).hexdigest()

    def _parameter_groups(self, model, admission):
        if self.head is None:
            return super()._parameter_groups(model, admission)
        values = dict(self.head.named_parameters())
        names = adapter_names(self.head)
        return (
            dict(head_adapters=[f"head.{name}" for name in names]),
            [],
            {f"head.{name}": values[name] for name in names},
        )

    def _check_runtime(self):
        super()._check_runtime()
        if self.head is not None and self.model.parameter_fingerprint() != self.native_sha256:
            raise ValueError("El ajuste cambió parámetros nativos fuera de la cabeza adaptada")

    def _extra_state(self):
        if self.head is None:
            return {}
        values = dict(self.head.named_parameters())
        return dict(
            head_adapters={
                name: values[name].detach().cpu().clone() for name in adapter_names(self.head)
            }
        )

    def _restore_extra(self, state):
        if self.head is not None:
            load_head(self.head, state.get("head_adapters"))


def load_head(head, saved):
    """Copiar las correcciones guardadas de la cabeza en su sitio."""
    values = dict(head.named_parameters())
    _require(
        isinstance(saved, dict) and list(saved) == adapter_names(head),
        "El estado no contiene las correcciones de la cabeza adaptada",
    )
    with torch.no_grad():
        for name, value in saved.items():
            _require(
                value.shape == values[name].shape and value.dtype == values[name].dtype,
                "Las correcciones de la cabeza no conservan forma o precisión",
            )
            values[name].copy_(value.to(values[name].device))


def _sources(dataset, folder, phases):
    """Índices de los tramos con las fases del padre, calentamiento incluido."""
    sources = {}
    for name, phase in phases.items():
        directory = Path(folder) / name
        manifest = prepare_observation_index(
            dataset, directory, phase=phase, resume=directory.exists()
        )
        sources[name] = FinancialObservationSource(dataset, manifest)
    return sources


def _parent_phases(report, names):
    declared = report["identity"]["sources"]
    _require(set(names) <= set(declared), "El padre no declara las fases de sus tramos")
    return {name: FinancialPhase(**declared[name]["phase"]) for name in names}


def run_candidate_posttraining(
    parent,
    view,
    output,
    *,
    case,
    matrix,
    digest,
    device="cuda:0",
    stop=None,
    optimizer_factory=None,
):
    """Postentrenar el estado elegido de una ventana de la candidata en esa misma ventana."""
    require_learning_allowed("el postentrenamiento de la GRU candidata")
    _require(device in ("cpu", "cuda:0"), "El dispositivo debe ser cpu o cuda:0 explícitos")
    cm.validate_case(case)
    adapter_case = case["adapter"]
    _require(
        (adapter_case is None or adapter_case["design"] == cm.CANDIDATE)
        and (adapter_case is None or adapter_case["matrix_sha256"] == digest),
        "El caso no pertenece a la GRU candidata de la matriz declarada",
    )
    parent, view, output = Path(parent), Path(view), Path(output)
    window, window_sha, report, _ = _anchor(parent, view)
    _require(window["seed"] == case["seed"], "El padre no se ajustó con la semilla del caso")
    dataset = CorpusDataset(view, input_policy=HISTORICAL_MASKED)
    contracts = _window(dataset)
    _prepare_output(view, output, dataset)
    outside_source(parent, output)
    request = dict(
        view_sha256=dataset.identity,
        matrix_sha256=digest,
        case=case,
        parent=dict(path=str(parent.resolve()), window_sha256=window_sha),
        device=device,
    )
    path = output / "window.json"
    if path.is_file():
        previous, _ = read_manifest(path, 8 * 1024**2)
        _require(previous.get("request") == request, "La petición de la ventana ha cambiado")
        if previous.get("status") == "completed":
            return previous
    output.mkdir(parents=True, exist_ok=True)
    sources = _sources(dataset, output / "indices", _parent_phases(report, PARTITIONS))
    adapter, parent_recipe, _, _ = anchor_adapter(
        parent, view, sources["train"].specification(), device=device
    )
    recipe = replace(parent_recipe, **cm.recipe_options(case))
    head, description = None, None
    if adapter_case is not None:
        seed = cm.component_seed(case, "head")
        head = adapted_head(adapter.model, adapter_case["points"], seed).to(device)
        description = dict(
            cm.describe(cm.candidate_targets(adapter_case["points"]), head), seed=seed
        )
    fold = next(iter(contracts.values()))["fold"]
    folder = output / "run"
    trainer = CandidatePosttrainer(
        adapter,
        recipe,
        posttraining=dict(
            kind=KIND,
            case=case,
            parent=dict(
                window_sha256=window_sha,
                checkpoint_sha256=window["checkpoint"]["sha256"],
                parameters_sha256=window["checkpoint"]["parameters_sha256"],
            ),
            adapter=description,
        ),
        head=head,
        train=sources["train"],
        validation=sources["validation"],
        heldout={name: sources[name] for name in HELDOUT},
        output=folder,
        world=WORLD,
        fold=fold["id"],
        optimizer_factory=optimizer_factory,
    )
    started = time.perf_counter()
    result = trainer.run(resume=(folder / "run.json").is_file(), stop=stop)
    if result["status"] != "completed":
        return dict(status=result["status"], final_test_opened=False)
    predictions = {}
    for name, record in result["predictions"].items():
        table_path = folder / record["path"]
        predictions[name] = dict(
            path=str(table_path.relative_to(output)),
            sha256=record["sha256"],
            rows=check_view_rows(pq.read_table(table_path), dataset, name),
            metrics=record["metrics"],
        )
    best = result["best_checkpoint"]
    document = dict(
        schema_version=1,
        kind=KIND,
        status="completed",
        request=request,
        fold=fold,
        markets=sorted(contracts),
        recipe=recipe.identity(),
        posttraining=trainer.identity["posttraining"],
        run=dict(
            path=str((folder / "run.json").relative_to(output)),
            sha256=sha256(folder / "run.json"),
            run_id=result["run_id"],
        ),
        checkpoint=dict(
            path=str((folder / best["path"]).relative_to(output)),
            sha256=best["sha256"],
            parameters_sha256=best["parameters_sha256"],
        ),
        global_step=result["global_step"],
        best_epoch=result["best_epoch"],
        best_score=result["best_score"],
        selection=result["selection"],
        sources={
            name: dict(index_sha256=source.identity, phase=asdict(source.phase))
            for name, source in sources.items()
        },
        predictions=predictions,
        seconds=time.perf_counter() - started,
        final_test_opened=False,
        finished_at_utc=datetime.now(UTC).isoformat(),
        **policy_identity(HISTORICAL_MASKED),
    )
    atomic_json(path, document)
    return document


def _completed(anchor, anchor_view):
    document, digest = read_manifest(Path(anchor) / "window.json", 8 * 1024**2)
    _require(
        document.get("kind") == KIND
        and document.get("status") == "completed"
        and document.get("final_test_opened") is False
        and document["request"]["view_sha256"] == sha256(anchor_view),
        "El ancla no es un postentrenamiento completo de la candidata sobre esa vista",
    )
    safe_destination(Path(anchor) / document["run"]["path"])
    _require(
        sha256(Path(anchor) / document["run"]["path"]) == document["run"]["sha256"],
        "La ejecución del ancla cambió",
    )
    return document, digest


def carry_candidate_posttraining(anchor, anchor_view, view, output, *, device="cuda:0", stop=None):
    """Variante B: calibración y evaluación posteriores con el estado elegido en el ancla."""
    require_learning_allowed("la predicción trasladada del postentrenamiento de la candidata")
    anchor, anchor_view, view = Path(anchor), Path(anchor_view), Path(view)
    document, digest = _completed(anchor, anchor_view)
    dataset = CorpusDataset(view, input_policy=HISTORICAL_MASKED)
    contracts = _window(dataset)
    anchor_meta, _ = read_manifest(anchor_view, 8 * 1024**2)
    anchor_fold, fold, months = carried_window(
        anchor_meta, dataset.manifest, input_policy=HISTORICAL_MASKED
    )
    output = _destination(output, dataset.roots.values())
    _prepare_output(view, output, dataset)
    output.mkdir(parents=True)
    parent = Path(document["request"]["parent"]["path"])
    _, _, parent_report, _ = _anchor(parent, anchor_view)
    folder = anchor / "run"
    run, _ = read_manifest(folder / "run.json", 16 * 1024**2)
    # Las fases trasladadas son las del traslado de la campaña base en esa ventana.
    sources = window_sources(dataset, output / "indices", CARRIED)
    adapter, parent_recipe, _, _ = anchor_adapter(
        parent, anchor_view, sources["calibration"].specification(), device=device
    )
    parameters = restore_selected(folder, adapter, carried=True)
    _require(
        parameters == document["checkpoint"]["parameters_sha256"],
        "Los parámetros cargados no son los elegidos en el ancla",
    )
    recipe = replace(parent_recipe, **cm.recipe_options(document["request"]["case"]))
    description = document["posttraining"]["adapter"]
    head = None
    if description is not None:
        head = adapted_head(
            adapter.model, document["request"]["case"]["adapter"]["points"], description["seed"]
        ).to(device)
        state = load_training_state(
            folder / "checkpoints",
            expected_identity=run["identity"],
            selection="best",
            expected_sha256=document["checkpoint"]["sha256"],
        )
        load_head(head, state["head_adapters"])
    target = output / "predictions"
    target.mkdir()
    predictor = CandidateHeadPredictor(
        adapter, recipe, head=head, sources=sources, output=target, world=WORLD, fold=fold["id"]
    )
    predictions = {}
    try:
        for name in CARRIED:
            path = target / f"{name}-predictions.parquet"
            metrics = predictor.evaluate(sources[name], stop=stop, destination=path)
            predictions[name] = dict(
                path=str(path.relative_to(output)),
                sha256=sha256(path),
                rows=check_view_rows(pq.read_table(path), dataset, name),
                metrics=metrics,
            )
    except _Pause:
        return dict(status="paused", final_test_opened=False)
    return _receipt(
        output,
        dict(
            model=KIND,
            anchor=dict(
                window_sha256=digest,
                checkpoint_sha256=document["checkpoint"]["sha256"],
                parameters_sha256=document["checkpoint"]["parameters_sha256"],
                case=document["request"]["case"],
                fold=anchor_fold,
            ),
            parent_recipe=parent_report["identity"]["recipe"],
            manifest_sha256=dataset.identity,
            fold=fold,
            markets=sorted(contracts),
            months_since_anchor_information=months,
            predictions=predictions,
            **policy_identity(HISTORICAL_MASKED),
        ),
    )
