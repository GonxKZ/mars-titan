"""Checkpoints de pesos fijados a mano, sin entrenamiento ni métricas científicas."""

import copy
import hashlib
import json
from pathlib import Path

import torch

from mars_titan.data.storage import sha256
from mars_titan.models.titans.config import canonical
from mars_titan.models.titans.financial import FinancialConfig, FinancialPredictor
from mars_titan.training.checkpoints import save_training_state
from mars_titan.training.mars_titan_run import _parameters_digest, case_recipe
from mars_titan.training.mars_titan_walk_forward import _readout, _variant


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    return sha256(path)


def values(module):
    return {
        key: item.detach().cpu().clone() if isinstance(item, torch.Tensor) else copy.deepcopy(item)
        for key, item in module.state_dict().items()
    }


def selected_fixture(root, source, *, hidden_size=32, quantiles=False):
    view = source["prefixes"]._manifest_paths[0]
    parent, anchor = root / "parent", root / "anchor"
    window = source.get("window", dict(id="fixture_window"))
    core_document = dict(
        predictor=dict(hidden_size=hidden_size, dtype="float64"),
        recipe=dict(loss="pinball" if quantiles else "mae"),
        walk_forward=dict(warmup_months=0),
    )
    if quantiles:
        core_document["predictor"]["head"] = "quantile_head_v1"
    parent_request = dict(
        variant="mac_online",
        seed=42,
        view_sha256=sha256(view),
        recipe_sha256=digest(core_document),
        search_case=None,
        window=window["id"],
    )
    model = (
        FinancialPredictor(
            FinancialConfig(
                source["spec"],
                variant="mac_online",
                seed=42,
                **{
                    key: value
                    for key, value in core_document["predictor"].items()
                    if key != "dtype"
                },
            ),
            dtype=torch.float64,
            device="cpu",
        )
        .eval()
        .requires_grad_(False)
    )
    state = values(model)
    state["head.bias"].fill_(0.125)
    model.load_state_dict(state)
    expected_core = values(model)
    parent_fit_identity = dict(kind="fixture_without_training", predictor=model.get_extra_state())
    core_path = save_training_state(
        parent / "fit/checkpoints",
        dict(global_step=0, model=values(model)),
        identity=parent_fit_identity,
        best=True,
    )
    core_reference = dict(path=str(core_path.relative_to(parent)), sha256=sha256(core_path))
    parent_fit = dict(
        schema_version=1,
        status="completed",
        final_test_opened=False,
        kind="titans_financial_chronological_run",
        run_id=digest(parent_fit_identity),
        identity=parent_fit_identity,
        best_checkpoint=dict(
            path=str(core_path.relative_to(parent / "fit")), sha256=sha256(core_path)
        ),
    )
    parent_fit_sha = write(parent / "fit/run.json", parent_fit)
    parent_identity = dict(
        recipe=core_document,
        request=parent_request,
        window=window,
        indices=dict(train=source["spec"].view_sha256),
    )
    if "phases" in source:
        parent_identity["phases"] = source["phases"]
    parent_report = dict(
        schema_version=1,
        kind="titans_walk_forward_window",
        status="completed",
        final_test_opened=False,
        request=parent_request,
        identity=parent_identity,
        run_id=digest(parent_identity),
        checkpoint=core_reference,
        predictions={},
        fit=dict(path="fit/run.json", sha256=parent_fit_sha, run_id=parent_fit["run_id"]),
    )
    parent_sha = write(parent / "run.json", parent_report)
    parent_link = dict(
        path=str(parent.resolve()),
        run_sha256=parent_sha,
        checkpoint_sha256=core_reference["sha256"],
    )
    components = dict(episodic_bank="m1")
    variant = _variant(model, parent_report, components)
    recipe_document = json.loads(
        Path("configs/titans/episodic-readout-historical-masked.json").read_text()
    )
    recipe_document["recipe"].update(
        loss="pinball" if quantiles else "mae",
        block_rows=4,
        neighbors=4,
        bank_capacity=source.get("bank_capacity", 16),
        max_working_bytes=32 * 1024**2,
        precision=None,
    )
    recipe = case_recipe(recipe_document, "lr1e-4")
    reader = _readout(variant, source["codec"], model, recipe, 42)
    reader.eval().requires_grad_(False)
    state = values(reader)
    state["step_logit"].fill_(0.375)
    reader.load_state_dict(state)
    expected_reader = values(reader)
    reader_digest = _parameters_digest(reader)
    fit_identity = dict(
        kind="fixture_without_training",
        dataset_sha256=sha256(view),
        admission="m1",
        parent=dict(parameters_sha256=model._parameter_id, predictor=model.get_extra_state()),
        readout=reader.get_extra_state(),
        recipe=recipe.identity(),
    )
    best = save_training_state(
        anchor / "fit/checkpoints",
        dict(
            global_step=0,
            model=values(reader),
            readout_sha256=reader_digest,
            optimizer=dict(unused_fixture=True),
            rng=dict(unused_fixture=True),
        ),
        identity=fit_identity,
        best=True,
    )
    best_reference = dict(
        path=str(best.relative_to(anchor)), sha256=sha256(best), readout_sha256=reader_digest
    )
    later = values(reader)
    later["step_logit"].fill_(0.875)
    # El contador ordena dos archivos de fixture y no corresponde a un paso de optimizador.
    save_training_state(
        anchor / "fit/checkpoints", dict(global_step=1, model=later), identity=fit_identity
    )
    fit = dict(
        schema_version=1,
        kind="mars_titan_readout_chronological_run",
        status="completed",
        final_test_opened=False,
        run_id=digest(fit_identity),
        identity=fit_identity,
        best_checkpoint=dict(best_reference, path=str(best.relative_to(anchor / "fit"))),
    )
    fit_sha = write(anchor / "fit/run.json", fit)
    request = dict(
        parent=parent_link,
        components=components,
        seed=42,
        search_case="lr1e-4",
        view_sha256=sha256(view),
        window=window["id"],
    )
    identity = dict(
        request=request,
        recipe=recipe_document,
        fit_run_id=fit["run_id"],
        variant=variant.identity(),
        variant_sha256=variant.fingerprint(),
        window=window,
    )
    if "phases" in source:
        identity["phases"] = source["phases"]
    selected = dict(
        schema_version=1,
        kind="mars_titan_selected_state",
        parent=parent_link,
        readout=best_reference,
        variant_sha256=variant.fingerprint(),
    )
    selected_sha = write(anchor / "selected.json", selected)
    report = dict(
        schema_version=1,
        kind="mars_titan_walk_forward_window",
        status="completed",
        final_test_opened=False,
        run_id=digest(identity),
        request=request,
        identity=identity,
        checkpoint=dict(path="selected.json", sha256=selected_sha),
        predictions={},
        fit=dict(
            path="fit/run.json",
            sha256=fit_sha,
            run_id=fit["run_id"],
            best_checkpoint=best_reference,
        ),
    )
    write(anchor / "run.json", report)
    return dict(
        anchor=anchor,
        view=view,
        source=source,
        core=expected_core,
        reader=expected_reader,
        report=report,
        selected=selected,
    )


def replace_selected_reader(selected, state):
    """Publicar otro archivo de fixture y enlazar sus descriptores sin alterar el padre."""
    anchor = selected["anchor"]
    fit_path = anchor / "fit/run.json"
    fit = json.loads(fit_path.read_text())
    state["global_step"] = 2
    path = save_training_state(
        anchor / "fit/checkpoints", state, identity=fit["identity"], best=True
    )
    reference = dict(
        path=str(path.relative_to(anchor)),
        sha256=sha256(path),
        readout_sha256=state.get("readout_sha256"),
    )
    fit["best_checkpoint"] = dict(reference, path=str(path.relative_to(anchor / "fit")))
    descriptor = copy.deepcopy(selected["selected"])
    descriptor["readout"] = reference
    report = copy.deepcopy(selected["report"])
    report["fit"]["best_checkpoint"] = reference
    report["fit"]["sha256"] = write(fit_path, fit)
    report["checkpoint"]["sha256"] = write(anchor / "selected.json", descriptor)
    write(anchor / "run.json", report)
