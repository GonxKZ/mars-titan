"""Titans-MAC dentro de la campaña con máscaras, en CPU y sin modificar pesos.

Se registra `titans_mac_online` en una campaña B sobre US con las vistas v2 del corpus
técnico. Sus ajustes y traslados usan los ejecutores reales con un optimizador que solo
registra gradientes. Los demás modelos son los dobles de `test_masked_campaign.py`. Se
comprueban trabajos, recibos, mismas filas, estado del ancla en los traslados, recibos
de ventana y el manifiesto de fuentes de la comparación.
"""

import json
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pyarrow.parquet as pq
import pytest
import torch

from mars_titan.data.input_policy import HISTORICAL_MASKED, STRICT_INPUTS
from mars_titan.data.storage import atomic_json, sha256
from mars_titan.environments.walk_forward_receipt import read_window_receipt
from mars_titan.training import mars_titan_walk_forward as mw
from mars_titan.training import masked_campaign as engine
from mars_titan.training import titans_walk_forward as wf
from mars_titan.training.campaign_plan import (
    TITANS,
    TITANS_SEARCHED,
    _titans,
    check_campaign,
    load_campaign,
    pending_families,
    plan_campaign,
)
from mars_titan.training.search_cases import SEARCHED, checked_predictor_changes
from tests.learning_hold_lowering import allowed_hold, lowers_learning_hold
from tests.training.test_masked_campaign import Recorder, doubles, write_campaign
from tests.training.test_titans_walk_forward import PT1, Factory, ShiftFirstTarget
from tests.training.test_walk_forward_v2_views import fixture

ARM = "titans_mac_online"
RECIPE = "configs/titans/chronological-training-historical-masked.json"
CASES = {"lr1e-4": 1e-4, "lr1e-3": 1e-3}


@pytest.fixture(scope="module")
def unfused():
    previous = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    yield
    torch.backends.mha.set_fastpath_enabled(previous)


@pytest.fixture(scope="module")
def permitted(tmp_path_factory):
    from mars_titan.training.learning_hold import HOLD_ENV

    path = allowed_hold(tmp_path_factory.mktemp("learning-hold") / "training-hold.json")
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv(HOLD_ENV, str(path))
        yield path


def titans_campaign(folder):
    """Campaña B reducida con un modelo Titans de una semilla y la receta de campaña reducida."""
    path = write_campaign(folder, arms=("gru", "ridge"))
    declared = json.loads((folder / "comparison.json").read_text())
    declared["arms"][ARM] = dict(family="titans_mac", output="quantile_head_v1", seeds=[42])
    declared["comparison"]["families"]["references_vs_zero"]["variants"].append(ARM)
    atomic_json(folder / "comparison.json", declared)
    recipe = json.loads(Path(RECIPE).read_text())
    recipe["predictor"].update(hidden_size=32, dtype="float64")
    recipe["recipe"].update(truncation=3, block_rows=2)
    atomic_json(folder / "titans.json", recipe)
    campaign = json.loads(path.read_text())
    campaign[TITANS] = dict(recipe="titans.json", arms={ARM: "mac_online"}, search_seed=42)
    atomic_json(path, campaign)
    return path


def executors(factory):
    result = doubles(Recorder())
    fit, carry = result[TITANS, "fit"], result[TITANS, "carry"]
    result[TITANS, "fit"] = dict(
        fit, run=partial(wf.titans_fit, device="cpu", optimizer_factory=factory)
    )
    result[TITANS, "carry"] = dict(carry, run=partial(wf.titans_carry, device="cpu"))
    return result


@pytest.fixture(scope="module")
def campaign_run(tmp_path_factory, unfused, permitted):
    root = tmp_path_factory.mktemp("titans-campaign")
    data = fixture(root / "data", ("US",))
    campaign = titans_campaign(root / "config")
    prepared = engine.prepare_views(campaign, data.parent, root / "views")
    views = {"US": root / "views" / "US"}
    factory = Factory()
    summary = engine.run_campaign(
        campaign,
        views,
        root / "out",
        executors=executors(factory),
        lease=nullcontext,
        stop=SimpleNamespace(requested=False),
    )
    return SimpleNamespace(
        root=root,
        campaign=campaign,
        views=views,
        output=root / "out",
        prepared=prepared,
        summary=summary,
        factory=factory,
        jobs=[job for job in plan_campaign(load_campaign(campaign)) if job["arm"] == ARM],
    )


def test_titans_jobs_follow_the_plan_with_the_same_rows_as_the_other_arms(campaign_run):
    summary, jobs = campaign_run.summary, campaign_run.jobs
    assert summary["status"] == "completed" and summary["completed"] == summary["planned"]
    # B sobre US: 7 ventanas reentrenadas con dos casos de búsqueda y 12 trasladadas.
    assert sum(job["kind"] == "fit" for job in jobs) == 7 * len(CASES)
    assert sum(job["kind"] == "carry" for job in jobs) == 12
    assert all(job["model"] == job["family"] == TITANS for job in jobs)
    assert campaign_run.factory.instances and all(
        item.calls == len(item.records) for item in campaign_run.factory.instances
    )
    windows = campaign_run.prepared["US"]["windows"]
    for job in jobs:
        receipt = json.loads(
            (campaign_run.output / "jobs" / job["id"] / "receipt.json").read_text()
        )
        gru = json.loads(
            (
                campaign_run.output
                / "jobs"
                / f"US/{job['window']}/gru"
                / ("search-gru-10" if job["kind"] == "fit" else "carry-s42")
                / "receipt.json"
            ).read_text()
        )
        if job["kind"] == "fit":
            # Cada búsqueda ajusta con la tasa de su caso y la declara en la petición.
            attempt = campaign_run.output / receipt["attempt"]
            report = json.loads((attempt / "run.json").read_text())
            fit = json.loads((attempt / "fit/run.json").read_text())
            assert report["request"]["search_case"] == job["candidate"]
            assert job["case"]["search_case"] == job["candidate"]
            assert fit["identity"]["recipe"]["learning_rate"] == CASES[job["candidate"]]
        for partition in ("calibration", "evaluation"):
            record = receipt["predictions"][partition]
            assert record["rows"] == windows[job["window"]]["counts"][partition]
            assert record["rows_sha256"] == gru["predictions"][partition]["rows_sha256"]
            table = pq.read_table(campaign_run.output / record["path"])
            assert {"quantile_0025", "quantile_0975"} <= set(table.column_names)
            assert table["prediction_at"].cast("int64").to_numpy().max() < wf.FINAL_TEST_US


def test_carried_windows_use_the_anchor_state_and_declare_their_memory_policy(campaign_run):
    output = campaign_run.output
    for job in campaign_run.jobs:
        receipt = json.loads((output / "jobs" / job["id"] / "receipt.json").read_text())
        if job["kind"] == "fit":
            report = json.loads((output / receipt["report"]["path"]).read_text())
            assert receipt["parent"] == dict(id=job["id"], sha256=report["checkpoint"]["sha256"])
            continue
        name, anchor = selected_search(output, job["anchor"])
        carry = json.loads((output / receipt["report"]["path"]).read_text())
        assert receipt["parent"] == anchor["parent"]
        assert carry["anchor"]["checkpoint_sha256"] == anchor["parent"]["sha256"]
        assert carry["anchor"]["search_case"] == name.removeprefix("search-")
        assert carry["months_since_anchor_information"] > 0
        policy = carry["memory_policy"]
        assert policy["name"] == wf.MEMORY_POLICY and policy["warmup_months"] == 12
        assert policy["fast_state"] == "never_transferred_from_the_anchor_reset_at_each_pass"
        assert carry["fold"]["id"] == job["window"] and carry["final_test_opened"] is False


def test_window_receipts_and_sources_include_the_titans_arm(campaign_run):
    output = campaign_run.output
    for job in campaign_run.jobs:
        path = output / "windows/US" / job["window"] / ARM / "seed-42" / "US.json"
        record = json.loads(path.read_text())
        checked = read_window_receipt(record)
        assert checked.fold == job["window"]
        # Una ventana reentrenada publica el caso de búsqueda elegido, no los dos.
        if job["kind"] == "fit":
            name, receipt = selected_search(output, job["window"])
            assert record["parent"]["id"].endswith(f"/{name}")
        else:
            receipt = json.loads((output / "jobs" / job["id"] / "receipt.json").read_text())
        assert record["parent"] == receipt["parent"]
    destination = engine.write_sources(campaign_run.campaign, campaign_run.views, output, "US")
    manifest = json.loads(destination.read_text())
    entries = manifest["arms"][ARM]["42"]
    assert len(entries) == 19
    assert all({"calibration", "evaluation"} <= set(entry) for entry in entries.values())


def test_check_lists_titans_as_connected_only_when_the_campaign_declares_it(tmp_path):
    path = titans_campaign(tmp_path / "config")
    checked = check_campaign(path)
    assert "titans_mac" not in checked["pending_families"]
    assert checked["counts"]["scopes"]["US"]["arms"][ARM] == {"42": dict(fit=14, carry=12)}
    jobs = [job for job in plan_campaign(load_campaign(path)) if job["arm"] == ARM]
    assert {(job["family"], job["model"]) for job in jobs} == {(TITANS, TITANS)}
    plain = load_campaign(path) | {TITANS: None}
    assert pending_families(plain)["titans_mac"]["arms"] == [ARM]


EXTENSION = "titans_mac_online_pt1"


def extended_campaign(folder, change=None):
    """Campaña reducida con PT1 como extensión del núcleo de `titans_mac_online`."""
    path = titans_campaign(folder)
    declared = json.loads((folder / "comparison.json").read_text())
    declared["arms"][EXTENSION] = dict(declared["arms"][ARM])
    declared["comparison"]["families"]["memory_stability"] = dict(
        kind="delta", base=ARM, variants=[EXTENSION]
    )
    campaign = json.loads(path.read_text())
    campaign[TITANS]["core_extensions"] = {EXTENSION: dict(base_arm=ARM, predictor_changes=PT1)}
    if change is not None:
        change(campaign, declared)
    atomic_json(folder / "comparison.json", declared)
    atomic_json(path, campaign)
    return path


def test_core_extensions_repeat_the_base_model_and_add_only_their_predictor_changes(tmp_path):
    path = extended_campaign(tmp_path / "config")
    loaded = load_campaign(path)
    candidates = loaded[TITANS]["candidates"]
    assert [case for case, _ in candidates[EXTENSION]] == [case for case, _ in candidates[ARM]]
    assert [value for _, value in candidates[EXTENSION]] == [
        dict(value, predictor_changes=PT1) for _, value in candidates[ARM]
    ]
    counts = check_campaign(path)["counts"]["scopes"]["US"]["arms"]
    assert counts[EXTENSION] == counts[ARM] == {"42": dict(fit=14, carry=12)}
    jobs = [job for job in plan_campaign(loaded) if job["arm"] == EXTENSION]
    assert {(job["family"], job["model"]) for job in jobs} == {(TITANS, TITANS)}
    fits = [job for job in jobs if job["kind"] == "fit"]
    assert fits and all(job["case"]["predictor_changes"] == PT1 for job in fits)
    run = SimpleNamespace(job=fits[0], case=fits[0]["case"], policy="historical_masked_2000_v1")
    assert wf._campaign_case(run) == fits[0]["case"]


def _base(campaign, declared):
    campaign[TITANS]["core_extensions"][EXTENSION]["base_arm"] = "titans_mac_frozen"


def _seeds(campaign, declared):
    declared["arms"][EXTENSION]["seeds"] = [42, 43]


def _foreign(campaign, declared):
    campaign[TITANS]["core_extensions"][EXTENSION]["predictor_changes"] = dict(
        hidden_size=dict(value=64)
    )


def _undeclared(campaign, declared):
    del campaign[TITANS]["core_extensions"]


def _empty(campaign, declared):
    campaign[TITANS]["core_extensions"] = {}


def _absent(campaign, declared):
    del declared["arms"][EXTENSION]
    del declared["comparison"]["families"]["memory_stability"]


def _repeated(campaign, declared):
    campaign[TITANS]["arms"][EXTENSION] = "mac_frozen"


def _fields(campaign, declared):
    campaign[TITANS]["core_extensions"][EXTENSION]["variant"] = "mac_online"


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (_base, "debe extender un modelo de la sección"),
        (_seeds, "con sus mismas semillas"),
        (_foreign, "gate_bias y memory_stability"),
        (_undeclared, "o una extensión declarada"),
        (_empty, "o una extensión declarada"),
        (_absent, "o una extensión declarada"),
        (_repeated, "o una extensión declarada"),
        (_fields, "debe extender un modelo de la sección"),
    ],
    ids=["base", "seeds", "foreign", "undeclared", "empty", "absent", "repeated", "fields"],
)
def test_core_extensions_reject_declarations_that_change_more_than_the_extension(
    tmp_path, change, message
):
    with pytest.raises(ValueError, match=message):
        load_campaign(extended_campaign(tmp_path / "config", change))


@lowers_learning_hold(
    "tests/training/test_titans_campaign.py::"
    "test_an_extension_fit_builds_its_changes_and_its_carry_rebuilds_them"
)
def test_an_extension_fit_builds_its_changes_and_its_carry_rebuilds_them(campaign_run, tmp_path):
    """El ejecutor de ajuste pasa los cambios del caso al predictor y el de traslado los
    reconstruye desde la petición del ancla. Sin ellos, el estado del ancla no
    correspondería a la arquitectura del predictor trasladado."""
    carry = next(job for job in campaign_run.jobs if job["kind"] == "carry")
    job = next(
        job
        for job in campaign_run.jobs
        if job["kind"] == "fit" and job["window"] == carry["anchor"]
    )
    windows = campaign_run.prepared["US"]["windows"]
    anchor, later = windows[job["window"]], windows[carry["window"]]
    run = SimpleNamespace(
        job=job,
        case=dict(job["case"], predictor_changes=PT1),
        view=Path(anchor["path"]),
        view_sha256=anchor["sha256"],
        folder=tmp_path / "fit",
        policy="historical_masked_2000_v1",
        stop=SimpleNamespace(requested=False),
        joint_epoch=None,
    )
    report = wf.titans_fit(run, device="cpu", optimizer_factory=Factory())
    assert report["status"] == "completed" and report["request"]["predictor_changes"] == PT1
    fit = json.loads((tmp_path / "fit" / "fit/run.json").read_text())["identity"]
    assert fit["predictor"]["memory_stability"] == PT1["memory_stability"]
    carried = wf.titans_carry(
        SimpleNamespace(
            anchor=dict(folder=tmp_path / "fit", view=anchor["path"]),
            view=later["path"],
            folder=tmp_path / "carry",
            stop=None,
        ),
        device="cpu",
    )
    assert carried["status"] == "completed"
    assert carried["anchor"]["run_sha256"] == sha256(tmp_path / "fit" / "run.json")


@pytest.mark.parametrize(
    "changes",
    [{}, dict(hidden_size=dict(value=32)), dict(memory_stability={}), dict(gate_bias=0.15)],
    ids=["empty", "foreign", "empty-option", "scalar"],
)
def test_predictor_changes_outside_the_extensions_are_rejected(changes):
    # La ventana Titans-MAC los comprueba justo después de la protección y antes de leer nada.
    assert checked_predictor_changes(None) is None
    assert checked_predictor_changes(PT1) == PT1
    with pytest.raises(ValueError, match="gate_bias y memory_stability"):
        checked_predictor_changes(changes)


def test_a_core_with_predictor_changes_cannot_be_the_mars_titan_parent(tmp_path):
    """Los lectores y B6 reconstruyen el padre con la receta común, así que PT1 no es padre."""
    view = tmp_path / "view.json"
    view.write_text("{}")
    report = dict(
        kind=mw.TITANS_KIND,
        status="completed",
        final_test_opened=False,
        checkpoint={},
        request=dict(variant="mac_online", seed=42, view_sha256=sha256(view)),
        predictions={},
    )
    atomic_json(tmp_path / "plain" / "run.json", report)
    report["request"]["predictor_changes"] = PT1
    atomic_json(tmp_path / "pt1" / "run.json", report)
    assert mw._parent(tmp_path / "plain", view, 42)[0]["request"].get("predictor_changes") is None
    with pytest.raises(ValueError, match="sin cambios del predictor"):
        mw._parent(tmp_path / "pt1", view, 42)


def test_the_frozen_parent_of_the_chain_rebuilds_the_predictor_changes_of_its_request():
    """El padre congelado de la cadena trivial de PT1 se construye con la caja y el recorte.

    PT1 no añade parámetros, así que solo el estado extra del predictor, que guarda su
    configuración, distingue sus pesos de los del núcleo. La etapa de adaptadores reconstruye
    el padre desde la petición de su ventana, y aquí se hace igual con el estado de un
    predictor recién creado, sin ajustar nada.
    """
    from mars_titan.models.titans.financial import FinancialConfig, FinancialPredictor
    from mars_titan.posttraining.chronological_windows import _predictor
    from tests.models.titans.test_financial_adapter import specification

    document = json.loads(Path(RECIPE).read_text())
    document["predictor"].update(hidden_size=32)
    spec = specification()
    request = dict(variant="mac_online", seed=42, predictor_changes=PT1)
    config = FinancialConfig(
        spec, variant="mac_online", seed=42, **wf.predictor_options(document, request)
    )
    state = FinancialPredictor(config, device="cpu", dtype=torch.float32).state_dict()
    rebuilt = _predictor(document, spec, request, torch.device("cpu"), state)
    assert rebuilt.config.identity() == config.identity()
    assert rebuilt.config.identity()["memory_stability"] == PT1["memory_stability"]
    # Sin los cambios de la petición, el predictor sería el núcleo sin PT1 y rechaza el estado.
    plain = dict(variant="mac_online", seed=42)
    with pytest.raises(ValueError, match="no coincide con el contrato"):
        _predictor(document, spec, plain, torch.device("cpu"), state)


def test_campaign_rejects_a_titans_recipe_without_the_protocol_rule(tmp_path):
    folder = tmp_path / "config"
    path = titans_campaign(folder)
    recipe = json.loads((folder / "titans.json").read_text())
    atomic_json(folder / "titans.json", recipe | dict(recipe=recipe["recipe"] | dict(epochs=2)))
    with pytest.raises(ValueError, match="regla de parada"):
        load_campaign(path)
    atomic_json(folder / "titans.json", recipe)
    declared = json.loads((folder / "comparison.json").read_text())
    declared["arms"]["titans_mac_frozen"] = declared["arms"][ARM]
    atomic_json(folder / "comparison.json", declared)
    campaign = json.loads(path.read_text())
    campaign[TITANS]["arms"] = {ARM: "mac_online", "titans_mac_frozen": "mac_online"}
    atomic_json(path, campaign)
    with pytest.raises(ValueError, match="variante distinta"):
        load_campaign(path)
    campaign[TITANS]["arms"] = {ARM: "mac_online", "titans_mac_frozen": "mac_frozen"}
    atomic_json(path, campaign)
    assert {
        arm: case[0][1]["variant"]
        for arm, case in load_campaign(path)[TITANS]["candidates"].items()
    } == {ARM: "mac_online", "titans_mac_frozen": "mac_frozen"}


def test_campaign_plan_repeats_the_searched_hyperparameters():
    # El plan, Titans-MAC, el lector y la GRU candidata buscan los mismos hiperparámetros.
    assert TITANS_SEARCHED == SEARCHED == ("learning_rate", "max_grad_norm")


def test_titans_section_requires_the_masked_input_policy(tmp_path):
    path = titans_campaign(tmp_path / "config")
    campaign = load_campaign(path)
    section = json.loads(path.read_text())[TITANS]
    arms, rule = campaign["comparison_config"]["arms"], campaign["rule"]
    count = len(campaign["neural"]["case_indices"])
    assert _titans(section, arms, rule, HISTORICAL_MASKED, path.parent, count) == campaign[TITANS]
    with pytest.raises(ValueError, match="política con máscaras"):
        _titans(section, arms, rule, STRICT_INPUTS, path.parent, count)


def test_fit_executor_returns_the_completed_window_only_for_its_view(campaign_run):
    job = next(job for job in campaign_run.jobs if job["kind"] == "fit")
    receipt = json.loads((campaign_run.output / "jobs" / job["id"] / "receipt.json").read_text())
    view = campaign_run.prepared["US"]["windows"][job["window"]]
    run = SimpleNamespace(
        job=job,
        case=job["case"],
        view=Path(view["path"]),
        view_sha256=view["sha256"],
        folder=campaign_run.output / receipt["attempt"],
        policy="historical_masked_2000_v1",
        stop=SimpleNamespace(requested=False),
        joint_epoch=None,
    )
    report = wf.titans_fit(run, device="cpu")
    assert report["status"] == "completed"
    assert report["checkpoint"]["sha256"] == receipt["parent"]["sha256"]
    with pytest.raises(ValueError, match="no confirma la vista"):
        wf.titans_fit(SimpleNamespace(**vars(run) | dict(view_sha256="0" * 64)), device="cpu")


def test_executors_declare_unfused_attention_only_while_their_job_runs(monkeypatch, tmp_path):
    """La campaña no fija fastpath para el proceso. Cada trabajo de Titans lo declara."""
    seen = []

    def window(*_args, **_kwargs):
        seen.append(torch.backends.mha.get_fastpath_enabled())
        return dict(status="completed", request=dict(view_sha256="v"))

    def carried(*_args, **_kwargs):
        seen.append(torch.backends.mha.get_fastpath_enabled())
        raise wf.Paused

    case = dict(recipe="r", variant="mac_online", seed=42, search_case="lr1e-4")
    monkeypatch.setattr(wf, "_campaign_case", lambda run: case)
    monkeypatch.setattr(wf, "view_protocol", lambda view: {})
    monkeypatch.setattr(wf, "run_titans_window", window)
    monkeypatch.setattr(wf, "carry_titans", carried)
    run = SimpleNamespace(
        view="v",
        view_sha256="v",
        job=dict(window="fold-000"),
        folder=tmp_path,
        stop=None,
        anchor=dict(folder=tmp_path, view="a"),
        joint_epoch=None,
    )
    previous = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(True)
    try:
        assert wf.titans_fit(run, device="cpu")["status"] == "completed"
        assert torch.backends.mha.get_fastpath_enabled() is True
        with pytest.raises(engine.Paused):
            wf.titans_carry(run, device="cpu")
        assert torch.backends.mha.get_fastpath_enabled() is True
    finally:
        torch.backends.mha.set_fastpath_enabled(previous)
    assert seen == [False, False]


def selected_search(output, window):
    """Búsqueda elegida en una ventana: menor MAE de validación y, si empatan, el nombre."""
    receipts = {
        path.parent.name: json.loads(path.read_text())
        for path in (output / "jobs" / f"US/{window}/{ARM}").glob("search-*/receipt.json")
    }
    assert len(receipts) == len(CASES)
    name = min(receipts, key=lambda key: (receipts[key]["score"], key))
    return name, receipts[name]


def carry_job(campaign_run):
    job = next(job for job in campaign_run.jobs if job["kind"] == "carry")
    _, receipt = selected_search(campaign_run.output, job["anchor"])
    windows = campaign_run.prepared["US"]["windows"]
    return (
        campaign_run.output / receipt["attempt"],
        windows[job["anchor"]]["path"],
        windows[job["window"]]["path"],
    )


def test_carry_repeats_bit_for_bit_and_rejects_incoherent_anchors(campaign_run, tmp_path):
    anchor, anchor_view, view = carry_job(campaign_run)
    first = wf.carry_titans(anchor, anchor_view, view, tmp_path / "first", device="cpu")
    again = wf.carry_titans(anchor, anchor_view, view, tmp_path / "again", device="cpu")
    for name in wf.CARRIED:
        assert first["predictions"][name]["sha256"] == again["predictions"][name]["sha256"]
    with pytest.raises(ValueError, match="directorio nuevo"):
        wf.carry_titans(anchor, anchor_view, view, tmp_path / "first", device="cpu")
    with pytest.raises(ValueError, match="no es la de su ajuste"):
        wf.carry_titans(anchor, view, view, tmp_path / "other-view", device="cpu")
    with pytest.raises(ValueError, match="posterior a la información del ancla"):
        wf.carry_titans(anchor, anchor_view, anchor_view, tmp_path / "same", device="cpu")


def test_carried_parameters_only_admit_another_view_of_the_same_input(campaign_run, tmp_path):
    anchor, _, _ = carry_job(campaign_run)
    report = json.loads((anchor / "run.json").read_text())
    payload = torch.load(
        anchor / report["checkpoint"]["path"], map_location="cpu", weights_only=True
    )
    model = payload["state"]["model"]
    extra = model["_extra_state"]
    configuration = extra["configuration"]

    class Target:
        def __init__(self, changes):
            self.loaded = None
            self.extra = dict(extra, configuration=configuration | changes)

        def get_extra_state(self):
            return self.extra

        def load_state_dict(self, value):
            self.loaded = value

    inputs = dict(configuration["inputs"], source_sha256="1" * 64, view_sha256="2" * 64)
    target = Target(dict(inputs=inputs))
    wf._carried_parameters(model, target)
    assert target.loaded["_extra_state"]["configuration"]["inputs"] == inputs
    for changes in (dict(variant="mac_frozen"), dict(hidden_size=64), dict(seed=43)):
        with pytest.raises(ValueError, match="no corresponde"):
            wf._carried_parameters(model, Target(changes))


def test_campaign_case_must_keep_the_planned_recipe(tmp_path):
    job = dict(seed=42)
    case = load_campaign(titans_campaign(tmp_path / "config"))[TITANS]["candidates"][ARM][0][1]
    run = SimpleNamespace(job=job, case=case, policy="historical_masked_2000_v1")
    assert wf._campaign_case(run) == case
    with pytest.raises(ValueError, match="caso"):
        wf._campaign_case(SimpleNamespace(job=job, case=case | dict(seed=43), policy=run.policy))
    changed = tmp_path / "titans.json"
    changed.write_text(Path(case["recipe"]).read_text() + "\n")
    with pytest.raises(ValueError, match="cambió"):
        wf._campaign_case(
            SimpleNamespace(job=job, case=case | dict(recipe=str(changed)), policy=run.policy)
        )
    assert sha256(changed) != case["recipe_sha256"]


def test_campaign_requires_the_quantiles_of_the_titans_arm(campaign_run, tmp_path):
    # Los dobles escriben cuantiles solo para las referencias neuronales.
    with pytest.raises(ValueError, match="Faltan columnas"):
        engine.run_campaign(
            campaign_run.campaign,
            campaign_run.views,
            tmp_path / "out",
            executors=doubles(Recorder()),
            lease=nullcontext,
            stop=SimpleNamespace(requested=False),
        )


class RepeatFirstRow:
    """Destino que entrega dos veces la primera fila resuelta."""

    def __init__(self, rows):
        self.rows, self.repeated = rows, False

    def append(self, record):
        if not self.repeated:
            self.rows.append(record)
            self.repeated = True
        self.rows.append(record)


@pytest.mark.parametrize(
    ("sink", "message"),
    [(RepeatFirstRow, "no concilian"), (ShiftFirstTarget, "1 filas difieren de la vista")],
    ids=["repeated", "target"],
)
def test_carry_reconciles_its_rows_before_writing(
    campaign_run, tmp_path, monkeypatch, sink, message
):
    from mars_titan.training.financial_run import ChronologicalInference

    predict = ChronologicalInference.predict
    monkeypatch.setattr(
        ChronologicalInference,
        "predict",
        lambda self, source, rows, **options: predict(self, source, sink(rows), **options),
    )
    anchor, anchor_view, view = carry_job(campaign_run)
    output = tmp_path / "carry"
    with pytest.raises(ValueError, match=message):
        wf.carry_titans(anchor, anchor_view, view, output, device="cpu")
    assert not list(output.glob("*.parquet")) and not (output / "carry.json").exists()
