"""Titans-MAC dentro de la campaña con máscaras, en CPU y sin modificar pesos.

Se registra `titans_mac_online` en una campaña B sobre US con las vistas v2 del corpus
técnico. Sus ajustes y traslados usan los ejecutores reales con un optimizador que solo
registra gradientes. Los demás brazos son los dobles de `test_masked_campaign.py`. Se
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
from mars_titan.training import masked_campaign as engine
from mars_titan.training import titans_walk_forward as wf
from mars_titan.training.campaign_plan import (
    TITANS,
    _titans,
    check_campaign,
    load_campaign,
    pending_families,
    plan_campaign,
)
from tests.training.test_masked_campaign import Recorder, doubles, write_campaign
from tests.training.test_titans_walk_forward import Factory, ShiftFirstTarget
from tests.training.test_walk_forward_v2_views import fixture

ARM = "titans_mac_online"


@pytest.fixture(scope="module")
def unfused():
    previous = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    yield
    torch.backends.mha.set_fastpath_enabled(previous)


@pytest.fixture(scope="module")
def permitted(tmp_path_factory):
    from mars_titan.training.learning_hold import HOLD_ENV

    path = tmp_path_factory.mktemp("learning-hold") / "training-hold.json"
    path.write_text(json.dumps({"training_allowed": True}), encoding="utf-8")
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv(HOLD_ENV, str(path))
        yield path


def titans_campaign(folder):
    """Campaña B reducida con un brazo Titans de una semilla y una receta técnica."""
    path = write_campaign(folder, arms=("gru", "ridge"))
    declared = json.loads((folder / "comparison.json").read_text())
    declared["arms"][ARM] = dict(family="titans_mac", output="quantile_head_v1", seeds=[42])
    declared["comparison"]["families"]["references_vs_zero"]["variants"].append(ARM)
    atomic_json(folder / "comparison.json", declared)
    recipe = json.loads(Path("configs/titans/chronological-training-quantile.json").read_text())
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
    # B sobre US: 7 ventanas reentrenadas con un ajuste y 12 trasladadas.
    assert sum(job["kind"] == "fit" for job in jobs) == 7
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
        anchor = json.loads(
            (output / "jobs" / f"US/{job['anchor']}/{ARM}/search-recipe/receipt.json").read_text()
        )
        carry = json.loads((output / receipt["report"]["path"]).read_text())
        assert receipt["parent"] == anchor["parent"]
        assert carry["anchor"]["checkpoint_sha256"] == anchor["parent"]["sha256"]
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
    assert checked["counts"]["scopes"]["US"]["arms"][ARM] == {"42": dict(fit=7, carry=12)}
    jobs = [job for job in plan_campaign(load_campaign(path)) if job["arm"] == ARM]
    assert {(job["family"], job["model"]) for job in jobs} == {(TITANS, TITANS)}
    plain = load_campaign(path) | {TITANS: None}
    assert pending_families(plain)["titans_mac"]["arms"] == [ARM]


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


def test_titans_section_requires_the_masked_input_policy(tmp_path):
    path = titans_campaign(tmp_path / "config")
    campaign = load_campaign(path)
    section = json.loads(path.read_text())[TITANS]
    arms, rule = campaign["comparison_config"]["arms"], campaign["rule"]
    assert _titans(section, arms, rule, HISTORICAL_MASKED, path.parent) == campaign[TITANS]
    with pytest.raises(ValueError, match="política con máscaras"):
        _titans(section, arms, rule, STRICT_INPUTS, path.parent)


def test_fit_executor_returns_the_completed_window_only_for_its_view(campaign_run):
    job = next(job for job in campaign_run.jobs if job["kind"] == "fit")
    receipt = json.loads((campaign_run.output / "jobs" / job["id"] / "receipt.json").read_text())
    view = campaign_run.prepared["US"]["windows"][job["window"]]
    case = load_campaign(campaign_run.campaign)[TITANS]["candidates"][ARM][0][1]
    run = SimpleNamespace(
        job=job,
        case=case,
        view=Path(view["path"]),
        view_sha256=view["sha256"],
        folder=campaign_run.output / receipt["attempt"],
        policy="historical_masked_2000_v1",
        stop=SimpleNamespace(requested=False),
    )
    report = wf.titans_fit(run, device="cpu")
    assert report["status"] == "completed"
    assert report["checkpoint"]["sha256"] == receipt["parent"]["sha256"]
    with pytest.raises(ValueError, match="no confirma la vista"):
        wf.titans_fit(SimpleNamespace(**vars(run) | dict(view_sha256="0" * 64)), device="cpu")


def carry_job(campaign_run):
    job = next(job for job in campaign_run.jobs if job["kind"] == "carry")
    receipt = json.loads(
        (
            campaign_run.output / "jobs" / f"US/{job['anchor']}/{ARM}/search-recipe/receipt.json"
        ).read_text()
    )
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
