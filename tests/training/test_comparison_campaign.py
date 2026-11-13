"""Motor de la campaña de comparación (#836): declaración, orden, cola y permiso.

Ninguna prueba ajusta un modelo. Los trabajos de la cola se sustituyen por un proceso que
solo escribe su registro, y `execute_job` por funciones que devuelven un informe.
"""

import copy
import json
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from mars_titan.training import comparison_campaign as campaign
from mars_titan.training.comparison_campaign import DONE, FAILED, PAUSED, PENDING, Job, Layout
from mars_titan.training.learning_hold import HOLD_ENV, learning_blocked
from mars_titan.training.window_indices import SHARED_INDICES_ENV
from tests.training.comparison_tables import declared, write_tables

RECIPE = "configs/titans/chronological-training-historical-sequence-v1.json"
RULE = dict(metric="session_mae", stopping="fixed_budget", patience=5, min_delta=0.0, max_epochs=2)
READER = "configs/titans/episodic-readout-historical-masked.json"
RIDGE = dict(
    machine="references",
    tier="core",
    runner="ridge",
    config=dict(identity="ridge_full_history_v1", alphas=[0.1, 1.0], batch_size=1024),
)


def declaration(**changes):
    document = dict(
        schema_version=1,
        kind=campaign.KIND,
        final_test_access=False,
        market="US",
        cohort=dict(size=16),
        origins=[
            dict(year=2019, fold="fold-014", role="development"),
            dict(year=2023, fold="fold-018", role="confirmation"),
        ],
        seeds=[42, 43],
        stopping=dict(fit=dict(RULE, initial_state="reported"), reader=dict(RULE)),
        execution=dict(
            pipeline=dict(decode_workers=2, prefetch_batches=2, group_cache_mib=512),
            workers=dict(references=1, mars_titan=2),
        ),
        recipes=dict(core=RECIPE, reader=READER),
        tiers=["core", "areas"],
        models=dict(
            transformer=dict(
                machine="references",
                tier="core",
                runner="fit",
                recipe="core",
                variant="transformer_direct",
            ),
            gru=dict(
                machine="references",
                tier="core",
                runner="fit",
                recipe="core",
                variant="transformer_direct",
                predictor_changes=dict(reference_encoder="gru"),
            ),
            rule=dict(
                machine="references",
                tier="core",
                runner="rule",
                config=dict(identity="momentum_reversal_v1"),
            ),
            mars_titan_m1=dict(
                machine="mars_titan",
                tier="core",
                runner="reader",
                recipe="reader",
                parent="titans_mac",
                components=dict(episodic_bank="m1"),
                search_case="lr1e-3",
            ),
            titans_mac=dict(
                machine="mars_titan", tier="core", runner="fit", recipe="core", variant="mac_online"
            ),
            area_network=dict(
                machine="mars_titan",
                tier="areas",
                runner="fit",
                recipe="core",
                variant="mac_online",
                predictor_changes=dict(holographic_network=dict(areas=19)),
            ),
        ),
    )
    document.update(changes)
    return document


def layout(tmp_path):
    return Layout(tmp_path / "state", tmp_path / "views")


def test_declared_campaign_of_the_repository_is_valid_and_splits_the_two_machines():
    document, digest = campaign.load_declaration()
    assert len(digest) == 64 and document["final_test_access"] is False
    machines = {model["machine"] for model in document["models"].values()}
    assert machines == set(campaign.MACHINES)
    for name, model in document["models"].items():
        mars = name.startswith(("titans_", "mars_titan", "area_", "asset_", "cross_asset"))
        assert (model["machine"] == "mars_titan") is mars, name
    # El segundo equipo solo ajusta Titans-MAC, MARS-TITAN y sus variantes.
    references = {
        name for name, model in document["models"].items() if model["machine"] == "references"
    }
    assert references == {
        "transformer",
        "gru",
        "lstm",
        "rnn",
        "dlinear",
        "ridge",
        "momentum_reversal",
        "lightgbm",
    }
    for machine in campaign.MACHINES:
        jobs = campaign.plan(document, machine)
        keys = [job.key for job in jobs]
        assert len(keys) == len(set(keys)) and jobs
        # Ningún lector aparece antes que su padre.
        for position, job in enumerate(jobs):
            assert job.needs is None or job.needs in keys[:position]
        # Los años añadidos empiezan cuando la primera etapa ha recorrido todos sus niveles.
        stages = [job.history for job in jobs]
        assert stages == sorted(stages) and True in stages
    # 2018 fue el origen del piloto que eligió la receta y 2024 es el test final cerrado.
    roles = {origin["year"]: origin["role"] for origin in document["origins"]}
    assert roles == {
        2019: "development",
        2020: "development",
        2021: "development",
        2022: "development",
        2023: "confirmation",
        2017: "history",
        2016: "history",
        2015: "history",
    }
    assert document["seeds"] == [42, 43, 44] and document["cohort"] == dict(size=1024)


def test_every_declared_model_passes_the_checks_of_its_runner():
    """La declaración real se comprueba con los validadores que usará cada trabajo."""
    from mars_titan.memory.mars_titan_variant import check_components
    from mars_titan.memory.mars_titan_variant import load_declaration as load_components
    from mars_titan.models.baselines import momentum_reversal, same_inputs_lightgbm
    from mars_titan.training import comparison_ridge, mars_titan_run
    from mars_titan.training.financial_run import load_recipe
    from mars_titan.training.search_cases import checked_predictor_changes
    from mars_titan.training.titans_walk_forward import VARIANTS

    document, _ = campaign.load_declaration()
    recipes = {name: campaign.ROOT / path for name, path in document["recipes"].items()}
    _, core = load_recipe(recipes["core"])
    # La receta común centra y escala las entradas con las filas de ajuste de cada ventana.
    assert core["predictor"]["learning_scale"] == dict(
        clip=5.0, price_returns=True, zero_median=True
    )
    assert "search_cases" not in core["walk_forward"] and "fit_months" not in core["walk_forward"]
    reader = mars_titan_run.load_recipe(recipes["reader"])
    assert list(reader["walk_forward"]["search_cases"]) == ["lr1e-3"]
    components = load_components()
    fits = stopping = 0
    for name, model in document["models"].items():
        runner = model["runner"]
        if runner == "fit":
            assert model["variant"] in VARIANTS and model["recipe"] == "core", name
            checked_predictor_changes(model.get("predictor_changes"))
            fits += 1
        elif runner == "reader":
            assert model["recipe"] == "reader" and model["search_case"] == "lr1e-3", name
            checked, correction = check_components(components, dict(model["components"]))
            assert correction is None and "episodic_bank" in checked, name
        elif runner == "ridge":
            comparison_ridge.checked_config(model["config"])
        elif runner == "rule":
            momentum_reversal.checked_config(model["config"])
        else:
            same_inputs_lightgbm.checked_config(model["config"])
        if runner in campaign.GPU_RUNNERS:
            rule = campaign.job_rule(document, model)
            stopping += rule == document["stopping"][runner]
    # Mismo presupuesto y misma regla para todos los modelos de un ejecutor.
    gpu = sum(model["runner"] in campaign.GPU_RUNNERS for model in document["models"].values())
    assert stopping == gpu and fits == 16
    assert document["stopping"]["fit"]["initial_state"] == "reported"
    assert "initial_state" not in document["stopping"]["reader"]
    assert {rule["min_delta"] for rule in document["stopping"].values()} == {0.0}
    assert {rule["stopping"] for rule in document["stopping"].values()} == {"fixed_budget"}


def test_declared_readers_can_replay_a_full_batch_with_a_full_bank():
    from mars_titan.models.titans.episodic_readout import EpisodicReadout, EpisodicReadoutConfig
    from mars_titan.models.titans.evidence_bridge import EvidenceBridgeConfig

    document, _ = campaign.load_declaration()
    core = json.loads((campaign.ROOT / document["recipes"]["core"]).read_text())
    recipe = json.loads((campaign.ROOT / document["recipes"]["reader"]).read_text())["recipe"]
    options = dict(
        codec_id="a" * 64,
        hidden_size=core["predictor"]["hidden_size"],
        max_batch=recipe["block_rows"],
        max_working_bytes=recipe["max_working_bytes"],
        neighbors=recipe["neighbors"],
        temperature=recipe["temperature"],
    )
    for name, model in document["models"].items():
        if model["runner"] != "reader":
            continue
        components = model["components"]
        bridge = None
        if "evidence_bridge" in components:
            geometry = dict(components["evidence_bridge"])
            geometry["regions"] = tuple(geometry["regions"])
            bridge = EvidenceBridgeConfig(**options, **geometry)
        config = EpisodicReadoutConfig(
            **options, evidence_bridge=bridge, refinements=components.get("refinements", 1)
        )
        required = EpisodicReadout(config).estimated_bytes(
            recipe["block_rows"], recipe["bank_capacity"], differentiable=True
        )
        assert required <= recipe["max_working_bytes"], (name, required)


def test_plan_orders_by_tier_seed_origin_and_model_with_parents_before_readers():
    document = campaign.checked_declaration(declaration())
    jobs = campaign.plan(document, "mars_titan")
    assert [job.key for job in jobs] == [
        "titans_mac/2019/seed-42",
        "mars_titan_m1/2019/seed-42",
        "titans_mac/2023/seed-42",
        "mars_titan_m1/2023/seed-42",
        "titans_mac/2019/seed-43",
        "mars_titan_m1/2019/seed-43",
        "titans_mac/2023/seed-43",
        "mars_titan_m1/2023/seed-43",
        "area_network/2019/seed-42",
        "area_network/2023/seed-42",
        "area_network/2019/seed-43",
        "area_network/2023/seed-43",
    ]
    reader = jobs[1]
    assert reader.needs == "titans_mac/2019/seed-42" and reader.runner == "reader"
    assert jobs[0].needs is None and jobs[0].fold == "fold-014" and jobs[2].fold == "fold-018"
    references = campaign.plan(document, "references")
    # El control sin semillas se planifica una vez por origen, con la primera semilla.
    assert [job.key for job in references if job.model == "rule"] == [
        "rule/2019/all-seeds",
        "rule/2023/all-seeds",
    ]
    assert references[0].key == "transformer/2019/seed-42"
    assert len(references) == 2 * 2 * 2 + 2
    with_ridge = declaration()
    with_ridge["models"]["ridge"] = RIDGE
    planned = campaign.plan(campaign.checked_declaration(with_ridge), "references")
    assert [(job.key, job.seed, job.runner) for job in planned if job.model == "ridge"] == [
        ("ridge/2019/all-seeds", None, "ridge"),
        ("ridge/2023/all-seeds", None, "ridge"),
    ]
    with pytest.raises(ValueError, match="equipo"):
        campaign.plan(document, "laptop")


def test_history_origins_form_a_second_stage_after_every_tier_of_the_first(tmp_path):
    # El orden declarado no decide la etapa: los años añadidos van después aunque se citen antes.
    origins = [
        dict(year=2017, fold="fold-012", role="history"),
        dict(year=2019, fold="fold-014", role="development"),
        dict(year=2016, fold="fold-011", role="history"),
        dict(year=2023, fold="fold-018", role="confirmation"),
    ]
    document = campaign.checked_declaration(declaration(origins=origins, seeds=[42]))
    jobs = campaign.plan(document, "mars_titan")
    assert [(job.key, job.stage) for job in jobs] == [
        ("titans_mac/2019/seed-42", "core"),
        ("mars_titan_m1/2019/seed-42", "core"),
        ("titans_mac/2023/seed-42", "core"),
        ("mars_titan_m1/2023/seed-42", "core"),
        ("area_network/2019/seed-42", "areas"),
        ("area_network/2023/seed-42", "areas"),
        ("titans_mac/2017/seed-42", "core/history"),
        ("mars_titan_m1/2017/seed-42", "core/history"),
        ("titans_mac/2016/seed-42", "core/history"),
        ("mars_titan_m1/2016/seed-42", "core/history"),
        ("area_network/2017/seed-42", "areas/history"),
        ("area_network/2016/seed-42", "areas/history"),
    ]
    assert [job.history for job in jobs] == [False] * 6 + [True] * 6
    assert {job.tier for job in jobs if job.history} == {"core", "areas"}
    # Un lector de un año añadido depende de su padre de ese mismo año.
    assert jobs[7].needs == "titans_mac/2017/seed-42" and jobs[7].fold == "fold-012"
    # Los controles sin semilla también se planifican una vez por origen en cada etapa.
    rules = [job for job in campaign.plan(document, "references") if job.model == "rule"]
    assert [(job.year, job.history) for job in rules] == [
        (2019, False),
        (2023, False),
        (2017, True),
        (2016, True),
    ]
    # La tabla de estado separa las etapas y el filtro por nivel alcanza a las dos.
    place = layout(tmp_path)
    assert list(campaign.summary(document, place, "mars_titan")["by_tier"]) == [
        "core",
        "areas",
        "core/history",
        "areas/history",
    ]
    limited = campaign.Queue(document, place, "mars_titan", tiers=["areas"])
    assert [job.stage for job in limited.jobs] == ["areas"] * 2 + ["areas/history"] * 2


@pytest.mark.parametrize(
    "change",
    [
        dict(kind="other"),
        dict(final_test_access=True),
        dict(market="CN"),
        dict(seeds=[42, 42]),
        dict(seeds=[]),
        dict(cohort=dict(size="16")),
        dict(tiers=["core"]),
        dict(tiers=["core", "core", "areas"]),
        dict(recipes=dict(core="configs/does-not-exist.json", reader=READER)),
        dict(origins=[dict(year=2024, fold="fold-019", role="development")]),
        dict(origins=[dict(year=2019, fold="fold-014", role="test")]),
        # Una campaña necesita al menos un origen que no sea un año añadido.
        dict(origins=[dict(year=2017, fold="fold-012", role="history")]),
        dict(
            origins=[
                dict(year=2019, fold="fold-014", role="development"),
                dict(year=2020, fold="fold-014", role="development"),
            ]
        ),
        dict(
            origins=[
                dict(year=2019, fold="fold-014", role="development"),
                dict(year=2019, fold="fold-015", role="development"),
            ]
        ),
        dict(stopping=dict(metric="session_ic", max_epochs=2)),
        dict(stopping=dict(fit=dict(RULE))),
        dict(stopping=dict(fit=dict(RULE), reader=dict(RULE, metric="session_ic"))),
        dict(stopping=dict(fit=dict(RULE), reader=dict(RULE, max_epochs=0))),
        dict(stopping=dict(fit=dict(RULE), reader=dict(RULE, max_epochs=2.0))),
        dict(stopping=dict(fit=dict(RULE, initial_state="selectable"), reader=dict(RULE))),
        # El lector conserva a su padre como candidato: no admite un estado inicial sin emitir.
        dict(stopping=dict(fit=dict(RULE), reader=dict(RULE, initial_state="reported"))),
        dict(stopping=dict(fit=dict(RULE, patience=0), reader=dict(RULE))),
        dict(execution=dict(pipeline=dict(decode_workers=2), workers=dict(references=1))),
        dict(
            execution=dict(
                pipeline=dict(decode_workers=99, prefetch_batches=2, group_cache_mib=512),
                workers=dict(references=1, mars_titan=1),
            )
        ),
    ],
)
def test_declaration_rejects_what_would_change_the_protocol(change):
    with pytest.raises(ValueError):
        campaign.checked_declaration(declaration(**change))


def test_models_declare_their_runner_fields_and_readers_a_mac_online_parent_of_their_machine():
    def with_model(name, **fields):
        document = declaration()
        document["models"][name] = dict(document["models"][name], **fields)
        return document

    campaign.checked_declaration(declaration())
    for document in (
        with_model("transformer", runner="train"),
        with_model("transformer", machine="cloud"),
        with_model("transformer", tier="unknown"),
        with_model("transformer", recipe="unknown"),
        with_model("transformer", extra=1),
        with_model("mars_titan_m1", parent="missing"),
        with_model("mars_titan_m1", parent="transformer"),
        with_model("mars_titan_m1", parent="area_network", tier="core"),
        with_model("titans_mac", variant="mac_frozen"),
    ):
        with pytest.raises(ValueError, match="modelo|lector"):
            campaign.checked_declaration(document)
    missing = declaration()
    missing["models"]["titans_mac"].pop("variant")
    with pytest.raises(ValueError, match="no declara sus campos"):
        campaign.checked_declaration(missing)
    # Ridge declara su configuración y nada de los ajustes con receta.
    ridge = declaration()
    ridge["models"]["ridge"] = RIDGE
    campaign.checked_declaration(ridge)
    for change in (dict(recipe="core"), dict(epochs=3), dict(config=None)):
        ridge["models"]["ridge"] = {
            key: value for key, value in dict(RIDGE, **change).items() if value is not None
        }
        with pytest.raises(ValueError, match="El modelo ridge no declara sus campos"):
            campaign.checked_declaration(ridge)


def test_authorization_is_a_campaign_file_that_the_learning_hold_reader_accepts(tmp_path):
    place = layout(tmp_path)
    assert not campaign.authorized(place, "a" * 64)
    with pytest.raises(ValueError, match="nota"):
        campaign.authorize(place, "a" * 64, note="  ")
    document = campaign.authorize(place, "a" * 64, note="Lanzamiento aprobado")
    assert document["training_allowed"] is True and document["final_test_access"] is False
    assert campaign.authorized(place, "a" * 64)
    # Una declaración distinta necesita su propio permiso.
    assert not campaign.authorized(place, "b" * 64)
    assert learning_blocked(place.authorization) is False
    environment = campaign._environment(place, campaign.checked_declaration(declaration()))
    assert environment[HOLD_ENV] == str(place.authorization)
    assert environment[SHARED_INDICES_ENV] == str(place.indices)
    assert environment["MARS_TITAN_DECODE_WORKERS"] == "2"
    assert environment["MARS_TITAN_GROUP_CACHE_MIB"] == "512"
    assert environment["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"


@pytest.mark.parametrize("changed", ["declaration", "commit", "changes"])
def test_authorization_cannot_mix_existing_jobs_with_another_execution_identity(tmp_path, changed):
    place = layout(tmp_path)
    identity = dict(declaration="a" * 64, commit="b" * 40, changes="c" * 64)
    campaign.authorize(
        place,
        identity["declaration"],
        note="Primera ejecución",
        commit=identity["commit"],
        changes=identity["changes"],
    )
    campaign._write_record(
        place,
        "transformer/2019/seed-42",
        dict(status=DONE, attempts=[dict(status=DONE)]),
    )
    campaign.authorize(
        place,
        identity["declaration"],
        note="Misma ejecución",
        commit=identity["commit"],
        changes=identity["changes"],
    )
    previous = place.authorization.read_bytes()
    identity[changed] = "d" * len(identity[changed])
    with pytest.raises(ValueError, match="otro estado"):
        campaign.authorize(
            place,
            identity["declaration"],
            note="Otra ejecución",
            commit=identity["commit"],
            changes=identity["changes"],
        )
    assert place.authorization.read_bytes() == previous


def test_layout_needs_absolute_paths_and_keeps_one_file_per_job(tmp_path):
    place = layout(tmp_path)
    key = "titans_mac/2019/seed-42"
    assert place.output(key) == tmp_path / "state/runs/titans_mac/2019/seed-42"
    assert place.record(key).name == "titans_mac--2019--seed-42.json"
    assert place.lock(key).parent == place.record(key).parent
    assert place.source("fold-014") == tmp_path / "views/fold-014/manifest.json"
    assert place.cohort(2019) == place.cohort(Job("m", 2019, "fold-014", 42, "fit", "core"))
    with pytest.raises(ValueError, match="absolutas"):
        Layout("relative/state", tmp_path)


def report(status="completed", **fields):
    return dict(
        status=status,
        run_id="r" * 64,
        kind="titans_walk_forward_window",
        fit=dict(best_epoch=2, best_score=0.01, global_step=64),
        predictions=dict(evaluation=dict(metrics=dict(session_mae=0.0102))),
        **fields,
    )


def test_run_job_records_the_result_and_returns_failures_to_the_queue(tmp_path, monkeypatch):
    document, place = campaign.checked_declaration(declaration()), layout(tmp_path)
    job = campaign.plan(document, "mars_titan")[0]
    outcomes = iter([RuntimeError("CUDA out of memory"), report("paused"), report()])

    def execute(*_, **__):
        outcome = next(outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(campaign, "execute_job", execute)
    assert campaign.read_record(place, job.key) == dict(status=PENDING, attempts=[])
    assert campaign.run_job(document, place, job) == PENDING
    record = campaign.read_record(place, job.key)
    assert record["attempts"][0]["status"] == "failed"
    assert record["attempts"][0]["error_type"] == "RuntimeError"
    assert "CUDA out of memory" in record["attempts"][0]["traceback"]
    assert campaign.run_job(document, place, job) == PAUSED
    assert campaign.job_state(place, job) == PAUSED
    assert campaign.run_job(document, place, job) == DONE
    record = campaign.read_record(place, job.key)
    assert [item["status"] for item in record["attempts"]] == ["failed", "paused", "completed"]
    assert "no conserva sus tablas" in record["result"].pop("scores_error")
    assert record["result"] == dict(
        run_id="r" * 64,
        kind="titans_walk_forward_window",
        best_epoch=2,
        best_score=0.01,
        global_step=64,
        session_mae=dict(evaluation=0.0102),
    )
    assert all(item["seconds"] >= 0 for item in record["attempts"])


@pytest.mark.parametrize("model,magnitude", [("transformer", True), ("rule", False)])
def test_finished_jobs_record_scores_from_their_prediction_tables(
    tmp_path, monkeypatch, model, magnitude
):
    document, place = campaign.checked_declaration(declaration()), layout(tmp_path)
    job = next(job for job in campaign.plan(document, "references") if job.model == model)
    write_tables(place.output(job.key))
    completed = dict(status="completed", predictions=declared())
    monkeypatch.setattr(campaign, "execute_job", lambda *_, **__: completed)
    assert campaign.run_job(document, place, job) == DONE
    measured = campaign.read_record(place, job.key)["result"]["scores"]
    assert measured["rows"] == 6 and measured["sessions"] == 2
    assert measured["scale"] == pytest.approx(0.5)
    assert (measured["raw"] is not None) is magnitude
    assert measured["rank_ic"] == pytest.approx(0.75)


def test_a_job_fails_for_good_after_the_declared_attempts_and_retry_returns_it(
    tmp_path, monkeypatch
):
    document, place = campaign.checked_declaration(declaration()), layout(tmp_path)
    parent, reader = campaign.plan(document, "mars_titan")[:2]

    def broken(*_, **__):
        raise ValueError("fallo permanente")

    monkeypatch.setattr(campaign, "execute_job", broken)
    states = [campaign.run_job(document, place, parent) for _ in range(campaign.MAX_ATTEMPTS)]
    assert states == [PENDING] * (campaign.MAX_ATTEMPTS - 1) + [FAILED]
    assert campaign.job_state(place, parent) == FAILED
    # El lector de un padre fallido queda bloqueado y no se ejecuta.
    assert campaign.job_state(place, reader) == campaign.BLOCKED
    assert campaign.retry_failed(document, place, "mars_titan") == 1
    assert campaign.job_state(place, parent) == PENDING
    attempts = campaign.read_record(place, parent.key)["attempts"]
    assert {item["status"] for item in attempts} == {"failed_before_retry"}
    monkeypatch.setattr(campaign, "execute_job", lambda *_, **__: report())
    assert campaign.run_job(document, place, parent) == DONE
    assert campaign.job_state(place, reader) == PENDING


def test_an_interrupted_job_stays_paused_and_a_stale_running_record_is_claimable(
    tmp_path, monkeypatch
):
    document, place = campaign.checked_declaration(declaration()), layout(tmp_path)
    job = campaign.plan(document, "mars_titan")[0]

    def interrupted(*_, **__):
        raise KeyboardInterrupt

    monkeypatch.setattr(campaign, "execute_job", interrupted)
    assert campaign.run_job(document, place, job) == PAUSED
    assert campaign.read_record(place, job.key)["attempts"][-1]["status"] == "interrupted"
    # Un apagado deja el registro en marcha sin proceso: la cola lo vuelve a coger.
    record = campaign.read_record(place, job.key)
    record["status"] = campaign.RUNNING
    campaign._write_record(place, job.key, record)
    assert campaign.job_state(place, job) == PENDING
    claimed, handle = campaign._claim(place, [job], set())
    assert claimed == job
    assert campaign._claim(place, [job], set()) == (None, None)
    handle.close()


WORKER = """
import json, sys, time
from pathlib import Path
record, log, key, mode = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4]
def save(data):
    temporary = record.with_suffix(".tmp")
    temporary.write_text(json.dumps(data))
    temporary.replace(record)
with log.open("a") as stream:
    stream.write(key + ("!preflight" if sys.argv[5] == "preflight" else "") + "\\n")
time.sleep(0.05)
if mode == "crash" and key.startswith("titans_mac/2023"):
    sys.exit(9)
if mode == "broken-model" and key.startswith("area_network"):
    sys.exit(9)
data = json.loads(record.read_text())
if sys.argv[5] == "preflight":
    # La comprobación inicial se detiene en la primera barrera del trabajo.
    data["status"] = "paused"
    data["attempts"][-1].update(status="paused", seconds=0.01, preflight=True)
    save(data)
    sys.exit(3)
data["status"] = "completed"
data["attempts"][-1].update(status="completed", seconds=0.05)
save(data)
"""


def queue(tmp_path, monkeypatch, *, workers, mode="ok", tiers=None):
    document, place = campaign.checked_declaration(declaration()), layout(tmp_path)
    script, executed = tmp_path / "worker.py", tmp_path / "executed.log"
    script.write_text(WORKER)
    instance = campaign.Queue(document, place, "mars_titan", workers=workers, tiers=tiers)
    instance.poll_seconds = 0.05

    def command(job, *, preflight=False):
        # El proceso real marca el trabajo en marcha y abre un intento antes de ejecutarlo.
        record = campaign.read_record(place, job.key)
        record["status"] = campaign.RUNNING
        record["attempts"].append({})
        campaign._write_record(place, job.key, record)
        arguments = [str(place.record(job.key)), str(executed), job.key, mode]
        return [sys.executable, str(script), *arguments, "preflight" if preflight else "run"]

    monkeypatch.setattr(instance, "_command", command)
    return instance, place, executed


def test_queue_stops_without_claiming_when_the_disk_runs_out(tmp_path, monkeypatch, capsys):
    instance, place, executed = queue(tmp_path, monkeypatch, workers=2)
    assert instance.min_free_gib == campaign.MIN_FREE_GIB == 1.0
    free = iter([5.0, 5.0, 0.4, 0.4, 0.4, 0.4])
    asked = []

    def measured(path):
        asked.append(path)
        return next(free)

    monkeypatch.setattr(campaign, "_free_gib", measured)
    result = instance.run(preflight=False)
    # Solo arrancan los trabajos reclamados mientras quedaba margen. Ninguno falla.
    done = sum(counts.get(DONE, 0) for counts in result["by_tier"].values())
    started = executed.read_text().split()
    assert 1 <= done <= len(started) <= 2
    assert all(
        attempt["status"] in (DONE, "interrupted")
        for key in started
        for attempt in campaign.read_record(place, key)["attempts"]
    )
    assert all(FAILED not in counts for counts in result["by_tier"].values())
    assert instance.stop.is_set() and set(asked) == {place.state}
    output = capsys.readouterr().out
    assert output.count("Quedan 0.4 GiB libres y la cola necesita 1.0") == 1
    # Con disco de nuevo, la misma orden termina la cola.
    monkeypatch.setattr(campaign, "_free_gib", lambda path: 1.0)
    again, _, _ = queue(tmp_path, monkeypatch, workers=1)
    result = again.run(preflight=False)
    assert result["by_tier"] == dict(core={DONE: 8}, areas={DONE: 4})


def test_the_declared_budget_sets_the_free_disk_margin(tmp_path):
    document = campaign.checked_declaration(declaration(budget=dict(min_free_gib=2.5)))
    assert campaign.Queue(document, layout(tmp_path), "references").min_free_gib == 2.5
    assert campaign._free_gib(tmp_path / "does" / "not" / "exist") == campaign._free_gib(tmp_path)
    assert campaign._free_gib(tmp_path) > 0
    for budget in (
        [],
        dict(other=1),
        dict(min_free_gib=-1),
        dict(min_free_gib="2"),
        dict(disk_gib=[]),
        dict(disk_gib=dict(laptop=3)),
        dict(disk_gib=dict(references=-1)),
        dict(disk_gib=dict(references=True)),
    ):
        with pytest.raises(ValueError, match="El presupuesto declara el disco estimado"):
            campaign.checked_declaration(declaration(budget=budget))
    campaign.checked_declaration(
        declaration(budget=dict(disk_gib=dict(references=4.5, mars_titan=12), min_free_gib=0))
    )


@pytest.mark.parametrize("value", ["mañana", "2026-12-05", "2026-12-05T00:00:00", None, 5])
def test_a_campaign_deadline_needs_an_explicit_timezone(value):
    with pytest.raises(ValueError, match="fecha límite"):
        campaign.checked_declaration(declaration(budget=dict(deadline_utc=value)))


def test_an_expired_campaign_starts_no_job_or_preflight(tmp_path, monkeypatch, capsys):
    deadline = "2026-12-04T23:00:00+00:00"
    document = campaign.checked_declaration(declaration(budget=dict(deadline_utc=deadline)))
    instance = campaign.Queue(document, layout(tmp_path), "mars_titan")
    monkeypatch.setattr(campaign, "_now", lambda: deadline)
    monkeypatch.setattr(instance, "_launch", lambda *_, **__: pytest.fail("fuera de plazo"))
    result = instance.run()
    assert instance.stop.is_set()
    assert result["seconds_spent"] == 0
    assert "fecha límite" in capsys.readouterr().out


def test_reaching_the_deadline_requests_a_checkpoint_from_running_children(tmp_path, monkeypatch):
    deadline = "2026-12-04T23:00:00+00:00"
    document = campaign.checked_declaration(declaration(budget=dict(deadline_utc=deadline)))
    instance = campaign.Queue(document, layout(tmp_path), "mars_titan")

    class Child:
        def __init__(self):
            self.signals = []

        def poll(self):
            return None

        def send_signal(self, signal):
            self.signals.append(signal)

    child = Child()
    instance.children["active"] = child
    monkeypatch.setattr(campaign, "_now", lambda: "2026-12-04T22:59:59+00:00")
    assert not instance._deadline_reached() and child.signals == []
    monkeypatch.setattr(campaign, "_now", lambda: deadline)
    assert instance._deadline_reached() and instance.stop.is_set()
    assert child.signals == [campaign.signal.SIGTERM]


@pytest.mark.parametrize("workers", [1, 3])
def test_queue_runs_every_job_once_and_readers_after_their_parent(tmp_path, monkeypatch, workers):
    instance, place, executed = queue(tmp_path, monkeypatch, workers=workers)
    result = instance.run(preflight=False)
    order = executed.read_text().split()
    assert sorted(order) == sorted(job.key for job in instance.jobs) and len(order) == 12
    for job in instance.jobs:
        if job.needs is not None:
            assert order.index(job.needs) < order.index(job.key)
    assert result["by_tier"] == dict(core={DONE: 8}, areas={DONE: 4})
    assert result["jobs"] == 12 and result["estimated_seconds_remaining"] == 0
    if workers == 1:
        assert order == [job.key for job in instance.jobs]
    # Una segunda orden no repite nada.
    again, _, executed = queue(tmp_path, monkeypatch, workers=workers)
    again.run(preflight=False)
    assert len(executed.read_text().split()) == 12
    assert (place.log("titans_mac/2019/seed-42")).read_text().count("===") == 1


def test_a_process_that_dies_is_retried_then_failed_and_does_not_stop_the_queue(
    tmp_path, monkeypatch
):
    instance, place, executed = queue(tmp_path, monkeypatch, workers=2, mode="crash")
    result = instance.run(preflight=False)
    order = executed.read_text().split()
    crashed = [key for key in order if key.startswith("titans_mac/2023")]
    assert len(crashed) == 2 * campaign.MAX_ATTEMPTS
    assert result["by_tier"]["core"] == {DONE: 4, FAILED: 2, campaign.BLOCKED: 2}
    assert result["by_tier"]["areas"] == {DONE: 4}
    record = campaign.read_record(place, "titans_mac/2023/seed-42")
    assert record["status"] == FAILED
    assert "código 9" in record["attempts"][-1]["error"]
    assert not any(key.startswith("mars_titan_m1/2023") for key in order)


def test_queue_can_be_limited_to_tiers_and_stops_claiming_after_a_stop_request(
    tmp_path, monkeypatch
):
    instance, _, executed = queue(tmp_path, monkeypatch, workers=1, tiers=["areas"])
    assert [job.tier for job in instance.jobs] == ["areas"] * 4
    instance.request_stop()
    instance.run()
    assert not executed.exists()
    with pytest.raises(ValueError, match="desconocido"):
        queue(tmp_path, monkeypatch, workers=1, tiers=["unknown"])
    with pytest.raises(ValueError, match="trabajadores"):
        queue(tmp_path, monkeypatch, workers=0)


def test_a_stop_before_the_first_barrier_is_an_interruption_not_a_failure(tmp_path, monkeypatch):
    instance, place, _ = queue(tmp_path, monkeypatch, workers=1)
    job = instance.jobs[0]
    campaign._write_record(place, job.key, dict(status=campaign.RUNNING, attempts=[{}]))
    instance.stop.set()
    instance._abnormal_exit(job, -15)
    record = campaign.read_record(place, job.key)
    assert record["status"] == PENDING and record["attempts"][-1]["status"] == "interrupted"
    instance.stop = threading.Event()
    campaign._write_record(place, job.key, dict(status=campaign.RUNNING, attempts=[{}]))
    instance._abnormal_exit(job, -9)
    assert campaign.read_record(place, job.key)["attempts"][-1]["status"] == "failed"
    # Un trabajo que ya escribió su estado no se toca.
    campaign._write_record(place, job.key, dict(status=DONE, attempts=[dict(status="completed")]))
    instance._abnormal_exit(job, 1)
    assert campaign.read_record(place, job.key)["status"] == DONE


def test_a_child_that_fails_before_writing_its_record_exhausts_its_retries(tmp_path, monkeypatch):
    document, place = campaign.checked_declaration(declaration()), layout(tmp_path)
    instance = campaign.Queue(document, place, "mars_titan")
    job = instance.jobs[0]
    monkeypatch.setattr(
        instance, "_command", lambda *_, **__: [sys.executable, "-c", "raise SystemExit(2)"]
    )
    for _ in range(3):
        assert instance._launch(job) == 2
    record = campaign.read_record(place, job.key)
    assert record["status"] == FAILED
    assert len(record["attempts"]) == 3
    assert all(
        item["status"] == "failed" and "código 2" in item["error"] for item in record["attempts"]
    )


def test_a_stop_during_process_creation_reaches_the_new_child(tmp_path, monkeypatch):
    instance, place, _ = queue(tmp_path, monkeypatch, workers=1)
    job = instance.jobs[0]

    class Child:
        stopped = False

        def poll(self):
            return None

        def send_signal(self, signal):
            assert signal == campaign.signal.SIGTERM
            self.stopped = True

        def wait(self):
            return -15 if self.stopped else 0

    def start(*_, **__):
        instance.request_stop()
        return Child()

    monkeypatch.setattr(campaign.subprocess, "Popen", start)
    assert instance._launch(job) == -15
    record = campaign.read_record(place, job.key)
    assert record["status"] == PENDING and record["attempts"][-1]["status"] == "interrupted"
    assert instance.children == {}


def test_a_child_failure_already_recorded_is_not_counted_twice(tmp_path, monkeypatch):
    instance, place, _ = queue(tmp_path, monkeypatch, workers=1)
    job = instance.jobs[0]
    script = tmp_path / "reported.py"
    script.write_text(
        "import json, sys\nfrom pathlib import Path\n"
        "path = Path(sys.argv[1])\nrecord = json.loads(path.read_text())\n"
        "record['status'] = 'pending'\nrecord['attempts'][-1]['status'] = 'failed'\n"
        "path.write_text(json.dumps(record))\nraise SystemExit(4)\n"
    )
    previous = instance._command

    def command(job, **options):
        previous(job, **options)
        return [sys.executable, str(script), str(place.record(job.key))]

    monkeypatch.setattr(instance, "_command", command)
    assert instance._launch(job) == 4
    record = campaign.read_record(place, job.key)
    assert record["status"] == PENDING and len(record["attempts"]) == 1


def test_prepare_writes_one_cohort_per_origin_from_its_window(tmp_path):
    from mars_titan.data.storage import atomic_json
    from tests.training.test_screening_cohorts import source

    document, place = campaign.checked_declaration(declaration()), layout(tmp_path)
    with pytest.raises(ValueError, match="Falta la vista"):
        campaign.prepare(document, place)
    atomic_json(place.source("fold-014"), source(2019))
    atomic_json(place.source("fold-018"), source(2023))
    receipts = campaign.prepare(document, place)
    assert set(receipts) == {"2019", "2023"}
    assert all(selection["size"] == 16 for selection in receipts.values())
    manifest = json.loads((place.cohort(2023) / "manifest.json").read_text())
    assert len(manifest["assets"]) == 16
    assert campaign.prepare(document, place) == receipts
    # La vista de otro origen en la carpeta de una ventana se detecta.
    other = Layout(tmp_path / "other", tmp_path / "other-views")
    atomic_json(other.source("fold-014"), source(2023))
    atomic_json(other.source("fold-018"), source(2023))
    with pytest.raises(ValueError, match="no es la ventana fold-014"):
        campaign.prepare(document, other)
    # Y también una ventana existente declarada con el año de otra.
    swapped = copy.deepcopy(document)
    swapped["origins"][0]["fold"] = "fold-018"
    with pytest.raises(ValueError, match="de ese año"):
        campaign.prepare(swapped, Layout(tmp_path / "swapped", tmp_path / "views"))


def test_summary_estimates_the_remaining_hours_from_completed_jobs(tmp_path, capsys):
    document, place = campaign.checked_declaration(declaration()), layout(tmp_path)
    jobs = campaign.plan(document, "mars_titan")
    empty = campaign.summary(document, place, "mars_titan")
    assert empty["estimated_seconds_remaining"] is None and empty["seconds_spent"] == 0
    assert empty["by_tier"]["core"] == {PENDING: 4, campaign.BLOCKED: 4}
    assert empty["failed"] == {} and empty["open_attempts"] == {}
    for job, seconds in ((jobs[0], 100.0), (jobs[2], 300.0), (jobs[8], 50.0)):
        campaign._write_record(
            place, job.key, dict(status=DONE, attempts=[dict(status="completed", seconds=seconds)])
        )
    result = campaign.summary(document, place, "mars_titan")
    assert result["seconds_spent"] == 450.0
    assert result["mean_seconds_by_model"] == dict(titans_mac=200.0, area_network=50.0)
    # Dos titans_mac a 200 s, tres redes de áreas a 50 s y cuatro lectores a la media global.
    assert result["estimated_seconds_remaining"] == pytest.approx(2 * 200 + 3 * 50 + 4 * 150)
    campaign._write_record(
        place,
        jobs[4].key,
        dict(
            status=FAILED, attempts=[dict(error="primer fallo"), dict(error="último fallo\ntraza")]
        ),
    )
    campaign._write_record(
        place,
        jobs[10].key,
        dict(status=campaign.RUNNING, attempts=[dict(started_at_utc="2026-11-13T07:00:00Z")]),
    )
    result = campaign.summary(document, place, "mars_titan")
    assert result["failed"] == {jobs[4].key: "último fallo"}
    assert result["open_attempts"] == {jobs[10].key: "2026-11-13T07:00:00Z"}
    campaign._print_summary(result)
    printed = capsys.readouterr().out
    assert f"fallido: {jobs[4].key}: último fallo" in printed
    assert f"con un intento abierto: {jobs[10].key}, desde 2026-11-13T07:00:00Z" in printed


def finished(place, key, scores):
    result = {} if scores is None else dict(scores=scores)
    campaign._write_record(place, key, dict(status=DONE, attempts=[], result=result))


def test_results_average_finished_jobs_against_the_zero_prediction_of_the_same_rows(
    tmp_path, capsys
):
    origins = [
        dict(year=2019, fold="fold-014", role="development"),
        dict(year=2023, fold="fold-018", role="confirmation"),
        dict(year=2017, fold="fold-012", role="history"),
    ]
    document = campaign.checked_declaration(declaration(origins=origins))
    place = layout(tmp_path)
    assert all(row["scored"] == 0 for row in campaign.results(document, place, "references"))
    scores = dict(raw=0.021, zero=0.02, calibrated=0.018, pearson_ic=0.1, rank_ic=0.2)
    finished(place, "transformer/2019/seed-42", scores)
    finished(place, "transformer/2019/seed-43", dict(scores, raw=0.019))
    finished(
        place,
        "transformer/2023/seed-42",
        dict(scores, raw=0.26, zero=0.2, calibrated=0.18, pearson_ic=0.4, rank_ic=0.5),
    )
    # La regla predice un orden y conserva su lectura calibrada.
    finished(place, "rule/2019/all-seeds", dict(scores, raw=None))
    finished(place, "gru/2017/seed-42", None)
    # Un trabajo en marcha no entra aunque ya tenga medidas de un intento anterior.
    campaign._write_record(
        place,
        "gru/2019/seed-42",
        dict(status=campaign.RUNNING, attempts=[], result=dict(scores=scores)),
    )
    rows = {
        (row["model"], row["history"]): row
        for row in campaign.results(document, place, "references")
    }
    main = rows["transformer", False]
    assert (main["done"], main["jobs"], main["scored"]) == (3, 4, 3)
    assert main["raw"] == pytest.approx(0.1)
    assert main["zero"] == pytest.approx(0.08)
    assert main["calibrated"] == pytest.approx(0.072)
    assert main["pearson_ic"] == pytest.approx(0.2)
    assert main["rank_ic"] == pytest.approx(0.3)
    assert main["difference_percent"] == pytest.approx(25)
    assert main["calibrated_percent"] == pytest.approx(-10)
    rule = rows["rule", False]
    assert (rule["done"], rule["jobs"], rule["scored"]) == (1, 2, 1)
    assert rule["raw"] is None and rule["zero"] == 0.02
    assert rule["difference_percent"] is None
    assert rule["calibrated_percent"] == pytest.approx(-10)
    history = rows["gru", True]
    assert (history["done"], history["jobs"], history["scored"]) == (1, 2, 0)
    assert history["raw"] is None and history["zero"] is None
    assert rows["gru", False]["done"] == 0 and rows["transformer", True]["done"] == 0

    path = tmp_path / "declaration.json"
    path.write_text(json.dumps(declaration(origins=origins)))
    arguments = ["--machine", "references", "--declaration", str(path), "--state", str(place.state)]
    assert campaign.main(["results", *arguments]) == 0
    printed = capsys.readouterr().out
    assert "equipo references, primera etapa" in printed
    assert "equipo references, años añadidos" in printed
    assert "factor elegido en la calibración" in printed
    line = next(text for text in printed.splitlines() if text.startswith("  transformer"))
    assert line.split() == [
        "transformer",
        "3/4",
        "0.100000",
        "0.080000",
        "+25.000",
        "-10.000",
        "+0.2000",
        "+0.3000",
    ]
    line = next(text for text in printed.splitlines() if text.startswith("  rule"))
    assert line.split() == ["rule", "1/2", "-", "0.020000", "-", "-10.000", "+0.1000", "+0.2000"]
    with pytest.raises(ValueError, match="La orden results necesita --machine"):
        campaign.main(["results", "--declaration", str(path), "--state", str(place.state)])


def test_command_line_plans_and_refuses_to_run_without_the_express_permission(
    tmp_path, capsys, monkeypatch
):
    path = tmp_path / "declaration.json"
    path.write_text(json.dumps(declaration()))
    common = ["--machine", "mars_titan", "--declaration", str(path), "--state"]
    state = str(tmp_path / "state")
    assert campaign.main(["plan", *common, state]) == 0
    assert "12 trabajos en mars_titan" in capsys.readouterr().out
    assert campaign.main(["status", *common, state]) == 0
    with pytest.raises(ValueError, match="permiso expreso"):
        campaign.main(["run", *common, state])
    with pytest.raises(ValueError, match="permiso expreso"):
        campaign.main(["job", *common, state, "--key", "titans_mac/2019/seed-42"])
    assert campaign.main(["authorize", *common, state, "--note", "Aprobado por el autor"]) == 0
    monkeypatch.setattr(campaign, "execute_job", lambda *_, **__: report())
    assert campaign.main(["job", *common, state, "--key", "titans_mac/2019/seed-42"]) == 0
    with pytest.raises(ValueError, match="no pertenece"):
        campaign.main(["job", *common, state, "--key", "transformer/2019/seed-42"])
    assert campaign.main(["retry", *common, state]) == 0
    assert Path(state, "authorization.json").is_file()


def test_running_one_tier_returns_success_when_that_tier_finishes(tmp_path, monkeypatch):
    path = tmp_path / "declaration.json"
    path.write_text(json.dumps(declaration()))
    place = layout(tmp_path)
    options = ["--declaration", str(path), "--state", str(place.state), "--machine", "mars_titan"]
    assert campaign.main(["authorize", *options, "--note", "Cola con procesos sustitutos"]) == 0

    def completed(self, **options):
        for job in self.jobs:
            campaign._write_record(self.layout, job.key, dict(status=DONE, attempts=[]))
        return campaign.summary(self.declaration, self.layout, self.machine)

    monkeypatch.setattr(campaign.Queue, "run", completed)
    assert campaign.main(["run", *options, "--tier", "areas"]) == 0
    result = campaign.summary(campaign.checked_declaration(declaration()), place, "mars_titan")
    assert result["by_tier"]["areas"] == {DONE: 4}
    assert result["by_tier"]["core"] == {PENDING: 4, campaign.BLOCKED: 4}


def local_views(place, tmp_path, years=((2019, "fold-014"), (2023, "fold-018")), **fields):
    """Vistas de prueba con sus raíces dentro de `tmp_path`, sin ningún archivo de datos.

    Como en la edición real, los datos preparados y las muestras son comunes y cada ventana
    tiene sus etiquetas.
    """
    from mars_titan.data.storage import atomic_json
    from tests.training.test_screening_cohorts import source

    for year, fold in years:
        roots = dict(
            prepared=str(tmp_path / "data/prepared"),
            samples=str(tmp_path / "data/samples"),
            labels=str(tmp_path / f"data/labels-{year}"),
        )
        atomic_json(place.source(fold), dict(source(year), roots=roots, **fields))


def cohort_files(place, year):
    cohort = json.loads((place.cohort(year) / "manifest.json").read_text())
    for asset in cohort["assets"]:
        for kind in ("prices", "samples", "labels"):
            root = Path(cohort["roots"]["prepared" if kind == "prices" else kind])
            yield root / asset["market"] / asset["symbol"] / f"{kind}.parquet"


def test_doctor_names_what_is_missing_before_a_machine_can_launch(tmp_path, monkeypatch):
    document, place = campaign.checked_declaration(declaration()), layout(tmp_path)
    monkeypatch.setattr(campaign, "_gpu_check", lambda: (True, "GPU de prueba"))
    monkeypatch.setattr(campaign, "_native_check", lambda _: (False, "sin enlace"))
    monkeypatch.setattr(campaign, "_available_memory_gib", lambda: 13.0)
    checks = {
        name: (state, detail)
        for name, state, detail in campaign.doctor(document, place, "mars_titan")
    }
    assert checks["vista 2019"][0] is False and checks["cohorte 2023"][0] is False
    assert checks["cuda:0"] == (True, "GPU de prueba")
    assert checks["enlace episódico nativo (M1, M2 y M3)"] == (False, "sin enlace")
    # Dos trabajadores declarados en este equipo y 6 GiB medidos por trabajo.
    assert (
        checks["memoria disponible"][0] is True
        and "12 GiB para 2" in checks["memoria disponible"][1]
    )
    assert checks["disco libre"][0] is True and "permiso expreso de esta declaración" not in checks
    local_views(place, tmp_path)
    campaign.prepare(document, place)
    checks = {name: state for name, state, _ in campaign.doctor(document, place, "mars_titan")}
    assert checks["vista 2019"] is True and checks["datos de la cohorte 2019"] is False
    for year in (2019, 2023):
        for path in cohort_files(place, year):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"x")
    # Falta un único archivo de una cohorte.
    victim = next(path for path in cohort_files(place, 2023) if path.name == "labels.parquet")
    victim.unlink()
    results = campaign.doctor(document, place, "mars_titan", digest="a" * 64, workers=3)
    checks = {name: (state, detail) for name, state, detail in results}
    assert checks["datos de la cohorte 2019"] == (True, "48 de 48 archivos")
    assert checks["datos de la cohorte 2023"] == (False, "47 de 48 archivos")
    assert checks["memoria disponible"][0] is False
    # El permiso pendiente es un aviso: no impide comprobar el resto.
    assert checks["permiso expreso de esta declaración"][0] is None
    campaign.authorize(
        place,
        "a" * 64,
        note="Aprobado",
        commit=campaign.code_commit(),
        changes=campaign.code_changes(),
    )
    results = campaign.doctor(document, place, "mars_titan", digest="a" * 64)
    assert dict((name, state) for name, state, _ in results)["permiso expreso de esta declaración"]
    # El equipo de las referencias no necesita el enlace nativo, y sin GPU no se pasa a CPU.
    monkeypatch.setattr(campaign, "_gpu_check", lambda: (False, "sin CUDA"))
    names = {name: state for name, state, _ in campaign.doctor(document, place, "references")}
    assert "enlace episódico nativo (M1, M2 y M3)" not in names and names["cuda:0"] is False
    budget = dict(document, budget=dict(disk_gib=dict(references=10**9, mars_titan=1)))
    names = {name: state for name, state, _ in campaign.doctor(budget, place, "references")}
    assert names["disco libre"] is False


def test_doctor_uses_the_memory_budget_of_each_machine(tmp_path, monkeypatch):
    document = campaign.checked_declaration(
        declaration(budget=dict(memory_gib=dict(references=12, mars_titan=6)))
    )
    monkeypatch.setattr(campaign, "_available_memory_gib", lambda: 10.0)
    monkeypatch.setattr(campaign, "_gpu_check", lambda: (True, "GPU de prueba"))
    monkeypatch.setattr(campaign, "_native_check", lambda _: (True, "enlace de prueba"))
    place = layout(tmp_path)
    for machine, expected in (("references", False), ("mars_titan", True)):
        checks = {
            name: state for name, state, _ in campaign.doctor(document, place, machine, workers=1)
        }
        assert checks["memoria disponible"] is expected
    for invalid in (dict(other=6), dict(references=-1), dict(mars_titan=True), []):
        with pytest.raises(ValueError, match="presupuesto"):
            campaign.checked_declaration(declaration(budget=dict(memory_gib=invalid)))


def test_ridge_alone_already_needs_cuda_and_cpu_controls_do_not(tmp_path, monkeypatch):
    document = declaration()
    for name in ("transformer", "gru"):
        del document["models"][name]
    place = layout(tmp_path)
    monkeypatch.setattr(campaign, "_gpu_check", lambda: (True, "GPU de prueba"))
    checked = campaign.checked_declaration(document)
    assert "cuda:0" not in [name for name, _, _ in campaign.doctor(checked, place, "references")]
    document["models"]["ridge"] = RIDGE
    checked = campaign.checked_declaration(document)
    assert "cuda:0" in [name for name, _, _ in campaign.doctor(checked, place, "references")]


def test_readers_without_a_bank_do_not_need_the_native_binding(tmp_path, monkeypatch):
    document = declaration()
    document["models"]["mars_titan_m1"]["components"] = dict(episodic_bank=campaign.NO_BANK)
    document, place = campaign.checked_declaration(document), layout(tmp_path)
    monkeypatch.setattr(campaign, "_gpu_check", lambda: (True, ""))
    monkeypatch.setattr(campaign, "_native_check", lambda _: pytest.fail("no debe cargarse"))
    names = [name for name, _, _ in campaign.doctor(document, place, "mars_titan")]
    assert "cuda:0" in names and not any("nativo" in name for name in names)


def test_native_and_gpu_probes_report_their_failure_without_raising(tmp_path, monkeypatch):
    place = layout(tmp_path)
    monkeypatch.setenv(campaign.NATIVE_ENV, str(tmp_path / "missing.so"))
    place = layout(tmp_path)
    assert place.native == tmp_path / "missing.so"
    passed, detail = campaign._native_check(place)
    assert passed is False and "FileNotFoundError" in detail
    monkeypatch.delenv(campaign.NATIVE_ENV)
    assert layout(tmp_path).native == campaign.DEFAULT_NATIVE
    assert campaign._environment(place, campaign.checked_declaration(declaration()))[
        campaign.NATIVE_ENV
    ] == str(tmp_path / "missing.so")
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert campaign._gpu_check() == (
        False,
        "torch.cuda.is_available() es falso. No se cambia a CPU",
    )

    def broken():
        raise RuntimeError("driver")

    monkeypatch.setattr(torch.cuda, "is_available", broken)
    assert campaign._gpu_check() == (False, "RuntimeError: driver")
    assert campaign._available_memory_gib() > 0


def test_export_and_import_carry_the_cohort_data_and_its_identity(tmp_path, capsys, monkeypatch):
    import hashlib

    from mars_titan.training import comparison_data

    home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    path = tmp_path / "declaration.json"
    path.write_text(json.dumps(declaration()))
    document, digest = campaign.load_declaration(path)
    place = Layout(home / "state", home / "views")
    factor = home / "data/prepared/US/SPY/prices.parquet"
    factor.parent.mkdir(parents=True)
    factor.write_bytes(b"factor")
    factors = dict(
        US=dict(prices_path=str(factor), prices_sha256=hashlib.sha256(b"factor").hexdigest())
    )
    local_views(place, home, market_factors=factors)
    campaign.prepare(document, place)
    for year in (2019, 2023):
        cohort_path = place.cohort(year) / "manifest.json"
        cohort = json.loads(cohort_path.read_text())
        for asset in cohort["assets"]:
            for kind in ("prices", "samples", "labels"):
                file = Path(cohort["roots"]["prepared" if kind == "prices" else kind]).joinpath(
                    asset["market"], asset["symbol"], f"{kind}.parquet"
                )
                file.parent.mkdir(parents=True, exist_ok=True)
                file.write_bytes(f"{kind}-{asset['symbol']}".encode())
                asset[f"{kind}_sha256"] = hashlib.sha256(file.read_bytes()).hexdigest()
        cohort_path.write_text(json.dumps(cohort))
    monkeypatch.setenv(campaign.NATIVE_ENV, str(home / "native/_episodic_native.so"))
    (home / "native").mkdir()
    (home / "native/_episodic_native.so").write_bytes(b"binding")
    common = ["--declaration", str(path), "--state", str(place.state), "--views", str(place.views)]
    with pytest.raises(ValueError, match="necesita --destination"):
        campaign.main(["export", *common])
    with pytest.raises(ValueError, match="necesita --machine"):
        campaign.main(["plan", *common])
    disk = tmp_path / "disk"
    assert campaign.main(["export", *common, "--destination", str(disk), "--part-gib", "0.5"]) == 0
    manifest = json.loads((disk / comparison_data.MANIFEST).read_text())
    assert manifest["identity"] == dict(
        declaration_sha256=digest,
        cohorts=campaign.cohort_identity(document, place),
        code_commit=campaign.code_commit(),
    )
    assert set(manifest["identity"]["cohorts"]) == {"2019", "2023"}
    names = [record["path"] for record in manifest["files"]]
    assert "native/_episodic_native.so" in names and "views/fold-014/manifest.json" in names
    assert "data/prepared/US/SPY/prices.parquet" in names
    assert "archivos" in capsys.readouterr().out
    # El otro equipo: mismos archivos, estado de campaña nuevo.
    victim = (
        home
        / "data/samples/US"
        / json.loads((place.cohort(2019) / "manifest.json").read_text())["assets"][0]["symbol"]
        / "samples.parquet"
    )
    content = victim.read_bytes()
    victim.unlink()
    other = ["--declaration", str(path), "--state", str(home / "other-state")]
    other += ["--views", str(place.views)]
    with pytest.raises(ValueError, match="necesita --source"):
        campaign.main(["import", *other])
    assert campaign.main(["import", *other, "--source", str(disk)]) == 0
    assert victim.read_bytes() == content and "1 archivos restaurados" in capsys.readouterr().out
    there = Layout(home / "other-state", place.views)
    assert json.loads(there.bundle.read_text()) == manifest["identity"]
    monkeypatch.setattr(campaign, "_gpu_check", lambda: (True, ""))
    monkeypatch.setattr(campaign, "_native_check", lambda _: (True, ""))
    name = "mismas cohortes que el equipo de origen"
    before = {n: s for n, s, _ in campaign.doctor(document, there, "mars_titan", digest=digest)}
    assert before[name] is False
    campaign.prepare(document, there)
    after = {n: s for n, s, _ in campaign.doctor(document, there, "mars_titan", digest=digest)}
    assert after[name] is True
    warning = "paquete exportado con otra declaración u otro commit"
    assert warning not in after
    changed = {n: s for n, s, _ in campaign.doctor(document, there, "mars_titan", digest="b" * 64)}
    assert changed[name] is True and changed[warning] is None
    # Con otro commit en este equipo, el código ya no es el del equipo de origen.
    monkeypatch.setattr(campaign, "code_commit", lambda *_: "f" * 40)
    other_code = {n: s for n, s, _ in campaign.doctor(document, there, "mars_titan", digest=digest)}
    assert other_code[name] is True and other_code[warning] is None
    monkeypatch.undo()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.setenv(campaign.NATIVE_ENV, str(home / "native/_episodic_native.so"))
    # Un modelo sin banco no lleva el enlace nativo en su paquete.
    plain = declaration()
    plain["models"]["mars_titan_m1"]["components"] = dict(episodic_bank=campaign.NO_BANK)
    path.write_text(json.dumps(plain))
    second = tmp_path / "second"
    second.mkdir()
    assert campaign.main(["export", *common, "--destination", str(second)]) == 0
    listed = json.loads((second / comparison_data.MANIFEST).read_text())["files"]
    assert not any("native" in record["path"] for record in listed)


def test_doctor_command_returns_two_only_for_real_failures(tmp_path, capsys, monkeypatch):
    path = tmp_path / "declaration.json"
    path.write_text(json.dumps(declaration()))
    results = iter(
        [
            [("datos", True, "bien"), ("permiso", None, "pendiente")],
            [("datos", False, "faltan")],
        ]
    )
    monkeypatch.setattr(campaign, "doctor", lambda *_, **__: next(results))
    arguments = ["doctor", "--machine", "references", "--declaration", str(path)]
    arguments += ["--state", str(tmp_path / "state")]
    assert campaign.main(arguments) == 0
    assert capsys.readouterr().out.splitlines() == ["bien  datos: bien", "aviso permiso: pendiente"]
    assert campaign.main(arguments) == 2
    assert capsys.readouterr().out.splitlines() == ["FALTA datos: faltan"]


def test_queue_starts_every_model_once_before_emptying_the_queue(tmp_path, monkeypatch, capsys):
    instance, place, executed = queue(tmp_path, monkeypatch, workers=1)
    result = instance.run()
    order = executed.read_text().split()
    # Primero, el primer trabajo de cada modelo que no es un lector, en el orden del plan.
    assert order[:2] == ["titans_mac/2019/seed-42!preflight", "area_network/2019/seed-42!preflight"]
    assert order[2:] == [job.key for job in instance.jobs]
    assert result["by_tier"] == dict(core={DONE: 8}, areas={DONE: 4})
    record = campaign.read_record(place, "titans_mac/2019/seed-42")
    assert [item["status"] for item in record["attempts"]] == ["paused", "completed"]
    assert record["attempts"][0]["preflight"] is True
    lines = capsys.readouterr().out.splitlines()
    assert lines == [
        "comprobación inicial bien: titans_mac/2019/seed-42",
        "comprobación inicial bien: area_network/2019/seed-42",
    ]
    log = place.log("titans_mac/2019/seed-42").read_text()
    assert log.count("(comprobación inicial)") == 1 and log.count("===") == 2
    # Una cola que ya arrancó sus modelos no repite la comprobación.
    again, _, executed = queue(tmp_path, monkeypatch, workers=1)
    assert again.preflight() == [] and len(executed.read_text().split()) == 14


def test_preflight_does_not_launch_a_job_already_locked_by_another_queue(tmp_path, monkeypatch):
    instance, place, executed = queue(tmp_path, monkeypatch, workers=1)
    parent = instance.jobs[0]
    claimed, handle = campaign._claim(place, [parent], set())
    assert claimed == parent
    try:
        instance.preflight()
        assert not any(parent.key in line for line in executed.read_text().splitlines())
        assert campaign.read_record(place, parent.key)["attempts"] == []
    finally:
        handle.close()


def test_a_model_that_cannot_start_is_reported_at_once_and_the_rest_continue(
    tmp_path, monkeypatch, capsys
):
    instance, place, executed = queue(tmp_path, monkeypatch, workers=1, mode="broken-model")
    result = instance.run()
    lines = capsys.readouterr().out.splitlines()
    assert lines == [
        "comprobación inicial bien: titans_mac/2019/seed-42",
        "comprobación inicial FALLA: area_network/2019/seed-42",
        "1 modelos no arrancan. Revisa sus registros",
    ]
    assert result["by_tier"] == dict(core={DONE: 8}, areas={FAILED: 4})
    record = campaign.read_record(place, "area_network/2019/seed-42")
    assert len(record["attempts"]) == campaign.MAX_ATTEMPTS and record["status"] == FAILED
    order = executed.read_text().split()
    assert order.count("area_network/2019/seed-42!preflight") == 1
    assert order.count("area_network/2019/seed-42") == campaign.MAX_ATTEMPTS - 1


def test_job_command_marks_the_initial_check_and_the_stop_reaches_running_children(
    tmp_path, monkeypatch
):
    document, place = campaign.checked_declaration(declaration()), layout(tmp_path)
    instance = campaign.Queue(document, place, "mars_titan", declaration_path=tmp_path / "d.json")
    job = instance.jobs[0]
    plain, checked = instance._command(job), instance._command(job, preflight=True)
    assert plain[1:4] == ["-m", "mars_titan.training.comparison_campaign", "job"]
    assert plain[4:] == ["--declaration", str(tmp_path / "d.json"), "--machine", "mars_titan"] + [
        "--key",
        job.key,
    ]
    assert checked == [*plain, "--preflight"]
    assert "--declaration" not in campaign.Queue(document, place, "mars_titan")._command(job)

    class Child:
        def __init__(self, code):
            self.code, self.signals = code, []

        def poll(self):
            return self.code

        def send_signal(self, value):
            self.signals.append(value)

    running, finished = Child(None), Child(0)
    instance.children = dict(a=running, b=finished)
    instance.request_stop()
    assert instance.stop.is_set()
    assert running.signals == [campaign.signal.SIGTERM] and finished.signals == []


def test_each_model_uses_the_rule_of_its_runner_with_its_own_epochs(tmp_path):
    document = declaration()
    document["models"]["area_network"]["epochs"] = 5
    document["models"]["mars_titan_m1"]["epochs"] = 3
    document = campaign.checked_declaration(document)
    models = document["models"]
    assert campaign.job_rule(document, models["titans_mac"]) == dict(RULE, initial_state="reported")
    assert campaign.job_rule(document, models["area_network"]) == dict(
        RULE, initial_state="reported", max_epochs=5
    )
    assert campaign.job_rule(document, models["mars_titan_m1"]) == dict(RULE, max_epochs=3)
    assert document["stopping"]["fit"]["max_epochs"] == 2
    for name, value in (("area_network", 0), ("area_network", "5"), ("mars_titan_m1", 1001)):
        broken = declaration()
        broken["models"][name]["epochs"] = value
        with pytest.raises(ValueError, match="máximo de épocas"):
            campaign.checked_declaration(broken)
    broken = declaration()
    broken["models"]["rule"]["epochs"] = 3
    with pytest.raises(ValueError, match="no declara sus campos"):
        campaign.checked_declaration(broken)


def test_the_permission_is_bound_to_the_commit_that_runs_the_campaign(tmp_path, monkeypatch):
    place = layout(tmp_path)
    commit = campaign.code_commit()
    assert commit is not None and len(commit) == 40 and int(commit, 16) >= 0
    assert campaign.code_commit(tmp_path) is None
    monkeypatch.setattr(campaign.subprocess, "run", lambda *_, **__: (_ for _ in ()).throw(OSError))
    assert campaign.code_commit() is None
    changes = "c" * 64
    document = campaign.authorize(place, "a" * 64, note="Aprobado", commit=commit, changes=changes)
    assert document["code_commit"] == commit
    assert document["code_changes_sha256"] == changes
    assert campaign.authorized(place, "a" * 64, commit=commit, changes=changes)
    assert not campaign.authorized(place, "a" * 64, commit=commit, changes="d" * 64)
    assert not campaign.authorized(place, "a" * 64, commit=commit)
    # Otro commit, o no poder leerlo, pide un permiso nuevo.
    assert not campaign.authorized(place, "a" * 64, commit="f" * 40)
    assert not campaign.authorized(place, "a" * 64)
    blocked = json.loads(place.authorization.read_text())
    blocked["training_allowed"] = False
    place.authorization.write_text(json.dumps(blocked))
    assert not campaign.authorized(place, "a" * 64, commit=commit)


@pytest.mark.parametrize(
    "added_path", ["src/added.py", "pyproject.toml", "uv.lock", ".python-version"]
)
def test_code_identity_includes_tracked_and_untracked_executable_files(tmp_path, added_path):
    def git(*args):
        return subprocess.run(["git", "-C", str(tmp_path), *args], check=True, capture_output=True)

    git("init")
    source = tmp_path / "src/model.py"
    source.parent.mkdir()
    source.write_text("x = 1\n")
    git("add", "src")
    git(
        "-c",
        "user.name=prueba",
        "-c",
        "user.email=prueba@example.invalid",
        "commit",
        "-m",
        "test: initialize code identity fixture",
    )
    assert campaign.code_changes(tmp_path) is None
    source.write_text("x = 2\n")
    first = campaign.code_changes(tmp_path)
    assert first is not None and len(first) == 64
    source.write_text("x = 3\n")
    assert campaign.code_changes(tmp_path) != first
    source.write_text("x = 1\n")
    (tmp_path / "notes.md").write_text("documentación local\n")
    assert campaign.code_changes(tmp_path) is None
    added = tmp_path / added_path
    added.write_text("value = 1\n")
    first = campaign.code_changes(tmp_path)
    assert first is not None
    added.write_text("value = 2\n")
    assert campaign.code_changes(tmp_path) != first
    git("add", added_path)
    git(
        "-c",
        "user.name=prueba",
        "-c",
        "user.email=prueba@example.invalid",
        "commit",
        "-m",
        "test: track dependency fixture",
    )
    assert campaign.code_changes(tmp_path) is None
    added.write_text("value = 3\n")
    assert campaign.code_changes(tmp_path) is not None


def test_unreadable_code_identity_cannot_be_reported_as_clean(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="identidad del código"):
        campaign.code_changes(tmp_path)
    monkeypatch.setattr(campaign, "code_commit", lambda: None)
    monkeypatch.setattr(campaign, "code_changes", lambda: None)
    name, passed, detail = campaign._code_check("a" * 64)
    assert name == "código y declaración" and passed is False
    assert "desconocido" in detail


def test_doctor_reports_local_changes_in_the_code_identity(monkeypatch):
    monkeypatch.setattr(campaign, "code_commit", lambda: "f" * 40)
    monkeypatch.setattr(campaign, "code_changes", lambda: None)
    assert campaign._code_check("a" * 64) == (
        "código y declaración",
        True,
        "commit ffffffffffff, declaración aaaaaaaaaaaa, sin cambios locales",
    )
    monkeypatch.setattr(campaign, "code_changes", lambda: "b" * 64)
    _, passed, detail = campaign._code_check("a" * 64)
    assert passed is None and "con cambios sin confirmar" in detail


def test_execute_job_gives_each_runner_its_window_rule_and_parent(tmp_path, monkeypatch):
    from mars_titan.training import common_control_windows as controls
    from mars_titan.training import comparison_ridge, mars_titan_walk_forward, titans_walk_forward

    document = declaration()
    document["models"]["area_network"]["epochs"] = 5
    document["models"]["ridge"] = RIDGE
    document["models"]["lightgbm"] = dict(
        machine="references",
        tier="core",
        runner="tabular",
        config=dict(identity="x"),
        fit_months=12,
        radius=1,
        min_assets=20,
    )
    document, place = campaign.checked_declaration(document), layout(tmp_path)
    job = campaign.plan(document, "mars_titan")[0]
    with pytest.raises(ValueError, match="Falta la cohorte de 2019"):
        campaign.execute_job(document, place, job)
    for year in (2019, 2023):
        (place.cohort(year)).mkdir(parents=True)
        (place.cohort(year) / "manifest.json").write_text("{}")
    calls = []

    def recorder(name):
        def call(*arguments, **options):
            stop = options.get("stop")
            calls.append((name, arguments, options, bool(stop is not None and stop.requested)))
            return dict(status="paused" if stop is not None and stop.requested else "completed")

        return call

    monkeypatch.setattr(titans_walk_forward, "view_protocol", lambda view: dict(view=str(view)))
    monkeypatch.setattr(titans_walk_forward, "run_titans_window", recorder("fit"))
    monkeypatch.setattr(mars_titan_walk_forward, "run_mars_titan_window", recorder("reader"))
    monkeypatch.setattr(controls, "run_rule_window", recorder("rule"))
    monkeypatch.setattr(controls, "run_tabular_window", recorder("tabular"))
    monkeypatch.setattr(comparison_ridge, "run_ridge_window", recorder("ridge"))
    jobs = {
        job.key: job for machine in campaign.MACHINES for job in campaign.plan(document, machine)
    }
    view = place.cohort(2019) / "manifest.json"

    assert campaign.execute_job(document, place, jobs["area_network/2019/seed-43"], device="cpu")
    name, arguments, options, stopped = calls.pop()
    assert name == "fit" and not stopped
    assert arguments == (view, dict(view=str(view)), "fold-014", campaign.ROOT / RECIPE)
    assert options["variant"] == "mac_online" and options["seed"] == 43
    assert options["device"] == "cpu"
    assert options["output"] == place.output("area_network/2019/seed-43")
    assert options["stopping"] == dict(RULE, initial_state="reported", max_epochs=5)
    assert options["predictor_changes"] == dict(holographic_network=dict(areas=19))

    campaign.execute_job(document, place, jobs["transformer/2023/seed-42"])
    name, arguments, options, _ = calls.pop()
    assert arguments[2] == "fold-018" and options["predictor_changes"] is None
    assert options["stopping"] == dict(RULE, initial_state="reported")

    campaign.execute_job(document, place, jobs["mars_titan_m1/2019/seed-42"])
    name, arguments, options, stopped = calls.pop()
    assert name == "reader" and not stopped
    assert arguments == (view, place.output("titans_mac/2019/seed-42"), campaign.ROOT / READER)
    assert options["components"] == dict(episodic_bank="m1") and options["search_case"] == "lr1e-3"
    assert options["stopping"] == dict(RULE) and options["seed"] == 42

    campaign.execute_job(document, place, jobs["rule/2019/all-seeds"])
    name, arguments, options, stopped = calls.pop()
    assert name == "rule" and arguments[2] == "fold-014" and not options["stop"].requested
    assert options["config"] == dict(identity="momentum_reversal_v1")

    campaign.execute_job(document, place, jobs["lightgbm/2023/all-seeds"])
    name, arguments, options, _ = calls.pop()
    assert name == "tabular" and options["seeds"] == [42, 43]
    assert options["max_new_fits"] == 1
    assert (options["fit_months"], options["radius"], options["min_assets"]) == (12, 1, 20)

    campaign.execute_job(document, place, jobs["ridge/2023/all-seeds"])
    name, arguments, options, _ = calls.pop()
    assert name == "ridge" and arguments == (place.cohort(2023) / "manifest.json",)
    assert not options.pop("stop").requested
    assert options == dict(output=place.output("ridge/2023/all-seeds"), config=RIDGE["config"])

    # La comprobación inicial pide la parada antes de empezar en los ejecutores sin padre.
    for key in (
        "titans_mac/2019/seed-42",
        "rule/2019/all-seeds",
        "lightgbm/2019/all-seeds",
        "ridge/2019/all-seeds",
    ):
        report = campaign.execute_job(document, place, jobs[key], preflight=True)
        assert report == dict(status="paused") and calls.pop()[3] is True
    assert not calls


@pytest.mark.parametrize("runner", ["rule", "ridge", "tabular"])
def test_origin_controls_receive_a_cooperative_stop_signal(tmp_path, monkeypatch, runner):
    from mars_titan.training import common_control_windows, comparison_ridge, titans_walk_forward

    document = declaration()
    document["models"]["ridge"] = RIDGE
    document["models"]["lightgbm"] = dict(
        machine="references",
        tier="core",
        runner="tabular",
        config=dict(identity="x"),
        fit_months=12,
        radius=1,
        min_assets=3,
    )
    document, place = campaign.checked_declaration(document), layout(tmp_path)
    job = next(job for job in campaign.plan(document, "references") if job.runner == runner)
    place.cohort(job.year).mkdir(parents=True)
    (place.cohort(job.year) / "manifest.json").write_text("{}")
    monkeypatch.setattr(titans_walk_forward, "view_protocol", lambda _: {})
    previous = campaign.signal.getsignal(campaign.signal.SIGTERM)

    def stopped(*_, stop=None, **__):
        assert stop is not None and not stop.requested
        handler = campaign.signal.getsignal(campaign.signal.SIGTERM)
        assert callable(handler)
        handler(campaign.signal.SIGTERM, None)
        assert stop.requested
        return dict(status="paused")

    monkeypatch.setattr(common_control_windows, "run_rule_window", stopped)
    monkeypatch.setattr(common_control_windows, "run_tabular_window", stopped)
    monkeypatch.setattr(comparison_ridge, "run_ridge_window", stopped)
    assert campaign.execute_job(document, place, job)["status"] == "paused"
    assert campaign.signal.getsignal(campaign.signal.SIGTERM) == previous
