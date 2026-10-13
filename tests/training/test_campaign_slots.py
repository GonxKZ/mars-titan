"""Ranuras de la campaña: concurrencia acotada, salidas iguales, parada y recuperación.

Los trabajos son los dobles de `slot_doubles`, que escriben filas nulas sin ajustar nada ni
usar CUDA. Cada trabajo GPU de una ejecución con varias ranuras corre en su proceso.
"""

import fcntl
import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from mars_titan.training import masked_campaign as engine
from mars_titan.training.campaign_plan import load_campaign, plan_campaign
from mars_titan.training.campaign_resources import (
    CONTEXT_MIB,
    MAX_OOM_RETRIES,
    Execution,
    JobResources,
    PeakEstimates,
    ResourcePool,
    check_plan,
    load_execution,
)
from mars_titan.training.campaign_slots import (
    SlotProcess,
    SlotTask,
    bound_vram,
    executor_name,
    new_event,
    parse_gpu_processes,
    wait,
)
from mars_titan.training.input_pipeline import PipelineOptions
from tests.hardware.platform_doubles import laptop
from tests.training import slot_doubles
from tests.training.test_masked_campaign import prepared, write_campaign  # noqa: F401

pytestmark = pytest.mark.usefixtures("learning_doubles")
GIB = 1024**3


def executors(cpu=()):
    return {
        key: dict(
            entry,
            run=slot_doubles.record_job,
            device="cpu" if key in cpu else entry["device"],
        )
        for key, entry in engine.EXECUTORS.items()
    }


def campaign_file(tmp_path):
    return write_campaign(tmp_path / "config", arms=("gru", "ridge"))


def log(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def slots(count, **fields):
    return Execution(
        gpu_slots=count,
        cpu_workers=fields.pop("cpu_workers", 1),
        threads_per_job=2,
        pipeline=PipelineOptions(decode_workers=2, prefetch_batches=2),
        **fields,
    )


def execute(campaign, views, output, execution, stop=None, cpu=()):
    return engine.run_campaign(
        campaign,
        views,
        output,
        executors=executors(cpu),
        lease=nullcontext,
        stop=stop or SimpleNamespace(requested=False),
        execution=execution,
    )


def receipts(output):
    """Recibos confirmados sin fechas ni huellas de otros recibos, que incluyen su fecha."""
    result = {}
    for path in sorted(Path(output, "jobs").rglob("receipt.json")):
        value = json.loads(path.read_text())
        identity = dict(value["identity"])
        sources = identity.pop("sources")
        result[identity["id"]] = (
            identity,
            sources.get("source"),
            sources.get("parent"),
            value["score"],
            {k: (v["sha256"], v["rows_sha256"]) for k, v in value["predictions"].items()},
        )
    return result


def overlapped(entries):
    spans = sorted((e["start"], e["end"], e["pid"]) for e in entries if e["event"] == "completed")
    return any(b[0] < a[1] and a[2] != b[2] for a, b in zip(spans, spans[1:], strict=False))


def test_slots_run_jobs_concurrently_with_the_same_receipts(prepared, tmp_path, monkeypatch):  # noqa: F811
    campaign = campaign_file(tmp_path)
    views = {"US": prepared.views["US"]}
    serial = execute(campaign, views, tmp_path / "serial", slots(1))
    assert serial["status"] == "completed"
    monkeypatch.setenv("SLOT_LOG", str(tmp_path / "slots.jsonl"))
    monkeypatch.setenv("SLOT_DELAY", "0.2")
    summary = execute(campaign, views, tmp_path / "slots", slots(3), cpu={("ridge", "fit")})
    assert summary["status"] == "completed"
    assert summary["completed"] == summary["planned"] == serial["planned"]
    assert receipts(tmp_path / "slots") == receipts(tmp_path / "serial")
    entries = log(tmp_path / "slots.jsonl")
    planned = [job["id"] for job in plan_campaign(load_campaign(campaign))]
    gpu = [e for e in entries if not ("/ridge/" in e["id"] and "/carry-" not in e["id"])]
    assert sorted(e["id"] for e in entries) == sorted(planned)
    assert overlapped(gpu)
    # Nunca más procesos simultáneos que ranuras GPU.
    events = sorted([(e["start"], 1) for e in gpu] + [(e["end"], -1) for e in gpu])
    active = peak = 0
    for _, change in events:
        active += change
        peak = max(peak, active)
    assert peak <= 3
    assert {e["environment"]["MARS_TITAN_DECODE_WORKERS"] for e in gpu} == {"2"}
    assert {e["environment"]["OMP_NUM_THREADS"] for e in gpu} == {"2"}
    # Cada proceso de una ranura trabaja en FP32 estricto, sin TF32.
    assert all(e["environment"]["tf32"] == [False, False] for e in gpu)
    assert summary["execution"]["gpu_slots"] == 3
    assert set(summary["usage"]) == {e["id"] for e in gpu}


def test_pause_with_slots_resumes_every_job_from_its_own_attempt(prepared, tmp_path, monkeypatch):  # noqa: F811
    campaign = campaign_file(tmp_path)
    views = {"US": prepared.views["US"]}
    stop_file = tmp_path / "stop"
    monkeypatch.setenv("SLOT_LOG", str(tmp_path / "slots.jsonl"))
    monkeypatch.setenv("SLOT_DELAY", "0.3")
    monkeypatch.setenv("SLOT_PAUSE_AFTER", "3")
    monkeypatch.setenv("SLOT_STOP_FILE", str(stop_file))
    output = tmp_path / "out"
    first = execute(campaign, views, output, slots(3), stop=slot_doubles.FileStop(stop_file))
    assert first["status"] == "paused"
    entries = log(tmp_path / "slots.jsonl")
    paused = {e["id"] for e in entries if e["event"] == "paused"}
    done = {e["id"] for e in entries if e["event"] == "completed"}
    assert paused and len(done) >= 3 and not paused & done
    stop_file.unlink()
    monkeypatch.delenv("SLOT_PAUSE_AFTER")
    monkeypatch.delenv("SLOT_DELAY")
    second = execute(campaign, views, output, slots(2))
    assert second["status"] == "completed"
    entries = log(tmp_path / "slots.jsonl")
    completed = [e["id"] for e in entries if e["event"] == "completed"]
    assert len(completed) == len(set(completed)) == len(plan_campaign(load_campaign(campaign)))
    # Los trabajos reanudables siguen en su intento, con el archivo parcial de la pausa.
    resumed = {e["id"] for e in entries if e["event"] == "completed" and e["resumed"]}
    assert {job for job in paused if "/carry-" not in job} <= resumed
    serial = execute(campaign, views, tmp_path / "serial", slots(1))
    assert serial["status"] == "completed"
    assert receipts(output) == receipts(tmp_path / "serial")


def test_a_crashed_slot_stops_the_campaign_after_draining_the_others(
    prepared,  # noqa: F811
    tmp_path,
    monkeypatch,
):
    campaign = campaign_file(tmp_path)
    views = {"US": prepared.views["US"]}
    planned = [job["id"] for job in plan_campaign(load_campaign(campaign))]
    monkeypatch.setenv("SLOT_LOG", str(tmp_path / "slots.jsonl"))
    monkeypatch.setenv("SLOT_DELAY", "0.2")
    monkeypatch.setenv("SLOT_CRASH", planned[3])
    with pytest.raises(RuntimeError, match="sin resultado"):
        execute(campaign, views, tmp_path / "out", slots(2))
    summary = json.loads((tmp_path / "out/summary.json").read_text())
    assert summary["status"] == "failed" and not summary["jobs"][planned[3]]
    monkeypatch.delenv("SLOT_CRASH")
    assert execute(campaign, views, tmp_path / "out", slots(2))["status"] == "completed"


def test_pool_admits_by_slots_vram_and_host_memory():
    execution = Execution(
        gpu_slots=2,
        cpu_workers=1,
        vram_budget_bytes=3 * GIB,
        host_budget_bytes=10 * GIB,
        host_reserve_bytes=GIB,
    )
    free = [32 * GIB]
    pool = ResourcePool(execution, available=lambda: free[0])
    small, large = JobResources("cuda", GIB, 2 * GIB), JobResources("cuda", 2 * GIB, 2 * GIB)
    pool.acquire(large)
    assert pool.admits(small) and not pool.admits(large)
    pool.acquire(small)
    # Las dos ranuras GPU ya están ocupadas.
    assert not pool.admits(small)
    cpu = JobResources("cpu", 0, 5 * GIB)
    assert pool.admits(cpu)
    pool.acquire(cpu)
    # El único trabajador CPU ya está ocupado.
    assert not pool.admits(JobResources("cpu", 0, GIB))
    pool.release(small)
    # La RAM declarada en curso (7 GiB) más la nueva (4 GiB) supera los 10 GiB.
    assert not pool.admits(JobResources("cuda", GIB, 4 * GIB))
    assert pool.admits(JobResources("cuda", GIB, GIB))
    free[0] = GIB + GIB // 2
    # La memoria disponible ya no cubre la reserva del anfitrión.
    assert not pool.admits(JobResources("cuda", GIB, GIB))


def write_execution(path, **changes):
    document = json.loads(
        Path("configs/baselines/historical-masked-campaign-execution.json").read_text()
    )
    for key, value in changes.items():
        document[key] = document[key] | value if isinstance(value, dict) else value
    path.write_text(json.dumps(document))
    return path


def test_declared_execution_matches_the_campaign_plan():
    campaign = load_campaign("configs/baselines/historical-masked-campaign-a.json")
    execution = load_execution("configs/baselines/historical-masked-campaign-execution.json")
    jobs = plan_campaign(campaign)
    check_plan(execution, jobs, engine.EXECUTORS)
    assert {job["model"] for job in jobs} <= set(execution.models)
    assert execution.gpu_slots >= 1 and execution.pipeline.decode_workers >= 1


@pytest.mark.parametrize(
    "changes",
    [
        dict(gpu=dict(slots=0)),
        dict(gpu=dict(slots=5)),
        dict(gpu=dict(slots=1, mps=True)),
        dict(host=dict(cpu_workers=9)),
        dict(pipeline=dict(decode_workers=17)),
        dict(kind="otra"),
        dict(
            models=dict(neural=dict(device="cuda", vram_mib=1024, host_mib=4096, arms=dict(gru={})))
        ),
        dict(
            models=dict(
                neural=dict(device="cuda", vram_mib=1024, host_mib=4096, arms=dict(gru=dict(lr=1)))
            )
        ),
        dict(
            models=dict(
                neural=dict(
                    device="cuda",
                    vram_mib=1024,
                    host_mib=4096,
                    arms=dict(gru=dict(scopes=dict(EU=dict(vram_mib=512)))),
                )
            )
        ),
    ],
)
def test_invalid_execution_is_rejected(tmp_path, changes):
    with pytest.raises(ValueError):
        load_execution(write_execution(tmp_path / "execution.json", **changes))


def test_the_declaration_names_the_profile_it_was_measured_on():
    execution = load_execution("configs/baselines/historical-masked-campaign-execution.json")
    assert execution.hardware_profile["name"] == "rtx4070-laptop"
    assert execution.record()["hardware_profile"] == dict(
        name="rtx4070-laptop", sha256=execution.hardware_profile["sha256"]
    )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        (dict(schema_version=1), "versión 2"),
        (dict(hardware_profile=None), "nombre del perfil"),
        (dict(hardware_profile="missing"), "No existe"),
        # 7.681 MiB de VRAM superan los 7.680 del perfil de la RTX 4070.
        (dict(gpu=dict(vram_budget_mib=7681)), "rtx4070-laptop"),
        (dict(host=dict(ram_budget_mib=24577)), "rtx4070-laptop"),
        # Tres ranuras y un trabajador CPU con 5 hilos son 20 hilos de los 16 del perfil.
        (dict(host=dict(threads_per_job=5)), "20 hilos"),
    ],
)
def test_budgets_must_fit_the_declared_profile(tmp_path, changes, message):
    with pytest.raises(ValueError, match=message):
        load_execution(write_execution(tmp_path / "execution.json", **changes))


def test_a_declaration_for_another_machine_stops_the_campaign_before_writing(
    prepared,  # noqa: F811
    tmp_path,
):
    # Los presupuestos de la RTX 4070 caben en la GB10, pero la plataforma no es la suya.
    execution = load_execution(
        write_execution(tmp_path / "execution.json", hardware_profile="dgx-gb10")
    )
    output = tmp_path / "out"
    with pytest.raises(ValueError, match="no corresponde al perfil dgx-gb10"):
        engine.run_campaign(
            campaign_file(tmp_path),
            {"US": prepared.views["US"]},
            output,
            executors=executors(),
            lease=nullcontext,
            stop=SimpleNamespace(requested=False),
            execution=execution,
            platform=laptop(),
        )
    assert not output.exists()


def test_job_that_cannot_fit_alone_is_rejected_before_running(tmp_path):
    path = write_execution(tmp_path / "execution.json", gpu=dict(vram_budget_mib=512))
    campaign = load_campaign("configs/baselines/historical-masked-campaign-a.json")
    with pytest.raises(ValueError, match="supera el presupuesto"):
        check_plan(load_execution(path), plan_campaign(campaign), engine.EXECUTORS)


def test_a_plan_out_of_dependency_order_is_rejected():
    jobs = plan_campaign(load_campaign("configs/baselines/historical-masked-campaign-a.json"))
    check_plan(Execution(), jobs, engine.EXECUTORS)
    index = next(i for i, job in enumerate(jobs) if job["depends"])
    dependency = next(i for i, job in enumerate(jobs) if job["id"] == jobs[index]["depends"][0])
    jobs.insert(index + 1, jobs.pop(dependency))
    with pytest.raises(ValueError, match="antes que alguna de sus dependencias"):
        check_plan(Execution(), jobs, engine.EXECUTORS)


def test_device_must_agree_with_the_executor(tmp_path):
    document = json.loads(
        Path("configs/baselines/historical-masked-campaign-execution.json").read_text()
    )
    document["models"]["neural"] = dict(device="cpu", host_mib=1024)
    path = tmp_path / "execution.json"
    path.write_text(json.dumps(document))
    campaign = load_campaign("configs/baselines/historical-masked-campaign-a.json")
    with pytest.raises(ValueError, match="su ejecutor usa"):
        check_plan(load_execution(path), plan_campaign(campaign), engine.EXECUTORS)


def test_a_result_sent_just_before_the_process_exits_is_not_lost():
    class Child:
        alive, exitcode = True, None

        def is_alive(self):
            return self.alive

        def join(self):
            pass

    class Receiver:
        sent = False

        def __init__(self, child):
            self.child = child

        def poll(self):
            ready = self.sent
            # El hijo envía su resultado y termina justo después de esta comprobación.
            self.sent, self.child.alive, self.child.exitcode = True, False, 0
            return ready

        def recv(self):
            return "completed", {}, {}

        def close(self):
            pass

    handle = SlotProcess.__new__(SlotProcess)
    handle.process = Child()
    handle.receiver, handle.result = Receiver(handle.process), None
    assert handle.poll() is None
    assert handle.poll() == ("completed", {}, {})


def test_every_campaign_executor_can_run_in_a_slot():
    # Una ranura importa su ejecutor por nombre en un proceso nuevo, así que ninguno de los
    # ejecutores de la campaña puede ser un cierre.
    for entry in engine.EXECUTORS.values():
        assert executor_name(entry["run"])


def test_tabular_state_is_released_before_each_launch(prepared, tmp_path, monkeypatch):  # noqa: F811
    campaign = campaign_file(tmp_path)
    views = {"US": prepared.views["US"]}
    released = []
    monkeypatch.setattr(engine, "_release_tabular", lambda *key: released.append(key))
    summary = execute(campaign, views, tmp_path / "out", slots(2), cpu={("ridge", "fit")})
    assert summary["status"] == "completed"
    planned = plan_campaign(load_campaign(campaign))
    assert sorted(released) == sorted((job["model"], job["kind"]) for job in planned)


def test_a_job_locked_by_another_process_is_not_run_twice(tmp_path):
    lock = tmp_path / ".job.lock"
    with lock.open("a") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        task = SlotTask(
            executor="tests.training.slot_doubles:record_job",
            run={},
            environment={},
            vram_bytes=0,
            lock=str(lock),
        )
        handle = SlotProcess(task, new_event())
        while handle.poll() is None:
            wait([handle], 1)
    status, error = handle.result[:2]
    assert status == "failed" and "otro proceso" in error["message"]


def test_the_declared_vram_bound_cannot_be_widened_by_the_job(monkeypatch):
    import torch

    calls = []
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda _: SimpleNamespace(total_memory=8 * GIB)
    )
    monkeypatch.setattr(
        torch.cuda,
        "set_per_process_memory_fraction",
        lambda value, device=None: calls.append(value),
    )
    assert bound_vram(2 * GIB) == 0.25
    torch.cuda.set_per_process_memory_fraction(0.9)
    torch.cuda.set_per_process_memory_fraction(0.1)
    assert calls == [0.25, 0.25, 0.1]


MIB = 1024**2


def test_a_job_that_runs_out_of_vram_is_repeated_alone_with_more(prepared, tmp_path, monkeypatch):  # noqa: F811
    campaign = campaign_file(tmp_path)
    views = {"US": prepared.views["US"]}
    planned = [job["id"] for job in plan_campaign(load_campaign(campaign))]
    marks = tmp_path / "marks"
    marks.mkdir()
    monkeypatch.setenv("SLOT_LOG", str(tmp_path / "slots.jsonl"))
    monkeypatch.setenv("SLOT_OOM", planned[1])
    monkeypatch.setenv("SLOT_MARKS", str(marks))
    execution = slots(2, vram_budget_bytes=4096 * MIB)
    summary = execute(campaign, views, tmp_path / "out", execution, cpu={("ridge", "fit")})
    assert summary["status"] == "completed"
    entries = [e for e in log(tmp_path / "slots.jsonl") if e["id"] == planned[1]]
    assert [e["event"] for e in entries] == ["oom", "completed"]
    first, second = (int(e["environment"]["MARS_TITAN_SLOT_VRAM_MIB"]) for e in entries)
    assert second >= first + 1024
    observed = json.loads((tmp_path / "out" / engine.OBSERVED_RESOURCES).read_text())
    job = next(job for job in plan_campaign(load_campaign(campaign)) if job["id"] == planned[1])
    assert observed["observed"][PeakEstimates.key(job)]["oom"] == 1
    assert execute(campaign, views, tmp_path / "serial", slots(1))["status"] == "completed"
    assert receipts(tmp_path / "out") == receipts(tmp_path / "serial")


def test_estimates_follow_the_observed_peak_within_the_budget(tmp_path):
    execution = Execution(
        gpu_slots=2,
        vram_budget_bytes=4096 * MIB,
        models={"neural": dict(default=JobResources("cuda", 1024 * MIB, GIB), scopes={}, arms={})},
    )
    job = dict(id="US/fold-000/gru/search-gru-00", model="neural", scope="US")
    path = tmp_path / "observed.json"
    estimates = PeakEstimates(execution, path)
    assert estimates.resources(job, "cuda").vram_bytes == 1024 * MIB
    estimates.observe(job, dict(peak_vram_reserved_bytes=256 * MIB))
    # Un pico menor que lo declarado no baja la estimación.
    assert estimates.resources(job, "cuda").vram_bytes == 1024 * MIB
    estimates.observe(job, dict(peak_vram_reserved_bytes=1536 * MIB))
    expected = (1536 + CONTEXT_MIB) * MIB
    assert PeakEstimates(execution, path).resources(job, "cuda").vram_bytes == expected
    current = estimates.resources(job, "cuda")
    for _ in range(MAX_OOM_RETRIES):
        assert estimates.grow(job, current)
        current = estimates.resources(job, "cuda")
    # La estimación crece hasta el presupuesto y no lo supera.
    assert current.vram_bytes == 4096 * MIB
    assert not estimates.grow(job, current)


def test_an_observed_peak_keeps_the_campaign_process(tmp_path):
    exclusive = JobResources("cuda", 2048 * MIB, GIB, campaign_process=True)
    execution = Execution(
        gpu_slots=2,
        vram_budget_bytes=4096 * MIB,
        models={"xgboost": dict(default=exclusive, scopes={}, arms={})},
    )
    job = dict(id="US/fold-000/xgboost/search-xgboost-00", model="xgboost", scope="US")
    estimates = PeakEstimates(execution, tmp_path / "observed.json")
    estimates.observe(job, dict(peak_vram_reserved_bytes=3072 * MIB))
    assert estimates.resources(job, "cuda") == JobResources(
        "cuda", 3584 * MIB, GIB, campaign_process=True
    )


def test_each_arm_corrects_only_its_own_estimate(tmp_path):
    declared = JobResources("cuda", 1792 * MIB, GIB)
    execution = Execution(
        gpu_slots=2,
        vram_budget_bytes=7000 * MIB,
        models={"titans_mac": dict(default=declared, scopes={}, arms={})},
    )
    online = dict(id="a", model="titans_mac", arm="titans_mac_online", scope="US")
    frozen = dict(online, id="b", arm="titans_mac_frozen")
    estimates = PeakEstimates(execution, tmp_path / "observed.json")
    estimates.observe(online, dict(peak_vram_reserved_bytes=5000 * MIB))
    assert estimates.resources(online, "cuda").vram_bytes == (5000 + CONTEXT_MIB) * MIB
    # El pico de un brazo grande no impide empaquetar los pequeños del mismo modelo.
    assert estimates.resources(frozen, "cuda") == declared


def test_arms_override_their_model_in_every_scope(tmp_path):
    document = json.loads(
        Path("configs/baselines/historical-masked-campaign-execution.json").read_text()
    )
    titans = document["models"]["titans_mac"]
    # La prueba fija sus propios recursos de modelo para no depender de las cifras medidas.
    titans.update(vram_mib=1792, scopes=dict(CN=dict(vram_mib=1280, host_mib=5120)))
    titans["arms"] = dict(
        titans_mac_online=dict(vram_mib=6000, scopes=dict(CN=dict(host_mib=4096)))
    )
    path = tmp_path / "execution.json"
    path.write_text(json.dumps(document))
    execution = load_execution(path)

    def resources(arm, scope):
        job = dict(id="x", model="titans_mac", arm=arm, scope=scope)
        return execution.resources(job, "cuda")

    # La VRAM del brazo sustituye a la del modelo en todos los ámbitos y su RAM solo en CN.
    assert resources("titans_mac_online", "US+CN") == JobResources("cuda", 6000 * MIB, 9 * GIB)
    assert resources("titans_mac_online", "CN") == JobResources("cuda", 6000 * MIB, 4 * GIB)
    # Los demás brazos conservan lo del modelo en cada ámbito.
    assert resources("titans_mac_frozen", "CN") == JobResources("cuda", 1280 * MIB, 5 * GIB)
    campaign = load_campaign("configs/baselines/historical-masked-campaign-a.json")
    check_plan(execution, plan_campaign(campaign), engine.EXECUTORS)
    for arms, message in (
        (dict(titans_mac_paper=dict(vram_mib=1024)), "brazos que no tiene"),
        (dict(titans_mac_online=dict(vram_mib=8000)), "supera el presupuesto"),
    ):
        titans["arms"] = arms
        path.write_text(json.dumps(document))
        with pytest.raises(ValueError, match=message):
            check_plan(load_execution(path), plan_campaign(campaign), engine.EXECUTORS)


def test_slots_launch_the_largest_ready_jobs_first():
    execution = Execution(
        gpu_slots=3,
        models={
            "neural": dict(default=JobResources("cuda", 512 * MIB, GIB), scopes={}, arms={}),
            "titans_mac": dict(default=JobResources("cuda", 1792 * MIB, GIB), scopes={}, arms={}),
        },
    )
    jobs = [
        dict(id=str(index), model=model, kind="fit", scope="US")
        for index, model in enumerate(["neural", "titans_mac", "neural", "titans_mac"])
    ]
    executors = {(m, "fit"): dict(device="cuda") for m in ("neural", "titans_mac")}
    estimates = PeakEstimates(execution, "/nonexistent/observed.json")
    order = engine._launch_order(jobs, execution, estimates, executors, READY, FITS)
    assert [job["id"] for job in order] == ["1", "3", "0", "2"]
    serial = Execution(gpu_slots=1, models=execution.models)
    assert engine._launch_order(jobs, serial, estimates, executors, READY, FITS) == jobs[:1]


def always(_job):
    return True


READY = FITS = always


def test_a_blocked_first_job_reserves_its_device():
    execution = Execution(
        gpu_slots=3,
        cpu_workers=2,
        models={
            "neural": dict(default=JobResources("cuda", 512 * MIB, GIB), scopes={}, arms={}),
            "xgboost": dict(default=JobResources("cuda", 4096 * MIB, GIB), scopes={}, arms={}),
            "ridge": dict(default=JobResources("cpu", 0, GIB), scopes={}, arms={}),
        },
    )
    models = ["neural", "xgboost", "neural", "ridge", "neural"]
    jobs = [dict(id=str(i), model=m, kind="fit", scope="US") for i, m in enumerate(models)]
    executors = {(m, "fit"): dict(device="cpu" if m == "ridge" else "cuda") for m in set(models)}
    estimates = PeakEstimates(execution, "/nonexistent/observed.json")

    def order(ready, fits):
        result = engine._launch_order(jobs, execution, estimates, executors, ready, fits)
        return [job["id"] for job in result]

    # El primer trabajo listo cabe: los grandes primero y los pequeños rellenan.
    assert order(READY, FITS) == ["1", "0", "2", "4", "3"]
    # El grande es el primero listo y no cabe: solo él y los de otro dispositivo.
    first_blocked = lambda job: job["id"] != "0"  # noqa: E731
    assert order(first_blocked, lambda job: job["id"] != "1") == ["1", "3"]
    # Si el que no cabe no es el primero listo, no reserva nada.
    assert order(READY, lambda job: job["id"] != "1") == ["1", "0", "2", "4", "3"]


def gpu_report(*processes, free="6000 MiB"):
    rows = "".join(
        f"<process_info><pid>{pid}</pid><type>{kind}</type>"
        f"<process_name>{name}</process_name></process_info>"
        for pid, kind, name in processes
    )
    return (
        f"<nvidia_smi_log><gpu><fb_memory_usage><free>{free}</free></fb_memory_usage>"
        f"<processes>{rows}</processes></gpu></nvidia_smi_log>"
    )


DESKTOP = ((7971, "C+G", "ptyxis"), (2999, "G", "/usr/bin/gnome-shell"))
SERVER = (51, "M", "/usr/bin/nvidia-cuda-mps-server")


def test_desktop_processes_do_not_block_the_slots():
    assert parse_gpu_processes(gpu_report(*DESKTOP), mps=False) == (6000 * 1024**2, [])


@pytest.mark.parametrize("kind", ["C", "M", "M+C"])
def test_other_compute_processes_block_the_slots(kind):
    _free, foreign = parse_gpu_processes(gpu_report(*DESKTOP, (9, kind, "python")), mps=True)
    assert foreign == [(9, kind, "python")]


def test_the_mps_server_is_admitted_only_when_mps_is_declared():
    assert parse_gpu_processes(gpu_report(SERVER), mps=True)[1] == []
    assert parse_gpu_processes(gpu_report(SERVER), mps=False)[1] == [SERVER]


@pytest.mark.parametrize(
    "report",
    [
        gpu_report((9, "X", "python")),
        gpu_report(("nine", "C", "python")),
        gpu_report(free="6 GiB"),
        "<nvidia_smi_log><gpu><fb_memory_usage><free>1 MiB</free></fb_memory_usage></gpu>"
        "</nvidia_smi_log>",
        "<nvidia_smi_log/>",
        "not xml",
    ],
)
def test_an_unreadable_gpu_report_is_rejected(report):
    with pytest.raises(ValueError):
        parse_gpu_processes(report, mps=False)


@pytest.mark.parametrize("ridge_mib", [1024, 4096])
def test_campaign_process_jobs_run_there(prepared, tmp_path, monkeypatch, ridge_mib):  # noqa: F811
    import os

    campaign = campaign_file(tmp_path)
    views = {"US": prepared.views["US"]}
    monkeypatch.setenv("SLOT_LOG", str(tmp_path / "slots.jsonl"))
    monkeypatch.setenv("SLOT_DELAY", "0.2")
    ridge = JobResources("cuda", ridge_mib * MIB, GIB, campaign_process=True)
    execution = slots(
        3,
        vram_budget_bytes=4096 * MIB,
        models=dict(
            neural=dict(default=JobResources("cuda", 1024 * MIB, GIB), scopes={}, arms={}),
            ridge=dict(default=ridge, scopes={}, arms={}),
        ),
    )
    assert execute(campaign, views, tmp_path / "out", execution)["status"] == "completed"
    entries = [e for e in log(tmp_path / "slots.jsonl") if e["event"] == "completed"]
    ridge_jobs = [e for e in entries if "/ridge/" in e["id"]]
    neural = [e for e in entries if "/ridge/" not in e["id"]]
    assert ridge_jobs and {e["pid"] for e in ridge_jobs} == {os.getpid()}
    assert os.getpid() not in {e["pid"] for e in neural}
    # Los trabajos del proceso de la campaña nunca se solapan entre sí.
    spans = sorted((e["start"], e["end"]) for e in ridge_jobs)
    assert all(a[1] <= b[0] for a, b in zip(spans, spans[1:], strict=False))
    shared = [
        (job, other)
        for job in ridge_jobs
        for other in neural
        if other["start"] < job["end"] and job["start"] < other["end"]
    ]
    # Con 1 GiB conviven con las ranuras. Con todo el presupuesto (más el contexto que ya
    # conserva la campaña) se ejecutan sin trabajos GPU al lado.
    assert bool(shared) == (ridge_mib < 4096)
    assert execute(campaign, views, tmp_path / "serial", slots(1))["status"] == "completed"
    assert receipts(tmp_path / "out") == receipts(tmp_path / "serial")


def test_the_campaign_context_counts_once_in_the_vram_budget():
    execution = Execution(gpu_slots=3, vram_budget_bytes=4096 * MIB)
    pool = ResourcePool(execution, available=lambda: 64 * GIB)
    slot = JobResources("cuda", 1792 * MIB, GIB)
    inside = JobResources("cuda", 1024 * MIB, GIB, campaign_process=True)
    pool.acquire(slot)
    assert pool.admits(JobResources("cuda", 2304 * MIB, GIB))
    pool.hold_context()
    # El contexto que conserva la campaña ocupa 512 MiB fuera de los trabajos en curso.
    assert not pool.admits(JobResources("cuda", 2304 * MIB, GIB))
    assert pool.admits(JobResources("cuda", 1792 * MIB, GIB))
    # Un trabajo de la campaña ya incluye ese contexto en su declaración.
    pool.acquire(inside)
    assert pool.admits(JobResources("cuda", 1280 * MIB, GIB))
    assert not pool.admits(JobResources("cuda", 1281 * MIB, GIB))
    # La campaña ejecuta sus trabajos de uno en uno: el segundo no ocupa una ranura esperando.
    second = JobResources("cuda", 256 * MIB, GIB, campaign_process=True)
    assert pool.waits_for_campaign(second) and not pool.admits(second)
    pool.release(inside)
    assert not pool.waits_for_campaign(second) and pool.admits(second)


@pytest.mark.parametrize(
    "ridge",
    [
        dict(device="cpu", host_mib=1536, campaign_process=True),
        dict(device="cuda", vram_mib=1024, host_mib=1536, campaign_process="sí"),
    ],
)
def test_only_cuda_jobs_are_declared_in_the_campaign_process(tmp_path, ridge):
    path = write_execution(tmp_path / "execution.json", models=dict(ridge=ridge))
    with pytest.raises(ValueError, match="proceso de la campaña"):
        load_execution(path)
