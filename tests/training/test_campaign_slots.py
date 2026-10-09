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
    Execution,
    JobResources,
    ResourcePool,
    check_plan,
    load_execution,
)
from mars_titan.training.campaign_slots import SlotProcess, SlotTask, bound_vram, new_event, wait
from mars_titan.training.input_pipeline import PipelineOptions
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
    assert not pool.admits(small)  # sin ranuras
    cpu = JobResources("cpu", 0, 5 * GIB)
    assert pool.admits(cpu)
    pool.acquire(cpu)
    assert not pool.admits(JobResources("cpu", 0, GIB))  # sin trabajadores CPU
    pool.release(small)
    assert not pool.admits(JobResources("cuda", GIB, 4 * GIB))  # RAM declarada: 7 + 4 > 10
    assert pool.admits(JobResources("cuda", GIB, GIB))
    free[0] = GIB + GIB // 2
    assert not pool.admits(JobResources("cuda", GIB, GIB))  # sin la reserva del anfitrión


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
    ],
)
def test_invalid_execution_is_rejected(tmp_path, changes):
    with pytest.raises(ValueError):
        load_execution(write_execution(tmp_path / "execution.json", **changes))


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
