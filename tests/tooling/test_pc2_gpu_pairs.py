from types import SimpleNamespace

import pytest

from scripts import run_pc2_gpu_pairs as pairs


def job(name, year=2023, stage="core"):
    return SimpleNamespace(model=name, year=year, stage=stage, key=f"{name}/{year}/seed-42")


@pytest.mark.parametrize(
    ("free", "gpu", "host", "expected"),
    [
        (8.7, 7000, 3000, "ram_wsl"),
        (15, 4500, 5000, "vram"),
        (15, 6500, 430, "ram_windows"),
        (15, None, 5000, "vram"),
        (15, 6500, None, "ram_windows"),
        (None, 6500, 4000, "ram_wsl"),
        (14, 6500, 4000, "par"),
    ],
)
def test_admission_requires_all_reserves(free, gpu, host, expected):
    assert pairs.admission(free, 6, gpu, host) == expected


def test_same_tier_and_tested_model_and_view():
    primary = job("transformer")
    candidates = [job("gru"), job("dlinear", 2022), job("mars_titan_m1", 2020, "core/history")]
    assert pairs.eligible_pair(primary, [primary, *candidates]) is None
    reader = job("mars_titan_m1", 2020)
    assert pairs.eligible_pair(primary, [primary, *candidates, reader]) is reader
    dlinear = job("dlinear")
    assert pairs.eligible_pair(dlinear, [dlinear, primary]) is primary


def test_no_parallel_work_after_memory_pressure(monkeypatch):
    import threading

    queue = SimpleNamespace(request_stop=threading.Event().set)
    seen = []
    queue.request_stop = lambda: seen.append("stop")
    closed = SimpleNamespace(wait=lambda _: False)
    pressure = threading.Event()
    monkeypatch.setattr(pairs, "gpu_free_mib", lambda: 900)
    monkeypatch.setattr(pairs, "host_free_mib", lambda: 4000)
    pairs.watch_resources(SimpleNamespace(_available_memory_gib=lambda: 3), queue, closed, pressure)
    assert seen == ["stop"] and pressure.is_set()


def wire(monkeypatch, tmp_path, mode):
    from mars_titan.training import comparison_campaign as campaign

    first, heavy, other = job("transformer"), job("gru"), job("dlinear")
    jobs = [first, heavy, other]
    status = {j.key: campaign.PENDING for j in jobs}
    groups = []

    class Queue:
        def __init__(self, *args, workers):
            import threading

            self.stop = threading.Event()
            self.workers = workers

        def preflight(self):
            return []

        def _deadline_reached(self):
            return False

        def _disk_exhausted(self):
            return False

        def run(self, *, preflight):
            assert not preflight
            groups.append((self.workers, [j.model for j in self.jobs]))
            for j in self.jobs:
                status[j.key] = campaign.DONE

        def request_stop(self):
            self.stop.set()

    monkeypatch.setattr(
        campaign,
        "load_declaration",
        lambda: ({"budget": {"memory_gib": {"mars_titan": 6}}}, "digest"),
    )
    monkeypatch.setattr(campaign, "Layout", lambda: SimpleNamespace(state=tmp_path))
    monkeypatch.setattr(campaign, "code_commit", lambda: pairs.WARMUP_RELEASE)
    monkeypatch.setattr(campaign, "code_changes", lambda: None)
    monkeypatch.setattr(campaign, "authorized", lambda *a, **kw: True)
    monkeypatch.setattr(campaign, "doctor", lambda *a, **kw: [("memoria", True, "ok")])
    monkeypatch.setattr(campaign, "_available_memory_gib", lambda: 16)
    monkeypatch.setattr(campaign, "Queue", Queue)
    monkeypatch.setattr(campaign, "plan", lambda *a: jobs)
    monkeypatch.setattr(
        campaign,
        "read_record",
        lambda _, k: {
            "attempts": ([] if status[k] == campaign.PENDING else [{}]),
            "status": status[k],
        },
    )
    monkeypatch.setattr(campaign, "job_state", lambda _, j, *args: status[j.key])
    monkeypatch.setattr(pairs, "reason", lambda *a: mode)
    monkeypatch.setattr(pairs.signal, "signal", lambda *a: None)
    return groups


def test_scheduler_pairs_only_confirmed_jobs(monkeypatch, tmp_path):
    groups = wire(monkeypatch, tmp_path, "par")
    assert pairs.main([]) == 0
    assert groups == [(2, ["transformer", "dlinear"]), (1, ["gru"])]


def test_insufficient_memory_executes_official_queue(monkeypatch, tmp_path):
    wire(monkeypatch, tmp_path, "ram_wsl")

    class Replaced(Exception):
        pass

    def execv(exe, argv):
        assert argv[1:] == [
            "-m",
            "mars_titan.training.comparison_campaign",
            "run",
            "--machine",
            "mars_titan",
            "--device",
            "cuda:0",
        ]
        raise Replaced()

    monkeypatch.setattr(pairs.os, "execv", execv)
    with pytest.raises(Replaced):
        pairs.main([])


def test_packed_release_keeps_one_worker_until_paired_inputs_are_measured(
    monkeypatch, tmp_path, capsys
):
    from mars_titan.training import comparison_campaign as campaign

    wire(monkeypatch, tmp_path, "par")
    monkeypatch.setattr(campaign, "code_commit", lambda: pairs.PACKED_RELEASE)
    assert pairs.main(["--assess"]) == 0
    assert '"admission": "serie_entradas_verificadas"' in capsys.readouterr().out


def test_unqualified_release_is_rejected(monkeypatch, tmp_path):
    from mars_titan.training import comparison_campaign as campaign

    wire(monkeypatch, tmp_path, "par")
    monkeypatch.setattr(campaign, "code_commit", lambda: "0" * 40)
    with pytest.raises(SystemExit, match="versión científica"):
        pairs.main(["--assess"])
