"""Coordinar presupuesto y recuperación con recibos simulados, sin lanzar CUDA."""

import hashlib
import json
import math
import runpy
import sys
from pathlib import Path

import pytest

from mars_titan.simulation import adaptive_campaign
from mars_titan.simulation.adaptation_scenarios import FAMILIES
from mars_titan.simulation.adaptive_campaign import (
    VARIANTS,
    auxiliary_gate,
    run_adaptive_campaign,
    select_grid,
)

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(path.read_text())


def write(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, sort_keys=True, separators=(",", ":")))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def settings():
    return read(ROOT / "configs/simulation/adaptive-campaign.json")


def pilot_costs(seconds=300):
    return [
        dict(
            variant=variant,
            seed=seed,
            status="completed",
            transitions=8192,
            budget_seconds=seconds,
            best={"mean_log_growth": seed},
        )
        for variant in VARIANTS
        for seed in (42, 43, 44)
    ]


def test_grid_uses_only_costs_and_fails_if_the_smallest_budget_does_not_fit():
    costs = pilot_costs()
    result = select_grid(costs, settings())
    assert result["transitions"] == 262144
    for item in costs:
        item["best"]["mean_log_growth"] = -1_000_000
    assert select_grid(costs, settings()) == result
    with pytest.raises(ValueError, match="65536"):
        select_grid(pilot_costs(2000), settings())


def test_auxiliary_gate_requires_gain_without_extra_ruin_or_drawdown():
    metrics = [
        dict(
            variant=variant,
            seed=seed,
            best=dict(
                episodes=128,
                ruin_count=0,
                mean_log_growth=0.011 if variant == "ppo_episodic_hmm" else 0.01,
                mean_max_drawdown=0.1,
            ),
        )
        for variant in ("ppo_hmm", "ppo_episodic_hmm")
        for seed in (42, 43, 44)
    ]
    assert auxiliary_gate(metrics, settings())["enabled"] is True
    metrics[-1]["best"]["ruin_count"] = 1
    assert auxiliary_gate(metrics, settings())["enabled"] is False
    metrics[-1]["best"].update(ruin_count=0, mean_max_drawdown=0.2)
    assert auxiliary_gate(metrics, settings())["enabled"] is False


@pytest.fixture
def catalog(tmp_path):
    root = tmp_path / "scenarios"
    root.mkdir()
    records = []
    for family_index, family in enumerate(FAMILIES):
        for split, count in (("train", 32), ("validation", 16), ("audit", 64)):
            for index in range(count):
                seed = (
                    family_index * 10000
                    + {"train": 0, "validation": 100, "audit": 200}[split]
                    + index
                )
                name = f"{family}-{split}-{seed}"
                record = dict(
                    name=name,
                    path=name,
                    family=family,
                    split=split,
                    seed=seed,
                    partition="train" if split == "train" else "validation",
                    evaluator_only=split == "audit",
                    warmup_sessions=64,
                )
                if split == "audit":
                    record.update(
                        manifest_sha256=hashlib.sha256(name.encode()).hexdigest(),
                        context_sha256="b" * 64,
                    )
                else:
                    directory = root / name
                    directory.mkdir()
                    market, context = directory / "market.parquet", directory / "context.parquet"
                    market.write_bytes(name.encode())
                    context.write_bytes((name + "-context").encode())
                    manifest = dict(
                        schema_version=1,
                        final_test_opened=False,
                        assets=16,
                        sessions=256,
                        file_bytes=market.stat().st_size,
                        file_sha256=hashlib.sha256(market.read_bytes()).hexdigest(),
                        identity=dict(
                            domain="synthetic",
                            partition=record["partition"],
                            parent_id="fixture",
                            source={"generator": dict(seed=seed, family=family, split=split)},
                        ),
                    )
                    record["manifest_sha256"] = write(directory / "manifest.json", manifest)
                    record["context_sha256"] = write(
                        directory / "context.json",
                        dict(
                            schema_version=1,
                            domain="synthetic",
                            market_manifest_sha256=record["manifest_sha256"],
                            fields=[dict(name="trading_enabled", unit="boolean")],
                            file=dict(
                                path="context.parquet",
                                bytes=context.stat().st_size,
                                sha256=hashlib.sha256(context.read_bytes()).hexdigest(),
                            ),
                        ),
                    )
                records.append(record)
    training = [item["manifest_sha256"] for item in records if item["split"] == "train"]
    hmm = dict(
        schema_version=1,
        fit_split="train",
        fit_scope="offline_train",
        inference="forward_filter_only",
        feature_indices=[4, 5, 6],
        train_manifest_sha256=training,
    )
    hmm_hash = write(root / "hmm.json", hmm)
    index = dict(
        schema_version=1,
        kind="adaptation_scenarios",
        status="completed",
        domain="synthetic",
        final_test_opened=False,
        parent_id="fixture",
        records=records,
        settings=dict(families=list(FAMILIES), worlds=dict(train=32, validation=16, audit=64)),
        market_feature_indices=[4, 5, 6],
        hmm=dict(path="hmm.json", sha256=hmm_hash),
    )
    write(root / "index.json", index)
    binary = tmp_path / "binary" / "mars-titan-ppo"
    binary.parent.mkdir()
    binary.write_bytes(b"fixture executable without GPU work")
    binary.chmod(0o700)
    build_body = "source_sha256[64]=" + "a" * 64 + "\nbuild_type[7]=Release\n"
    build_hash = hashlib.sha256(build_body.encode()).hexdigest()
    (binary.parent / "native-build-identity-Release.txt").write_text(
        f"sha256={build_hash}\n{build_body}"
    )
    return root / "index.json", binary, build_hash


class SimulatedExecutor:
    def __init__(self, catalog, codes=(), seconds=10, auxiliary=False):
        self.catalog = read(catalog[0])
        self.build_hash = catalog[2]
        self.codes = list(codes)
        self.seconds = seconds
        self.commands = []
        self.auxiliary = auxiliary

    def __call__(self, command, stop):
        self.commands.append(command)

        def option(name):
            return command[command.index(name) + 1]

        config = read(Path(option("--config")))
        output, watch = Path(option("--output")), Path(option("--watch-state"))
        code = self.codes.pop(0) if self.codes else 0
        before = (
            read(watch)
            if watch.exists()
            else dict(active_seconds=0, budget_seconds=0, starts=0, pauses=0)
        )
        active = 0 if code == 3 else self.seconds
        write(
            watch,
            dict(
                before,
                schema_version=2,
                identity_sha256=hashlib.sha256(str(output).encode()).hexdigest(),
                status={0: "completed", 2: "paused", 3: "waiting"}.get(code, "failed"),
                returncode=code,
                active_seconds=before["active_seconds"] + active,
                budget_seconds=before["budget_seconds"] + active,
                starts=before["starts"] + int(code != 3),
                pauses=before["pauses"] + int(code == 2),
                child_active=False,
                budget_complete=True,
                recoverable_checkpoint=code == 2,
                boot_id="fixture-boot",
                unobserved_reserve_seconds=75,
            ),
        )
        if code not in (0, 2):
            return code
        if "--audit-run" in command:
            return self.audit(command, config, output, code)
        score = (
            0.02 if self.auxiliary and config["agent"]["variant"] == "ppo_episodic_hmm" else 0.01
        )
        metrics = [
            dict(
                manifest_sha256=item["manifest_sha256"],
                net_return=score,
                max_drawdown=0.1,
                costs=2.0,
                turnover=0.2,
                steps=255,
                completed=True,
            )
            for item in self.catalog["records"]
            if item["split"] == "validation"
        ]
        best = dict(
            episodes=128,
            ruin_count=0,
            mean_log_growth=math.log1p(score),
            mean_max_drawdown=0.1,
            validation_metrics=metrics,
            transitions=0,
            optimizer_steps=0,
        )
        by_name = {row["name"]: row for row in self.catalog["records"]}
        sources = {
            split: [
                dict(manifest_sha256=item["manifest_sha256"], context_sha256=item["context_sha256"])
                for item in [
                    by_name[Path(command[index + 1]).name]
                    for index, value in enumerate(command)
                    if value == ("--train-tape" if split == "train" else "--validation-tape")
                ]
            ]
            for split in ("train", "validation")
        }
        identity = dict(
            configuration=config,
            sources=sources,
            native_build_sha256=self.build_hash,
            native_source_sha256="a" * 64,
            device="cuda:0",
            diagnostic=False,
        )
        identity_digest = adaptive_campaign._digest(identity)
        write(output / "identity.json", dict(identity=identity, sha256=identity_digest))
        report = dict(
            schema_version=2,
            kind="native_ppo",
            status="completed" if code == 0 else "paused",
            model=config["agent"]["variant"],
            agent_variant=config["agent"]["variant"],
            seed=config["training"]["seed"],
            transitions=config["training"]["total_transitions"] if code == 0 else 0,
            total_steps=config["training"]["total_transitions"],
            optimizer_steps=10,
            auxiliary_steps=2 if config["agent"]["variant"] in adaptive_campaign.AUXILIARY else 0,
            auxiliary_samples=96
            if config["agent"]["variant"] in adaptive_campaign.AUXILIARY
            else 0,
            device="cuda:0",
            diagnostic=False,
            final_test_opened=False,
            parent_frozen=True,
            stopping_reason="budget_exhausted" if code == 0 else "requested_pause",
            identity_sha256=identity_digest,
            best=best,
        )
        write(output / "run.json", report)
        if code == 0:
            checkpoint(output, config, identity_digest, best)
        return code

    def audit(self, command, config, output, code):
        campaign = output.parents[1]
        assert (campaign / "freeze.json").exists()
        frozen = read(campaign / "freeze.json")
        run = Path(command[command.index("--audit-run") + 1])
        selected = next(
            row for row in frozen["selections"] if row["output"] == str(run.relative_to(campaign))
        )
        by_name = {row["name"]: row for row in self.catalog["records"]}
        sources = [
            by_name[Path(command[index + 1]).name]
            for index, value in enumerate(command)
            if value == "--audit-tape"
        ]
        identity = dict(
            selected_identity_sha256=selected["training_identity_sha256"],
            selected_checkpoint=selected["bundle"],
            policy_sha256=selected["policy_sha256"],
            configuration=config,
            native_build_sha256=self.build_hash,
            native_source_sha256="a" * 64,
            device="cuda:0",
            diagnostic=False,
            final_test_opened=False,
            cost_bps=[0, 10, 25],
            sources=[
                dict(manifest_sha256=row["manifest_sha256"], context_sha256=row["context_sha256"])
                for row in sources
            ],
        )
        metrics = [
            dict(
                manifest_sha256=row["manifest_sha256"],
                cost_bps=cost,
                net_return=0.005,
                max_drawdown=0.1,
                costs=cost,
                turnover=0.2,
                steps=255,
                completed=True,
            )
            for row in sources
            for cost in (0, 10, 25)
        ]
        write(
            output / "audit.json",
            dict(
                schema_version=1,
                kind="native_ppo_audit",
                status="completed" if code == 0 else "paused",
                seed=config["training"]["seed"],
                identity=identity,
                identity_sha256=adaptive_campaign._digest(identity),
                confirmed_episodes=len(metrics) if code == 0 else 0,
                total_episodes=len(metrics),
                metrics=metrics,
            ),
        )
        return code


def checkpoint(output, config, identity_digest, best):
    payloads = {
        "metadata.json": json.dumps(
            dict(
                progress=dict(best=best),
                transitions=best["transitions"],
                optimizer_steps=best["optimizer_steps"],
            )
        ).encode(),
        "policy.pt": json.dumps(config, sort_keys=True).encode(),
        "rollout.pt": b"unit-test-rollout",
    }
    manifest = dict(
        schema_version=1,
        identity_sha256=identity_digest,
        files={
            name: dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
            for name, data in payloads.items()
        },
    )
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(encoded).hexdigest()
    record = dict(bundle="ppo-" + digest, sha256=digest)
    directory = output / record["bundle"]
    directory.mkdir(exist_ok=True)
    for name, data in payloads.items():
        (directory / name).write_bytes(data)
    (directory / "manifest.json").write_bytes(encoded)
    index = dict(
        schema_version=1, identity_sha256=identity_digest, recent=[record], best=record, retired=[]
    )
    write(output / "ppo-index.json", dict(payload=index, sha256=adaptive_campaign._digest(index)))


def prepare_audit_fixture(catalog):
    index = read(catalog[0])
    for row in index["records"]:
        if row["split"] != "audit":
            continue
        directory = catalog[0].parent / row["path"]
        directory.mkdir()
        market = directory / "market.parquet"
        context = directory / "context.parquet"
        market.write_bytes(row["name"].encode())
        context.write_bytes((row["name"] + "context").encode())
        row["manifest_sha256"] = write(
            directory / "manifest.json",
            dict(
                schema_version=1,
                final_test_opened=False,
                file_sha256=hashlib.sha256(market.read_bytes()).hexdigest(),
                file_bytes=market.stat().st_size,
                identity=dict(
                    domain="synthetic",
                    partition="validation",
                    parent_id="fixture",
                    source={
                        "generator": dict(seed=row["seed"], split="audit", family=row["family"])
                    },
                ),
            ),
        )
        row["context_sha256"] = write(
            directory / "context.json",
            dict(
                schema_version=1,
                domain="synthetic",
                market_manifest_sha256=row["manifest_sha256"],
                file=dict(
                    path="context.parquet",
                    bytes=context.stat().st_size,
                    sha256=hashlib.sha256(context.read_bytes()).hexdigest(),
                ),
            ),
        )
    write(catalog[0], index)


def test_pilot_only_uses_all_training_validation_sources_and_never_opens_audit(catalog, tmp_path):
    runner = SimulatedExecutor(catalog)
    result = run_adaptive_campaign(
        catalog[0],
        catalog[1],
        tmp_path / "campaign",
        pilot_only=True,
        executor=runner,
        sleep=lambda _: None,
    )
    assert result["phase"] == "pilot_complete" and result["chosen_transitions"] == 524288
    assert len(runner.commands) == 21
    assert result["budget_seconds"] == 210
    assert result["audit_opened"] is False
    for command in runner.commands:
        assert command.count("--train-tape") == 256
        assert command.count("--validation-tape") == 128
        assert "--audit-tape" not in command
        names = [
            Path(command[index + 1]).name
            for index, value in enumerate(command)
            if value == "--train-tape"
        ][:16]
        assert [name.split("-train-", 1)[0] for name in names] == list(FAMILIES) * 2
    before = len(runner.commands)
    recovered = run_adaptive_campaign(
        catalog[0],
        catalog[1],
        tmp_path / "campaign",
        resume=True,
        pilot_only=True,
        executor=runner,
        sleep=lambda _: None,
    )
    assert len(runner.commands) == before
    assert recovered["budget_seconds"] == result["budget_seconds"]


def test_pauses_charge_all_attempts_and_resume_with_a_finite_retry_limit(catalog, tmp_path):
    runner = SimulatedExecutor(catalog, codes=[2, 2, 2, 2], seconds=7)
    sleeps = []
    result = run_adaptive_campaign(
        catalog[0],
        catalog[1],
        tmp_path / "campaign",
        pilot_only=True,
        executor=runner,
        sleep=sleeps.append,
    )
    assert result["status"] == "blocked"
    assert result["budget_seconds"] == 28
    assert len(runner.commands) == 4 and sleeps == [30, 30, 30]
    assert all("--resume" in command for command in runner.commands[1:])


@pytest.mark.parametrize("code", [1, 75])
def test_errors_block_the_case_without_blind_retry(catalog, tmp_path, code):
    runner = SimulatedExecutor(catalog, codes=[code], seconds=9)
    result = run_adaptive_campaign(catalog[0], catalog[1], tmp_path / "campaign", executor=runner)
    assert result["status"] == "blocked" and result["budget_seconds"] == 9
    assert len(runner.commands) == 1


def test_gpu_waiting_does_not_consume_active_budget_or_pause_retries(catalog, tmp_path):
    runner = SimulatedExecutor(catalog, codes=[3, 0])
    sleeps = []
    result = run_adaptive_campaign(
        catalog[0],
        catalog[1],
        tmp_path / "campaign",
        pilot_only=True,
        executor=runner,
        sleep=sleeps.append,
    )
    assert result["budget_seconds"] == 210 and len(runner.commands) == 22
    assert sleeps == [30]
    assert "--resume" not in runner.commands[1]


def test_resume_rejects_changed_binary_or_configuration(catalog, tmp_path):
    runner = SimulatedExecutor(catalog)
    output = tmp_path / "campaign"
    run_adaptive_campaign(catalog[0], catalog[1], output, pilot_only=True, executor=runner)
    catalog[1].write_bytes(b"different executable")
    with pytest.raises(ValueError, match="identidad"):
        run_adaptive_campaign(catalog[0], catalog[1], output, resume=True, executor=runner)
    assert len(runner.commands) == 21


def test_hmm_with_validation_sources_is_rejected_before_creating_output(catalog, tmp_path):
    path = catalog[0].parent / "hmm.json"
    model = read(path)
    model["train_manifest_sha256"][0] = "f" * 64
    index = read(catalog[0])
    index["hmm"]["sha256"] = write(path, model)
    write(catalog[0], index)
    with pytest.raises(ValueError, match="HMM"):
        run_adaptive_campaign(catalog[0], catalog[1], tmp_path / "campaign", executor=lambda *_: 0)
    assert not (tmp_path / "campaign").exists()


@pytest.mark.parametrize("auxiliary,training_count", [(False, 21), (True, 27)])
def test_full_campaign_freezes_selection_before_audit_and_summarizes_real_receipts(
    catalog, tmp_path, monkeypatch, auxiliary, training_count
):
    prepare_audit_fixture(catalog)
    output = tmp_path / "campaign"
    original = adaptive_campaign._json

    def guarded(path, maximum=adaptive_campaign.JSON_LIMIT):
        if "-audit-" in path.parent.name:
            assert (output / "freeze.json").exists(), "Se abrió auditoría antes de congelar modelos"
        return original(path, maximum)

    monkeypatch.setattr(adaptive_campaign, "_json", guarded)
    runner = SimulatedExecutor(catalog, auxiliary=auxiliary)
    result = run_adaptive_campaign(
        catalog[0], catalog[1], output, executor=runner, sleep=lambda _: None
    )
    assert result["status"] == "completed" and result["selection_frozen"] is True
    assert result["auxiliary_gate"]["enabled"] is auxiliary
    assert len(runner.commands) == 21 + 2 * training_count
    assert result["budget_seconds"] == 10 * len(runner.commands)
    audited = [command for command in runner.commands if "--audit-run" in command]
    assert len(audited) == training_count
    assert all(
        command.count("--audit-tape") == 512 and "--train-tape" not in command
        for command in audited
    )
    summary = read(output / "summary.json")
    assert {row["family"] for row in summary["metrics"]} == set(FAMILIES)
    assert {row["seed"] for row in summary["metrics"]} == {42, 43, 44}
    assert {row["cost_bps"] for row in summary["metrics"] if row["split"] == "audit"} == {0, 10, 25}
    assert summary["audit_used_for_selection"] is False
    assert len(summary["runs"]) == 21 + 2 * training_count
    assert all(not Path(row["path"]).is_absolute() for row in summary["runs"])
    if auxiliary:
        assert len(summary["auxiliary_work"]) == 3
        assert all(row["same_optimizer_steps"] for row in summary["auxiliary_work"])
        assert all(row["same_auxiliary_samples"] for row in summary["auxiliary_work"])
        aux_runs = [row for row in summary["runs"] if row["stage"] == "auxiliary"]
        assert all(row["auxiliary_samples"] == 96 for row in aux_runs)
    before = len(runner.commands)
    recovered = run_adaptive_campaign(
        catalog[0], catalog[1], output, resume=True, executor=runner, sleep=lambda _: None
    )
    assert recovered["status"] == "completed" and len(runner.commands) == before


def test_budget_overrun_blocks_further_launches_instead_of_hiding_time(catalog, tmp_path):
    runner = SimulatedExecutor(catalog, seconds=12 * 3600 + 1)
    result = run_adaptive_campaign(catalog[0], catalog[1], tmp_path / "campaign", executor=runner)
    assert result["status"] == "blocked" and result["reason"] == "active_budget_overrun"
    assert result["budget_seconds"] == 12 * 3600 + 1 and len(runner.commands) == 1


def test_real_subprocess_executor_forwards_stop_only_to_its_child(tmp_path):
    ready = tmp_path / "ready.json"
    program = (
        "import signal,time,sys\nfrom pathlib import Path\n"
        "signal.signal(signal.SIGTERM, lambda *_: sys.exit(2))\n"
        "Path(sys.argv[1]).write_text('ready')\n"
        "while True: time.sleep(.01)\n"
    )
    result = adaptive_campaign._execute([sys.executable, "-c", program, str(ready)], ready.exists)
    assert result == 2 and ready.read_text() == "ready"


def campaign_instance(catalog, output, runner, *, config=adaptive_campaign.DEFAULT_CONFIG):
    inputs = adaptive_campaign.FrozenInputs(catalog[0], catalog[1], config)
    output.mkdir()
    return adaptive_campaign.Campaign(inputs, output, False, runner, lambda: False, lambda _: None)


@pytest.mark.parametrize("code", [0, 2, 1, 75])
def test_reconcile_closed_watch_after_journal_cut_without_duplicate_attempt(
    catalog, tmp_path, code
):
    runner = SimulatedExecutor(catalog, codes=[code])
    app = campaign_instance(catalog, tmp_path / "campaign", runner)
    case = app.state["cases"][0]
    case["status"] = "running"
    app.publish()
    runner(app.command(case), lambda: False)
    recovered = adaptive_campaign.Campaign(
        app.inputs, app.output, True, runner, lambda: False, lambda _: None
    )
    actual = recovered.state["cases"][0]
    assert recovered.run_case(actual) is (code in (0, 2))
    assert len(runner.commands) == (2 if code == 2 else 1)
    assert actual["budget_seconds"] == (20 if code == 2 else 10)
    assert actual["accounted_starts"] == actual["attempts"]


def test_existing_watcher_finishes_without_charging_provisional_reserve_twice(catalog, tmp_path):
    runner = SimulatedExecutor(catalog)
    app = campaign_instance(catalog, tmp_path / "campaign", runner)
    case = app.state["cases"][0]
    watch_path = app.output / case["watch"]

    def ongoing(command, stop):
        runner(command, stop)
        watch = read(watch_path)
        watch.update(child_active=True, parent_death_signal="SIGKILL", returncode=None)
        write(watch_path, watch)
        return 3

    def finish(_seconds):
        assert case["budget_seconds"] == 85
        watch = read(watch_path)
        watch.update(child_active=False, returncode=0)
        write(watch_path, watch)

    app.execute, app.sleep = ongoing, finish
    assert app.run_case(case)
    assert case["budget_seconds"] == case["active_seconds"] == 10
    assert len(runner.commands) == 1


def test_initial_reserve_is_charged_without_fabricating_observed_time(catalog, tmp_path):
    config = tmp_path / "settings.json"
    write(config, dict(settings(), initial_reserve_seconds=12 * 3600))
    (tmp_path / "adaptive-ppo.json").write_bytes(
        (ROOT / "configs/simulation/adaptive-ppo.json").read_bytes()
    )
    runner = SimulatedExecutor(catalog)
    app = campaign_instance(catalog, tmp_path / "campaign", runner, config=config)
    report = app.publish()
    assert report["initial_reserved_seconds"] == report["budget_seconds"] == 12 * 3600
    assert report["active_seconds"] == 0
    app.summary()
    pending = read(app.output / "summary.json")["runs"][0]
    assert pending["transitions"] == 0 and pending["planned_transitions"] == 8192
    assert app.remaining("pilot") == 12 * 3600 and app.remaining("audit") == 0
    case = dict(app.state["cases"][0], stage="audit")
    assert not app.run_case(case) and not runner.commands
    assert case["reason"] == "active_budget_exhausted"
    write(config, dict(settings(), initial_reserve_seconds=12 * 3600 + 1))
    with pytest.raises(ValueError, match="reserva inicial"):
        adaptive_campaign.FrozenInputs(catalog[0], catalog[1], config)


def test_training_receipt_rejects_identity_content_with_unchanged_seal(catalog, tmp_path):
    runner = SimulatedExecutor(catalog)
    app = campaign_instance(catalog, tmp_path / "campaign", runner)
    case = app.state["cases"][0]
    runner(app.command(case), lambda: False)
    path = app.output / case["output"] / "identity.json"
    identity = read(path)
    identity["identity"]["architecture"] = "another_model"
    write(path, identity)
    with pytest.raises(ValueError, match="identidad"):
        app.validate_training_receipt(case, "completed")


def test_setup_cut_before_first_journal_can_resume_without_starting_twice(
    catalog, tmp_path, monkeypatch
):
    runner = SimulatedExecutor(catalog)
    publish = adaptive_campaign.Campaign.publish

    def cut(_self):
        raise OSError("Corte de la prueba antes del primer diario")

    monkeypatch.setattr(adaptive_campaign.Campaign, "publish", cut)
    with pytest.raises(OSError, match="Corte"):
        campaign_instance(catalog, tmp_path / "campaign", runner)
    monkeypatch.setattr(adaptive_campaign.Campaign, "publish", publish)
    result = run_adaptive_campaign(
        catalog[0], catalog[1], tmp_path / "campaign", resume=True, pilot_only=True, executor=runner
    )
    assert result["phase"] == "pilot_complete" and len(runner.commands) == 21


def test_forced_wrapper_exit_with_guarded_child_blocks_further_attempts(catalog, tmp_path):
    runner = SimulatedExecutor(catalog)
    app = campaign_instance(catalog, tmp_path / "campaign", runner)
    case = app.state["cases"][0]

    def killed(command, stop):
        runner(command, stop)
        path = app.output / case["watch"]
        value = read(path)
        value.update(child_active=True, parent_death_signal="SIGKILL", returncode=None)
        write(path, value)
        return 75

    app.execute = killed
    assert not app.run_case(case)
    assert case["status"] == "blocked" and case["reason"] == "child_exit_75"
    assert case["budget_seconds"] == 85
    recovered = adaptive_campaign.Campaign(
        app.inputs, app.output, True, killed, lambda: False, lambda _: None
    )
    assert not recovered.run_case(recovered.state["cases"][0])
    assert len(runner.commands) == 1


def test_watch_lock_released_during_wait_is_reconciled_as_completed(catalog, tmp_path):
    runner = SimulatedExecutor(catalog)
    app = campaign_instance(catalog, tmp_path / "campaign", runner)

    def completed_during_lock_wait(command, stop):
        runner(command, stop)
        return 3

    app.execute = completed_during_lock_wait
    assert app.run_case(app.state["cases"][0])
    assert len(runner.commands) == 1


@pytest.mark.parametrize("field", ["optimizer_steps", "auxiliary_steps", "auxiliary_samples"])
def test_training_receipt_rejects_invalid_work_counters(catalog, tmp_path, field):
    runner = SimulatedExecutor(catalog)
    app = campaign_instance(catalog, tmp_path / "campaign", runner)
    case = app.state["cases"][0]
    runner(app.command(case), lambda: False)
    path = app.output / case["output"] / "run.json"
    receipt = read(path)
    receipt[field] = -1
    write(path, receipt)
    with pytest.raises(ValueError, match="entero"):
        app.validate_training_receipt(case, "completed")


def test_freeze_rejects_checkpoint_metadata_from_another_selection(catalog, tmp_path):
    runner = SimulatedExecutor(catalog)
    app = campaign_instance(catalog, tmp_path / "campaign", runner)
    case = app.state["cases"][0]
    assert app.run_case(case)
    case["stage"] = "main"
    wrong = dict(case["best"], transitions=16)
    checkpoint(
        app.output / case["output"],
        read(app.output / case["config"]),
        case["training_identity_sha256"],
        wrong,
    )
    with pytest.raises(ValueError, match="selección"):
        app.freeze()


@pytest.mark.parametrize("status,code", [("completed", 0), ("paused", 2), ("blocked", 1)])
def test_campaign_cli_preserves_state_codes_and_explicit_resume_flags(monkeypatch, status, code):
    namespace = runpy.run_path(str(ROOT / "scripts/run_adaptive_campaign.py"))
    calls = []

    def execute(*args, **kwargs):
        calls.append((args, kwargs))
        return dict(status=status)

    monkeypatch.setitem(namespace["main"].__globals__, "run_adaptive_campaign", execute)
    assert (
        namespace["main"](
            [
                "--scenarios",
                "index.json",
                "--binary",
                "native",
                "--output",
                "campaign",
                "--resume",
                "--pilot-only",
            ]
        )
        == code
    )
    assert calls[0][1]["resume"] is True and calls[0][1]["pilot_only"] is True
    assert calls[0][1]["stop"]() is False


def test_output_with_parent_components_cannot_enter_the_scenario_catalog(catalog, tmp_path):
    with pytest.raises(ValueError, match="separada"):
        run_adaptive_campaign(
            catalog[0], catalog[1], tmp_path / "outside/../scenarios/new-output", stop=lambda: True
        )
    assert not (catalog[0].parent / "new-output").exists()


@pytest.mark.parametrize("code", [0, 2])
def test_invalid_completion_or_pause_receipt_blocks_future_launches(catalog, tmp_path, code):
    runner = SimulatedExecutor(catalog, codes=[code])
    output = tmp_path / "campaign"

    def corrupt(command, stop):
        result = runner(command, stop)
        if code == 0:
            path = Path(command[command.index("--output") + 1]) / "identity.json"
            value = read(path)
            value["identity"]["architecture"] = "incoherent"
        else:
            path = Path(command[command.index("--watch-state") + 1])
            value = read(path)
            value["recoverable_checkpoint"] = False
        write(path, value)
        return result

    with pytest.raises(ValueError, match="identidad|checkpoint"):
        run_adaptive_campaign(catalog[0], catalog[1], output, executor=corrupt)
    case = read(output / "campaign.json")["payload"]["cases"][0]
    assert case["status"] == "blocked"
    result = run_adaptive_campaign(catalog[0], catalog[1], output, executor=corrupt, resume=True)
    assert result["status"] == "blocked" and len(runner.commands) == 1


@pytest.mark.parametrize("code,status", [(2, "paused"), (1, "blocked")])
def test_registry_exposes_pending_cases_before_first_execution_and_updates_on_stop(
    catalog, tmp_path, code, status
):
    runner = SimulatedExecutor(catalog, codes=[code])
    output = tmp_path / "campaign"
    stopped = False

    def first(command, stop):
        nonlocal stopped
        registry = read(output / "registry.json")
        assert registry["kind"] == "adaptive_campaign" and registry["schema_version"] == 1
        assert registry["planned_runs"] == len(registry["runs"]) == 21
        assert registry["runs"][0]["status"] == "running"
        assert all(row["status"] == "pending" for row in registry["runs"][1:])
        assert all(
            row["transitions"] == 0 and row["planned_transitions"] == 8192
            for row in registry["runs"]
        )
        stopped = True
        return runner(command, stop)

    run_adaptive_campaign(catalog[0], catalog[1], output, executor=first, stop=lambda: stopped)
    registry = read(output / "registry.json")
    assert registry["status"] == registry["runs"][0]["status"] == status
    allowed = {
        "path",
        "stage",
        "variant",
        "seed",
        "status",
        "config_sha256",
        "planned_transitions",
        "transitions",
    }
    assert all(
        set(row) == allowed
        and not Path(row["path"]).is_absolute()
        and len(row["config_sha256"]) == 64
        for row in registry["runs"]
    )
    assert str(tmp_path) not in (output / "registry.json").read_text()


def test_convergence_settings_keep_fixed_pilot_and_admit_only_declared_ceiling():
    path = ROOT / "configs/simulation/adaptive-convergence-campaign.json"
    config, base, _ = adaptive_campaign._settings(path)
    assert config["schema_version"] == 2 and config["pilot_transitions"] == 8192
    assert base["schema_version"] == 3
    assert base["selection"]["min_transitions"] == 131072
    assert select_grid(pilot_costs(10), config)["transitions"] == 1048576
    with pytest.raises(ValueError, match="1048576"):
        select_grid(pilot_costs(2000), config)


def convergence_receipt(tmp_path, *, early=True):
    config = read(ROOT / "configs/simulation/adaptive-ppo-convergence.json")
    transitions = 262144 if early else 1048576
    best = dict(transitions=0, optimizer_steps=0)
    report = dict(
        schema_version=3,
        transitions=transitions,
        optimizer_steps=256,
        selection=dict(config["selection"], policy="greedy_argmax"),
        stopping_reason="early_stop" if early else "budget_exhausted",
        evaluations=1 + transitions // 16384,
        stale_evaluations=(transitions - 131072) // 16384,
        best=best,
    )
    metadata = dict(
        schema_version=3,
        configuration=config,
        transitions=transitions,
        optimizer_steps=256,
        progress=dict(
            status="early_stopped" if early else "completed",
            best=best,
            evaluated_transitions=transitions,
            evaluated_optimizer_steps=256,
            evaluations=report["evaluations"],
            stale_evaluations=report["stale_evaluations"],
        ),
    )
    seal_convergence_checkpoint(tmp_path, metadata)
    return config, report, metadata


def seal_convergence_checkpoint(directory, metadata):
    temporary = directory / "pending"
    temporary.mkdir(exist_ok=True)
    digest = write(temporary / "metadata.json", metadata)
    manifest = dict(
        identity_sha256="a" * 64,
        files={
            "metadata.json": dict(sha256=digest, bytes=(temporary / "metadata.json").stat().st_size)
        },
    )
    digest = write(temporary / "manifest.json", manifest)
    bundle = "ppo-" + digest
    temporary.rename(directory / bundle)
    index = dict(identity_sha256="a" * 64, recent=[dict(bundle=bundle, sha256=digest)])
    write(
        directory / "ppo-index.json", dict(payload=index, sha256=adaptive_campaign._digest(index))
    )


@pytest.mark.parametrize("early", [True, False])
def test_convergence_receipt_requires_confirmed_terminal_selection(tmp_path, early):
    config, report, _ = convergence_receipt(tmp_path, early=early)
    campaign = object.__new__(adaptive_campaign.Campaign)
    campaign.validate_convergence_receipt(tmp_path, report, config, "a" * 64)


@pytest.mark.parametrize("corruption", ["minimum", "patience", "pending", "best", "schema", "seal"])
def test_convergence_receipt_rejects_forged_stopping_state(tmp_path, corruption):
    config, report, metadata = convergence_receipt(tmp_path)
    if corruption == "minimum":
        report["transitions"] = metadata["transitions"] = 131072
        report["evaluations"] = metadata["progress"]["evaluations"] = 9
        metadata["progress"]["evaluated_transitions"] = 131072
    elif corruption == "patience":
        report["stale_evaluations"] = metadata["progress"]["stale_evaluations"] = 7
    elif corruption == "pending":
        metadata["progress"]["evaluated_transitions"] -= 16384
    elif corruption == "best":
        report["best"]["transitions"] = 262144
    elif corruption == "schema":
        report["schema_version"] = 2
    else:
        index = read(tmp_path / "ppo-index.json")
        index["sha256"] = "b" * 64
        write(tmp_path / "ppo-index.json", index)
    if corruption in {"minimum", "patience", "pending"}:
        seal_convergence_checkpoint(tmp_path, metadata)
    campaign = object.__new__(adaptive_campaign.Campaign)
    with pytest.raises(ValueError):
        campaign.validate_convergence_receipt(tmp_path, report, config, "a" * 64)


def test_convergence_generated_pilot_disables_stopping_and_keeps_8192(tmp_path):
    from types import SimpleNamespace

    path = ROOT / "configs/simulation/adaptive-convergence-campaign.json"
    config, base, _ = adaptive_campaign._settings(path)
    campaign = object.__new__(adaptive_campaign.Campaign)
    campaign.inputs = SimpleNamespace(settings=config, base=base)
    campaign.output = tmp_path
    campaign.state = dict(cases=[])
    campaign.add_cases("pilot", ["ppo", "double_dqn"], 8192)
    campaign.add_cases("main", ["ppo"], 1048576)
    for case in campaign.state["cases"]:
        generated = read(tmp_path / case["config"])
        assert generated["training"]["seed"] in (42, 43, 44)
        if case["stage"] == "pilot":
            assert generated["training"]["total_transitions"] == 8192
            assert generated["selection"]["early_stopping"] is False
            assert generated["selection"]["min_transitions"] == 0
        else:
            assert generated["selection"] == base["selection"]


def test_v1_campaign_rejects_early_stop_receipt(catalog, tmp_path):
    runner = SimulatedExecutor(catalog)

    def premature(command, stop):
        code = runner(command, stop)
        output = Path(command[command.index("--output") + 1])
        receipt = read(output / "run.json")
        receipt.update(stopping_reason="early_stop", transitions=4096)
        write(output / "run.json", receipt)
        return code

    with pytest.raises(ValueError, match="presupuesto"):
        run_adaptive_campaign(
            catalog[0], catalog[1], tmp_path / "premature", executor=premature, sleep=lambda _: None
        )


def test_convergence_receipt_waits_for_declared_patience(tmp_path):
    config, report, metadata = convergence_receipt(tmp_path)
    config["selection"]["patience"] = 9
    report["selection"]["patience"] = 9
    seal_convergence_checkpoint(tmp_path, metadata)
    campaign = object.__new__(adaptive_campaign.Campaign)
    with pytest.raises(ValueError, match="paciencia"):
        campaign.validate_convergence_receipt(tmp_path, report, config, "a" * 64)
