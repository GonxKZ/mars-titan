"""Comparación financiera a partir de recibos sintéticos pequeños y confirmados."""

import copy
import csv
import hashlib
import json

import pytest

from mars_titan.data.storage import atomic_json, sha256

POLICIES = ("cash", "hold_initial", "rebalance_25", "rebalance_50", "rebalance_75", "rebalance_100")


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


class FinancialFixture:
    def __init__(self, root):
        self.campaign = root / "campaign"
        self.controls_path = root / "controls/controls.json"
        self.output = root / "comparison"
        self.seeds = [42, 43, 44]
        self.environment = dict(
            capital=10000, cost_bps=10, participation=0.01, score_scale=0.01, ruin_penalty=-20
        )
        self.sources = [
            dict(
                name=f"{family}-audit-{index}",
                family=family,
                seed=index,
                manifest_sha256=str(index) * 64,
                context_sha256="a" * 64,
                split="audit",
                evaluator_only=True,
                warmup_sessions=64,
                decision_start=64,
            )
            for index, family in ((1, "known_signal"), (2, "no_signal"))
        ]
        self.identity = dict(
            schema_version=1,
            kind="adaptive_campaign",
            final_test_opened=False,
            settings=dict(
                schema_version=1,
                variants=["ppo"],
                seeds=self.seeds,
                audit_cost_bps=[0, 10, 25],
                final_test_opened=False,
            ),
            base_configuration=dict(environment=self.environment, final_test_opened=False),
            index_sha256="c" * 64,
            source_manifests=[
                {k: row[k] for k in ("name", "manifest_sha256", "context_sha256", "split")}
                for row in self.sources
            ],
            source_order=dict(
                train=["b" * 64],
                validation=["d" * 64],
                audit=[row["manifest_sha256"] for row in self.sources],
            ),
        )
        self.records = {}
        self.cases = []
        self.selections = []
        for stage in ("pilot", "main", "audit"):
            for seed in self.seeds:
                output = f"{stage}/ppo-{seed}"
                configuration = f"configs/{'main' if stage == 'audit' else stage}-ppo-{seed}.json"
                config = dict(
                    environment=self.environment, training=dict(seed=seed), final_test_opened=False
                )
                atomic_json(self.campaign / configuration, config)
                case = dict(
                    id=f"{stage}-ppo-{seed}",
                    stage=stage,
                    variant="ppo",
                    seed=seed,
                    output=output,
                    status="completed",
                    config=configuration,
                    config_sha256=sha256(self.campaign / configuration),
                    transitions=0 if stage == "audit" else 128,
                    confirmed_transitions=0 if stage == "audit" else 128,
                )
                policy = hashlib.sha256(str(seed).encode()).hexdigest()
                if stage != "audit":
                    report = dict(
                        status="completed",
                        domain="synthetic",
                        final_test_opened=False,
                        seed=seed,
                        agent_variant="ppo",
                        identity_sha256=policy,
                        transitions=128,
                        optimizer_steps=4,
                        best=dict(transitions=0),
                    )
                    case["best"] = report["best"]
                else:
                    case["training_output"] = f"main/ppo-{seed}"
                    report = dict(
                        status="completed",
                        kind="native_ppo_audit",
                        model="ppo",
                        seed=seed,
                        domain="synthetic",
                        analysis_domain="technical",
                        final_test_opened=False,
                        partition="audit",
                        confirmed_episodes=6,
                        total_episodes=6,
                        confirmed_decisions=6 * 255,
                        identity=dict(
                            configuration=config,
                            cost_bps=[0, 10, 25],
                            policy_sha256=policy,
                            selected_identity_sha256=policy,
                            final_test_opened=False,
                            sources=[
                                dict(
                                    manifest_sha256=row["manifest_sha256"],
                                    context_sha256=row["context_sha256"],
                                    generator_seed=row["seed"],
                                )
                                for row in self.sources
                            ],
                        ),
                        metrics=[
                            self.metric(row, cost, 0.1 + 0.2 * (seed - 42) - 0.1 * index)
                            for cost in (0, 10, 25)
                            for index, row in enumerate(self.sources)
                        ],
                    )
                self.records[case["id"]] = report
                self.cases.append(case)
                if stage == "main":
                    self.selections.append(
                        dict(
                            case_id=case["id"],
                            output=output,
                            variant="ppo",
                            seed=seed,
                            config=configuration,
                            config_sha256=case["config_sha256"],
                            policy_sha256=policy,
                            training_identity_sha256=policy,
                        )
                    )
        self.freeze = dict(
            schema_version=1,
            identity_sha256=digest(self.identity),
            selections=self.selections,
            audit_sources=self.identity["source_manifests"],
            selection_uses="validation_only",
            final_test_opened=False,
        )
        self.state = dict(
            schema_version=1,
            identity_sha256=digest(self.identity),
            status="completed",
            phase="completed",
            cases=self.cases,
            choice=dict(transitions=128),
            gate=dict(enabled=False),
            audit_opened=True,
            budget_complete=True,
            active_case=None,
            updated_at="2026-01-01T00:00:00+00:00",
        )
        self.controls = dict(
            schema_version=1,
            kind="frozen_audit_financial_controls",
            status="completed",
            domain="synthetic",
            analysis_domain="technical",
            split="audit",
            final_test_opened=False,
            learning=False,
            decision_start=64,
            sources=copy.deepcopy(self.sources),
            cost_bps=[0, 10, 25],
            **{k: v for k, v in self.environment.items() if k != "cost_bps"},
            catalog_file_sha256="c" * 64,
            metrics=[
                dict(
                    self.metric(row, cost, 0 if policy == "cash" else 0.01),
                    policy=policy,
                    family=row["family"],
                    generator_seed=row["seed"],
                )
                for policy in POLICIES
                for cost in (0, 10, 25)
                for row in self.sources
            ],
        )
        for row in self.controls["metrics"]:
            if row["policy"] == "cash":
                row.update(costs=0.0, turnover=0.0, max_drawdown=0.0)
        self.controls.update(expected_episodes=36, completed_episodes=36, invalid_episodes=0)
        self.publish()

    @staticmethod
    def metric(source, cost, net_return):
        return dict(
            manifest_sha256=source["manifest_sha256"],
            cost_bps=cost,
            net_return=net_return,
            max_drawdown=0.02,
            costs=0.5 * cost,
            turnover=0.5,
            steps=255,
            completed=True,
            invalid_reason="",
        )

    def publish(self):
        atomic_json(self.campaign / "identity.json", self.identity)
        for case in self.cases:
            name = "audit.json" if case["stage"] == "audit" else "run.json"
            path = self.campaign / case["output"] / name
            atomic_json(path, self.records[case["id"]])
            case["receipt_sha256"] = sha256(path)
            for selected in self.selections:
                if selected["case_id"] == case["id"]:
                    selected["receipt_sha256"] = case["receipt_sha256"]
        self.freeze["identity_sha256"] = digest(self.identity)
        atomic_json(self.campaign / "freeze.json", self.freeze)
        self.state.update(
            identity_sha256=digest(self.identity),
            freeze_sha256=sha256(self.campaign / "freeze.json"),
        )
        atomic_json(
            self.campaign / "campaign.json", dict(payload=self.state, sha256=digest(self.state))
        )
        report = dict(
            schema_version=1,
            kind="adaptive_campaign",
            identity_sha256=digest(self.identity),
            status="completed",
            phase="completed",
            updated_at=self.state["updated_at"],
            completed_cases=len(self.cases),
            final_test_opened=False,
            parent_frozen=True,
            domain="synthetic",
            analysis_domain="technical",
            budget_complete=True,
            audit_opened=True,
            selection_frozen=True,
        )
        atomic_json(self.campaign / "run.json", report)
        rows = [
            dict(
                path=c["output"],
                stage=c["stage"],
                variant=c["variant"],
                seed=c["seed"],
                status=c["status"],
                config_sha256=c["config_sha256"],
                planned_transitions=c["transitions"],
                transitions=c["confirmed_transitions"],
            )
            for c in self.cases
        ]
        atomic_json(
            self.campaign / "registry.json",
            dict(
                schema_version=1,
                kind="adaptive_campaign",
                status="completed",
                planned_runs=len(rows),
                runs=rows,
            ),
        )
        self.controls.update(
            campaign_file_sha256=sha256(self.campaign / "campaign.json"),
            freeze_file_sha256=sha256(self.campaign / "freeze.json"),
            identity_file_sha256=sha256(self.campaign / "identity.json"),
            campaign_identity_sha256=digest(self.identity),
        )
        atomic_json(self.controls_path, self.controls)
        self.controls_sha256 = sha256(self.controls_path)

    def compare(self):
        from mars_titan.evaluation.financial_comparison import compare_financial_campaign

        return compare_financial_campaign(
            self.campaign, self.controls_path, self.output, controls_sha256=self.controls_sha256
        )


@pytest.fixture
def campaign(tmp_path):
    return FinancialFixture(tmp_path)


def test_pairs_worlds_after_averaging_policy_seeds(campaign):
    report = campaign.compare()
    assert report["population"]["worlds"] == 2
    assert report["population"]["policy_seeds"] == [42, 43, 44]
    assert report["counts"]["audit_episodes"] == 18
    assert report["counts"]["paired_world_rows"] == 36
    row = next(
        row
        for row in report["paired"]
        if row["control"] == "cash" and row["family"] == "all" and row["cost_bps"] == 10
    )
    assert row["worlds"] == 2
    assert row["policy_seeds"] == 3
    assert row["delta_net_return"] == pytest.approx(0.25)
    with (campaign.output / "paired_worlds.csv").open() as stream:
        pairs = list(csv.DictReader(stream))
    assert len(pairs) == 36
    assert {row["policy_seeds"] for row in pairs} == {"3"}
    assert str(campaign.campaign.parent) not in json.dumps(report)


@pytest.mark.parametrize(
    "defect", ["duplicate", "missing", "cost", "warmup", "cohort", "test", "selection"]
)
def test_rejects_incoherent_audits_and_controls_without_creating_output(campaign, defect):
    audit = campaign.records["audit-ppo-42"]
    if defect == "duplicate":
        audit["metrics"][-1] = copy.deepcopy(audit["metrics"][0])
    elif defect == "missing":
        audit["metrics"].pop()
    elif defect == "cost":
        audit["metrics"][0]["cost_bps"] = 5
    elif defect == "warmup":
        campaign.controls["sources"][0]["decision_start"] = 0
    elif defect == "cohort":
        campaign.controls["sources"][0]["manifest_sha256"] = "d" * 64
    elif defect == "test":
        audit["final_test_opened"] = True
    else:
        audit["identity"]["policy_sha256"] = "d" * 64
    campaign.publish()
    with pytest.raises(ValueError):
        campaign.compare()
    assert not campaign.output.exists()


@pytest.mark.parametrize("target", ["audit", "controls", "freeze", "identity"])
def test_rejects_corrupted_confirmed_bytes(campaign, target):
    path = {
        "audit": campaign.campaign / "audit/ppo-42/audit.json",
        "controls": campaign.controls_path,
        "freeze": campaign.campaign / "freeze.json",
        "identity": campaign.campaign / "identity.json",
    }[target]
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="huella"):
        campaign.compare()


def test_rejects_duplicate_control_world(campaign):
    campaign.controls["metrics"][-1] = copy.deepcopy(campaign.controls["metrics"][0])
    campaign.publish()
    with pytest.raises(ValueError, match="duplicad|población"):
        campaign.compare()


def test_rejects_nonfinite_metrics_even_if_the_control_hash_matches(campaign):
    campaign.controls["metrics"][8]["net_return"] = float("nan")
    campaign.controls_path.write_text(json.dumps(campaign.controls))
    campaign.controls_sha256 = sha256(campaign.controls_path)
    with pytest.raises(ValueError, match="finitos"):
        campaign.compare()


def test_known_ruin_keeps_the_declared_penalty_and_is_not_missing_data(campaign):
    audit = campaign.records["audit-ppo-42"]
    audit["metrics"][0].update(net_return=-1, max_drawdown=1, steps=100, invalid_reason="ruined")
    audit["confirmed_decisions"] = 5 * 255 + 100
    campaign.publish()
    report = campaign.compare()
    row = next(
        row
        for row in report["aggregates"]
        if row["kind"] == "learned"
        and row["seed"] == 42
        and row["family"] == "known_signal"
        and row["cost_bps"] == 0
    )
    assert row["mean_log_growth"] == -20
    assert row["mean_ruined"] == 1


def test_refuses_output_inside_sources_and_existing_destination(campaign):
    campaign.output = campaign.campaign / "result"
    with pytest.raises(ValueError, match="fuera|origen"):
        campaign.compare()
    campaign.output = campaign.campaign.parent / "existing"
    campaign.output.mkdir()
    with pytest.raises(FileExistsError):
        campaign.compare()


def test_missing_control_episode_and_changed_environment_are_rejected(campaign):
    campaign.controls["metrics"].pop()
    campaign.publish()
    with pytest.raises(ValueError, match="población"):
        campaign.compare()
    campaign.controls["capital"] = 20000
    campaign.publish()
    with pytest.raises(ValueError, match="parámetros"):
        campaign.compare()


def test_cli_emits_counts_and_registered_artifact_hashes(campaign, capsys):
    from mars_titan.evaluation.financial_comparison import main

    assert (
        main(
            [
                "--campaign",
                str(campaign.campaign),
                "--controls",
                str(campaign.controls_path),
                "--controls-sha256",
                campaign.controls_sha256,
                "--output",
                str(campaign.output),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["paired_world_rows"] == 36
    report = json.loads((campaign.output / "comparison.json").read_text())
    assert report["status"] == "completed"
    for artifact in report["artifacts"].values():
        assert sha256(campaign.output / artifact["path"]) == artifact["sha256"]
