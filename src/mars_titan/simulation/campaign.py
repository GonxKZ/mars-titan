"""Presupuesto financiero fijado antes de evaluar, con padres congelados."""

import fcntl
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from mars_titan.data.cohort_files import safe_destination
from mars_titan.data.storage import atomic_json, sha256
from mars_titan.training.run_receipts import initialize_receipt

from .environment import FinancialEnv
from .evaluation import evaluate, fixed_policy, learned_policy
from .training import FinancialTrainer, TrainConfig


def run_campaign(
    train, validation, output, config, *, resume=False, diagnostic=False, lease=None, stop=None
):
    required = {
        "schema_version",
        "seeds",
        "algorithms",
        "references",
        "cost_bps",
        "environment",
        "training",
        "final_test_opened",
    }
    if (
        set(config) != required
        or config["schema_version"] != 1
        or config["final_test_opened"] is not False
        or config["seeds"] != [42, 43, 44]
        or config["algorithms"] != ["ppo", "double_dqn"]
        or config["references"] != ["cash", "hold_initial", "rebalance_50"]
        or config["cost_bps"] != [0, 10, 25]
        or train.partition != "train"
        or not 1 <= len(validation) <= 12
        or any(
            t.partition != "validation"
            or t.assets != train.assets
            or t.currency != train.currency
            or t.identity["parent_id"] != train.identity["parent_id"]
            or t.domain != train.domain
            for t in validation
        )
        or len({t.sha256 for t in validation}) != len(validation)
        or (not diagnostic and (lease is None or lease.handle is None))
    ):
        raise ValueError("La campaña no conserva sus controles, fuentes o admisión CUDA")
    training = TrainConfig(**config["training"])
    training.validate()
    if train.domain == "synthetic":
        source = train.identity.get("source")
        validation_sources = [t.identity.get("source") for t in validation]
        if (
            not source
            or any(not item for item in validation_sources)
            or any(
                item["generator"]["seed"] == source["generator"]["seed"]
                for item in validation_sources
            )
        ):
            raise ValueError(
                "La evaluación sintética necesita semillas separadas del entrenamiento"
            )
    if diagnostic and training.total_steps > 32:
        raise ValueError("El diagnóstico en CPU admite como máximo 32 pasos por ajuste")
    identity = dict(
        config=config,
        training=asdict(training),
        train=train.sha256,
        validation=[t.sha256 for t in validation],
        diagnostic=diagnostic,
        code={
            name: sha256(Path(__file__).with_name(name))
            for name in ("campaign.py", "evaluation.py", "training.py")
        },
    )
    output = Path(output)
    safe_destination(output)
    if output.exists() and not resume or resume and not output.is_dir():
        raise ValueError("Usa una campaña nueva o recuperación explícita")
    output.mkdir(parents=True, exist_ok=resume)
    report = dict(
        schema_version=1,
        activity="rl",
        model="financial_comparison",
        status="running",
        domain="technical" if diagnostic else train.domain,
        identity=identity,
        final_test_opened=False,
        parent_frozen=True,
        training_runs=0,
        evaluations=0,
        total_steps=6 * training.total_steps,
        global_step=0,
    )

    def publish():
        report["updated_at"] = datetime.now(UTC).isoformat()
        atomic_json(output / "run.json", report)

    def assess(tape, policy, name, seed, cost, index):
        environment = dict(config["environment"], cost_bps=cost)
        result = evaluate(
            FinancialEnv(tape, **environment),
            policy,
            seed=seed,
            check_resources=lease.check if lease else None,
        )
        result.update(
            model=name,
            domain=report["domain"],
            status="completed",
            updated_at=datetime.now(UTC).isoformat(),
        )
        destination = output / "evaluations" / f"{name}-{seed}-{cost}-{index}"
        destination.mkdir(parents=True, exist_ok=True)
        atomic_json(destination / "run.json", result)
        report["evaluations"] += 1

    with (output / ".campaign.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        initialize_receipt(output, identity, record="run.json", lock=".campaign.lock")
        publish()
        try:
            for name in config["references"]:
                for index, tape in enumerate(validation):
                    for cost in config["cost_bps"]:
                        assess(tape, fixed_policy(name), name, 42, cost, index)
            for algorithm in config["algorithms"]:
                for seed in config["seeds"]:
                    trainer = FinancialTrainer(
                        FinancialEnv(train, **config["environment"]),
                        algorithm,
                        training,
                        seed=seed,
                        device="cpu" if diagnostic else "cuda:0",
                        diagnostic=diagnostic,
                        lease=lease,
                    )
                    destination = output / "training" / f"{algorithm}-{seed}"
                    result = trainer.run(destination, resume=destination.exists(), stop=stop)
                    report["global_step"] += result["global_step"]
                    if result["status"] != "completed":
                        report["status"] = result["status"]
                        return report
                    report["training_runs"] += 1
                    for index, tape in enumerate(validation):
                        for cost in config["cost_bps"]:
                            assess(tape, learned_policy(trainer), algorithm, seed, cost, index)
                    publish()
            report["status"] = "completed"
        except BaseException as error:
            report.update(status="failed", error_type=type(error).__name__)
            raise
        finally:
            publish()
    return report
