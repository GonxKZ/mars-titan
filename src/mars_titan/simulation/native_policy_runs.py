"""Ejecutores nativos de la etapa de políticas sobre cintas reconstruidas por ventana.

PPO y Double DQN usan ``mars-titan-ppo`` con la configuración de esquema 4, y KLPO terminal
usa ``mars-titan-klpo``. Un ajuste escribe su configuración junto al trabajo, lanza el
binario con ``scripts/run_native_ppo.py`` (admisión GPU, carga única, pausa y vigilancia
del padre), lee la selección en validación y evalúa la política elegida en la cinta de
evaluación con los tres costes declarados. Un traslado evalúa sin ajuste la política de su
ancla con la configuración de ese ajuste. Los binarios y el lanzador comprueban la
protección local del aprendizaje antes de leer cintas o crear salidas, igual que la etapa.

KLPO solo consume oleadas completas: un episodio por entorno, que recorre en ciclo las
cintas de ajuste. Usa las oleadas enteras que caben en el presupuesto de transiciones, así
que nunca supera el de PPO y puede quedarse por debajo en menos de una oleada.
"""

import json
import os
import subprocess
import sys
from pathlib import Path, PurePosixPath

from mars_titan.data.cohort_files import read_manifest
from mars_titan.data.storage import atomic_json, sha256

from .policy_plan import CARRY, FIT, _require

ROOT = Path(__file__).resolve().parents[3]
LAUNCHER = ROOT / "scripts" / "run_native_ppo.py"
BUILD = ROOT / "build" / "native" / "native-ppo-release"
BINARIES = {
    "native_ppo": ("MARS_TITAN_PPO_EXECUTABLE", "mars-titan-ppo"),
    "native_klpo": ("MARS_TITAN_KLPO_EXECUTABLE", "mars-titan-klpo"),
}
# Costes de `policy_evaluation.hpp`. La etapa debe declarar los mismos.
EVALUATION_COSTS = [0, 10, 25]
RECONSTRUCTED_SCHEMA = 4
ROLLOUT_BYTES = 128 * 1024**2
# Contrato klpo_terminal_token_full_v1: KL con peso uno y retorno terminal sin descuento.
KLPO_TERMINAL = dict(beta=1.0, gamma=1.0)
KLPO_GRADIENT_BLOCK = 8
PAUSED_CODES = {2: "paused", 3: "gpu_not_admitted", 4: "active_budget_exhausted"}
_MIB = 1024**2


def binary_path(engine):
    """Binario declarado en el entorno o compilado con el preset native-ppo-release."""
    variable, name = BINARIES[engine]
    return Path(os.environ.get(variable) or BUILD / name)


def probe_binary(engine, capability):
    """Preguntar al binario por sus capacidades sin leer datos. Devuelve su identidad."""
    path = binary_path(engine).resolve(strict=True)
    _require(path.is_file() and os.access(path, os.X_OK), f"{path} no es un ejecutable")
    result = subprocess.run(
        [str(path), "--capabilities"],
        env=dict(os.environ, CUDA_VISIBLE_DEVICES="-1"),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    _require(result.returncode == 0, f"{path.name} no informa de sus capacidades")
    report = json.loads(result.stdout)
    _require(
        report.get("schema_version") == 1
        and report.get("binary") == BINARIES[engine][1]
        and capability in report.get("capabilities", []),
        f"{path.name} no declara la capacidad {capability}",
    )
    return dict(report, path=str(path), binary_sha256=sha256(path))


def _environment(policies):
    """Entorno del motor. El plazo de pago ya está en la auditoría de cada cinta."""
    environment = policies["environment"]
    return {k: v for k, v in environment.items() if k != "dividend_payment_lag_sessions"}


def ppo_config(stage, job):
    """Configuración de esquema 4 de un brazo PPO o Double DQN con el presupuesto común."""
    policies = stage["policies"]
    entry, budget = policies["policies"][job["arm"]], policies["budget"]
    selection = policies["selection"]
    document = dict(
        schema_version=RECONSTRUCTED_SCHEMA,
        training=dict(
            total_transitions=budget["transitions"],
            rollout_transitions=budget["rollout_transitions"],
            workers=1,
            seed=job["seed"],
            rollout_bytes=ROLLOUT_BYTES,
        ),
        environments=budget["environments"],
        hyperparameters=dict(policies["hyperparameters"]),
        environment=_environment(policies),
        checkpoint_transitions=budget["evaluation_transitions"],
        selection=dict(
            min_delta=selection["min_delta"],
            patience=selection["patience"],
            early_stopping=selection["early_stopping"],
            metric=selection["metric"],
            min_transitions=0,
        ),
        final_test_opened=False,
        evaluation_transitions=budget["evaluation_transitions"],
        agent=dict(variant=entry["variant"], trading_field=None, markov_fields=[], hmm_file=None),
    )
    if entry["policy_objective"] is not None:
        document["policy_objective"] = dict(entry["policy_objective"])
    return document


def klpo_config(stage, job):
    """Configuración KLPO terminal con el presupuesto, el entorno y la selección comunes."""
    policies = stage["policies"]
    entry, budget = policies["policies"][job["arm"]], policies["budget"]
    hyper, selection = policies["hyperparameters"], policies["selection"]
    return dict(
        schema_version=1,
        kind="native_klpo_terminal",
        objective=entry["objective"],
        controller=entry["controller"],
        training=dict(total_transitions=budget["transitions"], seed=job["seed"], workers=1),
        environments=budget["environments"],
        adam=dict(learning_rate=hyper["learning_rate"], gradient_norm=hyper["gradient_norm"]),
        terminal=dict(KLPO_TERMINAL),
        confirmed_updates_per_reference=entry["confirmed_updates_per_reference"],
        gradient_block_episodes=KLPO_GRADIENT_BLOCK,
        environment=_environment(policies),
        evaluation_transitions=budget["evaluation_transitions"],
        selection=dict(
            metric=selection["metric"],
            min_delta=selection["min_delta"],
            patience=selection["patience"],
            early_stopping=selection["early_stopping"],
        ),
        final_test_opened=False,
    )


def klpo_waves(train, environments, transitions):
    """Oleadas completas que caben en el presupuesto y transiciones máximas de cada una."""
    wave = sum(len(train[lane % len(train)]) - 1 for lane in range(environments))
    return transitions // wave, wave


def _payload(path):
    """Contenido de un registro sellado por el binario, que verifica su huella al reanudar."""
    envelope, _ = read_manifest(path, 64 * _MIB)
    _require(
        isinstance(envelope, dict) and set(envelope) == {"payload", "sha256"},
        f"{path} no conserva su sello",
    )
    return envelope["payload"]


def _record(row, cost, manifest):
    """Episodio del binario con el registro de la etapa y el coste declarado."""
    _require(
        row["manifest_sha256"] == manifest and row["cost_bps"] == cost,
        "La evaluación nativa no corresponde a su cinta o a su coste",
    )
    keys = ("status", "reason", "net_return", "liquidated_net_return", "max_drawdown")
    return dict(
        cost_bps=cost,
        **{key: row[key] for key in keys},
        costs=row["costs"],
        turnover=row["turnover"],
        steps=row["steps"],
    )


class NativePolicyExecutor:
    """Ejecutor de un brazo aprendido con su binario nativo y el lanzador común.

    ``diagnostic`` lanza el diagnóstico explícito en CPU de los binarios, limitado a 32
    transiciones, y solo sirve para pruebas técnicas. ``stop_after`` pide una pausa al
    alcanzar esas transiciones, también antes de cualquier actualización.
    """

    def __init__(self, engine, *, diagnostic=False, stop_after=None, python=sys.executable):
        _require(engine in BINARIES, f"Motor nativo desconocido: {engine}")
        self.engine, self.diagnostic, self.stop_after, self.python = (
            engine,
            diagnostic,
            stop_after,
            python,
        )

    def config(self, stage, job):
        return (ppo_config if self.engine == "native_ppo" else klpo_config)(stage, job)

    def launch(self, config, output, *, train=(), validation=(), audit=None, tapes=()):
        command = [self.python, str(LAUNCHER), "--binary", str(binary_path(self.engine))]
        command += ["--config", str(config), "--output", str(output)]
        for option, paths in (
            ("--train-tape", train),
            ("--validation-tape", validation),
            ("--audit-tape", tapes),
        ):
            for path in paths:
                command += [option, str(path)]
        if audit is not None:
            command += ["--audit-run", str(audit)]
        elif self.stop_after is not None:
            command += ["--stop-after", str(self.stop_after)]
        if (output / "identity.json").is_file():
            command.append("--resume")
        if self.diagnostic:
            command.append("--diagnostic")
        # El lanzador importa el mismo código que la etapa, no otra instalación del paquete.
        paths = [str(ROOT / "src"), *filter(None, [os.environ.get("PYTHONPATH")])]
        environment = dict(os.environ, PYTHONPATH=os.pathsep.join(paths))
        result = subprocess.run(
            command, env=environment, capture_output=True, text=True, check=False
        )
        if result.returncode in PAUSED_CODES:
            return False
        if result.returncode != 0:
            raise RuntimeError(f"{self.engine} terminó con {result.returncode}: {result.stderr}")
        return True

    def __call__(self, job, tapes, folder, *, stage, resume, stop, anchor):
        policies = stage["policies"]
        _require(
            policies["evaluation_costs_bps"] == EVALUATION_COSTS,
            "Los costes de evaluación de la etapa no coinciden con los del motor nativo",
        )
        if job["kind"] == FIT:
            config = folder / "config.json"
            document = self.config(stage, job)
            if config.is_file():
                _require(
                    read_manifest(config)[0] == document,
                    f"{job['id']}: la configuración nativa cambió",
                )
            else:
                atomic_json(config, document)
            fit = folder / "fit"
            if not self.launch(
                config, fit, train=tapes.paths["train"], validation=[tapes.paths["validation"]]
            ):
                return dict(status="paused")
            run = read_manifest(fit / "run.json")[0]
            _require(run["status"] == "completed", f"{job['id']}: el ajuste no terminó")
            report = self.fit_report(job, run, policies)
        else:
            _require(job["kind"] == CARRY, "Solo los ajustes y traslados usan el motor nativo")
            fit = self.anchor_run(job, folder, anchor)
            config = fit.parent / "config.json"
            report = dict(transitions=0, updates=0, selection=None, policy=anchor["policy"])
        costs = policies["evaluation_costs_bps"]
        if tapes.evaluation is None:
            from .campaign_stage import episode

            records = [episode(cost, failure=tapes.failure["reason"]) for cost in costs]
        else:
            output = folder / "evaluation"
            if not self.launch(config, output, audit=fit, tapes=[tapes.paths["evaluation"]]):
                return dict(status="paused")
            evaluation = _payload(output / "evaluation.json")
            _require(
                evaluation["status"] == "completed"
                and evaluation["identity"]["policy_sha256"] == report["policy"]["sha256"],
                f"{job['id']}: la evaluación no corresponde a la política elegida",
            )
            manifest = sha256(Path(tapes.paths["evaluation"]) / "manifest.json")
            records = [
                _record(row, cost, manifest)
                for row, cost in zip(evaluation["metrics"], costs, strict=True)
            ]
        return dict(report, status="completed", evaluation=records)

    def fit_report(self, job, run, policies):
        if self.engine == "native_klpo":
            best = run["best"]
            sha, extra = best["actor_sha256"], dict(waves=run["consumed_waves"])
        else:
            sha, extra = run["selected_policy_sha256"], {}
        _require(isinstance(sha, str), f"{job['id']}: falta la huella de la política elegida")
        selection = dict(
            metric=policies["selection"]["metric"],
            partition="validation",
            policy="greedy_argmax",
            evaluations=run["evaluations"],
            best=run["best"],
        )
        return dict(
            transitions=run["transitions"],
            updates=run["optimizer_steps"],
            selection=selection,
            policy=dict(id=job["id"], sha256=sha),
            **extra,
        )

    @staticmethod
    def anchor_run(job, folder, anchor):
        """Ajuste del ancla: la etapa guarda su carpeta relativa a la salida."""
        parts = PurePosixPath(job["id"]).parts
        _require(
            folder.parts[-len(parts) - 2 :] == ("jobs", *parts, "run")
            and PurePosixPath(anchor["run"]).parts[0] == "jobs",
            "El traslado no encuentra la salida de la etapa",
        )
        root = folder.parents[len(parts) + 1]
        fit = root / anchor["run"] / "fit"
        _require((fit / "run.json").is_file(), f"{job['id']}: falta el ajuste de su ancla")
        return fit
