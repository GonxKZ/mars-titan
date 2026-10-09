"""Etapa de refuerzo de la campaña con máscaras: políticas financieras por ventana walk-forward.

La etapa parte de una campaña base confirmada (`training.masked_campaign`). En cada ventana
de política, todos los brazos observan las mismas cintas, montadas con las predicciones
congeladas y los recibos de cada brazo predictor declarado (`simulation.window_tapes`). Los
brazos son las variantes PPO identificadas, KLPO terminal como brazo principal del
contraste, Double DQN y tres referencias sin aprendizaje: efectivo, comprar y mantener y la
regla fija sobre la predicción. Comparten seis acciones, las semillas 42, 43 y 44 y el
presupuesto de transiciones declarado antes de evaluar. La selección usa el criterio de
cartera declarado sobre la cinta de validación, nunca el error del predictor.

Variante B: cada política se ajusta en la primera ventana de política y cada `period`
ventanas, como en la campaña base. Las intermedias evalúan sin ajuste la política
seleccionada en su ancla, sobre el universo del ancla.

Cada familia de brazos tiene un ejecutor y declara las capacidades del motor que necesita.
`run` se detiene con el bloqueo de aprendizaje antes de abrir fuentes, crear salidas o
lanzar binarios, y después exige todas las capacidades del plan antes de crear la salida.
Cada trabajo confirma un recibo con sus cintas, su informe y un registro por coste de
evaluación, también cuando el episodio falla. Los trabajos confirmados no se repiten y los
pendientes se reanudan en su carpeta.
"""

import argparse
import fcntl
import hashlib
import json
import math
import os
import re
from collections import Counter
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from mars_titan.data.cohort_files import read_manifest, safe_destination
from mars_titan.data.storage import atomic_json, outside_source, sha256
from mars_titan.environments.walk_forward_receipt import read_window_receipt
from mars_titan.training import masked_campaign
from mars_titan.training.campaign_plan import plan_campaign
from mars_titan.training.learning_hold import LearningHoldError, require_learning_allowed

from . import window_tapes
from .policy_plan import (
    CARRY,
    FIT,
    _number,
    _require,
    count_stage,
    load_stage,
    plan_stage,
)

RUN_KIND = "historical_masked_rl_stage_run"
RECEIPT_KIND = "masked_rl_job"
STATUSES = ("completed", "ruined", "failed")
_HEX = re.compile(r"[a-f0-9]{64}")

# Capacidades del motor que puede necesitar un ejecutor. Las que tienen sonda se comprueban
# con el motor instalado. Las demás son piezas que todavía no existen.
CAPABILITIES = {
    "native_accounting": dict(
        probe="native_library",
        pending="Compilar la biblioteca nativa de simulación (preset native-release)",
    ),
    "native_cn_a_share_rules": dict(
        probe="native_cn_rules",
        pending=(
            "El motor nativo debe aplicar las reglas de acciones A (lotes, resto impar, "
            "bandas diarias y timbre) que exige una cinta china reconstruida"
        ),
    ),
    "native_policy_reconstructed_tapes": dict(
        probe=None,
        pending=(
            "mars-titan-ppo solo admite fuentes sintéticas. Falta admitir cintas reconstruidas "
            "con su auditoría walk-forward, ajustar con el presupuesto declarado, seleccionar "
            "con el criterio de cartera y evaluar el estado elegido sin aprendizaje"
        ),
    ),
    "native_klpo_financial_runner": dict(
        probe=None,
        pending=(
            "KlpoLearningController no tiene orden ejecutable que recoja oleadas sobre cintas, "
            "seleccione en validación y evalúe el estado elegido"
        ),
    ),
}


class MissingCapability(RuntimeError):
    """El motor instalado no admite algo que el plan necesita."""


class Paused(Exception):
    """Parada solicitada o trabajo pendiente en una barrera confirmada."""


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


# Capacidades del motor


def _probe_tape():
    """Cinta sintética mínima de un activo A, solo para preguntar al motor por sus reglas."""
    from .market import MarketTape

    day = 86_400_000_000
    prices = np.tile([10.0, 10.0, 10.0, 10.0, 1e6], (3, 1, 1))
    times = 1_672_704_000_000_000 + day * np.arange(3, dtype=np.int64)
    return MarketTape(
        prices, times, ["CN/600000.SS"], np.zeros((3, 1)), domain="synthetic", currency="CNY"
    )


def probe_capabilities(library=None):
    """Estado de cada capacidad con el motor instalado, sin leer datos ni lanzar binarios."""
    from .environment import FinancialEnv
    from .market_rules import china_a_share_instrument
    from .native_runtime import load_library

    result = {}
    for name, entry in CAPABILITIES.items():
        reason = entry["pending"]
        try:
            if entry["probe"] == "native_library":
                load_library(library)
                reason = None
            elif entry["probe"] == "native_cn_rules":
                tape = _probe_tape()
                rules = {asset: china_a_share_instrument(asset) for asset in tape.assets}
                FinancialEnv(tape, backend="native", native_library=library, instruments=rules)
                reason = None
        except (ValueError, OSError) as error:
            reason = f"{entry['pending']}: {error}"
        result[name] = dict(available=reason is None, reason=reason)
    return result


def requirements(job, executors):
    """Capacidades que necesita un trabajo según su ejecutor y su mercado."""
    executor = executors[job["engine"]]
    names = list(executor["requires"])
    if executor["native"] and job["market"] == "CN":
        names.append("native_cn_a_share_rules")
    return names


def missing_capabilities(jobs, executors, available):
    """Capacidades ausentes agrupadas por motor y mercado."""
    missing = {}
    for job in jobs:
        for name in requirements(job, executors):
            if not available.get(name, {}).get("available"):
                key = f"{job['engine']}/{job['market']}"
                missing.setdefault(key, {})[name] = available.get(name, {}).get("reason")
    return missing


# Ejecutores


@dataclass(frozen=True)
class PolicyTapes:
    """Cintas comunes de un trabajo: universo, ajuste, validación y evaluación."""

    universe: tuple
    train: tuple
    validation: object
    evaluation: object
    failure: dict | None
    paths: dict
    identity: dict


def episode(cost, result=None, *, failure=None):
    """Registro de un episodio de evaluación, también cuando falla o se arruina."""
    record = dict(
        cost_bps=cost,
        status="failed",
        reason=failure,
        net_return=None,
        liquidated_net_return=None,
        max_drawdown=None,
        costs=None,
        turnover=None,
        steps=0,
    )
    if result is None:
        return record
    values = result["financial_validation"]
    completed = values["completed"]
    ruined = values["invalid_reason"] == "ruined"
    return dict(
        record,
        status="ruined" if ruined else "completed" if completed else "failed",
        reason=values["invalid_reason"],
        net_return=values["net_return"],
        liquidated_net_return=result["terminal_liquidation"]["net_return"],
        max_drawdown=values["max_drawdown"],
        costs=values["costs"],
        turnover=values["turnover"],
        steps=values["steps"],
    )


def market_rules(tape, market):
    """Reglas de acciones A para una cinta china y ninguna para EE. UU."""
    from .market_rules import china_a_share_instrument

    if market != "CN":
        return None
    return {asset: china_a_share_instrument(asset) for asset in tape.assets}


def evaluate_policy(tapes, policy, stage, market, *, backend, seed=42):
    """Evaluar una política fija o congelada en la cinta de evaluación con cada coste."""
    from .environment import FinancialEnv
    from .evaluation import evaluate

    policies = stage["policies"]
    environment = {
        key: value
        for key, value in policies["environment"].items()
        if key != "dividend_payment_lag_sessions"
    }
    records = []
    for cost in policies["evaluation_costs_bps"]:
        if tapes.evaluation is None:
            records.append(episode(cost, failure=tapes.failure["reason"]))
            continue
        env = FinancialEnv(
            tapes.evaluation,
            backend=backend,
            instruments=market_rules(tapes.evaluation, market),
            **dict(environment, cost_bps=cost),
        )
        records.append(episode(cost, evaluate(env, policy, seed=seed)))
    return records


def reference_executor(backend):
    """Ejecutor de las referencias sin aprendizaje con la contabilidad indicada."""
    from .evaluation import fixed_policy

    def run(job, tapes, folder, *, stage, resume, stop, anchor):
        records = evaluate_policy(
            tapes, fixed_policy(job["arm"]), stage, job["market"], backend=backend
        )
        return dict(
            status="completed",
            transitions=0,
            updates=0,
            selection=None,
            policy=None,
            evaluation=records,
        )

    return run


def _pending_engine(engine):
    def run(*_args, **_kwargs):
        raise MissingCapability(f"El ejecutor {engine} sobre cintas reconstruidas no existe")

    return run


EXECUTORS = {
    "reference": dict(
        run=reference_executor("native"), requires=("native_accounting",), native=True
    ),
    "native_ppo": dict(
        run=_pending_engine("native_ppo"),
        requires=("native_policy_reconstructed_tapes",),
        native=True,
    ),
    "native_klpo": dict(
        run=_pending_engine("native_klpo"),
        requires=("native_policy_reconstructed_tapes", "native_klpo_financial_runner"),
        native=True,
    ),
}


# Comprobación de informes y métricas


def _record(record, cost, failure):
    _require(
        isinstance(record, dict)
        and record.get("cost_bps") == cost
        and record.get("status") in STATUSES
        and type(record.get("steps")) is int
        and record["steps"] >= 0,
        f"Falta el episodio con coste {cost} o no conserva su estado",
    )
    if failure is not None:
        _require(
            record["status"] == "failed" and record["reason"] == failure,
            "Una cinta de evaluación fallida no puede producir episodios completos",
        )
    if record["status"] == "failed":
        _require(
            isinstance(record["reason"], str) and record["reason"],
            "Un episodio fallido conserva su motivo",
        )
        return
    values = [record.get(key) for key in ("net_return", "liquidated_net_return", "max_drawdown")]
    _require(
        all(_number(value, -math.inf) for value in values),
        "Un episodio terminado necesita métricas finitas",
    )
    if record["status"] == "completed":
        _require(record["liquidated_net_return"] > -1, "Un episodio sin ruina conserva patrimonio")


def check_report(stage, job, report, tapes, anchor=None):
    """Exigir el contrato del informe de un ejecutor antes de confirmar el trabajo."""
    policies = stage["policies"]
    _require(
        isinstance(report, dict)
        and report.get("status") == "completed"
        and type(report.get("updates")) is int
        and type(report.get("transitions")) is int
        and isinstance(report.get("evaluation"), list),
        f"{job['id']}: el ejecutor no devuelve un informe completo",
    )
    costs = policies["evaluation_costs_bps"]
    _require(
        len(report["evaluation"]) == len(costs),
        f"{job['id']}: cada coste necesita su episodio, también si falla",
    )
    failure = None if tapes.failure is None else tapes.failure["reason"]
    for record, cost in zip(report["evaluation"], costs, strict=True):
        _record(record, cost, failure)
    if job["kind"] == FIT:
        selection, policy = report.get("selection"), report.get("policy")
        _require(
            isinstance(selection, dict)
            and selection.get("metric") == policies["selection"]["metric"]
            and selection.get("partition") == "validation"
            and report["transitions"] == policies["budget"]["transitions"]
            and report["updates"] >= 0
            and isinstance(policy, dict)
            and set(policy) == {"id", "sha256"}
            and policy["id"] == job["id"]
            and _HEX.fullmatch(str(policy["sha256"])),
            f"{job['id']}: el ajuste no aplica el presupuesto, el criterio de cartera o "
            "no identifica la política elegida",
        )
    elif job["kind"] == CARRY:
        _require(
            report["policy"] == anchor["policy"]
            and report.get("selection") is None
            and report["transitions"] == report["updates"] == 0,
            f"{job['id']}: el traslado no evalúa sin ajuste la política de su ancla",
        )
    else:
        _require(
            report.get("policy") is None
            and report.get("selection") is None
            and report["transitions"] == report["updates"] == 0,
            f"{job['id']}: una referencia no aprende ni selecciona",
        )


def summarize(stage, receipts):
    """Métricas por brazo y coste con todos los episodios, también fallidos y arruinados.

    La media del crecimiento logarítmico liquidado se calcula sobre los episodios completos
    y se publica con su denominador. Fallos y ruinas se cuentan aparte y nunca se omiten.
    """
    order = [*stage["policies"]["policies"], *stage["policies"]["references"]]
    values = {}
    for receipt in receipts.values():
        identity = receipt["identity"]
        for record in receipt["evaluation"]:
            key = (identity["arm"], record["cost_bps"])
            values.setdefault(key, []).append(record)
    result = {}
    for arm in order:
        for cost in stage["policies"]["evaluation_costs_bps"]:
            records = values.get((arm, cost), [])
            if not records:
                continue
            completed = [r for r in records if r["status"] == "completed"]
            growth = [math.log1p(r["liquidated_net_return"]) for r in completed]
            reasons = Counter(r["reason"] for r in records if r["status"] == "failed")
            result.setdefault(arm, {})[str(cost)] = dict(
                episodes=len(records),
                completed=len(completed),
                ruined=sum(r["status"] == "ruined" for r in records),
                failed=sum(r["status"] == "failed" for r in records),
                failure_reasons=dict(sorted(reasons.items())),
                mean_liquidated_log_growth=math.fsum(growth) / len(growth) if growth else None,
                denominator="completed",
            )
    return result


# Ejecución


def _code():
    root = Path(__file__).parents[1]
    names = (
        "simulation/campaign_stage.py",
        "simulation/policy_plan.py",
        "simulation/window_tapes.py",
        "simulation/reconstructed_tape.py",
        "simulation/market.py",
        "simulation/storage.py",
        "simulation/environment.py",
        "simulation/evaluation.py",
        "simulation/portfolio.py",
        "environments/walk_forward_receipt.py",
        "training/masked_campaign.py",
        "training/campaign_plan.py",
    )
    return {name: sha256(root / name) for name in names}


def _base_receipts(base, campaign, stage):
    """Confirmar los trabajos base de los ámbitos y predictores de la etapa."""
    predictors = set(stage["policies"]["predictor"]["arms"])
    for job in plan_campaign(campaign):
        if job["scope"] not in stage["scopes"] or job["arm"] not in predictors:
            continue
        case, _, sources = base.resolve(job)
        receipt = base.confirmed(job, base.job_identity(job, case, sources))
        _require(receipt is not None, f"Falta confirmar {job['id']} en la campaña base")
        base.receipts[job["id"]] = receipt


def campaign_source(base, campaign_output, seed):
    """Recibo de ventana y predicciones emitidas del predictor elegido en la campaña base.

    El recibo publicado debe identificar al predictor elegido para la semilla y su huella
    de evaluación del mercado debe coincidir con la del trabajo confirmado.
    """

    def source(scope, market, window, predictor):
        folder = campaign_output / "windows" / scope / window / predictor / f"seed-{seed}"
        receipt = read_window_receipt(read_manifest(folder / f"{market}.json", 1024**2)[0])
        _, selected = base.selected(scope, window, predictor, seed)
        record = selected["predictions"]["evaluation"]
        expected = record["markets"].get(market)
        _require(
            receipt.parent == (selected["parent"]["id"], selected["parent"]["sha256"])
            and expected is not None
            and receipt.prediction_record(window_tapes.SEGMENT)
            == (expected["rows"], expected["sha256"]),
            f"El recibo de {scope}/{window}/{predictor} no corresponde al predictor elegido",
        )
        values = window_tapes.segment_predictions(
            campaign_output / record["path"], record["sha256"], market
        )
        return receipt, values

    return source


class _Tapes:
    """Universos y cintas confirmados de la etapa, con un conjunto abierto como máximo.

    `source(ámbito, mercado, ventana, predictor)` devuelve el recibo de la ventana y las
    predicciones emitidas en su tramo de evaluación. Cada recibo debe pertenecer a la
    ventana y al mercado pedidos, y los tramos de ajuste y validación deben terminar antes
    de la evaluación según los propios recibos.
    """

    def __init__(self, policies, source, edition, edition_id, output):
        self.policies, self.output = policies, output
        self._source, self.edition, self.edition_id = source, edition, edition_id
        self.lag = policies["environment"]["dividend_payment_lag_sessions"]
        self.key, self.current, self.evaluations = None, None, {}

    def source(self, job, window, predictor):
        receipt, values = self._source(job["scope"], job["market"], window, predictor)
        _require(
            receipt.fold == window and receipt.market == job["market"],
            f"El recibo pedido para {window} pertenece a otra ventana o a otro mercado",
        )
        return receipt, values

    def universe(self, job):
        """Universo del ancla con datos de ajuste y validación, guardado con su identidad."""
        predictor = self.policies["predictor"]["arms"][0]
        windows = [*job["train"], job["validation"]]
        sources = {window: self.source(job, window, predictor) for window in windows}
        identity = dict(
            edition_id=self.edition_id,
            rule=window_tapes.UNIVERSE_RULE,
            max_assets=self.policies["universe"]["max_assets"],
            predictor=predictor,
            segments={window: sources[window][0].sha256 for window in windows},
        )
        path = self.output / "universes" / job["scope"] / job["market"] / f"{job['anchor']}.json"
        safe_destination(path)
        if path.is_file():
            record = read_manifest(path, 8 * 1024**2)[0]
            _require(record["identity"] == identity, f"El universo de {path.name} ha cambiado")
            return tuple(record["assets"])
        admitted = {}
        for window, (receipt, values) in sources.items():
            role = "validation" if window == job["validation"] else "train"
            tape, _ = window_tapes.build_segment_tape(
                self.edition, receipt, values, market=job["market"], role=role, lag=self.lag
            )
            admitted[window] = window_tapes.admission(tape)
        assets = window_tapes.select_universe(
            [admitted[window] for window in job["train"]],
            admitted[job["validation"]],
            self.policies["universe"]["max_assets"],
        )
        atomic_json(path, dict(identity=identity, assets=assets))
        return tuple(assets)

    def tape(self, job, role, window, universe):
        """Cinta de un tramo restringida al universo, confirmada en disco o construida.

        Devuelve carpeta, cinta (o None si la evaluación excluye un activo del universo),
        fallo y tramo del recibo.
        """
        from .storage import read_tape, write_tape

        receipt, values = self.source(job, window, job["predictor"])
        folder = self.output / "tapes" / job["scope"] / job["market"] / job["predictor"]
        folder = folder / job["anchor"] / f"{role}-{window}"
        safe_destination(folder)
        bounds = receipt.segment(window_tapes.SEGMENT)
        expected = dict(receipt_sha256=receipt.sha256, universe_sha256=_digest(list(universe)))
        failure_path = folder / "failure.json"
        if failure_path.is_file():
            record = read_manifest(failure_path, 8 * 1024**2)[0]
            _require(record["identity"] == expected, f"La cinta fallida {folder.name} cambió")
            return folder, None, record["failure"], bounds
        if (folder / "manifest.json").is_file():
            tape = read_tape(folder)
        else:
            tape, report = window_tapes.build_segment_tape(
                self.edition,
                receipt,
                values,
                market=job["market"],
                role=role,
                lag=self.lag,
                # El universo guarda claves `mercado/símbolo` y la edición pide símbolos.
                symbols=[asset.split("/", 1)[1] for asset in universe],
            )
            if tuple(tape.assets) != universe:
                # Solo la evaluación puede excluir un activo del universo: el universo se
                # eligió entre los admitidos en ajuste y validación.
                _require(role == "evaluation", f"El universo no es admisible en {role}")
                failure = dict(reason="universe_assets_excluded", excluded=report["excluded"])
                atomic_json(failure_path, dict(identity=expected, failure=failure))
                return folder, None, failure, bounds
            write_tape(tape, folder)
        _require(
            tuple(tape.assets) == universe
            and [item["receipt_sha256"] for item in tape.identity["audit"]["walk_forward"]]
            == [receipt.sha256],
            f"La cinta de {folder.name} no corresponde a su recibo o su universo",
        )
        return folder, tape, None, bounds

    def open(self, job):
        """Cintas comunes del trabajo. Se reutilizan entre brazos, semillas y referencias."""
        key = (job["scope"], job["market"], job["predictor"], job["anchor"])
        if self.key != key:
            self.key, self.evaluations = None, {}
            universe = self.universe(job)
            train = [self.tape(job, "train", window, universe) for window in job["train"]]
            validation = self.tape(job, "validation", job["validation"], universe)
            self.current, self.key = (universe, train, validation), key
        universe, train, validation = self.current
        if job["window"] not in self.evaluations:
            self.evaluations = {
                job["window"]: self.tape(job, "evaluation", job["window"], universe)
            }
        evaluation = self.evaluations[job["window"]]
        segments = [item[3] for item in (*train, validation, evaluation)]
        _require(
            all(a[1] <= b[0] for a, b in zip(segments, segments[1:], strict=False)),
            f"La política de {job['window']} usaría tramos posteriores a su evaluación",
        )
        return PolicyTapes(
            universe=universe,
            train=tuple(item[1] for item in train),
            validation=validation[1],
            evaluation=evaluation[1],
            failure=evaluation[2],
            paths=dict(
                train=[str(item[0]) for item in train],
                validation=str(validation[0]),
                evaluation=None if evaluation[1] is None else str(evaluation[0]),
            ),
            identity=dict(
                universe_sha256=_digest(list(universe)),
                train=[item[1].sha256 for item in train],
                validation=validation[1].sha256,
                evaluation=None if evaluation[1] is None else evaluation[1].sha256,
                failure=evaluation[2],
            ),
        )


class _Stage:
    """Estado confirmado de la etapa y verificación de cada recibo."""

    def __init__(self, stage, tapes, output, identity, executors, stop):
        self.stage, self.tapes, self.output = stage, tapes, output
        self.identity_sha256 = _digest(identity)
        self.executors, self.stop = executors, stop
        self.receipts = {}

    def folder(self, job):
        return self.output / "jobs" / job["id"]

    def job_identity(self, job, tapes):
        fields = ("id", "scope", "market", "window", "anchor", "train", "validation")
        fields += ("predictor", "arm", "engine", "seed", "kind")
        identity = dict(
            stage_identity_sha256=self.identity_sha256,
            **{name: job[name] for name in fields},
            tapes=tapes.identity,
            anchor_fit=None,
        )
        if job["kind"] == CARRY:
            (anchor,) = job["depends"]
            _require(anchor in self.receipts, f"{job['id']} depende de {anchor}, sin confirmar")
            identity["anchor_fit"] = dict(
                job=anchor, receipt_sha256=self.receipts[anchor]["sha256"]
            )
        return identity

    def confirmed(self, job, identity):
        path = self.folder(job) / "receipt.json"
        if not path.is_file():
            return None
        receipt, digest = read_manifest(path, 8 * 1024**2)
        _require(receipt.get("identity") == identity, f"{job['id']} cambió de identidad")
        return dict(receipt, sha256=digest)

    def confirm(self, job, identity, tapes, report, anchor):
        check_report(self.stage, job, report, tapes, anchor)
        receipt = dict(
            schema_version=1,
            kind=RECEIPT_KIND,
            status="completed",
            identity=identity,
            run=str((self.folder(job) / "run").relative_to(self.output)),
            transitions=report["transitions"],
            updates=report["updates"],
            selection=report["selection"],
            policy=report["policy"],
            evaluation=report["evaluation"],
            final_test_opened=False,
            confirmed_at_utc=datetime.now(UTC).isoformat(),
        )
        path = self.folder(job) / "receipt.json"
        atomic_json(path, receipt)
        return dict(receipt, sha256=sha256(path))

    def execute(self, jobs):
        """Recorrer el plan en orden y confirmar cada trabajo."""
        for job in jobs:
            if self.stop.requested:
                raise Paused
            tapes = self.tapes.open(job)
            identity = self.job_identity(job, tapes)
            receipt = self.confirmed(job, identity)
            if receipt is None:
                require_learning_allowed(f"el trabajo {job['id']}")
                folder = self.folder(job) / "run"
                safe_destination(folder)
                resume = folder.exists()
                folder.mkdir(parents=True, exist_ok=True)
                anchor = self.receipts[job["depends"][0]] if job["kind"] == CARRY else None
                report = self.executors[job["engine"]]["run"](
                    job,
                    tapes,
                    folder,
                    stage=self.stage,
                    resume=resume,
                    stop=self.stop,
                    anchor=anchor,
                )
                if report.get("status") == "paused":
                    raise Paused
                receipt = self.confirm(job, identity, tapes, report, anchor)
            self.receipts[job["id"]] = receipt
        return "completed"


def _identity(stage, views, edition_id):
    campaign = stage["campaign"]
    return dict(
        schema_version=1,
        kind=RUN_KIND,
        stage_sha256=stage["sha256"],
        policies_sha256=stage["policies"]["sha256"],
        campaign_sha256=campaign["sha256"],
        edition_id=edition_id,
        variant=campaign["variant"],
        views={
            scope: {window: value["sha256"] for window, value in record["windows"].items()}
            for scope, record in views.items()
        },
        code=_code(),
        final_test_opened=False,
    )


def _summary(output, identity, jobs, state, status, **extra):
    planned = Counter(job["kind"] for job in jobs)
    done = Counter(job["kind"] for job in jobs if job["id"] in state.receipts)
    summary = dict(
        schema_version=1,
        kind=RUN_KIND,
        status=status,
        identity_sha256=_digest(identity),
        planned=dict(planned),
        completed=dict(done),
        metrics=summarize(state.stage, state.receipts),
        jobs={job["id"]: job["id"] in state.receipts for job in jobs},
        final_test_opened=False,
        updated_at_utc=datetime.now(UTC).isoformat(),
        **extra,
    )
    atomic_json(output / "summary.json", summary)
    return summary


def check_stage(path, *, library=None):
    """Validar, planificar y contar sin leer datos, e informar de las capacidades del motor."""
    stage = load_stage(path)
    jobs = plan_stage(stage)
    policies = stage["policies"]
    available = probe_capabilities(library)
    return dict(
        status="checked",
        name=stage["name"],
        variant=stage["campaign"]["variant"],
        stage_sha256=stage["sha256"],
        policies_sha256=policies["sha256"],
        campaign_sha256=stage["campaign"]["sha256"],
        predictor=policies["predictor"],
        universe=policies["universe"],
        selection=policies["selection"],
        budget=policies["budget"],
        seeds=policies["seeds"],
        contrasts=policies["contrasts"],
        counts=count_stage(stage, jobs),
        capabilities=available,
        missing_capabilities=missing_capabilities(jobs, EXECUTORS, available),
        scientific_training_started=False,
        final_test_opened=False,
    )


def run_stage(
    path, views, campaign_output, edition, output, *, executors=None, capabilities=None, stop=None
):
    """Ejecutar o reanudar la etapa sobre una campaña base confirmada.

    `executors` sustituye los ejecutores por familia y `capabilities` el estado del motor.
    El bloqueo de aprendizaje se comprueba antes de todo y antes de cada trabajo pendiente.
    Las capacidades del plan se exigen antes de abrir fuentes o crear la salida.
    """
    from mars_titan.training.checkpoints import StopRequest

    from .reconstructed_tape import read_edition

    require_learning_allowed("la etapa de políticas financieras de la campaña")
    stage = load_stage(path)
    jobs = plan_stage(stage)
    count_stage(stage, jobs)
    executors = dict(EXECUTORS if executors is None else executors)
    _require(
        {job["engine"] for job in jobs} <= set(executors)
        and all(set(entry) == {"run", "requires", "native"} for entry in executors.values()),
        "Falta un ejecutor para alguna familia de brazos",
    )
    available = probe_capabilities() if capabilities is None else capabilities
    missing = missing_capabilities(jobs, executors, available)
    if missing:
        raise MissingCapability(
            "El motor no admite todo el plan: " + json.dumps(missing, ensure_ascii=False)
        )
    campaign = stage["campaign"]
    _require(
        isinstance(views, dict) and set(views) == set(campaign["scopes"]),
        "Se necesitan las vistas de todos los ámbitos de la campaña base",
    )
    views = {scope: Path(value) for scope, value in views.items()}
    campaign_output, edition, output = Path(campaign_output), Path(edition), Path(output)
    safe_destination(output)
    for protected in (*views.values(), campaign_output, edition, Path("dataset")):
        outside_source(protected, output)
        outside_source(output, protected)
    edition_id = read_edition(edition)["edition_id"]
    _, base = masked_campaign._confirmed_state(campaign["path"], views, campaign_output)
    _base_receipts(base, campaign, stage)
    identity = _identity(stage, base.views, edition_id)
    output.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(output / ".lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        marker = output / "stage.json"
        if marker.exists():
            _require(
                read_manifest(marker, 8 * 1024**2)[0] == identity,
                "La salida pertenece a otra etapa, campaña, edición, vista o código",
            )
        else:
            _require(
                all(p.name in {".lock", "summary.json"} for p in output.iterdir()),
                "La salida sin identidad contiene artefactos ajenos",
            )
            atomic_json(marker, identity)
        source = campaign_source(base, campaign_output, stage["policies"]["predictor"]["seed"])
        tapes = _Tapes(stage["policies"], source, edition, edition_id, output)
        state = _Stage(stage, tapes, output, identity, executors, None)
        signals = StopRequest() if stop is None else nullcontext(stop)
        _summary(output, identity, jobs, state, "running")
        try:
            with signals as state.stop:
                status = state.execute(jobs)
        except Paused:
            status = "paused"
        except LearningHoldError as error:
            _summary(output, identity, jobs, state, "blocked", error=str(error))
            raise
        except BaseException as error:
            _summary(output, identity, jobs, state, "failed", error=str(error))
            raise
        return _summary(output, identity, jobs, state, status)
    finally:
        os.close(descriptor)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check", help="Validar, contar y revisar capacidades")
    execute = commands.add_parser("run", help="Ejecutar o reanudar la etapa")
    for command in (check, execute):
        command.add_argument("--stage", type=Path, required=True)
    execute.add_argument("--views", action="append", required=True)
    execute.add_argument("--campaign-output", type=Path, required=True)
    execute.add_argument("--edition", type=Path, required=True)
    execute.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "check":
        result = check_stage(args.stage)
    else:
        result = run_stage(
            args.stage,
            masked_campaign._views_argument(args.views),
            args.campaign_output,
            args.edition,
            args.output,
        )
        result.pop("jobs")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("status") in {"completed", "checked"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
