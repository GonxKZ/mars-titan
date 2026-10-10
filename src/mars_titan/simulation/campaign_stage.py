"""Etapa de refuerzo de la campaña con máscaras: políticas financieras por ventana walk-forward.

La etapa parte de una campaña base confirmada (`training.masked_campaign`). En cada ventana
de política, todos los brazos de un predictor observan las mismas cintas, montadas con sus
predicciones congeladas y sus recibos de ventana (`simulation.window_tapes`), y todos los
predictores comparten el universo del ancla. KLPO terminal, brazo principal del contraste, y
cinco referencias sin aprendizaje se aplican a todos los predictores con productor en la
campaña. Tres usan la predicción (efectivo, comprar y mantener y la regla fija del 50 %) y
dos no la usan: la cartera 1/N reequilibrada cada 21 sesiones, con los mismos costes y
reglas, y el índice de mercado comprado y mantenido en su propia cinta de un activo. Las variantes
PPO identificadas y Double DQN se comparan sobre los predictores del nivel de algoritmos.
Comparten seis acciones, las semillas 42, 43 y 44 y el presupuesto de transiciones
declarado antes de evaluar. La selección usa el criterio de cartera declarado sobre la
cinta de validación, nunca el error del predictor.

Un predictor sin predicciones del mercado en un tramo no detiene la etapa. Si le faltan en
el ajuste o la validación del ancla, sus ajustes y traslados no se ejecutan y registran
todos sus episodios como fallidos con el motivo `predictor_without_predictions`. Si le
faltan en la evaluación, los episodios de esa ventana fallan con el mismo motivo.

Variante B: cada política se ajusta en la primera ventana de política y cada `period`
ventanas, como en la campaña base. Las intermedias evalúan sin ajuste la política
seleccionada en su ancla, sobre el universo del ancla.

Cada familia de brazos tiene un ejecutor y declara las capacidades del motor que necesita.
`run` se detiene con el bloqueo de aprendizaje antes de abrir fuentes, crear salidas o
lanzar binarios, y después exige todas las capacidades del plan antes de crear la salida.
PPO y Double DQN se ejecutan con `mars-titan-ppo` y KLPO con `mars-titan-klpo`, ambos sobre
las cintas reconstruidas de la etapa (`simulation.native_policy_runs`). KLPO solo consume
oleadas completas: usa las que caben en el presupuesto de transiciones declarado.
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
import subprocess
from collections import Counter
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from mars_titan.data.cohort_files import read_manifest, safe_destination
from mars_titan.data.storage import atomic_json, outside_source, sha256
from mars_titan.environments.walk_forward_receipt import read_window_receipt
from mars_titan.posttraining import staged_chain
from mars_titan.training import campaign_schedule, masked_campaign
from mars_titan.training.campaign_plan import _arm_specs, plan_campaign
from mars_titan.training.learning_hold import LearningHoldError, require_learning_allowed

from . import native_policy_runs, window_tapes
from .policy_plan import (
    BASE_SELECTED,
    CARRY,
    FIT,
    MARKET_INDEX,
    REFERENCE,
    _number,
    _require,
    count_stage,
    count_tapes,
    load_stage,
    plan_stage,
    window_sensitivity,
)
from .reconstructed_tape import NoAdmittedAssets

RUN_KIND = "historical_masked_rl_stage_run"
RECEIPT_KIND = "masked_rl_job"
STATUSES = ("completed", "ruined", "failed")
_HEX = re.compile(r"[a-f0-9]{64}")

# Capacidades del motor que puede necesitar un ejecutor. Cada una se comprueba con el motor
# instalado: la biblioteca de simulación o la identidad que declaran los binarios de política.
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
        probe="native_ppo",
        pending=(
            "Compilar mars-titan-ppo (preset native-ppo-release) con el esquema 4, que ajusta "
            "sobre cintas reconstruidas, selecciona en validación y evalúa el estado elegido"
        ),
    ),
    "native_klpo_financial_runner": dict(
        probe="native_klpo",
        pending=(
            "Compilar mars-titan-klpo (preset native-ppo-release), que recoge oleadas KLPO "
            "sobre cintas reconstruidas, selecciona en validación y evalúa el estado elegido"
        ),
    ),
    "native_ppo_equity_and_costs": dict(
        probe="native_ppo",
        capability=native_policy_runs.EQUITY_AND_COSTS,
        pending=(
            "Compilar mars-titan-ppo con la evaluación que publica el patrimonio por sesión "
            "y acepta los costes declarados por la etapa"
        ),
    ),
    "native_klpo_equity_and_costs": dict(
        probe="native_klpo",
        capability=native_policy_runs.EQUITY_AND_COSTS,
        pending=(
            "Compilar mars-titan-klpo con la evaluación que publica el patrimonio por sesión "
            "y acepta los costes declarados por la etapa"
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


# Cierres de referencia de la consulta de reglas, con redondeos de medio céntimo.
RULE_QUERY_REFERENCES = (10.0, 9.99, 10.05, 2.675, 0.95)


def probe_cn_rules(library=None):
    """Preguntar a la biblioteca por las reglas de las acciones A sin construir ninguna cinta.

    Cargarla exige el contrato binario de reglas (tamaño del registro y paso v2). Después,
    los límites diarios de cada banda declarada para cada tablero se calculan en C++ y deben
    coincidir con los de Python. Ningún precio, activo ni sesión sale de esta consulta.
    """
    from .market_rules import PRICE_LIMITS
    from .native_runtime import load_library
    from .portfolio import Instrument

    native = load_library(library)
    for board, periods in PRICE_LIMITS.items():
        instrument = Instrument("CNY", price_limits=periods, rules=f"probe_{board}")
        for period in periods:
            for reference in RULE_QUERY_REFERENCES:
                expected = instrument.limits(reference, period.start)
                if native.price_limits(reference, period.band) != expected:
                    raise ValueError(f"Los límites de {board} no coinciden con los de Python")


def probe_capabilities(library=None):
    """Estado de cada capacidad con el motor instalado, sin leer datos.

    Los binarios de política solo se lanzan con `--capabilities`: declaran su nombre, sus
    capacidades y su identidad de compilación, que se conserva en el informe.
    """
    from .native_runtime import load_library

    result = {}
    for name, entry in CAPABILITIES.items():
        reason = entry["pending"]
        try:
            if entry["probe"] == "native_library":
                load_library(library)
                reason = None
            elif entry["probe"] == "native_cn_rules":
                probe_cn_rules(library)
                reason = None
            elif entry["probe"] in native_policy_runs.BINARIES:
                capability = entry.get("capability", name)
                identity = native_policy_runs.probe_binary(entry["probe"], capability)
                reason = None
        except (ValueError, OSError, subprocess.SubprocessError) as error:
            reason = f"{entry['pending']}: {error}"
        result[name] = dict(available=reason is None, reason=reason)
        if entry["probe"] in native_policy_runs.BINARIES and reason is None:
            result[name]["binary"] = {
                key: identity[key] for key in ("binary", "binary_sha256", "native_build_sha256")
            }
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
    unfit: dict | None
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
        equity=None,
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
        equity=None if not completed else result["equity"],
    )


def market_rules(tape, market):
    """Reglas de acciones A para una cinta china y ninguna para EE. UU."""
    from .market_rules import china_a_share_instrument

    if market != "CN":
        return None
    return {asset: china_a_share_instrument(asset) for asset in tape.assets}


def evaluate_policy(tapes, policy, stage, market, *, backend, seed=42, allocation=None):
    """Evaluar una política fija o congelada en la cinta de evaluación con cada coste."""
    from .environment import ALLOCATIONS, FinancialEnv
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
            allocation=ALLOCATIONS[0] if allocation is None else allocation,
            **dict(environment, cost_bps=cost),
        )
        records.append(episode(cost, evaluate(env, policy, seed=seed)))
    return records


def reference_executor(backend):
    """Ejecutor de las referencias sin aprendizaje con la contabilidad indicada."""
    from .evaluation import REFERENCE_ALLOCATIONS, fixed_policy

    def run(job, tapes, folder, *, stage, resume, stop, anchor):
        records = evaluate_policy(
            tapes,
            fixed_policy(job["arm"]),
            stage,
            job["market"],
            backend=backend,
            allocation=REFERENCE_ALLOCATIONS[job["arm"]],
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


def unfit_report(stage, tapes):
    """Informe de un ajuste o traslado sin datos de ajuste o validación: no se ejecuta."""
    reason = tapes.unfit["reason"]
    return dict(
        status="completed",
        transitions=0,
        updates=0,
        selection=None,
        policy=None,
        evaluation=[
            episode(cost, failure=reason) for cost in stage["policies"]["evaluation_costs_bps"]
        ],
    )


EXECUTORS = {
    "reference": dict(
        run=reference_executor("native"), requires=("native_accounting",), native=True
    ),
    "native_ppo": dict(
        run=native_policy_runs.NativePolicyExecutor("native_ppo"),
        requires=("native_policy_reconstructed_tapes", "native_ppo_equity_and_costs"),
        native=True,
    ),
    "native_klpo": dict(
        run=native_policy_runs.NativePolicyExecutor("native_klpo"),
        requires=(
            "native_policy_reconstructed_tapes",
            "native_klpo_financial_runner",
            "native_klpo_equity_and_costs",
        ),
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
            isinstance(record["reason"], str) and record["reason"] and record.get("equity") is None,
            "Un episodio fallido conserva su motivo y no publica patrimonio",
        )
        return
    values = [record.get(key) for key in ("net_return", "liquidated_net_return", "max_drawdown")]
    _require(
        all(_number(value, -math.inf) for value in values),
        "Un episodio terminado necesita métricas finitas",
    )
    if record["status"] == "completed":
        _require(record["liquidated_net_return"] > -1, "Un episodio sin ruina conserva patrimonio")
    _equity(record)


# Tolerancia relativa entre el patrimonio final de la serie y el retorno publicado. El motor
# nativo reconstruye la serie con sus recompensas logarítmicas.
EQUITY_TOLERANCE = 1e-9


def _equity(record):
    """Exigir el patrimonio por sesión de un episodio terminado, coherente con su retorno."""
    equity = record.get("equity")
    _require(
        isinstance(equity, dict)
        and set(equity) == {"basis", "close_times", "nav"}
        and isinstance(equity["close_times"], list)
        and isinstance(equity["nav"], list)
        and len(equity["close_times"]) == len(equity["nav"]) == record["steps"] + 1
        and all(type(value) is int for value in equity["close_times"])
        and all(
            a < b for a, b in zip(equity["close_times"], equity["close_times"][1:], strict=False)
        )
        and all(_number(value, 0) for value in equity["nav"]),
        "Un episodio terminado conserva su patrimonio en cada cierre",
    )
    nav = equity["nav"]
    ruined = record["status"] == "ruined"
    expected = nav[0] * (1 + record["net_return"])
    _require(
        nav[0] > 0
        and (nav[-1] == 0) is ruined
        and all(value > 0 for value in nav[:-1])
        and abs(nav[-1] - expected) <= EQUITY_TOLERANCE * max(1.0, abs(expected)),
        "El patrimonio por sesión no concilia con el retorno del episodio",
    )


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
    if job["kind"] != REFERENCE and tapes.unfit is not None:
        failure = tapes.unfit["reason"]
    for record, cost in zip(report["evaluation"], costs, strict=True):
        _record(record, cost, failure)
    if job["kind"] != REFERENCE and tapes.unfit is not None:
        _require(
            report.get("policy") is None
            and report.get("selection") is None
            and report["transitions"] == report["updates"] == 0,
            f"{job['id']}: sin predicciones de ajuste o validación no hay política que evaluar",
        )
    elif job["kind"] == FIT:
        selection, policy = report.get("selection"), report.get("policy")
        budget = policies["budget"]
        if job["engine"] == "native_klpo":
            # KLPO consume oleadas completas: las que caben en el presupuesto, sin superarlo.
            waves, wave = native_policy_runs.klpo_waves(
                tapes.train, budget["environments"], budget["transitions"]
            )
            spent = waves > 0 and report.get("waves") == waves
            spent = spent and 0 < report["transitions"] <= waves * wave
        else:
            spent = report["transitions"] == budget["transitions"]
        _require(
            isinstance(selection, dict)
            and selection.get("metric") == policies["selection"]["metric"]
            and selection.get("partition") == "validation"
            and spent
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
    """Métricas por predictor, brazo y coste con todos los episodios, también fallidos.

    La media del crecimiento logarítmico liquidado se calcula sobre los episodios completos
    y se publica con su denominador. Fallos y ruinas se cuentan aparte y nunca se omiten.
    """
    order = [*stage["policies"]["policies"], *stage["policies"]["references"]]
    values = {}
    for receipt in receipts.values():
        identity = receipt["identity"]
        for record in receipt["evaluation"]:
            key = (identity["predictor"], identity["arm"], record["cost_bps"])
            values.setdefault(key, []).append(record)
    result = {}
    for predictor in stage["predictors"]:
        for arm in order:
            for cost in stage["policies"]["evaluation_costs_bps"]:
                records = values.get((predictor, arm, cost), [])
                if not records:
                    continue
                completed = [r for r in records if r["status"] == "completed"]
                growth = [math.log1p(r["liquidated_net_return"]) for r in completed]
                reasons = Counter(r["reason"] for r in records if r["status"] == "failed")
                entry = result.setdefault(predictor, {}).setdefault(arm, {})
                entry[str(cost)] = dict(
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


def _base_receipts(base, campaign, stage, pairs=None):
    """Confirmar los trabajos base de los ámbitos y predictores de la etapa.

    También los de los auxiliares de los que parten, como los núcleos de CM-v1, porque el
    caso y la identidad de un brazo dependen de sus recibos. `pairs` limita la confirmación
    a esos pares (ámbito, ventana) al ejecutar una ventana.
    """
    predictors = set(stage["predictors"])
    predictors |= {
        spec["parent"]
        for spec in _arm_specs(campaign)
        if spec["arm"] in predictors and spec["parent"]
    }
    for job in plan_campaign(campaign):
        if job["scope"] not in stage["scopes"] or job["arm"] not in predictors:
            continue
        if pairs is not None and (job["scope"], job["window"]) not in pairs:
            continue
        case, _, sources = base.resolve(job)
        receipt = base.confirmed(job, base.job_identity(job, case, sources))
        _require(receipt is not None, f"Falta confirmar {job['id']} en la campaña base")
        base.receipts[job["id"]] = receipt


def campaign_source(base, campaign_output, seed):
    """Recibo de ventana y predicciones emitidas del predictor elegido en la campaña base.

    El recibo publicado debe identificar al predictor elegido para la semilla y su huella
    de evaluación del mercado debe coincidir con la del trabajo confirmado. Su
    `labels_used_until` debe ser la maduración real de las etiquetas que leyó ese predictor,
    recalculada aquí desde las vistas con `training.label_maturity`, de modo que una cinta
    nunca lleva predicciones de un predictor que ajustó pesos con sus filas. Si el trabajo
    no emitió filas del mercado en su evaluación, el recibo tampoco las declara y la fuente
    devuelve `None` en lugar de predicciones.
    """

    def source(scope, market, window, predictor):
        folder = campaign_output / "windows" / scope / window / predictor / f"seed-{seed}"
        receipt = read_window_receipt(read_manifest(folder / f"{market}.json", 1024**2)[0])
        _, selected = base.selected(scope, window, predictor, seed)
        record = selected["predictions"]["evaluation"]
        expected = record["markets"].get(market)
        declared = dict(receipt.predictions).get(window_tapes.SEGMENT)
        _require(
            receipt.parent == (selected["parent"]["id"], selected["parent"]["sha256"])
            and declared == (None if expected is None else (expected["rows"], expected["sha256"])),
            f"El recibo de {scope}/{window}/{predictor} no corresponde al predictor elegido",
        )
        _require(
            receipt.labels_used_until == base.labels_used_until(scope, window, selected),
            f"El recibo de {scope}/{window}/{predictor} no declara la maduración real de las "
            "etiquetas que leyó su predictor",
        )
        if expected is None:
            return receipt, None
        values = window_tapes.segment_predictions(
            campaign_output / record["path"], record["sha256"], market
        )
        return receipt, values

    return source


def chain_source(base, chain_output, seed, campaign_output):
    """Recibo y predicciones del predictor de la cadena de cada ventana.

    El predictor de la ventana k es el estado que el posentrenamiento elige con `val_k`:
    adaptador, continuación o padre congelado, y en la ventana 0 el estado elegido de la
    campaña base. Su `selection.json`, escrito el último, confirma la ventana: sin él no hay
    cinta. La selección se lee con `posttraining.staged_chain.read_selection`, el mismo
    lector que usa la etapa que la escribe, así que la regla de elección, los candidatos y
    la huella y el padre de cada recibo de mercado se comprueban con un único contrato.
    Además, el recibo confirmado del trabajo elegido debe tener la huella que fija la
    selección y las huellas de evaluación del mercado. Su `labels_used_until` debe ser la
    maduración real de las etiquetas de las vistas de la ventana y de la anterior (las del
    padre), recalculada aquí con `training.label_maturity`. Así ninguna cinta lleva
    predicciones de un estado que ajustó, eligió o calibró con etiquetas posteriores a su
    primera decisión.
    """
    from mars_titan.training.label_maturity import FIT_PARTITIONS, label_maturity

    maturity = {}

    def labels_used_until(scope, window):
        windows = base.views[scope]["windows"]
        names = list(windows)
        index = names.index(window)
        read = names[max(0, index - 1) : index + 1]
        for name in read:
            if (scope, name) not in maturity:
                maturity[(scope, name)] = label_maturity(windows[name]["path"], FIT_PARTITIONS)[0]
        return max(maturity[(scope, name)] for name in read)

    def source(scope, market, window, predictor):
        label = f"{scope}/{window}/{staged_chain.chain_arm(predictor)}"
        selection = staged_chain.read_selection(chain_output, scope, window, predictor, seed)
        _require(selection is not None, f"La cadena de {label} no tiene confirmada su selección")
        _require(
            market in selection["receipts"],
            f"La cadena de {label} no publica recibo de {market}",
        )
        receipt, selected = selection["receipts"][market], selection["selected"]
        # Solo la primera ventana elige el estado de la base, cuyo recibo está en la campaña.
        root = campaign_output if selection["parent_window"] is None else chain_output
        emitted, emitted_digest = read_manifest(
            root / "jobs" / selected["job"] / "receipt.json", 8 * 1024**2
        )
        record = emitted["predictions"][window_tapes.SEGMENT]
        expected = record["markets"].get(market)
        _require(
            emitted_digest == selected["receipt_sha256"]
            and expected is not None
            and dict(receipt.predictions).get(window_tapes.SEGMENT)
            == (expected["rows"], expected["sha256"]),
            f"El recibo de {market} de {label} no corresponde al estado elegido",
        )
        _require(
            receipt.labels_used_until == labels_used_until(scope, window),
            f"El recibo de {market} de {label} no declara la maduración real de las etiquetas "
            "que leyó el estado elegido",
        )
        values = window_tapes.segment_predictions(root / record["path"], record["sha256"], market)
        return receipt, values

    return source


def predictor_source(policies, base, campaign_output, chain_output=None):
    """Fuente de recibos y predicciones de las cintas según la regla declarada."""
    rule, seed = policies["predictor"]["source"], policies["predictor"]["seed"]
    if rule == BASE_SELECTED:
        return campaign_source(base, campaign_output, seed)
    _require(
        chain_output is not None,
        "Las cintas del predictor de la cadena necesitan la salida del posentrenamiento",
    )
    return chain_source(base, Path(chain_output), seed, campaign_output)


class _Tapes:
    """Universos y cintas confirmados de la etapa, con un conjunto abierto como máximo.

    `source(ámbito, mercado, ventana, predictor)` devuelve el recibo de la ventana y las
    predicciones emitidas en su tramo de evaluación, o `None` si no las hay. Cada recibo
    debe pertenecer a la ventana y al mercado pedidos, y los tramos de ajuste y validación
    deben terminar antes de la evaluación según los propios recibos. El universo de cada
    ancla es común a todos los predictores y se elige con `universe_predictor`.
    """

    def __init__(self, policies, source, edition, edition_id, output, universe_predictor):
        self.policies, self.output = policies, output
        self.universe_predictor = universe_predictor
        self._source, self.edition, self.edition_id = source, edition, edition_id
        self.lag = policies["environment"]["dividend_payment_lag_sessions"]
        self.key, self.current, self.evaluations = None, None, {}
        # Admisión de cada tramo con todos los activos, por recibo. Con la ventana en
        # expansión, cada ancla repite las ventanas anteriores y montar una cinta de EE. UU.
        # con todos sus activos cuesta en torno a un minuto.
        self.admissions = {}

    def source(self, job, window, predictor):
        receipt, values = self._source(job["scope"], job["market"], window, predictor)
        _require(
            receipt.fold == window and receipt.market == job["market"],
            f"El recibo pedido para {window} pertenece a otra ventana o a otro mercado",
        )
        return receipt, values

    def universe(self, job):
        """Universo del ancla con datos de ajuste y validación, guardado con su identidad."""
        predictor = self.universe_predictor
        windows = [*job["train"], job["validation"]]
        sources = {window: self.source(job, window, predictor) for window in windows}
        _require(
            all(values is not None for _, values in sources.values()),
            f"El predictor {predictor} del universo no tiene predicciones de {job['market']} "
            f"en el ajuste o la validación de {job['anchor']}",
        )
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
            key = (job["scope"], job["market"], window, receipt.sha256)
            if key not in self.admissions:
                tape, _ = window_tapes.build_segment_tape(
                    self.edition, receipt, values, market=job["market"], role="train", lag=self.lag
                )
                window_tapes.require_real_tape(tape, self.edition_id, f"universe-{window}")
                self.admissions[key] = window_tapes.admission(tape)
            admitted[window] = self.admissions[key]
        assets = window_tapes.select_universe(
            [admitted[window] for window in job["train"]],
            admitted[job["validation"]],
            self.policies["universe"]["max_assets"],
        )
        atomic_json(path, dict(identity=identity, assets=assets))
        return tuple(assets)

    def tape(self, job, role, window, universe, *, name=None):
        """Cinta de un tramo restringida al universo, confirmada en disco o construida.

        Devuelve carpeta, cinta, fallo y tramo del recibo. La cinta es None si el predictor
        no tiene predicciones del mercado en el tramo o si la evaluación excluye un activo del
        universo, y el fallo guarda el motivo. `name` separa la carpeta de una cinta con otros
        activos del mismo tramo, como la del índice de mercado.
        """
        from .storage import read_tape, write_tape

        receipt, values = self.source(job, window, job["predictor"])
        folder = self.output / "tapes" / job["scope"] / job["market"] / job["predictor"]
        folder = folder / job["anchor"] / (name or f"{role}-{window}")
        safe_destination(folder)
        bounds = receipt.segment(window_tapes.SEGMENT)
        expected = dict(receipt_sha256=receipt.sha256, universe_sha256=_digest(list(universe)))
        failure_path = folder / "failure.json"
        if failure_path.is_file():
            record = read_manifest(failure_path, 8 * 1024**2)[0]
            _require(record["identity"] == expected, f"La cinta fallida {folder.name} cambió")
            return folder, None, record["failure"], bounds
        if values is None:
            failure = dict(reason=window_tapes.NO_PREDICTIONS, window=window)
            atomic_json(failure_path, dict(identity=expected, failure=failure))
            return folder, None, failure, bounds
        if (folder / "manifest.json").is_file():
            tape = read_tape(folder)
        else:
            try:
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
            except NoAdmittedAssets as error:
                # Excluir todo el universo es el mismo fallo que excluir una parte de él.
                tape, report = None, dict(excluded=error.excluded)
            if tape is None or tuple(tape.assets) != universe:
                # Solo la evaluación puede excluir un activo del universo: el universo se
                # eligió entre los admitidos en ajuste y validación.
                _require(role == "evaluation", f"El universo no es admisible en {role}")
                failure = dict(reason="universe_assets_excluded", excluded=report["excluded"])
                atomic_json(failure_path, dict(identity=expected, failure=failure))
                return folder, None, failure, bounds
            # Una cinta china lleva las reglas de acciones A que exige el lector nativo.
            write_tape(tape, folder, instruments=market_rules(tape, job["market"]))
        # Ninguna cinta sintética ni de otra edición llega a un ejecutor, tampoco al reanudar
        # desde una cinta confirmada en disco.
        window_tapes.require_real_tape(tape, self.edition_id, folder.name)
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
        # El índice de mercado se evalúa en su propia cinta de un activo, del mismo tramo y
        # con el mismo recibo. No usa sus predicciones: reparte por igual entre lo valorado.
        symbol = self.policies[MARKET_INDEX][job["market"]] if job["arm"] == MARKET_INDEX else None
        key = (job["window"], symbol)
        if key not in self.evaluations:
            assets = universe if symbol is None else (f"{job['market']}/{symbol}",)
            name = None if symbol is None else f"index-{symbol}-{job['window']}"
            self.evaluations = {key: self.tape(job, "evaluation", job["window"], assets, name=name)}
        evaluation = self.evaluations[key]
        segments = [item[3] for item in (*train, validation, evaluation)]
        _require(
            all(a[1] <= b[0] for a, b in zip(segments, segments[1:], strict=False)),
            f"La política de {job['window']} usaría tramos posteriores a su evaluación",
        )
        # Sin predicciones en el ajuste o la validación no hay política que ajustar.
        unfit = next((item[2] for item in (*train, validation) if item[2] is not None), None)

        def sha(item):
            return None if item[1] is None else item[1].sha256

        return PolicyTapes(
            universe=universe,
            train=tuple(item[1] for item in train),
            validation=validation[1],
            evaluation=evaluation[1],
            failure=evaluation[2],
            unfit=unfit,
            paths=dict(
                train=[str(item[0]) for item in train],
                validation=str(validation[0]),
                evaluation=None if evaluation[1] is None else str(evaluation[0]),
            ),
            identity=dict(
                universe_sha256=_digest(list(universe)),
                train=[sha(item) for item in train],
                validation=sha(validation),
                evaluation=sha(evaluation),
                failure=evaluation[2],
                unfit=unfit,
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
            # El primer requisito de un traslado es el ajuste de su ancla. Los demás son
            # selecciones de la cadena, confirmadas en la etapa de posentrenamiento.
            anchor = job["depends"][0]
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
            # Oleadas KLPO consumidas. Los demás brazos no las tienen.
            waves=report.get("waves"),
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
                if job["kind"] != REFERENCE and tapes.unfit is not None:
                    # Sin datos de ajuste o validación no se lanza ningún ejecutor.
                    report = unfit_report(self.stage, tapes)
                else:
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
        # La sensibilidad de ventanas comparte archivos con la etapa principal, así que su
        # identidad es lo único que separa sus salidas.
        sensitivity=stage.get("sensitivity"),
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
        predictor_source=dict(
            rule=policies["predictor"]["source"],
            needs_chain_output=policies["predictor"]["source"] != BASE_SELECTED,
        ),
        data_policy=policies["data"]["policy"],
        edition_id=policies["data"]["edition_id"],
        levels=stage["levels"],
        universe_predictor=stage["universe_predictor"],
        universe=policies["universe"],
        selection=policies["selection"],
        budget=policies["budget"],
        seeds=policies["seeds"],
        contrasts=policies["contrasts"],
        counts=count_stage(stage, jobs),
        tapes_per_predictor=count_tapes(stage),
        window_sensitivity=dict(
            policies["window_sensitivity"],
            tapes_per_predictor=count_tapes(window_sensitivity(stage)),
        ),
        capabilities=available,
        missing_capabilities=missing_capabilities(jobs, EXECUTORS, available),
        scientific_training_started=False,
        final_test_opened=False,
    )


def run_stage(
    path,
    views,
    campaign_output,
    edition,
    output,
    *,
    executors=None,
    capabilities=None,
    stop=None,
    window=None,
    chain_output=None,
    sensitivity=False,
):
    """Ejecutar o reanudar la etapa sobre una campaña base confirmada.

    `executors` sustituye los ejecutores por familia y `capabilities` el estado del motor.
    `window` limita la etapa a una ventana de campaña y a la base confirmada de esa ventana.
    El bloqueo de aprendizaje se comprueba antes de todo y antes de cada trabajo pendiente.
    Las capacidades del plan se exigen antes de abrir fuentes o crear la salida. Con
    `sensitivity` se ejecuta la sensibilidad de ventanas declarada, que debe estar activada
    en la configuración y escribe en una salida con su propia identidad.
    """
    from mars_titan.training.checkpoints import StopRequest

    from .reconstructed_tape import read_edition

    require_learning_allowed("la etapa de políticas financieras de la campaña")
    stage = load_stage(path)
    if sensitivity:
        stage = window_sensitivity(stage)
        _require(
            stage["sensitivity"]["enabled"],
            f"La sensibilidad {stage['sensitivity']['id']} está declarada y desactivada. Solo "
            "se lanza si sobra presupuesto y se activa en la configuración de las políticas",
        )
    jobs = plan_stage(stage)
    count_stage(stage, jobs)
    pairs = None
    if window is not None:
        jobs, pairs = campaign_schedule.stage_window(stage["campaign"], jobs, window)
        # Cada política lee también las evaluaciones de sus ventanas de ajuste y validación.
        pairs |= {
            (job["scope"], name) for job in jobs for name in (*job["train"], job["validation"])
        }
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
    chained = () if chain_output is None else (Path(chain_output),)
    for protected in (*views.values(), campaign_output, edition, *chained, Path("dataset")):
        outside_source(protected, output)
        outside_source(output, protected)
    # Las políticas solo aprenden con la edición real declarada: su identidad se recalcula
    # desde el manifiesto y debe ser la de las políticas antes de leer ninguna cinta.
    edition_id = read_edition(edition)["edition_id"]
    _require(
        edition_id == stage["policies"]["data"]["edition_id"],
        "La edición no es la edición real declarada por las políticas de la etapa",
    )
    _, base = masked_campaign._confirmed_state(campaign["path"], views, campaign_output)
    _base_receipts(base, campaign, stage, pairs)
    # La fuente de las predicciones se resuelve antes de crear la salida.
    source = predictor_source(stage["policies"], base, campaign_output, chain_output)
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
        tapes = _Tapes(
            stage["policies"], source, edition, edition_id, output, stage["universe_predictor"]
        )
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
    execute.add_argument("--window", help="Ventana de campaña que se ejecuta")
    execute.add_argument("--chain-output", type=Path)
    execute.add_argument(
        "--sensitivity", action="store_true", help="Ejecutar la sensibilidad de ventanas activada"
    )
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
            window=args.window,
            chain_output=args.chain_output,
            sensitivity=args.sensitivity,
        )
        result.pop("jobs")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("status") in {"completed", "checked"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
