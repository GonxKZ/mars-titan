"""Paridad entre dispositivos de la evaluación congelada de una política, sin aprender.

Compara dos evaluaciones con `--decisions` de la misma política, con los mismos pesos, las
mismas cintas y los mismos costes, hechas en dispositivos distintos. CPU y cuBLAS en FP32
no producen logits idénticos bit a bit porque suman en otro orden, así que se declara una
tolerancia y se mide si alguna diferencia cambia una acción.

Cada episodio se recorre en orden. Mientras las acciones coinciden, las observaciones son
las mismas y los logits deben quedar dentro de la tolerancia. Con esa condición, una acción
distinta solo puede aparecer si el margen entre las dos mejores salidas de la referencia no
supera el doble de la tolerancia, porque cada una de esas salidas se mueve como mucho una
tolerancia. Es un empate casi exacto que el redondeo puede invertir y se publica como
divergencia. Tras ella las trayectorias ya no son comparables, así que solo se mide su
efecto en el patrimonio. Sin divergencias, el patrimonio por sesión debe ser idéntico,
porque la contabilidad es la misma en los dos casos y solo cambia la inferencia.
"""

import hashlib
import math
from pathlib import Path

from mars_titan.data.cohort_files import read_manifest

# Capacidad de los binarios que aceptan --decisions en la evaluación congelada.
DECISION_LOGITS = "native_policy_decision_logits"
DECISIONS_KIND = "native_policy_decisions"
EVALUATION_KIND = "native_policy_evaluation"
# Tolerancia declarada antes de medir y relativa a la escala de los logits de cada decisión.
# El error de redondeo de una suma FP32 crece con el número de términos y cuBLAS no suma en
# el orden de la CPU, así que se esperan diferencias de unos pocos épsilon relativos. El
# término absoluto cubre los logits cercanos a cero. La medida publica la mayor diferencia
# observada para que se vea su distancia a esta cota.
TOLERANCE = dict(atol=1e-5, rtol=1e-4)
MARGIN_LIMITS = (1e-6, 1e-5, 1e-4, 1e-3)
_MIB = 1024**2
_HEAD = b'{"payload":'


def _sealed(path, kind):
    """Contenido de un registro del binario con su huella comprobada.

    El binario escribe el sobre compacto y calcula la huella sobre el texto exacto del
    contenido. Python no escribe los reales con el mismo formato que nlohmann, así que la
    huella se comprueba sobre los bytes del archivo y no sobre una nueva serialización.
    """
    envelope, _ = read_manifest(path, 256 * _MIB)
    if (
        not isinstance(envelope, dict)
        or set(envelope) != {"payload", "sha256"}
        or not isinstance(envelope["sha256"], str)
    ):
        raise ValueError(f"{path} no conserva su sello")
    # Si el sobre no tiene exactamente esa forma, no se recorta nada y la huella no coincide.
    tail = b',"sha256":"' + envelope["sha256"].encode() + b'"}'
    body = path.read_bytes().removeprefix(_HEAD).removesuffix(tail)
    if hashlib.sha256(body).hexdigest() != envelope["sha256"]:
        raise ValueError(f"{path} no conserva su sello")
    payload = envelope["payload"]
    if payload.get("kind") != kind or payload.get("schema_version") != 1:
        raise ValueError(f"{path} no es un registro {kind} de la versión 1")
    return payload


def read_run(folder):
    """Evaluación completa y registro de decisiones de una salida con `--decisions`."""
    folder = Path(folder)
    evaluation = _sealed(folder / "evaluation.json", EVALUATION_KIND)
    decisions = _sealed(folder / "decisions.json", DECISIONS_KIND)
    if (
        evaluation["status"] != "completed"
        or evaluation["identity"].get("decisions") is not True
        or decisions["identity_sha256"] != evaluation["identity_sha256"]
        or decisions["device"] != evaluation["identity"]["device"]
    ):
        raise ValueError(f"{folder} no es una evaluación completa con su registro de decisiones")
    return evaluation, decisions


def bound(logits, tolerance):
    """Diferencia admitida entre dispositivos para las salidas de una decisión."""
    return tolerance["atol"] + tolerance["rtol"] * max(abs(value) for value in logits)


def margin(logits):
    """Distancia entre la mejor salida y la siguiente."""
    first, second = sorted(logits, reverse=True)[:2]
    return first - second


def first_argmax(logits):
    """Acción del binario: la primera salida máxima, como `std::max_element`."""
    return max(range(len(logits)), key=lambda index: (logits[index], -index))


def _episodes(evaluation, decisions):
    rows = {(row["cost_bps"], row["manifest_sha256"]): row for row in evaluation["metrics"]}
    for cost in decisions["costs"]:
        for tape in cost["tapes"]:
            key = (cost["cost_bps"], tape["manifest_sha256"])
            if key not in rows:
                raise ValueError("El registro de decisiones tiene episodios sin evaluación")
            yield key, tape["decisions"], rows.pop(key)
    if rows:
        raise ValueError("La evaluación tiene episodios sin registro de decisiones")


def _relative(a, b):
    """Diferencia relativa de dos patrimonios. Sin valorar frente a valorado es infinita."""
    if a is None or b is None:
        return 0.0 if a is b else math.inf
    return abs(a - b) / max(abs(a), abs(b), 1e-300)


def compare_runs(reference, other, *, tolerance=TOLERANCE):
    """Comparar la evaluación `other` con la de `reference`, ambas leídas con `read_run`.

    La referencia es la evaluación que se publica y fija el margen de cada empate. Devuelve
    los recuentos de decisiones, empates casi exactos y divergencias, las mayores
    diferencias de logits y el efecto de cada divergencia en el patrimonio. Lanza ValueError
    si las evaluaciones no son comparables o si alguna diferencia supera lo que el redondeo
    puede explicar.
    """
    (left, left_decisions), (right, right_decisions) = reference, other
    # La huella del archivo de la política incluye el estado del generador de cada
    # dispositivo, así que los pesos se comparan por la huella de sus parámetros.
    keys = ("kind", "tapes", "cost_bps", "optimizer_steps", "seed")
    if (
        left_decisions["parameter_fingerprint"] != right_decisions["parameter_fingerprint"]
        or left_decisions["outputs"] != right_decisions["outputs"]
        or any(left["identity"][key] != right["identity"][key] for key in keys)
        or left_decisions["device"] == right_decisions["device"]
    ):
        raise ValueError(
            "Las evaluaciones no comparan los mismos pesos y cintas en dos dispositivos"
        )
    left_episodes = list(_episodes(left, left_decisions))
    right_episodes = {
        key: (rows, metrics) for key, rows, metrics in _episodes(right, right_decisions)
    }
    if {key for key, _, _ in left_episodes} != set(right_episodes):
        raise ValueError("Las evaluaciones no recorren los mismos episodios")
    summary = dict(
        tolerance=dict(tolerance),
        outputs=left_decisions["outputs"],
        reference_device=left_decisions["device"],
        other_device=right_decisions["device"],
        parameter_fingerprint=left_decisions["parameter_fingerprint"],
        episodes=len(left_episodes),
        decisions=0,
        compared_decisions=0,
        exact_logits=0,
        max_abs_logit_difference=0.0,
        # Relativa a la mayor salida en valor absoluto de cada decisión, como la tolerancia.
        max_relative_logit_difference=0.0,
        near_ties=0,
        minimum_reference_margin=math.inf,
        # Decisiones comparadas con el margen de la referencia por debajo de cada umbral.
        margins_below={str(limit): 0 for limit in MARGIN_LIMITS},
        divergences=[],
        identical_equity_episodes=0,
    )
    for key, rows, metrics in left_episodes:
        other_rows, other_metrics = right_episodes[key]
        summary["decisions"] += len(rows)
        diverged = None
        for index, (a, b) in enumerate(zip(rows, other_rows, strict=False)):
            if a["cursor"] != b["cursor"] or a["cursor"] != index:
                raise ValueError("Las decisiones no siguen las sesiones del episodio")
            # El calentamiento de la memoria fija la acción sin mirar los logits, así que
            # esta comparación no lo cubre y se detiene en lugar de contarlo como empate.
            if a["mode"] != "greedy" or b["mode"] != "greedy":
                raise ValueError("La comparación solo cubre decisiones argmax")
            for record in (a, b):
                if record["action"] != first_argmax(record["logits"]):
                    raise ValueError("Una acción no es la primera salida máxima de sus logits")
            summary["compared_decisions"] += 1
            allowed = bound(a["logits"], tolerance)
            scale = max(abs(value) for value in a["logits"])
            largest = max(abs(x - y) for x, y in zip(a["logits"], b["logits"], strict=True))
            summary["exact_logits"] += int(largest == 0)
            summary["max_abs_logit_difference"] = max(summary["max_abs_logit_difference"], largest)
            summary["max_relative_logit_difference"] = max(
                summary["max_relative_logit_difference"],
                largest / scale if scale > 0 else (math.inf if largest > 0 else 0.0),
            )
            if largest > allowed:
                raise ValueError(
                    f"Logits fuera de la tolerancia en {key} y la sesión {a['cursor']}: "
                    f"{largest:.3e} > {allowed:.3e}"
                )
            gap = margin(a["logits"])
            summary["minimum_reference_margin"] = min(summary["minimum_reference_margin"], gap)
            for limit in MARGIN_LIMITS:
                summary["margins_below"][str(limit)] += int(gap < limit)
            summary["near_ties"] += int(gap <= 2 * allowed)
            if a["action"] != b["action"]:
                # Con los logits dentro de la tolerancia, este margen no supera 2 * allowed.
                diverged = dict(
                    cost_bps=key[0],
                    manifest_sha256=key[1],
                    cursor=a["cursor"],
                    reference_margin=gap,
                    other_margin=margin(b["logits"]),
                    allowed=allowed,
                    actions=[a["action"], b["action"]],
                )
                break
        nav, other_nav = metrics["equity"]["nav"], other_metrics["equity"]["nav"]
        if diverged is None:
            if (
                len(rows) != len(other_rows)
                or nav != other_nav
                or metrics["status"] != other_metrics["status"]
            ):
                raise ValueError(f"Con las mismas acciones el patrimonio de {key} cambia")
            summary["identical_equity_episodes"] += 1
            continue
        # Efecto en el capital de una divergencia por empate: el patrimonio final y la mayor
        # diferencia relativa por sesión desde la primera acción distinta.
        sessions = range(diverged["cursor"] + 1, min(len(nav), len(other_nav)))
        diverged.update(
            statuses=[metrics["status"], other_metrics["status"]],
            final_nav=[nav[-1], other_nav[-1]],
            final_relative_difference=_relative(nav[-1], other_nav[-1]),
            max_session_relative_difference=max(
                (_relative(nav[s], other_nav[s]) for s in sessions), default=0.0
            ),
        )
        summary["divergences"].append(diverged)
    if summary["compared_decisions"] == 0:
        summary["minimum_reference_margin"] = None
    return summary
