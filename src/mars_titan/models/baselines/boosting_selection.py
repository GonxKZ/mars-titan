"""Selección temporal de boosting tras evaluaciones completas por sesión."""

import math
from pathlib import Path

import numpy as np

from mars_titan.evaluation.session_metrics import SessionErrors


class ValidationCache:
    """Materializar una vez las modalidades, con límites de bloques, sesiones y disco."""

    def __init__(self, factory, directory, *, expected_rows, max_bytes):
        self.directory, self.blocks, self.bytes = Path(directory), 0, 0
        self.directory.mkdir(parents=True, exist_ok=False)
        sessions = SessionErrors()
        for values, target, markets, moments in factory():
            values, target = np.asarray(values), np.asarray(target)
            markets, moments = np.asarray(markets), np.asarray(moments)
            if (
                values.ndim != 2
                or len(values) != len(target)
                or values.dtype.kind not in "fiu"
                or target.dtype.kind not in "fiu"
                or not np.isfinite(values).all()
                or not np.isfinite(target).all()
                or values.nbytes + target.nbytes > 512 * 1024**2
                or self.blocks >= 1_000_000
            ):
                raise ValueError(
                    "El bloque de validación no cumple el presupuesto o las dimensiones"
                )
            sessions.update(markets, moments, np.zeros_like(target))
            # Las cuatro cabeceras NPY y el contenedor ZIP caben en este margen.
            needed = sum(array.nbytes for array in (values, target, markets, moments)) + 4096
            if self.bytes + needed > max_bytes:
                raise ValueError("La caché de validación supera el presupuesto de disco")
            path = self.directory / f"{self.blocks:06}.npz"
            # np.savez evita compresión repetida y conserva las etiquetas sin redondearlas.
            np.savez(path, values=values, target=target, markets=markets, moments=moments)
            self.bytes += path.stat().st_size
            if self.bytes > max_bytes:
                raise ValueError("La caché de validación supera el presupuesto de disco")
            self.blocks += 1
        if sessions.summary()["samples"] != expected_rows or not expected_rows:
            raise ValueError("La caché no conserva la población de validación")

    def __call__(self):
        for index in range(self.blocks):
            with np.load(self.directory / f"{index:06}.npz", allow_pickle=False) as block:
                yield tuple(block[key] for key in ("values", "target", "markets", "moments"))


class BoostingSelection:
    """El mínimo retrasa la parada, pero todos los modelos pueden ser seleccionados."""

    def __init__(self, policy, rounds, *, state=None):
        if (
            not isinstance(policy, dict)
            or set(policy) != {"schema_version", "minimum_rounds", "patience_rounds", "min_delta"}
            or type(policy["schema_version"]) is not int
            or policy["schema_version"] != 1
            or type(rounds) is not int
            or not 1 <= rounds <= 2000
            or type(policy["minimum_rounds"]) is not int
            or not 1 <= policy["minimum_rounds"] <= rounds
            or type(policy["patience_rounds"]) is not int
            or not 1 <= policy["patience_rounds"] <= rounds
            or type(policy["min_delta"]) not in (int, float)
            or not math.isfinite(policy["min_delta"])
            or policy["min_delta"] < 0
        ):
            raise ValueError("La selección de boosting necesita límites explícitos y válidos")
        self.policy, self.rounds = dict(policy), rounds
        self.state = dict(
            completed_rounds=0,
            selected_round=0,
            best_session_mae=None,
            last_session_mae=None,
            patience_session_mae=None,
            rounds_without_improvement=0,
            stop_reason=None,
        )
        if state is not None:
            self._restore(state)

    def _restore(self, state):
        if not isinstance(state, dict) or set(state) != set(self.state):
            raise ValueError("El estado de selección está incompleto")
        count, best, stale = (
            state["completed_rounds"],
            state["selected_round"],
            state["rounds_without_improvement"],
        )
        if (
            any(type(value) is not int for value in (count, best, stale))
            or not 1 <= best <= count <= self.rounds
            or not 0
            <= stale
            <= min(self.policy["patience_rounds"], max(0, count - self.policy["minimum_rounds"]))
            or any(
                type(state[key]) not in (int, float)
                or not math.isfinite(state[key])
                or state[key] < 0
                for key in ("best_session_mae", "last_session_mae", "patience_session_mae")
            )
            or state["best_session_mae"] > state["last_session_mae"]
            or state["best_session_mae"] > state["patience_session_mae"]
            or state["stop_reason"] != self._reason(count, stale)
        ):
            raise ValueError("El estado no conserva rondas, paciencia y métricas coherentes")
        self.state = dict(state)

    def _reason(self, count, stale):
        if stale >= self.policy["patience_rounds"]:
            return "validation_plateau"
        return "budget_exhausted" if count == self.rounds else None

    def observe(self, count, score):
        previous = self.state
        if (
            type(count) is not int
            or count != previous["completed_rounds"] + 1
            or previous["stop_reason"] is not None
            or type(score) not in (int, float)
            or not math.isfinite(score)
            or score < 0
        ):
            raise ValueError("Solo se confirman rondas consecutivas con una evaluación finita")
        state = dict(previous, completed_rounds=count, last_session_mae=score)
        if previous["best_session_mae"] is None or score < previous["best_session_mae"]:
            state.update(selected_round=count, best_session_mae=score)
        significant = previous["patience_session_mae"]
        if significant is None or significant - score > self.policy["min_delta"]:
            state.update(patience_session_mae=score, rounds_without_improvement=0)
        elif count > self.policy["minimum_rounds"]:
            state["rounds_without_improvement"] += 1
        state["stop_reason"] = self._reason(count, state["rounds_without_improvement"])
        self.state = state


def session_validation(model, factory, *, expected_rows):
    """Evaluar bloques sin ajustar transformaciones ni conservar predicciones completas."""
    sessions = SessionErrors()
    for values, target, markets, moments in factory():
        sessions.update(markets, moments, model.predict(values) - target)
    result = sessions.summary()
    if result["samples"] != expected_rows or not expected_rows:
        raise ValueError("La validación no recorre exactamente la población declarada")
    return result["session_mae"]


def selection_callback(xgb, selection, evaluate, confirm):
    """Usar la barrera de ronda de XGBoost sin consumir evaluaciones parciales."""

    class EvaluateRound(xgb.callback.TrainingCallback):
        def after_iteration(self, model, epoch, evals_log):
            score = evaluate(model)
            selection.observe(model.num_boosted_rounds(), score)
            confirm(model, dict(selection.state))
            return selection.state["stop_reason"] is not None

    return EvaluateRound()
