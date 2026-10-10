"""Selección temporal de boosting tras evaluaciones completas por sesión."""

import hashlib
import json
import math
import time

import numpy as np

from mars_titan.evaluation.session_metrics import SessionErrors


def _float32(values):
    """Bloque float32 contiguo, sin desbordar valores finitos al convertirlo."""
    try:
        with np.errstate(over="raise", invalid="raise"):
            return np.ascontiguousarray(values, dtype=np.float32)
    except FloatingPointError as error:
        raise ValueError("La validación no conserva valores finitos en float32") from error


def _base_score(booster):
    """Base que el predictor añade a la suma de hojas, leída de la configuración."""
    text = json.loads(booster.save_config())["learner"]["learner_model_param"]["base_score"]
    values = text.strip("[]").split(",")
    if len(values) != 1:
        raise ValueError("La validación incremental solo admite una salida")
    return np.float32(float(values[0]))


class ResidentValidation:
    """Validación leída una vez y conservada en RAM y, hasta un presupuesto, en la GPU.

    Conserva los bloques del lector, su orden y sus tipos, igual que la antigua caché
    en disco, así que `session_validation` agrupa los mismos errores en el mismo orden.
    Los valores se comprueban al cargarse y cada ronda solo predice. Varias
    configuraciones de una misma ventana pueden compartirla porque no depende de los
    árboles.

    Entre rondas consecutivas del mismo booster solo se evalúa el árbol nuevo. El
    predictor CUDA de XGBoost suma las hojas de cada fila en orden sobre un acumulador
    float32 y después añade la base, así que conservar esa suma y añadirle la hoja nueva
    da los mismos bits que predecir con todos los árboles. Cada reinicio compara toda la
    validación con la predicción completa y, si difiere, se usa siempre la completa.
    Cada ronda incremental vuelve a comparar el primer bloque y falla si difiere.
    """

    def __init__(self, factory, *, expected_rows, max_bytes):
        self.blocks, self.bytes, self.expected_rows = [], 0, expected_rows
        self.device, self.placed, self.device_bytes = None, 0, 0
        self.sums, self.incremental = None, None
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
                or len(self.blocks) >= 1_000_000
            ):
                raise ValueError(
                    "El bloque de validación no cumple el presupuesto o las dimensiones"
                )
            sessions.update(markets, moments, np.zeros_like(target))
            self.bytes += sum(array.nbytes for array in (values, target, markets, moments))
            if self.bytes > max_bytes:
                raise ValueError("La validación residente supera su presupuesto de memoria")
            self.blocks.append((values, target, markets, moments))
        if sessions.summary()["samples"] != expected_rows or not expected_rows:
            raise ValueError("La validación no conserva la población declarada")
        self.offsets = np.cumsum([0] + [len(block[1]) for block in self.blocks])

    def place(self, device_bytes):
        """Copiar a cuda:0, contiguos, los primeros bloques que quepan.

        Cada fila se predice con independencia del resto, así que la ubicación y el
        reparto en tramos no cambian ningún valor.
        """
        if type(device_bytes) is not int or device_bytes < 0:
            raise ValueError("El presupuesto de validación en la GPU no es válido")
        self.release_device()
        placed, size = 0, 0
        for values, *_ in self.blocks:
            if size + values.size * 4 > device_bytes:
                break
            placed, size = placed + 1, size + values.size * 4
        if not placed:
            return 0
        import cupy as cp

        with cp.cuda.Device(0):
            device = cp.empty((int(self.offsets[placed]), self.blocks[0][0].shape[1]), cp.float32)
            for index in range(placed):
                device[self.offsets[index] : self.offsets[index + 1]].set(
                    _float32(self.blocks[index][0])
                )
        self.device, self.placed, self.device_bytes = device, placed, size
        return size

    def release_device(self):
        self.device, self.placed, self.device_bytes, self.sums = None, 0, 0, None

    def __call__(self):
        yield from self.blocks

    def _parts(self):
        """Tramos predichos de una vez: los bloques residentes juntos y cada bloque en RAM."""
        first = [(0, self.placed)] if self.placed else []
        return first + [(index, index + 1) for index in range(self.placed, len(self.blocks))]

    def _values(self, part):
        import cupy as cp

        start, stop = part
        if stop <= self.placed:
            return self.device[self.offsets[start] : self.offsets[stop]]
        return cp.asarray(_float32(self.blocks[start][0]))

    def _full(self, model, part):
        """Predicción con todos los árboles, igual que `session_validation`."""
        start, stop = part
        if stop <= self.placed:
            return model.predict_device(self.device[self.offsets[start] : self.offsets[stop]])
        return model.predict(self.blocks[start][0])

    def _margins(self, booster, start, stop):
        """Suma float32 de las hojas de los árboles [start, stop) para cada tramo."""
        import cupy as cp

        sums = []
        for part in self._parts():
            values = self._values(part)
            zeros = cp.zeros(len(values), dtype=cp.float32)
            sums.append(
                booster.inplace_predict(
                    values, iteration_range=(start, stop), predict_type="margin", base_margin=zeros
                )
            )
        return sums

    def _restart(self, model, count):
        """Predicción completa y, si la suma de hojas la reproduce, nuevo estado incremental."""
        import cupy as cp

        full = [self._full(model, part) for part in self._parts()]
        self.sums = None
        if self.incremental is False or count < 1:
            return full
        with cp.cuda.Device(0):
            base = _base_score(model.booster)
            sums = self._margins(model.booster, 0, count)
            if all(
                np.array_equal((base + value).get(), expected)
                for value, expected in zip(sums, full, strict=True)
            ):
                self.sums = dict(booster=model.booster, count=count, base=base, values=sums)
                self.incremental = True
            else:
                self.incremental = False
        return full

    def _predictions(self, model):
        import cupy as cp

        booster, count = model.booster, model.booster.num_boosted_rounds()
        state = self.sums
        if state is None or state["booster"] is not booster or state["count"] != count - 1:
            return self._restart(model, count)
        with cp.cuda.Device(0):
            for value, leaf in zip(
                state["values"], self._margins(booster, count - 1, count), strict=True
            ):
                value += leaf
            state["count"] = count
            predictions = [(state["base"] + value).get() for value in state["values"]]
        if not all(np.isfinite(prediction).all() for prediction in predictions):
            raise ValueError("La predicción de boosting no es finita o tiene otra forma")
        first = (0, 1)
        if not np.array_equal(predictions[0][: self.offsets[1]], self._full(model, first)):
            self.sums = None
            raise ValueError("La suma incremental de hojas no reproduce la predicción completa")
        return predictions

    def evaluate(self, model):
        """MAE por sesión con las mismas predicciones que `session_validation`.

        Los bloques residentes en la GPU ya son float32 contiguos y finitos, así que
        se omiten la copia y la comprobación repetidas de cada ronda. Los errores se
        acumulan bloque a bloque en el orden del lector.
        """
        sessions = SessionErrors()
        for (start, stop), prediction in zip(self._parts(), self._predictions(model), strict=True):
            for index in range(start, stop):
                _, target, markets, moments = self.blocks[index]
                begin, end = (self.offsets[i] - self.offsets[start] for i in (index, index + 1))
                sessions.update(markets, moments, prediction[begin:end] - target)
        result = sessions.summary()
        if result["samples"] != self.expected_rows:
            raise ValueError("La validación no recorre exactamente la población declarada")
        return result["session_mae"]

    def release(self):
        self.release_device()
        self.blocks = []


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
    if isinstance(factory, ResidentValidation):
        if factory.expected_rows != expected_rows:
            raise ValueError("La validación no recorre exactamente la población declarada")
        return factory.evaluate(model)
    sessions = SessionErrors()
    for values, target, markets, moments in factory():
        sessions.update(markets, moments, model.predict(values) - target)
    result = sessions.summary()
    if result["samples"] != expected_rows or not expected_rows:
        raise ValueError("La validación no recorre exactamente la población declarada")
    return result["session_mae"]


def selection_callback(
    xgb, selection, evaluate, confirm, *, replay_model=None, stop_requested=None
):
    """Usar la barrera de ronda de XGBoost sin consumir evaluaciones parciales."""

    class EvaluateRound(xgb.callback.TrainingCallback):
        replayed_rounds = 0
        replay_seconds = 0.0

        def before_training(self, model):
            self.started = time.perf_counter()
            return model

        def after_iteration(self, model, epoch, evals_log):
            count = model.num_boosted_rounds()
            if replay_model is not None and count <= replay_model.num_boosted_rounds():
                self.replayed_rounds = count
                if count == replay_model.num_boosted_rounds():
                    # Solo normalizar los dos atributos de auditoría escritos por save().
                    model.set_attr(
                        **{
                            key: replay_model.attr(key)
                            for key in ("mars_external_schema", "mars_external_contract")
                        }
                    )
                    expected = hashlib.sha256(replay_model.save_raw(raw_format="ubj")).digest()
                    actual = hashlib.sha256(model.save_raw(raw_format="ubj")).digest()
                    if actual != expected:
                        raise ValueError("El prefijo reconstruido difiere del booster confirmado")
                self.replay_seconds = time.perf_counter() - self.started
                return bool(stop_requested and stop_requested())
            score = evaluate(model)
            selection.observe(count, score)
            paused = confirm(model, dict(selection.state))
            return bool(paused) or selection.state["stop_reason"] is not None

    return EvaluateRound()
