"""Entorno Gymnasium por cohortes, con recompensa diferida y sin etiquetas en la observación."""

import copy
import math
import re

import gymnasium as gym
import numpy as np

from mars_titan.evaluation.splits import PARTITIONS

from .actions import ActionGrid
from .cohorts import FINAL_TEST_START_US, VALIDATION_START_US, read_cohort, shapes_contract
from .walk_forward_receipt import WalkForwardWindow


class CausalPredictionEnv(gym.Env):
    """Recibir juntas las decisiones de un instante y publicar solo recompensas maduras."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        source,
        *,
        source_sha256,
        grid,
        shapes,
        max_assets,
        max_pending=65_536,
        max_observation_bytes=64 * 1024**2,
        partition="train",
        window=None,
    ):
        if (
            not callable(source)
            or not isinstance(grid, ActionGrid)
            or source_sha256 != grid.source_sha256
            or partition not in (PARTITIONS if window is not None else {"train", "validation"})
            or (window is not None and not isinstance(window, WalkForwardWindow))
            or type(max_pending) is not int
            or not 1 <= max_pending <= 65_536
        ):
            raise ValueError("La fuente, rejilla o presupuesto de etiquetas no es válido")
        declared = getattr(source, "partition", None)
        if declared is not None and declared != partition:
            # Los cortes fijos no distinguen una validación walk-forward anterior a 2023.
            raise ValueError("La fuente declara otra partición que la del entorno")
        if window is not None:
            # Los cortes de la ventana salen de su recibo. La fuente no puede declarar otros.
            start, end = window.segment(partition)
            bounds = getattr(source, "market_bounds", None) or {}
            if any(tuple(value) != (start, end, end) for value in bounds.values()):
                raise ValueError("La fuente declara otros cortes que la ventana walk-forward")
        self.shapes = shapes_contract(shapes, max_assets, max_observation_bytes)
        self.source, self.grid = source, grid
        self.max_assets, self.max_pending, self.max_bytes = (
            max_assets,
            max_pending,
            max_observation_bytes,
        )
        self.identity = dict(
            source_sha256=source_sha256,
            grid=grid.to_dict(),
            shapes={key: list(shape) for key, shape in self.shapes.items()},
            max_assets=max_assets,
            max_pending=max_pending,
            max_observation_bytes=max_observation_bytes,
            partition=partition,
            gymnasium=gym.__version__,
            final_test_opened=False,
        )
        self.observation_space = gym.spaces.Dict(
            {
                **{
                    key: gym.spaces.Box(-np.inf, np.inf, (max_assets, *shape), np.float32)
                    for key, shape in self.shapes.items()
                },
                "active": gym.spaces.MultiBinary(max_assets),
            }
        )
        self.action_space = gym.spaces.MultiDiscrete(np.full(max_assets, 21, dtype=np.int64))
        self.current, self.pending, self.position, self.clock, self.done = None, [], 0, 0, True
        self.started, self.origins = False, {}
        self.abandoned = dict(episodes=0, pending_credits=0)
        self.window = window
        if window is not None:
            self.identity["walk_forward"] = window.identity(partition)
            self.start, self.end = start, end
        else:
            self.start = 0 if partition == "train" else VALIDATION_START_US
            self.end = VALIDATION_START_US if partition == "train" else FINAL_TEST_START_US

    def _read(self, position):
        raw = self.source(position)
        current = (
            None if raw is None else read_cohort(raw, self.shapes, self.max_assets, self.max_bytes)
        )
        if current is not None and self.window is not None:
            # Decisión dentro del tramo y etiquetas maduras antes de su final, nunca en 2024.
            if not self.start <= current["prediction_at"] < self.end or (
                (current["target_available_at"] >= self.end).any()
            ):
                raise ValueError("La cohorte cruza el tramo de su ventana walk-forward")
        elif current is not None:
            train = current["prediction_at"] < VALIDATION_START_US
            if train != (self.identity["partition"] == "train") or (
                train and (current["target_available_at"] >= VALIDATION_START_US).any()
            ):
                raise ValueError(
                    "La cohorte cruza la partición declarada de entrenamiento o validación"
                )
        return current

    def _observation(self, current):
        result = {
            key: np.zeros((self.max_assets, *shape), dtype=np.float32)
            for key, shape in self.shapes.items()
        }
        result["active"] = np.zeros(self.max_assets, dtype=np.int8)
        if current is not None:
            n = len(current["asset_ids"])
            for key in self.shapes:
                result[key][:n] = current["inputs"][key]
            result["active"][:n] = 1
        return result

    def _info(self):
        return dict(
            prediction_at=self.clock,
            asset_ids=list(self.current["asset_ids"]) if self.current else [],
            pending=len(self.pending),
        )

    def reset(self, *, seed=None, options=None):
        if options:
            raise ValueError("No se admiten opciones de reinicio sin contrato")
        current = self._read(0)
        if current is None:
            raise ValueError("La fuente no contiene cohortes")
        observed = self._observation(current)
        if self.started and not self.done and self.position:
            # Un reinicio no emite créditos pendientes, pero queda contabilizado.
            self.abandoned = dict(
                episodes=self.abandoned["episodes"] + 1,
                pending_credits=self.abandoned["pending_credits"] + len(self.pending),
            )
        super().reset(seed=seed)
        if seed is not None:
            self.action_space.seed(seed)
        self.current, self.pending, self.position, self.clock = (
            current,
            [],
            0,
            current["prediction_at"],
        )
        self.done, self.started, self.origins = False, True, {}
        return observed, self._info()

    def step(self, action):
        if not self.started or self.done:
            raise RuntimeError("Reinicia el entorno antes de decidir o después de terminar")
        actions = np.asarray(action)
        if not self.action_space.contains(actions) or not np.issubdtype(actions.dtype, np.integer):
            raise ValueError("La acción no pertenece al espacio predictivo")
        n = len(self.current["asset_ids"])
        if len(self.pending) + n > self.max_pending:
            raise ValueError("La cola de etiquetas supera el presupuesto")
        following = self._read(self.position + 1)
        if following is not None and following["prediction_at"] <= self.clock:
            raise ValueError(
                "Las cohortes deben avanzar en UTC y agrupar cada instante una sola vez"
            )
        additions = [
            dict(
                prediction_at=self.clock,
                asset_id=asset,
                event_id=f"{self.clock:020d}/{asset}",
                action=int(actions[i]),
                target_available_at=int(self.current["target_available_at"][i]),
                target=float(self.current["target"][i]),
                position=self.position,
            )
            for i, asset in enumerate(self.current["asset_ids"])
        ]
        all_pending = self.pending + additions
        next_time = (
            following["prediction_at"]
            if following
            else max(e["target_available_at"] for e in all_pending)
        )
        matured = sorted(
            (e for e in all_pending if e["target_available_at"] <= next_time),
            key=lambda e: (e["target_available_at"], e["event_id"]),
        )
        confirmed = []
        # La cola puede contener más de un bloque, sin pedir un vector ilimitado a la rejilla.
        for start in range(0, len(matured), 4096):
            block = matured[start : start + 4096]
            rewards = self.grid.reward(
                np.array([e["action"] for e in block]), np.array([e["target"] for e in block])
            )
            confirmed.extend(
                dict(
                    {key: value for key, value in event.items() if key != "position"},
                    reward=float(reward),
                    prediction=float(self.grid.values[event["action"]]),
                )
                for event, reward in zip(block, rewards, strict=True)
            )
        reward = math.fsum(e["reward"] / len(confirmed) for e in confirmed)
        observed = self._observation(following)
        pending = [e for e in all_pending if e["target_available_at"] > next_time]
        origins = {**self.origins, self.position: self.current["sha256"]}
        used = {e["position"] for e in pending}
        self.origins = {p: digest for p, digest in origins.items() if p in used}
        self.pending = pending
        self.position, self.current, self.clock, self.done = (
            self.position + 1,
            following,
            next_time,
            following is None,
        )
        return observed, reward, self.done, False, dict(self._info(), matured=confirmed)

    def snapshot(self):
        """Exportar tipos JSON simples, sin etiquetas inmaduras, pesos ni copias del corpus."""
        if not self.started:
            raise RuntimeError("El entorno todavía no se ha iniciado")
        return copy.deepcopy(
            dict(
                schema_version=2,
                identity=self.identity,
                position=self.position,
                clock=self.clock,
                done=self.done,
                # Las etiquetas pendientes se releen de la fuente al restaurar.
                pending=[{k: v for k, v in e.items() if k != "target"} for e in self.pending],
                origins=[[position, digest] for position, digest in sorted(self.origins.items())],
                abandoned=self.abandoned,
                current_sha256=self.current["sha256"] if self.current else None,
                rng=self.np_random.bit_generator.state,
                action_rng=self.action_space.np_random.bit_generator.state,
                rng_seed=self.np_random_seed,
            )
        )

    def _sealed_targets(self, state):
        """Releer cada etiqueta pendiente en su cohorte de origen, verificada por su huella."""
        origins = state["origins"]
        if (
            not isinstance(origins, list)
            or len(origins) > self.max_pending
            or any(
                not isinstance(row, list)
                or len(row) != 2
                or type(row[0]) is not int
                or not 0 <= row[0] < state["position"]
                or not isinstance(row[1], str)
                or not re.fullmatch(r"[a-f0-9]{64}", row[1])
                for row in origins
            )
            or any(a[0] >= b[0] for a, b in zip(origins, origins[1:], strict=False))
            or {row[0] for row in origins} != {e["position"] for e in state["pending"]}
        ):
            raise ValueError("Los orígenes de las etiquetas pendientes no son coherentes")
        groups, targets = {}, {}
        for event in state["pending"]:
            groups.setdefault(event["position"], []).append(event)
        for position, digest in origins:
            cohort = self._read(position)
            if cohort is None or cohort["sha256"] != digest:
                raise ValueError("La cohorte de origen de una etiqueta pendiente ha cambiado")
            rows = {asset: i for i, asset in enumerate(cohort["asset_ids"])}
            for event in groups[position]:
                row = rows.get(event["asset_id"])
                if (
                    row is None
                    or cohort["prediction_at"] != event["prediction_at"]
                    or int(cohort["target_available_at"][row]) != event["target_available_at"]
                ):
                    raise ValueError("Una etiqueta pendiente no pertenece a su cohorte de origen")
                targets[event["event_id"]] = float(cohort["target"][row])
        return targets

    def restore(self, state):
        """Validar el estado completo antes de cambiar el cursor o las decisiones pendientes."""
        keys = {
            "schema_version",
            "identity",
            "position",
            "clock",
            "done",
            "pending",
            "origins",
            "abandoned",
            "current_sha256",
            "rng",
            "action_rng",
            "rng_seed",
        }
        abandoned = state.get("abandoned") if isinstance(state, dict) else None
        if (
            not isinstance(state, dict)
            or set(state) != keys
            or state["schema_version"] != 2
            or state["identity"] != self.identity
            or type(state["position"]) is not int
            or not 0 <= state["position"] <= 10_000_000
            or type(state["clock"]) is not int
            or not self.start <= state["clock"] < self.end
            or type(state["done"]) is not bool
            or not isinstance(state["pending"], list)
            or len(state["pending"]) > self.max_pending
            or type(state["rng_seed"]) is not int
            or not isinstance(abandoned, dict)
            or set(abandoned) != {"episodes", "pending_credits"}
            or any(type(v) is not int or not 0 <= v < 2**63 for v in abandoned.values())
        ):
            raise ValueError("El estado no concuerda con la identidad o los límites del entorno")
        current = self._read(state["position"])
        if (
            state["done"] != (current is None)
            or state["current_sha256"] != (current["sha256"] if current else None)
            or (current and current["prediction_at"] != state["clock"])
            or (state["done"] and state["pending"])
        ):
            raise ValueError("La fuente o la huella de la cohorte actual han cambiado")
        ids = set()
        fields = {
            "prediction_at",
            "asset_id",
            "event_id",
            "action",
            "target_available_at",
            "position",
        }
        for event in state["pending"]:
            if (
                not isinstance(event, dict)
                or set(event) != fields
                or type(event["prediction_at"]) is not int
                or not self.start <= event["prediction_at"] < state["clock"]
                or type(event["target_available_at"]) is not int
                or not state["clock"] < event["target_available_at"] < self.end
                or type(event["action"]) is not int
                or not 0 <= event["action"] < 21
                or type(event["position"]) is not int
                or not 0 <= event["position"] < state["position"]
                or not isinstance(event["asset_id"], str)
                or len(event["asset_id"]) > 67
                or event["event_id"] != f"{event['prediction_at']:020d}/{event['asset_id']}"
                or event["event_id"] in ids
            ):
                raise ValueError("Una etiqueta pendiente no conserva su identidad y maduración")
            ids.add(event["event_id"])
        targets = self._sealed_targets(state)
        rng, actions = np.random.default_rng(), np.random.default_rng()
        rng.bit_generator.state = copy.deepcopy(state["rng"])
        actions.bit_generator.state = copy.deepcopy(state["action_rng"])
        self.current, self.position, self.clock, self.done = (
            current,
            state["position"],
            state["clock"],
            state["done"],
        )
        self.pending = [
            dict(
                prediction_at=event["prediction_at"],
                asset_id=event["asset_id"],
                event_id=event["event_id"],
                action=event["action"],
                target_available_at=event["target_available_at"],
                target=targets[event["event_id"]],
                position=event["position"],
            )
            for event in state["pending"]
        ]
        self.origins = {position: digest for position, digest in state["origins"]}
        self.abandoned = dict(abandoned)
        self._np_random, self._np_random_seed = rng, state["rng_seed"]
        self.action_space.np_random.bit_generator.state = actions.bit_generator.state
        self.started = True
