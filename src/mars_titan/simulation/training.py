"""PPO y Double DQN recuperables sobre observaciones financieras comunes."""

import copy
import fcntl
import math
import os
import resource
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from mars_titan.data.storage import atomic_json, sha256
from mars_titan.training.checkpoints import (
    capture_rng,
    load_training_state,
    restore_rng,
    save_training_state,
)
from mars_titan.training.run_receipts import initialize_receipt

from .algorithms import FinancialNetwork, double_targets, generalized_advantage, ppo_objective
from .replay import Replay


@dataclass(frozen=True)
class TrainConfig:
    total_steps: int = 8192
    learning_rate: float = 0.0003
    gamma: float = 0.99
    batch_size: int = 64
    rollout_steps: int = 128
    ppo_epochs: int = 4
    gae_lambda: float = 0.95
    clip: float = 0.2
    entropy: float = 0.01
    value_weight: float = 0.5
    replay_capacity: int = 4096
    warmup_steps: int = 256
    target_interval: int = 256
    epsilon_start: float = 1
    epsilon_end: float = 0.05
    checkpoint_steps: int = 128
    gradient_norm: float = 1

    def validate(self):
        for key in (
            "total_steps",
            "batch_size",
            "rollout_steps",
            "ppo_epochs",
            "replay_capacity",
            "warmup_steps",
            "target_interval",
            "checkpoint_steps",
        ):
            value = getattr(self, key)
            if type(value) is not int or not 1 <= value <= 1_000_000:
                raise ValueError("El presupuesto de entrenamiento debe ser entero y acotado")
        if (
            self.rollout_steps > 4096
            or self.ppo_epochs > 32
            or self.batch_size > self.replay_capacity
        ):
            raise ValueError("El lote o recorrido supera el presupuesto")
        for key in (
            "learning_rate",
            "gamma",
            "gae_lambda",
            "clip",
            "entropy",
            "value_weight",
            "epsilon_start",
            "epsilon_end",
            "gradient_norm",
        ):
            value = getattr(self, key)
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Los hiperparámetros deben ser finitos y estar acotados")
        if (
            self.learning_rate == 0
            or self.gradient_norm == 0
            or self.epsilon_end > self.epsilon_start
        ):
            raise ValueError("La tasa, norma o exploración no son válidas")


def _cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu(item) for item in value)
    return copy.deepcopy(value)


class FinancialTrainer:
    def __init__(
        self, env, algorithm, config, *, seed, device="cuda:0", diagnostic=False, lease=None
    ):
        config.validate()
        if device == "cuda:0" and os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {
            ":4096:8",
            ":16:8",
        }:
            raise ValueError("Configura CUBLAS_WORKSPACE_CONFIG antes de iniciar PyTorch")
        if (
            algorithm not in {"ppo", "double_dqn"}
            or env.tape.partition != "train"
            or type(seed) is not int
            or not 0 <= seed < 2**32
            or device not in {"cpu", "cuda:0"}
            or (device == "cpu" and (not diagnostic or config.total_steps > 32))
            or (
                device == "cuda:0"
                and (lease is None or lease.record is None or lease.handle is None)
            )
        ):
            raise ValueError(
                "El entrenamiento necesita una fuente de train y admisión CUDA explícita"
            )
        self.env, self.algorithm, self.config, self.seed, self.device = (
            env,
            algorithm,
            config,
            seed,
            device,
        )
        self.lease = lease
        if device == "cpu":
            torch.set_num_threads(1)
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)
        torch.set_float32_matmul_precision("highest")
        self.network = FinancialNetwork(
            env.observation_space.shape[0], value_head=algorithm == "ppo"
        ).to(device)
        self.target = (
            copy.deepcopy(self.network).eval().requires_grad_(False)
            if algorithm == "double_dqn"
            else None
        )
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=config.learning_rate)
        self.rng = np.random.default_rng(seed)
        self.sampling = torch.Generator(device=device).manual_seed(seed)
        self.replay = (
            Replay(config.replay_capacity, env.observation_space.shape[0])
            if algorithm == "double_dqn"
            else None
        )
        self.rollout = []
        self.observation, _ = env.reset(seed=seed)
        self.step_count, self.episodes, self.updates = 0, 0, 0
        self.episode_return, self.returns = 0.0, []
        self.last_loss = None
        self.identity = dict(
            algorithm=algorithm,
            config=asdict(config),
            seed=seed,
            device=device,
            diagnostic=diagnostic,
            environment=env.identity,
            torch=str(torch.__version__),
            numerics=dict(
                parameter_dtype="float32",
                float32_matmul_precision=torch.get_float32_matmul_precision(),
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                cublas_workspace=os.environ.get("CUBLAS_WORKSPACE_CONFIG")
                if device == "cuda:0"
                else None,
            ),
            code={
                name: sha256(Path(__file__).with_name(name))
                for name in (
                    "training.py",
                    "algorithms.py",
                    "environment.py",
                    "portfolio.py",
                    "market.py",
                    "replay.py",
                )
            },
        )

    def _tensor(self, value):
        return torch.as_tensor(value, dtype=torch.float32, device=self.device)

    def _action(self):
        with torch.no_grad():
            result = self.network(self._tensor(self.observation)[None])
            if self.algorithm == "ppo":
                logits, value = result
                logp = torch.log_softmax(logits, dim=-1)
                action = torch.multinomial(logp.exp(), 1, generator=self.sampling)
                return int(action.item()), float(logp.gather(1, action).item()), float(value.item())
            fraction = min(1, self.step_count / max(1, self.config.total_steps // 2))
            epsilon = self.config.epsilon_start + fraction * (
                self.config.epsilon_end - self.config.epsilon_start
            )
            action = (
                int(self.rng.integers(6))
                if self.rng.random() < epsilon
                else int(result.argmax(1).item())
            )
            return action, 0.0, 0.0

    def _optimize(self, loss):
        if not torch.isfinite(loss):
            raise FloatingPointError("El objetivo financiero no es finito")
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.network.parameters(), self.config.gradient_norm, error_if_nonfinite=True
        )
        self.optimizer.step()
        self.last_loss, self.updates = float(loss.detach().cpu()), self.updates + 1

    def _double_update(self):
        if self.replay.size < self.config.batch_size or self.step_count < self.config.warmup_steps:
            return
        batch = self.replay.sample(self.config.batch_size, self.rng, self.device)
        with torch.no_grad():
            target = double_targets(
                batch["reward"],
                batch["terminated"],
                self.network(batch["following"]),
                self.target(batch["following"]),
                self.config.gamma,
            )
        selected = self.network(batch["observation"]).gather(1, batch["action"][:, None]).squeeze(1)
        self._optimize(F.smooth_l1_loss(selected, target))
        if self.step_count % self.config.target_interval == 0:
            self.target.load_state_dict(self.network.state_dict())

    def _ppo_update(self):
        if not self.rollout:
            return
        fields = {
            key: np.asarray([row[key] for row in self.rollout])
            for key in ("reward", "value", "next_value", "terminated", "truncated")
        }
        advantages, returns = generalized_advantage(
            fields["reward"],
            fields["value"],
            fields["next_value"],
            fields["terminated"],
            fields["truncated"],
            gamma=self.config.gamma,
            lam=self.config.gae_lambda,
        )
        observations = self._tensor(np.stack([row["observation"].numpy() for row in self.rollout]))
        actions = torch.tensor([row["action"] for row in self.rollout], device=self.device)
        old = self._tensor([row["logp"] for row in self.rollout])
        advantages = self._tensor(advantages)
        advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(
            1e-8
        )
        returns = self._tensor(returns)
        for _ in range(self.config.ppo_epochs):
            permutation = self.rng.permutation(len(self.rollout))
            for start in range(0, len(permutation), self.config.batch_size):
                indices = torch.tensor(
                    permutation[start : start + self.config.batch_size], device=self.device
                )
                logits, value = self.network(observations[indices])
                logp = torch.log_softmax(logits, dim=-1)
                selected = logp.gather(1, actions[indices, None]).squeeze(1)
                entropy = -(logp.exp() * logp).sum(1).mean()
                loss = ppo_objective(
                    selected, old[indices], advantages[indices], clip=self.config.clip
                )
                loss += (
                    self.config.value_weight * F.mse_loss(value, returns[indices])
                    - self.config.entropy * entropy
                )
                self._optimize(loss)
        self.rollout.clear()

    def advance(self):
        action, logp, value = self._action()
        following, reward, terminated, truncated, info = self.env.step(action)
        self.step_count += 1
        if info["reward_valid"]:
            self.episode_return += reward
            if self.algorithm == "double_dqn":
                self.replay.add(self.observation, following, action, reward, terminated)
                self._double_update()
            else:
                with torch.no_grad():
                    _, future = self.network(self._tensor(following)[None])
                self.rollout.append(
                    dict(
                        observation=torch.from_numpy(self.observation.copy()),
                        action=action,
                        logp=logp,
                        value=value,
                        next_value=float(future.item()),
                        reward=reward,
                        terminated=terminated,
                        truncated=truncated,
                    )
                )
                if len(self.rollout) >= self.config.rollout_steps:
                    self._ppo_update()
        elif self.rollout:
            self.rollout[-1]["truncated"] = True
        self.observation = following
        if terminated or truncated:
            if info["reward_valid"]:
                self.returns = (self.returns + [self.episode_return])[-128:]
            self.episode_return = 0.0
            self.episodes += 1
            self.observation, _ = self.env.reset(seed=self.seed + self.episodes)
        if self.lease:
            self.lease.check()

    def snapshot(self):
        return dict(
            global_step=self.step_count,
            network=_cpu(self.network.state_dict()),
            target=_cpu(self.target.state_dict()) if self.target is not None else None,
            optimizer=_cpu(self.optimizer.state_dict()),
            replay=self.replay.snapshot() if self.replay is not None else None,
            rollout=_cpu(self.rollout),
            environment=self.env.snapshot(),
            rng=capture_rng(self.device),
            sampling_rng=self.sampling.get_state().cpu(),
            numpy_rng=copy.deepcopy(self.rng.bit_generator.state),
            episodes=self.episodes,
            episode_return=self.episode_return,
            returns=self.returns,
            updates=self.updates,
            last_loss=self.last_loss,
        )

    def restore(self, state):
        if (
            not 0 <= state["global_step"] <= self.config.total_steps
            or len(state["rollout"]) > self.config.rollout_steps
        ):
            raise ValueError("El estado de entrenamiento excede su presupuesto")
        self.env.restore(state["environment"])
        self.network.load_state_dict(state["network"])
        if self.target is not None:
            self.target.load_state_dict(state["target"])
        self.optimizer.load_state_dict(state["optimizer"])
        if self.replay is not None:
            self.replay.restore(state["replay"])
        self.rollout = _cpu(state["rollout"])
        self.rng.bit_generator.state = state["numpy_rng"]
        self.sampling.set_state(state["sampling_rng"])
        restore_rng(state["rng"], self.device)
        self.step_count, self.episodes, self.updates = (
            state["global_step"],
            state["episodes"],
            state["updates"],
        )
        self.episode_return, self.returns, self.last_loss = (
            state["episode_return"],
            state["returns"],
            state["last_loss"],
        )
        self.observation = self.env._observation()

    def run(self, output, *, resume=False, stop_after=None, stop=None):
        output = Path(output)
        if output.exists() and not resume or resume and not output.is_dir():
            raise ValueError("Usa una ejecución nueva o recuperación explícita")
        output.mkdir(parents=True, exist_ok=resume)
        started = time.perf_counter()
        report = dict(
            schema_version=1,
            activity="rl",
            model=self.algorithm,
            status="running",
            identity=self.identity,
            domain="technical" if self.device == "cpu" else self.env.tape.domain,
            parent_frozen=True,
            final_test_opened=False,
            seed=self.seed,
            partition="train",
            started_at=datetime.now(UTC).isoformat(),
            resources=self.lease.record if self.lease else {"device": "cpu", "diagnostic": True},
        )
        with (output / ".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            confirmed = initialize_receipt(output, self.identity, record="run.json", lock=".lock")
            try:
                if confirmed and (output / "checkpoints" / "latest.json").is_file():
                    self.restore(
                        load_training_state(output / "checkpoints", expected_identity=self.identity)
                    )
                elif confirmed and (output / "checkpoints").exists():
                    raise ValueError("Falta el índice del estado confirmado")
                atomic_json(output / "run.json", report)
                save_training_state(output / "checkpoints", self.snapshot(), identity=self.identity)
                while self.step_count < self.config.total_steps:
                    if (stop and stop()) or (
                        stop_after is not None and self.step_count >= stop_after
                    ):
                        report["status"] = "paused"
                        break
                    self.advance()
                    if self.step_count % self.config.checkpoint_steps == 0:
                        save_training_state(
                            output / "checkpoints", self.snapshot(), identity=self.identity
                        )
                        report["global_step"] = self.step_count
                        atomic_json(output / "run.json", report)
                else:
                    if self.algorithm == "ppo":
                        self._ppo_update()
                    report["status"] = "completed"
                path = save_training_state(
                    output / "checkpoints", self.snapshot(), identity=self.identity
                )
                report["checkpoint"] = dict(path=str(path.relative_to(output)), sha256=sha256(path))
            except BaseException as error:
                report.update(status="failed", error_type=type(error).__name__)
                raise
            finally:
                report.update(
                    global_step=self.step_count,
                    total_steps=self.config.total_steps,
                    episodes=self.episodes,
                    updates=self.updates,
                    last_loss=self.last_loss,
                    updated_at=datetime.now(UTC).isoformat(),
                    parameters=sum(p.numel() for p in self.network.parameters()),
                    attempt_seconds=time.perf_counter() - started,
                    process_lifetime_peak_rss_bytes=resource.getrusage(
                        resource.RUSAGE_SELF
                    ).ru_maxrss
                    * 1024,
                    peak_vram_allocated_bytes=torch.cuda.max_memory_allocated(0)
                    if self.device == "cuda:0"
                    else None,
                )
                atomic_json(output / "run.json", report)
        return report
