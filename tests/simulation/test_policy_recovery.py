"""Recuperación coherente de un trabajo de políticas pausado, sin pasos de optimizador.

El ejecutor sustituto construye el entrenador Python de PPO o Double DQN sobre la cinta de
ajuste que la etapa montó con el recibo de su ventana. Avanza menos transiciones que el
calentamiento y que el recorrido, de modo que ninguna actualización llega a aplicarse, y
escribe a mano momentos de Adam para que el optimizador tenga un estado no trivial. La
etapa pausa el trabajo y lo reanuda en su carpeta con un entrenador nuevo, que solo
recibe el estado del punto de control confirmado. KLPO nativo sigue pendiente de un
ejecutor financiero, así que el trabajo KLPO recupera aquí el entrenador PPO Python.

La campaña reducida emite muy pocas predicciones y sus primeras sesiones no tienen
puntuación. Para que la cartera abra posiciones y deje órdenes pendientes, el entrenador
recorre los precios y sesiones de la cinta de ajuste con puntuaciones deterministas
positivas, identificadas como sintéticas.
"""

import dataclasses
import random

import numpy as np
import pytest
import torch

from mars_titan.data.storage import atomic_json
from mars_titan.simulation import campaign_stage
from mars_titan.simulation.environment import FinancialEnv
from mars_titan.simulation.market import MarketTape
from mars_titan.simulation.training import FinancialTrainer, TrainConfig
from mars_titan.training.checkpoints import load_training_state, save_training_state
from tests.simulation import rl_stage_fixture as fixture
from tests.simulation.test_recovery_atomicity import assert_state_equal

PAUSE, REST = 12, 10
JOBS = dict(
    double_dqn="US/US/fold-002/gru/double_dqn/fit-s42",
    ppo="US/US/fold-002/gru/klpo_terminal/fit-s42",
)
# El calentamiento y el recorrido superan las transiciones avanzadas: no hay actualización.
CONFIG = TrainConfig(
    total_steps=32,
    batch_size=4,
    rollout_steps=32,
    ppo_epochs=1,
    replay_capacity=64,
    warmup_steps=32,
    target_interval=1024,
)


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    return fixture.base_campaign(tmp_path_factory.mktemp("rl-recovery"), "A")


def scored(tape):
    """Precios y sesiones de la cinta de ajuste con puntuaciones sintéticas positivas."""
    sessions = np.arange(len(tape), dtype=np.float64)[:, None]
    scores = np.broadcast_to(0.01 + 0.005 * np.sin(sessions), tape.scores.shape).copy()
    return MarketTape(
        tape.prices,
        tape.close_times,
        tape.assets,
        scores,
        domain="synthetic",
        currency=tape.currency,
        open_times=tape.open_times,
        actions=tape.actions,
    )


def trainer_for(tape, algorithm, seed, stage):
    environment = {
        key: value
        for key, value in stage["policies"]["environment"].items()
        if key != "dividend_payment_lag_sessions"
    }
    # Con una participación mínima las órdenes se llenan en parte y siguen pendientes al
    # guardar, de modo que el punto de control también tiene que conservarlas.
    environment["participation"] = 1e-5
    # Los generadores globales parten del mismo estado en el ajuste y en la referencia.
    random.seed(7)
    np.random.seed(7)
    env = FinancialEnv(tape, backend="python", **environment)
    return FinancialTrainer(env, algorithm, CONFIG, seed=seed, device="cpu", diagnostic=True)


def hand_moments(trainer):
    """Momentos de Adam escritos a mano: el optimizador no ejecuta ningún paso."""
    state = trainer.optimizer.state_dict()
    moments = {}
    for index, parameter in enumerate(trainer.network.parameters()):
        values = torch.linspace(-1, 1, parameter.numel()).reshape(parameter.shape)
        moments[index] = dict(step=torch.tensor(3.0), exp_avg=values, exp_avg_sq=values.square())
    trainer.optimizer.load_state_dict(dict(state=moments, param_groups=state["param_groups"]))


def backward_only(trainer):
    """Sustituir la aplicación del paso por el cálculo de la pérdida y sus gradientes."""
    gradients = []

    def optimize(loss):
        trainer.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradients.append(
            (loss.detach().clone(), [p.grad.clone() for p in trainer.network.parameters()])
        )

    trainer._optimize = optimize
    trainer.config = dataclasses.replace(trainer.config, warmup_steps=1)
    return gradients


class RecoveringLearner(fixture.ScriptedLearner):
    """Pausa el trabajo indicado tras `PAUSE` transiciones y lo reanuda desde su estado."""

    def __init__(self, algorithm):
        super().__init__()
        self.algorithm, self.job = algorithm, JOBS[algorithm]
        self.paused = self.trainer = self.tape = None

    def __call__(self, job, tapes, folder, *, stage, resume, stop, anchor):
        if job["id"] == self.job:
            # Con una ventana de ajuste, el trabajo recorre la cinta de su único tramo.
            (train,) = tapes.train
            self.tape = scored(train)
            trainer = trainer_for(self.tape, self.algorithm, job["seed"], stage)
            if not resume:
                for _ in range(PAUSE):
                    trainer.advance()
                hand_moments(trainer)
                self.paused = trainer.snapshot()
                save_training_state(folder / "checkpoints", self.paused, identity=trainer.identity)
                atomic_json(folder / "checkpoint.json", dict(job=job["id"], transitions=PAUSE))
                self.calls.append(dict(id=job["id"], resume=resume, tapes=tapes, anchor=anchor))
                return dict(status="paused")
            trainer.restore(
                load_training_state(folder / "checkpoints", expected_identity=trainer.identity)
            )
            for _ in range(REST):
                trainer.advance()
            self.trainer = trainer
        return super().__call__(
            job, tapes, folder, stage=stage, resume=resume, stop=stop, anchor=anchor
        )


@pytest.mark.parametrize("algorithm", sorted(JOBS))
def test_a_paused_policy_job_resumes_portfolio_orders_rng_optimizer_and_replay(
    base, tmp_path, learning_doubles, algorithm
):
    first = RecoveringLearner(algorithm)
    assert fixture.run(base, tmp_path / "stage", first)["status"] == "paused"
    paused = first.paused
    # El punto de control contiene posiciones y una orden pendiente para la apertura siguiente.
    book = paused["environment"]["book"]["state"]
    assert book["positions"] and book["orders"]
    assert paused["environment"]["cursor"] > 0 and paused["global_step"] == PAUSE
    if algorithm == "double_dqn":
        assert paused["replay"]["size"] == PAUSE and paused["rollout"] == []
    else:
        assert paused["replay"] is None and len(paused["rollout"]) == PAUSE
    # Un ejecutor nuevo solo dispone del estado confirmado en la carpeta del trabajo.
    second = RecoveringLearner(algorithm)
    assert fixture.run(base, tmp_path / "stage", second)["status"] == "completed"
    assert [call["resume"] for call in second.calls if call["id"] == JOBS[algorithm]] == [True]
    restored = second.trainer
    stage = campaign_stage.load_stage(base.stage)
    reference = trainer_for(first.tape, algorithm, 42, stage)
    initial = {name: value.clone() for name, value in reference.network.state_dict().items()}
    for _ in range(PAUSE):
        reference.advance()
    hand_moments(reference)
    for _ in range(REST):
        reference.advance()
    # Cartera, órdenes, cursor, generadores, optimizador, replay y recorrido coinciden.
    assert_state_equal(restored.snapshot(), reference.snapshot())
    assert restored.updates == reference.updates == 0
    assert_state_equal(restored.network.state_dict(), initial)
    steps = {float(s["step"]) for s in restored.optimizer.state_dict()["state"].values()}
    assert steps == {3.0}
    # La actualización siguiente produce la misma pérdida y los mismos gradientes.
    expected, actual = backward_only(reference), backward_only(restored)
    update = "_double_update" if algorithm == "double_dqn" else "_ppo_update"
    getattr(reference, update)()
    getattr(restored, update)()
    assert actual and len(actual) == len(expected)
    for (loss, gradients), (other, others) in zip(actual, expected, strict=True):
        torch.testing.assert_close(loss, other, rtol=0, atol=0)
        assert_state_equal(gradients, others)
    assert_state_equal(restored.network.state_dict(), initial)


def test_a_checkpoint_of_another_seed_is_not_restored(base, tmp_path, learning_doubles):
    first = RecoveringLearner("double_dqn")
    fixture.run(base, tmp_path / "stage", first)
    folder = tmp_path / f"stage/jobs/{JOBS['double_dqn']}/run/checkpoints"
    stage = campaign_stage.load_stage(base.stage)
    other = trainer_for(first.tape, "double_dqn", 43, stage)
    with pytest.raises(ValueError):
        load_training_state(folder, expected_identity=other.identity)
