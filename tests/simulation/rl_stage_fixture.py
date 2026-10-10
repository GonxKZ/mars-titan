"""Campaña base reducida y edición sintética para recorrer la etapa de políticas sin aprender.

El protocolo US se recorta a cuatro ventanas, que evalúan de 2020 a 2023, y la comparación
a un brazo GRU con la semilla 42, como en `tests.posttraining.campaign_fixture`. Sus
ejecutores sustitutos escriben predicciones reales de un padre con pesos iniciales y la
campaña no aplica ningún paso de optimizador. La edición sintética, identificada como
fixture, cubre desde septiembre de 2019. Los ejecutores aprendidos se sustituyen por
`ScriptedLearner`, que no ajusta ninguna red y evalúa una política guionizada.
"""

import hashlib
import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from mars_titan.data.storage import atomic_json, sha256
from mars_titan.simulation import campaign_stage, native_policy_runs
from mars_titan.training import masked_campaign as engine
from mars_titan.training.learning_hold import HOLD_ENV
from tests.posttraining import campaign_fixture
from tests.simulation import unadjusted_edition_fixture as edition_fixture
from tests.training.test_walk_forward_v2_views import fixture

CONFIGS = Path("configs").resolve()
REFERENCES = ["cash", "hold_initial", "rebalance_50"]
HISTORY = "2019-09-01"


@contextmanager
def longer_history():
    """La edición sintética empieza en septiembre de 2019 para cubrir cuatro evaluaciones."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(edition_fixture, "HISTORY_START", HISTORY)
        yield


def write_edition(root, assets):
    with longer_history():
        return edition_fixture.write_edition(root, assets)


def policies(**changes):
    """Políticas reducidas: KLPO y Double DQN, las tres referencias y un año de ajuste.

    La campaña reducida produce los brazos GRU y LSTM. Los dos entran en el nivel completo
    y solo la GRU en el de algoritmos, que también fija el universo.
    """
    value = json.loads((CONFIGS / "simulation/historical-masked-rl-policies.json").read_text())
    value["levels"]["algorithms"].update(predictors=["gru"], arms=["double_dqn"])
    value.update(
        train_windows=1,
        universe=dict(rule="median_traded_value_in_validation_v1", max_assets=4),
        # Caben dos oleadas KLPO de dos episodios anuales.
        budget=dict(
            transitions=1024, environments=2, rollout_transitions=32, evaluation_transitions=32
        ),
        policies={key: value["policies"][key] for key in ("klpo_terminal", "double_dqn")},
        contrasts=dict(
            primary="klpo_terminal",
            controls=["double_dqn", "cash", "hold_initial", "rebalance_50"],
        ),
    )
    value["hyperparameters"]["minibatch_size"] = 16
    value.update(changes)
    return value


def write_configs(folder, variant):
    """Campaña base de cuatro ventanas y dos brazos, políticas reducidas y etapa."""
    campaign, _ = campaign_fixture.write_configs(folder, variant)
    protocol = json.loads((folder / "us-protocol.json").read_text())
    protocol["first_validation_start"] = "2019-04-01"
    atomic_json(folder / "us-protocol.json", protocol)
    # Un segundo predictor de la campaña para el nivel que recorre todos los productores.
    comparison = json.loads((folder / "comparison.json").read_text())
    comparison["arms"]["lstm"] = dict(comparison["arms"]["gru"])
    atomic_json(folder / "comparison.json", comparison)
    declared = json.loads(campaign.read_text())
    declared["neural"]["arms"] = dict(gru="gru", lstm="lstm")
    atomic_json(campaign, declared)
    atomic_json(folder / "rl-policies.json", policies())
    stage = json.loads(
        (CONFIGS / f"simulation/historical-masked-rl-stage-{variant.lower()}.json").read_text()
    )
    stage.update(
        campaign="campaign.json",
        policies="rl-policies.json",
        scopes=["US"],
        limits=dict(max_training_jobs=100, max_evaluation_jobs=100),
    )
    atomic_json(folder / "rl-stage.json", stage)
    return campaign, folder / "rl-stage.json"


def base_campaign(root, variant):
    """Vistas, campaña base con los dobles de la etapa de adaptadores y edición sintética."""
    campaign, stage = write_configs(root / "config", variant)
    data = fixture(root / "data", ("US",))
    hold = root / "hold.json"
    hold.write_text(json.dumps({"training_allowed": True}), encoding="utf-8")
    with pytest.MonkeyPatch.context() as patch:
        # La campaña base solo ejecuta dobles: ningún modelo se ajusta.
        patch.setenv(HOLD_ENV, str(hold))
        # Puntuaciones de validación fijas de los candidatos LSTM, como las de la GRU.
        patch.setitem(campaign_fixture.SCORES, "lstm-00", 0.03)
        patch.setitem(campaign_fixture.SCORES, "lstm-10", 0.025)
        engine.prepare_views(campaign, data.parent, root / "views")
        views = {"US": root / "views" / "US"}
        summary = engine.run_campaign(
            campaign,
            views,
            root / "campaign",
            executors=campaign_fixture.executors(),
            lease=campaign_fixture.CpuLease,
            stop=SimpleNamespace(requested=False),
        )
    assert summary["status"] == "completed"
    assets = {
        "US": [
            edition_fixture.Asset("A0000", base=20.0),
            edition_fixture.Asset("B0001", base=30.0),
        ]
    }
    write_edition(root / "edition", assets)
    return SimpleNamespace(
        campaign=campaign,
        stage=stage,
        views=views,
        output=root / "campaign",
        edition=root / "edition",
        root=root,
    )


def scripted_action(observation, step):
    """Acción fija por la huella de la observación: no aprende y no ve nada más."""
    return hashlib.sha256(observation.tobytes()).digest()[0] % 6


class ScriptedLearner:
    """Ejecutor sustituto de un brazo aprendido, sin redes ni pasos de optimizador.

    Un ajuste guarda una política guionizada como estado elegido y la evalúa con el
    criterio y el presupuesto declarados. Un traslado evalúa la política del ancla.
    `pause` pide una pausa en la primera llamada de esos trabajos y `mutate` altera el
    informe para comprobar que la etapa lo rechaza.
    """

    def __init__(self, *, pause=(), mutate=None, action=scripted_action):
        self.calls, self.pause, self.mutate, self.action = [], set(pause), mutate, action

    def __call__(self, job, tapes, folder, *, stage, resume, stop, anchor):
        self.calls.append(dict(id=job["id"], resume=resume, tapes=tapes, anchor=anchor))
        extra = {}
        if job["id"] in self.pause and not resume:
            atomic_json(folder / "checkpoint.json", dict(job=job["id"], transitions=32))
            return dict(status="paused")
        if job["kind"] == campaign_stage.FIT:
            if resume:
                assert (folder / "checkpoint.json").is_file()
            state = folder / "policy.json"
            atomic_json(state, dict(job=job["id"], seed=job["seed"], rule="observation_sha256"))
            policy = dict(id=job["id"], sha256=sha256(state))
            selection = dict(
                metric=stage["policies"]["selection"]["metric"], partition="validation"
            )
            budget = stage["policies"]["budget"]
            transitions = budget["transitions"]
            if job["engine"] == "native_klpo":
                # KLPO declara las oleadas completas que caben en el presupuesto.
                waves, wave = native_policy_runs.klpo_waves(
                    tapes.train, budget["environments"], transitions
                )
                extra, transitions = dict(waves=waves), waves * wave
        else:
            policy, selection, transitions = anchor["policy"], None, 0
        report = dict(
            **extra,
            status="completed",
            transitions=transitions,
            updates=0,
            selection=selection,
            policy=policy,
            evaluation=campaign_stage.evaluate_policy(
                tapes, self.action, stage, job["market"], backend="python"
            ),
        )
        return self.mutate(job, report) if self.mutate else report


def executors(learner, backend="python"):
    """Referencias con la contabilidad indicada y brazos aprendidos sustituidos."""
    reference = campaign_stage.reference_executor(backend)
    return {
        "reference": dict(run=reference, requires=(), native=False),
        "native_klpo": dict(run=learner, requires=(), native=False),
        "native_ppo": dict(run=learner, requires=(), native=False),
    }


def run(base, output, learner, *, stop=None, backend="python"):
    return campaign_stage.run_stage(
        base.stage,
        base.views,
        base.output,
        base.edition,
        output,
        executors=executors(learner, backend),
        capabilities={},
        stop=stop or SimpleNamespace(requested=False),
    )
