"""PPO, Double DQN y KLPO nativos sobre cintas reconstruidas por ventana, sin aprender.

Las cintas proceden de `policy_tape_fixture`: una edición sintética con el formato real y
ventanas mensuales de 2023 montadas con el mismo constructor que la etapa. Ninguna prueba
ejecuta un paso de optimizador. PPO se pausa antes de completar su primer recorrido, Double
DQN no actualiza antes de 256 pasos de entorno y termina con 32 transiciones, y KLPO se
pausa en la frontera de la oleada completa, antes de `update_ready`. La protección temporal
se permite solo para lanzar los binarios en diagnóstico CPU y cada prueba comprueba después
que no hubo actualizaciones. La selección cerrada de KLPO se fabrica en la prueba con el
actor inicial para recorrer su evaluación congelada.
"""

import copy
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from mars_titan.data.storage import sha256
from mars_titan.simulation import campaign_stage, native_policy_runs
from mars_titan.simulation.market import MarketTape
from mars_titan.simulation.storage import read_tape, write_tape
from mars_titan.training.learning_hold import HOLD_ENV
from tests.simulation.policy_tape_fixture import ROLES, monthly_window, write_policy_tapes

ROOT = Path(__file__).resolve().parents[2]
POLICIES = ROOT / "configs/simulation/historical-masked-rl-policies.json"


@pytest.fixture(scope="module")
def binaries():
    paths = {
        engine: native_policy_runs.binary_path(engine) for engine in ("native_ppo", "native_klpo")
    }
    if not all(path.is_file() for path in paths.values()):
        pytest.skip("Faltan mars-titan-ppo o mars-titan-klpo compilados")
    return paths


@pytest.fixture(scope="module")
def tapes(tmp_path_factory):
    root = tmp_path_factory.mktemp("policy-tapes")
    result = {market: write_policy_tapes(root, market) for market in ("US", "CN")}
    result["US-lag1"] = write_policy_tapes(root, "US", lag=1)
    return result


def diagnostic_stage(**budget):
    """Etapa declarada con un presupuesto de diagnóstico CPU de 32 transiciones."""
    policies = json.loads(POLICIES.read_text())
    policies["budget"] = dict(
        dict(transitions=32, environments=16, rollout_transitions=32, evaluation_transitions=32),
        **budget,
    )
    policies["hyperparameters"].update(minibatch_size=16, epochs=2)
    return dict(policies=policies)


def job(arm, engine, *, kind="fit", window="fold-012"):
    return dict(
        id=f"US/US/{window}/fixture/{arm}/{kind}-s42",
        kind=kind,
        arm=arm,
        engine=engine,
        seed=42,
        market="US",
    )


def write_config(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document))
    return path


def run(binary, hold, *arguments):
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="-1")
    environment[HOLD_ENV] = str(hold)
    result = subprocess.run(
        [str(binary), *map(str, arguments), "--device", "cpu", "--diagnostic"],
        env=environment,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    for name in ("AddressSanitizer", "UndefinedBehaviorSanitizer", "runtime error:"):
        assert name not in result.stderr, result.stderr
    return result


def sources(market_tapes, *, train=(0, 1)):
    arguments = []
    for index in train:
        arguments += ["--train-tape", market_tapes["train"][index][0]]
    return [*arguments, "--validation-tape", market_tapes["validation"][0][0]]


def read(path):
    return json.loads(Path(path).read_text())


def sealed(path):
    return read(path)["payload"]


def latest_metadata(output):
    bundle = read(output / "ppo-index.json")["payload"]["recent"][0]["bundle"]
    return read(output / bundle / "metadata.json")


@pytest.mark.parametrize("market", ["US", "CN"])
def test_ppo_pauses_on_real_tapes_before_its_first_update(
    binaries, tapes, tmp_path, learning_hold, market
):
    hold = learning_hold(True)
    config = write_config(
        tmp_path / "ppo.json",
        native_policy_runs.ppo_config(diagnostic_stage(), job("ppo_clip_full_kl", "native_ppo")),
    )
    output = tmp_path / "out"
    arguments = ["--config", config, "--output", output, *sources(tapes[market])]
    for extra in ([], ["--resume"]):
        result = run(binaries["native_ppo"], hold, *arguments, "--stop-after", 16, *extra)
        assert result.returncode == 2, result.stderr
        report = read(output / "run.json")
        assert (report["status"], report["transitions"], report["optimizer_steps"]) == (
            "paused",
            16,
            0,
        )
    assert report["domain"] == "technical" and report["selected_policy_sha256"] is None
    assert report["evaluations"] == 1 and "analysis_domain" not in report
    # Una selección abierta no se evalúa.
    audit = ["--audit-run", output, "--audit-tape", tapes[market]["evaluation"][0][0]]
    result = run(
        binaries["native_ppo"], hold, "--config", config, "--output", tmp_path / "open", *audit
    )
    assert result.returncode == 1 and "selección finalizada" in result.stderr
    assert not (tmp_path / "open").exists()
    identity = read(output / "identity.json")["identity"]
    assert identity["schema_version"] == 4 and identity["sources_contract"] == (
        "unadjusted_reconstructed_walk_forward_v1"
    )
    schema = identity["observation_schema"]
    assert schema["domain"] == "real" and schema["parent_id"] is None
    assert schema["historical_basis"].startswith(f"{market}/") and schema["context_fields"] == []
    train = identity["sources"]["train"]
    assert [item["role"] for item in train] == ["train", "train"]
    assert train[0]["last_close"] < train[1]["first_open"]
    assert train[1]["last_close"] < identity["sources"]["validation"][0]["first_open"]


@pytest.mark.parametrize("market", ["US", "CN"])
def test_double_dqn_selects_and_evaluates_the_chosen_state_without_updates(
    binaries, tapes, tmp_path, learning_hold, market
):
    hold = learning_hold(True)
    stage = diagnostic_stage(rollout_transitions=16)
    config = write_config(
        tmp_path / "dqn.json", native_policy_runs.ppo_config(stage, job("double_dqn", "native_ppo"))
    )
    fit, evaluated = tmp_path / "fit", tmp_path / "evaluation"
    result = run(
        binaries["native_ppo"], hold, "--config", config, "--output", fit, *sources(tapes[market])
    )
    assert result.returncode == 0, result.stderr
    report = read(fit / "run.json")
    assert (report["status"], report["transitions"], report["optimizer_steps"]) == (
        "completed",
        32,
        0,
    )
    assert report["model"] == "double_dqn" and report["best"]["optimizer_steps"] == 0
    evaluation_tape = tapes[market]["evaluation"][0]
    arguments = ["--config", config, "--output", evaluated, "--audit-run", fit]
    arguments += ["--audit-tape", evaluation_tape[0]]
    for extra in ([], ["--resume"]):
        result = run(binaries["native_ppo"], hold, *arguments, *extra)
        assert result.returncode == 0, result.stderr
    document = sealed(evaluated / "evaluation.json")
    assert document["status"] == "completed" and document["domain"] == "real"
    assert document["identity"]["policy_sha256"] == report["selected_policy_sha256"]
    assert document["identity"]["optimizer_steps"] == 0
    manifest = sha256(evaluation_tape[0] / "manifest.json")
    assert [(row["cost_bps"], row["manifest_sha256"]) for row in document["metrics"]] == [
        (cost, manifest) for cost in (0.0, 10.0, 25.0)
    ]
    for row in document["metrics"]:
        assert (
            row["status"] in ("completed", "ruined") and row["steps"] == len(evaluation_tape[1]) - 1
        )
        assert row["liquidated_net_return"] > -1
    # La venta final descuenta la tarifa declarada y, en China, el timbre de venta vigente en el
    # último cierre, como `terminal_liquidation`. China paga además el timbre en cada venta,
    # también con tarifa cero, y esta política termina invertida.
    zero = document["metrics"][0]
    assert (zero["net_return"] == zero["liquidated_net_return"]) is (market == "US")
    assert zero["liquidated_net_return"] <= zero["net_return"]
    assert (zero["costs"] == 0) is (market == "US")
    # Una cinta de evaluación que no es posterior a la validación se rechaza.
    early = ["--audit-run", fit, "--audit-tape", tapes[market]["early_evaluation"][0][0]]
    result = run(
        binaries["native_ppo"], hold, "--config", config, "--output", tmp_path / "early", *early
    )
    assert result.returncode == 1 and "posterior a la selección" in result.stderr
    assert not (tmp_path / "early").exists()


def test_klpo_collects_a_full_wave_and_pauses_before_update_ready(
    binaries, tapes, tmp_path, learning_hold
):
    hold = learning_hold(True)
    stage = diagnostic_stage(environments=1)
    config = write_config(
        tmp_path / "klpo.json",
        native_policy_runs.klpo_config(stage, job("klpo_terminal", "native_klpo")),
    )
    output = tmp_path / "out"
    us = tapes["US"]
    arguments = ["--config", config, "--output", output, *sources(us, train=(1,))]
    wave = len(us["train"][1][1]) - 1
    result = run(binaries["native_klpo"], hold, *arguments, "--stop-after", 7)
    assert result.returncode == 2, result.stderr
    report = read(output / "run.json")
    assert (report["collected_transitions"], report["consumed_waves"]) == (7, 0)
    assert (report["planned_waves"], report["wave_transitions"]) == (1, wave)
    assert latest_metadata(output)["phase"] == "collecting"
    for _ in range(2):
        result = run(binaries["native_klpo"], hold, *arguments, "--stop-after", wave, "--resume")
        assert result.returncode == 2, result.stderr
        report = read(output / "run.json")
        assert (report["collected_transitions"], report["transitions"]) == (wave, 0)
        assert (report["optimizer_steps"], report["consumed_waves"]) == (0, 0)
        metadata = latest_metadata(output)
        # La oleada completa queda confirmada antes de la actualización, sin ejecutarla.
        assert metadata["phase"] == "ready" and metadata["counters"]["confirmed_updates"] == 0
    assert len(read(output / "ppo-index.json")["payload"]["recent"]) == 2
    selection = sealed(output / "selection.json")
    assert selection["evaluated_waves"] == 0 and selection["best"]["wave"] == 0
    assert (output / selection["best"]["actor_file"]).is_file()
    experiment = sealed(output / "experiment.json")
    assert experiment["budget_rule"] == "complete_waves_within_declared_transitions"
    assert experiment["sources"]["train"][0]["role"] == "train"

    evaluation_tape = us["evaluation"][0][0]
    audit = ["--config", config, "--audit-run", output, "--audit-tape", evaluation_tape]
    result = run(binaries["native_klpo"], hold, *audit, "--output", tmp_path / "open")
    assert result.returncode == 1 and "selección cerrada" in result.stderr
    assert not (tmp_path / "open").exists()
    # Selección cerrada fabricada con el actor inicial, solo para recorrer la evaluación.
    closed = tmp_path / "closed"
    shutil.copytree(output, closed)
    path = closed / "selection.json"
    payload = dict(sealed(path), status="completed", evaluated_waves=1)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    path.write_text(
        json.dumps(dict(payload=payload, sha256=hashlib.sha256(encoded.encode()).hexdigest()))
    )
    audit[audit.index(output)] = closed
    result = run(binaries["native_klpo"], hold, *audit, "--output", tmp_path / "evaluated")
    assert result.returncode == 0, result.stderr
    document = sealed(tmp_path / "evaluated/evaluation.json")
    assert document["identity"]["policy_sha256"] == payload["best"]["actor_sha256"]
    assert document["identity"]["optimizer_steps"] == 0
    assert [row["cost_bps"] for row in document["metrics"]] == [0.0, 10.0, 25.0]


def test_klpo_rejects_a_budget_below_one_complete_wave(binaries, tapes, tmp_path, learning_hold):
    hold = learning_hold(True)
    us = tapes["US"]
    wave = len(us["train"][1][1]) - 1
    stage = diagnostic_stage(environments=1, transitions=wave - 1, evaluation_transitions=wave - 1)
    config = write_config(
        tmp_path / "klpo.json",
        native_policy_runs.klpo_config(stage, job("klpo_terminal", "native_klpo")),
    )
    output = tmp_path / "out"
    arguments = ["--config", config, "--output", output, *sources(us, train=(1,))]
    # La pausa pedida evita cualquier actualización aunque faltara la comprobación.
    result = run(binaries["native_klpo"], hold, *arguments, "--stop-after", 1)
    assert result.returncode == 1 and "ni una oleada completa" in result.stderr


def group_config(path, stage, objective="grpo_outcome_v1"):
    """Configuración de un brazo de grupo con el mismo esquema que KLPO."""
    stage = copy.deepcopy(stage)
    stage["policies"]["policies"]["grpo"] = dict(
        engine="native_group_relative",
        objective=objective,
        controller="group_relative_fresh_waves_v1",
        confirmed_updates_per_reference=2,
    )
    return write_config(
        path, native_policy_runs.klpo_config(stage, job("grpo", "native_group_relative"))
    )


def test_group_objectives_reject_lonely_lanes_and_unknown_identities(
    binaries, tapes, tmp_path, learning_hold
):
    hold = learning_hold(True)
    us = tapes["US"]
    # Un solo carril por cinta no deja línea base y se rechaza antes de recoger.
    config = group_config(tmp_path / "group.json", diagnostic_stage(environments=1))
    arguments = ["--config", config, "--output", tmp_path / "lonely", *sources(us, train=(1,))]
    result = run(binaries["native_klpo"], hold, *arguments, "--stop-after", 1)
    assert result.returncode == 1 and "dos carriles por cinta" in result.stderr
    stage = diagnostic_stage(environments=2)
    unknown = group_config(tmp_path / "unknown.json", stage, objective="grpo_plus_plus")
    arguments = ["--config", unknown, "--output", tmp_path / "unknown", *sources(us, train=(1,))]
    result = run(binaries["native_klpo"], hold, *arguments, "--stop-after", 1)
    assert result.returncode == 1 and "objetivo de grupo" in result.stderr
    assert not (tmp_path / "unknown").exists()


def blocking(tmp_path):
    path = tmp_path / "hold.json"
    path.write_text(json.dumps({"training_allowed": False}))
    return path


def test_the_learning_hold_stops_every_real_tape_command_before_outputs(binaries, tapes, tmp_path):
    hold = blocking(tmp_path)
    stage = diagnostic_stage(rollout_transitions=16)
    ppo = write_config(
        tmp_path / "ppo.json", native_policy_runs.ppo_config(stage, job("double_dqn", "native_ppo"))
    )
    klpo = write_config(
        tmp_path / "klpo.json",
        native_policy_runs.klpo_config(
            diagnostic_stage(environments=1), job("klpo_terminal", "native_klpo")
        ),
    )
    us = tapes["US"]
    evaluation = ["--audit-run", tmp_path / "run", "--audit-tape", us["evaluation"][0][0]]
    cases = [
        ("native_ppo", ppo, sources(us), "el entrenamiento nativo sobre cintas reconstruidas"),
        ("native_ppo", ppo, evaluation, "la evaluación nativa sobre cintas reconstruidas"),
        ("native_klpo", klpo, sources(us, train=(1,)), "el entrenamiento KLPO nativo"),
        ("native_klpo", klpo, evaluation, "la evaluación KLPO nativa"),
        (
            "native_klpo",
            group_config(tmp_path / "group.json", diagnostic_stage(environments=2)),
            sources(us, train=(1,)),
            "el entrenamiento nativo con objetivo de grupo",
        ),
    ]
    for number, (engine, config, arguments, action) in enumerate(cases):
        output = tmp_path / f"out-{number}"
        result = run(binaries[engine], hold, "--config", config, "--output", output, *arguments)
        assert result.returncode == 1 and f"Bloqueo de aprendizaje vigente: {action}" in (
            result.stderr
        )
        assert not output.exists()


def synthetic_tape(folder):
    day = 86_400_000_000
    times = 1_693_526_400_000_000 + day * np.arange(5, dtype=np.int64)
    prices = np.tile([10.0, 10.0, 10.0, 10.0, 1e6], (5, 1, 1))
    tape = MarketTape(
        prices, times, ["US/AAA"], np.zeros((5, 1)), domain="synthetic", currency="USD"
    )
    write_tape(tape, folder)
    return folder


def without_rules(source, folder):
    tape = read_tape(source)
    write_tape(tape, folder)
    return folder


def test_reconstructed_sources_keep_role_order_basis_and_rules(
    binaries, tapes, tmp_path, learning_hold
):
    hold = learning_hold(True)
    stage = diagnostic_stage(rollout_transitions=16)
    config = write_config(
        tmp_path / "dqn.json", native_policy_runs.ppo_config(stage, job("double_dqn", "native_ppo"))
    )
    us, lagged, cn = tapes["US"], tapes["US-lag1"], tapes["CN"]
    train = [us["train"][0][0], us["train"][1][0]]
    validation = us["validation"][0][0]
    cases = {
        "reversed_train": ([train[1], train[0]], validation, "empieza antes"),
        "validation_before_train": ([train[1]], train[0], "papel"),
        "evaluation_role_as_train": ([us["evaluation"][0][0]], validation, "papel"),
        "other_dividend_lag": (train, lagged["validation"][0][0], "base histórica"),
        "other_market": (train, cn["validation"][0][0], "base histórica"),
        "synthetic_source": (
            [synthetic_tape(tmp_path / "synthetic")],
            validation,
            "papel",
        ),
        "china_without_rules": (
            [without_rules(cn["train"][0][0], tmp_path / "bare")],
            cn["validation"][0][0],
            "reglas",
        ),
    }
    for name, (train_paths, validation_path, message) in cases.items():
        output = tmp_path / f"out-{name}"
        arguments = ["--config", config, "--output", output, "--validation-tape", validation_path]
        for path in train_paths:
            arguments += ["--train-tape", path]
        result = run(binaries["native_ppo"], hold, *arguments)
        assert result.returncode == 1 and message in result.stderr, (name, result.stderr)
        assert not output.exists(), name


def test_schema_four_rejects_masks_hmm_and_other_variants(binaries, tapes, tmp_path, learning_hold):
    hold = learning_hold(True)
    base = native_policy_runs.ppo_config(
        diagnostic_stage(rollout_transitions=16), job("double_dqn", "native_ppo")
    )
    changes = {
        "mask": dict(trading_field=0),
        "hmm": dict(markov_fields=[0, 1, 2]),
        "memory_variant": dict(variant="ppo_episodic"),
    }
    for name, change in changes.items():
        document = copy.deepcopy(base)
        document["agent"].update(change)
        config = write_config(tmp_path / f"{name}.json", document)
        output = tmp_path / f"out-{name}"
        result = run(
            binaries["native_ppo"],
            hold,
            "--config",
            config,
            "--output",
            output,
            *sources(tapes["US"]),
        )
        assert result.returncode == 1 and "esquema 4" in result.stderr, (name, result.stderr)
        assert not output.exists()


def policy_tapes(market_tapes, *, evaluation=True):
    train = market_tapes["train"]
    validation = market_tapes["validation"][0]
    evaluated = market_tapes["evaluation"][0]
    return campaign_stage.PolicyTapes(
        universe=tuple(validation[1].assets),
        train=tuple(tape for _, tape in train),
        validation=validation[1],
        evaluation=evaluated[1] if evaluation else None,
        failure=None if evaluation else dict(reason="universe_assets_excluded"),
        unfit=None,
        paths=dict(
            train=[str(path) for path, _ in train],
            validation=str(validation[0]),
            evaluation=str(evaluated[0]) if evaluation else None,
        ),
        identity={},
    )


def test_stage_executors_fit_carry_and_pause_through_the_launcher(
    binaries, tapes, tmp_path, learning_hold
):
    learning_hold(True)
    stage = diagnostic_stage(rollout_transitions=16)
    stage["policies"]["evaluation_costs_bps"] = [0, 10, 25]
    executor = native_policy_runs.NativePolicyExecutor("native_ppo", diagnostic=True)
    stop = None
    fit_job = job("double_dqn", "native_ppo")
    fit_folder = tmp_path / "jobs" / fit_job["id"] / "run"
    fit_folder.mkdir(parents=True)
    us = policy_tapes(tapes["US"])
    report = executor(fit_job, us, fit_folder, stage=stage, resume=False, stop=stop, anchor=None)
    assert report["status"] == "completed" and report["updates"] == 0
    assert report["transitions"] == 32 and report["policy"]["id"] == fit_job["id"]
    assert [record["cost_bps"] for record in report["evaluation"]] == [0, 10, 25]
    campaign_stage.check_report(stage, fit_job, report, us)
    # El traslado evalúa la política del ancla con su configuración, sin ajustar.
    carry_job = job("double_dqn", "native_ppo", kind="carry", window="fold-013")
    carry_folder = tmp_path / "jobs" / carry_job["id"] / "run"
    carry_folder.mkdir(parents=True)
    anchor = dict(run=f"jobs/{fit_job['id']}/run", policy=report["policy"])
    carried = executor(
        carry_job, us, carry_folder, stage=stage, resume=False, stop=stop, anchor=anchor
    )
    campaign_stage.check_report(stage, carry_job, carried, us, anchor)
    assert carried["evaluation"] == report["evaluation"] and not (carry_folder / "fit").exists()
    # Una evaluación sin cinta registra fallos con su motivo y no lanza el binario.
    failed = executor(
        carry_job,
        policy_tapes(tapes["US"], evaluation=False),
        tmp_path / "jobs" / carry_job["id"] / "run",
        stage=stage,
        resume=True,
        stop=stop,
        anchor=anchor,
    )
    assert {record["reason"] for record in failed["evaluation"]} == {"universe_assets_excluded"}
    # Un ancla con otra huella o una carpeta fuera de la etapa no se evalúan.
    forged = dict(anchor, policy=dict(report["policy"], sha256="c" * 64))
    with pytest.raises(ValueError, match="no corresponde a la política"):
        executor(carry_job, us, carry_folder, stage=stage, resume=True, stop=stop, anchor=forged)
    outside = tmp_path / "elsewhere" / "run"
    outside.mkdir(parents=True)
    with pytest.raises(ValueError, match="no encuentra la salida"):
        executor(carry_job, us, outside, stage=stage, resume=False, stop=stop, anchor=anchor)
    # Cada episodio debe corresponder a la cinta y al coste evaluados.
    row = dict(report["evaluation"][0], manifest_sha256="0" * 64)
    with pytest.raises(ValueError, match="no corresponde a su cinta"):
        native_policy_runs._record(row, 0, sha256(Path(us.paths["evaluation"]) / "manifest.json"))

    klpo_stage = diagnostic_stage(environments=1)
    klpo_stage["policies"]["evaluation_costs_bps"] = [0, 10, 25]
    klpo = native_policy_runs.NativePolicyExecutor("native_klpo", diagnostic=True, stop_after=5)
    klpo_job = job("klpo_terminal", "native_klpo")
    klpo_folder = tmp_path / "klpo" / "jobs" / klpo_job["id"] / "run"
    klpo_folder.mkdir(parents=True)
    single = policy_tapes(dict(tapes["US"], train=tapes["US"]["train"][1:]))
    paused = klpo(
        klpo_job, single, klpo_folder, stage=klpo_stage, resume=False, stop=stop, anchor=None
    )
    assert paused == dict(status="paused")
    assert read(klpo_folder / "fit/run.json")["optimizer_steps"] == 0
    # El ajuste pausado se reanuda en su carpeta y vuelve a pausarse en el mismo punto.
    again = klpo(
        klpo_job, single, klpo_folder, stage=klpo_stage, resume=True, stop=stop, anchor=None
    )
    assert again == dict(status="paused")
    assert read(klpo_folder / "fit/run.json")["collected_transitions"] == 5
    changed = copy.deepcopy(klpo_stage)
    changed["policies"]["selection"]["min_delta"] = 0.5
    with pytest.raises(ValueError, match="configuración nativa cambió"):
        klpo(klpo_job, single, klpo_folder, stage=changed, resume=True, stop=stop, anchor=None)


def test_stage_writes_chinese_tapes_with_their_a_share_rules(tapes, tmp_path):
    folder, tape = tapes["CN"]["validation"][0]
    window, values = monthly_window(
        "CN", ROLES[2][1], [asset.split("/")[1] for asset in tape.assets]
    )
    policies = dict(environment=dict(dividend_payment_lag_sessions=0), universe=dict(max_assets=8))
    edition = folder.parents[1] / "edition"
    stage_tapes = campaign_stage._Tapes(
        policies,
        lambda *_: (window, values),
        edition,
        json.loads((edition / "manifest.json").read_text())["edition_id"],
        tmp_path,
        "fixture",
    )
    job = dict(scope="CN", market="CN", predictor="fixture", anchor="fixture")
    written, built, failure, _ = stage_tapes.tape(
        job, "validation", window.fold, tuple(tape.assets)
    )
    assert failure is None and tuple(built.assets) == tuple(tape.assets)
    manifest = read(written / "manifest.json")
    assert manifest["schema_version"] == 2 and set(manifest["instruments"]) == set(tape.assets)
    assert manifest["instruments"] == read(folder / "manifest.json")["instruments"]


def test_declared_costs_and_session_equity_reach_the_frozen_evaluation(
    binaries, tapes, tmp_path, learning_hold
):
    hold = learning_hold(True)
    stage = diagnostic_stage(rollout_transitions=16)
    stage["policies"]["evaluation_costs_bps"] = [0, 5, 10, 20]
    executor = native_policy_runs.NativePolicyExecutor("native_ppo", diagnostic=True)
    fit_job = job("double_dqn", "native_ppo")
    folder = tmp_path / "jobs" / fit_job["id"] / "run"
    folder.mkdir(parents=True)
    us = policy_tapes(tapes["US"])
    report = executor(fit_job, us, folder, stage=stage, resume=False, stop=None, anchor=None)
    assert report["updates"] == 0
    campaign_stage.check_report(stage, fit_job, report, us)
    assert [record["cost_bps"] for record in report["evaluation"]] == [0, 5, 10, 20]
    document = sealed(folder / "evaluation" / "evaluation.json")
    assert document["identity"]["cost_bps"] == [0.0, 5.0, 10.0, 20.0]
    capital = stage["policies"]["environment"]["capital"]
    tape = us.evaluation
    for record in report["evaluation"]:
        equity = record["equity"]
        assert equity["basis"] == "close_valuation_from_log_rewards"
        assert equity["close_times"] == [int(t) for t in tape.close_times[: record["steps"] + 1]]
        assert equity["nav"][0] == capital and len(equity["nav"]) == record["steps"] + 1
        final = capital * (1 + record["net_return"])
        assert equity["nav"][-1] == pytest.approx(final, rel=1e-12)
    # Más coste nunca deja más patrimonio con las mismas decisiones deterministas.
    finals = [record["equity"]["nav"][-1] for record in report["evaluation"]]
    assert finals == sorted(finals, reverse=True)
    # Una serie que no concilia con su retorno no se confirma.
    broken = copy.deepcopy(report)
    broken["evaluation"][1]["equity"]["nav"][-1] *= 1.001
    with pytest.raises(ValueError, match="no concilia"):
        campaign_stage.check_report(stage, fit_job, broken, us)
    missing = copy.deepcopy(report)
    missing["evaluation"][0]["equity"] = None
    with pytest.raises(ValueError, match="patrimonio en cada cierre"):
        campaign_stage.check_report(stage, fit_job, missing, us)
    # El binario rechaza costes sin orden creciente o sin auditoría.
    config = folder / "config.json"
    evaluation = ["--audit-run", folder / "fit", "--audit-tape", us.paths["evaluation"]]
    for costs in ([10, 5], [5, 5], [-1], [2000]):
        arguments = [item for cost in costs for item in ("--evaluation-cost", cost)]
        result = run(
            binaries["native_ppo"],
            hold,
            "--config",
            config,
            "--output",
            tmp_path / "rejected",
            *evaluation,
            *arguments,
        )
        assert result.returncode == 1 and "costes de evaluación" in result.stderr, result.stderr
        assert not (tmp_path / "rejected").exists()
    result = run(
        binaries["native_ppo"],
        hold,
        "--config",
        config,
        "--output",
        tmp_path / "training",
        *sources(tapes["US"]),
        "--evaluation-cost",
        5,
    )
    assert result.returncode == 1 and "--audit-run" in result.stderr


def test_klpo_budget_counts_complete_waves_over_cycled_train_tapes(tapes):
    train = tuple(tape for _, tape in tapes["US"]["train"])
    lengths = [len(tape) - 1 for tape in train]
    waves, wave = native_policy_runs.klpo_waves(train, 3, 1000)
    assert wave == lengths[0] + lengths[1] + lengths[0]
    assert waves == 1000 // wave
    assert native_policy_runs.klpo_waves(train, 2, sum(lengths) - 1)[0] == 0
