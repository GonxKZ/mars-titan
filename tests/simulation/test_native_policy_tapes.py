"""PPO, Double DQN y KLPO nativos sobre cintas reconstruidas por ventana, sin aprender.

Las cintas proceden de `policy_tape_fixture`: una edición sintética con el formato real y
ventanas mensuales de 2023 montadas con el mismo constructor que la etapa. Ninguna prueba
ejecuta un paso de optimizador.

Con la protección cerrada, las pruebas usan los dos modos sin aprendizaje de los binarios
(#546). `--check-sources` lee y valida las cintas de ajuste y validación con los lectores del
ajuste y publica su identidad, y `--initial-evaluation` evalúa con argmax la política inicial
de la configuración sobre cintas de evaluación. Ninguno crea entrenador, controlador ni Adam
terminal, y la protección no se rebaja ni se consulta.

Las pruebas que lanzan un ajuste (PPO pausado antes de su primer recorrido, Double DQN por
debajo de su calentamiento, KLPO pausado antes de `update_ready` y los ejecutores de la
etapa) figuran en el inventario de `tests/learning_hold_lowering.py` y se omiten mientras la
protección real bloquee.
"""

import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from mars_titan.data.storage import sha256
from mars_titan.simulation import campaign_stage, device_parity, native_policy_runs, window_tapes
from mars_titan.simulation.market import MarketTape
from mars_titan.simulation.storage import read_tape, write_tape
from mars_titan.training.learning_hold import HOLD_ENV, hold_path, learning_blocked
from tests.learning_hold_lowering import lowers_learning_hold
from tests.simulation.policy_tape_fixture import (
    EDITION,
    ROLES,
    monthly_window,
    write_policy_tapes,
)
from tests.simulation.unadjusted_edition_fixture import (
    Asset,
    listing_status,
    tape_days,
    write_edition,
)
from tests.suite_support import missing_native

ROOT = Path(__file__).resolve().parents[2]
POLICIES = ROOT / "configs/simulation/historical-masked-rl-policies.json"
THIS = "tests/simulation/test_native_policy_tapes.py"


@pytest.fixture(scope="module")
def binaries():
    paths = {
        engine: native_policy_runs.binary_path(engine) for engine in ("native_ppo", "native_klpo")
    }
    if not all(path.is_file() for path in paths.values()):
        missing = next(engine for engine, path in paths.items() if not path.is_file())
        pytest.skip(missing_native(native_policy_runs.BINARIES[missing][0]))
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


@pytest.mark.native("MARS_TITAN_PPO_EXECUTABLE", "MARS_TITAN_KLPO_EXECUTABLE")
@pytest.mark.parametrize("market", ["US", "CN"])
def test_ppo_checks_real_sources_with_the_hold_closed(binaries, tapes, tmp_path, market):
    hold = blocking(tmp_path)
    config = write_config(
        tmp_path / "ppo.json",
        native_policy_runs.ppo_config(diagnostic_stage(), job("ppo_clip_full_kl", "native_ppo")),
    )
    output = tmp_path / "sources"
    arguments = ["--config", config, "--output", output, *sources(tapes[market])]
    result = run(binaries["native_ppo"], hold, *arguments, "--check-sources")
    assert result.returncode == 0, result.stderr
    # Ni checkpoints, ni selección, ni informe del ajuste: solo el registro de las fuentes.
    assert sorted(path.name for path in output.iterdir()) == [".run.lock", "sources.json"]
    report = sealed(output / "sources.json")
    assert report["kind"] == "native_policy_source_check" and report["status"] == "completed"
    assert (report["transitions"], report["optimizer_steps"]) == (0, 0)
    # El binario registra que el cerrojo de solo evaluación estaba activo.
    assert report["evaluation_only"] is True
    assert report["final_test_opened"] is False
    assert len(report["initial_parameter_fingerprint"]) == 64
    # Es la identidad con la que el ajuste escribiría identity.json.
    identity = report["identity"]
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
    # La salida no se reutiliza y la comprobación no usa la GPU.
    again = run(binaries["native_ppo"], hold, *arguments, "--check-sources")
    assert again.returncode == 1 and "La salida debe ser nueva" in again.stderr
    gpu = arguments[:3] + [tmp_path / "gpu", *arguments[4:]]
    result = subprocess.run(
        [str(binaries["native_ppo"]), *map(str, gpu), "--check-sources", "--device", "cuda:0"],
        env=dict(os.environ, CUDA_VISIBLE_DEVICES="-1", **{HOLD_ENV: str(hold)}),
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert result.returncode == 1 and "se ejecuta en CPU" in result.stderr
    assert not (tmp_path / "gpu").exists()
    # Los dos modos solo leen cintas reconstruidas: una configuración sintética se rechaza.
    synthetic = ROOT / "configs/simulation/native-ppo-diagnostic.json"
    for mode in (sources(tapes[market]), ["--audit-tape", tapes[market]["evaluation"][0][0]]):
        flag = "--check-sources" if "--train-tape" in mode else "--initial-evaluation"
        output = tmp_path / flag.strip("-")
        result = run(
            binaries["native_ppo"], hold, "--config", synthetic, "--output", output, *mode, flag
        )
        assert result.returncode == 1 and "solo admiten cintas reconstruidas" in result.stderr
        assert not output.exists()


@pytest.mark.native("MARS_TITAN_PPO_EXECUTABLE", "MARS_TITAN_KLPO_EXECUTABLE")
@lowers_learning_hold(f"{THIS}::test_ppo_pauses_on_real_tapes_before_its_first_update")
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
    # El ajuste publica la identidad que ya comprueba `--check-sources`.
    check = run(
        binaries["native_ppo"],
        hold,
        *arguments[:3],
        tmp_path / "sources",
        *arguments[4:],
        "--check-sources",
    )
    assert check.returncode == 0, check.stderr
    identity = read(output / "identity.json")["identity"]
    assert identity == sealed(tmp_path / "sources" / "sources.json")["identity"]


def double_dqn_config(path, stage=None):
    stage = diagnostic_stage(rollout_transitions=16) if stage is None else stage
    return write_config(path, native_policy_runs.ppo_config(stage, job("double_dqn", "native_ppo")))


def initial_evaluation(binary, hold, config, output, tape, *extra):
    return run(
        binary,
        hold,
        "--config",
        config,
        "--output",
        output,
        "--audit-tape",
        tape,
        "--initial-evaluation",
        *extra,
    )


@pytest.mark.native("MARS_TITAN_PPO_EXECUTABLE", "MARS_TITAN_KLPO_EXECUTABLE")
@pytest.mark.parametrize("market", ["US", "CN"])
def test_initial_evaluation_pays_the_declared_costs_and_the_final_exit(
    binaries, tapes, tmp_path, market
):
    """Evaluación congelada de la política inicial de Double DQN con la protección cerrada.

    Son los pesos que tendría el ajuste antes de su primera actualización, generados en CPU con
    la semilla de la configuración. Ninguna métrica mide un modelo.
    """
    hold = blocking(tmp_path)
    config = double_dqn_config(tmp_path / "dqn.json")
    evaluated, evaluation_tape = tmp_path / "evaluation", tapes[market]["evaluation"][0]
    for extra in ([], ["--resume"]):
        result = initial_evaluation(
            binaries["native_ppo"], hold, config, evaluated, evaluation_tape[0], *extra
        )
        assert result.returncode == 0, result.stderr
    document = sealed(evaluated / "evaluation.json")
    assert document["status"] == "completed" and document["final_test_opened"] is False
    identity = document["identity"]
    assert identity["kind"] == "native_ppo_initial_evaluation"
    assert identity["weights"] == "initial_from_configuration_seed"
    assert identity["evaluation_only"] is True
    assert (identity["optimizer_steps"], identity["transitions"], identity["seed"]) == (0, 0, 42)
    assert identity["tapes"][0]["role"] == "evaluation" and identity["decisions"] is False
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
    # Una cinta de ajuste no se evalúa y la orden no admite fuentes de ajuste.
    wrong = initial_evaluation(
        binaries["native_ppo"], hold, config, tmp_path / "train", tapes[market]["train"][0][0]
    )
    assert wrong.returncode == 1 and "papel" in wrong.stderr
    mixed = initial_evaluation(
        binaries["native_ppo"],
        hold,
        config,
        tmp_path / "mixed",
        evaluation_tape[0],
        *sources(tapes[market]),
    )
    assert mixed.returncode == 1 and "--initial-evaluation solo admite" in mixed.stderr
    assert not (tmp_path / "train").exists() and not (tmp_path / "mixed").exists()


@pytest.mark.native("MARS_TITAN_PPO_EXECUTABLE", "MARS_TITAN_KLPO_EXECUTABLE")
@lowers_learning_hold(
    f"{THIS}::test_double_dqn_selects_and_evaluates_the_chosen_state_without_updates"
)
@pytest.mark.parametrize("market", ["US", "CN"])
def test_double_dqn_selects_and_evaluates_the_chosen_state_without_updates(
    binaries, tapes, tmp_path, learning_hold, market
):
    hold = learning_hold(True)
    config = double_dqn_config(tmp_path / "dqn.json")
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
    # Sin actualizaciones, la política elegida es la inicial y su evaluación coincide con la
    # de `--initial-evaluation`, que comprueba costes y liquidación con la protección cerrada.
    initial = initial_evaluation(
        binaries["native_ppo"], hold, config, tmp_path / "initial", evaluation_tape[0]
    )
    assert initial.returncode == 0, initial.stderr
    assert document["metrics"] == sealed(tmp_path / "initial" / "evaluation.json")["metrics"]
    # Una cinta de evaluación que no es posterior a la validación se rechaza.
    early = ["--audit-run", fit, "--audit-tape", tapes[market]["early_evaluation"][0][0]]
    result = run(
        binaries["native_ppo"], hold, "--config", config, "--output", tmp_path / "early", *early
    )
    assert result.returncode == 1 and "posterior a la selección" in result.stderr
    assert not (tmp_path / "early").exists()


def klpo_config(path, stage=None):
    stage = diagnostic_stage(environments=1) if stage is None else stage
    return write_config(
        path, native_policy_runs.klpo_config(stage, job("klpo_terminal", "native_klpo"))
    )


@pytest.mark.native("MARS_TITAN_PPO_EXECUTABLE", "MARS_TITAN_KLPO_EXECUTABLE")
def test_klpo_checks_sources_and_plans_complete_waves_with_the_hold_closed(
    binaries, tapes, tmp_path
):
    hold = blocking(tmp_path)
    config = klpo_config(tmp_path / "klpo.json")
    us = tapes["US"]
    output = tmp_path / "sources"
    arguments = ["--config", config, "--output", output, *sources(us, train=(1,))]
    result = run(binaries["native_klpo"], hold, *arguments, "--check-sources")
    assert result.returncode == 0, result.stderr
    assert sorted(path.name for path in output.iterdir()) == [".run.lock", "sources.json"]
    report = sealed(output / "sources.json")
    assert report["kind"] == "native_policy_source_check" and report["evaluation_only"] is True
    assert (report["transitions"], report["optimizer_steps"]) == (0, 0)
    assert len(report["initial_parameter_fingerprint"]) == 64
    # La identidad del ajuste salvo la del controlador, que solo existe al ajustar.
    identity = report["identity"]
    wave = len(us["train"][1][1]) - 1
    assert identity["kind"] == "native_klpo" and "controller_identity_sha256" not in identity
    assert identity["budget_rule"] == "complete_waves_within_declared_transitions"
    assert (identity["wave_transitions"], identity["planned_waves"]) == (wave, 32 // wave)
    assert identity["sources"]["train"][0]["role"] == "train"
    assert identity["sources"]["validation"][0]["role"] == "validation"
    # Las oleadas que publica son las del presupuesto de la etapa.
    assert native_policy_runs.klpo_waves((us["train"][1][1],), 1, 32) == (32 // wave, wave)
    # La comprobación tampoco usa la GPU en KLPO.
    gpu = ["--config", config, "--output", tmp_path / "gpu", *sources(us, train=(1,))]
    result = subprocess.run(
        [str(binaries["native_klpo"]), *map(str, gpu), "--check-sources", "--device", "cuda:0"],
        env=dict(os.environ, CUDA_VISIBLE_DEVICES="-1", **{HOLD_ENV: str(hold)}),
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert result.returncode == 1 and "se ejecuta en CPU" in result.stderr
    assert not (tmp_path / "gpu").exists()


@pytest.mark.native("MARS_TITAN_PPO_EXECUTABLE", "MARS_TITAN_KLPO_EXECUTABLE")
def test_klpo_initial_evaluation_runs_without_a_controller(binaries, tapes, tmp_path):
    hold = blocking(tmp_path)
    config = klpo_config(tmp_path / "klpo.json")
    evaluated = tmp_path / "evaluation"
    result = initial_evaluation(
        binaries["native_klpo"], hold, config, evaluated, tapes["US"]["evaluation"][0][0]
    )
    assert result.returncode == 0, result.stderr
    document = sealed(evaluated / "evaluation.json")
    identity = document["identity"]
    assert identity["kind"] == "native_klpo_initial_evaluation"
    assert identity["weights"] == "initial_from_configuration_seed"
    assert identity["evaluation_only"] is True
    assert (identity["optimizer_steps"], identity["transitions"]) == (0, 0)
    assert [row["cost_bps"] for row in document["metrics"]] == [0.0, 10.0, 25.0]
    # El actor inicial es el de la recogida: el mismo que fija la comprobación de fuentes.
    check = run(
        binaries["native_klpo"],
        hold,
        "--config",
        config,
        "--output",
        tmp_path / "sources",
        *sources(tapes["US"], train=(1,)),
        "--check-sources",
    )
    assert check.returncode == 0, check.stderr
    fingerprint = sealed(tmp_path / "sources" / "sources.json")["initial_parameter_fingerprint"]
    assert identity["parameter_fingerprint"] == fingerprint


@pytest.mark.native("MARS_TITAN_PPO_EXECUTABLE", "MARS_TITAN_KLPO_EXECUTABLE")
@lowers_learning_hold(f"{THIS}::test_klpo_collects_a_full_wave_and_pauses_before_update_ready")
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


@pytest.fixture(scope="module")
def ending(tmp_path_factory):
    """Cintas con el diseño de la etapa: columnas fuera del universo y bajas sin precio.

    GON termina su serie a mitad de noviembre y END el 6 de diciembre, sin salida
    acreditada. El ajuste de septiembre y octubre deja END, GON y OUT fuera de su universo,
    con todas sus columnas ausentes. La validación de noviembre ya no admite GON y la
    evaluación de diciembre lleva la baja de END, la única puntuación positiva, de modo que
    cualquier reparto por puntuación la compra. Dos cintas de noviembre con GON en el
    universo, una de ajuste y otra de validación, solo sirven para comprobar el rechazo.
    """
    root = tmp_path_factory.mktemp("ending-tapes")
    days = tape_days("US")
    extra = [
        Asset("END", base=30.0, end=days.index("2023-12-06")),
        Asset("GON", base=35.0, end=days.index("2023-11-15")),
        Asset("OUT", base=60.0),
    ]
    write_edition(root / "edition", {"US": [*EDITION["US"], *extra]})
    status = listing_status(root / "edition")
    layout = ["AAA", "BBB", "CCC", "END", "GON", "OUT"]
    base = ["AAA", "BBB", "CCC"]
    plan = {
        "train-0": ("train", -4, base),
        "train-1": ("train", -3, base),
        "validation": ("validation", -2, [*base, "END"]),
        "evaluation": ("evaluation", -1, [*base, "END"]),
        "gone-train": ("train", -2, [*base, "GON"]),
        "gone-validation": ("validation", -2, [*base, "GON"]),
        "late-validation": ("validation", -1, base),
    }
    result = {}
    for name, (role, index, universe) in plan.items():
        score = lambda k, i, universe=universe: 0.03 if universe[i] == "END" else 0.01  # noqa: E731
        window, values = monthly_window("US", index, universe, score=score)
        tape, _ = window_tapes.build_segment_tape(
            root / "edition",
            window,
            values,
            market="US",
            role=role,
            lag=0,
            listing_status=status,
            symbols=layout,
            universe=universe,
        )
        write_tape(tape, root / name)
        result[name] = root / name, tape
    return result


def test_stage_tapes_mask_columns_outside_the_universe_and_carry_unpriced_exits(ending):
    """Las cintas de la fixture tienen la forma que la etapa entrega a los binarios."""
    tapes_ = {name: tape for name, (_, tape) in ending.items()}
    assert tapes_["train-0"].identity["audit"]["outside_universe"] == ["US/END", "US/GON", "US/OUT"]
    assert np.isnan(tapes_["train-0"].prices[:, 3:, 3]).all()
    assert {a.kind for a in tapes_["train-0"].actions} <= {"dividend", "split"}
    for name in ("gone-train", "gone-validation", "evaluation"):
        kinds = [a.kind for a in tapes_[name].actions if a.kind.endswith("delisting")]
        assert kinds == ["unpriced_delisting"], name


@pytest.mark.native("MARS_TITAN_PPO_EXECUTABLE", "MARS_TITAN_KLPO_EXECUTABLE")
def test_source_checks_accept_masked_columns_and_reject_an_unpriced_exit(
    binaries, ending, tmp_path
):
    """Las fuentes del ajuste se validan con `--check-sources`, con la protección cerrada."""
    hold = blocking(tmp_path)
    train = [ending["train-0"][0], ending["train-1"][0]]
    ppo = double_dqn_config(tmp_path / "dqn.json")
    klpo = klpo_config(tmp_path / "klpo.json", diagnostic_stage(environments=1))
    message = "Las cintas de ajuste y validación no admiten bajas sin precio de salida"
    rejected = {
        "train": ([train[0], ending["gone-train"][0]], ending["late-validation"][0]),
        "validation": (train, ending["gone-validation"][0]),
    }
    # KLPO con un entorno admite una sola cinta de ajuste: recibe la última de cada caso.
    for engine, config, kept in (("native_ppo", ppo, 0), ("native_klpo", klpo, -1)):
        for role, (train_paths, validation) in rejected.items():
            output = tmp_path / f"{engine}-{role}"
            arguments = ["--config", config, "--output", output, "--validation-tape", validation]
            for path in train_paths[kept:]:
                arguments += ["--train-tape", path]
            result = run(binaries[engine], hold, *arguments, "--check-sources")
            assert result.returncode == 1 and message in result.stderr, (engine, role)
            assert not output.exists()
    # Las columnas ausentes del diseño no impiden leer las fuentes del ajuste.
    output = tmp_path / "sources"
    arguments = ["--config", ppo, "--output", output, "--validation-tape", ending["validation"][0]]
    arguments += ["--train-tape", train[0], "--train-tape", train[1]]
    result = run(binaries["native_ppo"], hold, *arguments, "--check-sources")
    assert result.returncode == 0, result.stderr
    assert sealed(output / "sources.json")["status"] == "completed"


@pytest.mark.native("MARS_TITAN_PPO_EXECUTABLE", "MARS_TITAN_KLPO_EXECUTABLE")
def test_the_initial_evaluation_ends_an_episode_at_an_unpriced_exit(binaries, ending, tmp_path):
    """La evaluación congelada admite la baja sin precio y publica el episodio como fallido."""
    hold = blocking(tmp_path)
    config = double_dqn_config(tmp_path / "dqn.json")
    folder, tape = ending["evaluation"]
    output = tmp_path / "evaluation"
    result = initial_evaluation(binaries["native_ppo"], hold, config, output, folder)
    assert result.returncode == 0, result.stderr
    document = sealed(output / "evaluation.json")
    assert document["identity"]["optimizer_steps"] == 0
    at = tape.delisted_at["US/END"]
    for row in document["metrics"]:
        assert (row["status"], row["reason"], row["steps"]) in (
            ("completed", None, len(tape) - 1),
            ("failed", "unpriced_exit", at),
        ), row
        nav = row["equity"]["nav"]
        assert len(nav) == row["steps"] + 1 and all(value is not None for value in nav[:-1])
        assert (nav[-1] is None) is (row["status"] == "failed")
        assert (row["net_return"] is None) is (row["status"] == "failed")
    # Con esta semilla la política inicial compra END antes de su baja, así que la prueba
    # recorre el episodio truncado y no solo la admisión de la cinta.
    assert any(row["status"] == "failed" for row in document["metrics"])


# El ajuste de Double DQN sobre las columnas ausentes del diseño necesita la protección abierta.
# Mientras rija el bloqueo se omite y, después, usa la protección real sin rebajarla. Las
# fuentes y la evaluación de la baja sin precio se comprueban arriba con la protección cerrada.
@pytest.mark.native("MARS_TITAN_PPO_EXECUTABLE", "MARS_TITAN_KLPO_EXECUTABLE")
@pytest.mark.skipif(
    learning_blocked(), reason="Lanza un ajuste nativo y espera a que se levante el bloqueo"
)
def test_double_dqn_fits_over_masked_columns_without_updates(binaries, ending, tmp_path):
    fit = tmp_path / "fit"
    arguments = ["--config", double_dqn_config(tmp_path / "dqn.json"), "--output", fit]
    arguments += ["--validation-tape", ending["validation"][0]]
    arguments += ["--train-tape", ending["train-0"][0], "--train-tape", ending["train-1"][0]]
    result = run(binaries["native_ppo"], hold_path(), *arguments)
    assert result.returncode == 0, result.stderr
    report = read(fit / "run.json")
    assert (report["status"], report["transitions"], report["optimizer_steps"]) == (
        "completed",
        32,
        0,
    )


@pytest.mark.native("MARS_TITAN_PPO_EXECUTABLE", "MARS_TITAN_KLPO_EXECUTABLE")
def test_klpo_rejects_a_budget_below_one_complete_wave(binaries, tapes, tmp_path):
    hold = blocking(tmp_path)
    us = tapes["US"]
    wave = len(us["train"][1][1]) - 1
    stage = diagnostic_stage(environments=1, transitions=wave - 1, evaluation_transitions=wave - 1)
    config = klpo_config(tmp_path / "klpo.json", stage)
    output = tmp_path / "out"
    arguments = ["--config", config, "--output", output, *sources(us, train=(1,))]
    # La comprobación de fuentes lee el presupuesto con la misma función que el ajuste.
    result = run(binaries["native_klpo"], hold, *arguments, "--check-sources")
    assert result.returncode == 1 and "ni una oleada completa" in result.stderr
    assert not output.exists()


def group_config(path, stage, objective="grpo_outcome_v1"):
    """Configuración de una variante de grupo con el mismo esquema que KLPO."""
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


@pytest.mark.native("MARS_TITAN_PPO_EXECUTABLE", "MARS_TITAN_KLPO_EXECUTABLE")
def test_group_objectives_reject_lonely_lanes_and_unknown_identities(binaries, tapes, tmp_path):
    hold = blocking(tmp_path)
    us = tapes["US"]
    # Un solo carril por cinta no deja línea base y se rechaza antes de recoger.
    config = group_config(tmp_path / "group.json", diagnostic_stage(environments=1))
    arguments = ["--config", config, "--output", tmp_path / "lonely", *sources(us, train=(1,))]
    result = run(binaries["native_klpo"], hold, *arguments, "--check-sources")
    assert result.returncode == 1 and "dos carriles por cinta" in result.stderr
    stage = diagnostic_stage(environments=2)
    unknown = group_config(tmp_path / "unknown.json", stage, objective="grpo_plus_plus")
    arguments = ["--config", unknown, "--output", tmp_path / "unknown", *sources(us, train=(1,))]
    result = run(binaries["native_klpo"], hold, *arguments, "--check-sources")
    assert result.returncode == 1 and "objetivo de grupo" in result.stderr
    assert not (tmp_path / "lonely").exists() and not (tmp_path / "unknown").exists()


def blocking(tmp_path):
    path = tmp_path / "hold.json"
    path.write_text(json.dumps({"training_allowed": False}))
    return path


@pytest.mark.native("MARS_TITAN_PPO_EXECUTABLE", "MARS_TITAN_KLPO_EXECUTABLE")
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
    # Con la misma protección, los dos modos sin aprendizaje terminan y no la modifican.
    before = hold.read_bytes()
    admitted = [
        ("native_ppo", ppo, [*sources(us), "--check-sources"]),
        ("native_ppo", ppo, ["--audit-tape", us["evaluation"][0][0], "--initial-evaluation"]),
        ("native_klpo", klpo, [*sources(us, train=(1,)), "--check-sources"]),
        ("native_klpo", klpo, ["--audit-tape", us["evaluation"][0][0], "--initial-evaluation"]),
    ]
    for number, (engine, config, arguments) in enumerate(admitted):
        output = tmp_path / f"admitted-{number}"
        result = run(binaries[engine], hold, "--config", config, "--output", output, *arguments)
        assert result.returncode == 0, result.stderr
        assert "Bloqueo" not in result.stderr
    assert hold.read_bytes() == before


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


@pytest.mark.native("MARS_TITAN_PPO_EXECUTABLE", "MARS_TITAN_KLPO_EXECUTABLE")
def test_reconstructed_sources_keep_role_order_basis_and_rules(binaries, tapes, tmp_path):
    hold = blocking(tmp_path)
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
        # Los lectores son los del ajuste, que la comprobación recorre sin la protección.
        result = run(binaries["native_ppo"], hold, *arguments, "--check-sources")
        assert result.returncode == 1 and message in result.stderr, (name, result.stderr)
        assert not output.exists(), name


@pytest.mark.native("MARS_TITAN_PPO_EXECUTABLE", "MARS_TITAN_KLPO_EXECUTABLE")
def test_schema_four_rejects_masks_hmm_and_other_variants(binaries, tapes, tmp_path):
    hold = blocking(tmp_path)
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
            "--check-sources",
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


@pytest.mark.native("MARS_TITAN_PPO_EXECUTABLE", "MARS_TITAN_KLPO_EXECUTABLE")
@lowers_learning_hold(f"{THIS}::test_stage_executors_fit_carry_and_pause_through_the_launcher")
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
    universe = dict(rule=window_tapes.UNIVERSE_RULE, max_assets=8, ranking_sessions=20)
    policies = dict(environment=dict(dividend_payment_lag_sessions=0), universe=universe)
    edition = folder.parents[1] / "edition"
    stage_tapes = campaign_stage._Tapes(
        policies,
        lambda *_: (window, lambda: values),
        edition,
        json.loads((edition / "manifest.json").read_text())["edition_id"],
        tmp_path,
        "fixture",
        listing_status(edition),
    )
    job = dict(scope="CN", market="CN", predictor="fixture", anchor="fixture")
    assets = tuple(tape.assets)
    written, built, failure, _ = stage_tapes.tape(job, "validation", window.fold, assets, assets)
    assert failure is None and tuple(built.assets) == tuple(tape.assets)
    manifest = read(written / "manifest.json")
    assert manifest["schema_version"] == 2 and set(manifest["instruments"]) == set(tape.assets)
    assert manifest["instruments"] == read(folder / "manifest.json")["instruments"]


@pytest.mark.native("MARS_TITAN_PPO_EXECUTABLE", "MARS_TITAN_KLPO_EXECUTABLE")
def test_declared_costs_and_session_equity_reach_the_frozen_evaluation(binaries, tapes, tmp_path):
    """Costes y patrimonio por sesión de la evaluación inicial, con la protección cerrada.

    Cada episodio pasa por los mismos lectores que el ejecutor de la etapa
    (`native_policy_runs._record`) y que su confirmación (`campaign_stage._record`).
    """
    hold = blocking(tmp_path)
    config = double_dqn_config(tmp_path / "dqn.json")
    evaluation_tape = tapes["US"]["evaluation"][0]
    costs = [0, 5, 10, 20]
    declared = [item for cost in costs for item in ("--evaluation-cost", cost)]
    output = tmp_path / "evaluation"
    result = initial_evaluation(
        binaries["native_ppo"], hold, config, output, evaluation_tape[0], *declared
    )
    assert result.returncode == 0, result.stderr
    document = sealed(output / "evaluation.json")
    assert document["identity"]["cost_bps"] == [0.0, 5.0, 10.0, 20.0]
    manifest = sha256(evaluation_tape[0] / "manifest.json")
    records = [
        native_policy_runs._record(row, cost, manifest)
        for row, cost in zip(document["metrics"], costs, strict=True)
    ]
    capital = json.loads(config.read_text())["environment"]["capital"]
    tape = evaluation_tape[1]
    for record, cost in zip(records, costs, strict=True):
        campaign_stage._record(record, cost, None)
        equity = record["equity"]
        assert equity["basis"] == "close_valuation_from_log_rewards"
        assert equity["close_times"] == [int(t) for t in tape.close_times[: record["steps"] + 1]]
        assert equity["nav"][0] == capital and len(equity["nav"]) == record["steps"] + 1
        final = capital * (1 + record["net_return"])
        assert equity["nav"][-1] == pytest.approx(final, rel=1e-12)
    # Más coste nunca deja más patrimonio con las mismas decisiones deterministas.
    finals = [record["equity"]["nav"][-1] for record in records]
    assert finals == sorted(finals, reverse=True)
    # Una serie que no concilia con su retorno no se confirma.
    broken = copy.deepcopy(records[1])
    broken["equity"]["nav"][-1] *= 1.001
    with pytest.raises(ValueError, match="no concilia"):
        campaign_stage._record(broken, 5, None)
    missing = dict(records[0], equity=None)
    with pytest.raises(ValueError, match="patrimonio en cada cierre"):
        campaign_stage._record(missing, 0, None)
    # El binario rechaza costes sin orden creciente, también sin aprender.
    for wrong in ([10, 5], [5, 5], [-1], [2000]):
        arguments = [item for cost in wrong for item in ("--evaluation-cost", cost)]
        result = initial_evaluation(
            binaries["native_ppo"],
            hold,
            config,
            tmp_path / "rejected",
            evaluation_tape[0],
            *arguments,
        )
        assert result.returncode == 1 and "costes de evaluación" in result.stderr, result.stderr
        assert not (tmp_path / "rejected").exists()
    # El ajuste no admite costes de evaluación: lo rechaza al leer los argumentos.
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
    assert not (tmp_path / "training").exists()


@pytest.mark.native("MARS_TITAN_PPO_EXECUTABLE", "MARS_TITAN_KLPO_EXECUTABLE")
@lowers_learning_hold(f"{THIS}::test_the_stage_confirms_the_declared_costs_of_a_fit")
def test_the_stage_confirms_the_declared_costs_of_a_fit(binaries, tapes, tmp_path, learning_hold):
    learning_hold(True)
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
    broken = copy.deepcopy(report)
    broken["evaluation"][1]["equity"]["nav"][-1] *= 1.001
    with pytest.raises(ValueError, match="no concilia"):
        campaign_stage.check_report(stage, fit_job, broken, us)


def launch(binary, hold, *arguments, gpu=False):
    """Lanzar el binario con `scripts/run_native_ppo.py`, que aplica la puerta de Python."""
    environment = dict(os.environ, **{HOLD_ENV: str(hold)})
    if not gpu:
        environment["CUDA_VISIBLE_DEVICES"] = "-1"
    return subprocess.run(
        [
            sys.executable,
            str(native_policy_runs.LAUNCHER),
            "--binary",
            str(binary),
            *map(str, arguments),
            *([] if gpu else ["--diagnostic"]),
        ],
        env=environment,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )


@pytest.mark.native("MARS_TITAN_PPO_EXECUTABLE", "MARS_TITAN_KLPO_EXECUTABLE")
@pytest.mark.parametrize("engine", ["native_ppo", "native_klpo"])
def test_the_launcher_admits_the_modes_without_learning_with_the_hold_closed(
    binaries, tapes, tmp_path, engine
):
    hold = blocking(tmp_path)
    config = (double_dqn_config if engine == "native_ppo" else klpo_config)(tmp_path / "c.json")
    us = tapes["US"]
    train = sources(us) if engine == "native_ppo" else sources(us, train=(1,))
    evaluation = ["--audit-tape", us["evaluation"][0][0], "--initial-evaluation", "--decisions"]
    result = launch(
        binaries[engine], hold, "--config", config, "--output", tmp_path / "eval", *evaluation
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "eval" / "decisions.json").is_file()
    result = launch(
        binaries[engine],
        hold,
        "--config",
        config,
        "--output",
        tmp_path / "src",
        *train,
        "--check-sources",
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "src" / "sources.json").is_file()
    # La puerta de ajuste sigue cerrada para el entrenamiento y la auditoría.
    for number, arguments in enumerate(
        (train, ["--audit-run", tmp_path / "src", "--audit-tape", us["evaluation"][0][0]])
    ):
        output = tmp_path / f"blocked-{number}"
        result = launch(binaries[engine], hold, "--config", config, "--output", output, *arguments)
        assert result.returncode != 0 and "Bloqueo de aprendizaje vigente" in result.stderr
        assert not output.exists()
    # Los dos modos no se combinan, y la comprobación de fuentes no admite la GPU.
    both = [*evaluation, "--check-sources"]
    result = launch(binaries[engine], hold, "--config", config, "--output", tmp_path / "x", *both)
    assert result.returncode == 2 and "--check-sources" in result.stderr
    result = launch(
        binaries[engine],
        hold,
        "--config",
        config,
        "--output",
        tmp_path / "y",
        *train,
        "--check-sources",
        gpu=True,
    )
    assert result.returncode == 2 and "--check-sources" in result.stderr
    assert not (tmp_path / "x").exists() and not (tmp_path / "y").exists()


@pytest.mark.native("MARS_TITAN_PPO_EXECUTABLE", "MARS_TITAN_KLPO_EXECUTABLE")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Necesita cuda:0 visible")
@pytest.mark.parametrize("engine", ["native_ppo", "native_klpo"])
def test_the_initial_evaluation_on_cuda_stays_within_the_tolerance_of_the_cpu(
    binaries, tapes, tmp_path, engine
):
    """El modo sin aprendizaje en `cuda:0`, con la admisión de GPU del lanzador.

    Los pesos iniciales salen del generador de CPU en los dos dispositivos, así que las dos
    evaluaciones usan los mismos parámetros. La comparación es la de #439.
    """
    hold = blocking(tmp_path)
    config = (double_dqn_config if engine == "native_ppo" else klpo_config)(tmp_path / "c.json")
    evaluation = ["--audit-tape", tapes["US"]["evaluation"][0][0], "--initial-evaluation"]
    for name, gpu in (("cpu", False), ("cuda", True)):
        result = launch(
            binaries[engine],
            hold,
            "--config",
            config,
            "--output",
            tmp_path / name,
            *evaluation,
            "--decisions",
            gpu=gpu,
        )
        assert result.returncode == 0, result.stderr
    reference, other = (device_parity.read_run(tmp_path / name) for name in ("cuda", "cpu"))
    assert (reference[1]["device"], other[1]["device"]) == ("cuda:0", "cpu")
    summary = device_parity.compare_runs(reference, other)
    # Una divergencia solo puede ser un empate cercano y detiene la comparación de su episodio.
    assert summary["compared_decisions"] > 0
    assert all(item["reference_margin"] <= 2 * item["allowed"] for item in summary["divergences"])
    assert summary["episodes"] == 3 and reference[0]["identity"]["optimizer_steps"] == 0


def test_klpo_budget_counts_complete_waves_over_cycled_train_tapes(tapes):
    train = tuple(tape for _, tape in tapes["US"]["train"])
    lengths = [len(tape) - 1 for tape in train]
    waves, wave = native_policy_runs.klpo_waves(train, 3, 1000)
    assert wave == lengths[0] + lengths[1] + lengths[0]
    assert waves == 1000 // wave
    assert native_policy_runs.klpo_waves(train, 2, sum(lengths) - 1)[0] == 0
