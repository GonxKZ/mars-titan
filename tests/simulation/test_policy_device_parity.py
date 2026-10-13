"""Paridad CPU y CUDA de la evaluación congelada de las políticas, sin pasos de optimizador.

Las comparaciones usan registros fabricados con el formato que escriben los binarios, así
que no se lanza ninguna política. Las pruebas que lanzan binarios mantienen cerrada la
protección de aprendizaje y solo comprueban rechazos previos a leer cintas. La medida sobre
cintas reales espera a un modo de solo evaluación que la protección admita (#546).
"""

import copy
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path

import pytest

from mars_titan.simulation import device_parity, native_policy_runs
from mars_titan.training.learning_hold import HOLD_ENV
from tests.simulation.test_native_policy_tapes import (
    binaries,  # noqa: F401
    diagnostic_stage,
    job,
    sources,
    tapes,  # noqa: F401
    write_config,
)

ROOT = Path(__file__).resolve().parents[2]
STEPS = 4


def decisions(logits, *, start=0):
    return [
        dict(
            cursor=start + index, action=device_parity.first_argmax(row), mode="greedy", logits=row
        )
        for index, row in enumerate(logits)
    ]


def fake_run(device, logits, nav, *, fingerprint="f" * 64, status="completed"):
    """Evaluación y registro de un episodio con un coste, como los publica el binario."""
    identity = dict(
        kind="native_ppo_reconstructed_evaluation",
        tapes=[dict(manifest_sha256="m" * 64)],
        cost_bps=[0.0],
        optimizer_steps=0,
        seed=42,
        device=device,
        decisions=True,
    )
    evaluation = dict(
        kind=device_parity.EVALUATION_KIND,
        status="completed",
        identity=identity,
        identity_sha256=device,
        metrics=[
            dict(cost_bps=0.0, manifest_sha256="m" * 64, status=status, equity=dict(nav=list(nav)))
        ],
    )
    record = dict(
        kind=device_parity.DECISIONS_KIND,
        identity_sha256=device,
        device=device,
        parameter_fingerprint=fingerprint,
        outputs="q_values",
        costs=[
            dict(cost_bps=0.0, tapes=[dict(manifest_sha256="m" * 64, decisions=decisions(logits))])
        ],
    )
    return evaluation, record


LOGITS = [[0.1, 0.3, -0.2, 0.0, 0.05, 0.2] for _ in range(STEPS)]
NAV = [1.0e6, 1.001e6, 0.999e6, 1.002e6, 1.0e6]


def test_rounding_inside_the_tolerance_keeps_actions_and_equity():
    shifted = [[value + 3e-7 for value in row] for row in LOGITS]
    summary = device_parity.compare_runs(
        fake_run("cuda:0", LOGITS, NAV), fake_run("cpu", shifted, NAV)
    )
    assert summary["reference_device"] == "cuda:0" and summary["other_device"] == "cpu"
    assert summary["decisions"] == summary["compared_decisions"] == STEPS
    assert summary["divergences"] == [] and summary["identical_equity_episodes"] == 1
    assert summary["near_ties"] == 0 and summary["exact_logits"] == 0
    assert summary["max_abs_logit_difference"] == pytest.approx(3e-7, rel=1e-6)
    # La diferencia relativa se mide frente a la mayor salida, no frente al logit nulo.
    assert summary["max_relative_logit_difference"] == pytest.approx(1e-6, rel=1e-6)
    assert summary["minimum_reference_margin"] == pytest.approx(0.1)
    assert set(summary["margins_below"].values()) == {0}
    exact = device_parity.compare_runs(
        fake_run("cuda:0", LOGITS, NAV), fake_run("cpu", LOGITS, NAV)
    )
    assert exact["exact_logits"] == STEPS and exact["max_relative_logit_difference"] == 0


def test_a_near_tie_may_flip_the_action_and_its_capital_effect_is_published():
    tie = copy.deepcopy(LOGITS)
    tie[2][1], tie[2][5] = 0.3, 0.3 - 2e-6
    flipped = copy.deepcopy(tie)
    flipped[2][1], flipped[2][5] = 0.3 - 2e-6, 0.3 + 1e-6
    other_nav = [*NAV[:3], 1.0015e6, 0.9995e6]
    summary = device_parity.compare_runs(
        fake_run("cuda:0", tie, NAV), fake_run("cpu", flipped, other_nav)
    )
    (divergence,) = summary["divergences"]
    assert divergence["cursor"] == 2 and divergence["actions"] == [1, 5]
    assert divergence["reference_margin"] == pytest.approx(2e-6)
    assert divergence["other_margin"] == pytest.approx(3e-6)
    assert divergence["final_relative_difference"] == pytest.approx(5e-4, rel=1e-3)
    assert divergence["max_session_relative_difference"] == pytest.approx(5e-4, rel=1e-3)
    assert divergence["statuses"] == ["completed", "completed"]
    assert summary["compared_decisions"] == 3 and summary["near_ties"] == 1
    assert summary["minimum_reference_margin"] == pytest.approx(2e-6)
    assert summary["margins_below"] == {"1e-06": 0, "1e-05": 1, "0.0001": 1, "0.001": 1}
    assert summary["identical_equity_episodes"] == 0


def test_a_margin_up_to_twice_the_tolerance_counts_as_a_near_tie():
    allowed = device_parity.bound(LOGITS[0], device_parity.TOLERANCE)
    close = copy.deepcopy(LOGITS)
    close[1][5] = 0.3 - 1.5 * allowed
    summary = device_parity.compare_runs(
        fake_run("cuda:0", close, NAV), fake_run("cpu", close, NAV)
    )
    assert summary["near_ties"] == 1 and summary["divergences"] == []
    close[1][5] = 0.3 - 2.5 * allowed
    summary = device_parity.compare_runs(
        fake_run("cuda:0", close, NAV), fake_run("cpu", close, NAV)
    )
    assert summary["near_ties"] == 0


def test_the_thresholds_include_twice_the_tolerance_and_exclude_each_margin_limit():
    # Un margen igual al doble de la tolerancia todavía puede invertir la acción. La mayor
    # salida en valor absoluto es -0.3, así que la tolerancia no depende de las dos primeras.
    allowed = 1e-5 + 1e-4 * 0.3
    edge = [[2 * allowed, 0.0, -0.3, -0.1, -0.2, -0.25] for _ in range(STEPS)]
    assert device_parity.bound(edge[0], device_parity.TOLERANCE) == allowed
    assert device_parity.margin(edge[0]) == 2 * allowed
    summary = device_parity.compare_runs(fake_run("cuda:0", edge, NAV), fake_run("cpu", edge, NAV))
    assert summary["near_ties"] == STEPS
    # Cada umbral cuenta los márgenes estrictamente por debajo de él.
    limit = [[0.001, 0.0, -0.1, -0.2, -0.3, -0.4] for _ in range(STEPS)]
    assert device_parity.margin(limit[0]) == 1e-3
    summary = device_parity.compare_runs(
        fake_run("cuda:0", limit, NAV), fake_run("cpu", limit, NAV)
    )
    assert summary["margins_below"]["0.001"] == 0


def test_the_tolerance_grows_with_the_scale_of_the_logits():
    large = [[100 * value for value in row] for row in LOGITS]
    shifted = copy.deepcopy(large)
    shifted[0][0] += 1e-3
    summary = device_parity.compare_runs(
        fake_run("cuda:0", large, NAV), fake_run("cpu", shifted, NAV)
    )
    assert summary["max_abs_logit_difference"] == pytest.approx(1e-3, rel=1e-6)
    shifted[0][0] += 3e-3
    with pytest.raises(ValueError, match="fuera de la tolerancia"):
        device_parity.compare_runs(fake_run("cuda:0", large, NAV), fake_run("cpu", shifted, NAV))


def test_an_unvalued_equity_after_a_divergence_has_an_infinite_difference():
    tie = copy.deepcopy(LOGITS)
    tie[2][1], tie[2][5] = 0.3, 0.3 - 2e-6
    flipped = copy.deepcopy(tie)
    flipped[2][1], flipped[2][5] = 0.3 - 2e-6, 0.3 + 1e-6
    unvalued = [*NAV[:-1], None]
    (divergence,) = device_parity.compare_runs(
        fake_run("cuda:0", tie, unvalued), fake_run("cpu", flipped, NAV)
    )["divergences"]
    assert divergence["final_relative_difference"] == math.inf
    (divergence,) = device_parity.compare_runs(
        fake_run("cuda:0", tie, unvalued), fake_run("cpu", flipped, unvalued)
    )["divergences"]
    assert divergence["final_relative_difference"] == 0.0


def test_equity_before_the_first_different_action_must_match():
    tie = copy.deepcopy(LOGITS)
    tie[2][1], tie[2][5] = 0.3, 0.3 - 2e-6
    flipped = copy.deepcopy(tie)
    flipped[2][1], flipped[2][5] = 0.3 - 2e-6, 0.3 + 1e-6
    # La acción de la sesión 2 actúa desde la apertura de la 3: el cierre 2 ya debe coincidir.
    earlier = [*NAV[:2], 0.998e6, 1.0015e6, 0.9995e6]
    with pytest.raises(ValueError, match="antes de la primera acción distinta"):
        device_parity.compare_runs(fake_run("cuda:0", tie, NAV), fake_run("cpu", flipped, earlier))


def test_runs_with_the_same_identity_must_cover_the_same_episodes():
    reference = fake_run("cuda:0", LOGITS, NAV)
    other = fake_run("cpu", LOGITS, NAV)
    other[0]["metrics"][0]["manifest_sha256"] = "n" * 64
    other[1]["costs"][0]["tapes"][0]["manifest_sha256"] = "n" * 64
    with pytest.raises(ValueError, match="mismos episodios"):
        device_parity.compare_runs(reference, other)


def test_an_episode_without_decisions_publishes_no_margin():
    summary = device_parity.compare_runs(
        fake_run("cuda:0", [], NAV[:1]), fake_run("cpu", [], NAV[:1])
    )
    assert summary["compared_decisions"] == 0 and summary["minimum_reference_margin"] is None
    assert summary["identical_equity_episodes"] == 1


def test_differences_that_rounding_cannot_explain_stop_the_comparison():
    reference = fake_run("cuda:0", LOGITS, NAV)
    far = copy.deepcopy(LOGITS)
    far[1][0] += 1e-3
    with pytest.raises(ValueError, match="fuera de la tolerancia"):
        device_parity.compare_runs(reference, fake_run("cpu", far, NAV))
    # Una acción distinta con margen amplio tampoco se admite, aunque los logits sean iguales.
    run_ = fake_run("cpu", LOGITS, NAV)
    run_[1]["costs"][0]["tapes"][0]["decisions"][1]["action"] = 5
    with pytest.raises(ValueError, match="primera salida máxima"):
        device_parity.compare_runs(reference, run_)
    with pytest.raises(ValueError, match="patrimonio"):
        device_parity.compare_runs(reference, fake_run("cpu", LOGITS, [*NAV[:-1], 1.0]))
    with pytest.raises(ValueError, match="patrimonio"):
        device_parity.compare_runs(reference, fake_run("cpu", LOGITS, NAV, status="failed"))
    with pytest.raises(ValueError, match="patrimonio"):
        device_parity.compare_runs(reference, fake_run("cpu", LOGITS[:-1], NAV))
    with pytest.raises(ValueError, match="mismos pesos"):
        device_parity.compare_runs(reference, fake_run("cpu", LOGITS, NAV, fingerprint="0" * 64))
    with pytest.raises(ValueError, match="mismos pesos"):
        device_parity.compare_runs(reference, fake_run("cuda:0", LOGITS, NAV))
    for key, value in (("seed", 7), ("cost_bps", [10.0]), ("tapes", []), ("optimizer_steps", 1)):
        changed = fake_run("cpu", LOGITS, NAV)
        changed[0]["identity"][key] = value
        with pytest.raises(ValueError, match="mismos pesos"):
            device_parity.compare_runs(reference, changed)
    outputs = fake_run("cpu", LOGITS, NAV)
    outputs[1]["outputs"] = "logits"
    with pytest.raises(ValueError, match="mismos pesos"):
        device_parity.compare_runs(reference, outputs)
    # Tampoco se admite que las dos ejecuciones se salten la misma sesión.
    runs = fake_run("cuda:0", LOGITS, NAV), fake_run("cpu", LOGITS, NAV)
    for skipped in runs[1:], runs:
        for run_ in skipped:
            run_[1]["costs"][0]["tapes"][0]["decisions"][1]["cursor"] = 2
        with pytest.raises(ValueError, match="sesiones del episodio"):
            device_parity.compare_runs(*runs)
    reference = fake_run("cuda:0", LOGITS, NAV)
    missing = fake_run("cpu", LOGITS, NAV)
    missing[1]["costs"][0]["tapes"] = []
    with pytest.raises(ValueError, match="sin registro de decisiones"):
        device_parity.compare_runs(reference, missing)
    unknown = fake_run("cpu", LOGITS, NAV)
    unknown[1]["costs"][0]["tapes"][0]["manifest_sha256"] = "n" * 64
    with pytest.raises(ValueError, match="episodios sin evaluación"):
        device_parity.compare_runs(reference, unknown)
    # El calentamiento de la memoria fija la acción y queda fuera de esta comparación.
    warmup = fake_run("cpu", LOGITS, NAV)
    warmup[1]["costs"][0]["tapes"][0]["decisions"][0].update(action=1, mode="warmup")
    with pytest.raises(ValueError, match="argmax"):
        device_parity.compare_runs(reference, warmup)


def write_sealed(path, payload):
    """Sobre compacto como el del binario, con la huella del texto exacto del contenido."""
    text = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(text).hexdigest().encode()
    path.write_bytes(b'{"payload":' + text + b',"sha256":"' + digest + b'"}')


def test_a_record_is_read_only_with_the_seal_of_its_exact_bytes(tmp_path):
    evaluation, record = fake_run("cuda:0", LOGITS, NAV)
    evaluation.update(schema_version=1, identity_sha256="i" * 64)
    record.update(schema_version=1, identity_sha256="i" * 64)
    write_sealed(tmp_path / "evaluation.json", evaluation)
    write_sealed(tmp_path / "decisions.json", record)
    assert device_parity.read_run(tmp_path) == (evaluation, record)
    # El mismo contenido con otro formato ya no es el texto que se selló.
    path = tmp_path / "decisions.json"
    raw = path.read_bytes()
    path.write_bytes(raw.replace(b'"payload":{', b'"payload": {', 1))
    with pytest.raises(ValueError, match="sello"):
        device_parity.read_run(tmp_path)
    path.write_bytes(raw.replace(b"0.3", b"0.4", 1))
    with pytest.raises(ValueError, match="sello"):
        device_parity.read_run(tmp_path)
    path.write_bytes(raw)
    for change in (dict(status="paused"), dict(identity_sha256="j" * 64)):
        write_sealed(tmp_path / "evaluation.json", dict(evaluation, **change))
        with pytest.raises(ValueError, match="evaluación completa"):
            device_parity.read_run(tmp_path)
    write_sealed(tmp_path / "evaluation.json", evaluation)
    write_sealed(tmp_path / "decisions.json", dict(record, device="cpu"))
    with pytest.raises(ValueError, match="evaluación completa"):
        device_parity.read_run(tmp_path)
    write_sealed(tmp_path / "decisions.json", dict(record, kind=device_parity.EVALUATION_KIND))
    with pytest.raises(ValueError, match="native_policy_decisions"):
        device_parity.read_run(tmp_path)
    write_sealed(tmp_path / "decisions.json", dict(record, schema_version=2))
    with pytest.raises(ValueError, match="versión 1"):
        device_parity.read_run(tmp_path)
    # Una evaluación que no declara el registro no se lee con él.
    write_sealed(tmp_path / "decisions.json", record)
    undeclared = dict(evaluation, identity=dict(evaluation["identity"], decisions=False))
    write_sealed(tmp_path / "evaluation.json", undeclared)
    with pytest.raises(ValueError, match="evaluación completa"):
        device_parity.read_run(tmp_path)
    # Un campo añadido al sobre queda fuera del sello aunque la huella cubra su texto.
    write_sealed(tmp_path / "evaluation.json", evaluation)
    text = json.dumps(record, sort_keys=True, separators=(",", ":")).encode() + b',"extra":1'
    digest = hashlib.sha256(text).hexdigest().encode()
    path.write_bytes(b'{"payload":' + text + b',"sha256":"' + digest + b'"}')
    with pytest.raises(ValueError, match="sello"):
        device_parity.read_run(tmp_path)


def test_ties_follow_the_first_maximum_like_the_binary():
    assert device_parity.first_argmax([0.2, 0.5, 0.5, 0.1, 0.0, 0.0]) == 1
    assert device_parity.margin([0.2, 0.5, 0.5, 0.1, 0.0, 0.0]) == 0
    assert device_parity.bound([0.0] * 6, device_parity.TOLERANCE) == 1e-5


def direct(binary, hold, *arguments):
    """Lanzar un binario en el diagnóstico CPU con la protección cerrada, para sus rechazos."""
    environment = dict(os.environ, **{HOLD_ENV: str(hold)}, CUDA_VISIBLE_DEVICES="-1")
    return subprocess.run(
        [str(binary), *map(str, arguments), "--device", "cpu", "--diagnostic"],
        env=environment,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )


@pytest.mark.parametrize("engine", ["native_ppo", "native_klpo"])
def test_binaries_announce_the_record_and_refuse_it_outside_the_audit(
    binaries,  # noqa: F811
    tapes,  # noqa: F811
    tmp_path,
    learning_hold,
    engine,
):
    """Los dos binarios anuncian el registro y lo rechazan al leer los argumentos de un ajuste.

    La protección queda cerrada: el rechazo ocurre antes de comprobarla y no se lee ninguna
    cinta. El registro en sí se comprueba en CTest sobre `run_frozen_evaluation`.
    """
    native_policy_runs.probe_binary(engine, device_parity.DECISION_LOGITS)
    stage = diagnostic_stage(environments=1)
    arm = (
        job("double_dqn", "native_ppo") if engine == "native_ppo" else job("klpo_terminal", engine)
    )
    document = (
        native_policy_runs.ppo_config(stage, arm)
        if engine == "native_ppo"
        else native_policy_runs.klpo_config(stage, arm)
    )
    config = write_config(tmp_path / "config.json", document)
    output = tmp_path / "out"
    us = tapes["US"]
    arguments = ["--config", config, "--output", output, *sources(us, train=(1,)), "--decisions"]
    rejected = direct(binaries[engine], learning_hold(False), *arguments)
    assert rejected.returncode == 1
    assert "--decisions solo se admite con --audit-run" in rejected.stderr
    assert not output.exists()


def test_a_synthetic_audit_rejects_decisions_before_reading_anything(
    binaries,  # noqa: F811
    tmp_path,
    learning_hold,
):
    # La auditoría sintética ya publica su traza con probabilidades y no escribe este registro.
    config = json.loads((ROOT / "configs/simulation/native-ppo-diagnostic.json").read_text())
    config.update(
        schema_version=2,
        environments=16,
        evaluation_transitions=32,
        agent=dict(variant="ppo", trading_field=10, markov_fields=[], hmm_file=None),
    )
    config["training"].update(rollout_transitions=32)
    config["hyperparameters"]["minibatch_size"] = 16
    path = write_config(tmp_path / "adaptive.json", config)
    output = tmp_path / "audit"
    arguments = ["--config", path, "--audit-run", tmp_path / "run", "--output", output]
    result = direct(
        binaries["native_ppo"],
        learning_hold(False),
        *arguments,
        "--audit-tape",
        tmp_path / "world",
        "--decisions",
    )
    assert result.returncode == 1
    assert "--decisions solo se admite sobre cintas reconstruidas" in result.stderr
    assert not output.exists()
