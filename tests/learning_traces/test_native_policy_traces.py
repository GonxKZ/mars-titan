"""Trazas de aprendizaje de la etapa RL nativa con el contrato de Python (#448).

Los binarios se lanzan en diagnóstico CPU sobre las cintas reconstruidas de la fixture y
ninguna prueba ejecuta un paso de optimizador: Double DQN no actualiza antes de 256 pasos de
entorno, PPO se pausa antes de completar su primer recorrido y KLPO antes de `update_ready`.
Por eso solo se registran validaciones. Las filas de las actualizaciones usan la misma
escritura, que comprueban las pruebas nativas de `learning_traces_tests`.
"""

from pathlib import Path

import pytest

from mars_titan.learning_traces.hooks import unit_config
from mars_titan.learning_traces.recorder import read_traces
from mars_titan.simulation import campaign_stage, native_policy_runs
from tests.learning_hold_lowering import lowers_learning_hold
from tests.learning_traces.parity import trace_rows
from tests.simulation.test_native_policy_tapes import (
    diagnostic_stage,
    job,
    read,
    run,
    sealed,
    sources,
    write_config,
)

TRACED = ["--learning-trace-every", 1, "--learning-trace-max-bytes", 16 * 1024**2]
# Prefijo de las pruebas de este archivo en el inventario de la protección (#546).
KEY = "tests/learning_traces/test_native_policy_traces.py::"
# Campos deterministas del informe. La huella del archivo de la política cambia en cada
# ejecución aunque los pesos sean los mismos, porque PyTorch guarda un identificador de
# serialización aleatorio, así que la paridad se comprueba con la selección y sus métricas.
REPORT_FIELDS = ("status", "transitions", "optimizer_steps", "evaluations", "best")
VALIDATION = {
    "rl.selection_score",
    "rl.mean_log_growth",
    "rl.mean_liquidated_log_growth",
    "rl.ruin_count",
    "rl.episodes",
    "rl.mean_max_drawdown",
    "rl.improved",
    "rl.stale_evaluations",
}


@lowers_learning_hold(KEY + "test_double_dqn_traces_keep_the_run_and_record_each_validation")
@pytest.mark.native("MARS_TITAN_PPO_EXECUTABLE", "MARS_TITAN_KLPO_EXECUTABLE")
def test_double_dqn_traces_keep_the_run_and_record_each_validation(
    binaries, tapes, tmp_path, learning_hold
):
    hold = learning_hold(True)
    stage = diagnostic_stage(rollout_transitions=16, evaluation_transitions=16)
    config = write_config(
        tmp_path / "dqn.json", native_policy_runs.ppo_config(stage, job("double_dqn", "native_ppo"))
    )
    reports = {}
    for name, extra in (("plain", []), ("traced", TRACED)):
        output = tmp_path / name
        arguments = ["--config", config, "--output", output, *sources(tapes["US"]), *extra]
        result = run(binaries["native_ppo"], hold, *arguments)
        assert result.returncode == 0, result.stderr
        reports[name] = read(output / "run.json")
    plain, traced = reports["plain"], reports["traced"]
    assert {key: plain[key] for key in REPORT_FIELDS} == {key: traced[key] for key in REPORT_FIELDS}
    assert traced["optimizer_steps"] == 0 and not (tmp_path / "plain/traces").exists()
    manifest, rows = trace_rows(tmp_path / "traced/traces")
    assert manifest["step_unit"] == "transition" and manifest["identity"]["family"] == "double_dqn"
    assert manifest["identity"]["run_identity_sha256"] == traced["identity_sha256"]
    assert manifest["config"] == dict(every=1, max_bytes=16 * 1024**2, enabled=True)
    assert {row["phase"] for row in rows} == {"validation"}
    assert {row["metric"] for row in rows} == VALIDATION
    assert sorted({row["step"] for row in rows}) == [0, 16, 32]
    best = {row["step"]: row["value"] for row in rows if row["metric"] == "rl.mean_log_growth"}
    assert best[traced["best"]["transitions"]] == traced["best"]["mean_log_growth"]
    # Una parte por checkpoint confirmado con validación, cada una con su marca.
    native = manifest["native"]["parts"]
    assert [part["commit"] for part in native] == [[0, 1], [16, 2], [32, 3]]


@lowers_learning_hold(KEY + "test_ppo_traces_are_not_duplicated_by_a_pause_and_resume")
@pytest.mark.native("MARS_TITAN_PPO_EXECUTABLE", "MARS_TITAN_KLPO_EXECUTABLE")
def test_ppo_traces_are_not_duplicated_by_a_pause_and_resume(
    binaries, tapes, tmp_path, learning_hold
):
    hold = learning_hold(True)
    config = write_config(
        tmp_path / "ppo.json",
        native_policy_runs.ppo_config(diagnostic_stage(), job("ppo_clip_full_kl", "native_ppo")),
    )
    output = tmp_path / "out"
    arguments = ["--config", config, "--output", output, *sources(tapes["US"]), *TRACED]
    result = run(binaries["native_ppo"], hold, *arguments, "--stop-after", 8)
    assert result.returncode == 2, result.stderr
    first = read_traces(output / "traces")[1].to_pylist()
    # Restos de una escritura cortada después del último checkpoint confirmado.
    (output / "traces/parts/part-00000002.parquet").write_bytes(b"sin confirmar")
    result = run(binaries["native_ppo"], hold, *arguments, "--stop-after", 16, "--resume")
    assert result.returncode == 2, result.stderr
    report = read(output / "run.json")
    assert (report["transitions"], report["optimizer_steps"], report["evaluations"]) == (16, 0, 1)
    manifest, rows = trace_rows(output / "traces")
    assert rows == first and {row["step"] for row in rows} == {0}
    assert manifest["identity"]["family"] == "ppo_clip_full_kl_v1"
    assert not (output / "traces/parts/part-00000002.parquet").exists()


@lowers_learning_hold(KEY + "test_klpo_traces_record_the_initial_validation_once")
@pytest.mark.native("MARS_TITAN_PPO_EXECUTABLE", "MARS_TITAN_KLPO_EXECUTABLE")
def test_klpo_traces_record_the_initial_validation_once(binaries, tapes, tmp_path, learning_hold):
    hold = learning_hold(True)
    stage = diagnostic_stage(environments=1)
    config = write_config(
        tmp_path / "klpo.json",
        native_policy_runs.klpo_config(stage, job("klpo_terminal", "native_klpo")),
    )
    output = tmp_path / "out"
    us = tapes["US"]
    arguments = ["--config", config, "--output", output, *sources(us, train=(1,)), *TRACED]
    wave = len(us["train"][1][1]) - 1
    result = run(binaries["native_klpo"], hold, *arguments, "--stop-after", 7)
    assert result.returncode == 2, result.stderr
    for _ in range(2):
        result = run(binaries["native_klpo"], hold, *arguments, "--stop-after", wave, "--resume")
        assert result.returncode == 2, result.stderr
    report = read(output / "run.json")
    assert (report["optimizer_steps"], report["consumed_waves"]) == (0, 0)
    manifest, rows = trace_rows(output / "traces")
    assert manifest["step_unit"] == "wave" and manifest["identity"]["family"] == "klpo_terminal"
    assert {(row["step"], row["phase"]) for row in rows} == {(0, "validation")}
    selection = sealed(output / "selection.json")
    score = {row["metric"]: row["value"] for row in rows}["rl.selection_score"]
    assert score == selection["best"]["score"]
    assert manifest["native"]["parts"][0]["commit"] == [0, 1]


@pytest.mark.native("MARS_TITAN_PPO_EXECUTABLE", "MARS_TITAN_KLPO_EXECUTABLE")
def test_the_frozen_evaluation_rejects_traces(binaries, tapes, tmp_path, learning_hold):
    hold = learning_hold(True)
    config = write_config(
        tmp_path / "ppo.json",
        native_policy_runs.ppo_config(diagnostic_stage(), job("ppo_clip_full_kl", "native_ppo")),
    )
    audit = ["--audit-run", tmp_path / "fit", "--audit-tape", tapes["US"]["evaluation"][0][0]]
    arguments = ["--config", config, "--output", tmp_path / "evaluation", *audit, *TRACED]
    result = run(binaries["native_ppo"], hold, *arguments)
    assert result.returncode == 1 and "trazas" in result.stderr
    assert not (tmp_path / "evaluation").exists()


def test_each_engine_takes_the_cadence_of_its_step_unit():
    traces = dict(every=50, max_bytes=2 * 1024**2, every_by_unit=dict(transition=4096))
    ppo = native_policy_runs.NativePolicyExecutor("native_ppo").trace_options(traces)
    klpo = native_policy_runs.NativePolicyExecutor("native_klpo").trace_options(traces)
    assert ppo == ["--learning-trace-every", "4096", "--learning-trace-max-bytes", "2097152"]
    assert klpo == ["--learning-trace-every", "50", "--learning-trace-max-bytes", "2097152"]
    assert native_policy_runs.NativePolicyExecutor("native_ppo").trace_options(None) == []
    assert unit_config(traces, "optimizer_step").every == 50
    with pytest.raises(ValueError, match="unidades"):
        unit_config(dict(traces, every_by_unit=dict(epoch=2)), "transition")


def test_declared_traces_require_binaries_that_write_them(tmp_path, learning_doubles):
    stage = Path("configs/simulation/historical-masked-rl-stage-a.json")
    everything = {name: dict(available=True, reason=None) for name in campaign_stage.CAPABILITIES}
    without = dict(everything, native_ppo_learning_traces=dict(available=False, reason="viejo"))
    traces = dict(every=50, max_bytes=2 * 1024**2)
    with pytest.raises(campaign_stage.MissingCapability, match="native_ppo_learning_traces"):
        campaign_stage.run_stage(
            stage,
            {},
            tmp_path / "c",
            tmp_path / "e",
            tmp_path / "out",
            listing_status=tmp_path / "listing.parquet",
            capabilities=without,
            traces=traces,
        )
    assert not (tmp_path / "out").exists()
    # Sin trazas declaradas, la capacidad no se exige y la etapa sigue hasta pedir las vistas.
    with pytest.raises(ValueError, match="vistas"):
        campaign_stage.run_stage(
            stage,
            {},
            tmp_path / "c",
            tmp_path / "e",
            tmp_path / "out",
            listing_status=tmp_path / "listing.parquet",
            capabilities=without,
        )
