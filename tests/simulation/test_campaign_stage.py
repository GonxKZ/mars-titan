"""Etapa de políticas por ventana sobre una campaña base reducida, sin aprender.

La campaña base de `rl_stage_fixture` escribe recibos de ventana y predicciones reales de
un padre con pesos iniciales. Los brazos aprendidos se sustituyen por `ScriptedLearner`,
que no tiene redes ni optimizador, y las referencias usan la contabilidad Python. Las
pruebas admiten la etapa con la protección temporal permitida de `learning_doubles`.
"""

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mars_titan.data.cohort_files import read_manifest
from mars_titan.data.storage import atomic_json
from mars_titan.environments.walk_forward_receipt import read_window_receipt
from mars_titan.simulation import campaign_stage, native_policy_runs, window_tapes
from mars_titan.simulation.native_runtime import library_path
from mars_titan.simulation.storage import read_tape
from mars_titan.training.learning_hold import LearningHoldError
from tests.simulation import rl_stage_fixture as fixture

REPOSITORY = {
    v: Path(f"configs/simulation/historical-masked-rl-stage-{v.lower()}.json") for v in "AB"
}


@pytest.fixture(scope="module")
def base_a(tmp_path_factory):
    return fixture.base_campaign(tmp_path_factory.mktemp("rl-a"), "A")


@pytest.fixture(scope="module")
def base_b(tmp_path_factory):
    return fixture.base_campaign(tmp_path_factory.mktemp("rl-b"), "B")


def receipts(output):
    return {
        str(path.parent.relative_to(output / "jobs")): json.loads(path.read_text())
        for path in (output / "jobs").rglob("receipt.json")
    }


def published(base, window):
    path = base.output / "windows/US" / window / "gru/seed-42/US.json"
    return read_window_receipt(read_manifest(path, 1024**2)[0])


def us(day):
    return int(np.datetime64(day, "us").astype(np.int64))


def test_variant_a_runs_every_policy_on_the_same_causal_tapes(base_a, tmp_path, learning_doubles):
    learner = fixture.ScriptedLearner()
    summary = fixture.run(base_a, tmp_path / "stage", learner)
    assert summary["status"] == "completed"
    # KLPO y referencias sobre GRU y LSTM, y Double DQN solo sobre la GRU.
    assert summary["planned"] == summary["completed"] == dict(fit=18, reference=12)
    found = receipts(tmp_path / "stage")
    assert len(found) == 30
    stage = campaign_stage.load_stage(base_a.stage)
    folds = stage["campaign"]["comparison_config"]["resolved_scopes"]["US"]["windows"]
    by_window, universes = {}, {}
    for receipt in found.values():
        identity = receipt["identity"]
        # Todas las políticas y referencias de un predictor y una ventana ven las mismas
        # cintas, y todos los predictores de la ventana comparten el universo.
        key = (identity["window"], identity["predictor"])
        by_window.setdefault(key, set()).add(json.dumps(identity["tapes"]))
        universes.setdefault(identity["window"], set()).add(identity["tapes"]["universe_sha256"])
        assert len(receipt["evaluation"]) == 3 and receipt["final_test_opened"] is False
        if identity["kind"] == "fit":
            assert receipt["selection"] == dict(
                metric="ruin_count_then_mean_liquidated_log_growth", partition="validation"
            )
            assert receipt["policy"]["id"] == identity["id"]
            if identity["arm"] == "klpo_terminal":
                # Dos oleadas completas de dos episodios anuales, sin superar el presupuesto.
                assert receipt["waves"] == 2 and 0 < receipt["transitions"] <= 1024
            else:
                assert receipt["transitions"] == 1024 and receipt["waves"] is None
        else:
            assert receipt["policy"] is None and receipt["transitions"] == 0
    assert {key: len(values) for key, values in by_window.items()} == {
        (window, predictor): 1
        for window in ("fold-002", "fold-003")
        for predictor in ("gru", "lstm")
    }
    assert {window: len(values) for window, values in universes.items()} == {
        "fold-002": 1,
        "fold-003": 1,
    }
    # Cada cinta procede del recibo publicado por la campaña base para su ventana, y su
    # predictor terminó de ver etiquetas antes de cada decisión del tramo.
    roots = tmp_path / "stage/tapes/US/US/gru"
    expected = {
        "fold-002": ["train-fold-000", "validation-fold-001", "evaluation-fold-002"],
        "fold-003": ["train-fold-001", "validation-fold-002", "evaluation-fold-003"],
    }
    for anchor, names in expected.items():
        assert sorted(p.name for p in (roots / anchor).iterdir()) == sorted(names)
        ends = []
        for name in names:
            tape = read_tape(roots / anchor / name)
            window = name.split("-", 1)[1]
            start, end = (us(day) for day in folds[window]["evaluation"])
            assert [s["receipt_sha256"] for s in tape.identity["audit"]["walk_forward"]] == [
                published(base_a, window).sha256
            ]
            assert start <= tape.close_times[0] and tape.close_times[-1] < end <= us("2024-01-01")
            assert max(tape.identity["audit"]["prediction_fit_ends"]) < tape.close_times[0]
            ends.append((start, end))
        assert ends[0][1] <= ends[1][0] and ends[1][1] <= ends[2][0]
    metrics = summary["metrics"]
    assert list(metrics) == ["gru", "lstm"]
    assert list(metrics["gru"]) == ["klpo_terminal", "double_dqn", *fixture.REFERENCES]
    assert list(metrics["lstm"]) == ["klpo_terminal", *fixture.REFERENCES]
    for predictor in ("gru", "lstm"):
        assert metrics[predictor]["klpo_terminal"]["10"]["episodes"] == 6
        assert metrics[predictor]["cash"]["0"]["episodes"] == 2
        assert metrics[predictor]["cash"]["25"]["mean_liquidated_log_growth"] == 0.0
    assert all(
        entry["denominator"] == "completed"
        for arms in metrics.values()
        for arm in arms.values()
        for entry in arm.values()
    )


def test_variant_b_carries_the_anchor_policy_on_the_anchor_universe(
    base_b, tmp_path, learning_doubles
):
    learner = fixture.ScriptedLearner()
    summary = fixture.run(base_b, tmp_path / "stage", learner)
    assert summary["planned"] == summary["completed"] == dict(fit=9, carry=9, reference=12)
    found = receipts(tmp_path / "stage")
    for job_id, receipt in found.items():
        identity = receipt["identity"]
        if identity["kind"] != "carry":
            continue
        anchor = found[identity["anchor_fit"]["job"]]
        assert identity["window"] == "fold-003" and identity["anchor"] == "fold-002"
        assert receipt["policy"] == anchor["policy"] and receipt["selection"] is None
        assert identity["tapes"]["train"] == anchor["identity"]["tapes"]["train"]
        assert identity["tapes"]["validation"] == anchor["identity"]["tapes"]["validation"]
        assert identity["tapes"]["evaluation"] != anchor["identity"]["tapes"]["evaluation"]
        assert job_id.endswith(f"carry-s{identity['seed']}")
    carries = [call for call in learner.calls if call["anchor"] is not None]
    assert len(carries) == 9 and all(call["tapes"].evaluation is not None for call in carries)
    # La evaluación trasladada se monta en el universo del ancla.
    assert (tmp_path / "stage/tapes/US/US/gru/fold-002/evaluation-fold-003").is_dir()
    assert not (tmp_path / "stage/universes/US/US/fold-003.json").exists()


@pytest.fixture
def lstm_without(monkeypatch):
    """La LSTM no emitió filas del mercado en las ventanas indicadas de la campaña base."""

    def select(*windows):
        real = campaign_stage.campaign_source

        def source(base, output, seed):
            read = real(base, output, seed)

            def missing(scope, market, window, predictor):
                receipt, values = read(scope, market, window, predictor)
                return receipt, None if predictor == "lstm" and window in windows else values

            return missing

        monkeypatch.setattr(campaign_stage, "campaign_source", source)

    return select


def outcomes(found, predictor, arm):
    return {
        (r["identity"]["window"], r["identity"]["kind"]): {e["reason"] for e in r["evaluation"]}
        for r in found.values()
        if r["identity"]["predictor"] == predictor and r["identity"]["arm"] == arm
    }


@pytest.mark.parametrize("variant", "AB")
def test_a_predictor_without_training_predictions_fails_its_policies_without_running_them(
    variant, base_a, base_b, tmp_path, learning_doubles, lstm_without
):
    base = dict(A=base_a, B=base_b)[variant]
    # La LSTM no predijo el primer año de ajuste de la ventana de 2022.
    lstm_without("fold-000")
    learner = fixture.ScriptedLearner()
    summary = fixture.run(base, tmp_path / "stage", learner)
    assert summary["status"] == "completed"
    reason = window_tapes.NO_PREDICTIONS
    found = receipts(tmp_path / "stage")
    klpo = outcomes(found, "lstm", "klpo_terminal")
    if variant == "A":
        # La política de 2023 se ajusta con 2021 y valida con 2022, que sí tienen predicciones.
        assert klpo == {("fold-002", "fit"): {reason}, ("fold-003", "fit"): {None}}
        called = {call["id"] for call in learner.calls if "/lstm/" in call["id"]}
        assert called == {f"US/US/fold-003/lstm/klpo_terminal/fit-s{s}" for s in (42, 43, 44)}
    else:
        # En B la ventana de 2023 traslada la política de 2022, que no se pudo ajustar.
        assert klpo == {("fold-002", "fit"): {reason}, ("fold-003", "carry"): {reason}}
        assert not any("/lstm/" in call["id"] for call in learner.calls)
    for receipt in found.values():
        if reason in {e["reason"] for e in receipt["evaluation"]}:
            assert receipt["policy"] is None and receipt["transitions"] == 0
            assert receipt["selection"] is None
    # Las referencias solo usan la evaluación y la GRU no cambia.
    assert all(value == {None} for value in outcomes(found, "lstm", "cash").values())
    assert all(value == {None} for value in outcomes(found, "gru", "klpo_terminal").values())
    failed = summary["metrics"]["lstm"]["klpo_terminal"]["10"]
    assert failed["failure_reasons"] == {reason: 3 if variant == "A" else 6}


def test_a_predictor_without_evaluation_predictions_fails_those_episodes(
    base_a, tmp_path, learning_doubles, lstm_without
):
    lstm_without("fold-003")
    learner = fixture.ScriptedLearner()
    assert fixture.run(base_a, tmp_path / "stage", learner)["status"] == "completed"
    reason = window_tapes.NO_PREDICTIONS
    found = receipts(tmp_path / "stage")
    # La política de 2023 se ajusta con datos completos y solo falla su evaluación.
    assert outcomes(found, "lstm", "klpo_terminal")[("fold-003", "fit")] == {reason}
    assert outcomes(found, "lstm", "cash") == {
        ("fold-002", "reference"): {None},
        ("fold-003", "reference"): {reason},
    }
    called = [call for call in learner.calls if call["id"].startswith("US/US/fold-003/lstm/")]
    assert len(called) == 3 and all(call["tapes"].evaluation is None for call in called)


def test_a_paused_policy_resumes_in_its_folder_and_confirmed_jobs_are_not_repeated(
    base_a, tmp_path, learning_doubles
):
    first = "US/US/fold-002/gru/klpo_terminal/fit-s42"
    paused = fixture.ScriptedLearner(pause={first})
    summary = fixture.run(base_a, tmp_path / "stage", paused)
    assert summary["status"] == "paused" and summary["completed"] == {}
    assert [call["id"] for call in paused.calls] == [first]
    assert (tmp_path / f"stage/jobs/{first}/run/checkpoint.json").is_file()
    resumed = fixture.ScriptedLearner()
    assert fixture.run(base_a, tmp_path / "stage", resumed)["status"] == "completed"
    assert resumed.calls[0] == dict(resumed.calls[0], id=first, resume=True)
    assert not any(call["resume"] for call in resumed.calls[1:])
    again = fixture.ScriptedLearner()
    summary = fixture.run(base_a, tmp_path / "stage", again)
    assert summary["status"] == "completed" and again.calls == []


def test_a_stop_request_pauses_between_jobs(base_a, tmp_path, learning_doubles):
    stop = SimpleNamespace(requested=False)

    class Stopping(fixture.ScriptedLearner):
        def __call__(self, *args, **kwargs):
            report = super().__call__(*args, **kwargs)
            stop.requested = len(self.calls) == 2
            return report

    learner = Stopping()
    summary = fixture.run(base_a, tmp_path / "stage", learner, stop=stop)
    assert summary["status"] == "paused" and summary["completed"] == dict(fit=2)
    rest = fixture.ScriptedLearner()
    assert fixture.run(base_a, tmp_path / "stage", rest)["status"] == "completed"
    assert len(rest.calls) == 16 and not any(call["resume"] for call in rest.calls)


def test_the_learning_hold_stops_the_stage_before_outputs_and_between_jobs(
    base_a, tmp_path, learning_hold
):
    blocked = learning_hold(False)
    with pytest.raises(LearningHoldError, match="etapa de políticas"):
        fixture.run(base_a, tmp_path / "stage", fixture.ScriptedLearner())
    assert not (tmp_path / "stage").exists()
    allowed = learning_hold(True)

    class Reinstating(fixture.ScriptedLearner):
        def __call__(self, *args, **kwargs):
            report = super().__call__(*args, **kwargs)
            allowed.write_text(blocked.read_text())
            return report

    learner = Reinstating()
    with pytest.raises(LearningHoldError, match="fold-002/gru/klpo_terminal/fit-s43"):
        fixture.run(base_a, tmp_path / "stage", learner)
    assert len(learner.calls) == 1
    summary = json.loads((tmp_path / "stage/summary.json").read_text())
    assert summary["status"] == "blocked" and summary["completed"] == dict(fit=1)


@pytest.mark.parametrize("variant", "AB")
def test_missing_engine_capabilities_stop_the_stage_before_reading_sources(
    variant, tmp_path, learning_doubles
):
    everything = {name: dict(available=True, reason=None) for name in campaign_stage.CAPABILITIES}
    without_cn = dict(everything, native_cn_a_share_rules=dict(available=False, reason="CN"))
    with pytest.raises(campaign_stage.MissingCapability) as error:
        campaign_stage.run_stage(
            REPOSITORY[variant],
            {},
            tmp_path / "c",
            tmp_path / "e",
            tmp_path / "out",
            capabilities=without_cn,
        )
    message = str(error.value)
    # China depende de las reglas nativas en todos sus brazos y EE. UU. no las necesita.
    assert all(f"{engine}/CN" in message for engine in ("reference", "native_ppo", "native_klpo"))
    assert "/US" not in message and not (tmp_path / "out").exists()
    # Con los ejecutores actuales faltan además las piezas nativas de las políticas.
    with pytest.raises(campaign_stage.MissingCapability, match="native_klpo_financial_runner"):
        campaign_stage.run_stage(
            REPOSITORY[variant],
            {},
            tmp_path / "c",
            tmp_path / "e",
            tmp_path / "out",
            capabilities=dict(
                everything, native_klpo_financial_runner=dict(available=False, reason="x")
            ),
        )
    assert not (tmp_path / "out").exists()


def missing_binaries(monkeypatch, root):
    """Sustituir los binarios de política por rutas inexistentes."""
    for variable, name in native_policy_runs.BINARIES.values():
        monkeypatch.setenv(variable, str(root / name))


def test_capability_requirements_follow_engine_and_market(monkeypatch, tmp_path):
    missing_binaries(monkeypatch, tmp_path)
    executors = campaign_stage.EXECUTORS
    job = dict(engine="native_klpo", market="CN")
    assert campaign_stage.requirements(job, executors) == [
        "native_policy_reconstructed_tapes",
        "native_klpo_financial_runner",
        "native_klpo_equity_and_costs",
        "native_cn_a_share_rules",
    ]
    assert campaign_stage.requirements(dict(engine="native_ppo", market="US"), executors) == [
        "native_policy_reconstructed_tapes",
        "native_ppo_equity_and_costs",
    ]
    assert campaign_stage.requirements(dict(engine="reference", market="US"), executors) == [
        "native_accounting"
    ]
    probed = campaign_stage.probe_capabilities(library="/nonexistent/library.so")
    assert probed["native_accounting"]["available"] is False
    assert probed["native_cn_a_share_rules"]["available"] is False
    for name in (
        "native_policy_reconstructed_tapes",
        "native_klpo_financial_runner",
        "native_ppo_equity_and_costs",
        "native_klpo_equity_and_costs",
    ):
        assert probed[name]["available"] is False
        assert probed[name]["reason"].startswith(campaign_stage.CAPABILITIES[name]["pending"])
    for engine in ("native_ppo", "native_klpo"):
        run = executors[engine]["run"]
        assert isinstance(run, native_policy_runs.NativePolicyExecutor) and run.engine == engine


def test_the_hold_stops_the_stage_before_probing_or_launching_binaries(
    learning_hold, tmp_path, monkeypatch
):
    learning_hold(False)
    launched = []

    def forbidden(*args, **kwargs):
        launched.append(args)
        raise AssertionError("La etapa lanzó un proceso con el bloqueo vigente")

    monkeypatch.setattr(campaign_stage, "probe_capabilities", forbidden)
    monkeypatch.setattr(native_policy_runs.subprocess, "run", forbidden)
    with pytest.raises(LearningHoldError):
        campaign_stage.run_stage(
            REPOSITORY["A"], {}, tmp_path / "c", tmp_path / "e", tmp_path / "out"
        )
    assert launched == [] and not (tmp_path / "out").exists()


def test_policy_binaries_declare_their_capabilities_and_identity():
    for engine in native_policy_runs.BINARIES:
        if not native_policy_runs.binary_path(engine).is_file():
            pytest.skip("Faltan mars-titan-ppo o mars-titan-klpo compilados")
    probed = campaign_stage.probe_capabilities()
    for name, binary in (
        ("native_policy_reconstructed_tapes", "mars-titan-ppo"),
        ("native_klpo_financial_runner", "mars-titan-klpo"),
        ("native_ppo_equity_and_costs", "mars-titan-ppo"),
        ("native_klpo_equity_and_costs", "mars-titan-klpo"),
    ):
        assert probed[name]["available"] is True and probed[name]["reason"] is None
        identity = probed[name]["binary"]
        assert identity["binary"] == binary and len(identity["binary_sha256"]) == 64
        assert len(identity["native_build_sha256"]) == 64


def fake_binary(path, report):
    """Ejecutable que responde a `--capabilities` con el informe dado o falla sin él."""
    body = "exit 1" if report is None else f"printf '%s' '{json.dumps(report)}'"
    path.write_text(f"#!/bin/sh\n{body}\n")
    path.chmod(0o755)
    return path


def test_a_binary_without_the_capability_or_with_another_name_is_unavailable(monkeypatch, tmp_path):
    # Un binario anterior al esquema 4, otro ejecutable o uno que falla no cuentan.
    declared = ["native_policy_reconstructed_tapes"]
    older = dict(schema_version=1, binary="mars-titan-ppo", capabilities=[])
    cases = {
        "older": (older, "declara"),
        "renamed": (dict(older, binary="mars-titan-sim", capabilities=declared), "declara"),
        "failing": (None, "no informa"),
    }
    missing_binaries(monkeypatch, tmp_path)
    for name, (report, message) in cases.items():
        variable = native_policy_runs.BINARIES["native_ppo"][0]
        monkeypatch.setenv(variable, str(fake_binary(tmp_path / name, report)))
        with pytest.raises(ValueError, match=message):
            native_policy_runs.probe_binary("native_ppo", declared[0])
        probed = campaign_stage.probe_capabilities()["native_policy_reconstructed_tapes"]
        assert probed["available"] is False and message in probed["reason"]
        assert "binary" not in probed


def _pop(job, report):
    report["evaluation"].pop()
    return report


def _set(**values):
    def change(job, report):
        if job["kind"] == "fit":
            report.update(values)
        return report

    return change


def _selection(**values):
    def change(job, report):
        if job["kind"] == "fit":
            report["selection"] = dict(report["selection"], **values)
        return report

    return change


def _fewer(job, report):
    # PPO y Double DQN gastan todo el presupuesto y KLPO todas las oleadas que caben en él.
    if job["kind"] == "fit":
        if job["engine"] == "native_klpo":
            report["waves"] -= 1
        else:
            report["transitions"] = 32
    return report


def _beyond_waves(job, report):
    # KLPO declara sus oleadas completas, pero no puede gastar más de lo que caben.
    if job["kind"] == "fit" and job["engine"] == "native_klpo":
        report["transitions"] += 10**6
    return report


def _episode(**values):
    def change(job, report):
        report["evaluation"][0] = dict(report["evaluation"][0], **values)
        return report

    return change


def _reverse(job, report):
    report["evaluation"].reverse()
    return report


INVALID = {
    "drops_an_episode": (_pop, "cada coste"),
    "predictor_mae": (_selection(metric="session_mae"), "criterio de cartera"),
    "selects_on_evaluation": (_selection(partition="evaluation"), "criterio de cartera"),
    "fewer_transitions": (_fewer, "presupuesto"),
    "klpo_beyond_its_waves": (_beyond_waves, "presupuesto"),
    "policy_of_another_job": (
        _set(policy=dict(id="US/US/fold-002/gru/double_dqn/fit-s42", sha256="a" * 64)),
        "política elegida",
    ),
    "completed_without_metrics": (_episode(liquidated_net_return=None), "métricas finitas"),
    "failed_without_reason": (_episode(status="failed", reason=None), "motivo"),
    "costs_reordered": (_reverse, "coste 0"),
    "unknown_status": (_episode(status="skipped"), "coste 0"),
}


@pytest.mark.parametrize("name", sorted(INVALID))
def test_reports_that_hide_episodes_or_break_the_contract_are_rejected(
    base_a, tmp_path, learning_doubles, name
):
    change, message = INVALID[name]
    with pytest.raises(ValueError, match=message):
        fixture.run(base_a, tmp_path / "stage", fixture.ScriptedLearner(mutate=change))
    summary = json.loads((tmp_path / "stage/summary.json").read_text())
    assert summary["status"] == "failed" and receipts(tmp_path / "stage") == {}


def test_a_carry_must_evaluate_the_policy_of_its_anchor(base_b, tmp_path, learning_doubles):
    def other(job, report):
        if job["kind"] == "carry":
            report["policy"] = dict(report["policy"], sha256="c" * 64)
        return report

    with pytest.raises(ValueError, match="política de su ancla"):
        fixture.run(base_b, tmp_path / "stage", fixture.ScriptedLearner(mutate=other))


def test_a_reference_that_learns_is_rejected(base_a, tmp_path, learning_doubles):
    executors = fixture.executors(fixture.ScriptedLearner())
    honest = executors["reference"]["run"]

    def learning(*args, **kwargs):
        return dict(honest(*args, **kwargs), updates=1)

    executors["reference"] = dict(executors["reference"], run=learning)
    with pytest.raises(ValueError, match="no aprende"):
        campaign_stage.run_stage(
            base_a.stage,
            base_a.views,
            base_a.output,
            base_a.edition,
            tmp_path / "stage",
            executors=executors,
            capabilities={},
            stop=SimpleNamespace(requested=False),
        )


def copied(base, root):
    """Copia de la salida de la campaña base que una prueba puede alterar."""
    shutil.copytree(base.output, root / "campaign")
    return SimpleNamespace(**dict(vars(base), output=root / "campaign"))


def _swap_receipt(base):
    # El recibo de 2021 se sustituye por el de 2023, de un ajuste posterior. Su padre no es
    # el elegido para 2021 y la etapa lo rechaza antes de comprobar la ventana, que
    # `test_window_tapes` contrasta con una fuente que sí respeta el padre.
    folder = base.output / "windows/US"
    shutil.copy(folder / "fold-003/gru/seed-42/US.json", folder / "fold-001/gru/seed-42/US.json")


def _parent(base):
    path = base.output / "windows/US/fold-000/gru/seed-42/US.json"
    record = json.loads(path.read_text())
    record["parent"]["sha256"] = "d" * 64
    atomic_json(path, record)


def _predictions(base):
    receipt = json.loads(
        next((base.output / "jobs").rglob("search-gru-10/receipt.json")).read_text()
    )
    path = base.output / receipt["predictions"]["evaluation"]["path"]
    path.write_bytes(path.read_bytes() + b"\0")


TAMPERED = {
    "receipt_of_a_later_fit": (_swap_receipt, "fold-001/gru no corresponde"),
    "receipt_of_another_parent": (_parent, "predictor elegido"),
    "changed_predictions": (_predictions, "ha cambiado"),
}


@pytest.mark.parametrize("name", sorted(TAMPERED))
def test_tampered_base_campaign_is_rejected_before_any_policy_runs(
    base_a, tmp_path, learning_doubles, name
):
    change, message = TAMPERED[name]
    base = copied(base_a, tmp_path)
    change(base)
    learner = fixture.ScriptedLearner()
    with pytest.raises(ValueError, match=message):
        fixture.run(base, tmp_path / "stage", learner)
    assert learner.calls == []


def test_changed_tapes_are_rejected_on_resume(base_a, tmp_path, learning_doubles):
    fixture.run(base_a, tmp_path / "stage", fixture.ScriptedLearner())
    market = tmp_path / "stage/tapes/US/US/gru/fold-002/train-fold-000/market.parquet"
    market.write_bytes(market.read_bytes() + b"\0")
    with pytest.raises(ValueError, match="integridad"):
        fixture.run(base_a, tmp_path / "stage", fixture.ScriptedLearner())


def test_script_checks_the_policy_stage_and_runs_it_only_without_the_hold(
    learning_hold, tmp_path, capsys, monkeypatch
):
    import runpy

    missing_binaries(monkeypatch, tmp_path)
    script = runpy.run_path("scripts/run_masked_campaign.py", run_name="script")
    assert script["main"](["rl", "check", "--stage", str(REPOSITORY["B"])]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "checked" and report["variant"] == "B"
    assert (report["counts"]["training_jobs"], report["counts"]["carried_jobs"]) == (456, 912)
    assert set(report["missing_capabilities"]) >= {"native_klpo/US", "native_ppo/CN"}
    learning_hold(False)
    arguments = ["rl", "run", "--stage", str(REPOSITORY["A"]), "--views", f"US={tmp_path}"]
    arguments += ["--campaign-output", str(tmp_path / "campaign"), "--edition", str(tmp_path)]
    arguments += ["--output", str(tmp_path / "out")]
    with pytest.raises(LearningHoldError):
        script["main"](arguments)
    assert not (tmp_path / "out").exists()


@pytest.mark.skipif(library_path() is None, reason="Falta la biblioteca nativa de simulación")
def test_references_with_native_accounting_match_the_python_accounting(
    base_a, tmp_path, learning_doubles
):
    # El ejecutor de referencias del repositorio usa la contabilidad nativa en EE. UU.
    python = fixture.run(base_a, tmp_path / "python", fixture.ScriptedLearner())
    native = fixture.run(base_a, tmp_path / "native", fixture.ScriptedLearner(), backend="native")
    assert python["status"] == native["status"] == "completed"
    found = {name: receipts(tmp_path / name) for name in ("python", "native")}
    references = [job for job, r in found["python"].items() if r["identity"]["kind"] == "reference"]
    # Tres referencias en dos ventanas para cada uno de los dos predictores.
    assert len(references) == 12
    for job in references:
        for ours, theirs in zip(
            found["python"][job]["evaluation"], found["native"][job]["evaluation"], strict=True
        ):
            assert ours.keys() == theirs.keys() and ours["status"] == theirs["status"]
            for key, value in ours.items():
                if isinstance(value, float):
                    assert theirs[key] == pytest.approx(value, rel=1e-12, abs=1e-12), key
                else:
                    assert theirs[key] == value, key
