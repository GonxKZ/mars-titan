"""Retención v2 ventana a ventana sobre la campaña reducida, en CPU y sin ajustes.

La campaña y la ablación del fixture ya están confirmadas con pesos iniciales y ningún
paso de optimizador. Las fases de ajuste se sustituyen por órdenes que solo registran la
ventana. Los agregados, la regeneración y la liberación son los reales: se comprueba que el
informe final sale idéntico sin abrir filas, que solo se libera lo que se regenera bit a
bit y que el recorrido se reanuda sin repetir nada.
"""

import copy
import json
import shutil
from types import SimpleNamespace

import pytest

from mars_titan.data import prediction_files
from mars_titan.data.storage import atomic_json
from mars_titan.evaluation import walk_forward_comparison as comparison
from mars_titan.training import campaign_plan as plan
from mars_titan.training import masked_campaign as engine
from mars_titan.training import modality_ablation_stage as ablation
from mars_titan.training import rolling_retention as rolling
from mars_titan.training import rolling_storage
from mars_titan.training.learning_hold import HOLD_ENV
from tests.posttraining.campaign_fixture import CpuLease
from tests.training.test_modality_ablation_stage import RUNNING, base, on_cpu  # noqa: F401
from tests.training.test_prediction_regeneration import flipped, strict_numerics  # noqa: F401

DECLARATION = "configs/baselines/historical-masked-retention-v2.json"
VOLATILE = {"created_at_utc", "resources", "sources_sha256"}


def schedule(base, folder):  # noqa: F811
    campaign = plan.load_campaign(base.campaign)
    windows = campaign["comparison_config"]["resolved_scopes"]["US"]["windows"]
    path = folder / "schedule.json"
    atomic_json(
        path,
        dict(
            schema_version=1,
            campaign=campaign["sha256"],
            windows=[dict(window=w, scopes={"US": w}) for w in windows],
        ),
    )
    return path, campaign


class Calls:
    def __init__(self):
        self.calls = []

    def runners(self):
        return {
            phase: (
                lambda window, phase=phase: (
                    self.calls.append((phase, window)) or dict(status="completed")
                )
            )
            for phase in ("base", "ablation")
        }


@pytest.fixture(scope="module")
def walked(base, tmp_path_factory):  # noqa: F811
    """Recorrido completo con un trabajo cuya regeneración no repite un bit."""
    root = tmp_path_factory.mktemp("rolling")
    # Copia propia de la campaña: la liberación no toca el fixture compartido.
    output = root / "campaign"
    shutil.copytree(base.output, output, symlinks=True)
    with pytest.MonkeyPatch.context() as patch:
        on_cpu(patch)
        patch.setenv(HOLD_ENV, str(base.hold))
        staged = root / "ablation"
        ablation.run_stage(base.stage, base.views, output, staged, lease=CpuLease, stop=RUNNING)
        masked = ablation.write_sources(base.stage, base.views, output, staged, "US")
        primary = engine.write_sources(base.campaign, base.views, output, "US")
        expected = comparison.evaluate_walk_forward(
            base.comparison, primary, "US", ablation_sources=masked
        )
        path, campaign = schedule(base, root)
        windows = rolling.load_schedule(path, campaign)
        jobs = [j for j in plan.plan_campaign(campaign) if j["kind"] == engine.FIT]
        odd = jobs[0]["id"]
        available = engine.regenerators()

        def regenerate(run):
            if run.job["id"] == odd:
                return flipped(available["neural", engine.FIT])(run)
            return available["neural", engine.FIT](run)

        state = rolling.Rolling(
            rolling.load_retention(DECLARATION),
            base.campaign,
            base.views,
            output,
            windows,
            ablation=dict(stage=base.stage, output=staged),
            regenerators={**available, ("neural", engine.FIT): regenerate},
        )
        calls = Calls()
        result = rolling.run_rolling(state, calls.runners())
        again = Calls()
        resumed = rolling.run_rolling(state, again.runners())
        with pytest.raises(prediction_files.PredictionsReleased):
            comparison.evaluate_walk_forward(
                base.comparison, primary, "US", ablation_sources=masked
            )
        report = comparison.evaluate_walk_forward(
            base.comparison,
            primary,
            "US",
            ablation_sources=masked,
            aggregates=state.folder / "aggregates",
        )
        yield dict(
            base=base,
            state=state,
            windows=windows,
            jobs=jobs,
            odd=odd,
            staged=staged,
            expected=expected,
            report=report,
            result=result,
            resumed=resumed,
            calls=calls.calls,
            again=again.calls,
        )


def tables_of(state, job_id, output):
    receipt = json.loads((output / "jobs" / job_id / "receipt.json").read_text())
    report_path = output / receipt["report"]["path"]
    report = json.loads(report_path.read_text())
    return {
        partition: (report_path.parent / record["path"], record["sha256"])
        for partition, record in report["predictions"].items()
        if partition != "train"
    }


def test_phases_run_once_per_window_in_order_and_resume_without_repeating(walked):
    ids = [row["id"] for row in walked["windows"]]
    assert walked["calls"] == [(phase, w) for w in ids for phase in ("base", "ablation")]
    assert walked["result"]["status"] == walked["resumed"]["status"] == "completed"
    assert walked["again"] == []
    ledger = walked["state"].ledger()
    for window in ids:
        assert set(ledger["windows"][window]["phases"]) == {
            "base",
            "ablation",
            "aggregates",
            "release",
        }


def test_the_final_comparison_from_aggregates_equals_the_one_that_read_the_rows(walked):
    expected, sessions = walked["expected"]
    report, from_aggregates = walked["report"]
    assert {k: v for k, v in report.items() if k not in VOLATILE} == {
        k: v for k, v in expected.items() if k not in VOLATILE
    }
    assert from_aggregates.equals(sessions)


def test_only_bit_exact_regenerations_release_their_rows(walked):
    state = walked["state"]
    for job in walked["jobs"]:
        tables = tables_of(state, job["id"], state.output)
        states = {prediction_files.verify(path, digest) for path, digest in tables.values()}
        if job["id"] == walked["odd"]:
            # Un bit distinto: las filas se compactan sin pérdida y se conservan.
            assert states == {prediction_files.COMPACTED}
            for path, digest in tables.values():
                restored = prediction_files.read(path, digest)
                record = prediction_files.entry(path)
                assert prediction_files.content_digest(restored) == record["content_sha256"]
        else:
            assert states == {prediction_files.RELEASED}
            for path, _ in tables.values():
                record = prediction_files.entry(path)
                assert record["job"] == job["id"] and record["stage"] == "base"
    reports = state.folder / "regeneration-reports"
    decided = [json.loads(path.read_text()) for path in reports.glob("*.json")]
    assert sum(not item["identical"] for item in decided) == 1
    assert not (state.folder / "regeneration").exists() or not any(
        (state.folder / "regeneration").iterdir()
    )


def test_masked_predictions_are_released_after_their_aggregates(walked):
    stage = ablation.load_stage(walked["state"].ablation["stage"])
    for job in ablation.plan_stage(stage):
        receipt = json.loads((walked["staged"] / "jobs" / job["id"] / "receipt.json").read_text())
        record = receipt["prediction"]
        assert (
            prediction_files.verify(walked["staged"] / record["path"], record["sha256"])
            == prediction_files.RELEASED
        )


def test_shared_rows_tables_are_kept_only_while_a_compacted_table_needs_them(walked):
    state = walked["state"]
    tables = list((state.folder / "rows").rglob("rows-*.parquet"))
    referenced = set()
    for record in state.output.rglob(prediction_files.RETENTION_FILE):
        referenced |= prediction_files.rows_references(record.parent)
    assert tables and {path.resolve() for path in tables} == referenced


def test_the_campaign_resumes_over_released_rows_without_running_any_job(walked, monkeypatch):
    """Los recibos con filas liberadas se confirman y ningún ejecutor vuelve a correr."""
    shared = walked["base"]

    def refuse(run):
        raise AssertionError(f"{run.job['id']} no debe volver a ejecutarse")

    on_cpu(monkeypatch)
    monkeypatch.setenv(HOLD_ENV, str(shared.hold))
    executors = {key: dict(entry, run=refuse) for key, entry in engine.EXECUTORS.items()}
    summary = engine.run_campaign(
        shared.campaign,
        shared.views,
        walked["state"].output,
        executors=executors,
        lease=CpuLease,
        stop=RUNNING,
    )
    assert summary["status"] == "completed"


def test_policy_needs_keep_each_evaluation_until_its_last_reading_window():
    positions = {("US", f"fold-{i:03d}"): i for i in range(8)}
    jobs = [
        dict(
            scope="US",
            window=f"fold-{w:03d}",
            train=[f"fold-{w - 4 + k:03d}" for k in range(3)],
            validation=f"fold-{w - 1:03d}",
            predictor="gru",
        )
        for w in range(4, 8)
    ]
    needs = rolling_storage.policy_needs(jobs, positions, 42)
    assert needs[("US", "fold-000", "gru", 42)] == 4
    assert needs[("US", "fold-003", "gru", 42)] == 7
    assert needs[("US", "fold-007", "gru", 42)] == 7
    assert ("US", "fold-000", "transformer", 42) not in needs


def test_a_policy_input_is_compacted_and_released_only_after_its_last_reader(
    base,  # noqa: F811
    tmp_path,
    monkeypatch,
):
    on_cpu(monkeypatch)
    monkeypatch.setenv(HOLD_ENV, str(base.hold))
    output = tmp_path / "campaign"
    # Copia propia de la campaña para poder liberar sin tocar el fixture compartido.
    shutil.copytree(base.output, output, symlinks=True)
    path, campaign = schedule(base, tmp_path)
    windows = rolling.load_schedule(path, campaign)
    state = rolling.Rolling(
        rolling.load_retention(DECLARATION), base.campaign, base.views, output, windows
    )
    first = windows[0]["scopes"]["US"]
    kept, _ = state.base_state(0)[0].selected("US", first, "gru", 42)
    monkeypatch.setattr(
        rolling.Rolling, "policy_keep", lambda self, state, index: {kept: len(windows) - 1}
    )
    calls = Calls()
    rolling.run_rolling(state, calls.runners())
    tables = tables_of(state, kept, output)
    # Tras la última ventana que la lee, también se libera.
    assert {prediction_files.verify(p, d) for p, d in tables.values()} == {
        prediction_files.RELEASED
    }
    ledger = state.ledger()["windows"]
    assert ledger[first]["phases"]["release"]["kept_for_policies"] == 1
    assert ledger[windows[-1]["id"]]["phases"]["release"]["kept_for_policies"] == 0


@pytest.mark.parametrize(
    "edit,message",
    [
        (lambda d: d["release"]["stages"].append("adapters"), "regeneradas bit a bit"),
        (lambda d: d["release"].update(otherwise="release"), "regeneradas bit a bit"),
        (lambda d: d["numerics"].update(cudnn_allow_tf32=True), "FP32 estricto"),
        (lambda d: d.update(aggregates=[]), "agregados"),
        (lambda d: d.update(status="draft"), "contrato"),
        (lambda d: d["phases"].reverse(), "contrato"),
    ],
)
def test_the_declaration_is_fixed_before_results(tmp_path, edit, message):
    document = json.loads(open(DECLARATION).read())
    assert rolling.load_retention(DECLARATION)["sha256"]
    changed = copy.deepcopy(document)
    edit(changed)
    atomic_json(tmp_path / "retention.json", changed)
    with pytest.raises(ValueError, match=message):
        rolling.load_retention(tmp_path / "retention.json")


def test_the_schedule_must_cover_exactly_the_campaign_windows(base, tmp_path):  # noqa: F811
    path, campaign = schedule(base, tmp_path)
    document = json.loads(path.read_text())
    for edit, message in (
        (lambda d: d.update(campaign="0" * 64), "no pertenece"),
        (lambda d: d["windows"].pop(), "exactamente"),
        (lambda d: d["windows"].append(copy.deepcopy(d["windows"][0])), "repite"),
        (lambda d: d["windows"][0]["scopes"].update(CN="fold-000"), "no corresponde"),
    ):
        changed = copy.deepcopy(document)
        edit(changed)
        atomic_json(tmp_path / "other.json", changed)
        with pytest.raises(ValueError, match=message):
            rolling.load_schedule(tmp_path / "other.json", campaign)


def test_default_runners_require_the_window_filter_of_the_stages(base, monkeypatch):  # noqa: F811
    def without_window(path, views, output, *, storage=None, stop=None):
        return {}

    monkeypatch.setattr(engine, "run_campaign", without_window)
    with pytest.raises(ValueError, match="no admite una ventana"):
        rolling.default_runners(base.campaign, base.views, base.output)


class Walk:
    """Recorrido mínimo: registra las fases y prohíbe agregar o liberar."""

    windows = [dict(id="w0", scopes={}), dict(id="w1", scopes={})]
    disk = None

    def __init__(self):
        self.marked = []

    def done(self, window, phase):
        return (window, phase) in self.marked

    def mark(self, window, phase, **record):
        self.marked.append((window, phase))

    def aggregates(self, row):
        raise AssertionError("Una fase pausada no llega a los agregados")

    def release(self, index):
        raise AssertionError("Una fase pausada no libera nada")


def test_a_paused_phase_stops_the_walk_before_aggregates_and_release():
    walk = Walk()
    runners = dict(
        base=lambda window: dict(status="completed"),
        ablation=lambda window: dict(status="paused"),
    )
    result = rolling.run_rolling(walk, runners)
    assert result == dict(status="paused", window="w0", windows={})
    assert walk.marked == [("w0", "base")]
    stopped = Walk()
    result = rolling.run_rolling(stopped, runners, stop=SimpleNamespace(requested=True))
    assert result["status"] == "paused" and stopped.marked == []


def test_a_window_that_does_not_fit_is_refused_before_any_phase(base, tmp_path, monkeypatch):  # noqa: F811
    from mars_titan.training.campaign_storage import DiskBudgetError, load_storage

    on_cpu(monkeypatch)
    monkeypatch.setenv(HOLD_ENV, str(base.hold))
    path, campaign = schedule(base, tmp_path)
    windows = rolling.load_schedule(path, campaign)
    extras = json.loads(open("reports/engineering/campaign-storage-20261009/extras.json").read())
    storage = load_storage("configs/baselines/historical-masked-campaign-storage.json")
    free = dict(bytes=0)

    def usage(_):
        return SimpleNamespace(free=free["bytes"])

    disk = dict(
        storage="configs/baselines/historical-masked-campaign-storage.json",
        extras=extras,
        usage=usage,
    )
    state = rolling.Rolling(
        rolling.load_retention(DECLARATION),
        base.campaign,
        base.views,
        base.output,
        windows,
        ablation=dict(stage=base.stage, output=tmp_path / "unused"),
        disk=disk,
    )
    increment, _ = state.disk_projection(windows[0])
    assert increment["bytes"] > 0 and increment["window"] == windows[0]["id"]
    free["bytes"] = storage["margin_bytes"] + increment["bytes"] - 1
    calls = Calls()
    with pytest.raises(DiskBudgetError, match="supera"):
        rolling.run_rolling(state, calls.runners())
    assert calls.calls == [] and not state.ledger_path.exists()
    free["bytes"] += 1
    launched = state.require_disk(windows[0])
    assert launched["peak_bytes"] == increment["bytes"]


def test_the_campaign_script_refuses_the_walk_until_the_stages_accept_a_window(
    base,  # noqa: F811
    tmp_path,
):
    """Sin el filtro `window` de la campaña A v2 no se ejecuta ni se escribe nada."""
    import runpy

    script = runpy.run_path("scripts/run_masked_campaign.py", run_name="script")
    path, _ = schedule(base, tmp_path)
    arguments = ["rolling", "--retention", DECLARATION, "--schedule", str(path)]
    arguments += ["--campaign", str(base.campaign), "--views", f"US={base.views['US']}"]
    arguments += ["--output", str(tmp_path / "campaign")]
    arguments += ["--storage", "configs/baselines/historical-masked-campaign-storage.json"]
    arguments += ["--extras", "reports/engineering/campaign-storage-20261009/extras.json"]
    with pytest.raises(ValueError, match="no admite una ventana"):
        script["main"](arguments)
    assert not (tmp_path / "campaign").exists()


def test_a_regeneration_error_keeps_the_rows_and_is_retried_while_a_comparison_is_final(
    tmp_path,
):
    """Un fallo antes de comparar no decide nada. Una comparación escrita sí es definitiva."""
    from tests.data.test_prediction_files import writer, written

    state = rolling.Rolling.__new__(rolling.Rolling)
    state.folder = tmp_path / "retention"
    job = dict(id="US/fold-000/gru/search-a", scope="US", window="fold-000")
    path, digest = written(tmp_path / "attempt", writer("neural"))
    tables = {"evaluation": (path, digest)}
    attempts = []

    def failing(destination):
        attempts.append("error")
        raise MemoryError("sin memoria")

    def identical(destination):
        attempts.append("identical")
        destination.mkdir(parents=True)
        return dict(identical=True)

    def differs(destination):
        attempts.append("differs")
        return dict(identical=False)

    totals = dict(released=0, not_regenerable=0)
    state._release_job("base", job, tables, failing, totals)
    assert prediction_files.verify(path, digest) == prediction_files.COMPACTED
    assert totals == dict(released=0, not_regenerable=1)
    # En la ventana siguiente se vuelve a intentar y, si coincide, se libera.
    state._release_job("base", job, tables, identical, totals)
    assert prediction_files.verify(path, digest) == prediction_files.RELEASED
    assert attempts == ["error", "identical"] and totals["released"] == 1
    assert not (state.folder / "regeneration").exists() or not any(
        (state.folder / "regeneration").iterdir()
    )
    other, other_digest = written(tmp_path / "other", writer("neural", seed=2))
    second = dict(job, id="US/fold-000/gru/search-b")
    state._release_job("base", second, {"evaluation": (other, other_digest)}, differs, totals)
    state._release_job("base", second, {"evaluation": (other, other_digest)}, identical, totals)
    # La comparación distinta ya escrita decide: no se regenera otra vez ni se libera.
    assert attempts[-1] == "differs"
    assert prediction_files.verify(other, other_digest) == prediction_files.COMPACTED
    assert totals["not_regenerable"] == 2
