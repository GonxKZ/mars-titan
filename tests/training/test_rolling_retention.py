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
from mars_titan.evaluation import long_short_comparison as portfolio
from mars_titan.evaluation import walk_forward_comparison as comparison
from mars_titan.simulation import policy_plan
from mars_titan.training import campaign_plan as plan
from mars_titan.training import masked_campaign as engine
from mars_titan.training import modality_ablation_stage as ablation
from mars_titan.training import rolling_retention as rolling
from mars_titan.training import rolling_storage
from mars_titan.training.learning_hold import HOLD_ENV
from tests.posttraining.campaign_fixture import CpuLease
from tests.simulation import rl_stage_fixture
from tests.simulation.unadjusted_edition_fixture import Asset
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


@pytest.fixture(scope="module")
def unconsumed(tmp_path_factory):
    """Retención sin las políticas como consumidoras, para recorrer solo base y ablación."""
    document = json.loads(open(DECLARATION).read())
    document["consumers"] = {}
    path = tmp_path_factory.mktemp("retention") / "retention.json"
    atomic_json(path, document)
    return path


@pytest.fixture(scope="module")
def edition(tmp_path_factory):
    """Edición de precios sintética para la cartera larga y corta de la comparación."""
    root = tmp_path_factory.mktemp("edition")
    rl_stage_fixture.write_edition(
        root, {"US": [Asset("A0000", base=20.0), Asset("B0001", base=30.0)]}
    )
    return root


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
def walked(base, edition, unconsumed, tmp_path_factory):  # noqa: F811
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
        expected_portfolio = portfolio.evaluate_long_short(base.comparison, primary, "US", edition)
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
            rolling.load_retention(unconsumed),
            base.campaign,
            base.views,
            output,
            windows,
            ablation=dict(stage=base.stage, output=staged),
            regenerators={**available, ("neural", engine.FIT): regenerate},
            edition=edition,
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
        with pytest.raises(prediction_files.PredictionsReleased):
            portfolio.evaluate_long_short(base.comparison, primary, "US", edition)
        from_aggregates = portfolio.evaluate_long_short(
            base.comparison, primary, "US", edition, aggregates=state.folder / "aggregates"
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
            portfolio=(expected_portfolio, from_aggregates),
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
    (expected_portfolio, expected_books), (portfolio_report, books) = walked["portfolio"]
    volatile = {"created_at_utc", "resources"}
    assert {k: v for k, v in portfolio_report.items() if k not in volatile} == {
        k: v for k, v in expected_portfolio.items() if k not in volatile
    }
    assert books.equals(expected_books)
    ledger = walked["state"].ledger()["windows"]
    for row in walked["windows"]:
        assert set(ledger[row["id"]]["phases"]["aggregates"]["written"]["US"]) == {
            "walk_forward",
            "long_short",
        }


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
    stage = dict(
        policies=dict(predictor=dict(seed=42, source=policy_plan.BASE_SELECTED)),
        universe_predictor="gru",
        scopes=["US"],
    )
    needs = rolling_storage.policy_needs(stage, jobs, positions)
    assert needs[("US", "fold-000", "gru", 42)] == 4
    assert needs[("US", "fold-003", "gru", 42)] == 7
    assert needs[("US", "fold-007", "gru", 42)] == 7
    assert ("US", "fold-000", "transformer", 42) not in needs


def test_with_the_chain_the_policies_only_read_the_base_in_the_first_window():
    # Desde la ventana 1 la cadena publica tablas de los adaptadores, que solo se compactan.
    stage = policy_plan.load_stage("configs/simulation/historical-masked-rl-stage-a.json")
    jobs = policy_plan.plan_stage(stage)
    resolved = stage["campaign"]["comparison_config"]["resolved_scopes"]
    pairs = [(scope, window) for scope in stage["scopes"] for window in resolved[scope]["windows"]]
    positions = {pair: index for index, pair in enumerate(pairs)}
    chained = rolling_storage.policy_needs(stage, jobs, positions)
    assert chained and {window for _, window, _, _ in chained} == {"fold-000"}
    assert {predictor for *_, predictor, _ in chained} <= set(stage["predictors"])
    predictor = dict(stage["policies"]["predictor"], source=policy_plan.BASE_SELECTED)
    selected = dict(stage, policies=dict(stage["policies"], predictor=predictor))
    everything = rolling_storage.policy_needs(selected, jobs, positions)
    assert set(chained) < set(everything)
    assert all(everything[key] == last for key, last in chained.items())


def test_a_policy_input_is_compacted_and_released_only_after_its_last_reader(
    base,  # noqa: F811
    edition,
    tmp_path,
    monkeypatch,
    unconsumed,
):
    on_cpu(monkeypatch)
    monkeypatch.setenv(HOLD_ENV, str(base.hold))
    output = tmp_path / "campaign"
    # Copia propia de la campaña para poder liberar sin tocar el fixture compartido.
    shutil.copytree(base.output, output, symlinks=True)
    path, campaign = schedule(base, tmp_path)
    windows = rolling.load_schedule(path, campaign)
    state = rolling.Rolling(
        rolling.load_retention(unconsumed),
        base.campaign,
        base.views,
        output,
        windows,
        edition=edition,
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
        (lambda d: d.pop("consumers"), "contrato"),
        (lambda d: d["consumers"]["rl"].update(release_after="aggregates"), "consumidoras"),
        (lambda d: d["consumers"].update(ablation=dict(reads="evaluation")), "consumidoras"),
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


def test_the_policy_stage_comes_with_the_retention_that_declares_it(
    base,  # noqa: F811
    unconsumed,
    edition,
    tmp_path,
):
    path, campaign = schedule(base, tmp_path)
    windows = rolling.load_schedule(path, campaign)
    policies = dict(stage=tmp_path / "rl.json", output=tmp_path / "rl")
    for declaration, rl in ((DECLARATION, None), (unconsumed, policies)):
        with pytest.raises(ValueError, match="acompañar a la retención"):
            rolling.Rolling(
                rolling.load_retention(declaration),
                base.campaign,
                base.views,
                base.output,
                windows,
                rl=rl,
                edition=edition,
            )


def test_an_evaluation_read_by_a_policy_is_released_only_after_its_receipt(
    base,  # noqa: F811
    edition,
    tmp_path,
    monkeypatch,
):
    from mars_titan.simulation import campaign_stage as policy_stage

    on_cpu(monkeypatch)
    monkeypatch.setenv(HOLD_ENV, str(base.hold))
    output = tmp_path / "campaign"
    shutil.copytree(base.output, output, symlinks=True)
    path, campaign = schedule(base, tmp_path)
    windows = rolling.load_schedule(path, campaign)
    policies = tmp_path / "rl"
    state = rolling.Rolling(
        rolling.load_retention(DECLARATION),
        base.campaign,
        base.views,
        output,
        windows,
        rl=dict(stage=tmp_path / "rl.json", output=policies),
        edition=edition,
    )
    first = windows[0]["scopes"]["US"]
    read, _ = state.base_state(0)[0].selected("US", first, "gru", 42)
    # Una sola política de la primera ventana lee la evaluación del predictor elegido.
    reader = dict(id=f"US/US/{first}/gru/native_klpo/fit-s42", scope="US", window=first)
    monkeypatch.setattr(rolling.Rolling, "policy_stage", lambda self: (dict(sha256="5" * 64), []))
    monkeypatch.setattr(rolling.Rolling, "policy_reads", lambda self, s, index: {read: [reader]})
    tables = tables_of(state, read, output)

    def states():
        return {prediction_files.verify(p, d) for p, d in tables.values()}

    with pytest.raises(ValueError, match="Faltan 1 recibos de las políticas"):
        state.release(0)
    assert states() == {prediction_files.PRESENT}
    marker = dict(kind=policy_stage.RUN_KIND, stage_sha256="5" * 64)
    atomic_json(policies / "stage.json", marker)
    receipt = dict(
        kind=policy_stage.RECEIPT_KIND,
        status="completed",
        identity=dict(id=reader["id"], stage_identity_sha256="6" * 64),
    )
    atomic_json(policies / "jobs" / reader["id"] / "receipt.json", receipt)
    # Un recibo de otra ejecución de la etapa no cuenta.
    with pytest.raises(ValueError, match="Faltan 1 recibos de las políticas"):
        state.release(0)
    assert states() == {prediction_files.PRESENT}
    receipt["identity"]["stage_identity_sha256"] = policy_stage._digest(marker)
    atomic_json(policies / "jobs" / reader["id"] / "receipt.json", receipt)
    totals = state.release(0)
    assert totals["released"] >= 1 and states() == {prediction_files.RELEASED}


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


def test_the_walk_needs_the_price_edition_when_the_comparison_declares_the_portfolio(
    base,  # noqa: F811
    tmp_path,
    unconsumed,
):
    path, campaign = schedule(base, tmp_path)
    windows = rolling.load_schedule(path, campaign)
    assert comparison.LONG_SHORT_FIELD in campaign["comparison_config"]
    with pytest.raises(ValueError, match="falta la edición"):
        rolling.Rolling(
            rolling.load_retention(unconsumed), base.campaign, base.views, base.output, windows
        )


def test_default_runners_require_the_window_filter_of_the_stages(base, monkeypatch):  # noqa: F811
    def without_window(path, views, output, *, storage=None, stop=None):
        return {}

    monkeypatch.setattr(engine, "run_campaign", without_window)
    with pytest.raises(ValueError, match="no admite una ventana"):
        rolling.default_runners(base.campaign, base.views, base.output)


def test_staged_runners_give_each_stage_a_fresh_report_and_the_chain(
    base,  # noqa: F811
    monkeypatch,
    tmp_path,
):
    """Con el walk-forward por etapas, cada etapa recibe un informe calculado en ese momento.

    La campaña reducida es de la versión 1, así que el diseño se añade a la campaña cargada.
    Las etapas son dobles que registran sus argumentos, sin ajustar nada.
    """
    from mars_titan.posttraining import campaign_stage as adapter_stage
    from mars_titan.simulation import campaign_stage as rl_stage
    from mars_titan.training import campaign_chain, chain_disjunction

    load = plan.load_campaign
    monkeypatch.setattr(
        plan,
        "load_campaign",
        lambda path: dict(load(path), walk_forward_stages=campaign_chain.DESIGN),
    )
    calls = {}

    def adapters(path, views, campaign_output, output, *, stop=None, window=None, **options):
        calls["adapters"] = dict(options, window=window)
        return dict(status="completed")

    def policies(path, views, campaign, edition, output, *, stop=None, window=None, **options):
        calls["rl"] = dict(options, window=window)
        return dict(status="completed")

    monkeypatch.setattr(adapter_stage, "run_stage", adapters)
    monkeypatch.setattr(rl_stage, "run_stage", policies)
    # Copia propia de la campaña: el informe se guarda en su carpeta de retención.
    output = tmp_path / "campaign"
    shutil.copytree(base.output, output, symlinks=True)
    stages = dict(
        adapters=dict(stage="adapters.json", output=tmp_path / "adapters"),
        rl=dict(stage="rl.json", output=tmp_path / "rl", edition=tmp_path / "edition"),
    )
    runners = rolling.default_runners(base.campaign, base.views, output, stages=stages)
    runners["adapters"]("fold-000")
    path = calls["adapters"]["disjunction"]
    assert path == output / "retention/disjunction/fold-000-adapters.json"
    report = json.loads(path.read_text())
    assert report["kind"] == chain_disjunction.KIND and report["failures"] == []
    assert report["campaign_sha256"] == plan.load_campaign(base.campaign)["sha256"]
    runners["rl"]("fold-001")
    assert calls["rl"]["chain_output"] == tmp_path / "adapters"
    assert calls["rl"]["disjunction"] == output / "retention/disjunction/fold-001-rl.json"
    # Un informe con fallos se guarda para revisarlo y detiene el recorrido antes de la etapa.
    calls.clear()
    monkeypatch.setattr(
        chain_disjunction, "verify", lambda *args, **options: dict(report, failures=["x"])
    )
    with pytest.raises(ValueError, match="1 fallos antes de rl en fold-002"):
        runners["rl"]("fold-002")
    assert calls == {}
    assert json.loads((output / "retention/disjunction/fold-002-rl.json").read_text())[
        "failures"
    ] == ["x"]


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


def test_a_window_that_does_not_fit_is_refused_before_any_phase(
    base,  # noqa: F811
    edition,
    tmp_path,
    monkeypatch,
    unconsumed,
):
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
        rolling.load_retention(unconsumed),
        base.campaign,
        base.views,
        base.output,
        windows,
        ablation=dict(stage=base.stage, output=tmp_path / "unused"),
        disk=disk,
        edition=edition,
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


def test_the_campaign_script_accepts_the_window_and_still_refuses_an_incomplete_walk(
    base,  # noqa: F811
    tmp_path,
):
    """Con el filtro `window` de la campaña A v2, el recorrido pasa a la siguiente guarda.

    La base y las etapas ya admiten una ventana, así que el recorrido pasa esa comprobación
    y se detiene porque la retención declara las políticas como consumidoras y la orden no
    trae su etapa. No ejecuta ni escribe nada.
    """
    from mars_titan.posttraining import campaign_stage as adapter_stage
    from mars_titan.training import masked_campaign as engine
    from mars_titan.training import modality_ablation_stage as ablation_stage

    for runner in (engine.run_campaign, ablation_stage.run_stage, adapter_stage.run_stage):
        assert rolling._supports_window(runner)
    import runpy

    script = runpy.run_path("scripts/run_masked_campaign.py", run_name="script")
    path, _ = schedule(base, tmp_path)
    arguments = ["rolling", "--retention", DECLARATION, "--schedule", str(path)]
    arguments += ["--campaign", str(base.campaign), "--views", f"US={base.views['US']}"]
    arguments += ["--output", str(tmp_path / "campaign")]
    arguments += ["--storage", "configs/baselines/historical-masked-campaign-storage.json"]
    arguments += ["--extras", "reports/engineering/campaign-storage-20261009/extras.json"]
    with pytest.raises(ValueError, match="acompañar a la retención"):
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


def test_masked_predictions_without_aggregates_are_compacted_not_released(
    base,  # noqa: F811
    edition,
    tmp_path,
    monkeypatch,
    unconsumed,
):
    """Si la comparación no declara la ablación, sus filas no tienen agregados y se conservan."""
    on_cpu(monkeypatch)
    monkeypatch.setenv(HOLD_ENV, str(base.hold))
    output = tmp_path / "campaign"
    shutil.copytree(base.output, output, symlinks=True)
    staged = tmp_path / "ablation"
    ablation.run_stage(base.stage, base.views, output, staged, lease=CpuLease, stop=RUNNING)
    path, campaign = schedule(base, tmp_path)
    windows = rolling.load_schedule(path, campaign)
    state = rolling.Rolling(
        rolling.load_retention(unconsumed),
        base.campaign,
        base.views,
        output,
        windows,
        ablation=dict(stage=base.stage, output=staged),
        edition=edition,
    )
    declared = state.comparison_config()
    without = {k: v for k, v in declared.items() if k != comparison.ABLATION_FIELD}
    monkeypatch.setattr(rolling.Rolling, "comparison_config", lambda self: without)
    rolling.run_rolling(state, Calls().runners())
    for job in ablation.plan_stage(ablation.load_stage(base.stage)):
        receipt = json.loads((staged / "jobs" / job["id"] / "receipt.json").read_text())
        record = receipt["prediction"]
        state_of = prediction_files.verify(staged / record["path"], record["sha256"])
        assert state_of == prediction_files.COMPACTED


@pytest.mark.parametrize("mark", [dict(regenerable=False), dict(kind=plan.ONLINE)])
def test_jobs_declared_not_regenerable_are_compacted_without_regenerating(
    base,  # noqa: F811
    edition,
    tmp_path,
    monkeypatch,
    mark,
    unconsumed,
):
    """Un trabajo con `regenerable=False` o un control en línea se conserva sin regenerar."""
    on_cpu(monkeypatch)
    monkeypatch.setenv(HOLD_ENV, str(base.hold))
    output = tmp_path / "campaign"
    shutil.copytree(base.output, output, symlinks=True)
    path, campaign = schedule(base, tmp_path)
    windows = rolling.load_schedule(path, campaign)
    state = rolling.Rolling(
        rolling.load_retention(unconsumed),
        base.campaign,
        base.views,
        output,
        windows,
        edition=edition,
    )
    target = state.base_state(0)[1][0]["id"]
    original = rolling.Rolling.base_state

    def marked(self, index):
        found, jobs = original(self, index)
        return found, [dict(job, **mark) if job["id"] == target else job for job in jobs]

    monkeypatch.setattr(rolling.Rolling, "base_state", marked)
    rolling.run_rolling(state, Calls().runners())
    tables = tables_of(state, target, output)
    for path_of, digest in tables.values():
        assert prediction_files.verify(path_of, digest) == prediction_files.COMPACTED
        record = prediction_files.entry(path_of)
        restored = prediction_files.read(path_of, digest)
        assert prediction_files.content_digest(restored) == record["content_sha256"]
    report = state.folder / "regeneration-reports" / f"{rolling._name(f'base:{target}')}.json"
    assert not report.exists()
    release = state.ledger()["windows"][windows[0]["id"]]["phases"]["release"]
    assert release["declared_not_regenerable"] == 1


def test_a_plateau_of_a_joint_fit_has_no_tables_to_release(
    base,  # noqa: F811
    edition,
    tmp_path,
    monkeypatch,
    unconsumed,
):
    """La meseta de un ajuste conjunto no escribe tablas: la liberación no la regenera."""
    on_cpu(monkeypatch)
    monkeypatch.setenv(HOLD_ENV, str(base.hold))
    output = tmp_path / "campaign"
    shutil.copytree(base.output, output, symlinks=True)
    path, campaign = schedule(base, tmp_path)
    windows = rolling.load_schedule(path, campaign)
    state = rolling.Rolling(
        rolling.load_retention(unconsumed),
        base.campaign,
        base.views,
        output,
        windows,
        edition=edition,
    )
    original = rolling.Rolling.base_state
    added = []

    def with_plateau(self, index):
        # La meseta va antes que su continuación y su recibo solo copia el informe en espera.
        found, jobs = original(self, index)
        target = jobs[0]
        head, _, name = target["id"].rpartition("/")
        plateau = dict(target, id=f"{head}/plateau-{name}", phase=plan.PLATEAU)
        copy = output / "jobs" / plateau["id"] / "plateau-report.json"
        copy.parent.mkdir(parents=True, exist_ok=True)
        copy.write_text(json.dumps(dict(status="awaiting_joint_stop", individual_stop_epoch=1)))
        found.receipts[plateau["id"]] = dict(
            status=engine.PLATEAU_STATUS,
            report=dict(path=str(copy.relative_to(output))),
            predictions={},
        )
        added.append(plateau["id"])
        return found, [plateau, *jobs]

    monkeypatch.setattr(rolling.Rolling, "base_state", with_plateau)
    rolling.run_rolling(state, Calls().runners())
    assert added
    reports = state.folder / "regeneration-reports"
    assert not any((reports / f"{rolling._name(f'base:{key}')}.json").exists() for key in added)
    # La continuación sí se regenera y se libera como cualquier ajuste.
    release = state.ledger()["windows"][windows[0]["id"]]["phases"]["release"]
    assert release["released"] > 0
