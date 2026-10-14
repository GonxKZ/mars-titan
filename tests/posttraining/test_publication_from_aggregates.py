"""Paso final de la campaña desde los agregados del recorrido ventana a ventana.

La campaña A reducida de `campaign_fixture` (dos ventanas US y GRU con la semilla 42 y
pesos iniciales) se completa con su etapa de adaptadores, cuyo optimizador solo registra
gradientes y nunca cambia pesos. La retención v2 la recorre después con la publicación
declarada: guarda los agregados de cada ventana y libera las filas que regenera bit a bit.
Se comprueba que la publicación final desde los agregados coincide exactamente con los
informes que leían las filas antes de liberarlas, que un corte entre los agregados y la
liberación se reanuda sin perder ni duplicar nada y que una publicación fallida no deja
destino. Al final, el informe de integridad (#442): cada mutación dirigida rompe el código
principal o un recibo y debe detener la publicación o la ventana. No se ejecuta ningún paso
de optimizador ni se abre 2024. Como la etapa y la regeneración solo se recorren con la
protección rebajada, el módulo se omite mientras rige el bloqueo de aprendizaje (#546).
"""

import json
import runpy
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from mars_titan.data import prediction_files
from mars_titan.data.storage import sha256
from mars_titan.evaluation import comparison_matrix, forecast_scores
from mars_titan.evaluation import session_table_contrasts as tables
from mars_titan.evaluation import walk_forward_comparison as walk
from mars_titan.integrity import declared_runs as runs
from mars_titan.integrity import publication_gate as gate
from mars_titan.integrity import window_selection as selection
from mars_titan.posttraining import campaign_stage
from mars_titan.posttraining import stage_comparison as compare
from mars_titan.training import campaign_publication as publication
from mars_titan.training import masked_campaign as engine
from mars_titan.training import rolling_retention as rolling
from mars_titan.training.campaign_plan import load_campaign, plan_campaign
from mars_titan.training.learning_hold import HOLD_ENV
from tests.learning_hold_lowering import lowers_learning_hold
from tests.posttraining import recording_optimizer
from tests.posttraining.campaign_fixture import CpuLease, base_campaign, quantile_reference
from tests.posttraining.real_only import real_data_only
from tests.training import publication_fixture
from tests.training.test_rolling_retention import schedule, unconsumed  # noqa: F401

# La etapa de adaptadores y la regeneración del recorrido se detienen con la protección.
pytestmark = lowers_learning_hold("tests/posttraining/test_publication_from_aggregates.py")
ROOT = Path(__file__).resolve().parents[2]
VOLATILE = {"created_at_utc", "resources", "artifacts"}
# La regeneración repite el ejecutor de la campaña reducida, que predice con los pesos
# iniciales de la semilla. Sale idéntica bit a bit y las filas de la base se liberan.
REGENERATORS = {("neural", engine.FIT): quantile_reference}
RUNNING = SimpleNamespace(requested=False)


class Cut(Exception):
    """Corte simulado del proceso entre dos pasos del recorrido."""


def plain(declared):
    """Comparación sin cartera ni ablación.

    La cartera necesita una edición con los activos del corpus y la ablación su propia etapa.
    Las dos tienen ya su prueba desde agregados en `test_rolling_retention`.
    """
    for section in (walk.LONG_SHORT_FIELD, walk.ABLATION_FIELD):
        declared.pop(section)
    # La versión 2 declara solo los estratos de presencia.
    declared["schema_version"] = 2


def stable(report, *volatile):
    return {key: value for key, value in report.items() if key not in {*VOLATILE, *volatile}}


def without_paths(value):
    """Manifiesto de fuentes sin rutas: solo huellas, vistas y políticas de cada archivo."""
    if isinstance(value, dict):
        return {key: without_paths(item) for key, item in value.items() if key != "path"}
    return value


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    """Campaña y etapa confirmadas, e informes calculados desde las filas sin liberar."""
    root = tmp_path_factory.mktemp("publication")
    base = base_campaign(root, "A", comparison=plain)
    adapters = root / "adapters"
    deterministic = torch.are_deterministic_algorithms_enabled()
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv(HOLD_ENV, str(base.hold))
        record = recording_optimizer.install(patch)
        with real_data_only():
            summary = campaign_stage.run_stage(
                base.stage,
                base.views,
                base.output,
                adapters,
                lease=CpuLease,
                stop=RUNNING,
                device="cpu",
            )
    torch.use_deterministic_algorithms(deterministic)
    assert summary["status"] == "completed" and record.optimizers
    declared = publication_fixture.declare(root / "declared", base.campaign, stage=base.stage)
    primary = engine.write_sources(base.campaign, base.views, base.output, "US")
    comparison = Path(base.campaign).parent / "comparison.json"
    folder = root / "expected"
    walked = walk.write_walk_forward(comparison, primary, "US", folder / "walk-forward")
    loaded = comparison_matrix.load_matrix(declared.matrix)
    manifest = tables.write_sources(
        folder / "matrix-sources.json",
        "US",
        [dict(kind=tables.WALK_SOURCE, path=folder / "walk-forward" / "comparison.json")],
        views=loaded["views"],
        config=loaded["comparison_config"],
    )
    matrix = comparison_matrix.evaluate(declared.matrix, manifest, "US")
    sources = compare.write_sources(
        declared.stage, "US", "gru", base_sources=primary, stage_output=adapters
    )
    _, stage_report, stage_sessions, _ = compare.evaluate(declared.stage, sources, "US", "gru")
    return SimpleNamespace(
        base=base,
        adapters=adapters,
        declared=declared,
        walk=(walked, pq.read_table(folder / "walk-forward" / "sessions.parquet")),
        matrix=matrix,
        stage=(stage_report, stage_sessions, sources),
    )


def copied(prepared, folder, retention):
    """Recorrido sobre una copia propia de la campaña y de la etapa."""
    output, adapters = folder / "campaign", folder / "adapters"
    shutil.copytree(prepared.base.output, output, symlinks=True)
    shutil.copytree(prepared.adapters, adapters, symlinks=True)
    path, campaign = schedule(prepared.base, folder)
    return rolling.Rolling(
        rolling.load_retention(retention),
        prepared.base.campaign,
        prepared.base.views,
        output,
        rolling.load_schedule(path, campaign),
        adapters=dict(stage=prepared.base.stage, output=adapters),
        regenerators=REGENERATORS,
        publication=prepared.declared.publication,
    )


def runners():
    # La base y la etapa ya están confirmadas: sus fases solo se registran.
    return {phase: (lambda window: dict(status="completed")) for phase in ("base", "adapters")}


def publish(prepared, state, destination):
    return publication.run_publication(
        prepared.declared.publication,
        prepared.base.views,
        state.output,
        destination,
        adapter_output=state.adapters["output"],
        aggregates=state.folder / "aggregates",
    )


def base_tables(state, window):
    """Estado de las tablas de calibración y evaluación de los ajustes base de una ventana."""
    found = {}
    for job in plan_campaign(state.campaign):
        if job["kind"] != engine.FIT or job["window"] != window:
            continue
        receipt = json.loads((state.output / "jobs" / job["id"] / "receipt.json").read_text())
        report_path = state.output / receipt["report"]["path"]
        report = json.loads(report_path.read_text())
        for partition in ("calibration", "evaluation"):
            record = report["predictions"][partition]
            path = report_path.parent / record["path"]
            found[job["id"], partition] = prediction_files.verify(path, record["sha256"])
    return found


def assert_published_from_aggregates(prepared, state, destination, receipt):
    walked, sessions = prepared.walk
    report = json.loads((destination / "walk-forward" / "US" / "comparison.json").read_text())
    assert stable(report) == stable(json.loads(json.dumps(walked)))
    assert pq.read_table(destination / "walk-forward" / "US" / "sessions.parquet").equals(sessions)
    expected, expected_sessions, sources = prepared.stage
    folder = destination / "posttraining" / "US" / "gru"
    report = json.loads((folder / "comparison.json").read_text())
    # Las fuentes de la etapa guardan rutas relativas a la copia recorrida, así que su
    # huella cambia. Sin las rutas, el manifiesto es el mismo archivo a archivo.
    assert stable(report, "sources_sha256") == stable(
        json.loads(json.dumps(expected)), "sources_sha256"
    )
    published = state.adapters["output"] / "sources" / "US" / "gru.json"
    assert without_paths(json.loads(published.read_text())) == without_paths(
        json.loads(sources.read_text())
    )
    assert report["sources_sha256"] == sha256(published)
    assert pq.read_table(folder / "sessions.parquet").equals(expected_sessions)
    matrix = json.loads((destination / "matrix" / "US" / "matrix.json").read_text())
    expected_matrix = json.loads(json.dumps(prepared.matrix))
    assert matrix["views"] == expected_matrix["views"]
    assert matrix["families"] == expected_matrix["families"]
    assert receipt["status"] == "completed" and receipt["from_aggregates"] is True
    assert receipt["posttraining"]["parents"] == {"US/gru": 1}
    stored = json.loads((destination / "publication.json").read_text())
    assert stored == json.loads(json.dumps(receipt))
    files = {
        str(path.relative_to(destination))
        for path in destination.rglob("*")
        if path.is_file() and path.name != "publication.json"
    }
    assert files == set(receipt["artifacts"])
    for name, digest in receipt["artifacts"].items():
        assert sha256(destination / name) == digest, name
    assert not (destination / publication.PARTIAL_MARK).exists()
    assert not destination.with_name(f".{destination.name}.partial").exists()


@pytest.fixture(scope="module")
def walked(prepared, unconsumed, tmp_path_factory):  # noqa: F811
    root = tmp_path_factory.mktemp("walked")
    state = copied(prepared, root, unconsumed)
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv(HOLD_ENV, str(prepared.base.hold))
        result = rolling.run_rolling(state, runners())
        primary = engine.write_sources(
            prepared.base.campaign, prepared.base.views, state.output, "US"
        )
        sources = compare.write_sources(
            prepared.declared.stage,
            "US",
            "gru",
            base_sources=primary,
            stage_output=state.adapters["output"],
        )
        with pytest.raises(prediction_files.PredictionsReleased):
            compare.evaluate(prepared.declared.stage, sources, "US", "gru")
        # El paso final por la orden de la campaña, como se ejecutará.
        destination = root / "published"
        script = runpy.run_path(str(ROOT / "scripts/run_masked_campaign.py"), run_name="script")
        arguments = ["publication", "run", "--declaration", str(prepared.declared.publication)]
        arguments += ["--views", f"US={prepared.base.views['US']}", "--output", str(state.output)]
        arguments += ["--destination", str(destination)]
        arguments += ["--adapter-output", str(state.adapters["output"])]
        arguments += ["--aggregates", str(state.folder / "aggregates")]
        assert script["main"](arguments) == 0
    receipt = json.loads((destination / "publication.json").read_text())
    return SimpleNamespace(state=state, result=result, destination=destination, receipt=receipt)


def test_the_publication_from_aggregates_equals_the_reports_that_read_the_rows(prepared, walked):
    assert walked.result["status"] == "completed"
    assert_published_from_aggregates(prepared, walked.state, walked.destination, walked.receipt)


def test_the_walk_saves_the_stage_aggregates_before_releasing_the_rows(walked):
    state = walked.state
    ledger = state.ledger()["windows"]
    # Solo la segunda ventana tiene trabajos de la etapa, y por tanto comparación.
    assert "posttraining" not in ledger["fold-000"]["phases"]["aggregates"]["written"]["US"]
    record = ledger["fold-001"]["phases"]["aggregates"]["written"]["US"]["posttraining"]["gru"]
    path = state.folder / record["path"]
    assert path == state.folder / "aggregates" / "posttraining" / "gru" / "US" / "fold-001.npz"
    assert sha256(path) == record["sha256"] and path.stat().st_size == record["bytes"]
    # Las filas del modelo base que lee la comparación están liberadas.
    states = base_tables(state, "fold-001")
    assert states and set(states.values()) == {prediction_files.RELEASED}
    sources = state.adapters["output"] / "sources" / "windows" / "fold-001" / "US" / "gru.json"
    assert list(json.loads(sources.read_text())["windows"]) == ["fold-001"]


def test_a_cut_between_the_aggregates_and_the_release_loses_and_repeats_nothing(
    prepared,
    unconsumed,  # noqa: F811
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv(HOLD_ENV, str(prepared.base.hold))
    state = copied(prepared, tmp_path, unconsumed)
    stored = state.folder / "aggregates" / "posttraining" / "gru" / "US" / "fold-001.npz"
    mark, release = rolling.Rolling.mark, rolling.Rolling.release

    def cut_before_marking(self, window, phase, **record):
        if (window, phase) == ("fold-001", "aggregates"):
            raise Cut
        return mark(self, window, phase, **record)

    # Primer corte: los agregados están escritos y todavía no constan en el registro.
    monkeypatch.setattr(rolling.Rolling, "mark", cut_before_marking)
    with pytest.raises(Cut):
        rolling.run_rolling(state, runners())
    assert stored.is_file() and not state.done("fold-001", "aggregates")
    assert set(base_tables(state, "fold-001").values()) == {prediction_files.PRESENT}

    def cut_before_releasing(self, index):
        if index == 1:
            raise Cut
        return release(self, index)

    # Segundo corte: la reanudación repite los agregados, los confirma y se corta al liberar.
    monkeypatch.setattr(rolling.Rolling, "mark", mark)
    monkeypatch.setattr(rolling.Rolling, "release", cut_before_releasing)
    with pytest.raises(Cut):
        rolling.run_rolling(state, runners())
    record = state.ledger()["windows"]["fold-001"]["phases"]["aggregates"]["written"]["US"]
    confirmed = record["posttraining"]["gru"]["sha256"]
    assert sha256(stored) == confirmed and not state.done("fold-001", "release")
    assert set(base_tables(state, "fold-001").values()) == {prediction_files.PRESENT}

    # La última reanudación libera sin volver a escribir los agregados confirmados.
    monkeypatch.setattr(rolling.Rolling, "release", release)
    assert rolling.run_rolling(state, runners())["status"] == "completed"
    assert sha256(stored) == confirmed
    assert set(base_tables(state, "fold-001").values()) == {prediction_files.RELEASED}
    written = {
        str(path.relative_to(state.folder / "aggregates"))
        for path in (state.folder / "aggregates").rglob("*")
        if path.is_file()
    }
    assert {name for name in written if name.startswith("posttraining/")} == {
        "posttraining/gru/US/fold-001.npz",
        "posttraining/gru/US/fold-001.integrity.json",
    }
    leftovers = [
        path
        for folder in (state.folder / "aggregates", state.adapters["output"] / "sources")
        for path in folder.rglob(".*")
    ]
    assert leftovers == []
    destination = tmp_path / "published"
    receipt = publish(prepared, state, destination)
    assert_published_from_aggregates(prepared, state, destination, receipt)


def test_a_failed_publication_leaves_no_destination_and_is_repeated(
    prepared, walked, tmp_path, monkeypatch
):
    destination = tmp_path / "published"
    partial = destination.with_name(f".{destination.name}.partial")
    write = compare.write

    def fail(*args, **kwargs):
        raise Cut

    monkeypatch.setattr(compare, "write", fail)
    with pytest.raises(Cut):
        publish(prepared, walked.state, destination)
    assert not destination.exists() and (partial / publication.PARTIAL_MARK).is_file()
    monkeypatch.setattr(compare, "write", write)
    receipt = publish(prepared, walked.state, destination)
    assert_published_from_aggregates(prepared, walked.state, destination, receipt)
    # Una carpeta provisional sin la marca no es nuestra y no se borra.
    other = tmp_path / "other"
    foreign = other.with_name(f".{other.name}.partial")
    foreign.mkdir()
    (foreign / "notes.txt").write_text("ajeno")
    with pytest.raises(ValueError, match="no es una publicación provisional"):
        publish(prepared, walked.state, other)
    assert (foreign / "notes.txt").read_text() == "ajeno" and not other.exists()
    with pytest.raises(ValueError, match="debe ser nueva"):
        publish(prepared, walked.state, destination)


def test_the_walk_with_the_adapter_stage_needs_the_publication_of_that_stage(
    prepared,
    unconsumed,  # noqa: F811
    tmp_path,
):
    base = prepared.base
    path, campaign = schedule(base, tmp_path)
    windows = rolling.load_schedule(path, campaign)
    retention = rolling.load_retention(unconsumed)
    stage = dict(stage=base.stage, output=prepared.adapters)

    def walker(**options):
        return rolling.Rolling(
            retention, base.campaign, base.views, base.output, windows, **options
        )

    with pytest.raises(ValueError, match="van juntas"):
        walker(adapters=stage)
    with pytest.raises(ValueError, match="van juntas"):
        walker(publication=prepared.declared.publication)
    repository = ROOT / "configs/evaluation/historical-masked-publication-a.json"
    with pytest.raises(ValueError, match="declara otra campaña"):
        walker(adapters=stage, publication=repository)
    assert walker(adapters=stage, publication=prepared.declared.publication).publication


# Informe de integridad que bloquea la publicación (#442).


def test_the_publication_carries_a_passed_integrity_report(prepared, walked):
    destination, receipt = walked.destination, walked.receipt
    report = json.loads((destination / gate.REPORT).read_text())
    assert report["passed"] and report["failing"] == []
    assert receipt["integrity"] == dict(
        path=gate.REPORT, sha256=sha256(destination / gate.REPORT), passed=True
    )
    assert set(report["comparisons"]) == {"walk-forward/US", "posttraining/US/gru"}
    for checked in report["comparisons"].values():
        assert checked["from_aggregates"] and checked["declared"]["passed"]
        assert set(checked["windows"]["records"]) <= {"fold-000", "fold-001"}
    runs = report["runs"]
    jobs = plan_campaign(load_campaign(prepared.base.campaign))
    assert runs["campaign"]["planned"] == len(jobs)
    assert runs["campaign"]["missing"] == [] and runs["campaign"]["undeclared"] == []
    # La etapa cuenta sus ajustes, sus padres congelados y la selección de cada ventana.
    assert runs["posttraining"]["planned"] == runs["posttraining"]["confirmed"] > 0
    assert set(report["selection"]) == {"base", "partitions", "chain"}
    assert report["selection"]["base"]["searched_groups"] == 2
    assert report["selection"]["base"]["window_receipts"] == 2
    # Cada ventana del recorrido guardó su registro de integridad antes de liberar.
    ledger = walked.state.ledger()["windows"]
    for window in ("fold-000", "fold-001"):
        record = ledger[window]["phases"]["aggregates"]["written"]["US"]["integrity"]
        stored = json.loads((walked.state.folder / record["path"]).read_text())
        assert stored["passed"] and sha256(walked.state.folder / record["path"]) == record["sha256"]


def test_a_publication_whose_sessions_differ_from_the_aggregates_is_stopped(
    prepared, walked, tmp_path, monkeypatch
):
    original = walk._session_tables

    def shifted(windows, arm, seed):
        # Un solo ulp de diferencia en el MAE de la primera sesión de cada modelo.
        tables = original(windows, arm, seed)
        first = tables[0]
        mae = first.column("mae").to_numpy().copy()
        mae[0] = np.nextafter(mae[0], np.inf)
        tables[0] = first.set_column(first.schema.get_field_index("mae"), "mae", pa.array(mae))
        return tables

    monkeypatch.setattr(walk, "_session_tables", shifted)
    destination = tmp_path / "published"
    with pytest.raises(ValueError, match="La integridad de la publicación falla"):
        publish(prepared, walked.state, destination)
    partial = destination.with_name(f".{destination.name}.partial")
    assert not destination.exists() and (partial / publication.PARTIAL_MARK).is_file()
    report = json.loads((partial / gate.REPORT).read_text())
    assert not report["passed"]
    assert {"comparisons/walk-forward/US", "comparisons/posttraining/US/gru"} <= set(
        report["failing"]
    )


def test_a_window_that_fails_its_recalculation_is_not_released(
    prepared,
    unconsumed,  # noqa: F811
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv(HOLD_ENV, str(prepared.base.hold))
    state = copied(prepared, tmp_path, unconsumed)
    original = forecast_scores._mean

    def biased(panel, values):
        return original(panel, values) * panel.session_samples / (panel.session_samples + 1)

    monkeypatch.setattr(forecast_scores, "_mean", biased)
    with pytest.raises(ValueError, match="La integridad de US fold-000 falla"):
        rolling.run_rolling(state, runners())
    record = json.loads(
        (state.folder / "aggregates" / "US" / "fold-000.integrity.json").read_text()
    )
    assert record["passed"] is False and record["scores"]["failures"] > 0
    # La ventana no consta como agregada y sus filas siguen sin liberar.
    assert not state.done("fold-000", "aggregates")
    assert set(base_tables(state, "fold-000").values()) == {prediction_files.PRESENT}


def test_the_chain_disjunction_blocks_the_publication(prepared, walked, tmp_path, monkeypatch):
    from mars_titan.training import chain_disjunction

    verify = chain_disjunction.verify

    def failing(*args, **kwargs):
        report = verify(*args, **kwargs)
        return dict(report, failures=["US/fold-001: una etiqueta del padre madura tarde"])

    monkeypatch.setattr(chain_disjunction, "verify", failing)
    destination = tmp_path / "published"
    with pytest.raises(ValueError, match="selection/chain"):
        publish(prepared, walked.state, destination)
    assert not destination.exists()


@pytest.fixture
def campaign_copy(prepared, tmp_path):
    """Copia de la salida de la campaña base para alterar recibos sin tocar la original."""
    output = tmp_path / "campaign"
    shutil.copytree(prepared.base.output, output, symlinks=True)
    campaign = load_campaign(prepared.base.campaign)
    return SimpleNamespace(campaign=campaign, jobs=plan_campaign(campaign), output=output)


def _edit(path, change):
    document = json.loads(path.read_text())
    change(document)
    path.write_text(json.dumps(document))


def test_runs_report_every_declared_job_with_its_attempts(campaign_copy):
    copy = campaign_copy
    result = runs.check_output(copy.jobs, copy.output)
    assert result["passed"] and result["planned"] == result["reported"] == len(copy.jobs)
    assert result["unconfirmed_attempts"] == 0
    # Un intento interrumpido se cuenta y no esconde el trabajo.
    job = copy.jobs[0]["id"]
    (copy.output / "jobs" / job / "attempt-0099").mkdir()
    result = runs.check_output(copy.jobs, copy.output)
    assert result["passed"] and result["unconfirmed_attempts"] == 1
    assert next(entry for entry in result["jobs"] if entry["id"] == job)["attempts"] == 2


@pytest.mark.parametrize("defect", ["undeclared", "missing"])
def test_runs_reject_an_undeclared_or_a_missing_job(campaign_copy, defect):
    copy = campaign_copy
    first = copy.output / "jobs" / copy.jobs[0]["id"] / "receipt.json"
    if defect == "undeclared":
        extra = copy.output / "jobs" / "US" / "fold-000" / "gru" / "search-gru-99"
        extra.mkdir(parents=True)
        shutil.copy(first, extra / "receipt.json")
    else:
        first.unlink()
    result = runs.check_output(copy.jobs, copy.output)
    assert not result["passed"]
    if defect == "undeclared":
        assert result["undeclared"] == ["US/fold-000/gru/search-gru-99"]
    else:
        assert result["missing"] == [copy.jobs[0]["id"]]


def test_runs_count_the_chain_selection_of_each_window(prepared, tmp_path):
    adapters = tmp_path / "adapters"
    shutil.copytree(prepared.adapters, adapters, symlinks=True)
    loaded = publication.load_publication(prepared.declared.publication)
    campaign, stage = loaded["campaign_config"], loaded["posttraining"]
    _, result = gate._runs(campaign, prepared.base.output, stage, adapters)
    chains = [entry for entry in result["posttraining"]["jobs"] if entry["status"] == "selected"]
    assert result["posttraining"]["passed"] and chains
    # Una ventana de la cadena sin su selección confirmada falta en el informe.
    selections = sorted(adapters.glob("windows/**/selection.json"))
    assert len(selections) == len(chains)
    selections[-1].unlink()
    _, result = gate._runs(campaign, prepared.base.output, stage, adapters)
    assert not result["posttraining"]["passed"] and len(result["posttraining"]["missing"]) == 1


@pytest.mark.parametrize("defect", ["seed", "window", "model"])
def test_a_published_comparison_must_keep_every_declared_seed_and_window(prepared, defect):
    walked_report, _ = prepared.walk
    report = json.loads(json.dumps(walked_report))
    config = walk.load_config(Path(prepared.base.campaign).parent / "comparison.json")
    primary = engine.write_sources(
        prepared.base.campaign, prepared.base.views, prepared.base.output, "US"
    )
    sources = walk.load_sources(primary, config, "US")
    assert runs.check_published(report, sources)["passed"]
    seeds = report["arms"]["gru"]["seeds"]
    if defect == "seed":
        seeds["43"] = seeds.pop("42")
    elif defect == "window":
        seeds["42"]["windows"].pop("fold-001")
    else:
        report["arms"]["gru_best"] = report["arms"]["gru"]
    result = runs.check_published(report, sources)
    assert not result["passed"] and len(result["differing"]) == 1


def test_selection_follows_the_validation_before_each_window(campaign_copy):
    copy = campaign_copy
    result = selection.check_selection(copy.campaign, copy.jobs, copy.output)
    assert result["passed"], result["failures"]
    # En cada ventana, la búsqueda de gru con menor MAE de validación del doble de la campaña.
    assert result["chosen"] == {
        f"US/{window}/gru": dict(job=f"US/{window}/gru/search-gru-10", searches=2)
        for window in ("fold-000", "fold-001")
    }
    partitions = selection.check_partitions(copy.campaign["comparison_config"], ["US"])
    assert partitions["passed"] and partitions["windows"] == 2


def _search_receipt(copy, name):
    return copy.output / "jobs" / "US" / "fold-000" / "gru" / f"search-{name}" / "receipt.json"


def _window_receipt(copy, window):
    return copy.output / "windows" / "US" / window / "gru" / "seed-42" / "US.json"


@pytest.mark.parametrize(
    ("defect", "message"),
    [
        ("score", "la puntuación del recibo no es la de su validación"),
        ("labels", "La última etiqueta usada debe madurar antes de la evaluación"),
        ("parent", "el recibo no nombra el estado elegido"),
    ],
)
def test_selection_rejects_a_state_chosen_or_declared_against_the_rule(
    campaign_copy, defect, message
):
    copy = campaign_copy
    if defect == "score":
        # El recibo dice que la otra búsqueda ganó, sin que lo diga su validación.
        _edit(_search_receipt(copy, "gru-00"), lambda receipt: receipt.update(score=0.0))
    elif defect == "labels":
        evaluation = copy.campaign["comparison_config"]["resolved_scopes"]["US"]["windows"]
        start = int(np.datetime64(evaluation["fold-001"]["evaluation"][0], "us").astype(np.int64))
        _edit(_window_receipt(copy, "fold-001"), lambda r: r.update(labels_used_until=start))
    else:
        _edit(_window_receipt(copy, "fold-000"), lambda r: r["parent"].update(sha256="0" * 64))
    result = selection.check_selection(copy.campaign, copy.jobs, copy.output)
    assert not result["passed"]
    assert any(message in failure for failure in result["failures"]), result["failures"]


def test_partitions_out_of_order_or_another_evaluated_partition_fail(campaign_copy):
    config = json.loads(json.dumps(campaign_copy.campaign["comparison_config"]))
    window = config["resolved_scopes"]["US"]["windows"]["fold-001"]
    window["calibration"], window["evaluation"] = window["evaluation"], window["calibration"]
    result = selection.check_partitions(config, ["US"])
    assert result["failures"] == ["US/fold-001: tramos fuera de orden"]
    config["partition"] = "calibration"
    assert len(selection.check_partitions(config, ["US"])["failures"]) == 2
