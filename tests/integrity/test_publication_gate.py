"""Conexión del informe de integridad con la publicación, el recorrido y la etapa (#442).

Las pruebas de extremo a extremo de `tests/posttraining/test_publication_from_aggregates.py`
recorren la campaña reducida y solo corren con la protección de aprendizaje levantada. Estas
comprueban la misma conexión sin ejecutar ninguna campaña: las comprobaciones costosas se
sustituyen por resultados conocidos y la etapa de adaptadores es la de `Stage`, con recibos y
predicciones escritos en la prueba. No hay modelos ajustados ni pasos de optimizador.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mars_titan.data.storage import sha256
from mars_titan.evaluation import comparison_matrix, long_short_comparison
from mars_titan.evaluation import session_table_contrasts as tables
from mars_titan.evaluation import walk_forward_comparison as walk
from mars_titan.integrity import publication_gate as gate
from mars_titan.posttraining import campaign_stage
from mars_titan.posttraining import stage_comparison as compare
from mars_titan.training import campaign_publication as publication
from mars_titan.training import masked_campaign as engine
from mars_titan.training import rolling_retention as rolling
from tests.integrity.test_comparison_integrity import biased_mean
from tests.posttraining.test_stage_comparison import Stage

ROOT = Path(__file__).resolve().parents[2]
DECLARATION = ROOT / "configs/evaluation/historical-masked-publication-a-v2.json"
CHAIN_FAILURE = "US/fold-001: una etiqueta del padre madura tarde"


def _loaded(design=None, stages=False):
    campaign = dict(name="campaign", sha256="1" * 64, comparison_config={}, scopes=["US"])
    if stages:
        campaign["walk_forward_stages"] = dict(windows=2)
    posttraining = None if design is None else dict(stage=dict(design=design))
    return dict(
        name="publication",
        sha256="2" * 64,
        campaign_config=campaign,
        posttraining=posttraining,
        budget_differences={},
    )


@pytest.fixture
def budget_result(monkeypatch):
    """Presupuesto superado que una prueba puede cambiar antes de construir el informe."""
    state = SimpleNamespace(passed=True, calls=[])

    def check(campaign, jobs, output, declarations):
        state.calls.append(declarations)
        return dict(kind="fit_budget", passed=state.passed)

    monkeypatch.setattr(gate.fit_budget, "check", check)
    return state


@pytest.fixture
def chain_calls(monkeypatch, budget_result):
    """Secciones superadas salvo la disjunción de la cadena, que registra sus llamadas."""
    calls = []

    def runs(campaign, output, posttraining, adapter_output):
        return [], dict(campaign=dict(passed=True))

    def chain(campaign, views, output, posttraining):
        calls.append(posttraining)
        return dict(selections=1, scopes=["US"], failures=[CHAIN_FAILURE], passed=False)

    monkeypatch.setattr(gate, "_runs", runs)
    monkeypatch.setattr(gate.window_selection, "check_selection", lambda *a: dict(passed=True))
    monkeypatch.setattr(gate.window_selection, "check_partitions", lambda *a: dict(passed=True))
    monkeypatch.setattr(gate.window_selection, "check_chain", chain)
    return calls


@pytest.mark.parametrize(
    ("design", "stages", "checked"),
    [
        (campaign_stage.STAGED, False, True),
        (None, True, True),
        (campaign_stage.ANCHORED, False, False),
        (None, False, False),
    ],
    ids=["staged-stage", "staged-campaign", "anchored-stage", "no-stage"],
)
def test_the_staged_design_adds_the_chain_disjunction_to_the_report(
    chain_calls, tmp_path, design, stages, checked
):
    adapters = None if design is None else tmp_path / "adapters"
    report = gate.build(_loaded(design, stages), {}, tmp_path, [], adapter_output=adapters)
    assert ("chain" in report["selection"]) is checked
    assert report["failing"] == (["selection/chain"] if checked else [])
    assert report["passed"] is not checked and report["training_executed"] is False
    # Sin etapa publicada, la disjunción solo recorre la campaña.
    assert chain_calls == ([adapters] if checked else [])


def test_the_budget_of_the_campaign_is_one_more_section_that_blocks(
    chain_calls, budget_result, tmp_path
):
    loaded = _loaded()
    assert gate.build(loaded, {}, tmp_path, [])["budget"]["campaign"]["passed"]
    assert budget_result.calls == [loaded["budget_differences"]]
    budget_result.passed = False
    report = gate.build(loaded, {}, tmp_path, [])
    assert report["failing"] == ["budget/campaign"] and not report["passed"]


def test_a_failed_report_is_kept_and_stops_the_publication(chain_calls, tmp_path):
    report = gate.build(_loaded(campaign_stage.STAGED), {}, tmp_path, [], adapter_output=tmp_path)
    folder = tmp_path / "partial"
    folder.mkdir()
    with pytest.raises(
        ValueError, match="La integridad de la publicación falla en selection/chain"
    ):
        gate.write(folder, report)
    assert json.loads((folder / gate.REPORT).read_text())["failing"] == ["selection/chain"]
    passed = dict(report, failing=[], passed=True)
    assert gate.write(folder, passed) == dict(
        path=folder / gate.REPORT, sha256=sha256(folder / gate.REPORT)
    )


@pytest.fixture
def stubbed_publication(monkeypatch):
    """Publicación de A v2 con cada comparación sustituida por un archivo vacío.

    Solo queda real el orden de `run_publication`: carpeta provisional, informe de
    integridad y nombre del destino. `report` es el informe que devolverá el gate.
    """
    state = SimpleNamespace(report=None, built=0, comparisons=None)

    def walk_forward(config, sources, scope, folder, **options):
        Path(folder).mkdir(parents=True)
        (Path(folder) / "comparison.json").write_text("{}")

    def build(loaded, views, output, comparisons, *, adapter_output=None):
        state.built += 1
        state.comparisons = comparisons
        assert {entry["label"] for entry in comparisons} >= {"walk-forward/US+CN"}
        return state.report

    monkeypatch.setattr(engine, "write_sources", lambda *a, **k: Path("sources.json"))
    monkeypatch.setattr(walk, "write_walk_forward", walk_forward)
    monkeypatch.setattr(long_short_comparison, "write_long_short", lambda *a, **k: None)
    monkeypatch.setattr(tables, "write_sources", lambda path, *a, **k: path)
    monkeypatch.setattr(comparison_matrix, "write", lambda *a, **k: dict(families={}))
    monkeypatch.setattr(compare, "write_sources", lambda *a, **k: Path("stage.json"))
    monkeypatch.setattr(compare, "write", lambda *a, **k: dict(windows=[]))
    monkeypatch.setattr(gate, "build", build)
    return state


def _publish(tmp_path, destination):
    # A v2 declara la retención e interferencia, que pide el calendario de regímenes.
    calendar = tmp_path / "regimes.json"
    calendar.write_text("{}")
    return publication.run_publication(
        DECLARATION,
        {},
        tmp_path / "campaign",
        destination,
        adapter_output=tmp_path / "adapters",
        edition=tmp_path / "edition",
        regimes=calendar,
    )


def test_the_publication_stops_on_a_failed_report_and_carries_a_passed_one(
    stubbed_publication, tmp_path
):
    destination = tmp_path / "published"
    stubbed_publication.report = dict(passed=False, failing=["runs/campaign"])
    with pytest.raises(ValueError, match="La integridad de la publicación falla en runs/campaign"):
        _publish(tmp_path, destination)
    partial = destination.with_name(f".{destination.name}.partial")
    assert not destination.exists() and (partial / publication.PARTIAL_MARK).is_file()
    assert json.loads((partial / gate.REPORT).read_text())["passed"] is False
    # Cada comparación walk-forward llega al informe con la matriz que la lee.
    walked = [e for e in stubbed_publication.comparisons if e["label"].startswith("walk-forward/")]
    assert len(walked) == 3 and all(
        entry["matrix"] == partial / "matrix" / entry["scope"] / "matrix.json" for entry in walked
    )
    stubbed_publication.report = dict(passed=True, failing=[])
    receipt = _publish(tmp_path, destination)
    assert stubbed_publication.built == 2
    assert receipt["integrity"] == dict(
        path=gate.REPORT, sha256=sha256(destination / gate.REPORT), passed=True
    )
    assert receipt["artifacts"][gate.REPORT] == receipt["integrity"]["sha256"]


def test_the_walk_checks_each_window_before_recording_its_aggregates(tmp_path, monkeypatch):
    """Fase de agregados del recorrido con fuentes y comprobación sustituidas."""
    checked = []

    def write_checked(folder, config, sources, window, ablation=None, *, liquidity):
        checked.append((window, liquidity))
        raise ValueError(f"La integridad de US {window} falla: revisa el registro")

    def unchecked(*args, **kwargs):
        raise AssertionError("Los agregados se guardan sin su registro de integridad")

    monkeypatch.setattr(engine, "write_sources", lambda *a, **k: tmp_path / "sources.json")
    monkeypatch.setattr(rolling.comparison, "restrict_windows", lambda config, *a: config)
    monkeypatch.setattr(rolling.comparison, "load_sources", lambda *a: dict(scope="US"))
    monkeypatch.setattr(rolling.comparison, "scope_config", lambda config, scope: config)
    monkeypatch.setattr(rolling.window_aggregates, "write", unchecked)
    monkeypatch.setattr(rolling.comparison_integrity, "write_checked", write_checked)
    # El efecto de la selección no interviene en la integridad y se sustituye.
    monkeypatch.setattr(engine, "search_records", lambda *a, **k: (None, {}))
    monkeypatch.setattr(
        rolling.selection_effect,
        "write_window",
        lambda *a, **k: dict(path="a.selection.json", sha256="c"),
    )
    walker = SimpleNamespace(
        comparison_config=dict,
        folder=tmp_path,
        campaign_path=tmp_path / "campaign.json",
        views={},
        output=tmp_path / "out",
        ablation=None,
        edition=None,
        unrecorded_platform=None,
        _liquidity=None,
        _record=lambda record: record,
        posttraining_aggregates=lambda *a: {},
    )
    row = dict(id="fold-000", scopes={"US": "fold-000"})
    with pytest.raises(ValueError, match="La integridad de US fold-000 falla"):
        rolling.Rolling.aggregates(walker, row)
    # Sin estratos de liquidez declarados, el registro se guarda sin asignador.
    assert checked == [("fold-000", None)]
    monkeypatch.setattr(
        rolling.comparison_integrity,
        "write_checked",
        lambda *a, **k: (dict(path="a.npz", sha256="a"), dict(path="a.integrity.json", sha256="b")),
    )
    written = rolling.Rolling.aggregates(walker, row)
    assert written["US"]["integrity"] == dict(path="a.integrity.json", sha256="b")
    assert set(written["US"]) == {"walk_forward", "integrity", "selection_effect"}


def test_the_stage_checks_each_window_before_its_rows_are_released(tmp_path, monkeypatch):
    """Etapa de `Stage`: recibos y predicciones escritos en la prueba, sin ejecutar nada."""
    stage = Stage(tmp_path / "stage")

    def aggregate(root):
        return compare.write_window_aggregates(
            stage.declaration,
            "US",
            "gru",
            "fold-001",
            base_sources=stage.window_base_sources("fold-001"),
            stage_output=stage.output,
            root=root,
        )

    written = aggregate(tmp_path / "aggregates")
    folder = compare.aggregates_folder(tmp_path / "aggregates", "gru") / "US"
    assert written["integrity"] == dict(
        path=folder / "fold-001.integrity.json",
        sha256=sha256(folder / "fold-001.integrity.json"),
        passed=True,
    )
    assert json.loads((folder / "fold-001.integrity.json").read_text())["passed"]
    biased_mean(monkeypatch)
    with pytest.raises(ValueError, match="La integridad de US fold-001 falla"):
        aggregate(tmp_path / "biased")
    record = (
        compare.aggregates_folder(tmp_path / "biased", "gru") / "US" / "fold-001.integrity.json"
    )
    assert json.loads(record.read_text())["passed"] is False
