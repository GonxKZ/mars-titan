"""Declaración del paso final: cada campaña publica la matriz y la etapa de su comparación.

Solo se cargan declaraciones y configuraciones, salvo en la retención e interferencia, que
publica la comparación de un estudio sintético con predicciones escritas en la prueba. No se
ajusta nada. El recorrido completo, desde los agregados por ventana, está en
`tests/posttraining/test_publication_from_aggregates.py`.
"""

import importlib
import json
import runpy
from pathlib import Path

import pytest

from mars_titan.data.storage import atomic_json, sha256
from mars_titan.evaluation import comparison_matrix
from mars_titan.evaluation import walk_forward_comparison as walk
from mars_titan.training import campaign_plan as plan
from mars_titan.training import campaign_publication as publication

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "configs"
DECLARATIONS = dict(
    A=CONFIGS / "evaluation/historical-masked-publication-a.json",
    A_V2=CONFIGS / "evaluation/historical-masked-publication-a-v2.json",
)
EXPECTED = dict(
    A=dict(
        campaign="baselines/historical-masked-campaign-a.json",
        comparison="evaluation/historical-masked-2000-comparison.json",
        matrix="evaluation/comparison-matrix-a.json",
        posttraining="posttraining/historical-masked-adapter-comparison-a.json",
    ),
    A_V2=dict(
        campaign="baselines/historical-masked-campaign-a-v2.json",
        comparison="evaluation/historical-masked-2000-joint-comparison.json",
        matrix="evaluation/comparison-matrix-a-v2.json",
        posttraining="posttraining/historical-masked-adapter-comparison-a-v2.json",
    ),
)
# Modelos de integración que la comparación conjunta de A v2 declara desde #525.
INTEGRATION_ARMS = {
    "mars_titan_b6",
    "mars_titan_b6_bias",
    "mars_titan_m1_k4_first_read",
    "mars_titan_b6_regime",
    "mars_titan_b6_calendar",
    "mars_titan_b6_regime_banks",
    "mars_titan_b6_calendar_banks",
}


@pytest.fixture(scope="module", params=sorted(DECLARATIONS))
def declared(request):
    return request.param, publication.load_publication(DECLARATIONS[request.param])


def test_each_campaign_publishes_the_matrix_and_the_stage_of_its_own_comparison(declared):
    variant, loaded = declared
    expected = {key: (CONFIGS / value).resolve() for key, value in EXPECTED[variant].items()}
    campaign = loaded["campaign_config"]
    assert Path(campaign["path"]) == expected["campaign"]
    assert Path(campaign["comparison_path"]) == expected["comparison"]
    assert Path(loaded["matrix_path"]) == expected["matrix"]
    matrix = loaded["matrix"]
    # La matriz lee la misma comparación que la campaña, con la misma huella.
    assert (expected["matrix"].parent / matrix["comparison"]).resolve() == expected["comparison"]
    assert matrix["comparison_config"]["sha256"] == campaign["comparison_config"]["sha256"]
    assert Path(loaded["posttraining_path"]) == expected["posttraining"]
    stage = loaded["posttraining"]["stage"]
    assert stage["campaign"]["sha256"] == campaign["sha256"]
    summary = comparison_matrix.summary(matrix)
    assert (summary["families"], summary["contrasts"]) == (47, 362)
    assert loaded["final_test_opened"] is False


def test_the_v2_matrix_reads_the_integration_arms_from_its_joint_comparison():
    loaded = publication.load_publication(DECLARATIONS["A_V2"])
    arms = set(loaded["campaign_config"]["comparison_config"]["arms"])
    matrix = loaded["matrix"]
    # Ya no quedan condicionados: sus contrastes se estiman con la campaña.
    assert INTEGRATION_ARMS <= arms and not INTEGRATION_ARMS & set(matrix["conditional_arms"])
    assert set(matrix["conditional_arms"]) == {"titans_reference_mac"}
    assert {matrix["arms"].kind(arm) for arm in INTEGRATION_ARMS} == {"declared"}
    # El resto de la matriz es la de A: mismas preguntas, linajes y vistas.
    matrix_a = json.loads((CONFIGS / EXPECTED["A"]["matrix"]).read_text())
    matrix_v2 = json.loads((CONFIGS / EXPECTED["A_V2"]["matrix"]).read_text())
    changed = {key for key in matrix_a if matrix_a[key] != matrix_v2[key]}
    assert changed == {"name", "comparison", "conditional_arms"}


def rewrite(tmp_path, variant, **changes):
    """Copia de una declaración con rutas absolutas y los campos cambiados."""
    source = DECLARATIONS[variant]
    document = json.loads(source.read_text())
    for key in ("campaign", "comparison_matrix", "posttraining_comparison"):
        document[key] = str((source.parent / document[key]).resolve())
    for key, value in changes.items():
        document[key] = value if value is None else str((CONFIGS / value).resolve())
    path = tmp_path / "publication.json"
    atomic_json(path, document)
    return path


def test_a_matrix_of_another_campaign_is_rejected(tmp_path):
    path = rewrite(tmp_path, "A_V2", comparison_matrix=EXPECTED["A"]["matrix"])
    with pytest.raises(ValueError, match="no lee la comparación de la campaña"):
        publication.load_publication(path)


def test_a_posttraining_comparison_of_another_campaign_is_rejected(tmp_path):
    path = rewrite(tmp_path, "A", posttraining_comparison=EXPECTED["A_V2"]["posttraining"])
    with pytest.raises(ValueError, match="no deriva de una etapa de la campaña"):
        publication.load_publication(path)


@pytest.mark.parametrize(
    "edit",
    [
        dict(status="draft"),
        dict(final_test_opened=True),
        dict(kind="comparison_matrix"),
        dict(extra=1),
        dict(name="no válido"),
    ],
    ids=["status", "final_test", "kind", "extra_field", "name"],
)
def test_the_declaration_is_fixed_before_results(tmp_path, edit):
    path = rewrite(tmp_path, "A")
    document = json.loads(path.read_text())
    document.update(edit)
    atomic_json(path, document)
    with pytest.raises(ValueError, match="no cumple su contrato"):
        publication.load_publication(path)


def test_without_posttraining_comparison_only_the_campaign_is_published(tmp_path):
    loaded = publication.load_publication(rewrite(tmp_path, "A", posttraining_comparison=None))
    assert loaded["posttraining"] is None and loaded["posttraining_path"] is None


def test_the_campaign_script_checks_the_declaration_without_reading_predictions(capsys):
    script = runpy.run_path(str(ROOT / "scripts/run_masked_campaign.py"), run_name="script")
    arguments = ["publication", "check", "--declaration", str(DECLARATIONS["A_V2"])]
    assert script["main"](arguments) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "checked" and result["final_test_opened"] is False
    campaign = plan.load_campaign(CONFIGS / EXPECTED["A_V2"]["campaign"])
    assert result["campaign"] == dict(name=campaign["name"], sha256=campaign["sha256"])
    assert result["matrix"]["families"] == 47
    assert result["posttraining"]["scopes"] == ["US+CN"]
    # Desde #525 entra mars_titan_m1_k4_first_read. Los ocho B6 y las dos extensiones PT1
    # solo trasladan el padre congelado, sin adaptadores, así que no tienen comparación
    # postentrenada.
    parents = result["posttraining"]["parents"]
    assert len(parents) == 21 and "mars_titan_m1_k4_first_read" in parents
    assert not {
        arm for arm in parents if arm.startswith(("mars_titan_b6", "titans_mac_online_pt1"))
    }


def test_the_campaign_plan_registers_the_publication_as_its_last_stage():
    declared = plan.LATER_STAGES["campaign_publication"]
    assert declared["issue"] == 28 and declared["pending"] == []
    assert Path(declared["config"]).resolve() == (CONFIGS / EXPECTED["A"]["matrix"]).resolve()
    assert Path(declared["stages"]["A"]).resolve() == DECLARATIONS["A"]
    assert Path(declared["joint_stage"]).resolve() == DECLARATIONS["A_V2"]
    module, _, function = declared["entry"].partition(":")
    assert getattr(importlib.import_module(module), function) is publication.run_publication
    # Es la última entrada: se ejecuta cuando todas las etapas están confirmadas.
    assert list(plan.LATER_STAGES)[-1] == "campaign_publication"


# Retención e interferencia en las revisitas de régimen (#528)


def test_both_campaigns_need_the_calendar_and_edition_their_comparison_declares(
    tmp_path, monkeypatch
):
    destination = tmp_path / "published"
    for variant in sorted(DECLARATIONS):
        config = publication.load_publication(DECLARATIONS[variant])["campaign_config"]
        assert walk.RETENTION_FIELD in config["comparison_config"]
    with pytest.raises(ValueError, match="falta el calendario de regímenes"):
        publication.run_publication(
            DECLARATIONS["A_V2"],
            {},
            tmp_path / "campaign",
            destination,
            adapter_output=tmp_path / "adapters",
            edition=tmp_path / "edition",
        )
    # Se detiene antes de crear la carpeta provisional.
    assert list(tmp_path.iterdir()) == []
    # Una comparación sin la sección no admite un calendario.
    loaded = publication.load_publication(DECLARATIONS["A_V2"])
    campaign = loaded["campaign_config"]
    plain = {k: v for k, v in campaign["comparison_config"].items() if k != walk.RETENTION_FIELD}
    changed = dict(loaded, campaign_config=dict(campaign, comparison_config=plain))
    monkeypatch.setattr(publication, "load_publication", lambda path: changed)
    with pytest.raises(ValueError, match="sobra el calendario"):
        publication.run_publication(
            DECLARATIONS["A_V2"],
            {},
            tmp_path / "campaign",
            destination,
            adapter_output=tmp_path / "adapters",
            edition=tmp_path / "edition",
            regimes=tmp_path / "calendar.json",
        )
    # Una comparación con estratos de liquidez y sin cartera también pide la edición.
    stratified = {
        k: v for k, v in campaign["comparison_config"].items() if k != walk.LONG_SHORT_FIELD
    }
    assert walk.LIQUIDITY_FIELD in stratified
    changed = dict(loaded, campaign_config=dict(campaign, comparison_config=stratified))
    monkeypatch.setattr(publication, "load_publication", lambda path: changed)
    with pytest.raises(ValueError, match="estratos de liquidez: falta la edición"):
        publication.run_publication(
            DECLARATIONS["A_V2"],
            {},
            tmp_path / "campaign",
            destination,
            adapter_output=tmp_path / "adapters",
            regimes=tmp_path / "calendar.json",
        )
    assert not destination.exists()


@pytest.fixture(scope="module")
def retention_study(tmp_path_factory):
    """Estudio sintético con la sección de retención, su calendario y sus agregados."""
    from mars_titan.evaluation import window_aggregates
    from tests.evaluation import test_walk_forward_comparison as base
    from tests.evaluation.test_walk_forward_predictive_ability import long_decisions
    from tests.evaluation.test_walk_forward_retention import section, write_calendar

    root = tmp_path_factory.mktemp("retention")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(base, "decisions", long_decisions)
        study = base.Study(root / "study")
        study.config[walk.RETENTION_FIELD] = section()
        study.publish()
    calendar = write_calendar(root / "regimes" / "calendar.json")
    config = walk.load_config(study.config_path)
    sources = walk.load_sources(study.sources_path, config, study.scope)
    aggregates = root / "aggregates"
    for window in sources["windows"]:
        window_aggregates.write(aggregates, config, sources, window)
    rows, _ = walk.evaluate_walk_forward(
        study.config_path, study.sources_path, study.scope, regimes=calendar
    )
    return study, calendar, aggregates, rows


def test_the_publication_from_aggregates_carries_the_complete_retention_section(
    retention_study, tmp_path, monkeypatch
):
    """`run_publication` de A v2 con la comparación del estudio sintético.

    La campaña reducida solo se recorre rebajando la protección de aprendizaje. Aquí la
    publicación conserva su orden y sus argumentos, y cada comparación walk-forward se
    calcula de verdad, desde los agregados y con el calendario que recibe, sobre el estudio.
    Las demás piezas (fuentes, cartera, matriz y etapa) se sustituyen por archivos vacíos, y
    el informe de integridad por uno superado (sus pruebas están en `tests/integrity`).
    """
    from mars_titan.evaluation import long_short_comparison
    from mars_titan.evaluation import session_table_contrasts as tables
    from mars_titan.integrity import publication_gate
    from mars_titan.posttraining import stage_comparison
    from mars_titan.training import masked_campaign as engine

    study, calendar, aggregates, rows = retention_study
    calls = []
    real = walk.write_walk_forward

    def walk_forward(config_path, sources_path, scope, output, **options):
        calls.append(options)
        return real(study.config_path, study.sources_path, study.scope, output, **options)

    monkeypatch.setattr(walk, "write_walk_forward", walk_forward)
    monkeypatch.setattr(engine, "write_sources", lambda *a, **k: study.sources_path)
    monkeypatch.setattr(long_short_comparison, "write_long_short", lambda *a, **k: None)
    monkeypatch.setattr(tables, "write_sources", lambda path, *a, **k: path)
    monkeypatch.setattr(comparison_matrix, "write", lambda *a, **k: dict(families={}))
    monkeypatch.setattr(stage_comparison, "write_sources", lambda *a, **k: study.sources_path)
    parents = []

    def parent(*args, **options):
        parents.append(options)
        return dict(windows=[])

    monkeypatch.setattr(stage_comparison, "write", parent)
    monkeypatch.setattr(publication_gate, "build", lambda *a, **k: dict(passed=True, failing=[]))
    destination = tmp_path / "published"
    receipt = publication.run_publication(
        DECLARATIONS["A_V2"],
        {},
        tmp_path / "campaign",
        destination,
        adapter_output=tmp_path / "adapters",
        aggregates=aggregates,
        edition=tmp_path / "edition",
        regimes=calendar,
    )
    scopes = publication.load_publication(DECLARATIONS["A_V2"])["campaign_config"]["scopes"]
    edition = tmp_path / "edition"
    assert len(calls) == len(scopes) and all(
        options["aggregates"] == aggregates
        and options["regimes"] == calendar
        and options["edition"] == edition
        for options in calls
    )
    # Cada padre recibe la edición para sus estratos de liquidez heredados.
    assert parents and all(
        options["aggregates"] == aggregates and options["edition"] == edition for options in parents
    )
    expected = rows[walk.RETENTION_FIELD]
    assert expected["status"] == "computed"
    for scope in scopes:
        report = json.loads((destination / "walk-forward" / scope / "comparison.json").read_text())
        assert report[walk.RETENTION_FIELD] == expected
    assert receipt["from_aggregates"] is True
    assert receipt["regimes"] == dict(sha256=sha256(calendar))
