"""Declaración del paso final: cada campaña publica la matriz y la etapa de su comparación.

Solo se cargan declaraciones y configuraciones. No se lee ninguna predicción ni se ajusta
nada. El recorrido completo, desde los agregados por ventana, está en
`tests/posttraining/test_publication_from_aggregates.py`.
"""

import importlib
import json
import runpy
from pathlib import Path

import pytest

from mars_titan.data.storage import atomic_json
from mars_titan.evaluation import comparison_matrix
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
# Modelos de integración que la matriz declara y la comparación conjunta todavía no tiene.
PENDING_IN_V2 = {"mars_titan_b6", "mars_titan_b6_bias", "mars_titan_m1_k4_first_read"}


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
    assert (summary["families"], summary["contrasts"]) == (47, 356)
    assert loaded["final_test_opened"] is False


def test_the_v2_matrix_leaves_pending_the_arms_its_joint_comparison_lacks():
    loaded = publication.load_publication(DECLARATIONS["A_V2"])
    arms = set(loaded["campaign_config"]["comparison_config"]["arms"])
    conditional = loaded["matrix"]["conditional_arms"]
    assert PENDING_IN_V2 <= set(conditional) and not PENDING_IN_V2 & arms
    for arm in PENDING_IN_V2:
        assert conditional[arm]["issue"] == 437
        assert "comparación conjunta de A v2 todavía no declara" in conditional[arm]["condition"]
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
    assert len(result["posttraining"]["parents"]) == 20


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
