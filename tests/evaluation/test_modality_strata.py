"""Declaración previa de los estratos por presencia de noticias y fundamentales.

Solo se validan configuraciones. No se lee ningún dato, no se ajusta ningún modelo y no se
ejecuta ningún paso de optimizador.
"""

import copy
import json

import pytest

from mars_titan.data.input_policy import STRICT_INPUTS
from mars_titan.evaluation import modality_strata as strata
from mars_titan.evaluation import walk_forward_comparison as walk
from tests.evaluation.test_walk_forward_comparison import CONFIG, Study

SECTION = json.loads(CONFIG.read_text())["modality_strata"]


def declare(study, **changes):
    study.config["schema_version"] = 2
    study.config["modality_strata"] = dict(copy.deepcopy(SECTION), **changes)
    study.publish()
    return study


def test_declared_configuration_is_a_secondary_analysis_fixed_before_results():
    config = walk.load_config(CONFIG)
    section = config["modality_strata"]
    assert config["schema_version"] == 2
    assert section["status"] == "secondary_descriptive" and section["declared_at"] == "2026-10-09"
    assert section["use"] == strata.USE and "not_for_model_selection" in section["use"]
    assert section["strata"] == strata.STRATA and section["focus"] == "news_and_fundamentals"
    assert section["always_present"] == ["prices", "charts", "macro"]
    assert section["metrics"] == ["mae"] and section["recalibrate"] is False
    assert (section["min_rows"], section["min_sessions"]) == (1000, 50)


@pytest.mark.parametrize(
    "edit,message",
    [
        (lambda s: s.pop("min_sessions"), "exactamente sus campos"),
        (lambda s: s.update(extra=1), "exactamente sus campos"),
        (lambda s: s.update(status="primary"), "secundario"),
        (lambda s: s.update(use="model_selection"), "secundario"),
        (lambda s: s.update(declared_at="2026-13-09"), "secundario"),
        (lambda s: s.update(presence_source="predictions"), "secundario"),
        (lambda s: s.update(always_present=["prices", "charts"]), "secundario"),
        (lambda s: s.update(multiplicity="none"), "secundario"),
        (
            lambda s: s["strata"].update(news_only=dict(news=False, fundamentals=True)),
            "cuatro patrones",
        ),
        (lambda s: s["strata"].pop("neither"), "cuatro patrones"),
        (lambda s: s.update(focus="all"), "cuatro patrones"),
        (lambda s: s.update(metrics=["mse"]), "empezar por el MAE"),
        (lambda s: s.update(metrics=["mae", "mae"]), "empezar por el MAE"),
        (lambda s: s.update(metrics=["mae", "coverage"]), "empezar por el MAE"),
        (lambda s: s.update(recalibrate=True), "volver a calibrar"),
        (lambda s: s.update(min_rows=0), "min_rows"),
        (lambda s: s.update(min_rows=True), "min_rows"),
        (lambda s: s.update(min_sessions=2.5), "min_sessions"),
    ],
)
def test_declaration_rejects_any_change_to_its_contract(edit, message):
    section = copy.deepcopy(SECTION)
    edit(section)
    with pytest.raises(ValueError, match=message):
        strata.declaration(section, walk.SERIES_METRICS)


@pytest.mark.parametrize(
    "edit,message",
    [
        (lambda c: c.pop("modality_strata"), "contrato"),
        (lambda c: c.update(schema_version=1), "contrato"),
        (lambda c: c.update(schema_version=True), "contrato"),
        (lambda c: c.update(input_policy=STRICT_INPUTS), "máscaras"),
    ],
)
def test_configuration_versions_require_their_own_fields(tmp_path, edit, message):
    study = declare(Study(tmp_path, scope="US"))
    edit(study.config)
    study.publish()
    with pytest.raises(ValueError, match=message):
        walk.load_config(study.config_path)
