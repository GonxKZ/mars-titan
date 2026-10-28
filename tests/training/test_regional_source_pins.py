"""Fuentes de C1/C6 y rechazo de su congelación sin registrar, sin crear entrenadores."""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from mars_titan.models.titans.regional_fusion import RegionalFusionConfig
from mars_titan.models.titans.regional_variants import (
    RegionAxisMixingConfig,
    SharedWorkspaceConfig,
)
from mars_titan.training import financial_run, mars_titan_walk_forward
from mars_titan.training.financial_run import ChronologicalRecipe, ChronologicalTrainer
from tests.models.titans.test_financial_adapter import specification

MODULE = "mars_titan.models.titans.regional_variants"
OPERATOR = RegionalFusionConfig(topology="ring")
VARIANTS = (
    RegionAxisMixingConfig(OPERATOR),
    RegionAxisMixingConfig(OPERATOR, mixing="none"),
    SharedWorkspaceConfig(OPERATOR),
    SharedWorkspaceConfig(OPERATOR, write="dense", write_top_k=None),
)


def source_reader(variant):
    # Solo _code y _check_runtime. No se ejecuta __init__ ni se crea un optimizador.
    reader = object.__new__(ChronologicalTrainer)
    reader.predictor = SimpleNamespace(
        config=SimpleNamespace(
            temporal_encoder=None,
            regional_variant=variant,
            cross_asset_attention=None,
        )
    )
    # La receta decide si F2 y F3 añaden su código.
    reader.recipe = financial_run.ChronologicalRecipe()
    return reader


@pytest.mark.parametrize("variant", VARIANTS)
def test_active_regional_operator_source_is_included_in_the_training_identity(variant):
    expected = financial_run.importlib.import_module(MODULE).__file__
    sources = source_reader(variant)._code()
    assert MODULE in sources
    assert sources[MODULE] == hashlib.sha256(Path(expected).read_bytes()).hexdigest()


def test_disabled_operator_does_not_add_a_regional_source():
    assert MODULE not in source_reader(None)._code()


def test_runtime_rejects_a_changed_active_operator_file_without_executing_it(tmp_path, monkeypatch):
    path = tmp_path / "operator.py"
    path.write_text("first source snapshot\n")
    original = financial_run.importlib.import_module

    def imported(name, *args):
        return SimpleNamespace(__file__=path) if name == MODULE else original(name, *args)

    monkeypatch.setattr(financial_run.importlib, "import_module", imported)
    reader = source_reader(VARIANTS[0])
    reader.recipe = ChronologicalRecipe()
    reader.kernel_policy = None
    reader.identity = dict(implementation=reader._code(), numerics=financial_run._numerics())
    reader._check_runtime()
    path.write_text("a different operator implementation\n")
    with pytest.raises(ValueError, match="código"):
        reader._check_runtime()


@pytest.mark.parametrize("variant", VARIANTS)
def test_frozen_loader_refuses_unregistered_regional_variants_before_build_or_read(
    tmp_path, monkeypatch, variant
):
    document = dict(predictor=dict(hidden_size=32, dtype="float64"))
    report = dict(request=dict(predictor_changes=dict(regional_variant=asdict(variant))))

    def forbidden(*_args, **_options):
        pytest.fail("Se construyó o leyó un padre regional sin registro congelado")

    monkeypatch.setattr(mars_titan_walk_forward, "FinancialPredictor", forbidden)
    monkeypatch.setattr(mars_titan_walk_forward, "read_manifest", forbidden)
    with pytest.raises(ValueError, match="regional"):
        mars_titan_walk_forward._frozen_parent(
            document, specification(), 42, "cpu", tmp_path, report
        )


@pytest.mark.parametrize("in_recipe", (False, True))
def test_parent_declaration_is_refused_before_checkpoint_verification(
    tmp_path, monkeypatch, in_recipe
):
    view = tmp_path / "view.json"
    view.write_text("{}\n")
    regional = asdict(VARIANTS[0])
    predictor = dict(hidden_size=32, dtype="float64")
    request = dict(
        variant="mac_online", seed=42, view_sha256=hashlib.sha256(view.read_bytes()).hexdigest()
    )
    if in_recipe:
        predictor["regional_variant"] = regional
    else:
        request["predictor_changes"] = dict(regional_variant=regional)
    report = dict(
        kind=mars_titan_walk_forward.TITANS_KIND,
        status="completed",
        final_test_opened=False,
        checkpoint={},
        request=request,
        identity=dict(recipe=dict(predictor=predictor)),
    )
    (tmp_path / "run.json").write_text(json.dumps(report))
    monkeypatch.setattr(
        mars_titan_walk_forward,
        "_verify",
        lambda *_a: pytest.fail(
            "Se verificó el checkpoint de un consumidor regional sin registrar"
        ),
    )
    with pytest.raises(ValueError, match="regional"):
        mars_titan_walk_forward._parent(tmp_path, view, 42)
