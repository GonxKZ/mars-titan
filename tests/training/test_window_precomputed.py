"""Admisión y procedencia de precomputación, sin abrir datos ni crear optimizadores."""

import hashlib
import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest

KINDS = ["kronlstm_precomputed_v1"]


@pytest.mark.parametrize("kind", KINDS)
def test_precomputed_kind_reaches_the_existing_financial_and_window_factories(kind):
    from mars_titan.training import window_variant_run as window
    from mars_titan.training.search_cases import checked_predictor_changes
    from tests.models.titans.test_financial_adapter import raw_batch, specification
    from tests.training.test_window_variants import recipe

    assert checked_predictor_changes({"temporal_encoder": kind}) == {"temporal_encoder": kind}
    options = recipe(temporal_encoder=kind, loss="mae")
    spec = specification()
    dataset = SimpleNamespace(
        temporal=SimpleNamespace(contract={"parent_sha256": spec.source_sha256}),
        identity=spec.view_sha256,
        manifest={"representation": spec.representation},
    )
    model = window._predictor(dataset, raw_batch(), options, "cpu")
    assert model.config.temporal_encoder == kind
    assert model.price_encoder.configuration["backend"] == "pytorch_precomputed_cpu_qkv64_v1"
    assert model.price_encoder.configuration["rule"] == (
        "scalar" if kind.startswith("kronlstm") else "vector"
    )
    sources = window._sources(options)
    name = "models/kronlstm/precomputed.py"
    assert name in sources
    source = Path(window.__file__).resolve().parents[1] / name
    assert sources[name] == hashlib.sha256(source.read_bytes()).hexdigest()


@pytest.mark.parametrize("kind", [None, "kronlstm_v1", "vector_delta_v1", *KINDS])
def test_frozen_and_window_source_manifests_identify_the_optional_scan(kind):
    from mars_titan.models.titans.frozen_financial import _implementation
    from mars_titan.training import window_variant_run as window
    from tests.training.test_window_variants import recipe

    active = kind is not None
    module_name = "mars_titan.models.kronlstm.precomputed"
    frozen = _implementation(kronlstm=active)
    assert (module_name in frozen) is active
    sources = window._sources(recipe(temporal_encoder=kind))
    assert ("models/kronlstm/precomputed.py" in sources) is active
    if active:
        module = importlib.import_module(module_name)
        assert frozen[module_name] == hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()


@pytest.mark.parametrize("kind", KINDS)
def test_trainers_inherit_the_real_precomputed_source_without_initializing_training(kind):
    from mars_titan.posttraining.readout_adapters import ReadoutAdapterTrainer
    from mars_titan.training.financial_run import ChronologicalRecipe, ChronologicalTrainer
    from mars_titan.training.mars_titan_run import ReadoutTrainer

    module = importlib.import_module("mars_titan.models.kronlstm.precomputed")
    expected = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
    for trainer_type in (ChronologicalTrainer, ReadoutTrainer, ReadoutAdapterTrainer):
        trainer = object.__new__(trainer_type)
        # `_code` lee estos campos de la configuración real del predictor.
        config = SimpleNamespace(
            temporal_encoder=kind,
            regional_variant=None,
            cross_asset_attention=None,
            asset_groups=None,
            holographic_network=None,
            axial_time_region=None,
            mature_reliability=None,
        )
        trainer.predictor = SimpleNamespace(config=config)
        # La receta decide si F2 y F3 añaden su código.
        trainer.recipe = ChronologicalRecipe()
        assert trainer._code()[module.__name__] == expected


@pytest.mark.parametrize("kind", KINDS)
def test_precomputed_window_does_not_admit_the_transformer_only_dust_path(kind):
    from tests.training.test_window_variants import dust_options, recipe

    with pytest.raises(ValueError, match="Dust|Transformer"):
        recipe(temporal_encoder=kind, estimator="dust", dust=dust_options())
