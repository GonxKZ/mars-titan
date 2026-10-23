"""Huellas del adaptador del lector, sin construir ni ejecutar un ajuste."""

import hashlib
import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("temporal_encoder", [None, "kronlstm_v1", "vector_delta_v1"])
def test_adapter_code_inherits_real_parent_sources_and_conditional_encoder(temporal_encoder):
    from mars_titan.models.titans.frozen_financial import _implementation
    from mars_titan.posttraining import readout_adapters as adapters
    from mars_titan.training import mars_titan_run as parent

    # _code solo necesita esta configuración. __init__ crearía el optimizador del ajuste.
    trainer = object.__new__(adapters.ReadoutAdapterTrainer)
    trainer.predictor = SimpleNamespace(config=SimpleNamespace(temporal_encoder=temporal_encoder))
    actual = trainer._code()

    inherited = parent.ReadoutTrainer._code(trainer)
    expected = _implementation(kronlstm=temporal_encoder is not None)
    for name in (*parent._OWN_MODULES, *adapters._OWN):
        source = Path(importlib.import_module(name).__file__)
        expected[name] = hashlib.sha256(source.read_bytes()).hexdigest()
    assert actual == expected
    assert all(actual[name] == digest for name, digest in inherited.items())
    assert set(adapters._OWN) <= actual.keys()

    kron_modules = {
        "mars_titan.models.kronlstm",
        "mars_titan.models.kronlstm.rule",
        "mars_titan.models.kronlstm.price_encoder",
    }
    if temporal_encoder is None:
        assert kron_modules.isdisjoint(actual)
    else:
        assert kron_modules <= actual.keys()
        for name in kron_modules:
            source = Path(importlib.import_module(name).__file__)
            assert actual[name] == hashlib.sha256(source.read_bytes()).hexdigest()
