"""La identidad registra también la implementación del validador compartido."""

import hashlib
import importlib
from pathlib import Path


def test_both_consumers_pin_the_shared_stopping_validator_source():
    from mars_titan.models.titans.frozen_financial import _implementation
    from mars_titan.models.titans.mechanism_consumer import _code

    name = "mars_titan.models.titans.bridge_stopping_record"
    module = importlib.import_module(name)
    digest = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
    assert _implementation().get(name) == digest
    assert _code().get(name) == digest
