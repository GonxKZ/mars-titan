"""Reutilización de introspección dentro de una auditoría, nunca entre verificaciones."""

import inspect
from collections import Counter

import pytest
import torch
from test_frozen_financial import frozen
from test_frozen_financial import frozen_backend as frozen_backend


def test_audit_inspects_each_shared_module_class_once_and_keeps_all_instances(monkeypatch):
    engine, _ = frozen()
    counts = Counter()
    original = inspect.getmembers

    def counted(kind, predicate=None):
        counts[kind] += 1
        return original(kind, predicate)

    monkeypatch.setattr(inspect, "getmembers", counted)
    engine.verify()
    assert len(list(engine.predictor.modules())) > len(counts)
    assert counts[torch.nn.Linear] == 1
    assert all(count == 1 for count in counts.values())


def test_repeated_class_audit_still_rejects_a_hook_on_a_later_instance():
    engine, _ = frozen()
    modules = [value for value in engine.predictor.modules() if type(value) is torch.nn.Linear]
    assert len(modules) > 1
    handle = modules[-1].register_forward_hook(lambda *_: None)
    try:
        with pytest.raises(ValueError, match="hooks"):
            engine.verify()
    finally:
        handle.remove()
    engine.verify()


def test_class_methods_are_reinspected_on_the_next_audit(monkeypatch):
    engine, _ = frozen()
    engine.verify()
    original = torch.nn.Linear.forward

    def changed(self, value):
        return original(self, value)

    with monkeypatch.context() as patch:
        patch.setattr(torch.nn.Linear, "forward", changed)
        with pytest.raises(ValueError, match="identidad"):
            engine.verify()
    engine.verify()
