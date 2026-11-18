"""Recuperación del calentamiento del padre con las mismas emisiones y cronología."""

import pytest
import torch

from mars_titan.training.checkpoints import StopRequest
from mars_titan.training.financial_run import Paused
from mars_titan.training.validation_warmup import ValidationWarmup
from tests.training.test_mars_titan_run import build, explicit_fastpath, native, parent, shared

__all__ = ["explicit_fastpath", "native", "shared"]


def compare_pass(reader, source):
    rows = []
    metrics = reader.evaluate(source, rows=rows)
    return rows, metrics


@pytest.mark.parametrize("admission", ["m0", "m1", "m2", "m3"])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.cuda)])
def test_reuse_skips_only_the_frozen_history(
    shared, tmp_path, native, admission, device, monkeypatch
):
    from mars_titan.training import validation_warmup

    _, streams = shared
    monkeypatch.setenv(validation_warmup.SWITCH, "0")
    options = dict(admission=admission, device=device, model=parent(streams, device=device))
    reference = build(streams, tmp_path / "ref", native, **options)
    ref = compare_pass(reference, streams["validation"])
    monkeypatch.setenv(validation_warmup.SWITCH, "1")
    if device == "cpu":
        monkeypatch.setattr(validation_warmup, "configured", lambda device: ValidationWarmup())
    engine = build(streams, tmp_path / "fit", native, **options)
    assert compare_pass(engine, streams["validation"]) == ref
    assert compare_pass(engine, streams["validation"]) == ref
    cache = engine.validation_warmup
    assert cache.hits == 1 and cache.events > 0
    assert 0 < cache.tensor_bytes < cache.limit
    with torch.no_grad():
        engine.readout.step_logit.add_(0.5)
        reference.readout.step_logit.add_(0.5)
    hot = compare_pass(engine, streams["validation"])
    assert hot == compare_pass(reference, streams["validation"])


def test_warmup_can_be_interrupted_and_the_cache_is_rebuilt(shared, tmp_path, monkeypatch):
    _, streams = shared
    from mars_titan.training import validation_warmup

    monkeypatch.setattr(
        validation_warmup, "configured", lambda device: validation_warmup.ValidationWarmup()
    )
    engine = build(streams, tmp_path / "fit", None, admission="m0")
    stop = StopRequest()
    stop.request_stop()
    with pytest.raises(Paused):
        engine.evaluate(streams["validation"], stop=stop)
    assert not engine.validation_warmup.events
    compare_pass(engine, streams["validation"])
    assert engine.validation_warmup.events


def test_tiny_limit_keeps_streaming_and_preserves_predictions(shared, tmp_path, monkeypatch):
    _, streams = shared
    ref = compare_pass(
        build(streams, tmp_path / "ref", None, admission="m0"), streams["validation"]
    )
    from mars_titan.training import validation_warmup

    monkeypatch.setattr(
        validation_warmup, "configured", lambda device: validation_warmup.ValidationWarmup(limit=1)
    )
    engine = build(streams, tmp_path / "fit", None, admission="m0")
    assert compare_pass(engine, streams["validation"]) == ref
    assert compare_pass(engine, streams["validation"]) == ref
    assert engine.validation_warmup.hits == 0
    assert engine.validation_warmup.tensor_bytes == 0


def test_configuration_requires_an_explicit_cuda_device(monkeypatch):
    from mars_titan.training import validation_warmup

    monkeypatch.delenv(validation_warmup.SWITCH, raising=False)
    assert validation_warmup.configured(torch.device("cuda:0")) is None
    monkeypatch.setenv(validation_warmup.SWITCH, "1")
    assert validation_warmup.configured(torch.device("cpu")) is None
    assert isinstance(validation_warmup.configured(torch.device("cuda:0")), ValidationWarmup)
    monkeypatch.setenv(validation_warmup.SWITCH, "true")
    with pytest.raises(ValueError, match="0 o 1"):
        validation_warmup.configured(torch.device("cuda:0"))


def test_capture_leaves_space_for_the_scored_pass_and_temporary_copies(monkeypatch):
    from mars_titan.training.validation_warmup import RESERVE

    cache = ValidationWarmup(limit=1024)
    device = torch.device("cuda:0")
    for free, accepted in [(RESERVE + 2047, False), (RESERVE + 2048, True)]:
        monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device, free=free: (free, free))
        assert cache._room(device, 1024) is accepted
    assert not cache._room(device, 1025)


@pytest.mark.parametrize("operation", ["export_state", "restore_state"])
def test_optional_copy_oom_replays_the_same_validation(shared, tmp_path, monkeypatch, operation):
    _, streams = shared
    validation = streams["validation"]
    reference = compare_pass(build(streams, tmp_path / "ref", None, admission="m0"), validation)
    from mars_titan.training import validation_warmup

    monkeypatch.setattr(validation_warmup, "configured", lambda device: ValidationWarmup())
    engine = build(streams, tmp_path / "fit", None, admission="m0")
    if operation == "restore_state":
        assert compare_pass(engine, validation) == reference
    with monkeypatch.context() as failure:

        def unavailable(*args, **kwargs):
            raise torch.OutOfMemoryError("Copia opcional sin espacio")

        failure.setattr(engine.predictor, operation, unavailable)
        assert compare_pass(engine, validation) == reference
    assert engine.validation_warmup.events == 0
    assert engine.validation_warmup.receipt()["allocation_fallbacks"] == 1
    assert compare_pass(engine, validation) == reference
    assert engine.validation_warmup.events > 0


def test_cache_rechecks_source_and_frozen_parent(shared, tmp_path, monkeypatch):
    from mars_titan.training import validation_warmup

    _, streams = shared
    monkeypatch.setattr(validation_warmup, "configured", lambda device: ValidationWarmup())
    engine = build(streams, tmp_path / "fit", None, admission="m0")
    validation = streams["validation"]
    compare_pass(engine, validation)
    with monkeypatch.context() as changed:
        changed.setattr(validation, "identity", "0" * 64)
        with pytest.raises(ValueError, match="índice|origen"):
            compare_pass(engine, validation)
    with torch.no_grad():
        engine.predictor.head.weight.add_(1)
    with pytest.raises(ValueError, match="parámetros|padre|sellad|identidad"):
        compare_pass(engine, validation)
    assert engine.validation_warmup.hits == 0
