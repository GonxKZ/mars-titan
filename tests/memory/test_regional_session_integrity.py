"""Rechazar un ledger incoherente antes de entrar en el ejecutor nativo."""

import sys
from contextlib import contextmanager

import pytest
from test_regional_program_session import engines, limits, session
from test_regional_program_session import four_flow_source as four_flow_source
from test_regional_program_session import frozen_backend as frozen_backend
from test_regional_program_session import native as native
from test_regional_program_session import no_target_estimation as no_target_estimation

pytestmark = pytest.mark.native("MARS_TITAN_EPISODIC_NATIVE")


@contextmanager
def native_steps(executor):
    """Observar la entrada C sin sustituir métodos de la sesión identificada."""
    calls = []

    def observe(frame, event, argument):
        if (
            event == "c_call"
            and getattr(argument, "__self__", None) is executor
            and getattr(argument, "__name__", None) == "step"
        ):
            calls.append("step")
            raise AssertionError("El evento incoherente alcanzó el ejecutor nativo")

    previous = sys.getprofile()
    sys.setprofile(observe)
    try:
        yield calls
    finally:
        sys.setprofile(previous)


@pytest.mark.parametrize("active", [True, False], ids=["missing_ledger", "unexpected_ledger"])
def test_inconsistent_regional_ledger_is_rejected_before_native_step_or_mutation(
    native, four_flow_source, tmp_path, active
):
    data = engines(four_flow_source, "cpu", active=active)
    output = tmp_path / "integrity"
    options = dict(regional_work_limits=limits()) if active else {}
    with session(native, output, data, **options) as run:
        before_head = (output / "latest.json").read_bytes()
        index = output / "regional-work.json"
        before_index = index.read_bytes() if active else None
        run._regional_work = None if active else object()
        boundaries = []

        def fault(boundary):
            boundaries.append(boundary)
            raise AssertionError("El rechazo regional llegó después del hook nativo")

        with native_steps(run._executor) as calls, pytest.raises(ValueError, match="regional"):
            run.step(four_flow_source["batches"][126], [], fault=fault)
        assert not calls and not boundaries
        assert (output / "latest.json").read_bytes() == before_head
        if active:
            assert index.read_bytes() == before_index
        else:
            assert not index.exists()
