"""La lectura de metadatos también rechaza cambios del contrato de recogida."""

import pytest
from test_financial_mechanisms import engines
from test_financial_session import moment
from test_financial_session_controls import four_flow_source as four_flow_source
from test_financial_session_controls import no_target_estimation as no_target_estimation
from test_frozen_financial import frozen_backend as frozen_backend
from test_native_episode_backend import native as native
from test_native_mechanism_collection import session

pytestmark = pytest.mark.native("MARS_TITAN_EPISODIC_NATIVE")


@pytest.mark.parametrize("field", ["disabled", "method", "limit"])
def test_native_collection_guard_rejects_changes_before_using_state_or_resolver(
    native, four_flow_source, tmp_path, field
):
    data = engines(four_flow_source, "cpu")
    with session(native, tmp_path / field, data) as run:
        run.step(four_flow_source["batches"][126], [])
        before = (run.output / "latest.json").read_bytes()
        invoked = []
        if field == "disabled":
            run._binding.collections = None
        elif field == "method":
            original = run._binding.collections.resolve

            def substituted(*args, **kwargs):
                invoked.append(True)
                return original(*args, **kwargs)

            run._binding.collections.resolve = substituted
        else:
            object.__setattr__(run._collection_config, "max_decisions", 256)
        with pytest.raises(ValueError):
            if field == "method":
                run.step([], [], kind="settlement", cutoff=moment(127))
            else:
                run.mechanism_state()
        assert invoked == []
        assert (run.output / "latest.json").read_bytes() == before
