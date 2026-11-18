"""Los indicadores escalares conservan el primer fallo y el valor original de los tensores."""

import pytest
import torch

from mars_titan.models.titans import state


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.cuda)])
def test_finite_route_preserves_error_order_and_values(device):
    from mars_titan.models.titans.host_finite import enabled, first_failure, on_host

    for number in (
        0.0,
        -0.0,
        torch.finfo(torch.float32).max,
        torch.finfo(torch.float32).tiny,
        float("nan"),
        float("inf"),
        -float("inf"),
    ):
        for boolean in (True, False):
            value = torch.tensor(number, device=device)
            flag = torch.tensor(boolean, device=device)
            for items in ((value, flag), (flag, value), (value,), (flag,)):
                group = state._PendingChecks()
                for i, item in enumerate(items):
                    group.add(item, str(i))
                expected = group.first_failure()
                groups = {}
                for index, item in enumerate(items):
                    groups.setdefault(item.dtype, []).append(index)
                if torch.float32 in groups:
                    assert first_failure(group.flags, group.messages, groups) == expected
                with on_host():
                    assert enabled()
                    assert group.first_failure() == expected
                assert not enabled()
