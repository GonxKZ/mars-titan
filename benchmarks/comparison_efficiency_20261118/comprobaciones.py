"""Resolver indicadores escalares CUDA conservando el primer error del recorrido."""

import math

import torch

from mars_titan.models.titans import state

REFERENCE = state._PendingChecks.first_failure
REAL = {torch.float16, torch.bfloat16, torch.float32, torch.float64}


def host_finite(self):
    if not self.flags:
        return None
    types = {flag.dtype for flag in self.flags}
    num = types - {torch.bool}
    if len(num) != 1 or not num <= REAL or self.flags[0].device.type != "cuda":
        return REFERENCE(self)
    dtype = next(iter(num))
    if torch.bool not in types:
        values = torch.stack(self.flags).tolist()
        for value, msg in zip(values, self.messages, strict=True):
            if not math.isfinite(value):
                return msg
        return None
    numbers = []
    booleans = []
    for flag in self.flags:
        (booleans if flag.dtype == torch.bool else numbers).append(flag)
    values = torch.cat((torch.stack(numbers), torch.stack(booleans).to(dtype=dtype))).tolist()
    i, j = 0, len(numbers)
    for flag, msg in zip(self.flags, self.messages, strict=True):
        if flag.dtype == torch.bool:
            passed = bool(values[j])
            j += 1
        else:
            passed = math.isfinite(values[i])
            i += 1
        if not passed:
            return msg
    return None


def edge_cases():
    count = 0
    for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        for number in (
            0.0,
            -0.0,
            float(torch.finfo(dtype).max),
            float(torch.finfo(dtype).tiny),
            float("nan"),
            float("inf"),
            -float("inf"),
        ):
            for flag in (True, False):
                value = torch.tensor(number, dtype=dtype, device="cuda:0")
                truth = torch.tensor(flag, device="cuda:0")
                for items in ((value, truth), (truth, value), (value,)):
                    pending = state._PendingChecks()
                    for i, x in enumerate(items):
                        pending.add(x, str(i))
                    expected = next(
                        (
                            str(i)
                            for i, x in enumerate(items)
                            if not (
                                bool(x.item()) if x.dtype == torch.bool else math.isfinite(x.item())
                            )
                        ),
                        None,
                    )
                    assert REFERENCE(pending) == host_finite(pending) == expected
                    count += 1
    for items in ((), (True,), (False, True)):
        p = state._PendingChecks()
        for i, v in enumerate(items):
            p.add(torch.tensor(v, device="cuda:0"), str(i))
        assert REFERENCE(p) == host_finite(p)
        count += 1
    return count
