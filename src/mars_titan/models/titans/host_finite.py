"""Clasificar en CPU indicadores CUDA escalares sin cambiar el orden de los errores."""

import math
from contextlib import contextmanager
from contextvars import ContextVar

import torch

_ACTIVE = ContextVar("titans_finite_scalars_on_host", default=False)


def enabled():
    return _ACTIVE.get()


@contextmanager
def on_host():
    token = _ACTIVE.set(True)
    try:
        yield
    finally:
        _ACTIVE.reset(token)


def first_failure(flags, messages, groups):
    """Los indicadores FP32 ya son normas. Solo hay que clasificar esas normas y los booleanos."""
    if torch.bool not in groups:
        values = torch.stack(flags).tolist()
        return next(
            (msg for value, msg in zip(values, messages, strict=True) if not math.isfinite(value)),
            None,
        )
    numbers = torch.stack([flags[i] for i in groups[torch.float32]])
    booleans = torch.stack([flags[i] for i in groups[torch.bool]])
    values = torch.cat((numbers, booleans.to(dtype=torch.float32))).tolist()
    numeric, boolean = 0, len(numbers)
    for flag, message in zip(flags, messages, strict=True):
        if flag.dtype == torch.bool:
            passed = bool(values[boolean])
            boolean += 1
        else:
            passed = math.isfinite(values[numeric])
            numeric += 1
        if not passed:
            return message
    return None
