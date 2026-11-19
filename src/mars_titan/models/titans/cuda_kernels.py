"""Kernels CUDA para la escritura MAC de una capa y la selección estable de episodios."""

from functools import lru_cache
from importlib.util import find_spec

import torch


@lru_cache(maxsize=1)
def _programs():
    # Importar el controlador JIT únicamente en una ruta que vaya a lanzar un kernel.
    import triton
    import triton.language as tl

    @triton.jit
    def _write_tile(
        weight,
        previous,
        gradient,
        alpha,
        eta,
        theta,
        output,
        velocity,
        elements,
        WIDTH: tl.constexpr,
        GRAD_TRANSPOSED: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        pos = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        valid = pos < elements
        row = pos // WIDTH
        batch = row // WIDTH
        grad_pos = pos
        if GRAD_TRANSPOSED:
            grad_pos = batch * WIDTH * WIDTH + (pos % WIDTH) * WIDTH + row % WIDTH
        g = tl.load(gradient + grad_pos, valid, 0.0)
        old = tl.load(previous + pos, valid, 0.0)
        w = tl.load(weight + pos, valid, 0.0)
        forget = tl.load(alpha + row, valid, 0.0)
        retain = tl.load(eta + batch, valid, 0.0)
        step = tl.load(theta + batch, valid, 0.0)
        v = retain * old - step * g
        tl.store(velocity + pos, v, valid)
        tl.store(output + pos, (1.0 - forget) * w + v, valid)

    @triton.jit
    def _stable_neighbors(scores, indices, columns, COUNT: tl.constexpr, WIDTH: tl.constexpr):
        row = tl.program_id(0)
        col = tl.arange(0, WIDTH)
        value = tl.load(scores + row * columns + col, col < columns, 0.0)
        # El orden entero conserva el FP64, iguala ambos ceros y coloca NaN primero,
        # como argsort descendente. El índice menor decide cualquier empate.
        bits = value.to(tl.uint64, bitcast=True)
        sign = tl.full((WIDTH,), 0x8000000000000000, tl.uint64)
        magnitude = bits & (sign - 1)
        bits = tl.where(magnitude == 0, bits ^ bits, bits)
        rank = tl.where((bits & sign) != 0, ~bits, bits ^ sign)
        rank = tl.where(
            magnitude > 0x7FF0000000000000, tl.full((WIDTH,), 0xFFFFFFFFFFFFFFFF, tl.uint64), rank
        )
        rank = tl.where(col < columns, rank, 0)
        for position in tl.static_range(COUNT):
            high = tl.max(rank, 0)
            first = tl.min(tl.where(rank == high, col, WIDTH), 0)
            tl.store(indices + row * COUNT + position, first.to(tl.int64))
            rank = tl.where(col == first, 0, rank)

    return triton, _write_tile, _stable_neighbors


class _MemoryWrite(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, previous, gradient, alpha, eta, theta):
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(weight, previous, gradient, alpha, eta, theta)
        return _launch_write(weight, previous, gradient, alpha, eta, theta)

    @staticmethod
    def backward(ctx, weight_grad, velocity_grad):
        weight, previous, gradient, alpha, eta, theta = ctx.saved_tensors
        total = velocity_grad
        dw = da = None
        if weight_grad is not None:
            total = weight_grad if total is None else total + weight_grad
            dw = weight_grad * (1 - alpha)
            da = -(weight_grad * weight).sum_to_size(alpha.shape)
        if total is None:
            return dw, None, None, da, None, None
        neg = -total
        return (
            dw,
            total * eta,
            neg * theta,
            da,
            (total * previous).sum_to_size(eta.shape),
            (neg * gradient).sum_to_size(theta.shape),
        )


def _launch_write(weight, previous, gradient, alpha, eta, theta):
    following = torch.empty_like(weight)
    velocity = torch.empty_like(previous)
    triton, write_tile, _ = _programs()
    size = weight.numel()
    with torch.cuda.device(weight.device):
        write_tile[(triton.cdiv(size, 512),)](
            weight,
            previous,
            gradient,
            alpha,
            eta,
            theta,
            following,
            velocity,
            size,
            weight.shape[-1],
            not gradient.is_contiguous(),
            512,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return following, velocity


@lru_cache(maxsize=16)
def _supported_device(device):
    # La paridad numérica y el rendimiento se midieron en SM 12.0.
    return (
        device.type == "cuda"
        and find_spec("triton") is not None
        and torch.version.cuda is not None
        and torch.cuda.get_device_capability(device) == (12, 0)
    )


def _write_layout(weight, previous, gradient, gates):
    if weight.ndim != 3:
        return False
    batch, rows, width = weight.shape
    if not batch or (rows, width) != (64, 64):
        return False
    alpha, eta, theta = gates
    if previous.shape != weight.shape or gradient.shape != weight.shape:
        return False
    if alpha.shape != (batch, rows, 1) or eta.shape != (batch, 1, 1) or theta.shape != eta.shape:
        return False
    args = (weight, previous, gradient, alpha, eta, theta)
    return (
        all(t.dtype == torch.float32 and t.device == weight.device for t in args)
        and all(t.is_contiguous() for t in (weight, previous, alpha, eta, theta))
        and (gradient.is_contiguous() or gradient.stride() == (rows * width, 1, width))
    )


def try_memory_write(weights, velocities, gradients, gates, *, differentiable):
    """Validar todas las capas antes de reservar salidas o lanzar la primera escritura."""
    if not _supported_device(weights[0].device):
        return None
    layers = tuple(zip(weights, velocities, gradients, strict=True))
    if not all(_write_layout(*layer, gates) for layer in layers):
        return None
    write = _MemoryWrite.apply if differentiable and torch.is_grad_enabled() else _launch_write
    return tuple(zip(*(write(*layer, *gates) for layer in layers), strict=True))


def stable_neighbors(scores, count):
    """Primeros K índices de scores FP64 con el mismo orden total y estable de PyTorch."""
    if (
        not _supported_device(scores.device)
        or scores.dtype != torch.float64
        or scores.ndim != 2
        or not scores.is_contiguous()
        or scores.shape[0] == 0
        or not 1 <= count <= min(8, scores.shape[1])
        or not 32 <= scores.shape[1] <= 1024
    ):
        return torch.argsort(scores, dim=-1, descending=True, stable=True)[:, :count]
    rows, columns = scores.shape
    result = torch.empty((rows, count), device=scores.device, dtype=torch.int64)
    triton, _, select_neighbors = _programs()
    with torch.cuda.device(scores.device):
        select_neighbors[(rows,)](
            scores, result, columns, count, triton.next_power_of_2(columns), num_warps=4
        )
    return result
