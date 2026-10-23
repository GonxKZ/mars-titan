"""Referencia diferenciable de la revisión escalar y su control vectorial.

Adaptado de fla/ops/kronlstm/naive.py en NX-AI/KronLSTM 3d800c9a.
Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li.
Licencia MIT conservada en el paquete. No requiere FLA ni kernels nuevos.
"""

import math

import torch

from mars_titan.models.titans.state import check_finite, deferred_checks, require_true

V_NORM_FLOOR = 1e-18
MAX_RECURRENT_ELEMENTS = 2**24
_DTYPES = (torch.float32, torch.float64)


def _tensor(value, name):
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.is_nested
        or value.dtype not in _DTYPES
        or value.device.type not in {"cpu", "cuda"}
    ):
        raise ValueError(f"{name} debe ser un tensor denso float32 o float64 en CPU o CUDA")


def _validate(q, k, v, g, beta, target, initial, rule, decoupled, scale):
    for name, value in (("q", q), ("k", k), ("v", v), ("beta", beta)):
        _tensor(value, name)
    if q.ndim != 4 or k.shape != q.shape or v.ndim != 4 or v.shape[:3] != q.shape[:3]:
        raise ValueError("q/k deben ser [B,T,H,K] y v debe ser [B,T,H,V]")
    batch, steps, heads, keys = q.shape
    values = v.shape[-1]
    if min(batch, steps, heads, keys, values) < 1:
        raise ValueError("La recurrencia necesita dimensiones positivas")
    if batch * steps * heads * keys * values > MAX_RECURRENT_ELEMENTS:
        raise ValueError("La recurrencia supera el presupuesto lógico de elementos")
    if beta.shape != q.shape[:3]:
        raise ValueError("beta debe tener forma [B,T,H]")
    if type(rule) is not str or rule not in {"scalar", "vector"} or type(decoupled) is not bool:
        raise ValueError("La regla o decouple_erase no pertenecen al contrato")
    if decoupled and target is None:
        raise ValueError("decouple_erase requiere target_gate")
    if scale is not None and (
        type(scale) not in (int, float) or not math.isfinite(scale) or scale <= 0
    ):
        raise ValueError("La escala debe ser positiva y finita")
    optional = (("g", g), ("target_gate", target), ("initial_state", initial))
    for name, value in optional:
        if value is not None:
            _tensor(value, name)
    if g is not None and tuple(g.shape) not in (tuple(q.shape[:3]), tuple(q.shape)):
        raise ValueError("g debe ser [B,T,H] o [B,T,H,K], sin broadcasting")
    if target is not None and target.shape != q.shape[:3]:
        raise ValueError("target_gate debe tener forma [B,T,H]")
    if initial is not None and initial.shape != (batch, heads, keys, values):
        raise ValueError("initial_state debe tener forma [B,H,K,V]")
    all_values = (q, k, v, beta, *(value for _, value in optional if value is not None))
    if any(value.device != q.device for value in all_values):
        raise ValueError("Todos los tensores deben compartir dispositivo")
    if (
        k.dtype != q.dtype
        or v.dtype != q.dtype
        or (initial is not None and initial.dtype != q.dtype)
    ):
        raise ValueError("q/k/v y el estado deben compartir precisión")
    if torch.is_autocast_enabled(q.device.type):
        raise ValueError("La referencia KronLSTM no admite autocast")


def kronlstm_scan(
    q,
    k,
    v,
    g,
    beta,
    *,
    target_gate,
    initial_state=None,
    rule="scalar",
    decouple_erase=True,
    scale=None,
):
    """Actualizar y consultar después de escribir, con estado [B,H,K,V] explícito.

    q/k/v llegan proyectados. La capa normaliza q/k/v y el operador no repite esa
    transformación. k se presupone unitario, como en la referencia publicada. g contiene
    log-olvido por cabeza o dimensión. El suelo de norma afecta solo a la división.
    El control vectorial conserva la escritura y borra el residual vectorial.
    El presupuesto cuenta estados lógicos por paso, no certifica el pico del allocator.
    """
    _validate(q, k, v, g, beta, target_gate, initial_state, rule, decouple_erase, scale)
    with deferred_checks():
        for name, value in (
            ("q", q),
            ("k", k),
            ("v", v),
            ("beta", beta),
            ("g", g),
            ("target_gate", target_gate),
            ("estado inicial", initial_state),
        ):
            if value is not None:
                check_finite(value, name)
        require_true(((beta >= 0) & (beta <= 1)).all(), "beta debe estar entre cero y uno")
        if target_gate is not None:
            require_true(
                ((target_gate >= 0) & (target_gate <= 1)).all(),
                "target_gate debe estar entre cero y uno",
            )
        if g is not None:
            require_true((g <= 0).all(), "El log-olvido g no puede ser positivo")
        batch, steps, heads, keys = q.shape
        values = v.shape[-1]
        scale = keys**-0.5 if scale is None else scale
        state = (
            q.new_zeros(batch, heads, keys, values)
            if initial_state is None
            else initial_state.clone()
        )
        beta = beta.to(q.dtype)
        target_gate = target_gate.to(q.dtype) if target_gate is not None else None
        g = g.to(q.dtype) if g is not None else None
        norm = v.norm(dim=-1).square()
        check_finite(norm, "La norma cuadrática de v")
        cap = norm.clamp_min(V_NORM_FLOOR).reciprocal()
        outputs = []
        for step in range(steps):
            with deferred_checks():
                key, value = k[:, step], v[:, step]
                if g is not None:
                    decay = g[:, step].exp()
                    state = state * (decay[..., None] if g.ndim == 4 else decay[..., None, None])
                erase = beta[:, step]
                target = 1 if target_gate is None else target_gate[:, step]
                if rule == "scalar":
                    association = torch.einsum("bhk,bhkv,bhv->bh", key, state, value)
                    if decouple_erase:
                        strength = (
                            cap[:, step] * (norm[:, step] * target)
                            - cap[:, step] * erase * association
                        )
                    else:
                        strength = cap[:, step] * erase * (norm[:, step] * target - association)
                    check_finite(strength, "La revisión escalar KronLSTM")
                    state = state + strength[..., None, None] * key[..., None] * value[..., None, :]
                else:
                    recalled = torch.einsum("bhk,bhkv->bhv", key, state)
                    write = target if decouple_erase else erase * target
                    residual = write[..., None] * value - erase[..., None] * recalled
                    check_finite(residual, "La revisión vectorial")
                    state = state + key[..., None] * residual[..., None, :]
                check_finite(state, "El estado KronLSTM actualizado")
                output = torch.einsum("bhk,bhkv->bhv", q[:, step] * scale, state)
                check_finite(output, "La salida KronLSTM")
                outputs.append(output)
        return torch.stack(outputs, dim=1), state
