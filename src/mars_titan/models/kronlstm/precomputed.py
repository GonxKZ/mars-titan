"""Ejecución opcional que precomputa términos independientes del estado recurrente.

Adaptado de la referencia local basada en NX-AI/KronLSTM 3d800c9a.
Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li.
Licencia MIT conservada en el paquete. Mantiene las comprobaciones por paso.
"""

import torch

from mars_titan.models.titans.state import check_finite, deferred_checks, require_true

from .rule import V_NORM_FLOOR, _validate


def kronlstm_scan_precomputed(
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
    """Usar el contrato de la referencia y conservar su orden de revisión y lectura.

    La precomputación reduce operaciones dentro del bucle a cambio de temporales
    adicionales. No fusiona estados, aproxima la norma ni omite diagnósticos.
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
        scaled_q = q * scale
        decays = g.exp() if g is not None else None
        desired = norm * (1 if target_gate is None else target_gate)
        if rule == "scalar":
            erase_scale = cap * beta
            wanted = cap * desired if decouple_erase else None
        else:
            write = 1 if target_gate is None else target_gate
            if not decouple_erase:
                write = beta * write
            written = write[..., None] * v if isinstance(write, torch.Tensor) else write * v
        outputs = []
        for step in range(steps):
            with deferred_checks():
                key, value = k[:, step], v[:, step]
                if g is not None:
                    decay = decays[:, step]
                    state = state * (decay[..., None] if g.ndim == 4 else decay[..., None, None])
                if rule == "scalar":
                    association = torch.einsum("bhk,bhkv,bhv->bh", key, state, value)
                    if decouple_erase:
                        strength = wanted[:, step] - erase_scale[:, step] * association
                    else:
                        strength = erase_scale[:, step] * (desired[:, step] - association)
                    check_finite(strength, "La revisión escalar KronLSTM")
                    state = state + strength[..., None, None] * key[..., None] * value[..., None, :]
                else:
                    recalled = torch.einsum("bhk,bhkv->bhv", key, state)
                    residual = written[:, step] - beta[:, step, :, None] * recalled
                    check_finite(residual, "La revisión vectorial")
                    state = state + key[..., None] * residual[..., None, :]
                check_finite(state, "El estado KronLSTM actualizado")
                output = torch.einsum("bhk,bhkv->bhv", scaled_q[:, step], state)
                check_finite(output, "La salida KronLSTM")
                outputs.append(output)
        return torch.stack(outputs, dim=1), state
