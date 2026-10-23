"""Candidato de préstamo de pesos, conservado fuera de la implementación activa."""

from dataclasses import replace

import torch


def _associative_problem(self, keys, state, differentiable):
    if self.bank_readout is None:
        local = tuple(
            (w.clone() if differentiable else w.detach().clone()).requires_grad_(True)
            for w in self.readable_weights(state)
        )
        return local, self._apply_memory(keys, local)
    if self.config.momentum_copies == 3:
        # La inferencia solo lee el estado anterior y publica matrices nuevas. Detach
        # presta su almacenamiento sin conectar el gradiente externo. Los tensores de
        # inference_mode y la ruta diferenciable sí necesitan una copia local.
        local = tuple(
            (w.clone() if differentiable or w.is_inference() else w.detach()).requires_grad_(True)
            for bank in self._banks(state)
            for w in bank
        )
        depth = self.config.depth
        outputs = torch.stack(
            [self._apply_memory(keys, local[r * depth : (r + 1) * depth]) for r in range(3)],
            dim=2,
        )
        return local, self.bank_readout(keys, outputs)
    # La variable de derivación es una traslación compartida por los bancos. Su
    # gradiente suma las parciales de cada región sin fingir que se lee una media de pesos.
    local = tuple(torch.zeros_like(w, requires_grad=True) for w in state.weights)
    if not differentiable:
        state = replace(
            state,
            weights=tuple(w.detach() for w in state.weights),
            levels=tuple(w.detach() for w in state.levels),
        )
    return local, self._read_memory(keys, state, local)
