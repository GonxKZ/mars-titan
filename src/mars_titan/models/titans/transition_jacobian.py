"""Jacobianos completos de las transiciones de un flujo, solo en dimensiones pequeñas.

`MACProjectionControl` mide `RᵀJR` con productos Jacobiano-vector. Aquí se obtiene J
denso para contrastar esa compresión y estudiar secuencias de operadores con
`cm.operator_dynamics`. Hay dos transiciones:

- La transición rápida de MAC, `z' = F(z, x)` con `z = (vec W₁…W_L, vec m₁…m_L)`. Incluye
  lectura previa, atención, tasas dependientes de su salida y actualización asociativa.
- Cada refinamiento del lector episódico, `z_{k+1} = z_k + η tanh(u_k)` con η = σ(s) y
  `u_k = W[z_k, base, read(z_k), presencia] + b`. Su operador local es `I + η D_z f_k`.
  La base es fija y los episodios elegidos son constantes a trozos, así que la derivada
  vale lejos de empates del orden de los vecinos.

Las matrices son medidas en coma flotante y no certifican cotas.
"""

from dataclasses import replace

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from .mac import TitansMAC

MAX_DENSE_ORDER = 256


def fast_state_point(state):
    """Vector z de un estado MAC: pesos rápidos y momentum por capas, en orden de filas."""
    memory = state.memory
    return torch.cat([value.flatten() for value in (*memory.weights, *memory.momentum)])


def fast_state_transition(mac, token, state):
    """F(·, x) de un flujo con el token fijo. Devuelve z' con la misma disposición que z."""
    memory = state.memory
    dim, depth = mac.config.memory.dim, mac.config.memory.depth

    def transition(value):
        pieces = tuple(piece.reshape(1, dim, dim).clone() for piece in value.split(dim**2))
        local = replace(memory, weights=pieces[:depth], momentum=pieces[depth:])
        _, following = mac(token, replace(state, memory=local), differentiable=True)
        return fast_state_point(following)

    return transition


def _single_flow(mac, token, state):
    if not isinstance(mac, TitansMAC) or mac.config.memory_mode != "online":
        raise ValueError("El Jacobiano rápido necesita Titans-MAC online")
    mac._validate_state(state)
    if state.memory.steps.shape != (1,) or token.shape[:2] != (1, 1):
        raise ValueError("El Jacobiano denso se calcula para un flujo y un token")
    order = 2 * mac.config.memory.depth * mac.config.memory.dim**2
    if order > MAX_DENSE_ORDER:
        raise ValueError("El Jacobiano denso solo se admite hasta orden 256")


def fast_state_jacobian(mac, token, state):
    """J = ∂F/∂z completo en el estado dado, con SDPA Math como el control C."""
    _single_flow(mac, token, state)
    point = fast_state_point(state).detach()
    with torch.enable_grad(), sdpa_kernel(SDPBackend.MATH):
        jacobian = torch.autograd.functional.jacobian(
            fast_state_transition(mac, token.detach(), state), point
        )
    return jacobian.detach()


def fast_state_trajectory(mac, tokens, state):
    """J_t de una secuencia de tokens de un flujo, avanzando con la transición ordinaria."""
    if tokens.ndim != 3 or tokens.shape[0] != 1 or not 1 <= tokens.shape[1] <= 256:
        raise ValueError("La trayectoria necesita [1, T, D] con 1 ≤ T ≤ 256")
    jacobians = []
    for index in range(tokens.shape[1]):
        token = tokens[:, index : index + 1]
        jacobians.append(fast_state_jacobian(mac, token, state))
        with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
            _, state = mac(token, state)
    return torch.stack(jacobians), state


def refinement_jacobians(readout, z_base, snapshot=None, *, context_id=None, cutoff=None):
    """J_k = ∂z_{k+1}/∂z_k de cada refinamiento de una fila, con base y episodios fijos.

    En `per_step` cada paso usa los episodios que elige en su estado. En `first_read`
    todos usan los del primer paso, como `EpisodicReadout.forward`: a partir del segundo,
    la lectura se restringe a esas posiciones y las devuelve.
    """
    # El lector importa el predictor y este el control C, que usa este módulo.
    from .episodic_readout import EpisodicReadout

    if not isinstance(readout, EpisodicReadout):
        raise ValueError("Se necesita el lector episódico")
    hidden = readout.config.hidden_size
    if not isinstance(z_base, torch.Tensor) or z_base.shape != (1, hidden):
        raise ValueError("El Jacobiano del refinamiento se calcula para una fila")
    base = z_base.detach()
    fixed = readout.config.episode_selection == "first_read"
    state, first, steps = base, None, []
    for _ in range(readout.config.refinements):
        with torch.no_grad():
            _, chosen = readout._read(state, snapshot, context_id, cutoff, first)
        if fixed and first is None:
            first = chosen

        def step(value, positions=chosen):
            read, _ = readout._read(value, snapshot, context_id, cutoff, positions)
            return readout.refine(value, base, read)

        with torch.enable_grad():
            steps.append(torch.autograd.functional.jacobian(step, state).reshape(hidden, hidden))
        with torch.no_grad():
            state = step(state)
    return torch.stack(steps).detach()
