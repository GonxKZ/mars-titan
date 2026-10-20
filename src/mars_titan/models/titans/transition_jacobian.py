"""Jacobianos completos de las transiciones de un flujo, solo en dimensiones pequeñas.

`MACProjectionControl` mide `RᵀJR` con productos Jacobiano-vector. Aquí se obtiene J
denso para contrastar esa compresión y estudiar secuencias de operadores con
`cm.operator_dynamics`. Hay dos transiciones:

- La transición rápida de MAC, `z' = F(z, x)` con `z = (vec W₁…W_L, vec m₁…m_L)`. Incluye
  lectura previa, atención, tasas dependientes de su salida y actualización asociativa.
  Con la convolución de la sección 4.4, z añade las ventanas de k, v y q, porque la
  transición depende de ellas. La de q solo se desplaza y recibe W_Q x, que no depende de z.
- Cada refinamiento del lector episódico, `z_{k+1} = z_k + η tanh(u_k)` con η = σ(s) y
  `u_k = W[z_k, base, read(z_k), presencia] + b`. Su operador local es `I + η D_z f_k`.
  La base es fija y los episodios elegidos son constantes a trozos, así que la derivada
  vale lejos de empates del orden de los vecinos.

Las matrices son medidas en coma flotante y no certifican cotas.
"""

from dataclasses import replace

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from .config import MACConfig
from .mac import TitansMAC

MAX_DENSE_ORDER = 256


def fast_state_dimension(config: MACConfig) -> int:
    """Orden de z: matrices y momentum, con convolución las tres ventanas de un flujo y, con
    PT5, los niveles ocultos de cada capa, y las estadísticas de plasticidad si están activas."""
    memory = config.memory
    matrices = (2 + memory.hidden_levels) * memory.depth
    return matrices * memory.dim**2 + 3 * memory.window * memory.dim + memory.modulation_elements


def fast_state_point(state):
    """Vector z con pesos, momentum, ventanas, niveles y estadísticas de plasticidad."""
    memory = state.memory
    values = (
        *memory.weights,
        *memory.momentum,
        *memory.convolution,
        *state.convolution,
        *memory.levels,
        *memory.modulation,
    )
    return torch.cat([value.flatten() for value in values])


def fast_state_transition(mac, token, state):
    """F(·, x) de un flujo con el token fijo. Devuelve z' con la misma disposición que z."""
    memory = state.memory
    dim, depth = mac.config.memory.dim, mac.config.memory.depth
    window, count = mac.config.memory.window, len(memory.convolution)
    windows_total = count + len(state.convolution)
    sizes = [dim**2] * (2 * depth) + [window * dim] * windows_total + [dim**2] * len(memory.levels)
    sizes += [value.numel() for value in memory.modulation]

    def transition(value):
        pieces = value.split(sizes)
        matrices = tuple(piece.reshape(1, dim, dim).clone() for piece in pieces[: 2 * depth])
        windows = tuple(
            piece.reshape(1, window, dim).clone()
            for piece in pieces[2 * depth : 2 * depth + windows_total]
        )
        levels_start = 2 * depth + windows_total
        modulation_start = levels_start + len(memory.levels)
        levels = tuple(
            piece.reshape(1, dim, dim).clone() for piece in pieces[levels_start:modulation_start]
        )
        modulation = tuple(
            piece.reshape_as(original).clone()
            for piece, original in zip(pieces[modulation_start:], memory.modulation, strict=True)
        )
        local = replace(
            memory,
            weights=matrices[:depth],
            momentum=matrices[depth:],
            convolution=windows[:count],
            levels=levels,
            modulation=modulation,
        )
        _, following = mac(
            token, replace(state, memory=local, convolution=windows[count:]), differentiable=True
        )
        return fast_state_point(following)

    return transition


def _single_flow(mac, token, state):
    if not isinstance(mac, TitansMAC) or mac.config.memory_mode != "online":
        raise ValueError("El Jacobiano rápido necesita Titans-MAC online")
    mac._validate_state(state)
    if state.memory.steps.shape != (1,) or token.shape[:2] != (1, 1):
        raise ValueError("El Jacobiano denso se calcula para un flujo y un token")
    order = fast_state_dimension(mac.config)
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
