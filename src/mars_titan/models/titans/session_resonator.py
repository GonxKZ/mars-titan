"""Resonador holográfico de descomposición latente transversal `session_resonator_v1` (CSRF).

Descompone cada representación centrada de la sesión (mismo mercado y mismo instante de decisión)
en un factor compuesto ligado mediante enlace Walsh-Hadamard más un residuo idiosincrático, y
calcula un contexto de interacción conjunta por celda *leave-one-out* en tiempo y memoria lineales
O(N · (T(K_1 + K_2)H + K_1 K_2 H)), sin formar nunca matrices N × N:

    n_i = LayerNorm(e_i),     ẽ_i = n_i − (1 / N) Σ_{j ∈ S} n_j
    B(a, b) = Q_H ((Q_H a) ⊙ (Q_H b)),     B*(x, b) = B(x, b)

con Q_H la matriz ortogonal simétrica normalizada de Walsh-Hadamard de tamaño H = 2^m. Con dos
diccionarios aprendidos normalizados a RMS unitario (‖Ã_u‖_2 = ‖B̃_v‖_2 = √H) e inicializados con
filas ortogonales, partiendo de b̂_i^(0) = (1 / K_2) Σ_q B̃_q, se ejecutan T iteraciones
diferenciables:

    p_i^(A,τ) = softmax_comp( B*(ẽ_i, b̂_i^(τ−1)) Ã^T / √H ),     â_i^(τ) = p_i^(A,τ) Ã
    p_i^(B,τ) = softmax_comp( B*(ẽ_i, â_i^(τ))   B̃^T / √H ),     b̂_i^(τ) = p_i^(B,τ) B̃

donde `softmax_comp` resta a cada columna de logits su LogMeanExp en la sesión
(`logsumexp(Z, dim=0) − log N`) antes de la softmax por fila. Esa normalización competitiva
(equivalente a un paso de Sinkhorn en espacio logarítmico) es estrictamente causal dentro de la
sesión, invariante a sesgos aditivos de popularidad por código y previene estructuralmente el
colapso de diccionarios sin requerir pérdidas auxiliares.

Tras converger, se forman las pertenencias conjuntas M_{i,uv} = p_{i,u}^(A,T) p_{i,v}^(B,T), el
residuo desconvolucionado r_i = ẽ_i − B(â_i^(T), b̂_i^(T)) y el contexto *leave-one-out*:

    c_i = Σ_{u,v} M_{i,uv} · [ Σ_{j ≠ i} M_{j,uv} ẽ_j / (Σ_{j ≠ i} M_{j,uv} + ε) ]

que se calcula exactamente en O(N K_1 K_2 H) sin tensores N × K_1 K_2 × H. Finalmente:

    e'_i = e_i + g ⊙ W_out LayerNorm_out([r_i, c_i])

con W_out inicializada (no nula) y la ganancia vectorial g iniciada a cero: el predictor arranca
bit a bit idéntico a su control sin encadenar dos ceros en serie, de modo que g recibe gradiente
no nulo ya en el primer paso. Para sesiones de un único activo (N = 1), la actualización es cero.

Controles negativos integrados con los mismos parámetros y la misma inicialización:
- `resonator`: CSRF completo con T ∈ [1, 4] iteraciones (por defecto T = 2).
- `no_iteration`: control de diccionario aditivo sin enlace Walsh-Hadamard ni iteración (T = 0).
- `shuffled_peers`: placebo con el mismo cómputo del resonador pero permutando determinísticamente
  las asignaciones y factores entre los pares de la sesión (clave: semilla, mercado, instante).
"""

import hashlib
import math
from dataclasses import dataclass

import torch
from torch import nn

from .config import LAYER_NORM_EPS, bounded_integer, require_identity
from .state import check_finite

SESSION_RESONATOR = "session_resonator_v1"
RESONATOR = "resonator"
NO_ITERATION = "no_iteration"
SHUFFLED_PEERS = "shuffled_peers"
MODES = (RESONATOR, NO_ITERATION, SHUFFLED_PEERS)
# Pseudo-observaciones en el denominador del contexto leave-one-out: estabiliza celdas casi vacías
# y evita cancelación catastrófica en FP32 cuando un activo concentra masa en una celda.
CELL_EPS = 0.1
# Épsilon suave en la normalización a RMS unitario de los átomos del diccionario.
NORM_EPS = 1e-6


def walsh_hadamard_matrix(size, dtype=torch.float32):
    """Matriz ortogonal simétrica normalizada de Walsh-Hadamard Q_H (Sylvester), H = 2^m."""
    bounded_integer(size, "dimensión de Walsh-Hadamard", 2, 4096)
    if (size & (size - 1)) != 0:
        raise ValueError("La dimensión del resonador Walsh-Hadamard debe ser potencia de dos")
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("La matriz Walsh-Hadamard necesita precisión FP32 o FP64")
    matrix = torch.ones((1, 1), dtype=torch.float64)
    while matrix.shape[0] < size:
        top = torch.cat((matrix, matrix), dim=1)
        bottom = torch.cat((matrix, -matrix), dim=1)
        matrix = torch.cat((top, bottom), dim=0)
    return (matrix / math.sqrt(size)).to(dtype=dtype)


@dataclass(frozen=True)
class SessionResonatorConfig:
    """Hiperparámetros identificados del resonador holográfico transversal."""

    codes_a: int = 8
    codes_b: int = 8
    iterations: int = 2
    mode: str = RESONATOR

    def __post_init__(self):
        bounded_integer(self.codes_a, "códigos del diccionario A de CSRF", 2, 32)
        bounded_integer(self.codes_b, "códigos del diccionario B de CSRF", 2, 32)
        if type(self.mode) is not str or self.mode not in MODES:
            raise ValueError(f"El modo de CSRF debe ser uno de {MODES}")
        if self.mode == NO_ITERATION:
            if type(self.iterations) is not int or self.iterations != 0:
                raise ValueError("El control no_iteration de CSRF exige iterations=0")
        else:
            bounded_integer(self.iterations, "iteraciones del resonador CSRF", 1, 4)

    def identity(self):
        result = dict(
            schema_version=1,
            kind=SESSION_RESONATOR,
            codes_a=self.codes_a,
            codes_b=self.codes_b,
            iterations=self.iterations,
            mode=self.mode,
            groups="same_market_and_prediction_instant_in_the_block",
            position="after_technical_branch_before_head",
            centering="layer_norm_then_subtract_session_mean",
            binding="walsh_hadamard_hadamard_product_orthogonal_self_adjoint"
            if self.mode != NO_ITERATION
            else "none_additive_mean_dictionary_control",
            dictionary_normalization="unit_rms_per_atom_orthogonal_init",
            competition="session_column_log_mean_exp_centered_softmax",
            cell_context="exact_leave_one_out_joint_cell_mean",
            cell_eps=CELL_EPS,
            output="residual_concat_cell_context_layer_norm_projection_zero_vector_gain",
            singleton_session="identity_zero_update",
            state="none",
            parameter_updates="none_in_forward",
        )
        if self.mode == SHUFFLED_PEERS:
            result.update(
                shuffle="non_identity_row_permutation_within_session_by_seed_market_and_instant"
            )
        return result


def parse_session_resonator(raw):
    """Aceptar la configuración, sus cuatro campos explícitos o None."""
    if raw is None:
        return None
    if type(raw) is SessionResonatorConfig:
        raw.__post_init__()
        return raw
    if isinstance(raw, dict) and set(raw) == set(SessionResonatorConfig.__dataclass_fields__):
        return SessionResonatorConfig(**raw)
    raise ValueError(
        "session_resonator debe declarar codes_a, codes_b, iterations y mode, o ser None"
    )


def _initial_dictionary(codes, hidden_size, dtype):
    """Inicializar átomos ortogonales de norma √H (RMS unitario) con signo QR canónico."""
    raw = torch.randn(codes, hidden_size, dtype=torch.float64)
    if codes <= hidden_size:
        q, r = torch.linalg.qr(raw.T, mode="reduced")
        diag = torch.diagonal(r)
        signs = torch.where(diag >= 0, torch.ones_like(diag), -torch.ones_like(diag))
        atoms = (q * signs.unsqueeze(0)).T * math.sqrt(hidden_size)
    else:
        norms = torch.sqrt(raw.square().sum(dim=-1, keepdim=True) + NORM_EPS)
        atoms = raw * (math.sqrt(hidden_size) / norms)
    return atoms.to(dtype=dtype)


class SessionResonator(nn.Module):
    """Resonador holográfico transversal por sesión `(market, prediction_at)`."""

    session_level = True

    def __init__(self, hidden_size, config, *, seed, dtype=torch.float32):
        super().__init__()
        bounded_integer(hidden_size, "dimensión de CSRF", 2, 4096)
        if (hidden_size & (hidden_size - 1)) != 0:
            raise ValueError("La dimensión de CSRF debe ser potencia de dos")
        bounded_integer(seed, "semilla de CSRF", 0, 2**32 - 1)
        if type(config) is not SessionResonatorConfig:
            raise ValueError("CSRF necesita SessionResonatorConfig")
        config.__post_init__()
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("CSRF necesita precisión FP32 o FP64")
        self.config, self.hidden_size, self.seed = config, hidden_size, seed
        # Semilla derivada del predictor sin consumir el generador global: el candidato y sus
        # controles (`no_iteration`, `shuffled_peers`) arrancan con los mismos pesos bit a bit.
        derived = hashlib.sha256(f"{SESSION_RESONATOR}/{seed}".encode()).digest()
        with torch.device("cpu"), torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(int.from_bytes(derived[:7], "little"))
            self.norm = nn.LayerNorm(hidden_size, eps=LAYER_NORM_EPS, dtype=dtype)
            self.dict_a = nn.Parameter(_initial_dictionary(config.codes_a, hidden_size, dtype))
            self.dict_b = nn.Parameter(_initial_dictionary(config.codes_b, hidden_size, dtype))
            self.out_norm = nn.LayerNorm(2 * hidden_size, eps=LAYER_NORM_EPS, dtype=dtype)
            # Proyección inicializada no nula para no encadenar dos ceros en serie con `gain`.
            self.out_proj = nn.Linear(2 * hidden_size, hidden_size, bias=False, dtype=dtype)
        self.gain = nn.Parameter(torch.zeros(hidden_size, dtype=dtype))
        self.register_buffer("hadamard", walsh_hadamard_matrix(hidden_size, dtype=dtype), False)

    def get_extra_state(self):
        state = dict(configuration=self.config.identity(), hidden_size=self.hidden_size)
        if self.config.mode == SHUFFLED_PEERS:
            state["permutation_seed"] = self.seed
        return state

    def set_extra_state(self, state):
        require_identity(state, self.get_extra_state())

    def bind(self, left, right):
        """Enlace/desenlace bilineal ortogonal Walsh-Hadamard B(a, b) = Q_H((Q_H a) ⊙ (Q_H b))."""
        q = self.hadamard
        return ((left @ q) * (right @ q)) @ q

    def _unit_rms(self, dictionary):
        scale = math.sqrt(self.hidden_size)
        rms_norm = torch.sqrt(dictionary.square().sum(dim=-1, keepdim=True) + NORM_EPS)
        return dictionary * (scale / rms_norm)

    def _competitive_softmax(self, logits):
        """Softmax competitiva: resta el LogMeanExp por código en la sesión antes de normalizar."""
        log_mean_exp = torch.logsumexp(logits, dim=0, keepdim=True) - math.log(len(logits))
        return torch.softmax(logits - log_mean_exp, dim=-1)

    def _session_permutation(self, key, rows):
        """Permutación determinista no identidad dentro de la sesión para `shuffled_peers`."""
        market, moment = key
        text = f"{SESSION_RESONATOR}/shuffle/{self.seed}/{market}/{moment}"
        digest = hashlib.sha256(text.encode()).digest()
        generator = torch.Generator().manual_seed(int.from_bytes(digest[:7], "little"))
        order = torch.randperm(rows, generator=generator)
        if rows >= 2 and torch.equal(order, torch.arange(rows)):
            order = torch.roll(order, shifts=1, dims=0)
        return order

    def groups(self, markets, prediction_at):
        """Clave y filas de cada sesión del bloque, en el orden de primera aparición."""
        sessions = {}
        for row, key in enumerate(zip(markets, prediction_at, strict=True)):
            sessions.setdefault(key, []).append(row)
        return list(sessions.items())

    def _leave_one_out_context(self, centered, m_flat):
        """Contexto de celda leave-one-out exacto en O(N K_1 K_2 H) sin matrices N × N."""
        cell_weight = m_flat.sum(dim=0)
        cell_sum = m_flat.T @ centered
        denom = (cell_weight.unsqueeze(0) - m_flat).clamp_min(0.0) + CELL_EPS
        alpha = m_flat / denom
        self_scale = (alpha * m_flat).sum(dim=1, keepdim=True)
        return alpha @ cell_sum - self_scale * centered

    def _factorize_session(self, normalized_group, a_h, b_h, key):
        """Factorizar una sesión de tamaño N >= 2 y devolver (residuo, contexto, p_a, p_b, M)."""
        rows = len(normalized_group)
        q = self.hadamard
        inv_sqrt_h = 1.0 / math.sqrt(self.hidden_size)
        centered = normalized_group - normalized_group.mean(dim=0, keepdim=True)
        e_h = centered @ q
        if self.config.mode == NO_ITERATION:
            p_a = self._competitive_softmax((e_h @ a_h.T) * inv_sqrt_h)
            p_b = self._competitive_softmax((e_h @ b_h.T) * inv_sqrt_h)
            recon_h = 0.5 * (p_a @ a_h + p_b @ b_h)
        else:
            b_hat_h = b_h.mean(dim=0, keepdim=True).expand(rows, -1)
            for _ in range(self.config.iterations):
                p_a = self._competitive_softmax(((e_h * b_hat_h) @ a_h.T) * inv_sqrt_h)
                a_hat_h = p_a @ a_h
                p_b = self._competitive_softmax(((e_h * a_hat_h) @ b_h.T) * inv_sqrt_h)
                b_hat_h = p_b @ b_h
            recon_h = a_hat_h * b_hat_h
        m_flat = (p_a.unsqueeze(2) * p_b.unsqueeze(1)).reshape(rows, -1)
        if self.config.mode == SHUFFLED_PEERS:
            perm = self._session_permutation(key, rows)
            if centered.is_cuda:
                perm = perm.pin_memory().to(device=centered.device, non_blocking=True)
            else:
                perm = perm.to(device=centered.device)
            recon_h = recon_h.index_select(0, perm)
            m_flat = m_flat.index_select(0, perm)
            p_a = p_a.index_select(0, perm)
            p_b = p_b.index_select(0, perm)
        residual = centered - recon_h @ q
        peer_context = self._leave_one_out_context(centered, m_flat)
        return centered, residual, peer_context, p_a, p_b, m_flat

    def _checked(self, encoded, markets, prediction_at):
        reference = self.gain
        if (
            not isinstance(encoded, torch.Tensor)
            or encoded.layout != torch.strided
            or encoded.ndim != 2
            or encoded.shape[1] != self.hidden_size
            or len(encoded) == 0
            or encoded.dtype != reference.dtype
            or encoded.device != reference.device
            or len(markets) != len(encoded)
            or len(prediction_at) != len(encoded)
        ):
            raise ValueError("CSRF necesita una representación, un mercado y un instante por fila")
        if torch.is_autocast_enabled(encoded.device.type):
            raise ValueError("CSRF exige FP32 o FP64 explícitos, sin autocast")
        check_finite(encoded, "La representación de entrada al resonador de sesión")

    def forward(self, encoded, markets, prediction_at):
        self._checked(encoded, markets, prediction_at)
        q = self.hadamard
        a_h = self._unit_rms(self.dict_a) @ q
        b_h = self._unit_rms(self.dict_b) @ q
        normalized = self.norm(encoded)
        sessions = self.groups(markets, prediction_at)
        if len(sessions) == 1:
            key, members = sessions[0]
            if len(members) < 2:
                result = encoded + self.gain * torch.zeros_like(encoded)
            else:
                _, residual, peer_context, _, _, _ = self._factorize_session(
                    normalized, a_h, b_h, key
                )
                joined = torch.cat((residual, peer_context), dim=-1)
                update = self.out_proj(self.out_norm(joined))
                check_finite(update, "La proyección del resonador de sesión")
                result = encoded + self.gain * update
            check_finite(result, "La representación con el resonador de sesión")
            return result
        mixed = torch.zeros_like(encoded)
        for key, members in sessions:
            size = len(members)
            if size < 2:
                continue
            # `groups()` recorre las filas en orden estrictamente creciente sin duplicados.
            contiguous = members[-1] - members[0] + 1 == size
            if contiguous:
                start, stop = members[0], members[0] + size
                group_norm = normalized[start:stop]
            else:
                if encoded.is_cuda:
                    pinned = torch.tensor(members, dtype=torch.int64, pin_memory=True)
                    index = pinned.to(device=encoded.device, non_blocking=True)
                else:
                    index = torch.tensor(members, dtype=torch.int64, device=encoded.device)
                group_norm = normalized.index_select(0, index)
            _, residual, peer_context, _, _, _ = self._factorize_session(group_norm, a_h, b_h, key)
            part = self.out_proj(self.out_norm(torch.cat((residual, peer_context), dim=-1)))
            if contiguous:
                mixed[start:stop] = part
            else:
                mixed = mixed.index_copy(0, index, part)
        check_finite(mixed, "La proyección del resonador de sesión")
        result = encoded + self.gain * mixed
        check_finite(result, "La representación con el resonador de sesión")
        return result

    @torch.no_grad()
    def diagnostics(self, encoded, markets, prediction_at):
        """Métricas de salud de los diccionarios (perplejidad, entropía, cuota máxima y residuo)."""
        self._checked(encoded, markets, prediction_at)
        q = self.hadamard
        a_h = self._unit_rms(self.dict_a) @ q
        b_h = self._unit_rms(self.dict_b) @ q
        normalized = self.norm(encoded)
        sessions = self.groups(markets, prediction_at)
        p_a_list, p_b_list, m_list = [], [], []
        centered_energy = 0.0
        residual_energy = 0.0
        for key, members in sessions:
            if len(members) < 2:
                continue
            index = torch.tensor(members, dtype=torch.int64, device=encoded.device)
            centered, residual, _, p_a, p_b, m_flat = self._factorize_session(
                normalized.index_select(0, index), a_h, b_h, key
            )
            p_a_list.append(p_a)
            p_b_list.append(p_b)
            m_list.append(m_flat)
            centered_energy += centered.square().sum().item()
            residual_energy += residual.square().sum().item()

        def _usage_stats(probs, count):
            mean_p = torch.cat(probs, dim=0).mean(dim=0)
            entropy = -(mean_p * mean_p.clamp_min(1e-12).log()).sum().item()
            return math.exp(entropy), entropy / math.log(count), mean_p.max().item()

        if not p_a_list:
            return dict(
                perplexity_a=float(self.config.codes_a),
                perplexity_b=float(self.config.codes_b),
                perplexity_cells=float(self.config.codes_a * self.config.codes_b),
                normalized_entropy_a=1.0,
                normalized_entropy_b=1.0,
                max_code_share_a=1.0 / self.config.codes_a,
                max_code_share_b=1.0 / self.config.codes_b,
                residual_energy_ratio=1.0,
                gain_l2=self.gain.norm().item(),
            )
        perp_a, ent_a, max_a = _usage_stats(p_a_list, self.config.codes_a)
        perp_b, ent_b, max_b = _usage_stats(p_b_list, self.config.codes_b)
        perp_cells, _, _ = _usage_stats(m_list, self.config.codes_a * self.config.codes_b)
        ratio = residual_energy / centered_energy if centered_energy > 0.0 else 0.0
        return dict(
            perplexity_a=perp_a,
            perplexity_b=perp_b,
            perplexity_cells=perp_cells,
            normalized_entropy_a=ent_a,
            normalized_entropy_b=ent_b,
            max_code_share_a=max_a,
            max_code_share_b=max_b,
            residual_energy_ratio=ratio,
            gain_l2=self.gain.norm().item(),
        )
