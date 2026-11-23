"""Cerebro de atractores de Hopfield escasos con inhibición lateral de orden (RSHRB).

Tras MAC o tras la rama técnica, cada activo i de la sesión S (mismo mercado y mismo instante
de decisión t) dispone de su representación e_i ∈ R^H. El módulo normaliza la entrada al espacio
de atractores z_i^(0) = LN_0(e_i) y ejecuta J pasos de descenso en energía sobre P prototipos
asociativos aprendidos Ξ = [ξ_1, …, ξ_P] ∈ R^{P × H} (normalizados a norma cuadrática media ≤ 1,
ξ̃_p = ξ_p / √(media(ξ_p²) + ε_ξ)) con afinidad escasa de Tsallis α = 1,5 (entmax-1.5):

    ẑ_i^(k) = LN(z_i^(k))
    α_i^(k) = entmax_{1.5}( (β / √H) · ẑ_i^(k) ξ̃ᵀ )  ∈ Δ^{P-1}

donde β ∈ (β_min, β_max) es la temperatura inversa aprendida y acotada. En cada paso k se suman
dos campos derivados de la energía (atracción hacia los prototipos activos y repulsión lateral
frente a los pares que compiten por los mismos prototipos), ejecutados en O(N P H) sin construir
ninguna matriz N × N:

    δ_i^{atract,(k)} = Σ_p α_{ip}^(k) ξ̃_p
    n_p^(k) = Σ_{j ∈ S} α_{jp}^(k),        S_p^(k) = Σ_{j ∈ S} α_{jp}^(k) ẑ_j^(k)
    δ_i^{repul,(k)}  = Σ_p α_{ip}^(k) (n_p^(k) ẑ_i^(k) − S_p^(k)) / (n_p^(k) + ε_c)
                     = Σ_{j≠i} (Σ_p α_{ip}^(k) α_{jp}^(k) / (n_p^(k) + ε_c)) (ẑ_i^(k) − ẑ_j^(k))
    z_i^(k+1)        = z_i^(k) + η_a δ_i^{atract,(k)} + η_r δ_i^{repul,(k)}

con pasos aprendidos y estrictamente acotados η_a, η_r ∈ (0, η_max) y amortiguamiento por
congestión ε_c = 1,0. La salida inyecta el desplazamiento neto Δz_i = z_i^(J) − z_i^(0) mediante
una puerta vectorial inicializada a cero sobre una proyección Kaiming no nula (sin doble cero):

    e'_i = e_i + g ⊙ (W_out LN_out(Δz_i)),    g_0 = 0,  LN_out sin sesgo (LN_out(0) = 0).

Propiedades algebraicas y numéricas exactas:
- Escasez exacta de prototipos: entmax-1.5 asigna peso exactamente cero a prototipos ajenos, de
  modo que activos de regímenes disjuntos no ejercen repulsión espuria entre sí.
- Conservación de suma cero (3.ª ley de Newton): Σ_{i ∈ S} δ_i^{repul} = 0 idénticamente para
  todo ε_c > 0; la inhibición lateral solo separa rangos relativos y nunca desplaza la media.
- Invariancia a traslaciones uniformes: ẑ_j → ẑ_j + c deja δ_i^{repul} idénticamente invariante.
- Amortiguamiento automático de atípicos solitarios y sesiones N = 1: si un activo no comparte
  prototipos con pares (o N = 1), n_p ẑ_i − S_p = 0 y el denominador n_p + 1 ≥ 1 nunca amplifica
  ruido de alta frecuencia.
- Causalidad estricta y sin estado entre sesiones: solo lee filas de la misma tupla
  (mercado, instante) y no utiliza ni regresa rangos de etiquetas maduras.

Modos con exactamente los mismos parámetros y la misma inicialización:
- `repulsive`: mecanismo RSHRB completo (atracción escasa + inhibición lateral de orden).
- `attractive_only`: ablación con η_r = 0 (atracción local fila a fila, sin lectura de pares).
- `repulsive_only`: ablación con η_a = 0 (inhibición lateral pura entre pares co-activados).
- `shuffled_peers`: control placebo que conserva la atracción y permuta deterministamente el
  campo repulsivo δ^{repul} entre las filas de cada sesión (semilla derivada de predictor,
  mercado e instante), preservando distribución marginal y suma cero pero rompiendo la
  correspondencia de pares.
"""

import hashlib
import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import LAYER_NORM_EPS, bounded_integer, require_identity
from .state import check_finite

HOPFIELD_RANK_BRAIN = "repulsive_sparse_hopfield_rank_brain_v1"
REPULSIVE = "repulsive"
ATTRACTIVE_ONLY = "attractive_only"
REPULSIVE_ONLY = "repulsive_only"
SHUFFLED_PEERS = "shuffled_peers"
MODES = (REPULSIVE, ATTRACTIVE_ONLY, REPULSIVE_ONLY, SHUFFLED_PEERS)
MIN_PROTOTYPES = 4
MAX_PROTOTYPES = 128
MIN_STEPS = 1
MAX_STEPS = 4
BETA_MIN = 0.25
BETA_MAX = 8.0
BETA_INIT = 1.0
STEP_BOUND = 0.5
STEP_INIT = 0.25
CROWDING_PRIOR = 1.0
PROTOTYPE_NORM_EPS = 1e-8
POSITIONS = dict(
    after_mac="after_mac_before_head",
    after_technical="after_technical_branch_before_head",
)


def _logit(fraction: float) -> float:
    return math.log(fraction) - math.log1p(-fraction)


class _Entmax15(torch.autograd.Function):
    """Transformación escasa entmax-1.5 exacta con VJP analítico sin división por cero."""

    @staticmethod
    def forward(ctx, logits: Tensor) -> Tensor:
        shifted = (logits - logits.amax(dim=-1, keepdim=True)) * 0.5
        ordered, _ = torch.sort(shifted, dim=-1, descending=True)
        # Cualquier entrada <= -1 nunca entra en el soporte activo porque (0 - x)^2 >= 1 y
        # tau pertenece a [-1, 0). Acotar a -2 evita desbordamientos en x^2 para entradas extremas.
        safe = ordered.clamp_min(-2.0)
        ranks = torch.arange(1, logits.shape[-1] + 1, dtype=logits.dtype, device=logits.device)
        cumsum = safe.cumsum(dim=-1)
        cumsum_sq = safe.square().cumsum(dim=-1)
        mean = cumsum / ranks
        ss = (cumsum_sq - ranks * mean.square()).clamp_min(0.0)
        # Criterio exacto y monótono: f(r) = ss_r + r (m_r - x_(r))^2.
        support = (ss < 1.0) & ((ss + ranks * (mean - safe).square()) < 1.0)
        k = support.sum(dim=-1, keepdim=True).clamp_min(1)
        tau_all = mean - torch.sqrt((1.0 - ss).clamp_min(0.0) / ranks)
        tau = tau_all.gather(-1, k - 1)
        root = (shifted - tau).clamp_min(0.0)
        probs = root.square()
        ctx.save_for_backward(root)
        return probs

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tensor:
        (root,) = ctx.saved_tensors
        # Para cualquier distribución p en el símplex, sum(sqrt(p)) >= sum(p) = 1.0 estrictamente.
        denom = root.sum(dim=-1, keepdim=True)
        weighted = (root * grad_output).sum(dim=-1, keepdim=True) / denom
        return root * (grad_output - weighted)


def entmax15(logits: Tensor) -> Tensor:
    """Proyección escasa de Tsallis α = 1,5 sobre el símplex en la última dimensión."""
    if not isinstance(logits, Tensor) or logits.ndim < 1 or logits.shape[-1] < 1:
        raise ValueError("entmax15 necesita un tensor no vacío en su última dimensión")
    if logits.dtype not in (torch.float32, torch.float64):
        raise ValueError("entmax15 exige precisión FP32 o FP64")
    return _Entmax15.apply(logits)


@dataclass(frozen=True)
class HopfieldRankBrainConfig:
    """Contrato del cerebro de atractores de Hopfield escasos con inhibición lateral (RSHRB)."""

    prototypes: int = 32
    steps: int = 2
    mode: str = REPULSIVE
    position: str = "after_technical"

    def __post_init__(self) -> None:
        bounded_integer(
            self.prototypes, "número de prototipos de RSHRB", MIN_PROTOTYPES, MAX_PROTOTYPES
        )
        bounded_integer(self.steps, "pasos de descenso energético de RSHRB", MIN_STEPS, MAX_STEPS)
        if type(self.mode) is not str or self.mode not in MODES:
            raise ValueError(f"El modo de RSHRB debe ser uno de {MODES}")
        if type(self.position) is not str or self.position not in POSITIONS:
            raise ValueError("La posición de RSHRB debe ser after_mac o after_technical")

    def identity(self) -> dict:
        groups = (
            "each_row_with_itself"
            if self.mode == ATTRACTIVE_ONLY
            else "same_market_and_prediction_instant_in_the_block"
        )
        dynamics = dict(
            repulsive="sparse_entmax15_prototype_attraction_plus_crowding_damped_lateral_repulsion",
            attractive_only="sparse_entmax15_prototype_attraction_only_zero_repulsion",
            repulsive_only="crowding_damped_prototype_collision_lateral_repulsion_only",
            shuffled_peers="sparse_attraction_plus_session_permuted_lateral_repulsion_by_seed",
        )[self.mode]
        return dict(
            schema_version=1,
            kind=HOPFIELD_RANK_BRAIN,
            prototypes=self.prototypes,
            steps=self.steps,
            mode=self.mode,
            position=POSITIONS[self.position],
            groups=groups,
            activation="tsallis_entmax_1_5_exact_sparse_hopfield",
            prototype_norm="smooth_rms_unit_sphere",
            beta_bounds=[BETA_MIN, BETA_MAX],
            step_bound=STEP_BOUND,
            crowding_prior=CROWDING_PRIOR,
            repulsion="zero_sum_shift_invariant_prototype_collision_energy_gradient",
            dynamics=dynamics,
            normalization="unit_entry_layer_norm_step_layer_norm_and_bias_free_readout_layer_norm",
            output="residual_with_zero_initialized_vector_gate_over_kaiming_projection",
            state="working_only_reset_per_session",
            parameter_updates="none_in_forward",
        )


def parse_hopfield_rank_brain(raw: object) -> HopfieldRankBrainConfig | None:
    """Aceptar la configuración, sus valores declarados o None."""
    if raw is None:
        return None
    if type(raw) is HopfieldRankBrainConfig:
        raw.__post_init__()
        return raw
    required = {"prototypes", "steps", "mode"}
    allowed = required | {"position"}
    if isinstance(raw, dict) and required <= set(raw) <= allowed:
        return HopfieldRankBrainConfig(**raw)
    raise ValueError(
        "hopfield_rank_brain debe declarar prototypes, steps y mode "
        "(y opcionalmente position), o ser None"
    )


class RepulsiveSparseHopfieldRankBrain(nn.Module):
    """Descenso energético en O(J · N · P · H) con atracción escasa e inhibición lateral."""

    session_level = True

    def __init__(
        self,
        hidden_size: int,
        config: HopfieldRankBrainConfig,
        *,
        seed: int,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        bounded_integer(hidden_size, "dimensión de RSHRB", 1, 4096)
        bounded_integer(seed, "semilla de RSHRB", 0, 2**32 - 1)
        if type(config) is not HopfieldRankBrainConfig:
            raise ValueError("RSHRB necesita HopfieldRankBrainConfig")
        config.__post_init__()
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("RSHRB necesita precisión FP32 o FP64")
        self.config = config
        self.hidden_size = hidden_size
        self.prototypes_count = config.prototypes
        self.steps = config.steps
        self.seed = seed

        # Semilla aislada e independiente de mode y position: variante y controles arrancan iguales.
        derived = hashlib.sha256(f"{HOPFIELD_RANK_BRAIN}/{seed}".encode()).digest()
        with torch.device("cpu"), torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(int.from_bytes(derived[:7], "little"))
            self.norm = nn.LayerNorm(hidden_size, eps=LAYER_NORM_EPS, dtype=dtype)
            self.prototypes = nn.Parameter(
                torch.randn((config.prototypes, hidden_size), dtype=dtype)
            )
            self.readout_norm = nn.LayerNorm(
                hidden_size, eps=LAYER_NORM_EPS, bias=False, dtype=dtype
            )
            self.out = nn.Linear(hidden_size, hidden_size, bias=False, dtype=dtype)

        beta_fraction = (BETA_INIT - BETA_MIN) / (BETA_MAX - BETA_MIN)
        step_fraction = STEP_INIT / STEP_BOUND
        self.beta_logit = nn.Parameter(torch.tensor(_logit(beta_fraction), dtype=dtype))
        self.eta_a_logit = nn.Parameter(torch.tensor(_logit(step_fraction), dtype=dtype))
        self.eta_r_logit = nn.Parameter(torch.tensor(_logit(step_fraction), dtype=dtype))
        # Puerta vectorial a cero sobre W_out Kaiming no nula: identidad inicial y gradiente vivo.
        self.gate = nn.Parameter(torch.zeros(hidden_size, dtype=dtype))

    def get_extra_state(self) -> dict:
        state = dict(configuration=self.config.identity(), hidden_size=self.hidden_size)
        if self.config.mode == SHUFFLED_PEERS:
            state["permutation_seed"] = self.seed
        return state

    def set_extra_state(self, state: object) -> None:
        require_identity(state, self.get_extra_state())

    def _normalized_prototypes(self) -> Tensor:
        rms = torch.sqrt(self.prototypes.square().mean(dim=-1, keepdim=True) + PROTOTYPE_NORM_EPS)
        return self.prototypes / rms

    def _beta(self) -> Tensor:
        return BETA_MIN + (BETA_MAX - BETA_MIN) * torch.sigmoid(self.beta_logit)

    def _eta_a(self) -> Tensor:
        return STEP_BOUND * torch.sigmoid(self.eta_a_logit)

    def _eta_r(self) -> Tensor:
        return STEP_BOUND * torch.sigmoid(self.eta_r_logit)

    def groups(self, markets, prediction_at):
        """Clave y filas de cada sesión del bloque, en el orden de primera aparición."""
        sessions = {}
        for row, key in enumerate(zip(markets, prediction_at, strict=True)):
            sessions.setdefault(key, []).append(row)
        return list(sessions.items())

    def _permutation(self, key, rows: int) -> Tensor:
        market, instant = key
        text = f"{HOPFIELD_RANK_BRAIN}/{self.seed}/{market}/{instant}"
        digest = hashlib.sha256(text.encode()).digest()
        generator = torch.Generator(device="cpu").manual_seed(int.from_bytes(digest[:7], "little"))
        return torch.randperm(rows, generator=generator)

    def _repulsive_field(self, z_hat: Tensor, alpha: Tensor) -> Tensor:
        """Campo repulsivo exacto O(N P H) con suma cero e invariancia a traslaciones."""
        n_p = alpha.sum(dim=0)
        inv_denom = 1.0 / (n_p + CROWDING_PRIOR)
        w_bar = n_p * inv_denom
        z_bar = (alpha.transpose(0, 1) @ z_hat) * inv_denom.unsqueeze(-1)
        self_scale = alpha @ w_bar
        return self_scale.unsqueeze(-1) * z_hat - (alpha @ z_bar)

    def _session_displacement(
        self,
        values: Tensor,
        prototypes: Tensor,
        scale: Tensor,
        eta_a: Tensor | None,
        eta_r: Tensor | None,
        perm: Tensor | None,
    ) -> Tensor:
        """Desplazamiento neto Δz = z^(J) − z^(0) para una sesión completa de N filas."""
        rows = len(values)
        z0 = F.layer_norm(values, (self.hidden_size,), eps=self.norm.eps)
        z = z0
        for _ in range(self.config.steps):
            z_hat = self.norm(z)
            logits = (z_hat @ prototypes.transpose(0, 1)) * scale
            alpha = entmax15(logits)
            step = None
            if eta_a is not None:
                step = eta_a * (alpha @ prototypes)
            if eta_r is not None:
                if rows == 1:
                    repul = z_hat.new_zeros(z_hat.shape) + (alpha.sum() + z_hat.sum() + eta_r) * 0
                else:
                    repul = self._repulsive_field(z_hat, alpha)
                    if perm is not None:
                        repul = repul.index_select(0, perm)
                repul_step = eta_r * repul
                step = repul_step if step is None else step + repul_step
            z = z + step
        return z - z0

    def forward(self, encoded: Tensor, markets, prediction_at) -> Tensor:
        reference = self.gate
        if (
            not isinstance(encoded, Tensor)
            or encoded.layout != torch.strided
            or encoded.ndim != 2
            or encoded.shape[1] != self.hidden_size
            or len(encoded) == 0
            or encoded.dtype != reference.dtype
            or encoded.device != reference.device
            or len(markets) != len(encoded)
            or len(prediction_at) != len(encoded)
        ):
            raise ValueError("RSHRB necesita una representación, un mercado y un instante por fila")
        if torch.is_autocast_enabled(encoded.device.type):
            raise ValueError("RSHRB exige FP32 o FP64 explícitos, sin autocast")
        check_finite(encoded, "La representación de entrada a RSHRB")

        prototypes = self._normalized_prototypes()
        scale = self._beta() / math.sqrt(self.hidden_size)
        mode = self.config.mode
        eta_a = None if mode == REPULSIVE_ONLY else self._eta_a()
        eta_r = None if mode == ATTRACTIVE_ONLY else self._eta_r()

        if mode == ATTRACTIVE_ONLY:
            displacement = self._session_displacement(encoded, prototypes, scale, eta_a, None, None)
        else:
            sessions = self.groups(markets, prediction_at)
            if len(sessions) == 1:
                key, _ = sessions[0]
                perm = None
                if mode == SHUFFLED_PEERS and len(encoded) > 1:
                    order = self._permutation(key, len(encoded))
                    perm = (
                        order.pin_memory().to(device=encoded.device, non_blocking=True)
                        if encoded.is_cuda
                        else order.to(encoded.device)
                    )
                displacement = self._session_displacement(
                    encoded, prototypes, scale, eta_a, eta_r, perm
                )
            else:
                displacement = encoded.new_zeros(encoded.shape)
                for key, members in sessions:
                    size = len(members)
                    contiguous = members[-1] - members[0] + 1 == size and members == list(
                        range(members[0], members[0] + size)
                    )
                    if contiguous:
                        start, stop = members[0], members[0] + size
                        part_in = encoded[start:stop]
                    else:
                        if encoded.is_cuda:
                            index = torch.tensor(members, dtype=torch.int64, pin_memory=True).to(
                                device=encoded.device, non_blocking=True
                            )
                        else:
                            index = torch.tensor(members, dtype=torch.int64, device=encoded.device)
                        part_in = encoded.index_select(0, index)
                    perm = None
                    if mode == SHUFFLED_PEERS and size > 1:
                        order = self._permutation(key, size)
                        perm = (
                            order.pin_memory().to(device=encoded.device, non_blocking=True)
                            if encoded.is_cuda
                            else order.to(encoded.device)
                        )
                    part_out = self._session_displacement(
                        part_in, prototypes, scale, eta_a, eta_r, perm
                    )
                    if contiguous:
                        displacement = displacement.clone()
                        displacement[start:stop] = part_out
                    else:
                        displacement = displacement.index_copy(0, index, part_out)

        residual = self.out(self.readout_norm(displacement))
        check_finite(residual, "La proyección de RSHRB")
        result = encoded + self.gate * residual
        check_finite(result, "La representación con RSHRB")
        return result
