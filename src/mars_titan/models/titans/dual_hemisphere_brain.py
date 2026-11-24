"""Cerebro holográfico de doble hemisferio con ortogonalización transversal (CODHB).

Implementación de `dual_hemisphere_brain_v1`: añade dentro de `FinancialPredictor` un segundo
hemisferio estructuralmente distinto e independiente del bloque gráfico y de la rama técnica
(`technical_branch`), acoplado mediante proyección ortogonal transversal diferenciable sobre el
subespacio completo del hemisferio primario detenido en gradiente (`stopgrad`).

Diseño motivado por los hechos medidos de la campaña:
1. El hemisferio secundario h^{rel}_i se alimenta EXCLUSIVAMENTE de:
   - la estructura relacional entre activos de la ventana de precios y volumen (momento residual,
     rendimientos intradía/nocturnos y choques de volumen frente a pares co-móviles en retorno y
     en volumen, más acoplamiento/desacoplamiento estructural de pares), calculada en O(N·T·F)
     con cancelación leave-one-out exacta y pesos de suma cero (sin matrices N×N ni eco propio);
   - las modalidades no técnicas (noticias, fundamentales y contexto macro con modulación FiLM
     para sobrevivir al centrado transversal), con difusión causal entre pares co-móviles y sus
     bits de presencia. Nunca lee `charts` ni los 46 rasgos técnicos.
2. Para resolver la debilidad teórica de la ortogonalización coordenada a coordenada (donde dos
   vectores ortogonales dimensión a dimensión pueden combinarse en `W_out` o `head` para producir
   predicciones escalares correlacionadas), la ortogonalización proyecta fuera del SUBESPACIO
   COMPLETO de la representación primaria congelada en gradiente E = stopgrad(e^{core}):

       H̄ = (H^{rel} − mean_S(H^{rel})) / sqrt(var_S(H^{rel}) + ε_rms)
       Ê = normalizar_columnas_activas(E − mean_S(E))
       G = (1/N) Êᵀ Ê,    C = (1/N) Êᵀ H̄,    λ = ridge · tr(G) + ε_abs
       L Lᵀ = G + λ I_H   (Cholesky sin sincronizar el dispositivo)
       H⊥ = H̄ − Ê (G + λ I_H)⁻¹ C

   Propiedades algebraicas exactas:
   - Normalizar H̄ ANTES de proyectar evita reamplificar residuos O(λ) si H^{rel} cae en col(Ê).
   - Para cualquier proyección lineal W_out, ganancia g y vector de cabeza w_head (incluso con
     `zero_median=True` al inicio y para los 5 cuantiles de `QuantileHead`), la contribución
     escalar Δŷ = w_headᵀ (g ⊙ W_out H⊥) satisface Cov_S(Δŷ, ŷ^{core}) = O(λ) ≈ 0.
   - ∂Δ / ∂e^{core} ≡ 0 exactamente por `stopgrad`: el ortogonalizador nunca distorsiona los
     gradientes del tronco primario ni de la memoria rápida MAC.
   - Inicialización neutra sin doble cero: W_out arranca Kaiming (no nula) y la puerta vectorial
     g arranca en cero, e'_i = e_i + g ⊙ W_out h⊥_i, de modo que el predictor parte bit a bit de
     su control y g recibe gradiente desde el primer paso.

Modos científicos con exactamente los mismos parámetros e inicialización:
- `orthogonal`: CODHB completo (hemisferio relacional no técnico + proyección de subespacio).
- `unorthogonalized`: mismo hemisferio relacional sin restar la proyección sobre Ê (aísla el
  efecto propio de la ortogonalización transversal).
- `shuffled_peers`: control placebo que permuta deterministamente el hemisferio dentro de cada
  sesión (mercado, instante) antes de ortogonalizar (misma distribución marginal y ortogonalidad,
  correspondencia activo-pares destruida).
- `self_only`: control placebo estrictamente local por fila (sin mezcla entre activos ni
  ortogonalización transversal; separa la señal entre activos del aumento puro de capacidad).
"""

import hashlib
import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import bounded_integer, require_identity
from .state import check_finite, require_true

DUAL_HEMISPHERE_BRAIN = "dual_hemisphere_brain_v1"
ORTHOGONAL = "orthogonal"
UNORTHOGONALIZED = "unorthogonalized"
SELF_ONLY = "self_only"
SHUFFLED_PEERS = "shuffled_peers"
MODES = (ORTHOGONAL, UNORTHOGONALIZED, SELF_ONLY, SHUFFLED_PEERS)

MIN_DIM = 4
MAX_DIM = 256
DEFAULT_DIM = 32
MIN_RIDGE = 1e-6
MAX_RIDGE = 1.0
DEFAULT_RIDGE = 1e-4

RELATIONAL_FEATURES = 36
NON_CHART_PRESENCE = 4
SMOOTH_CLIP = 5.0
BASIS_EPS = 1e-8
PEER_EPS = 1e-6
STD_EPS = 1e-8
RMS_EPS = 1e-5
CORE_EPS = 1e-8
ORTHO_FLOOR = 1e-6
_VOL_PEER_COLS = (0, 1, 4, 8)
_SHORT_RET_COLS = (0, 1, 4, 5, 6, 7, 8, 9, 10, 11)
_LONG_RET_COLS = (2, 3)


def _finite_float(value: object, low: float, high: float, message: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not low <= value <= high
    ):
        raise ValueError(message)
    return float(value)


def _smooth_clip(values: Tensor, limit: float = SMOOTH_CLIP) -> Tensor:
    """Acotación suave monótona x / sqrt(1 + (x / limit)²), en (−limit, limit)."""
    scaled = values / limit
    return values * torch.rsqrt(1.0 + scaled.square())


def _time_unit_basis(series: Tensor, mask: Tensor) -> Tensor:
    """Centrar en el tiempo sobre pasos presentes y normalizar en L2 suave (‖z_i‖_2 < 1)."""
    counts = mask.sum(dim=-1, keepdim=True).clamp(min=1).to(series.dtype)
    mean = series.sum(dim=-1, keepdim=True) / counts
    centered = torch.where(mask, series - mean, torch.zeros_like(series))
    norm = torch.sqrt(centered.square().sum(dim=-1, keepdim=True) + BASIS_EPS)
    return centered / norm


def _window_sum(series: Tensor, horizon: int) -> Tensor:
    width = min(horizon, series.shape[1])
    return series[:, series.shape[1] - width :].sum(dim=1)


def _window_mean(series: Tensor, mask: Tensor, horizon: int) -> Tensor:
    width = min(horizon, series.shape[1])
    total = series[:, series.shape[1] - width :].sum(dim=1)
    count = mask[:, mask.shape[1] - width :].sum(dim=1).clamp(min=1).to(series.dtype)
    return total / count


def _standardize_session(values: Tensor) -> Tensor:
    """Centrado y estandarización transversal por columna dentro de la sesión."""
    centered = values - values.mean(dim=0, keepdim=True)
    scale = torch.sqrt(centered.square().mean(dim=0, keepdim=True) + STD_EPS)
    return centered / scale


def loo_peer_readout(basis: Tensor, values: Tensor) -> tuple[Tensor, Tensor]:
    """Lectura leave-one-out exacta de pares ponderada por correlación en O(N·K·(K+F)).

    Para cada fila i y cada par j ≠ i con correlación empírica ρ_ij = basis_i · basis_j:

        ρ̄_i = (1/(N−1)) Σ_{j≠i} ρ_ij
        σ_i = sqrt((1/(N−1)) Σ_{j≠i} (ρ_ij − ρ̄_i)² + PEER_EPS)
        P_i = ((1/(N−1)) Σ_{j≠i} (ρ_ij − ρ̄_i) · values_j) / σ_i

    Propiedades exactas: cero eco propio (∂P_i/∂values_i = 0) y pesos de suma cero
    (invariante a cualquier desplazamiento uniforme values_j → values_j + c).
    """
    rows = basis.shape[0]
    if rows <= 2:
        # Con 0 o 1 par (N <= 2), la covarianza centrada leave-one-out es idénticamente cero.
        zero_out = values * 0.0 + basis.sum() * 0.0
        zero_std = basis.new_full((rows,), math.sqrt(PEER_EPS))
        return zero_out, zero_std
    inv_peers = 1.0 / float(rows - 1)
    norm_sq = basis.square().sum(dim=-1)
    basis_sum = basis.sum(dim=0)
    values_sum = values.sum(dim=0)

    gram_bv = basis.transpose(0, 1) @ values
    raw_dot = (basis @ gram_bv - norm_sq.unsqueeze(-1) * values) * inv_peers
    rho_bar = (basis @ basis_sum - norm_sq) * inv_peers
    loo_mean = (values_sum.unsqueeze(0) - values) * inv_peers
    cov = raw_dot - rho_bar.unsqueeze(-1) * loo_mean

    gram_bb = basis.transpose(0, 1) @ basis
    rho_sq_sum = ((basis @ gram_bb) * basis).sum(dim=-1) - norm_sq.square()
    rho_var = (rho_sq_sum * inv_peers - rho_bar.square()).clamp(min=0.0)
    peer_std = torch.sqrt(rho_var + PEER_EPS)
    return cov / peer_std.unsqueeze(-1), peer_std


def _extract_price_channels(
    prices: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Rendimientos nocturnos, intradía, cierre-cierre, rango y volumen de la ventana."""
    steps, channels = prices.shape[1], prices.shape[2]
    opening, high, low, close = prices[..., :4].unbind(-1)
    volume = prices[..., 4]
    if channels == 6:
        present = prices[..., 5] > 0
    else:
        present = torch.ones((prices.shape[0], steps), dtype=torch.bool, device=prices.device)
    index = torch.arange(steps, device=prices.device).unsqueeze(0).expand_as(close)
    last = torch.where(present, index, torch.full_like(index, -1)).cummax(dim=-1).values
    previous = torch.cat((torch.full_like(last[:, :1], -1), last[:, :-1]), dim=-1)
    known = present & (previous >= 0)
    prev_close = close.gather(-1, previous.clamp(min=0))
    zero = torch.zeros((), dtype=prices.dtype, device=prices.device)
    ret_on = torch.where(known, opening - prev_close, zero)
    ret_id = torch.where(present, close - opening, zero)
    ret_cc = torch.where(known, close - prev_close, zero)
    hl_range = torch.where(present, (high - low).clamp(min=0.0), zero)
    vol = torch.where(present, volume, zero)
    return ret_cc, ret_id, ret_on, hl_range, vol, present, known


def _relational_price_volume(prices: Tensor) -> tuple[Tensor, Tensor]:
    """36 señales relacionales de precio/volumen entre pares y base Z_full de la sesión."""
    rows, steps, _ = prices.shape
    ret_cc, ret_id, ret_on, hl_range, vol, present, known = _extract_price_channels(prices)
    if rows <= 1:
        z_full = _time_unit_basis(ret_cc, known)
        return prices.new_zeros((rows, RELATIONAL_FEATURES)), z_full

    zero = torch.zeros((), dtype=prices.dtype, device=prices.device)
    cnt_known = known.sum(dim=0, keepdim=True).clamp(min=1).to(prices.dtype)
    cnt_present = present.sum(dim=0, keepdim=True).clamp(min=1).to(prices.dtype)
    mkt_cc = ret_cc.sum(dim=0, keepdim=True) / cnt_known
    mkt_id = ret_id.sum(dim=0, keepdim=True) / cnt_present
    mkt_on = ret_on.sum(dim=0, keepdim=True) / cnt_known
    mkt_vol = vol.sum(dim=0, keepdim=True) / cnt_present
    mkt_range = hl_range.sum(dim=0, keepdim=True) / cnt_present

    resid_cc = torch.where(known, ret_cc - mkt_cc, zero)
    resid_id = torch.where(present, ret_id - mkt_id, zero)
    resid_on = torch.where(known, ret_on - mkt_on, zero)
    resid_vol = torch.where(present, vol - mkt_vol, zero)
    resid_range = torch.where(present, hl_range - mkt_range, zero)

    signals = torch.stack(
        (
            _window_sum(resid_cc, 1),
            _window_sum(resid_cc, 5),
            _window_sum(resid_cc, 21),
            _window_sum(resid_cc, steps),
            _window_sum(resid_id, 1),
            _window_sum(resid_id, 5),
            _window_sum(resid_on, 1),
            _window_sum(resid_on, 5),
            _window_mean(resid_vol, present, 1),
            _window_mean(resid_vol, present, 5) - _window_mean(resid_vol, present, steps),
            _window_mean(resid_range, present, 5),
            _window_sum(resid_cc * resid_vol, 5),
        ),
        dim=-1,
    )
    x_std = _standardize_session(signals)

    z_full = _time_unit_basis(resid_cc, known)
    hist_end = steps - 5 if steps >= 12 else steps
    z_hist = _time_unit_basis(resid_cc[:, :hist_end], known[:, :hist_end])
    rec_start = max(0, steps - 5)
    z_rec = _time_unit_basis(resid_cc[:, rec_start:], known[:, rec_start:])
    z_vol = _time_unit_basis(resid_vol, present)

    # Pares históricos sin solape temporal para horizontes de 1 y 5 sesiones; ventana completa
    # para horizontes de 21 y 63 sesiones.
    short_idx = torch.tensor(_SHORT_RET_COLS, dtype=torch.int64, device=prices.device)
    long_idx = torch.tensor(_LONG_RET_COLS, dtype=torch.int64, device=prices.device)
    vol_idx = torch.tensor(_VOL_PEER_COLS, dtype=torch.int64, device=prices.device)

    peer_short, _ = loo_peer_readout(z_hist, x_std.index_select(1, short_idx))
    peer_long, ret_peer_std = loo_peer_readout(z_full, x_std.index_select(1, long_idx))
    peer_ret = torch.zeros_like(x_std)
    peer_ret = peer_ret.index_copy(1, short_idx, peer_short)
    peer_ret = peer_ret.index_copy(1, long_idx, peer_long)
    resid_ret = x_std - peer_ret

    peer_vol, vol_peer_std = loo_peer_readout(z_vol, x_std.index_select(1, vol_idx))
    resid_vol_sig = x_std.index_select(1, vol_idx) - peer_vol

    # 4 escalares estructurales del grafo de pares: acoplamiento en retorno, acoplamiento en
    # volumen, correlación con el mercado equiponderado y alineación reciente frente a histórica.
    mkt_basis = _time_unit_basis(mkt_cc.expand(rows, steps), known)
    asset_basis = _time_unit_basis(ret_cc, known)
    market_corr = (asset_basis * mkt_basis).sum(dim=-1)

    inv_peers = 1.0 / float(rows - 1)
    zh_sq = z_hist.square().sum(dim=-1)
    zr_sq = z_rec.square().sum(dim=-1)
    cross_hr = z_hist.transpose(0, 1) @ z_rec
    align_raw = ((z_hist @ cross_hr) * z_rec).sum(dim=-1) - zh_sq * zr_sq
    v_hist = (
        ((z_hist @ (z_hist.transpose(0, 1) @ z_hist)) * z_hist).sum(dim=-1) - zh_sq.square()
    ).clamp(min=0.0) * inv_peers
    v_rec = (
        ((z_rec @ (z_rec.transpose(0, 1) @ z_rec)) * z_rec).sum(dim=-1) - zr_sq.square()
    ).clamp(min=0.0) * inv_peers
    peer_align = (align_raw * inv_peers) / (
        torch.sqrt(v_hist + PEER_EPS) * torch.sqrt(v_rec + PEER_EPS)
    )

    structural = torch.stack(
        (
            _standardize_session(ret_peer_std.unsqueeze(-1)).squeeze(-1),
            _standardize_session(vol_peer_std.unsqueeze(-1)).squeeze(-1),
            market_corr,
            peer_align,
        ),
        dim=-1,
    )
    features = torch.cat((peer_ret, resid_ret, peer_vol, resid_vol_sig, structural), dim=-1)
    return _smooth_clip(features), z_full


def _self_only_price_volume(prices: Tensor) -> Tensor:
    """36 rasgos puramente locales por activo (cero lectura de otros activos de la sesión)."""
    steps = prices.shape[1]
    ret_cc, ret_id, ret_on, hl_range, vol, present, known = _extract_price_channels(prices)
    cnt_k = known.sum(dim=-1).clamp(min=1).to(prices.dtype)
    cnt_p = present.sum(dim=-1).clamp(min=1).to(prices.dtype)
    ret_mean = ret_cc.sum(dim=-1) / cnt_k
    ret_var = (
        torch.where(known, (ret_cc - ret_mean.unsqueeze(-1)).square(), torch.zeros_like(ret_cc))
        .sum(dim=-1)
        .div(cnt_k)
    )
    ret_scale = torch.sqrt(ret_var + 1e-6)
    vol_mean = vol.sum(dim=-1) / cnt_p
    vol_var = (
        torch.where(present, (vol - vol_mean.unsqueeze(-1)).square(), torch.zeros_like(vol))
        .sum(dim=-1)
        .div(cnt_p)
    )
    vol_scale = torch.sqrt(vol_var + 1e-6)
    range_mean = hl_range.sum(dim=-1) / cnt_p
    range_scale = range_mean + 1e-4

    own_12 = torch.stack(
        (
            _window_sum(ret_cc, 1) / ret_scale,
            _window_sum(ret_cc, 5) / (ret_scale * math.sqrt(min(5, steps))),
            _window_sum(ret_cc, 21) / (ret_scale * math.sqrt(min(21, steps))),
            _window_sum(ret_cc, steps) / (ret_scale * math.sqrt(steps)),
            _window_sum(ret_id, 1) / ret_scale,
            _window_sum(ret_id, 5) / (ret_scale * math.sqrt(min(5, steps))),
            _window_sum(ret_on, 1) / ret_scale,
            _window_sum(ret_on, 5) / (ret_scale * math.sqrt(min(5, steps))),
            (_window_mean(vol, present, 1) - vol_mean) / vol_scale,
            (_window_mean(vol, present, 5) - vol_mean) / vol_scale,
            (_window_mean(hl_range, present, 5) - range_mean) / range_scale,
            _window_sum(ret_cc * (vol - vol_mean.unsqueeze(-1)), 5) / (ret_scale * vol_scale),
        ),
        dim=-1,
    )
    vol_idx = torch.tensor(_VOL_PEER_COLS, dtype=torch.int64, device=prices.device)
    own_4 = own_12.index_select(1, vol_idx)
    rec_vol = torch.sqrt(_window_mean(ret_cc.square(), known, 5) + 1e-8) / ret_scale
    own_struct = torch.stack(
        (
            torch.log(ret_scale + 1e-4),
            torch.log(vol_scale + 1e-4),
            torch.log(range_scale),
            rec_vol - 1.0,
        ),
        dim=-1,
    )
    features = torch.cat((own_12, own_12, own_4, own_4, own_struct), dim=-1)
    return _smooth_clip(features)


@dataclass(frozen=True)
class DualHemisphereBrainConfig:
    """Contrato del cerebro holográfico de doble hemisferio con ortogonalización (CODHB)."""

    dim: int = DEFAULT_DIM
    ridge: float = DEFAULT_RIDGE
    mode: str = ORTHOGONAL

    def __post_init__(self) -> None:
        bounded_integer(self.dim, "dimensión del segundo hemisferio CODHB", MIN_DIM, MAX_DIM)
        ridge = _finite_float(
            self.ridge,
            MIN_RIDGE,
            MAX_RIDGE,
            f"ridge de CODHB debe ser un real finito en [{MIN_RIDGE}, {MAX_RIDGE}]",
        )
        if type(self.mode) is not str or self.mode not in MODES:
            raise ValueError(f"El modo de CODHB debe ser uno de {MODES}")
        object.__setattr__(self, "ridge", ridge)

    def identity(self) -> dict:
        groups = (
            "each_row_with_itself"
            if self.mode == SELF_ONLY
            else "same_market_and_prediction_instant_in_the_block"
        )
        orthogonalization = dict(
            orthogonal="full_subspace_cholesky_gram_schmidt_against_stopgrad_core",
            unorthogonalized="none_session_centered_column_rms_only",
            shuffled_peers="session_permuted_hemisphere_then_full_subspace_gram_schmidt",
            self_only="none_row_local_rms_only",
        )[self.mode]
        return dict(
            schema_version=1,
            kind=DUAL_HEMISPHERE_BRAIN,
            dim=self.dim,
            ridge=self.ridge,
            ortho_floor=ORTHO_FLOOR,
            mode=self.mode,
            groups=groups,
            position="after_session_readings_and_technical_before_crrm_and_head",
            hemisphere_inputs=(
                "non_technical_modalities_and_cross_asset_price_volume_relations_"
                "excluding_charts_and_technical_branch"
            ),
            relational_features=RELATIONAL_FEATURES,
            peer_operator="exact_leave_one_out_zero_sum_correlation_weighted_peers",
            macro_conditioning="film_one_plus_tanh_before_session_centering",
            orthogonalization=orthogonalization,
            core_gradient="stopgrad_on_primary_hemisphere_representation",
            output="residual_with_zero_initialized_vector_gate_over_kaiming_projection",
            state="none",
            parameter_updates="none_in_forward",
        )


def parse_dual_hemisphere_brain(raw: object) -> DualHemisphereBrainConfig | None:
    """Aceptar DualHemisphereBrainConfig, su diccionario declarado o None."""
    if raw is None:
        return None
    if type(raw) is DualHemisphereBrainConfig:
        raw.__post_init__()
        return raw
    required = {"dim", "mode"}
    allowed = required | {"ridge"}
    if isinstance(raw, dict) and required <= set(raw) <= allowed:
        return DualHemisphereBrainConfig(**raw)
    raise ValueError(
        "dual_hemisphere_brain debe declarar dim y mode (y opcionalmente ridge), o ser None"
    )


class DualHemisphereBrain(nn.Module):
    """Segundo hemisferio relacional no técnico acoplado por ortogonalización transversal."""

    session_level = True

    def __init__(
        self,
        hidden_size: int,
        dimensions: dict[str, int],
        config: DualHemisphereBrainConfig,
        *,
        seed: int,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        bounded_integer(hidden_size, "dimensión oculta de CODHB", 1, 4096)
        bounded_integer(seed, "semilla de CODHB", 0, 2**32 - 1)
        if type(config) is not DualHemisphereBrainConfig:
            raise ValueError("CODHB necesita DualHemisphereBrainConfig")
        config.__post_init__()
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("CODHB exige precisión FP32 o FP64")
        if (
            not isinstance(dimensions, dict)
            or not {"prices", "news", "fundamentals", "macro"} <= set(dimensions)
            or dimensions["prices"] not in (5, 6)
        ):
            raise ValueError(
                "CODHB necesita las dimensiones de las modalidades no técnicas "
                "(prices, news, fundamentals, macro)"
            )
        for name in ("news", "fundamentals", "macro"):
            bounded_integer(dimensions[name], f"dimensión {name} de CODHB", 1, 2048)

        self.config = config
        self.hidden_size = hidden_size
        self.seed = seed
        self.news_dim = int(dimensions["news"])
        self.fundamentals_dim = int(dimensions["fundamentals"])
        self.macro_dim = int(dimensions["macro"])
        dim = config.dim

        # Semilla aislada e independiente de mode/ridge: candidato y controles arrancan iguales.
        derived = hashlib.sha256(f"{DUAL_HEMISPHERE_BRAIN}/{seed}".encode()).digest()
        with torch.device("cpu"), torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(int.from_bytes(derived[:7], "little"))
            self.rel_proj = nn.Linear(RELATIONAL_FEATURES + NON_CHART_PRESENCE, dim, dtype=dtype)
            self.news_proj = nn.Linear(self.news_dim, dim, dtype=dtype)
            self.fundamentals_proj = nn.Linear(self.fundamentals_dim, dim, dtype=dtype)
            self.macro_proj = nn.Linear(self.macro_dim, dim, dtype=dtype)
            self.hemisphere_norm = nn.LayerNorm(2 * dim, dtype=dtype)
            self.fc1 = nn.Linear(2 * dim, dim, dtype=dtype)
            self.fc2 = nn.Linear(dim, dim, dtype=dtype)
            self.out = nn.Linear(dim, hidden_size, bias=False, dtype=dtype)
        # Macro FiLM arranca neutro (1 + tanh(0) = 1, derivada 1) y puerta vectorial en cero sobre
        # W_out Kaiming no nula (sin doble cero en serie).
        nn.init.zeros_(self.macro_proj.weight)
        nn.init.zeros_(self.macro_proj.bias)
        self.gate = nn.Parameter(torch.zeros(hidden_size, dtype=dtype))

    def get_extra_state(self) -> dict:
        state = dict(
            configuration=self.config.identity(),
            hidden_size=self.hidden_size,
            non_technical_dimensions=dict(
                news=self.news_dim,
                fundamentals=self.fundamentals_dim,
                macro=self.macro_dim,
            ),
        )
        if self.config.mode == SHUFFLED_PEERS:
            state["permutation_seed"] = self.seed
        return state

    def set_extra_state(self, state: object) -> None:
        require_identity(state, self.get_extra_state())

    def groups(self, markets, prediction_at):
        """Clave y filas de cada sesión del bloque, en el orden de primera aparición."""
        sessions = {}
        for row, key in enumerate(zip(markets, prediction_at, strict=True)):
            sessions.setdefault(key, []).append(row)
        return list(sessions.items())

    def _permutation(self, key: tuple[str, int], rows: int) -> Tensor:
        market, instant = key
        text = f"{DUAL_HEMISPHERE_BRAIN}/shuffle/{self.seed}/{market}/{instant}"
        digest = hashlib.sha256(text.encode()).digest()
        generator = torch.Generator(device="cpu").manual_seed(int.from_bytes(digest[:7], "little"))
        order = torch.randperm(rows, generator=generator)
        # Garantiza romper la identidad para todo N >= 2 de forma determinista.
        if rows >= 2 and torch.equal(order, torch.arange(rows, dtype=torch.int64)):
            order = torch.roll(order, shifts=1)
        return order

    def _hemisphere_session(
        self,
        raw_prices: Tensor,
        news: Tensor,
        fundamentals: Tensor,
        macro: Tensor,
        presence: Tensor,
    ) -> Tensor:
        """Calcular h^{rel} [N, dim] en una sesión sin tocar `charts` ni el núcleo primario."""
        dtype = news.dtype
        pres_non_chart = torch.stack(
            (presence[:, 0], presence[:, 1], presence[:, 3], presence[:, 4]), dim=-1
        ).to(dtype=dtype)

        if self.config.mode == SELF_ONLY:
            rel_features = _self_only_price_volume(raw_prices)
            z_full = None
        else:
            rel_features, z_full = _relational_price_volume(raw_prices)

        u_self = F.silu(self.rel_proj(torch.cat((rel_features, pres_non_chart), dim=-1)))
        news_emb = F.silu(self.news_proj(news)) * presence[:, 1:2].to(dtype=dtype)
        fund_emb = F.silu(self.fundamentals_proj(fundamentals)) * presence[:, 3:4].to(dtype=dtype)
        u_self = u_self + news_emb + fund_emb

        if self.config.mode == SELF_ONLY:
            u_peer = u_self
        else:
            u_peer, _ = loo_peer_readout(z_full, u_self)

        combined = self.hemisphere_norm(torch.cat((u_self, u_peer), dim=-1))
        hidden = self.fc1(combined)
        macro_mod = torch.where(
            presence[:, 4:5],
            1.0 + torch.tanh(self.macro_proj(macro)),
            torch.ones_like(hidden),
        )
        return self.fc2(F.silu(macro_mod * hidden))

    def _orthogonalize_session(self, hemisphere: Tensor, core: Tensor) -> Tensor:
        """Proyectar el segundo hemisferio fuera del subespacio completo de `stopgrad(core)`."""
        rows = hemisphere.shape[0]
        if self.config.mode == SELF_ONLY:
            rms = torch.sqrt(hemisphere.square().mean(dim=-1, keepdim=True) + RMS_EPS)
            return hemisphere / rms
        if rows <= 1:
            return hemisphere * 0.0

        h_centered = hemisphere - hemisphere.mean(dim=0, keepdim=True)
        h_rms = torch.sqrt(h_centered.square().mean(dim=0, keepdim=True) + RMS_EPS)
        h_norm = h_centered / h_rms
        if self.config.mode == UNORTHOGONALIZED:
            return h_norm

        core_detached = core.detach()
        e_centered = core_detached - core_detached.mean(dim=0, keepdim=True)
        e_var = e_centered.square().mean(dim=0, keepdim=True)
        active = e_var > 0
        e_scale = torch.sqrt(torch.where(active, e_var, torch.ones_like(e_var)) + CORE_EPS)
        e_norm = torch.where(active, e_centered / e_scale, torch.zeros_like(e_centered))

        inv_n = 1.0 / float(rows)
        gram = (e_norm.transpose(0, 1) @ e_norm) * inv_n
        gram = 0.5 * (gram + gram.transpose(0, 1))
        cross = (e_norm.transpose(0, 1) @ h_norm) * inv_n
        trace = torch.diagonal(gram).sum()
        lam = self.config.ridge * trace + ORTHO_FLOOR
        eye = torch.eye(self.hidden_size, dtype=hemisphere.dtype, device=hemisphere.device)
        # Sin sincronizar la GPU: el código de estado de Cholesky se comprueba diferido.
        chol, info = torch.linalg.cholesky_ex(gram + lam * eye, check_errors=False)
        require_true(
            (info == 0).all(),
            "El sistema de Gram de CODHB no es definido positivo",
        )
        coeffs = torch.cholesky_solve(cross, chol)
        return h_norm - e_norm @ coeffs

    def _session_output(
        self,
        key: tuple[str, int],
        core: Tensor,
        raw_prices: Tensor,
        news: Tensor,
        fundamentals: Tensor,
        macro: Tensor,
        presence: Tensor,
    ) -> Tensor:
        hemisphere = self._hemisphere_session(raw_prices, news, fundamentals, macro, presence)
        if self.config.mode == SHUFFLED_PEERS and hemisphere.shape[0] > 1:
            order = self._permutation(key, hemisphere.shape[0]).to(hemisphere.device)
            hemisphere = hemisphere.index_select(0, order)
        return self._orthogonalize_session(hemisphere, core)

    def forward(
        self,
        encoded: Tensor,
        raw_prices: Tensor,
        news: Tensor,
        fundamentals: Tensor,
        macro: Tensor,
        presence: Tensor | None,
        markets,
        prediction_at,
    ) -> Tensor:
        reference = self.gate
        if (
            not isinstance(encoded, Tensor)
            or encoded.layout != torch.strided
            or encoded.ndim != 2
            or encoded.shape[1] != self.hidden_size
            or len(encoded) == 0
            or encoded.dtype != reference.dtype
            or encoded.device != reference.device
            or not isinstance(raw_prices, Tensor)
            or raw_prices.ndim != 3
            or raw_prices.shape[0] != len(encoded)
            or raw_prices.shape[1] < 2
            or raw_prices.shape[2] not in (5, 6)
            or raw_prices.dtype != reference.dtype
            or raw_prices.device != reference.device
            or not isinstance(news, Tensor)
            or news.shape != (len(encoded), self.news_dim)
            or news.dtype != reference.dtype
            or news.device != reference.device
            or not isinstance(fundamentals, Tensor)
            or fundamentals.shape != (len(encoded), self.fundamentals_dim)
            or fundamentals.dtype != reference.dtype
            or fundamentals.device != reference.device
            or not isinstance(macro, Tensor)
            or macro.shape != (len(encoded), self.macro_dim)
            or macro.dtype != reference.dtype
            or macro.device != reference.device
            or len(markets) != len(encoded)
            or len(prediction_at) != len(encoded)
        ):
            raise ValueError(
                "CODHB necesita representación, modalidades no técnicas, mercado e instante"
            )
        if presence is None:
            presence = torch.ones((len(encoded), 5), dtype=torch.bool, device=encoded.device)
        elif (
            not isinstance(presence, Tensor)
            or presence.dtype != torch.bool
            or presence.shape != (len(encoded), 5)
            or presence.device != encoded.device
        ):
            raise ValueError("CODHB necesita los cinco bits booleanos de presencia por fila")
        if torch.is_autocast_enabled(encoded.device.type):
            raise ValueError("CODHB exige precisión FP32 o FP64 explícita, sin autocast")

        check_finite(encoded, "La representación primaria de entrada a CODHB")
        check_finite(raw_prices, "Los precios de entrada a CODHB")
        check_finite(news, "Las noticias de entrada a CODHB")
        check_finite(fundamentals, "Los fundamentales de entrada a CODHB")
        check_finite(macro, "El contexto macro de entrada a CODHB")

        rows = len(encoded)
        if self.config.mode == SELF_ONLY:
            orth = self._session_output(
                ("self", 0), encoded, raw_prices, news, fundamentals, macro, presence
            )
        else:
            sessions = self.groups(markets, prediction_at)
            if len(sessions) == 1:
                key, _ = sessions[0]
                orth = self._session_output(
                    key, encoded, raw_prices, news, fundamentals, macro, presence
                )
            else:
                orth = encoded.new_zeros((rows, self.config.dim))
                for key, members in sessions:
                    if encoded.is_cuda:
                        index = torch.tensor(members, dtype=torch.int64, pin_memory=True).to(
                            device=encoded.device, non_blocking=True
                        )
                    else:
                        index = torch.tensor(members, dtype=torch.int64, device=encoded.device)
                    part = self._session_output(
                        key,
                        encoded.index_select(0, index),
                        raw_prices.index_select(0, index),
                        news.index_select(0, index),
                        fundamentals.index_select(0, index),
                        macro.index_select(0, index),
                        presence.index_select(0, index),
                    )
                    orth = orth.index_copy(0, index, part)

        residual = self.out(orth)
        check_finite(residual, "La proyección ortogonal de CODHB")
        result = encoded + self.gate * residual
        check_finite(result, "La representación con cerebro de doble hemisferio CODHB")
        return result
