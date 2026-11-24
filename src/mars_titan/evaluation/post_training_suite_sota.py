"""Entorno Stage-2 de Post-Entrenamiento SOTA y Exportador de Trazas/Tablas TFM.

Consume limpiamente las emisiones y trazas de cualquiera de las 4 familias independientes
pre-entrenadas (`titans_original`, `holographic_brain`, `mmarrs_titan`, `titans_cognitive_core`)
bajo causalidad estricta (`cal_split -> eval_split`) y cumplimiento inviolable del sello `2024+`:

  1. `OrderLevelDecoupler` (FM-7 / F7-b): ajusta escala puntual causal `lambda > 0` (Huber / LAD)
     en calibración reduciendo MAE/RMSE sin alterar un solo bit el orden transversal (Rank IC).
  2. `Stage2PostTrainingPipeline`:
     - (a) Desacoplamiento Orden+Nivel (FM-7 / F7-b).
     - (b) Calibración Conforme `NormalizedConformalPID` (estratificación Mondrian por deciles de
       volatilidad para intervalos 80%/90%/95% no cruzados) + `VennAbersCalibrator` (`[p0, p1]`).
     - (c) Combinación causal ortogonal `OAS-ICIR` + contracción risk-parity entre familias,
       gobernador metacognitivo `MetacognitiveGovernor.decide_torch` + optimizador convexo GPU
       `ConvexPortfolioOptimizerPPlus.solve_torch`, tablero granular de 6 dimensiones (32 métricas)
       y contraste pareado Stage-2 vs Stage-1 con errores estándar HAC Newey-West.
  3. `TFMTraceAndTableExporter`:
     - Consolida `telemetry.jsonl`, `experiment_memory.jsonl` (si existe), evaluaciones 6D,
       veredictos `GainValidationGate` y resultados Stage-2.
     - Exporta las 4 tablas listas para tesis en Markdown y LaTeX (`booktabs`) + resumen JSON:
       (i) Comparativa maestra de las 4 familias, (ii) Controles emparejados y validación de
       ganancia, (iii) Trazas diagnósticas Tool-Use / MoE / Loop-Depth / Neuro-Plasticity, y
       (iv) Ganancia Stage-1 Pre-Training vs Stage-2 Post-Training.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy import optimize

from mars_titan.calibration.adaptive_conformal_pid import (
    NormalizedConformalPID,
    make_stratified_groups,
    replay_normalized_pid,
)
from mars_titan.calibration.venn_abers_ltt import VennAbersCalibrator
from mars_titan.environments.cohorts import FINAL_TEST_START_US
from mars_titan.evaluation import online_aggregator as oa
from mars_titan.evaluation.enhanced_posttraining import (
    oas_shrink_covariance,
    shrinkage_icir_weights,
)
from mars_titan.evaluation.granular_evaluator import (
    QUANTILE_LEVELS_9,
    evaluate_granular_six_dimensions,
)
from mars_titan.evaluation.metacognitive_agent import (
    ConvexPortfolioOptimizerPPlus,
    MetacognitiveGovernor,
)
from mars_titan.models.mmarrs.frontier_modules_sota import (
    compute_cross_family_disagreement_risk,
)
from mars_titan.models.mmarrs.long_running_harness import (
    GainVerdict,
    SealedTestViolationError,
    validate_no_sealed_leakage,
)
from mars_titan.models.mmarrs.telemetry import read_telemetry_jsonl

LocallyNormalizedConformalPID = NormalizedConformalPID

FOUR_FAMILIES: tuple[str, ...] = (
    "titans_original",
    "holographic_brain",
    "mmarrs_titan",
    "titans_cognitive_core",
)

EXTENDED_QUANTILE_LEVELS_11: tuple[float, ...] = (
    0.01,
    0.025,
    0.05,
    0.10,
    0.25,
    0.50,
    0.75,
    0.90,
    0.95,
    0.975,
    0.99,
)

_QUANTILE_9_IN_11_IDX: tuple[int, ...] = (0, 2, 3, 4, 5, 6, 7, 8, 10)
_STANDARD_Z9 = np.array(
    [-2.326348, -1.644854, -1.281552, -0.674490, 0.0, 0.674490, 1.281552, 1.644854, 2.326348],
    dtype=np.float64,
)
_BLEND_MODES: tuple[str, ...] = ("oas_icir_rp", "oas_icir", "risk_parity", "equal")
_POINT_LOSSES: tuple[str, ...] = ("huber", "lad")
_STAGE2_MARKETS: tuple[str, ...] = ("US", "CN")


def _utc_now_iso() -> str:
    return datetime.now(tz=UTC).isoformat()


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _is_gzip_file(path: Path) -> bool:
    if path.suffix == ".gz":
        return True
    try:
        with path.open("rb") as fh:
            return fh.read(2) == b"\x1f\x8b"
    except OSError:
        return False


def _json_clean(obj: Any) -> Any:
    """Convierte estructuras arbitrarias a JSON estricto (NaN/Inf -> None, sin pérdidas)."""
    if obj is None or isinstance(obj, bool | str | int):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return _json_clean(obj.tolist())
    if isinstance(obj, np.generic):
        return _json_clean(obj.item())
    if isinstance(obj, torch.Tensor):
        return _json_clean(obj.detach().cpu().tolist())
    if isinstance(obj, Mapping):
        return {str(k): _json_clean(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_json_clean(v) for v in obj]
    return str(obj)


def _read_jsonl_optional(path: str | Path | None) -> list[dict[str, Any]]:
    """Lector transparente `.jsonl` y `.jsonl.gz` (comprimido sin pérdida)."""
    if path is None:
        return []
    p = Path(path)
    if not p.is_file():
        return []
    opener = gzip.open if _is_gzip_file(p) else open
    records: list[dict[str, Any]] = []
    with opener(p, "rt", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSONL corrupto en {p} línea {line_no}: {exc}") from exc
    return records


def read_compressed_trace_bundle(path: str | Path) -> list[dict[str, Any]]:
    """Lee un paquete de trazas `.jsonl.gz` o `.jsonl` de forma transparente y sin pérdidas."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"No existe el paquete de trazas: {p}")
    return _read_jsonl_optional(p)


def write_compressed_jsonl(
    path: str | Path,
    records: Sequence[Mapping[str, Any]],
    *,
    compresslevel: int = 6,
) -> dict[str, Any]:
    """Escribe JSONL ultra-compacto (`separators=(',',':')`) en `.jsonl.gz` o `.jsonl`."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f".{p.name}.tmp.{os.getpid()}")

    raw_chunks: list[bytes] = []
    for rec in records:
        clean = _json_clean(rec)
        line = json.dumps(clean, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        raw_chunks.append(line.encode("utf-8"))
    raw_bytes = b"".join(raw_chunks)
    digest = hashlib.sha256(raw_bytes).hexdigest()

    use_gz = p.suffix == ".gz"
    if use_gz:
        with gzip.open(tmp, "wb", compresslevel=compresslevel) as fh:
            fh.write(raw_bytes)
    else:
        tmp.write_bytes(raw_bytes)
    os.replace(tmp, p)

    disk_bytes = int(p.stat().st_size)
    raw_len = len(raw_bytes)
    ratio = (float(raw_len) / float(max(disk_bytes, 1))) if raw_len > 0 else 1.0
    return dict(
        path=str(p),
        compressed=use_gz,
        n_records=len(records),
        uncompressed_bytes=raw_len,
        compressed_bytes=disk_bytes,
        compression_ratio=round(ratio, 3),
        sha256_uncompressed=digest,
    )


def _safe_float(val: Any) -> float | None:
    if val is None or isinstance(val, bool):
        return None
    try:
        fv = float(val)
    except (TypeError, ValueError):
        return None
    return fv if math.isfinite(fv) else None


def _latex_escape(text: str) -> str:
    """Escapa caracteres especiales de LaTeX preservando legibilidad en celdas de texto."""
    replacements = (
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("~", r"\textasciitilde{}"),
        ("^", r"\textasciicircum{}"),
    )
    out = str(text)
    for src, dst in replacements:
        out = out.replace(src, dst)
    return out


def _fmt_num(
    val: Any,
    spec: str = ".4f",
    *,
    signed: bool = False,
    latex: bool = False,
) -> str:
    fv = _safe_float(val)
    if fv is None:
        return "--" if latex else "—"
    prefix = "+" if (signed and fv > 0.0) else ""
    return f"{prefix}{fv:{spec}}"


@dataclass
class OrderLevelDecoupler:
    """Desacoplamiento Orden+Nivel (FM-7 / F7-b) preservando estrictamente el Rank IC.

    Ajusta una transformación afín estrictamente creciente `y_hat = lambda * s + c` con
    `lambda > 0` minimizando pérdida robusta LAD o Huber en el tramo de calibración. Al ser
    `lambda` estrictamente positivo, el orden relativo de todas las parejas de activos dentro de
    cada sesión es idéntico bit-a-bit (delta Rank IC == 0.0 exacto).
    """

    loss: str = "huber"
    huber_delta: float = 0.02
    lambda_min: float = 1e-4
    lambda_max: float = 2.0
    fit_intercept: bool = True
    lambda_scale: float = 1.0
    intercept: float = 0.0
    cal_raw_loss: float = 0.0
    cal_fitted_loss: float = 0.0

    def __post_init__(self) -> None:
        if self.loss not in _POINT_LOSSES:
            raise ValueError(f"loss inválido {self.loss!r}; válidos: {_POINT_LOSSES}")
        if self.huber_delta <= 0.0:
            raise ValueError("huber_delta debe ser > 0")
        if self.lambda_min <= 0.0 or self.lambda_max <= self.lambda_min:
            raise ValueError("Se requiere 0 < lambda_min < lambda_max")

    def _loss_value(self, resid: np.ndarray) -> float:
        abs_r = np.abs(resid)
        if self.loss == "lad":
            return float(np.mean(abs_r))
        d = self.huber_delta
        return float(np.mean(np.where(abs_r <= d, 0.5 * resid * resid, d * (abs_r - 0.5 * d))))

    def fit(self, signal: np.ndarray, target: np.ndarray) -> OrderLevelDecoupler:
        """Ajusta `(lambda_scale > 0, intercept)` estrictamente sobre calibración."""
        s = np.asarray(signal, dtype=np.float64).reshape(-1)
        y = np.asarray(target, dtype=np.float64).reshape(-1)
        if len(s) != len(y) or len(s) == 0:
            raise ValueError("signal y target deben ser no vacíos y de igual longitud")
        valid = np.isfinite(s) & np.isfinite(y)
        if not np.any(valid):
            raise ValueError("No hay observaciones finitas para OrderLevelDecoupler")
        sv = s[valid]
        yv = y[valid]

        self.cal_raw_loss = self._loss_value(yv - sv)

        def _best_intercept(lam: float) -> float:
            if not self.fit_intercept:
                return 0.0
            diff = yv - lam * sv
            if self.loss == "lad":
                return float(np.median(diff))
            # Punto fijo rápido Huber-M en 1D para el intercepto óptimo
            c = float(np.median(diff))
            d = self.huber_delta
            for _ in range(8):
                r = diff - c
                w = np.where(np.abs(r) <= d, 1.0, d / np.maximum(np.abs(r), 1e-12))
                c_next = float(np.sum(w * diff) / max(float(np.sum(w)), 1e-12))
                if abs(c_next - c) < 1e-10:
                    return c_next
                c = c_next
            return c

        def _obj(lam: float) -> float:
            c = _best_intercept(lam)
            return self._loss_value(yv - (lam * sv + c))

        res = optimize.minimize_scalar(
            _obj,
            bounds=(self.lambda_min, self.lambda_max),
            method="bounded",
            options={"xatol": 1e-7},
        )
        lam_opt = float(np.clip(res.x, self.lambda_min, self.lambda_max))
        c_opt = _best_intercept(lam_opt)
        self.lambda_scale = lam_opt
        self.intercept = c_opt
        self.cal_fitted_loss = self._loss_value(yv - (lam_opt * sv + c_opt))
        return self

    def transform(self, signal: np.ndarray) -> np.ndarray:
        """Aplica la transformación monótona estricta `lambda_scale * signal + intercept`."""
        s = np.asarray(signal, dtype=np.float64)
        return self.lambda_scale * s + self.intercept

    def to_dict(self) -> dict[str, Any]:
        return dict(
            loss=self.loss,
            huber_delta=self.huber_delta,
            lambda_scale=self.lambda_scale,
            intercept=self.intercept,
            cal_raw_loss=self.cal_raw_loss,
            cal_fitted_loss=self.cal_fitted_loss,
        )


@dataclass(frozen=True)
class Stage2Config:
    """Configuración inmutable del pipeline Stage-2 de post-entrenamiento SOTA."""

    sealed_test_start_year: int = 2024
    # (a) Desacoplamiento Orden+Nivel (FM-7 / F7-b)
    point_loss: str = "huber"
    huber_delta: float = 0.02
    lambda_min: float = 1e-4
    lambda_max: float = 2.0
    # (b) Calibración conforme Mondrian (80/90/95%) + Venn-Abers
    n_vol_strata: int = 10
    conformal_nominals: tuple[float, ...] = (0.80, 0.90, 0.95)
    conformal_scale_mode: str = "hybrid"
    pid_kp: float = 0.05
    pid_ki: float = 0.10
    pid_kd: float = 0.02
    venn_abers_half_life: float | None = 63.0
    # (c) Mezcla ortogonal OAS-ICIR + risk-parity + gobernador A5-A8 + cartera P+
    blend_mode: str = "oas_icir_rp"
    oas_lam: float = 0.25
    oas_half_life: float = 63.0
    oas_se_shrinkage: float = 0.50
    risk_parity_mix: float = 0.25
    epistemic_kappa: float = 0.35
    ood_threshold: float = 2.5
    entropy_weight: float = 0.002
    loop_energy_weight: float = 0.002
    risk_aversion: float = 2.0
    factor_penalty: float = 15.0
    l1_cost: float = 0.001
    impact_cost: float = 0.01
    gross_limit: float = 1.0
    pos_cap: float = 0.05
    # (c.3) Riesgo de desacuerdo epistémico entre familias en P+ (#195 #5C; 0.0 = control exacto)
    disagreement_kappa: float = 0.0
    cka_weights: tuple[float, ...] | None = None
    cka_matrix: tuple[tuple[float, ...], ...] | None = None
    spread_bps: float = 5.0
    impact_eta: float = 0.15
    aum_musd: float = 50.0
    market: str = "US"
    device: str = "cpu"
    min_assets_per_session: int = 10

    def __post_init__(self) -> None:
        if self.point_loss not in _POINT_LOSSES:
            raise ValueError(f"point_loss inválido {self.point_loss!r}")
        if self.blend_mode not in _BLEND_MODES:
            raise ValueError(f"blend_mode inválido {self.blend_mode!r}; válidos: {_BLEND_MODES}")
        if self.market not in _STAGE2_MARKETS:
            raise ValueError(
                f"market inválido {self.market!r}; válidos: {_STAGE2_MARKETS} "
                "(FM-8 prohíbe mezclar mercados)"
            )
        if self.n_vol_strata < 1 or self.n_vol_strata > 20:
            raise ValueError("n_vol_strata debe estar en [1, 20]")
        if not (0.0 <= self.risk_parity_mix <= 1.0):
            raise ValueError("risk_parity_mix debe estar en [0, 1]")
        if self.gross_limit <= 0.0 or self.pos_cap <= 0.0:
            raise ValueError("gross_limit y pos_cap deben ser > 0")
        if self.disagreement_kappa < 0.0:
            raise ValueError("disagreement_kappa debe ser >= 0.0")
        if self.sealed_test_start_year < 2001:
            raise ValueError("sealed_test_start_year debe ser >= 2001")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_cross_family_disagreement_stack(
    family_preds: Sequence[np.ndarray | torch.Tensor]
    | Mapping[str, np.ndarray | torch.Tensor]
    | np.ndarray
    | torch.Tensor,
    *,
    realized_vol: np.ndarray | torch.Tensor,
    exposures: np.ndarray | torch.Tensor | None = None,
    w_prev: np.ndarray | torch.Tensor | None = None,
    disagreement_kappa: float = 0.0,
    cka_weights: Sequence[float] | np.ndarray | torch.Tensor | None = None,
    cka_matrix: Sequence[Sequence[float]] | np.ndarray | torch.Tensor | None = None,
    vol_scale: float | None = None,
    horizon_idx: int = 0,
    risk_aversion: float = 2.0,
    factor_penalty: float = 15.0,
    l1_cost: float = 0.001,
    impact_cost: float = 0.01,
    gross_limit: float = 1.0,
    pos_cap: float = 0.05,
    device: str = "cpu",
) -> dict[str, Any]:
    """Apilado epistémico entre familias (`#195 #5C`) alimentando `sigma_eff` a cartera `P+`.

    Cuando `disagreement_kappa == 0.0`, `sigma_eff == realized_vol` bit-a-bit (control emparejado
    exacto); cuando `disagreement_kappa > 0.0`, el desacuerdo entre familias infla la volatilidad
    efectiva `sqrt(realized_vol^2 + kappa * Var_f(u_f) * scale)` reduciendo exposición en `P+`.
    Soporta predicciones 1D (`[F, N]`) y multi-horizonte 2D (`[F, N, H]` vía `horizon_idx`).
    """
    if disagreement_kappa < 0.0:
        raise ValueError(f"disagreement_kappa debe ser >= 0.0; recibido {disagreement_kappa}")
    risk_out = compute_cross_family_disagreement_risk(
        family_preds,
        realized_vol,
        kappa=float(disagreement_kappa),
        cka_weights=cka_weights,
        cka_matrix=cka_matrix,
        vol_scale=vol_scale,
    )
    dev = (
        torch.device(device)
        if (str(device).startswith("cuda") and torch.cuda.is_available())
        else torch.device("cpu")
    )
    u_bar_np = np.asarray(risk_out["consensus_u"], dtype=np.float64)
    sigma_eff_np = np.asarray(risk_out["sigma_eff"], dtype=np.float64)
    dis_var_np = np.asarray(risk_out["disagreement_var"], dtype=np.float64)
    dis_std_np = np.asarray(risk_out["disagreement_std"], dtype=np.float64)
    real_vol_np = np.asarray(realized_vol, dtype=np.float64)

    if u_bar_np.ndim == 2:
        n_h = int(u_bar_np.shape[1])
        h_i = int(horizon_idx)
        if not (0 <= h_i < n_h):
            raise ValueError(f"horizon_idx ({h_i}) fuera de rango [0, {n_h})")
        u_bar_1d = u_bar_np[:, h_i]
        sigma_eff_1d = sigma_eff_np[:, h_i] if sigma_eff_np.ndim == 2 else sigma_eff_np
        real_vol_1d = real_vol_np[:, h_i] if real_vol_np.ndim == 2 else real_vol_np
    elif u_bar_np.ndim == 1:
        u_bar_1d = u_bar_np
        sigma_eff_1d = sigma_eff_np.reshape(-1)
        real_vol_1d = real_vol_np.reshape(-1)
    else:
        raise ValueError(f"consensus_u debe ser 1D o 2D; recibido shape={u_bar_np.shape}")

    pos_vols = real_vol_1d[np.isfinite(real_vol_1d) & (real_vol_1d > 0.0)]
    alpha_scale = float(np.median(pos_vols)) if pos_vols.size > 0 else 0.02
    if not math.isfinite(alpha_scale) or alpha_scale <= 0.0:
        alpha_scale = 0.02
    signal_t = torch.as_tensor(u_bar_1d * alpha_scale, device=dev, dtype=torch.float32)
    vol_t = torch.as_tensor(sigma_eff_1d, device=dev, dtype=torch.float32)
    exp_t = (
        torch.as_tensor(np.asarray(exposures, dtype=np.float64), device=dev, dtype=torch.float32)
        if exposures is not None
        else None
    )
    w_prev_t = (
        torch.as_tensor(np.asarray(w_prev, dtype=np.float64), device=dev, dtype=torch.float32)
        if w_prev is not None
        else None
    )
    opt_p = ConvexPortfolioOptimizerPPlus(
        risk_aversion=risk_aversion,
        factor_penalty=factor_penalty,
        l1_cost=l1_cost,
        impact_cost=impact_cost,
        gross_limit=gross_limit,
    )
    w_t = opt_p.solve_torch(
        signal_t, vol=vol_t, exposures=exp_t, w_prev=w_prev_t, pos_cap=pos_cap
    )
    w_np = w_t.detach().cpu().numpy().astype(np.float64)
    ratio = sigma_eff_1d / np.maximum(real_vol_1d, 1e-6)
    return dict(
        consensus_u=u_bar_np,
        u_bar=u_bar_np,
        disagreement_var=dis_var_np,
        disagreement_std=dis_std_np,
        sigma_eff=sigma_eff_np,
        family_weights=np.asarray(risk_out["weights"], dtype=np.float64),
        portfolio_weights=w_np,
        mean_sigma_inflation_ratio=float(np.mean(ratio)),
        disagreement_kappa=float(disagreement_kappa),
        horizon_idx=int(horizon_idx),
    )


@dataclass(frozen=True)
class Stage2SplitData:
    """Emisiones alineadas de una o varias familias pre-entrenadas en un tramo causal."""

    family_preds: dict[str, np.ndarray]
    target: np.ndarray
    bounds: list[tuple[int, int]]
    prediction_at: np.ndarray
    matures_at: np.ndarray
    quantiles: np.ndarray | None = None
    family_quantiles: dict[str, np.ndarray] | None = None
    quantile_levels: tuple[float, ...] = QUANTILE_LEVELS_9
    volatility: np.ndarray | None = None
    liquidity: np.ndarray | None = None
    exposures: np.ndarray | None = None
    markets: np.ndarray | None = None
    hopfield_energy: np.ndarray | None = None
    router_entropy: np.ndarray | None = None
    loop_energy: np.ndarray | None = None
    session_years: np.ndarray | None = None
    asset_ids: np.ndarray | None = None


def _augment_quantiles_to_11(
    q: np.ndarray, levels: tuple[float, ...]
) -> tuple[np.ndarray, tuple[float, ...]]:
    """Alinea cuantiles al mallado de 11 niveles que incluye pares 80%, 90% y 95% sin cruces."""
    q_arr = np.asarray(q, dtype=np.float64)
    if len(levels) == len(EXTENDED_QUANTILE_LEVELS_11) and all(
        abs(a - b) < 1e-6 for a, b in zip(levels, EXTENDED_QUANTILE_LEVELS_11, strict=True)
    ):
        return np.sort(q_arr, axis=1), EXTENDED_QUANTILE_LEVELS_11

    lv_arr = np.asarray(levels, dtype=np.float64)
    q_sorted = np.sort(q_arr, axis=1)
    # Interpolación lineal monótona vectorizada sobre los niveles solicitados
    out = np.empty((q_sorted.shape[0], len(EXTENDED_QUANTILE_LEVELS_11)), dtype=np.float64)
    for col_i, target_lv in enumerate(EXTENDED_QUANTILE_LEVELS_11):
        exact = np.flatnonzero(np.abs(lv_arr - target_lv) < 1e-6)
        if exact.size > 0:
            out[:, col_i] = q_sorted[:, int(exact[0])]
        else:
            pos = int(np.searchsorted(lv_arr, target_lv, side="right"))
            lo = max(0, min(pos - 1, len(lv_arr) - 2))
            hi = lo + 1
            frac = (target_lv - lv_arr[lo]) / max(lv_arr[hi] - lv_arr[lo], 1e-12)
            out[:, col_i] = (1.0 - frac) * q_sorted[:, lo] + frac * q_sorted[:, hi]
    return np.sort(out, axis=1), EXTENDED_QUANTILE_LEVELS_11


def _recenter_quantiles(
    q_ext: np.ndarray, new_median: np.ndarray, levels: tuple[float, ...]
) -> np.ndarray:
    """Recentra los cuantiles en `new_median` preservando su dispersión relativa sin cruces."""
    med_idxs = [i for i, lv in enumerate(levels) if abs(lv - 0.50) < 1e-6]
    if not med_idxs:
        return q_ext.copy()
    med_i = med_idxs[0]
    shift = np.asarray(new_median, dtype=np.float64) - q_ext[:, med_i]
    return np.sort(q_ext + shift[:, None], axis=1)


def _compute_session_net1_series(
    *,
    y: np.ndarray,
    pred: np.ndarray,
    bounds: list[tuple[int, int]],
    portfolio_weights: np.ndarray | None,
    volatility: np.ndarray | None,
    liquidity: np.ndarray | None,
    spread_bps: float,
    impact_eta: float,
    aum_musd: float,
    asset_ids: np.ndarray | None = None,
    market: str = "US",
    cn_stamp_tax_bps: float = 10.0,
    cn_limit_lock_threshold: float = 0.098,
    cn_limit_lock_slippage_bps: float = 50.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Calcula retornos netos 1x y turnover por sesión con alineación exacta por `asset_ids`."""
    n = len(y)
    if portfolio_weights is None:
        w_all = np.zeros(n, dtype=np.float64)
        for lo, hi in bounds:
            z = pred[lo:hi] - np.mean(pred[lo:hi])
            s = float(np.sum(np.abs(z)))
            if s > 1e-12:
                w_all[lo:hi] = z / s
    else:
        w_all = np.asarray(portfolio_weights, dtype=np.float64)

    if volatility is None:
        vol_all = np.full(n, 0.02, dtype=np.float64)
    else:
        v_raw = np.nan_to_num(np.asarray(volatility, dtype=np.float64), nan=0.02)
        if float(np.min(v_raw)) < 0.0 or float(np.median(v_raw)) > 0.25:
            vol_all = 0.02 * np.exp(0.5 * np.clip(v_raw, -3.0, 3.0))
        else:
            vol_all = np.clip(v_raw, 0.003, 0.25)

    if liquidity is not None:
        l_raw = np.nan_to_num(np.asarray(liquidity, dtype=np.float64), nan=0.0)
        l_med = float(np.median(l_raw))
        if l_med > 50.0:
            adv_usd = np.clip(l_raw, 1e4, 1e12)
        elif l_med >= 6.0:
            adv_usd = np.exp(np.clip(l_raw, 8.0, 25.0))
        else:
            adv_usd = 1e7 * np.exp(np.clip(l_raw, -4.0, 4.0))
    else:
        adv_usd = np.full(n, 1e7, dtype=np.float64)

    spread_rate = float(spread_bps) * 1e-4
    stamp_rate = (float(cn_stamp_tax_bps) * 1e-4) if market == "CN" else 0.0
    lock_slip_rate = (float(cn_limit_lock_slippage_bps) * 1e-4) if market == "CN" else 0.0

    net1 = np.empty(len(bounds), dtype=np.float64)
    turnovers = np.empty(len(bounds), dtype=np.float64)
    w_prev_vec: np.ndarray | None = None
    prev_map: dict[Any, tuple[float, float, float]] = {}  # aid -> (w, vol, adv)
    aids = None if asset_ids is None else np.asarray(asset_ids)

    for s_idx, (lo, hi) in enumerate(bounds):
        wt = w_all[lo:hi]
        yt = y[lo:hi]
        vt = vol_all[lo:hi]
        advt = adv_usd[lo:hi]
        gross = float(np.dot(wt, yt))
        locked = (
            (np.abs(yt) >= cn_limit_lock_threshold)
            if market == "CN"
            else np.zeros(len(yt), dtype=bool)
        )

        exit_to = 0.0
        exit_cost = 0.0
        if aids is not None:
            cur_ids = aids[lo:hi].tolist()
            w_prev_aligned = np.array(
                [prev_map[aid][0] if aid in prev_map else 0.0 for aid in cur_ids],
                dtype=np.float64,
            )
            dw = np.abs(wt - w_prev_aligned)
            sell_act = np.maximum(w_prev_aligned - wt, 0.0)
            cur_set = set(cur_ids)
            for old_aid, (old_w, old_v, old_adv) in prev_map.items():
                if old_aid not in cur_set and abs(old_w) > 0.0:
                    abs_ow = abs(old_w)
                    exit_to += abs_ow
                    part_ex = min(max(abs_ow * (aum_musd * 1e6) / max(old_adv, 1e4), 0.0), 1.0)
                    exit_cost += abs_ow * (
                        spread_rate + impact_eta * old_v * math.sqrt(part_ex)
                    ) + max(old_w, 0.0) * stamp_rate
            prev_map = {
                cur_ids[i]: (float(wt[i]), float(vt[i]), float(advt[i]))
                for i in range(len(cur_ids))
            }
        else:
            same_prev = w_prev_vec is not None and len(w_prev_vec) == len(wt)
            w_prev_aligned = w_prev_vec if same_prev else np.zeros_like(wt)
            dw = np.abs(wt - w_prev_aligned)
            sell_act = np.maximum(w_prev_aligned - wt, 0.0)
            w_prev_vec = wt

        trade_usd = dw * (aum_musd * 1e6)
        part = np.clip(trade_usd / np.maximum(advt, 1e4), 0.0, 1.0)
        c_lin = (
            float(np.sum(dw) * spread_rate)
            + float(np.sum(sell_act) * stamp_rate)
            + float(np.sum(dw[locked]) * lock_slip_rate)
        )
        c_imp = float(np.sum(dw * impact_eta * vt * np.sqrt(part)))
        turnovers[s_idx] = float(np.sum(dw)) + exit_to
        net1[s_idx] = gross - (c_lin + c_imp + exit_cost)
    return net1, turnovers


class Stage2PostTrainingPipeline:
    """Pipeline causal Stage-2: FM-7 + Conformal PID Mondrian + Venn-Abers + OAS-ICIR + P+/A5-A8."""

    def __init__(self, config: Stage2Config | None = None) -> None:
        self.config = config or Stage2Config()

    def _validate_split(self, split: Stage2SplitData, name: str) -> list[str]:
        if not split.family_preds:
            raise ValueError(f"{name}.family_preds no puede estar vacío")
        y = np.asarray(split.target, dtype=np.float64)
        pred_at = np.asarray(split.prediction_at, dtype=np.int64)
        mat_at = np.asarray(split.matures_at, dtype=np.int64)
        n = len(y)
        if n == 0 or len(pred_at) != n or len(mat_at) != n:
            raise ValueError(f"Longitudes incompatibles en {name}")
        if not np.isfinite(y).all():
            raise ValueError(f"Valores no finitos en {name}.target")

        # Sello estricto 2024+
        if int(pred_at.max()) >= FINAL_TEST_START_US:
            raise SealedTestViolationError(
                f"Violación del sello 2024+ en {name}.prediction_at (>= FINAL_TEST_START_US)"
            )
        if int(mat_at.max()) >= FINAL_TEST_START_US:
            raise SealedTestViolationError(
                f"Violación del sello 2024+ en {name}.matures_at (>= FINAL_TEST_START_US)"
            )
        if split.session_years is not None:
            validate_no_sealed_leakage(
                [int(yr) for yr in np.asarray(split.session_years).tolist()],
                sealed_test_start_year=self.config.sealed_test_start_year,
            )
        if not bool(np.all(mat_at > pred_at)):
            raise ValueError(f"Etiquetas en {name} deben madurar estrictamente tras prediction_at")

        if not split.bounds:
            raise ValueError(f"{name}.bounds no puede estar vacío")
        prev_hi = 0
        prev_t: int | None = None
        for lo, hi in split.bounds:
            if lo != prev_hi or hi <= lo or hi > n:
                raise ValueError(f"{name}.bounds debe particionar [0, {n}) contiguamente")
            t_cur = int(pred_at[lo])
            if prev_t is not None and t_cur < prev_t:
                raise ValueError(f"Orden cronológico violado en {name}.bounds")
            prev_t = t_cur
            prev_hi = hi
        if prev_hi != n:
            raise ValueError(f"{name}.bounds no cubre todas las filas ({prev_hi} != {n})")

        # Ordenar nombres de familias: primero las 4 familias canónicas si están presentes
        ordered_fams = [f for f in FOUR_FAMILIES if f in split.family_preds]
        ordered_fams.extend(sorted(f for f in split.family_preds if f not in ordered_fams))
        for fam in ordered_fams:
            arr = np.asarray(split.family_preds[fam], dtype=np.float64)
            if len(arr) != n or not np.isfinite(arr).all():
                raise ValueError(f"Predicciones inválidas o no finitas para familia {fam!r}")
        if split.quantiles is not None:
            q_arr = np.asarray(split.quantiles, dtype=np.float64)
            if q_arr.ndim != 2 or len(q_arr) != n or not np.isfinite(q_arr).all():
                raise ValueError(f"Cuantiles inválidos o no finitos en {name}.quantiles")
        if split.asset_ids is not None and len(split.asset_ids) != n:
            raise ValueError(f"Longitud incompatible de {name}.asset_ids")
        return ordered_fams

    def _resolve_device(self) -> torch.device:
        if self.config.device.startswith("cuda") and torch.cuda.is_available():
            return torch.device(self.config.device)
        return torch.device("cpu")

    @staticmethod
    def _resolve_vol(split: Stage2SplitData) -> np.ndarray:
        n = len(split.target)
        if split.volatility is not None:
            v = np.nan_to_num(np.asarray(split.volatility, dtype=np.float64), nan=0.02)
            return np.clip(np.where(v > 0.0, v, 0.02), 1e-4, 1.0)
        if split.quantiles is not None and split.quantiles.shape[1] >= 2:
            q = np.asarray(split.quantiles, dtype=np.float64)
            width = np.maximum(q[:, -1] - q[:, 0], 1e-4)
            return np.clip(width / 4.0, 1e-4, 1.0)
        return np.full(n, 0.02, dtype=np.float64)

    def _resolve_markets(self, split: Stage2SplitData) -> np.ndarray:
        if split.markets is not None:
            return np.asarray(split.markets).astype(str)
        return np.full(len(split.target), self.config.market, dtype=object).astype(str)

    @staticmethod
    def _fit_vol_strata(
        vol_cal: np.ndarray,
        vol_eval: np.ndarray,
        markets_cal: np.ndarray,
        markets_eval: np.ndarray,
        n_strata: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float]]:
        """Ajusta bordes de deciles de volatilidad SOLO en calibración y asigna estratos."""
        if n_strata <= 1:
            s_cal = np.zeros(len(vol_cal), dtype=np.int64)
            s_eval = np.zeros(len(vol_eval), dtype=np.int64)
            return (
                make_stratified_groups(markets_cal, s_cal),
                make_stratified_groups(markets_eval, s_eval),
                s_eval,
                [],
            )

        probs = np.linspace(0.0, 1.0, n_strata + 1)[1:-1]
        raw_edges = np.unique(np.quantile(vol_cal, probs))
        s_cal = np.searchsorted(raw_edges, vol_cal, side="right").astype(np.int64)
        s_eval = np.searchsorted(raw_edges, vol_eval, side="right").astype(np.int64)

        # Clamping causal vectorizado por mercado vía `np.searchsorted` (0 bucles sobre filas!)
        m_cal_str = np.asarray(markets_cal, dtype=str)
        m_eval_str = np.asarray(markets_eval, dtype=str)
        known_mkts = sorted(set(m_cal_str.tolist()))
        valid_by_mkt: dict[str, np.ndarray] = {
            mkt: np.sort(np.unique(s_cal[m_cal_str == mkt])) for mkt in known_mkts
        }
        fallback_mkt = known_mkts[0]

        use_mkt_arr = np.where(np.isin(m_eval_str, known_mkts), m_eval_str, fallback_mkt)
        s_eval_clamped = np.empty(len(vol_eval), dtype=np.int64)
        max_s = max((int(a[-1]) for a in valid_by_mkt.values() if a.size > 0), default=0)
        max_str_len = max((len(m) for m in known_mkts), default=1) + 2 + len(str(max_s))
        g_eval = np.empty(len(vol_eval), dtype=f"<U{max_str_len}")

        for mkt, allowed in valid_by_mkt.items():
            mask = use_mkt_arr == mkt
            if not bool(np.any(mask)):
                continue
            vals = s_eval[mask]
            pos = np.searchsorted(allowed, vals, side="left")
            right_idx = np.minimum(pos, len(allowed) - 1)
            left_idx = np.maximum(pos - 1, 0)
            d_left = np.abs(vals - allowed[left_idx])
            d_right = np.abs(allowed[right_idx] - vals)
            # En empate exacto (`d_left == d_right`) elige índice menor, paridad bit a bit `argmin`
            nearest = np.where(d_left <= d_right, allowed[left_idx], allowed[right_idx])
            s_eval_clamped[mask] = nearest
            lut = np.array([f"{mkt}|s{k}" for k in range(max_s + 1)], dtype=g_eval.dtype)
            g_eval[mask] = lut[nearest]

        g_cal = make_stratified_groups(m_cal_str, s_cal)
        return g_cal, g_eval, s_eval_clamped, [float(x) for x in raw_edges.tolist()]

    def _compute_causal_family_blending(
        self,
        cal_split: Stage2SplitData,
        eval_split: Stage2SplitData,
        families: list[str],
    ) -> dict[str, Any]:
        """Calcula pesos causales OAS-ICIR + risk-parity sobre `(cal_split -> eval_split)`."""
        n_fam = len(families)
        cal_bounds = cal_split.bounds
        eval_bounds = eval_split.bounds
        n_cal_s = len(cal_bounds)
        n_eval_s = len(eval_bounds)
        n_tot_s = n_cal_s + n_eval_s

        # 1) Estandarizar rangos transversales por sesión en cal y eval (sin fuga entre sesiones)
        def _std_matrix(split: Stage2SplitData) -> np.ndarray:
            mat = np.empty((len(split.target), n_fam), dtype=np.float64)
            for lo, hi in split.bounds:
                for m_i, fam in enumerate(families):
                    mat[lo:hi, m_i] = oa.standardized_ranks(split.family_preds[fam][lo:hi]).astype(
                        np.float64
                    )
            return mat

        z_cal = _std_matrix(cal_split)
        z_eval = _std_matrix(eval_split)

        # Seleccionar familia Stage-1 de referencia ESTRICTAMENTE en cal_split
        cal_ics = np.zeros((n_cal_s, n_fam), dtype=np.float64)
        cal_covs = np.zeros((n_cal_s, n_fam, n_fam), dtype=np.float64)
        cal_t = np.empty(n_cal_s, dtype=np.int64)
        cal_mat = np.empty(n_cal_s, dtype=np.int64)
        for s_i, (lo, hi) in enumerate(cal_bounds):
            cal_t[s_i] = int(cal_split.prediction_at[lo])
            cal_mat[s_i] = int(cal_split.matures_at[lo:hi].max())
            y_seg = cal_split.target[lo:hi]
            for m_i in range(n_fam):
                ric = oa.rank_ic_vector(z_cal[lo:hi, m_i], y_seg)
                cal_ics[s_i, m_i] = ric if math.isfinite(ric) else 0.0
            if n_fam == 1:
                cal_covs[s_i, 0, 0] = max(float(np.var(z_cal[lo:hi, 0])), 1e-6)
            elif hi - lo < 2:
                cal_covs[s_i] = np.eye(n_fam, dtype=np.float64)
            else:
                cal_covs[s_i] = np.cov(z_cal[lo:hi], rowvar=False)

        eval_ics = np.zeros((n_eval_s, n_fam), dtype=np.float64)
        eval_covs = np.zeros((n_eval_s, n_fam, n_fam), dtype=np.float64)
        eval_t = np.empty(n_eval_s, dtype=np.int64)
        eval_mat = np.empty(n_eval_s, dtype=np.int64)
        for s_i, (lo, hi) in enumerate(eval_bounds):
            eval_t[s_i] = int(eval_split.prediction_at[lo])
            eval_mat[s_i] = int(eval_split.matures_at[lo:hi].max())
            y_seg = eval_split.target[lo:hi]
            for m_i in range(n_fam):
                ric = oa.rank_ic_vector(z_eval[lo:hi, m_i], y_seg)
                eval_ics[s_i, m_i] = ric if math.isfinite(ric) else 0.0
            if n_fam == 1:
                eval_covs[s_i, 0, 0] = max(float(np.var(z_eval[lo:hi, 0])), 1e-6)
            elif hi - lo < 2:
                eval_covs[s_i] = np.eye(n_fam, dtype=np.float64)
            else:
                eval_covs[s_i] = np.cov(z_eval[lo:hi], rowvar=False)

        cal_mean_ic = cal_ics.mean(axis=0)
        best_fam_idx = int(np.argmax(cal_mean_ic))
        ref_family = families[best_fam_idx]

        all_ics = np.vstack([cal_ics, eval_ics])
        all_covs = np.concatenate([cal_covs, eval_covs], axis=0)
        all_t = np.concatenate([cal_t, eval_t])
        all_mat = np.concatenate([cal_mat, eval_mat])

        if n_fam == 1 or self.config.blend_mode == "equal":
            weights_all = np.full((n_tot_s, n_fam), 1.0 / n_fam, dtype=np.float64)
        else:
            w_oas = shrinkage_icir_weights(
                all_ics,
                all_covs,
                lam=self.config.oas_lam,
                half_life=self.config.oas_half_life,
                session_times=all_t,
                session_matures=all_mat,
                se_shrinkage=self.config.oas_se_shrinkage,
                cov_shrinkage="oas",
                use_current_cov=True,
                n_obs_cov=float(max(len(cal_split.target) // max(n_cal_s, 1), 20)),
            )
            # Pesos causales de paridad de riesgo / inversa de varianza OAS-contraída
            w_rp = np.empty((n_tot_s, n_fam), dtype=np.float64)
            n_obs_s = float(max(len(cal_split.target) // max(n_cal_s, 1), 20))
            for t_i in range(n_tot_s):
                shrunk_c, _ = oas_shrink_covariance(all_covs[t_i], n_obs=n_obs_s)
                inv_vol = 1.0 / np.sqrt(np.maximum(np.diag(shrunk_c), 1e-6))
                w_rp[t_i] = inv_vol / max(float(inv_vol.sum()), 1e-12)

            if self.config.blend_mode == "oas_icir":
                weights_all = w_oas
            elif self.config.blend_mode == "risk_parity":
                weights_all = w_rp
            else:  # oas_icir_rp
                mix = self.config.risk_parity_mix
                w_comb = (1.0 - mix) * w_oas + mix * w_rp
                weights_all = w_comb / np.maximum(w_comb.sum(axis=1, keepdims=True), 1e-12)

        w_cal = weights_all[:n_cal_s]
        w_eval = weights_all[n_cal_s:]

        raw_blend_cal = np.empty(len(cal_split.target), dtype=np.float64)
        for s_i, (lo, hi) in enumerate(cal_bounds):
            raw_blend_cal[lo:hi] = z_cal[lo:hi] @ w_cal[s_i]

        raw_blend_eval = np.empty(len(eval_split.target), dtype=np.float64)
        for s_i, (lo, hi) in enumerate(eval_bounds):
            raw_blend_eval[lo:hi] = z_eval[lo:hi] @ w_eval[s_i]

        return dict(
            families=families,
            reference_family=ref_family,
            cal_family_mean_ic={
                fam: float(cal_mean_ic[i]) for i, fam in enumerate(families)
            },
            z_cal=z_cal,
            z_eval=z_eval,
            cal_ics=cal_ics,
            cal_mat=cal_mat,
            eval_t=eval_t,
            eval_mat=eval_mat,
            w_cal=w_cal,
            w_eval=w_eval,
            raw_blend_cal=raw_blend_cal,
            raw_blend_eval=raw_blend_eval,
        )

    @staticmethod
    def _resolve_quantiles(
        split: Stage2SplitData,
        *,
        blend_weights: np.ndarray,
        families: list[str],
        fallback_pred: np.ndarray,
        vol: np.ndarray,
    ) -> tuple[np.ndarray, tuple[float, ...]]:
        if split.quantiles is not None:
            return np.asarray(split.quantiles, dtype=np.float64), split.quantile_levels
        if split.family_quantiles is not None:
            first_q = np.asarray(split.family_quantiles[families[0]], dtype=np.float64)
            q_blend = np.zeros_like(first_q, dtype=np.float64)
            for s_i, (lo, hi) in enumerate(split.bounds):
                for m_i, fam in enumerate(families):
                    q_f = np.asarray(split.family_quantiles[fam][lo:hi], dtype=np.float64)
                    q_blend[lo:hi] += blend_weights[s_i, m_i] * q_f
            return q_blend, split.quantile_levels
        # Cuantiles gaussianos causales base si el tramo no proporcionó cuantiles explícitos
        q_base = fallback_pred[:, None] + vol[:, None] * _STANDARD_Z9[None, :]
        return q_base, QUANTILE_LEVELS_9

    def run(
        self,
        cal_split: Stage2SplitData,
        eval_split: Stage2SplitData,
    ) -> dict[str, Any]:
        """Ejecuta el pipeline causal Stage-2 completo sobre `(cal_split -> eval_split)`."""
        cal_fams = self._validate_split(cal_split, "cal_split")
        eval_fams = self._validate_split(eval_split, "eval_split")
        if set(cal_fams) != set(eval_fams):
            raise ValueError(
                f"Familias incompatibles entre cal_split {cal_fams} y eval_split {eval_fams}"
            )
        if int(cal_split.prediction_at.max()) >= int(eval_split.prediction_at.min()):
            raise ValueError(
                "Orden cronológico violado entre cal_split y eval_split: "
                f"{int(cal_split.prediction_at.max())} >= {int(eval_split.prediction_at.min())}"
            )
        if int(cal_split.matures_at.max()) > int(eval_split.prediction_at.min()):
            raise ValueError(
                "Frontera causal de madurez violada entre cal_split y eval_split: "
                f"{int(cal_split.matures_at.max())} > {int(eval_split.prediction_at.min())}"
            )

        cfg = self.config
        n_fam = len(cal_fams)
        vol_cal = self._resolve_vol(cal_split)
        vol_eval = self._resolve_vol(eval_split)
        mkt_cal = self._resolve_markets(cal_split)
        mkt_eval = self._resolve_markets(eval_split)

        # (c.1) Combinación ortogonal causal OAS-ICIR + risk-parity entre familias
        blend = self._compute_causal_family_blending(cal_split, eval_split, cal_fams)

        # (a) Desacoplamiento Orden+Nivel (FM-7 / F7-b) ajustado EXCLUSIVAMENTE en cal_split
        decoupler = OrderLevelDecoupler(
            loss=cfg.point_loss,
            huber_delta=cfg.huber_delta,
            lambda_min=cfg.lambda_min,
            lambda_max=cfg.lambda_max,
        ).fit(blend["raw_blend_cal"], cal_split.target)
        s2_pred_cal = decoupler.transform(blend["raw_blend_cal"])
        s2_pred_eval = decoupler.transform(blend["raw_blend_eval"])

        # (b.1) Calibración conforme Mondrian por deciles de volatilidad (80%/90%/95%)
        q_cal_raw, q_cal_lv = self._resolve_quantiles(
            cal_split,
            blend_weights=blend["w_cal"],
            families=cal_fams,
            fallback_pred=s2_pred_cal,
            vol=vol_cal,
        )
        q_eval_raw, q_eval_lv = self._resolve_quantiles(
            eval_split,
            blend_weights=blend["w_eval"],
            families=cal_fams,
            fallback_pred=s2_pred_eval,
            vol=vol_eval,
        )
        q_cal_11, ext_levels = _augment_quantiles_to_11(q_cal_raw, q_cal_lv)
        q_eval_11, _ = _augment_quantiles_to_11(q_eval_raw, q_eval_lv)
        q_cal_11 = _recenter_quantiles(q_cal_11, s2_pred_cal, ext_levels)
        q_eval_11 = _recenter_quantiles(q_eval_11, s2_pred_eval, ext_levels)

        cal_groups, eval_groups, eval_strata, vol_edges = self._fit_vol_strata(
            vol_cal, vol_eval, mkt_cal, mkt_eval, cfg.n_vol_strata
        )
        pid = NormalizedConformalPID.start(
            target=cal_split.target,
            quantiles=q_cal_11,
            groups=cal_groups,
            levels=ext_levels,
            nominals=cfg.conformal_nominals,
            scale_mode=cfg.conformal_scale_mode,
            cal_scales=vol_cal,
            kp=cfg.pid_kp,
            ki=cfg.pid_ki,
            kd=cfg.pid_kd,
        )
        q_eval_11_cal, _, pid_updates = replay_normalized_pid(
            pid,
            q_eval_11,
            eval_groups,
            eval_split.prediction_at,
            eval_split.target,
            eval_split.matures_at,
            scales=vol_eval,
        )
        q_eval_9_cal = np.sort(q_eval_11_cal[:, list(_QUANTILE_9_IN_11_IDX)], axis=1)

        # Métricas conformes explícitas 80% / 90% / 95% + dispersión Mondrian por decil
        y_ev = np.asarray(eval_split.target, dtype=np.float64)
        conf_metrics: dict[str, Any] = dict(
            scale_mode=cfg.conformal_scale_mode,
            n_vol_strata=len(set(eval_strata.tolist())),
            vol_edges=vol_edges,
            pid_updates_count=len(pid_updates),
            pid_state_sha256=pid.state_sha256(),
        )
        for nom in cfg.conformal_nominals:
            lo_lv = round((1.0 - nom) / 2.0, 6)
            hi_lv = round(1.0 - (1.0 - nom) / 2.0, 6)
            lo_i = [i for i, lv in enumerate(ext_levels) if abs(lv - lo_lv) < 1e-6][0]
            hi_i = [i for i, lv in enumerate(ext_levels) if abs(lv - hi_lv) < 1e-6][0]
            ql = q_eval_11_cal[:, lo_i]
            qh = q_eval_11_cal[:, hi_i]
            alpha = 1.0 - nom
            cov = float(np.mean((y_ev >= ql) & (y_ev <= qh)))
            width = float(np.mean(qh - ql))
            winkler = float(
                np.mean(
                    (qh - ql)
                    + (2.0 / alpha) * np.maximum(ql - y_ev, 0.0)
                    + (2.0 / alpha) * np.maximum(y_ev - qh, 0.0)
                )
            )
            tag = int(round(nom * 100))
            conf_metrics[f"cov_{tag}"] = cov
            conf_metrics[f"width_{tag}"] = width
            conf_metrics[f"winkler_{tag}"] = winkler
            if tag == 80:
                dec_covs: dict[str, float] = {}
                for st_id in sorted(set(eval_strata.tolist())):
                    m_st = eval_strata == st_id
                    dec_covs[f"s{st_id}"] = float(
                        np.mean((y_ev[m_st] >= ql[m_st]) & (y_ev[m_st] <= qh[m_st]))
                    )
                conf_metrics["decile_cov_80"] = dec_covs
                vals_d = list(dec_covs.values())
                conf_metrics["decile_cov_80_min"] = min(vals_d) if vals_d else cov
                conf_metrics["decile_cov_80_max"] = max(vals_d) if vals_d else cov
                conf_metrics["decile_cov_80_std"] = (
                    float(np.std(vals_d)) if len(vals_d) > 1 else 0.0
                )

        # (b.2) Calibración probabilística Venn-Abers (`[p0, p1]`, anchura epistémica `p1 - p0`)
        y_bin_cal = (cal_split.target > 0.0).astype(np.float64)
        if cfg.venn_abers_half_life is not None:
            cal_sess_order = np.empty(len(cal_split.target), dtype=np.float64)
            for s_i, (lo, hi) in enumerate(cal_split.bounds):
                cal_sess_order[lo:hi] = float(s_i)
            va_cal = VennAbersCalibrator.fit_with_decay(
                s2_pred_cal,
                y_bin_cal,
                session_order=cal_sess_order,
                half_life=cfg.venn_abers_half_life,
            )
        else:
            va_cal = VennAbersCalibrator.fit(s2_pred_cal, y_bin_cal)
        va_eval = va_cal.predict(s2_pred_eval)

        # (c.2) Gobernador Metacognitivo A5-A8 (`decide_torch`) + Cartera Convexa GPU P+
        dev = self._resolve_device()
        governor = MetacognitiveGovernor(
            n_specialists=n_fam,
            epistemic_kappa=cfg.epistemic_kappa,
            ood_threshold=cfg.ood_threshold,
            entropy_weight=cfg.entropy_weight,
            loop_energy_weight=cfg.loop_energy_weight,
        )
        portfolio_opt = ConvexPortfolioOptimizerPPlus(
            risk_aversion=cfg.risk_aversion,
            factor_penalty=cfg.factor_penalty,
            l1_cost=cfg.l1_cost,
            impact_cost=cfg.impact_cost,
            gross_limit=cfg.gross_limit,
        )

        # Calentar gobernador de forma estrictamente causal con sesiones maduras de cal_split
        for s_i in range(len(cal_split.bounds)):
            governor.update_matured(blend["cal_ics"][s_i])

        n_eval = len(y_ev)
        stage2_weights = np.zeros(n_eval, dtype=np.float64)
        active_rates: list[float] = []
        disagreement_ratios: list[float] = []
        disagreement_vars: list[float] = []
        w_prev_t: torch.Tensor | None = None
        prev_w_by_asset: dict[Any, float] = {}
        eval_aids = None if eval_split.asset_ids is None else np.asarray(eval_split.asset_ids)
        pending_gov: list[tuple[int, np.ndarray]] = []
        z_eval = blend["z_eval"]
        w_eval_blend = blend["w_eval"]

        for s_i, (lo, hi) in enumerate(eval_split.bounds):
            now_us = int(blend["eval_t"][s_i])
            n_s = hi - lo
            # Madurar exclusivamente cohortes con matures_at <= now_us
            if pending_gov:
                ready = [item for item in pending_gov if item[0] <= now_us]
                if ready:
                    pending_gov = [item for item in pending_gov if item[0] > now_us]
                    ready.sort(key=lambda item: item[0])
                    for _, ics_m in ready:
                        governor.update_matured(ics_m)

            # Alinear pesos de creencia con OAS-ICIR + amortiguación CUSUM del gobernador
            w_eff = w_eval_blend[s_i] * np.where(governor.decay_alerts, 0.15, 1.0)
            w_eff_sum = float(w_eff.sum())
            governor.specialist_weights = (
                (w_eff / w_eff_sum)
                if w_eff_sum > 1e-12
                else np.full(n_fam, 1.0 / n_fam, dtype=np.float64)
            )

            spec_seg = decoupler.transform(z_eval[lo:hi])
            spec_t = torch.as_tensor(spec_seg, device=dev, dtype=torch.float32)
            va_w_t = torch.as_tensor(va_eval["uncertainty"][lo:hi], device=dev, dtype=torch.float32)
            hop_t = (
                torch.as_tensor(eval_split.hopfield_energy[lo:hi], device=dev, dtype=torch.float32)
                if eval_split.hopfield_energy is not None
                else None
            )
            ent_t = (
                torch.as_tensor(eval_split.router_entropy[lo:hi], device=dev, dtype=torch.float32)
                if eval_split.router_entropy is not None
                else None
            )
            l_eng_t = (
                torch.as_tensor(eval_split.loop_energy[lo:hi], device=dev, dtype=torch.float32)
                if eval_split.loop_energy is not None
                else None
            )
            costs_t = torch.full((n_s,), cfg.l1_cost * 0.25, device=dev, dtype=torch.float32)

            dec_t = governor.decide_torch(
                specialist_preds=spec_t,
                va_widths=va_w_t,
                hopfield_energy=hop_t,
                router_entropy=ent_t,
                loop_energy=l_eng_t,
                estimated_costs=costs_t,
            )
            vol_seg = vol_eval[lo:hi]
            if cfg.disagreement_kappa > 0.0 and n_fam >= 2:
                fam_seg_map = {fam: eval_split.family_preds[fam][lo:hi] for fam in cal_fams}
                dis_out = compute_cross_family_disagreement_risk(
                    fam_seg_map,
                    vol_seg,
                    kappa=float(cfg.disagreement_kappa),
                    cka_weights=cfg.cka_weights,
                    cka_matrix=cfg.cka_matrix,
                )
                sigma_eff_seg = np.asarray(dis_out["sigma_eff"], dtype=np.float64)
                disagreement_ratios.append(
                    float(np.mean(sigma_eff_seg / np.maximum(vol_seg, 1e-6)))
                )
                disagreement_vars.append(float(np.mean(dis_out["disagreement_var"])))
                vol_t = torch.as_tensor(sigma_eff_seg, device=dev, dtype=torch.float32)
            else:
                vol_t = torch.as_tensor(vol_seg, device=dev, dtype=torch.float32)
            exp_t = (
                torch.as_tensor(eval_split.exposures[lo:hi], device=dev, dtype=torch.float32)
                if eval_split.exposures is not None
                else None
            )
            # Alinear `w_prev` por `asset_ids` en paneles irregulares (evita reseteo P0-1)
            if eval_aids is not None:
                cur_aids = eval_aids[lo:hi].tolist()
                if prev_w_by_asset:
                    w_prev_in: torch.Tensor | None = torch.tensor(
                        [prev_w_by_asset.get(aid, 0.0) for aid in cur_aids],
                        device=dev,
                        dtype=torch.float32,
                    )
                else:
                    w_prev_in = None
            else:
                w_prev_in = (
                    w_prev_t if (w_prev_t is not None and w_prev_t.shape[0] == n_s) else None
                )

            # `ConvexPortfolioOptimizerPPlus.solve_torch` normaliza columnas de B nativamente
            w_opt_t = portfolio_opt.solve_torch(
                dec_t["gated_signal"],
                vol=vol_t,
                exposures=exp_t,
                w_prev=w_prev_in,
                pos_cap=cfg.pos_cap,
            )
            active_rates.append(float(dec_t["active_mask"].float().mean().cpu().item()))
            w_opt_np = w_opt_t.detach().cpu().numpy()
            stage2_weights[lo:hi] = w_opt_np
            w_prev_t = w_opt_t.detach()
            if eval_aids is not None:
                prev_w_by_asset = {
                    aid: float(w_opt_np[i]) for i, aid in enumerate(eval_aids[lo:hi].tolist())
                }

            # Encolar ICs maduros de la sesión actual para madurar únicamente en t >= matures_at
            y_seg = y_ev[lo:hi]
            sess_ics = np.array(
                [
                    oa.rank_ic_vector(spec_seg[:, j], y_seg)
                    for j in range(n_fam)
                ],
                dtype=np.float64,
            )
            sess_ics = np.nan_to_num(sess_ics, nan=0.0)
            pending_gov.append((int(blend["eval_mat"][s_i]), sess_ics))

        # (c.3) Informes 6D completos: cada familia Stage-1, referencia Stage-1 y Stage-2 SOTA
        q_eval_9_raw = np.sort(
            _augment_quantiles_to_11(q_eval_raw, q_eval_lv)[0][:, list(_QUANTILE_9_IN_11_IDX)],
            axis=1,
        )
        family_stage1_reports: dict[str, dict[str, dict[str, object]]] = {}
        for fam in cal_fams:
            p_fam = np.asarray(eval_split.family_preds[fam], dtype=np.float64)
            q_fam_raw = (
                np.sort(
                    _augment_quantiles_to_11(
                        np.asarray(eval_split.family_quantiles[fam], dtype=np.float64),
                        eval_split.quantile_levels,
                    )[0][:, list(_QUANTILE_9_IN_11_IDX)],
                    axis=1,
                )
                if (eval_split.family_quantiles is not None and fam in eval_split.family_quantiles)
                else q_eval_9_raw
            )
            family_stage1_reports[fam] = evaluate_granular_six_dimensions(
                y=y_ev,
                pred=p_fam,
                bounds=eval_split.bounds,
                quantiles=q_fam_raw,
                volatility=vol_eval,
                liquidity=eval_split.liquidity,
                exposures=eval_split.exposures,
                asset_ids=eval_split.asset_ids,
                n_trials=max(1, n_fam),
                spread_bps=cfg.spread_bps,
                impact_eta=cfg.impact_eta,
                aum_musd=cfg.aum_musd,
                market=cfg.market,  # type: ignore[arg-type]
            )

        ref_fam = blend["reference_family"]
        stage1_pred_ref = np.asarray(eval_split.family_preds[ref_fam], dtype=np.float64)
        stage1_report = family_stage1_reports[ref_fam]

        stage2_report = evaluate_granular_six_dimensions(
            y=y_ev,
            pred=s2_pred_eval,
            bounds=eval_split.bounds,
            quantiles=q_eval_9_cal,
            probabilities=va_eval["p_minimax"],
            volatility=vol_eval,
            liquidity=eval_split.liquidity,
            exposures=eval_split.exposures,
            portfolio_weights=stage2_weights,
            asset_ids=eval_split.asset_ids,
            n_trials=max(1, n_fam),
            spread_bps=cfg.spread_bps,
            impact_eta=cfg.impact_eta,
            aum_musd=cfg.aum_musd,
            market=cfg.market,  # type: ignore[arg-type]
        )

        # (c.4) Contraste pareado Stage-2 vs Stage-1 (Newey-West HAC + deltas 6D)
        ic_s1_list: list[float] = []
        ic_s2_list: list[float] = []
        for lo, hi in eval_split.bounds:
            if hi - lo < cfg.min_assets_per_session:
                continue
            r1 = oa.rank_ic_vector(stage1_pred_ref[lo:hi], y_ev[lo:hi])
            r2 = oa.rank_ic_vector(s2_pred_eval[lo:hi], y_ev[lo:hi])
            if math.isfinite(r1) and math.isfinite(r2):
                ic_s1_list.append(r1)
                ic_s2_list.append(r2)

        d_ic_arr = np.asarray(ic_s2_list, dtype=np.float64) - np.asarray(
            ic_s1_list, dtype=np.float64
        )
        t_ic_nw, se_ic_nw, _ = (
            oa.newey_west_hac(d_ic_arr) if len(d_ic_arr) >= 2 else (None, None, 0)
        )

        net1_s1, to_s1 = _compute_session_net1_series(
            y=y_ev,
            pred=stage1_pred_ref,
            bounds=eval_split.bounds,
            portfolio_weights=None,
            volatility=vol_eval,
            liquidity=eval_split.liquidity,
            spread_bps=cfg.spread_bps,
            impact_eta=cfg.impact_eta,
            aum_musd=cfg.aum_musd,
            asset_ids=eval_split.asset_ids,
            market=cfg.market,
        )
        net1_s2, to_s2 = _compute_session_net1_series(
            y=y_ev,
            pred=s2_pred_eval,
            bounds=eval_split.bounds,
            portfolio_weights=stage2_weights,
            volatility=vol_eval,
            liquidity=eval_split.liquidity,
            spread_bps=cfg.spread_bps,
            impact_eta=cfg.impact_eta,
            aum_musd=cfg.aum_musd,
            asset_ids=eval_split.asset_ids,
            market=cfg.market,
        )
        d_net1 = net1_s2 - net1_s1
        t_net1_nw, se_net1_nw, _ = (
            oa.newey_west_hac(d_net1) if len(d_net1) >= 2 else (None, None, 0)
        )

        # Trazas granulares por sesión para archivo comprimido `.jsonl.gz` sin pérdidas
        session_traces: list[dict[str, Any]] = []
        for s_i, (lo, hi) in enumerate(eval_split.bounds):
            r1_s = oa.rank_ic_vector(stage1_pred_ref[lo:hi], y_ev[lo:hi])
            r2_s = oa.rank_ic_vector(s2_pred_eval[lo:hi], y_ev[lo:hi])
            session_traces.append(
                dict(
                    session_idx=s_i,
                    prediction_at=int(blend["eval_t"][s_i]),
                    n_assets=int(hi - lo),
                    stage1_rank_ic=r1_s if math.isfinite(r1_s) else None,
                    stage2_rank_ic=r2_s if math.isfinite(r2_s) else None,
                    stage1_net1=float(net1_s1[s_i]),
                    stage2_net1=float(net1_s2[s_i]),
                    stage1_turnover=float(to_s1[s_i]),
                    stage2_turnover=float(to_s2[s_i]),
                    active_rate=float(active_rates[s_i]),
                    va_mean_uncertainty=float(np.mean(va_eval["uncertainty"][lo:hi])),
                    blend_weights={
                        fam: float(w_eval_blend[s_i, j]) for j, fam in enumerate(cal_fams)
                    },
                )
            )

        s1_d1, s2_d1 = stage1_report["dim1_value"], stage2_report["dim1_value"]
        s1_d2, s2_d2 = stage1_report["dim2_ranking"], stage2_report["dim2_ranking"]
        s1_d4, s2_d4 = stage1_report["dim4_uncertainty"], stage2_report["dim4_uncertainty"]
        s1_d6, s2_d6 = stage1_report["dim6_economics"], stage2_report["dim6_economics"]

        crps_s1 = float(s1_d4["crps"])
        crps_s2 = float(s2_d4["crps"])
        paired_gain_contrast = dict(
            stage1_reference_family=ref_fam,
            sessions_paired=int(len(d_ic_arr)),
            rank_ic_t_nw_paired=t_ic_nw,
            rank_ic_se_hac_paired=se_ic_nw,
            net_return_1x_t_nw_paired=t_net1_nw,
            net_return_1x_se_hac_paired=se_net1_nw,
            delta_rank_ic=float(s2_d2["rank_ic_mean"]) - float(s1_d2["rank_ic_mean"]),
            delta_pure_alpha_ic=(
                float(s2_d2["pure_alpha_ic_mean"]) - float(s1_d2["pure_alpha_ic_mean"])
            ),
            delta_mae_bps=float(s2_d1["mae_bps"]) - float(s1_d1["mae_bps"]),
            delta_rmse_bps=float(s2_d1["rmse_bps"]) - float(s1_d1["rmse_bps"]),
            delta_crps=crps_s2 - crps_s1,
            crps_improvement=crps_s1 - crps_s2,
            delta_cov_80=float(s2_d4.get("cov_80", 0.0)) - float(s1_d4.get("cov_80", 0.0)),
            delta_cov_90=float(s2_d4.get("cov_90", 0.0)) - float(s1_d4.get("cov_90", 0.0)),
            delta_net_sharpe_1x=float(s2_d6["net_sharpe_1x"]) - float(s1_d6["net_sharpe_1x"]),
            delta_net_sharpe_2x=float(s2_d6["net_sharpe_2x"]) - float(s1_d6["net_sharpe_2x"]),
            delta_net_sharpe_3x=float(s2_d6["net_sharpe_3x"]) - float(s1_d6["net_sharpe_3x"]),
            delta_max_drawdown_1x=(
                float(s2_d6["max_drawdown_1x"]) - float(s1_d6["max_drawdown_1x"])
            ),
            delta_capacity_musd=float(s2_d6["capacity_musd"]) - float(s1_d6["capacity_musd"]),
            delta_turnover=float(s2_d6["turnover_mean"]) - float(s1_d6["turnover_mean"]),
            delta_dsr=float(s2_d6["dsr"]) - float(s1_d6["dsr"]),
        )

        mean_blend_w = {
            fam: float(blend["w_eval"][:, i].mean()) for i, fam in enumerate(cal_fams)
        }
        epistemic_disagreement = dict(
            disagreement_kappa=float(cfg.disagreement_kappa),
            enabled=bool(cfg.disagreement_kappa > 0.0 and n_fam >= 2),
            mean_sigma_inflation_ratio=(
                float(np.mean(disagreement_ratios)) if disagreement_ratios else 1.0
            ),
            mean_disagreement_var=(
                float(np.mean(disagreement_vars)) if disagreement_vars else 0.0
            ),
        )
        return dict(
            config=cfg.to_dict(),
            families=cal_fams,
            decoupler=decoupler.to_dict(),
            conformal_calibration=conf_metrics,
            venn_abers=dict(
                p0=va_eval["p0"],
                p1=va_eval["p1"],
                p_minimax=va_eval["p_minimax"],
                uncertainty=va_eval["uncertainty"],
                mean_uncertainty=float(np.mean(va_eval["uncertainty"])),
                n_cal=va_cal.n_cal,
                effective_n=va_cal.effective_n,
            ),
            blending=dict(
                blend_mode=cfg.blend_mode,
                reference_family=ref_fam,
                cal_family_mean_ic=blend["cal_family_mean_ic"],
                eval_weights_by_session=blend["w_eval"],
                mean_weights=mean_blend_w,
            ),
            governor=dict(
                mean_active_rate=float(np.mean(active_rates)) if active_rates else 1.0,
                cusum_alerts=int(governor.decay_alerts.sum()),
                final_specialist_weights=governor.specialist_weights.tolist(),
            ),
            epistemic_disagreement=epistemic_disagreement,
            stage2_predictions=s2_pred_eval,
            calibrated_quantiles_9=q_eval_9_cal,
            calibrated_quantiles_11=q_eval_11_cal,
            portfolio_weights=stage2_weights,
            family_stage1_reports=family_stage1_reports,
            stage1_report=stage1_report,
            stage2_report=stage2_report,
            paired_gain_contrast=paired_gain_contrast,
            session_traces=session_traces,
            asset_aligned_turnover_mean=float(np.mean(to_s2)),
        )

    def run_cross_family_disagreement_stack(
        self,
        family_preds: Sequence[np.ndarray | torch.Tensor]
        | Mapping[str, np.ndarray | torch.Tensor]
        | np.ndarray
        | torch.Tensor,
        *,
        realized_vol: np.ndarray | torch.Tensor,
        exposures: np.ndarray | torch.Tensor | None = None,
        w_prev: np.ndarray | torch.Tensor | None = None,
        disagreement_kappa: float | None = None,
        cka_weights: Sequence[float] | np.ndarray | torch.Tensor | None = None,
        cka_matrix: Sequence[Sequence[float]] | np.ndarray | torch.Tensor | None = None,
        vol_scale: float | None = None,
        horizon_idx: int = 0,
    ) -> dict[str, Any]:
        """Método de conveniencia de `Stage2PostTrainingPipeline` para apilado de desacuerdo P+."""
        cfg = self.config
        return run_cross_family_disagreement_stack(
            family_preds,
            realized_vol=realized_vol,
            exposures=exposures,
            w_prev=w_prev,
            disagreement_kappa=(
                cfg.disagreement_kappa if disagreement_kappa is None else float(disagreement_kappa)
            ),
            cka_weights=cfg.cka_weights if cka_weights is None else cka_weights,
            cka_matrix=cfg.cka_matrix if cka_matrix is None else cka_matrix,
            vol_scale=vol_scale,
            horizon_idx=horizon_idx,
            risk_aversion=cfg.risk_aversion,
            factor_penalty=cfg.factor_penalty,
            l1_cost=cfg.l1_cost,
            impact_cost=cfg.impact_cost,
            gross_limit=cfg.gross_limit,
            pos_cap=cfg.pos_cap,
            device=cfg.device,
        )


def _extract_6d_row(entry: Mapping[str, Any]) -> dict[str, Any]:
    src = entry.get("granular_report", entry)
    d2 = src.get("dim2_ranking", {}) if isinstance(src, Mapping) else {}
    d4 = src.get("dim4_uncertainty", {}) if isinstance(src, Mapping) else {}
    d5 = src.get("dim5_error_autopsy", {}) if isinstance(src, Mapping) else {}
    d6 = src.get("dim6_economics", {}) if isinstance(src, Mapping) else {}
    return dict(
        variation=str(entry.get("variation", "—")),
        rank_ic=_safe_float(d2.get("rank_ic_mean")),
        pure_alpha_ic=_safe_float(d2.get("pure_alpha_ic_mean")),
        t_nw=_safe_float(d2.get("rank_ic_t_nw")),
        crps=_safe_float(d4.get("crps")),
        cov_80=_safe_float(d4.get("cov_80")),
        net_sr_1x=_safe_float(d6.get("net_sharpe_1x")),
        net_sr_2x=_safe_float(d6.get("net_sharpe_2x")),
        net_sr_3x=_safe_float(d6.get("net_sharpe_3x")),
        max_dd=_safe_float(d6.get("max_drawdown_1x")),
        dsr=_safe_float(d6.get("dsr")),
        capacity_musd=_safe_float(d6.get("capacity_musd")),
        dim5=d5 if isinstance(d5, Mapping) else {},
    )


def _aggregate_telemetry_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Consolida métricas diagnósticas desde una secuencia de registros `telemetry.jsonl`."""
    n = len(records)
    if n == 0:
        return dict(n_logged_steps=0, anomaly_count=0, anomaly_codes=[])

    def _mean_nested(section: str, key: str) -> float | None:
        vals: list[float] = []
        for rec in records:
            sec = rec.get(section)
            if isinstance(sec, Mapping):
                fv = _safe_float(sec.get(key))
                if fv is not None:
                    vals.append(fv)
        return float(sum(vals) / len(vals)) if vals else None

    # Distribución de herramientas del Cognitive Core (top tool + entropía + poda)
    tool_acc: dict[str, list[float]] = {}
    for rec in records:
        dom = rec.get("domain")
        if isinstance(dom, Mapping) and isinstance(dom.get("cognitive_tool_usage"), Mapping):
            for t_name, t_val in dom["cognitive_tool_usage"].items():
                fv = _safe_float(t_val)
                if fv is not None:
                    tool_acc.setdefault(str(t_name), []).append(fv)
    mean_tool_usage = {k: sum(v) / len(v) for k, v in tool_acc.items() if v}
    top_tool = (
        max(mean_tool_usage.items(), key=lambda kv: kv[1])[0] if mean_tool_usage else None
    )

    # Neuro-plasticidad desde `domain` o `extra` (Titans PT1 + Neuro-Brain 3-factor/replay)
    def _mean_dom_or_extra(key: str) -> float | None:
        v = _mean_nested("domain", key)
        return v if v is not None else _mean_nested("extra", key)

    # Ratio Karpathy ||ΔW|| / ||W|| mediano entre órganos
    ur_vals: list[float] = []
    for rec in records:
        organs = rec.get("organs")
        if isinstance(organs, Mapping):
            for snap in organs.values():
                if isinstance(snap, Mapping):
                    fv = _safe_float(snap.get("update_to_weight_ratio"))
                    if fv is not None:
                        ur_vals.append(fv)

    loss_totals = [
        fv
        for rec in records
        if isinstance(rec.get("losses"), Mapping)
        and (fv := _safe_float(rec["losses"].get("total"))) is not None
    ]
    vrams = [
        fv
        for rec in records
        if isinstance(rec.get("hardware"), Mapping)
        and (fv := _safe_float(rec["hardware"].get("cuda_vram_allocated_mb"))) is not None
    ]
    tputs = [
        fv
        for rec in records
        if isinstance(rec.get("hardware"), Mapping)
        and (fv := _safe_float(rec["hardware"].get("sessions_per_sec"))) is not None
    ]

    anom_codes: set[str] = set()
    anom_count = 0
    for rec in records:
        anoms = rec.get("anomalies")
        if isinstance(anoms, Sequence) and not isinstance(anoms, str | bytes):
            anom_count += len(anoms)
            for item in anoms:
                if isinstance(item, Mapping) and item.get("code"):
                    anom_codes.add(str(item["code"]))

    return dict(
        n_logged_steps=n,
        loss_first=loss_totals[0] if loss_totals else None,
        loss_last=loss_totals[-1] if loss_totals else None,
        moe_router_entropy_mean=_mean_nested("domain", "moe_router_entropy"),
        moe_load_max_min_mean=_mean_nested("domain", "moe_load_max_min_ratio"),
        moe_expert_corr_mean=_mean_nested("domain", "moe_expert_mean_corr"),
        loop_k_used_mean=_mean_nested("domain", "loop_k_used"),
        loop_delta_energy_neg_pct_mean=_mean_nested("domain", "loop_delta_energy_neg_pct"),
        loop_max_spectral_norm_mean=_mean_nested("domain", "loop_max_spectral_norm"),
        pt1_alpha_min_mean=_mean_nested("domain", "pt1_alpha_min"),
        pt1_eta_max_mean=_mean_nested("domain", "pt1_eta_max"),
        pt1_theta_max_mean=_mean_nested("domain", "pt1_theta_max"),
        neuromod_ne_slow_shift_mean=_mean_nested("domain", "neuromod_ne_slow_shift"),
        hrr_ood_energy_mean=_mean_nested("domain", "hrr_ood_energy_mean"),
        cognitive_tool_entropy_mean=_mean_nested("domain", "cognitive_tool_entropy"),
        cognitive_pruned_context_ratio_mean=_mean_nested(
            "domain", "cognitive_pruned_context_ratio"
        ),
        cognitive_top_tool=top_tool,
        cognitive_tool_usage_mean=mean_tool_usage,
        three_factor_dopamine_mean=_mean_dom_or_extra("three_factor_dopamine"),
        replay_priority_mean=_mean_dom_or_extra("replay_priority"),
        karpathy_update_ratio_median=float(np.median(ur_vals)) if ur_vals else None,
        peak_cuda_vram_mb=max(vrams) if vrams else 0.0,
        mean_sessions_per_sec=sum(tputs) / len(tputs) if tputs else 0.0,
        anomaly_count=anom_count,
        anomaly_codes=sorted(anom_codes),
    )


def _render_md_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines) + "\n"


def _render_latex_table(
    *,
    caption: str,
    label: str,
    col_spec: str,
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
) -> str:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        " & ".join(headers) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(str(c) for c in row) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines) + "\n"


class TFMTraceAndTableExporter:
    """Exportador de trazas de ejecución y 4 tablas Markdown/LaTeX listas para el TFM."""

    def __init__(
        self,
        *,
        family_evaluations: Mapping[str, Mapping[str, Any]] | None = None,
        gain_verdicts: Mapping[str, GainVerdict | Mapping[str, Any]] | None = None,
        stage2_results: Mapping[str, Any] | None = None,
        telemetry_paths: Mapping[str, str | Path] | str | Path | None = None,
        telemetry_records: Mapping[str, Sequence[Mapping[str, Any]]]
        | Sequence[Mapping[str, Any]]
        | None = None,
        experiment_memory_path: str | Path | None = None,
        experiment_records: Sequence[Mapping[str, Any]] | None = None,
    ) -> None:
        self._family_evals: dict[str, Mapping[str, Any]] = (
            dict(family_evaluations) if family_evaluations is not None else {}
        )
        self._gain_verdicts: dict[str, dict[str, Any]] = {}
        if gain_verdicts is not None:
            for k, v in gain_verdicts.items():
                self._gain_verdicts[str(k)] = v.to_dict() if isinstance(v, GainVerdict) else dict(v)

        # Normalizar stage2_results: acepta un único resultado o un mapa {nombre: resultado}
        self._stage2_results: dict[str, Mapping[str, Any]] = {}
        if stage2_results is not None:
            if "paired_gain_contrast" in stage2_results and "stage2_report" in stage2_results:
                self._stage2_results["stage2_sota"] = stage2_results
            else:
                for k, v in stage2_results.items():
                    if isinstance(v, Mapping):
                        self._stage2_results[str(k)] = v

        # Cargar telemetría JSONL / JSONL.GZ (mapa por familia o archivo único)
        self._telemetry_by_run: dict[str, list[dict[str, Any]]] = {}
        if isinstance(telemetry_paths, str | Path):
            p_tel = Path(telemetry_paths)
            recs = _read_jsonl_optional(p_tel) if _is_gzip_file(p_tel) else read_telemetry_jsonl(
                p_tel
            )
            if recs:
                self._telemetry_by_run["default"] = recs
        elif isinstance(telemetry_paths, Mapping):
            for name, path in telemetry_paths.items():
                p_tel = Path(path)
                recs = (
                    _read_jsonl_optional(p_tel)
                    if _is_gzip_file(p_tel)
                    else read_telemetry_jsonl(p_tel)
                )
                if recs:
                    self._telemetry_by_run[str(name)] = recs

        if isinstance(telemetry_records, Mapping):
            for name, recs in telemetry_records.items():
                self._telemetry_by_run[str(name)] = [dict(r) for r in recs]
        elif telemetry_records is not None:
            self._telemetry_by_run["default"] = [dict(r) for r in telemetry_records]

        # Cargar ledger de memoria de experimentos (`experiment_memory.jsonl`, opcional)
        loaded_exp = _read_jsonl_optional(experiment_memory_path)
        if experiment_records is not None:
            loaded_exp.extend(dict(r) for r in experiment_records)
        self._experiment_records: list[dict[str, Any]] = loaded_exp

    def _ordered_family_names(self) -> list[str]:
        ordered = [f for f in FOUR_FAMILIES if f in self._family_evals]
        ordered.extend(sorted(k for k in self._family_evals if k not in ordered))
        return ordered

    def _build_table_1_rows(self, *, latex: bool) -> tuple[list[str], list[list[str]]]:
        headers = (
            [
                "Family",
                "Rank IC",
                "Pure Alpha IC",
                r"$t_{\mathrm{NW}}$",
                "CRPS",
                "Cov80",
                "Net SR (1x)",
                "Net SR (2x)",
                "MaxDD",
                "DSR",
            ]
            if latex
            else [
                "Family",
                "Rank IC",
                "Pure Alpha IC",
                "t_NW",
                "CRPS",
                "Cov80",
                "Net SR (1x)",
                "Net SR (2x)",
                "MaxDD",
                "DSR",
            ]
        )
        rows: list[list[str]] = []
        for fam in self._ordered_family_names():
            m = _extract_6d_row(self._family_evals[fam])
            label = _latex_escape(fam) if latex else fam
            rows.append(
                [
                    label,
                    _fmt_num(m["rank_ic"], ".4f", latex=latex),
                    _fmt_num(m["pure_alpha_ic"], ".4f", latex=latex),
                    _fmt_num(m["t_nw"], ".2f", latex=latex),
                    _fmt_num(m["crps"], ".5f", latex=latex),
                    _fmt_num(m["cov_80"], ".3f", latex=latex),
                    _fmt_num(m["net_sr_1x"], ".2f", latex=latex),
                    _fmt_num(m["net_sr_2x"], ".2f", latex=latex),
                    _fmt_num(m["max_dd"], ".4f", latex=latex),
                    _fmt_num(m["dsr"], ".3f", latex=latex),
                ]
            )
        if not rows:
            placeholder = "--" if latex else "—"
            rows.append([placeholder] * len(headers))
        return headers, rows

    def _build_table_2_rows(self, *, latex: bool) -> tuple[list[str], list[list[str]]]:
        headers = (
            [
                "Variation",
                "Controls",
                r"$\min \Delta\mathrm{IC}$",
                r"$\min \Delta\mathrm{PureIC}$",
                r"$\min \mathrm{CRPS\ Imp.}$",
                r"$\min \Delta\mathrm{SR}_{1\times}$",
                "DSR",
                r"$t_{\mathrm{NW}}$",
                "Verdict",
            ]
            if latex
            else [
                "Variation",
                "Controls",
                "min dRankIC",
                "min dPureIC",
                "min CRPS Imp.",
                "min dNetSR1x",
                "DSR",
                "t_NW",
                "Verdict",
            ]
        )
        rows: list[list[str]] = []
        for name in sorted(self._gain_verdicts):
            vd = self._gain_verdicts[name]
            deltas = vd.get("deltas_vs_controls", {})
            d_ric = (
                min((float(d["delta_rank_ic"]) for d in deltas.values()), default=None)
                if isinstance(deltas, Mapping) and deltas
                else None
            )
            d_pure = (
                min((float(d["delta_pure_alpha_ic"]) for d in deltas.values()), default=None)
                if isinstance(deltas, Mapping) and deltas
                else None
            )
            d_crps = (
                min((float(d["crps_improvement"]) for d in deltas.values()), default=None)
                if isinstance(deltas, Mapping) and deltas
                else None
            )
            d_sr1 = (
                min((float(d["delta_net_sharpe_1x"]) for d in deltas.values()), default=None)
                if isinstance(deltas, Mapping) and deltas
                else None
            )
            cm = vd.get("candidate_metrics", {})
            n_ctrl = len(vd.get("controls_evaluated", []))
            verdict_str = str(vd.get("verdict", "REJECT"))
            label = _latex_escape(name) if latex else name
            v_cell = (
                rf"\textbf{{{_latex_escape(verdict_str)}}}" if latex else f"**{verdict_str}**"
            )
            rows.append(
                [
                    label,
                    str(n_ctrl),
                    _fmt_num(d_ric, ".4f", signed=True, latex=latex),
                    _fmt_num(d_pure, ".4f", signed=True, latex=latex),
                    _fmt_num(d_crps, ".5f", signed=True, latex=latex),
                    _fmt_num(d_sr1, ".2f", signed=True, latex=latex),
                    _fmt_num(cm.get("dsr"), ".3f", latex=latex),
                    _fmt_num(cm.get("rank_ic_t_nw"), ".2f", latex=latex),
                    v_cell,
                ]
            )
        if not rows:
            placeholder = "--" if latex else "—"
            rows.append([placeholder] * len(headers))
        return headers, rows

    def _build_table_3_rows(self, *, latex: bool) -> tuple[list[str], list[list[str]]]:
        headers = (
            [
                "Run / Family",
                "MoE Ent.",
                "MoE Max/Min",
                "Loop $k$",
                r"$\Delta E_{\mathrm{fin}}<0\,\%$",
                r"PT1 $\alpha_{\min}$",
                "Tool Ent.",
                "Pruned Ctx",
                "Neuro DA/Repl.",
                r"$\|\Delta W\|/\|W\|$",
            ]
            if latex
            else [
                "Run / Family",
                "MoE Ent.",
                "MoE Max/Min",
                "Loop k",
                "dE_fin<0 %",
                "PT1 alpha_min",
                "Tool Ent.",
                "Pruned Ctx",
                "Neuro DA/Repl.",
                "||dW||/||W||",
            ]
        )
        rows: list[list[str]] = []
        seen_names: set[str] = set()
        for run_name in sorted(self._telemetry_by_run):
            seen_names.add(run_name)
            agg = _aggregate_telemetry_records(self._telemetry_by_run[run_name])
            neuro_val = (
                agg["three_factor_dopamine_mean"]
                if agg["three_factor_dopamine_mean"] is not None
                else agg["neuromod_ne_slow_shift_mean"]
            )
            label = _latex_escape(run_name) if latex else run_name
            rows.append(
                [
                    label,
                    _fmt_num(agg["moe_router_entropy_mean"], ".3f", latex=latex),
                    _fmt_num(agg["moe_load_max_min_mean"], ".2f", latex=latex),
                    _fmt_num(agg["loop_k_used_mean"], ".2f", latex=latex),
                    _fmt_num(agg["loop_delta_energy_neg_pct_mean"], ".1f", latex=latex),
                    _fmt_num(agg["pt1_alpha_min_mean"], ".3f", latex=latex),
                    _fmt_num(agg["cognitive_tool_entropy_mean"], ".3f", latex=latex),
                    _fmt_num(agg["cognitive_pruned_context_ratio_mean"], ".3f", latex=latex),
                    _fmt_num(neuro_val, ".3f", latex=latex),
                    _fmt_num(agg["karpathy_update_ratio_median"], ".2e", latex=latex),
                ]
            )

        # Completar con diagnósticos `dim5_error_autopsy` si no hubo telemetría para esa familia
        for fam in self._ordered_family_names():
            if fam in seen_names:
                continue
            d5 = _extract_6d_row(self._family_evals[fam])["dim5"]
            if not d5:
                continue
            label = _latex_escape(fam) if latex else fam
            de_pct = (
                100.0 * float(d5["delta_energy_neg_rate"])
                if _safe_float(d5.get("delta_energy_neg_rate")) is not None
                else None
            )
            rows.append(
                [
                    label,
                    _fmt_num(d5.get("router_entropy_mean"), ".3f", latex=latex),
                    _fmt_num(d5.get("load_max_min_mean"), ".2f", latex=latex),
                    _fmt_num(d5.get("k_used_mean"), ".2f", latex=latex),
                    _fmt_num(de_pct, ".1f", latex=latex),
                    _fmt_num(d5.get("pt1_alpha_min"), ".3f", latex=latex),
                    _fmt_num(d5.get("cognitive_tool_entropy"), ".3f", latex=latex),
                    _fmt_num(d5.get("cognitive_pruned_context_ratio"), ".3f", latex=latex),
                    _fmt_num(d5.get("three_factor_dopamine"), ".3f", latex=latex),
                    _fmt_num(None, ".2e", latex=latex),
                ]
            )
        if not rows:
            placeholder = "--" if latex else "—"
            rows.append([placeholder] * len(headers))
        return headers, rows

    def _build_table_4_rows(self, *, latex: bool) -> tuple[list[str], list[list[str]]]:
        headers = (
            [
                "Run (Ref.)",
                r"$\Delta$Rank IC",
                r"$t_{\mathrm{NW}}$",
                r"$\Delta$CRPS",
                "Cov80/90/95",
                r"$\Delta\mathrm{SR}_{1\times}$",
                r"$\Delta\mathrm{SR}_{2\times}$",
                r"$\Delta\mathrm{SR}_{3\times}$",
                r"$\Delta$MaxDD",
                r"$\Delta$Cap (M\$)",
            ]
            if latex
            else [
                "Run (Ref.)",
                "dRank IC",
                "t_NW (paired)",
                "dCRPS",
                "Cov80/90/95",
                "dNetSR 1x",
                "dNetSR 2x",
                "dNetSR 3x",
                "dMaxDD",
                "dCap (M$)",
            ]
        )
        rows: list[list[str]] = []
        for run_name in sorted(self._stage2_results):
            res = self._stage2_results[run_name]
            c = res.get("paired_gain_contrast", {})
            conf = res.get("conformal_calibration", {})
            ref_f = str(c.get("stage1_reference_family", "stage1"))
            run_label = f"{run_name} ({ref_f})"
            if latex:
                run_label = _latex_escape(run_label)
            cov_str = "/".join(
                _fmt_num(conf.get(f"cov_{t}"), ".2f", latex=latex) for t in (80, 90, 95)
            )
            rows.append(
                [
                    run_label,
                    _fmt_num(c.get("delta_rank_ic"), ".4f", signed=True, latex=latex),
                    _fmt_num(c.get("rank_ic_t_nw_paired"), ".2f", signed=True, latex=latex),
                    _fmt_num(c.get("delta_crps"), ".5f", signed=True, latex=latex),
                    cov_str,
                    _fmt_num(c.get("delta_net_sharpe_1x"), ".2f", signed=True, latex=latex),
                    _fmt_num(c.get("delta_net_sharpe_2x"), ".2f", signed=True, latex=latex),
                    _fmt_num(c.get("delta_net_sharpe_3x"), ".2f", signed=True, latex=latex),
                    _fmt_num(c.get("delta_max_drawdown_1x"), ".4f", signed=True, latex=latex),
                    _fmt_num(c.get("delta_capacity_musd"), ".1f", signed=True, latex=latex),
                ]
            )
        if not rows:
            placeholder = "--" if latex else "—"
            rows.append([placeholder] * len(headers))
        return headers, rows

    def export_markdown_tables(
        self, *, out_dir: str | Path | None = None
    ) -> dict[str, str]:
        """Exporta las 4 tablas TFM en Markdown GFM (y opcionalmente las escribe a disco)."""
        specs = (
            (
                "master_four_family_comparison",
                "### Table 1: Four-Family Master Comparison (6-Dimension Granular Metrics)",
                self._build_table_1_rows(latex=False),
            ),
            (
                "paired_control_gain_validation",
                "### Table 2: Paired Control & Gain Validation Verdicts",
                self._build_table_2_rows(latex=False),
            ),
            (
                "diagnostic_trace_summary",
                "### Table 3: Tool-Use / MoE / Loop-Depth / Neuro-Plasticity Diagnostic Traces",
                self._build_table_3_rows(latex=False),
            ),
            (
                "stage1_vs_stage2_gain",
                "### Table 4: Stage-1 Pre-Training vs Stage-2 Post-Training Paired Gains",
                self._build_table_4_rows(latex=False),
            ),
        )
        tables: dict[str, str] = {}
        sections: list[str] = []
        for key, title, (hdrs, rows) in specs:
            md_block = f"{title}\n\n{_render_md_table(hdrs, rows)}"
            tables[key] = md_block
            sections.append(md_block)

        tables["combined"] = "\n".join(sections)
        if out_dir is not None:
            base = Path(out_dir)
            base.mkdir(parents=True, exist_ok=True)
            for key, content in tables.items():
                fname = "tfm_all_tables.md" if key == "combined" else f"{key}.md"
                _atomic_write_text(base / fname, content)
        return tables

    def export_latex_tables(
        self, *, out_dir: str | Path | None = None
    ) -> dict[str, str]:
        """Exporta las 4 tablas TFM en LaTeX `booktabs` (y opcionalmente las escribe a disco)."""
        specs = (
            (
                "master_four_family_comparison",
                "Four-Family Master Comparison across 6-Dimension Granular OOS Metrics.",
                "tab:four_family_master",
                "lrrrrrrrrr",
                self._build_table_1_rows(latex=True),
            ),
            (
                "paired_control_gain_validation",
                "Paired Matched-Control Gain Validation Verdicts (PASS/REJECT).",
                "tab:paired_control_gain",
                "lrrrrrrrr",
                self._build_table_2_rows(latex=True),
            ),
            (
                "diagnostic_trace_summary",
                "Tool-Use, MoE, Loop-Depth, and Neuro-Plasticity Diagnostic Trace Summary.",
                "tab:diagnostic_traces",
                "lrrrrrrrrr",
                self._build_table_3_rows(latex=True),
            ),
            (
                "stage1_vs_stage2_gain",
                "Stage-1 Pre-Training vs Stage-2 Post-Training Paired Gain Contrasts.",
                "tab:stage1_vs_stage2_gain",
                "lrrrcrrrrr",
                self._build_table_4_rows(latex=True),
            ),
        )
        tables: dict[str, str] = {}
        sections: list[str] = []
        for key, caption, label, col_spec, (hdrs, rows) in specs:
            tex = _render_latex_table(
                caption=caption, label=label, col_spec=col_spec, headers=hdrs, rows=rows
            )
            tables[key] = tex
            sections.append(tex)

        tables["combined"] = "\n".join(sections)
        if out_dir is not None:
            base = Path(out_dir)
            base.mkdir(parents=True, exist_ok=True)
            for key, content in tables.items():
                fname = "tfm_all_tables.tex" if key == "combined" else f"{key}.tex"
                _atomic_write_text(base / fname, content)
        return tables

    def export_execution_trace_summary(
        self, *, output_path: str | Path | None = None
    ) -> dict[str, Any]:
        """Consolida telemetría, memoria de experimentos, veredictos y métricas Stage-2 en JSON."""
        telemetry_summary = {
            run_name: _aggregate_telemetry_records(recs)
            for run_name, recs in sorted(self._telemetry_by_run.items())
        }

        verdict_counts: dict[str, int] = {"KEEP": 0, "REJECT": 0, "INVESTIGATE": 0, "BUG_CAUGHT": 0}
        by_family: dict[str, int] = {}
        rejected_roots: list[dict[str, str]] = []
        for rec in self._experiment_records:
            v = str(rec.get("verdict", "UNKNOWN"))
            verdict_counts[v] = verdict_counts.get(v, 0) + 1
            fam = str(rec.get("family", "unknown"))
            by_family[fam] = by_family.get(fam, 0) + 1
            if v in ("REJECT", "BUG_CAUGHT"):
                rejected_roots.append(
                    dict(
                        experiment_id=str(rec.get("experiment_id", "")),
                        family=fam,
                        variation=str(rec.get("variation", "")),
                        verdict=v,
                        root_cause=str(rec.get("failure_mode_or_root_cause", "")),
                    )
                )

        stage2_summary: dict[str, Any] = {}
        for run_name, res in sorted(self._stage2_results.items()):
            va = res.get("venn_abers", {})
            bl = res.get("blending", {})
            stage2_summary[run_name] = dict(
                decoupler=res.get("decoupler", {}),
                conformal_calibration=res.get("conformal_calibration", {}),
                venn_abers_mean_uncertainty=_safe_float(va.get("mean_uncertainty")),
                mean_blend_weights=bl.get("mean_weights", {}),
                governor=res.get("governor", {}),
                paired_gain_contrast=res.get("paired_gain_contrast", {}),
            )

        summary: dict[str, Any] = dict(
            generated_at_utc=_utc_now_iso(),
            families_present=self._ordered_family_names(),
            tables_manifest=[
                "master_four_family_comparison",
                "paired_control_gain_validation",
                "diagnostic_trace_summary",
                "stage1_vs_stage2_gain",
            ],
            telemetry_summary=telemetry_summary,
            experiment_memory_summary=dict(
                total_experiments=len(self._experiment_records),
                verdict_counts=verdict_counts,
                experiments_by_family=by_family,
                rejected_root_causes=rejected_roots,
            ),
            gain_verdicts_summary={
                k: dict(verdict=v.get("verdict"), promoted=bool(v.get("promoted", False)))
                for k, v in sorted(self._gain_verdicts.items())
            },
            stage2_summary=stage2_summary,
        )

        if output_path is not None:
            out_p = Path(output_path)
            compact = json.dumps(
                _json_clean(summary), sort_keys=True, separators=(",", ":"), allow_nan=False
            )
            if out_p.suffix == ".gz":
                out_p.parent.mkdir(parents=True, exist_ok=True)
                tmp = out_p.with_name(f".{out_p.name}.tmp.{os.getpid()}")
                with gzip.open(tmp, "wt", encoding="utf-8", compresslevel=6) as fh:
                    fh.write(compact + "\n")
                os.replace(tmp, out_p)
            else:
                _atomic_write_text(out_p, compact + "\n")
        return summary

    def export_compressed_trace_bundle(
        self,
        output_path: str | Path,
        *,
        compresslevel: int = 6,
    ) -> dict[str, Any]:
        """Exportador primario de trazas ultra-compactas comprimidas sin pérdida (`.jsonl.gz`).

        Preserva con 100 % de fidelidad cada evaluación 6D por familia, veredicto de ganancia,
        resumen y trazas por sesión de Stage-2, pasos de telemetría y registros de memoria
        experimental usando `gzip` y `separators=(',',':')` para mínima huella en disco.
        """
        records: list[dict[str, Any]] = []
        counts: dict[str, int] = {}

        def _push(rec_type: str, payload: Mapping[str, Any]) -> None:
            entry = dict(payload)
            entry["record_type"] = rec_type
            records.append(entry)
            counts[rec_type] = counts.get(rec_type, 0) + 1

        _push(
            "bundle_header",
            dict(
                schema_version=1,
                generated_at_utc=_utc_now_iso(),
                families_present=self._ordered_family_names(),
                telemetry_runs=sorted(self._telemetry_by_run),
                stage2_runs=sorted(self._stage2_results),
                experiment_records=len(self._experiment_records),
            ),
        )

        for fam in self._ordered_family_names():
            _push("family_evaluation_6d", dict(family=fam, report=self._family_evals[fam]))

        for name in sorted(self._gain_verdicts):
            _push("gain_verdict", dict(name=name, verdict=self._gain_verdicts[name]))

        for run_name in sorted(self._stage2_results):
            res = self._stage2_results[run_name]
            va = res.get("venn_abers", {})
            bl = res.get("blending", {})
            _push(
                "stage2_summary",
                dict(
                    run_name=run_name,
                    decoupler=res.get("decoupler", {}),
                    conformal_calibration=res.get("conformal_calibration", {}),
                    venn_abers_mean_uncertainty=_safe_float(va.get("mean_uncertainty")),
                    mean_blend_weights=bl.get("mean_weights", {}),
                    governor=res.get("governor", {}),
                    stage1_report=res.get("stage1_report", {}),
                    stage2_report=res.get("stage2_report", {}),
                    paired_gain_contrast=res.get("paired_gain_contrast", {}),
                ),
            )
            for s_tr in res.get("session_traces", []):
                _push("stage2_session_trace", dict(run_name=run_name, **dict(s_tr)))

        for run_name in sorted(self._telemetry_by_run):
            for rec in self._telemetry_by_run[run_name]:
                _push("telemetry_step", dict(run_name=run_name, payload=rec))

        for exp in self._experiment_records:
            _push("experiment_record", dict(payload=exp))

        meta = write_compressed_jsonl(output_path, records, compresslevel=compresslevel)
        meta["record_counts"] = counts
        return meta


__all__ = [
    "EXTENDED_QUANTILE_LEVELS_11",
    "FOUR_FAMILIES",
    "LocallyNormalizedConformalPID",
    "OrderLevelDecoupler",
    "Stage2Config",
    "Stage2PostTrainingPipeline",
    "Stage2SplitData",
    "TFMTraceAndTableExporter",
    "read_compressed_trace_bundle",
    "run_cross_family_disagreement_stack",
    "write_compressed_jsonl",
]
