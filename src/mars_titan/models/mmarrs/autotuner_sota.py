"""Motor SOTA de auto-ajuste determinista y control óptimo de ajuste para los 4 modelos Titans.

Diseñado para maximizar capacidad predictiva sin sobreajuste ni subajuste en las 4 familias
independientes (`titans_original`, `holographic_brain`, `mmarrs_titan`, `titans_cognitive_core`)
sobre NVIDIA RTX PRO 6000 Blackwell (`sm_120`) y CPU en modo FP32 estricto o BF16 AMP:

  1. `SearchSpaceSpec` + `TPEHyperbandAutoTuner`:
     - Búsqueda TPE determinista (cero dependencia Optuna) + poda ASHA / Successive Halving.
     - Busca `(lr, weight_decay, hidden_dim, router_lr_scale, modality_dropout, ema_decay,
       w_crps, w_sharpe, w_energy)` exclusivamente sobre validación purgada (bloqueo duro de
       `2024+` mediante `SealedTestViolationError`).
     - Registra el 100 % de los ensayos (completados, podados y fallidos) en un ledger determinista
       JSONL con huella SHA-256, calcula el recorte de multiplicidad de Bailey & López de Prado +
       `deflated_sharpe_ratio` (`granular_evaluator.py`) y ejecuta comprobación de acantilados de
       sensibilidad (`check_parameter_sensitivity_cliff`) para rechazar picos frágiles.
  2. `OptimalFitController`:
     - Parada temprana adaptativa suavizada probada en DGX (`smoothing=3`, `patience=5`, decaimiento
       automático de LR `0.5x` hasta 2 veces y promedio exacto de pesos Top-3 / Model Soup).
     - Diagnóstico dual por época (`underfitting` vs `healthy` vs `overfitting`) a partir de
       `(train_ic, val_ic, val_crps, update_weight_ratio)`.
  3. `shrink_and_perturb` (Ash & Adams, NeurIPS):
     - Rompe la pérdida de plasticidad a lo largo de 2000–2023 contrayendo normas con perturbación
       determinista de inicialización fresca.
  4. `build_sota_optimizer` + `MuonHybridOptimizer` + `ScheduleFreeAdamW`:
     - `adamw_fused` (con LR `3x` menor en routers MoE), `muon_hybrid` (ortogonalización
       Newton-Schulz de 5 iteraciones en matrices ocultas 2D + AdamW en 1D/embeddings/cabezas/
       routers, 100 % PyTorch nativo en GPU con cero sincronizaciones host) y `schedule_free_adamw`
       con interpolación train/eval.
  5. `rank_ensemble_predictions` + `evaluate_rank_ensemble`:
     - Ensamblado determinista por rangos de 3 semillas (`+0.002..+0.004` IC probado en DGX).
"""

from __future__ import annotations

import gzip
import hashlib
import inspect
import json
import math
import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from scipy import stats
from torch import nn

from mars_titan.evaluation.granular_evaluator import EULER_GAMMA, deflated_sharpe_ratio
from mars_titan.evaluation.online_aggregator import average_ranks, rank_ic_vector

from .hardware_blackwell import BlackwellRuntimeConfig, build_optimizer
from .long_running_harness import validate_no_sealed_leakage

ParamKind = Literal["float", "log_float", "int", "categorical"]
FitRegime = Literal["underfitting", "healthy", "overfitting"]
RankEnsembleMethod = Literal["rank_zscore", "rank_uniform", "gaussian_rank"]

DEFAULT_SEARCH_PARAM_NAMES: tuple[str, ...] = (
    "lr",
    "weight_decay",
    "hidden_dim",
    "router_lr_scale",
    "modality_dropout",
    "ema_decay",
    "w_crps",
    "w_sharpe",
    "w_energy",
)

SOTA_OPTIMIZER_TYPES: tuple[str, ...] = (
    "adamw_fused",
    "muon_hybrid",
    "schedule_free_adamw",
)

_ROUTER_PARAM_TOKENS: tuple[str, ...] = (
    "router",
    "balance_bias",
    "halt_gate",
    "mix_gate",
)

_HEAD_OR_EMBED_TOKENS: tuple[str, ...] = (
    "embed",
    "alpha_head",
    "q_median",
    "q_deltas_lo",
    "q_deltas_hi",
    "tb_head",
    "turn_head",
    "vol_head",
    "epistemic_head",
    "factor_load",
)


class TrialPruned(Exception):
    """Excepción de control lanzada cuando ASHA poda un ensayo en un escalón intermedio."""

    def __init__(self, message: str = "trial_pruned", *, resource: int = 0) -> None:
        super().__init__(message)
        self.resource = int(resource)


def _json_sanitize(obj: Any) -> Any:
    """Convierte valores arbitrarios a JSON estricto RFC 8259 (NaN/Inf -> `None`)."""
    if obj is None or isinstance(obj, bool | str | int):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return _json_sanitize(obj.tolist())
    if isinstance(obj, torch.Tensor):
        if obj.numel() == 1:
            v = float(obj.detach().cpu().item())
            return v if math.isfinite(v) else None
        return _json_sanitize(obj.detach().cpu().tolist())
    if isinstance(obj, dict):
        return {str(k): _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_json_sanitize(v) for v in obj]
    if hasattr(obj, "item") and callable(obj.item):
        try:
            return _json_sanitize(obj.item())
        except Exception:
            pass
    return str(obj)


def _canonical_sha256(payload: Any) -> str:
    clean = _json_sanitize(payload)
    raw = json.dumps(clean, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def validate_purged_validation_split(
    train_years: Sequence[int],
    val_years: Sequence[int],
    *,
    sealed_test_start_year: int = 2024,
) -> None:
    """Valida orden temporal estricto (train < val) y bloqueo total de años `>= 2024`."""
    if not train_years or not val_years:
        raise ValueError("train_years y val_years no pueden estar vacíos")
    validate_no_sealed_leakage(train_years, sealed_test_start_year=sealed_test_start_year)
    validate_no_sealed_leakage(val_years, sealed_test_start_year=sealed_test_start_year)
    max_train = max(int(y) for y in train_years)
    min_val = min(int(y) for y in val_years)
    if max_train >= min_val:
        raise ValueError(
            f"Violación de orden cronológico / fuga temporal: max(train_years)={max_train} "
            f">= min(val_years)={min_val}"
        )


# ---------------------------------------------------------------------------
# 1. Espacio de búsqueda, TPE determinista + ASHA + multiplicidad + acantilados
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParamSpec:
    """Definición validada de una dimensión del espacio de hiperparámetros."""

    name: str
    kind: ParamKind
    low: float | None = None
    high: float | None = None
    choices: tuple[Any, ...] | None = None

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("ParamSpec.name no puede estar vacío")
        if self.kind not in ("float", "log_float", "int", "categorical"):
            raise ValueError(f"ParamSpec.kind inválido: {self.kind!r}")
        if self.kind == "categorical":
            if not self.choices or len(self.choices) < 2:
                raise ValueError(
                    f"Parámetro categórico {self.name!r} requiere al menos 2 choices"
                )
        else:
            if self.low is None or self.high is None:
                raise ValueError(f"Parámetro numérico {self.name!r} requiere low y high")
            if float(self.low) >= float(self.high):
                raise ValueError(
                    f"Parámetro {self.name!r}: low ({self.low}) debe ser < high ({self.high})"
                )
            if self.kind == "log_float" and float(self.low) <= 0.0:
                raise ValueError(f"Parámetro log_float {self.name!r} requiere low > 0")

    def to_unit(self, value: Any) -> float:
        """Proyecta un valor numérico del parámetro al intervalo normalizado `[0, 1]`."""
        if self.kind == "categorical":
            assert self.choices is not None
            idx = self.choices.index(value)
            return float(idx) / float(max(len(self.choices) - 1, 1))
        assert self.low is not None and self.high is not None
        val = float(value)
        if self.kind == "log_float":
            lo = math.log(float(self.low))
            hi = math.log(float(self.high))
            u = (math.log(max(val, float(self.low))) - lo) / (hi - lo)
        else:
            u = (val - float(self.low)) / (float(self.high) - float(self.low))
        return float(min(max(u, 0.0), 1.0))

    def from_unit(self, u: float) -> Any:
        """Invierte un escalar en `[0, 1]` al dominio tipado del parámetro."""
        uc = float(min(max(u, 0.0), 1.0))
        if self.kind == "categorical":
            assert self.choices is not None
            idx = int(round(uc * (len(self.choices) - 1)))
            return self.choices[min(max(idx, 0), len(self.choices) - 1)]
        assert self.low is not None and self.high is not None
        if self.kind == "log_float":
            lo = math.log(float(self.low))
            hi = math.log(float(self.high))
            return float(math.exp(lo + uc * (hi - lo)))
        if self.kind == "int":
            lo_i = int(round(self.low))
            hi_i = int(round(self.high))
            return int(min(max(round(lo_i + uc * (hi_i - lo_i)), lo_i), hi_i))
        return float(float(self.low) + uc * (float(self.high) - float(self.low)))

    def sample_random(self, rng: np.random.Generator) -> Any:
        """Muestreo uniforme determinista (en escala log para `log_float`)."""
        if self.kind == "categorical":
            assert self.choices is not None
            return self.choices[int(rng.integers(0, len(self.choices)))]
        if self.kind == "int":
            assert self.low is not None and self.high is not None
            return int(rng.integers(int(round(self.low)), int(round(self.high)) + 1))
        return self.from_unit(float(rng.uniform(0.0, 1.0)))


@dataclass(frozen=True)
class SearchSpaceSpec:
    """Contrato inmutable del espacio de búsqueda para cualquiera de las 4 familias Titans."""

    params: tuple[ParamSpec, ...]
    sealed_test_start_year: int = 2024

    def __post_init__(self) -> None:
        if not self.params:
            raise ValueError("SearchSpaceSpec requiere al menos un ParamSpec")
        names = [p.name for p in self.params]
        if len(set(names)) != len(names):
            raise ValueError(f"Nombres de parámetros duplicados en SearchSpaceSpec: {names}")
        if self.sealed_test_start_year < 2001:
            raise ValueError("sealed_test_start_year debe ser >= 2001")

    @property
    def param_names(self) -> tuple[str, ...]:
        return tuple(p.name for p in self.params)

    def by_name(self, name: str) -> ParamSpec:
        for p in self.params:
            if p.name == name:
                return p
        raise KeyError(f"Parámetro desconocido {name!r}")

    def sample_random(self, rng: np.random.Generator) -> dict[str, Any]:
        return {p.name: p.sample_random(rng) for p in self.params}

    def validate_params(self, params: Mapping[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for p in self.params:
            if p.name not in params:
                raise ValueError(f"Falta el parámetro obligatorio {p.name!r}")
            val = params[p.name]
            if p.kind == "categorical":
                assert p.choices is not None
                if val not in p.choices:
                    raise ValueError(f"{p.name}={val!r} fuera de choices={p.choices}")
                out[p.name] = val
            else:
                assert p.low is not None and p.high is not None
                fv = float(val)
                if not math.isfinite(fv) or fv < float(p.low) - 1e-12 or fv > float(p.high) + 1e-12:
                    raise ValueError(
                        f"{p.name}={val!r} fuera del rango [{p.low}, {p.high}]"
                    )
                out[p.name] = int(round(fv)) if p.kind == "int" else fv
        return out

    def to_dict(self) -> dict[str, Any]:
        return dict(
            sealed_test_start_year=self.sealed_test_start_year,
            params=[asdict(p) for p in self.params],
        )

    @classmethod
    def default_mmarrs_space(
        cls,
        *,
        sealed_test_start_year: int = 2024,
    ) -> SearchSpaceSpec:
        """Espacio canónico de 9 hiperparámetros para MMARRS-TITAN y las 4 familias Titans."""
        return cls(
            sealed_test_start_year=sealed_test_start_year,
            params=(
                ParamSpec("lr", "log_float", low=1e-4, high=5e-3),
                ParamSpec("weight_decay", "log_float", low=1e-5, high=2e-2),
                ParamSpec("hidden_dim", "categorical", choices=(64, 96, 128, 192, 256)),
                ParamSpec("router_lr_scale", "float", low=0.10, high=0.50),
                ParamSpec("modality_dropout", "float", low=0.0, high=0.30),
                ParamSpec("ema_decay", "float", low=0.95, high=0.999),
                ParamSpec("w_crps", "log_float", low=5.0, high=30.0),
                ParamSpec("w_sharpe", "log_float", low=0.05, high=1.0),
                ParamSpec("w_energy", "log_float", low=0.05, high=1.0),
            ),
        )


@dataclass
class TrialRecord:
    """Registro auditable de un ensayo (completado, podado o fallido)."""

    trial_id: int
    params: dict[str, Any]
    status: Literal["completed", "pruned", "failed"]
    score: float | None
    intermediate_scores: dict[int, float] = field(default_factory=dict)
    max_resource_reached: int = 0
    metrics: dict[str, Any] = field(default_factory=dict)
    returns: np.ndarray | None = None
    error: str | None = None
    config_sha256: str = ""

    def __post_init__(self) -> None:
        if not self.config_sha256:
            self.config_sha256 = _canonical_sha256(self.params)

    def to_audit_dict(self) -> dict[str, Any]:
        """Serialización JSON estricta para el ledger append-only de auditoría TFM."""
        ret_summary: dict[str, float] | None = None
        if self.returns is not None:
            r = np.asarray(self.returns, dtype=np.float64)
            r = r[np.isfinite(r)]
            if len(r) > 0:
                ret_summary = dict(
                    n_obs=int(len(r)),
                    mean=float(np.mean(r)),
                    std=float(np.std(r, ddof=1)) if len(r) > 1 else 0.0,
                )
        return _json_sanitize(
            dict(
                trial_id=self.trial_id,
                config_sha256=self.config_sha256,
                params=dict(self.params),
                status=self.status,
                score=self.score,
                intermediate_scores={
                    str(k): v for k, v in sorted(self.intermediate_scores.items())
                },
                max_resource_reached=self.max_resource_reached,
                metrics=dict(self.metrics),
                returns_summary=ret_summary,
                error=self.error,
            )
        )


class TrialContext:
    """Contexto pasado al objetivo para reportar puntuaciones intermedias y aplicar poda ASHA."""

    def __init__(
        self,
        tuner: TPEHyperbandAutoTuner,
        *,
        trial_id: int,
        rungs: tuple[int, ...],
        seed: int,
    ) -> None:
        self._tuner = tuner
        self.trial_id = int(trial_id)
        self.rungs = rungs
        self.max_resource = int(rungs[-1])
        self.seed = int(seed)
        self.intermediate_scores: dict[int, float] = {}
        self.max_resource_reached: int = 0

    def report(self, score: float, *, resource: int) -> bool:
        """Registra la puntuación en `resource` y devuelve `True` si ASHA ordena podar."""
        res_i = int(resource)
        val = float(score)
        self.intermediate_scores[res_i] = val
        self.max_resource_reached = max(self.max_resource_reached, res_i)
        return self._tuner._evaluate_asha_prune(self.trial_id, res_i, val)

    def check_prune(self, score: float, *, resource: int) -> None:
        """Registra la puntuación y lanza `TrialPruned` si ASHA descarta el ensayo."""
        if self.report(score, resource=resource):
            raise TrialPruned(
                f"ASHA pruned trial {self.trial_id} at resource={resource} (score={score:.6g})",
                resource=resource,
            )


def compute_multiplicity_haircut(
    scores: Sequence[float | None],
    *,
    n_total_trials: int,
) -> dict[str, float]:
    """Recorte de multiplicidad Bailey & López de Prado sobre todas las pruebas intentadas."""
    finite = np.asarray(
        [float(s) for s in scores if s is not None and math.isfinite(float(s))],
        dtype=np.float64,
    )
    if len(finite) == 0:
        return dict(
            best_score=float("nan"),
            score_std=0.0,
            expected_max_offset=0.0,
            haircut_score=float("nan"),
            n_total_trials=int(max(n_total_trials, 0)),
        )
    best_s = float(np.max(finite))
    n_eff = max(int(n_total_trials), len(finite), 1)
    score_std = float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0
    if n_eff <= 1 or score_std <= 1e-12:
        offset = 0.0
    else:
        z1 = float(stats.norm.ppf(1.0 - 1.0 / float(n_eff)))
        z2 = float(stats.norm.ppf(1.0 - 1.0 / (float(n_eff) * math.e)))
        offset = max(0.0, score_std * ((1.0 - EULER_GAMMA) * z1 + EULER_GAMMA * z2))
    return dict(
        best_score=best_s,
        score_std=score_std,
        expected_max_offset=offset,
        haircut_score=best_s - offset,
        n_total_trials=int(n_eff),
    )


def _extract_objective_score(raw: Any) -> float:
    if isinstance(raw, Mapping):
        return float(raw["score"])
    return float(raw)


def check_parameter_sensitivity_cliff(
    space: SearchSpaceSpec,
    center_params: Mapping[str, Any],
    objective_fn: Callable[[dict[str, Any]], Any],
    *,
    delta_frac: float = 0.15,
    max_drop_ratio: float = 0.35,
    max_mean_drop_ratio: float = 0.25,
    retention_ratio: float = 0.50,
    min_robustness_fraction: float = 0.60,
    boundary_tol: float = 0.05,
    include_joint_diagonals: bool = True,
) -> dict[str, Any]:
    """Auditoría de acantilados de sensibilidad (`ml4t-sensitivity-analysis`).

    Perturba cada parámetro individualmente (`±delta_frac` en espacio normalizado) y pares
    simultáneos en diagonal para distinguir mesetas amplias de picos frágiles sobreajustados.
    """
    center = space.validate_params(center_params)
    center_score = _extract_objective_score(objective_fn(dict(center)))
    scale_denom = max(abs(center_score), 1e-8)

    per_param: dict[str, dict[str, Any]] = {}
    all_neighbor_scores: list[float] = []
    cliff_params: list[str] = []
    boundary_params: list[str] = []
    numeric_specs: list[ParamSpec] = []

    for spec in space.params:
        neighbors: list[dict[str, Any]] = []
        if spec.kind == "categorical":
            assert spec.choices is not None
            idx = spec.choices.index(center[spec.name])
            for adj in (idx - 1, idx + 1):
                if 0 <= adj < len(spec.choices):
                    cand = dict(center)
                    cand[spec.name] = spec.choices[adj]
                    neighbors.append(cand)
        else:
            numeric_specs.append(spec)
            u0 = spec.to_unit(center[spec.name])
            if u0 <= boundary_tol or u0 >= 1.0 - boundary_tol:
                boundary_params.append(spec.name)
            seen_vals: set[Any] = {center[spec.name]}
            for sign in (-1.0, 1.0):
                u_cand = float(min(max(u0 + sign * delta_frac, 0.0), 1.0))
                val_cand = spec.from_unit(u_cand)
                if spec.kind == "int" and val_cand == center[spec.name]:
                    assert spec.low is not None and spec.high is not None
                    step_val = int(center[spec.name]) + (1 if sign > 0 else -1)
                    if int(round(spec.low)) <= step_val <= int(round(spec.high)):
                        val_cand = step_val
                if val_cand not in seen_vals:
                    seen_vals.add(val_cand)
                    cand = dict(center)
                    cand[spec.name] = val_cand
                    neighbors.append(cand)

        p_scores: list[float] = []
        for cand in neighbors:
            try:
                sc = _extract_objective_score(objective_fn(cand))
            except Exception:
                sc = float("nan")
            p_scores.append(sc)
            all_neighbor_scores.append(sc)

        finite_p = [s for s in p_scores if math.isfinite(s)]
        has_nonfinite = len(finite_p) < len(p_scores)
        if finite_p:
            min_p = float(min(finite_p))
            mean_p = float(np.mean(finite_p))
            max_drop = max(0.0, center_score - min_p)
            rel_drop = max_drop / scale_denom
        else:
            min_p = float("nan")
            mean_p = float("nan")
            max_drop = float("inf")
            rel_drop = float("inf")

        is_param_cliff = bool(has_nonfinite or rel_drop > max_drop_ratio)
        if is_param_cliff:
            cliff_params.append(spec.name)

        per_param[spec.name] = dict(
            center_value=center[spec.name],
            neighbor_scores=p_scores,
            min_neighbor_score=min_p,
            mean_neighbor_score=mean_p,
            max_relative_drop=rel_drop,
            is_cliff=is_param_cliff,
        )

    # Perturbaciones conjuntas de 2 parámetros (regla ml4t-sensitivity-analysis: no solo OAT)
    joint_scores: list[float] = []
    if include_joint_diagonals and len(numeric_specs) >= 2:
        n_diag = min(len(numeric_specs), 4)
        for i in range(n_diag):
            for j in range(i + 1, n_diag):
                sp_i, sp_j = numeric_specs[i], numeric_specs[j]
                ui, uj = sp_i.to_unit(center[sp_i.name]), sp_j.to_unit(center[sp_j.name])
                for sign in (-1.0, 1.0):
                    cand = dict(center)
                    cand[sp_i.name] = sp_i.from_unit(
                        min(max(ui + sign * delta_frac, 0.0), 1.0)
                    )
                    cand[sp_j.name] = sp_j.from_unit(
                        min(max(uj + sign * delta_frac, 0.0), 1.0)
                    )
                    same_i = cand[sp_i.name] == center[sp_i.name]
                    same_j = cand[sp_j.name] == center[sp_j.name]
                    if same_i and same_j:
                        continue
                    try:
                        sc = _extract_objective_score(objective_fn(cand))
                    except Exception:
                        sc = float("nan")
                    joint_scores.append(sc)
                    all_neighbor_scores.append(sc)

    finite_all = np.asarray(
        [s for s in all_neighbor_scores if math.isfinite(s)], dtype=np.float64
    )
    if len(finite_all) > 0 and math.isfinite(center_score):
        mean_nb = float(np.mean(finite_all))
        min_nb = float(np.min(finite_all))
        std_nb = float(np.std(finite_all, ddof=1)) if len(finite_all) > 1 else 0.0
        max_rel_drop = max(0.0, center_score - min_nb) / scale_denom
        mean_rel_drop = max(0.0, center_score - mean_nb) / scale_denom
        if center_score > 0.0:
            threshold = retention_ratio * center_score
        else:
            threshold = center_score - abs(center_score) * (1.0 - retention_ratio)
        robustness_fraction = float(np.sum(finite_all >= threshold)) / float(
            max(len(all_neighbor_scores), 1)
        )
        plateau_score = 0.5 * center_score + 0.5 * mean_nb
    else:
        mean_nb = float("nan")
        min_nb = float("nan")
        std_nb = 0.0
        max_rel_drop = float("inf")
        mean_rel_drop = float("inf")
        robustness_fraction = 0.0
        plateau_score = float("-inf")

    is_cliff = bool(
        (not math.isfinite(center_score))
        or len(cliff_params) > 0
        or mean_rel_drop > max_mean_drop_ratio
        or robustness_fraction < min_robustness_fraction
    )

    return dict(
        center_params=center,
        center_score=center_score,
        n_neighbors=len(all_neighbor_scores),
        mean_neighbor_score=mean_nb,
        min_neighbor_score=min_nb,
        neighbor_std=std_nb,
        max_relative_drop=max_rel_drop,
        mean_relative_drop=mean_rel_drop,
        robustness_fraction=robustness_fraction,
        plateau_score=plateau_score,
        cliff_params=cliff_params,
        boundary_params=boundary_params,
        is_cliff=is_cliff,
        joint_scores=joint_scores,
        per_param=per_param,
    )


def _trunc_norm_pdf_1d(x: np.ndarray, centers: np.ndarray, bw: float) -> np.ndarray:
    """Matriz de densidades gaussianas truncadas en `[0, 1]` de forma `[K_cand, M_centers]`."""
    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    z_hi = 0.5 * (1.0 + np.vectorize(math.erf)((1.0 - centers) / bw * inv_sqrt2))
    z_lo = 0.5 * (1.0 + np.vectorize(math.erf)((0.0 - centers) / bw * inv_sqrt2))
    norm_mass = np.maximum(z_hi - z_lo, 1e-6)
    diff = (x[:, None] - centers[None, :]) / bw
    pdf = np.exp(-0.5 * diff * diff) / (bw * math.sqrt(2.0 * math.pi))
    return pdf / norm_mass[None, :]


def _sample_trunc_norm_scalar(rng: np.random.Generator, mu: float, bw: float) -> float:
    for _ in range(8):
        cand = float(rng.normal(mu, bw))
        if 0.0 <= cand <= 1.0:
            return cand
    return float(min(max(mu, 0.0), 1.0))


def _kde_bandwidth(values: np.ndarray, *, n_dims: int = 1, consider_prior: bool = True) -> float:
    """Ancho de banda Scott multivariado con anclaje al prior (evita colapso prematuro)."""
    if len(values) < 2:
        return 0.25
    # Incluir punto medio del prior (0.5) al estilo Optuna `consider_prior=True` evita que
    # un clúster inicial estrecho colapse el ancho de banda antes de explorar el dominio.
    pts = np.append(values, 0.5) if consider_prior else values
    std = float(np.std(pts, ddof=1))
    scott_exp = -1.0 / float(max(n_dims, 1) + 4)
    bw = 1.06 * max(std, 0.08) * (float(len(values)) ** scott_exp)
    # Suelo adaptativo suave (más amplio al inicio, más fino conforme crecen observaciones)
    floor_bw = max(0.08, 0.22 / (1.0 + 0.08 * float(len(values))))
    return float(min(max(bw, floor_bw), 0.40))


class TPEHyperbandAutoTuner:
    """Auto-tuner TPE determinista (cero Optuna) + poda ASHA + auditoría DSR de multiplicidad."""

    def __init__(
        self,
        space: SearchSpaceSpec,
        *,
        min_resource: int = 1,
        max_resource: int = 9,
        reduction_factor: int = 3,
        n_startup_trials: int = 8,
        tpe_gamma: float = 0.25,
        n_ei_candidates: int = 48,
        exploration_beta: float = 1.5,
        min_trials_for_pruning: int | None = None,
        seed: int = 42,
    ) -> None:
        if min_resource < 1 or max_resource < min_resource:
            raise ValueError("Se requiere 1 <= min_resource <= max_resource")
        if reduction_factor < 2:
            raise ValueError("reduction_factor debe ser >= 2")
        if not (0.05 <= tpe_gamma <= 0.5):
            raise ValueError("tpe_gamma debe pertenecer a [0.05, 0.5]")
        if n_ei_candidates < 4:
            raise ValueError("n_ei_candidates debe ser >= 4")
        if exploration_beta < 0.0:
            raise ValueError("exploration_beta debe ser >= 0")

        self.space = space
        self.min_resource = int(min_resource)
        self.max_resource = int(max_resource)
        self.reduction_factor = int(reduction_factor)
        self.n_startup_trials = int(max(n_startup_trials, 2))
        self.tpe_gamma = float(tpe_gamma)
        self.n_ei_candidates = int(n_ei_candidates)
        self.exploration_beta = float(exploration_beta)
        self.min_trials_for_pruning = int(
            min_trials_for_pruning if min_trials_for_pruning is not None else reduction_factor
        )
        self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed)

        rungs: list[int] = []
        r = self.min_resource
        while r < self.max_resource:
            rungs.append(r)
            r *= self.reduction_factor
        if not rungs or rungs[-1] != self.max_resource:
            rungs.append(self.max_resource)
        self.rungs: tuple[int, ...] = tuple(rungs)
        self._rung_set: frozenset[int] = frozenset(self.rungs)

        self._ledger: list[TrialRecord] = []
        self._rung_scores: dict[int, list[float]] = {rk: [] for rk in self.rungs}
        self._startup_queue: list[dict[str, Any]] = self._build_stratified_startup()

    def _build_stratified_startup(self) -> list[dict[str, Any]]:
        """Genera un diseño inicial estratificado (LHS + balanceo categórico determinista)."""
        n = self.n_startup_trials
        columns: dict[str, list[Any]] = {}
        for spec in self.space.params:
            if spec.kind == "categorical":
                assert spec.choices is not None
                k = len(spec.choices)
                vals = [spec.choices[i % k] for i in range(n)]
                order = self._rng.permutation(n)
                columns[spec.name] = [vals[int(i)] for i in order]
            else:
                # Latin Hypercube estratificado en [0, 1]
                bins = (np.arange(n, dtype=np.float64) + self._rng.uniform(0.0, 1.0, size=n)) / n
                order = self._rng.permutation(n)
                columns[spec.name] = [spec.from_unit(float(bins[int(i)])) for i in order]
        return [{spec.name: columns[spec.name][i] for spec in self.space.params} for i in range(n)]

    @property
    def n_total_trials(self) -> int:
        """Número total honesto de ensayos (completados + podados + fallidos)."""
        return len(self._ledger)

    @property
    def ledger(self) -> tuple[TrialRecord, ...]:
        """Vista inmutable de todos los registros de ensayos."""
        return tuple(self._ledger)

    def get_trial(self, trial_id: int) -> TrialRecord | None:
        if 0 <= trial_id < len(self._ledger):
            return self._ledger[trial_id]
        return None

    def _evaluate_asha_prune(self, trial_id: int, resource: int, score: float) -> bool:
        del trial_id
        if not math.isfinite(score):
            return True
        # En ASHA solo se evalúa poda en los escalones geométricos declarados (`self.rungs`)
        if resource not in self._rung_set:
            return False
        if resource >= self.max_resource:
            self._rung_scores.setdefault(resource, []).append(score)
            return False
        bucket = self._rung_scores.setdefault(resource, [])
        bucket.append(score)
        if len(bucket) < self.min_trials_for_pruning:
            return False
        # Promover únicamente el 1 / reduction_factor superior del escalón
        keep_quantile = 1.0 - (1.0 / float(self.reduction_factor))
        threshold = float(np.quantile(np.asarray(bucket, dtype=np.float64), keep_quantile))
        return bool(score < threshold)

    def suggest(self) -> dict[str, Any]:
        """Propone la siguiente configuración mediante TPE multivariado determinista (BOHB)."""
        trial_idx = len(self._ledger)
        if trial_idx < len(self._startup_queue):
            return dict(self._startup_queue[trial_idx])

        # Recolectar ensayos con puntuación finita dando prioridad al presupuesto completo
        completed: list[tuple[float, dict[str, Any]]] = []
        pruned_or_partial: list[tuple[float, dict[str, Any]]] = []
        for rec in self._ledger:
            if rec.status == "completed" and rec.score is not None and math.isfinite(rec.score):
                completed.append((float(rec.score), rec.params))
            else:
                s = rec.score
                if s is None and rec.intermediate_scores:
                    s = rec.intermediate_scores[max(rec.intermediate_scores.keys())]
                if s is not None and math.isfinite(float(s)):
                    pruned_or_partial.append((float(s), rec.params))

        total_obs = len(completed) + len(pruned_or_partial)
        if total_obs < max(self.n_startup_trials, 4):
            return self.space.sample_random(self._rng)

        # Ordenar completados arriba y podados en el grupo inferior (apple-to-apple en max_resource)
        completed.sort(key=lambda item: item[0], reverse=True)
        pruned_or_partial.sort(key=lambda item: item[0], reverse=True)
        ranked = completed + pruned_or_partial

        n_good = max(2, min(total_obs - 2, int(math.ceil(self.tpe_gamma * total_obs))))
        good_params = [p for _, p in ranked[:n_good]]
        bad_params = [p for _, p in ranked[n_good:]]

        n_cand = max(self.n_ei_candidates, 48)
        prior_w = 1.0
        cat_tau = 0.35  # probabilidad de exploración suave entre categorías en el núcleo conjunto

        # Precalcular anchos de banda por dimensión numérica
        num_specs = [sp for sp in self.space.params if sp.kind != "categorical"]
        cat_specs = [sp for sp in self.space.params if sp.kind == "categorical"]

        good_num = {
            sp.name: np.asarray([sp.to_unit(p[sp.name]) for p in good_params], dtype=np.float64)
            for sp in num_specs
        }
        bad_num = {
            sp.name: np.asarray([sp.to_unit(p[sp.name]) for p in bad_params], dtype=np.float64)
            for sp in num_specs
        }
        n_num = len(num_specs)
        bw_good = {
            sp.name: _kde_bandwidth(good_num[sp.name], n_dims=n_num, consider_prior=True)
            for sp in num_specs
        }
        bw_bad = {
            sp.name: _kde_bandwidth(bad_num[sp.name], n_dims=n_num, consider_prior=True)
            for sp in num_specs
        }

        # Muestrear candidatos desde la mezcla multivariada l(z) + prior uniforme
        p_prior = prior_w / (float(n_good) + prior_w)
        cand_dicts: list[dict[str, Any]] = []
        cand_units: dict[str, np.ndarray] = {
            sp.name: np.empty(n_cand, dtype=np.float64) for sp in num_specs
        }
        cand_cats: dict[str, list[Any]] = {sp.name: [] for sp in cat_specs}

        for c_i in range(n_cand):
            cand: dict[str, Any] = {}
            if float(self._rng.uniform(0.0, 1.0)) < p_prior:
                for sp in num_specs:
                    uv = float(self._rng.uniform(0.0, 1.0))
                    cand_units[sp.name][c_i] = uv
                    cand[sp.name] = sp.from_unit(uv)
                for sp in cat_specs:
                    assert sp.choices is not None
                    cv = sp.choices[int(self._rng.integers(0, len(sp.choices)))]
                    cand_cats[sp.name].append(cv)
                    cand[sp.name] = cv
            else:
                src_i = int(self._rng.integers(0, n_good))
                src_p = good_params[src_i]
                for sp in num_specs:
                    uv = _sample_trunc_norm_scalar(
                        self._rng, float(good_num[sp.name][src_i]), bw_good[sp.name]
                    )
                    cand_units[sp.name][c_i] = uv
                    cand[sp.name] = sp.from_unit(uv)
                for sp in cat_specs:
                    assert sp.choices is not None
                    if float(self._rng.uniform(0.0, 1.0)) < cat_tau:
                        cv = sp.choices[int(self._rng.integers(0, len(sp.choices)))]
                    else:
                        cv = src_p[sp.name]
                    cand_cats[sp.name].append(cv)
                    cand[sp.name] = cv
            cand_dicts.append(cand)

        # Evaluar log l(z) - log g(z) usando núcleo producto multivariado
        prior_density = 1.0
        for sp in cat_specs:
            assert sp.choices is not None
            prior_density *= 1.0 / float(len(sp.choices))

        k_good = np.ones((n_cand, len(good_params)), dtype=np.float64)
        k_bad = np.ones((n_cand, len(bad_params)), dtype=np.float64)
        for sp in num_specs:
            k_good *= _trunc_norm_pdf_1d(cand_units[sp.name], good_num[sp.name], bw_good[sp.name])
            k_bad *= _trunc_norm_pdf_1d(cand_units[sp.name], bad_num[sp.name], bw_bad[sp.name])
        for sp in cat_specs:
            assert sp.choices is not None
            k_ch = float(len(sp.choices))
            p_same = 1.0 - cat_tau + cat_tau / k_ch
            p_diff = cat_tau / k_ch
            c_arr = np.asarray(cand_cats[sp.name], dtype=object)[:, None]
            g_arr = np.asarray([p[sp.name] for p in good_params], dtype=object)[None, :]
            b_arr = np.asarray([p[sp.name] for p in bad_params], dtype=object)[None, :]
            k_good *= np.where(c_arr == g_arr, p_same, p_diff)
            k_bad *= np.where(c_arr == b_arr, p_same, p_diff)

        sum_good = np.sum(k_good, axis=1)
        sum_bad = np.sum(k_bad, axis=1)
        l_denom = float(len(good_params)) + prior_w
        g_denom = float(len(bad_params)) + prior_w
        l_pdf = (prior_w * prior_density + sum_good) / l_denom
        g_pdf = (prior_w * prior_density + sum_bad) / g_denom
        log_ei = np.log(np.maximum(l_pdf, 1e-18)) - np.log(np.maximum(g_pdf, 1e-18))
        # Bono UCB de densidad epistémica: evita bloqueo prematuro en ramas categóricas/locales
        # poco visitadas usando el conteo efectivo de vecinos ya calculado en el núcleo KDE.
        if self.exploration_beta > 0.0:
            local_n = sum_good + sum_bad
            log_ei = log_ei + self.exploration_beta / np.sqrt(1.0 + local_n)

        best_idx = int(np.argmax(log_ei))
        return cand_dicts[best_idx]

    def record_external_trial(
        self,
        params: Mapping[str, Any],
        *,
        score: float | None,
        status: Literal["completed", "pruned", "failed"] = "completed",
        returns: np.ndarray | None = None,
        metrics: Mapping[str, Any] | None = None,
        error: str | None = None,
    ) -> TrialRecord:
        """Registra manualmente un ensayo evaluado externamente (mantiene conteo honesto)."""
        valid_p = self.space.validate_params(params)
        rec = TrialRecord(
            trial_id=len(self._ledger),
            params=valid_p,
            status=status,
            score=float(score) if score is not None else None,
            max_resource_reached=self.max_resource if status == "completed" else 0,
            metrics=dict(metrics) if metrics else {},
            returns=np.asarray(returns, dtype=np.float64) if returns is not None else None,
            error=error,
        )
        self._ledger.append(rec)
        return rec

    def optimize(
        self,
        objective_fn: Callable[..., Any],
        *,
        n_trials: int,
        train_years: Sequence[int] | None = None,
        val_years: Sequence[int] | None = None,
    ) -> dict[str, Any]:
        """Ejecuta `n_trials` de búsqueda TPE+ASHA validando el sello `2024+` antes de iniciar."""
        if n_trials < 1:
            raise ValueError("n_trials debe ser >= 1")
        if train_years is not None and val_years is not None:
            validate_purged_validation_split(
                train_years,
                val_years,
                sealed_test_start_year=self.space.sealed_test_start_year,
            )
        elif val_years is not None:
            validate_no_sealed_leakage(
                val_years, sealed_test_start_year=self.space.sealed_test_start_year
            )
        elif train_years is not None:
            validate_no_sealed_leakage(
                train_years, sealed_test_start_year=self.space.sealed_test_start_year
            )

        sig = inspect.signature(objective_fn)
        accepts_ctx = len(sig.parameters) >= 2

        for _ in range(n_trials):
            trial_id = len(self._ledger)
            params = self.suggest()
            ctx = TrialContext(
                self,
                trial_id=trial_id,
                rungs=self.rungs,
                seed=self.seed + trial_id + 1,
            )
            try:
                raw = objective_fn(dict(params), ctx) if accepts_ctx else objective_fn(dict(params))
                if isinstance(raw, Mapping):
                    score = float(raw["score"])
                    rets_raw = raw.get("returns")
                    rets = (
                        np.asarray(rets_raw, dtype=np.float64)
                        if rets_raw is not None
                        else None
                    )
                    metrics = dict(raw.get("metrics") or {})
                else:
                    score = float(raw)
                    rets = None
                    metrics = {}
                status: Literal["completed", "pruned", "failed"] = (
                    "completed" if math.isfinite(score) else "failed"
                )
                rec = TrialRecord(
                    trial_id=trial_id,
                    params=params,
                    status=status,
                    score=score if math.isfinite(score) else None,
                    intermediate_scores=dict(ctx.intermediate_scores),
                    max_resource_reached=max(ctx.max_resource_reached, self.max_resource),
                    metrics=metrics,
                    returns=rets,
                    error=None if math.isfinite(score) else "non_finite_objective_score",
                )
            except TrialPruned as exc:
                last_score = (
                    ctx.intermediate_scores[max(ctx.intermediate_scores)]
                    if ctx.intermediate_scores
                    else None
                )
                valid_last = last_score is not None and math.isfinite(last_score)
                rec = TrialRecord(
                    trial_id=trial_id,
                    params=params,
                    status="pruned",
                    score=last_score if valid_last else None,
                    intermediate_scores=dict(ctx.intermediate_scores),
                    max_resource_reached=ctx.max_resource_reached,
                    error=str(exc),
                )
            except Exception as exc:
                rec = TrialRecord(
                    trial_id=trial_id,
                    params=params,
                    status="failed",
                    score=None,
                    intermediate_scores=dict(ctx.intermediate_scores),
                    max_resource_reached=ctx.max_resource_reached,
                    error=f"{type(exc).__name__}: {exc}",
                )
            self._ledger.append(rec)

        return self.summary()

    def _best_trial(self) -> TrialRecord | None:
        completed = [
            r for r in self._ledger if r.status == "completed" and r.score is not None
        ]
        pool = completed if completed else [r for r in self._ledger if r.score is not None]
        if not pool:
            return None
        return max(pool, key=lambda r: float(r.score))  # type: ignore[arg-type]

    def summary(self) -> dict[str, Any]:
        """Genera el informe determinista de multiplicidad, DSR y mejor configuración."""
        n_total = len(self._ledger)
        n_completed = sum(1 for r in self._ledger if r.status == "completed")
        n_pruned = sum(1 for r in self._ledger if r.status == "pruned")
        n_failed = sum(1 for r in self._ledger if r.status == "failed")

        all_scores = [r.score for r in self._ledger]
        haircut = compute_multiplicity_haircut(all_scores, n_total_trials=n_total)
        best = self._best_trial()

        if best is not None and best.returns is not None:
            emp_sharpes: list[float] = []
            for rec in self._ledger:
                if rec.returns is not None:
                    r_arr = np.asarray(rec.returns, dtype=np.float64)
                    r_arr = r_arr[np.isfinite(r_arr)]
                    if len(r_arr) >= 5:
                        sd_r = float(np.std(r_arr, ddof=1))
                        if sd_r > 1e-12:
                            emp_sharpes.append(float(np.mean(r_arr) / sd_r))
            dsr_rep = deflated_sharpe_ratio(
                best.returns,
                n_trials=max(n_total, 1),
                trial_sharpes=emp_sharpes if len(emp_sharpes) >= 2 else None,
                trial_sharpes_annualized=False,
            )
            dsr_report: dict[str, Any] = dict(dsr_rep, from_returns=True, n_trials=n_total)
        else:
            dsr_report = dict(
                sr_annual=0.0,
                sr_threshold=0.0,
                dsr=0.5,
                from_returns=False,
                n_trials=n_total,
            )

        best_raw = float(best.score) if (best is not None and best.score is not None) else None
        audit_rows = [r.to_audit_dict() for r in self._ledger]
        return dict(
            n_total_trials=n_total,
            n_completed=n_completed,
            n_pruned=n_pruned,
            n_failed=n_failed,
            rungs=list(self.rungs),
            best_trial_id=best.trial_id if best is not None else None,
            best_params=dict(best.params) if best is not None else None,
            best_raw_score=best_raw,
            score_std=haircut["score_std"],
            multiplicity_haircut_offset=haircut["expected_max_offset"],
            haircut_score=haircut["haircut_score"],
            dsr_report=dsr_report,
            ledger_sha256=_canonical_sha256(audit_rows),
        )

    def select_robust_best(
        self,
        objective_fn: Callable[[dict[str, Any]], Any],
        *,
        top_k_candidates: int = 5,
        delta_frac: float = 0.15,
        max_drop_ratio: float = 0.35,
        min_robustness_fraction: float = 0.60,
    ) -> dict[str, Any]:
        """Evalúa los mejores candidatos contra acantilados y rechaza picos frágiles."""
        completed = [
            r for r in self._ledger if r.status == "completed" and r.score is not None
        ]
        if not completed:
            raise ValueError("No existen ensayos completados para selección robusta")
        ranked = sorted(completed, key=lambda r: float(r.score), reverse=True)  # type: ignore[arg-type]
        candidates = ranked[: max(1, int(top_k_candidates))]

        evaluations: list[tuple[TrialRecord, dict[str, Any]]] = []
        rejected_cliff_ids: list[int] = []
        for rec in candidates:
            sens = check_parameter_sensitivity_cliff(
                self.space,
                rec.params,
                objective_fn,
                delta_frac=delta_frac,
                max_drop_ratio=max_drop_ratio,
                min_robustness_fraction=min_robustness_fraction,
            )
            evaluations.append((rec, sens))
            if sens["is_cliff"]:
                rejected_cliff_ids.append(rec.trial_id)

        non_cliff = [(r, s) for r, s in evaluations if not s["is_cliff"]]
        pool = non_cliff if non_cliff else evaluations
        chosen_rec, chosen_sens = max(
            pool, key=lambda item: float(item[1]["plateau_score"])
        )
        return dict(
            selected_trial_id=chosen_rec.trial_id,
            selected_params=dict(chosen_rec.params),
            selected_raw_score=float(chosen_rec.score),  # type: ignore[arg-type]
            selected_sensitivity=chosen_sens,
            rejected_cliff_trial_ids=rejected_cliff_ids,
            all_candidates_cliff=len(non_cliff) == 0,
            n_candidates_checked=len(evaluations),
        )

    def write_audit_ledger(
        self,
        path: str | Path,
        *,
        compress: bool | None = None,
    ) -> Path:
        """Escribe el ledger determinista en JSONL o `.jsonl.gz` atómico sin pérdida.

        Si `compress` es `None`, detecta automáticamente compresión gzip por sufijo `.gz`.
        Fija `filename=""` y `mtime=0` para que el archivo `.jsonl.gz` sea 100 % determinista
        byte-a-byte.
        """
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        use_gz = out_path.suffix == ".gz" if compress is None else bool(compress)
        tmp = out_path.with_name(f".{out_path.name}.tmp.{os.getpid()}")
        if use_gz:
            with (
                tmp.open("wb") as raw_fh,
                gzip.GzipFile(
                    filename="", mode="wb", fileobj=raw_fh, compresslevel=6, mtime=0
                ) as gz_fh,
            ):
                for rec in self._ledger:
                    line = (
                        json.dumps(
                            rec.to_audit_dict(), sort_keys=True, separators=(",", ":")
                        )
                        + "\n"
                    )
                    gz_fh.write(line.encode("utf-8"))
        else:
            with tmp.open("w", encoding="utf-8") as fh:
                for rec in self._ledger:
                    line = json.dumps(
                        rec.to_audit_dict(), sort_keys=True, separators=(",", ":")
                    )
                    fh.write(line + "\n")
        os.replace(tmp, out_path)
        return out_path


def append_audit_record(
    path: str | Path,
    record: TrialRecord | Mapping[str, Any],
    *,
    compress: bool | None = None,
) -> Path:
    """Añade un registro al ledger JSONL o `.jsonl.gz` (miembro gzip RFC 1952 concatenado)."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = record.to_audit_dict() if isinstance(record, TrialRecord) else _json_sanitize(record)
    line = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"

    if compress is None:
        if out_path.is_file() and out_path.stat().st_size >= 2:
            with out_path.open("rb") as rf:
                use_gz = rf.read(2) == b"\x1f\x8b"
        else:
            use_gz = out_path.suffix == ".gz"
    else:
        use_gz = bool(compress)

    if use_gz:
        with (
            out_path.open("ab") as raw_fh,
            gzip.GzipFile(
                filename="", mode="wb", fileobj=raw_fh, compresslevel=6, mtime=0
            ) as gz_fh,
        ):
            gz_fh.write(line.encode("utf-8"))
    else:
        with out_path.open("a", encoding="utf-8") as fh:
            fh.write(line)
    return out_path


def read_audit_ledger(path: str | Path) -> list[dict[str, Any]]:
    """Lee de forma transparente un ledger `.jsonl` o `.jsonl.gz` (detectando bytes mágicos)."""
    p = Path(path)
    if not p.is_file():
        return []
    with p.open("rb") as raw_fh:
        magic = raw_fh.read(2)
    is_gz = magic == b"\x1f\x8b"
    rows: list[dict[str, Any]] = []
    opener = gzip.open(p, "rt", encoding="utf-8") if is_gz else p.open("r", encoding="utf-8")
    with opener as fh:
        for line_no, raw in enumerate(fh, start=1):
            s = raw.strip()
            if not s:
                continue
            try:
                rows.append(json.loads(s))
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSONL corrupto en {p} línea {line_no}: {exc}") from exc
    return rows


# ---------------------------------------------------------------------------
# 2. Controlador de ajuste óptimo (sin sobreajuste ni subajuste) + Top-3 Soup
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OptimalFitConfig:
    """Contrato inmutable de parada adaptativa suavizada y diagnóstico dual de ajuste."""

    smoothing: int = 3
    patience: int = 5
    min_delta: float = 1e-4
    lr_decay_factor: float = 0.5
    max_lr_decays: int = 2
    top_k_soup: int = 3
    # Umbrales de subajuste (underfitting)
    underfit_train_ic_threshold: float = 0.015
    underfit_update_ratio_threshold: float = 5e-5
    auto_adjust_underfitting: bool = False
    underfit_lr_boost: float = 1.5
    max_underfit_boosts: int = 1
    # Umbrales de sobreajuste (overfitting)
    overfit_gap_threshold: float = 0.03
    overfit_crps_rise_ratio: float = 0.05
    overfit_wd_bump_patience: int = 2
    wd_bump_factor: float = 1.5
    max_weight_decay: float = 5e-2
    auto_bump_weight_decay: bool = True
    auto_restore_soup_on_stop: bool = True

    def __post_init__(self) -> None:
        if self.smoothing < 1:
            raise ValueError("smoothing debe ser >= 1")
        if self.patience < 1:
            raise ValueError("patience debe ser >= 1")
        if not (0.0 < self.lr_decay_factor < 1.0):
            raise ValueError("lr_decay_factor debe pertenecer a (0, 1)")
        if self.max_lr_decays < 0:
            raise ValueError("max_lr_decays debe ser >= 0")
        if self.top_k_soup < 1:
            raise ValueError("top_k_soup debe ser >= 1")
        if self.underfit_update_ratio_threshold <= 0.0:
            raise ValueError("underfit_update_ratio_threshold debe ser > 0")
        if self.overfit_gap_threshold <= 0.0:
            raise ValueError("overfit_gap_threshold debe ser > 0")
        if self.wd_bump_factor < 1.0:
            raise ValueError("wd_bump_factor debe ser >= 1.0")


@dataclass(frozen=True)
class FitEpochDecision:
    """Decisión estructurada por época emitida por `OptimalFitController`."""

    epoch: int
    regime: FitRegime
    train_ic: float
    val_ic: float
    smoothed_val_ic: float
    ic_gap: float
    val_crps: float | None
    update_weight_ratio: float | None
    improved: bool
    epochs_since_best: int
    lr_decayed: bool
    lr_decays_used: int
    wd_bumped: bool
    should_stop: bool
    soup_restored: bool
    recommended_actions: tuple[str, ...]
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _snapshot_state_dict_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


class OptimalFitController:
    """Controlador de ajuste óptimo: parada suavizada, diagnóstico dual y Top-3 Model Soup."""

    def __init__(self, config: OptimalFitConfig | None = None) -> None:
        self.config = config or OptimalFitConfig()
        self._val_ic_history: list[float] = []
        self._train_ic_history: list[float] = []
        self._train_loss_history: list[float] = []
        self._decisions: list[FitEpochDecision] = []

        self.best_smoothed_val_ic: float = float("-inf")
        self.best_val_ic: float = float("-inf")
        self.best_val_crps: float | None = None
        self.epochs_since_best: int = 0
        self.lr_decays_used: int = 0
        self.underfit_boosts_used: int = 0
        self.overfit_streak: int = 0
        self.wd_bumps_used: int = 0

        # Lista ordenada descendentemente por val_ic: [(val_ic, epoch, state_dict_cpu), ...]
        self._top_checkpoints: list[tuple[float, int, dict[str, torch.Tensor]]] = []
        self._param_names: set[str] | None = None

    @property
    def history(self) -> tuple[FitEpochDecision, ...]:
        return tuple(self._decisions)

    @property
    def top_checkpoint_epochs(self) -> tuple[int, ...]:
        return tuple(ep for _, ep, _ in self._top_checkpoints)

    def _record_top_checkpoint(self, epoch: int, val_ic: float, model: nn.Module) -> None:
        if not math.isfinite(val_ic):
            return
        self._param_names = {name for name, _ in model.named_parameters()}
        k = self.config.top_k_soup
        if len(self._top_checkpoints) < k or val_ic > self._top_checkpoints[-1][0]:
            snap = _snapshot_state_dict_cpu(model)
            self._top_checkpoints.append((float(val_ic), int(epoch), snap))
            self._top_checkpoints.sort(key=lambda item: (item[0], -item[1]), reverse=True)
            if len(self._top_checkpoints) > k:
                self._top_checkpoints = self._top_checkpoints[:k]

    def build_model_soup_state_dict(self) -> dict[str, torch.Tensor]:
        """Construye el `state_dict` promedio exacto (Top-K Model Soup) solo sobre parámetros.

        Promedia únicamente las claves de `named_parameters()`; todos los buffers (memorias
        episódicas `slot_keys`/`proto_*`, `balance_bias`, contadores, máscaras) se copian
        intactos de la mejor época (#1) para jamás mezclar estados episódicos incompatibles.
        """
        if not self._top_checkpoints:
            raise RuntimeError("No hay checkpoints almacenados en OptimalFitController")
        n = len(self._top_checkpoints)
        best_sd = self._top_checkpoints[0][2]
        if n == 1:
            return {k: v.clone() for k, v in best_sd.items()}

        param_names = self._param_names
        averaged: dict[str, torch.Tensor] = {}
        for key, ref_t in best_sd.items():
            is_param = (key in param_names) if param_names is not None else True
            if is_param and ref_t.dtype.is_floating_point:
                acc = torch.zeros_like(ref_t, dtype=torch.float64)
                for _, _, sd in self._top_checkpoints:
                    acc.add_(sd[key].to(dtype=torch.float64))
                acc.div_(float(n))
                averaged[key] = acc.to(dtype=ref_t.dtype)
            else:
                # Todos los buffers (flotantes episódicos o enteros/bool) provienen de época #1
                averaged[key] = ref_t.clone()
        return averaged

    def restore_top_k_soup(self, model: nn.Module) -> dict[str, torch.Tensor]:
        """Carga el `state_dict` promediado Top-K en `model` y lo devuelve."""
        soup_sd = self.build_model_soup_state_dict()
        model.load_state_dict(soup_sd, strict=True)
        return soup_sd

    def _diagnose_regime(
        self,
        *,
        train_ic: float,
        val_ic: float,
        val_crps: float | None,
        train_loss: float | None,
        update_weight_ratio: float | None,
    ) -> tuple[FitRegime, list[str], list[str]]:
        c = self.config
        reasons: list[str] = []
        actions: list[str] = []
        ic_gap = train_ic - val_ic

        # 1) Comprobar señales de SOBREAJUSTE (overfitting)
        overfit_signals = 0
        if ic_gap >= c.overfit_gap_threshold and (
            val_ic < self.best_val_ic - c.min_delta or ic_gap >= 1.5 * c.overfit_gap_threshold
        ):
            overfit_signals += 1
            reasons.append(f"widening_ic_gap({ic_gap:.4f} >= {c.overfit_gap_threshold:.4f})")

        if (
            val_crps is not None
            and math.isfinite(val_crps)
            and self.best_val_crps is not None
            and val_crps > self.best_val_crps * (1.0 + c.overfit_crps_rise_ratio)
            and train_ic >= val_ic
        ):
            overfit_signals += 1
            reasons.append(
                f"rising_val_crps({val_crps:.5f} > {self.best_val_crps:.5f})"
            )

        if (
            len(self._train_ic_history) >= 2
            and len(self._val_ic_history) >= 2
            and train_ic > self._train_ic_history[-1] + 0.003
            and val_ic < self._val_ic_history[-1] - 0.003
            and ic_gap >= 0.5 * c.overfit_gap_threshold
        ):
            overfit_signals += 1
            reasons.append("diverging_train_up_val_down")

        if overfit_signals > 0:
            actions.extend(
                ["decay_lr", "bump_weight_decay", "early_stop_and_restore_top_k_soup"]
            )
            return "overfitting", reasons, actions

        # 2) Comprobar señales de SUBAJUSTE (underfitting)
        low_train = train_ic < c.underfit_train_ic_threshold
        low_val = val_ic < c.underfit_train_ic_threshold
        tiny_updates = (
            update_weight_ratio is not None
            and math.isfinite(update_weight_ratio)
            and update_weight_ratio < c.underfit_update_ratio_threshold
        )
        loss_stagnant = (
            train_loss is not None
            and math.isfinite(train_loss)
            and len(self._train_loss_history) >= 2
            and train_loss >= self._train_loss_history[-1] - 1e-4
            and train_loss >= self._train_loss_history[-2] - 1e-4
        )
        ic_stagnant_low = (
            len(self._train_ic_history) >= 2
            and low_train
            and all(v < c.underfit_train_ic_threshold for v in self._train_ic_history[-2:])
        )

        if low_train and low_val and (tiny_updates or loss_stagnant or ic_stagnant_low):
            if tiny_updates:
                reasons.append(
                    f"tiny_update_weight_ratio({update_weight_ratio:.2e} < "
                    f"{c.underfit_update_ratio_threshold:.2e})"
                )
            if loss_stagnant:
                reasons.append("stagnant_high_train_loss")
            if ic_stagnant_low:
                reasons.append(f"persistently_low_train_ic({train_ic:.4f})")
            actions.extend(
                ["increase_lr", "relax_modality_dropout", "shrink_and_perturb_warm_restart"]
            )
            return "underfitting", reasons, actions

        reasons.append("balanced_generalization_and_updates")
        return "healthy", reasons, actions

    def step(
        self,
        *,
        epoch: int,
        train_ic: float,
        val_ic: float,
        val_crps: float | None = None,
        train_loss: float | None = None,
        update_weight_ratio: float | None = None,
        model: nn.Module | None = None,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> FitEpochDecision:
        """Evalúa una época, actualiza Top-K Soup, aplica decaimiento/WD y decide parada."""
        c = self.config
        tr_ic = float(train_ic)
        va_ic = float(val_ic)
        crps_v = float(val_crps) if (val_crps is not None and math.isfinite(val_crps)) else None
        loss_v = (
            float(train_loss) if (train_loss is not None and math.isfinite(train_loss)) else None
        )
        upd_v = (
            float(update_weight_ratio)
            if (update_weight_ratio is not None and math.isfinite(update_weight_ratio))
            else None
        )

        # 1) Diagnóstico dual ANTES de mutar best_val_ic / best_val_crps
        regime, reasons, actions = self._diagnose_regime(
            train_ic=tr_ic,
            val_ic=va_ic,
            val_crps=crps_v,
            train_loss=loss_v,
            update_weight_ratio=upd_v,
        )

        # 2) Historial suavizado de val_ic (ventana rodante de tamaño `smoothing`)
        if math.isfinite(va_ic):
            self._val_ic_history.append(va_ic)
            window = self._val_ic_history[-c.smoothing :]
            smoothed = float(sum(window) / len(window))
            if va_ic > self.best_val_ic:
                self.best_val_ic = va_ic
        else:
            smoothed = float("nan")

        if math.isfinite(tr_ic):
            self._train_ic_history.append(tr_ic)
        if loss_v is not None:
            self._train_loss_history.append(loss_v)
        if crps_v is not None and (self.best_val_crps is None or crps_v < self.best_val_crps):
            self.best_val_crps = crps_v

        if model is not None and math.isfinite(va_ic):
            self._record_top_checkpoint(epoch, va_ic, model)

        # 3) Control de mejora sobre `smoothed_val_ic`
        improved = bool(
            math.isfinite(smoothed) and smoothed > self.best_smoothed_val_ic + c.min_delta
        )
        if improved:
            self.best_smoothed_val_ic = smoothed
            self.epochs_since_best = 0
        else:
            self.epochs_since_best += 1

        # 4) Respuesta adaptativa al régimen (subajuste / sobreajuste)
        wd_bumped = False
        if regime == "overfitting":
            self.overfit_streak += 1
            if (
                c.auto_bump_weight_decay
                and optimizer is not None
                and self.overfit_streak >= c.overfit_wd_bump_patience
                and self.overfit_streak % c.overfit_wd_bump_patience == 0
            ):
                for group in optimizer.param_groups:
                    cur_wd = float(group.get("weight_decay", 0.0))
                    if cur_wd > 0.0:
                        new_wd = min(cur_wd * c.wd_bump_factor, c.max_weight_decay)
                        if new_wd > cur_wd:
                            group["weight_decay"] = new_wd
                            wd_bumped = True
                if wd_bumped:
                    self.wd_bumps_used += 1
        else:
            self.overfit_streak = 0

        if (
            regime == "underfitting"
            and c.auto_adjust_underfitting
            and optimizer is not None
            and self.underfit_boosts_used < c.max_underfit_boosts
        ):
            for group in optimizer.param_groups:
                group["lr"] = float(group["lr"]) * c.underfit_lr_boost
            self.underfit_boosts_used += 1

        # 5) Parada adaptativa suavizada: decay de LR `0.5x` hasta `max_lr_decays` veces -> stop
        lr_decayed = False
        should_stop = False
        soup_restored = False
        if self.epochs_since_best >= c.patience:
            if self.lr_decays_used < c.max_lr_decays:
                if optimizer is not None:
                    for group in optimizer.param_groups:
                        group["lr"] = float(group["lr"]) * c.lr_decay_factor
                self.lr_decays_used += 1
                self.epochs_since_best = 0
                lr_decayed = True
            else:
                should_stop = True
                if c.auto_restore_soup_on_stop and model is not None and self._top_checkpoints:
                    self.restore_top_k_soup(model)
                    soup_restored = True

        decision = FitEpochDecision(
            epoch=int(epoch),
            regime=regime,
            train_ic=tr_ic,
            val_ic=va_ic,
            smoothed_val_ic=smoothed,
            ic_gap=tr_ic - va_ic,
            val_crps=crps_v,
            update_weight_ratio=upd_v,
            improved=improved,
            epochs_since_best=self.epochs_since_best,
            lr_decayed=lr_decayed,
            lr_decays_used=self.lr_decays_used,
            wd_bumped=wd_bumped,
            should_stop=should_stop,
            soup_restored=soup_restored,
            recommended_actions=tuple(actions),
            reasons=tuple(reasons),
        )
        self._decisions.append(decision)
        return decision


# ---------------------------------------------------------------------------
# 3. Shrink-and-Perturb (Ash & Adams, NeurIPS) para romper pérdida de plasticidad
# ---------------------------------------------------------------------------


def _stable_calc_dtype(dtype: torch.dtype) -> torch.dtype:
    """Promociona FP16/BF16 a FP32 para estabilidad numérica preservando FP32 y FP64."""
    return torch.float64 if dtype == torch.float64 else torch.float32


@torch.no_grad()
def shrink_and_perturb(
    model: nn.Module,
    *,
    shrink: float = 0.8,
    perturb: float = 0.02,
    seed: int = 42,
    preserve_norm_scales: bool = True,
) -> dict[str, Any]:
    """Aplica Shrink-and-Perturb determinista (`theta <- shrink*theta + perturb*theta_init`).

    Preserva la escala unitaria de capas de normalización (`LayerNorm`/`RMSNorm`/`BatchNorm`)
    cuando `preserve_norm_scales=True` y contrae todos los pesos de proyección con ruido fresco
    escalado a `1/sqrt(fan_in)` para restaurar plasticidad en entrenamientos multi-década.
    """
    if not (0.0 < shrink <= 1.0):
        raise ValueError("shrink debe pertenecer a (0, 1]")
    if perturb < 0.0:
        raise ValueError("perturb debe ser >= 0")

    sq_before_by_dev: dict[torch.device, torch.Tensor] = {}
    sq_after_by_dev: dict[torch.device, torch.Tensor] = {}
    finite_by_dev: dict[torch.device, torch.Tensor] = {}
    n_updated = 0

    for idx, (name, param) in enumerate(model.named_parameters()):
        if not param.requires_grad or not param.dtype.is_floating_point or param.numel() == 0:
            continue
        dev = param.device
        calc_dt = _stable_calc_dtype(param.dtype)
        p_f = param.detach().to(dtype=calc_dt)
        sq_b = (p_f * p_f).sum().to(dtype=torch.float64)
        if dev not in sq_before_by_dev:
            sq_before_by_dev[dev] = sq_b
            sq_after_by_dev[dev] = torch.zeros((), device=dev, dtype=torch.float64)
            finite_by_dev[dev] = torch.ones((), device=dev, dtype=torch.bool)
        else:
            sq_before_by_dev[dev] = sq_before_by_dev[dev] + sq_b

        is_norm_weight = (
            preserve_norm_scales
            and param.ndim == 1
            and "norm" in name.lower()
            and not name.endswith(".bias")
        )
        if is_norm_weight:
            # Contraer suavemente hacia identidad (1.0) para no amortiguar activaciones
            p_new = p_f * float(shrink) + (1.0 - float(shrink))
        else:
            p_new = p_f * float(shrink)
            if perturb > 0.0 and not name.endswith(".bias"):
                if param.ndim >= 2:
                    fan_in = int(param.shape[1])
                elif param.ndim == 1:
                    fan_in = int(param.shape[0])
                else:
                    fan_in = 1
                init_std = 1.0 / math.sqrt(max(fan_in, 1))
                gen = torch.Generator(device=dev)
                gen.manual_seed(int(seed) * 1_000_003 + idx)
                noise = torch.randn(param.shape, generator=gen, device=dev, dtype=calc_dt)
                p_new = p_new + (float(perturb) * init_std) * noise

        param.copy_(p_new.to(dtype=param.dtype))
        p_after = param.detach().to(dtype=calc_dt)
        finite_by_dev[dev] = finite_by_dev[dev] & torch.isfinite(p_after).all()
        sq_after_by_dev[dev] = sq_after_by_dev[dev] + (p_after * p_after).sum().to(
            dtype=torch.float64
        )
        n_updated += 1

    # Sincronización única por dispositivo (mata las ~450 pausas .cpu().item() por llamada)
    sq_before = 0.0
    sq_after = 0.0
    all_finite = True
    if sq_before_by_dev:
        devs = list(sq_before_by_dev.keys())
        primary = devs[0]
        total_sq_b = sum(
            (sq_before_by_dev[d].to(device=primary) for d in devs[1:]),
            sq_before_by_dev[primary],
        )
        total_sq_a = sum(
            (sq_after_by_dev[d].to(device=primary) for d in devs[1:]),
            sq_after_by_dev[primary],
        )
        total_fin = torch.stack(
            [finite_by_dev[d].to(device=primary) for d in devs]
        ).all().to(dtype=torch.float64)
        synced = torch.stack([total_sq_b, total_sq_a, total_fin]).cpu().tolist()
        sq_before = float(synced[0])
        sq_after = float(synced[1])
        all_finite = bool(synced[2] > 0.5)

    norm_before = math.sqrt(max(sq_before, 0.0))
    norm_after = math.sqrt(max(sq_after, 0.0))
    return dict(
        norm_before=norm_before,
        norm_after=norm_after,
        norm_ratio=norm_after / max(norm_before, 1e-12),
        n_tensors_updated=n_updated,
        all_finite=all_finite,
        shrink=float(shrink),
        perturb=float(perturb),
        seed=int(seed),
    )


# ---------------------------------------------------------------------------
# 4. Optimizadores SOTA: Muon (Newton-Schulz 5-iter) + Schedule-Free AdamW + Factory
# ---------------------------------------------------------------------------


def zeropower_via_newtonschulz5(
    grad: torch.Tensor,
    *,
    steps: int = 5,
    eps: float = 1e-7,
    use_bf16: bool = False,
) -> torch.Tensor:
    """Ortogonalización matricial quintic Newton-Schulz en 5 iteraciones (Muon, 2D o 3D `bmm`).

    Calcula la norma estrictamente en FP32/FP64 con margen de seguridad `1.01` ANTES del cast
    a `bfloat16` para impedir desbordamiento/subdesbordamiento o `sigma_max > 1` bajo BF16 AMP.
    Opera 100 % sobre tensores en dispositivo sin ninguna sincronización host (`.item()`/`.cpu()`).
    """
    if grad.ndim not in (2, 3):
        raise ValueError(
            f"zeropower_via_newtonschulz5 requiere tensor 2D o 3D; recibido ndim={grad.ndim}"
        )
    if steps < 1:
        raise ValueError("steps debe ser >= 1")

    orig_dtype = grad.dtype
    calc_dtype = _stable_calc_dtype(orig_dtype)
    x_f = grad.to(dtype=calc_dtype)
    transposed = x_f.shape[-2] > x_f.shape[-1]
    if transposed:
        x_f = x_f.transpose(-2, -1)

    # Norma estrictamente en FP32/FP64 + margen 1.01 antes de cast BF16 (evita divergencia NS-5)
    norm = x_f.norm(dim=(-2, -1), keepdim=True)
    x_f = x_f / (norm * 1.01 + eps)

    if use_bf16 and grad.is_cuda:
        x = x_f.to(dtype=torch.bfloat16)
    else:
        x = x_f

    a, b, c = 3.4445, -4.7750, 2.0315
    if x.ndim == 3:
        for _ in range(steps):
            a_mat = torch.bmm(x, x.transpose(-2, -1))
            b_mat = b * a_mat + c * torch.bmm(a_mat, a_mat)
            x = a * x + torch.bmm(b_mat, x)
    else:
        for _ in range(steps):
            a_mat = torch.matmul(x, x.transpose(-2, -1))
            b_mat = b * a_mat + c * torch.matmul(a_mat, a_mat)
            x = a * x + torch.matmul(b_mat, x)

    if transposed:
        x = x.transpose(-2, -1)
    return x.to(dtype=orig_dtype)


def _is_router_param(name: str) -> bool:
    low = name.lower()
    return any(tok in low for tok in _ROUTER_PARAM_TOKENS)


def _is_head_or_embedding_param(name: str) -> bool:
    low = name.lower()
    if low.startswith("head.") or ".head." in low:
        return True
    return any(tok in low for tok in _HEAD_OR_EMBED_TOKENS)


def _partition_model_parameters(
    model: nn.Module,
    *,
    lr: float,
    weight_decay: float,
    router_lr_scale: float,
    enable_muon: bool,
    muon_lr_scale: float = 1.0,
) -> list[dict[str, Any]]:
    """Particiona parámetros en grupos Muon 2D/3D-MoE ocultos, AdamW y routers (`3x`)."""
    muon_hidden: list[nn.Parameter] = []
    adamw_decay: list[nn.Parameter] = []
    adamw_no_decay: list[nn.Parameter] = []
    router_decay: list[nn.Parameter] = []
    router_no_decay: list[nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_router = _is_router_param(name)
        no_decay = param.ndim <= 1 or name.endswith(".bias") or "norm" in name.lower()
        if is_router:
            (router_no_decay if no_decay else router_decay).append(param)
            continue

        eligible_muon = (
            enable_muon
            and (
                (param.ndim == 2 and min(param.shape[0], param.shape[1]) >= 2)
                or (
                    param.ndim == 3
                    and param.shape[0] >= 1
                    and min(param.shape[1], param.shape[2]) >= 2
                )
            )
            and not no_decay
            and not _is_head_or_embedding_param(name)
        )
        if eligible_muon:
            muon_hidden.append(param)
        elif no_decay:
            adamw_no_decay.append(param)
        else:
            adamw_decay.append(param)

    groups: list[dict[str, Any]] = []
    if muon_hidden:
        groups.append(
            dict(
                params=muon_hidden,
                lr=lr * muon_lr_scale,
                weight_decay=weight_decay,
                use_muon=True,
                group_role="muon_hidden",
            )
        )
    if adamw_decay:
        groups.append(
            dict(
                params=adamw_decay,
                lr=lr,
                weight_decay=weight_decay,
                use_muon=False,
                group_role="adamw_decay",
            )
        )
    if adamw_no_decay:
        groups.append(
            dict(
                params=adamw_no_decay,
                lr=lr,
                weight_decay=0.0,
                use_muon=False,
                group_role="adamw_no_decay",
            )
        )
    if router_decay:
        groups.append(
            dict(
                params=router_decay,
                lr=lr * router_lr_scale,
                weight_decay=weight_decay,
                use_muon=False,
                group_role="router_decay",
            )
        )
    if router_no_decay:
        groups.append(
            dict(
                params=router_no_decay,
                lr=lr * router_lr_scale,
                weight_decay=0.0,
                use_muon=False,
                group_role="router_no_decay",
            )
        )
    return groups


class MuonHybridOptimizer(torch.optim.Optimizer):
    """Optimizador híbrido Muon (NS-5 batcheado por `bmm` en 2D/3D-MoE) + AdamW `_foreach_*`.

    Vectoriza actualizaciones con `torch._foreach_*` y `torch.bmm` por forma matricial para
    eliminar la tormenta de lanzamientos CUDA, manteniendo seguimiento de `step` por parámetro
    y cero sincronizaciones host en `step()`.
    """

    def __init__(
        self,
        params: nn.Module | Iterable[torch.Tensor] | Iterable[dict[str, Any]],
        *,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        router_lr_scale: float = 1.0 / 3.0,
        muon_lr_scale: float = 1.0,
        rms_match_adamw: bool = True,
        use_bf16_ns: bool = False,
    ) -> None:
        if lr <= 0.0:
            raise ValueError("lr debe ser > 0")
        if not (0.0 <= momentum < 1.0):
            raise ValueError("momentum debe pertenecer a [0, 1)")
        if ns_steps < 1:
            raise ValueError("ns_steps debe ser >= 1")

        if isinstance(params, nn.Module):
            param_groups: Any = _partition_model_parameters(
                params,
                lr=lr,
                weight_decay=weight_decay,
                router_lr_scale=router_lr_scale,
                enable_muon=True,
                muon_lr_scale=muon_lr_scale,
            )
        else:
            param_groups = params

        defaults = dict(
            lr=float(lr),
            weight_decay=float(weight_decay),
            momentum=float(momentum),
            nesterov=bool(nesterov),
            ns_steps=int(ns_steps),
            betas=(float(betas[0]), float(betas[1])),
            eps=float(eps),
            use_muon=False,
            group_role="adamw_default",
            rms_match_adamw=bool(rms_match_adamw),
            use_bf16_ns=bool(use_bf16_ns),
            step=0,
        )
        super().__init__(param_groups, defaults)
        for group in self.param_groups:
            if group.get("use_muon", False):
                for p in group["params"]:
                    if p.ndim not in (2, 3):
                        raise ValueError(
                            f"Los grupos Muon aceptan tensores 2D o 3D; recibido ndim={p.ndim}"
                        )

    @classmethod
    def from_model(
        cls,
        model: nn.Module,
        *,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        router_lr_scale: float = 1.0 / 3.0,
        **kwargs: Any,
    ) -> MuonHybridOptimizer:
        return cls(
            model,
            lr=lr,
            weight_decay=weight_decay,
            router_lr_scale=router_lr_scale,
            **kwargs,
        )

    @torch.no_grad()
    def step(self, closure: Callable[[], torch.Tensor] | None = None) -> torch.Tensor | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = float(group["lr"])
            wd = float(group["weight_decay"])
            use_muon = bool(group.get("use_muon", False))

            if use_muon:
                mom = float(group["momentum"])
                nesterov = bool(group["nesterov"])
                ns_steps = int(group["ns_steps"])
                rms_match = bool(group["rms_match_adamw"])
                use_bf16_ns = bool(group["use_bf16_ns"])

                # Agrupar por (device, calc_dtype) para operaciones `_foreach_*` y `bmm`
                muon_buckets: dict[
                    tuple[torch.device, torch.dtype],
                    tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]],
                ] = {}
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    calc_dt = _stable_calc_dtype(p.dtype)
                    g = p.grad.detach()
                    if g.dtype != calc_dt:
                        g = g.to(dtype=calc_dt)
                    st = self.state[p]
                    if "momentum_buffer" not in st:
                        st["momentum_buffer"] = torch.zeros_like(g)
                    st["step"] = int(st.get("step", 0)) + 1
                    key = (p.device, calc_dt)
                    if key not in muon_buckets:
                        muon_buckets[key] = ([], [], [])
                    ps_b, gs_b, bufs_b = muon_buckets[key]
                    ps_b.append(p)
                    gs_b.append(g)
                    bufs_b.append(st["momentum_buffer"])

                for (_, calc_dt), (ps_b, gs_b, bufs_b) in muon_buckets.items():
                    torch._foreach_mul_(bufs_b, mom)
                    torch._foreach_add_(bufs_b, gs_b)
                    g_effs = (
                        list(torch._foreach_add(gs_b, bufs_b, alpha=mom))
                        if nesterov
                        else bufs_b
                    )

                    # Agrupar por forma matricial (rows, cols) para ortogonalización `torch.bmm`
                    shape_groups: dict[tuple[int, int], list[int]] = {}
                    for i, g_eff in enumerate(g_effs):
                        rc = (int(g_eff.shape[-2]), int(g_eff.shape[-1]))
                        shape_groups.setdefault(rc, []).append(i)

                    orths: list[torch.Tensor | None] = [None] * len(ps_b)
                    alphas: list[float] = [0.0] * len(ps_b)
                    for (rows, cols), idxs in shape_groups.items():
                        scale = (
                            0.2 * math.sqrt(float(max(rows, cols)))
                            if rms_match
                            else math.sqrt(max(1.0, float(rows) / float(cols)))
                        )
                        alpha_val = -lr * scale
                        if len(idxs) == 1 and g_effs[idxs[0]].ndim == 2:
                            i0 = idxs[0]
                            orths[i0] = zeropower_via_newtonschulz5(
                                g_effs[i0], steps=ns_steps, use_bf16=use_bf16_ns
                            )
                            alphas[i0] = alpha_val
                        else:
                            slices: list[torch.Tensor] = []
                            counts: list[int] = []
                            for idx in idxs:
                                t = g_effs[idx]
                                if t.ndim == 2:
                                    slices.append(t.unsqueeze(0))
                                    counts.append(1)
                                else:
                                    slices.append(t)
                                    counts.append(int(t.shape[0]))
                            batched_in = (
                                slices[0] if len(slices) == 1 else torch.cat(slices, dim=0)
                            )
                            batched_out = zeropower_via_newtonschulz5(
                                batched_in, steps=ns_steps, use_bf16=use_bf16_ns
                            )
                            split_outs = (
                                [batched_out]
                                if len(counts) == 1
                                else list(torch.split(batched_out, counts, dim=0))
                            )
                            for idx, out_chunk in zip(idxs, split_outs, strict=True):
                                orths[idx] = (
                                    out_chunk.squeeze(0)
                                    if g_effs[idx].ndim == 2
                                    else out_chunk
                                )
                                alphas[idx] = alpha_val

                    same_ps: list[torch.Tensor] = []
                    same_orths: list[torch.Tensor] = []
                    same_alphas: list[float] = []
                    cast_ps: list[torch.Tensor] = []
                    cast_orths: list[torch.Tensor] = []
                    cast_alphas: list[float] = []
                    for p, orth_t, a_v in zip(ps_b, orths, alphas, strict=True):
                        assert orth_t is not None
                        if p.dtype == calc_dt:
                            same_ps.append(p)
                            same_orths.append(orth_t)
                            same_alphas.append(a_v)
                        else:
                            cast_ps.append(p)
                            cast_orths.append(orth_t)
                            cast_alphas.append(a_v)

                    if same_ps:
                        if wd != 0.0:
                            torch._foreach_mul_(same_ps, 1.0 - lr * wd)
                        scaled_u = torch._foreach_mul(same_orths, same_alphas)
                        torch._foreach_add_(same_ps, scaled_u)
                    if cast_ps:
                        p_fs = [p.detach().to(dtype=calc_dt) for p in cast_ps]
                        if wd != 0.0:
                            torch._foreach_mul_(p_fs, 1.0 - lr * wd)
                        scaled_u = torch._foreach_mul(cast_orths, cast_alphas)
                        torch._foreach_add_(p_fs, scaled_u)
                        for p, p_f in zip(cast_ps, p_fs, strict=True):
                            p.copy_(p_f.to(dtype=p.dtype))
            else:
                beta1, beta2 = group["betas"]
                eps = float(group["eps"])
                adam_buckets: dict[
                    tuple[torch.device, torch.dtype, bool, int],
                    tuple[
                        list[torch.Tensor],
                        list[torch.Tensor],
                        list[torch.Tensor],
                        list[torch.Tensor],
                    ],
                ] = {}
                max_step_in_group = int(group.get("step", 0))
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    calc_dt = _stable_calc_dtype(p.dtype)
                    g = p.grad.detach()
                    if g.dtype != calc_dt:
                        g = g.to(dtype=calc_dt)
                    st = self.state[p]
                    if "exp_avg" not in st:
                        st["exp_avg"] = torch.zeros_like(g)
                        st["exp_avg_sq"] = torch.zeros_like(g)
                    p_step = int(st.get("step", 0)) + 1
                    st["step"] = p_step
                    if p_step > max_step_in_group:
                        max_step_in_group = p_step
                    bkey = (p.device, calc_dt, p.dtype == calc_dt, p_step)
                    if bkey not in adam_buckets:
                        adam_buckets[bkey] = ([], [], [], [])
                    ps_b, gs_b, avgs_b, sqs_b = adam_buckets[bkey]
                    ps_b.append(p)
                    gs_b.append(g)
                    avgs_b.append(st["exp_avg"])
                    sqs_b.append(st["exp_avg_sq"])

                if not adam_buckets:
                    continue
                group["step"] = max_step_in_group

                for (_, calc_dt, same_dt, step_n), (ps_b, gs_b, avgs_b, sqs_b) in (
                    adam_buckets.items()
                ):
                    step_size = lr / (1.0 - (beta1**step_n))
                    bc2_sqrt = math.sqrt(1.0 - (beta2**step_n))
                    torch._foreach_mul_(avgs_b, beta1)
                    torch._foreach_add_(avgs_b, gs_b, alpha=1.0 - beta1)
                    torch._foreach_mul_(sqs_b, beta2)
                    torch._foreach_addcmul_(sqs_b, gs_b, gs_b, value=1.0 - beta2)
                    denoms = torch._foreach_sqrt(sqs_b)
                    torch._foreach_div_(denoms, bc2_sqrt)
                    torch._foreach_add_(denoms, eps)
                    if same_dt:
                        if wd != 0.0:
                            torch._foreach_mul_(ps_b, 1.0 - lr * wd)
                        torch._foreach_addcdiv_(ps_b, avgs_b, denoms, value=-step_size)
                    else:
                        p_fs = [p.detach().to(dtype=calc_dt) for p in ps_b]
                        if wd != 0.0:
                            torch._foreach_mul_(p_fs, 1.0 - lr * wd)
                        torch._foreach_addcdiv_(p_fs, avgs_b, denoms, value=-step_size)
                        for p, p_f in zip(ps_b, p_fs, strict=True):
                            p.copy_(p_f.to(dtype=p.dtype))

        return loss


class ScheduleFreeAdamW(torch.optim.Optimizer):
    """Schedule-Free AdamW vectorizado (`_foreach_*`, Defazio et al., NeurIPS 2024).

    Mantiene iterados `(x, z)` en `float32` y conmuta `p.data` entre el punto interpolado de
    entrenamiento `y = (1 - beta1) * z + beta1 * x` (`train()`) y el promedio `x` (`eval()`).
    Lanza `RuntimeError` si `step()` se invoca en modo `eval()` salvo `allow_auto_train=True`.
    Cero dependencias externas y cero sincronizaciones host en `step()`.
    """

    def __init__(
        self,
        params: nn.Module | Iterable[torch.Tensor] | Iterable[dict[str, Any]],
        *,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-4,
        warmup_steps: int = 0,
        weight_lr_power: float = 2.0,
        router_lr_scale: float = 1.0 / 3.0,
        allow_auto_train: bool = False,
    ) -> None:
        if lr <= 0.0:
            raise ValueError("lr debe ser > 0")
        if warmup_steps < 0:
            raise ValueError("warmup_steps debe ser >= 0")

        if isinstance(params, nn.Module):
            param_groups: Any = _partition_model_parameters(
                params,
                lr=lr,
                weight_decay=weight_decay,
                router_lr_scale=router_lr_scale,
                enable_muon=False,
            )
        else:
            param_groups = params

        defaults = dict(
            lr=float(lr),
            betas=(float(betas[0]), float(betas[1])),
            eps=float(eps),
            weight_decay=float(weight_decay),
            warmup_steps=int(warmup_steps),
            weight_lr_power=float(weight_lr_power),
            step=0,
            weight_sum=0.0,
            train_mode=True,
        )
        super().__init__(param_groups, defaults)
        self.allow_auto_train = bool(allow_auto_train)

    @torch.no_grad()
    def eval(self) -> None:
        """Coloca los parámetros del modelo en el iterado promediado `x_t` para evaluación."""
        for group in self.param_groups:
            if not group.get("train_mode", True):
                continue
            same_ps: list[torch.Tensor] = []
            same_xs: list[torch.Tensor] = []
            for p in group["params"]:
                st = self.state.get(p)
                if st is not None and "x" in st:
                    x_t = st["x"]
                    if x_t.dtype == p.dtype:
                        same_ps.append(p)
                        same_xs.append(x_t)
                    else:
                        p.copy_(x_t.to(dtype=p.dtype))
            if same_ps:
                torch._foreach_copy_(same_ps, same_xs)
            group["train_mode"] = False

    @torch.no_grad()
    def train(self) -> None:
        """Restaura el punto interpolado `y_t = (1 - beta1) * z_t + beta1 * x_t` para entrenar."""
        for group in self.param_groups:
            if group.get("train_mode", True):
                continue
            beta1 = float(group["betas"][0])
            buckets: dict[
                tuple[torch.device, torch.dtype, bool],
                tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]],
            ] = {}
            for p in group["params"]:
                st = self.state.get(p)
                if st is not None and "x" in st and "z" in st:
                    x_t, z_t = st["x"], st["z"]
                    bkey = (p.device, x_t.dtype, p.dtype == x_t.dtype)
                    if bkey not in buckets:
                        buckets[bkey] = ([], [], [])
                    ps_b, xs_b, zs_b = buckets[bkey]
                    ps_b.append(p)
                    xs_b.append(x_t)
                    zs_b.append(z_t)
            for (_, _, same_dt), (ps_b, xs_b, zs_b) in buckets.items():
                ys_b = torch._foreach_lerp(zs_b, xs_b, beta1)
                if same_dt:
                    torch._foreach_copy_(ps_b, ys_b)
                else:
                    for p, y_t in zip(ps_b, ys_b, strict=True):
                        p.copy_(y_t.to(dtype=p.dtype))
            group["train_mode"] = True

    @torch.no_grad()
    def step(self, closure: Callable[[], torch.Tensor] | None = None) -> torch.Tensor | None:
        if any(not group.get("train_mode", True) for group in self.param_groups):
            if self.allow_auto_train:
                self.train()
            else:
                raise RuntimeError(
                    "ScheduleFreeAdamW.step() invocado en modo eval(): los gradientes deben "
                    "calcularse en y_t tras llamar explícitamente a optimizer.train()."
                )

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = float(group["lr"])
            beta1, beta2 = group["betas"]
            eps = float(group["eps"])
            wd = float(group["weight_decay"])
            warmup_steps = int(group["warmup_steps"])
            r_pow = float(group["weight_lr_power"])

            sf_buckets: dict[
                tuple[torch.device, torch.dtype, bool, int, float],
                tuple[
                    list[torch.Tensor],
                    list[torch.Tensor],
                    list[torch.Tensor],
                    list[torch.Tensor],
                    list[torch.Tensor],
                    list[torch.Tensor],
                ],
            ] = {}
            max_step = int(group.get("step", 0))
            max_wsum = float(group.get("weight_sum", 0.0))

            for p in group["params"]:
                if p.grad is None:
                    continue
                calc_dt = _stable_calc_dtype(p.dtype)
                grad = p.grad.detach()
                if grad.dtype != calc_dt:
                    grad = grad.to(dtype=calc_dt)
                y = p.detach() if p.dtype == calc_dt else p.detach().to(dtype=calc_dt)
                st = self.state[p]
                if "z" not in st:
                    st["z"] = y.clone()
                    st["x"] = y.clone()
                    st["exp_avg_sq"] = torch.zeros_like(y)
                    st["step"] = 0
                    st["weight_sum"] = 0.0
                p_step = int(st.get("step", 0)) + 1
                st["step"] = p_step
                sched = (
                    min(1.0, float(p_step) / float(warmup_steps)) if warmup_steps > 0 else 1.0
                )
                lr_t = lr * sched
                weight = (lr_t**r_pow) if lr_t > 0.0 else 0.0
                w_sum = float(st.get("weight_sum", 0.0)) + weight
                st["weight_sum"] = w_sum
                if p_step > max_step:
                    max_step = p_step
                    max_wsum = w_sum
                bkey = (p.device, calc_dt, p.dtype == calc_dt, p_step, w_sum)
                if bkey not in sf_buckets:
                    sf_buckets[bkey] = ([], [], [], [], [], [])
                ps_b, gs_b, ys_b, xs_b, zs_b, vs_b = sf_buckets[bkey]
                ps_b.append(p)
                gs_b.append(grad)
                ys_b.append(y)
                xs_b.append(st["x"])
                zs_b.append(st["z"])
                vs_b.append(st["exp_avg_sq"])

            if not sf_buckets:
                continue
            group["step"] = max_step
            group["weight_sum"] = max_wsum

            for (_, _, same_dt, p_step, w_sum), (ps_b, gs_b, ys_b, xs_b, zs_b, vs_b) in (
                sf_buckets.items()
            ):
                sched = (
                    min(1.0, float(p_step) / float(warmup_steps)) if warmup_steps > 0 else 1.0
                )
                lr_t = lr * sched
                weight = (lr_t**r_pow) if lr_t > 0.0 else 0.0
                ckp1 = (weight / w_sum) if w_sum > 0.0 else 0.0
                bc2_sqrt = math.sqrt(1.0 - (beta2**p_step))

                torch._foreach_mul_(vs_b, beta2)
                torch._foreach_addcmul_(vs_b, gs_b, gs_b, value=1.0 - beta2)
                denoms = torch._foreach_sqrt(vs_b)
                torch._foreach_div_(denoms, bc2_sqrt)
                torch._foreach_add_(denoms, eps)
                upds = torch._foreach_div(gs_b, denoms)
                if wd != 0.0:
                    torch._foreach_add_(upds, ys_b, alpha=wd)
                torch._foreach_add_(zs_b, upds, alpha=-lr_t)
                torch._foreach_lerp_(xs_b, zs_b, ckp1)
                y_nexts = torch._foreach_lerp(zs_b, xs_b, beta1)
                if same_dt:
                    torch._foreach_copy_(ps_b, y_nexts)
                else:
                    for p, y_next in zip(ps_b, y_nexts, strict=True):
                        p.copy_(y_next.to(dtype=p.dtype))
            group["train_mode"] = True

        return loss


def build_sota_optimizer(
    model: nn.Module,
    *,
    optimizer_type: str = "adamw_fused",
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    router_lr_scale: float = 1.0 / 3.0,
    cfg: BlackwellRuntimeConfig | None = None,
    muon_momentum: float = 0.95,
    muon_ns_steps: int = 5,
    muon_lr_scale: float = 1.0,
    muon_rms_match_adamw: bool = True,
    sf_warmup_steps: int = 0,
    sf_beta1: float = 0.9,
    sf_beta2: float = 0.999,
) -> torch.optim.Optimizer:
    """Fábrica oficial SOTA: `adamw_fused`, `muon_hybrid` (NS-5) o `schedule_free_adamw`."""
    kind = str(optimizer_type).strip().lower()
    if kind == "adamw_fused":
        base_cfg = cfg or BlackwellRuntimeConfig()
        rt_cfg = BlackwellRuntimeConfig(
            precision_mode=base_cfg.precision_mode,
            compile_mode=base_cfg.compile_mode,
            use_triton_fused_ops=base_cfg.use_triton_fused_ops,
            use_fused_adamw=base_cfg.use_fused_adamw,
            use_pinned_prefetch=base_cfg.use_pinned_prefetch,
            router_lr_scale=router_lr_scale,
            vram_fraction_per_worker=base_cfg.vram_fraction_per_worker,
            allocator_conf=base_cfg.allocator_conf,
            cuda_arch_list=base_cfg.cuda_arch_list,
            deterministic=base_cfg.deterministic,
        )
        return build_optimizer(model, lr=lr, weight_decay=weight_decay, cfg=rt_cfg)

    if kind == "muon_hybrid":
        use_bf16_ns = bool(cfg is not None and cfg.precision_mode == "bf16_amp")
        return MuonHybridOptimizer(
            model,
            lr=lr,
            weight_decay=weight_decay,
            momentum=muon_momentum,
            ns_steps=muon_ns_steps,
            router_lr_scale=router_lr_scale,
            muon_lr_scale=muon_lr_scale,
            rms_match_adamw=muon_rms_match_adamw,
            use_bf16_ns=use_bf16_ns,
        )

    if kind == "schedule_free_adamw":
        return ScheduleFreeAdamW(
            model,
            lr=lr,
            betas=(sf_beta1, sf_beta2),
            weight_decay=weight_decay,
            warmup_steps=sf_warmup_steps,
            router_lr_scale=router_lr_scale,
        )

    raise ValueError(
        f"optimizer_type inválido {optimizer_type!r}; válidos: {SOTA_OPTIMIZER_TYPES}"
    )


# ---------------------------------------------------------------------------
# 5. Ensamblado determinista por rangos de 3+ semillas (Rank Ensemble Builder)
# ---------------------------------------------------------------------------


def _rank_transform_1d(values: np.ndarray, *, method: RankEnsembleMethod) -> np.ndarray:
    """Transformación determinista por rangos (con empates promediados y seguridad NaN)."""
    arr = np.asarray(values, dtype=np.float64)
    out = np.full(arr.shape, np.nan, dtype=np.float64)
    finite_mask = np.isfinite(arr)
    n_valid = int(np.sum(finite_mask))
    if n_valid == 0:
        return out
    if n_valid == 1:
        out[finite_mask] = 0.0
        return out

    valid_vals = arr[finite_mask]
    if float(np.ptp(valid_vals)) <= 0.0:
        out[finite_mask] = 0.0
        return out

    ranks = average_ranks(valid_vals)  # 1..n_valid, determinista vía mergesort
    if method == "rank_zscore":
        centered = ranks - float(np.mean(ranks))
        std = float(np.std(centered, ddof=0))
        transformed = centered / std if std > 1e-12 else np.zeros_like(centered)
    elif method == "rank_uniform":
        transformed = (ranks - 0.5) / float(n_valid) - 0.5
    elif method == "gaussian_rank":
        u = np.clip((ranks - 0.5) / float(n_valid), 1e-6, 1.0 - 1e-6)
        transformed = stats.norm.ppf(u)
    else:
        raise ValueError(f"method inválido para rank ensemble: {method!r}")

    out[finite_mask] = transformed
    return out


def _as_numpy_float64(item: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(item, torch.Tensor):
        return item.detach().cpu().numpy().astype(np.float64, copy=False)
    return np.asarray(item, dtype=np.float64)


def rank_ensemble_predictions(
    seed_predictions: Sequence[np.ndarray | torch.Tensor],
    *,
    bounds: Sequence[tuple[int, int]] | None = None,
    weights: Sequence[float] | None = None,
    method: RankEnsembleMethod = "rank_zscore",
) -> np.ndarray:
    """Combina predicciones de múltiples semillas por rangos transversales deterministas.

    Soporta paneles 2D `[T, N]` (una sesión por fila) o vectores apilados 1D `[M]` con `bounds`
    (o una sola sesión 1D si `bounds=None`). Invariante a escalas arbitrarias entre semillas y
    tolerante a `NaN` parciales.
    """
    if len(seed_predictions) == 0:
        raise ValueError("seed_predictions no puede estar vacío")

    arrays = [_as_numpy_float64(p) for p in seed_predictions]
    ref_shape = arrays[0].shape
    if any(a.shape != ref_shape for a in arrays[1:]):
        raise ValueError(f"Todas las semillas deben compartir forma; esperada {ref_shape}")
    if arrays[0].ndim not in (1, 2):
        raise ValueError(f"Se esperan arrays 1D o 2D; recibido ndim={arrays[0].ndim}")

    n_seeds = len(arrays)
    if weights is None:
        w = np.full(n_seeds, 1.0 / float(n_seeds), dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64)
        if len(w) != n_seeds or np.any(w < 0.0) or float(np.sum(w)) <= 0.0:
            raise ValueError("weights debe tener misma longitud que seed_predictions y suma > 0")
        w = w / float(np.sum(w))

    def _combine_slice(slices: list[np.ndarray]) -> np.ndarray:
        transformed = np.vstack([_rank_transform_1d(s, method=method) for s in slices])  # [S, K]
        finite = np.isfinite(transformed)
        w_mat = np.where(finite, w[:, None], 0.0)
        w_sum = np.sum(w_mat, axis=0)
        safe_vals = np.where(finite, transformed, 0.0)
        num = np.sum(w_mat * safe_vals, axis=0)
        out_slice = np.full(transformed.shape[1], np.nan, dtype=np.float64)
        valid = w_sum > 0.0
        out_slice[valid] = num[valid] / w_sum[valid]
        return out_slice

    if arrays[0].ndim == 2:
        t_len, n_assets = ref_shape
        out = np.empty((t_len, n_assets), dtype=np.float32)
        for t in range(t_len):
            out[t] = _combine_slice([a[t] for a in arrays]).astype(np.float32)
        return out

    # Caso 1D (sesión única o sesiones apiladas con `bounds`)
    seg_bounds = list(bounds) if bounds is not None else [(0, ref_shape[0])]
    out_1d = np.full(ref_shape[0], np.nan, dtype=np.float32)
    for lo, hi in seg_bounds:
        if hi <= lo:
            continue
        out_1d[lo:hi] = _combine_slice([a[lo:hi] for a in arrays]).astype(np.float32)
    return out_1d


def evaluate_rank_ensemble(
    seed_predictions: Sequence[np.ndarray | torch.Tensor],
    targets: np.ndarray | torch.Tensor,
    *,
    bounds: Sequence[tuple[int, int]] | None = None,
    weights: Sequence[float] | None = None,
    method: RankEnsembleMethod = "rank_zscore",
) -> dict[str, Any]:
    """Evalúa Rank IC por semilla frente al ensemble por rangos y mide ganancia + diversidad."""
    y = _as_numpy_float64(targets)
    seeds_np = [_as_numpy_float64(p) for p in seed_predictions]
    ens = rank_ensemble_predictions(
        seeds_np, bounds=bounds, weights=weights, method=method
    )

    def _mean_rank_ic(pred: np.ndarray) -> float:
        ics: list[float] = []
        if pred.ndim == 2:
            for t in range(pred.shape[0]):
                mask = np.isfinite(pred[t]) & np.isfinite(y[t])
                if int(np.sum(mask)) >= 3:
                    v = rank_ic_vector(pred[t][mask], y[t][mask])
                    if math.isfinite(v):
                        ics.append(v)
        else:
            seg_bounds = list(bounds) if bounds is not None else [(0, len(pred))]
            for lo, hi in seg_bounds:
                p_s, y_s = pred[lo:hi], y[lo:hi]
                mask = np.isfinite(p_s) & np.isfinite(y_s)
                if int(np.sum(mask)) >= 3:
                    v = rank_ic_vector(p_s[mask], y_s[mask])
                    if math.isfinite(v):
                        ics.append(v)
        return float(np.mean(ics)) if ics else float("nan")

    seed_ics = [_mean_rank_ic(p) for p in seeds_np]
    ens_ic = _mean_rank_ic(ens.astype(np.float64))
    mean_seed_ic = float(np.nanmean(seed_ics))
    max_seed_ic = float(np.nanmax(seed_ics))

    return dict(
        n_seeds=len(seeds_np),
        method=method,
        seed_rank_ics=seed_ics,
        mean_seed_rank_ic=mean_seed_ic,
        max_seed_rank_ic=max_seed_ic,
        ensemble_rank_ic=ens_ic,
        gain_over_mean_seed=ens_ic - mean_seed_ic,
        gain_over_best_seed=ens_ic - max_seed_ic,
    )


__all__ = [
    "DEFAULT_SEARCH_PARAM_NAMES",
    "SOTA_OPTIMIZER_TYPES",
    "FitEpochDecision",
    "FitRegime",
    "MuonHybridOptimizer",
    "OptimalFitConfig",
    "OptimalFitController",
    "ParamKind",
    "ParamSpec",
    "RankEnsembleMethod",
    "ScheduleFreeAdamW",
    "SearchSpaceSpec",
    "TPEHyperbandAutoTuner",
    "TrialContext",
    "TrialPruned",
    "TrialRecord",
    "append_audit_record",
    "build_sota_optimizer",
    "check_parameter_sensitivity_cliff",
    "compute_multiplicity_haircut",
    "evaluate_rank_ensemble",
    "rank_ensemble_predictions",
    "read_audit_ledger",
    "shrink_and_perturb",
    "validate_purged_validation_split",
    "zeropower_via_newtonschulz5",
]
