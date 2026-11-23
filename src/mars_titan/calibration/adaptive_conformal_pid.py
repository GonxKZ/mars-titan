"""Calibración conforme normalizada localmente con controlador PID en línea (CQR-NPID).

Resuelve dos defectos medidos en los intervalos conformes estáticos de 2013 (`A2` / `F1`):
1. Colapso en el decil extremo de volatilidad (0,687 frente a 0,80 nominal en `combined_cqr`):
   la puntuación de conformidad aditiva `max(q_lo - y, y - q_hi)` suma una constante fija a
   activos tranquilos y a activos muy volátiles. Al normalizar por una escala causal estricta
   `s_i > 0` (anchura de cuantiles del núcleo, volatilidad realizada conocida `vol_21`, o híbrido),
   `e_i = max(q_lo - y, y - q_hi) / s_i`, la corrección multiplicativa `Q_t * s_i` iguala la
   cobertura condicional en los 10 deciles de volatilidad y estrecha la anchura media.
2. Deriva de cobertura ante cambios de régimen (trimestres tranquilos frente a crisis):
   el controlador PID conforme en línea (Angelopoulos, Candès y Tibshirani, NeurIPS 2023)
   actualiza `Q_t` usando exclusivamente cohortes ya maduras (`matures_at <= prediction_at`) y
   los intervalos realmente emitidos cuando se decidió cada cohorte:
   - P (seguidor de cuantil): `q_track <- q_track + kp * (err_c - alpha)`
   - I (integrador saturado): `err_sum <- decay * err_sum + (err_c - alpha)`;
     `I_term = ki * tanh(c_sat * err_sum)` (acotado, sin *windup*)
   - D (momento de puntuación normalizada madura): `kd * (ema_fast - ema_slow)`

Contrato causal y sello 2024:
- Rechaza cualquier instante `prediction_at` o `matures_at` >= `FINAL_TEST_START_US`.
- Cada cohorte solo actualiza el estado al madurar (`matures_at <= now`), usando el `Q` que
  estuvo vigente al emitir esa cohorte (sin fuga de correcciones futuras).
- Estado serializable con `export()` / `restore()` y huella SHA-256 canónica (`state_sha256()`).
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from mars_titan.environments.cohorts import FINAL_TEST_START_US

KIND = "normalized_conformal_pid_v1"
DEFAULT_LEVELS = (0.025, 0.1, 0.5, 0.9, 0.975)
DEFAULT_NOMINALS = (0.8, 0.95)
SCALE_MODES = ("width", "volatility", "hybrid")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _finite_sample_quantile(scores: np.ndarray, nominal: float) -> float:
    arr = np.sort(np.asarray(scores, dtype=np.float64))
    n = int(arr.size)
    _require(n > 0, "Se requiere al menos una puntuación de calibración")
    rank = math.ceil((n + 1) * nominal)
    return float(arr[-1]) if rank > n else float(arr[rank - 1])


def _interval_indices(levels: tuple[float, ...], nominals: tuple[float, ...]) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for nom in nominals:
        lo_level = round((1.0 - nom) / 2.0, 6)
        hi_level = round(1.0 - (1.0 - nom) / 2.0, 6)
        lo_idx = [i for i, lv in enumerate(levels) if abs(lv - lo_level) < 1e-6]
        hi_idx = [i for i, lv in enumerate(levels) if abs(lv - hi_level) < 1e-6]
        _require(
            len(lo_idx) == 1 and len(hi_idx) == 1,
            f"Los niveles {levels} no contienen el par simétrico para nominal={nom}",
        )
        pairs.append((lo_idx[0], hi_idx[0]))
    return pairs


def make_stratified_groups(markets: np.ndarray, strata: np.ndarray) -> np.ndarray:
    """Combina mercado y estrato causal (p. ej. decil de volatilidad) en claves de grupo Mondrian."""
    m = np.asarray(markets).astype(str)
    s = np.asarray(strata)
    _require(len(m) == len(s), "markets y strata deben tener la misma longitud")
    return np.array([f"{mk}|s{int(st)}" for mk, st in zip(m.tolist(), s.tolist(), strict=True)])


def _compute_row_scale(
    quantiles: np.ndarray,
    lo_idx: int,
    hi_idx: int,
    *,
    scale_mode: str,
    external_scales: np.ndarray | None,
    scale_floor: float,
    scale_ceiling: float,
    hybrid_ratio: float,
) -> np.ndarray:
    """Escala causal estrictamente positiva y acotada `scale_floor <= s_i <= scale_ceiling`."""
    width = np.maximum(quantiles[:, hi_idx] - quantiles[:, lo_idx], 0.0)
    if scale_mode == "width":
        return np.clip(width, scale_floor, scale_ceiling)
    _require(
        external_scales is not None and len(external_scales) == len(quantiles),
        f"scale_mode={scale_mode!r} requiere `scales` causal con la misma longitud que quantiles",
    )
    ext = np.asarray(external_scales, dtype=np.float64)
    ext_clean = np.where(np.isfinite(ext) & (ext > 0.0), ext, np.nan)
    fallback = np.clip(width, scale_floor, scale_ceiling)
    ext_filled = np.where(np.isfinite(ext_clean), ext_clean, fallback)
    if scale_mode == "volatility":
        return np.clip(ext_filled, scale_floor, scale_ceiling)
    scaled_ext = ext_filled * hybrid_ratio
    raw_h = np.sqrt(np.maximum(width, scale_floor) * np.maximum(scaled_ext, scale_floor))
    return np.clip(raw_h, scale_floor, scale_ceiling)


def _enforce_monotone_quantiles(q: np.ndarray) -> np.ndarray:
    """Garantiza cuantiles no decrecientes por fila sin alterar la mediana."""
    out = np.array(q, dtype=np.float64, copy=True)
    # Reordenación exacta fila a fila solo en filas con cruce (<0.1% típico)
    crossed = np.any(np.diff(out, axis=1) < 0.0, axis=1)
    if np.any(crossed):
        out[crossed] = np.sort(out[crossed], axis=1)
    return out


@dataclass
class _ChannelState:
    """Estado PID por (mercado, nominal)."""

    q_track: float
    err_sum: float = 0.0
    ema_fast: float = 0.0
    ema_slow: float = 0.0
    updates: int = 0

    def to_dict(self) -> dict[str, Any]:
        return dict(
            q_track=float(self.q_track),
            err_sum=float(self.err_sum),
            ema_fast=float(self.ema_fast),
            ema_slow=float(self.ema_slow),
            updates=int(self.updates),
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> _ChannelState:
        return cls(
            q_track=float(data["q_track"]),
            err_sum=float(data["err_sum"]),
            ema_fast=float(data["ema_fast"]),
            ema_slow=float(data["ema_slow"]),
            updates=int(data["updates"]),
        )


@dataclass
class NormalizedConformalPID:
    """Calibrador conforme normalizado localmente con controlador PID causal por mercado."""

    levels: tuple[float, ...]
    nominals: tuple[float, ...]
    pairs: list[tuple[int, int]]
    scale_mode: str
    scale_floors: dict[str, list[float]]
    scale_ceilings: dict[str, list[float]]
    hybrid_ratios: dict[str, list[float]]
    kp: float
    ki: float
    kd: float
    c_sat: float
    err_decay: float
    ema_fast_alpha: float
    ema_slow_alpha: float
    q_min: float
    q_max: float
    channels: dict[str, list[_ChannelState]]
    pending: list[dict[str, Any]] = field(default_factory=list)
    emissions_count: int = 0
    updates_count: int = 0
    last_time_by_group: dict[str, int] = field(default_factory=dict)

    @classmethod
    def start(
        cls,
        target: np.ndarray,
        quantiles: np.ndarray,
        groups: np.ndarray,
        *,
        levels: tuple[float, ...] = DEFAULT_LEVELS,
        nominals: tuple[float, ...] = DEFAULT_NOMINALS,
        scale_mode: str = "width",
        cal_scales: np.ndarray | None = None,
        floor_quantile: float = 0.05,
        ceiling_quantile: float = 0.995,
        kp: float = 0.05,
        ki: float = 0.10,
        kd: float = 0.0,
        c_sat: float = 2.0,
        err_decay: float = 0.98,
        ema_fast_alpha: float = 0.25,
        ema_slow_alpha: float = 0.05,
        q_min: float = -0.5,
        q_max: float = 10.0,
    ) -> NormalizedConformalPID:
        """Ajusta la CQR normalizada inicial en calibración por grupo y nominal."""
        _require(scale_mode in SCALE_MODES, f"scale_mode inválido: {scale_mode!r}")
        _require(
            kp >= 0.0 and ki >= 0.0 and kd >= 0.0 and c_sat > 0.0 and 0.0 < err_decay <= 1.0,
            "Parámetros PID inválidos",
        )
        y = np.asarray(target, dtype=np.float64)
        q = np.asarray(quantiles, dtype=np.float64)
        g = np.asarray(groups).astype(str)
        levels_t = tuple(float(v) for v in levels)
        nominals_t = tuple(float(v) for v in nominals)
        pairs = _interval_indices(levels_t, nominals_t)
        _require(
            q.ndim == 2 and q.shape[1] == len(levels_t) and len(y) == len(q) == len(g) and len(y) > 0,
            "Dimensiones incompatibles en calibración de NormalizedConformalPID",
        )
        _require(
            bool(np.isfinite(y).all() and np.isfinite(q).all()),
            "Calibración requiere target y quantiles finitos",
        )

        ext_all = None if cal_scales is None else np.asarray(cal_scales, dtype=np.float64)
        if scale_mode != "width":
            _require(
                ext_all is not None and len(ext_all) == len(y),
                f"scale_mode={scale_mode!r} requiere cal_scales con la misma longitud",
            )

        channels: dict[str, list[_ChannelState]] = {}
        scale_floors: dict[str, list[float]] = {}
        scale_ceilings: dict[str, list[float]] = {}
        hybrid_ratios: dict[str, list[float]] = {}

        for grp in sorted(set(g.tolist())):
            mask = g == grp
            yg = y[mask]
            qg = q[mask]
            eg = None if ext_all is None else ext_all[mask]
            grp_channels: list[_ChannelState] = []
            grp_floors: list[float] = []
            grp_ceilings: list[float] = []
            grp_ratios: list[float] = []

            for nom, (lo_i, hi_i) in zip(nominals_t, pairs, strict=True):
                w = np.maximum(qg[:, hi_i] - qg[:, lo_i], 0.0)
                pos_w = w[w > 0.0]
                w_floor = float(np.quantile(pos_w, floor_quantile)) if pos_w.size > 0 else 1e-4
                w_floor = max(w_floor, 1e-6)
                w_ceil = float(np.quantile(pos_w, ceiling_quantile)) if pos_w.size > 0 else 1.0
                w_ceil = max(w_ceil, w_floor * 2.0)

                ratio = 1.0
                if scale_mode == "volatility":
                    pos_e = eg[np.isfinite(eg) & (eg > 0.0)]  # type: ignore[index]
                    s_floor = float(np.quantile(pos_e, floor_quantile)) if pos_e.size > 0 else w_floor
                    s_floor = max(s_floor, 1e-6)
                    s_ceil = float(np.quantile(pos_e, ceiling_quantile)) if pos_e.size > 0 else w_ceil
                    s_ceil = max(s_ceil, s_floor * 2.0)
                elif scale_mode == "hybrid":
                    pos_e = eg[np.isfinite(eg) & (eg > 0.0)]  # type: ignore[index]
                    med_w = float(np.median(pos_w)) if pos_w.size > 0 else 1e-2
                    med_e = float(np.median(pos_e)) if pos_e.size > 0 else med_w
                    ratio = med_w / max(med_e, 1e-8)
                    s_floor = w_floor
                    s_ceil = w_ceil
                else:
                    s_floor = w_floor
                    s_ceil = w_ceil

                s = _compute_row_scale(
                    qg,
                    lo_i,
                    hi_i,
                    scale_mode=scale_mode,
                    external_scales=eg,
                    scale_floor=s_floor,
                    scale_ceiling=s_ceil,
                    hybrid_ratio=ratio,
                )
                norm_scores = np.maximum(qg[:, lo_i] - yg, yg - qg[:, hi_i]) / s
                q0 = _finite_sample_quantile(norm_scores, nom)
                q0_clamped = float(np.clip(q0, q_min, q_max))
                grp_floors.append(s_floor)
                grp_ceilings.append(s_ceil)
                grp_ratios.append(float(ratio))
                grp_channels.append(
                    _ChannelState(q_track=q0_clamped, err_sum=0.0, ema_fast=q0_clamped, ema_slow=q0_clamped)
                )

            channels[grp] = grp_channels
            scale_floors[grp] = grp_floors
            scale_ceilings[grp] = grp_ceilings
            hybrid_ratios[grp] = grp_ratios

        return cls(
            levels=levels_t,
            nominals=nominals_t,
            pairs=pairs,
            scale_mode=scale_mode,
            scale_floors=scale_floors,
            scale_ceilings=scale_ceilings,
            hybrid_ratios=hybrid_ratios,
            kp=float(kp),
            ki=float(ki),
            kd=float(kd),
            c_sat=float(c_sat),
            err_decay=float(err_decay),
            ema_fast_alpha=float(ema_fast_alpha),
            ema_slow_alpha=float(ema_slow_alpha),
            q_min=float(q_min),
            q_max=float(q_max),
            channels=channels,
        )

    def current_q(self, group: str, nom_idx: int) -> float:
        """Corrección normalizada `Q_t` actual para `(group, nom_idx)`."""
        st = self.channels[group][nom_idx]
        i_term = self.ki * math.tanh(self.c_sat * st.err_sum) if self.ki > 0.0 else 0.0
        d_term = self.kd * (st.ema_fast - st.ema_slow) if self.kd > 0.0 else 0.0
        return float(np.clip(st.q_track + i_term + d_term, self.q_min, self.q_max))

    def _mature_until(self, group: str, now_us: int) -> list[dict[str, Any]]:
        """Procesa todas las cohortes pendientes de `group` con `matures_at <= now_us`."""
        applied: list[dict[str, Any]] = []
        remaining: list[dict[str, Any]] = []
        ready: list[dict[str, Any]] = []
        for item in self.pending:
            if item["group"] == group and item["matures_at"] <= now_us:
                ready.append(item)
            else:
                remaining.append(item)
        self.pending = remaining
        ready.sort(key=lambda r: (r["matures_at"], r["prediction_at"]))
        for item in ready:
            errs = item["miscoverages"]
            q_meds = item["score_quantiles"]
            for j, nom in enumerate(self.nominals):
                alpha = 1.0 - nom
                err_c = float(errs[j])
                gap = err_c - alpha
                st = self.channels[group][j]
                st.q_track = float(np.clip(st.q_track + self.kp * gap, self.q_min, self.q_max))
                st.err_sum = self.err_decay * st.err_sum + gap
                sc_q = float(q_meds[j])
                st.ema_fast = (1.0 - self.ema_fast_alpha) * st.ema_fast + self.ema_fast_alpha * sc_q
                st.ema_slow = (1.0 - self.ema_slow_alpha) * st.ema_slow + self.ema_slow_alpha * sc_q
                st.updates += 1
            self.updates_count += 1
            applied.append(
                dict(
                    group=group,
                    prediction_at=item["prediction_at"],
                    matures_at=item["matures_at"],
                    miscoverages=list(errs),
                )
            )
        return applied

    def emit_cohort(
        self,
        quantiles: np.ndarray,
        group: str,
        prediction_at: int,
        *,
        scales: np.ndarray | None = None,
        target: np.ndarray | None = None,
        matures_at: int | None = None,
    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        """Madura cohortes pasadas, emite intervalos para la cohorte actual y encola su madurez."""
        _require(group in self.channels, f"Grupo desconocido en NormalizedConformalPID: {group!r}")
        now_us = int(prediction_at)
        _require(now_us < FINAL_TEST_START_US, "Violación del sello de 2024 en prediction_at")
        prev_t = self.last_time_by_group.get(group)
        _require(
            prev_t is None or now_us >= prev_t,
            f"Orden cronológico violado en {group}: {now_us} < {prev_t}",
        )
        self.last_time_by_group[group] = now_us

        updates = self._mature_until(group, now_us)
        q_in = np.asarray(quantiles, dtype=np.float64)
        q_out = q_in.copy()
        emitted_q = [self.current_q(group, j) for j in range(len(self.nominals))]

        s_list: list[np.ndarray] = []
        for j, (lo_i, hi_i) in enumerate(self.pairs):
            s = _compute_row_scale(
                q_in,
                lo_i,
                hi_i,
                scale_mode=self.scale_mode,
                external_scales=scales,
                scale_floor=self.scale_floors[group][j],
                scale_ceiling=self.scale_ceilings[group][j],
                hybrid_ratio=self.hybrid_ratios[group][j],
            )
            s_list.append(s)
            q_out[:, lo_i] = q_in[:, lo_i] - emitted_q[j] * s
            q_out[:, hi_i] = q_in[:, hi_i] + emitted_q[j] * s

        q_out = _enforce_monotone_quantiles(q_out)
        self.emissions_count += 1

        if target is not None and matures_at is not None:
            mat_us = int(matures_at)
            _require(mat_us < FINAL_TEST_START_US, "Violación del sello de 2024 en matures_at")
            _require(mat_us > now_us, "matures_at debe ser estrictamente posterior a prediction_at")
            y = np.asarray(target, dtype=np.float64)
            miscovs: list[float] = []
            score_qs: list[float] = []
            for j, (nom, (lo_i, hi_i)) in enumerate(zip(self.nominals, self.pairs, strict=True)):
                covered = (y >= q_out[:, lo_i]) & (y <= q_out[:, hi_i])
                miscovs.append(float(1.0 - np.mean(covered)))
                norm_sc = np.maximum(q_in[:, lo_i] - y, y - q_in[:, hi_i]) / s_list[j]
                score_qs.append(float(np.quantile(norm_sc, nom)))
            self.pending.append(
                dict(
                    group=group,
                    prediction_at=now_us,
                    matures_at=mat_us,
                    miscoverages=miscovs,
                    score_quantiles=score_qs,
                )
            )
        return q_out, updates

    def export(self) -> dict[str, Any]:
        return dict(
            kind=KIND,
            levels=list(self.levels),
            nominals=list(self.nominals),
            scale_mode=self.scale_mode,
            scale_floors=self.scale_floors,
            scale_ceilings=self.scale_ceilings,
            hybrid_ratios=self.hybrid_ratios,
            kp=self.kp,
            ki=self.ki,
            kd=self.kd,
            c_sat=self.c_sat,
            err_decay=self.err_decay,
            ema_fast_alpha=self.ema_fast_alpha,
            ema_slow_alpha=self.ema_slow_alpha,
            q_min=self.q_min,
            q_max=self.q_max,
            channels={g: [c.to_dict() for c in chs] for g, chs in sorted(self.channels.items())},
            pending=list(self.pending),
            emissions_count=self.emissions_count,
            updates_count=self.updates_count,
            last_time_by_group=dict(sorted(self.last_time_by_group.items())),
        )

    @classmethod
    def restore(cls, state: dict[str, Any]) -> NormalizedConformalPID:
        _require(state.get("kind") == KIND, f"Estado incompatible: {state.get('kind')!r}")
        levels = tuple(float(v) for v in state["levels"])
        nominals = tuple(float(v) for v in state["nominals"])
        return cls(
            levels=levels,
            nominals=nominals,
            pairs=_interval_indices(levels, nominals),
            scale_mode=str(state["scale_mode"]),
            scale_floors={str(k): [float(x) for x in v] for k, v in state["scale_floors"].items()},
            scale_ceilings={str(k): [float(x) for x in v] for k, v in state["scale_ceilings"].items()},
            hybrid_ratios={str(k): [float(x) for x in v] for k, v in state["hybrid_ratios"].items()},
            kp=float(state["kp"]),
            ki=float(state["ki"]),
            kd=float(state["kd"]),
            c_sat=float(state["c_sat"]),
            err_decay=float(state["err_decay"]),
            ema_fast_alpha=float(state["ema_fast_alpha"]),
            ema_slow_alpha=float(state["ema_slow_alpha"]),
            q_min=float(state["q_min"]),
            q_max=float(state["q_max"]),
            channels={
                str(g): [_ChannelState.from_dict(c) for c in chs]
                for g, chs in state["channels"].items()
            },
            pending=[dict(item) for item in state["pending"]],
            emissions_count=int(state["emissions_count"]),
            updates_count=int(state["updates_count"]),
            last_time_by_group={str(k): int(v) for k, v in state["last_time_by_group"].items()},
        )

    def state_sha256(self) -> str:
        return hashlib.sha256(_canonical_json(self.export()).encode("utf-8")).hexdigest()


def replay_normalized_pid(
    calibrator: NormalizedConformalPID,
    quantiles: np.ndarray,
    groups: np.ndarray,
    prediction_at: np.ndarray,
    target: np.ndarray,
    matures_at: np.ndarray,
    *,
    scales: np.ndarray | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]], list[dict[str, Any]]]:
    """Reproduce causalmente el flujo de cohortes (grupo, prediction_at) en orden cronológico."""
    q = np.asarray(quantiles, dtype=np.float64)
    g = np.asarray(groups).astype(str)
    times = np.asarray(prediction_at, dtype=np.int64)
    y = np.asarray(target, dtype=np.float64)
    mats = np.asarray(matures_at, dtype=np.int64)
    sc = None if scales is None else np.asarray(scales, dtype=np.float64)
    n = len(q)
    _require(
        len(g) == n and len(times) == n and len(y) == n and len(mats) == n,
        "Longitudes incompatibles en replay_normalized_pid",
    )
    if n == 0:
        return np.empty_like(q), [], []
    _require(int(times.max()) < FINAL_TEST_START_US, "Violación del sello de 2024 en prediction_at")
    _require(int(mats.max()) < FINAL_TEST_START_US, "Violación del sello de 2024 en matures_at")
    _require(bool(np.all(mats > times)), "Todas las etiquetas deben madurar tras prediction_at")

    # Agrupar cohortes en orden cronológico estricto por (prediction_at, group)
    order = np.lexsort((g, times))
    t_ord = times[order]
    g_ord = g[order]
    change = np.r_[True, (t_ord[1:] != t_ord[:-1]) | (g_ord[1:] != g_ord[:-1])]
    starts = np.flatnonzero(change)
    ends = np.r_[starts[1:], n]

    emitted = np.empty_like(q)
    emissions: list[dict[str, Any]] = []
    all_updates: list[dict[str, Any]] = []
    for s, e in zip(starts, ends, strict=True):
        idx = order[s:e]
        grp = str(g_ord[s])
        now_us = int(t_ord[s])
        mat_us = int(mats[idx].max())
        q_coh, upd = calibrator.emit_cohort(
            q[idx],
            grp,
            now_us,
            scales=None if sc is None else sc[idx],
            target=y[idx],
            matures_at=mat_us,
        )
        emitted[idx] = q_coh
        emissions.append(dict(group=grp, prediction_at=now_us, rows=int(e - s)))
        all_updates.extend(upd)
    return emitted, emissions, all_updates
