"""Calibración probabilística Inductive Venn-Abers O(N log N) y puerta de riesgo SessionLTT.

Resuelve los dos problemas medidos en `F2` (`fiabilidad/2013/resumen.md`):
1. `VennAbersCalibrator` (Vovk y Petej, 2014):
   - La regresión isótona cruda (`isotonic_fit`) sobreajusta las colas extremas y empeora el
     Brier score frente a la tasa base en el tercio líquido (0,2516 > 0,2498).
   - Inductive Venn-Abers calcula un intervalo probabilístico exacto `[p0(x), p1(x)]` con
     garantía de calibración válida en muestra finita bajo intercambiabilidad (y con pesos de
     decaimiento temporal opcionales según Barber et al., 2023).
   - La probabilidad minimax de pérdida logarítmica `p = p1 / (1 - p0 + p1)` regulariza
     automáticamente las colas escasas y la anchura epistémica `p1 - p0` cuantifica la falta
     de evidencia en esa zona del score.
   - Implementación exacta O(N log^2 N) mediante árboles persistentes de envolvente convexa
     inferior (GCM) con *binary lifting* acelerada en C99/OpenMP (y respaldo puro en Python
     idéntico bit a bit): 100.000 filas de calibración + 100.000 de test en < 0,1 s.
2. `SessionLTTGate` (Learn-then-Test a nivel de sesión, Angelopoulos et al., 2021):
   - En lugar de exigir > 90 % de acierto de signo por activo individual (matemáticamente
     inalcanzable en retornos diarios de renta variable), certifica con control exacto de FWER
     la fiabilidad agregada a nivel de sesión o ventana de `block_size` sesiones:
       `P(IC_t > margin | gate_score_t >= lambda) >= target_reliability`
     con nivel de error de familia `<= delta` mediante prueba binomial exacta y contraste
     secuencial fijo (*fixed-sequence testing*) o Bonferroni sobre la rejilla fijada en
     calibración. Si ningún umbral cumple la garantía, devuelve `chosen_threshold = None`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.stats import binom

from mars_titan.memory import _fast_kernels as fk

Z95 = 1.959963984540054


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def brier_score(probabilities: np.ndarray, outcomes: np.ndarray) -> float:
    """Puntuación de Brier media `mean((p - y)^2)`."""
    p = np.asarray(probabilities, dtype=np.float64)
    y = np.asarray(outcomes, dtype=np.float64)
    _require(
        len(p) == len(y) and len(p) > 0, "brier_score requiere arrays no vacíos de igual longitud"
    )
    return float(np.mean((p - y) ** 2))


def expected_calibration_error(
    probabilities: np.ndarray,
    outcomes: np.ndarray,
    *,
    n_bins: int = 10,
) -> float:
    """Error de calibración esperado (ECE) ponderado por tamaño de intervalo."""
    p = np.asarray(probabilities, dtype=np.float64)
    y = np.asarray(outcomes, dtype=np.float64)
    _require(len(p) == len(y) and len(p) > 0 and n_bins >= 1, "Entradas inválidas para ECE")
    bins = np.clip((p * n_bins).astype(np.int64), 0, n_bins - 1)
    n = float(len(p))
    ece = 0.0
    for b in range(n_bins):
        mask = bins == b
        cnt = int(mask.sum())
        if cnt > 0:
            ece += (cnt / n) * abs(float(p[mask].mean()) - float(y[mask].mean()))
    return float(ece)


def _venn_abers_precompute_python(
    w_cum: np.ndarray,
    y_cum: np.ndarray,
    w_test: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Respaldo puro en Python del algoritmo exacto de envolvente persistente O(U log^2 U)."""
    u_count = len(w_cum) - 1
    log_n = max(1, (u_count + 1).bit_length())

    # 1) Árbol persistente de envolvente convexa inferior de prefijos P[0..k]
    parent_l = [-1] * (u_count + 1)
    stack = [0]
    for k in range(1, u_count + 1):
        wk, yk = float(w_cum[k]), float(y_cum[k])
        while len(stack) >= 2:
            a, b = stack[-2], stack[-1]
            if (float(y_cum[b]) - float(y_cum[a])) * (wk - float(w_cum[b])) >= (
                yk - float(y_cum[b])
            ) * (float(w_cum[b]) - float(w_cum[a])):
                stack.pop()
            else:
                break
        parent_l[k] = stack[-1]
        stack.append(k)

    up_l = [[-1] * log_n for _ in range(u_count + 1)]
    for v in range(u_count + 1):
        up_l[v][0] = parent_l[v]
        for j in range(1, log_n):
            p = up_l[v][j - 1]
            up_l[v][j] = -1 if p == -1 else up_l[p][j - 1]

    # 2) Árbol persistente de envolvente convexa inferior de sufijos P[k..U]
    next_r = [-1] * (u_count + 1)
    stack_r = [u_count]
    for k in range(u_count - 1, -1, -1):
        wk, yk = float(w_cum[k]), float(y_cum[k])
        while len(stack_r) >= 2:
            b, c = stack_r[-1], stack_r[-2]
            if (float(y_cum[b]) - yk) * (float(w_cum[c]) - float(w_cum[b])) >= (
                float(y_cum[c]) - float(y_cum[b])
            ) * (float(w_cum[b]) - wk):
                stack_r.pop()
            else:
                break
        next_r[k] = stack_r[-1]
        stack_r.append(k)

    up_r = [[-1] * log_n for _ in range(u_count + 1)]
    for v in range(u_count, -1, -1):
        up_r[v][0] = next_r[v]
        for j in range(1, log_n):
            nxt = up_r[v][j - 1]
            up_r[v][j] = -1 if nxt == -1 else up_r[nxt][j - 1]

    w_list = w_cum.tolist()
    y_list = y_cum.tolist()

    def _move_right(u: int, w: int, dx: float, dy: float) -> bool:
        nw = next_r[w]
        if nw == -1:
            return False
        return (y_list[nw] - y_list[w]) * ((w_list[w] + dx) - w_list[u]) < (
            (y_list[w] + dy) - y_list[u]
        ) * (w_list[nw] - w_list[w])

    def _best_right(u: int, k_right: int, dx: float, dy: float) -> float:
        cur = k_right
        if _move_right(u, cur, dx, dy):
            for j in range(log_n - 1, -1, -1):
                cand = up_r[cur][j]
                if cand != -1 and _move_right(u, cand, dx, dy):
                    cur = cand
            cur = next_r[cur]
        return ((y_list[cur] + dy) - y_list[u]) / ((w_list[cur] + dx) - w_list[u])

    def _move_left(u: int, k_right: int, dx: float, dy: float) -> bool:
        pu = parent_l[u]
        if pu == -1:
            return False
        s = _best_right(u, k_right, dx, dy)
        return (y_list[u] - y_list[pu]) > s * (w_list[u] - w_list[pu])

    def _solve_bridge(k_left: int, k_right: int, dx: float, dy: float) -> float:
        cur = k_left
        if _move_left(cur, k_right, dx, dy):
            for j in range(log_n - 1, -1, -1):
                cand = up_l[cur][j]
                if cand != -1 and _move_left(cand, k_right, dx, dy):
                    cur = cand
            cur = parent_l[cur]
        return _best_right(cur, k_right, dx, dy)

    gap_p0 = np.empty(u_count + 1, dtype=np.float64)
    gap_p1 = np.empty(u_count + 1, dtype=np.float64)
    tie_p0 = np.empty(u_count, dtype=np.float64)
    tie_p1 = np.empty(u_count, dtype=np.float64)
    for k in range(u_count + 1):
        gap_p0[k] = _solve_bridge(k, k, w_test, 0.0)
        gap_p1[k] = _solve_bridge(k, k, w_test, w_test)
    for k in range(u_count):
        tie_p0[k] = _solve_bridge(k, k + 1, w_test, 0.0)
        tie_p1[k] = _solve_bridge(k, k + 1, w_test, w_test)
    return gap_p0, gap_p1, tie_p0, tie_p1


@dataclass
class VennAbersCalibrator:
    """Predictor probabilístico exacto Inductive Venn-Abers (Vovk y Petej, 2014)."""

    unique_x: np.ndarray
    gap_p0: np.ndarray
    gap_p1: np.ndarray
    tie_p0: np.ndarray
    tie_p1: np.ndarray
    n_cal: int
    effective_n: float
    test_weight: float

    @classmethod
    def fit(
        cls,
        scores: np.ndarray,
        outcomes: np.ndarray,
        *,
        weights: np.ndarray | None = None,
        test_weight: float = 1.0,
    ) -> VennAbersCalibrator:
        """Precalcula `[p0, p1]` exactos para todos los intervalos y empates en O(N log^2 N)."""
        x = np.asarray(scores, dtype=np.float64)
        y = np.asarray(outcomes, dtype=np.float64)
        _require(
            x.ndim == 1
            and y.ndim == 1
            and len(x) == len(y)
            and len(x) > 0
            and bool(np.isfinite(x).all())
            and bool(np.isfinite(y).all())
            and bool(np.all((y >= 0.0) & (y <= 1.0))),
            "VennAbersCalibrator requiere scores finitos e outcomes en [0, 1]",
        )
        if weights is None:
            w = np.ones(len(x), dtype=np.float64)
        else:
            w = np.asarray(weights, dtype=np.float64)
            _require(
                len(w) == len(x) and bool(np.isfinite(w).all()) and bool(np.all(w > 0.0)),
                "weights debe tener la misma longitud y ser estrictamente positivo",
            )
        _require(test_weight > 0.0 and math.isfinite(test_weight), "test_weight debe ser > 0")

        uniq, inv = np.unique(x, return_inverse=True)
        w_u = np.bincount(inv, weights=w, minlength=len(uniq))
        y_u = np.bincount(inv, weights=w * y, minlength=len(uniq))
        w_cum = np.concatenate(([0.0], np.cumsum(w_u, dtype=np.float64)))
        y_cum = np.concatenate(([0.0], np.cumsum(y_u, dtype=np.float64)))

        native = fk.venn_abers_precompute(w_cum, y_cum, float(test_weight))
        if native is not None:
            gap_p0, gap_p1, tie_p0, tie_p1 = native
        else:
            gap_p0, gap_p1, tie_p0, tie_p1 = _venn_abers_precompute_python(
                w_cum, y_cum, float(test_weight)
            )

        sw = float(w.sum())
        n_eff = float((sw * sw) / np.sum(w * w))
        return cls(
            unique_x=uniq,
            gap_p0=gap_p0,
            gap_p1=gap_p1,
            tie_p0=tie_p0,
            tie_p1=tie_p1,
            n_cal=int(len(x)),
            effective_n=n_eff,
            test_weight=float(test_weight),
        )

    @classmethod
    def fit_with_decay(
        cls,
        scores: np.ndarray,
        outcomes: np.ndarray,
        *,
        session_order: np.ndarray,
        half_life: float,
    ) -> VennAbersCalibrator:
        """Venn-Abers ponderado temporalmente con semivida exponencial (Barber et al., 2023)."""
        _require(half_life > 0.0 and math.isfinite(half_life), "half_life debe ser positivo")
        t = np.asarray(session_order, dtype=np.float64)
        _require(len(t) == len(scores) and bool(np.isfinite(t).all()), "session_order incompatible")
        ages = float(t.max()) - t
        weights = np.power(2.0, -ages / float(half_life))
        return cls.fit(scores, outcomes, weights=weights, test_weight=1.0)

    def predict(self, scores: np.ndarray) -> dict[str, np.ndarray]:
        """Evalúa `[p0, p1]`, probabilidad minimax `p1 / (1 - p0 + p1)` y anchura `p1 - p0`."""
        x = np.asarray(scores, dtype=np.float64)
        u_count = len(self.unique_x)
        pos = np.searchsorted(self.unique_x, x, side="left")
        safe_pos = np.minimum(pos, u_count - 1)
        is_tie = (pos < u_count) & (self.unique_x[safe_pos] == x)
        p0 = np.where(is_tie, self.tie_p0[safe_pos], self.gap_p0[pos])
        p1 = np.where(is_tie, self.tie_p1[safe_pos], self.gap_p1[pos])
        p0 = np.clip(p0, 0.0, 1.0)
        p1 = np.clip(np.maximum(p1, p0), 0.0, 1.0)
        denom = 1.0 - p0 + p1
        p_minimax = np.where(denom > 0.0, p1 / denom, 0.5)
        return dict(
            p0=p0,
            p1=p1,
            p_minimax=p_minimax,
            uncertainty=p1 - p0,
        )


def _wilson_ci(hits: int, total: int, z: float = Z95) -> dict[str, Any] | None:
    if total <= 0:
        return None
    p = hits / total
    denom = 1.0 + z * z / total
    centre = (p + z * z / (2.0 * total)) / denom
    half = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denom
    return dict(
        rate=float(p), low=float(centre - half), high=float(centre + half), hits=hits, total=total
    )


@dataclass
class SessionLTTGate:
    """Puerta de fiabilidad por sesión con control de riesgo Learn-then-Test (FWER <= delta)."""

    chosen_threshold: float | None
    target_reliability: float
    delta: float
    margin: float
    block_size: int
    method: str
    tests: list[dict[str, Any]]

    @classmethod
    def fit(
        cls,
        gate_scores: np.ndarray,
        session_metrics: np.ndarray,
        *,
        target_reliability: float = 0.90,
        delta: float = 0.05,
        margin: float = 0.0,
        thresholds: np.ndarray | list[float] | None = None,
        quantiles: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8),
        method: str = "holm",
        block_size: int = 1,
        min_active: int = 5,
    ) -> SessionLTTGate:
        """Elige el umbral causal mínimo que certifica `P(metric > margin | score >= lam) >= p`.

        Los umbrales donde incluso el 100 % de aciertos no puede alcanzar `p <= alpha` (i.e.
        `target_reliability ** n_units > delta`, determinado SOLO por `gate_scores`, sin mirar
        `session_metrics`) se marcan no contrastables sin gastar error de familia (FWER exacto).

        Métodos de control de FWER <= `delta`:
        - `method="holm"` (por defecto): procedimiento step-down de Holm-Bonferroni sobre los
          umbrales contrastables (control exacto de FWER en muestra finita sin exigir monotonía).
        - `method="fixed_sequence"`: prueba secuencial fija desde el umbral contrastable más
          estricto hacia abajo, parando en el primer fallo.
        - `method="bonferroni"`: exige `p <= delta / M_testable` en cada umbral contrastable.
        - `block_size > 1`: agrupa las sesiones seleccionadas en bloques consecutivos no solapados
          y contrasta la media del bloque `> margin` (robusto a autocorrelación).
        """
        _require(0.0 < target_reliability < 1.0, "target_reliability debe estar en (0, 1)")
        _require(0.0 < delta < 1.0, "delta debe estar en (0, 1)")
        _require(method in ("holm", "fixed_sequence", "bonferroni"), f"method inválido: {method!r}")
        _require(type(block_size) is int and block_size >= 1, "block_size debe ser entero >= 1")

        g = np.asarray(gate_scores, dtype=np.float64)
        m = np.asarray(session_metrics, dtype=np.float64)
        _require(len(g) == len(m) and len(g) > 0, "gate_scores y session_metrics incompatibles")
        valid = np.isfinite(g) & np.isfinite(m)
        g = g[valid]
        m = m[valid]

        if thresholds is None:
            raw_grid = [float(np.quantile(g, q)) if q > 0.0 else -math.inf for q in quantiles]
            grid = sorted(set(raw_grid))
        else:
            grid = sorted({float(v) for v in thresholds})
        _require(len(grid) > 0, "La rejilla de umbrales no puede estar vacía")

        # 1) Calcular unidades, aciertos y p-valor binomial exacto por umbral
        records: dict[float, dict[str, Any]] = {}
        testable_thrs: list[float] = []
        # Número mínimo de unidades para que incluso 100% aciertos pueda rechazar a nivel delta
        min_power_units = max(
            min_active, int(math.ceil(math.log(delta) / math.log(target_reliability)))
        )

        for thr in grid:
            active_m = m[g >= thr]
            if block_size > 1:
                n_blk = len(active_m) // block_size
                if n_blk > 0:
                    blk_means = (
                        active_m[: n_blk * block_size].reshape(n_blk, block_size).mean(axis=1)
                    )
                    n_units = int(n_blk)
                    hits = int(np.sum(blk_means > margin))
                else:
                    n_units, hits = 0, 0
            else:
                n_units = int(len(active_m))
                hits = int(np.sum(active_m > margin))

            testable = n_units >= min_power_units
            if testable:
                pval = float(binom.cdf(n_units - hits, n_units, 1.0 - target_reliability))
                testable_thrs.append(thr)
            else:
                pval = 1.0

            records[thr] = dict(
                threshold=float(thr),
                units=n_units,
                hits=hits,
                rate=(hits / n_units) if n_units > 0 else None,
                p_value=pval,
                testable=testable,
                valid=False,
            )

        # 2) Control de FWER exacto sobre los umbrales contrastables (fijados solo por X)
        valid_thresholds: list[float] = []
        m_test = len(testable_thrs)
        if m_test > 0:
            if method == "bonferroni":
                alpha_b = delta / m_test
                for thr in testable_thrs:
                    if records[thr]["p_value"] <= alpha_b:
                        records[thr]["valid"] = True
                        valid_thresholds.append(thr)
            elif method == "holm":
                order_p = sorted(testable_thrs, key=lambda t: (records[t]["p_value"], t))
                for rank_i, thr in enumerate(order_p):
                    if records[thr]["p_value"] <= delta / (m_test - rank_i):
                        records[thr]["valid"] = True
                        valid_thresholds.append(thr)
                    else:
                        break
            else:
                # fixed_sequence: recorrer umbrales contrastables de mayor a menor
                for thr in reversed(testable_thrs):
                    if records[thr]["p_value"] <= delta:
                        records[thr]["valid"] = True
                        valid_thresholds.append(thr)
                    else:
                        break

        chosen = min(valid_thresholds) if valid_thresholds else None
        tests_sorted = [records[thr] for thr in grid]
        return cls(
            chosen_threshold=chosen,
            target_reliability=float(target_reliability),
            delta=float(delta),
            margin=float(margin),
            block_size=block_size,
            method=method,
            tests=tests_sorted,
        )

    def evaluate(self, gate_scores: np.ndarray, session_metrics: np.ndarray) -> dict[str, Any]:
        """Evalúa la puerta fijada en calibración sobre el tramo de evaluación."""
        g = np.asarray(gate_scores, dtype=np.float64)
        m = np.asarray(session_metrics, dtype=np.float64)
        valid = np.isfinite(g) & np.isfinite(m)
        g, m = g[valid], m[valid]
        base_ic = float(m.mean()) if len(m) else None
        base_hit = float(np.mean(m > self.margin)) if len(m) else None
        if self.chosen_threshold is None or len(m) == 0:
            return dict(
                chosen_threshold=self.chosen_threshold,
                sessions=int(len(m)),
                active_sessions=0,
                coverage=0.0,
                base_ic=base_ic,
                base_hit_rate=base_hit,
                active_ic=None,
                unconditional_ic=0.0,
                hit_rate=None,
                wilson=None,
                block_wilson=None,
            )

        mask = g >= self.chosen_threshold
        active_m = m[mask]
        n_act = int(mask.sum())
        cov = n_act / len(m)
        hits = int(np.sum(active_m > self.margin))
        blk_wilson = None
        if self.block_size > 1 and n_act >= self.block_size:
            n_blk = n_act // self.block_size
            blk_means = (
                active_m[: n_blk * self.block_size].reshape(n_blk, self.block_size).mean(axis=1)
            )
            blk_wilson = _wilson_ci(int(np.sum(blk_means > self.margin)), n_blk)
        return dict(
            chosen_threshold=self.chosen_threshold,
            sessions=int(len(m)),
            active_sessions=n_act,
            coverage=float(cov),
            base_ic=base_ic,
            base_hit_rate=base_hit,
            active_ic=float(active_m.mean()) if n_act > 0 else None,
            unconditional_ic=float(np.mean(np.where(mask, m, 0.0))),
            hit_rate=(hits / n_act) if n_act > 0 else None,
            wilson=_wilson_ci(hits, n_act),
            block_wilson=blk_wilson,
        )
