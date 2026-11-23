# ruff: noqa: E501
"""Núcleos nativos C99/OpenMP deterministas (IEEE-754 estricto) para post-entrenamiento.

Se compilan una sola vez en caché (`~/.cache/mars-titan/fast_kernels/`) con
`gcc -O3 -fno-fast-math -march=native -fopenmp -shared -fPIC` (aprovechando SVE/NEON en
DGX Grace ARM64 y AVX2/AVX-512 en Colab x86_64) y ofrecen respaldo transparente si no hay
compilador disponible.

Invariantes:
- Aritmética IEEE-754 estricta (`-fno-fast-math`): ninguna reordenación no determinista.
- El resultado es idéntico con cualquier número de hilos (`1..N`) porque cada hilo procesa
  filas independientes.
- Sin dependencias externas: solo `ctypes` y `gcc` estándar de Linux.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

_C_SOURCE = r"""
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <limits.h>
#ifdef _OPENMP
#include <omp.h>
#endif

/* ============================================================================
 * 1. Conformal ponderado por régimen (`regime_weighted_conformal`):
 *    1 pasada de exp() por fila evaluada (en lugar de 3), orden secuencial en
 *    el primer intervalo con sumas por bloque y recorrido inverso en cuantiles
 *    altos (> 0.5). Cota exacta -50.0 sobre (l - max_l): exp(-50) = 1.9e-22,
 *    muy por debajo de la precisión FP64 relativa a sum_w >= 1.0 y de tol=1e-9.
 * ============================================================================ */
void rwc_session_1d_f64(
    int64_t n_now,
    int64_t n_past,
    int64_t n_intervals,
    const double *now_x,
    const double *past_x,         /* ordenado según orders[0] */
    const double *past_c,         /* -(r_j - r_t)^2 * inv_2h2, en orders[0] */
    double inv_2h2,
    const int64_t *remap_orders,  /* [n_intervals, n_past]: pos en orders[0] */
    const double *sorted_scores,  /* [n_intervals, n_past] */
    const double *nominals,       /* [n_intervals] */
    double tol,
    double *out_corr,             /* [n_now, n_intervals] */
    double *out_eff,              /* [n_now] */
    int n_threads
) {
    if (n_now <= 0 || n_past <= 0 || n_intervals <= 0) return;
    if (n_threads < 1) n_threads = 1;
    int64_t n_blocks = (n_past + 255) / 256;

    #pragma omp parallel num_threads(n_threads)
    {
        double *w = (double *)malloc((size_t)n_past * sizeof(double));
        double *blk_sum = (double *)malloc((size_t)n_blocks * sizeof(double));

        #pragma omp for schedule(dynamic, 8)
        for (int64_t i = 0; i < n_now; ++i) {
            double q = now_x[i];
            double max_l = -INFINITY;
            for (int64_t j = 0; j < n_past; ++j) {
                double dx = past_x[j] - q;
                double l = past_c[j] - dx * dx * inv_2h2;
                if (l > max_l) max_l = l;
            }
            double cutoff = max_l - 50.0;
            double sum_w = 0.0;
            double sum_w2 = 0.0;
            for (int64_t b = 0; b < n_blocks; ++b) {
                int64_t j0 = b * 256;
                int64_t j1 = j0 + 256;
                if (j1 > n_past) j1 = n_past;
                double bs = 0.0;
                double bs2 = 0.0;
                for (int64_t j = j0; j < j1; ++j) {
                    if (past_c[j] <= cutoff) {
                        w[j] = 0.0;
                        continue;
                    }
                    double dx = past_x[j] - q;
                    double l = past_c[j] - dx * dx * inv_2h2;
                    if (l <= cutoff) {
                        w[j] = 0.0;
                        continue;
                    }
                    double wj = exp(l - max_l);
                    w[j] = wj;
                    bs += wj;
                    bs2 += wj * wj;
                }
                blk_sum[b] = bs;
                sum_w += bs;
                sum_w2 += bs2;
            }
            out_eff[i] = (sum_w2 > 0.0) ? ((sum_w * sum_w) / sum_w2) : 0.0;

            /* Intervalo 0: recorrido por bloques sobre w[] secuencial */
            {
                double thresh = (nominals[0] - tol) * sum_w;
                double acc = 0.0;
                int64_t b = 0;
                while (b + 1 < n_blocks && acc + blk_sum[b] < thresh) {
                    acc += blk_sum[b];
                    ++b;
                }
                int64_t j0 = b * 256;
                int64_t j1 = j0 + 256;
                if (j1 > n_past) j1 = n_past;
                int64_t idx = n_past - 1;
                for (int64_t j = j0; j < j1; ++j) {
                    acc += w[j];
                    if (acc >= thresh) {
                        idx = j;
                        break;
                    }
                }
                out_corr[i * n_intervals + 0] = sorted_scores[idx];
            }

            /* Intervalos k >= 1 */
            for (int64_t k = 1; k < n_intervals; ++k) {
                const int64_t *rmap = remap_orders + k * n_past;
                double nom = nominals[k];
                int64_t idx = n_past - 1;
                if (nom > 0.5) {
                    double tail_lim = (1.0 - nom + tol) * sum_w;
                    double tail = 0.0;
                    int64_t p = n_past - 1;
                    while (p > 0) {
                        double next_tail = tail + w[rmap[p]];
                        if (next_tail > tail_lim) break;
                        tail = next_tail;
                        --p;
                    }
                    idx = p;
                } else {
                    double thresh = (nom - tol) * sum_w;
                    double acc = 0.0;
                    for (int64_t p = 0; p < n_past; ++p) {
                        acc += w[rmap[p]];
                        if (acc >= thresh) {
                            idx = p;
                            break;
                        }
                    }
                }
                out_corr[i * n_intervals + k] = sorted_scores[k * n_past + idx];
            }
        }
        free(blk_sum);
        free(w);
    }
}

/* ============================================================================
 * 2. Memoria por vecinos (`neighbour_residual_memory`):
 *    Top-k determinista en FP32 con desempate por índice más antiguo del banco
 *    (menor índice global) + softmax FP64 en orden descendente exacto + suma
 *    ponderada para M modelos (y placebo opcional) sin materializar tensores
 *    intermedios de [rows, k, models].
 * ============================================================================ */
static inline int nrm_better(float v1, int64_t i1, float v2, int64_t i2) {
    return (v1 > v2) || (v1 == v2 && i1 < i2);
}

static inline void nrm_sift_down(float *hv, int64_t *hi, int64_t k, int64_t pos) {
    float v = hv[pos];
    int64_t idx = hi[pos];
    while (1) {
        int64_t left = 2 * pos + 1;
        if (left >= k) break;
        int64_t right = left + 1;
        int64_t worst = left;
        if (right < k && nrm_better(hv[left], hi[left], hv[right], hi[right])) {
            worst = right;
        }
        if (nrm_better(v, idx, hv[worst], hi[worst])) {
            hv[pos] = hv[worst];
            hi[pos] = hi[worst];
            pos = worst;
        } else {
            break;
        }
    }
    hv[pos] = v;
    hi[pos] = idx;
}

void nrm_heap_update_tile_f32(
    int64_t rows,
    int64_t tile_size,
    int64_t col_offset,
    int64_t k,
    const float *sims,       /* [rows, tile_size] */
    int exclude_own,
    const int64_t *q_assets, /* [rows] o NULL */
    const int64_t *b_assets, /* [tile_size] o NULL */
    float *heap_v,           /* [rows, k] */
    int64_t *heap_i,         /* [rows, k] */
    int64_t *heap_filled,    /* [rows] */
    int n_threads
) {
    if (rows <= 0 || tile_size <= 0 || k <= 0) return;
    if (n_threads < 1) n_threads = 1;

    #pragma omp parallel for schedule(static) num_threads(n_threads)
    for (int64_t r = 0; r < rows; ++r) {
        const float *row = sims + r * tile_size;
        float *hv = heap_v + r * k;
        int64_t *hi = heap_i + r * k;
        int64_t filled = heap_filled[r];
        float root_v = (filled == k) ? hv[0] : -INFINITY;
        int64_t root_i = (filled == k) ? hi[0] : INT64_MAX;
        int64_t qa = exclude_own ? q_assets[r] : -1;

        for (int64_t j = 0; j < tile_size; ++j) {
            float v = (exclude_own && b_assets[j] == qa) ? -INFINITY : row[j];
            int64_t gidx = col_offset + j;
            if (filled < k) {
                hv[filled] = v;
                hi[filled] = gidx;
                ++filled;
                if (filled == k) {
                    for (int64_t p = k / 2 - 1; p >= 0; --p) nrm_sift_down(hv, hi, k, p);
                    root_v = hv[0];
                    root_i = hi[0];
                }
            } else if (v > root_v || (v == root_v && gidx < root_i)) {
                hv[0] = v;
                hi[0] = gidx;
                nrm_sift_down(hv, hi, k, 0);
                root_v = hv[0];
                root_i = hi[0];
            }
        }
        heap_filled[r] = filled;
    }
}

void nrm_heap_finalize_gather_f64(
    int64_t rows,
    int64_t k,
    int64_t models,
    float *heap_v,            /* [rows, k] */
    int64_t *heap_i,          /* [rows, k] */
    const int64_t *heap_filled,
    const double *values,     /* [total_bank, models] */
    const double *shuffled,   /* [total_bank, models] o NULL */
    double tau,
    double *out_reads,        /* [rows, models] */
    double *out_placebo,      /* [rows, models] o NULL */
    int n_threads
) {
    if (rows <= 0 || k <= 0 || models <= 0) return;
    if (n_threads < 1) n_threads = 1;

    #pragma omp parallel num_threads(n_threads)
    {
        float *sv = (float *)malloc((size_t)k * sizeof(float));
        int64_t *si = (int64_t *)malloc((size_t)k * sizeof(int64_t));
        double *w = (double *)malloc((size_t)k * sizeof(double));

        #pragma omp for schedule(static)
        for (int64_t r = 0; r < rows; ++r) {
            float *hv = heap_v + r * k;
            int64_t *hi = heap_i + r * k;
            int64_t filled = heap_filled[r];
            if (filled < k && filled > 0) {
                for (int64_t p = filled / 2 - 1; p >= 0; --p) nrm_sift_down(hv, hi, filled, p);
            }
            int64_t n_cur = filled;
            for (int64_t pos = n_cur - 1; pos >= 0; --pos) {
                sv[pos] = hv[0];
                si[pos] = hi[0];
                --n_cur;
                if (n_cur > 0) {
                    hv[0] = hv[n_cur];
                    hi[0] = hi[n_cur];
                    nrm_sift_down(hv, hi, n_cur, 0);
                }
            }
            double best = (filled > 0) ? (double)sv[0] : -INFINITY;
            double sum_w = 0.0;
            if (isfinite(best)) {
                for (int64_t p = 0; p < filled; ++p) {
                    double vp = (double)sv[p];
                    double wp = isfinite(vp) ? exp((vp - best) / tau) : 0.0;
                    w[p] = wp;
                    sum_w += wp;
                }
            }
            double *orow = out_reads + r * models;
            for (int64_t m = 0; m < models; ++m) orow[m] = 0.0;
            if (sum_w > 0.0) {
                double inv = 1.0 / sum_w;
                for (int64_t p = 0; p < filled; ++p) {
                    double wp = w[p] * inv;
                    const double *vrow = values + si[p] * models;
                    for (int64_t m = 0; m < models; ++m) orow[m] += wp * vrow[m];
                }
            }
            if (shuffled != NULL && out_placebo != NULL) {
                double *prow = out_placebo + r * models;
                for (int64_t m = 0; m < models; ++m) prow[m] = 0.0;
                if (sum_w > 0.0) {
                    double inv = 1.0 / sum_w;
                    for (int64_t p = 0; p < filled; ++p) {
                        double wp = w[p] * inv;
                        const double *srow = shuffled + si[p] * models;
                        for (int64_t m = 0; m < models; ++m) prow[m] += wp * srow[m];
                    }
                }
            }
        }
        free(w);
        free(si);
        free(sv);
    }
}

/* ============================================================================
 * 3. Variante con decaimiento temporal causal opcional en memoria por vecinos:
 *    w_ij = softmax_j(q_i·q_j / tau - age_j * log(2) / recency_half_life)
 * ============================================================================ */
void nrm_heap_finalize_gather_recency_f64(
    int64_t rows,
    int64_t k,
    int64_t models,
    float *heap_v,
    int64_t *heap_i,
    const int64_t *heap_filled,
    const double *values,
    const double *shuffled,
    const double *recency_bias,  /* [total_bank]: -age * ln(2) / half_life */
    double tau,
    double *out_reads,
    double *out_placebo,
    int n_threads
) {
    if (rows <= 0 || k <= 0 || models <= 0) return;
    if (n_threads < 1) n_threads = 1;

    #pragma omp parallel num_threads(n_threads)
    {
        float *sv = (float *)malloc((size_t)k * sizeof(float));
        int64_t *si = (int64_t *)malloc((size_t)k * sizeof(int64_t));
        double *logits = (double *)malloc((size_t)k * sizeof(double));
        double *w = (double *)malloc((size_t)k * sizeof(double));

        #pragma omp for schedule(static)
        for (int64_t r = 0; r < rows; ++r) {
            float *hv = heap_v + r * k;
            int64_t *hi = heap_i + r * k;
            int64_t filled = heap_filled[r];
            if (filled < k && filled > 0) {
                for (int64_t p = filled / 2 - 1; p >= 0; --p) nrm_sift_down(hv, hi, filled, p);
            }
            int64_t n_cur = filled;
            for (int64_t pos = n_cur - 1; pos >= 0; --pos) {
                sv[pos] = hv[0];
                si[pos] = hi[0];
                --n_cur;
                if (n_cur > 0) {
                    hv[0] = hv[n_cur];
                    hi[0] = hi[n_cur];
                    nrm_sift_down(hv, hi, n_cur, 0);
                }
            }
            double best = -INFINITY;
            for (int64_t p = 0; p < filled; ++p) {
                double vp = (double)sv[p];
                double lp = isfinite(vp) ? (vp / tau + recency_bias[si[p]]) : -INFINITY;
                logits[p] = lp;
                if (lp > best) best = lp;
            }
            double sum_w = 0.0;
            if (isfinite(best)) {
                for (int64_t p = 0; p < filled; ++p) {
                    double lp = logits[p];
                    double wp = isfinite(lp) ? exp(lp - best) : 0.0;
                    w[p] = wp;
                    sum_w += wp;
                }
            }
            double *orow = out_reads + r * models;
            for (int64_t m = 0; m < models; ++m) orow[m] = 0.0;
            if (sum_w > 0.0) {
                double inv = 1.0 / sum_w;
                for (int64_t p = 0; p < filled; ++p) {
                    double wp = w[p] * inv;
                    const double *vrow = values + si[p] * models;
                    for (int64_t m = 0; m < models; ++m) orow[m] += wp * vrow[m];
                }
            }
            if (shuffled != NULL && out_placebo != NULL) {
                double *prow = out_placebo + r * models;
                for (int64_t m = 0; m < models; ++m) prow[m] = 0.0;
                if (sum_w > 0.0) {
                    double inv = 1.0 / sum_w;
                    for (int64_t p = 0; p < filled; ++p) {
                        double wp = w[p] * inv;
                        const double *srow = shuffled + si[p] * models;
                        for (int64_t m = 0; m < models; ++m) prow[m] += wp * srow[m];
                    }
                }
            }
        }
        free(w);
        free(logits);
        free(si);
        free(sv);
    }
}

/* ============================================================================
 * 5. Exact Inductive Venn-Abers O(U log^2 U) via persistent convex hull trees
 * ============================================================================
 */

static inline int va_move_right(
    int64_t u,
    int64_t w,
    double dx,
    double dy,
    const double *W,
    const double *Y,
    const int64_t *next_r
) {
    int64_t nw = next_r[w];
    if (nw < 0) return 0;
    double lhs = (Y[nw] - Y[w]) * ((W[w] + dx) - W[u]);
    double rhs = ((Y[w] + dy) - Y[u]) * (W[nw] - W[w]);
    return lhs < rhs;
}

static inline double va_best_right(
    int64_t u,
    int64_t k_right,
    double dx,
    double dy,
    int log_n,
    const double *W,
    const double *Y,
    const int64_t *next_r,
    const int64_t *up_r
) {
    int64_t cur = k_right;
    if (va_move_right(u, cur, dx, dy, W, Y, next_r)) {
        for (int j = log_n - 1; j >= 0; --j) {
            int64_t cand = up_r[cur * log_n + j];
            if (cand >= 0 && va_move_right(u, cand, dx, dy, W, Y, next_r)) {
                cur = cand;
            }
        }
        cur = next_r[cur];
    }
    return ((Y[cur] + dy) - Y[u]) / ((W[cur] + dx) - W[u]);
}

static inline int va_move_left(
    int64_t u,
    int64_t k_right,
    double dx,
    double dy,
    int log_n,
    const double *W,
    const double *Y,
    const int64_t *parent_l,
    const int64_t *next_r,
    const int64_t *up_r
) {
    int64_t pu = parent_l[u];
    if (pu < 0) return 0;
    double s = va_best_right(u, k_right, dx, dy, log_n, W, Y, next_r, up_r);
    return (Y[u] - Y[pu]) > s * (W[u] - W[pu]);
}

static inline double va_solve_bridge(
    int64_t k_left,
    int64_t k_right,
    double dx,
    double dy,
    int log_n,
    const double *W,
    const double *Y,
    const int64_t *parent_l,
    const int64_t *up_l,
    const int64_t *next_r,
    const int64_t *up_r
) {
    int64_t cur = k_left;
    if (va_move_left(cur, k_right, dx, dy, log_n, W, Y, parent_l, next_r, up_r)) {
        for (int j = log_n - 1; j >= 0; --j) {
            int64_t cand = up_l[cur * log_n + j];
            if (cand >= 0 && va_move_left(cand, k_right, dx, dy, log_n, W, Y, parent_l, next_r, up_r)) {
                cur = cand;
            }
        }
        cur = parent_l[cur];
    }
    return va_best_right(cur, k_right, dx, dy, log_n, W, Y, next_r, up_r);
}

int venn_abers_precompute_f64(
    int64_t u_count,
    const double *W,
    const double *Y,
    double w_test,
    double *gap_p0,
    double *gap_p1,
    double *tie_p0,
    double *tie_p1,
    int num_threads
) {
    int64_t n = u_count + 1;
    int log_n = 1;
    while ((1LL << log_n) <= n) log_n++;

    int64_t *parent_l = (int64_t *)malloc((size_t)n * sizeof(int64_t));
    int64_t *next_r = (int64_t *)malloc((size_t)n * sizeof(int64_t));
    int64_t *stack = (int64_t *)malloc((size_t)n * sizeof(int64_t));
    int64_t *up_l = (int64_t *)malloc((size_t)n * (size_t)log_n * sizeof(int64_t));
    int64_t *up_r = (int64_t *)malloc((size_t)n * (size_t)log_n * sizeof(int64_t));
    if (!parent_l || !next_r || !stack || !up_l || !up_r) {
        free(parent_l); free(next_r); free(stack); free(up_l); free(up_r);
        return -1;
    }

    /* Prefix lower convex hull tree */
    parent_l[0] = -1;
    int64_t sp = 0;
    stack[sp++] = 0;
    for (int64_t k = 1; k <= u_count; ++k) {
        double wk = W[k], yk = Y[k];
        while (sp >= 2) {
            int64_t a = stack[sp - 2];
            int64_t b = stack[sp - 1];
            if ((Y[b] - Y[a]) * (wk - W[b]) >= (yk - Y[b]) * (W[b] - W[a])) {
                sp--;
            } else {
                break;
            }
        }
        parent_l[k] = stack[sp - 1];
        stack[sp++] = k;
    }
    for (int64_t v = 0; v <= u_count; ++v) {
        up_l[v * log_n + 0] = parent_l[v];
        for (int j = 1; j < log_n; ++j) {
            int64_t p = up_l[v * log_n + j - 1];
            up_l[v * log_n + j] = (p < 0) ? -1 : up_l[p * log_n + j - 1];
        }
    }

    /* Suffix lower convex hull tree */
    next_r[u_count] = -1;
    sp = 0;
    stack[sp++] = u_count;
    for (int64_t k = u_count - 1; k >= 0; --k) {
        double wk = W[k], yk = Y[k];
        while (sp >= 2) {
            int64_t b = stack[sp - 1];
            int64_t c = stack[sp - 2];
            if ((Y[b] - yk) * (W[c] - W[b]) >= (Y[c] - Y[b]) * (W[b] - wk)) {
                sp--;
            } else {
                break;
            }
        }
        next_r[k] = stack[sp - 1];
        stack[sp++] = k;
    }
    for (int64_t v = u_count; v >= 0; --v) {
        up_r[v * log_n + 0] = next_r[v];
        for (int j = 1; j < log_n; ++j) {
            int64_t nxt = up_r[v * log_n + j - 1];
            up_r[v * log_n + j] = (nxt < 0) ? -1 : up_r[nxt * log_n + j - 1];
        }
    }

    #ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(num_threads) if(num_threads > 1 && u_count >= 512)
    #endif
    for (int64_t k = 0; k <= u_count; ++k) {
        gap_p0[k] = va_solve_bridge(k, k, w_test, 0.0, log_n, W, Y, parent_l, up_l, next_r, up_r);
        gap_p1[k] = va_solve_bridge(k, k, w_test, w_test, log_n, W, Y, parent_l, up_l, next_r, up_r);
        if (k < u_count) {
            tie_p0[k] = va_solve_bridge(k, k + 1, w_test, 0.0, log_n, W, Y, parent_l, up_l, next_r, up_r);
            tie_p1[k] = va_solve_bridge(k, k + 1, w_test, w_test, log_n, W, Y, parent_l, up_l, next_r, up_r);
        }
    }

    free(parent_l);
    free(next_r);
    free(stack);
    free(up_l);
    free(up_r);
    return 0;
}
"""

_LIB = None
_TRIED = False


def _cache_dir() -> Path:
    base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return base / "mars-titan" / "fast_kernels"


def load_library() -> ctypes.CDLL | None:
    """Compila (con caché por huella SHA-256) y carga los núcleos C99/OpenMP, o None si falla."""
    global _LIB, _TRIED
    if _TRIED:
        return _LIB
    _TRIED = True
    if os.environ.get("MARS_TITAN_DISABLE_NATIVE_POSTTRAINING", "0") == "1":
        return None
    compiler = shutil.which("gcc") or shutil.which("cc")
    if compiler is None:
        return None
    key = hashlib.sha256(_C_SOURCE.encode("utf-8")).hexdigest()[:20]
    cache = _cache_dir()
    so_path = cache / f"posttraining_fast_{key}.so"
    try:
        if not so_path.is_file():
            cache.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(dir=cache) as tmp:
                c_path = Path(tmp) / "kernels.c"
                out_path = Path(tmp) / "kernels.so"
                c_path.write_text(_C_SOURCE, encoding="utf-8")
                cmd = [
                    compiler,
                    "-O3",
                    "-fno-fast-math",
                    "-march=native",
                    "-fopenmp",
                    "-shared",
                    "-fPIC",
                    str(c_path),
                    "-lm",
                    "-o",
                    str(out_path),
                ]
                done = subprocess.run(cmd, capture_output=True, check=False)
                if done.returncode != 0:
                    # Respaldo sin -fopenmp ni -march=native si el compilador es antiguo
                    cmd_min = [
                        compiler,
                        "-O3",
                        "-fno-fast-math",
                        "-shared",
                        "-fPIC",
                        str(c_path),
                        "-lm",
                        "-o",
                        str(out_path),
                    ]
                    done = subprocess.run(cmd_min, capture_output=True, check=False)
                    if done.returncode != 0:
                        return None
                os.replace(out_path, so_path)
        lib = ctypes.CDLL(str(so_path))
        lib.rwc_session_1d_f64.argtypes = [
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_double,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_double,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        lib.nrm_heap_update_tile_f32.argtypes = [
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        lib.nrm_heap_finalize_gather_f64.argtypes = [
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_double,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        lib.nrm_heap_finalize_gather_recency_f64.argtypes = [
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_double,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        lib.venn_abers_precompute_f64.argtypes = [
            ctypes.c_int64,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_double,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        lib.venn_abers_precompute_f64.restype = ctypes.c_int
        _LIB = lib
    except OSError:
        _LIB = None
    return _LIB


def default_threads(limit: int = 8) -> int:
    """Hilos por defecto respetando el entorno (`OMP_NUM_THREADS` o `MARS_TITAN_POST_THREADS`)."""
    raw = os.environ.get("MARS_TITAN_POST_THREADS") or os.environ.get("OMP_NUM_THREADS")
    if raw:
        try:
            val = int(raw)
            if val >= 1:
                return min(val, limit)
        except ValueError:
            pass
    cpus = os.cpu_count() or 1
    return max(1, min(cpus // 2, limit))


def rwc_session_1d(
    now_x: np.ndarray,
    past_x_ord0: np.ndarray,
    past_c_ord0: np.ndarray,
    inv_2h2: float,
    remap_orders: np.ndarray,
    sorted_scores: np.ndarray,
    nominals: np.ndarray,
    tol: float,
    *,
    threads: int | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Ejecuta el núcleo C99 exacto para una sesión de conformal ponderado por régimen."""
    lib = load_library()
    if lib is None:
        return None
    now_x = np.ascontiguousarray(now_x, dtype=np.float64)
    past_x_ord0 = np.ascontiguousarray(past_x_ord0, dtype=np.float64)
    past_c_ord0 = np.ascontiguousarray(past_c_ord0, dtype=np.float64)
    remap_orders = np.ascontiguousarray(remap_orders, dtype=np.int64)
    sorted_scores = np.ascontiguousarray(sorted_scores, dtype=np.float64)
    nominals = np.ascontiguousarray(nominals, dtype=np.float64)
    n_now = len(now_x)
    n_past = len(past_x_ord0)
    n_intervals = len(nominals)
    out_corr = np.empty((n_now, n_intervals), dtype=np.float64)
    out_eff = np.empty(n_now, dtype=np.float64)
    nt = default_threads(8) if threads is None else max(1, int(threads))
    lib.rwc_session_1d_f64(
        ctypes.c_int64(n_now),
        ctypes.c_int64(n_past),
        ctypes.c_int64(n_intervals),
        now_x.ctypes.data_as(ctypes.c_void_p),
        past_x_ord0.ctypes.data_as(ctypes.c_void_p),
        past_c_ord0.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_double(float(inv_2h2)),
        remap_orders.ctypes.data_as(ctypes.c_void_p),
        sorted_scores.ctypes.data_as(ctypes.c_void_p),
        nominals.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_double(float(tol)),
        ctypes.c_void_p(out_corr.ctypes.data),
        ctypes.c_void_p(out_eff.ctypes.data),
        ctypes.c_int(nt),
    )
    return out_corr, out_eff


def venn_abers_precompute(
    w_cum: np.ndarray,
    y_cum: np.ndarray,
    w_test: float,
    *,
    threads: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Precalcula en C99/OpenMP los 4 arrays exactos de Venn-Abers en O(U log^2 U)."""
    lib = load_library()
    if lib is None:
        return None
    w_c = np.ascontiguousarray(w_cum, dtype=np.float64)
    y_c = np.ascontiguousarray(y_cum, dtype=np.float64)
    u_count = len(w_c) - 1
    if u_count <= 0 or len(y_c) != len(w_c):
        return None
    gap_p0 = np.empty(u_count + 1, dtype=np.float64)
    gap_p1 = np.empty(u_count + 1, dtype=np.float64)
    tie_p0 = np.empty(u_count, dtype=np.float64)
    tie_p1 = np.empty(u_count, dtype=np.float64)
    nt = default_threads(4) if threads is None else max(1, int(threads))
    rc = lib.venn_abers_precompute_f64(
        ctypes.c_int64(u_count),
        w_c.ctypes.data_as(ctypes.c_void_p),
        y_c.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_double(float(w_test)),
        gap_p0.ctypes.data_as(ctypes.c_void_p),
        gap_p1.ctypes.data_as(ctypes.c_void_p),
        tie_p0.ctypes.data_as(ctypes.c_void_p),
        tie_p1.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(nt),
    )
    if rc != 0:
        return None
    return gap_p0, gap_p1, tie_p0, tie_p1
