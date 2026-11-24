"""Física de mercado v4 vectorizada en PyTorch GPU/CPU y almacén multimodal zero-copy.

Preserva paridad matemática y causalidad estricta bit-a-bit respecto a `market_physics_v4.py`,
eliminando bucles Python sobre los `N` activos para saturar los 188 SMs y 1.792 GB/s GDDR7 de la
NVIDIA RTX PRO 6000 Blackwell (`sm_120`):
  1. `batched_asset_microstructure_v4_torch`: calcula los 24 rasgos de `ASSET_V4_NAMES` para todos
     los `N` activos a la vez (`[T, N] -> [T, N, 24]`) mediante sumas acumuladas O(T*N), varianza
     centrada anti-cancelación, máscara Day-0 en `ret_cc[0]`/`ret_co[0]` y `F.conv1d` cacheado
     para diferenciación fraccionaria (FFD) en FP32 por defecto (FP64 opcional para paridad).
  2. `batched_panel_physics_v4_torch`: RMT Marchenko-Pastur con ruta rápida exacta de Gram dual
     `W×W` cuando `W < N` (253× menos FLOPs en `N=2600, W=63`) y fallback por lotes FP32 cuando
     existen NaNs parciales; transporte óptimo de Wasserstein 1D causal en FP32 (2× más rápido);
     Lead-Lag centrado con contracción James-Stein; sorpresas macro EWMA + betas ridge por lotes
     y `high_snr_context [T, N, 8]`.
  3. `batched_multimodal_fusion_v4_torch`: novedad de noticias, deriva visual, frescura PIT y
     tailwind macro en O(log T) con decay broadcastable `[T, N, 1]` y chunking sobre `N` (<1 GB
     VRAM incluso en `T=500, N=2600, D=768`).
  4. `ZeroCopySessionStore`: almacenamiento `mmap` (`np.load(..., mmap_mode="r")`) compartido en
     page-cache del SO entre 6–8 workers CUDA MPS con búferes fijados high-water `max_N` sin
     reasignaciones en paneles ragged, staging por slabs contiguos pre-fijados con hilo de copia
     asíncrono (`>25 GB/s`) y modo opcional `gpu_resident=True`.
"""

from __future__ import annotations

import contextlib
import ctypes
import hashlib
import json
import math
import warnings
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F

from .market_physics_v4 import ASSET_V4_NAMES, LIMIT, MULTIMODAL_V4_NAMES, _ffd_weights

STORE_ARRAY_KEYS: tuple[str, ...] = (
    "asset_ids",
    "prices",
    "fundamentals",
    "macro",
    "news",
    "charts",
    "modality_mask",
    "physics_v4",
    "high_snr_context",
    "exposures",
    "rmt_loadings",
    "target_returns",
)

RMT_REGIME_EXTRA_NAMES: tuple[str, ...] = (
    "rmt_gap_ratio_1_2",
    "rmt_top1_vel_21",
)

_STORE_SCHEMA_VERSION = "mmarrs_v4_zerocopy_mmap_1"
_L2_CHUNK_TARGET_ELEMS = 4_000_000  # ~32 MB en FP64 -> cabe íntegro en la L2 de 128 MB de sm_120
_RMT_MATRIX_CHUNK_ELEMS = 8_000_000  # ~64 MB en FP64 para matrices N×N o lotes W×N
_FFD_WEIGHT_CACHE: dict[tuple[float, torch.dtype, torch.device], torch.Tensor] = {}


def _l2_chunk_elems(dtype: torch.dtype) -> int:
    """Presupuesto de elementos L2 (~32 MB FP64 / ~64 MB FP32 dentro de 128 MB L2 `sm_120`)."""
    return _L2_CHUNK_TARGET_ELEMS if dtype == torch.float64 else _L2_CHUNK_TARGET_ELEMS * 4


def _as_float_tensor(
    x: torch.Tensor | np.ndarray,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        target_dev = x.device if device is None else device
        if x.dtype == dtype and x.device == target_dev:
            return x
        return x.to(device=target_dev, dtype=dtype)
    np_dtype = np.float64 if dtype == torch.float64 else np.float32
    t = torch.as_tensor(np.asarray(x, dtype=np_dtype), dtype=dtype)
    return t.to(device=device) if device is not None else t


def _as_f64_tensor(
    x: torch.Tensor | np.ndarray,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    return _as_float_tensor(x, dtype=torch.float64, device=device)


def _as_bool_tensor(
    x: torch.Tensor | np.ndarray,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        t = x.to(dtype=torch.bool)
        return t.to(device=device) if (device is not None and t.device != device) else t
    t = torch.as_tensor(np.asarray(x, dtype=bool), dtype=torch.bool)
    return t.to(device=device) if device is not None else t


def _to_numpy_contiguous(x: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return np.ascontiguousarray(x.detach().cpu().numpy())
    return np.ascontiguousarray(np.asarray(x))


def _zero_copy_tensor(arr: np.ndarray) -> torch.Tensor:
    """Convierte vista NumPy/memmap solo lectura a Tensor zero-copy sin avisos ruidosos."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The given NumPy array is not writable",
            category=UserWarning,
        )
        return torch.from_numpy(arr)


def _get_ffd_weight_tensor(
    d: float,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Caché de pesos FFD residentes en dispositivo para evitar copias H2D/syncs en cada llamada."""
    key = (float(d), dtype, device)
    cached = _FFD_WEIGHT_CACHE.get(key)
    if cached is not None:
        return cached
    w_np = _ffd_weights(d)
    w_rev = torch.from_numpy(np.ascontiguousarray(w_np[::-1])).to(device=device, dtype=dtype)
    w_rev = w_rev.view(1, 1, len(w_np)).contiguous()
    _FFD_WEIGHT_CACHE[key] = w_rev
    return w_rev


def _rolling_sum_2d(x: torch.Tensor, window: int) -> torch.Tensor:
    """Suma rodante causal O(T*N) sobre eje temporal 0 para tensor `[T, N]`."""
    t_len, n_assets = x.shape
    if t_len == 0 or window <= 0:
        return torch.zeros_like(x)
    clean = torch.where(torch.isfinite(x), x, 0.0)
    zero = torch.zeros((1, n_assets), dtype=x.dtype, device=x.device)
    csum = torch.cat([zero, torch.cumsum(clean, dim=0)], dim=0)
    if window >= t_len:
        return csum[1:]
    out = csum[1:].clone()
    out[window:] = csum[window + 1 :] - csum[1 : t_len - window + 1]
    return out


def _rolling_mean_2d(x: torch.Tensor, window: int) -> torch.Tensor:
    """Media rodante causal O(T*N) ignorando valores no finitos."""
    fin = torch.isfinite(x)
    cnt = _rolling_sum_2d(fin.to(x.dtype), window)
    tot = _rolling_sum_2d(torch.where(fin, x, 0.0), window)
    return torch.where(cnt > 0.0, tot / cnt.clamp_min(1.0), 0.0)


def _rolling_var_2d(x: torch.Tensor, window: int) -> torch.Tensor:
    """Varianza muestral rodante causal O(T*N) con anclaje causal por bloques (cero cancelación)."""
    t_len, n_assets = x.shape
    out = torch.zeros_like(x)
    if t_len == 0 or n_assets == 0 or window < 2:
        return out
    block_t = 256
    fin_all = torch.isfinite(x)
    zero = torch.zeros((1, n_assets), dtype=torch.float64, device=x.device)
    for b0 in range(0, t_len, block_t):
        b1 = min(b0 + block_t, t_len)
        s0 = max(0, b0 - window + 1)
        x_blk = x[s0:b1].to(torch.float64)
        f_blk = fin_all[s0:b1]
        # Ancla estrictamente causal en s0 (<= b0): elimina desplazamientos grandes antes de x^2
        ax = torch.where(f_blk[0:1], x_blk[0:1], 0.0)
        xs = torch.where(f_blk, x_blk - ax, 0.0)
        l_blk = b1 - s0
        c_cnt = torch.cat([zero, torch.cumsum(f_blk.to(torch.float64), dim=0)], dim=0)
        c_s1 = torch.cat([zero, torch.cumsum(xs, dim=0)], dim=0)
        c_s2 = torch.cat([zero, torch.cumsum(xs * xs, dim=0)], dim=0)
        hi = torch.arange(b0 - s0 + 1, l_blk + 1, device=x.device)
        lo = (hi - window).clamp_min(0)
        cnt = c_cnt[hi] - c_cnt[lo]
        s1 = c_s1[hi] - c_s1[lo]
        s2 = c_s2[hi] - c_s2[lo]
        denom = cnt.clamp_min(1.0)
        ss = (s2 - (s1 * s1) / denom).clamp_min(0.0)
        ss = torch.where(ss <= 1e-12 * s2, 0.0, ss)
        var_b = torch.where(cnt >= 2.0, ss / (cnt - 1.0).clamp_min(1.0), 0.0)
        out[b0:b1] = var_b.to(dtype=x.dtype)
    return out


def _apply_ffd_2d(series: torch.Tensor, d: float) -> torch.Tensor:
    """Convolución causal 1D por lotes (`F.conv1d`) con pesos cacheados para FFD."""
    t_len, n_assets = series.shape
    out = torch.zeros_like(series)
    if t_len == 0 or n_assets == 0:
        return out
    w_rev = _get_ffd_weight_tensor(d, series.dtype, series.device)
    k = int(w_rev.shape[-1])
    if t_len >= k:
        x_n1t = series.transpose(0, 1).unsqueeze(1).contiguous()
        conv = F.conv1d(x_n1t, w_rev)  # [N, 1, T - K + 1]
        out[k - 1 :] = conv.squeeze(1).transpose(0, 1)
    std = torch.sqrt(_rolling_var_2d(out, 63))
    return torch.where(std > 1e-12, out / std.clamp_min(1e-12), 0.0)


def _rolling_levy_area_2d(
    x: torch.Tensor,
    y: torch.Tensor,
    window: int,
    *,
    dx_in: torch.Tensor | None = None,
    dy_in: torch.Tensor | None = None,
) -> torch.Tensor:
    """Área firmada de Lévy normalizada causal O(T*N) con anclaje causal por bloques."""
    t_len, n_assets = x.shape
    out = torch.zeros_like(x)
    if t_len < window or window < 2 or n_assets == 0:
        return out
    w_eff = window - 1
    dx_src = (x[1:] - x[:-1]) if dx_in is None else dx_in
    dy_src = (y[1:] - y[:-1]) if dy_in is None else dy_in
    dx_full = torch.where(torch.isfinite(dx_src), dx_src, 0.0)
    dy_full = torch.where(torch.isfinite(dy_src), dy_src, 0.0)

    # Bloques causales de 256 sesiones en FP64 O(T*N): elimina deriva de cumsum y 30x más rápido
    block_t = 256
    zero = torch.zeros((1, n_assets), dtype=torch.float64, device=x.device)
    for b0 in range(window - 1, t_len, block_t):
        b1 = min(b0 + block_t, t_len)
        s0 = b0 - w_eff  # índice inicial en [0, T-1] necesario para la primera ventana del bloque
        dx_blk = dx_full[s0 : b1 - 1].to(torch.float64)  # [L-1, N]
        dy_blk = dy_full[s0 : b1 - 1].to(torch.float64)  # [L-1, N]
        # Reconstruir trayectoria anclada en 0 al inicio del bloque s0 (cero offset DC)
        x_rel = torch.cat([zero, torch.cumsum(dx_blk, dim=0)], dim=0)  # [L, N]
        y_rel = torch.cat([zero, torch.cumsum(dy_blk, dim=0)], dim=0)  # [L, N]
        cross = x_rel[:-1] * dy_blk - y_rel[:-1] * dx_blk  # [L-1, N]

        c_cross = torch.cat([zero, torch.cumsum(cross, dim=0)], dim=0)  # [L, N]
        c_dx = x_rel
        c_dy = y_rel
        c_dx2 = torch.cat([zero, torch.cumsum(dx_blk * dx_blk, dim=0)], dim=0)
        c_dy2 = torch.cat([zero, torch.cumsum(dy_blk * dy_blk, dim=0)], dim=0)

        hi = torch.arange(b0 - s0, b1 - s0, device=x.device)  # [bc]
        lo = hi - w_eff  # [bc] >= 0
        win_cross = c_cross[hi] - c_cross[lo]
        win_dx = c_dx[hi] - c_dx[lo]
        win_dy = c_dy[hi] - c_dy[lo]
        raw = win_cross - x_rel[lo] * win_dy + y_rel[lo] * win_dx
        norm_x = torch.sqrt((c_dx2[hi] - c_dx2[lo]).clamp_min(0.0))
        norm_y = torch.sqrt((c_dy2[hi] - c_dy2[lo]).clamp_min(0.0))
        denom = norm_x * norm_y
        val = torch.where(denom > 1e-12, raw / denom.clamp_min(1e-12), 0.0)
        out[b0:b1] = val.to(dtype=x.dtype)
    return out


def _rough_hurst_local_2d(log_vol: torch.Tensor, window: int) -> torch.Tensor:
    """Estimador causal del exponente de Hurst local (Gatheral) vectorizado sobre `[T, N]`."""
    t_len = log_vol.shape[0]
    out = torch.full_like(log_vol, 0.5)
    if t_len < max(window, 6):
        return out
    d1 = torch.zeros_like(log_vol)
    d4 = torch.zeros_like(log_vol)
    d1[1:] = (log_vol[1:] - log_vol[:-1]) ** 2
    d4[4:] = (log_vol[4:] - log_vol[:-4]) ** 2
    m1 = _rolling_mean_2d(d1, window)
    m4 = _rolling_mean_2d(d4, window)
    valid = (m1 > 1e-12) & (m4 > 1e-12)
    ratio = m4.clamp_min(1e-12) / m1.clamp_min(1e-12)
    h_est = torch.where(valid, 0.5 * torch.log(ratio) / math.log(4.0), 0.5)
    out[window:] = h_est[window:].clamp(0.02, 0.98)
    return out


def _causal_affine_scan(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    init: torch.Tensor | float = 0.0,
) -> torch.Tensor:
    """Scan prefijo asociativo estrictamente causal O(log T) para `s[t] = a[t]*s[t-1] + b[t]`.

    Acepta `a` broadcastable sobre las dimensiones finales de `b` (p. ej. `a: [T, N, 1]` con
    `b: [T, N, D]`) sin materializar jamás copias expandidas de `a`, utilizando doble búfer
    ping-pong in-place para minimizar la huella de VRAM.
    """
    t_len = b.shape[0]
    if t_len == 0:
        return torch.empty_like(b)
    a_cur = a.clone()
    b_cur = b.clone()
    if t_len > 1:
        a_buf = torch.empty_like(a_cur)
        b_buf = torch.empty_like(b_cur)
        step = 1
        while step < t_len:
            a_buf[:step].copy_(a_cur[:step])
            torch.mul(a_cur[step:], a_cur[:-step], out=a_buf[step:])
            b_buf[:step].copy_(b_cur[:step])
            torch.addcmul(b_cur[step:], a_cur[step:], b_cur[:-step], out=b_buf[step:])
            a_cur, a_buf = a_buf, a_cur
            b_cur, b_buf = b_buf, b_cur
            step <<= 1
    if isinstance(init, torch.Tensor):
        return a_cur * init.unsqueeze(0) + b_cur
    if init != 0.0:
        return a_cur * float(init) + b_cur
    return b_cur


def _rolling_window_corr_2d(
    x: torch.Tensor,
    y: torch.Tensor,
    window: int,
) -> torch.Tensor:
    """Correlación de Pearson causal exacta sobre `[t-window+1 .. t]` en bloques L2 (`[-1, 1]`)."""
    t_len, n_assets = x.shape
    out = torch.zeros_like(x)
    if t_len == 0 or n_assets == 0 or window < 2:
        return out
    pad = torch.full((window - 1, n_assets), float("nan"), dtype=x.dtype, device=x.device)
    xw_all = torch.cat([pad, x], dim=0).unfold(0, window, 1)  # [T, N, W]
    yw_all = torch.cat([pad, y], dim=0).unfold(0, window, 1)  # [T, N, W]
    chunk_t = max(1, min(t_len, _l2_chunk_elems(x.dtype) // max(n_assets * window, 1)))
    eps_sq_scale = 64.0 * (torch.finfo(x.dtype).eps ** 2)
    for t0 in range(0, t_len, chunk_t):
        t1 = min(t0 + chunk_t, t_len)
        xw = xw_all[t0:t1]
        yw = yw_all[t0:t1]
        ok = torch.isfinite(xw) & torch.isfinite(yw)
        cnt = ok.sum(dim=-1)
        cnt_f = cnt.to(x.dtype)
        denom_cnt = cnt_f.clamp_min(1.0).unsqueeze(-1)
        mx = torch.where(ok, xw, 0.0).sum(dim=-1, keepdim=True) / denom_cnt
        my = torch.where(ok, yw, 0.0).sum(dim=-1, keepdim=True) / denom_cnt
        dx = torch.where(ok, xw - mx, 0.0)
        dy = torch.where(ok, yw - my, 0.0)
        sxx = (dx * dx).sum(dim=-1)
        syy = (dy * dy).sum(dim=-1)
        mx_s = mx.squeeze(-1)
        my_s = my.squeeze(-1)
        sxx = torch.where(sxx <= eps_sq_scale * (mx_s * mx_s) * cnt_f, 0.0, sxx)
        syy = torch.where(syy <= eps_sq_scale * (my_s * my_s) * cnt_f, 0.0, syy)
        sxy = (dx * dy).sum(dim=-1)
        denom = torch.sqrt((sxx * syy).clamp_min(0.0))
        out[t0:t1] = torch.where(
            (cnt >= 2) & (denom > 1e-12),
            (sxy / denom.clamp_min(1e-12)).clamp(-1.0, 1.0),
            0.0,
        )
    return out


def _rolling_window_skew_kurt_2d(
    x: torch.Tensor,
    window: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Asimetría y exceso de curtosis centrales exactos sobre `[t-window+1 .. t]` en bloques L2."""
    t_len, n_assets = x.shape
    skew = torch.zeros_like(x)
    kurt = torch.zeros_like(x)
    if t_len == 0 or n_assets == 0 or window < 3:
        return skew, kurt
    pad = torch.full((window - 1, n_assets), float("nan"), dtype=x.dtype, device=x.device)
    xw_all = torch.cat([pad, x], dim=0).unfold(0, window, 1)  # [T, N, W]
    chunk_t = max(1, min(t_len, _l2_chunk_elems(x.dtype) // max(n_assets * window, 1)))
    eps_sq_scale = 64.0 * (torch.finfo(x.dtype).eps ** 2)
    for t0 in range(0, t_len, chunk_t):
        t1 = min(t0 + chunk_t, t_len)
        xw = xw_all[t0:t1]
        ok = torch.isfinite(xw)
        cnt = ok.sum(dim=-1)
        cnt_f = cnt.to(x.dtype)
        denom_cnt = cnt_f.clamp_min(1.0)
        mx = torch.where(ok, xw, 0.0).sum(dim=-1, keepdim=True) / denom_cnt.unsqueeze(-1)
        dx = torch.where(ok, xw - mx, 0.0)
        dx2 = dx * dx
        ss2 = dx2.sum(dim=-1)
        mx_s = mx.squeeze(-1)
        const_win = ss2 <= eps_sq_scale * (mx_s * mx_s) * cnt_f
        m2 = torch.where(const_win, 0.0, ss2 / denom_cnt)
        m3 = (dx2 * dx).sum(dim=-1) / denom_cnt
        m4 = (dx2 * dx2).sum(dim=-1) / denom_cnt
        std = torch.sqrt(m2.clamp_min(0.0))
        valid = (cnt >= 3) & (~const_win) & (std > 1e-12)
        skew[t0:t1] = torch.where(valid, m3 / (std * std * std).clamp_min(1e-36), 0.0)
        kurt[t0:t1] = torch.where(valid, m4 / (m2 * m2).clamp_min(1e-36) - 3.0, 0.0)
    return skew, kurt


def detect_unadjusted_split_jumps(
    open_: torch.Tensor | np.ndarray,
    close: torch.Tensor | np.ndarray,
    *,
    threshold: float = 0.55,
) -> dict[str, object]:
    """Audita saltos overnight `|log(O_t / C_{t-1})| >= threshold` indicativos de splits crudos."""
    o = _to_numpy_contiguous(open_).astype(np.float64, copy=False)
    c = _to_numpy_contiguous(close).astype(np.float64, copy=False)
    if o.shape != c.shape or o.shape[0] < 2:
        return dict(n_jumps=0, max_abs_overnight_log_jump=0.0, threshold=float(threshold))
    valid = np.isfinite(o[1:]) & np.isfinite(c[:-1]) & (o[1:] > 0.0) & (c[:-1] > 0.0)
    jumps = np.where(
        valid,
        np.abs(np.log(np.maximum(o[1:], 1e-12) / np.maximum(c[:-1], 1e-12))),
        0.0,
    )
    return dict(
        n_jumps=int((jumps >= threshold).sum()),
        max_abs_overnight_log_jump=float(jumps.max()) if jumps.size else 0.0,
        threshold=float(threshold),
    )


def gaussian_rank_features(
    x: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    clip_val: float = LIMIT,
) -> torch.Tensor:
    """Rangos gaussianos transversales por sesión `Phi^{-1}((r_i - 0.5) / n)` sin syncs host.

    Replica `session_ranks.gaussian_ranks` (Beasley-Erickson-Allison / Gu-Kelly-Xiu) con
    empates promediados. Acepta `[N, F]` (sección cruzada en dim 0) o `[T, N, F]` (en dim 1
    de forma independiente por sesión `t`, 100% causal). Valores ausentes/no finitos -> `0.0`.
    """
    if x.ndim not in (2, 3):
        raise ValueError(f"Se espera tensor 2D [N, F] o 3D [T, N, F]; recibido ndim={x.ndim}")
    squeeze_2d = x.ndim == 2
    x3 = x.unsqueeze(0) if squeeze_2d else x
    calc_dtype = torch.float64 if x3.dtype == torch.float64 else torch.float32
    vals = x3.to(dtype=calc_dtype)
    present = torch.isfinite(vals)
    if mask is not None:
        m3 = mask.unsqueeze(0) if (squeeze_2d and mask.ndim == 2) else mask
        present = present & m3.to(device=vals.device, dtype=torch.bool)

    if vals.shape[1] == 0:
        out_empty = torch.zeros_like(vals, dtype=x.dtype)
        return out_empty.squeeze(0) if squeeze_2d else out_empty

    # Mover eje de activos N a la última dimensión: [T, F, N]
    vals_tfn = vals.transpose(1, 2).contiguous()
    pres_tfn = present.transpose(1, 2).contiguous()
    masked = torch.where(pres_tfn, vals_tfn, float("inf"))
    ordered = masked.sort(dim=-1).values
    lower = torch.searchsorted(ordered, masked, right=False)
    upper = torch.searchsorted(ordered, masked, right=True)
    rank = (lower + upper + 1).to(calc_dtype) * 0.5
    count = pres_tfn.sum(dim=-1, keepdim=True).to(calc_dtype).clamp_min(1.0)
    prob = (rank - 0.5) / count
    safe_prob = torch.where(pres_tfn, prob, 0.5)
    gauss = torch.special.ndtri(safe_prob).clamp(-clip_val, clip_val)
    gauss = torch.where(pres_tfn, gauss, 0.0)
    out = gauss.transpose(1, 2).to(dtype=x.dtype)
    return out.squeeze(0) if squeeze_2d else out


def batched_asset_microstructure_v4_torch(
    open_: torch.Tensor | np.ndarray,
    high: torch.Tensor | np.ndarray,
    low: torch.Tensor | np.ndarray,
    close: torch.Tensor | np.ndarray,
    volume: torch.Tensor | np.ndarray,
    *,
    compute_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Calcula los 24 rasgos causales de `ASSET_V4_NAMES` para todos los `N` activos en paralelo.

    Contrato de entrada:
      - OHLC deben estar **ajustados por splits y dividendos** (`detect_unadjusted_split_jumps`).
      - Días de suspensión (`volume <= 0`) y Day-0 (`ret_cc[0]`/`ret_co[0]`) se enmascaran como
        NaN/neutrales para evitar saturar Amihud a `8.0` o sesgar ventanas iniciales con `0.0`.
      - Acepta tensores 2D `[T, N]` (devuelve `[T, N, 24]`) o 1D `[T]` (devuelve `[T, 24]`).
      - Cero bucles Python sobre `N` ni `T`; cero copias H2D ocultas; `float32` en `[-8, 8]`.
    """
    # Preservar precisión FP64 en log-precios si las entradas son FP64 (previene overflow >3.4e38)
    in_is_f64 = (isinstance(open_, torch.Tensor) and open_.dtype == torch.float64) or (
        isinstance(open_, np.ndarray) and open_.dtype == np.float64
    )
    prep_dtype = torch.float64 if (in_is_f64 or compute_dtype == torch.float64) else compute_dtype

    o_raw = _as_float_tensor(open_, dtype=prep_dtype)
    dev = o_raw.device
    h_raw = _as_float_tensor(high, dtype=prep_dtype, device=dev)
    l_raw = _as_float_tensor(low, dtype=prep_dtype, device=dev)
    c_raw = _as_float_tensor(close, dtype=prep_dtype, device=dev)
    v_raw = _as_float_tensor(volume, dtype=prep_dtype, device=dev)

    if not (o_raw.shape == h_raw.shape == l_raw.shape == c_raw.shape == v_raw.shape):
        raise ValueError("open_, high, low, close y volume deben tener exactamente la misma forma")
    if o_raw.ndim not in (1, 2):
        raise ValueError(f"Se esperan tensores 1D [T] o 2D [T, N]; recibido ndim={o_raw.ndim}")

    squeeze_1d = o_raw.ndim == 1
    if squeeze_1d:
        o_raw = o_raw.unsqueeze(1)
        h_raw = h_raw.unsqueeze(1)
        l_raw = l_raw.unsqueeze(1)
        c_raw = c_raw.unsqueeze(1)
        v_raw = v_raw.unsqueeze(1)

    t_len, n_assets = o_raw.shape
    n_feats = len(ASSET_V4_NAMES)
    if t_len == 0 or n_assets == 0:
        shape = (t_len, n_feats) if squeeze_1d else (t_len, n_assets, n_feats)
        return torch.empty(shape, dtype=torch.float32, device=dev)

    # Cero torch.tensor(scalar, device=dev) -> cero copias H2D ni stream syncs
    o = o_raw.clamp_min(1e-12)
    h = torch.maximum(h_raw, o)
    l_ = torch.minimum(l_raw.clamp_min(1e-12), o)
    c = c_raw.clamp_min(1e-12)
    h = torch.maximum(h, c)
    l_ = torch.minimum(l_, c)
    traded = torch.isfinite(v_raw) & (v_raw > 0.0)
    v = v_raw.clamp_min(0.0)

    log_o_p = torch.log(o)
    log_h_p = torch.log(h)
    log_l_p = torch.log(l_)
    log_c_p = torch.log(c)
    log_v_p = torch.log1p(v)
    dollar_vol_p = (c * v).clamp_min(1.0)
    hl_lin_p = h - l_
    clv_p = torch.where(
        hl_lin_p > 1e-12,
        ((c - l_) - (h - c)) / hl_lin_p.clamp_min(1e-12),
        0.0,
    )

    # Diferencias de log-precios y spreads de 2 días antes de convertir a compute_dtype
    u = (log_h_p - log_o_p).to(dtype=compute_dtype)
    d = (log_l_p - log_o_p).to(dtype=compute_dtype)
    hl = (log_h_p - log_l_p).to(dtype=compute_dtype)
    co_c = (log_c_p - log_o_p).to(dtype=compute_dtype)
    log_c = log_c_p.to(dtype=compute_dtype)
    log_v = log_v_p.to(dtype=compute_dtype)
    dollar_vol = dollar_vol_p.to(dtype=compute_dtype)
    clv = clv_p.to(dtype=compute_dtype)

    # Day-0 ret_cc[0] y ret_co[0] enmascarados como NaN (sin sesgo 0.0 en warmup)
    ret_cc = torch.full_like(log_c, float("nan"))
    ret_oc = co_c
    ret_co = torch.full_like(log_c, float("nan"))
    if t_len > 1:
        ret_cc[1:] = (log_c_p[1:] - log_c_p[:-1]).to(dtype=compute_dtype)
        ret_co[1:] = (log_o_p[1:] - log_c_p[:-1]).to(dtype=compute_dtype)

    log_v_traded = torch.where(traded, log_v, float("nan"))

    # 1) Garman-Klass, Rogers-Satchell y Yang-Zhang (días volume <= 0 enmascarados como NaN)
    gk_raw = (0.5 * hl * hl - (2.0 * math.log(2.0) - 1.0) * co_c * co_c).clamp_min(0.0)
    rs_raw = (u * (u - co_c) + d * (d - co_c)).clamp_min(0.0)
    gk_daily = torch.where(traded, gk_raw, float("nan"))
    rs_daily = torch.where(traded, rs_raw, float("nan"))
    ret_oc_t = torch.where(traded, ret_oc, float("nan"))
    ret_co_t = torch.where(traded, ret_co, float("nan"))

    def _ann_log_vol(daily_var: torch.Tensor, win: int) -> torch.Tensor:
        return torch.log(
            torch.sqrt(_rolling_mean_2d(daily_var, win).clamp_min(1e-10) * 252.0) * 100.0 + 1e-6
        )

    gk_21 = _ann_log_vol(gk_daily, 21)
    gk_63 = _ann_log_vol(gk_daily, 63)
    rs_21 = _ann_log_vol(rs_daily, 21)
    rs_63 = _ann_log_vol(rs_daily, 63)

    def _yz(win: int) -> torch.Tensor:
        k_yz = 0.34 / (1.34 + (win + 1.0) / max(win - 1.0, 1.0))
        v_o = _rolling_var_2d(ret_co_t, win)
        v_c = _rolling_var_2d(ret_oc_t, win)
        v_rs = _rolling_mean_2d(rs_daily, win)
        total = (v_o + k_yz * v_c + (1.0 - k_yz) * v_rs).clamp_min(1e-10)
        return torch.log(torch.sqrt(total * 252.0) * 100.0 + 1e-6)

    yz_21 = _yz(21)
    yz_63 = _yz(63)

    # 2) Volatilidad rugosa (Gatheral) + vol-of-vol
    daily_log_sigma = torch.where(traded, 0.5 * torch.log(gk_raw.clamp_min(1e-10)), float("nan"))
    hurst_21 = _rough_hurst_local_2d(daily_log_sigma, 21)
    hurst_63 = _rough_hurst_local_2d(daily_log_sigma, 63)
    vov_21 = torch.sqrt(_rolling_var_2d(daily_log_sigma, 21))

    # 3) Spread implícito Corwin-Schultz (2012) & Abdi-Ranaldo (2017)
    beta_cs = torch.zeros_like(log_c)
    gamma_cs = torch.zeros_like(log_c)
    if t_len > 1:
        beta_cs[1:] = hl[1:] ** 2 + hl[:-1] ** 2
        h2_log = torch.maximum(log_h_p[1:], log_h_p[:-1])
        l2_log = torch.minimum(log_l_p[1:], log_l_p[:-1])
        gamma_cs[1:] = ((h2_log - l2_log).clamp_min(0.0) ** 2).to(dtype=compute_dtype)
    denom_cs = 3.0 - 2.0 * math.sqrt(2.0)
    alpha_cs = (
        (torch.sqrt(2.0 * beta_cs) - torch.sqrt(beta_cs)) / denom_cs
        - torch.sqrt(gamma_cs / denom_cs)
    ).clamp_min(0.0)
    cs_daily = torch.where(traded, 2.0 * torch.tanh(0.5 * alpha_cs), float("nan"))
    cs_21 = torch.log1p(_rolling_mean_2d(cs_daily, 21).clamp_min(0.0) * 100.0)

    eta_mid_p = 0.5 * (log_h_p + log_l_p)
    ar_sq_raw = torch.zeros_like(log_c)
    if t_len > 1:
        ar_sq_raw[1:] = (
            4.0 * (log_c_p[:-1] - eta_mid_p[:-1]) * (log_c_p[:-1] - eta_mid_p[1:])
        ).to(dtype=compute_dtype)
    ar_sq = torch.where(traded, ar_sq_raw, float("nan"))
    ar_21 = torch.log1p(torch.sqrt(_rolling_mean_2d(ar_sq, 21).clamp_min(0.0)) * 100.0)

    # 4) Amihud direccional: días volume <= 0 y t=0 enmascarados como NaN/neutrales
    illiq = torch.where(
        traded & torch.isfinite(ret_cc), ret_cc.abs() / dollar_vol * 1e6, float("nan")
    )
    amihud_up_21 = torch.log1p(
        _rolling_mean_2d(torch.where(traded & (ret_cc > 0.0), illiq, float("nan")), 21)
    )
    amihud_dn_21 = torch.log1p(
        _rolling_mean_2d(torch.where(traded & (ret_cc < 0.0), illiq, float("nan")), 21)
    )
    amihud_asym_21 = amihud_dn_21 - amihud_up_21

    # 5) Flujo firmado (CLV * volumen) y correlación exacta precio-volumen en [-1, 1]
    signed_v = torch.where(traded, clv * (log_v - _rolling_mean_2d(log_v_traded, 63)), 0.0)
    sv_5 = _rolling_mean_2d(signed_v, 5)
    sv_21 = _rolling_mean_2d(signed_v, 21)
    pv_corr_21 = _rolling_window_corr_2d(ret_cc, log_v_traded, 21)

    # 6) Asimetría y curtosis centrales exactas de la ventana (21d)
    skew_21, kurt_21 = _rolling_window_skew_kurt_2d(ret_cc, 21)

    # 7) Diferenciación fraccionaria (López de Prado) vía F.conv1d batched con caché en dispositivo
    ffd_035 = _apply_ffd_2d(log_c, 0.35)
    ffd_050 = _apply_ffd_2d(log_c, 0.50)

    # 8) Firmas de trayectoria de orden 2 (incrementos exactos desde prep_dtype -> cero deriva FP32)
    log_hl100_p = torch.log1p((log_h_p - log_l_p) * 100.0)
    if t_len > 1:
        d_log_c = (log_c_p[1:] - log_c_p[:-1]).to(dtype=compute_dtype)
        d_log_v = (log_v_p[1:] - log_v_p[:-1]).to(dtype=compute_dtype)
        d_log_hl = (log_hl100_p[1:] - log_hl100_p[:-1]).to(dtype=compute_dtype)
    else:
        d_log_c = torch.empty((0, n_assets), dtype=compute_dtype, device=dev)
        d_log_v = torch.empty((0, n_assets), dtype=compute_dtype, device=dev)
        d_log_hl = torch.empty((0, n_assets), dtype=compute_dtype, device=dev)
    sig_pv_5 = _rolling_levy_area_2d(log_c, log_v, 5, dx_in=d_log_c, dy_in=d_log_v)
    sig_pv_21 = _rolling_levy_area_2d(log_c, log_v, 21, dx_in=d_log_c, dy_in=d_log_v)
    sig_pr_21 = _rolling_levy_area_2d(
        log_c, log_hl100_p.to(dtype=compute_dtype), 21, dx_in=d_log_c, dy_in=d_log_hl
    )

    mat = torch.stack(
        [
            gk_21,
            gk_63,
            rs_21,
            rs_63,
            yz_21,
            yz_63,
            hurst_21,
            hurst_63,
            vov_21,
            cs_21,
            ar_21,
            amihud_up_21,
            amihud_dn_21,
            amihud_asym_21,
            sv_5,
            sv_21,
            pv_corr_21,
            skew_21,
            kurt_21,
            ffd_035,
            ffd_050,
            sig_pv_5,
            sig_pv_21,
            sig_pr_21,
        ],
        dim=-1,
    )
    mat = torch.nan_to_num(mat, nan=0.0, posinf=LIMIT, neginf=-LIMIT).clamp(-LIMIT, LIMIT)
    out = mat.to(dtype=torch.float32)
    return out.squeeze(1) if squeeze_1d else out


def _rmt_spectrum_summary(
    eigvals: torch.Tensor,
    w_lens_t: torch.Tensor,
    n_assets: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Calcula resumen espectral + `(gap_ratio_1_2_log, top1_share)` gratis de `eigvals [B, N]`."""
    dtype = eigvals.dtype
    q_ratio = float(n_assets) / w_lens_t.clamp_min(1.0)
    lambda_plus = (1.0 + torch.sqrt(q_ratio)) ** 2  # [B]
    signal_mask = eigvals > lambda_plus.unsqueeze(-1)
    n_sig = signal_mask.sum(dim=-1)

    noise_mask = ~signal_mask
    noise_cnt = noise_mask.sum(dim=-1).clamp_min(1).to(dtype)
    sigma_bulk2 = (eigvals * noise_mask.to(dtype)).sum(dim=-1) / noise_cnt
    reest = (n_sig > 0) & (n_sig < n_assets)
    lam_eff = torch.where(reest, sigma_bulk2 * lambda_plus, lambda_plus)
    signal_mask = eigvals > lam_eff.unsqueeze(-1)
    n_sig = signal_mask.sum(dim=-1)

    tr = eigvals.sum(dim=-1).clamp_min(1e-12)
    k_abs = min(5, n_assets)
    absorption = eigvals[:, :k_abs].sum(dim=-1) / tr
    probs = eigvals / tr.unsqueeze(-1)
    probs_safe = probs.clamp_min(1e-15)
    ent_terms = torch.where(probs > 1e-15, probs * torch.log(probs_safe), 0.0)
    entropy = -ent_terms.sum(dim=-1) / max(math.log(n_assets), 1e-12)

    # Rasgos gratuitos de rotación 2-factores (0 eigendecomps extra!)
    lam_1 = eigvals[:, 0]
    lam_2 = eigvals[:, 1] if n_assets >= 2 else torch.ones_like(lam_1)
    top1_share = (lam_1 / tr).clamp(0.0, 1.0)
    gap_ratio_log = torch.log((lam_1 / lam_2.clamp_min(1e-6)).clamp(1.0, 100.0))
    return n_sig, signal_mask, lam_eff, absorption, entropy, gap_ratio_log, top1_share


def _canonicalize_loading_signs(raw_load: torch.Tensor) -> torch.Tensor:
    """Fija signo canónico determinista por columna `k` usando el pivote de máximo |valor|."""
    if raw_load.shape[-1] == 0:
        return raw_load
    pivots = raw_load.abs().argmax(dim=1, keepdim=True)
    pivot_vals = torch.gather(raw_load, 1, pivots)
    signs = torch.where(pivot_vals < 0.0, -1.0, 1.0).to(dtype=raw_load.dtype)
    return raw_load * signs


def _rmt_dual_gram_path(
    z: torch.Tensor,
    active: torch.Tensor,
    w_lens_t: torch.Tensor,
    *,
    top_k: int,
    return_clean_corr: bool,
) -> dict[str, torch.Tensor]:
    """Ruta rápida exacta dual `W×W` cuando `W < N` y cada activo activo está completo en `[W]`.

    Explota que `C_active = X^T X` (con `X = Z / sqrt(W_eff - 1)`) comparte todos los autovalores
    no nulos con el Gram pequeño `G = X X^T` de tamaño `[B, W, W]` (`253×` menos FLOPs en
    `W=63, N=2600`), recuperando las cargas exactas vía `X^T U_W` y la correlación limpia vía
    resolución de la identidad `noise_mean * I_N + (X^T U_W) diag(w) (X^T U_W)^T`.
    """
    bsz, w_max, n_assets = z.shape
    dev = z.device
    dtype = z.dtype

    inv_sqrt_df = torch.rsqrt((w_lens_t - 1.0).clamp_min(1.0))  # [B]
    x = z * inv_sqrt_df.view(bsz, 1, 1)  # [B, W, N] (columnas inactivas son exactamente 0)
    gram = torch.bmm(x, x.transpose(1, 2))  # [B, W, W]
    gram = 0.5 * (gram + gram.transpose(1, 2))

    mu, u_vecs = torch.linalg.eigh(gram)
    mu = mu.flip(-1).clamp_min(0.0)  # [B, W] descendente
    u_vecs = u_vecs.flip(-1)  # [B, W, W] columnas descendentes

    # Espectro completo exacto [B, N]: autovalores de G + n_inactive unos (diag=1) + ceros nulos
    n_act = active.sum(dim=1, keepdim=True)  # [B, 1]
    n_inact = n_assets - n_act  # [B, 1]
    w_idx = torch.arange(w_max, device=dev).unsqueeze(0)  # [1, W]
    head_eigs = torch.where(w_idx < n_act, mu, 1.0).to(dtype)
    rem_inact = n_inact - (w_max - n_act).clamp_min(0)
    tail_idx = torch.arange(n_assets - w_max, device=dev).unsqueeze(0)  # [1, N - W]
    tail_eigs = torch.where(tail_idx < rem_inact, 1.0, 0.0).to(dtype)
    eigvals_unsorted = torch.cat([head_eigs, tail_eigs], dim=1)  # [B, N]
    eigvals = torch.sort(eigvals_unsorted, dim=-1, descending=True).values

    n_sig, signal_mask, lam_eff, absorption, entropy, gap_ratio_log, top1_share = (
        _rmt_spectrum_summary(eigvals, w_lens_t, n_assets)
    )

    # Cargas top_k: X^T u_k == v_k * sqrt(mu_k) exactamente sin división por sqrt(mu_k)
    avail = min(top_k, n_assets)
    loadings = torch.zeros((bsz, n_assets, top_k), dtype=dtype, device=dev)
    if avail > 0:
        k_g = min(avail, w_max)
        # Enmascarar modos de G más allá de n_act (autovalor 0) para que sus cargas sean 0
        valid_g = (torch.arange(k_g, device=dev).unsqueeze(0) < n_act).to(dtype)
        raw_load = torch.bmm(x.transpose(1, 2), u_vecs[:, :, :k_g]) * valid_g.unsqueeze(1)
        raw_load = _canonicalize_loading_signs(raw_load)

        # Caso general: si algún activo inactivo (autovalor 1.0) entra en el top-k, fusionar exacto
        if avail > w_max:
            inact_in_top = n_inact.squeeze(1) > 0
        else:
            inact_in_top = (n_inact.squeeze(1) > 0) & (mu[:, k_g - 1] < 1.0 + 1e-6)
        if not bool(inact_in_top.any()):
            loadings[:, :, :k_g] = raw_load
        else:
            loadings[:, :, :k_g] = raw_load
            inact_flags = inact_in_top.tolist()
            for b_i, needs_merge in enumerate(inact_flags):
                if not needs_merge:
                    continue
                n_act_i = int(n_act[b_i, 0].tolist())
                g_eigs = mu[b_i, : min(k_g, n_act_i)].tolist()
                inact_assets = [j for j, flag in enumerate(active[b_i].tolist()) if not flag]
                p_g = 0
                p_in = 0
                for col_k in range(avail):
                    g_val = g_eigs[p_g] if p_g < len(g_eigs) else -1.0
                    in_val = 1.0 if p_in < len(inact_assets) else -1.0
                    if g_val >= in_val and g_val >= 0.0:
                        loadings[b_i, :, col_k] = raw_load[b_i, :, p_g]
                        p_g += 1
                    elif in_val > 0.0:
                        loadings[b_i, :, col_k].zero_()
                        loadings[b_i, inact_assets[p_in], col_k] = 1.0
                        p_in += 1
                    else:
                        loadings[b_i, :, col_k].zero_()

    res: dict[str, torch.Tensor] = dict(
        n_signal_modes=n_sig.to(torch.int64),
        absorption_ratio=absorption,
        spectral_entropy=entropy,
        loadings=loadings,
        rmt_gap_ratio_1_2=gap_ratio_log,
        rmt_top1_share=top1_share,
    )

    if return_clean_corr:
        diag_idx = torch.arange(n_assets, device=dev)
        noise_mask2 = ~signal_mask
        noise_cnt2 = noise_mask2.sum(dim=-1).clamp_min(1).to(dtype)
        noise_mean = ((eigvals * noise_mask2.to(dtype)).sum(dim=-1) / noise_cnt2).clamp_min(1e-4)

        mu_valid = w_idx < n_act
        mu_sig = mu_valid & (mu > lam_eff.unsqueeze(-1))
        mu_clean = torch.where(mu_sig, mu.clamp_min(1e-4), noise_mean.unsqueeze(-1))
        w_mode = torch.where(
            mu_sig & (mu > 1e-12),
            (mu_clean - noise_mean.unsqueeze(-1)) / mu.clamp_min(1e-12),
            0.0,
        )
        xu = torch.bmm(x.transpose(1, 2), u_vecs)  # [B, N, W]
        clean_corr = torch.bmm(xu * w_mode.unsqueeze(1), xu.transpose(1, 2))
        clean_corr[:, diag_idx, diag_idx] = (
            clean_corr[:, diag_idx, diag_idx] + noise_mean.unsqueeze(-1)
        )
        clean_corr = 0.5 * (clean_corr + clean_corr.transpose(1, 2))
        diag_sqrt = torch.sqrt(torch.diagonal(clean_corr, dim1=1, dim2=2).clamp_min(1e-8))
        clean_corr = clean_corr / (diag_sqrt.unsqueeze(2) * diag_sqrt.unsqueeze(1))
        clean_corr[:, diag_idx, diag_idx] = 1.0

        raw_corr = torch.bmm(x.transpose(1, 2), x).clamp(-1.0, 1.0)
        raw_corr = 0.5 * (raw_corr + raw_corr.transpose(1, 2))
        raw_corr[:, diag_idx, diag_idx] = 1.0
        res["raw_corr"] = raw_corr
        res["clean_corr"] = clean_corr

    return res


def _rmt_full_nxn_path(
    z: torch.Tensor,
    valid: torch.Tensor,
    w_lens_t: torch.Tensor,
    *,
    top_k: int,
    return_clean_corr: bool,
    min_overlap: int,
) -> dict[str, torch.Tensor]:
    """Ruta completa `N×N` con solape por pares `cnt_ij - 1` sub-chunkeada para acotar VRAM."""
    bsz, _, n_assets = z.shape
    dev = z.device
    dtype = z.dtype
    sub_b = max(1, _RMT_MATRIX_CHUNK_ELEMS // max(n_assets * n_assets, 1))

    n_sig_out = torch.zeros(bsz, dtype=torch.int64, device=dev)
    abs_out = torch.zeros(bsz, dtype=dtype, device=dev)
    ent_out = torch.zeros(bsz, dtype=dtype, device=dev)
    gap_out = torch.zeros(bsz, dtype=dtype, device=dev)
    top1_out = torch.zeros(bsz, dtype=dtype, device=dev)
    load_out = torch.zeros((bsz, n_assets, top_k), dtype=dtype, device=dev)
    raw_out = (
        torch.zeros((bsz, n_assets, n_assets), dtype=dtype, device=dev)
        if return_clean_corr
        else None
    )
    clean_out = (
        torch.zeros((bsz, n_assets, n_assets), dtype=dtype, device=dev)
        if return_clean_corr
        else None
    )
    diag_idx = torch.arange(n_assets, device=dev)
    avail = min(top_k, n_assets)

    for b0 in range(0, bsz, sub_b):
        b1 = min(b0 + sub_b, bsz)
        z_s = z[b0:b1]
        v_f = valid[b0:b1].to(dtype)
        wl_s = w_lens_t[b0:b1]

        cnt_ij = torch.bmm(v_f.transpose(1, 2), v_f)  # [bs, N, N]
        min_ov = (
            torch.minimum(torch.full_like(wl_s, float(min_overlap)), wl_s)
            .clamp_min(3.0)
            .view(b1 - b0, 1, 1)
        )
        raw_corr = torch.bmm(z_s.transpose(1, 2), z_s) / (cnt_ij - 1.0).clamp_min(1.0)
        corr = torch.where(cnt_ij >= min_ov, raw_corr.clamp(-1.0, 1.0), 0.0)
        corr = 0.5 * (corr + corr.transpose(1, 2))
        corr[:, diag_idx, diag_idx] = 1.0

        eigvals, eigvecs = torch.linalg.eigh(corr)
        eigvals = eigvals.flip(-1).clamp_min(0.0)
        eigvecs = eigvecs.flip(-1)

        n_sig, signal_mask, _, absorption, entropy, gap_ratio_log, top1_share = (
            _rmt_spectrum_summary(eigvals, wl_s, n_assets)
        )
        n_sig_out[b0:b1] = n_sig.to(torch.int64)
        abs_out[b0:b1] = absorption
        ent_out[b0:b1] = entropy
        gap_out[b0:b1] = gap_ratio_log
        top1_out[b0:b1] = top1_share

        if avail > 0:
            top_vecs = eigvecs[:, :, :avail]
            top_vals = eigvals[:, :avail].clamp_min(0.0)
            load_out[b0:b1, :, :avail] = _canonicalize_loading_signs(
                top_vecs * torch.sqrt(top_vals).unsqueeze(1)
            )

        if return_clean_corr:
            assert raw_out is not None and clean_out is not None
            noise_mask2 = ~signal_mask
            noise_cnt2 = noise_mask2.sum(dim=-1).clamp_min(1).to(dtype)
            noise_mean = (
                (eigvals * noise_mask2.to(dtype)).sum(dim=-1) / noise_cnt2
            ).clamp_min(1e-4)
            cleaned_eigs = torch.where(
                noise_mask2, noise_mean.unsqueeze(-1), eigvals
            ).clamp_min(1e-4)
            clean_corr = torch.bmm(eigvecs * cleaned_eigs.unsqueeze(1), eigvecs.transpose(1, 2))
            clean_corr = 0.5 * (clean_corr + clean_corr.transpose(1, 2))
            diag_sqrt = torch.sqrt(torch.diagonal(clean_corr, dim1=1, dim2=2).clamp_min(1e-8))
            clean_corr = clean_corr / (diag_sqrt.unsqueeze(2) * diag_sqrt.unsqueeze(1))
            clean_corr[:, diag_idx, diag_idx] = 1.0
            raw_out[b0:b1] = corr
            clean_out[b0:b1] = clean_corr

    res: dict[str, torch.Tensor] = dict(
        n_signal_modes=n_sig_out,
        absorption_ratio=abs_out,
        spectral_entropy=ent_out,
        loadings=load_out,
        rmt_gap_ratio_1_2=gap_out,
        rmt_top1_share=top1_out,
    )
    if return_clean_corr:
        assert raw_out is not None and clean_out is not None
        res["raw_corr"] = raw_out
        res["clean_corr"] = clean_out
    return res


def _rmt_batch_core(
    r_batch: torch.Tensor,
    *,
    top_k: int,
    time_mask: torch.Tensor | None = None,
    w_lens: torch.Tensor | None = None,
    return_clean_corr: bool = False,
    min_overlap: int = 5,
    compute_dtype: torch.dtype = torch.float32,
    assume_no_nans: bool = False,
) -> dict[str, torch.Tensor]:
    """Núcleo RMT Marchenko-Pastur `[B, W, N]` con ruta dual `W×W` exacta y fallback `N×N`."""
    r_b = r_batch.to(dtype=compute_dtype)
    bsz, w_max, n_assets = r_b.shape
    dev = r_b.device
    if w_lens is None:
        w_lens_t = torch.full((bsz,), float(w_max), dtype=compute_dtype, device=dev)
    else:
        w_lens_t = w_lens.to(device=dev, dtype=compute_dtype)

    if w_max < 3 or n_assets < 2:
        out: dict[str, torch.Tensor] = dict(
            n_signal_modes=torch.zeros(bsz, dtype=torch.int64, device=dev),
            absorption_ratio=torch.zeros(bsz, dtype=compute_dtype, device=dev),
            spectral_entropy=torch.zeros(bsz, dtype=compute_dtype, device=dev),
            loadings=torch.zeros((bsz, n_assets, top_k), dtype=compute_dtype, device=dev),
            rmt_gap_ratio_1_2=torch.zeros(bsz, dtype=compute_dtype, device=dev),
            rmt_top1_share=torch.zeros(bsz, dtype=compute_dtype, device=dev),
        )
        if return_clean_corr:
            eye = (
                torch.eye(n_assets, dtype=compute_dtype, device=dev)
                .unsqueeze(0)
                .expand(bsz, -1, -1)
            )
            out["raw_corr"] = eye.clone()
            out["clean_corr"] = eye.clone()
        return out

    valid = torch.isfinite(r_b)
    if time_mask is not None:
        valid = valid & time_mask.unsqueeze(-1)
    cnt_raw = valid.sum(dim=1).to(compute_dtype)  # [B, N]
    cnt = cnt_raw.clamp_min(1.0)
    r_zero = torch.where(valid, r_b, 0.0)
    means = r_zero.sum(dim=1) / cnt  # [B, N]
    centered = torch.where(valid, r_b - means.unsqueeze(1), 0.0)
    stds = torch.sqrt((centered * centered).sum(dim=1) / (cnt - 1.0).clamp_min(1.0))  # [B, N]
    active = stds > 1e-12
    z = torch.where(
        active.unsqueeze(1),
        centered / stds.clamp_min(1e-12).unsqueeze(1),
        0.0,
    )

    ac_ctx = (
        torch.amp.autocast(device_type=dev.type, enabled=False)
        if dev.type in ("cuda", "cpu")
        else contextlib.nullcontext()
    )
    with ac_ctx:
        if w_max < n_assets:
            if assume_no_nans:
                return _rmt_dual_gram_path(
                    z,
                    active,
                    w_lens_t,
                    top_k=top_k,
                    return_clean_corr=return_clean_corr,
                )
            dual_ok = (w_lens_t >= 3.0) & (
                (~active) | (cnt_raw == w_lens_t.unsqueeze(1))
            ).all(dim=1)
            if bool(dual_ok.all()):
                return _rmt_dual_gram_path(
                    z,
                    active,
                    w_lens_t,
                    top_k=top_k,
                    return_clean_corr=return_clean_corr,
                )
            if bool(dual_ok.any()):
                flags = dual_ok.tolist()
                dual_idx = [i for i, f in enumerate(flags) if f]
                full_idx = [i for i, f in enumerate(flags) if not f]
                res_d = _rmt_dual_gram_path(
                    z[dual_idx],
                    active[dual_idx],
                    w_lens_t[dual_idx],
                    top_k=top_k,
                    return_clean_corr=return_clean_corr,
                )
                res_f = _rmt_full_nxn_path(
                    z[full_idx],
                    valid[full_idx],
                    w_lens_t[full_idx],
                    top_k=top_k,
                    return_clean_corr=return_clean_corr,
                    min_overlap=min_overlap,
                )
                merged: dict[str, torch.Tensor] = dict(
                    n_signal_modes=torch.empty(bsz, dtype=torch.int64, device=dev),
                    absorption_ratio=torch.empty(bsz, dtype=compute_dtype, device=dev),
                    spectral_entropy=torch.empty(bsz, dtype=compute_dtype, device=dev),
                    loadings=torch.empty((bsz, n_assets, top_k), dtype=compute_dtype, device=dev),
                    rmt_gap_ratio_1_2=torch.empty(bsz, dtype=compute_dtype, device=dev),
                    rmt_top1_share=torch.empty(bsz, dtype=compute_dtype, device=dev),
                )
                if return_clean_corr:
                    merged["raw_corr"] = torch.empty(
                        (bsz, n_assets, n_assets), dtype=compute_dtype, device=dev
                    )
                    merged["clean_corr"] = torch.empty(
                        (bsz, n_assets, n_assets), dtype=compute_dtype, device=dev
                    )
                for k in merged:
                    merged[k][dual_idx] = res_d[k]
                    merged[k][full_idx] = res_f[k]
                return merged

        return _rmt_full_nxn_path(
            z,
            valid,
            w_lens_t,
            top_k=top_k,
            return_clean_corr=return_clean_corr,
            min_overlap=min_overlap,
        )


def batched_rmt_clean_correlation_torch(
    returns_window: torch.Tensor | np.ndarray,
    *,
    top_k: int = 5,
    return_clean_corr: bool = True,
    min_overlap: int = 5,
    compute_dtype: torch.dtype = torch.float32,
    return_extended: bool = False,
) -> dict[str, torch.Tensor]:
    """Limpieza espectral RMT Marchenko-Pastur en PyTorch sobre `[W, N]` o `[B, W, N]`."""
    r = _as_float_tensor(returns_window, dtype=compute_dtype)
    if r.ndim == 2:
        res = _rmt_batch_core(
            r.unsqueeze(0),
            top_k=top_k,
            return_clean_corr=return_clean_corr,
            min_overlap=min_overlap,
            compute_dtype=compute_dtype,
        )
        out: dict[str, torch.Tensor] = dict(
            n_signal_modes=res["n_signal_modes"][0],
            absorption_ratio=res["absorption_ratio"][0].to(torch.float32),
            spectral_entropy=res["spectral_entropy"][0].to(torch.float32),
            loadings=res["loadings"][0].to(torch.float32),
        )
        if return_clean_corr:
            out["raw_corr"] = res["raw_corr"][0].to(torch.float32)
            out["clean_corr"] = res["clean_corr"][0].to(torch.float32)
        if return_extended:
            out["rmt_gap_ratio_1_2"] = res["rmt_gap_ratio_1_2"][0].to(torch.float32)
            out["rmt_top1_share"] = res["rmt_top1_share"][0].to(torch.float32)
        return out
    if r.ndim != 3:
        raise ValueError(f"Se espera tensor 2D [W, N] o 3D [B, W, N]; recibido ndim={r.ndim}")

    res = _rmt_batch_core(
        r,
        top_k=top_k,
        return_clean_corr=return_clean_corr,
        min_overlap=min_overlap,
        compute_dtype=compute_dtype,
    )
    out = dict(
        n_signal_modes=res["n_signal_modes"],
        absorption_ratio=res["absorption_ratio"].to(torch.float32),
        spectral_entropy=res["spectral_entropy"].to(torch.float32),
        loadings=res["loadings"].to(torch.float32),
    )
    if return_clean_corr:
        out["raw_corr"] = res["raw_corr"].to(torch.float32)
        out["clean_corr"] = res["clean_corr"].to(torch.float32)
    if return_extended:
        out["rmt_gap_ratio_1_2"] = res["rmt_gap_ratio_1_2"].to(torch.float32)
        out["rmt_top1_share"] = res["rmt_top1_share"].to(torch.float32)
    return out


def _sorted_linear_quantiles(
    vals: torch.Tensor,
    grid: torch.Tensor,
    min_valid: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Calcula cuantiles lineales exactos en `[B, M]` tolerando NaNs en 1 única ordenación."""
    dtype = vals.dtype
    fin = torch.isfinite(vals)
    cnt = fin.sum(dim=1)  # [B]
    filled = torch.where(fin, vals, float("inf"))
    sorted_v = torch.sort(filled, dim=1).values
    sorted_safe = torch.where(torch.isfinite(sorted_v), sorted_v, 0.0)
    max_idx = vals.shape[1] - 1
    pos = (cnt - 1).clamp_min(0).to(dtype).unsqueeze(1) * grid.unsqueeze(0)
    lo = pos.floor().to(torch.long).clamp(0, max_idx)
    hi = pos.ceil().to(torch.long).clamp(0, max_idx)
    frac = pos - lo.to(dtype)
    v_lo = torch.gather(sorted_safe, 1, lo)
    v_hi = torch.gather(sorted_safe, 1, hi)
    return v_lo + frac * (v_hi - v_lo), cnt >= min_valid


def batched_causal_wasserstein_shift_torch(
    returns_panel: torch.Tensor | np.ndarray,
    *,
    short_window: int = 5,
    long_window: int = 40,
    maturity_lag: int = 1,
    n_quantiles: int = 64,
    compute_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Distancia 1D de Wasserstein estrictamente causal vectorizada en bloques L2 (FP32 radix)."""
    if maturity_lag < 0:
        raise ValueError("maturity_lag debe ser >= 0 para preservar causalidad estricta")
    r = _as_float_tensor(returns_panel, dtype=compute_dtype)
    t_len, n_assets = r.shape
    dev = r.device
    out = torch.zeros(t_len, dtype=compute_dtype, device=dev)
    t_first = maturity_lag - 1 + short_window + long_window
    if (
        short_window <= 0
        or long_window <= 0
        or n_assets == 0
        or t_first >= t_len
        or t_first < 0
    ):
        return out.to(torch.float32)

    grid = torch.linspace(0.01, 0.99, n_quantiles, dtype=compute_dtype, device=dev)
    b_total = t_len - t_first
    chunk_b = max(
        1, min(b_total, _l2_chunk_elems(compute_dtype) // max(long_window * n_assets, 1))
    )

    end_long_max = b_total - 1 + long_window
    end_short_max = end_long_max + short_window
    long_windows = r[:end_long_max].unfold(0, long_window, 1)  # [b_total, N, long_window]
    short_windows = r[long_window:end_short_max].unfold(0, short_window, 1)  # [b_total, N, short]

    for b0 in range(0, b_total, chunk_b):
        b1 = min(b0 + chunk_b, b_total)
        s_flat = short_windows[b0:b1].reshape(b1 - b0, -1)
        l_flat = long_windows[b0:b1].reshape(b1 - b0, -1)
        qs, s_ok = _sorted_linear_quantiles(s_flat, grid, min_valid=10)
        ql, l_ok = _sorted_linear_quantiles(l_flat, grid, min_valid=20)
        dist = (qs - ql).abs().mean(dim=1)
        out[t_first + b0 : t_first + b1] = torch.where(s_ok & l_ok, dist, 0.0)

    return out.to(torch.float32)


def batched_causal_lead_lag_spillover_torch(
    returns_panel: torch.Tensor | np.ndarray,
    liquidity_panel: torch.Tensor | np.ndarray,
    *,
    window: int = 40,
    leader_frac: float = 0.2,
    maturity_lag: int = 1,
    compute_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Señal dirigida Lead-Lag causal (`leader_ret[t - maturity_lag]`) centrada sin cancelación."""
    if maturity_lag < 1:
        raise ValueError("maturity_lag debe ser >= 1 para preservar observabilidad causal en t")
    r = _as_float_tensor(returns_panel, dtype=compute_dtype)
    dev = r.device
    liq = _as_float_tensor(liquidity_panel, dtype=compute_dtype, device=dev)
    t_len, n_assets = r.shape
    out = torch.zeros((t_len, n_assets), dtype=compute_dtype, device=dev)
    if t_len <= maturity_lag or n_assets == 0:
        return out.to(torch.float32)

    n_leaders = max(1, int(round(n_assets * leader_frac)))
    ok = torch.isfinite(liq) & torch.isfinite(r)
    ok_cnt = ok.sum(dim=1)
    liq_masked = torch.where(ok, liq, float("-inf"))
    order = torch.argsort(liq_masked, dim=1, stable=True)
    top_idx = order[:, -n_leaders:]
    r_ok = torch.where(ok, r, 0.0)
    top_rets = torch.gather(r_ok, 1, top_idx)
    leader_ret = torch.where(ok_cnt >= n_leaders, top_rets.mean(dim=1), 0.0)

    t_start = 2 * maturity_lag + 9
    if window < 10 or t_start >= t_len:
        return out.to(torch.float32)

    # Pares causales centrados en bloques L2 vía unfold (elimina cancelación E[X^2]-E[X]^2)
    x_seq = leader_ret[:-maturity_lag]  # [T - lag]
    y_seq = r[maturity_lag:]  # [T - lag, N]
    pad_x = torch.full((window - 1,), float("nan"), dtype=compute_dtype, device=dev)
    pad_y = torch.full((window - 1, n_assets), float("nan"), dtype=compute_dtype, device=dev)
    xw_all = torch.cat([pad_x, x_seq], dim=0).unfold(0, window, 1)  # [T - lag, W]
    yw_all = torch.cat([pad_y, y_seq], dim=0).unfold(0, window, 1)  # [T - lag, N, W]

    # Para t in [t_start, T-1]: e_idx = t - 2*lag + 1 => ventana e_idx - 1 in [9, T - 2*lag)
    n_valid = t_len - t_start
    xw_valid = xw_all[9 : 9 + n_valid]  # [B, W]
    yw_valid = yw_all[9 : 9 + n_valid]  # [B, N, W]
    lead_cur = leader_ret[t_start - maturity_lag : t_len - maturity_lag]  # [B]

    chunk_b = max(1, min(n_valid, _l2_chunk_elems(compute_dtype) // max(n_assets * window, 1)))
    eps_sq_scale = 64.0 * (torch.finfo(compute_dtype).eps ** 2)
    for b0 in range(0, n_valid, chunk_b):
        b1 = min(b0 + chunk_b, n_valid)
        xw = xw_valid[b0:b1]  # [bc, W]
        yw = yw_valid[b0:b1]  # [bc, N, W]
        in_win = torch.isfinite(xw)  # [bc, W]
        l_vec = in_win.sum(dim=-1).to(compute_dtype).clamp_min(1.0)  # [bc]
        xm = torch.where(in_win, xw, 0.0).sum(dim=-1) / l_vec  # [bc]
        xc = torch.where(in_win, xw - xm.unsqueeze(-1), 0.0)  # [bc, W]
        var_x = (xc * xc).sum(dim=-1)  # [bc]
        var_x = torch.where(var_x <= eps_sq_scale * (xm * xm) * l_vec, 0.0, var_x)

        ok_y = in_win.unsqueeze(1) & torch.isfinite(yw)  # [bc, N, W]
        cnt_y = ok_y.sum(dim=-1).to(compute_dtype).clamp_min(1.0)  # [bc, N]
        ym = torch.where(ok_y, yw, 0.0).sum(dim=-1) / cnt_y  # [bc, N]
        yc = torch.where(ok_y, yw - ym.unsqueeze(-1), 0.0)  # [bc, N, W]
        cov_xy = torch.bmm(yc, xc.unsqueeze(-1)).squeeze(-1)  # [bc, N]
        var_y = (yc * yc).sum(dim=-1)  # [bc, N]
        var_y = torch.where(var_y <= eps_sq_scale * (ym * ym) * cnt_y, 0.0, var_y)

        var_x_col = var_x.unsqueeze(1)
        beta = cov_xy / (var_x_col + 1e-10)
        denom_c = torch.sqrt((var_x_col * var_y.clamp_min(1e-12)).clamp_min(0.0))
        corr = torch.where(var_y > 1e-12, cov_xy / denom_c.clamp_min(1e-12), 0.0)
        r2 = (corr * corr).clamp(0.0, 0.999)
        dof = (l_vec - 2.0).clamp_min(1.0).unsqueeze(1)
        t2 = dof * r2 / (1.0 - r2).clamp_min(1e-6)
        shrink = (t2 / (t2 + 3.0)).clamp(0.0, 1.0)

        signal = shrink * beta * (lead_cur[b0:b1] - xm).unsqueeze(1)
        signal = torch.where(var_x_col > 1e-12, signal, 0.0)
        out[t_start + b0 : t_start + b1] = torch.nan_to_num(
            signal, nan=0.0, posinf=0.0, neginf=0.0
        )
    return out.to(torch.float32)


def batched_macro_surprises_and_betas_torch(
    returns_panel: torch.Tensor | np.ndarray,
    macro: torch.Tensor | np.ndarray,
    *,
    halflife: float = 21.0,
    beta_window: int = 63,
    maturity_lag: int = 1,
    ridge_lambda: float = 1.0,
    compute_dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sorpresas macro EWMA en scan afín O(log T) y betas ridge por lotes (`linalg.solve`)."""
    if maturity_lag < 0:
        raise ValueError("maturity_lag debe ser >= 0 para preservar causalidad estricta")
    r = _as_float_tensor(returns_panel, dtype=compute_dtype)
    dev = r.device
    m = _as_float_tensor(macro, dtype=compute_dtype, device=dev)
    if m.ndim == 1:
        m = m.unsqueeze(1)
    t_len, n_assets = r.shape
    k_mac = m.shape[1]
    if t_len == 0 or k_mac == 0:
        return (
            torch.zeros((t_len, k_mac), dtype=torch.float32, device=dev),
            torch.zeros((t_len, n_assets, k_mac), dtype=torch.float32, device=dev),
        )

    decay = math.pow(0.5, 1.0 / max(halflife, 1.0))
    fin = torch.isfinite(m)
    m_safe = torch.where(fin, m, 0.0)

    # Conteo de observaciones finitas estrictamente previas a t
    seen_after = torch.cumsum(fin.to(compute_dtype), dim=0)
    seen_before = torch.zeros_like(seen_after)
    if t_len > 1:
        seen_before[1:] = seen_after[:-1]

    alpha = torch.where(seen_before < 1.0, 1.0, 1.0 - decay).to(compute_dtype)
    a_mean = torch.where(fin, 1.0 - alpha, 1.0)
    b_mean = torch.where(fin, alpha * m_safe, 0.0)
    mean_after = _causal_affine_scan(a_mean, b_mean, init=0.0)
    mean_before = torch.zeros_like(mean_after)
    if t_len > 1:
        mean_before[1:] = mean_after[:-1]

    diff = torch.where(fin, m_safe - mean_before, 0.0)
    a_var = torch.where(fin, decay, 1.0).to(compute_dtype)
    b_var = torch.where(fin, (1.0 - decay) * (diff * diff), 0.0)
    var_after = _causal_affine_scan(a_var, b_var, init=1.0)
    var_before = torch.ones_like(var_after)
    if t_len > 1:
        var_before[1:] = var_after[:-1]

    surprises = torch.zeros((t_len, k_mac), dtype=compute_dtype, device=dev)
    if t_len > 1:
        std = torch.sqrt(var_before[1:].clamp_min(1e-6))
        z_mac = ((m_safe[1:] - mean_before[1:]) / std).clamp(-6.0, 6.0)
        surprises[1:] = torch.where(fin[1:] & (seen_before[1:] >= 3.0), z_mac, 0.0)

    # Betas ridge dinámicas vectorizadas por lotes con torch.linalg.solve (sin syncs CPU<->GPU)
    betas = torch.zeros((t_len, n_assets, k_mac), dtype=compute_dtype, device=dev)
    min_obs = max(k_mac + 4, 12)
    t_start = max(0, min_obs + maturity_lag - 1)
    if beta_window < min_obs or t_start >= t_len or n_assets == 0:
        return surprises.to(torch.float32), betas.to(torch.float32)

    valid_t = torch.arange(t_start, t_len, device=dev)
    e_idx = valid_t - maturity_lag + 1
    s_idx = (e_idx - beta_window).clamp_min(0)
    l_vec = (e_idx - s_idx).to(compute_dtype)

    xx = surprises.unsqueeze(2) * surprises.unsqueeze(1)  # [T, K, K]
    p_xx = torch.cat(
        [torch.zeros((1, k_mac, k_mac), dtype=compute_dtype, device=dev), torch.cumsum(xx, dim=0)],
        dim=0,
    )
    eye = torch.eye(k_mac, dtype=compute_dtype, device=dev) * ridge_lambda
    gram = (p_xx[e_idx] - p_xx[s_idx]) + eye.unsqueeze(0) * (l_vec / 20.0).view(-1, 1, 1)

    y_clean = torch.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
    n_chunk = max(1, min(n_assets, _l2_chunk_elems(compute_dtype) // max(t_len * k_mac, 1)))
    for n0 in range(0, n_assets, n_chunk):
        n1 = min(n0 + n_chunk, n_assets)
        xy = surprises.unsqueeze(2) * y_clean[:, n0:n1].unsqueeze(1)  # [T, K, nc]
        p_xy = torch.cat(
            [
                torch.zeros((1, k_mac, n1 - n0), dtype=compute_dtype, device=dev),
                torch.cumsum(xy, dim=0),
            ],
            dim=0,
        )
        rhs = p_xy[e_idx] - p_xy[s_idx]  # [B, K, nc]
        with torch.amp.autocast(device_type=dev.type, enabled=False):
            if ridge_lambda > 0.0:
                sol, _ = torch.linalg.solve_ex(gram, rhs, check_errors=False)
            else:
                try:
                    sol = torch.linalg.solve(gram, rhs)
                except RuntimeError:  # pragma: no cover - fallback solo si ridge_lambda == 0
                    sol = torch.linalg.lstsq(gram, rhs).solution
        betas[valid_t, n0:n1, :] = sol.transpose(1, 2).clamp(-LIMIT, LIMIT)

    return surprises.to(torch.float32), betas.to(torch.float32)


def _cross_sectional_percentile_ranks(x: torch.Tensor) -> torch.Tensor:
    """Rangos percentiles transversales causales en `[-1, 1]` vectorizados sobre `[T, N]`."""
    t_len, n_assets = x.shape
    dev = x.device
    dtype = x.dtype
    if t_len == 0 or n_assets == 0:
        return torch.zeros((t_len, n_assets), dtype=dtype, device=dev)
    fin = torch.isfinite(x)
    cnt = fin.sum(dim=1, keepdim=True)  # [T, 1]
    filled = torch.where(fin, x, float("inf"))
    order = torch.argsort(filled, dim=1, stable=True)
    pos = torch.arange(n_assets, dtype=dtype, device=dev).unsqueeze(0)
    denom = (cnt - 1).clamp_min(1).to(dtype)
    valid_pos = (pos < cnt) & (cnt > 1)
    sorted_ranks = torch.where(valid_pos, -1.0 + 2.0 * pos / denom, 0.0)
    dst = torch.zeros_like(sorted_ranks)
    dst.scatter_(1, order, sorted_ranks)
    return dst


def compute_rmt_regime_rotation_features(
    rmt_gap_ratio_raw: torch.Tensor,
    rmt_top1_share: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None = None,
    fast_span: int = 5,
    slow_span: int = 21,
    z_window: int = 63,
    clip_val: float = LIMIT,
) -> dict[str, torch.Tensor]:
    """Calcula `rmt_gap_ratio_1_2` estandarizado y `rmt_top1_vel_21` causales sin eigh extra."""
    t_len = rmt_gap_ratio_raw.shape[0]
    dev = rmt_gap_ratio_raw.device
    if t_len == 0:
        z0 = torch.zeros(0, dtype=torch.float32, device=dev)
        return dict(
            rmt_gap_ratio_1_2=z0,
            rmt_gap_ratio_1_2_raw=z0,
            rmt_top1_share=z0,
            rmt_top1_vel_21=z0,
        )

    ac_ctx = (
        torch.amp.autocast(device_type=dev.type, enabled=False)
        if dev.type in ("cuda", "cpu")
        else contextlib.nullcontext()
    )
    with ac_ctx:
        gap_f64 = rmt_gap_ratio_raw.to(device=dev, dtype=torch.float64)
        sh_f64 = rmt_top1_share.to(device=dev, dtype=torch.float64)
        valid = torch.isfinite(gap_f64) & torch.isfinite(sh_f64)
        if valid_mask is not None:
            valid = valid & valid_mask.to(device=dev, dtype=torch.bool)

        gap_safe = torch.where(valid, gap_f64, 0.0)
        sh_safe = torch.where(valid, sh_f64, 0.0).clamp(0.0, 1.0)
        m_f = valid.to(torch.float64)

        # 1) Estandarización causal rodante de `log(clamp(lam_1 / max(lam_2, 1e-6), 1, 100))`
        zero1 = torch.zeros(1, dtype=torch.float64, device=dev)
        c_cnt = torch.cat([zero1, torch.cumsum(m_f, dim=0)], dim=0)
        c_s1 = torch.cat([zero1, torch.cumsum(gap_safe, dim=0)], dim=0)
        c_s2 = torch.cat([zero1, torch.cumsum(gap_safe * gap_safe, dim=0)], dim=0)
        hi = torch.arange(1, t_len + 1, device=dev)
        lo = (hi - max(int(z_window), 2)).clamp_min(0)
        cnt = c_cnt[hi] - c_cnt[lo]
        s1 = c_s1[hi] - c_s1[lo]
        s2 = c_s2[hi] - c_s2[lo]
        denom = cnt.clamp_min(1.0)
        mean_w = s1 / denom
        css = (s2 - (s1 * s1) / denom).clamp_min(0.0)
        css = torch.where(css <= 1e-12 * s2, 0.0, css)
        var_w = torch.where(cnt >= 2.0, css / (cnt - 1.0).clamp_min(1.0), 0.0)
        std_w = torch.sqrt(var_w.clamp_min(0.0))
        gap_z = torch.where(
            valid & (cnt >= 5.0) & (std_w > 1e-6),
            (gap_safe - mean_w) / std_w.clamp_min(1e-6),
            0.0,
        ).clamp(-clip_val, clip_val)

        # 2) Velocidad causal `ema_5(lambda_1_share) - ema_21(lambda_1_share)` en O(log T)
        seen_after = torch.cumsum(m_f, dim=0)
        seen_before = torch.zeros_like(seen_after)
        if t_len > 1:
            seen_before[1:] = seen_after[:-1]
        is_first = valid & (seen_before < 0.5)

        def _causal_ema(span: int) -> torch.Tensor:
            alpha = 2.0 / (float(max(int(span), 1)) + 1.0)
            decay = 1.0 - alpha
            a_s = torch.where(valid, torch.where(is_first, 0.0, decay), 1.0)
            b_s = torch.where(valid, torch.where(is_first, sh_safe, alpha * sh_safe), 0.0)
            return _causal_affine_scan(a_s, b_s, init=0.0)

        ema_fast = _causal_ema(fast_span)
        ema_slow = _causal_ema(slow_span)
        top1_vel = torch.where(valid, ema_fast - ema_slow, 0.0).clamp(-clip_val, clip_val)

    return dict(
        rmt_gap_ratio_1_2=gap_z.to(torch.float32),
        rmt_gap_ratio_1_2_raw=gap_safe.clamp(0.0, clip_val).to(torch.float32),
        rmt_top1_share=sh_safe.to(torch.float32),
        rmt_top1_vel_21=top1_vel.to(torch.float32),
    )


def batched_panel_physics_v4_torch(
    returns: torch.Tensor | np.ndarray,
    liquidity: torch.Tensor | np.ndarray,
    volatility: torch.Tensor | np.ndarray,
    *,
    macro: torch.Tensor | np.ndarray | None = None,
    rmt_window: int = 63,
    top_k_rmt: int = 5,
    maturity_lag: int = 1,
    lag_vol_liq: bool = False,
    compute_dtype: torch.dtype = torch.float32,
    return_extended: bool = False,
) -> dict[str, torch.Tensor]:
    """Física transversal completa v4 en PyTorch GPU/CPU con paridad causal exacta.

    Contrato de observabilidad en `t`:
      - `maturity_lag >= 1` obligatorio: toda magnitud dependiente de `returns` usa estrictamente
        sesiones `<= t - maturity_lag` (nunca retornos objetivo futuros `y_t`).
      - Si `volatility`/`liquidity` provienen del cierre de `t`, pase `lag_vol_liq=True`.
      - Por defecto opera en `compute_dtype=torch.float32` con ruta rápida dual `W×W` RMT en
        Blackwell (`compute_dtype=torch.float64` disponible para paridad de referencia).
    """
    if maturity_lag < 1:
        raise ValueError("maturity_lag debe ser >= 1 para preservar observabilidad causal en t")
    r = _as_float_tensor(returns, dtype=compute_dtype)
    dev = r.device
    liq = _as_float_tensor(liquidity, dtype=compute_dtype, device=dev)
    vol = _as_float_tensor(volatility, dtype=compute_dtype, device=dev)
    if r.ndim != 2 or liq.shape != r.shape or vol.shape != r.shape:
        raise ValueError("returns, liquidity y volatility deben ser tensores 2D [T, N] idénticos")

    t_len, n_assets = r.shape
    rmt_abs = torch.zeros(t_len, dtype=compute_dtype, device=dev)
    rmt_ent = torch.zeros(t_len, dtype=compute_dtype, device=dev)
    rmt_nmodes_raw = torch.zeros(t_len, dtype=torch.int64, device=dev)
    rmt_loadings = torch.zeros((t_len, n_assets, top_k_rmt), dtype=compute_dtype, device=dev)
    rmt_gap_raw = torch.zeros(t_len, dtype=compute_dtype, device=dev)
    rmt_top1_sh = torch.zeros(t_len, dtype=compute_dtype, device=dev)
    rmt_active_mask = torch.zeros(t_len, dtype=torch.bool, device=dev)

    # 1) Dispersión transversal estrictamente causal (std poblacional en end_m - 1)
    cross_disp = torch.zeros(t_len, dtype=compute_dtype, device=dev)
    fin_r = torch.isfinite(r)
    if t_len > 0 and n_assets > 0:
        cnt_r = fin_r.sum(dim=1)
        r_clean = torch.where(fin_r, r, 0.0)
        row_mean = r_clean.sum(dim=1, keepdim=True) / cnt_r.clamp_min(1).unsqueeze(1).to(
            compute_dtype
        )
        row_diff = torch.where(fin_r, r - row_mean, 0.0)
        row_std = torch.sqrt(
            (row_diff * row_diff).sum(dim=1) / cnt_r.clamp_min(1).to(compute_dtype)
        )
        row_std = torch.where(cnt_r >= 3, row_std, 0.0)
        start_cd = maturity_lag + 1
        if start_cd < t_len:
            cross_disp[start_cd:] = row_std[1 : t_len - maturity_lag]

    # 2) RMT Marchenko-Pastur por lotes (ruta rápida dual W×W cuando W < N)
    min_rmt = max(15, top_k_rmt + 2)
    if n_assets >= 2 and t_len > 0 and rmt_window >= min_rmt:
        if rmt_window < n_assets:
            # Ruta dual W×W: cada ventana usa O(W*N) en vez de O(N*N) -> lotes grandes en GPU
            chunk_b = max(1, _RMT_MATRIX_CHUNK_ELEMS // max(rmt_window * n_assets, 1))
            panel_no_nans = bool(fin_r.all())
        else:
            chunk_b = max(1, _RMT_MATRIX_CHUNK_ELEMS // max(n_assets * n_assets, 1))
            panel_no_nans = False
        t_min_active = max(0, min_rmt + maturity_lag - 1)
        t_warm_end = min(t_len, rmt_window + maturity_lag - 1)

        # 2a) Prefijo warm-up: start_m == 0 y min_rmt <= end_m < rmt_window
        if t_warm_end > t_min_active:
            warm_t = torch.arange(t_min_active, t_warm_end, device=dev)
            n_warm = t_warm_end - t_min_active
            max_w = t_warm_end - maturity_lag
            w_idx = torch.arange(max_w, device=dev).unsqueeze(0)
            for c0 in range(0, n_warm, chunk_b):
                c1 = min(c0 + chunk_b, n_warm)
                wt = warm_t[c0:c1]
                em = wt - maturity_lag + 1  # [bc]
                tmask = w_idx < em.unsqueeze(1)  # [bc, max_w]
                r_sub = r[:max_w].unsqueeze(0).expand(c1 - c0, -1, -1)
                res_w = _rmt_batch_core(
                    r_sub,
                    top_k=top_k_rmt,
                    time_mask=tmask,
                    w_lens=em,
                    return_clean_corr=False,
                    compute_dtype=compute_dtype,
                    assume_no_nans=panel_no_nans,
                )
                rmt_abs[wt] = res_w["absorption_ratio"]
                rmt_ent[wt] = res_w["spectral_entropy"]
                rmt_nmodes_raw[wt] = res_w["n_signal_modes"]
                rmt_loadings[wt] = res_w["loadings"]
                rmt_gap_raw[wt] = res_w["rmt_gap_ratio_1_2"]
                rmt_top1_sh[wt] = res_w["rmt_top1_share"]
                rmt_active_mask[wt] = True

        # 2b) Régimen estacionario: ventanas fijas [rmt_window, N] vía unfold sin copia
        t_steady_start = max(t_min_active, rmt_window + maturity_lag - 1)
        if t_steady_start < t_len:
            steady_t = torch.arange(t_steady_start, t_len, device=dev)
            max_em = t_len - maturity_lag
            all_win = r[:max_em].unfold(0, rmt_window, 1).transpose(1, 2)
            s_starts = steady_t - maturity_lag + 1 - rmt_window
            n_steady = t_len - t_steady_start
            for c0 in range(0, n_steady, chunk_b):
                c1 = min(c0 + chunk_b, n_steady)
                st = steady_t[c0:c1]
                r_sub = all_win[s_starts[c0:c1]]
                res_s = _rmt_batch_core(
                    r_sub,
                    top_k=top_k_rmt,
                    return_clean_corr=False,
                    compute_dtype=compute_dtype,
                    assume_no_nans=panel_no_nans,
                )
                rmt_abs[st] = res_s["absorption_ratio"]
                rmt_ent[st] = res_s["spectral_entropy"]
                rmt_nmodes_raw[st] = res_s["n_signal_modes"]
                rmt_loadings[st] = res_s["loadings"]
                rmt_gap_raw[st] = res_s["rmt_gap_ratio_1_2"]
                rmt_top1_sh[st] = res_s["rmt_top1_share"]
                rmt_active_mask[st] = True

    # 3) Transporte óptimo de Wasserstein 1D causal (FP32) y Lead-Lag centrado
    wass = batched_causal_wasserstein_shift_torch(
        r,
        short_window=5,
        long_window=min(rmt_window, 40),
        maturity_lag=maturity_lag,
        compute_dtype=compute_dtype,
    )
    lead_lag = batched_causal_lead_lag_spillover_torch(
        r,
        liq,
        window=min(rmt_window, 40),
        maturity_lag=maturity_lag,
        compute_dtype=compute_dtype,
    )

    # 4) Sorpresas y betas macro dinámicas
    if macro is not None:
        m_surp, m_betas = batched_macro_surprises_and_betas_torch(
            r,
            macro,
            beta_window=rmt_window,
            maturity_lag=maturity_lag,
            compute_dtype=compute_dtype,
        )
        if m_surp.shape[1] > 0:
            ms_c = m_surp.to(compute_dtype)
            macro_stress = torch.sqrt((ms_c * ms_c).mean(dim=1))
        else:
            macro_stress = torch.zeros(t_len, dtype=compute_dtype, device=dev)
    else:
        m_surp = torch.zeros((t_len, 0), dtype=torch.float32, device=dev)
        m_betas = torch.zeros((t_len, n_assets, 0), dtype=torch.float32, device=dev)
        macro_stress = torch.zeros(t_len, dtype=compute_dtype, device=dev)

    # 5) Descriptor de contexto de ALTO SNR [T, N, 8]
    vol_pct = _cross_sectional_percentile_ranks(vol)
    liq_pct = _cross_sectional_percentile_ranks(liq)
    if lag_vol_liq and maturity_lag > 0:
        z_pad = torch.zeros((min(maturity_lag, t_len), n_assets), dtype=compute_dtype, device=dev)
        if maturity_lag < t_len:
            vol_pct = torch.cat([z_pad, vol_pct[:-maturity_lag]], dim=0)
            liq_pct = torch.cat([z_pad, liq_pct[:-maturity_lag]], dim=0)
        else:
            vol_pct = torch.zeros_like(vol_pct)
            liq_pct = torch.zeros_like(liq_pct)
    rmt_nmodes_frac = rmt_nmodes_raw.to(compute_dtype) / max(float(n_assets), 1.0)
    wass_c = wass.to(compute_dtype)

    high_snr_ctx = torch.stack(
        [
            rmt_abs.unsqueeze(1).expand(t_len, n_assets),
            rmt_ent.unsqueeze(1).expand(t_len, n_assets),
            rmt_nmodes_frac.unsqueeze(1).expand(t_len, n_assets),
            (wass_c * 50.0).clamp(0.0, LIMIT).unsqueeze(1).expand(t_len, n_assets),
            (cross_disp * 25.0).clamp(0.0, LIMIT).unsqueeze(1).expand(t_len, n_assets),
            macro_stress.clamp(0.0, LIMIT).unsqueeze(1).expand(t_len, n_assets),
            vol_pct,
            liq_pct,
        ],
        dim=-1,
    ).to(dtype=torch.float32)

    out: dict[str, torch.Tensor] = dict(
        rmt_absorption=rmt_abs.to(torch.float32),
        rmt_entropy=rmt_ent.to(torch.float32),
        rmt_n_signal_modes=rmt_nmodes_raw,
        rmt_loadings=rmt_loadings.to(torch.float32),
        wasserstein=wass,
        cross_dispersion=cross_disp.to(torch.float32),
        lead_lag=lead_lag,
        macro_surprises=m_surp,
        macro_betas=m_betas,
        high_snr_context=high_snr_ctx,
    )
    if return_extended:
        ext_rmt = compute_rmt_regime_rotation_features(
            rmt_gap_raw,
            rmt_top1_sh,
            valid_mask=rmt_active_mask,
        )
        out["rmt_gap_ratio_1_2"] = ext_rmt["rmt_gap_ratio_1_2"]
        out["rmt_gap_ratio_1_2_raw"] = ext_rmt["rmt_gap_ratio_1_2_raw"]
        out["rmt_top1_share"] = ext_rmt["rmt_top1_share"]
        out["rmt_top1_vel_21"] = ext_rmt["rmt_top1_vel_21"]
        out["extra_rmt_regime_features"] = torch.stack(
            [
                ext_rmt["rmt_gap_ratio_1_2"].unsqueeze(1).expand(t_len, n_assets),
                ext_rmt["rmt_top1_vel_21"].unsqueeze(1).expand(t_len, n_assets),
            ],
            dim=-1,
        )
    return out


def batched_multimodal_fusion_v4_torch(
    *,
    news_emb: torch.Tensor | np.ndarray,
    news_present: torch.Tensor | np.ndarray,
    chart_emb: torch.Tensor | np.ndarray,
    chart_present: torch.Tensor | np.ndarray,
    fund_vals: torch.Tensor | np.ndarray,
    fund_present: torch.Tensor | np.ndarray,
    macro_surprises: torch.Tensor | np.ndarray,
    macro_betas: torch.Tensor | np.ndarray,
    halflife: float = 21.0,
    compute_dtype: torch.dtype = torch.float32,
    device: str | torch.device | None = None,
) -> dict[str, torch.Tensor]:
    """Extrae los 12 rasgos de `MULTIMODAL_V4_NAMES` en O(log T) sin OOM (decay `[T,N,1]` + chunks).

    Chunkea sobre el eje de activos `N` para mantener el workspace GPU muy por debajo de 1 GB
    incluso para paneles `T=500, N=2600, D=768`.
    """
    if device is not None:
        dev = torch.device(device)
    elif isinstance(news_emb, torch.Tensor):
        dev = news_emb.device
    elif isinstance(news_present, torch.Tensor):
        dev = news_present.device
    else:
        dev = torch.device("cpu")

    t_len, n_assets, d_news = (int(x) for x in news_emb.shape)
    d_chart = int(chart_emb.shape[2])
    d_fund = int(fund_vals.shape[2])

    np_t = _as_bool_tensor(news_present, device=dev)
    cp_t = _as_bool_tensor(chart_present, device=dev)
    fp_t = _as_bool_tensor(fund_present, device=dev)
    ms_raw = _as_float_tensor(macro_surprises, dtype=compute_dtype, device=dev)
    mb_raw = _as_float_tensor(macro_betas, dtype=compute_dtype, device=dev)

    decay = math.pow(0.5, 1.0 / max(halflife, 1.0))

    has_macro = (
        torch.isfinite(ms_raw).any(dim=1)
        if ms_raw.shape[1] > 0
        else torch.zeros(t_len, dtype=torch.bool, device=dev)
    )
    macro_p = has_macro.unsqueeze(1).expand(t_len, n_assets)

    if t_len == 0 or n_assets == 0:
        return dict(
            features=torch.empty(
                (t_len, n_assets, len(MULTIMODAL_V4_NAMES)), dtype=torch.float32, device=dev
            ),
            modality_presence=torch.empty((t_len, n_assets, 4), dtype=torch.float32, device=dev),
        )

    t_idx = torch.arange(t_len, dtype=torch.int64, device=dev).unsqueeze(1).expand(t_len, n_assets)
    np_f = np_t.to(compute_dtype)

    # Presupuesto de workspace por chunk: <= 8M elementos (~32 MB FP32 -> pico total < 150 MB)
    scan_chunk_budget = _L2_CHUNK_TARGET_ELEMS if compute_dtype == torch.float64 else 8_000_000

    # 1) Novedad semántica de noticias (decay escalar [T, nc, 1], chunked sobre N, cero copias)
    novelty = torch.zeros((t_len, n_assets), dtype=compute_dtype, device=dev)
    n_chunk_news = max(1, min(n_assets, scan_chunk_budget // max(t_len * max(d_news, 1), 1)))
    for n0 in range(0, n_assets, n_chunk_news):
        n1 = min(n0 + n_chunk_news, n_assets)
        ne_c = torch.nan_to_num(
            _as_float_tensor(news_emb[:, n0:n1], dtype=compute_dtype, device=dev), nan=0.0
        )
        np_c = np_t[:, n0:n1]
        np_col = np_c.unsqueeze(-1)  # [T, nc, 1]
        a_nm = torch.where(np_col, decay, 1.0).to(compute_dtype)  # [T, nc, 1] (sin expandir D!)
        b_nm = torch.where(np_col, (1.0 - decay) * ne_c, 0.0)
        mem_after = _causal_affine_scan(a_nm, b_nm, init=0.0)
        # t = 0: memoria previa es 0 -> m_norm == 0 -> novelty[0] = np_f[0]
        novelty[0, n0:n1] = np_f[0, n0:n1]
        if t_len > 1:
            ne_tail = ne_c[1:]
            mem_prev = mem_after[:-1]
            n_norm = torch.linalg.vector_norm(ne_tail, dim=-1)
            m_norm = torch.linalg.vector_norm(mem_prev, dim=-1)
            cos = (ne_tail * mem_prev).sum(dim=-1) / (n_norm * m_norm).clamp_min(1e-8)
            nov_raw = 1.0 - cos.clamp(-1.0, 1.0)
            novelty[1:, n0:n1] = torch.where(
                np_c[1:] & (m_norm > 1e-8), nov_raw, np_f[1:, n0:n1]
            )

    last_news = torch.cummax(torch.where(np_t, t_idx, -253), dim=0).values
    news_age = (t_idx - last_news).clamp(max=252).to(compute_dtype)
    news_int = _causal_affine_scan(
        torch.full_like(np_f, decay), (1.0 - decay) * np_f, init=0.0
    )

    # 2) Deriva visual de gráficos (decay escalar [T, nc, 1], chunked sobre N)
    c_drift = torch.zeros((t_len, n_assets), dtype=compute_dtype, device=dev)
    inv_sqrt_dc = 1.0 / math.sqrt(max(d_chart, 1))
    n_chunk_chart = max(1, min(n_assets, scan_chunk_budget // max(t_len * max(d_chart, 1), 1)))
    for n0 in range(0, n_assets, n_chunk_chart):
        n1 = min(n0 + n_chunk_chart, n_assets)
        ce_c = torch.nan_to_num(
            _as_float_tensor(chart_emb[:, n0:n1], dtype=compute_dtype, device=dev), nan=0.0
        )
        cp_c = cp_t[:, n0:n1]
        cp_col = cp_c.unsqueeze(-1)  # [T, nc, 1]
        a_cm = torch.where(cp_col, decay, 1.0).to(compute_dtype)
        b_cm = torch.where(cp_col, (1.0 - decay) * ce_c, 0.0)
        cmem_after = _causal_affine_scan(a_cm, b_cm, init=0.0)
        d0 = torch.linalg.vector_norm(ce_c[0], dim=-1) * inv_sqrt_dc
        c_drift[0, n0:n1] = torch.where(cp_c[0], d0, 0.0)
        if t_len > 1:
            d_tail = torch.linalg.vector_norm(ce_c[1:] - cmem_after[:-1], dim=-1) * inv_sqrt_dc
            c_drift[1:, n0:n1] = torch.where(cp_c[1:], d_tail, 0.0)

    # 3) Cambio fundamental Point-in-Time y frescura (chunked sobre N)
    last_fp = torch.cummax(torch.where(fp_t, t_idx, -1), dim=0).values
    prev_fp = torch.full_like(last_fp, -1)
    if t_len > 1:
        prev_fp[1:] = last_fp[:-1]
    f_delta = torch.zeros((t_len, n_assets), dtype=compute_dtype, device=dev)
    inv_sqrt_df = 1.0 / math.sqrt(max(d_fund, 1))
    n_chunk_fund = max(1, min(n_assets, scan_chunk_budget // max(t_len * max(d_fund, 1), 1)))
    for n0 in range(0, n_assets, n_chunk_fund):
        n1 = min(n0 + n_chunk_fund, n_assets)
        fv_c = torch.nan_to_num(
            _as_float_tensor(fund_vals[:, n0:n1], dtype=compute_dtype, device=dev), nan=0.0
        )
        pf_c = prev_fp[:, n0:n1]
        gather_idx = pf_c.clamp_min(0).unsqueeze(-1).expand(t_len, n1 - n0, d_fund)
        fmem_before = torch.where(
            (pf_c >= 0).unsqueeze(-1), torch.gather(fv_c, 0, gather_idx), 0.0
        )
        f_dist = torch.linalg.vector_norm(fv_c - fmem_before, dim=-1) * inv_sqrt_df
        f_delta[:, n0:n1] = torch.where(fp_t[:, n0:n1], f_dist, 0.0)

    changed = fp_t & (f_delta > 1e-6)
    last_chg = torch.cummax(torch.where(changed, t_idx, -253), dim=0).values
    fund_age = (t_idx - last_chg).clamp(max=252).to(compute_dtype)

    # 4) Macro tailwind: dot(asset_macro_beta_t, macro_surprise_t)
    if ms_raw.shape[1] > 0:
        ms_c = torch.nan_to_num(ms_raw, nan=0.0)
        mb_c = torch.nan_to_num(mb_raw, nan=0.0)
        tailwind = torch.einsum("tnk,tk->tn", mb_c, ms_c)
        align_abs = torch.einsum("tnk,tk->tn", mb_c.abs(), ms_c.abs())
    else:
        tailwind = torch.zeros((t_len, n_assets), dtype=compute_dtype, device=dev)
        align_abs = torch.zeros((t_len, n_assets), dtype=compute_dtype, device=dev)

    cp_f = cp_t.to(compute_dtype)
    fp_f = fp_t.to(compute_dtype)
    mp_f = macro_p.to(compute_dtype)
    completeness = (np_f + cp_f + fp_f + mp_f) / 4.0

    feats = torch.stack(
        [
            novelty,
            np_f,
            torch.log1p(news_age),
            news_int,
            c_drift,
            cp_f,
            f_delta,
            fp_f,
            torch.log1p(fund_age),
            tailwind,
            align_abs,
            completeness,
        ],
        dim=-1,
    )
    feats = torch.nan_to_num(feats, nan=0.0, posinf=LIMIT, neginf=-LIMIT).clamp(-LIMIT, LIMIT)
    presence = torch.stack([np_t, cp_t, fp_t, macro_p], dim=-1).to(dtype=torch.float32)
    return dict(features=feats.to(dtype=torch.float32), modality_presence=presence)


class ZeroCopySessionView:
    """Vista cronológica sin copia sobre un `ZeroCopySessionStore` compatible con `Sequence`."""

    def __init__(
        self,
        store: ZeroCopySessionStore,
        indices: np.ndarray,
        *,
        gaussian_rank_keys: Sequence[str] | None = None,
        apply_missingness_featurizer: bool | None = None,
    ) -> None:
        self._store = store
        self._indices = np.ascontiguousarray(indices, dtype=np.int64)
        self._gaussian_rank_keys = (
            tuple(gaussian_rank_keys) if gaussian_rank_keys is not None else None
        )
        self._apply_missingness_featurizer = apply_missingness_featurizer

    @property
    def indices(self) -> np.ndarray:
        return self._indices

    @property
    def session_years(self) -> np.ndarray:
        return np.asarray(self._store.session_years[self._indices], dtype=np.int64)

    def __len__(self) -> int:
        return int(len(self._indices))

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if isinstance(idx, slice):
            return ZeroCopySessionView(
                self._store,
                self._indices[idx],
                gaussian_rank_keys=self._gaussian_rank_keys,
                apply_missingness_featurizer=self._apply_missingness_featurizer,
            )  # type: ignore[return-value]
        raw_i = int(self._indices[idx])
        return self._store.get_session(
            raw_i,
            zero_copy=True,
            gaussian_rank_keys=self._gaussian_rank_keys,
            apply_missingness_featurizer=self._apply_missingness_featurizer,
        )

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        for raw_i in self._indices:
            yield self._store.get_session(
                int(raw_i),
                zero_copy=True,
                gaussian_rank_keys=self._gaussian_rank_keys,
                apply_missingness_featurizer=self._apply_missingness_featurizer,
            )

    def iter_pinned_h2d(
        self,
        device: str | torch.device = "cpu",
        *,
        use_pinned: bool = True,
        gaussian_rank_keys: Sequence[str] | None = None,
        apply_missingness_featurizer: bool | None = None,
        slab_sessions: int = 64,
        slab_max_bytes: int = 128 * 1024 * 1024,
        gpu_resident: bool = False,
    ) -> Iterator[dict[str, torch.Tensor]]:
        """Transmite sesiones con staging por slabs pre-fijados y stream CUDA."""
        eff_keys = (
            gaussian_rank_keys if gaussian_rank_keys is not None else self._gaussian_rank_keys
        )
        eff_feat = (
            apply_missingness_featurizer
            if apply_missingness_featurizer is not None
            else self._apply_missingness_featurizer
        )
        return self._store.iter_pinned_h2d(
            device=device,
            indices=self._indices,
            use_pinned=use_pinned,
            gaussian_rank_keys=eff_keys,
            apply_missingness_featurizer=eff_feat,
            slab_sessions=slab_sessions,
            slab_max_bytes=slab_max_bytes,
            gpu_resident=gpu_resident,
        )


def _sota_enrichment_module() -> Any:
    """Importación perezosa de `dataset_enrichment_sota` (evita ciclos con `gaussian_rank`)."""
    from . import dataset_enrichment_sota as des

    return des


def _market_macro_indices_module() -> Any:
    """Importación perezosa de `market_indices_macro_sota` (evita ciclos con `gaussian_rank`)."""
    from . import market_indices_macro_sota as mims

    return mims


def _remap_encoded_modality_columns(x: torch.Tensor) -> torch.Tensor:
    """Remapea columnas de `encoded-history-v3.1` a orden MMARRS `(0, 3, 4, 1, 2)` (0 syncs)."""
    if x.ndim >= 2 and x.shape[-1] == 5:
        return torch.stack((x[..., 0], x[..., 3], x[..., 4], x[..., 1], x[..., 2]), dim=-1)
    return x


def _apply_missingness_featurizer_to_session(
    batch: dict[str, torch.Tensor],
    featurizer: Any,
    *,
    modality_mask_layout: str = "model",
    fundamental_layout: Literal["auto", "encoded_v31_78", "raw"] = "auto",
    macro_layout: Literal["auto", "encoded_v31_420", "raw"] = "auto",
) -> dict[str, torch.Tensor]:
    """Aplica `MissingnessResilientFeaturizer` a una sesión `[N, ...]` con 0 host-syncs."""
    work = batch
    if modality_mask_layout == "encoded_v31":
        work = dict(batch)
        if isinstance(work.get("modality_mask"), torch.Tensor):
            work["modality_mask"] = _remap_encoded_modality_columns(work["modality_mask"])
        if isinstance(work.get("modality_age"), torch.Tensor):
            work["modality_age"] = _remap_encoded_modality_columns(work["modality_age"])
    return featurizer.featurize_batch(
        work,
        fundamental_layout=fundamental_layout,
        macro_layout=macro_layout,
    )


class ZeroCopySessionStore:
    """Almacén multimodal empaquetado sobre `mmap` (`np.load(..., mmap_mode='r')`).

    Mantiene los 24 años (`2000–2023`, ~6.000 sesiones × 2.600 activos) en una sola copia del
    page-cache del SO compartida entre 6–8 workers CUDA MPS en 1× RTX PRO 6000 (96 GB VRAM),
    con `asset_ids: int64[N]` canónicos por sesión, indexación O(1), purga/embargo anti-fuga
    multi-horizonte obligatorios, rangos gaussianos transversales opcionales y H2D de alto caudal
    mediante búferes fijados high-water `max_N`, slabs contiguos pre-fijados y modo `gpu_resident`.
    """

    CANONICAL_KEYS: tuple[str, ...] = STORE_ARRAY_KEYS

    def __init__(
        self,
        store_dir: Path,
        *,
        arrays: dict[str, np.ndarray],
        session_arrays: dict[str, np.ndarray] | None = None,
        session_years: np.ndarray,
        session_offsets: np.ndarray | None,
        layout: str,
        manifest: dict[str, Any],
        gaussian_rank_keys: Sequence[str] = (),
        apply_missingness_featurizer: bool = False,
        featurizer_kwargs: Mapping[str, Any] | None = None,
        modality_mask_layout: str | None = None,
    ) -> None:
        eff_mod_layout = str(
            modality_mask_layout
            if modality_mask_layout is not None
            else manifest.get("modality_mask_layout", "model")
        )
        if eff_mod_layout not in ("model", "encoded_v31"):
            raise ValueError(
                f"modality_mask_layout inválido {eff_mod_layout!r}; "
                "válidos: ('model', 'encoded_v31')"
            )
        self.store_dir = Path(store_dir)
        self._arrays = arrays
        self._session_arrays: dict[str, np.ndarray] = dict(session_arrays or {})
        self.session_years = session_years
        self.session_offsets = session_offsets
        self.layout = layout
        self.manifest = manifest
        self.gaussian_rank_keys = tuple(gaussian_rank_keys)
        self.apply_missingness_featurizer = bool(apply_missingness_featurizer)
        self.modality_mask_layout = eff_mod_layout
        self._featurizer_kwargs: dict[str, Any] = dict(featurizer_kwargs or {})
        self._featurizer_cache: Any | None = None
        self._pinned_slab_cache: list[dict[str, torch.Tensor]] = [{}, {}]
        self._pinned_session_cache: list[dict[str, torch.Tensor]] = [{}, {}]
        self._gpu_resident_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
        self._h2d_streams: dict[int, torch.cuda.Stream] = {}
        self._h2d_events: dict[int, tuple[torch.cuda.Event, torch.cuda.Event]] = {}
        self._copy_executor: ThreadPoolExecutor | None = None

    def _get_missingness_featurizer(self) -> Any:
        if self._featurizer_cache is None:
            des = _sota_enrichment_module()
            self._featurizer_cache = des.MissingnessResilientFeaturizer(
                **self._featurizer_kwargs
            ).eval()
        return self._featurizer_cache

    def clear_pinned_cache(self) -> None:
        """Libera los búferes fijados (`pin_memory`) reutilizables y el hilo de copia."""
        self._pinned_slab_cache = [{}, {}]
        self._pinned_session_cache = [{}, {}]
        if self._copy_executor is not None:
            self._copy_executor.shutdown(wait=True)
            self._copy_executor = None

    def __del__(self) -> None:  # pragma: no cover - limpieza defensiva de hilo
        ex = getattr(self, "_copy_executor", None)
        if ex is not None:
            ex.shutdown(wait=False)

    def clear_gpu_cache(self) -> None:
        """Libera los slabs residentes en VRAM creados con `gpu_resident=True`."""
        self._gpu_resident_cache.clear()

    @staticmethod
    def _validate_or_build_asset_ids(
        np_arrays: dict[str, np.ndarray],
        *,
        layout: str,
        t_len: int,
        offsets: np.ndarray | None,
    ) -> np.ndarray:
        """Valida `asset_ids: int64` únicos por sesión o genera IDs canónicos `0..N-1`."""
        if "asset_ids" in np_arrays:
            raw_ids = np_arrays["asset_ids"]
            if not np.issubdtype(raw_ids.dtype, np.integer):
                raise ValueError(
                    f"asset_ids debe tener dtype entero (int64); recibido {raw_ids.dtype}"
                )
            ids = np.ascontiguousarray(raw_ids, dtype=np.int64)
            if layout == "dense":
                if ids.ndim != 2 or ids.shape[0] != t_len:
                    raise ValueError(
                        f"asset_ids en layout='dense' debe tener forma [T, N]; recibido {ids.shape}"
                    )
                for s_i in range(t_len):
                    row = ids[s_i]
                    if len(np.unique(row)) != len(row):
                        raise ValueError(
                            f"asset_ids duplicados detectados dentro de la sesión {s_i}"
                        )
            else:
                assert offsets is not None
                if ids.ndim != 1 or ids.shape[0] != int(offsets[-1]):
                    raise ValueError(
                        f"asset_ids en layout='ragged' debe tener forma [TotalRows]; "
                        f"recibido {ids.shape}"
                    )
                for s_i in range(t_len):
                    row = ids[int(offsets[s_i]) : int(offsets[s_i + 1])]
                    if len(np.unique(row)) != len(row):
                        raise ValueError(
                            f"asset_ids duplicados detectados dentro de la sesión {s_i}"
                        )
            return ids

        sample_arr = next(iter(np_arrays.values()))
        if layout == "dense":
            n_assets = int(sample_arr.shape[1]) if sample_arr.ndim >= 2 else 1
            return np.ascontiguousarray(
                np.broadcast_to(np.arange(n_assets, dtype=np.int64)[None, :], (t_len, n_assets))
            )
        assert offsets is not None
        chunks = [
            np.arange(int(offsets[s_i + 1] - offsets[s_i]), dtype=np.int64)
            for s_i in range(t_len)
        ]
        return np.ascontiguousarray(np.concatenate(chunks) if chunks else np.empty(0, np.int64))

    @classmethod
    def save(
        cls,
        store_dir: str | Path,
        *,
        session_years: np.ndarray | Sequence[int],
        arrays: Mapping[str, np.ndarray | torch.Tensor] | None = None,
        session_arrays: Mapping[str, np.ndarray | torch.Tensor] | None = None,
        session_offsets: np.ndarray | Sequence[int] | None = None,
        modality_mask_layout: str = "model",
        **named_arrays: np.ndarray | torch.Tensor,
    ) -> Path:
        """Persiste tensores contiguos `.npy` + `asset_ids.npy` + `manifest.json` para `mmap`."""
        if modality_mask_layout not in ("model", "encoded_v31"):
            raise ValueError(
                f"modality_mask_layout inválido {modality_mask_layout!r}; "
                "válidos: ('model', 'encoded_v31')"
            )
        out_dir = Path(store_dir)
        years = np.ascontiguousarray(np.asarray(session_years, dtype=np.int64))
        if years.ndim != 1 or len(years) == 0:
            raise ValueError("session_years debe ser un vector 1D no vacío")
        if len(years) > 1 and np.any(np.diff(years) < 0):
            raise ValueError("session_years debe estar en orden estrictamente cronológico")

        merged: dict[str, np.ndarray | torch.Tensor] = {}
        if arrays is not None:
            merged.update(arrays)
        merged.update(named_arrays)
        if not merged:
            raise ValueError("Se requiere al menos un array multimodal para guardar el store")

        t_len = int(len(years))
        offsets: np.ndarray | None = None
        if session_offsets is not None:
            offsets = np.ascontiguousarray(np.asarray(session_offsets, dtype=np.int64))
            if offsets.ndim != 1 or len(offsets) != t_len + 1:
                raise ValueError("session_offsets debe tener longitud len(session_years) + 1")
            if offsets[0] != 0 or np.any(np.diff(offsets) < 0):
                raise ValueError("session_offsets debe comenzar en 0 y ser no decreciente")
            layout = "ragged"
            expected_rows = int(offsets[-1])
        else:
            layout = "dense"
            expected_rows = t_len

        np_arrays: dict[str, np.ndarray] = {}
        for key, raw_val in merged.items():
            arr = _to_numpy_contiguous(raw_val)
            if arr.ndim < 1 or arr.shape[0] != expected_rows:
                raise ValueError(
                    f"Array {key!r} tiene eje 0 = {arr.shape[0] if arr.ndim else None}, "
                    f"esperado {expected_rows} para layout={layout!r}"
                )
            np_arrays[key] = arr

        np_arrays["asset_ids"] = cls._validate_or_build_asset_ids(
            np_arrays, layout=layout, t_len=t_len, offsets=offsets
        )

        np_session_arrays: dict[str, np.ndarray] = {}
        if session_arrays is not None:
            for skey, raw_sval in session_arrays.items():
                if skey in np_arrays:
                    raise ValueError(
                        f"Clave de sesión {skey!r} colisiona con un array por activo del store"
                    )
                sarr = _to_numpy_contiguous(raw_sval)
                if sarr.ndim < 1 or sarr.shape[0] != t_len:
                    raise ValueError(
                        f"Session array {skey!r} tiene eje 0 = "
                        f"{sarr.shape[0] if sarr.ndim else None}, esperado {t_len}"
                    )
                np_session_arrays[skey] = sarr

        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "session_years.npy", years)
        if offsets is not None:
            np.save(out_dir / "session_offsets.npy", offsets)

        shapes: dict[str, list[int]] = {}
        dtypes: dict[str, str] = {}
        total_bytes = int(years.nbytes + (offsets.nbytes if offsets is not None else 0))
        for key, arr in np_arrays.items():
            np.save(out_dir / f"{key}.npy", arr)
            shapes[key] = list(arr.shape)
            dtypes[key] = str(arr.dtype)
            total_bytes += int(arr.nbytes)
        for skey, sarr in np_session_arrays.items():
            np.save(out_dir / f"{skey}.npy", sarr)
            shapes[skey] = list(sarr.shape)
            dtypes[skey] = str(sarr.dtype)
            total_bytes += int(sarr.nbytes)

        manifest = dict(
            schema_version=_STORE_SCHEMA_VERSION,
            layout=layout,
            modality_mask_layout=modality_mask_layout,
            keys=list(np_arrays.keys()),
            shapes=shapes,
            dtypes=dtypes,
            n_sessions=t_len,
            year_min=int(years[0]),
            year_max=int(years[-1]),
            total_bytes=total_bytes,
        )
        if np_session_arrays:
            manifest["session_keys"] = list(np_session_arrays.keys())
        tmp_manifest = out_dir / "manifest.json.tmp"
        tmp_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        tmp_manifest.replace(out_dir / "manifest.json")
        return out_dir

    @classmethod
    def from_sessions(
        cls,
        store_dir: str | Path,
        sessions: Sequence[Mapping[str, np.ndarray | torch.Tensor]],
        session_years: Sequence[int] | np.ndarray,
        *,
        apply_missingness_featurizer: bool = False,
        featurizer_kwargs: Mapping[str, Any] | None = None,
        modality_mask_layout: str = "model",
    ) -> ZeroCopySessionStore:
        """Empaqueta sesiones en `ZeroCopySessionStore` (layout dense o ragged)."""
        if not sessions:
            raise ValueError("sessions no puede estar vacío")
        if len(sessions) != len(session_years):
            raise ValueError("sessions y session_years deben tener exactamente la misma longitud")

        first_keys = tuple(sessions[0].keys())
        if not first_keys:
            raise ValueError("Cada sesión debe contener al menos una clave tensorial")

        counts: list[int] = []
        for idx, sess in enumerate(sessions):
            if set(sess.keys()) != set(first_keys):
                raise ValueError(f"Claves inconsistentes en la sesión {idx}")
            n_i = int(_to_numpy_contiguous(sess[first_keys[0]]).shape[0])
            for key in first_keys[1:]:
                if int(_to_numpy_contiguous(sess[key]).shape[0]) != n_i:
                    raise ValueError(
                        f"Número de activos inconsistente para {key!r} en sesión {idx}"
                    )
            counts.append(n_i)

        uniform = len(set(counts)) == 1
        packed: dict[str, np.ndarray] = {}
        offsets: np.ndarray | None = None
        if uniform:
            for key in first_keys:
                packed[key] = np.stack(
                    [_to_numpy_contiguous(s[key]) for s in sessions], axis=0
                )
        else:
            offsets = np.concatenate([[0], np.cumsum(np.asarray(counts, dtype=np.int64))])
            for key in first_keys:
                packed[key] = np.concatenate(
                    [_to_numpy_contiguous(s[key]) for s in sessions], axis=0
                )

        cls.save(
            store_dir,
            session_years=session_years,
            arrays=packed,
            session_offsets=offsets,
            modality_mask_layout=modality_mask_layout,
        )
        return cls.load(
            store_dir,
            mmap_mode="r",
            apply_missingness_featurizer=apply_missingness_featurizer,
            featurizer_kwargs=featurizer_kwargs,
            modality_mask_layout=modality_mask_layout,
        )

    @classmethod
    def load(
        cls,
        store_dir: str | Path,
        *,
        mmap_mode: str = "r",
        gaussian_rank_keys: Sequence[str] = (),
        apply_missingness_featurizer: bool = False,
        featurizer_kwargs: Mapping[str, Any] | None = None,
        modality_mask_layout: str | None = None,
    ) -> ZeroCopySessionStore:
        """Carga el almacén vía `np.load(..., mmap_mode='r')` compartiendo page-cache del SO."""
        in_dir = Path(store_dir)
        manifest_path = in_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"No se encontró manifest.json en {in_dir}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        years = np.load(in_dir / "session_years.npy", mmap_mode=mmap_mode)
        offsets_path = in_dir / "session_offsets.npy"
        offsets = np.load(offsets_path, mmap_mode=mmap_mode) if offsets_path.exists() else None
        arrays = {
            key: np.load(in_dir / f"{key}.npy", mmap_mode=mmap_mode)
            for key in manifest["keys"]
        }
        session_arrays = {
            skey: np.load(in_dir / f"{skey}.npy", mmap_mode=mmap_mode)
            for skey in manifest.get("session_keys", ())
        }
        return cls(
            in_dir,
            arrays=arrays,
            session_arrays=session_arrays,
            session_years=years,
            session_offsets=offsets,
            layout=str(manifest["layout"]),
            manifest=manifest,
            gaussian_rank_keys=gaussian_rank_keys,
            apply_missingness_featurizer=apply_missingness_featurizer,
            featurizer_kwargs=featurizer_kwargs,
            modality_mask_layout=modality_mask_layout,
        )

    def __len__(self) -> int:
        return int(len(self.session_years))

    def keys(self) -> tuple[str, ...]:
        return tuple(self._arrays.keys())

    def session_keys(self) -> tuple[str, ...]:
        """Claves de arrays agregados por sesión `[T, ...]` (p. ej. `market_regime_indices`)."""
        return tuple(self._session_arrays.keys())

    @property
    def market_regime_indices(self) -> np.ndarray | None:
        """Vista memmap `[T, 18]` de índices de régimen de mercado si están adjuntos."""
        return self._session_arrays.get("market_regime_indices")

    def raw_array(self, key: str) -> np.ndarray:
        """Devuelve la vista/memmap subyacente para inspección o verificación zero-copy."""
        if key in self._arrays:
            return self._arrays[key]
        return self._session_arrays[key]

    def _normalize_index(self, index: int) -> int:
        t_len = len(self)
        idx = index + t_len if index < 0 else index
        if idx < 0 or idx >= t_len:
            raise IndexError(f"Índice de sesión {index} fuera de rango [0, {t_len})")
        return idx

    def get_session(
        self,
        index: int,
        *,
        zero_copy: bool = True,
        gaussian_rank_keys: Sequence[str] | None = None,
        apply_missingness_featurizer: bool | None = None,
    ) -> dict[str, torch.Tensor]:
        """Recupera una sesión en O(1) compartiendo memoria del memmap sin copiar."""
        idx = self._normalize_index(index)
        rank_set = set(
            self.gaussian_rank_keys if gaussian_rank_keys is None else gaussian_rank_keys
        )
        eff_feat = (
            self.apply_missingness_featurizer
            if apply_missingness_featurizer is None
            else bool(apply_missingness_featurizer)
        )
        out: dict[str, torch.Tensor] = {}
        if self.layout == "dense":
            for key, arr in self._arrays.items():
                view = arr[idx]
                t = _zero_copy_tensor(view) if zero_copy else torch.from_numpy(view.copy())
                if key in rank_set and t.dtype.is_floating_point and t.ndim == 2:
                    t = gaussian_rank_features(t)
                out[key] = t
        else:
            assert self.session_offsets is not None
            r0 = int(self.session_offsets[idx])
            r1 = int(self.session_offsets[idx + 1])
            for key, arr in self._arrays.items():
                view = arr[r0:r1]
                t = _zero_copy_tensor(view) if zero_copy else torch.from_numpy(view.copy())
                if key in rank_set and t.dtype.is_floating_point and t.ndim == 2:
                    t = gaussian_rank_features(t)
                out[key] = t
        for skey, sarr in self._session_arrays.items():
            sview = sarr[idx]
            out[skey] = _zero_copy_tensor(sview) if zero_copy else torch.from_numpy(sview.copy())
        if eff_feat:
            out = _apply_missingness_featurizer_to_session(
                out,
                self._get_missingness_featurizer(),
                modality_mask_layout=self.modality_mask_layout,
            )
        elif self.modality_mask_layout == "encoded_v31":
            if isinstance(out.get("modality_mask"), torch.Tensor):
                out["modality_mask"] = _remap_encoded_modality_columns(out["modality_mask"])
            if isinstance(out.get("modality_age"), torch.Tensor):
                out["modality_age"] = _remap_encoded_modality_columns(out["modality_age"])
        return out

    def get_slice(
        self, start: int, stop: int, *, zero_copy: bool = True
    ) -> dict[str, torch.Tensor]:
        """Devuelve un corte contiguo `[start:stop]` de sesiones como tensores zero-copy O(1)."""
        t_len = len(self)
        s0 = max(0, start + t_len if start < 0 else start)
        s1 = min(t_len, stop + t_len if stop < 0 else stop)
        if s1 < s0:
            s1 = s0
        out: dict[str, torch.Tensor] = {}
        if self.layout == "dense":
            for key, arr in self._arrays.items():
                view = arr[s0:s1]
                out[key] = _zero_copy_tensor(view) if zero_copy else torch.from_numpy(view.copy())
        else:
            assert self.session_offsets is not None
            r0 = int(self.session_offsets[s0])
            r1 = int(self.session_offsets[s1])
            for key, arr in self._arrays.items():
                view = arr[r0:r1]
                out[key] = _zero_copy_tensor(view) if zero_copy else torch.from_numpy(view.copy())
        for skey, sarr in self._session_arrays.items():
            sview = sarr[s0:s1]
            out[skey] = _zero_copy_tensor(sview) if zero_copy else torch.from_numpy(sview.copy())
        return out

    def __getitem__(self, idx: int | slice) -> dict[str, torch.Tensor] | ZeroCopySessionView:
        if isinstance(idx, slice):
            indices = np.arange(len(self), dtype=np.int64)[idx]
            return ZeroCopySessionView(self, indices)
        return self.get_session(int(idx), zero_copy=True)

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        for idx in range(len(self)):
            yield self.get_session(idx, zero_copy=True)

    def slice_years(
        self,
        start_year: int,
        end_year: int,
        *,
        sealed_test_start_year: int = 2024,
        allow_sealed_test: bool = False,
    ) -> ZeroCopySessionView:
        """Vista cronológica por rango inclusivo de años con sello estricto `2024+`."""
        if end_year < start_year:
            raise ValueError("end_year debe ser >= start_year")
        if end_year >= sealed_test_start_year and not allow_sealed_test:
            raise ValueError(
                f"Acceso denegado: bloque {sealed_test_start_year}+ sellado "
                "(use allow_sealed_test=True únicamente tras congelar el modelo final)"
            )
        years = np.asarray(self.session_years, dtype=np.int64)
        idx = np.flatnonzero((years >= start_year) & (years <= end_year))
        return ZeroCopySessionView(self, idx)

    def chronological_splits(
        self,
        *,
        train_start_year: int = 2000,
        train_end_year: int = 2022,
        val_year: int = 2023,
        sealed_test_start_year: int = 2024,
        active_horizons: Sequence[int] = (1, 5, 10, 21, 42, 63),
        purge_sessions: int | None = None,
        embargo_sessions: int | None = None,
        allow_zero_purge: bool = False,
        allow_sealed_test: bool = False,
    ) -> dict[str, Any]:
        """Partición temporal estricta (`2000–2022` train, `2023` val, `2024+` sellado).

        Exige por defecto `purge_sessions >= max(active_horizons)` y
        `embargo_sessions >= max(5, ceil(0.2 * max_h))` para impedir solape multi-horizonte
        entre train y val (use `allow_zero_purge=True` únicamente en pruebas sintéticas cortas).
        """
        if train_end_year < train_start_year:
            raise ValueError("train_end_year debe ser >= train_start_year")
        if val_year <= train_end_year:
            raise ValueError("val_year debe ser estrictamente posterior a train_end_year")
        if val_year >= sealed_test_start_year:
            raise ValueError("val_year debe preceder estrictamente al test sellado (2024+)")

        horizons = [int(h) for h in active_horizons]
        if not horizons or any(h < 1 for h in horizons):
            raise ValueError("active_horizons debe contener al menos un horizonte >= 1")
        max_h = max(horizons)
        min_purge = max_h
        min_embargo = max(5, math.ceil(0.2 * max_h))

        eff_purge = min_purge if purge_sessions is None else int(purge_sessions)
        eff_embargo = min_embargo if embargo_sessions is None else int(embargo_sessions)
        if eff_purge < 0 or eff_embargo < 0:
            raise ValueError("purge_sessions y embargo_sessions deben ser >= 0")
        if not allow_zero_purge:
            if eff_purge < min_purge:
                raise ValueError(
                    f"purge_sessions={eff_purge} viola el mínimo anti-fuga "
                    f"max(active_horizons)={min_purge}; pase allow_zero_purge=True solo en tests"
                )
            if eff_embargo < min_embargo:
                raise ValueError(
                    f"embargo_sessions={eff_embargo} viola el mínimo anti-fuga "
                    f"max(5, ceil(0.2*{max_h}))={min_embargo}; "
                    "pase allow_zero_purge=True solo en tests"
                )

        years = np.asarray(self.session_years, dtype=np.int64)
        train_idx = np.flatnonzero((years >= train_start_year) & (years <= train_end_year))
        val_idx = np.flatnonzero(
            (years > train_end_year)
            & (years <= val_year)
            & (years < sealed_test_start_year)
        )
        sealed_idx = np.flatnonzero(years >= sealed_test_start_year)

        if eff_purge > 0:
            train_idx = (
                train_idx[:-eff_purge] if len(train_idx) > eff_purge else np.empty(0, np.int64)
            )
        if eff_embargo > 0:
            val_idx = (
                val_idx[eff_embargo:] if len(val_idx) > eff_embargo else np.empty(0, np.int64)
            )

        if allow_sealed_test:
            exposed_sealed = sealed_idx
            n_withheld = 0
        else:
            exposed_sealed = np.empty(0, dtype=np.int64)
            n_withheld = int(len(sealed_idx))

        return dict(
            train=ZeroCopySessionView(self, train_idx),
            val=ZeroCopySessionView(self, val_idx),
            sealed_test=ZeroCopySessionView(self, exposed_sealed),
            train_indices=train_idx,
            val_indices=val_idx,
            purge_sessions=eff_purge,
            embargo_sessions=eff_embargo,
            max_horizon=max_h,
            n_sealed_withheld=n_withheld,
            sealed_test_accessible=bool(allow_sealed_test),
        )

    def _is_contiguous_span(self, idx_arr: np.ndarray) -> bool:
        n = len(idx_arr)
        if n <= 1:
            return True
        if int(idx_arr[-1]) - int(idx_arr[0]) != n - 1:
            return False
        return bool(np.all(np.diff(idx_arr) == 1))

    def _iter_gpu_resident(
        self,
        dev: torch.device,
        idx_arr: np.ndarray,
        rank_tuple: tuple[str, ...],
        apply_featurizer: bool = False,
    ) -> Iterator[dict[str, torch.Tensor]]:
        """Modo residente en VRAM: carga 1 sola vez el bloque en GPU y entrega vistas O(1)."""
        is_contig = self._is_contiguous_span(idx_arr)
        if is_contig:
            s0 = self._normalize_index(int(idx_arr[0]))
            s1 = s0 + len(idx_arr)
            cache_key: tuple[Any, ...] = (
                str(dev),
                "contig",
                s0,
                s1,
                rank_tuple,
                bool(apply_featurizer),
            )
        else:
            norm_idx = np.asarray([self._normalize_index(int(i)) for i in idx_arr], dtype=np.int64)
            cache_key = (
                str(dev),
                "gather",
                norm_idx.tobytes(),
                rank_tuple,
                bool(apply_featurizer),
            )

        entry = self._gpu_resident_cache.get(cache_key)
        rank_set = set(rank_tuple)
        if entry is None:
            gpu_arrays: dict[str, torch.Tensor] = {}
            featurizer = self._get_missingness_featurizer() if apply_featurizer else None
            if is_contig:
                s0 = self._normalize_index(int(idx_arr[0]))
                s1 = s0 + len(idx_arr)
                if self.layout == "dense":
                    for key, arr in self._arrays.items():
                        t_gpu = _zero_copy_tensor(arr[s0:s1]).to(dev, non_blocking=False)
                        if key in rank_set and t_gpu.dtype.is_floating_point and t_gpu.ndim == 3:
                            t_gpu = gaussian_rank_features(t_gpu)
                        if (
                            featurizer is None
                            and self.modality_mask_layout == "encoded_v31"
                            and key in ("modality_mask", "modality_age")
                        ):
                            t_gpu = _remap_encoded_modality_columns(t_gpu)
                        gpu_arrays[key] = t_gpu
                    for skey, sarr in self._session_arrays.items():
                        gpu_arrays[skey] = _zero_copy_tensor(sarr[s0:s1]).to(
                            dev, non_blocking=False
                        )
                    if featurizer is not None:
                        sess_gpu = [
                            _apply_missingness_featurizer_to_session(
                                {key: t[i] for key, t in gpu_arrays.items()},
                                featurizer,
                                modality_mask_layout=self.modality_mask_layout,
                            )
                            for i in range(s1 - s0)
                        ]
                        entry = dict(mode="list", sessions=sess_gpu)
                    else:
                        entry = dict(mode="dense", arrays=gpu_arrays, n_sessions=s1 - s0)
                else:
                    assert self.session_offsets is not None
                    r0 = int(self.session_offsets[s0])
                    r1 = int(self.session_offsets[s1])
                    rel_offsets = np.asarray(self.session_offsets[s0 : s1 + 1], dtype=np.int64) - r0
                    for key, arr in self._arrays.items():
                        t_gpu = _zero_copy_tensor(arr[r0:r1]).to(dev, non_blocking=False)
                        if key in rank_set and t_gpu.dtype.is_floating_point and t_gpu.ndim == 2:
                            t_gpu = t_gpu.clone()
                            for loc_i in range(s1 - s0):
                                o0 = int(rel_offsets[loc_i])
                                o1 = int(rel_offsets[loc_i + 1])
                                t_gpu[o0:o1] = gaussian_rank_features(t_gpu[o0:o1])
                        if (
                            featurizer is None
                            and self.modality_mask_layout == "encoded_v31"
                            and key in ("modality_mask", "modality_age")
                        ):
                            t_gpu = _remap_encoded_modality_columns(t_gpu)
                        gpu_arrays[key] = t_gpu
                    gpu_sess_arrays = {
                        skey: _zero_copy_tensor(sarr[s0:s1]).to(dev, non_blocking=False)
                        for skey, sarr in self._session_arrays.items()
                    }
                    if featurizer is not None:
                        sess_gpu = []
                        for loc_i in range(s1 - s0):
                            o0 = int(rel_offsets[loc_i])
                            o1 = int(rel_offsets[loc_i + 1])
                            raw_loc = {key: t[o0:o1] for key, t in gpu_arrays.items()}
                            for skey, st in gpu_sess_arrays.items():
                                raw_loc[skey] = st[loc_i]
                            sess_gpu.append(
                                _apply_missingness_featurizer_to_session(
                                    raw_loc,
                                    featurizer,
                                    modality_mask_layout=self.modality_mask_layout,
                                )
                            )
                        entry = dict(mode="list", sessions=sess_gpu)
                    else:
                        entry = dict(
                            mode="ragged",
                            arrays=gpu_arrays,
                            session_arrays=gpu_sess_arrays,
                            rel_offsets=rel_offsets,
                            n_sessions=s1 - s0,
                        )
            else:
                norm_idx = np.asarray(
                    [self._normalize_index(int(i)) for i in idx_arr], dtype=np.int64
                )
                sess_list: list[dict[str, torch.Tensor]] = []
                for raw_i in norm_idx:
                    cpu_b = self.get_session(
                        int(raw_i),
                        zero_copy=True,
                        gaussian_rank_keys=(),
                        apply_missingness_featurizer=False,
                    )
                    gpu_b: dict[str, torch.Tensor] = {}
                    for key, src in cpu_b.items():
                        t_gpu = src.to(dev, non_blocking=False)
                        if key in rank_set and t_gpu.dtype.is_floating_point and t_gpu.ndim == 2:
                            t_gpu = gaussian_rank_features(t_gpu)
                        gpu_b[key] = t_gpu
                    if featurizer is not None:
                        # `get_session(..., apply_missingness_featurizer=False)` ya remapeó
                        # `encoded_v31 -> model` una vez; pasar `"model"` evita doble permutación!
                        gpu_b = _apply_missingness_featurizer_to_session(
                            gpu_b,
                            featurizer,
                            modality_mask_layout="model",
                        )
                    sess_list.append(gpu_b)
                entry = dict(mode="list", sessions=sess_list)
            self._gpu_resident_cache[cache_key] = entry

        mode = entry["mode"]
        if mode == "dense":
            arrays = entry["arrays"]
            for i in range(entry["n_sessions"]):
                yield {key: t[i] for key, t in arrays.items()}
        elif mode == "ragged":
            arrays = entry["arrays"]
            sess_arrays = entry.get("session_arrays", {})
            rel_offsets = entry["rel_offsets"]
            for i in range(entry["n_sessions"]):
                o0 = int(rel_offsets[i])
                o1 = int(rel_offsets[i + 1])
                out_rag = {key: t[o0:o1] for key, t in arrays.items()}
                for skey, st in sess_arrays.items():
                    out_rag[skey] = st[i]
                yield out_rag
        else:
            yield from entry["sessions"]

    def _iter_contiguous_slab_h2d(
        self,
        dev: torch.device,
        s0: int,
        s1: int,
        *,
        use_pinned: bool,
        rank_set: set[str],
        apply_featurizer: bool,
        slab_sessions: int,
        slab_max_bytes: int,
    ) -> Iterator[dict[str, torch.Tensor]]:
        """Staging contiguo por slabs pre-fijados con copia CPU solapada en hilo secundario."""
        n_total = s1 - s0
        if self.layout == "dense":
            span_bytes = sum(
                int(arr[s0:s1].nbytes) for arr in self._arrays.values()
            )
        else:
            assert self.session_offsets is not None
            r_start = int(self.session_offsets[s0])
            r_stop = int(self.session_offsets[s1])
            span_bytes = sum(
                int(arr[r_start:r_stop].nbytes) for arr in self._arrays.values()
            )
        avg_sess_bytes = max(1, span_bytes // max(n_total, 1))
        eff_slab = max(
            1,
            min(int(slab_sessions), n_total, max(1, int(slab_max_bytes // avg_sess_bytes))),
        )
        slab_ranges = [
            (a0, min(a0 + eff_slab, s1)) for a0 in range(s0, s1, eff_slab)
        ]

        # Preasignar / reusar búferes high-water para el tamaño máximo de slab
        if use_pinned:
            max_sess_lead = max(a1 - a0 for a0, a1 in slab_ranges)
            if self.layout == "dense":
                max_lead = max_sess_lead
            else:
                assert self.session_offsets is not None
                max_lead = max(
                    int(self.session_offsets[a1]) - int(self.session_offsets[a0])
                    for a0, a1 in slab_ranges
                )
            n_slots = min(2, len(slab_ranges))
            for slot in range(n_slots):
                bufs = self._pinned_slab_cache[slot]
                for key, arr in self._arrays.items():
                    sample_t = _zero_copy_tensor(arr[:1])
                    tail_shape = sample_t.shape[1:]
                    prev = bufs.get(key)
                    if (
                        prev is None
                        or prev.shape[0] < max_lead
                        or prev.shape[1:] != tail_shape
                        or prev.dtype != sample_t.dtype
                    ):
                        bufs[key] = torch.empty(
                            (max_lead, *tail_shape), dtype=sample_t.dtype, pin_memory=True
                        )
                for skey, sarr in self._session_arrays.items():
                    sample_t = _zero_copy_tensor(sarr[:1])
                    tail_shape = sample_t.shape[1:]
                    prev = bufs.get(skey)
                    if (
                        prev is None
                        or prev.shape[0] < max_sess_lead
                        or prev.shape[1:] != tail_shape
                        or prev.dtype != sample_t.dtype
                    ):
                        bufs[skey] = torch.empty(
                            (max_sess_lead, *tail_shape), dtype=sample_t.dtype, pin_memory=True
                        )

        dev_idx = 0 if dev.index is None else int(dev.index)
        stream = self._h2d_streams.get(dev_idx)
        if stream is None:
            stream = torch.cuda.Stream(device=dev)
            self._h2d_streams[dev_idx] = stream
        ev_pair = self._h2d_events.get(dev_idx)
        if ev_pair is None:
            ev_pair = (torch.cuda.Event(), torch.cuda.Event())
            self._h2d_events[dev_idx] = ev_pair
        slot_events = [ev_pair[0], ev_pair[1]]
        slot_in_flight = [False, False]

        def _copy_slab_to_pinned(a0: int, a1: int, slot: int) -> int:
            bufs = self._pinned_slab_cache[slot]
            s_lead = a1 - a0
            for skey, sarr in self._session_arrays.items():
                src = _zero_copy_tensor(sarr[a0:a1])
                dst = bufs[skey][:s_lead]
                nbytes = src.numel() * src.element_size()
                if nbytes > 0 and src.is_contiguous() and dst.is_contiguous():
                    ctypes.memmove(dst.data_ptr(), src.data_ptr(), nbytes)
                else:
                    dst.copy_(src, non_blocking=False)
            if self.layout == "dense":
                lead = s_lead
                for key, arr in self._arrays.items():
                    src = _zero_copy_tensor(arr[a0:a1])
                    dst = bufs[key][:lead]
                    nbytes = src.numel() * src.element_size()
                    if nbytes > 0 and src.is_contiguous() and dst.is_contiguous():
                        # ctypes.memmove evita contención OpenMP entre 8 workers CUDA MPS
                        ctypes.memmove(dst.data_ptr(), src.data_ptr(), nbytes)
                    else:
                        dst.copy_(src, non_blocking=False)
                return lead
            assert self.session_offsets is not None
            r0 = int(self.session_offsets[a0])
            r1 = int(self.session_offsets[a1])
            lead = r1 - r0
            for key, arr in self._arrays.items():
                src = _zero_copy_tensor(arr[r0:r1])
                dst = bufs[key][:lead]
                nbytes = src.numel() * src.element_size()
                if nbytes > 0 and src.is_contiguous() and dst.is_contiguous():
                    ctypes.memmove(dst.data_ptr(), src.data_ptr(), nbytes)
                else:
                    dst.copy_(src, non_blocking=False)
            return lead

        def _enqueue_slab_h2d(a0: int, a1: int, slot: int, lead: int) -> dict[str, torch.Tensor]:
            gpu_slab: dict[str, torch.Tensor] = {}
            s_lead = a1 - a0
            if use_pinned:
                bufs = self._pinned_slab_cache[slot]
                for key in self._arrays:
                    gpu_slab[key] = bufs[key][:lead].to(dev, non_blocking=True)
                for skey in self._session_arrays:
                    gpu_slab[skey] = bufs[skey][:s_lead].to(dev, non_blocking=True)
                slot_events[slot].record(stream)
                slot_in_flight[slot] = True
            else:
                if self.layout == "dense":
                    for key, arr in self._arrays.items():
                        gpu_slab[key] = _zero_copy_tensor(arr[a0:a1]).to(dev, non_blocking=True)
                else:
                    assert self.session_offsets is not None
                    r0 = int(self.session_offsets[a0])
                    r1 = int(self.session_offsets[a1])
                    for key, arr in self._arrays.items():
                        gpu_slab[key] = _zero_copy_tensor(arr[r0:r1]).to(dev, non_blocking=True)
                for skey, sarr in self._session_arrays.items():
                    gpu_slab[skey] = _zero_copy_tensor(sarr[a0:a1]).to(dev, non_blocking=True)
            return gpu_slab

        executor: ThreadPoolExecutor | None = None
        if use_pinned and len(slab_ranges) > 1:
            if self._copy_executor is None:
                self._copy_executor = ThreadPoolExecutor(max_workers=1)
            executor = self._copy_executor
        pending_copy: Future[int] | None = None
        try:
            # Slab 0: staging inicial + DMA asíncrono + solapamiento inmediato con copia slab 1
            a0_0, a1_0 = slab_ranges[0]
            lead_0 = _copy_slab_to_pinned(a0_0, a1_0, 0) if use_pinned else (a1_0 - a0_0)
            with torch.cuda.stream(stream):
                cur_gpu_slab = _enqueue_slab_h2d(a0_0, a1_0, 0, lead_0)
            if executor is not None and len(slab_ranges) > 1:
                na0, na1 = slab_ranges[1]
                pending_copy = executor.submit(_copy_slab_to_pinned, na0, na1, 1)

            for s_idx, (a0, a1) in enumerate(slab_ranges):
                cur_stream = torch.cuda.current_stream(dev)
                cur_stream.wait_stream(stream)
                for tensor in cur_gpu_slab.values():
                    tensor.record_stream(cur_stream)

                # Preparar DMA del siguiente slab mientras el consumidor recorre el slab actual
                next_gpu_slab: dict[str, torch.Tensor] | None = None
                if s_idx + 1 < len(slab_ranges):
                    next_slot = (s_idx + 1) & 1
                    na0, na1 = slab_ranges[s_idx + 1]
                    if pending_copy is not None:
                        next_lead = pending_copy.result()
                        pending_copy = None
                    elif use_pinned:
                        if slot_in_flight[next_slot]:
                            slot_events[next_slot].synchronize()
                            slot_in_flight[next_slot] = False
                        next_lead = _copy_slab_to_pinned(na0, na1, next_slot)
                    else:
                        next_lead = na1 - na0
                    with torch.cuda.stream(stream):
                        next_gpu_slab = _enqueue_slab_h2d(na0, na1, next_slot, next_lead)
                    # Lanzar copia CPU del slab s_idx + 2 solapada con DMA y cómputo
                    if executor is not None and s_idx + 2 < len(slab_ranges):
                        nn_slot = (s_idx + 2) & 1
                        if slot_in_flight[nn_slot]:
                            slot_events[nn_slot].synchronize()
                            slot_in_flight[nn_slot] = False
                        nna0, nna1 = slab_ranges[s_idx + 2]
                        pending_copy = executor.submit(_copy_slab_to_pinned, nna0, nna1, nn_slot)

                featurizer = self._get_missingness_featurizer() if apply_featurizer else None
                if self.layout == "dense":
                    if rank_set:
                        for key in rank_set:
                            t_g = cur_gpu_slab.get(key)
                            if t_g is not None and t_g.dtype.is_floating_point and t_g.ndim == 3:
                                cur_gpu_slab[key] = gaussian_rank_features(t_g)
                    for loc_i in range(a1 - a0):
                        out_sess = {key: t[loc_i] for key, t in cur_gpu_slab.items()}
                        if featurizer is not None:
                            out_sess = _apply_missingness_featurizer_to_session(
                                out_sess,
                                featurizer,
                                modality_mask_layout=self.modality_mask_layout,
                            )
                        elif self.modality_mask_layout == "encoded_v31":
                            if isinstance(out_sess.get("modality_mask"), torch.Tensor):
                                out_sess["modality_mask"] = _remap_encoded_modality_columns(
                                    out_sess["modality_mask"]
                                )
                            if isinstance(out_sess.get("modality_age"), torch.Tensor):
                                out_sess["modality_age"] = _remap_encoded_modality_columns(
                                    out_sess["modality_age"]
                                )
                        yield out_sess
                else:
                    assert self.session_offsets is not None
                    r0_base = int(self.session_offsets[a0])
                    for sess_i in range(a0, a1):
                        o0 = int(self.session_offsets[sess_i]) - r0_base
                        o1 = int(self.session_offsets[sess_i + 1]) - r0_base
                        loc_s = sess_i - a0
                        out_sess = {
                            key: (t[loc_s] if key in self._session_arrays else t[o0:o1])
                            for key, t in cur_gpu_slab.items()
                        }
                        if rank_set:
                            for key in rank_set:
                                t_g = out_sess.get(key)
                                if (
                                    t_g is not None
                                    and t_g.dtype.is_floating_point
                                    and t_g.ndim == 2
                                ):
                                    out_sess[key] = gaussian_rank_features(t_g)
                        if featurizer is not None:
                            out_sess = _apply_missingness_featurizer_to_session(
                                out_sess,
                                featurizer,
                                modality_mask_layout=self.modality_mask_layout,
                            )
                        elif self.modality_mask_layout == "encoded_v31":
                            if isinstance(out_sess.get("modality_mask"), torch.Tensor):
                                out_sess["modality_mask"] = _remap_encoded_modality_columns(
                                    out_sess["modality_mask"]
                                )
                            if isinstance(out_sess.get("modality_age"), torch.Tensor):
                                out_sess["modality_age"] = _remap_encoded_modality_columns(
                                    out_sess["modality_age"]
                                )
                        yield out_sess

                if next_gpu_slab is not None:
                    cur_gpu_slab = next_gpu_slab
        finally:
            if pending_copy is not None:
                pending_copy.result()
            for slot in range(2):
                if slot_in_flight[slot]:
                    slot_events[slot].synchronize()
                    slot_in_flight[slot] = False

    def iter_pinned_h2d(
        self,
        device: str | torch.device = "cpu",
        *,
        indices: Sequence[int] | np.ndarray | None = None,
        use_pinned: bool = True,
        gaussian_rank_keys: Sequence[str] | None = None,
        apply_missingness_featurizer: bool | None = None,
        slab_sessions: int = 64,
        slab_max_bytes: int = 128 * 1024 * 1024,
        gpu_resident: bool = False,
    ) -> Iterator[dict[str, torch.Tensor]]:
        """Itera sesiones con H2D de alto caudal en CUDA (slabs pre-fijados / high-water / GPU).

        En CPU entrega vistas zero-copy sin invocar APIs CUDA. En GPU CUDA:
          - `gpu_resident=True`: carga una sola vez el bloque en VRAM y sirve vistas O(1).
          - Bloques contiguos con `slab_sessions > 1`: staging por slabs contiguos pre-fijados
            con copia `mmap -> pinned` en hilo secundario solapada con DMA (`>25 GB/s`).
          - `slab_sessions == 1` o índices no contiguos: doble búfer fijado high-water `max_N`
            (crecimiento monotónico sin `cudaHostAlloc`/`cudaFreeHost` por sesión en ragged).
        """
        dev = torch.device(device)
        idx_arr = (
            np.arange(len(self), dtype=np.int64)
            if indices is None
            else np.asarray(indices, dtype=np.int64)
        )
        if len(idx_arr) == 0:
            return

        eff_rank_tuple = tuple(
            self.gaussian_rank_keys if gaussian_rank_keys is None else gaussian_rank_keys
        )
        eff_feat = (
            self.apply_missingness_featurizer
            if apply_missingness_featurizer is None
            else bool(apply_missingness_featurizer)
        )

        if dev.type != "cuda":
            for raw_i in idx_arr:
                yield self.get_session(
                    int(raw_i),
                    zero_copy=True,
                    gaussian_rank_keys=eff_rank_tuple,
                    apply_missingness_featurizer=eff_feat,
                )
            return

        if gpu_resident:
            yield from self._iter_gpu_resident(
                dev, idx_arr, eff_rank_tuple, apply_featurizer=eff_feat
            )
            return

        if slab_sessions > 1 and len(idx_arr) > 1 and self._is_contiguous_span(idx_arr):
            s0 = self._normalize_index(int(idx_arr[0]))
            s1 = s0 + len(idx_arr)
            yield from self._iter_contiguous_slab_h2d(
                dev,
                s0,
                s1,
                use_pinned=use_pinned,
                rank_set=set(eff_rank_tuple),
                apply_featurizer=eff_feat,
                slab_sessions=slab_sessions,
                slab_max_bytes=slab_max_bytes,
            )
            return

        # Ruta por sesión con doble búfer fijado high-water max_N (cero realloc en ragged)
        if self.layout == "ragged" and self.session_offsets is not None:
            norm_idx = np.asarray([self._normalize_index(int(i)) for i in idx_arr], dtype=np.int64)
            counts = self.session_offsets[norm_idx + 1] - self.session_offsets[norm_idx]
            max_n = int(counts.max()) if len(counts) > 0 else 0
        else:
            max_n = 0

        stream = torch.cuda.Stream(device=dev)
        staging: list[dict[str, torch.Tensor]] = [{}, {}]
        slot_events = [torch.cuda.Event(), torch.cuda.Event()]
        slot_in_flight = [False, False]

        def _stage_and_send(raw_i: int, slot: int) -> dict[str, torch.Tensor]:
            if use_pinned and slot_in_flight[slot]:
                slot_events[slot].synchronize()
                slot_in_flight[slot] = False
            cpu_batch = self.get_session(
                raw_i,
                zero_copy=True,
                gaussian_rank_keys=eff_rank_tuple,
                apply_missingness_featurizer=False,
            )
            out_gpu: dict[str, torch.Tensor] = {}
            bufs = staging[slot]
            for key, src in cpu_batch.items():
                if use_pinned:
                    n_t = src.shape[0]
                    cap_n = max(max_n, n_t)
                    prev = bufs.get(key)
                    if (
                        prev is None
                        or prev.shape[0] < n_t
                        or prev.shape[1:] != src.shape[1:]
                        or prev.dtype != src.dtype
                    ):
                        prev = torch.empty(
                            (cap_n, *src.shape[1:]), dtype=src.dtype, pin_memory=True
                        )
                        bufs[key] = prev
                    dst = prev if n_t == prev.shape[0] else prev[:n_t]
                    dst.copy_(src, non_blocking=False)
                    out_gpu[key] = dst.to(dev, non_blocking=True)
                else:
                    out_gpu[key] = src.to(dev, non_blocking=True)
            if use_pinned:
                slot_events[slot].record(stream)
                slot_in_flight[slot] = True
            return out_gpu

        featurizer = self._get_missingness_featurizer() if eff_feat else None
        with torch.cuda.stream(stream):
            nxt = _stage_and_send(int(idx_arr[0]), 0)
        for pos in range(len(idx_arr)):
            cur_stream = torch.cuda.current_stream(dev)
            cur_stream.wait_stream(stream)
            cur = nxt
            for tensor in cur.values():
                tensor.record_stream(cur_stream)
            if pos + 1 < len(idx_arr):
                with torch.cuda.stream(stream):
                    nxt = _stage_and_send(int(idx_arr[pos + 1]), (pos + 1) & 1)
            if featurizer is not None:
                # `_stage_and_send` llama a `get_session`, que ya remapeó `encoded_v31 -> model`!
                cur = _apply_missingness_featurizer_to_session(
                    cur, featurizer, modality_mask_layout="model"
                )
            yield cur

    def with_sota_enrichment(
        self,
        output_dir: str | Path,
        *,
        ohlcv: Mapping[str, np.ndarray | torch.Tensor] | None = None,
        ohlcv_layout: Literal["store", "canonical"] = "store",
        ohlcv_price_cols: tuple[int, int, int, int, int] = (0, 1, 2, 3, 4),
        prices_are_ohlcv: bool = False,
        append_to_physics_v4: bool = True,
        store_enriched_key: str = "enriched_sota",
        apply_missingness_featurizer: bool = True,
        featurizer_kwargs: Mapping[str, Any] | None = None,
        modality_mask_layout: str | None = None,
        fundamental_layout: Literal["auto", "encoded_v31_78", "raw"] = "auto",
        macro_layout: Literal["auto", "encoded_v31_420", "raw"] = "auto",
        staleness_halflife_days: float = 63.0,
        device: str | torch.device = "cpu",
        output_dtype: np.dtype | type = np.float32,
        sealed_test_start_year: int = 2024,
        allow_sealed_test: bool = False,
        run_audit: bool = False,
        include_market_macro_indices: bool = False,
    ) -> tuple[ZeroCopySessionStore, dict[str, Any]]:
        """Construye un `ZeroCopySessionStore` enriquecido SOTA (señales causales + resiliencia)."""
        return enrich_session_store_sota(
            self,
            output_dir,
            ohlcv=ohlcv,
            ohlcv_layout=ohlcv_layout,
            ohlcv_price_cols=ohlcv_price_cols,
            prices_are_ohlcv=prices_are_ohlcv,
            append_to_physics_v4=append_to_physics_v4,
            store_enriched_key=store_enriched_key,
            apply_missingness_featurizer=apply_missingness_featurizer,
            featurizer_kwargs=featurizer_kwargs,
            modality_mask_layout=modality_mask_layout,
            fundamental_layout=fundamental_layout,
            macro_layout=macro_layout,
            staleness_halflife_days=staleness_halflife_days,
            device=device,
            output_dtype=output_dtype,
            sealed_test_start_year=sealed_test_start_year,
            allow_sealed_test=allow_sealed_test,
            run_audit=run_audit,
            include_market_macro_indices=include_market_macro_indices,
        )

    def memory_footprint_report(self, *, workers: int = 6) -> dict[str, object]:
        """Informe de huella page-cache compartida para despliegue MPS en RTX PRO 6000."""
        per_key_bytes = {k: int(v.nbytes) for k, v in self._arrays.items()}
        for sk, sv in self._session_arrays.items():
            per_key_bytes[sk] = int(sv.nbytes)
        total_bytes = sum(per_key_bytes.values()) + int(self.session_years.nbytes)
        if self.session_offsets is not None:
            total_bytes += int(self.session_offsets.nbytes)
        all_mmap = all(
            isinstance(v, np.memmap)
            for v in (*self._arrays.values(), *self._session_arrays.values())
        )
        avg_session_bytes = total_bytes / max(len(self), 1)
        pinned_staging_bytes_total = int(math.ceil(avg_session_bytes * 2.0 * max(workers, 1)))
        return dict(
            layout=self.layout,
            n_sessions=len(self),
            is_mmap_backed=all_mmap,
            shared_ram_duplication_factor=1 if all_mmap else max(workers, 1),
            total_bytes=total_bytes,
            total_mib=round(total_bytes / (1024.0**2), 3),
            workers=workers,
            pinned_staging_bytes_total=pinned_staging_bytes_total,
            per_key_bytes=per_key_bytes,
        )


def _derive_causal_ohlcv_from_prices(
    prices_tnf: np.ndarray,
    *,
    ohlcv_price_cols: tuple[int, int, int, int, int] = (0, 1, 2, 3, 4),
    prices_are_ohlcv: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    """Extrae OHLCV crudo (si `prices_are_ohlcv=True`) o deriva proxy causal sin mirar futuro."""
    t_len, n_assets, f_dim = prices_tnf.shape
    if prices_are_ohlcv:
        if f_dim < max(ohlcv_price_cols) + 1:
            raise ValueError(
                f"prices_are_ohlcv=True requiere al menos {max(ohlcv_price_cols) + 1} columnas; "
                f"recibido {f_dim}"
            )
        o_c, h_c, l_c, c_c, v_c = ohlcv_price_cols
        return (
            prices_tnf[:, :, o_c].astype(np.float32, copy=False),
            prices_tnf[:, :, h_c].astype(np.float32, copy=False),
            prices_tnf[:, :, l_c].astype(np.float32, copy=False),
            prices_tnf[:, :, c_c].astype(np.float32, copy=False),
            prices_tnf[:, :, v_c].astype(np.float32, copy=False),
            "store_prices_ohlcv",
        )

    # Proxy estrictamente causal (por activo en `<= t`) — nunca depende de valores futuros t > t_cut
    p0 = prices_tnf[:, :, 0].astype(np.float64, copy=False)
    valid = np.isfinite(p0)
    ret = np.where(valid, np.clip(p0 * 0.015, -0.25, 0.25), 0.0)
    log_c = np.log(100.0) + np.cumsum(ret, axis=0)
    c_px = np.where(valid, np.exp(np.clip(log_c, -10.0, 15.0)), np.nan)
    o_px = c_px.copy()
    if t_len > 1:
        o_px[1:] = np.where(valid[1:] & valid[:-1], c_px[:-1], c_px[1:])
    spread = (
        np.clip(np.abs(np.nan_to_num(prices_tnf[:, :, 1], nan=0.0)) * 0.01, 0.001, 0.08)
        if f_dim > 1
        else np.full((t_len, n_assets), 0.01, dtype=np.float64)
    )
    h_px = np.maximum(o_px, c_px) * np.exp(spread)
    l_px = np.minimum(o_px, c_px) * np.exp(-spread)
    v_sig = (
        np.clip(np.nan_to_num(prices_tnf[:, :, 2], nan=0.0), -4.0, 4.0)
        if f_dim > 2
        else np.zeros((t_len, n_assets), dtype=np.float64)
    )
    vol = np.where(valid, np.exp(12.0 + v_sig), 0.0)
    return (
        o_px.astype(np.float32),
        h_px.astype(np.float32),
        l_px.astype(np.float32),
        c_px.astype(np.float32),
        vol.astype(np.float32),
        "derived_causal_proxy",
    )


def _scatter_ragged_to_panel(
    arr: np.ndarray,
    row_t: np.ndarray,
    row_col: np.ndarray,
    t_len: int,
    n_uniq: int,
    *,
    fill_value: float = float("nan"),
) -> np.ndarray:
    tail = arr.shape[1:]
    dtype = np.float32 if np.issubdtype(arr.dtype, np.floating) else arr.dtype
    out = np.full((t_len, n_uniq, *tail), fill_value, dtype=dtype)
    out[row_t, row_col] = arr.astype(dtype, copy=False)
    return out


def enrich_session_store_sota(
    store: ZeroCopySessionStore,
    output_dir: str | Path,
    *,
    ohlcv: Mapping[str, np.ndarray | torch.Tensor] | None = None,
    ohlcv_layout: Literal["store", "canonical"] = "store",
    ohlcv_price_cols: tuple[int, int, int, int, int] = (0, 1, 2, 3, 4),
    prices_are_ohlcv: bool = False,
    append_to_physics_v4: bool = True,
    store_enriched_key: str = "enriched_sota",
    apply_missingness_featurizer: bool = True,
    featurizer_kwargs: Mapping[str, Any] | None = None,
    modality_mask_layout: str | None = None,
    fundamental_layout: Literal["auto", "encoded_v31_78", "raw"] = "auto",
    macro_layout: Literal["auto", "encoded_v31_420", "raw"] = "auto",
    staleness_halflife_days: float = 63.0,
    device: str | torch.device = "cpu",
    output_dtype: np.dtype | type = np.float32,
    sealed_test_start_year: int = 2024,
    allow_sealed_test: bool = False,
    run_audit: bool = False,
    include_market_macro_indices: bool = False,
) -> tuple[ZeroCopySessionStore, dict[str, Any]]:
    """Enriquece causalmente un `ZeroCopySessionStore` (dense o ragged) con señales SOTA.

    - Aplica sello estricto `< 2024` (salvo `allow_sealed_test=True`).
    - Calcula `ENRICHED_SOTA_NAMES` en precisión maestra FP32 (bajo `_autocast_disabled`,
      usando `solve_ex(..., check_errors=False)` sin syncs host).
    - Aplica opcionalmente `MissingnessResilientFeaturizer` sobre las modalidades base y
      `physics_v4` base **antes** de concatenar `enriched_sota` (cero recorte de señales dispersas
      CN) y remapeando `ENCODED_TO_MODEL_MODALITY_INDEX`.
    - Persiste un nuevo `ZeroCopySessionStore` sin mutar jamás el almacén fuente.
    """
    des = _sota_enrichment_module()
    years = np.asarray(store.session_years, dtype=np.int64)
    max_year = int(np.max(years)) if years.size else 0
    if max_year >= sealed_test_start_year and not allow_sealed_test:
        raise ValueError(
            f"Acceso denegado: bloque {sealed_test_start_year}+ sellado en store "
            f"(max_year={max_year}); pase allow_sealed_test=True únicamente en evaluación final"
        )

    out_path = Path(output_dir)
    out_res = out_path.resolve()
    src_res = store.store_dir.resolve()
    if out_res == src_res or src_res in out_res.parents:
        raise ValueError(
            f"Directorio fuente de solo lectura protegido: output_dir={out_res} "
            f"no puede coincidir ni estar dentro de store_dir={src_res}"
        )

    eff_mod_layout = str(
        modality_mask_layout if modality_mask_layout is not None else store.modality_mask_layout
    )
    if eff_mod_layout not in ("model", "encoded_v31"):
        raise ValueError(f"modality_mask_layout inválido: {eff_mod_layout!r}")
    if ohlcv_layout not in ("store", "canonical"):
        raise ValueError(f"ohlcv_layout inválido: {ohlcv_layout!r}")

    dev = torch.device(device)
    t_len = len(store)
    raw_ids = np.asarray(store.raw_array("asset_ids"), dtype=np.int64)

    # Mapear layout dense/ragged a panel causal alineado `[T, N_uniq]` por `asset_ids`
    if store.layout == "dense" and bool(np.all(raw_ids == raw_ids[0:1])):
        needs_scatter = False
        n_panel = int(raw_ids.shape[1])
        row_t = np.empty(0, dtype=np.int64)
        row_col = np.empty(0, dtype=np.int64)
    else:
        needs_scatter = True
        if store.layout == "dense":
            n_row_t = int(raw_ids.shape[1])
            flat_ids = raw_ids.reshape(-1)
            row_t = np.repeat(np.arange(t_len, dtype=np.int64), n_row_t)
        else:
            assert store.session_offsets is not None
            offs = np.asarray(store.session_offsets, dtype=np.int64)
            counts = np.diff(offs)
            flat_ids = raw_ids
            row_t = np.repeat(np.arange(t_len, dtype=np.int64), counts)
        uniq_ids = np.unique(flat_ids)
        n_panel = int(len(uniq_ids))
        row_col = np.searchsorted(uniq_ids, flat_ids).astype(np.int64)

    def _to_panel(key: str, *, fill: float = float("nan")) -> np.ndarray | None:
        if key not in store.keys():
            return None
        raw_a = store.raw_array(key)
        arr = np.array(raw_a, copy=True) if not raw_a.flags.writeable else np.asarray(raw_a)
        if not needs_scatter:
            return arr
        flat = arr.reshape(-1, *arr.shape[2:]) if store.layout == "dense" else arr
        return _scatter_ragged_to_panel(flat, row_t, row_col, t_len, n_panel, fill_value=fill)

    # 1) Obtener OHLCV causal `[T, N_panel]` (alineado siempre con `asset_ids` en dense/ragged!)
    ohlcv_source = "explicit_ohlcv"
    if ohlcv is not None:
        o_p, h_p, l_p, c_p, v_p = (
            _to_numpy_contiguous(ohlcv[k]).astype(np.float32, copy=False)
            for k in ("open", "high", "low", "close", "volume")
        )
        if needs_scatter and ohlcv_layout == "store":
            if store.layout == "ragged":
                if o_p.ndim != 1 or o_p.shape[0] != raw_ids.shape[0]:
                    raise ValueError(
                        f"ohlcv en layout='ragged' debe tener forma {raw_ids.shape}; "
                        f"recibido {o_p.shape}"
                    )
                o_p = _scatter_ragged_to_panel(o_p, row_t, row_col, t_len, n_panel)
                h_p = _scatter_ragged_to_panel(h_p, row_t, row_col, t_len, n_panel)
                l_p = _scatter_ragged_to_panel(l_p, row_t, row_col, t_len, n_panel)
                c_p = _scatter_ragged_to_panel(c_p, row_t, row_col, t_len, n_panel)
                v_p = _scatter_ragged_to_panel(v_p, row_t, row_col, t_len, n_panel, fill_value=0.0)
            else:
                if o_p.shape != raw_ids.shape:
                    raise ValueError(
                        f"ohlcv en layout='dense' con ohlcv_layout='store' debe tener forma "
                        f"{raw_ids.shape}; recibido {o_p.shape}"
                    )
                o_p = _scatter_ragged_to_panel(o_p.reshape(-1), row_t, row_col, t_len, n_panel)
                h_p = _scatter_ragged_to_panel(h_p.reshape(-1), row_t, row_col, t_len, n_panel)
                l_p = _scatter_ragged_to_panel(l_p.reshape(-1), row_t, row_col, t_len, n_panel)
                c_p = _scatter_ragged_to_panel(c_p.reshape(-1), row_t, row_col, t_len, n_panel)
                v_p = _scatter_ragged_to_panel(
                    v_p.reshape(-1), row_t, row_col, t_len, n_panel, fill_value=0.0
                )
        elif ohlcv_layout == "canonical" and o_p.shape != (t_len, n_panel):
            raise ValueError(
                f"ohlcv_layout='canonical' exige forma {(t_len, n_panel)}; recibido {o_p.shape}"
            )
    elif all(k in store.keys() for k in ("open", "high", "low", "close", "volume")):
        ohlcv_source = "store_ohlcv_keys"
        o_p = _to_panel("open")
        h_p = _to_panel("high")
        l_p = _to_panel("low")
        c_p = _to_panel("close")
        v_p = _to_panel("volume", fill=0.0)
        assert (
            o_p is not None
            and h_p is not None
            and l_p is not None
            and c_p is not None
            and v_p is not None
        )
    else:
        prices_panel = _to_panel("prices")
        if prices_panel is None:
            raise ValueError("El store no contiene `prices` ni OHLCV explícito para enriquecer")
        o_p, h_p, l_p, c_p, v_p, ohlcv_source = _derive_causal_ohlcv_from_prices(
            prices_panel,
            ohlcv_price_cols=ohlcv_price_cols,
            prices_are_ohlcv=prices_are_ohlcv,
        )

    # 2) Modalidades auxiliares y exposiciones de pares `[T, N_panel, ...]`
    peer_parts: list[np.ndarray] = []
    for pk in ("exposures", "rmt_loadings"):
        pa = _to_panel(pk, fill=0.0)
        if pa is not None and pa.ndim == 3 and pa.shape[-1] > 0:
            peer_parts.append(pa.astype(np.float32, copy=False))
    peer_exp = np.concatenate(peer_parts, axis=-1) if peer_parts else None

    mod_mask_p = _to_panel("modality_mask", fill=0.0)
    if mod_mask_p is not None and mod_mask_p.shape[-1] == 5 and eff_mod_layout == "encoded_v31":
        mod_mask_p = mod_mask_p[..., list(des.ENCODED_TO_MODEL_MODALITY_INDEX)]
    mod_age_p = _to_panel("modality_age", fill=0.0)
    if mod_age_p is not None and mod_age_p.shape[-1] == 5 and eff_mod_layout == "encoded_v31":
        mod_age_p = mod_age_p[..., list(des.ENCODED_TO_MODEL_MODALITY_INDEX)]

    mod_kw: dict[str, Any] = dict(
        peer_exposures=peer_exp,
        fundamentals=_to_panel("fundamentals"),
        fundamentals_mask=(mod_mask_p[..., 1] > 0.5) if mod_mask_p is not None else None,
        fundamentals_age=mod_age_p[..., 1] if mod_age_p is not None else None,
        fundamental_layout=fundamental_layout,
        news=_to_panel("news"),
        news_mask=(mod_mask_p[..., 3] > 0.5) if mod_mask_p is not None else None,
        charts=_to_panel("charts"),
        charts_mask=(mod_mask_p[..., 4] > 0.5) if mod_mask_p is not None else None,
        macro=_to_panel("macro"),
        macro_mask=(mod_mask_p[..., 2] > 0.5) if mod_mask_p is not None else None,
        staleness_halflife_days=staleness_halflife_days,
    )

    # 3) Calcular las `ENRICHED_SOTA_DIM` señales ortogonales SOTA en FP32 maestro
    #    (+ subconjunto ortogonal KEEP `[T, N, 7]` e índices de régimen `[T, 18]` si opt-in)
    mims = _market_macro_indices_module() if include_market_macro_indices else None
    market_regime_panel: np.ndarray | None = None
    with torch.no_grad(), torch.amp.autocast(device_type=dev.type, enabled=False):
        o_t = torch.from_numpy(np.array(o_p, dtype=np.float32, copy=True)).to(device=dev)
        h_t = torch.from_numpy(np.array(h_p, dtype=np.float32, copy=True)).to(device=dev)
        l_t = torch.from_numpy(np.array(l_p, dtype=np.float32, copy=True)).to(device=dev)
        c_t = torch.from_numpy(np.array(c_p, dtype=np.float32, copy=True)).to(device=dev)
        v_t = torch.from_numpy(np.array(v_p, dtype=np.float32, copy=True)).to(device=dev)
        enriched_panel_t = des.enrich_full_dataset_sota_torch(
            o_t, h_t, l_t, c_t, v_t, **mod_kw
        )
        base_enriched_panel = (
            enriched_panel_t.detach().cpu().numpy().astype(np.float32, copy=False)
        )
        if include_market_macro_indices:
            assert mims is not None
            mims_out = mims.compute_market_indices_and_macro_sota_torch(
                o_t,
                h_t,
                l_t,
                c_t,
                v_t,
                macro=mod_kw["macro"],
                macro_mask=mod_kw["macro_mask"],
                macro_layout=macro_layout,
                staleness_halflife_days=staleness_halflife_days,
            )
            keep_asset_np = (
                mims.select_keep_asset_index_macro_features(
                    mims_out["asset_index_macro_features"]
                )
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32, copy=False)
            )
            market_regime_panel = (
                mims_out["market_regime_indices"]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32, copy=False)
            )
            enriched_panel = np.concatenate([base_enriched_panel, keep_asset_np], axis=-1)
        else:
            enriched_panel = base_enriched_panel

    enriched_names: tuple[str, ...] = (
        tuple(des.ENRICHED_SOTA_NAMES) + tuple(mims.KEEP_ASSET_INDEX_MACRO_NAMES)
        if include_market_macro_indices and mims is not None
        else tuple(des.ENRICHED_SOTA_NAMES)
    )

    # Devolver al layout exacto del store original (dense o ragged)
    if not needs_scatter:
        enriched_layout = enriched_panel
    elif store.layout == "dense":
        enriched_layout = enriched_panel[row_t, row_col].reshape(
            t_len, raw_ids.shape[1], enriched_panel.shape[-1]
        )
    else:
        enriched_layout = enriched_panel[row_t, row_col]

    # 4) Limpiar modalidades base y `physics_v4` base ANTES de concatenar `enriched_layout`
    #    (preserva 100% la amplitud de señales dispersas CN como `pit_fund_value`, P1-2!)
    out_arrays: dict[str, np.ndarray] = {
        k: np.asarray(store.raw_array(k)).copy() for k in store.keys()
    }
    if (
        "modality_mask" in out_arrays
        and out_arrays["modality_mask"].shape[-1] == 5
        and eff_mod_layout == "encoded_v31"
    ):
        out_arrays["modality_mask"] = np.ascontiguousarray(
            out_arrays["modality_mask"][..., list(des.ENCODED_TO_MODEL_MODALITY_INDEX)]
        )
    if (
        "modality_age" in out_arrays
        and out_arrays["modality_age"].shape[-1] == 5
        and eff_mod_layout == "encoded_v31"
    ):
        out_arrays["modality_age"] = np.ascontiguousarray(
            out_arrays["modality_age"][..., list(des.ENCODED_TO_MODEL_MODALITY_INDEX)]
        )

    out_np_dtype = np.dtype(output_dtype)
    if apply_missingness_featurizer:
        featurizer = des.MissingnessResilientFeaturizer(**dict(featurizer_kwargs or {})).eval()
        fresh_list: list[np.ndarray] = []
        with torch.no_grad(), torch.amp.autocast(device_type=dev.type, enabled=False):
            if store.layout == "dense":
                for s_i in range(t_len):
                    sess_t = {
                        k: torch.from_numpy(np.ascontiguousarray(v[s_i])).to(device=dev)
                        for k, v in out_arrays.items()
                    }
                    clean_s = featurizer.featurize_batch(
                        sess_t,
                        fundamental_layout=fundamental_layout,
                        macro_layout=macro_layout,
                    )
                    for k in out_arrays:
                        if k in clean_s and clean_s[k].dtype.is_floating_point:
                            out_arrays[k][s_i] = (
                                clean_s[k].detach().cpu().numpy().astype(out_arrays[k].dtype)
                            )
                    if "modality_freshness" in clean_s:
                        fresh_list.append(
                            clean_s["modality_freshness"].detach().cpu().numpy().astype(np.float32)
                        )
                if len(fresh_list) == t_len:
                    out_arrays["modality_freshness"] = np.ascontiguousarray(
                        np.stack(fresh_list, axis=0).astype(out_np_dtype, copy=False)
                    )
            else:
                assert store.session_offsets is not None
                offs = np.asarray(store.session_offsets, dtype=np.int64)
                for s_i in range(t_len):
                    r0, r1 = int(offs[s_i]), int(offs[s_i + 1])
                    sess_t = {
                        k: torch.from_numpy(np.ascontiguousarray(v[r0:r1])).to(device=dev)
                        for k, v in out_arrays.items()
                    }
                    clean_s = featurizer.featurize_batch(
                        sess_t,
                        fundamental_layout=fundamental_layout,
                        macro_layout=macro_layout,
                    )
                    for k in out_arrays:
                        if k in clean_s and clean_s[k].dtype.is_floating_point:
                            out_arrays[k][r0:r1] = (
                                clean_s[k].detach().cpu().numpy().astype(out_arrays[k].dtype)
                            )
                    if "modality_freshness" in clean_s:
                        fresh_list.append(
                            clean_s["modality_freshness"].detach().cpu().numpy().astype(np.float32)
                        )
                if len(fresh_list) == t_len:
                    out_arrays["modality_freshness"] = np.ascontiguousarray(
                        np.concatenate(fresh_list, axis=0).astype(out_np_dtype, copy=False)
                    )

    out_arrays[store_enriched_key] = np.ascontiguousarray(
        enriched_layout.astype(out_np_dtype, copy=False)
    )
    if append_to_physics_v4:
        if "physics_v4" in out_arrays:
            base_phys = out_arrays["physics_v4"].astype(np.float32, copy=False)
            out_arrays["physics_v4"] = np.ascontiguousarray(
                np.concatenate([base_phys, enriched_layout], axis=-1).astype(
                    out_np_dtype, copy=False
                )
            )
        else:
            out_arrays["physics_v4"] = out_arrays[store_enriched_key].copy()

    offsets_arg = (
        np.asarray(store.session_offsets, dtype=np.int64)
        if store.session_offsets is not None
        else None
    )
    session_arrays_arg: dict[str, np.ndarray] | None = None
    if market_regime_panel is not None:
        session_arrays_arg = {
            "market_regime_indices": np.ascontiguousarray(
                market_regime_panel.astype(out_np_dtype, copy=False)
            )
        }
    ZeroCopySessionStore.save(
        out_path,
        session_years=years,
        arrays=out_arrays,
        session_arrays=session_arrays_arg,
        session_offsets=offsets_arg,
        modality_mask_layout="model",
    )

    # Contrato SHA-256 determinista + metadatos en manifest.json (escritura atómica)
    h_ctx = hashlib.sha256()
    h_ctx.update(json.dumps(list(enriched_names)).encode("utf-8"))
    h_ctx.update(years.tobytes())
    h_ctx.update(out_arrays[store_enriched_key].tobytes())
    if session_arrays_arg is not None and mims is not None:
        h_ctx.update(json.dumps(list(mims.MARKET_REGIME_INDEX_NAMES)).encode("utf-8"))
        h_ctx.update(session_arrays_arg["market_regime_indices"].tobytes())
    sha256_contract = h_ctx.hexdigest()

    manifest_path = out_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["modality_mask_layout"] = "model"
    sota_meta: dict[str, Any] = dict(
        enriched_sota_names=list(enriched_names),
        n_enriched_features=len(enriched_names),
        store_enriched_key=store_enriched_key,
        append_to_physics_v4=bool(append_to_physics_v4),
        apply_missingness_featurizer=bool(apply_missingness_featurizer),
        include_market_macro_indices=bool(include_market_macro_indices),
        ohlcv_source=ohlcv_source,
        strictly_pre_2024=bool(max_year < sealed_test_start_year),
        sha256_contract=sha256_contract,
    )
    if include_market_macro_indices and mims is not None:
        sota_meta["keep_asset_index_macro_names"] = list(mims.KEEP_ASSET_INDEX_MACRO_NAMES)
        sota_meta["market_regime_index_names"] = list(mims.MARKET_REGIME_INDEX_NAMES)
        sota_meta["market_regime_index_dim"] = int(mims.MARKET_REGIME_INDEX_DIM)
    manifest["sota_enrichment"] = sota_meta
    tmp_m = out_path / "manifest.json.tmp"
    tmp_m.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    tmp_m.replace(manifest_path)

    report: dict[str, Any] = dict(
        output_dir=str(out_path),
        layout=store.layout,
        n_sessions=t_len,
        n_panel_assets=n_panel,
        n_enriched_features=len(enriched_names),
        enriched_sota_names=list(enriched_names),
        include_market_macro_indices=bool(include_market_macro_indices),
        ohlcv_source=ohlcv_source,
        apply_missingness_featurizer=bool(apply_missingness_featurizer),
        strictly_pre_2024=bool(max_year < sealed_test_start_year),
        sha256_contract=sha256_contract,
    )
    if include_market_macro_indices and mims is not None:
        report["keep_asset_index_macro_names"] = list(mims.KEEP_ASSET_INDEX_MACRO_NAMES)
        report["market_regime_index_names"] = list(mims.MARKET_REGIME_INDEX_NAMES)
        report["market_regime_index_dim"] = int(mims.MARKET_REGIME_INDEX_DIM)

    if run_audit:
        y_panel = _to_panel("target_returns")
        if y_panel is not None:
            y_1d = y_panel[:, :, 0] if y_panel.ndim == 3 else y_panel
        else:
            y_1d = np.zeros((t_len, n_panel), dtype=np.float32)
            if t_len > 1:
                c_ok = np.isfinite(c_p) & (c_p > 0.0)
                y_1d[:-1] = np.where(
                    c_ok[1:] & c_ok[:-1],
                    np.log(np.maximum(c_p[1:], 1e-12) / np.maximum(c_p[:-1], 1e-12)),
                    np.nan,
                )
        if include_market_macro_indices and mims is not None:
            def _combined_causality_fn() -> tuple[torch.Tensor, torch.Tensor, int]:
                t_cut = max(1, (2 * t_len) // 3)
                o0 = torch.from_numpy(np.array(o_p, dtype=np.float32, copy=True))
                h0 = torch.from_numpy(np.array(h_p, dtype=np.float32, copy=True))
                l0 = torch.from_numpy(np.array(l_p, dtype=np.float32, copy=True))
                c0 = torch.from_numpy(np.array(c_p, dtype=np.float32, copy=True))
                v0 = torch.from_numpy(np.array(v_p, dtype=np.float32, copy=True))
                o_m, h_m, l_m, c_m, v_m = (
                    o0.clone(),
                    h0.clone(),
                    l0.clone(),
                    c0.clone(),
                    v0.clone(),
                )
                c_m[t_cut + 1 :] *= 3.5
                h_m[t_cut + 1 :] *= 4.0
                l_m[t_cut + 1 :] *= 0.3
                v_m[t_cut + 1 :] *= 12.0
                mod_mut: dict[str, Any] = {}
                for mk, mv in mod_kw.items():
                    if isinstance(mv, np.ndarray):
                        mt = mv.copy()
                        if mt.ndim >= 1 and mt.shape[0] == t_len:
                            if mt.dtype == np.bool_:
                                mt[t_cut + 1 :] = ~mt[t_cut + 1 :]
                            elif "age" in mk.lower():
                                mt[t_cut + 1 :] = np.maximum(mt[t_cut + 1 :], 0.0) + 100.0
                            else:
                                mt[t_cut + 1 :] = -mt[t_cut + 1 :] * 2.5 - 3.0
                        mod_mut[mk] = mt
                    else:
                        mod_mut[mk] = mv
                with torch.no_grad(), torch.amp.autocast(device_type="cpu", enabled=False):
                    fa_b = des.enrich_full_dataset_sota_torch(o0, h0, l0, c0, v0, **mod_kw)
                    fa_m = mims.compute_market_indices_and_macro_sota_torch(
                        o0,
                        h0,
                        l0,
                        c0,
                        v0,
                        macro=mod_kw["macro"],
                        macro_mask=mod_kw["macro_mask"],
                        macro_layout=macro_layout,
                        staleness_halflife_days=staleness_halflife_days,
                    )
                    fa_k = mims.select_keep_asset_index_macro_features(
                        fa_m["asset_index_macro_features"]
                    )
                    fa_r = fa_m["market_regime_indices"].unsqueeze(1).expand(t_len, n_panel, -1)
                    fa = torch.cat([fa_b, fa_k, fa_r], dim=-1)

                    fb_b = des.enrich_full_dataset_sota_torch(o_m, h_m, l_m, c_m, v_m, **mod_mut)
                    fb_m = mims.compute_market_indices_and_macro_sota_torch(
                        o_m,
                        h_m,
                        l_m,
                        c_m,
                        v_m,
                        macro=mod_mut["macro"],
                        macro_mask=mod_mut["macro_mask"],
                        macro_layout=macro_layout,
                        staleness_halflife_days=staleness_halflife_days,
                    )
                    fb_k = mims.select_keep_asset_index_macro_features(
                        fb_m["asset_index_macro_features"]
                    )
                    fb_r = fb_m["market_regime_indices"].unsqueeze(1).expand(t_len, n_panel, -1)
                    fb = torch.cat([fb_b, fb_k, fb_r], dim=-1)
                return fa, fb, t_cut

            report["quality_audit"] = des.audit_dataset_enrichment_quality(
                enriched_panel,
                y_1d,
                feature_names=enriched_names,
                causality_fn=_combined_causality_fn if t_len >= 4 else None,
                run_stress=True,
            )
            report["market_macro_audit"] = mims.audit_market_indices_macro_value(
                o_p,
                h_p,
                l_p,
                c_p,
                v_p,
                y_1d,
                macro=mod_kw["macro"],
                macro_mask=mod_kw["macro_mask"],
                macro_layout=macro_layout,
                existing_enriched=base_enriched_panel,
                market_label="STORE",
            )
        else:
            report["quality_audit"] = des.audit_dataset_enrichment_quality(
                enriched_panel,
                y_1d,
                feature_names=enriched_names,
                ohlcv=(o_p, h_p, l_p, c_p, v_p),
                modalities=mod_kw,
                run_stress=True,
            )

    loaded = ZeroCopySessionStore.load(
        out_path,
        mmap_mode="r",
        modality_mask_layout="model",
    )
    return loaded, report


def __getattr__(name: str) -> Any:
    if name in (
        "ENCODED_TO_MODEL_MODALITY_INDEX",
        "ENRICHED_SOTA_DIM",
        "ENRICHED_SOTA_NAMES",
        "MissingnessResilientFeaturizer",
        "audit_dataset_enrichment_quality",
        "enrich_full_dataset_sota_torch",
    ):
        return getattr(_sota_enrichment_module(), name)
    if name in (
        "ASSET_INDEX_MACRO_DIM",
        "ASSET_INDEX_MACRO_NAMES",
        "FACTOR_PORTFOLIO_DIM",
        "FACTOR_PORTFOLIO_NAMES",
        "KEEP_ASSET_INDEX_MACRO_DIM",
        "KEEP_ASSET_INDEX_MACRO_NAMES",
        "MARKET_REGIME_INDEX_DIM",
        "MARKET_REGIME_INDEX_NAMES",
        "audit_market_indices_macro_value",
        "compute_market_indices_and_macro_sota_torch",
        "select_keep_asset_index_macro_features",
    ):
        return getattr(_market_macro_indices_module(), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
