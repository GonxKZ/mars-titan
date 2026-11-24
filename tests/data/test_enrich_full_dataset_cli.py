"""Pruebas TDD para el CLI oficial `scripts/enrich_full_dataset_sota.py` (Wave-3A)."""

from __future__ import annotations

import datetime
import gzip
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mars_titan.data.dataset_enrichment_sota import ENRICHED_SOTA_DIM, ENRICHED_SOTA_NAMES

_CLI_PATH = Path(__file__).resolve().parents[2] / "scripts" / "enrich_full_dataset_sota.py"


def _load_cli_module() -> Any:
    if not _CLI_PATH.is_file():
        raise FileNotFoundError(f"Falta el CLI oficial: {_CLI_PATH}")
    spec = importlib.util.spec_from_file_location("enrich_full_dataset_sota_cli", _CLI_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_synthetic_materialization_tree(
    root: Path,
    *,
    market: str = "US",
    tickers: tuple[str, ...] = ("AAA", "BBB", "CCC", "DDD"),
    n_pre_sessions: int = 90,
    n_post2024_sessions: int = 5,
    fund_missing_prob: float = 0.70,
    seed: int = 42,
) -> tuple[list[str], list[int]]:
    """Crea una réplica mínima de `prepared-v3.1`, `encoded-history-v3.1` y `targets-v3.1`."""
    rng = np.random.default_rng(seed)
    start_date = datetime.date(2023, 6, 1)
    pre_dates: list[datetime.date] = []
    cur = start_date
    while len(pre_dates) < n_pre_sessions:
        if cur.weekday() < 5 and cur.year < 2024:
            pre_dates.append(cur)
        cur += datetime.timedelta(days=1)

    post_dates = [
        datetime.date(2024, 1, 2) + datetime.timedelta(days=i) for i in range(n_post2024_sessions)
    ]
    all_dates = pre_dates + post_dates
    all_str = [d.isoformat() for d in all_dates]
    epoch0 = datetime.date(1970, 1, 1)
    pre_epochs = [(d - epoch0).days for d in pre_dates]

    t_tot = len(all_dates)
    macro_shared = rng.normal(0.0, 1.0, size=(t_tot, 420)).astype(np.float32)
    macro_shared[:, 140:280] = 1.0
    macro_shared[:, 280:420] = 0.5

    for idx_t, tkr in enumerate(tickers):
        rets = 0.0004 + 0.016 * rng.standard_normal(t_tot)
        close = (40.0 + 15.0 * idx_t) * np.exp(np.cumsum(rets))
        open_ = close * np.exp(0.003 * rng.standard_normal(t_tot))
        high = np.maximum(open_, close) * np.exp(np.abs(0.006 * rng.standard_normal(t_tot)))
        low = np.minimum(open_, close) * np.exp(-np.abs(0.006 * rng.standard_normal(t_tot)))
        volume = np.exp(13.0 + 0.5 * rng.standard_normal(t_tot))
        # Inyectar un halt (`volume == 0`)
        volume[12] = 0.0

        pdir = root / "prepared-v3.1" / "prepared" / market / tkr
        pdir.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.table(
                {
                    "open": pa.array(open_, type=pa.float64()),
                    "high": pa.array(high, type=pa.float64()),
                    "low": pa.array(low, type=pa.float64()),
                    "close": pa.array(close, type=pa.float64()),
                    "volume": pa.array(volume, type=pa.float64()),
                    "session": pa.array(all_str, type=pa.string()),
                }
            ),
            pdir / "prices.parquet",
        )

        # `encoded-history-v3.1`: orden `(prices, news, charts, fundamentals, macro)`
        fund_mat = np.zeros((t_tot, 78), dtype=np.float32)
        f_has = rng.random(t_tot) >= fund_missing_prob
        if f_has.any():
            fund_mat[f_has, :26] = rng.normal(23.5, 1.0, size=(int(f_has.sum()), 26)).astype(
                np.float32
            )
            fund_mat[f_has, 26:52] = 1.0
            fund_mat[f_has, 52:78] = rng.uniform(0.1, 4.0, size=(int(f_has.sum()), 26)).astype(
                np.float32
            )
        news_has = rng.random(t_tot) > 0.75
        news_mat = np.zeros((t_tot, 384), dtype=np.float32)
        if news_has.any():
            news_mat[news_has] = rng.standard_normal((int(news_has.sum()), 384)).astype(np.float32)
        chart_mat = rng.standard_normal((t_tot, 512)).astype(np.float32)
        pres = np.stack(
            [
                np.ones(t_tot, dtype=bool),
                news_has,
                np.ones(t_tot, dtype=bool),
                f_has,
                np.ones(t_tot, dtype=bool),
            ],
            axis=-1,
        )

        edir = root / "encoded-history-v3.1" / "samples" / market / tkr
        edir.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.table(
                {
                    "session": pa.array(all_str, type=pa.string()),
                    "presence": pa.FixedSizeListArray.from_arrays(
                        pa.array(pres.reshape(-1), type=pa.bool_()), 5
                    ),
                    "fundamentals": pa.FixedSizeListArray.from_arrays(
                        pa.array(fund_mat.reshape(-1), type=pa.float32()), 78
                    ),
                    "macro": pa.FixedSizeListArray.from_arrays(
                        pa.array(macro_shared.reshape(-1), type=pa.float32()), 420
                    ),
                    "news": pa.FixedSizeListArray.from_arrays(
                        pa.array(news_mat.reshape(-1), type=pa.float32()), 384
                    ),
                    "charts": pa.FixedSizeListArray.from_arrays(
                        pa.array(chart_mat.reshape(-1), type=pa.float32()), 512
                    ),
                }
            ),
            edir / "samples.parquet",
        )

        # `targets-v3.1`: retorno futuro 1d con `target_available_at` real (última fila None)
        fwd = np.empty(t_tot, dtype=np.float64)
        fwd[:-1] = np.log(close[1:] / close[:-1])
        fwd[-1] = np.nan
        pred_ts = [
            datetime.datetime(d.year, d.month, d.day, 21, 5, tzinfo=datetime.UTC) for d in all_dates
        ]
        avail_ts: list[datetime.datetime | None] = [
            datetime.datetime(d.year, d.month, d.day, 21, 5, tzinfo=datetime.UTC)
            for d in all_dates[1:]
        ] + [None]
        tdir = root / "targets-v3.1" / "labels" / market / tkr
        tdir.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.table(
                {
                    "sample_row": pa.array(np.arange(t_tot, dtype=np.int64)),
                    "prediction_at": pa.array(pred_ts, type=pa.timestamp("us", tz="UTC")),
                    "target_available_at": pa.array(avail_ts, type=pa.timestamp("us", tz="UTC")),
                    "target": pa.array(
                        [None if np.isnan(x) else float(x) for x in fwd], type=pa.float64()
                    ),
                }
            ),
            tdir / "labels.parquet",
        )

    return [d.isoformat() for d in pre_dates], pre_epochs


def _make_synthetic_lab_cache(
    cache_dir: Path,
    *,
    market: str = "US",
    tickers: tuple[str, ...] = ("AAA", "BBB", "CCC", "DDD"),
    epoch_days: list[int],
    include_x: bool = True,
    seed: int = 7,
) -> None:
    rng = np.random.default_rng(seed)
    cache_dir.mkdir(parents=True, exist_ok=True)
    n_sess = len(epoch_days)
    n_assets = len(tickers)
    n_rows = n_sess * n_assets

    bounds = np.arange(0, n_rows + 1, n_assets, dtype=np.int64)
    days = np.repeat(np.asarray(epoch_days, dtype=np.int32), n_assets)
    assets = np.tile(np.arange(n_assets, dtype=np.int32), n_sess)
    sp = np.zeros(n_sess, dtype=np.int8)
    sp[-max(1, n_sess // 5) :] = 3
    y = (0.015 * rng.standard_normal(n_rows)).astype(np.float64)

    np.save(cache_dir / "bounds.npy", bounds)
    np.save(cache_dir / "day.npy", days)
    np.save(cache_dir / "asset.npy", assets)
    np.save(cache_dir / "session_part.npy", sp)
    np.save(cache_dir / "y.npy", y)

    base_names = [
        "alpha_percent",
        "alpha_t",
        "beta",
        "log_residual_volatility",
        "has_market_model",
        "return_5",
        "return_21",
        "return_63",
        "log_dollar_volume_21",
        "return_1",
        "intraday_1",
        "range_1",
    ]
    if include_x:
        x = rng.normal(0.0, 0.02, size=(n_rows, len(base_names))).astype(np.float32)
        x[:, 8] = rng.normal(14.0, 1.0, size=n_rows).astype(np.float32)
        x[:, 11] = np.abs(rng.normal(0.02, 0.005, size=n_rows)).astype(np.float32)
        np.save(cache_dir / "X.npy", x)

    names_doc = dict(
        market=market,
        version="synthetic_test_v1",
        assets=[f"{market}/{t}" for t in tickers],
        base=base_names,
        names=base_names,
        rows=n_rows,
        sessions=n_sess,
    )
    (cache_dir / "names.json").write_text(json.dumps(names_doc, indent=2), encoding="utf-8")


def test_cli_fm8_market_isolation_2024_seal_and_readonly_source_guards(tmp_path: Path):
    """FM-8: rechaza cruce US/CN, datos >= 2024-01-01 y escritura dentro de origen solo lectura."""
    cli = _load_cli_module()
    pre_dates, pre_epochs = _make_synthetic_materialization_tree(
        tmp_path / "mat", market="CN", tickers=("000001.SZ", "000002.SZ", "000008.SZ")
    )
    assert all(d < "2024-01-01" for d in pre_dates)

    # 1) Cache CN pasada con --market US -> ValueError FM-8
    cn_cache = tmp_path / "cn_cache"
    _make_synthetic_lab_cache(
        cn_cache,
        market="CN",
        tickers=("000001.SZ", "000002.SZ", "000008.SZ"),
        epoch_days=pre_epochs,
    )
    with pytest.raises(ValueError, match="FM-8"):
        cli.run_enrichment_pipeline(
            mode="lab_cache",
            market="US",
            lab_cache_dir=cn_cache,
            materialization_root=tmp_path / "mat",
            output_dir=tmp_path / "out_bad_mkt",
            device="cpu",
            dry_run=True,
        )

    # 2) Cache con fecha 2024-01-01 (epoch 19723) -> ValueError sello 2024+
    bad24_cache = tmp_path / "bad24_cache"
    _make_synthetic_lab_cache(
        bad24_cache,
        market="CN",
        tickers=("000001.SZ", "000002.SZ", "000008.SZ"),
        epoch_days=pre_epochs + [19723],
    )
    with pytest.raises(ValueError, match="2024"):
        cli.run_enrichment_pipeline(
            mode="lab_cache",
            market="CN",
            lab_cache_dir=bad24_cache,
            materialization_root=tmp_path / "mat",
            output_dir=tmp_path / "out_bad24",
            device="cpu",
            dry_run=True,
        )

    # 3) `--output-dir` igual o dentro del directorio fuente -> ValueError
    with pytest.raises(ValueError, match="solo lectura"):
        cli.run_enrichment_pipeline(
            mode="lab_cache",
            market="CN",
            lab_cache_dir=cn_cache,
            materialization_root=tmp_path / "mat",
            output_dir=cn_cache / "sub_forbidden",
            device="cpu",
            audit_only=True,
        )


def test_cli_lab_cache_mode_with_and_without_x_npy_dry_run_audit_only_and_materialize(
    tmp_path: Path,
):
    """`--mode lab_cache` soporta cachés con `X.npy` y esqueletos tipo `x4-us-2015` sin `X.npy`."""
    cli = _load_cli_module()
    tickers = tuple(f"U{i:02d}" for i in range(20))
    _, pre_epochs = _make_synthetic_materialization_tree(
        tmp_path / "mat", market="US", tickers=tickers, n_pre_sessions=80, n_post2024_sessions=3
    )

    # A) Caché completa con X.npy: --dry-run (0 files), --audit-only (.json.gz), full (.npz)
    full_cache = tmp_path / "x3_us_syn"
    _make_synthetic_lab_cache(
        full_cache, market="US", tickers=tickers, epoch_days=pre_epochs, include_x=True
    )

    out_dry = tmp_path / "out_dry"
    rep_dry = cli.run_enrichment_pipeline(
        mode="lab_cache",
        market="US",
        lab_cache_dir=full_cache,
        materialization_root=tmp_path / "mat",
        output_dir=out_dry,
        device="cpu",
        max_assets=20,
        dry_run=True,
    )
    assert not out_dry.exists()
    assert rep_dry["strictly_pre_2024"] is True
    assert rep_dry["quality_audit"]["causality_check"]["bit_exact_causal"] is True
    assert rep_dry["quality_audit"]["missingness_stress"]["all_passed"] is True
    assert rep_dry["n_enriched_features"] == ENRICHED_SOTA_DIM == len(ENRICHED_SOTA_NAMES) == 24

    out_aud = tmp_path / "out_audit_only"
    rep_aud = cli.run_enrichment_pipeline(
        mode="lab_cache",
        market="US",
        lab_cache_dir=full_cache,
        materialization_root=tmp_path / "mat",
        output_dir=out_aud,
        device="cpu",
        max_assets=20,
        audit_only=True,
    )
    report_path = Path(rep_aud["report_path"])
    assert report_path.is_file() and report_path.name.endswith(".json.gz")
    assert rep_aud["sidecar_path"] is None
    loaded_rep = json.loads(gzip.decompress(report_path.read_bytes()).decode("utf-8"))
    assert loaded_rep["sha256_contract"] == rep_aud["sha256_contract"]

    out_mat = tmp_path / "out_materialize_fp16"
    rc = cli.main(
        [
            "--mode",
            "lab_cache",
            "--market",
            "US",
            "--lab-cache",
            str(full_cache),
            "--materialization-root",
            str(tmp_path / "mat"),
            "--output-dir",
            str(out_mat),
            "--device",
            "cpu",
            "--max-assets",
            "20",
            "--dtype",
            "fp16",
        ]
    )
    assert rc == 0
    npz_files = list(out_mat.glob("*.npz"))
    gz_files = list(out_mat.glob("*.json.gz"))
    assert len(npz_files) == 1 and len(gz_files) == 1
    with np.load(npz_files[0]) as data:
        assert data["enriched_sota"].dtype == np.float16
        assert data["enriched_sota"].shape[-1] == len(ENRICHED_SOTA_NAMES)
        assert np.isfinite(data["enriched_sota"]).all()

    # B) Caché esqueleto SIN `X.npy` (como `x4-us-2015`): enriquece vía `materialization-root`
    skel_cache = tmp_path / "x4_us_2015_syn"
    _make_synthetic_lab_cache(
        skel_cache, market="US", tickers=tickers, epoch_days=pre_epochs, include_x=False
    )
    rep_skel = cli.run_enrichment_pipeline(
        mode="lab_cache",
        market="US",
        lab_cache_dir=skel_cache,
        materialization_root=tmp_path / "mat",
        output_dir=tmp_path / "out_skel",
        device="cpu",
        max_assets=20,
        audit_only=True,
    )
    assert rep_skel["lab_cache_audit"]["has_x_npy"] is False
    assert rep_skel["quality_audit"]["causality_check"]["bit_exact_causal"] is True
    assert rep_skel["quality_audit"]["missingness_stress"]["all_passed"] is True


def test_cli_full_history_streaming_chunk_parity_2024_seal_and_cn_99pct_missing(tmp_path: Path):
    """`--mode full_history` sella >=2024, tolera CN 99%/100% ausente y empata exacto en chunks."""
    cli = _load_cli_module()
    tickers = tuple(f"0000{i:02d}.SZ" for i in range(1, 37))
    # CN 2006–2013 tiene 100% fundamentales ausentes y 2014–2023 ~99.2% ausentes:
    # 36 activos × 150 sesiones = 5400 celdas; inyectamos 50 obs (0.93% obs -> 99.07% ausente!).
    _make_synthetic_materialization_tree(
        tmp_path / "mat_cn",
        market="CN",
        tickers=tickers,
        n_pre_sessions=150,
        n_post2024_sessions=6,
        fund_missing_prob=1.0,
        seed=101,
    )
    rng_cn = np.random.default_rng(10101)
    # 25 sesiones × 2 activos rotando -> 50/5400 = 0.93% obs (99.07% ausente)
    active_sessions = list(range(5, 145, 5))[:25]
    per_ticker_rows: dict[str, list[int]] = {t: [] for t in tickers}
    for k_idx, s_row in enumerate(active_sessions):
        per_ticker_rows[tickers[(2 * k_idx) % len(tickers)]].append(s_row)
        per_ticker_rows[tickers[(2 * k_idx + 1) % len(tickers)]].append(s_row)
    for tkr, rows in per_ticker_rows.items():
        if not rows:
            continue
        sf = tmp_path / "mat_cn" / f"encoded-history-v3.1/samples/CN/{tkr}/samples.parquet"
        tbl = pq.read_table(sf)
        f_np = cli._extract_fixed_list_2d(tbl["fundamentals"], 78).copy()
        p_np = cli._extract_fixed_bool_2d(tbl["presence"], 5).copy()
        for s_row in rows:
            f_np[s_row, :26] = rng_cn.normal(23.8, 1.0, size=26).astype(np.float32)
            f_np[s_row, 26:52] = 1.0
            f_np[s_row, 52:78] = 4.5
            p_np[s_row, 3] = True
        tbl_m = tbl.set_column(
            tbl.schema.get_field_index("fundamentals"),
            "fundamentals",
            pa.FixedSizeListArray.from_arrays(pa.array(f_np.reshape(-1), type=pa.float32()), 78),
        ).set_column(
            tbl.schema.get_field_index("presence"),
            "presence",
            pa.FixedSizeListArray.from_arrays(pa.array(p_np.reshape(-1), type=pa.bool_()), 5),
        )
        pq.write_table(tbl_m, sf)

    out_chunk2 = tmp_path / "out_cn_chunk2"
    rep_c2 = cli.run_enrichment_pipeline(
        mode="full_history",
        market="CN",
        materialization_root=tmp_path / "mat_cn",
        output_dir=out_chunk2,
        device="cpu",
        max_assets=0,
        asset_chunk_size=10,
        dtype="fp32",
    )
    out_chunk_all = tmp_path / "out_cn_chunk_all"
    rep_all = cli.run_enrichment_pipeline(
        mode="full_history",
        market="CN",
        materialization_root=tmp_path / "mat_cn",
        output_dir=out_chunk_all,
        device="cpu",
        max_assets=0,
        asset_chunk_size=64,
        dtype="fp32",
    )

    # Sello 2024+ estricto: ninguna fecha >= 2024-01-01 y filas 2024 descartadas contabilizadas
    assert rep_c2["strictly_pre_2024"] is True
    assert rep_c2["max_session_date"] < "2024-01-01"
    assert rep_c2["n_sealed_2024_rows_dropped"] == len(tickers) * 6
    assert rep_c2["n_sealed_2024_targets_dropped"] == len(tickers)
    assert rep_c2["quality_audit"]["causality_check"]["bit_exact_causal"] is True
    assert rep_c2["quality_audit"]["missingness_stress"]["all_passed"] is True
    assert rep_c2["sha256_contract"] == rep_all["sha256_contract"]

    # Paridad bit a bit exacta entre streaming `asset_chunk_size=10` vs `asset_chunk_size=64`!
    with np.load(rep_c2["sidecar_path"]) as d2, np.load(rep_all["sidecar_path"]) as d_all:
        np.testing.assert_array_equal(d2["enriched_sota"], d_all["enriched_sota"])
        assert np.isfinite(d2["enriched_sota"]).all()
        assert d2["enriched_sota"].shape == (150, len(tickers), len(ENRICHED_SOTA_NAMES))


def test_cli_adversarial_199_sealed_2024_target_mask_parity_fp32_slabs_and_fail_closed(
    tmp_path: Path,
):
    """Regresión `#199`: sello `target_available_at>=2024`, paridad NaN/FP32-slab y fail-closed."""
    import torch

    from mars_titan.data.dataset_enrichment_sota import (
        ENCODED_TO_MODEL_MODALITY_INDEX,
        enrich_full_dataset_sota_torch,
    )

    cli = _load_cli_module()
    tickers = tuple(f"A{i:02d}" for i in range(20))
    pre_dates, pre_epochs = _make_synthetic_materialization_tree(
        tmp_path / "mat_199",
        market="US",
        tickers=tickers,
        n_pre_sessions=80,
        n_post2024_sessions=4,
        fund_missing_prob=0.50,
        seed=199,
    )
    # Inyectar un embedding news todo-NaN con presence=True en ticker A00 para probar paridad P1-3
    aaa_s = tmp_path / "mat_199" / "encoded-history-v3.1/samples/US/A00/samples.parquet"
    tbl = pq.read_table(aaa_s)
    news_np = cli._extract_fixed_list_2d(tbl["news"], 384).copy()
    pres_np = cli._extract_fixed_bool_2d(tbl["presence"], 5).copy()
    news_np[20] = np.nan
    pres_np[20, 1] = True  # encoded index 1 == news
    tbl_mod = tbl.set_column(
        tbl.schema.get_field_index("news"),
        "news",
        pa.FixedSizeListArray.from_arrays(pa.array(news_np.reshape(-1), type=pa.float32()), 384),
    ).set_column(
        tbl.schema.get_field_index("presence"),
        "presence",
        pa.FixedSizeListArray.from_arrays(pa.array(pres_np.reshape(-1), type=pa.bool_()), 5),
    )
    pq.write_table(tbl_mod, aaa_s)

    # 1) P0-1: Último retorno objetivo pre-2024 (`target_available_at` en 2024-01-02) DEBE ser NaN
    panel = cli._ingest_tickers_streaming(
        market="US",
        tickers=tickers,
        session_dates=pre_dates,
        materialization_root=tmp_path / "mat_199",
        asset_chunk_size=6,
        device=torch.device("cpu"),
        include_targets=True,
    )
    assert np.isnan(panel["target"][-1]).all(), (
        "Fuga P0-1: retorno 2024-01-02 se filtró en target[-1]"
    )
    assert panel["n_sealed_2024_targets_dropped"] == len(tickers)
    # P1-3a: fila 20 todo-NaN con presence=True -> news_mask_p[20, 0] debe ser False!
    assert bool(panel["news_mask"][20, 0]) is False

    # 2) P1-3b: Paridad FP32 exacta entre `fund_slab_mb=0.05` (memmap+slabs) y 3D monolítico!
    out_slab = tmp_path / "out_tiny_slab"
    rep_slab = cli.run_enrichment_pipeline(
        mode="full_history",
        market="US",
        materialization_root=tmp_path / "mat_199",
        output_dir=out_slab,
        device="cpu",
        asset_chunk_size=6,
        fund_slab_mb=0.05,
        dtype="fp32",
    )
    assert rep_slab["n_sealed_2024_targets_dropped"] == len(tickers)
    assert rep_slab["fund_uses_memmap"] is True

    # Comparar contra `enrich_full_dataset_sota_torch` monolítico 3D (news 384, charts 512)
    t_len, n_a = len(pre_dates), len(tickers)
    news_3d = np.zeros((t_len, n_a, 384), dtype=np.float32)
    charts_3d = np.zeros((t_len, n_a, 512), dtype=np.float32)
    fund_3d = np.zeros((t_len, n_a, 78), dtype=np.float32)
    pres_m_3d = np.zeros((t_len, n_a, 5), dtype=bool)
    for j, tkr in enumerate(tickers):
        et = pq.read_table(
            tmp_path / "mat_199" / f"encoded-history-v3.1/samples/US/{tkr}/samples.parquet"
        )
        news_3d[:, j] = cli._extract_fixed_list_2d(et["news"], 384)[:t_len]
        charts_3d[:, j] = cli._extract_fixed_list_2d(et["charts"], 512)[:t_len]
        fund_3d[:, j] = cli._extract_fixed_list_2d(et["fundamentals"], 78)[:t_len]
        pe = cli._extract_fixed_bool_2d(et["presence"], 5)[:t_len]
        pres_m_3d[:, j] = pe[:, list(ENCODED_TO_MODEL_MODALITY_INDEX)]

    ref_3d = enrich_full_dataset_sota_torch(
        panel["open"],
        panel["high"],
        panel["low"],
        panel["close"],
        panel["volume"],
        fundamentals=fund_3d,
        fundamentals_mask=pres_m_3d[:, :, 1],
        fundamental_layout="encoded_v31_78",
        news=news_3d,
        news_mask=pres_m_3d[:, :, 3],
        charts=charts_3d,
        charts_mask=pres_m_3d[:, :, 4],
        macro=panel["macro"],
        macro_mask=pres_m_3d[:, :, 2],
    ).numpy()
    with np.load(rep_slab["sidecar_path"]) as ds:
        np.testing.assert_allclose(ds["enriched_sota"], ref_3d, atol=1e-6, rtol=1e-6)

    # 3) P1-4: Fail-closed gate (inyectar fuga sintética |IC| > 0.10 -> sin .npz y exit code != 0)
    leak_cache = tmp_path / "leak_cache"
    _make_synthetic_lab_cache(
        leak_cache, market="US", tickers=tickers, epoch_days=pre_epochs, include_x=True
    )
    y_leak = ref_3d[:, :, 0].reshape(-1).astype(np.float64)
    np.save(leak_cache / "y.npy", y_leak)

    out_fail = tmp_path / "out_fail_closed"
    with pytest.raises(RuntimeError, match="quality gate"):
        cli.run_enrichment_pipeline(
            mode="lab_cache",
            market="US",
            lab_cache_dir=leak_cache,
            materialization_root=tmp_path / "mat_199",
            output_dir=out_fail,
            device="cpu",
            max_assets=20,
            dtype="fp32",
        )
    assert list(out_fail.glob("*.npz")) == [], "Fail-closed: jamás escribir .npz si falla auditoría"
    assert len(list(out_fail.glob("*.json.gz"))) == 1
    rc_fail = cli.main(
        [
            "--mode",
            "lab_cache",
            "--market",
            "US",
            "--lab-cache",
            str(leak_cache),
            "--materialization-root",
            str(tmp_path / "mat_199"),
            "--output-dir",
            str(tmp_path / "out_fail_cli"),
            "--device",
            "cpu",
            "--max-assets",
            "20",
        ]
    )
    assert rc_fail == 1
    assert list((tmp_path / "out_fail_cli").glob("*.npz")) == []


def test_cli_include_market_macro_indices_opt_in_backwards_compat_and_fail_closed(
    tmp_path: Path,
):
    """#231: `--include-market-macro-indices` añade 7 KEEP + `[T, 18]` regime y mantiene compat."""
    from mars_titan.data.market_indices_macro_sota import (
        FACTOR_PORTFOLIO_DIM,
        KEEP_ASSET_INDEX_MACRO_DIM,
        KEEP_ASSET_INDEX_MACRO_NAMES,
        MARKET_REGIME_INDEX_DIM,
        MARKET_REGIME_INDEX_NAMES,
    )

    cli = _load_cli_module()
    tickers = tuple(f"M{i:02d}" for i in range(18))
    _make_synthetic_materialization_tree(
        tmp_path / "mat_231",
        market="US",
        tickers=tickers,
        n_pre_sessions=90,
        n_post2024_sessions=4,
        fund_missing_prob=0.50,
        seed=231,
    )

    # 1) Default OFF (`include_market_macro_indices=False`): 24 dims, sin `market_regime_indices`
    out_off = tmp_path / "out_off"
    rep_off = cli.run_enrichment_pipeline(
        mode="full_history",
        market="US",
        materialization_root=tmp_path / "mat_231",
        output_dir=out_off,
        device="cpu",
        asset_chunk_size=6,
    )
    assert rep_off["n_enriched_features"] == ENRICHED_SOTA_DIM == 24
    with np.load(rep_off["sidecar_path"]) as d_off:
        assert d_off["enriched_sota"].shape == (90, len(tickers), 24)
        assert "market_regime_indices" not in d_off.files

    # 2) Opt-in ON vía CLI `--include-market-macro-indices`: 31 dims + `[T, 18]` regime + `[T, 5]`
    out_on = tmp_path / "out_on"
    rc = cli.main(
        [
            "--mode",
            "full_history",
            "--market",
            "US",
            "--materialization-root",
            str(tmp_path / "mat_231"),
            "--output-dir",
            str(out_on),
            "--device",
            "cpu",
            "--asset-chunk-size",
            "6",
            "--include-market-macro-indices",
        ]
    )
    assert rc == 0
    npz_files = list(out_on.glob("*.npz"))
    gz_files = list(out_on.glob("*.json.gz"))
    assert len(npz_files) == 1 and len(gz_files) == 1
    rep_on = json.loads(gzip.decompress(gz_files[0].read_bytes()).decode("utf-8"))
    assert rep_on["include_market_macro_indices"] is True
    assert rep_on["strictly_pre_2024"] is True
    assert rep_on["n_sealed_2024_rows_dropped"] == len(tickers) * 4
    assert rep_on["n_sealed_2024_targets_dropped"] == len(tickers)
    assert rep_on["n_enriched_features"] == ENRICHED_SOTA_DIM + KEEP_ASSET_INDEX_MACRO_DIM == 31
    assert (
        tuple(rep_on["enriched_sota_names"]) == ENRICHED_SOTA_NAMES + KEEP_ASSET_INDEX_MACRO_NAMES
    )
    assert "stock_rel_to_vw_5" not in rep_on["enriched_sota_names"]
    assert rep_on["quality_audit"]["causality_check"]["bit_exact_causal"] is True
    assert rep_on["quality_audit"]["causality_check"]["max_abs_diff"] == 0.0
    assert rep_on["market_macro_audit"]["overall_passed"] is True
    assert rep_on["market_macro_audit"]["bf16_autocast_check"]["max_abs_diff"] == 0.0

    with np.load(npz_files[0]) as d_on, np.load(rep_off["sidecar_path"]) as d_off:
        assert d_on["enriched_sota"].shape == (90, len(tickers), 31)
        # Primeras 24 cols son bit-a-bit idénticas al modo OFF!
        np.testing.assert_array_equal(d_on["enriched_sota"][..., :24], d_off["enriched_sota"])
        assert np.isfinite(d_on["enriched_sota"]).all()
        assert d_on["market_regime_indices"].shape == (90, MARKET_REGIME_INDEX_DIM)
        assert np.isfinite(d_on["market_regime_indices"]).all()
        assert tuple(d_on["market_regime_index_names"].tolist()) == MARKET_REGIME_INDEX_NAMES
        assert d_on["factor_returns"].shape == (90, FACTOR_PORTFOLIO_DIM)
        assert np.isfinite(d_on["factor_returns"]).all()

    # 3) Paridad bit-a-bit entre streaming chunk=6 vs chunk=64 con `--include-market-macro-indices`
    out_on_c64 = tmp_path / "out_on_c64"
    rep_c64 = cli.run_enrichment_pipeline(
        mode="full_history",
        market="US",
        materialization_root=tmp_path / "mat_231",
        output_dir=out_on_c64,
        device="cpu",
        asset_chunk_size=64,
        include_market_macro_indices=True,
    )
    assert rep_c64["sha256_contract"] == rep_on["sha256_contract"]
    with np.load(npz_files[0]) as d6, np.load(rep_c64["sidecar_path"]) as d64:
        np.testing.assert_array_equal(d6["enriched_sota"], d64["enriched_sota"])
        np.testing.assert_array_equal(d6["market_regime_indices"], d64["market_regime_indices"])


