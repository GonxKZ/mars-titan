"""Pruebas TDD para la sonda de concurrencia (1, 2, 4 y 6 trabajos) en Colab RTX PRO 6000."""

from __future__ import annotations

import json
from pathlib import Path

import concurrency_probe as cp
import pytest


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _site_sheet(*, ram_gib: float = 96.0, vram_mib: int = 97887, allow_tf32: bool = False) -> dict:
    return {
        "schema_version": 1,
        "kind": "machine_site_v1",
        "site": "google-colab-rtx-pro-6000",
        "platform": {"system": "Linux", "machine": "x86_64", "release": "6.8.0"},
        "cpu": {"model": "AMD EPYC Turin", "logical": 24},
        "memory_kib": int(ram_gib * 1024 * 1024),
        "python": "3.12.3",
        "packages": {"torch": "2.14.0+cu130", "triton": "3.8.0", "numpy": "2.5.3"},
        "gpu": {
            "name": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
            "memory_mib": vram_mib,
            "driver": "590.48.01",
            "capability": [12, 0],
        },
        "torch": {
            "cuda_runtime": "13.0",
            "cudnn": 92400,
            "cuda_available": True,
            "matmul_precision": "highest",
            "allow_tf32": allow_tf32,
        },
        "disk": {"total_bytes": 236 * 1024**3, "free_bytes": 120 * 1024**3},
    }


def _make_level_fixture(
    root: Path,
    *,
    workers: int,
    per_job_steps: int,
    per_job_wall_s: float,
    per_job_rss_gib: float,
    per_job_vram_reserved_gib: float,
    per_job_vram_alloc_gib: float = 4.2,
    oom_jobs: int = 0,
    host_ram_sample_used_gib: float | None = None,
    gpu_vram_sample_used_mib: int | None = None,
    year: int = 2013,
) -> Path:
    level_dir = root / f"w{workers}"
    models = [f"titans_mac_probe_{idx}" for idx in range(workers)]
    for idx, model in enumerate(models):
        is_oom = idx < oom_jobs
        key = f"{model}/{year}/seed-{42 + idx}"
        attempt = {
            "started_at_utc": "2026-11-23T10:00:00+00:00",
            "finished_at_utc": "2026-11-23T10:10:00+00:00",
            "pid": 1000 + idx,
            "status": "failed" if is_oom else "completed",
            "seconds": per_job_wall_s,
            "process_wall_seconds": per_job_wall_s + 2.0,
            "process_exit_code": 137 if is_oom else 0,
            "cuda_limit_bytes": 16 * 1024**3,
            "resources": {
                "wall_seconds": per_job_wall_s,
                "cpu_seconds": per_job_wall_s * 0.9,
                "peak_rss_bytes": int(per_job_rss_gib * 1024**3),
                "rss_scope": "process_lifetime",
                "device": "cuda:0",
                "site": "google-colab-rtx-pro-6000",
                "cuda_allocated_peak_bytes": int(per_job_vram_alloc_gib * 1024**3),
                "cuda_reserved_peak_bytes": int(per_job_vram_reserved_gib * 1024**3),
                "gpu_name": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
                "gpu_capability": [12, 0],
            },
        }
        if is_oom:
            attempt["error_type"] = "CUDAOutOfMemoryError"
            attempt["error"] = "CUDA out of memory"
        q_doc = {
            "status": "failed" if is_oom else "completed",
            "model": model,
            "year": year,
            "seed": 42 + idx,
            "tier": "core",
            "attempts": [attempt],
            "result": {"global_step": 0 if is_oom else per_job_steps},
        }
        _write_json(level_dir / "queue" / f"{key.replace('/', '--')}.json", q_doc)

        fit_doc = {
            "schema_version": 1,
            "kind": "titans_financial_chronological_run",
            "status": "failed" if is_oom else "completed",
            "final_test_opened": False,
            "global_step": 0 if is_oom else per_job_steps,
            "history": [
                {"epoch": 0, "global_step": 0, "train": None},
                {
                    "epoch": 1,
                    "global_step": 0 if is_oom else per_job_steps,
                    "train": {"updates": 0 if is_oom else per_job_steps, "segments": 371},
                },
            ],
            "attempts": [{"seconds": per_job_wall_s}],
        }
        _write_json(
            level_dir / "runs" / model / str(year) / f"seed-{42 + idx}" / "fit" / "run.json",
            fit_doc,
        )

    if host_ram_sample_used_gib is not None or gpu_vram_sample_used_mib is not None:
        sample = {
            "timestamp_utc": "2026-11-23T10:05:00+00:00",
            "host_ram_total_gib": 96.0,
            "host_ram_used_gib": host_ram_sample_used_gib
            if host_ram_sample_used_gib is not None
            else 20.0,
            "gpu_mem_used_mib": gpu_vram_sample_used_mib
            if gpu_vram_sample_used_mib is not None
            else 10000,
            "gpu_mem_total_mib": 97887,
            "gpu_util_pct": 85.0,
        }
        (level_dir / "samples.jsonl").write_text(json.dumps(sample) + "\n", encoding="utf-8")
    return level_dir


def test_parse_meminfo_and_nvidia_smi_exact_values_and_errors():
    mem_text = (
        "MemTotal:       98304000 kB\n"
        "MemFree:        12000000 kB\n"
        "MemAvailable:   65536000 kB\n"
        "SwapTotal:             0 kB\n"
        "SwapFree:              0 kB\n"
    )
    parsed = cp.parse_meminfo(mem_text)
    assert parsed["MemTotal"] == 98304000
    assert parsed["MemAvailable"] == 65536000
    assert parsed["used_kib"] == 98304000 - 65536000

    with pytest.raises(ValueError, match="MemAvailable"):
        cp.parse_meminfo("MemTotal: 1000 kB\n")

    gpu = cp.parse_nvidia_smi_memory("24576, 97887, 88\n")
    assert gpu == {"used_mib": 24576, "total_mib": 97887, "util_pct": 88}

    with pytest.raises(ValueError):
        cp.parse_nvidia_smi_memory("99999, 97887, 50\n")


def test_site_sheet_validation_rejects_tf32_and_accepts_strict_fp32(tmp_path):
    ok_path = tmp_path / "equipo_ok.json"
    _write_json(ok_path, _site_sheet(allow_tf32=False))
    loaded = cp.load_site_sheet(ok_path)
    assert loaded["fp32_strict_ok"] is True
    assert loaded["host_ram_total_gib"] == pytest.approx(96.0, rel=1e-4)
    assert loaded["gpu_vram_total_gib"] == pytest.approx(97887 / 1024.0, rel=1e-4)

    bad_path = tmp_path / "equipo_tf32.json"
    _write_json(bad_path, _site_sheet(allow_tf32=True))
    bad = cp.load_site_sheet(bad_path)
    assert bad["fp32_strict_ok"] is False


def test_sealed_2024_blocked_across_nested_int_str_and_flags():
    cp.assert_no_sealed_2024({"year": 2023, "fold": "fold-018", "final_test_opened": False})
    for bad in (
        {"year": 2024},
        {"year": "2024"},
        {"nested": [{"origin": 2024}]},
        {"path": "/content/state/packed-inputs/2024/train.arrow"},
        {"fold": "fold-019"},
        {"final_test_access": True},
        {"final_test_opened": True},
    ):
        with pytest.raises(ValueError, match="2024|sellado|final_test"):
            cp.assert_no_sealed_2024(bad)


def test_analyze_levels_1_2_4_6_and_recommend_optimal_safe_concurrency(tmp_path):
    site = cp.load_site_sheet(tmp_path / "site.json") if False else _site_sheet(ram_gib=96.0)
    site_path = tmp_path / "equipo.json"
    _write_json(site_path, site)
    site_info = cp.load_site_sheet(site_path)

    # 1 trabajador: 371 pasos en 180 s -> 123.67 pasos/min agregados
    d1 = _make_level_fixture(
        tmp_path,
        workers=1,
        per_job_steps=371,
        per_job_wall_s=180.0,
        per_job_rss_gib=12.0,
        per_job_vram_reserved_gib=7.5,
        host_ram_sample_used_gib=18.0,
        gpu_vram_sample_used_mib=8500,
    )
    # 2 trabajadores: 371 pasos en 190 s cada uno -> 234.32 pasos/min agregados (+89.5%)
    d2 = _make_level_fixture(
        tmp_path,
        workers=2,
        per_job_steps=371,
        per_job_wall_s=190.0,
        per_job_rss_gib=13.0,
        per_job_vram_reserved_gib=7.5,
        host_ram_sample_used_gib=30.0,
        gpu_vram_sample_used_mib=16800,
    )
    # 4 trabajadores: 371 pasos en 220 s cada uno -> 404.73 pasos/min agregados (+72.7%, RAM 58 GiB
    # = 60.4%)
    d4 = _make_level_fixture(
        tmp_path,
        workers=4,
        per_job_steps=371,
        per_job_wall_s=220.0,
        per_job_rss_gib=14.0,
        per_job_vram_reserved_gib=7.5,
        host_ram_sample_used_gib=58.0,
        gpu_vram_sample_used_mib=34000,
    )
    # 6 trabajadores: satura RAM del anfitrión (88 GiB / 96 GiB = 91.7% > 85%) -> inseguro
    d6 = _make_level_fixture(
        tmp_path,
        workers=6,
        per_job_steps=371,
        per_job_wall_s=270.0,
        per_job_rss_gib=15.0,
        per_job_vram_reserved_gib=7.5,
        host_ram_sample_used_gib=88.0,
        gpu_vram_sample_used_mib=51000,
    )

    reports = [
        cp.analyze_level_state(
            d1, workers=1, site_sheet=site_info, samples_path=d1 / "samples.jsonl"
        ),
        cp.analyze_level_state(
            d2, workers=2, site_sheet=site_info, samples_path=d2 / "samples.jsonl"
        ),
        cp.analyze_level_state(
            d4, workers=4, site_sheet=site_info, samples_path=d4 / "samples.jsonl"
        ),
        cp.analyze_level_state(
            d6, workers=6, site_sheet=site_info, samples_path=d6 / "samples.jsonl"
        ),
    ]

    assert reports[0]["aggregate_steps_per_min"] == pytest.approx(371.0 / 3.0, rel=1e-6)
    assert reports[2]["aggregate_steps_per_min"] == pytest.approx(
        4 * 371.0 / (220.0 / 60.0), rel=1e-6
    )
    assert reports[3]["host_ram_pct"] == pytest.approx(88.0 / 96.0 * 100.0, rel=1e-3)

    rec = cp.recommend_concurrency(reports)
    assert rec["recommended_workers"] == 4
    assert rec["status"] == "ok"
    by_w = {row["workers"]: row for row in rec["levels"]}
    assert by_w[4]["safe"] is True
    assert by_w[6]["safe"] is False
    assert "host_ram_exceeded" in by_w[6]["unsafe_reasons"]
    assert by_w[4]["speedup_vs_1"] == pytest.approx(
        reports[2]["aggregate_steps_per_min"] / reports[0]["aggregate_steps_per_min"], rel=1e-6
    )


def test_recommend_rejects_oom_and_diminishing_returns(tmp_path):
    site_path = tmp_path / "equipo.json"
    _write_json(site_path, _site_sheet(ram_gib=180.0))
    site_info = cp.load_site_sheet(site_path)

    # 1 -> 2 gana bien; 4 solo gana +5% (< 10% umbral); 6 tiene OOM
    d1 = _make_level_fixture(
        tmp_path,
        workers=1,
        per_job_steps=300,
        per_job_wall_s=180.0,
        per_job_rss_gib=10.0,
        per_job_vram_reserved_gib=7.0,
        host_ram_sample_used_gib=20.0,
    )
    d2 = _make_level_fixture(
        tmp_path,
        workers=2,
        per_job_steps=300,
        per_job_wall_s=200.0,
        per_job_rss_gib=10.0,
        per_job_vram_reserved_gib=7.0,
        host_ram_sample_used_gib=32.0,
    )
    d4 = _make_level_fixture(
        tmp_path,
        workers=4,
        per_job_steps=300,
        per_job_wall_s=385.0,
        per_job_rss_gib=10.0,
        per_job_vram_reserved_gib=7.0,
        host_ram_sample_used_gib=55.0,
    )
    d6 = _make_level_fixture(
        tmp_path,
        workers=6,
        per_job_steps=300,
        per_job_wall_s=400.0,
        per_job_rss_gib=10.0,
        per_job_vram_reserved_gib=7.0,
        oom_jobs=1,
        host_ram_sample_used_gib=75.0,
    )
    reports = [
        cp.analyze_level_state(d, workers=w, site_sheet=site_info, samples_path=d / "samples.jsonl")
        for w, d in ((1, d1), (2, d2), (4, d4), (6, d6))
    ]
    rec = cp.recommend_concurrency(reports, min_marginal_gain_pct=10.0)
    assert rec["recommended_workers"] == 2
    by_w = {row["workers"]: row for row in rec["levels"]}
    assert by_w[6]["safe"] is False
    assert "oom_detected" in by_w[6]["unsafe_reasons"]


def test_build_colab_probe_commands_and_cli_dry_run(tmp_path, capsys):
    plan = cp.build_colab_probe_commands(
        kit_dir=Path("/content/mars-titan/kit/scripts"),
        repo_dir=Path("/content/mars-titan/repo"),
        state_root=Path("/content/mars-titan/estado/sonda"),
        views_dir=Path("/home/gonzalo/.local/state/mars-titan/views/US"),
        declaration="configs/experiments/us-campaign-v11-screen1.json",
        origin=2013,
    )
    assert [entry["workers"] for entry in plan] == [1, 2, 4, 6]
    for entry in plan:
        assert "colab_origin.sh" in entry["command"]
        assert f"--trabajadores {entry['workers']}" in entry["command"]
        assert entry["env"]["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
        assert entry["env"]["TRITON_F32_DEFAULT"] == "ieee"
        assert entry["env"]["MARS_TITAN_SITE"] == "google-colab-rtx-pro-6000"

    with pytest.raises(ValueError, match="2024"):
        cp.build_colab_probe_commands(
            kit_dir=Path("/k"),
            repo_dir=Path("/r"),
            state_root=Path("/s"),
            views_dir=Path("/v"),
            declaration="d.json",
            origin=2024,
        )

    code = cp.main(
        [
            "plan",
            "--kit",
            "/content/mars-titan/kit/scripts",
            "--repo",
            "/content/mars-titan/repo",
            "--state-root",
            "/content/mars-titan/estado/sonda",
            "--views",
            "/views/US",
            "--declaration",
            "configs/experiments/us-campaign-v11-screen1.json",
            "--origin",
            "2013",
        ]
    )
    assert code == 0
    out = json.loads(capsys.readouterr().out)
    assert len(out["levels"]) == 4


def test_analyze_rejects_symlinks_and_2024_in_state(tmp_path):
    d1 = _make_level_fixture(
        tmp_path,
        workers=1,
        per_job_steps=100,
        per_job_wall_s=60.0,
        per_job_rss_gib=10.0,
        per_job_vram_reserved_gib=6.0,
        year=2024,
    )
    with pytest.raises(ValueError, match="2024"):
        cp.analyze_level_state(d1, workers=1)

    d_ok = _make_level_fixture(
        tmp_path / "sym",
        workers=1,
        per_job_steps=100,
        per_job_wall_s=60.0,
        per_job_rss_gib=10.0,
        per_job_vram_reserved_gib=6.0,
        year=2013,
    )
    target = tmp_path / "outside.json"
    target.write_text("{}", encoding="utf-8")
    (d_ok / "queue" / "link.json").symlink_to(target)
    with pytest.raises(ValueError, match="(?i)enlace"):
        cp.analyze_level_state(d_ok, workers=1)


def test_pass3_adversarial_2024_variants_vram_fp32_and_no_samples_fallback(tmp_path):
    # Variantes de evasión de 2024 (claves plurales, fechas compactas/ISO, flags truthy no bool)
    for sneaky in (
        {"origins": [2013, 2024]},
        {"years": ["2023", "2024"]},
        {"date": "20240102"},
        {"date": "2024-03-15"},
        {"final_test_access": 1},
        {"final_test_opened": "yes"},
    ):
        with pytest.raises(ValueError, match="2024|sellado|final_test"):
            cp.assert_no_sealed_2024(sneaky)

    # Sin samples.jsonl: debe estimar VRAM = suma reservada + workers * 512 MiB y RAM = suma RSS
    d2 = _make_level_fixture(
        tmp_path / "nosamp",
        workers=2,
        per_job_steps=400,
        per_job_wall_s=120.0,
        per_job_rss_gib=11.0,
        per_job_vram_reserved_gib=8.0,
    )
    site_path = tmp_path / "equipo_vram.json"
    _write_json(
        site_path, _site_sheet(ram_gib=128.0, vram_mib=20480, allow_tf32=False)
    )  # 20 GiB VRAM
    site_info = cp.load_site_sheet(site_path)
    rep = cp.analyze_level_state(d2, workers=2, site_sheet=site_info)
    assert rep["observed_host_ram_peak_gib"] == pytest.approx(22.0, rel=1e-6)
    assert rep["observed_vram_peak_gib"] == pytest.approx(
        16.0 + 2 * 0.5, rel=1e-6
    )  # 17 GiB / 20 GiB = 85%, libre 3 GiB < 6 GiB

    rec = cp.recommend_concurrency([rep])
    assert rec["status"] == "unsafe"
    assert rec["recommended_workers"] is None
    assert "vram_margin_low" in rec["levels"][0]["unsafe_reasons"]

    # Violación FP32 estricta (allow_tf32=True) debe marcar nivel como inseguro
    site_tf32 = tmp_path / "equipo_bad_tf32.json"
    _write_json(site_tf32, _site_sheet(ram_gib=128.0, vram_mib=97887, allow_tf32=True))
    rep_tf32 = cp.analyze_level_state(d2, workers=2, site_sheet=cp.load_site_sheet(site_tf32))
    rec_tf32 = cp.recommend_concurrency([rep_tf32])
    assert rec_tf32["status"] == "unsafe"
    assert "fp32_strict_violation" in rec_tf32["levels"][0]["unsafe_reasons"]

    # extract_job_telemetry sin fit_report usa queue_record["result"]["global_step"] y falla con
    # wall_s <= 0
    q_only = {
        "status": "completed",
        "model": "titans_mac",
        "year": 2013,
        "seed": 42,
        "attempts": [{"status": "completed", "seconds": 60.0, "resources": {"wall_seconds": 60.0}}],
        "result": {"global_step": 180},
    }
    row = cp.extract_job_telemetry(q_only, None, job_key="titans_mac/2013/seed-42")
    assert row["steps"] == 180
    assert row["steps_per_min"] == pytest.approx(180.0)

    q_zero_wall = {
        "status": "completed",
        "model": "titans_mac",
        "year": 2013,
        "seed": 42,
        "attempts": [{"status": "completed", "seconds": 0.0, "resources": {"wall_seconds": 0.0}}],
        "result": {"global_step": 100},
    }
    with pytest.raises(ValueError, match="wall_seconds"):
        cp.extract_job_telemetry(q_zero_wall, None, job_key="titans_mac/2013/seed-42")
