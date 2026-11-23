"""Pruebas TDD para el plan de pruebas y barrido sm_120 de fused_memory_write.py."""

from __future__ import annotations

import json

import kernel_sm120_plan as kp
import pytest
import torch


def test_strict_fp32_deterministic_guard_enforces_all_flags_and_rejects_autocast():
    state = kp.enforce_strict_fp32_deterministic(check_triton=True)
    assert state["deterministic_algorithms"] is True
    assert state["matmul_allow_tf32"] is False
    assert state["cudnn_allow_tf32"] is False
    assert state["matmul_precision"] == "highest"
    assert state["cudnn_deterministic"] is True
    assert state["cudnn_benchmark"] is False
    assert state["triton_fp32_default"] == "ieee"

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        with pytest.raises(RuntimeError, match="autocast"):
            kp.enforce_strict_fp32_deterministic(check_triton=True)


def test_ulp_distance_fp32_exact_including_signed_zero_and_negatives():
    a = torch.tensor([1.0, -1.0, 0.0, -0.0, 1.0], dtype=torch.float32)
    b = torch.tensor(
        [
            torch.nextafter(
                torch.tensor(1.0, dtype=torch.float32), torch.tensor(2.0, dtype=torch.float32)
            ).item(),
            torch.nextafter(
                torch.tensor(-1.0, dtype=torch.float32), torch.tensor(-2.0, dtype=torch.float32)
            ).item(),
            -0.0,
            torch.nextafter(
                torch.tensor(0.0, dtype=torch.float32), torch.tensor(1.0, dtype=torch.float32)
            ).item(),
            -1.0,
        ],
        dtype=torch.float32,
    )
    dist = kp.ulp_distance_fp32(a, b)
    assert dist[0].item() == 1
    assert dist[1].item() == 1
    assert dist[2].item() == 0  # +0.0 y -0.0 son 0 ULP
    assert dist[3].item() == 1
    assert dist[4].item() > 1_000_000_000  # cruce de signo 1.0 vs -1.0

    with pytest.raises(ValueError, match="float32"):
        kp.ulp_distance_fp32(torch.ones(2, dtype=torch.float64), torch.ones(2, dtype=torch.float64))
    with pytest.raises(ValueError, match="finitos"):
        kp.ulp_distance_fp32(
            torch.tensor([float("nan")], dtype=torch.float32),
            torch.tensor([0.0], dtype=torch.float32),
        )


def test_tensor_sha256_detects_single_ulp_and_shape_or_dtype_changes():
    x = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    next_one = torch.nextafter(
        torch.tensor(1.0, dtype=torch.float32), torch.tensor(2.0, dtype=torch.float32)
    ).item()
    y = torch.tensor([[next_one, 2.0]], dtype=torch.float32)
    assert kp.tensor_sha256(x) == kp.tensor_sha256(x.clone())
    assert kp.tensor_sha256(x) != kp.tensor_sha256(y)
    assert kp.tensor_sha256(x) != kp.tensor_sha256(x.reshape(2, 1))
    assert kp.tensor_sha256(x) != kp.tensor_sha256(x.to(torch.float64))


def test_level0_bitwise_and_level1_fp64_parity_cpu_harness():
    kp.enforce_strict_fp32_deterministic(check_triton=True)
    report = kp.run_cpu_verification_plan(batches=(1, 7, 13), layer_norms=(True, False), seed=42)
    assert report["strict_fp32_verified"] is True
    assert report["level0_all_passed"] is True
    assert report["level1_all_passed"] is True
    assert report["all_passed"] is True
    for case in report["cases"]:
        assert case["level0"]["deterministic_bitwise"] is True
        assert case["level0"]["reference_match_bitwise"] is True
        assert case["level0"]["batch_invariant_bitwise"] is True
        assert case["level0"]["noncontiguous_match_bitwise"] is True
        assert len(case["level0"]["output_hashes"]) == 5
        assert len(case["level0"]["grad_hashes"]) == 9
        assert case["level1"]["passed"] is True
        assert case["level1"]["statistical_equivalent"] is True
        assert case["level1"]["max_ulp_fwd"] <= kp.LEVEL1_FP32_FWD_MAX_ULP
        assert case["level1"]["max_ulp_bwd"] <= kp.LEVEL1_FP32_BWD_MAX_ULP


def test_level1_rejects_excessive_error_tf32_drift_or_nan():
    ref = torch.linspace(-1.0, 1.0, 64, dtype=torch.float64)
    # Ruido simulando deriva TF32 (~1e-3) -> debe fallar Nivel 1
    tf32_like = (ref + 2e-3).to(torch.float32)
    res_drift = kp.evaluate_level1_fp64_parity(
        (tf32_like,), (tf32_like,), (ref,), (ref,), stat_margin=1e-4
    )
    assert res_drift["passed"] is False
    assert res_drift["violations"]

    # NaN debe fallar inmediatamente
    with_nan = ref.to(torch.float32)
    with_nan[0] = float("nan")
    res_nan = kp.evaluate_level1_fp64_parity((with_nan,), (ref.to(torch.float32),), (ref,), (ref,))
    assert res_nan["passed"] is False
    assert any("finito" in v.lower() or "nan" in v.lower() for v in res_nan["violations"])

    # Equivalencia estadística TOST detecta sesgo sistemático dentro de atol/rtol si excede
    # stat_margin
    biased = (ref + 5e-7).to(torch.float32)
    tost_fail = kp.tost_zero_bias(biased.to(torch.float64) - ref, margin=1e-7)
    assert tost_fail["equivalent"] is False
    tost_ok = kp.tost_zero_bias(ref.to(torch.float32).to(torch.float64) - ref, margin=1e-6)
    assert tost_ok["equivalent"] is True


def test_sm120_occupancy_and_sweep_grid_enforce_99kib_cta_48_warps_and_power_of_two():
    # Granularidad 256 regs/warp, 48 warps/SM, 99 KiB/CTA
    occ = kp.sm120_occupancy(num_warps=16, regs_per_thread=100, smem_bytes=2048)
    assert occ["regs_per_block"] == 53248  # ceil(100*32/256)*256 * 16 = 3328 * 16
    assert occ["active_blocks_per_sm"] == 1
    assert occ["occupancy"] == pytest.approx(16 / 48.0)

    # Superar 99 KiB/CTA debe rechazarse (trampa H100/B200 228 KiB)
    with pytest.raises(ValueError, match="99 KiB"):
        kp.sm120_occupancy(num_warps=8, regs_per_thread=64, smem_bytes=100 * 1024)

    # Rejilla residente en registros para D=64 (kernel actual sin bucle for)
    reg_grid = kp.generate_sm120_sweep_grid(
        dim=64, regime="register_resident", num_warps_list=(2, 4, 8, 16), num_stages_list=(1, 2, 3)
    )
    assert reg_grid["autotune_forbidden"] is True
    assert reg_grid["arch"] == "sm_120"
    assert len(reg_grid["valid_configs"]) == 4  # 1 stage canónico (sin bucle for) x 4 num_warps
    assert any(
        e["reason"] == "redundant_num_stages_without_loop" for e in reg_grid["eliminated_configs"]
    )

    # Rejilla tiled para D=128: debe filtrar configuraciones > 99 KiB/CTA y bloques no potencia de 2
    tiled_grid = kp.generate_sm120_sweep_grid(
        dim=128,
        regime="tiled_smem",
        block_ms=(32, 64, 96, 128),
        block_ks=(32, 64, 128),
        num_warps_list=(4, 8),
        num_stages_list=(2, 3, 4),
    )
    assert tiled_grid["valid_configs"]
    for cfg in tiled_grid["valid_configs"]:
        assert cfg["smem_bytes"] <= kp.SM120_SPEC.max_smem_per_block_bytes
        assert (cfg["block_m"] & (cfg["block_m"] - 1)) == 0
        assert (cfg["block_k"] & (cfg["block_k"] - 1)) == 0
    reasons = {e["reason"] for e in tiled_grid["eliminated_configs"]}
    assert "non_power_of_two_block" in reasons
    assert "smem_exceeds_sm120_99kib_cta_limit" in reasons


def test_offline_ptx_audit_of_actual_repo_v11_fused_memory_write_on_sm120():
    audit = kp.audit_repo_v11_kernel_sm120_ptx(num_warps_fwd=4, num_warps_bwd=8, num_stages=2)
    assert audit["arch"] == "sm_120"
    assert audit["forward"]["atomic_ops"] == 0
    assert audit["forward"]["tf32_ops"] == 0
    assert audit["backward"]["atomic_ops"] == 0
    assert audit["backward"]["tf32_ops"] == 0
    assert audit["forward"]["shared_bytes"] <= kp.SM120_SPEC.max_smem_per_block_bytes
    assert audit["backward"]["shared_bytes"] <= kp.SM120_SPEC.max_smem_per_block_bytes


def test_cli_cpu_verify_and_grid_output_json(capsys):
    code = kp.main(["--mode", "grid", "--dim", "64", "--regime", "register_resident"])
    assert code == 0
    out = json.loads(capsys.readouterr().out)
    assert out["arch"] == "sm_120"
    assert out["autotune_forbidden"] is True


def test_pass3_adversarial_fp32_verify_level0_negatives_ulp_limit_and_gpu_guard():
    # 1. verify_strict_fp32_deterministic detecta cada bandera violada sin silenciarla
    kp.enforce_strict_fp32_deterministic(check_triton=True)
    import triton

    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        with pytest.raises(RuntimeError, match="Contrato FP32"):
            kp.verify_strict_fp32_deterministic(check_triton=True)
        torch.backends.cuda.matmul.allow_tf32 = False

        torch.backends.cudnn.benchmark = True
        with pytest.raises(RuntimeError, match="Contrato FP32"):
            kp.verify_strict_fp32_deterministic(check_triton=True)
        torch.backends.cudnn.benchmark = False

        triton.knobs.language.fp32_default = "tf32"
        with pytest.raises(RuntimeError, match="fp32_default"):
            kp.verify_strict_fp32_deterministic(check_triton=True)
    finally:
        kp.enforce_strict_fp32_deterministic(check_triton=True)

    # 2. Nivel 0 negativo: detecta 1 ULP contra referencia, no determinismo, varianza de lote y bug
    # de stride
    in64, up64 = kp.make_write_inputs(3, dtype=torch.float64, seed=99)
    in32 = tuple(t.to(torch.float32) for t in in64)
    up32 = tuple(t.to(torch.float32) for t in up64)

    def ref_perturbed(w, m, k, v, g, **kw):
        (nw1, nw2), (ns1, ns2), elem = kp.reference_write_fp32_deterministic(w, m, k, v, g, **kw)
        delta = torch.zeros_like(nw1)
        val0 = nw1[0, 0, 0].detach()
        delta[0, 0, 0] = torch.nextafter(val0, val0 + 1.0) - val0
        return (nw1 + delta, nw2), (ns1, ns2), elem

    res_ref_diff = kp.evaluate_level0_bitwise(
        ref_perturbed, kp.reference_write_fp32_deterministic, in32, up32, layer_norm=True, repeats=2
    )
    assert res_ref_diff["passed"] is False
    assert res_ref_diff["reference_match_bitwise"] is False

    call_n = {"i": 0}

    def nondet_fn(w, m, k, v, g, **kw):
        call_n["i"] += 1
        (nw1, nw2), (ns1, ns2), elem = kp.reference_write_fp32_deterministic(w, m, k, v, g, **kw)
        return (nw1 + (1e-6 if call_n["i"] == 2 else 0.0), nw2), (ns1, ns2), elem

    res_nondet = kp.evaluate_level0_bitwise(
        nondet_fn, kp.reference_write_fp32_deterministic, in32, up32, layer_norm=True, repeats=2
    )
    assert res_nondet["passed"] is False
    assert res_nondet["deterministic_bitwise"] is False

    def stride_bug_fn(w, m, k, v, g, **kw):
        # Si recibe tensor no contiguo, inyecta perturbación simulando bug de stride
        (nw1, nw2), (ns1, ns2), elem = kp.reference_write_fp32_deterministic(w, m, k, v, g, **kw)
        if not w[0].is_contiguous():
            nw1 = nw1 + 1e-4
        return (nw1, nw2), (ns1, ns2), elem

    res_stride = kp.evaluate_level0_bitwise(
        stride_bug_fn, kp.reference_write_fp32_deterministic, in32, up32, layer_norm=True, repeats=2
    )
    assert res_stride["passed"] is False
    assert res_stride["noncontiguous_match_bitwise"] is False

    def batch_bug_fn(w, m, k, v, g, **kw):
        # La fila 0 sale distinta según cuántas filas la acompañan en el lote
        (nw1, nw2), (ns1, ns2), elem = kp.reference_write_fp32_deterministic(w, m, k, v, g, **kw)
        return (nw1 + (1e-4 if w[0].shape[0] > 1 else 0.0), nw2), (ns1, ns2), elem

    res_batch = kp.evaluate_level0_bitwise(
        batch_bug_fn, batch_bug_fn, in32, up32, layer_norm=True, repeats=2
    )
    assert res_batch["passed"] is False
    assert res_batch["batch_invariant_bitwise"] is False
    assert res_batch["deterministic_bitwise"] and res_batch["reference_match_bitwise"]
    assert res_batch["noncontiguous_match_bitwise"]

    # 3. Nivel 1 detecta exceso de ULP aunque rtol/atol sean laxos + TOST rechaza media 0 con SE
    # ancho
    ref = torch.ones(64, dtype=torch.float64)
    shifted = (ref + 1e-4).to(torch.float32)  # ~839 ULP > límite 100, dentro de rtol=1e-2
    res_ulp = kp.evaluate_level1_fp64_parity(
        (shifted,),
        (ref.to(torch.float32),),
        (ref,),
        (ref,),
        fwd_rtol=1e-2,
        fwd_atol=1e-6,
        fwd_max_ulp=100,
        stat_margin=1e-2,
    )
    assert res_ulp["passed"] is False
    assert any("ULP" in v for v in res_ulp["violations"])

    wide_zero_mean = torch.tensor([-1.0, 1.0] * 10, dtype=torch.float64)
    assert kp.tost_zero_bias(wide_zero_mean, margin=0.05)["equivalent"] is False

    # 4. run_sm120_gpu_sweep en CPU falla con etiqueta [no verificado en GPU]
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match=r"no verificado en GPU"):
            kp.run_sm120_gpu_sweep()
