"""Comprobación CUDA M2 y M3 con etiquetas manuales y parámetros congelados."""

import gc
import hashlib
import json
import os
import random
import resource
import subprocess
import time

import numpy as np
import test_financial_session_m2 as session_helpers
import test_financial_session_m3 as m3_session
import torch
from test_financial_session import moment
from test_financial_session_controls import FLOWS
from test_financial_session_controls import four_flow_source as four_flow_source
from test_financial_session_controls import no_target_estimation as no_target_estimation
from test_native_episode_backend import native as native
from test_native_episode_backend import record

from mars_titan.hardware import hardware_profiles
from mars_titan.hardware.platform_identity import platform_identity
from mars_titan.models.titans.episodic_readout import (
    EpisodicReadout,
    EpisodicReadoutConfig,
    copy_readout_parameters,
)
from mars_titan.models.titans.financial import (
    FinancialConfig,
    FinancialPredictor,
    copy_paired_parameters,
)
from mars_titan.models.titans.frozen_financial import FrozenFinancialConsumer
from mars_titan.models.titans.local_control import MACProjectionConfig

CAP = 128 * 1024**2
LABELS = {
    moment(125): (2.0, 8.0, 4.0, 6.0),
    moment(126): (3.0, 7.0, 5.0, 1.0),
    moment(127): (12.0, 9.0, 11.0, 10.0),
    moment(128): (3.5, 6.5, 4.5, 5.5),
}


def manual_labels(native, predictions, index):
    return [
        native.Feedback(p.id, 0, moment(index), LABELS[p.decision_at][FLOWS.index(p.asset)])
        for p in predictions
    ]


def parameter_tensors(consumer, *, include_buffers=True):
    result = {}
    for role, model in (("predictor", consumer.predictor), ("readout", consumer.readout)):
        values = list(model.named_parameters())
        if include_buffers:
            values.extend(model.named_buffers())
        for name, value in values:
            result[f"{role}/{name}"] = value.detach().cpu().clone()
    return result


def tensor_digest(values):
    digest = hashlib.sha256()
    for name, value in sorted(values.items()):
        digest.update(name.encode())
        digest.update(str((value.dtype, tuple(value.shape))).encode())
        digest.update(value.contiguous().numpy().tobytes())
    return digest.hexdigest()


def consumer(source, dtype, device, parent=None):
    model = (
        FinancialPredictor(
            FinancialConfig(source["spec"], variant="mac_online", hidden_size=32, seed=42),
            local_control=MACProjectionConfig(
                mode="disabled", rank=1, grid_size=4, frequency=1, max_flows=2, seed=91
            ),
            dtype=torch.float32,
            device="cpu",
        )
        .to(device=device, dtype=dtype)
        .eval()
        .requires_grad_(False)
    )
    readout = (
        EpisodicReadout(
            EpisodicReadoutConfig(
                source["codec"].fingerprint(), hidden_size=32, refinements=1, neighbors=4, seed=42
            ),
            dtype=torch.float32,
            device="cpu",
        )
        .to(device=device, dtype=dtype)
        .eval()
        .requires_grad_(False)
    )
    copies = None
    if parent is not None:
        copies = dict(
            predictor=copy_paired_parameters(parent.predictor, model),
            readout=copy_readout_parameters(parent.readout, readout),
        )
    return FrozenFinancialConsumer(model, readout=readout), copies


def validate_indices(native, result):
    scope = native.MemoryScope()
    scope.world, scope.partition, scope.fold, scope.representation = (
        "manual_oracle",
        "validation",
        "0",
        "64",
    )
    oracle = native.EpisodicMemory(scope, 73, 2, 2)
    errors, gaps = {}, []
    for event, decision in enumerate((None, 125, 126, 127, 128, None)):
        if decision is not None:
            for flow in FLOWS:
                identifier = len(errors) + 1
                point = result["points"][flow, moment(decision)]
                label = LABELS[moment(decision)][FLOWS.index(flow)]
                errors[identifier] = abs(label - point)
                oracle.write(record(native, identifier), 100)
        indices = result["indices"][event]
        selected = sorted(errors, key=lambda key: (-errors[key], key))
        assert indices["selective"] == tuple(selected[:1])
        assert indices["recent"] == ((len(errors),) if errors else ())
        assert indices["reservoir"] == tuple(sorted(r.id for r in oracle.retained_records()))
        assert result["scores"][event] == {key: errors[key] for key in selected[:1]}
        if errors:
            gap = errors[selected[0]] - errors[selected[1]]
            assert gap > 0.1, "El fixture no separa suficientemente las puntuaciones M2"
            gaps.append(gap)
            receipt = result["receipts"][event]
            union = set().union(*map(set, indices.values()))
            assert receipt["physical_slots"] == 4
            assert receipt["unique_episodes"] == len(union) <= 4
            assert receipt["duplicate_slots"] == 4 - len(union)
    assert result["indices"][1]["selective"] != result["indices"][1]["recent"]
    assert result["final"]["applied"] == 16 > 4
    return min(gaps)


def compare(left, right, *, rtol, atol, exact_recovery=False):
    assert left["points"].keys() == right["points"].keys()
    keys = sorted(left["points"])
    expected = np.array([left["points"][key] for key in keys])
    actual = np.array([right["points"][key] for key in keys])
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)
    assert left["indices"] == right["indices"]
    assert left["receipts"] == right["receipts"]
    assert left["fast"].keys() == right["fast"].keys()
    for key in left["fast"]:
        torch.testing.assert_close(left["fast"][key], right["fast"][key], rtol=rtol, atol=atol)
    if exact_recovery:
        assert left["final"] == right["final"]
        assert left["scores"] == right["scores"]
    return float(np.max(np.abs(actual - expected)))


def cuda_preflight():
    """Hardware, límite del asignador y aritmética sin TF32 ni fastpath, comunes a M2 y M3."""
    hardware = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.free,driver_version",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()
    assert torch.cuda.is_available(), "CUDA es obligatoria, no se admite fallback"
    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)
    # La GPU debe ser la del perfil declarado, no una cualquiera con CUDA.
    hardware_profiles.check_profile(hardware_profiles.declared_profile(), platform_identity())
    torch.cuda.set_per_process_memory_fraction(CAP / properties.total_memory, device)
    free, _ = torch.cuda.mem_get_info(device)
    assert free >= CAP, "La memoria CUDA libre no cubre el límite del asignador"
    torch.cuda.reset_peak_memory_stats(device)
    torch.set_num_threads(2)
    # Solo se puede fijar una vez por proceso, antes de cualquier trabajo paralelo.
    if torch.get_num_interop_threads() != 2:
        torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.backends.mha.set_fastpath_enabled(False)
    return device, hardware, free


def test_cuda_m2_manual_parity_overflow_and_recovery(
    native, four_flow_source, tmp_path, monkeypatch
):
    # No se fija una compilación concreta: el enlace incorpora rutas absolutas de include y
    # solo se reproduce en su worktree. load_native ya comprueba contrato y runtime, y el
    # recibo registra la huella del binario realmente cargado.
    loaded = os.path.realpath(os.environ["MARS_TITAN_EPISODIC_NATIVE"])
    with open(loaded, "rb") as handle:
        assert native.binary_sha256 == hashlib.sha256(handle.read()).hexdigest()
    started = time.perf_counter()
    device, hardware, free = cuda_preflight()
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    cpu_rng = torch.random.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state(device).clone()
    monkeypatch.setattr(session_helpers, "labels", manual_labels)
    source = four_flow_source
    assert source["label_origin"] == "manual_fixture_no_estimation"
    output = tmp_path
    (output / "preflight.json").write_text(
        json.dumps(
            dict(
                hardware=hardware,
                device="cuda:0",
                cap_torch_bytes=CAP,
                free_after_context_bytes=free,
                native_sha256=native.binary_sha256,
                torch=torch.__version__,
                cuda=torch.version.cuda,
                mha_fastpath=False,
                tf32=False,
                matmul_precision="highest",
            ),
            indent=2,
        )
        + "\n"
    )
    records, common_parameters = [], None
    for dtype, rtol, atol in ((torch.float32, 5e-5, 3e-6), (torch.float64, 2e-9, 2e-10)):
        cpu, _ = consumer(source, dtype, "cpu")
        gpu, copies = consumer(source, dtype, device, parent=cpu)
        before = parameter_tensors(cpu)
        assert tensor_digest(before) == tensor_digest(parameter_tensors(gpu))
        cast_parameters = tensor_digest(
            {
                name: value.float()
                for name, value in parameter_tensors(cpu, include_buffers=False).items()
            }
        )
        if common_parameters is None:
            common_parameters = cast_parameters
        assert cast_parameters == common_parameters
        paths, times, margins = {}, {}, {}
        for name, engine, recover in (
            ("cpu", cpu, False),
            ("cpu_recovered", cpu, True),
            ("cuda", gpu, False),
            ("cuda_recovered", gpu, True),
        ):
            torch.cuda.synchronize(device)
            begin = time.perf_counter()
            paths[name] = session_helpers.m2_trajectory(
                native, source, engine, tmp_path / f"{dtype}-{name}", block_rows=4, recover=recover
            )
            torch.cuda.synchronize(device)
            times[name] = time.perf_counter() - begin
            margins[name] = validate_indices(native, paths[name])
        error = compare(paths["cpu"], paths["cuda"], rtol=rtol, atol=atol)
        compare(paths["cpu"], paths["cpu_recovered"], rtol=0, atol=0, exact_recovery=True)
        compare(paths["cuda"], paths["cuda_recovered"], rtol=0, atol=0, exact_recovery=True)
        for engine in (cpu, gpu):
            engine.verify()
            assert tensor_digest(parameter_tensors(engine)) == tensor_digest(before)
            assert all(p.grad is None for p in engine.predictor.parameters())
            assert all(p.grad is None for p in engine.readout.parameters())
        records.append(
            dict(
                dtype=str(dtype),
                rtol=rtol,
                atol=atol,
                max_prediction_error=error,
                synchronized_walk_seconds=times,
                minimum_selective_error_margin=margins,
                parameter_digest=tensor_digest(before),
                paired_copies=copies,
                retained_ids=paths["cuda"]["indices"],
                final_receipt=paths["cuda"]["receipts"][-1],
                exact_recovery_each_device=True,
                parameters_unchanged=True,
            )
        )
        (output / f"case-{dtype}.json").write_text(
            json.dumps(records[-1], ensure_ascii=False, indent=2) + "\n"
        )
        del cpu, gpu, before, paths, engine
        gc.collect()
        torch.cuda.empty_cache()
    assert random.getstate() == python_rng
    after_numpy = np.random.get_state()
    assert after_numpy[0] == numpy_rng[0] and after_numpy[2:] == numpy_rng[2:]
    np.testing.assert_array_equal(after_numpy[1], numpy_rng[1])
    assert torch.equal(cpu_rng, torch.random.get_rng_state())
    assert torch.equal(cuda_rng, torch.cuda.get_rng_state(device))
    peak = torch.cuda.max_memory_allocated(device)
    assert peak <= CAP
    report = dict(
        cases=records,
        hardware=hardware,
        torch=torch.__version__,
        cuda=torch.version.cuda,
        native_sha256=native.binary_sha256,
        cap_torch_bytes=CAP,
        peak_torch_bytes=peak,
        peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
        wall_seconds=time.perf_counter() - started,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        B_mem=4,
        quotas=[2, 1, 1],
        K=1,
        C="disabled",
        cm_m="disabled",
        admission="m2",
        mac_backend="math",
        mha_fastpath=False,
        tf32=False,
        global_rng_unchanged=True,
        fixture_flows=4,
        trajectories_per_dtype=4,
        emitted_per_trajectory=20,
        matured_per_trajectory=16,
        pending_per_trajectory=4,
        labels="manual_fixture_no_estimation",
        input_digests=[row.input_digest for rows in source["batches"].values() for row in rows],
        common_fp32_parameter_digest=common_parameters,
        parameter_recipe="Inicialización FP32 común y conversión explícita antes del sellado",
        target_estimation_calls=0,
        optimizer_steps=0,
        transfer_bytes_measured=False,
        scientific_evaluation=False,
        scope="Composición técnica con parámetros congelados, sin evaluar utilidad predictiva.",
    )
    destination = output / "report.json"
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


# M3: mismos índices que M2 con la puntuación de error, anomalía y relevancia.
# `MARS_TITAN_M3_CHECK_DEVICE=cpu` ensaya la lógica sin GPU y no acredita CUDA.
M3_DEVICE = os.environ.get("MARS_TITAN_M3_CHECK_DEVICE", "cuda:0")


def m3_margin(receipts):
    """Menor distancia entre la puntuación elegida por el selectivo y la mejor descartada.

    Compiten las ofertas de cada evento y el selectivo anterior. Un margen amplio frente a
    la tolerancia del dispositivo evita que los IDs dependan del redondeo.
    """
    known, before, margins = {}, set(), []
    for indices, receipt, _ in receipts:
        if receipt is None:
            continue
        known.update((row[0], row[4]) for row in receipt["components"])
        selected = set(indices["selective"])
        contenders = ({row[0] for row in receipt["components"]} | before) - selected
        if contenders and selected:
            margins.append(min(known[i] for i in selected) - max(known[i] for i in contenders))
        before = selected
    return min(margins)


def compare_m3(left, right, *, rtol, atol):
    """IDs, ofertas y rasgos de entrada exactos. Error, puntuación y predicciones con tolerancia."""
    (receipts, _, diagnostics, points), (other, _, other_diagnostics, other_points) = left, right
    assert points.keys() == other_points.keys()
    keys = sorted(points)
    expected = np.array([points[key] for key in keys])
    actual = np.array([other_points[key] for key in keys])
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)
    assert diagnostics["B_mem"] == other_diagnostics["B_mem"] == 4
    assert diagnostics["physical_slots"] == other_diagnostics["physical_slots"]
    measured, reference = [], []
    for (indices, receipt, features), (o_indices, o_receipt, o_features) in zip(
        receipts, other, strict=True
    ):
        assert indices == o_indices and (receipt is None) == (o_receipt is None)
        assert features.keys() == o_features.keys()
        for key, raw in features.items():
            assert raw[1:] == o_features[key][1:]
            reference.append(raw[0])
            measured.append(o_features[key][0])
        if receipt is None:
            continue
        assert {k: v for k, v in receipt.items() if k != "components"} == {
            k: v for k, v in o_receipt.items() if k != "components"
        }
        for row, o_row in zip(receipt["components"], o_receipt["components"], strict=True):
            assert (row[0], row[2], row[3], row[5]) == (o_row[0], o_row[2], o_row[3], o_row[5])
            reference.extend((row[1], row[4]))
            measured.extend((o_row[1], o_row[4]))
    np.testing.assert_allclose(measured, reference, rtol=rtol, atol=atol)
    return float(np.max(np.abs(actual - expected)))


def test_cuda_m3_scores_parity_and_recovery(native, four_flow_source, tmp_path):
    started = time.perf_counter()
    cuda = M3_DEVICE.startswith("cuda")
    hardware = None
    if cuda:
        device, hardware, _ = cuda_preflight()
    else:
        device = torch.device(M3_DEVICE)
        torch.backends.mha.set_fastpath_enabled(False)
    python_rng, cpu_rng = random.getstate(), torch.random.get_rng_state().clone()
    # Noticias, fundamentales y últimos precios distintos por flujo: los tres componentes varían.
    source = m3_session.varied(four_flow_source)
    assert source["label_origin"] == "manual_fixture_no_estimation"
    records = []
    for dtype, rtol, atol in ((torch.float32, 5e-5, 3e-6), (torch.float64, 2e-9, 2e-10)):
        cpu, _ = consumer(source, dtype, "cpu")
        other, copies = consumer(source, dtype, device, parent=cpu)
        before = parameter_tensors(cpu)
        assert tensor_digest(before) == tensor_digest(parameter_tensors(other))
        paths, times = {}, {}
        for name, engine, recover in (
            ("cpu", cpu, False),
            ("cpu_recovered", cpu, True),
            ("device", other, False),
            ("device_recovered", other, True),
        ):
            if cuda:
                torch.cuda.synchronize(device)
            begin = time.perf_counter()
            paths[name] = m3_session.trajectory(
                native, source, engine, tmp_path / f"m3-{dtype}-{name}", recover=recover
            )
            if cuda:
                torch.cuda.synchronize(device)
            times[name] = time.perf_counter() - begin
        assert paths["cpu"] == paths["cpu_recovered"]
        assert paths["device"] == paths["device_recovered"]
        error = compare_m3(paths["cpu"], paths["device"], rtol=rtol, atol=atol)
        receipts, final, _, _ = paths["cpu"]
        margin = m3_margin(receipts)
        assert margin > 1e-4, "El fixture no separa suficientemente las puntuaciones M3"
        assert (final["issued"], final["applied"]) == (20, 16)
        masks = {row[5] for _, receipt, _ in receipts if receipt for row in receipt["components"]}
        assert masks == {0, 1, 2, 3}
        for engine in (cpu, other):
            engine.verify()
            assert tensor_digest(parameter_tensors(engine)) == tensor_digest(before)
            assert all(p.grad is None for p in engine.predictor.parameters())
            assert all(p.grad is None for p in engine.readout.parameters())
        records.append(
            dict(
                dtype=str(dtype),
                rtol=rtol,
                atol=atol,
                max_prediction_error=error,
                minimum_selective_score_margin=margin,
                synchronized_walk_seconds=times,
                paired_copies=copies,
                retained_ids=[indices for indices, _, _ in paths["device"][0]],
                final_receipt=paths["device"][0][-1][1],
                exact_recovery_each_device=True,
                parameters_unchanged=True,
            )
        )
        del cpu, other, before, paths, engine
        gc.collect()
        if cuda:
            torch.cuda.empty_cache()
    assert random.getstate() == python_rng
    assert torch.equal(cpu_rng, torch.random.get_rng_state())
    report = dict(
        cases=records,
        device=str(device),
        hardware=hardware,
        torch=torch.__version__,
        cuda=torch.version.cuda,
        native_sha256=native.binary_sha256,
        peak_torch_bytes=torch.cuda.max_memory_allocated(device) if cuda else None,
        wall_seconds=time.perf_counter() - started,
        B_mem=4,
        quotas=[2, 1, 1],
        K=1,
        C="disabled",
        cm_m="disabled",
        admission="m3",
        scalers_sha256=m3_session.M3.scalers.fingerprint(),
        labels="manual_fixture_no_estimation",
        target_estimation_calls=0,
        optimizer_steps=0,
        scientific_evaluation=False,
        scope="Composición técnica con parámetros congelados, sin evaluar utilidad predictiva.",
    )
    if cuda:
        assert report["peak_torch_bytes"] <= CAP
    (tmp_path / "m3-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
