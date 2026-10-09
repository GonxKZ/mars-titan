"""Comprobación CUDA de la GRU episódica en el coordinador, con etiquetas manuales.

Se ejecuta de forma explícita. `MARS_TITAN_GRU_CHECK_DEVICE=cpu` ensaya la misma lógica
sin GPU y no acredita CUDA.
"""

import json
import os
import random
import resource
import subprocess
import time

import numpy as np
import pytest
import torch
from test_financial_session import moment
from test_financial_session_controls import FLOWS
from test_financial_session_controls import four_flow_source as four_flow_source
from test_financial_session_controls import no_target_estimation as no_target_estimation
from test_native_episode_backend import native as native

from mars_titan.memory.candidate_bank import CandidateBankConfig
from mars_titan.memory.financial_session import FinancialPhase, FinancialSession
from mars_titan.models.candidate.episode_codec import FrozenCandidateCodec
from mars_titan.models.candidate.frozen_consumer import FrozenCandidateConsumer
from mars_titan.models.candidate.input_adapter import CandidateInputAdapter

CAP = 128 * 1024**2
DEVICE = os.environ.get("MARS_TITAN_GRU_CHECK_DEVICE", "cuda:0")
EVENTS = (125, 126, 127, 128)
LABELS = {
    moment(125): (2.0, 8.0, 4.0, 6.0),
    moment(126): (3.0, 7.0, 5.0, 1.0),
    moment(127): (12.0, 9.0, 11.0, 10.0),
    moment(128): (3.5, 6.5, 4.5, 5.5),
}
TOLERANCES = {torch.float32: (2e-4, 2e-6), torch.float64: (1e-8, 1e-10)}


def frozen(adapter, refinements):
    adapter.model.eval()
    for value in adapter.model.named_parameters().values():
        value.requires_grad_(False)
    return FrozenCandidateConsumer(adapter, refinements=refinements), FrozenCandidateCodec(adapter)


def pair(source, dtype, refinements):
    cpu = CandidateInputAdapter(source["spec"], dtype=dtype)
    device = CandidateInputAdapter.restore(cpu.export_state(), source["spec"], device=DEVICE)
    return frozen(cpu, refinements), frozen(device, refinements)


def parameters(consumer):
    return {
        name: value.detach().cpu().clone()
        for name, value in consumer.model.named_parameters().items()
    }


def trajectory(native, source, engine, output, *, admission, cut):
    consumer, codec = engine

    def session(resume=False):
        return FinancialSession(
            output,
            native=native,
            consumer=consumer,
            codec=codec,
            prefixes=source["prefixes"],
            retention=CandidateBankConfig(capacity=4, seed=73),
            phase=FinancialPhase("validation", moment(125), moment(125), moment(200), moment(201)),
            world="manual_gru_cuda",
            fold="0",
            admission=admission,
            block_rows=4,
            resume=resume,
        )

    points, run = {}, session()
    try:
        for index in EVENTS:
            pending = [p for p in run._executor.pending() if p.decision_at < moment(index)]
            feedback = [
                native.Feedback(p.id, 0, moment(index), LABELS[p.decision_at][FLOWS.index(p.asset)])
                for p in pending
            ]
            if cut and index == 127:

                def fail(boundary):
                    if boundary == native.Boundary.before_commit:
                        raise RuntimeError("Corte de comprobación GRU")

                with pytest.raises(RuntimeError, match="Corte de comprobación"):
                    run.step(source["batches"][index], feedback, fault=fail)
                run = session(resume=True)
            result = run.step(source["batches"][index], feedback)
            points.update({(p.asset, p.decision_at): p.value for p in result.predictions})
        bundle = run._bundle(run.snapshot()["state"])
        bank, episodes = run._bank(bundle["bank"])
        final = run.snapshot()
        view = bank.read_view()
    finally:
        run.close()
    return dict(
        points=points,
        ids=view["ids"].tolist(),
        keys=view["keys"].clone(),
        values=view["values"].clone(),
        labels=view["labels"].clone(),
        seen=bank.seen,
        decisions=sorted((e["flow_id"], e["prediction_at"]) for e in episodes.values()),
        counts={key: final[key] for key in ("observed", "issued", "applied", "cursor")},
    )


def compare(reference, actual, rtol, atol):
    keys = sorted(reference["points"])
    assert keys == sorted(actual["points"])
    expected = np.array([reference["points"][key] for key in keys])
    observed = np.array([actual["points"][key] for key in keys])
    np.testing.assert_allclose(observed, expected, rtol=rtol, atol=atol)
    assert reference["ids"] == actual["ids"] and reference["decisions"] == actual["decisions"]
    for name in ("keys", "values", "labels"):
        assert torch.equal(reference[name], actual[name]), name
    assert reference["counts"] == actual["counts"] and reference["seen"] == actual["seen"]
    return float(np.max(np.abs(observed - expected)))


def test_gru_session_device_parity_and_recovery(native, four_flow_source, tmp_path):
    started = time.perf_counter()
    rehearsal = DEVICE == "cpu"
    hardware = "cpu_rehearsal"
    device = torch.device(DEVICE)
    if not rehearsal:
        hardware = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free,driver_version",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip()
        assert torch.cuda.is_available(), "CUDA es obligatoria, no se admite fallback"
        properties = torch.cuda.get_device_properties(device)
        torch.cuda.set_per_process_memory_fraction(CAP / properties.total_memory, device)
        free, _ = torch.cuda.mem_get_info(device)
        assert free >= CAP, "La memoria CUDA libre no cubre el límite del asignador"
        torch.cuda.reset_peak_memory_stats(device)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
    torch.set_num_threads(2)
    rngs = random.getstate(), np.random.get_state()[1].copy(), torch.random.get_rng_state()
    source = four_flow_source
    assert source["label_origin"] == "manual_fixture_no_estimation"
    cases = []
    for dtype, refinements, admission in (
        (torch.float32, 1, "m1"),
        (torch.float64, 1, "m1"),
        (torch.float64, 4, "m1"),
        (torch.float64, 1, "m0"),
    ):
        rtol, atol = TOLERANCES[dtype]
        cpu, other = pair(source, dtype, refinements)
        before = parameters(cpu[0])
        assert all(torch.equal(before[k], v) for k, v in parameters(other[0]).items())
        assert cpu[1].fingerprint() == other[1].fingerprint()
        name = f"{dtype}-k{refinements}-{admission}"
        paths, seconds = {}, {}
        for label, engine, cut in (
            ("cpu", cpu, False),
            ("device", other, False),
            ("device_recovered", other, True),
        ):
            if not rehearsal:
                torch.cuda.synchronize(device)
            begin = time.perf_counter()
            paths[label] = trajectory(
                native, source, engine, tmp_path / f"{name}-{label}", admission=admission, cut=cut
            )
            if not rehearsal:
                torch.cuda.synchronize(device)
            seconds[label] = time.perf_counter() - begin
        error = compare(paths["cpu"], paths["device"], rtol, atol)
        compare(paths["device"], paths["device_recovered"], 0, 0)
        for engine in (cpu, other):
            engine[0].verify()
            assert all(torch.equal(before[k], v) for k, v in parameters(engine[0]).items())
        assert paths["cpu"]["seen"] == (12 if admission == "m1" else 0)
        cases.append(
            dict(
                case=name,
                rtol=rtol,
                atol=atol,
                max_prediction_error=error,
                seconds=seconds,
                retained_ids=paths["device"]["ids"],
                exact_recovery_on_device=True,
                parameters_unchanged=True,
            )
        )
    assert random.getstate() == rngs[0]
    np.testing.assert_array_equal(np.random.get_state()[1], rngs[1])
    assert torch.equal(rngs[2], torch.random.get_rng_state())
    report = dict(
        device=DEVICE,
        rehearsal_without_cuda=rehearsal,
        hardware=hardware,
        torch=torch.__version__,
        cuda=torch.version.cuda,
        native_sha256=native.binary_sha256,
        cases=cases,
        cap_torch_bytes=None if rehearsal else CAP,
        peak_torch_bytes=None if rehearsal else torch.cuda.max_memory_allocated(device),
        peak_reserved_bytes=None if rehearsal else torch.cuda.max_memory_reserved(device),
        wall_seconds=time.perf_counter() - started,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        capacity=4,
        labels="manual_fixture_no_estimation",
        optimizer_steps=0,
        target_estimation_calls=0,
        scientific_evaluation=False,
    )
    if not rehearsal:
        assert report["peak_torch_bytes"] <= CAP
    destination = os.environ.get("MARS_TITAN_GRU_CHECK_REPORT")
    if destination:
        with open(destination, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
