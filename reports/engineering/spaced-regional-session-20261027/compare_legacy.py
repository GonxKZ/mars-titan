"""Contrastar la sesión sin E2 con el enlace anterior sobre los mismos fixtures."""

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import torch
from torch.optim.optimizer import register_optimizer_step_pre_hook

import mars_titan
from mars_titan.memory import financial_session
from mars_titan.memory.native_backend import load_native

ROOT = Path(__file__).resolve().parents[3]
BASE = "babe9f780fcf73ed4e54ba92bddb0d2194fefff0"
SOURCE = "src/mars_titan/memory/spaced_session.py"
sys.path[:0] = [str(ROOT / "tests/memory"), str(ROOT / "tests/models/titans")]
from test_financial_session_controls import (  # noqa: E402
    flow_source,
    no_target_estimation,
    paired_consumers,
)
from test_spaced_session import feedback, open_session  # noqa: E402


def run(native, source, consumer, output):
    points = []
    with open_session(native, source, consumer, output, capacity=4) as session:
        previous = []
        for at in (125, 126, 127, 128):
            event = session.step(source["batches"][at], feedback(native, previous, at))
            previous = event.predictions
            points.extend((row.asset, row.decision_at, row.value) for row in previous)
        return dict(
            contract_id=session.contract_id,
            snapshot=session.snapshot(),
            points=points,
            retained=session.retained_episodes(),
        )


def main():
    if not Path(mars_titan.__file__).resolve().is_relative_to(ROOT / "src"):
        raise RuntimeError("La comparación necesita las fuentes propias con PYTHONPATH=src:.")
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    assert not torch.cuda.is_initialized()
    old = subprocess.check_output(["git", "-C", str(ROOT), "show", f"{BASE}:{SOURCE}"])
    path = out / "legacy_spaced_session.py"
    path.write_bytes(old)
    name = "mars_titan.memory._legacy_spaced_verification"
    spec = importlib.util.spec_from_file_location(name, path)
    legacy = importlib.util.module_from_spec(spec)
    sys.modules[name] = legacy
    spec.loader.exec_module(legacy)
    native = load_native()
    guard = no_target_estimation.__wrapped__()
    next(guard)
    attempts = 0

    def forbidden(*_):
        nonlocal attempts
        attempts += 1
        raise RuntimeError("La comparación no admite pasos de optimizador")

    hook = register_optimizer_step_pre_hook(forbidden)
    attention = torch.backends.mha.get_fastpath_enabled()
    original = financial_session.spaced_session
    try:
        torch.backends.mha.set_fastpath_enabled(False)
        source = flow_source(out / "inputs")
        consumer = paired_consumers(source)["disabled"]
        current = run(native, source, consumer, out / "current")
        financial_session.spaced_session = legacy
        previous = run(native, source, consumer, out / "legacy")
        assert current == previous
        result = dict(
            kind="technical_legacy_non_e2_session_parity",
            base_commit=BASE,
            legacy_source_sha256=hashlib.sha256(old).hexdigest(),
            current_source_sha256=hashlib.sha256((ROOT / SOURCE).read_bytes()).hexdigest(),
            probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            native_sha256=native.binary_sha256,
            torch=str(torch.__version__),
            device="cpu",
            cpu_threads=2,
            contract_id_equal=True,
            complete_snapshot_equal=True,
            retained_episodes_equal=True,
            emitted_points=len(current["points"]),
            emitted_sha256=hashlib.sha256(
                json.dumps(current["points"], sort_keys=True).encode()
            ).hexdigest(),
            optimizer_attempts=attempts,
            cuda_initialized=torch.cuda.is_initialized(),
            scope="Enlace del coordinador sin E2 sobre las mismas fuentes y parámetros.",
            limits=[
                "No compara contratos E2 antiguos. Estos conservan su runtime fijado.",
                "Solo fixtures CPU y etiquetas manuales. No es una evaluación predictiva.",
            ],
        )
        (out / "parity.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(result, ensure_ascii=False))
    finally:
        financial_session.spaced_session = original
        torch.backends.mha.set_fastpath_enabled(attention)
        guard.close()
        hook.remove()
        sys.modules.pop(name, None)


if __name__ == "__main__":
    main()
