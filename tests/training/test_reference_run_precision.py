"""Precisión declarada por un caso de referencia: validación, flags e identidad.

La ejecución se detiene al escribir el primer `run.json`, antes de leer lotes de ajuste o
de llamar al optimizador. Sin el campo la identidad no cambia.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mars_titan.data.storage import atomic_json
from mars_titan.training import kernel_policy as policy
from mars_titan.training import reference_run

CASE = dict(
    kind="gru",
    loss="mae",
    learning_rate=1e-3,
    seed=7,
    epochs=1,
    huber_delta=0.01,
    architecture=dict(hidden_size=32, layers=1, dropout=0.0),
)
DIMENSIONS = dict(prices=5, news=4, charts=3, fundamentals=2, macro=6)


class Reached(Exception):
    """El recorrido llegó al primer informe sin ajustar nada."""


@pytest.fixture(autouse=True)
def restore_numerics(monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    flags = (
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
        torch.get_float32_matmul_precision(),
        torch.are_deterministic_algorithms_enabled(),
        torch.backends.cudnn.benchmark,
    )
    yield
    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = flags[:2]
    torch.set_float32_matmul_precision(flags[2])
    torch.use_deterministic_algorithms(flags[3])
    torch.backends.cudnn.benchmark = flags[4]


def first_report(tmp_path, monkeypatch, case, *, batch_size=16, write=False, resume=False):
    """Recorrer `run_reference_case` en CPU hasta su primer informe y devolverlo.

    Con `write`, el informe se escribe antes de detenerse. Con `resume`, el recorrido se
    detiene justo después de aceptar el informe escrito, si su identidad coincide.
    """
    batch = dict(
        inputs={
            name: np.zeros((2, 8, size) if name == "prices" else (2, size), np.float32)
            for name, size in DIMENSIONS.items()
        }
    )
    dataset = SimpleNamespace(
        manifest=dict(counts=dict(train=2, validation=2), scope="test", cohort_complete=True),
        context=8,
        roots={},
        identity="0" * 64,
        assets=[dict(market="US", counts=dict(train=2))],
        cache_limit=0,
        cache_sample_tables=False,
        cache_entry_limit=0,
        cached_bytes=0,
        batches=lambda **_: iter([batch]),
    )

    def stop(path, report):
        if write:
            atomic_json(path, report)
        raise Reached(report)

    def resumed(*_):
        raise Reached("reanudada")

    if resume:
        monkeypatch.setattr(reference_run, "bind_joint_epoch", resumed)
    monkeypatch.setattr(reference_run, "require_learning_allowed", lambda _: None)
    monkeypatch.setattr(reference_run, "require_cuda", lambda: torch.device("cpu"))
    monkeypatch.setattr(reference_run, "configured_corpus", lambda *_, **__: dataset)
    monkeypatch.setattr(reference_run, "atomic_json", stop)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *_: "prueba")
    monkeypatch.setattr(torch.optim.AdamW, "step", lambda *_: pytest.fail("paso del optimizador"))
    with pytest.raises(Reached) as reached:
        reference_run.run_reference_case(
            tmp_path / "manifest.json", tmp_path / "run", case, batch_size=batch_size, resume=resume
        )
    value = reached.value.args[0]
    return value if resume else value["identity"]


def test_case_without_precision_keeps_the_process_numerics_and_identity(tmp_path, monkeypatch):
    torch.backends.cudnn.allow_tf32 = True
    identity = first_report(tmp_path, monkeypatch, dict(CASE))
    assert "kernel_policy" not in identity
    assert "precision" not in identity["case"]
    assert identity["precision"] == "float32"
    assert identity["numerics"]["cudnn_allow_tf32"] is True
    assert torch.backends.cudnn.allow_tf32 is True


def test_declared_precision_applies_the_policy_before_recording_numerics(tmp_path, monkeypatch):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    case = dict(CASE, precision=policy.FP32_STRICT)
    identity = first_report(tmp_path, monkeypatch, case)
    assert identity["kernel_policy"] == policy.kernel_policy_identity(policy.FP32_STRICT)
    assert identity["case"]["precision"] == "fp32_strict"
    assert identity["numerics"]["cudnn_allow_tf32"] is False
    assert identity["numerics"]["cuda_matmul_allow_tf32"] is False
    assert identity["numerics"]["float32_matmul_precision"] == "highest"
    without = first_report(tmp_path / "plain", monkeypatch, dict(CASE))
    assert {key for key in identity if identity[key] != without.get(key)} == {
        "case",
        "kernel_policy",
    }


@pytest.mark.parametrize("value", [None, "fp32", "bf16", "FP32_STRICT", 1, True])
def test_invalid_precision_is_rejected_before_touching_the_process(value):
    torch.backends.cudnn.allow_tf32 = True
    with pytest.raises(ValueError, match="precisión"):
        reference_run._options(dict(CASE, precision=value), 256, 60, 0)
    assert torch.backends.cudnn.allow_tf32 is True
    reference_run._options(dict(CASE, precision=policy.FP32_STRICT), 256, 60, 0)


def test_continuation_requires_the_same_declared_precision(tmp_path, monkeypatch):
    parent = tmp_path / "parent"
    parent.mkdir()
    identity = dict(
        manifest_sha256="0" * 64,
        case=dict(CASE),
        batch_size=256,
        weighting="natural",
    )
    monkeypatch.setattr(
        reference_run, "read_json", lambda _: dict(status="completed", identity=identity)
    )
    dataset = SimpleNamespace(identity="0" * 64)
    strict = dict(CASE, precision=policy.FP32_STRICT)
    with pytest.raises(ValueError, match="origen"):
        reference_run._parent(parent, dataset, strict, None, 256, "natural")
    # El mismo caso sin precisión supera esa comparación y solo falla al buscar el checkpoint.
    with pytest.raises(KeyError, match="checkpoint"):
        reference_run._parent(parent, dataset, dict(CASE), None, 256, "natural")


def test_a_numeric_change_after_starting_is_caught_before_saving(tmp_path, monkeypatch):
    first_report(tmp_path / "first", monkeypatch, dict(CASE, precision=policy.FP32_STRICT))

    def change_after_report(path, report):
        # Otro código del proceso vuelve a permitir TF32 tras registrar la identidad.
        torch.backends.cudnn.allow_tf32 = True

    def saved(*_, **__):
        raise Reached("guardado")

    monkeypatch.setattr(reference_run, "atomic_json", change_after_report)
    monkeypatch.setattr(reference_run, "save_training_state", saved)
    for name in ("reset_peak_memory_stats", "max_memory_allocated", "max_memory_reserved"):
        monkeypatch.setattr(torch.cuda, name, lambda *_: 0)
    with pytest.raises(ValueError, match="política de precisión"):
        reference_run.run_reference_case(
            tmp_path / "manifest.json",
            tmp_path / "run",
            dict(CASE, precision=policy.FP32_STRICT),
        )


@pytest.mark.parametrize(
    ("case", "batch_size"),
    [
        (dict(CASE, precision=policy.FP32_STRICT), 32),
        (dict(CASE), 16),
        (dict(CASE, precision=policy.FP32_STRICT, cuda_graphs=True), 16),
    ],
    ids=["other_batch", "without_precision", "with_graphs"],
)
def test_resume_keeps_the_batch_precision_and_graph_step_it_started_with(
    tmp_path, monkeypatch, case, batch_size
):
    declared = dict(CASE, precision=policy.FP32_STRICT)
    first_report(tmp_path, monkeypatch, declared, write=True)
    assert (tmp_path / "run/run.json").is_file()
    # Con las mismas opciones, la reanudación acepta el informe y sigue.
    assert first_report(tmp_path, monkeypatch, declared, resume=True) == "reanudada"
    # Cualquier opción distinta cambia la identidad y la reanudación se rechaza.
    with pytest.raises(ValueError, match="identidad o configuración"):
        first_report(tmp_path, monkeypatch, case, batch_size=batch_size, resume=True)
