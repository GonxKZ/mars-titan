"""Conversión de las trazas del registrador en paquetes del observatorio.

El registrador recibe tensores escritos en la prueba en CPU. No hay pasadas hacia atrás
ni optimizador: solo se comprueba que el formato largo llega al paquete binario sin
perder pasos, ausencias, cadencia ni el corte por presupuesto.
"""

import json
import math
import os
import struct
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from mars_titan.learning_traces.recorder import TraceConfig, TraceRecorder
from mars_titan.observatory.traces import series_from_learning_traces, write_trace_bundle

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "export_learning_traces.py"
IDENTITY = dict(arm="fixture", seed=42, window="fold-000")


def convert(rows):
    return series_from_learning_traces(*map(list, zip(*rows, strict=True)))


def test_long_rows_become_one_series_per_metric_group_and_statistic():
    series = convert(
        [
            (4, "train", "grad_l2", "memory", "value", 0.5),
            (0, "train", "grad_l2", "memory", "value", 0.25),
            (0, "train", "grad_l2", "head", "value", 1.0),
            (0, "train", "surprise", "memory", "mean", 0.1),
            (0, "train", "surprise", "memory", "finite_fraction", 1.0),
            (0, "train", "policy.entropy", "all", "value", 0.7),
            (0, "train", "loss", "all", "value", 0.02),
        ]
    )
    found = {item["id"]: item for item in series}
    gradient = found["grad_l2.memory.value.train"]
    # Las normas por grupo de parámetros se quedan juntas en optimización.
    assert gradient["group"] == "optimization" and found["grad_l2.head.value.train"]
    assert gradient["x"] == [0, 4] and gradient["y"] == [0.25, 0.5]
    assert gradient["label"] == "Norma L2 del gradiente · memory"
    assert gradient["unit"] == "norma L2"
    assert found["surprise.memory.mean.train"]["group"] == "titans"
    assert found["surprise.memory.finite_fraction.train"]["unit"] == "fracción"
    assert found["policy.entropy.all.value.train"]["group"] == "rl"
    assert found["loss.all.value.train"]["group"] == "optimization"


def test_absences_are_kept_and_repeated_steps_keep_the_later_row():
    series = convert(
        [
            (0, "train", "loss", "all", "value", 1.0),
            (2, "train", "loss", "all", "value", math.nan),
            (0, "train", "loss", "all", "value", 0.75),
        ]
    )
    assert series[0]["x"] == [0, 2]
    assert series[0]["y"][0] == 0.75 and math.isnan(series[0]["y"][1])


def test_phases_are_named_only_when_there_is_more_than_one():
    series = convert(
        [
            (0, "train", "loss", "all", "value", 1.0),
            (0, "validation", "loss", "all", "value", 2.0),
        ]
    )
    assert [item["label"] for item in series] == ["loss · all · train", "loss · all · validation"]


def test_identifiers_stay_valid_and_unique_after_normalisation():
    series = convert(
        [
            (0, "train", "a:b", "all", "value", 1.0),
            (0, "train", "a/b", "all", "value", 2.0),
        ]
    )
    assert [item["id"] for item in series] == ["a_b.all.value.train", "a_b.all.value.train.2"]


@pytest.mark.parametrize(
    "rows",
    [
        [(-1, "train", "loss", "all", "value", 1.0)],
        [(1.5, "train", "loss", "all", "value", 1.0)],
    ],
)
def test_invalid_steps_are_rejected(rows):
    with pytest.raises(ValueError):
        convert(rows)


def test_columns_of_different_length_are_rejected():
    with pytest.raises(ValueError):
        series_from_learning_traces([0], ["train"], ["loss"], ["all"], ["value"], [])


@pytest.mark.parametrize(("cadence", "truncated_at"), [(0, None), (2.0, None), (None, -1)])
def test_bundles_reject_invalid_cadence_or_cut(tmp_path, cadence, truncated_at):
    with pytest.raises(ValueError):
        write_trace_bundle(
            tmp_path,
            "bad",
            run_id="r",
            attempt_id="a",
            model_id="m",
            provenance="fixture",
            x_unit="optimizer_step",
            series=[{"id": "a", "group": "titans", "label": "x", "x": [1], "y": [0]}],
            cadence=cadence,
            truncated_at=truncated_at,
        )


def record(folder, steps, max_bytes=1 << 20):
    recorder = TraceRecorder(folder, TraceConfig(every=2, max_bytes=max_bytes), IDENTITY)
    for step in steps:
        recorder.tensor("activation", torch.arange(4.0) * (step + 1), group="memory")
        recorder.scalar("loss", 1.0 / (step + 1))
        recorder.flush(step, "train")
    return recorder


def export(source, output):
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--traces",
            str(source),
            "--output",
            str(output),
            "--name",
            "fixture-traces",
            "--run-id",
            "fixture-run",
            "--attempt-id",
            "attempt-0001",
            "--model-id",
            "fixture",
            "--provenance",
            "fixture",
        ],
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
        # El subproceso importa el paquete de este árbol aunque el entorno esté
        # instalado desde otra copia del repositorio.
        env=os.environ | {"CUDA_VISIBLE_DEVICES": "-1", "PYTHONPATH": str(ROOT / "src")},
    )


def test_cli_converts_recorder_parts_into_a_bundle(tmp_path):
    source = tmp_path / "traces"
    record(source, [0, 2, 4])
    before = sorted(path.name for path in (source / "parts").iterdir())
    result = export(source, tmp_path / "bundles")
    assert result.returncode == 0, result.stderr
    manifest = json.loads((tmp_path / "bundles" / "fixture-traces.json").read_text())
    assert manifest["cadence"] == 2 and manifest["truncated_at"] is None
    assert manifest["provenance"] == "fixture" and manifest["x_unit"] == "optimizer_step"
    # Siete estadísticos de la activación y el escalar de la pérdida.
    assert len(manifest["series"]) == 8
    loss = next(item for item in manifest["series"] if item["id"] == "loss.all.value.train")
    blob = (tmp_path / "bundles" / manifest["blob"]).read_bytes()
    x = struct.unpack_from("<3d", blob, loss["x"]["offset"])
    y = struct.unpack_from("<3f", blob, loss["y"]["offset"])
    assert x == (0.0, 2.0, 4.0)
    assert y == pytest.approx((1.0, 1 / 3, 1 / 5), rel=1e-7)
    assert sorted(path.name for path in (source / "parts").iterdir()) == before


def test_cli_carries_the_step_where_the_recorder_ran_out_of_budget(tmp_path):
    source = tmp_path / "traces"
    probe = record(tmp_path / "probe", [0])
    part = probe.bytes
    recorder = record(source, [0, 2, 4], max_bytes=part + part // 2)
    assert recorder.exhausted_at == 2
    result = export(source, tmp_path / "bundles")
    assert result.returncode == 0, result.stderr
    manifest = json.loads((tmp_path / "bundles" / "fixture-traces.json").read_text())
    assert manifest["truncated_at"] == 2
    assert all(item["x"]["length"] == 1 for item in manifest["series"])
