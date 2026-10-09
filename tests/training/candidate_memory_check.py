"""Memoria de activaciones del entrenador de la GRU candidata, sin pasos de optimizador.

Se ejecuta de forma explícita sobre el corpus técnico sintético con las dimensiones reales de
entrada (precios 64×5, noticias 384, gráficos 512, 15 conceptos fundamentales y 140 indicadores
macro) y la receta propuesta en FP32. Compara el backward único del tramo con la acumulación
por bloques y la recomputación de un bloque con `torch.utils.checkpoint`.

Variables: `MARS_TITAN_CANDIDATE_MEMORY_CHECK_DEVICE` (cpu por defecto o cuda:0),
`MARS_TITAN_CANDIDATE_MEMORY_CHECK_ASSETS` (activos por instante, 128,256 por defecto) y
`MARS_TITAN_CANDIDATE_MEMORY_CHECK_REPORT` (JSON acumulativo).
"""

import json
import os
import statistics
import time
from pathlib import Path

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from mars_titan.data.input_policy import HISTORICAL_MASKED, MODALITIES
from mars_titan.memory import financial_observations as api
from mars_titan.models.titans.financial_inputs import DecisionBatch
from mars_titan.training.corpus_inputs import CorpusDataset
from tests.training.chronological_fixture import chronological_corpus, phases
from tests.training.test_candidate_accumulation import SavedActivations, first_block
from tests.training.test_candidate_run import named_records, trainer

DEVICE = os.environ.get("MARS_TITAN_CANDIDATE_MEMORY_CHECK_DEVICE", "cpu")
ASSETS = [
    int(value)
    for value in os.environ.get("MARS_TITAN_CANDIDATE_MEMORY_CHECK_ASSETS", "128,256").split(",")
]
WIDTHS = (384, 512, 15, 140)
RECIPE = dict(update_instants=8, block_rows=128, bank_capacity=1024)
pytestmark = [
    pytest.mark.skipif(
        not os.environ.get("MARS_TITAN_EPISODIC_NATIVE")
        or (DEVICE == "cuda:0" and not torch.cuda.is_available()),
        reason="Faltan el enlace nativo o cuda:0",
    ),
    pytest.mark.usefixtures("learning_doubles"),
]


def report(entry):
    destination = os.environ.get("MARS_TITAN_CANDIDATE_MEMORY_CHECK_REPORT")
    if destination:
        path = Path(destination)
        records = json.loads(path.read_text()) if path.exists() else []
        records.append(dict(entry, device=DEVICE, torch=torch.__version__))
        path.write_text(json.dumps(records, indent=2))


def streams(root, assets):
    manifest = chronological_corpus(
        root / "corpus", assets=assets, last="2023-01-31", widths=WIDTHS
    )
    dataset = CorpusDataset(manifest, input_policy=HISTORICAL_MASKED)
    result = {}
    for phase in phases():
        index = api.prepare_observation_index(
            dataset, root / f"index-{phase.partition}", phase=phase
        )
        result[phase.partition] = api.FinancialObservationSource(dataset, index)
    return result


def model(sources):
    from mars_titan.models.candidate.input_adapter import CandidateInputAdapter

    specification = sources["train"].specification()
    cpu = CandidateInputAdapter(specification, dtype=torch.float32, parameter_seed=42)
    if DEVICE == "cpu":
        return cpu
    return CandidateInputAdapter.restore(cpu.export_state(), specification, device=DEVICE)


CONFIGURATIONS = (
    dict(accumulation_rows=None, recompute=False),
    dict(accumulation_rows=128, recompute=False),
    dict(accumulation_rows=1024, recompute=False),
    dict(accumulation_rows=None, recompute=True),
    dict(accumulation_rows=128, recompute=True),
)


def allocated_peak(prof):
    """Pico neto de memoria del asignador de ATen en CPU durante la ventana perfilada."""
    events = sorted(
        (event.start_ns(), event.nbytes())
        for event in prof.profiler.kineto_results.events()
        if event.name() == "[memory]" and event.device_type() == torch.autograd.DeviceType.CPU
    )
    current = peak = 0
    for _, delta in events:
        current += delta
        peak = max(peak, current)
    return peak


def measured_run(sources, root, options, *, profiled):
    engine = trainer(sources, root, model=model(sources), **RECIPE, **options)
    tracker = SavedActivations(engine.model.named_parameters().values())
    cuda = DEVICE.startswith("cuda")
    if cuda:
        torch.cuda.synchronize(0)
        torch.cuda.reset_peak_memory_stats(0)
        base = torch.cuda.memory_allocated(0)
    activities = [torch.profiler.ProfilerActivity.CPU]
    start = time.perf_counter()
    if profiled and not cuda:
        with torch.profiler.profile(activities=activities, profile_memory=True) as prof, tracker:
            engine.run()
        peak = allocated_peak(prof)
    else:
        with tracker:
            engine.run()
        if cuda:
            torch.cuda.synchronize(0)
        peak = torch.cuda.max_memory_allocated(0) - base if cuda else None
    return engine, tracker, peak, time.perf_counter() - start


@pytest.mark.parametrize("assets", ASSETS)
def test_peak_memory_by_accumulation_and_recompute(tmp_path, assets):
    sources = streams(tmp_path, assets)
    reference, peaks = None, {}
    for index, options in enumerate(CONFIGURATIONS):
        # Una pasada sin perfilador para el tiempo y otra con él para el pico en CPU.
        engine, tracker, _, seconds = measured_run(
            sources, tmp_path / f"time-{index}", options, profiled=False
        )
        _, _, peak, _ = measured_run(sources, tmp_path / f"memory-{index}", options, profiled=True)
        records = named_records(engine)
        reference = reference or records
        difference = max(
            float((right[name].float() - value.float()).abs().max())
            for left, right in zip(reference, records, strict=True)
            for name, value in left.items()
            if value is not None
        )
        scale = max(
            float(value.abs().max())
            for left in reference
            for value in left.values()
            if value is not None
        )
        train = engine.history[1]["train"]
        peaks[index] = peak
        report(
            dict(
                check="peak_memory",
                assets=assets,
                **options,
                update_instants=RECIPE["update_instants"],
                block_rows=RECIPE["block_rows"],
                dtype="float32",
                peak_allocated_bytes=peak,
                peak_allocated_bytes_per_asset=peak / assets,
                peak_saved_bytes_outside_recompute=tracker.peak,
                parameter_bytes=sum(
                    v.numel() * v.element_size() for v in engine.model.named_parameters().values()
                ),
                seconds=seconds,
                updates=train["updates"],
                labels_in_loss=train["labels_in_loss"],
                labels_without_graph=train["labels_without_graph"],
                predictions=train["predictions"],
                max_abs_gradient_difference=difference,
                max_abs_gradient=scale,
                gradients_bitwise_equal=all(
                    (value is None and right[name] is None) or torch.equal(value, right[name])
                    for left, right in zip(reference, records, strict=True)
                    for name, value in left.items()
                ),
            )
        )
        assert tracker.current == 0 and train["updates"] >= 2
        assert difference <= 1e-6 * scale
    assert peaks[3] < peaks[0] and peaks[4] < peaks[0]


def timed(function, repeats=5):
    function()
    if DEVICE.startswith("cuda"):
        torch.cuda.synchronize(0)
    seconds = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        if DEVICE.startswith("cuda"):
            torch.cuda.synchronize(0)
        seconds.append(time.perf_counter() - start)
    return statistics.median(seconds)


def test_recompute_cost_on_one_block(tmp_path):
    sources = streams(tmp_path, RECIPE["block_rows"])
    engine = trainer(sources, tmp_path / "run", model=model(sources), **RECIPE)
    engine.model.train()
    _, batch = first_block(engine, sources["train"], RECIPE["block_rows"])
    tensors = DecisionBatch.from_validated(batch, device=engine.device, dtype=engine.dtype)
    inputs = engine.native.CandidateInputs(
        *(tensors.inputs[name] for name in MODALITIES), tensors.presence
    )
    encoded = engine.codec.encode(batch)
    memory = engine.model.snapshot(
        torch.from_numpy(encoded.keys.copy()).to(engine.device),
        torch.from_numpy(encoded.values.copy()).to(engine.device),
        torch.linspace(-0.02, 0.02, len(encoded.keys), dtype=engine.dtype, device=engine.device),
        torch.arange(1, len(encoded.keys) + 1),
        engine.model.representation_id(),
    )
    target = torch.zeros(len(batch.flow_ids), dtype=engine.dtype, device=engine.device)

    def forward():
        return engine.model.forward(inputs, memory, 1).quantiles

    def step(recompute):
        engine.optimizer.zero_grad(set_to_none=True)
        quantiles = checkpoint(forward, use_reentrant=False) if recompute else forward()
        engine._loss(quantiles, target, reduction="sum").backward()

    measured = {}
    for recompute in (False, True):
        tracker = SavedActivations(engine.model.named_parameters().values())
        with tracker:
            quantiles = checkpoint(forward, use_reentrant=False) if recompute else forward()
        held = tracker.current
        del quantiles
        step(recompute)
        gradients = [None if v.grad is None else v.grad.clone() for v in engine.trainable]
        measured[recompute] = dict(
            seconds=timed(lambda recompute=recompute: step(recompute)),
            held_bytes=held,
            gradients=gradients,
        )
    with torch.no_grad():
        frozen = timed(forward)
    exact = all(
        (left is None and right is None) or torch.equal(left, right)
        for left, right in zip(
            measured[False]["gradients"], measured[True]["gradients"], strict=True
        )
    )
    inputs_bytes = sum(
        t.numel() * t.element_size() for t in (*tensors.inputs.values(), tensors.presence)
    )
    rows = len(batch.flow_ids)
    report(
        dict(
            check="recompute_one_block",
            rows=rows,
            dtype="float32",
            seconds_forward_backward=measured[False]["seconds"],
            seconds_checkpoint_forward_backward=measured[True]["seconds"],
            seconds_forward_without_graph=frozen,
            held_saved_bytes_per_row=measured[False]["held_bytes"] / rows,
            held_saved_bytes_per_row_with_checkpoint=measured[True]["held_bytes"] / rows,
            input_bytes_per_row=inputs_bytes / rows,
            gradients_bitwise_equal=exact,
        )
    )
    assert exact
