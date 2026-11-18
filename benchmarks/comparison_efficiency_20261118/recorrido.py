"""Repetición técnica de tramos reales, sin cambios de parámetros ni escrituras en la cola."""

import argparse
import hashlib
import json
import resource
import statistics
import struct
import time
from contextlib import ExitStack, closing
from dataclasses import replace
from itertools import islice
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from entrada import OUT, SEED, STATE, YEAR, background, packed_stream, sources

from mars_titan.data.storage import atomic_json, sha256
from mars_titan.memory.episodic_codec import FrozenEpisodeCodec
from mars_titan.training import campaign_throughput as perf
from mars_titan.training import financial_run as core
from mars_titan.training import mars_titan_run as readers
from mars_titan.training import mars_titan_walk_forward as rw
from mars_titan.training import titans_walk_forward as tw
from mars_titan.training.checkpoints import load_training_state
from mars_titan.training.kernel_policy import apply_kernel_policy
from mars_titan.training.selection import with_rule


def read(path):
    return json.loads(path.read_text())


def parameter_digest(engine):
    h = hashlib.sha256()
    modules = [engine.predictor]
    if isinstance(engine, readers.ReadoutTrainer):
        modules.append(engine.readout)
    for module in modules:
        for name, tensor in sorted(module.named_parameters()):
            h.update(name.encode() + b"\0" + tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class FixedWeights(perf._NoStepOptimizer):
    capture = False

    def step(self):
        super().step()
        if self.capture:
            h = hashlib.sha256()
            for i, x in enumerate(self.parameters()):
                h.update(struct.pack("<I?", i, x.grad is not None))
                if x.grad is not None:
                    h.update(x.grad.detach().cpu().contiguous().numpy().tobytes())
            self.grads.append(h.hexdigest())


class Predictions:
    def __init__(self):
        self.digest = hashlib.sha256()
        self.count = 0

    def append(self, record):
        if record[0] == "prediction":
            flow, at, value = record[2:5]
            self.digest.update(flow.encode() + b"\0" + struct.pack("<qd", int(at), float(value)))
            self.count += 1


def build(model, src, spec, seed=SEED, *, year=YEAR):
    anchor = STATE / "runs" / model / str(year) / f"seed-{seed}"
    report = read(anchor / "run.json")
    req, identity = report["request"], report["identity"]
    assert report["status"] == "completed" and report["final_test_opened"] is False
    assert src["train"].identity == identity["indices"]["train"]
    fit_path = anchor / report["fit"]["path"]
    assert sha256(fit_path) == report["fit"]["sha256"]
    fit = read(fit_path)
    saved = load_training_state(
        anchor / "fit/checkpoints",
        expected_identity=fit["identity"],
        selection="best",
        expected_sha256=fit["best_checkpoint"]["sha256"],
    )
    shared = dict(
        train=src["train"],
        validation=src["validation"],
        output=OUT / "sin-entrenamiento",
        optimizer_factory=FixedWeights,
    )
    if model.startswith("mars_titan_"):
        parent = Path(req["parent"]["path"])
        pr, ph = rw._parent(parent, STATE / "cohorts" / str(year) / "manifest.json", seed)
        assert (
            ph == req["parent"]["run_sha256"]
            and pr["checkpoint"]["sha256"] == req["parent"]["checkpoint_sha256"]
        )
        predictor = rw._frozen_parent(
            pr["identity"]["recipe"], spec, seed, "cuda:0", parent, pr, allow_context=True
        )
        fam = rw._mars_family(req["components"])
        variant = fam.variant(predictor, pr)
        codec = FrozenEpisodeCodec(spec)
        recipe = with_rule(
            rw.case_recipe(identity["recipe"], req["search_case"]), identity["stopping_rule"]
        )
        readout = rw._readout(variant, codec, predictor, recipe, seed)
        extra = {"scalers": rw._anchor_scalers(fit)} if variant.admission == "m3" else {}
        engine = readers.ReadoutTrainer(
            predictor,
            readout,
            recipe,
            admission=variant.admission,
            retention=fam.retention(recipe, variant, **extra),
            native=rw._native(variant.admission),
            codec=codec,
            world=fam.world,
            fold=identity["window"]["id"],
            **shared,
        )
        engine._load_readout(saved)
    else:
        doc = identity["recipe"]
        recipe = with_rule(tw.case_recipe(doc, req.get("search_case")), identity["stopping_rule"])
        predictor, _ = tw._predictor(
            doc,
            spec,
            req["variant"],
            seed,
            "cuda:0",
            req.get("local_control"),
            req.get("predictor_changes"),
        )
        predictor.load_state_dict(saved["model"])
        engine = core.ChronologicalTrainer(predictor, recipe, **shared)
    engine.optimizer.initial = [x.detach().clone() for x in engine.optimizer.parameters()]
    assert engine.identity["recipe"] == fit["identity"]["recipe"]
    return engine, report, anchor


def profile_slice(
    engine,
    spec,
    mode,
    capture,
    *,
    warm=4,
    until=16,
    gate=None,
    trace=None,
    source_range=None,
    rng_seed=SEED,
):
    is_reader = isinstance(engine, readers.ReadoutTrainer)
    mod = readers if is_reader else core
    source = engine.train
    block = engine.recipe.block_rows
    file = OUT / "train-late-1024.arrow"
    if source_range is None:
        meta = read(file.with_suffix(".json"))
        start, count = meta["start"], meta["events"]
    else:
        if mode != "original":
            raise ValueError("El intervalo directo solo admite el lector de la fuente")
        start, count = source_range
    first = source.metadata["groups"][start][0]
    original = source.batched_events
    current = engine._new_pass("train") if is_reader else core._Pass()
    optimizer = engine.optimizer
    optimizer.calls = 0
    optimizer.grads = []
    optimizer.capture = capture
    engine.global_step = 0
    engine.audit = Predictions() if capture else None
    torch.manual_seed(rng_seed)
    wait = 0.0
    calls = 0

    def events(*, start_cursor=0, block_rows):
        assert start_cursor == start and block_rows == block
        if mode == "original":
            seq = original(start_cursor=start, block_rows=block)
        else:
            seq = background(
                packed_stream(source, spec, file, start, count, meta["block"], output_block=block),
                source.dataset.pipeline.prefetch_batches,
            )
        with closing(seq):
            for ev in islice(seq, count):
                # Solo se pueden resolver decisiones emitidas dentro de esta repetición.
                if ev.labels and any(label[1] < first for label in ev.labels):
                    ev = replace(
                        ev, labels=tuple(label for label in ev.labels if label[1] >= first)
                    )
                yield ev

    prepared = mod.prepared

    def timed_prepared(*args):
        nonlocal wait, calls
        seq = prepared(*args)
        with closing(seq):
            while True:
                before = time.perf_counter()
                try:
                    val = next(seq)
                except StopIteration:
                    return
                wait += time.perf_counter() - before
                calls += 1
                yield val

    marks = {}
    phases = {
        name: 0.0
        for name in (
            ("_labels", "_observe", "_update", "_admit")
            if is_reader
            else ("_labels", "_observe", "_update")
        )
    }

    def wall_method(name, fn):
        def timed(*args, **kwargs):
            t = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                phases[name] += time.perf_counter() - t

        return timed

    class Budget:
        @property
        def requested(self):
            i = optimizer.calls
            if i not in marks:
                if i in (warm, until):
                    torch.cuda.synchronize(0)
                if i == warm:
                    if gate is not None:
                        gate()
                    if trace is not None:
                        trace.start()
                    torch.cuda.reset_peak_memory_stats(0)
                marks[i] = (
                    time.perf_counter(),
                    current.counters["observations"],
                    current.counters["labels"],
                    wait,
                    calls,
                    phases.copy(),
                )
                if i == until and trace is not None:
                    trace.finish()
            return i >= until

    budget = Budget()
    before = time.perf_counter()
    with ExitStack() as stack:
        stack.enter_context(patch.object(source, "batched_events", events))
        stack.enter_context(patch.object(mod, "prepared", timed_prepared))
        for name in phases:
            stack.enter_context(
                patch.object(engine, name, wall_method(name, getattr(engine, name)))
            )
        try:
            engine._train_pass(
                current,
                dict(epoch=0, phase="train", event=start, stage="start"),
                budget,
                lambda *args: None,
            )
        except core.Paused:
            pass
        else:
            raise AssertionError("El tramo no llegó a la barrera")
    end = time.perf_counter()
    a, b = marks[warm], marks[until]
    t = b[0] - a[0]
    assert optimizer.calls == until and optimizer.unchanged() and b[1] > a[1]
    result = dict(
        mode=mode,
        with_parity_audit=capture,
        samples=b[1] - a[1],
        labels=b[2] - a[2],
        events=b[4] - a[4],
        seconds=t,
        monotonic_start=a[0],
        monotonic_finish=b[0],
        rows_per_second=(b[1] - a[1]) / t,
        exposed_input_seconds=b[3] - a[3],
        main_thread_phase_seconds={k: b[5][k] - a[5][k] for k in phases},
        start_to_first_barrier_s=marks[min(marks)][0] - before,
        pass_seconds=end - before,
        warm_segments=warm,
        measured_segments=until - warm,
        segment_p50_s=statistics.median(
            marks[i][0] - marks[i - 1][0] for i in range(warm + 1, until + 1)
        ),
        segment_p95_s=float(
            np.quantile([marks[i][0] - marks[i - 1][0] for i in range(warm + 1, until + 1)], 0.95)
        ),
        allocated_bytes=torch.cuda.max_memory_allocated(0),
        reserved_bytes=torch.cuda.max_memory_reserved(0),
        rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    )
    if capture:
        assert engine.audit.count > 0 and len(optimizer.grads) == until
        result["parity"] = dict(
            gradients=optimizer.grads,
            predictions=engine.audit.digest.hexdigest(),
            prediction_count=engine.audit.count,
            counters=dict(current.counters),
        )
    return result


def validation(engine, spec, outer, anchor, mode, expected):
    source = engine.validation
    spec = source.specification()
    block = engine.recipe.block_rows
    file = OUT / f"validation-{block}.arrow"
    meta = read(file.with_suffix(".json"))
    native = source.batched_events

    def packed(*, start_cursor=0, block_rows):
        assert block_rows == block
        return background(
            packed_stream(source, spec, file, 0, meta["events"], block, start_cursor),
            source.dataset.pipeline.prefetch_batches,
        )

    with patch.object(source, "batched_events", native if mode == "original" else packed):
        rows = tw.PredictionRows(engine.quantiles)
        torch.cuda.synchronize(0)
        torch.cuda.reset_peak_memory_stats(0)
        before = time.perf_counter()
        metrics = engine.evaluate(source, rows=rows)
        torch.cuda.synchronize(0)
        seconds = time.perf_counter() - before
        got = pa.concat_tables(rows.finish())
        if not got.equals(expected):
            raise AssertionError("Discrepancia con el Parquet de la campaña")
    return dict(
        mode=mode,
        seconds=seconds,
        rows=got.num_rows,
        metrics=metrics,
        hits=engine.validation_warmup.receipt()["hits"],
        allocated_bytes=torch.cuda.max_memory_allocated(0),
        reserved_bytes=torch.cuda.max_memory_reserved(0),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", choices=("titans_mac", "transformer", "mars_titan_m2"), required=True
    )
    parser.add_argument("--validation", action="store_true")
    parser.add_argument("--copies", action="store_true")
    parser.add_argument("--flags", action="store_true")
    parser.add_argument("--kernels", action="store_true")
    args = parser.parse_args()
    dest = OUT / (args.model + "-cuda.json")
    assert not dest.exists()
    t = time.perf_counter()
    rw._cuda("cuda:0")
    assert torch.cuda.is_available()
    torch.cuda.set_device("cuda:0")
    apply_kernel_policy("fp32_strict")
    guard = perf._forbid_steps()
    try:
        with tw.unfused_attention():
            src = sources()
            spec = src["train"].specification()
            engine, outer, anchor = build(args.model, src, spec)
            initial = parameter_digest(engine)
            result = dict(
                model=args.model,
                year=YEAR,
                seed=SEED,
                device=torch.cuda.get_device_name(0),
                torch=str(torch.__version__),
                setup_s=time.perf_counter() - t,
                block=engine.recipe.block_rows,
                weights_changed=False,
                real_optimizer_steps=0,
                source=src["train"].identity,
                regime="FP32 estricto, tramo tardío de ajuste, sin modificar pesos",
                passes=[],
                validation=[],
            )
            timing = (
                ("original", True),
                ("preparada", True),
                ("original", False),
                ("preparada", False),
                ("preparada", False),
                ("original", False),
            )
            for mode, capture in timing:
                case = profile_slice(engine, spec, mode, capture)
                if len(result["passes"]) == 1:
                    assert case["parity"] == result["passes"][0]["parity"], (
                        "No coinciden emisiones, contadores o gradientes"
                    )
                result["passes"].append(case)
                atomic_json(dest, result)
                print(
                    args.model,
                    mode,
                    capture,
                    case["seconds"],
                    case["exposed_input_seconds"],
                    case["samples"],
                    flush=True,
                )
            if args.flags:
                from comprobaciones import edge_cases, host_finite

                from mars_titan.models.titans import state

                result["flag_edge_cases"] = edge_cases()
                result["flag_trials"] = []
                for alternative, capture in (
                    (True, True),
                    (False, False),
                    (True, False),
                    (True, False),
                    (False, False),
                ):
                    with ExitStack() as stack:
                        if alternative:
                            stack.enter_context(
                                patch.object(state._PendingChecks, "first_failure", host_finite)
                            )
                        case = profile_slice(
                            engine,
                            spec,
                            "preparada",
                            capture,
                            warm=4 if capture else 2,
                            until=16 if capture else 10,
                        )
                    if capture:
                        assert case["parity"] == result["passes"][0]["parity"]
                    case.update(finite_scalars_on_host=alternative)
                    result["flag_trials"].append(case)
                    atomic_json(dest, result)
                    print("indicadores", alternative, capture, case["seconds"], flush=True)
            if args.copies:
                from transferencias import grouped

                from mars_titan.models.titans.financial_inputs import DecisionBatch

                result["copy_trials"] = []
                for summed, capture in (
                    (True, True),
                    (False, False),
                    (True, False),
                    (True, False),
                    (False, False),
                ):
                    with ExitStack() as stack:
                        if summed:
                            stack.enter_context(
                                patch.object(DecisionBatch, "from_validated", classmethod(grouped))
                            )
                        # La comprobación de paridad abarca los mismos 16 tramos que la referencia.
                        case = profile_slice(
                            engine,
                            spec,
                            "preparada",
                            capture,
                            warm=4 if capture else 2,
                            until=16 if capture else 10,
                        )
                    if capture:
                        assert case["parity"] == result["passes"][0]["parity"], (
                            "Las transferencias agrupadas cambiaron el cálculo"
                        )
                    case.update(grouped_transfers=summed)
                    result["copy_trials"].append(case)
                    atomic_json(dest, result)
                    print("transferencias", summed, capture, case["seconds"], flush=True)
            if args.kernels:

                class Trace:
                    def start(self):
                        self.prof = torch.profiler.profile(
                            activities=[
                                torch.profiler.ProfilerActivity.CPU,
                                torch.profiler.ProfilerActivity.CUDA,
                            ],
                            record_shapes=False,
                            with_stack=False,
                            profile_memory=False,
                        )
                        self.prof.start()
                        self.range = torch.profiler.record_function("auditoria.tramo_cuda")
                        self.range.__enter__()

                    def finish(self):
                        self.range.__exit__(None, None, None)
                        self.prof.stop()
                        self.prof.export_chrome_trace(str(OUT / (args.model + "-kernels.json")))

                try:
                    case = profile_slice(
                        engine, spec, "preparada", False, warm=2, until=3, trace=Trace()
                    )
                    result["profile_trial"] = case
                    print("perfil", args.model, case["samples"], flush=True)
                except RuntimeError as err:
                    result["profile_error"] = str(err)
                    print("perfil no disponible", err, flush=True)
                atomic_json(dest, result)
            if args.validation:
                engine.audit = None
                table = anchor / outer["predictions"]["validation"]["path"]
                assert sha256(table) == outer["predictions"]["validation"]["sha256"]
                expected = pq.read_table(table)
                for mode in ("preparada", "original", "preparada", "preparada", "original"):
                    case = validation(engine, spec, outer, anchor, mode, expected)
                    if result["validation"]:
                        assert case["metrics"] == result["validation"][0]["metrics"]
                    result["validation"].append(case)
                    atomic_json(dest, result)
                    print("validación", mode, case["seconds"], case["hits"], flush=True)
            assert parameter_digest(engine) == initial
            result.update(
                elapsed_s=time.perf_counter() - t, complete=True, parameters_sha256=initial
            )
            atomic_json(dest, result)
    finally:
        guard.remove()


if __name__ == "__main__":
    main()
