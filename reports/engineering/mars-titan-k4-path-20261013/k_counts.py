"""Trabajo contado del lector M1 con K = 1, 2 y 4 en la fixture técnica, sin cronometrar.

Uso: k_counts.py ÁRBOL SALIDA.json DISPOSITIVO TRABAJO

Para cada K recorre `ReadoutTrainer._train_pass` de M1 con `RecordingOptimizer` y cuenta
llamadas por función (cProfile, solo ncalls), lanzamientos de núcleos, copias y esperas del
perfilador de PyTorch y sincronizaciones de `set_sync_debug_mode`. La diferencia entre K = 4
y K = 1 es el trabajo de tres refinamientos más.
"""

import cProfile
import json
import os
import pstats
import shutil
import sys
import warnings
from pathlib import Path

TREE, OUT, DEVICE, WORK = sys.argv[1:5]
sys.path[:0] = [TREE, f"{TREE}/src"]

import torch  # noqa: E402
from torch.profiler import ProfilerActivity, profile  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("highest")
torch.backends.mha.set_fastpath_enabled(False)

from mars_titan.memory.episodic_codec import FrozenEpisodeCodec  # noqa: E402
from mars_titan.memory.native_backend import load_native  # noqa: E402
from mars_titan.models.titans.episodic_readout import (  # noqa: E402
    EpisodicReadout,
    EpisodicReadoutConfig,
)
from mars_titan.models.titans.financial import FinancialConfig, FinancialPredictor  # noqa: E402
from mars_titan.training import campaign_throughput as ct  # noqa: E402
from mars_titan.training import mars_titan_run as mt  # noqa: E402
from tests.training import test_financial_run as tfr  # noqa: E402
from tests.training import test_mars_titan_run as tmt  # noqa: E402

cuda = DEVICE.startswith("cuda")
work = Path(WORK)
if work.exists():
    shutil.rmtree(work)
work.mkdir(parents=True)
_, streams = tfr.corpus(work / "corpus")
native = load_native(os.environ["MARS_TITAN_EPISODIC_NATIVE"])
guard = ct._forbid_steps()
specification = streams["train"].specification()


class Never:
    requested = False


results = {}
for refinements in (1, 2, 4):
    config = FinancialConfig(
        specification, variant="mac_online", hidden_size=32, seed=42, head="quantile_head_v1"
    )
    parent = FinancialPredictor(config, dtype=torch.float32).to(DEVICE)
    parent = parent.eval().requires_grad_(False)
    codec = FrozenEpisodeCodec(specification)
    readout = EpisodicReadout(
        EpisodicReadoutConfig(
            codec.fingerprint(),
            hidden_size=32,
            refinements=refinements,
            mode="bank",
            max_working_bytes=128 * 1024**2,
            seed=42,
            episode_selection="per_step",
        ),
        dtype=torch.float32,
    ).to(DEVICE)
    plan = tmt.recipe()
    engine = mt.ReadoutTrainer(
        parent,
        readout,
        plan,
        admission="m1",
        retention=mt.retention_config(plan, "m1"),
        native=native,
        codec=codec,
        train=streams["train"],
        validation=streams["validation"],
        output=work / f"k{refinements}",
        world="fixture",
        fold="0",
        optimizer_factory=tfr.RecordingOptimizer,
        audit=True,
    )
    run = engine._new_pass("train")
    profiler = cProfile.Profile()
    if cuda:
        torch.cuda.synchronize(0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if cuda:
            torch.cuda.set_sync_debug_mode("warn")
        activities = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if cuda else [])
        with profile(activities=activities) as prof:
            profiler.enable()
            engine._train_pass(
                run, dict(epoch=0, phase="train", event=0, stage="start"), Never(), ct._no_save
            )
            profiler.disable()
            if cuda:
                torch.cuda.synchronize(0)
        if cuda:
            torch.cuda.set_sync_debug_mode(0)
    runtime = {item.key: item.count for item in prof.key_averages()}
    stats = pstats.Stats(profiler)
    calls = {}
    for (filename, line, name), (_, ncalls, *_rest) in stats.stats.items():
        if "mars_titan" in filename:
            short = filename.split("src/mars_titan/")[-1]
            calls[f"{short}:{line}:{name}"] = ncalls
    results[f"k{refinements}"] = dict(
        predictions=sum(entry[0] == "prediction" for entry in engine.audit),
        optimizer_calls=engine.optimizer.calls,
        host_syncs=sum("synchroniz" in str(w.message).lower() for w in caught),
        cuda_launch_kernel=runtime.get("cudaLaunchKernel", 0)
        + runtime.get("cudaLaunchKernelExC", 0),
        cuda_memcpy_async=runtime.get("cudaMemcpyAsync", 0),
        cuda_stream_synchronize=runtime.get("cudaStreamSynchronize", 0),
        aten_ops={
            key: value
            for key, value in sorted(runtime.items(), key=lambda item: -item[1])[:60]
            if key.startswith("aten::")
        },
        function_calls=calls,
    )
    print(
        refinements,
        {k: v for k, v in results[f"k{refinements}"].items() if not isinstance(v, dict)},
        flush=True,
    )
guard.remove()
Path(OUT).write_text(json.dumps(dict(tree=TREE, device=DEVICE, results=results), indent=1))
shutil.rmtree(work)
