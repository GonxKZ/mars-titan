"""Trabajo de anfitrión por recorrido de #549 en la fixture técnica, contado y no cronometrado.

Uso: work_counts.py ÁRBOL SALIDA.json TRABAJO

Con el código del árbol indicado, en cuda:0 y FP32 estricto, recorre `ChronologicalTrainer.run()`
de `mac_online` y `ReadoutTrainer._train_pass` de M1 con `RecordingOptimizer`, que solo copia
gradientes. Durante el recorrido (sin la preparación de la fixture) cuenta:
- llamadas a SHA-256 y bytes resumidos, envolviendo `hashlib.sha256`;
- llamadas a `json.dumps` y caracteres producidos;
- sincronizaciones con el anfitrión que señala `torch.cuda.set_sync_debug_mode("warn")`;
- llamadas a `cudaLaunchKernel`, copias y esperas del perfilador de PyTorch.
Son cuentas del mismo recorrido determinista, así que no dependen de la carga del equipo.
"""

import hashlib
import json
import os
import shutil
import sys
import warnings
from pathlib import Path

TREE, OUT, WORK = sys.argv[1:4]
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

DEVICE = "cuda:0"
work = Path(WORK)
if work.exists():
    shutil.rmtree(work)
work.mkdir(parents=True)
_, streams = tfr.corpus(work / "corpus")
native = load_native(os.environ["MARS_TITAN_EPISODIC_NATIVE"])
guard = ct._forbid_steps()
real_sha256, real_dumps = hashlib.sha256, json.dumps
counts = {}


class Counted:
    def __init__(self, data=b"", **kwargs):
        self._hash = real_sha256(data, **kwargs)
        counts["sha256_calls"] += 1
        counts["sha256_bytes"] += memoryview(data).nbytes

    def update(self, data):
        counts["sha256_bytes"] += memoryview(data).nbytes
        self._hash.update(data)

    def __getattr__(self, name):
        return getattr(self._hash, name)


def dumps(*args, **kwargs):
    text = real_dumps(*args, **kwargs)
    counts["json_dumps_calls"] += 1
    counts["json_dumps_chars"] += len(text)
    return text


def counted(name, function):
    counts.clear()
    counts.update(sha256_calls=0, sha256_bytes=0, json_dumps_calls=0, json_dumps_chars=0)
    hashlib.sha256, json.dumps = Counted, dumps
    torch.cuda.synchronize(0)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            torch.cuda.set_sync_debug_mode("warn")
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                observations = function()
                torch.cuda.synchronize(0)
            torch.cuda.set_sync_debug_mode(0)
    finally:
        hashlib.sha256, json.dumps = real_sha256, real_dumps
    runtime = {item.key: item.count for item in prof.key_averages()}
    syncs = [str(w.message) for w in caught if "synchroniz" in str(w.message).lower()]
    result = dict(
        counts,
        observations=observations,
        host_syncs=len(syncs),
        cuda_launch_kernel=runtime.get("cudaLaunchKernel", 0)
        + runtime.get("cudaLaunchKernelExC", 0),
        cuda_memcpy_async=runtime.get("cudaMemcpyAsync", 0),
        cuda_stream_synchronize=runtime.get("cudaStreamSynchronize", 0),
    )
    print(name, json.dumps(result), flush=True)
    return result


def predictor(variant, **options):
    specification = streams["train"].specification()
    config = FinancialConfig(specification, variant=variant, hidden_size=32, seed=42, **options)
    return FinancialPredictor(config, dtype=torch.float32).to(DEVICE)


class Never:
    requested = False


results = {}
engine = tfr.trainer(streams, work / "titans", model=predictor("mac_online"))


def titans():
    report = engine.run()
    assert report["status"] == "completed"
    return sum(entry[0] == "prediction" for entry in engine.audit)


results["titans/mac_online"] = counted("titans/mac_online", titans)

parent = predictor("mac_online", head="quantile_head_v1").eval().requires_grad_(False)
codec = FrozenEpisodeCodec(streams["train"].specification())
readout = EpisodicReadout(
    EpisodicReadoutConfig(
        codec.fingerprint(),
        hidden_size=32,
        refinements=1,
        mode="bank",
        max_working_bytes=128 * 1024**2,
        seed=42,
        episode_selection="per_step",
    ),
    dtype=torch.float32,
).to(DEVICE)
plan = tmt.recipe()
reader = mt.ReadoutTrainer(
    parent,
    readout,
    plan,
    admission="m1",
    retention=mt.retention_config(plan, "m1"),
    native=native,
    codec=codec,
    train=streams["train"],
    validation=streams["validation"],
    output=work / "readout",
    world="fixture",
    fold="0",
    optimizer_factory=tfr.RecordingOptimizer,
    audit=True,
)


def readout_pass():
    run = reader._new_pass("train")
    reader._train_pass(
        run, dict(epoch=0, phase="train", event=0, stage="start"), Never(), ct._no_save
    )
    return sum(entry[0] == "prediction" for entry in reader.audit)


results["readout/m1"] = counted("readout/m1", readout_pass)
guard.remove()
Path(OUT).write_text(
    json.dumps(dict(tree=TREE, torch=torch.__version__, results=results), indent=1)
)
shutil.rmtree(work)
