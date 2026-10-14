"""Captura de emisiones y gradientes de #549 sobre la fixture técnica, sin pasos.

Uso: parity_capture.py ÁRBOL SALIDA.pt DISPOSITIVO DTYPE TRABAJO

Con el código del árbol indicado recorre:
- `ChronologicalTrainer.run()` de cada variante de Titans-MAC con `RecordingOptimizer`, que
  solo copia los gradientes de cada llamada.
- `ReadoutTrainer._train_pass` de M0, M1 y M3 hasta el cierre del tramo de ajuste, con el
  mismo sustituto. `run()` del lector exige la protección levantada, así que se recorre el
  tramo como hace `chronological_replay.py`.
El gancho de `campaign_throughput` prohíbe cualquier paso de `torch.optim` y al final se
comprueba que ningún parámetro ha cambiado. Guarda la auditoría completa, los gradientes por
nombre y las predicciones pendientes para compararlas bit a bit entre árboles.
"""

import copy
import os
import shutil
import sys
from pathlib import Path

TREE, OUT, DEVICE, DTYPE, WORK = sys.argv[1:6]
sys.path[:0] = [TREE, f"{TREE}/src"]

import torch  # noqa: E402

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
from mars_titan.models.titans.financial import (  # noqa: E402
    VARIANTS,
    FinancialConfig,
    FinancialPredictor,
)
from mars_titan.training import campaign_throughput as ct  # noqa: E402
from mars_titan.training import mars_titan_run as mt  # noqa: E402
from tests.training import test_financial_run as tfr  # noqa: E402
from tests.training import test_mars_titan_run as tmt  # noqa: E402

assert Path(mt.__file__).resolve().is_relative_to(Path(TREE).resolve()), mt.__file__
dtype = dict(float64=torch.float64, float32=torch.float32)[DTYPE]
work = Path(WORK)
if work.exists():
    shutil.rmtree(work)
work.mkdir(parents=True)
_, streams = tfr.corpus(work / "corpus")
native = load_native(os.environ["MARS_TITAN_EPISODIC_NATIVE"])
guard = ct._forbid_steps()


def snapshot(*modules):
    return [
        value.detach().clone()
        for module in modules
        for group in (module.parameters(), module.buffers())
        for value in group
    ]


def unchanged(before, *modules):
    after = snapshot(*modules)
    return len(before) == len(after) and all(
        torch.equal(a, b) for a, b in zip(before, after, strict=True)
    )


def predictor(variant, **options):
    specification = streams["train"].specification()
    config = FinancialConfig(specification, variant=variant, hidden_size=32, seed=42, **options)
    return FinancialPredictor(config, dtype=dtype).to(DEVICE)


def by_name(module, records):
    names = {id(value): name for name, value in module.named_parameters()}
    return [
        {names[key]: None if value is None else value.cpu() for key, value in record.items()}
        for record in records
    ]


class Never:
    requested = False


def closing(engine, states):
    """Exportar a CPU los estados de todos los flujos antes de que el cierre los descarte."""
    original, model = engine._close, engine.predictor

    def close(run):
        flows, size = list(run.flows), model.config.max_batch
        with torch.no_grad():
            states.append(
                [
                    model.export_state_cpu(run.flows.gather(flows[i : i + size]))
                    for i in range(0, len(flows), size)
                ]
            )
        return original(run)

    engine._close = close


captures = {}
for variant in VARIANTS:
    model = predictor(variant)
    before = snapshot(model)
    engine = tfr.trainer(streams, work / f"titans-{variant}", model=model)
    states = []
    closing(engine, states)
    report = engine.run()
    assert report["status"] == "completed", report["status"]
    assert unchanged(before, model), variant
    captures[f"titans/{variant}"] = dict(
        audit=engine.audit,
        grads=by_name(model, engine.optimizer.records),
        calls=engine.optimizer.calls,
        states=states,
        identity=engine.identity,
        report={key: report[key] for key in ("status", "history") if key in report},
    )
    print(variant, len(engine.audit), engine.optimizer.calls, len(states), flush=True)

for admission in ("m0", "m1", "m3"):
    # Cabeza de cuantiles, como `tmt.parent`, porque la receta del lector usa pinball.
    parent = predictor("mac_online", head="quantile_head_v1").eval().requires_grad_(False)
    codec = FrozenEpisodeCodec(streams["train"].specification())
    config = EpisodicReadoutConfig(
        codec.fingerprint(),
        hidden_size=32,
        refinements=1,
        mode="no_bank" if admission == "m0" else "bank",
        max_working_bytes=128 * 1024**2,
        seed=42,
        episode_selection="per_step",
    )
    readout = EpisodicReadout(config, dtype=dtype).to(DEVICE)
    plan = tmt.recipe()
    extra = dict(scalers=tmt.scalers_for(streams, plan)) if admission == "m3" else {}
    engine = mt.ReadoutTrainer(
        parent,
        readout,
        plan,
        admission=admission,
        retention=mt.retention_config(plan, admission, **extra),
        native=native,
        codec=codec,
        train=streams["train"],
        validation=streams["validation"],
        output=work / f"readout-{admission}",
        world="fixture",
        fold="0",
        optimizer_factory=tfr.RecordingOptimizer,
        audit=True,
    )
    before = snapshot(parent, readout)
    states = []
    closing(engine, states)
    run = engine._new_pass("train")
    cursor = dict(epoch=0, phase="train", event=0, stage="start")
    metrics = engine._train_pass(run, cursor, Never(), ct._no_save)
    assert unchanged(before, parent, readout), admission
    captures[f"readout/{admission}"] = dict(
        audit=engine.audit,
        grads=by_name(readout, engine.optimizer.records),
        calls=engine.optimizer.calls,
        metrics=copy.deepcopy(metrics),
        states=states,
        identity=getattr(engine, "identity", None),
    )
    print(admission, len(engine.audit), engine.optimizer.calls, len(states), flush=True)

guard.remove()
torch.save(dict(tree=TREE, device=DEVICE, dtype=DTYPE, captures=captures), OUT)
shutil.rmtree(work)
