"""Lanzamientos y bits de una lectura con K refinamientos en formas de la campaña, sin tiempos.

Uso: k_shapes.py ÁRBOL SALIDA.pt

Con el código del árbol, en cuda:0 y FP32 estricto, crea una instantánea fija de 1.024
episodios y un estado de 512 flujos con `hidden_size` 64 y 8 vecinos. Para K = 1, 2 y 4 y en
los dos modos (sin gradiente, y con gradiente más un backward de la suma del estado sobre los
parámetros del lector) cuenta lanzamientos de núcleos y copias de una llamada tras un
calentamiento, y guarda estados, lecturas y gradientes para compararlos bit a bit.
"""

import sys
from collections import Counter

TREE, OUT = sys.argv[1:3]
sys.path[:0] = [f"{TREE}/src"]

import torch  # noqa: E402
from torch.profiler import ProfilerActivity, profile  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("highest")

from mars_titan.models.titans import episodic_readout as api  # noqa: E402
from mars_titan.models.titans.config import MAX_STATE_BYTES  # noqa: E402

CODEC, CONTEXT, DEVICE = "a" * 64, "b" * 64, "cuda:0"
EPISODES, FLOWS, WIDTH = 1024, 512, 64
generator = torch.Generator().manual_seed(549)
keys = torch.randn((EPISODES, 64), generator=generator, dtype=torch.float64)
keys = (keys / keys.norm(dim=-1, keepdim=True)).float()
snapshot = api.EpisodeSnapshot.create(
    keys=keys,
    values=torch.randn((EPISODES, 64), generator=generator, dtype=torch.float32),
    labels=torch.randn(EPISODES, generator=generator, dtype=torch.float64) / 100,
    ids=torch.arange(1, EPISODES + 1, dtype=torch.int64) * 10,
    decision_at=torch.ones(EPISODES, dtype=torch.int64),
    available_at=torch.ones(EPISODES, dtype=torch.int64),
    maturity_at=torch.full((EPISODES,), 2, dtype=torch.int64),
    cutoff=10,
    codec_id=CODEC,
    context_id=CONTEXT,
    device=DEVICE,
    dtype=torch.float32,
)
start = torch.randn((FLOWS, WIDTH), generator=generator, dtype=torch.float32).to(DEVICE)


def call(model, differentiable):
    for parameter in model.parameters():
        parameter.grad = None
    z = start.clone().requires_grad_(differentiable)
    result = model(z, snapshot, context_id=CONTEXT, cutoff=10, differentiable=differentiable)
    if differentiable:
        result.state.sum().backward()
    return result


captures, counts = {}, {}
for refinements in (1, 2, 4):
    for selection in ("per_step", "first_read"):
        config = api.EpisodicReadoutConfig(
            CODEC,
            hidden_size=WIDTH,
            refinements=refinements,
            max_batch=FLOWS,
            max_working_bytes=MAX_STATE_BYTES,
            episode_selection=selection,
        )
        model = api.EpisodicReadout(config, dtype=torch.float32).to(DEVICE)
        for differentiable in (False, True):
            name = f"k{refinements}/{selection}/{'grad' if differentiable else 'nograd'}"
            call(model, differentiable)
            torch.cuda.synchronize(0)
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                result = call(model, differentiable)
                torch.cuda.synchronize(0)
            runtime = Counter({item.key: item.count for item in prof.key_averages()})
            kernels = Counter(
                event.name
                for event in prof.events()
                if event.device_type == torch.autograd.DeviceType.CUDA
            )
            counts[name] = dict(
                launches=runtime["cudaLaunchKernel"] + runtime["cudaLaunchKernelExC"],
                memcpy=runtime["cudaMemcpyAsync"],
                stream_synchronize=runtime["cudaStreamSynchronize"],
                kernels=dict(kernels.most_common()),
            )
            captures[name] = dict(
                state=result.state.detach().cpu(),
                reads=[
                    (
                        r.values.detach().cpu(),
                        r.weights.detach().cpu(),
                        r.ids.cpu(),
                        r.presence.cpu(),
                    )
                    for r in result.reads
                ],
                grads={
                    key: None if value.grad is None else value.grad.cpu()
                    for key, value in model.named_parameters()
                },
            )
            print(name, {k: v for k, v in counts[name].items() if k != "kernels"}, flush=True)
torch.save(dict(tree=TREE, counts=counts, captures=captures), OUT)
