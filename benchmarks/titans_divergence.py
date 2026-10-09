"""Separar la sensibilidad propia de la memoria de Titans-MAC de los errores de precisión.

Estudia por qué las trayectorias CPU y CUDA de MAC con residual y LayerNorm se separan con
el número de pasos y qué precisión es aceptable para su memoria. Hay cinco órdenes:

- `extract` guarda tokens fusionados reales de una vista: el codificador inicial de
  `FinancialPredictor` (semilla de la receta) aplicado a las observaciones de unos flujos.
- `trajectories` recorre MAC sin grafo exterior con cada precisión y guarda salidas, estados
  en puntos de control y estadísticas de las normalizaciones.
- `lyapunov` estima en FP64 el exponente máximo de la transición rápida con el método de
  Benettin y el crecimiento libre de perturbaciones pequeñas.
- `floor` mide cuánto se separan copias exactas de un flujo según su fila en el lote.
- `report` compara cada precisión con la referencia FP64 de CPU y escribe el recibo.

Solo ejecuta pasos hacia delante y la derivada interna de la regla asociativa. No crea
optimizadores, no usa etiquetas y no modifica parámetros. BF16 se emula: el núcleo rechaza
autocast, así que las operandos y el resultado de cada `linear`, `bmm` y SDPA se redondean a
bfloat16 y el resto se calcula en FP32. Es una cota optimista del error de autocast real.
"""

import argparse
import math
import platform
import subprocess
import sys
import time
from contextlib import ExitStack, contextmanager
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import torch
from torch.nn import functional as F

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from titans_gate_retention import STREAMS, tokens_for  # noqa: E402
from titans_mac_output_scale import CANDIDATES, build  # noqa: E402

from mars_titan.data.storage import atomic_json  # noqa: E402

# (dispositivo, dtype de cálculo, aritmética)
PRECISIONS = {
    "cpu_float64": ("cpu", torch.float64, "exact"),
    "cuda_float64": ("cuda:0", torch.float64, "exact"),
    "cpu_float32": ("cpu", torch.float32, "exact"),
    "cuda_float32": ("cuda:0", torch.float32, "exact"),
    "cuda_tf32": ("cuda:0", torch.float32, "tf32"),
    "cuda_bf16_emulated": ("cuda:0", torch.float32, "bf16"),
    "cpu_bf16_emulated": ("cpu", torch.float32, "bf16"),
}
# Error de redondeo unitario de cada aritmética, para predecir el horizonte de separación.
UNIT_ROUNDOFF = {"float64": 2.0**-53, "float32": 2.0**-24, "tf32": 2.0**-11, "bf16": 2.0**-8}
# reproduction: parámetros y tokens FP64 como titans_mac_output_scale.py.
# precision: parámetros y tokens redondeados a FP32 y comunes a todas las precisiones.
STUDIES = ("reproduction", "precision")
CAMPAIGN = "gate_bias_residual_layer_norm"
CONTRAST = "gate_bias"
CHECKPOINTS = (64, 256, 512, 1024, 1536, 2048, 3072, 4096)
THRESHOLDS = (1e-6, 1e-3, 1e-2)
LAYER_NORM_EPS = 1e-5
LEVELS = ("min", "p01", "p50")


def _round_bf16(value):
    return value if value is None else value.to(torch.bfloat16).to(value.dtype)


@contextmanager
def arithmetic(mode):
    """Fijar TF32 o emular BF16 durante el recorrido y restaurar el estado previo."""
    saved = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    originals = dict(linear=F.linear, sdpa=F.scaled_dot_product_attention, bmm=torch.bmm)
    torch.backends.cuda.matmul.allow_tf32 = mode == "tf32"
    torch.backends.cudnn.allow_tf32 = mode == "tf32"
    if mode == "bf16":

        def linear(values, weight, bias=None):
            return _round_bf16(
                originals["linear"](_round_bf16(values), _round_bf16(weight), _round_bf16(bias))
            )

        def sdpa(query, key, value, *args, **kwargs):
            rounded = (_round_bf16(item) for item in (query, key, value))
            return _round_bf16(originals["sdpa"](*rounded, *args, **kwargs))

        def bmm(left, right):
            return _round_bf16(originals["bmm"](_round_bf16(left), _round_bf16(right)))

        F.linear, F.scaled_dot_product_attention, torch.bmm = linear, sdpa, bmm
    try:
        yield
    finally:
        F.linear, F.scaled_dot_product_attention = originals["linear"], originals["sdpa"]
        torch.bmm = originals["bmm"]
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = saved


class NormalizationProbe:
    """Registrar la varianza de entrada de cada LayerNorm y la norma de entrada de normalize."""

    def __init__(self):
        self.variances, self.norms = [], []
        self.originals = dict(layer_norm=F.layer_norm, normalize=F.normalize)

    def __enter__(self):
        def layer_norm(values, *args, **kwargs):
            self.variances.append(values.detach().var(dim=-1, unbiased=False).min())
            return self.originals["layer_norm"](values, *args, **kwargs)

        def normalize(values, *args, **kwargs):
            self.norms.append(values.detach().norm(dim=-1).min())
            return self.originals["normalize"](values, *args, **kwargs)

        F.layer_norm, F.normalize = layer_norm, normalize
        return self

    def __exit__(self, *exc):
        F.layer_norm, F.normalize = self.originals["layer_norm"], self.originals["normalize"]

    def summary(self):
        norm = torch.stack(self.norms).double().cpu()
        result = dict(
            layer_norm_calls=len(self.variances),
            normalize_calls=len(self.norms),
            normalize_min_input_norm=float(norm.min()),
        )
        if not self.variances:
            return result
        variance = torch.stack(self.variances).double().cpu()
        quantiles = torch.tensor([0.0, 0.01, 0.5], dtype=torch.float64)
        # Ganancia máxima del Jacobiano de LN, 1/sqrt(var + eps), en la llamada peor condicionada.
        return dict(
            result,
            layer_norm_input_variance_quantiles=dict(
                zip(LEVELS, torch.quantile(variance, quantiles).tolist(), strict=True)
            ),
            layer_norm_calls_below_10_eps=int((variance < 10 * LAYER_NORM_EPS).sum()),
            layer_norm_max_gain=float((variance.min() + LAYER_NORM_EPS) ** -0.5),
        )


def model(candidate, *, flows, dim, parameters, dtype, device):
    """MAC del candidato con parámetros iniciales en `parameters` y cálculo en `dtype`."""
    mac, _ = build(CANDIDATES[candidate], dim=dim, flows=flows, dtype=parameters, device="cpu")
    return mac.to(device=device, dtype=dtype)


def state_point(state):
    """[flujos, 4·D²]: pesos rápidos y momentum por capa, en el orden de `fast_state_point`."""
    memory = state.memory
    return torch.cat([value.flatten(1) for value in (*memory.weights, *memory.momentum)], dim=1)


def _with_point(state, point, dim):
    pieces = [piece.reshape(-1, dim, dim).clone() for piece in point.split(dim * dim, dim=1)]
    depth = len(state.memory.weights)
    memory = replace(state.memory, weights=tuple(pieces[:depth]), momentum=tuple(pieces[depth:]))
    return replace(state, memory=memory)


def stream_tokens(stream, *, study, dim, flows, length, real=None):
    """Tokens [pasos, flujos, 1, D] en FP64. En el estudio de precisión, redondeados a FP32."""
    if stream == "real":
        tokens = real[:length, :flows].to(torch.float64)
    else:
        values = tokens_for(
            stream, dim=dim, flows=flows, length=length, dtype=torch.float64, device="cpu"
        )
        tokens = torch.stack(values)
    if study == "precision":
        tokens = tokens.to(torch.float32).to(torch.float64)
    return tokens


def run_trajectory(candidate, tokens, *, precision, study, dim, probe=True):
    """Recorrer MAC sin grafo exterior. Devuelve salidas por paso y estados por punto de control."""
    device, dtype, mode = PRECISIONS[precision]
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("La precisión pedida necesita cuda:0 visible")
    length, flows = tokens.shape[0], tokens.shape[1]
    parameters = torch.float64 if study == "reproduction" else torch.float32
    mac = model(candidate, flows=flows, dim=dim, parameters=parameters, dtype=dtype, device=device)
    inputs = tokens.to(device=device, dtype=dtype)
    outputs = torch.empty(length, flows, dim, dtype=dtype, device=device)
    states = {}
    started = time.perf_counter()
    with ExitStack() as stack:
        stack.enter_context(arithmetic(mode))
        stack.enter_context(torch.no_grad())
        recorder = stack.enter_context(NormalizationProbe()) if probe else None
        state = mac.initial_state(flows)
        for step in range(1, length + 1):
            output, state = mac(inputs[step - 1], state)
            outputs[step - 1] = output
            if step in CHECKPOINTS or step == length:
                states[step] = state_point(state).clone()
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    return dict(
        candidate=candidate,
        precision=precision,
        study=study,
        outputs=outputs.double().cpu(),
        states={step: value.double().cpu() for step, value in states.items()},
        normalization=None if recorder is None else recorder.summary(),
        seconds=time.perf_counter() - started,
        device_name=torch.cuda.get_device_name(0) if device.startswith("cuda") else "cpu",
    )


def lyapunov(candidate, tokens, *, dim, epsilon=1e-8, renormalize=8, directions=3, seed=7):
    """Exponente máximo por paso con Benettin en FP64, por flujo y dirección inicial.

    Las filas perturbadas son copias de cada flujo con su mismo token. Cada `renormalize`
    pasos se acumula log(‖δ‖/δ₀) y la separación vuelve a medir δ₀ en su dirección.
    """
    length, flows = tokens.shape[0], tokens.shape[1]
    rows = flows * (1 + directions)
    mac = model(
        candidate, flows=rows, dim=dim, parameters=torch.float32, dtype=torch.float64, device="cpu"
    )
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        state = mac.initial_state(rows)
        point = state_point(state)
        reference = point[:flows]
        scale = epsilon * reference.norm(dim=1)
        delta = torch.randn(directions * flows, point.shape[1], generator=generator)
        delta = delta.double() / delta.double().norm(dim=1, keepdim=True)
        base = scale.repeat(directions)
        point = torch.cat([reference, reference.repeat(directions, 1) + delta * base[:, None]])
        state = _with_point(state, point, dim)
        logs = torch.zeros(directions * flows, dtype=torch.float64)
        running = {}
        for step in range(1, length + 1):
            _, state = mac(tokens[step - 1].repeat(1 + directions, 1, 1), state)
            if step % renormalize:
                continue
            point = state_point(state)
            reference = point[:flows].repeat(directions, 1)
            separation = point[flows:] - reference
            size = separation.norm(dim=1)
            if not torch.all(size > 0):
                raise ValueError("La perturbación se anuló y el exponente no está definido")
            logs += torch.log(size / base)
            point = torch.cat([point[:flows], reference + separation * (base / size)[:, None]])
            state = _with_point(state, point, dim)
            if step in CHECKPOINTS or step == length:
                running[step] = (logs / step).mean().item()
    per_row = (logs / (length - length % renormalize)).reshape(directions, flows)
    return dict(
        exponent_per_step=per_row.mean().item(),
        exponent_min=per_row.min().item(),
        exponent_max=per_row.max().item(),
        exponent_by_flow=per_row.mean(dim=0).tolist(),
        running_mean=running,
        epsilon=epsilon,
        renormalize=renormalize,
        directions=directions,
        seed=seed,
    )


def free_growth(
    candidate, tokens, *, dim, epsilon, directions=2, seed=11, device="cpu", dtype=torch.float64
):
    """Separación relativa de la salida tras perturbar el estado inicial, sin renormalizar.

    Devuelve la separación por paso y dirección y las salidas [pasos, 1 + direcciones,
    flujos, D], con la referencia en la primera posición.

    Con `epsilon=0` las filas son copias exactas y solo las separa el redondeo de su posición
    en el lote, que mide el piso de invariancia por filas del dispositivo.
    """
    length, flows = tokens.shape[0], tokens.shape[1]
    rows = flows * (1 + directions)
    mac = model(
        candidate, flows=rows, dim=dim, parameters=torch.float32, dtype=dtype, device=device
    )
    generator = torch.Generator().manual_seed(seed)
    inputs = tokens.to(device=device, dtype=dtype)
    distances = torch.empty(length, directions, dtype=dtype, device=device)
    outputs = torch.empty(length, 1 + directions, flows, dim, dtype=dtype, device=device)
    with torch.no_grad(), arithmetic("exact"):
        state = mac.initial_state(rows)
        point = state_point(state)
        reference = point[:flows]
        delta = torch.randn(directions * flows, point.shape[1], generator=generator)
        delta = (delta / delta.norm(dim=1, keepdim=True)).to(device=device, dtype=dtype)
        size = (epsilon * reference.norm(dim=1)).repeat(directions)
        point = torch.cat([reference, reference.repeat(directions, 1) + delta * size[:, None]])
        state = _with_point(state, point, dim)
        for step in range(1, length + 1):
            output, state = mac(inputs[step - 1].repeat(1 + directions, 1, 1), state)
            pieces = output.reshape(1 + directions, flows, dim)
            outputs[step - 1] = pieces
            gap = (pieces[1:] - pieces[:1]).flatten(1).norm(dim=1)
            distances[step - 1] = gap / pieces[0].norm()
    return distances.double().cpu(), outputs.double().cpu()


def relative_curve(outputs, reference):
    """‖o_t − r_t‖ / ‖r_t‖ por paso, sobre flujos y dimensiones."""
    gap = (outputs - reference).flatten(1).norm(dim=1)
    return gap / reference.flatten(1).norm(dim=1)


def horizons(curve):
    """Primer paso (1-indexado) en que la separación alcanza cada umbral, o None."""
    result = {}
    for threshold in THRESHOLDS:
        reached = torch.nonzero(curve >= threshold)
        result[f"{threshold:g}"] = None if not len(reached) else int(reached[0]) + 1
    return result


def growth_rate(curve, low=1e-12, high=1e-3):
    """Pendiente de log(separación) por paso en la fase exponencial, por mínimos cuadrados."""
    steps = torch.arange(1, len(curve) + 1, dtype=torch.float64)
    mask = (curve > low) & (curve < high)
    if int(mask.sum()) < 16:
        return None
    x, y = steps[mask], torch.log(curve[mask])
    x = x - x.mean()
    return float((x * (y - y.mean())).sum() / (x * x).sum())


def climate(outputs):
    """Promedios de la segunda mitad: comparables aunque las trayectorias se separen."""
    half = outputs[len(outputs) // 2 :]
    return dict(
        output_norm=half.norm(dim=-1).mean().item(),
        output_abs_p90=torch.quantile(half.abs().flatten(), 0.9).item(),
    )


def climate_gap(mine, theirs):
    return {key: abs(mine[key] - theirs[key]) / abs(theirs[key]) for key in theirs}


def _save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _streams(args):
    names = list(STREAMS)
    real = None
    if args.real_tokens is not None:
        real = torch.load(args.real_tokens, weights_only=True)["tokens"]
        names.append("real")
    return names, real


def _length(stream, args, real):
    return min(args.length, real.shape[0]) if stream == "real" else args.length


def command_trajectories(args):
    names, real = _streams(args)
    for precision in args.precisions:
        for study in args.studies:
            candidates = tuple(CANDIDATES) if study == "reproduction" else (CAMPAIGN, CONTRAST)
            for stream in names if study == "precision" else STREAMS:
                for candidate in candidates:
                    if study == "reproduction" and not precision.endswith("float64"):
                        continue
                    path = args.cache / study / stream / candidate / f"{precision}.pt"
                    if path.exists() and not args.overwrite:
                        continue
                    tokens = stream_tokens(
                        stream,
                        study=study,
                        dim=args.dim,
                        flows=args.flows,
                        length=_length(stream, args, real),
                        real=real,
                    )
                    result = run_trajectory(
                        candidate, tokens, precision=precision, study=study, dim=args.dim
                    )
                    _save(path, result)
                    print(path.relative_to(args.cache), f"{result['seconds']:.1f} s", flush=True)


def command_lyapunov(args):
    names, real = _streams(args)
    for stream in names:
        for candidate in CANDIDATES:
            path = args.cache / "lyapunov" / stream / f"{candidate}.pt"
            if path.exists() and not args.overwrite:
                continue
            tokens = stream_tokens(
                stream,
                study="precision",
                dim=args.dim,
                flows=args.flows,
                length=_length(stream, args, real),
                real=real,
            )
            started = time.perf_counter()
            result = dict(benettin=lyapunov(candidate, tokens, dim=args.dim))
            if candidate == CAMPAIGN:
                result["free_growth"] = {}
                for epsilon in (1e-15, 1e-7):
                    distances, outputs = free_growth(
                        candidate, tokens, dim=args.dim, epsilon=epsilon
                    )
                    reference = climate(outputs[:, 0])
                    # Dispersión del clima entre trayectorias FP64 exactas que solo difieren
                    # por la perturbación inicial: la vara de medir de las demás precisiones.
                    spread = [
                        climate_gap(climate(outputs[:, index]), reference)
                        for index in range(1, outputs.shape[1])
                    ]
                    result["free_growth"][f"{epsilon:g}"] = dict(
                        distances=distances,
                        climate_spread={key: max(row[key] for row in spread) for key in reference},
                    )
            result["seconds"] = time.perf_counter() - started
            _save(path, result)
            print(path.relative_to(args.cache), result["benettin"]["exponent_per_step"], flush=True)


def command_floor(args):
    """Piso de invariancia por filas de cada dispositivo y precisión, con copias exactas."""
    names, real = _streams(args)
    for stream in names:
        for dtype in (torch.float64, torch.float32):
            path = args.cache / "floor" / stream / f"{args.device}-{dtype}.pt".replace(":", "")
            if path.exists() and not args.overwrite:
                continue
            tokens = stream_tokens(
                stream,
                study="precision",
                dim=args.dim,
                flows=args.flows,
                length=_length(stream, args, real),
                real=real,
            )
            distances, _ = free_growth(
                CAMPAIGN, tokens, dim=args.dim, epsilon=0.0, device=args.device, dtype=dtype
            )
            _save(path, dict(device=args.device, dtype=str(dtype), distances=distances))
            print(path.relative_to(args.cache), distances.max().item(), flush=True)


def _compare(result, reference):
    curve = relative_curve(result["outputs"], reference["outputs"])
    length = len(curve)
    state_gaps = {
        str(step): (
            (result["states"][step] - reference["states"][step]).norm()
            / reference["states"][step].norm()
        ).item()
        for step in reference["states"]
    }
    mine = climate(result["outputs"])
    return dict(
        output_relative_distance={
            str(step): curve[step - 1].item() for step in CHECKPOINTS if step <= length
        },
        state_relative_distance=state_gaps,
        horizons=horizons(curve),
        growth_rate_per_step=growth_rate(curve),
        climate_relative_difference=climate_gap(mine, climate(reference["outputs"])),
        climate=mine,
        normalization=result["normalization"],
        seconds=result["seconds"],
    )


def command_report(args):
    names, real = _streams(args)
    results = dict(reproduction={}, precision={}, lyapunov={})
    for study in STUDIES:
        for stream in names if study == "precision" else STREAMS:
            folder = args.cache / study / stream
            for candidate_folder in sorted(folder.glob("*")) if folder.exists() else []:
                reference_path = candidate_folder / "cpu_float64.pt"
                if not reference_path.exists():
                    continue
                reference = torch.load(reference_path, weights_only=False)
                entry = results[study].setdefault(stream, {})[candidate_folder.name] = dict(
                    reference_climate=climate(reference["outputs"]),
                    reference_normalization=reference["normalization"],
                    steps=len(reference["outputs"]),
                )
                for path in sorted(candidate_folder.glob("*.pt")):
                    if path.stem == "cpu_float64":
                        continue
                    result = torch.load(path, weights_only=False)
                    entry[path.stem] = _compare(result, reference)
    for stream in names:
        folder = args.cache / "lyapunov" / stream
        for path in sorted(folder.glob("*.pt")) if folder.exists() else []:
            value = torch.load(path, weights_only=False)
            record = dict(benettin=value["benettin"], seconds=value["seconds"])
            for epsilon, growth in value.get("free_growth", {}).items():
                curve = growth["distances"].max(dim=1).values
                record.setdefault("free_growth", {})[epsilon] = dict(
                    climate_spread=growth["climate_spread"],
                    horizons=horizons(curve),
                    growth_rate_per_step=growth_rate(curve),
                    output_relative_distance={
                        str(step): curve[step - 1].item()
                        for step in CHECKPOINTS
                        if step <= len(curve)
                    },
                )
            exponent = value["benettin"]["exponent_per_step"]
            if exponent > 0:
                record["predicted_horizon_1e-2"] = {
                    name: math.log(1e-2 / roundoff) / exponent
                    for name, roundoff in UNIT_ROUNDOFF.items()
                }
            results["lyapunov"].setdefault(stream, {})[path.stem] = record
    results["floor"] = {}
    for stream in names:
        folder = args.cache / "floor" / stream
        for path in sorted(folder.glob("*.pt")) if folder.exists() else []:
            value = torch.load(path, weights_only=False)
            curve = value["distances"].max(dim=1).values
            first = torch.nonzero(curve > 0)
            results["floor"].setdefault(stream, {})[path.stem] = dict(
                device=value["device"],
                dtype=value["dtype"],
                first_nonzero_step=None if not len(first) else int(first[0]) + 1,
                horizons=horizons(curve),
                output_relative_distance={
                    str(step): curve[step - 1].item() for step in CHECKPOINTS if step <= len(curve)
                },
            )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    cuda = torch.cuda.is_available()
    receipt = dict(
        schema_version=1,
        recorded_at_utc=datetime.now(UTC).isoformat(),
        commit=commit,
        command="uv run --no-sync python benchmarks/titans_divergence.py "
        "{extract,trajectories,lyapunov,floor,report} ...",
        environment="OMP_NUM_THREADS=2, MKL_NUM_THREADS=2, CUBLAS_WORKSPACE_CONFIG=:4096:8",
        scope=(
            "Trayectorias de MAC sin grafo exterior, sin optimizador, sin etiquetas y sin "
            "cambiar parámetros. No es una medida de utilidad predictiva"
        ),
        hardware=dict(
            machine=platform.machine(),
            cuda_name=torch.cuda.get_device_name(0) if cuda else None,
        ),
        versions=dict(
            python=platform.python_version(),
            torch=torch.__version__,
            cuda=torch.version.cuda,
            cudnn=torch.backends.cudnn.version() if cuda else None,
        ),
        shapes=dict(dim=args.dim, depth=2, flows=args.flows, heads=4, persistent_tokens=4),
        candidates={
            name: dict(gate_bias=None if bias is None else vars(bias), residual_layer_norm=residual)
            for name, (bias, residual) in CANDIDATES.items()
        },
        precisions={
            name: dict(device=device, dtype=str(dtype).removeprefix("torch."), arithmetic=mode)
            for name, (device, dtype, mode) in PRECISIONS.items()
        },
        bf16_emulation=(
            "Operandos y resultado de linear, bmm y SDPA redondeados a bfloat16, acumulación y "
            "resto de operaciones en FP32. Cota optimista del error de autocast real"
        ),
        studies=dict(
            reproduction="Parámetros y tokens FP64 como titans_mac_output_scale.py",
            precision="Parámetros iniciales y tokens redondeados a FP32 y comunes a todas",
        ),
        streams=dict(
            fused_norm_1="N(0, I/D)",
            unit_variance="N(0, I)",
            recurring_unit="cuatro prototipos N(0, I) alternos más 0,3·N(0, I)",
            real="Tokens fusionados reales de `extract`, con el codificador inicial",
        ),
        real_tokens=None if args.real_tokens is None else _real_metadata(args.real_tokens),
        reference="cpu_float64",
        metrics=dict(
            output_relative_distance="‖o_t − r_t‖/‖r_t‖ sobre flujos y dimensiones",
            state_relative_distance="‖z − z_ref‖/‖z_ref‖ con pesos rápidos y momentum",
            horizons="primer paso con distancia de salida igual o mayor que el umbral",
            growth_rate_per_step="pendiente de log(distancia) entre 1e-12 y 1e-3",
            climate="norma media y percentil 90 de |salida| en la segunda mitad del recorrido",
            climate_spread="diferencia de clima entre trayectorias FP64 con perturbación inicial",
            lyapunov="Benettin en FP64, δ₀ = 1e-8·‖z₀‖, renormalización cada 8 pasos",
            free_growth="máximo sobre direcciones de la distancia relativa de salida",
            floor="copias exactas de cada flujo en el mismo lote, separadas solo por su fila",
        ),
        unit_roundoff=UNIT_ROUNDOFF,
        results=results,
        optimizer_steps=0,
        training_runs=0,
    )
    atomic_json(args.output, receipt)


def _real_metadata(path):
    payload = torch.load(path, weights_only=True)
    return dict(
        path=str(path),
        shape=list(payload["tokens"].shape),
        **{key: value for key, value in payload.items() if key != "tokens"},
    )


def command_extract(args):
    """Tokens fusionados reales de `flows` activos con más observaciones en el tramo de ajuste."""
    from collections import defaultdict

    from mars_titan.data.input_policy import MODALITIES
    from mars_titan.models.titans.financial_inputs import DecisionBatch
    from mars_titan.training import campaign_throughput as throughput
    from mars_titan.training import titans_walk_forward as titans
    from mars_titan.training.financial_run import load_recipe

    _, document = load_recipe(args.recipe)
    seed = args.seed
    _, sources, _ = throughput._chronological_sources(args.view, document, args.work)
    source = sources["train"]
    specification = source.specification()
    predictor, _ = titans._predictor(document, specification, "mac_online", seed, "cpu")
    candidate = (
        document["predictor"]["gate_bias"] is not None
        and document["predictor"]["memory_residual_layer_norm"]
    )
    if not candidate or predictor.config.hidden_size != args.dim:
        raise ValueError("La receta no corresponde al candidato de campaña del estudio")
    reference = model(
        CAMPAIGN,
        flows=args.flows,
        dim=args.dim,
        parameters=torch.float32,
        dtype=torch.float32,
        device="cpu",
    )
    same = all(
        torch.equal(a, b)
        for a, b in zip(
            reference.state_dict().values(), predictor.mac.state_dict().values(), strict=True
        )
        if isinstance(a, torch.Tensor)
    )
    if not same:
        raise ValueError("El MAC del estudio no coincide con el del predictor de la receta")
    counts = sorted(
        (
            (asset["counts"]["train"], f"{asset['market']}/{asset['symbol']}")
            for asset in source.dataset.assets
        ),
        reverse=True,
    )
    chosen = sorted(flow for _, flow in counts[: args.flows])
    tokens, checked = defaultdict(list), False
    with torch.no_grad():
        for event in source.batched_events(block_rows=256):
            for raw in event.inputs:
                batch = DecisionBatch.from_corpus(raw, specification)
                rows = [i for i, flow in enumerate(batch.flow_ids) if flow in chosen]
                if not rows:
                    continue
                selected = batch.select(rows)
                inputs, presence = selected.inputs, selected.presence
                # Mismo cálculo que FinancialPredictor.prepare hasta el token fusionado.
                pieces = [predictor.price_encoder(inputs["prices"])]
                for index, name in enumerate(MODALITIES[1:], 1):
                    pieces.append(
                        predictor.encoders[name](inputs[name]) * presence[:, index : index + 1]
                    )
                pieces.append(presence.to(torch.float32))
                fused = predictor.fusion(torch.cat(pieces, dim=-1))
                if not checked:
                    state = predictor.initial_state(selected.flow_ids)
                    prepared = predictor.prepare(selected, state)
                    if not torch.equal(prepared.detached_tokens, fused):
                        raise ValueError("El token reconstruido no coincide con prepare")
                    checked = True
                for row, flow in enumerate(selected.flow_ids):
                    tokens[flow].append(fused[row].clone())
            if all(len(tokens[flow]) >= args.length for flow in chosen):
                break
    length = min(len(tokens[flow]) for flow in chosen)
    stacked = torch.stack([torch.stack(tokens[flow][:length]) for flow in chosen], dim=1)
    stacked = stacked.unsqueeze(2)
    payload = dict(
        tokens=stacked,
        flows=chosen,
        view=str(args.view),
        recipe=str(args.recipe),
        seed=seed,
        encoder="FinancialPredictor inicial, mac_online, semilla de la receta",
        steps=length,
        token_norm_mean=stacked.norm(dim=-1).mean().item(),
    )
    _save(args.output, payload)
    print(args.output, list(stacked.shape), payload["token_norm_mean"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    extract = commands.add_parser("extract")
    extract.add_argument("--view", type=Path, required=True)
    extract.add_argument("--recipe", type=Path, required=True)
    extract.add_argument("--work", type=Path, required=True)
    extract.add_argument("--output", type=Path, required=True)
    extract.add_argument("--seed", type=int, default=42)
    extract.add_argument("--dim", type=int, default=64)
    extract.add_argument("--flows", type=int, default=4)
    extract.add_argument("--length", type=int, default=4096)
    for name in ("trajectories", "lyapunov", "floor", "report"):
        command = commands.add_parser(name)
        command.add_argument("--cache", type=Path, required=True)
        command.add_argument("--real-tokens", type=Path)
        command.add_argument("--dim", type=int, default=64)
        command.add_argument("--flows", type=int, default=4)
        command.add_argument("--length", type=int, default=4096)
        command.add_argument("--overwrite", action="store_true")
    commands.choices["trajectories"].add_argument(
        "--precisions", nargs="+", choices=tuple(PRECISIONS), required=True
    )
    commands.choices["trajectories"].add_argument(
        "--studies", nargs="+", choices=STUDIES, default=list(STUDIES)
    )
    commands.choices["report"].add_argument("--output", type=Path, required=True)
    commands.choices["floor"].add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args(argv)
    torch.set_num_threads(args.threads)
    dict(
        extract=command_extract,
        trajectories=command_trajectories,
        lyapunov=command_lyapunov,
        floor=command_floor,
        report=command_report,
    )[args.command](args)


if __name__ == "__main__":
    main()
