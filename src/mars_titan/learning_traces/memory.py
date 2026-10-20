"""Trazas del interior de la memoria neuronal de Titans-MAC (#448).

`MemoryObserver` se engancha a cada `NeuralMemory` del modelo. Mientras está activo, la
memoria le entrega en cada token las puertas de olvido (α), momentum (η) y tasa interna
(θ), la pérdida asociativa y los pesos rápidos y el momentum ya actualizados. El observador
acumula en el dispositivo, sin sincronizar, la suma, la suma de cuadrados, el mínimo y el
máximo de cada magnitud, y para las puertas cuántos valores quedan por debajo de
`SATURATION` o por encima de `1 - SATURATION`. Así se ve si las puertas se saturan o si el
olvido borra la memoria en pocos tokens.

Al registrar, cada magnitud da su media, desviación típica, mínimo, máximo y, en las
puertas, las fracciones saturadas, con el nombre del módulo como grupo (por ejemplo
`blocks.0.memory`). Las normas de los pesos rápidos y del momentum se miden por flujo y
capa. La pérdida asociativa se divide entre el número de flujos del lote.

Lo acumulado cubre los recorridos hacia delante desde el paso o el punto de control
anterior. Solo lee copias separadas del grafo y no consume números aleatorios. Con el
observador inactivo la memoria no le llama, y sin trazas no se engancha a nada.
"""

import torch

from mars_titan.models.titans.neural_memory import NeuralMemory

from .hooks import TraceObserver
from .recorder import trace_name

# Umbral fijo de saturación de las puertas, declarado antes de cualquier ejecución.
SATURATION = 0.01
GATES = ("alpha", "eta", "theta")


class _Running:
    """Suma, suma de cuadrados, mínimo y máximo en float64 sobre el dispositivo."""

    def __init__(self, device, saturation=False):
        zero = torch.zeros((), dtype=torch.float64, device=device)
        self.count = 0
        self.total, self.squares = zero.clone(), zero.clone()
        self.minimum = torch.full((), float("inf"), dtype=torch.float64, device=device)
        self.maximum = torch.full((), float("-inf"), dtype=torch.float64, device=device)
        self.low = zero.clone() if saturation else None
        self.high = zero.clone() if saturation else None

    def add(self, value):
        x = value.detach().reshape(-1).to(torch.float64)
        self.count += x.numel()
        self.total += x.sum()
        self.squares += x.square().sum()
        self.minimum = torch.minimum(self.minimum, x.min())
        self.maximum = torch.maximum(self.maximum, x.max())
        if self.low is not None:
            self.low += (x < SATURATION).sum()
            self.high += (x > 1 - SATURATION).sum()

    def record(self, recorder, metric, group):
        if not self.count:
            return
        mean = self.total / self.count
        variance = (self.squares / self.count - mean.square()).clamp(min=0)
        recorder.scalar(f"{metric}.mean", mean, group=group)
        recorder.scalar(f"{metric}.std", variance.sqrt(), group=group)
        recorder.scalar(f"{metric}.min", self.minimum, group=group)
        recorder.scalar(f"{metric}.max", self.maximum, group=group)
        if self.low is not None:
            recorder.scalar(f"{metric}.low_fraction", self.low / self.count, group=group)
            recorder.scalar(f"{metric}.high_fraction", self.high / self.count, group=group)


class _Layer:
    """Acumuladores de una memoria mientras el observador está activo."""

    def __init__(self, name, device):
        self.name, self.device, self.tokens = name, device, 0
        self.shared_writes = self.shared_rows = 0
        self.values = {}

    def running(self, key, saturation=False):
        if key not in self.values:
            self.values[key] = _Running(self.device, saturation)
        return self.values[key]

    def __call__(
        self,
        *,
        alpha,
        eta,
        theta,
        loss,
        weights,
        momentum,
        shared_rows=None,
        plasticity=(),
        capture_applied=False,
    ):
        prefix = "memory"
        if shared_rows is None:
            self.tokens += 1
        else:
            self.shared_writes += 1
            self.shared_rows += shared_rows
            prefix += ".shared"
            self.running(f"{prefix}.rows_per_write").add(loss.new_tensor(shared_rows))
        for key, value in (("alpha", alpha), ("eta", eta), ("theta", theta)):
            self.running(f"{prefix}.{key}", saturation=key != "theta").add(value)
        flows = weights[0].shape[0]
        # La división se hace en float64, como el resto de acumulados.
        self.running(f"{prefix}.associative_loss").add(loss.detach().to(torch.float64) / flows)
        for index, (weight, step) in enumerate(zip(weights, momentum, strict=True)):
            self.running(f"{prefix}.fast_weight_l2.{index}").add(
                torch.linalg.vector_norm(weight.detach(), dim=(1, 2), dtype=torch.float64)
            )
            self.running(f"{prefix}.momentum_l2.{index}").add(
                torch.linalg.vector_norm(step.detach(), dim=(1, 2), dtype=torch.float64)
            )
        for index, update in enumerate(plasticity):
            for key, value in (
                ("write_gain", update.write_gain),
                ("coherence", update.coherence),
                ("eligibility", update.history[2]),
            ):
                self.running(f"{prefix}.plasticity.{key}.{index}").add(value)
            if capture_applied:
                self.running(f"{prefix}.plasticity.capture_gain.{index}").add(update.capture_gain)


class MemoryObserver(TraceObserver):
    """Observa todas las memorias neuronales de los módulos durante los pasos registrados."""

    def __init__(self, modules):
        self.memories = [
            (trace_name(f"{key}.{name}" if name else key), module)
            for key, root in modules.items()
            # La GRU candidata nativa no es un módulo de PyTorch ni contiene memorias neuronales.
            if isinstance(root, torch.nn.Module)
            for name, module in root.named_modules()
            if isinstance(module, NeuralMemory)
        ]
        self.layers = {}

    def __bool__(self):
        return bool(self.memories)

    def start(self):
        # Un paso puede reunir varios recorridos hacia delante. Lo acumulado se conserva
        # hasta `record`, que lo guarda en el paso y empieza de cero.
        for name, module in self.memories:
            if name not in self.layers:
                self.layers[name] = _Layer(name, module._reference().device)
            module.observer = self.layers[name]

    def stop(self):
        for _, module in self.memories:
            module.observer = None

    def reset(self):
        self.layers = {}

    def record(self, recorder):
        layers, self.layers = self.layers, {}
        for name, layer in layers.items():
            if layer.tokens:
                recorder.scalar("memory.tokens", float(layer.tokens), group=name)
            if layer.shared_writes:
                recorder.scalar("memory.shared.writes", float(layer.shared_writes), group=name)
                recorder.scalar("memory.shared.rows", float(layer.shared_rows), group=name)
            for metric, running in layer.values.items():
                running.record(recorder, metric, name)
