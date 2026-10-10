"""Paso de ajuste de una referencia neuronal repetido con un CUDA Graph.

El grafo captura forward, pérdida y backward con la forma del lote declarado y repite los
mismos núcleos que la ruta eager, así que la aritmética no cambia. El recorte y el
optimizador quedan fuera del grafo y leen los gradientes que este escribe.

Un lote con otra forma, como el último parcial de cada época, sigue la ruta eager. Esa ruta
crea gradientes nuevos, por eso antes de cada repetición los parámetros vuelven a apuntar a
los gradientes que escribe el grafo.

Las comprobaciones de finitud del Transformer y la de la pérdida se calculan dentro del
grafo y se leen después con una sola copia al host, en el mismo orden y con los mismos
mensajes que la ruta eager. Como el optimizador va después, ningún peso cambia si falla una.

El dropout usa el generador de CUDA registrado en el grafo, y cada repetición avanza su
desplazamiento igual que la llamada eager. El calentamiento previo a la captura sí consume
números aleatorios, así que el estado del generador se guarda antes y se restaura después.
La equivalencia bit a bit se comprueba por familia en las pruebas, no se supone.
"""

import torch

from mars_titan.models.baselines.transformer import first_flagged, nonfinite_flags

# Familias con salida y gradientes idénticos bit a bit frente a la ruta eager, según
# tests/training/test_reference_step_graph.py.
GRAPH_KINDS = ("rnn", "lstm", "gru", "dlinear", "transformer")
LOSS_MESSAGE = "La pérdida no es finita"
_WARMUP = 3


def _device_inputs(batch, device):
    inputs = {name: torch.from_numpy(value).to(device) for name, value in batch["inputs"].items()}
    presence = batch.get("presence")
    return inputs, None if presence is None else torch.from_numpy(presence).to(device)


def _signature(batch):
    arrays = [*sorted(batch["inputs"].items()), ("presence", batch.get("presence"))]
    return tuple(
        (name, None if value is None else (value.shape, value.dtype)) for name, value in arrays
    )


class ReferenceStepGraph:
    """Forward, pérdida y backward de `model` con lotes de `batch_size` filas.

    `loss` recibe la salida, el objetivo en float32 y los pesos por fila (o None) y
    devuelve la predicción puntual y la pérdida media, como la ruta eager del ajuste.
    """

    def __init__(self, model, loss, batch_size):
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("El lote capturado debe ser un entero positivo")
        self.model, self.loss, self.batch_size = model, loss, batch_size
        self.graph = self.signature = None
        self.replays = 0

    def admits(self, batch):
        """Solo el lote completo con la forma capturada repite el grafo."""
        if len(batch["target"]) != self.batch_size:
            return False
        return self.signature is None or _signature(batch) == self.signature

    def _compute(self):
        output, checks = self.model.forward_pending(self.inputs, self.presence)
        prediction, loss = self.loss(output, self.target, self.weights)
        flags = nonfinite_flags([*checks, (loss, LOSS_MESSAGE)])
        loss.backward()
        return prediction, flags, [message for _, message in checks] + [LOSS_MESSAGE]

    def _capture(self, batch, target, weights):
        device = target.device
        self.inputs, self.presence = _device_inputs(batch, device)
        self.target = target.clone()
        self.weights = None if weights is None else weights.clone()
        state = torch.cuda.get_rng_state(device)
        side = torch.cuda.Stream(device)
        side.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(side):
            for _ in range(_WARMUP):
                self.model.zero_grad(set_to_none=True)
                self._compute()
        torch.cuda.current_stream(device).wait_stream(side)
        # La captura empieza sin gradientes: el backward capturado los escribe, no los suma.
        self.model.zero_grad(set_to_none=True)
        graph = torch.cuda.CUDAGraph()
        # Solo este hilo captura. Los lectores de datos de otros hilos pueden seguir usando
        # CUDA (memoria fijada, copias) sin invalidar la captura.
        with torch.cuda.graph(graph, capture_error_mode="thread_local"):
            self.prediction, self.flags, self.messages = self._compute()
        # La captura no fija la semilla ni el desplazamiento, que el grafo lee al repetirse.
        # Basta con devolver el generador al estado anterior al calentamiento.
        torch.cuda.set_rng_state(state, device)
        self.gradients = [parameter.grad for parameter in self.model.parameters()]
        self.graph, self.signature = graph, _signature(batch)

    def step(self, batch, target, weights=None):
        """Repetir el paso con el lote y devolver la predicción puntual del grafo.

        La predicción es un búfer del grafo: debe usarse antes de la siguiente repetición.
        """
        if not self.admits(batch):
            raise ValueError("El lote no tiene la forma capturada")
        if self.graph is None:
            self._capture(batch, target, weights)
        else:
            if (weights is None) != (self.weights is None):
                raise ValueError("Los pesos por fila no corresponden a los capturados")
            for name, value in batch["inputs"].items():
                self.inputs[name].copy_(torch.from_numpy(value))
            if self.presence is not None:
                self.presence.copy_(torch.from_numpy(batch["presence"]))
            self.target.copy_(target)
            if weights is not None:
                self.weights.copy_(weights)
        for parameter, gradient in zip(self.model.parameters(), self.gradients, strict=True):
            parameter.grad = gradient
        self.graph.replay()
        self.replays += 1
        message = first_flagged(self.flags, self.messages)
        if message is not None:
            raise ValueError(message)
        return self.prediction
