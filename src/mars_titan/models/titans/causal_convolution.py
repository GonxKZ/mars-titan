"""Convolución causal en profundidad con ventana explícita, sin estado oculto en el módulo."""

import math

import torch
from torch import Tensor, nn


class CausalDepthwiseConvolution(nn.Module):
    """u_t[c] = Σ_j w[c, j] · p_{t−K+1+j}[c] para cada canal c, sin sesgo.

    La ventana guarda las K − 1 proyecciones anteriores del mismo flujo y empieza en ceros,
    que equivale al relleno solo por la izquierda de ShortConvolution. La suma se hace en
    orden fijo de j con operaciones elemento a elemento, así que cada posición produce los
    mismos bits tanto si la secuencia llega entera como si llega en trozos.
    """

    def __init__(self, dim: int, kernel: int, *, dtype: torch.dtype) -> None:
        super().__init__()
        # Disposición [canales, 1, K] de nn.Conv1d con groups=dim y su inicialización por
        # defecto, que hereda ShortConvolution: U(−1/√K, 1/√K).
        self.weight = nn.Parameter(torch.empty(dim, 1, kernel, dtype=dtype))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, values: Tensor, window: Tensor) -> tuple[Tensor, Tensor]:
        """values [lote, T, dim] y ventana [lote, K − 1, dim]. Devuelve u y la ventana nueva."""
        length = values.shape[1]
        extended = torch.cat((window, values), dim=1)
        kernel = self.weight[:, 0]
        output = extended[:, :length] * kernel[:, 0]
        for offset in range(1, kernel.shape[1]):
            output = output + extended[:, offset : offset + length] * kernel[:, offset]
        return output, extended[:, length:].clone(memory_format=torch.contiguous_format)
