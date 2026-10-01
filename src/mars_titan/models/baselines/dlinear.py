"""Descomposición DLinear dentro de la ventana observada, con salida de un paso."""

# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 DLinear Authors. All rights reserved.
# Adaptación con validación de formas y salida por canal para la fusión multimodal.

import torch
from torch import nn
from torch.nn import functional as F


class DLinear(nn.Module):
    """Dos mapas temporales compartidos entre canales, sin consultar el futuro."""

    def __init__(self, context: int = 64, kernel_size: int = 25):
        super().__init__()
        if (
            type(context) is not int
            or type(kernel_size) is not int
            or context < 2
            or not 1 <= kernel_size <= context
            or kernel_size % 2 != 1
        ):
            raise ValueError("La ventana debe contener una media móvil impar y positiva")
        self.context, self.kernel_size = context, kernel_size
        self.seasonal = nn.Linear(context, 1)
        self.trend = nn.Linear(context, 1)

    def decompose(self, values: torch.Tensor):
        if values.ndim != 3 or values.shape[1] != self.context or values.shape[2] < 1:
            raise ValueError("La entrada debe ser una ventana [lote, contexto, canales]")
        padding = self.kernel_size // 2
        padded = torch.cat(
            [values[:, :1].expand(-1, padding, -1), values, values[:, -1:].expand(-1, padding, -1)],
            dim=1,
        )
        trend = F.avg_pool1d(padded.transpose(1, 2), self.kernel_size, stride=1).transpose(1, 2)
        return values - trend, trend

    def forward(self, values: torch.Tensor):
        seasonal, trend = self.decompose(values)
        return (
            self.seasonal(seasonal.transpose(1, 2)) + self.trend(trend.transpose(1, 2))
        ).squeeze(-1)
