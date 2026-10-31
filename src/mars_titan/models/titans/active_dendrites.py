"""Operador contextual N2, todavía sin productor temporal ni conexión financiera.

Para la unidad i y el segmento j, r_ij = U_ij c. Se elige el primer segmento
con mayor |r_ij| y se conserva su signo en d_i. La adaptación financiera usa
h'_i = h_i [1 + tanh(a_i) tanh(d_i / 2)], con a_i inicialmente cero.

Iyer y otros (2022), sección 3, usan h_i sigmoid(d_i) y añaden kWTA y pesos
dispersos. Aquí se conserva su selección firmada, pero la ganancia residual
permite paridad inicial y con contexto nulo. No se implementan aquellas
otras piezas ni se heredan sus resultados. La fecha del contexto corresponde
al productor, aún pendiente, y no puede validarse a partir de sus valores.
"""

import math

import torch
from torch import nn

from .config import MAX_BLOCK_ROWS, bounded_integer
from .state import check_finite

MAX_PARAMETER_ELEMENTS = 2**20
MAX_RESPONSE_ELEMENTS = 2**20


class ActiveDendrites(nn.Module):
    """Modular cada fila sin escribir memoria ni cambiar parámetros en forward."""

    def __init__(self, hidden_size, context_size, *, segments=4, seed=42, dtype=torch.float32):
        super().__init__()
        bounded_integer(hidden_size, "unidades de N2", 1, 4096)
        bounded_integer(context_size, "dimensión del contexto de N2", 1, 4096)
        bounded_integer(segments, "segmentos de N2", 1, 64)
        bounded_integer(seed, "semilla de N2", 0, 2**32 - 1)
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("N2 exige float32 o float64 explícitos")
        if hidden_size * (segments * context_size + 1) > MAX_PARAMETER_ELEMENTS:
            raise ValueError("Los parámetros de N2 superan su presupuesto de elementos")
        self.hidden_size, self.context_size, self.segments, self.seed = (
            hidden_size,
            context_size,
            segments,
            seed,
        )
        generator = torch.Generator(device="cpu").manual_seed(seed)
        weights = torch.empty(hidden_size, segments, context_size, device="cpu", dtype=dtype)
        radius = 1 / math.sqrt(context_size)
        weights.uniform_(-radius, radius, generator=generator)
        self.segment_weights = nn.Parameter(weights)
        self.gain = nn.Parameter(torch.zeros(hidden_size, device="cpu", dtype=dtype))

    def identity(self):
        return dict(
            kind="active_dendrites_v1",
            scope="neural_operator_only_pending_temporal_integration",
            hidden_size=self.hidden_size,
            context_size=self.context_size,
            segments=self.segments,
            seed=self.seed,
            selection="first_signed_absolute_maximum",
            modulation="h_times_one_plus_tanh_gain_times_tanh_half_selected_response",
            initial="random_segments_and_zero_gain",
            state="none",
            context_provenance="not_checked_by_this_neural_operator",
            max_parameter_elements=MAX_PARAMETER_ELEMENTS,
            max_response_elements=MAX_RESPONSE_ELEMENTS,
        )

    def _validate(self, values, context):
        reference = self.segment_weights
        if reference.dtype not in (torch.float32, torch.float64) or torch.is_autocast_enabled(
            reference.device.type
        ):
            raise ValueError("N2 exige precisión explícita, sin autocast")
        if (
            reference.shape != (self.hidden_size, self.segments, self.context_size)
            or self.gain.shape != (self.hidden_size,)
            or self.gain.dtype != reference.dtype
            or self.gain.device != reference.device
        ):
            raise ValueError("Los parámetros de N2 no corresponden a su geometría")
        for tensor, width in ((values, self.hidden_size), (context, self.context_size)):
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.layout != torch.strided
                or tensor.ndim != 2
                or tensor.shape[1] != width
                or tensor.dtype != reference.dtype
                or tensor.device != reference.device
            ):
                raise ValueError(
                    "Las entradas de N2 tienen forma, precisión o dispositivo inválidos"
                )
            check_finite(tensor, "Una entrada de N2")
        if values.shape[0] != context.shape[0]:
            raise ValueError("El contexto de N2 debe conservar las filas de la representación")
        bounded_integer(values.shape[0], "filas de N2", 0, MAX_BLOCK_ROWS)
        if values.shape[0] * self.hidden_size * self.segments > MAX_RESPONSE_ELEMENTS:
            raise ValueError("Las respuestas de N2 superan su presupuesto de elementos")
        check_finite(reference, "Los segmentos de N2")
        check_finite(self.gain, "Las ganancias de N2")

    def forward(self, values, context):
        self._validate(values, context)
        responses = torch.einsum("bc,hsc->bhs", context, self.segment_weights)
        check_finite(responses, "Las respuestas dendríticas de N2")
        winners = responses.abs().argmax(dim=-1, keepdim=True)
        selected = responses.gather(-1, winners).squeeze(-1)
        # tanh(d/2) equivale a 2 sigmoid(d) - 1, sin la resta cercana a cero.
        output = values * (1 + self.gain.tanh() * (selected * 0.5).tanh())
        check_finite(output, "La salida de N2")
        return output
