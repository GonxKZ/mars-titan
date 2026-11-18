"""Alternativa local de dos transferencias H2D para el camino FP32 de la campaña."""

import numpy as np
import torch

from mars_titan.models.titans.financial_inputs import CPUDecisionBatch, device_tensor


def grouped(cls, batch, *, device="cuda:0", dtype=torch.float32):
    if (
        type(batch) is not CPUDecisionBatch
        or torch.device(device) != torch.device("cuda:0")
        or dtype != torch.float32
    ):
        raise ValueError("Esta prueba requiere el lote CUDA FP32 de la campaña")
    batch.verify()
    locations = {}
    end = 0
    for key, array in batch.inputs.items():
        if array.dtype != np.float32 or not array.flags.c_contiguous:
            raise ValueError("Entrada FP32 no contigua")
        offset = (end + 63) // 64 * 64
        locations[key] = (offset, array.size, array.shape)
        end = offset + array.size
    staging = torch.empty(end, dtype=torch.float32, pin_memory=True)
    buf = staging.numpy()
    previous = 0
    for key, (offset, n, shape) in locations.items():
        if offset > previous:
            buf[previous:offset] = 0
        buf[offset : offset + n].reshape(shape)[...] = batch.inputs[key]
        previous = offset + n
    dense = staging.to(device=device, non_blocking=True)
    tensors = {k: dense.narrow(0, off, n).view(shape) for k, (off, n, shape) in locations.items()}
    return cls._create(
        tensors,
        device_tensor(batch.presence, device, torch.bool),
        batch.flow_ids,
        batch.sample_ids,
        batch.prediction_at,
        batch.input_available_at,
        batch.input_contract_id,
        batch.input_digest,
    )
