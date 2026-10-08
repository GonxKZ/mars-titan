"""Consulta congelada de palabras en CPU con activaciones en el dispositivo del modelo."""

import torch


def check_cpu_word_table(table):
    if (
        type(table) is not torch.nn.Embedding
        or table.training
        or table.weight.requires_grad
        or table.weight.device.type != "cpu"
        or table.weight.dtype != torch.float32
        or table.max_norm is not None
    ):
        raise ValueError("La tabla de palabras debe estar congelada en CPU y FP32")


def cpu_word_values(table, indices, destination: torch.device):
    """Consulta un lote acotado antes de trasladarlo, sin mover los IDs a CUDA."""
    if torch.is_grad_enabled():
        raise ValueError("La consulta CPU solo admite inferencia congelada")
    check_cpu_word_table(table)
    if (
        indices.device.type != "cpu"
        or indices.dtype != torch.int64
        or indices.ndim != 2
        or not 1 <= indices.shape[0] <= 32
        or not 1 <= indices.shape[1] <= 128
    ):
        raise ValueError("La consulta de palabras supera el contrato de fragmentos")
    return table(indices).to(device=destination)


def place_cpu_word_embeddings(model, destination: torch.device):
    """Mueve el modelo sin materializar la tabla completa en el dispositivo de destino."""
    table = model.get_input_embeddings()
    check_cpu_word_table(table)
    model.set_input_embeddings(torch.nn.Identity())
    try:
        model.to(destination)
    finally:
        model.set_input_embeddings(table)
    return model
