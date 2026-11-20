"""Presupuesto de memoria CUDA: el del portátil por defecto y el de quien llama con None."""

import subprocess
from types import SimpleNamespace

import pytest
import torch

from mars_titan.data import embeddings
from mars_titan.training import mars_titan_walk_forward, titans_walk_forward

GIB = 1024**3


@pytest.fixture
def cuda(monkeypatch):
    """Una GPU simulada de 128 GiB que registra cada fracción de memoria que se fija."""
    fractions = []
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=0))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: None)
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda index: SimpleNamespace(total_memory=128 * GIB)
    )
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda index: (100 * GIB, 128 * GIB))
    monkeypatch.setattr(
        torch.cuda,
        "set_per_process_memory_fraction",
        lambda fraction, device=None: fractions.append((fraction, device)),
    )
    return fractions


def test_the_default_budget_is_the_six_gib_of_the_laptop(cuda):
    assert embeddings.require_cuda() == torch.device("cuda:0")
    assert cuda == [(6 / 128, 0)]
    embeddings.require_cuda(max_bytes=200 * GIB)
    assert cuda[-1] == (1.0, 0)


def test_without_a_budget_the_caller_keeps_its_own_limit(cuda):
    assert embeddings.require_cuda(max_bytes=None) == torch.device("cuda:0")
    assert cuda == []
    embeddings.require_cuda(max_bytes=None, min_free_bytes=50 * GIB)
    with pytest.raises(RuntimeError, match="reserva de codificación"):
        embeddings.require_cuda(max_bytes=None, min_free_bytes=101 * GIB)
    assert cuda == []


@pytest.mark.parametrize("budget", [0, -1, 6.0, True, "6"])
def test_an_invalid_budget_is_rejected_before_touching_the_device(cuda, budget):
    with pytest.raises(ValueError, match="presupuesto CUDA"):
        embeddings.require_cuda(max_bytes=budget)
    with pytest.raises(ValueError, match="presupuesto CUDA"):
        embeddings.require_cuda(max_bytes=None, min_free_bytes=-1)
    assert cuda == []


def test_a_campaign_window_never_lowers_the_budget_of_its_campaign(cuda, monkeypatch):
    from mars_titan.training import campaign_slots

    # La campaña fija sus 16 GiB y después cada ventana exige CUDA: el tope no cambia.
    assert campaign_slots.bound_vram(16 * GIB) == 16 / 128
    assert cuda == [(16 / 128, 0)]
    titans_walk_forward.require_window_cuda()
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    mars_titan_walk_forward._cuda("cuda:0")
    assert cuda == [(16 / 128, 0)]
    # Con el valor por defecto, el asignador de la campaña quedaba en los 6 GiB del portátil.
    embeddings.require_cuda()
    assert cuda[-1] == (6 / 128, 0)
