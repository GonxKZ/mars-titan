"""Empaquetado cuDNN de la GRU candidata nativa, sin optimizador ni datos reales.

En CUDA los cuatro pesos de la GRU deben ser vistas de un único bloque con el formato de cuDNN,
para que `at::gru` no los compacte en cada llamada. El empaquetado copia valores exactos, así
que salidas y gradientes coinciden bit a bit con la ruta que compacta. En CPU no cambia nada.
"""

import os
import sys
import tempfile
import warnings

import pytest
import torch

from mars_titan.memory.native_backend import load_native

pytestmark = pytest.mark.skipif(
    not os.environ.get("MARS_TITAN_EPISODIC_NATIVE"), reason="Falta el enlace nativo compilado"
)
GRU = ("price_weight_ih", "price_weight_hh", "price_bias_ih", "price_bias_hh")
WARNING = "contiguous chunk of memory"


def candidate(dtype, device):
    native = load_native()
    config = native.CandidateConfig()
    config.dimensions = [5, 3, 6, 3, 3]
    config.normalization_id = "manual-historical-v1"
    config.input_policy = "historical_masked_2000_v1"
    return native, native.Candidate(config, dtype, device)


def inputs(native, dtype, device, rows=4):
    generator = torch.Generator().manual_seed(7)
    dtype = getattr(torch, dtype)

    def normal(*shape):
        return torch.randn(*shape, generator=generator, dtype=dtype).to(device)

    # Bloques con valores, observación y edad coherentes con la política histórica.
    def block(width):
        values = normal(rows, width)
        return torch.cat((values, torch.ones_like(values), torch.zeros_like(values)), 1)

    return native.CandidateInputs(
        normal(rows, 64, 5),
        normal(rows, 3),
        normal(rows, 6),
        block(1),
        block(1),
        torch.ones(rows, 5, dtype=torch.bool, device=device),
    )


def cuda_candidate(dtype):
    if not torch.cuda.is_available():
        pytest.skip("Comprobación cuda:0 sin dispositivo visible")
    try:
        return candidate(dtype, "cuda:0")
    except ValueError as error:
        if "sin backend CUDA" in str(error):
            pytest.skip("El enlace se compiló sin backend CUDA")
        raise


def stderr_warnings(function):
    """Contar el aviso de cuDNN como aviso de Python o en el descriptor 2.

    Las funciones de PyTorch lo convierten en un aviso de Python. El enlace pybind del
    candidato no instala ese manejador y LibTorch lo escribe directamente en stderr.
    """
    sys.stderr.flush()
    saved = os.dup(2)
    with tempfile.TemporaryFile(mode="w+b") as sink:
        os.dup2(sink.fileno(), 2)
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = function()
        finally:
            sys.stderr.flush()
            os.dup2(saved, 2)
            os.close(saved)
        sink.seek(0)
        text = sink.read().decode(errors="replace")
    return result, text.count(WARNING) + sum(WARNING in str(item.message) for item in caught)


def packed(parameters):
    """Los cuatro pesos son vistas contiguas y consecutivas de un único almacenamiento."""
    values = [parameters[name] for name in GRU]
    storage = values[0].untyped_storage()
    offset = 0
    for value in values:
        if (
            value.untyped_storage().data_ptr() != storage.data_ptr()
            or not value.is_contiguous()
            or value.storage_offset() != offset
        ):
            return False
        offset += value.numel()
    return storage.nbytes() == offset * values[0].element_size()


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_cuda_weights_are_one_cudnn_block_and_survive_in_place_copies(dtype):
    native, model = cuda_candidate(dtype)
    parameters = model.named_parameters()
    assert packed(parameters)
    # La copia en el sitio de `candidate_run._load_parameters` conserva el bloque.
    with torch.no_grad():
        for name in GRU:
            parameters[name].copy_(parameters[name].detach().clone() * 0.5)
    current = model.named_parameters()
    assert packed(current)
    assert all(current[name] is parameters[name] for name in GRU)
    data = inputs(native, dtype, "cuda:0")
    model.train()

    def step():
        output = model.forward(data, model.empty_memory(), 2)
        output.quantiles.square().sum().backward()
        return output

    _, warned = stderr_warnings(step)
    assert warned == 0
    assert all(current[name].grad is not None for name in GRU)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("training", [True, False])
def test_packed_gru_matches_the_compacting_path_bit_for_bit(dtype, training):
    """Mismos pesos en bloque y en tensores sueltos: la ruta que compacta es la referencia."""
    native, model = cuda_candidate(dtype)
    weights = [model.named_parameters()[name] for name in GRU]
    loose = [value.detach().clone().requires_grad_(True) for value in weights]
    assert not packed(dict(zip(GRU, loose, strict=True)))
    prices = torch.randn(8, 64, 5, dtype=getattr(torch, dtype), device="cuda:0")
    start = torch.zeros(1, 8, 128, dtype=prices.dtype, device=prices.device)

    def gru(values):
        return torch.gru(prices, start, values, True, 1, 0.0, training, False, True)

    (expected, expected_hidden), warned = stderr_warnings(lambda: gru(loose))
    (observed, observed_hidden), clean = stderr_warnings(lambda: gru(weights))
    assert warned >= 1 and clean == 0
    assert torch.equal(observed, expected) and torch.equal(observed_hidden, expected_hidden)
    if training:
        expected.square().sum().backward()
        observed.square().sum().backward()
        for reference, value in zip(loose, weights, strict=True):
            assert torch.equal(value.grad, reference.grad)


def test_restored_cuda_candidate_is_packed_and_reproduces_its_outputs():
    native, model = cuda_candidate("float32")
    data = inputs(native, "float32", "cuda:0")
    expected = model.forward(data, model.empty_memory(), 1).quantiles
    restored = native.Candidate.load_state(
        model.save_state(), "cuda:0", "historical_masked_2000_v1"
    )
    assert packed(restored.named_parameters())
    assert restored.parameter_fingerprint() == model.parameter_fingerprint()
    assert torch.equal(restored.forward(data, restored.empty_memory(), 1).quantiles, expected)
    on_cpu = native.Candidate.load_state(model.save_state(), "cpu", "historical_masked_2000_v1")
    assert on_cpu.parameter_fingerprint() == model.parameter_fingerprint()


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_cpu_weights_keep_their_own_storage(dtype):
    _, model = candidate(dtype, "cpu")
    parameters = model.named_parameters()
    assert len({parameters[name].untyped_storage().data_ptr() for name in GRU}) == len(GRU)
