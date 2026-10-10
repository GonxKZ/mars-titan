"""Misma precisión numérica en todas las familias, sin abrir datos ni usar la GPU.

La comparación usa FP32 estricto en todas las familias: `float32_matmul_precision = highest`,
sin TF32 en cuBLAS ni en cuDNN y sin autocast. Las recetas no fijan hoy esos indicadores. Cada
entrenador los lee del proceso, los guarda en su identidad y la vuelve a comprobar al
reanudar. Se comprueba que todas las familias registran los tres indicadores leídos del
proceso con el mismo valor, que una receta solo puede declarar FP32 y que ningún módulo del
paquete activa autocast, escalado de pérdidas ni tipos de 16 bits.
"""

import json
import re
from pathlib import Path

import pytest
import torch

from mars_titan.training import candidate_run, financial_run, mars_titan_run, reference_run

STRICT = ("highest", False, False)
# Así guarda cada entrenador en su identidad los tres indicadores, que son la precisión de
# matmul FP32, TF32 en cuBLAS y TF32 en cuDNN.
TRAINERS = {
    "titans_mac_and_cm_v1_cores": lambda: _flags(
        financial_run._numerics(), "matmul_precision", "matmul_allow_tf32", "cudnn_allow_tf32"
    ),
    "mars_titan_and_cm_v1_arms": lambda: _flags(
        mars_titan_run._numerics(), "matmul_precision", "matmul_allow_tf32", "cudnn_allow_tf32"
    ),
    "gru_episodic": lambda: _flags(
        candidate_run._numerics(), "matmul_precision", "matmul_allow_tf32", "cudnn_allow_tf32"
    ),
    "neural_references": lambda: _flags(
        reference_run.scientific_identity()["numerics"],
        "float32_matmul_precision",
        "cuda_matmul_allow_tf32",
        "cudnn_allow_tf32",
    ),
}
RECIPES = [
    Path("configs/titans/chronological-training-historical-masked.json"),
    Path("configs/titans/episodic-readout-historical-masked.json"),
    Path("configs/titans/cm-v1-factorial.json"),
    Path("configs/titans/mars-titan-extensions.json"),
    Path("configs/candidate/chronological-training.json"),
    *sorted(Path("configs/baselines").glob("historical-masked-campaign-*.json")),
]
# Estos son los valores admitidos para las claves de precisión que una receta pueda declarar.
ALLOWED = {
    "dtype": {"float32"},
    "precision": {"float32", "fp32_strict"},
    "float32_matmul_precision": {"highest"},
    "matmul_precision": {"highest"},
}
FORBIDDEN_KEYS = {"autocast", "amp", "mixed_precision", "allow_tf32", "tf32", "loss_scale"}
REDUCED = re.compile(
    r"torch\.autocast|torch\.(cuda|cpu)\.amp|torch\.amp\b|GradScaler|\.half\(\)|\.bfloat16\(\)"
    r"|torch\.(float16|bfloat16|half)\b|\"(float16|bfloat16)\""
)


def _flags(numerics, precision, matmul, cudnn):
    return numerics[precision], numerics[matmul], numerics[cudnn]


@pytest.fixture
def process_flags(monkeypatch):
    """Fijar los tres indicadores del proceso y restaurarlos al terminar."""
    previous = (
        torch.get_float32_matmul_precision(),
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
    )
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *_: "sin GPU en la prueba")

    def apply(precision, matmul, cudnn):
        torch.set_float32_matmul_precision(precision)
        torch.backends.cuda.matmul.allow_tf32 = matmul
        torch.backends.cudnn.allow_tf32 = cudnn

    yield apply
    apply(*previous)


@pytest.mark.parametrize("flags", [STRICT, ("highest", False, True), ("high", True, True)], ids=str)
def test_every_family_records_the_three_indicators_read_from_the_process(process_flags, flags):
    process_flags(*flags)
    recorded = {family: read() for family, read in TRAINERS.items()}
    assert recorded == dict.fromkeys(TRAINERS, flags)


def _declared(value, path=()):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _declared(item, (*path, key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _declared(item, (*path, index))
    elif path:
        yield path, value


def _violations(document):
    """Devuelve las claves de precisión de una receta con un valor distinto de FP32."""
    found = []
    for path, value in _declared(document):
        key = path[-1] if isinstance(path[-1], str) else path[-2]
        if (key in FORBIDDEN_KEYS and value not in (False, None)) or (
            key in ALLOWED and value not in ALLOWED[key]
        ):
            found.append(path)
    return found


@pytest.mark.parametrize("recipe", RECIPES, ids=lambda path: path.stem)
def test_recipes_only_declare_fp32(recipe):
    assert _violations(json.loads(recipe.read_text())) == []


@pytest.mark.parametrize(
    "document",
    [
        {"dtype": "bfloat16"},
        {"precision": "tf32"},
        {"training": {"autocast": True}},
        {"cases": [{"float32_matmul_precision": "high"}]},
        {"selection": {"allow_tf32": True}},
    ],
    ids=str,
)
def test_reduced_precision_in_a_recipe_is_reported(document):
    assert len(_violations(document)) == 1


def test_no_module_enables_autocast_loss_scaling_or_16_bit_types():
    sources = sorted(Path("src/mars_titan").rglob("*.py"))
    assert len(sources) > 100
    found = [
        f"{path}:{number}"
        for path in sources
        for number, line in enumerate(path.read_text().splitlines(), 1)
        if REDUCED.search(line)
    ]
    assert found == []
    for sample in ("with torch.autocast('cuda'):", "x.half()", "dtype=torch.bfloat16"):
        assert REDUCED.search(sample)
