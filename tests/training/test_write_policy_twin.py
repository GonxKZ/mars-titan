"""Gemelos de escritura de Titans-MAC: la misma red ajustada, evaluada escribiendo o sin escribir.

Es la prueba causal «misma red, evaluada sin escribir» (22 de noviembre de 2026). Con el estado
elegido de un ajuste `mac_online`, el gemelo `mac_frozen` repite su propia ventana sin escribir
en la memoria, y al revés. Las pruebas ajustan con el registrador de gradientes, que no mueve
pesos, y los controles emparejados parten de los parámetros de `mac_online`: cada gemelo debe
repetir entonces bit a bit la ventana que se ajustó con la otra variante. Ese es el oráculo.
"""

import hashlib
import json

import pytest
import torch

from mars_titan.models.titans.config import canonical
from mars_titan.models.titans.financial import _tensor_digest
from mars_titan.training import titans_walk_forward as wf
from mars_titan.training.checkpoints import load_training_state
from tests.training.test_titans_walk_forward import (  # noqa: F401
    _compare,
    base,
    learning_doubles_module,
    scaled_recipe,
    unfused_attention,
    window,
)

pytestmark = pytest.mark.usefixtures("learning_doubles")


def twin(anchor, view, output, **options):
    options.setdefault("regenerate", True)
    with unfused_attention():
        return wf.carry_titans(anchor, view, view, output, device="cpu", **options)


def chosen_tensors(anchor):
    """Huellas de los tensores del estado elegido de una ventana, leídas de su checkpoint."""
    report = json.loads((anchor / "run.json").read_text())
    fit = json.loads((anchor / "fit/run.json").read_text())
    state = load_training_state(
        anchor / "fit/checkpoints",
        expected_identity=fit["identity"],
        selection="best",
        expected_sha256=report["checkpoint"]["sha256"],
    )
    return {
        name: _tensor_digest(value)
        for name, value in state["model"].items()
        if isinstance(value, torch.Tensor)
    }


@pytest.mark.parametrize(
    ("fitted", "evaluated"), [("mac_online", "mac_frozen"), ("mac_frozen", "mac_online")]
)
def test_the_twin_is_the_fitted_network_evaluated_with_the_other_write_policy(
    base,  # noqa: F811
    tmp_path,
    fitted,
    evaluated,
):
    runs = base["root"] / "runs"
    receipt = twin(runs / fitted, base["view"], tmp_path / "twin", evaluated_variant=evaluated)
    switch = receipt["write_policy"]
    assert (switch["fitted_variant"], switch["evaluated_variant"]) == (fitted, evaluated)
    assert receipt["anchor"]["variant"] == fitted and receipt["regenerated"] is True
    assert set(receipt["predictions"]) == set(wf.PREDICTED)
    for partition in wf.PREDICTED:
        assert _compare(runs / evaluated, tmp_path / "twin", partition), partition
    # Escribir o no en la memoria cambia lo emitido: el gemelo no repite su propio ajuste.
    assert not all(_compare(runs / fitted, tmp_path / "twin", p) for p in wf.PREDICTED)


def test_the_twin_loads_every_tensor_of_the_chosen_state_including_the_scales(base, tmp_path):  # noqa: F811
    scaled = scaled_recipe(tmp_path)
    with unfused_attention():
        window(base["view"], base["protocol"], scaled, tmp_path / "online")
    receipt = twin(
        tmp_path / "online", base["view"], tmp_path / "twin", evaluated_variant="mac_frozen"
    )
    switch, tensors = receipt["write_policy"], chosen_tensors(tmp_path / "online")
    assert switch["tensors"] == len(tensors)
    assert switch["tensors_sha256"] == hashlib.sha256(canonical(tensors).encode()).hexdigest()
    # Las escalas de aprendizaje son buffers: la copia emparejada no las lleva y el gemelo sí.
    assert "head.output_scale" in switch["buffers"]
    assert any(name.startswith("learning_scaler.") for name in switch["buffers"])


@pytest.mark.parametrize(
    ("options", "message"),
    [
        (dict(regenerate=False, evaluated_variant="mac_frozen"), "propia ventana"),
        (dict(evaluated_variant="mac_online"), "de mac_online a mac_frozen"),
        (dict(evaluated_variant="mac_disabled"), "de mac_online a mac_frozen"),
        (dict(evaluated_variant="transformer_direct"), "de mac_online a mac_frozen"),
        (dict(evaluated_variant="mac_frozen", modality_ablation="mask_news"), "propia ventana"),
    ],
)
def test_the_write_switch_only_twins_online_and_frozen_in_the_anchor_window(
    base,  # noqa: F811
    tmp_path,
    options,
    message,
):
    with pytest.raises(ValueError, match=message):
        twin(base["root"] / "runs" / "mac_online", base["view"], tmp_path / "x", **options)
    assert not (tmp_path / "x").exists()


def _contracts(variant, mode, *, hidden=8, dtype="torch.float64", training=True, **inputs):
    inputs = dict(dict(source_sha256="a", view_sha256="b", columns=5), **inputs)
    return {
        "_extra_state": dict(
            configuration=dict(variant=variant, hidden_size=hidden, inputs=inputs),
            mac_contract=dict(memory_mode=mode, heads=4),
            training=training,
        ),
        "mac._extra_state": dict(configuration=dict(memory_mode=mode, heads=4), dtype=dtype),
    }


def test_only_the_variant_and_the_memory_mode_change_between_the_state_and_its_twin():
    weight = torch.arange(3.0)
    model = {"w": weight, **_contracts("mac_online", "online")}
    expected = {"w": torch.zeros(3), **_contracts("mac_frozen", "frozen", training=False)}
    payload = wf._switched_payload(model, expected)
    assert payload["w"] is weight
    # El modo train/eval es el del estado guardado, como en un traslado.
    assert payload["_extra_state"] == dict(expected["_extra_state"], training=True)
    assert payload["mac._extra_state"] == expected["mac._extra_state"]
    # Como en un traslado, el índice de entrada del gemelo puede tener otra huella.
    moved = {"w": torch.zeros(3), **_contracts("mac_frozen", "frozen", source_sha256="c")}
    assert wf._switched_payload(model, moved)["_extra_state"] == dict(
        moved["_extra_state"], training=True
    )
    for other in (
        _contracts("mac_frozen", "frozen", columns=6),
        _contracts("mac_frozen", "frozen", hidden=16),
        _contracts("mac_frozen", "frozen", dtype="torch.float32"),
        _contracts("mac_frozen", "disabled"),
        _contracts("mac_disabled", "frozen"),
    ):
        with pytest.raises(ValueError, match="cambia algo más que la escritura"):
            wf._switched_payload(model, {"w": torch.zeros(3), **other})
    with pytest.raises(ValueError, match="mismas entradas"):
        wf._switched_payload(model, dict(expected, extra=torch.zeros(1)))
