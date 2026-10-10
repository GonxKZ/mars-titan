"""PT6: fusión condicionada por la presencia y su alternativa trivial, sin optimizadores.

Las pruebas calculan salidas y gradientes con lotes técnicos. Para comprobar la modulación
con pesos distintos de cero se asignan valores fijos a mano, sin ningún paso de ajuste.
"""

import json
from pathlib import Path

import pytest
import torch

from mars_titan.models.baselines.multimodal import MODALITIES
from mars_titan.models.titans.presence_modulation import (
    FILM,
    WIDE_FUSION,
    PresenceFiLM,
    film_parameters,
    wide_parameters_per_unit,
    wide_units,
)
from tests.models.titans.test_financial_adapter import api, raw_batch, specification

ROOT = Path(__file__).resolve().parents[3]
DECLARATION = ROOT / "configs/evaluation/presence-modulation-comparison.json"
SHARED = ("price_encoder", "encoders", "fusion", "head", "mac")


def build(modulation=None, *, variant="mac_online", seed=42, historical=True):
    config = api().FinancialConfig(
        specification(historical=historical),
        variant=variant,
        hidden_size=32,
        layers=1,
        seed=seed,
        presence_modulation=modulation,
    )
    return api().FinancialPredictor(config, dtype=torch.float64)


def decisions(*, complete=False):
    raw = raw_batch(absent=not complete)
    return api().DecisionBatch.from_corpus(raw, specification(), dtype=torch.float64)


def prepare(model, batch, *, differentiable=False):
    state = model.initial_state(batch.flow_ids, differentiable=differentiable)
    return model.prepare(batch, state, differentiable=differentiable)


def fill(model, prefix, seed=5):
    """Cargar valores fijos distintos de cero en un submódulo, sin optimizador."""
    generator = torch.Generator().manual_seed(seed)
    state = model.state_dict()
    for name, value in state.items():
        if name.startswith(prefix + "."):
            state[name] = 0.3 * torch.randn(value.shape, generator=generator, dtype=value.dtype)
    model.load_state_dict(state)


def fusion_input(model, batch):
    observed = {}
    handle = model.fusion[0].register_forward_pre_hook(
        lambda module, arguments: observed.update(value=arguments[0].detach().clone())
    )
    try:
        prepared = prepare(model, batch)
    finally:
        handle.remove()
    return observed["value"], prepared


@pytest.mark.parametrize("modulation", [FILM, WIDE_FUSION])
@pytest.mark.parametrize("complete", [True, False])
def test_both_options_start_exactly_at_the_control_with_the_same_seed(modulation, complete):
    control, option = build(), build(modulation)
    for name, value in control.state_dict().items():
        if name.split(".", 1)[0] in SHARED:
            assert torch.equal(option.state_dict()[name], value), name
    batch = decisions(complete=complete)
    expected, actual = prepare(control, batch), prepare(option, batch)
    torch.testing.assert_close(actual.detached_tokens, expected.detached_tokens, rtol=0, atol=0)
    torch.testing.assert_close(actual.point_predictions, expected.point_predictions, rtol=0, atol=0)


def test_film_is_the_identity_on_complete_rows_whatever_its_weights():
    control, film = build(), build(FILM)
    fill(film, "presence_modulation")
    batch = decisions()
    assert batch.presence[0].all() and not batch.presence[1].all()
    expected, actual = prepare(control, batch), prepare(film, batch)
    # La fila completa no cambia. La incompleta sí, porque ahora la modulación actúa.
    torch.testing.assert_close(
        actual.detached_tokens[0], expected.detached_tokens[0], rtol=0, atol=0
    )
    assert not torch.equal(actual.detached_tokens[1], expected.detached_tokens[1])


def test_film_rescales_present_blocks_by_the_absence_of_others_and_keeps_absent_ones_zero():
    film = build(FILM)
    fill(film, "presence_modulation")
    batch = decisions()
    observed, _ = fusion_input(film, batch)
    hidden = 32
    presence = batch.presence
    for index in range(len(MODALITIES)):
        block = observed[1, index * hidden : (index + 1) * hidden]
        if not presence[1, index]:
            assert torch.count_nonzero(block) == 0
    torch.testing.assert_close(observed[:, -5:], presence.double(), rtol=0, atol=0)
    # Fórmula escrita a mano para la fila incompleta: z' = p (z + (Γa) z + Βa).
    control = build()
    reference, _ = fusion_input(control, batch)
    blocks = reference[1, :-5].reshape(len(MODALITIES), hidden)
    absence = (~presence[1]).double()
    module = film.presence_modulation
    scale = module.scale.detach() @ absence
    shift = module.shift.detach() @ absence
    expected = presence[1].double()[:, None] * (blocks + scale * blocks + shift)
    torch.testing.assert_close(observed[1, :-5], expected.flatten(), rtol=1e-12, atol=1e-12)


def test_film_gradient_comes_only_from_rows_with_some_absence():
    complete = build(FILM)
    prepare(
        complete, decisions(complete=True), differentiable=True
    ).point_predictions.sum().backward()
    for parameter in complete.presence_modulation.parameters():
        assert parameter.grad is None or torch.count_nonzero(parameter.grad) == 0
    partial = build(FILM)
    prepare(partial, decisions(), differentiable=True).point_predictions.sum().backward()
    absent = (~decisions().presence).any(dim=0)
    for parameter in partial.presence_modulation.parameters():
        assert parameter.grad.abs().sum() > 0
        # Solo las columnas de modalidades que faltan en alguna fila pueden recibir gradiente.
        assert torch.count_nonzero(parameter.grad[:, :, ~absent]) == 0


def test_wide_branch_learns_through_its_output_and_starts_without_effect():
    wide = build(WIDE_FUSION)
    prepare(wide, decisions(), differentiable=True).point_predictions.sum().backward()
    branch = wide.fusion_branch
    # Con D en cero solo D recibe gradiente en el primer paso, como en una capa nueva.
    assert branch.project.grad.abs().sum() > 0
    assert torch.count_nonzero(branch.expand.grad) == 0
    fill(wide, "fusion_branch")
    control = build()
    batch = decisions(complete=True)
    # Sin interacción con la presencia la rama cambia también las filas completas.
    assert not torch.equal(
        prepare(wide, batch).detached_tokens, prepare(control, batch).detached_tokens
    )


@pytest.mark.parametrize("hidden", [32, 64, 128])
def test_parameter_counts_are_recorded_and_the_wide_branch_matches_film_within_five_percent(hidden):
    assert film_parameters(hidden) == 2 * 5 * hidden * 5
    film = PresenceFiLM(hidden)
    assert sum(p.numel() for p in film.parameters()) == film_parameters(hidden)
    units = wide_units(hidden)
    wide = units * wide_parameters_per_unit(hidden)
    assert abs(wide - film_parameters(hidden)) / film_parameters(hidden) < 0.05
    # Ninguna otra cantidad entera de unidades queda más cerca de FiLM.
    for other in (units - 1, units + 1):
        distance = abs(other * wide_parameters_per_unit(hidden) - film_parameters(hidden))
        assert distance >= abs(wide - film_parameters(hidden))


def test_identity_records_the_option_and_its_parameters_only_when_declared():
    control, film, wide = build(), build(FILM), build(WIDE_FUSION)
    assert "presence_modulation" not in control.config.identity()
    assert film.config.identity()["presence_modulation"] == dict(
        name=FILM, parameters=1600, initial="zero"
    )
    recorded = wide.config.identity()["presence_modulation"]
    assert recorded["name"] == WIDE_FUSION and recorded["film_parameters"] == 1600
    assert recorded["parameters"] == sum(p.numel() for p in wide.fusion_branch.parameters())
    added = {name for name in film.state_dict()} - {name for name in control.state_dict()}
    assert added == {"presence_modulation.scale", "presence_modulation.shift"}
    added = {name for name in wide.state_dict()} - {name for name in control.state_dict()}
    assert added == {"fusion_branch.expand", "fusion_branch.project"}
    with pytest.raises(ValueError, match="mismo contrato|identidad|contrato"):
        film.load_state_dict(wide.state_dict())


@pytest.mark.parametrize("value", ["film", "wide", 1, True])
def test_unknown_options_and_the_strict_policy_are_rejected(value):
    with pytest.raises(ValueError, match="presence_modulation"):
        build(value)
    with pytest.raises(ValueError, match="presence_modulation"):
        build(FILM, historical=False)


def test_comparison_is_declared_with_its_control_rule_and_budget_but_not_executed():
    declaration = json.loads(DECLARATION.read_text())
    recipe = json.loads((ROOT / declaration["base_recipe"]).read_text())
    assert declaration["status"] == "declared_not_executed"
    assert declaration["final_test_opened"] is False and declaration["executions"] == 0
    roles = {name: arm["role"] for name, arm in declaration["arms"].items()}
    assert sorted(roles.values()) == ["control", "innovation", "trivial_alternative"]
    base = {key: value for key, value in recipe["predictor"].items() if key != "dtype"}
    identities = {}
    for arm in declaration["arms"].values():
        config = api().FinancialConfig(
            specification(), variant=declaration["variant"], **(base | arm["predictor_changes"])
        )
        identities[arm["role"]] = config.identity()
    control = identities["control"]
    for role in ("innovation", "trivial_alternative"):
        changed = {key for key in identities[role] if identities[role][key] != control.get(key)}
        assert changed == {"presence_modulation"}
    assert identities["innovation"]["presence_modulation"]["name"] == FILM
    assert identities["trivial_alternative"]["presence_modulation"]["name"] == WIDE_FUSION
    # Igual presupuesto de búsqueda, selección y parada para los tres modelos.
    assert {"search_cases", "optimizer_updates", "stopping_rule", "seeds"} <= set(
        declaration["equal"]
    )
    budget = declaration["budget"]
    assert budget["consumed"] == dict(gpu_hours=0, executions=0, optimizer_updates=0)
    assert declaration["pilot"]["gpu_hours_max"] == budget["declared"]["gpu_hours_max"] == 6
    assert declaration["metric"]["rows"] == "rows_with_some_absent_modality"
