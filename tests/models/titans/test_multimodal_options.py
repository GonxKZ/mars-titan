"""Opciones multimodales WP6: subbloque técnico en `charts` y compuerta de presencia.

Las pruebas siguen desarrollo guiado por pruebas en CPU (FP32 y FP64), sin optimizadores ni
modificar el comportamiento por defecto.
"""

import copy
import math
from dataclasses import asdict

import numpy as np
import pytest
import torch

from mars_titan.data.input_policy import HISTORICAL_MASKED, STRICT_INPUTS, policy_identity
from mars_titan.models.baselines.multimodal import MODALITIES
from mars_titan.models.learning_scale import (
    RELATIVE_FLOOR,
    LearningScale,
    LearningScaler,
    fit_statistics,
    parse_learning_scale,
)
from mars_titan.models.quantile_head import QUANTILE_HEAD, pinball_loss
from mars_titan.models.titans.financial import (
    VARIANTS,
    DecisionBatch,
    FinancialConfig,
    FinancialInputSpec,
    FinancialPredictor,
    copy_paired_parameters,
)
from mars_titan.models.titans.frozen_financial import FrozenFinancialConsumer
from mars_titan.models.titans.presence_modulation import (
    MACRO_FILM,
    MODALITY_GATINGS,
    PRESENCE_SOFTMAX,
    PresenceModalityGate,
    modality_gating_parameters,
)
from mars_titan.models.titans.regional_fusion import RegionalFusionConfig
from mars_titan.models.titans.session_ic_objective import session_correlations
from mars_titan.models.titans.technical_branch import TechnicalDenseConfig
from mars_titan.training.financial_run import ChronologicalInference, ChronologicalRecipe
from mars_titan.training.search_cases import checked_predictor_changes

# 8 columnas visuales + 4 columnas técnicas al final de `charts` (= 12).
VISUAL_COLS = 8
TECH_COLS = 4
CHART_COLS = VISUAL_COLS + TECH_COLS
DIMS = dict(prices=5, news=4, charts=CHART_COLS, fundamentals=6, macro=6)
HIDDEN = 32
BASE_SCALE = dict(clip=5.0, price_returns=True, zero_median=True)
SPLIT_SCALE = dict(BASE_SCALE, chart_technical_columns=TECH_COLS)
DENSE_BRANCH = TechnicalDenseConfig(columns=TECH_COLS, hidden=16, depth=2)


@pytest.fixture(autouse=True)
def explicit_fastpath():
    previous = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    yield
    torch.backends.mha.set_fastpath_enabled(previous)


def spec(*, historical=True):
    policy = HISTORICAL_MASKED if historical else STRICT_INPUTS
    representation = dict(
        **policy_identity(policy),
        fundamental_concepts=["assets", "equity"],
        macro_indicators=["rate", "cpi"],
        encoders={"fixture": "frozen-v1"},
        representation_code={"fixture.py": "c" * 64},
        text_aggregation="mean",
        context_sessions=64,
        news_lookback_sessions=5,
    )
    return FinancialInputSpec(
        source_sha256="a" * 64,
        view_sha256="b" * 64,
        representation=representation,
        dimensions=DIMS,
        input_policy=policy,
    )


def fit_corpus(*, rows=160, seed=17, visual_scale=20.0, tech_scale=0.05):
    """Lote de ajuste donde la escala visual y la técnica difieren en varios órdenes."""
    rng = np.random.default_rng(seed)
    close = np.cumsum(rng.normal(scale=0.015, size=(rows, 64)), axis=1)
    close -= close[:, :1]
    opening = close + rng.normal(scale=0.004, size=close.shape)
    high = np.maximum(opening, close) + np.abs(rng.normal(scale=0.003, size=close.shape))
    low = np.minimum(opening, close) - np.abs(rng.normal(scale=0.003, size=close.shape))
    volume = np.log1p(np.abs(rng.normal(size=close.shape)))
    prices = np.stack([opening, high, low, close, volume], axis=-1).astype(np.float32)

    news = rng.normal(0.5, 1.5, size=(rows, DIMS["news"])).astype(np.float32)
    visual = rng.normal(2.0, visual_scale, size=(rows, VISUAL_COLS)).astype(np.float32)
    technical = rng.normal(-0.2, tech_scale, size=(rows, TECH_COLS)).astype(np.float32)
    # Última columna técnica casi constante para ejercitar su propio suelo relativo.
    technical[:, -1] = 0.0
    charts = np.concatenate([visual, technical], axis=1)

    presence = np.ones((rows, 5), dtype=np.bool_)
    presence[::5, 1] = False
    news[::5] = 0.0

    inputs = dict(prices=prices, news=news, charts=charts)
    for name, mean in (("fundamentals", 5.0), ("macro", -2.0)):
        third = DIMS[name] // 3
        values = rng.normal(mean, 1.2, size=(rows, third))
        observed = rng.random(size=(rows, third)) < 0.75
        ages = np.abs(rng.normal(2.0, 0.5, size=(rows, third)))
        inputs[name] = np.concatenate(
            [values * observed, observed, ages * observed], axis=1
        ).astype(np.float32)
        presence[:, MODALITIES.index(name)] = observed.any(axis=1)
    targets = rng.normal(scale=0.02, size=rows)
    return inputs, presence, targets


def decision_batch(
    *,
    historical=True,
    absent=True,
    at=1_609_459_200_000_000,
    dtype=torch.float64,
    seed=23,
):
    rng = np.random.default_rng(seed)
    rows = 4
    prices = rng.normal(0.0, 0.02, size=(rows, 64, DIMS["prices"])).astype(np.float32)
    prices[..., 1] = np.maximum(prices[..., 0], prices[..., 3]) + np.abs(prices[..., 1])
    prices[..., 2] = np.minimum(prices[..., 0], prices[..., 3]) - np.abs(prices[..., 2])
    prices[..., 4] = np.abs(prices[..., 4])
    news = rng.normal(0.5, 1.2, size=(rows, DIMS["news"])).astype(np.float32)
    visual = rng.normal(2.0, 10.0, size=(rows, VISUAL_COLS)).astype(np.float32)
    technical = rng.normal(-0.2, 0.05, size=(rows, TECH_COLS)).astype(np.float32)
    charts = np.concatenate([visual, technical], axis=1)

    fundamentals = np.zeros((rows, DIMS["fundamentals"]), dtype=np.float32)
    fundamentals[:, :2] = rng.normal(5.0, 1.0, size=(rows, 2))
    fundamentals[:, 2:4] = 1.0
    fundamentals[:, 4:6] = np.abs(rng.normal(2.0, 0.3, size=(rows, 2)))

    # Misma macro en todas las filas de la sesión, como en la realidad.
    macro_row = np.array([-1.8, -2.3, 1.0, 1.0, 1.5, 2.1], dtype=np.float32)
    macro = np.broadcast_to(macro_row, (rows, DIMS["macro"])).copy()

    presence = np.ones((rows, 5), dtype=np.bool_)
    if absent and historical:
        # Fila 0: completa. Fila 1: sin noticias ni fundamentales. Fila 2: sin macro.
        # Fila 3: solo precios y gráficos.
        presence[1, [1, 3]] = False
        presence[2, 4] = False
        presence[3, [1, 3, 4]] = False
        news[~presence[:, 1]] = 0.0
        fundamentals[~presence[:, 3]] = 0.0
        macro[~presence[:, 4]] = 0.0

    raw = dict(
        inputs=dict(
            prices=prices, news=news, charts=charts, fundamentals=fundamentals, macro=macro
        ),
        sample_ids=[f"US/A{i}/{at}" for i in range(rows)],
        prediction_at=np.full(rows, at, dtype="datetime64[us]"),
        input_available_at=np.full(rows, at - 1, dtype="datetime64[us]"),
    )
    if historical:
        raw["presence"] = presence
    return DecisionBatch.from_corpus(raw, spec(historical=historical), dtype=dtype)


def build_predictor(
    *,
    variant="mac_online",
    head=QUANTILE_HEAD,
    historical=True,
    dtype=torch.float64,
    fitted=True,
    fit_kwargs=None,
    **options,
):
    config = FinancialConfig(
        spec(historical=historical),
        variant=variant,
        hidden_size=HIDDEN,
        layers=1,
        seed=42,
        head=head,
        **options,
    )
    model = FinancialPredictor(config, dtype=dtype)
    if fitted and config.learning_scale is not None:
        inputs, presence, targets = fit_corpus(**(fit_kwargs or {}))
        model.set_learning_statistics(
            fit_statistics(config.learning_scale, inputs, presence, targets)
        )
    return model


def prepare(model, batch, *, differentiable=False):
    state = model.initial_state(batch.flow_ids, differentiable=differentiable)
    return model.prepare(batch, state, differentiable=differentiable)


def capture_fusion_input(model, batch):
    observed = {}
    handle = model.fusion[0].register_forward_pre_hook(
        lambda module, arguments: observed.update(value=arguments[0].detach().clone())
    )
    try:
        prepared = prepare(model, batch)
    finally:
        handle.remove()
    return observed["value"], prepared


def fill_module(module, seed=7, scale=0.4):
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.copy_(
                scale * torch.randn(parameter.shape, generator=generator, dtype=parameter.dtype)
            )


# ===========================================================================
# 0. PARIDAD BIT A BIT POR DEFECTO
# ===========================================================================


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("variant", VARIANTS)
def test_defaults_preserve_identity_outputs_losses_and_gradients_bit_for_bit(dtype, variant):
    plain = build_predictor(variant=variant, dtype=dtype, learning_scale=BASE_SCALE)
    explicit = build_predictor(
        variant=variant,
        dtype=dtype,
        learning_scale=dict(BASE_SCALE, chart_technical_columns=None),
        modality_gating=None,
    )
    assert plain.config.identity() == explicit.config.identity()
    assert plain._config_id() == explicit._config_id()
    assert plain._parameter_id == explicit._parameter_id
    assert "chart_technical_block" not in plain.config.identity()["learning_scale"]
    assert "modality_gating" not in plain.config.identity()

    batch = decision_batch(dtype=dtype)
    target = torch.tensor([0.01, -0.02, 0.005, -0.01], dtype=dtype)
    out_plain = prepare(plain, batch, differentiable=True)
    out_explicit = prepare(explicit, batch, differentiable=True)
    assert torch.equal(out_plain.detached_tokens, out_explicit.detached_tokens)
    assert torch.equal(out_plain.working_state, out_explicit.working_state)
    assert torch.equal(out_plain.quantiles, out_explicit.quantiles)
    assert torch.equal(out_plain.point_predictions, out_explicit.point_predictions)

    loss_plain = pinball_loss(out_plain.quantiles, target)
    loss_explicit = pinball_loss(out_explicit.quantiles, target)
    assert torch.equal(loss_plain, loss_explicit)
    grads_plain = torch.autograd.grad(loss_plain, tuple(plain.parameters()), allow_unused=True)
    grads_explicit = torch.autograd.grad(
        loss_explicit, tuple(explicit.parameters()), allow_unused=True
    )
    for g1, g2 in zip(grads_plain, grads_explicit, strict=True):
        if g1 is None:
            assert g2 is None
        else:
            assert torch.equal(g1, g2)


# ===========================================================================
# 1A. SUBBLOQUE TÉCNICO EN LEARNING_SCALE: SUELO `_floored` INDEPENDIENTE
# ===========================================================================


def _independent_robust(array):
    array = array.astype(np.float64)
    low, med, high = np.quantile(array, (0.16, 0.5, 0.84), axis=0)
    return med, np.maximum((high - low) / 2.0, 0.5 * array.std(axis=0))


def _independent_floor(scales):
    tiny = float(np.finfo(np.float32).tiny)
    pos = scales[scales >= tiny]
    if not len(pos):
        return np.ones_like(scales)
    floor = max(RELATIVE_FLOOR * float(np.median(pos)), tiny)
    return np.maximum(scales, floor)


def test_chart_technical_columns_separates_the_relative_scale_floor_from_visual_dims():
    inputs, presence, targets = fit_corpus(visual_scale=20.0, tech_scale=0.05)
    # Con el cálculo antiguo, las 8 dimensiones visuales (~20) fuerzan el suelo de las 4 técnicas
    # a 0.1 * ~20 = ~2.0, aplastando la dispersión real (~0.05) de los rasgos técnicos.
    coupled = fit_statistics(LearningScale(), inputs, presence, targets)
    separated = fit_statistics(
        LearningScale(chart_technical_columns=TECH_COLS), inputs, presence, targets
    )

    vis_center, vis_raw = _independent_robust(inputs["charts"][:, :VISUAL_COLS])
    tech_center, tech_raw = _independent_robust(inputs["charts"][:, VISUAL_COLS:])
    expected_center = np.concatenate([vis_center, tech_center]).astype(np.float32)
    expected_scale = np.concatenate(
        [_independent_floor(vis_raw), _independent_floor(tech_raw)]
    ).astype(np.float32)

    charts_sep = separated["modalities"]["charts"]
    np.testing.assert_allclose(charts_sep["center"], expected_center, rtol=0, atol=0)
    np.testing.assert_allclose(charts_sep["scale"], expected_scale, rtol=0, atol=0)

    # Confirmar el contraste real frente al acoplado: la escala técnica ya no es aplastada (~2.0)
    # sino que conserva su propia escala (~0.05) y su propio suelo (0.1 * ~0.05 = ~0.005).
    coupled_tech = np.asarray(coupled["modalities"]["charts"]["scale"][VISUAL_COLS:])
    sep_tech = np.asarray(charts_sep["scale"][VISUAL_COLS:])
    assert np.all(coupled_tech > 1.0)
    assert np.all(sep_tech[:-1] < 0.2) and np.all(sep_tech[:-1] > 0.01)
    assert sep_tech[-1] == pytest.approx(RELATIVE_FLOOR * float(np.median(tech_raw[:-1])), rel=1e-6)
    # Dirección inversa: visual pequeño (~0.02) y técnico grande (~15.0) con primera columna
    # visual constante -> su suelo depende de la mediana visual, nunca de la técnica.
    inv_inputs, inv_presence, inv_targets = fit_corpus(visual_scale=0.02, tech_scale=15.0)
    inv_inputs["charts"][:, 0] = 1.0
    inv_sep = fit_statistics(
        LearningScale(chart_technical_columns=TECH_COLS), inv_inputs, inv_presence, inv_targets
    )
    _, inv_vis_raw = _independent_robust(inv_inputs["charts"][:, :VISUAL_COLS])
    assert inv_sep["modalities"]["charts"]["scale"][0] == pytest.approx(
        RELATIVE_FLOOR * float(np.median(inv_vis_raw[1:])), rel=1e-6
    )
    # Las demás modalidades no cambian bit a bit.
    for other in ("prices", "news", "fundamentals", "macro"):
        assert separated["modalities"][other] == coupled["modalities"][other]


def test_chart_technical_columns_isolates_constant_subblocks_and_preserves_absences():
    inputs, presence, targets = fit_corpus(visual_scale=15.0, tech_scale=0.2)
    # Bloque visual completamente constante: su escala cae a 1.0 sin alterar el subbloque técnico.
    inputs["charts"][:, :VISUAL_COLS] = 3.5
    presence[::4, 2] = False
    inputs["charts"][::4] = 0.0
    stats = fit_statistics(
        LearningScale(chart_technical_columns=TECH_COLS), inputs, presence, targets
    )
    scales = stats["modalities"]["charts"]["scale"]
    assert scales[:VISUAL_COLS] == [1.0] * VISUAL_COLS
    _, tech_raw = _independent_robust(inputs["charts"][presence[:, 2], VISUAL_COLS:])
    np.testing.assert_allclose(
        scales[VISUAL_COLS:], _independent_floor(tech_raw).astype(np.float32), rtol=0, atol=0
    )

    # Al escalar, una fila con `charts` ausente sigue quedando en cero exacto.
    scaler = LearningScaler(
        LearningScale(chart_technical_columns=TECH_COLS), DIMS, dtype=torch.float64
    )
    scaler.load(stats)
    tensors = {k: torch.tensor(v, dtype=torch.float64) for k, v in inputs.items()}
    scaled = scaler(tensors, torch.tensor(presence))
    assert torch.count_nonzero(scaled["charts"][~presence[:, 2]]) == 0


def test_chart_technical_columns_identity_parsing_and_validation():
    assert "chart_technical_block" not in LearningScale().identity()
    assert LearningScale(chart_technical_columns=None).identity() == LearningScale().identity()
    identity = LearningScale(chart_technical_columns=TECH_COLS).identity()
    assert identity["chart_technical_block"] == dict(
        columns=TECH_COLS,
        position="last_columns_of_charts",
        floor="independent_relative_floor_per_subblock_v1",
    )
    assert parse_learning_scale(SPLIT_SCALE) == LearningScale(chart_technical_columns=TECH_COLS)
    assert parse_learning_scale(BASE_SCALE) == LearningScale()

    for bad in (0, -1, 513, True, 4.0, "4"):
        with pytest.raises(ValueError, match="chart_technical_columns"):
            LearningScale(chart_technical_columns=bad)

    # No puede igualar ni superar la anchura total del bloque `charts`.
    inputs, presence, targets = fit_corpus()
    for too_wide in (CHART_COLS, CHART_COLS + 1):
        with pytest.raises(ValueError, match="chart_technical_columns"):
            fit_statistics(
                LearningScale(chart_technical_columns=too_wide), inputs, presence, targets
            )
        with pytest.raises(ValueError, match="chart_technical_columns"):
            LearningScaler(LearningScale(chart_technical_columns=too_wide), DIMS)
        with pytest.raises(ValueError, match="chart_technical_columns"):
            build_predictor(
                learning_scale=dict(BASE_SCALE, chart_technical_columns=too_wide), fitted=False
            )

    # Si se declaran a la vez `chart_technical_columns` y `technical_branch`, deben coincidir.
    with pytest.raises(ValueError, match="chart_technical_columns"):
        build_predictor(
            learning_scale=dict(BASE_SCALE, chart_technical_columns=TECH_COLS - 1),
            technical_branch=asdict(DENSE_BRANCH),
            fitted=False,
        )


def test_chart_technical_columns_prevents_leakage_across_visual_perturbation_and_future_rows():
    inputs, presence, targets = fit_corpus(rows=180, visual_scale=10.0, tech_scale=0.1)
    # Separar un tramo pasado (ajuste) y un tramo futuro (evaluación).
    fit_inputs = {k: v[:120].copy() for k, v in inputs.items()}
    fit_presence = presence[:120].copy()
    fit_targets = targets[:120].copy()

    stats_a = fit_statistics(
        LearningScale(chart_technical_columns=TECH_COLS), fit_inputs, fit_presence, fit_targets
    )
    # Multiplicar por 100 el bloque visual del ajuste NO puede cambiar ni un bit de las escalas
    # técnicas.
    altered_vis = copy.deepcopy(fit_inputs)
    altered_vis["charts"][:, :VISUAL_COLS] *= 100.0
    stats_b = fit_statistics(
        LearningScale(chart_technical_columns=TECH_COLS), altered_vis, fit_presence, fit_targets
    )
    assert (
        stats_a["modalities"]["charts"]["center"][VISUAL_COLS:]
        == stats_b["modalities"]["charts"]["center"][VISUAL_COLS:]
    )
    assert (
        stats_a["modalities"]["charts"]["scale"][VISUAL_COLS:]
        == stats_b["modalities"]["charts"]["scale"][VISUAL_COLS:]
    )

    # Las estadísticas de `fit` no dependen de los datos del futuro: el escalado de las filas
    # pasadas es idéntico independientemente de los valores de las filas futuras.
    scaler = LearningScaler(
        LearningScale(chart_technical_columns=TECH_COLS), DIMS, dtype=torch.float64
    )
    scaler.load(stats_a)
    eval_tensors = {k: torch.tensor(v[120:], dtype=torch.float64) for k, v in inputs.items()}
    eval_presence = torch.tensor(presence[120:])
    first = scaler(eval_tensors, eval_presence)["charts"]
    perturbed_future = {k: v.clone() for k, v in eval_tensors.items()}
    perturbed_future["charts"][-1] = 1e4
    second = scaler(perturbed_future, eval_presence)["charts"]
    assert torch.equal(first[:-1], second[:-1])


# ===========================================================================
# 1B. SUPRESIÓN INDEPENDIENTE `charts_visual` / `charts_technical` Y MODALITY_DROPOUT
# ===========================================================================


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_removing_charts_visual_zeroes_only_visual_columns_and_preserves_technical_branch(dtype):
    model = build_predictor(
        dtype=dtype,
        learning_scale=SPLIT_SCALE,
        technical_branch=asdict(DENSE_BRANCH),
        removed_modalities=dict(names=["charts_visual"]),
    )
    # Abrir la salida de la rama técnica para verificar que sigue activa.
    fill_module(model.technical_branch.outer, seed=19, scale=0.2)
    model._seal_parameters()

    batch = decision_batch(dtype=dtype)
    # Cambiar arbitrariamente las columnas visuales no cambia ni un bit de la salida ni el token.
    altered_inputs = dict(batch.inputs)
    altered_charts = batch.inputs["charts"].clone()
    altered_charts[:, :VISUAL_COLS] = torch.randn_like(altered_charts[:, :VISUAL_COLS]) * 500.0
    altered_inputs["charts"] = altered_charts
    other = batch.derived(altered_inputs, "prueba_visual_cambiado")

    out_a = prepare(model, batch)
    out_b = prepare(model, other)
    assert torch.equal(out_a.detached_tokens, out_b.detached_tokens)
    assert torch.equal(out_a.working_state, out_b.working_state)
    assert torch.equal(out_a.quantiles, out_b.quantiles)
    assert torch.equal(out_a.point_predictions, out_b.point_predictions)

    # Cambiar las columnas técnicas SÍ cambia tanto el token como el residuo técnico.
    tech_changed = dict(batch.inputs)
    t_charts = batch.inputs["charts"].clone()
    t_charts[:, VISUAL_COLS:] += 0.5
    tech_changed["charts"] = t_charts
    out_c = prepare(model, batch.derived(tech_changed, "prueba_tecnico_cambiado"))
    assert not torch.equal(out_a.detached_tokens, out_c.detached_tokens)
    assert not torch.equal(out_a.working_state, out_c.working_state)

    # Verificación independiente exacta de la entrada al codificador `charts`: las columnas
    # visuales valen 0.0 incluso tras `learning_scaler`, mientras las técnicas coinciden con el
    # escalador y el bit de presencia `charts` sigue activo.
    seen_charts = {}
    hook = model.encoders["charts"].register_forward_pre_hook(
        lambda m, args: seen_charts.update(x=args[0].detach().clone())
    )
    try:
        fusion_in, _ = capture_fusion_input(model, batch)
    finally:
        hook.remove()
    assert torch.count_nonzero(seen_charts["x"][:, :VISUAL_COLS]) == 0
    expected_scaled = model.learning_scaler(batch.inputs, batch.presence)["charts"]
    torch.testing.assert_close(
        seen_charts["x"][:, VISUAL_COLS:], expected_scaled[:, VISUAL_COLS:], rtol=0, atol=0
    )
    assert bool(fusion_in[:, 5 * HIDDEN + MODALITIES.index("charts")].all())

    # Gradiente estrictamente cero en pesos visuales y no nulo en pesos técnicos y rama técnica.
    diff_out = prepare(model, batch, differentiable=True)
    loss = pinball_loss(diff_out.quantiles, torch.tensor([0.01, -0.02, 0.03, -0.01], dtype=dtype))
    loss.backward()
    chart_weight_grad = model.encoders["charts"][0].weight.grad
    assert torch.isfinite(chart_weight_grad).all()
    assert torch.count_nonzero(chart_weight_grad[:, :VISUAL_COLS]) == 0
    assert chart_weight_grad[:, VISUAL_COLS:].abs().sum() > 0
    assert model.technical_branch.outer.weight.grad.abs().sum() > 0


def test_removing_charts_visual_works_as_single_variable_using_technical_branch_width():
    """En el núcleo v11 (`learning_scale` sin `chart_technical_columns`, `technical_branch` sí),
    basta declarar `removed_modalities={"names": ["charts_visual"]}`."""
    model = build_predictor(
        learning_scale=BASE_SCALE,
        technical_branch=asdict(DENSE_BRANCH),
        removed_modalities=dict(names=["charts_visual"]),
    )
    assert model.config.removed_modalities == ("charts_visual",)
    assert model.config.identity()["removed_modalities"]["chart_technical_columns"] == TECH_COLS
    batch = decision_batch()
    changed = dict(batch.inputs)
    changed["charts"] = batch.inputs["charts"].clone()
    changed["charts"][:, :VISUAL_COLS] += 99.0
    assert torch.equal(
        prepare(model, batch).detached_tokens,
        prepare(model, batch.derived(changed, "visual_diff")).detached_tokens,
    )


def test_removing_charts_technical_zeroes_only_technical_columns_and_preserves_visual_dims():
    model = build_predictor(
        learning_scale=SPLIT_SCALE,
        removed_modalities=dict(names=["charts_technical"]),
    )
    batch = decision_batch()
    changed_tech = dict(batch.inputs)
    changed_tech["charts"] = batch.inputs["charts"].clone()
    changed_tech["charts"][:, VISUAL_COLS:] = 42.0
    out_a = prepare(model, batch)
    out_b = prepare(model, batch.derived(changed_tech, "tech_diff"))
    assert torch.equal(out_a.detached_tokens, out_b.detached_tokens)
    assert torch.equal(out_a.quantiles, out_b.quantiles)

    changed_vis = dict(batch.inputs)
    changed_vis["charts"] = batch.inputs["charts"].clone()
    changed_vis["charts"][:, :VISUAL_COLS] += 10.0
    assert not torch.equal(
        out_a.detached_tokens,
        prepare(model, batch.derived(changed_vis, "vis_diff")).detached_tokens,
    )

    seen_charts = {}
    hook = model.encoders["charts"].register_forward_pre_hook(
        lambda m, args: seen_charts.update(x=args[0].detach().clone())
    )
    try:
        prepare(model, batch)
    finally:
        hook.remove()
    assert torch.count_nonzero(seen_charts["x"][:, VISUAL_COLS:]) == 0
    expected_scaled = model.learning_scaler(batch.inputs, batch.presence)["charts"]
    torch.testing.assert_close(
        seen_charts["x"][:, :VISUAL_COLS], expected_scaled[:, :VISUAL_COLS], rtol=0, atol=0
    )

    diff_out = prepare(model, batch, differentiable=True)
    pinball_loss(
        diff_out.quantiles, torch.tensor([0.01, -0.02, 0.03, -0.01], dtype=torch.float64)
    ).backward()
    grad = model.encoders["charts"][0].weight.grad
    assert torch.count_nonzero(grad[:, VISUAL_COLS:]) == 0
    assert grad[:, :VISUAL_COLS].abs().sum() > 0


def test_chart_subblock_removal_validation_and_session_ic_column_guards():
    # Rechazar combinaciones redundantes o sin anchura técnica declarada.
    for names in (
        ["charts", "charts_visual"],
        ["charts", "charts_technical"],
        ["charts_visual", "charts_technical"],
    ):
        with pytest.raises(ValueError, match="removed_modalities"):
            build_predictor(
                learning_scale=SPLIT_SCALE, removed_modalities=dict(names=names), fitted=False
            )

    with pytest.raises(ValueError, match="chart_technical_columns|technical_branch"):
        build_predictor(
            learning_scale=BASE_SCALE,
            removed_modalities=dict(names=["charts_visual"]),
            fitted=False,
        )

    # `charts_technical` sí es incompatible con `technical_branch`.
    with pytest.raises(ValueError, match="technical_branch"):
        build_predictor(
            learning_scale=SPLIT_SCALE,
            technical_branch=asdict(DENSE_BRANCH),
            removed_modalities=dict(names=["charts_technical"]),
            fitted=False,
        )

    # Combinar una modalidad entera con un subbloque del gráfico conserva el orden canónico y
    # anula ambos (presencia en falso para `news` y prefijo visual en cero para `charts`).
    combo = build_predictor(
        learning_scale=SPLIT_SCALE,
        removed_modalities=dict(names=["charts_visual", "news"]),
    )
    assert combo.config.removed_modalities == ("news", "charts_visual")
    combo_in, _ = capture_fusion_input(combo, decision_batch(absent=False))
    assert not combo_in[:, 5 * HIDDEN + MODALITIES.index("news")].any()
    assert bool(combo_in[:, 5 * HIDDEN + MODALITIES.index("charts")].all())

    # `financial_run.ChronologicalInference` admite `charts_visual` con liquidity/offset, pero
    # rechaza `charts_technical`.
    recipe_liq = ChronologicalRecipe(
        loss="pinball",
        session_ic_term=dict(weight=3.0, minimum_rows=3, liquidity_column=-2, liquidity_power=0.5),
    )
    recipe_off = ChronologicalRecipe(
        loss="pinball",
        session_ic_term=dict(weight=3.0, minimum_rows=3, offset_column=-1, offset_scale=0.01),
    )
    vis_removed = build_predictor(
        learning_scale=SPLIT_SCALE,
        technical_branch=asdict(DENSE_BRANCH),
        removed_modalities=dict(names=["charts_visual"]),
    )
    assert ChronologicalInference(vis_removed, recipe_liq).liquidity
    assert ChronologicalInference(vis_removed, recipe_off).offset

    tech_removed = build_predictor(
        learning_scale=SPLIT_SCALE,
        removed_modalities=dict(names=["charts_technical"]),
    )
    with pytest.raises(ValueError, match="liquidez"):
        ChronologicalInference(tech_removed, recipe_liq)
    with pytest.raises(ValueError, match="desplazamiento"):
        ChronologicalInference(tech_removed, recipe_off)


def test_modality_dropout_with_chart_technical_columns_drops_only_visual_slice():
    model = build_predictor(
        learning_scale=SPLIT_SCALE,
        technical_branch=asdict(DENSE_BRANCH),
        modality_dropout=0.5,
    )
    model.train(True)
    batch = decision_batch(absent=False)
    # Buscar una época en la que alguna fila sufra dropout sobre `charts`.
    seen = {}

    def capture(module, arguments):
        seen["charts"] = arguments[0].detach().clone()

    handle = model.encoders["charts"].register_forward_pre_hook(capture)
    dropped_rows = None
    try:
        for epoch in range(20):
            model.set_fit_epoch(epoch)
            prepare(model, batch)
            vis_zero = seen["charts"][:, :VISUAL_COLS].abs().sum(dim=1) == 0
            if vis_zero.any():
                dropped_rows = vis_zero
                break
    finally:
        handle.remove()
    assert dropped_rows is not None and dropped_rows.any()
    expected = model.learning_scaler(batch.inputs, batch.presence)["charts"]
    # En las filas donde cayó el bloque visual, las columnas técnicas siguen intactas y escaladas.
    torch.testing.assert_close(
        seen["charts"][dropped_rows, VISUAL_COLS:],
        expected[dropped_rows, VISUAL_COLS:],
        rtol=0,
        atol=0,
    )
    # Y el comportamiento es determinista al repetir la misma época.
    first = prepare(model, batch)
    second = prepare(model, batch)
    assert torch.equal(first.detached_tokens, second.detached_tokens)

    # Si además `charts_visual` está retirado, el dropout no reintroduce nunca valores visuales.
    both = build_predictor(
        learning_scale=SPLIT_SCALE,
        technical_branch=asdict(DENSE_BRANCH),
        removed_modalities=dict(names=["charts_visual"]),
        modality_dropout=0.5,
    )
    both.train(True)
    both.set_fit_epoch(0)
    seen_both = {}
    h2 = both.encoders["charts"].register_forward_pre_hook(
        lambda m, args: seen_both.update(charts=args[0].detach().clone())
    )
    try:
        prepare(both, batch)
    finally:
        h2.remove()
    assert torch.count_nonzero(seen_both["charts"][:, :VISUAL_COLS]) == 0
    torch.testing.assert_close(
        seen_both["charts"][:, VISUAL_COLS:], expected[:, VISUAL_COLS:], rtol=0, atol=0
    )


# ===========================================================================
# 2. COMPUERTA DE MODALIDADES ENMASCARADA POR PRESENCIA (`presence_softmax_v1`)
# ===========================================================================


def _independent_presence_softmax_gate(blocks, presence, gate_weight, gate_bias):
    """Cuenta independiente fila a fila en Python puro + math.exp."""
    rows, modalities, hidden = blocks.shape
    out = torch.zeros_like(blocks)
    weights = torch.zeros((rows, modalities), dtype=blocks.dtype)
    for b in range(rows):
        present_idx = [m for m in range(modalities) if bool(presence[b, m])]
        if not present_idx:
            continue
        logits = {}
        for m in present_idx:
            dot = sum(float(blocks[b, m, h]) * float(gate_weight[m, h]) for h in range(hidden))
            logits[m] = dot + float(gate_bias[m])
        peak = max(logits.values())
        exps = {m: math.exp(logits[m] - peak) for m in present_idx}
        total = sum(exps.values())
        for m in present_idx:
            alpha = exps[m] / total
            weights[b, m] = alpha
            out[b, m] = (modalities * alpha) * blocks[b, m]
    return out, weights


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_presence_modality_gate_matches_independent_calculation_and_conserves_weight(dtype):
    gate = PresenceModalityGate(6, dtype=dtype)
    fill_module(gate, seed=31, scale=0.7)
    rng = torch.Generator().manual_seed(32)
    blocks = torch.randn(4, 5, 6, generator=rng, dtype=dtype)
    presence = torch.tensor(
        [
            [True, True, True, True, True],
            [True, False, True, False, True],
            [True, True, True, False, False],
            [True, False, False, False, False],
        ]
    )
    expected_out, expected_alpha = _independent_presence_softmax_gate(
        blocks, presence, gate.gate_weight.detach(), gate.gate_bias.detach()
    )
    actual_alpha = gate.weights(blocks, presence)
    actual_out = gate(blocks, presence)

    tol = 1e-6 if dtype == torch.float32 else 1e-12
    torch.testing.assert_close(actual_alpha, expected_alpha, rtol=tol, atol=tol)
    torch.testing.assert_close(actual_out, expected_out, rtol=tol, atol=tol)
    # Conservación exacta de peso total (= 5) en toda fila con al menos una modalidad presente.
    torch.testing.assert_close(
        actual_alpha.sum(dim=1), torch.ones(4, dtype=dtype), rtol=tol, atol=tol
    )
    # Modalidades ausentes estrictamente en cero.
    assert torch.count_nonzero(actual_alpha[~presence]) == 0
    assert torch.count_nonzero(actual_out[~presence]) == 0

    # Verificar que tanto `gate_weight` como `gate_bias` influyen por separado.
    bias_only = PresenceModalityGate(6, dtype=dtype)
    with torch.no_grad():
        bias_only.gate_bias.copy_(torch.tensor([1.0, -1.0, 0.5, 0.0, -0.5], dtype=dtype))
    assert not torch.allclose(
        bias_only.weights(blocks, presence)[0], torch.full((5,), 0.2, dtype=dtype)
    )

    # Rechazar entradas mal formadas en el módulo.
    with pytest.raises(ValueError, match="bloques"):
        gate(blocks[:, :4], presence)
    with pytest.raises(ValueError, match="bloques"):
        gate(blocks.to(torch.float16), presence)
    with pytest.raises(ValueError, match="presencia"):
        gate(blocks, presence.to(dtype=dtype))


def test_presence_modality_gate_ignores_absent_garbage_and_handles_extreme_logits_and_empty_row():
    gate = PresenceModalityGate(4, dtype=torch.float32)
    fill_module(gate, seed=41, scale=500.0)
    blocks = torch.randn(3, 5, 4, generator=torch.Generator().manual_seed(42), dtype=torch.float32)
    presence = torch.tensor(
        [
            [True, False, True, False, True],
            [True, True, True, False, True],
            [False, False, False, False, False],
        ]
    )
    clean_blocks = blocks.clone()
    clean_blocks[~presence] = 0.0
    dirty_blocks = blocks.clone()
    dirty_blocks[~presence] = 1e6

    clean_blocks.requires_grad_(True)
    dirty_blocks.requires_grad_(True)
    out_clean = gate(clean_blocks, presence)
    out_dirty = gate(dirty_blocks, presence)

    assert torch.isfinite(out_clean).all() and torch.isfinite(out_dirty).all()
    assert torch.equal(out_clean, out_dirty)
    assert torch.count_nonzero(out_clean[2]) == 0

    g_clean = torch.autograd.grad(
        out_clean.square().sum(), (clean_blocks, gate.gate_weight, gate.gate_bias)
    )
    g_dirty = torch.autograd.grad(
        out_dirty.square().sum(), (dirty_blocks, gate.gate_weight, gate.gate_bias)
    )
    for left, right in zip(g_clean, g_dirty, strict=True):
        assert torch.isfinite(left).all() and torch.isfinite(right).all()
        assert torch.equal(left, right)
    assert torch.count_nonzero(g_clean[0][~presence]) == 0
    # Modalidad 3 ausente en todas las filas: sus parámetros de compuerta reciben gradiente 0.
    assert torch.count_nonzero(g_clean[1][3]) == 0 and g_clean[2][3] == 0


def test_presence_modality_gate_is_strictly_row_independent():
    gate = PresenceModalityGate(8, dtype=torch.float64)
    fill_module(gate, seed=51, scale=0.5)
    blocks = torch.randn(5, 5, 8, generator=torch.Generator().manual_seed(52), dtype=torch.float64)
    presence = torch.tensor(
        [
            [True, True, True, True, True],
            [True, False, True, True, True],
            [True, True, True, False, True],
            [True, False, True, False, True],
            [True, False, True, False, False],
        ]
    )
    full = gate(blocks, presence)
    # Evaluar una sola fila coincide con evaluarla dentro del lote completo.
    for row in range(len(blocks)):
        single = gate(blocks[row : row + 1], presence[row : row + 1])
        torch.testing.assert_close(single, full[row : row + 1], rtol=1e-13, atol=1e-13)
    # Cambiar otra fila no altera ni un bit de la fila 1.
    perturbed = blocks.clone()
    perturbed[0] *= -7.0
    assert torch.equal(gate(perturbed, presence)[1], full[1])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_predictor_modality_gating_starts_exact_on_complete_rows_and_reweights_incomplete_rows(
    dtype,
):
    control = build_predictor(dtype=dtype)
    gated = build_predictor(dtype=dtype, modality_gating=PRESENCE_SOFTMAX)

    # Mismos pesos del tronco (no consume generador global).
    assert set(gated.state_dict()) - set(control.state_dict()) == {
        "modality_gating.gate_weight",
        "modality_gating.gate_bias",
    }
    for name, value in control.state_dict().items():
        if torch.is_tensor(value):
            assert torch.equal(gated.state_dict()[name], value), name

    batch = decision_batch(dtype=dtype, absent=True)
    ref_in, ref_out = capture_fusion_input(control, batch)
    got_in, got_out = capture_fusion_input(gated, batch)
    tol = 1e-6 if dtype == torch.float32 else 1e-12

    # Fila 0 es completa: en el inicio (logits=0) la compuerta vale 5 * (1/5) = 1.0 -> idéntica.
    torch.testing.assert_close(got_in[0], ref_in[0], rtol=tol, atol=tol)
    torch.testing.assert_close(
        got_out.detached_tokens[0], ref_out.detached_tokens[0], rtol=tol, atol=tol
    )

    # Fila 1 tiene k=3 modalidades presentes (0, 2, 4): en el inicio cada bloque presente se
    # reescala por 5/3 y los ausentes (1, 3) siguen en 0.
    blocks_ref = ref_in[:, :-5].reshape(-1, 5, HIDDEN)
    blocks_got = got_in[:, :-5].reshape(-1, 5, HIDDEN)
    for row in range(1, len(batch.presence)):
        k = int(batch.presence[row].sum())
        factor = 5.0 / k
        for m in range(5):
            if batch.presence[row, m]:
                torch.testing.assert_close(
                    blocks_got[row, m], blocks_ref[row, m] * factor, rtol=tol, atol=tol
                )
            else:
                assert torch.count_nonzero(blocks_got[row, m]) == 0
    # Los bits de presencia al final de fusion_input permanecen intactos.
    torch.testing.assert_close(got_in[:, -5:], batch.presence.to(dtype=dtype), rtol=0, atol=0)

    # Con pesos no nulos en la compuerta coincide con la cuenta independiente sobre los bloques.
    fill_module(gated.modality_gating, seed=61, scale=0.6)
    gated._seal_parameters()
    weighted_in, _ = capture_fusion_input(gated, batch)
    expected_blocks, _ = _independent_presence_softmax_gate(
        blocks_ref,
        batch.presence,
        gated.modality_gating.gate_weight.detach(),
        gated.modality_gating.gate_bias.detach(),
    )
    torch.testing.assert_close(weighted_in[:, :-5], expected_blocks.flatten(1), rtol=tol, atol=tol)
    # En fila completa también actúa cuando los pesos no son nulos.
    assert not torch.equal(weighted_in[0, :-5], ref_in[0, :-5])


@pytest.mark.parametrize("head", [QUANTILE_HEAD, "scalar"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_gradients_are_finite_when_initial_prediction_is_constant_zero(head, dtype):
    """Regla 4: con la última capa en cero (`zero_median=True`), todos los gradientes son finitos
    tanto bajo pinball/L1 como bajo el término de IC por sesión."""
    model = build_predictor(
        head=head,
        dtype=dtype,
        learning_scale=SPLIT_SCALE,
        modality_gating=PRESENCE_SOFTMAX,
        technical_branch=asdict(DENSE_BRANCH),
        removed_modalities=dict(names=["charts_visual"]),
    )
    batch = decision_batch(dtype=dtype, absent=True)
    out = prepare(model, batch, differentiable=True)
    # Predicción puntual inicialmente constante (cero exacto).
    assert torch.equal(out.point_predictions, torch.zeros_like(out.point_predictions))

    target = torch.tensor([0.02, -0.01, 0.015, -0.025], dtype=dtype)
    sessions = tuple(("US", batch.prediction_at[0]) for _ in range(len(target)))
    ic_term = -session_correlations(out.point_predictions, target, sessions, minimum_rows=4).mean()
    if head == QUANTILE_HEAD:
        loss = pinball_loss(out.quantiles, target) + 3.0 * ic_term
    else:
        loss = (out.point_predictions - target).abs().mean() + 3.0 * ic_term
    grads = torch.autograd.grad(loss, tuple(model.parameters()), allow_unused=True)
    for name, grad in zip((n for n, _ in model.named_parameters()), grads, strict=True):
        if grad is not None:
            assert torch.isfinite(grad).all(), f"Gradiente no finito en {name}"
    if head == QUANTILE_HEAD:
        named = dict(zip((n for n, _ in model.named_parameters()), grads, strict=True))
        # Con QuantileHead las colas propagan señal no nula desde el paso 0 a la compuerta.
        assert named["modality_gating.gate_weight"].abs().sum() > 0
        assert named["modality_gating.gate_bias"].abs().sum() > 0


def test_modality_gating_learns_from_complete_rows_at_zero_start():
    model = build_predictor(modality_gating=PRESENCE_SOFTMAX)
    batch = decision_batch(absent=False)
    assert bool(batch.presence.all())
    out = prepare(model, batch, differentiable=True)
    out.quantiles.sum().backward()
    assert model.modality_gating.gate_weight.grad.abs().sum() > 0
    assert model.modality_gating.gate_bias.grad.abs().sum() > 0


def test_modality_gating_identity_state_pairing_frozen_and_composition():
    assert MODALITY_GATINGS == (PRESENCE_SOFTMAX,)
    assert modality_gating_parameters(HIDDEN) == 5 * (HIDDEN + 1)

    control = build_predictor()
    gated = build_predictor(modality_gating=PRESENCE_SOFTMAX)
    assert "modality_gating" not in control.config.identity()
    assert gated.config.identity()["modality_gating"] == dict(
        name=PRESENCE_SOFTMAX,
        parameters=5 * (HIDDEN + 1),
        initial="zero_logits_uniform_over_present_modalities",
        scale="modality_count_times_masked_softmax",
    )
    assert sum(p.numel() for p in gated.modality_gating.parameters()) == modality_gating_parameters(
        HIDDEN
    )
    assert checked_predictor_changes(dict(modality_gating=PRESENCE_SOFTMAX)) == dict(
        modality_gating=PRESENCE_SOFTMAX
    )

    # Carga de estado y emparejamiento entre variantes.
    fill_module(gated.modality_gating, seed=71, scale=0.3)
    gated._seal_parameters()
    twin = build_predictor(modality_gating=PRESENCE_SOFTMAX)
    twin.load_state_dict(gated.state_dict())
    batch = decision_batch()
    assert torch.equal(prepare(twin, batch).quantiles, prepare(gated, batch).quantiles)
    with pytest.raises(ValueError, match="contrato"):
        control.load_state_dict(gated.state_dict())

    target = build_predictor(variant="mac_frozen", modality_gating=PRESENCE_SOFTMAX)
    copy_paired_parameters(gated, target)
    assert torch.equal(target.modality_gating.gate_weight, gated.modality_gating.gate_weight)

    # Consumidor congelado y `fused_tokens` (F5).
    gated.eval().requires_grad_(False)
    consumer = FrozenFinancialConsumer(gated)
    assert consumer.identity()["modules"]["predictor/modality_gating"]["settings"] == {
        "hidden_size": HIDDEN
    }
    frozen_out = consumer.prepare(batch, gated.initial_state(batch.flow_ids), context_id="wp6-ctx")
    direct_out = prepare(gated, batch)
    assert torch.equal(frozen_out.quantiles, direct_out.quantiles)
    assert torch.equal(gated.fused_tokens(batch.inputs, batch.presence), direct_out.detached_tokens)

    # Composición limpia con `presence_modulation="macro_film_v1"`: macro modula primero y la
    # compuerta de presencia repondera los bloques modulados.
    both = build_predictor(presence_modulation=MACRO_FILM, modality_gating=PRESENCE_SOFTMAX)
    fill_module(both.presence_modulation, seed=72, scale=0.2)
    fill_module(both.modality_gating, seed=73, scale=0.2)
    both._seal_parameters()
    macro_only = build_predictor(presence_modulation=MACRO_FILM)
    macro_only.presence_modulation.load_state_dict(both.presence_modulation.state_dict())
    macro_only._seal_parameters()
    both_in, _ = capture_fusion_input(both, batch)
    macro_in, _ = capture_fusion_input(macro_only, batch)
    expected_composed = both.modality_gating(
        macro_in[:, :-5].reshape(-1, 5, HIDDEN), batch.presence
    )
    torch.testing.assert_close(
        both_in[:, :-5], expected_composed.flatten(1), rtol=1e-12, atol=1e-12
    )

    # Validaciones de rechazo.
    for bad in ("softmax", "", 1, True):
        with pytest.raises(ValueError, match="modality_gating"):
            build_predictor(modality_gating=bad)
    with pytest.raises(ValueError, match="modality_gating"):
        build_predictor(historical=False, modality_gating=PRESENCE_SOFTMAX)
    with pytest.raises(ValueError, match="modality_gating"):
        build_predictor(
            modality_gating=PRESENCE_SOFTMAX,
            regional_fusion=RegionalFusionConfig(),
        )
    for bad_change in (dict(modality_gating={}), dict(modality_gating=True)):
        with pytest.raises(ValueError, match="cambios del predictor"):
            checked_predictor_changes(bad_change)


def test_both_options_compose_cleanly_deterministically_across_all_variants():
    batch = decision_batch(dtype=torch.float32)
    for variant in VARIANTS:
        first = build_predictor(
            variant=variant,
            dtype=torch.float32,
            learning_scale=SPLIT_SCALE,
            technical_branch=asdict(DENSE_BRANCH),
            removed_modalities=dict(names=["charts_visual"]),
            modality_gating=PRESENCE_SOFTMAX,
        )
        second = build_predictor(
            variant=variant,
            dtype=torch.float32,
            learning_scale=SPLIT_SCALE,
            technical_branch=asdict(DENSE_BRANCH),
            removed_modalities=dict(names=["charts_visual"]),
            modality_gating=PRESENCE_SOFTMAX,
        )
        out_1 = prepare(first, batch)
        out_2 = prepare(second, batch)
        assert torch.equal(out_1.quantiles, out_2.quantiles)
        assert torch.equal(out_1.detached_tokens, out_2.detached_tokens)
        assert torch.isfinite(out_1.quantiles).all()
