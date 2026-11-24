"""Recorrido cronológico con memoria rápida factorizada LFM-SW (Idea 4).

Comprueba causalidad estricta (sin fugas del futuro), reinicio de estado entre fases,
acumulación por repetición (`accumulation_rows`) con y sin `observe_without_graph`,
puntos de control y reanudación bit a bit, modos placebo y receta tipo campaña (`zero_median` +
`session_ic_term` + `technical_branch`).
"""

from dataclasses import asdict

import numpy as np
import pytest
import torch

from mars_titan.data.input_policy import HISTORICAL_MASKED
from mars_titan.memory import financial_observations as observations
from mars_titan.models.learning_scale import fit_statistics
from mars_titan.models.quantile_head import PINBALL, QUANTILE_HEAD
from mars_titan.models.titans.factor_memory import (
    EVAL_FROZEN,
    ONLINE,
    RANDOM_BASIS,
    FactorMemoryConfig,
    LowRankFactorMemory,
)
from mars_titan.models.titans.financial import FinancialConfig, FinancialPredictor
from mars_titan.training import financial_run
from mars_titan.training.checkpoints import load_training_state
from mars_titan.training.corpus_inputs import CorpusDataset
from mars_titan.training.financial_run import ChronologicalRecipe, ChronologicalTrainer
from tests.training.chronological_fixture import chronological_corpus, decision, phases
from tests.training.test_financial_run import (
    StopAtStep,
    entries,
    named_records,
)
from tests.training.test_financial_run_session_objective import (
    ASSETS,
    explicit_fastpath,  # noqa: F401
    shared,  # noqa: F401
    streams,
    trainer,
)


def no_warmup_streams(root):
    """Corpus con ajuste sin calentamiento (como en la campaña real) para probar F_0."""
    dataset = CorpusDataset(
        chronological_corpus(root / "corpus", assets=ASSETS),
        input_policy=HISTORICAL_MASKED,
    )
    train_phase, val_phase = phases(warmup="2022-11-15")[0], phases()[1]
    result = {}
    for phase in (train_phase, val_phase):
        manifest = observations.prepare_observation_index(
            dataset, root / f"index-{phase.partition}", phase=phase
        )
        result[phase.partition] = observations.FinancialObservationSource(dataset, manifest)
    return result


PLAIN = dict(session_rank_inputs=None, session_ic_objective=None)
CFG_ONLINE = FactorMemoryConfig(factors=4, ns_steps=3, mode=ONLINE)
CFG_FROZEN = FactorMemoryConfig(factors=4, ns_steps=3, mode=EVAL_FROZEN)
CFG_RANDOM = FactorMemoryConfig(factors=4, ns_steps=3, mode=RANDOM_BASIS)


def make_model(sources, cfg=CFG_ONLINE, *, open_out=False, dtype=torch.float64):
    spec = sources["train"].specification()
    config = FinancialConfig(
        spec,
        hidden_size=32,
        variant="mac_online",
        memory_residual_layer_norm=True,
        gate_bias=dict(alpha_half_life=256.0, eta=0.5, theta=0.05),
        factor_memory=cfg,
    )
    model = FinancialPredictor(config, dtype=dtype)
    if open_out and model.factor_memory is not None:
        with torch.no_grad():
            gen = torch.Generator().manual_seed(77)
            model.factor_memory.out.weight.copy_(
                torch.randn(32, 32, generator=gen, dtype=dtype) / 8
            )
        model._seal_parameters()
    return model


def assert_same_gradients(actual, expected, *, rtol=0.0, atol=0.0):
    assert len(actual) == len(expected) > 1
    for left, right in zip(actual, expected, strict=True):
        assert left.keys() == right.keys()
        for name, value in right.items():
            if value is None:
                assert left[name] is None, name
            else:
                torch.testing.assert_close(left[name], value, rtol=rtol, atol=atol)


def test_disabled_option_keeps_trainer_audit_gradients_and_checkpoint_bit_exact(
    shared,  # noqa: F811
    tmp_path,
):
    control = trainer(shared, tmp_path / "control", model=make_model(shared, cfg=None))
    explicit = trainer(shared, tmp_path / "explicit", model=make_model(shared, cfg=None))
    r_ctrl = control.run()
    r_exp = explicit.run()
    assert r_ctrl["status"] == r_exp["status"] == "completed"
    assert explicit.identity == control.identity
    assert explicit.audit == control.audit
    assert_same_gradients(named_records(explicit), named_records(control))
    assert "factor_memory" not in control._export(financial_run._Pass())


def test_strict_causality_perturbing_future_never_changes_past_predictions_or_gradients(tmp_path):
    cut = decision("2022-12-09")
    runs = {}
    for name, perturb in (("base", None), ("future", cut)):
        sources = streams(tmp_path / name, perturb_after=perturb)
        model = make_model(sources, CFG_ONLINE, open_out=True)
        runs[name] = trainer(
            sources, tmp_path / name / "run", model=model, truncation=2, block_rows=16
        )
        runs[name].run()
    base_preds = entries(runs["base"].audit, "prediction")
    fut_preds = entries(runs["future"].audit, "prediction")
    # Nada posterior a `cut` puede alterar predicciones en `<= cut`.
    assert [e for e in base_preds if e[3] <= cut] == [e for e in fut_preds if e[3] <= cut]
    # Después de `cut`, la perturbación sí cambia predicciones (la prueba es no vacua).
    assert [e for e in base_preds if e[3] > cut] != [e for e in fut_preds if e[3] > cut]
    # Tampoco cambian los gradientes de ninguna actualización en `<= cut`.
    steps = [u[2] for u in entries(runs["base"].audit, "update") if u[1] <= cut]
    assert steps
    assert_same_gradients(
        named_records(runs["base"])[: len(steps)],
        named_records(runs["future"])[: len(steps)],
    )


def test_factor_state_resets_at_every_phase_boundary_and_warms_up_causally(
    shared,  # noqa: F811
    tmp_path,
    monkeypatch,
):
    seen = []
    original_step = LowRankFactorMemory.step

    def spy(self, encoded, flow_ids, markets, prediction_at, state, *, differentiable=False):
        seen.append((self.training, state.updates, state.last_updated_at, min(prediction_at)))
        return original_step(
            self, encoded, flow_ids, markets, prediction_at, state, differentiable=differentiable
        )

    monkeypatch.setattr(LowRankFactorMemory, "step", spy)
    model = make_model(shared, CFG_ONLINE, open_out=True)
    engine = trainer(
        shared, tmp_path / "phases", model=model, truncation=2, epochs=2, block_rows=16
    )
    report = engine.run()
    assert report["status"] == "completed"
    # Cada cambio de fase (val epoch 0, train epoch 0, val epoch 1, train epoch 1, val epoch 2)
    # arranca exactamente en updates=(0, 0) y last_updated_at=(-1, -1).
    starts = [
        i
        for i, (_, updates, last_at, _) in enumerate(seen)
        if updates == (0, 0) and last_at == (-1, -1)
    ]
    assert len(starts) >= 4
    # Dentro de cada fase las actualizaciones avanzan de forma estrictamente creciente.
    for start, end in zip(starts, [*starts[1:], len(seen)], strict=True):
        segment = seen[start:end]
        assert len(segment) > 2
        totals = [sum(u) for _, u, _, _ in segment]
        assert totals == list(range(len(segment)))


def test_accumulation_replay_and_observe_without_graph_match_direct_gradients(
    shared,  # noqa: F811
    tmp_path,
):
    direct = trainer(
        shared,
        tmp_path / "direct",
        model=make_model(shared, CFG_ONLINE, open_out=True),
        truncation=2,
        block_rows=16,
    )
    direct.run()
    replayed = trainer(
        shared,
        tmp_path / "replayed",
        model=make_model(shared, CFG_ONLINE, open_out=True),
        truncation=2,
        block_rows=16,
        accumulation_rows=16,
    )
    replayed.run()
    nograph = trainer(
        shared,
        tmp_path / "nograph",
        model=make_model(shared, CFG_ONLINE, open_out=True),
        truncation=2,
        block_rows=16,
        accumulation_rows=16,
        observe_without_graph=True,
    )
    nograph.run()
    assert replayed.audit == direct.audit
    assert nograph.audit == direct.audit
    ref = named_records(direct)
    assert_same_gradients(named_records(replayed), ref, rtol=1e-9, atol=1e-12)
    assert_same_gradients(named_records(nograph), ref, rtol=1e-9, atol=1e-12)
    # En shared hay calentamiento previo: los 6 parámetros activos tras calentamiento reciben
    # gradiente no nulo. Sin calentamiento en ajuste, initial_factors también recibe gradiente.
    lfmsw_names = [n for n in ref[0] if n.startswith("factor_memory.")]
    assert len(lfmsw_names) == 7
    for name in lfmsw_names:
        if name != "factor_memory.initial_factors":
            assert any(r[name] is not None and r[name].abs().sum() > 0 for r in ref), name
    # Sin calentamiento (política de la campaña), initial_factors recibe gradiente exacto en
    # ambos modos (directo y repetición por acumulación).
    cold = no_warmup_streams(tmp_path / "cold")
    cold_direct = trainer(
        cold,
        tmp_path / "cold_direct",
        model=make_model(cold, CFG_ONLINE, open_out=True),
        truncation=2,
        block_rows=16,
    )
    cold_direct.run()
    cold_replay = trainer(
        cold,
        tmp_path / "cold_replay",
        model=make_model(cold, CFG_ONLINE, open_out=True),
        truncation=2,
        block_rows=16,
        accumulation_rows=16,
        observe_without_graph=True,
    )
    cold_replay.run()
    cold_ref = named_records(cold_direct)
    assert_same_gradients(named_records(cold_replay), cold_ref, rtol=1e-9, atol=1e-12)
    assert any(
        r["factor_memory.initial_factors"] is not None
        and r["factor_memory.initial_factors"].abs().sum() > 0
        for r in cold_ref
    )
    # Con accumulation_rows menor que los flujos del tramo se rechaza (sesión completa).
    too_small = trainer(
        shared,
        tmp_path / "small",
        model=make_model(shared, CFG_ONLINE, open_out=True),
        truncation=2,
        block_rows=16,
        accumulation_rows=2,
    )
    with pytest.raises(ValueError, match="repetir todos los flujos"):
        too_small.run()


@pytest.mark.parametrize("acc", [None, 16])
def test_resume_after_interruption_reproduces_continuous_run_bit_for_bit(
    shared,  # noqa: F811
    tmp_path,
    acc,
):
    options = dict(
        truncation=2,
        epochs=2,
        block_rows=16,
        checkpoint_updates=2,
        accumulation_rows=acc,
        selection=dict(metric="session_mae", patience=3, min_delta=0.0, stopping="fixed_budget"),
    )

    def build_engine(name):
        return trainer(
            shared,
            tmp_path / name,
            model=make_model(shared, CFG_ONLINE, open_out=True),
            **options,
        )

    continuous = build_engine(f"continuous-{acc}")
    expected = continuous.run()
    first = build_engine(f"interrupted-{acc}")
    stop = StopAtStep(4)
    stop.trainer = first
    assert first.run(stop=stop)["status"] == "paused"
    state = load_training_state(
        tmp_path / f"interrupted-{acc}/checkpoints", expected_identity=first.identity
    )
    saved = state["run"]["factor_memory"]
    assert saved["configuration"] == CFG_ONLINE.identity()
    assert sum(saved["updates"]) > 0
    second = build_engine(f"interrupted-{acc}")
    resumed = second.run(resume=True)
    assert resumed["status"] == "completed"
    assert first.audit + second.audit == continuous.audit
    assert_same_gradients(named_records(first) + named_records(second), named_records(continuous))
    assert resumed["history"] == expected["history"]


def test_modes_online_eval_frozen_and_random_basis_differ_as_expected(
    shared,  # noqa: F811
    tmp_path,
):
    results = {}
    for name, cfg in (("online", CFG_ONLINE), ("frozen", CFG_FROZEN), ("random", CFG_RANDOM)):
        engine = trainer(
            shared,
            tmp_path / name,
            model=make_model(shared, cfg, open_out=True),
            truncation=2,
            block_rows=16,
        )
        engine.run()
        results[name] = engine
    train_on = entries(results["online"].audit, "prediction", "train")
    train_fr = entries(results["frozen"].audit, "prediction", "train")
    train_rd = entries(results["random"].audit, "prediction", "train")
    val_on = entries(results["online"].audit, "prediction", "validation")
    val_fr = entries(results["frozen"].audit, "prediction", "validation")
    val_rd = entries(results["random"].audit, "prediction", "validation")
    # eval_frozen coincide en ajuste bit a bit con online, pero diverge en validación.
    assert train_fr == train_on
    assert val_fr != val_on
    # random_basis diverge tanto en ajuste como en validación.
    assert train_rd != train_on
    assert val_rd != val_on


def test_core_like_recipe_with_zero_median_and_session_ic_takes_finite_steps(tmp_path):
    sources = no_warmup_streams(tmp_path / "core_streams")
    spec = sources["train"].specification()
    gen = np.random.default_rng(7)
    stats_inputs = {
        name: gen.normal(
            0.0, 1.0, size=(120, 64, width) if name == "prices" else (120, width)
        ).astype(np.float32)
        for name, width in spec.dimensions.items()
    }
    for name in ("fundamentals", "macro"):
        stats_inputs[name][:, 1::3] = 1.0
        stats_inputs[name][:, 2::3] = np.abs(stats_inputs[name][:, 2::3])
    config = FinancialConfig(
        spec,
        hidden_size=32,
        variant="mac_online",
        head=QUANTILE_HEAD,
        gate_bias=dict(alpha_half_life=256.0, eta=0.5, theta=0.05),
        memory_residual_layer_norm=True,
        learning_scale=dict(clip=5.0, price_returns=True, zero_median=True),
        technical_branch=dict(
            columns=min(8, spec.dimensions["charts"]), hidden=32, depth=2, session_ranks=True
        ),
        median_gain="learned_median_gain_v1",
        factor_memory=asdict(CFG_ONLINE),
    )
    model = FinancialPredictor(config, dtype=torch.float32)
    stats = fit_statistics(
        config.learning_scale,
        stats_inputs,
        np.ones((120, 5), dtype=np.bool_),
        gen.normal(0, 0.02, 120),
    )
    model.set_learning_statistics(stats)
    # Optimizador real (AdamW) con 2 épocas: en la época 1 se abre out.weight y se ajustan las
    # puertas; en la época 2 (sin calentamiento) el gradiente alcanza también initial_factors.
    recipe = ChronologicalRecipe(
        truncation=2,
        loss=PINBALL,
        epochs=2,
        block_rows=16,
        accumulation_rows=16,
        observe_without_graph=True,
        max_grad_norm=1.0,
        learning_rate=1e-3,
        selection=dict(metric="session_mae", patience=5, min_delta=0.0, stopping="fixed_budget"),
        session_ic_term=dict(weight=3.0, minimum_rows=3),
    )
    before = {
        n: p.detach().clone() for n, p in model.named_parameters() if n.startswith("factor_memory.")
    }
    engine = ChronologicalTrainer(
        model,
        recipe,
        train=sources["train"],
        validation=sources["validation"],
        output=tmp_path / "core_like",
        audit=True,
    )
    report = engine.run()
    assert report["status"] == "completed"
    # El punto de recuperación guarda el estado final tras las 2 épocas (best podría ser época 0).
    latest = load_training_state(
        tmp_path / "core_like/checkpoints", expected_identity=engine.identity
    )
    trained = {k: v for k, v in latest["model"].items() if k.startswith("factor_memory.")}
    for name, before_tensor in before.items():
        after_tensor = trained[name]
        assert torch.isfinite(after_tensor).all(), name
        assert not torch.equal(after_tensor, before_tensor), f"Sin cambio en {name}"


def test_checkpoint_export_restore_barrier_and_code_identity(shared, tmp_path):  # noqa: F811
    engine = trainer(shared, tmp_path / "lfmsw", model=make_model(shared, CFG_ONLINE))
    control = trainer(shared, tmp_path / "control", model=make_model(shared, None))
    payload = engine._export(financial_run._Pass())
    assert payload["factor_memory"] is None
    plain = control._export(financial_run._Pass())
    assert "factor_memory" not in plain
    for target, saved in ((control, payload), (engine, plain)):
        with pytest.raises(ValueError, match="memoria factorial"):
            target._restore(saved)
    assert engine._restore(payload).factor_memory is None
    # Exportar con `factor_start` sin cerrar en barrera se rechaza.
    dirty = financial_run._Pass(factor_start=None)
    with pytest.raises(ValueError, match="barrera"):
        engine._export(dirty)
    # `_close` debe limpiar tanto `factor_memory` como `factor_start` en el propio `_Pass`.
    populated = financial_run._Pass(
        factor_memory=engine.predictor.initial_factor_memory(),
        factor_start=engine.predictor.initial_factor_memory(),
    )
    engine._close(populated)
    assert populated.factor_memory is None
    assert populated.factor_start is financial_run._UNSET
    mod = "mars_titan.models.titans.factor_memory"
    assert mod in engine._code() and mod not in control._code()
    assert ASSETS >= 5


def test_clip_groups_split_block_rejection_and_frozen_parent_guard(shared, tmp_path):  # noqa: F811
    import hashlib
    import json

    from mars_titan.training import mars_titan_walk_forward
    from mars_titan.training.financial_run import clip_parameter_groups
    from tests.models.titans.test_financial_adapter import specification

    model = make_model(shared, CFG_ONLINE)
    groups = clip_parameter_groups(model)
    lfmsw_params = [n for n, _ in model.named_parameters() if n.startswith("factor_memory.")]
    assert len(lfmsw_params) == 7
    for name in lfmsw_params:
        assert name in groups["mac"], name

    # Partir un instante en bloques menores que la sesión se rechaza.
    split = trainer(shared, tmp_path / "split", model=model, block_rows=2)
    with pytest.raises(ValueError, match="sesión completa"):
        split.run()

    # mars_titan_walk_forward rechaza un padre con LFM-SW antes de construir consumidor congelado.
    view = tmp_path / "view.json"
    view.write_text("{}\n")
    report = dict(
        kind=mars_titan_walk_forward.TITANS_KIND,
        status="completed",
        final_test_opened=False,
        checkpoint={},
        request=dict(
            variant="mac_online",
            seed=42,
            view_sha256=hashlib.sha256(view.read_bytes()).hexdigest(),
            predictor_changes=dict(factor_memory=asdict(CFG_ONLINE)),
        ),
        identity=dict(recipe=dict(predictor=dict(hidden_size=32, dtype="float64"))),
    )
    (tmp_path / "run.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="memoria factorial"):
        mars_titan_walk_forward._parent(tmp_path, view, 42)
    with pytest.raises(ValueError, match="memoria factorial"):
        mars_titan_walk_forward._frozen_parent(
            dict(predictor=dict(hidden_size=32, dtype="float64")),
            specification(),
            42,
            "cpu",
            tmp_path,
            report,
        )


def test_coexistence_with_crrm_and_session_hologram_in_trainer_replay_and_resume(
    shared,  # noqa: F811
    tmp_path,
):
    """LFM-SW convive con CRRM y session_hologram en ajuste, replay, pausa y reanudación."""
    from mars_titan.models.titans.cross_sectional_rank_memory import (
        CrossSectionalRankMemoryConfig,
    )
    from mars_titan.models.titans.session_hologram import SessionHologramConfig

    holo_cfg = SessionHologramConfig(
        bands=4, phase_dim=8, mode="interferometric", position="after_mac"
    )
    crrm_cfg = CrossSectionalRankMemoryConfig(
        dim=8, time_constants=(5, 21), ridge=1.0, minimum_rows=3, mode="matured_rank", seed=0
    )

    def build_combo():
        spec = shared["train"].specification()
        cfg = FinancialConfig(
            spec,
            hidden_size=32,
            variant="mac_online",
            memory_residual_layer_norm=True,
            gate_bias=dict(alpha_half_life=256.0, eta=0.5, theta=0.05),
            session_hologram=holo_cfg,
            factor_memory=CFG_ONLINE,
            cross_sectional_rank_memory=crrm_cfg,
        )
        m = FinancialPredictor(cfg, dtype=torch.float64)
        with torch.no_grad():
            m.session_hologram.gate.fill_(0.1)
            m.factor_memory.out.weight.fill_(0.05)
            m.cross_sectional_rank_memory.out.weight.fill_(0.1)
        m._seal_parameters()
        return m

    options = dict(
        truncation=2,
        epochs=2,
        block_rows=16,
        accumulation_rows=16,
        observe_without_graph=True,
        checkpoint_updates=2,
        selection=dict(metric="session_mae", patience=3, min_delta=0.0, stopping="fixed_budget"),
    )
    continuous = trainer(shared, tmp_path / "combo_cont", model=build_combo(), **options)
    expected = continuous.run()
    assert expected["status"] == "completed"

    first = trainer(shared, tmp_path / "combo_int", model=build_combo(), **options)
    stop = StopAtStep(3)
    stop.trainer = first
    assert first.run(stop=stop)["status"] == "paused"
    saved = load_training_state(
        tmp_path / "combo_int/checkpoints", expected_identity=first.identity
    )["run"]
    assert "factor_memory" in saved and "rank_memory" in saved
    assert sum(saved["factor_memory"]["updates"]) > 0

    second = trainer(shared, tmp_path / "combo_int", model=build_combo(), **options)
    resumed = second.run(resume=True)
    assert resumed["status"] == "completed"
    assert first.audit + second.audit == continuous.audit
    assert_same_gradients(named_records(first) + named_records(second), named_records(continuous))
    assert resumed["history"] == expected["history"]
