"""Pruebas TDD de la regla de parada adaptativa en `ChronologicalTrainer` y la ventana.

Verifica:
1. Comportamiento por defecto (opciones ausentes o en su valor por defecto) bit a bit idéntico.
2. Suavizado en `ChronologicalTrainer` con curva ruidosa de máximo tardío, curva monótona
   empeorando y cambio retroactivo de mejor época.
3. Reducción de tasa de aprendizaje con factor exacto, retorno a la mejor época guardada y
   reanudación exacta tras interrupción antes, durante (`lr_decay`) y después de una reducción.
4. Promedio de pesos `emit: {"average": k}` exacto frente a media manual en FP32 y FP64,
   `k = 1` idéntico a `"best"`, y prueba explícita de que memorias (`mac.persistent`,
   `mac.memory.initial_weights.*`), búferes y estado en línea provienen de la mejor época.
5. Numérica estricta: gradientes finitos con predicción constante en cero (FP32 y FP64).
6. Informe por ajuste (`stopping_report.json` y `fit["stopping_report"]`) con curva completa
   (incluido `session_rank_ic` de entrenamiento) y diagnóstico explícito.
7. Integración con `run_titans_window`, `comparison_campaign` y prueba sin fugas.
"""

import json
import math

import numpy as np
import pyarrow.parquet as pq
import pytest
import torch
from scipy.stats import spearmanr

from mars_titan.data.storage import sha256
from mars_titan.models.titans.financial import FinancialConfig, FinancialPredictor
from mars_titan.training import comparison_campaign as campaign
from mars_titan.training import titans_walk_forward as wf
from mars_titan.training.checkpoints import StopRequest, load_training_state
from mars_titan.training.financial_run import (
    ChronologicalRecipe,
    ChronologicalTrainer,
    parameter_roles,
)
from mars_titan.training.learning_hold import HOLD_ENV
from mars_titan.training.selection import (
    FIXED_BUDGET,
    SESSION_MAE,
    SESSION_RANK_IC,
    VALIDATION_PLATEAU,
)
from tests.learning_hold_lowering import allowed_hold
from tests.training.test_comparison_campaign import declaration as base_declaration
from tests.training.test_financial_run import (
    SELECTION,
    RecordingOptimizer,
    StopAtStep,
    corpus,
)
from tests.training.test_titans_walk_forward import (
    decision,
    unfused_attention,
)
from tests.training.test_titans_walk_forward import (
    recipe as build_wf_recipe,
)
from tests.training.test_titans_walk_forward import (
    table as wf_table,
)
from tests.training.test_titans_walk_forward import (
    views as build_wf_views,
)
from tests.training.test_titans_walk_forward import (
    window as run_wf_window,
)

PLATEAU_SELECTION = dict(
    metric=SESSION_MAE,
    stopping=VALIDATION_PLATEAU,
    patience=2,
    min_delta=0.0,
    initial_state="reported",
)


@pytest.fixture(autouse=True)
def explicit_fastpath():
    previous = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    yield
    torch.backends.mha.set_fastpath_enabled(previous)


@pytest.fixture(scope="module")
def shared(tmp_path_factory):
    return corpus(tmp_path_factory.mktemp("adaptive-stopping"))[1]


@pytest.fixture(autouse=True)
def allow_local_training(tmp_path, monkeypatch):
    hold = allowed_hold(tmp_path / "allow-training.json")
    monkeypatch.setenv(HOLD_ENV, str(hold))
    return hold


def make_predictor(streams, *, variant="mac_online", seed=42, dtype=torch.float64):
    specification = streams["train"].specification()
    config = FinancialConfig(specification, variant=variant, hidden_size=32, seed=seed)
    return FinancialPredictor(config, dtype=dtype)


class DriftOptimizer:
    """Optimizador determinista y verificable sin heredar de AdamW cuando no hace falta."""

    def __init__(self, groups, *, lr=1e-2):
        self.param_groups = [
            dict(group, params=list(group["params"]), lr=float(lr)) for group in groups
        ]
        self.calls = 0
        self.seen_lrs = []

    def step(self):
        self.calls += 1
        self.seen_lrs.append([group["lr"] for group in self.param_groups])
        with torch.no_grad():
            for group in self.param_groups:
                step_lr = group["lr"]
                for param in group["params"]:
                    if param.grad is not None:
                        param.add_(param.grad, alpha=-step_lr)

    def zero_grad(self, set_to_none=True):
        assert set_to_none is True
        for group in self.param_groups:
            for param in group["params"]:
                param.grad = None

    def state_dict(self):
        return dict(
            calls=self.calls,
            lrs=[group["lr"] for group in self.param_groups],
            roles=[group["role"] for group in self.param_groups],
        )

    def load_state_dict(self, state):
        assert state["roles"] == [group["role"] for group in self.param_groups]
        self.calls = state["calls"]
        for group, lr in zip(self.param_groups, state["lrs"], strict=True):
            group["lr"] = float(lr)


def make_trainer(
    streams,
    output,
    *,
    selection,
    epochs=8,
    dtype=torch.float64,
    variant="mac_online",
    learning_rate=1e-2,
    optimizer_factory=None,
    checkpoint_updates=256,
    model=None,
    **extra,
):
    recipe = ChronologicalRecipe(
        truncation=3,
        epochs=epochs,
        block_rows=2,
        learning_rate=learning_rate,
        selection=selection,
        checkpoint_updates=checkpoint_updates,
        **extra,
    )
    return ChronologicalTrainer(
        model or make_predictor(streams, variant=variant, dtype=dtype),
        recipe,
        train=streams["train"],
        validation=streams["validation"],
        output=output,
        optimizer_factory=optimizer_factory,
    )


def scripted_validation(values):
    iterator = iter(values)

    def evaluate(source, *, stop=None, rows=None):
        val = next(iterator)
        return dict(session_mae=val, session_mse=val**2, labels=1, samples=1, session_count=1)

    return evaluate


def test_default_and_explicit_default_options_keep_identical_recipe_and_run_identity(
    shared, tmp_path
):
    base = ChronologicalRecipe(truncation=3, epochs=4, block_rows=2, selection=PLATEAU_SELECTION)
    with_smooth1 = ChronologicalRecipe(
        truncation=3,
        epochs=4,
        block_rows=2,
        selection=dict(PLATEAU_SELECTION, smoothing=1),
    )
    with_emit_best = ChronologicalRecipe(
        truncation=3,
        epochs=4,
        block_rows=2,
        selection=dict(PLATEAU_SELECTION, emit="best"),
    )
    with_emit_k1 = ChronologicalRecipe(
        truncation=3,
        epochs=4,
        block_rows=2,
        selection=dict(PLATEAU_SELECTION, emit={"average": 1}),
    )
    assert (
        base.identity()
        == with_smooth1.identity()
        == with_emit_best.identity()
        == with_emit_k1.identity()
    )
    assert "smoothing_rule" not in base.identity()
    assert "learning_rate_decay_rule" not in base.identity()
    assert "emit_weight_averaging" not in base.identity()

    # Cualquier opción no por defecto cambia la identidad de la receta y del entrenador.
    changed = [
        ChronologicalRecipe(
            truncation=3, epochs=4, block_rows=2, selection=dict(PLATEAU_SELECTION, smoothing=3)
        ),
        ChronologicalRecipe(
            truncation=3,
            epochs=4,
            block_rows=2,
            selection=dict(PLATEAU_SELECTION, learning_rate_decay={"factor": 0.5, "reductions": 2}),
        ),
        ChronologicalRecipe(
            truncation=3,
            epochs=4,
            block_rows=2,
            selection=dict(PLATEAU_SELECTION, emit={"average": 2}),
        ),
    ]
    for recipe in changed:
        assert recipe.identity() != base.identity()

    # Rechazar combinación learning_rate_decay + schedule_free.
    with pytest.raises(ValueError, match="schedule_free"):
        ChronologicalRecipe(
            truncation=3,
            epochs=4,
            block_rows=2,
            schedule_free=dict(warmup_steps=0, beta1=0.9),
            selection=dict(PLATEAU_SELECTION, learning_rate_decay={"factor": 0.5, "reductions": 1}),
        )

    # Con opciones ausentes vs por defecto, el ajuste entero produce resultados bit a bit iguales.
    reports = {}
    for label, sel in (
        ("absent", PLATEAU_SELECTION),
        ("defaults", dict(PLATEAU_SELECTION, smoothing=1, emit="best")),
        ("k1", dict(PLATEAU_SELECTION, smoothing=1, emit={"average": 1})),
    ):
        engine = make_trainer(
            shared,
            tmp_path / label,
            selection=sel,
            epochs=3,
            optimizer_factory=lambda groups: DriftOptimizer(groups, lr=1e-2),
        )
        reports[label] = engine.run()
    assert reports["absent"]["run_id"] == reports["defaults"]["run_id"] == reports["k1"]["run_id"]
    assert (
        reports["absent"]["best_epoch"]
        == reports["defaults"]["best_epoch"]
        == reports["k1"]["best_epoch"]
    )
    assert (
        reports["absent"]["best_score"]
        == reports["defaults"]["best_score"]
        == reports["k1"]["best_score"]
    )
    assert (
        reports["absent"]["best_checkpoint"]["sha256"]
        == reports["defaults"]["best_checkpoint"]["sha256"]
        == reports["k1"]["best_checkpoint"]["sha256"]
    )
    assert "epoch_states" not in reports["absent"]


def test_trainer_smoothing_reaches_late_peak_and_promotes_retroactive_best_checkpoint(
    shared, tmp_path, monkeypatch
):
    # Curva con pico espurio en ep 2 y máximo real tardío en ep 7; luego ep 8 empeora y
    # promueve retroactivamente ep 7 sobre ep 8.
    scores = [0.90, 0.50, 0.30, 0.45, 0.40, 0.36, 0.25, 0.18, 0.20, 0.60, 0.65, 0.70]
    plain_sel = dict(PLATEAU_SELECTION, patience=3)
    smooth_sel = dict(plain_sel, smoothing=3)

    plain = make_trainer(
        shared,
        tmp_path / "plain",
        selection=plain_sel,
        epochs=11,
        optimizer_factory=lambda groups: DriftOptimizer(groups, lr=5e-3),
    )
    smooth = make_trainer(
        shared,
        tmp_path / "smooth",
        selection=smooth_sel,
        epochs=11,
        optimizer_factory=lambda groups: DriftOptimizer(groups, lr=5e-3),
    )
    monkeypatch.setattr(plain, "evaluate", scripted_validation(scores))
    monkeypatch.setattr(smooth, "evaluate", scripted_validation(scores))

    def fast_pass(engine):
        def _pass(run, cursor, stop, save):
            engine.global_step += 1
            with torch.no_grad():
                engine.predictor.head.bias.add_(1.0)
                engine.predictor._seal_parameters()
            return dict(session_mae=0.1, session_mse=0.01, mean_loss=0.1, updates=1, labels=1)

        return _pass

    monkeypatch.setattr(plain, "_train_pass", fast_pass(plain))
    monkeypatch.setattr(smooth, "_train_pass", fast_pass(smooth))

    initial_bias = smooth.predictor.head.bias.item()
    plain_report = plain.run()
    smooth_report = smooth.run()

    # Sin suavizado para en época 5 atrapado en ep 2.
    assert plain_report["best_epoch"] == 2
    assert plain_report["history"][-1]["epoch"] == 5
    # Con suavizado llega al pico real (ep 7) y para en ep 10; el estado cargado es el de ep 7.
    assert smooth_report["best_epoch"] == 7
    assert smooth_report["history"][-1]["epoch"] == 10
    assert smooth_report["best_score"] == pytest.approx(scores[7])
    assert smooth_report["best_smoothed_score"] == pytest.approx(
        (scores[6] + scores[7] + scores[8]) / 3.0
    )
    # El bias sumó 1.0 por época -> en época 7 vale exactamente initial_bias + 7.0.
    assert smooth.predictor.head.bias.item() == pytest.approx(initial_bias + 7.0)
    best_saved = load_training_state(
        tmp_path / "smooth/checkpoints", expected_identity=smooth.identity, selection="best"
    )
    assert best_saved["model"]["head.bias"].item() == pytest.approx(initial_bias + 7.0)


def test_learning_rate_decay_scales_exact_factor_restores_best_and_matches_resumes_bit_for_bit(
    shared, tmp_path, monkeypatch
):
    # Real AdamW sobre CPU en FP64.
    # Curva: ep0=0.9, ep1=0.30 (best=1), ep2=0.40 (s=1), ep3=0.45 (s=2 -> reducción 1, restaura 1)
    # ep4=0.35 (s=1), ep5=0.38 (s=2 -> reducción 2, restaura 1 aunque best no cambió!)
    # ep6=0.25 (best=6, s=0), ep7=0.26 (s=1), ep8=0.27 (s=2 -> parada definitiva).
    curve = [0.90, 0.30, 0.40, 0.45, 0.35, 0.38, 0.25, 0.26, 0.27]
    sel = dict(
        PLATEAU_SELECTION,
        patience=2,
        learning_rate_decay={"factor": 0.25, "reductions": 2},
    )

    def build(name):
        engine = make_trainer(
            shared,
            tmp_path / name,
            selection=sel,
            epochs=10,
            learning_rate=1e-2,
            checkpoint_updates=2,
        )
        return engine

    # 1. Ejecución continua de referencia, registrando lr efectivo en cada época.
    continuous = build("continuous")
    observed_lrs = []
    real_train_pass = continuous._train_pass

    def spy_train_pass(run, cursor, stop, save):
        if cursor["stage"] == "start":
            observed_lrs.append(
                (cursor["epoch"] + 1, [g["lr"] for g in continuous.optimizer.param_groups])
            )
        return real_train_pass(run, cursor, stop, save)

    monkeypatch.setattr(continuous, "_train_pass", spy_train_pass)
    monkeypatch.setattr(continuous, "evaluate", scripted_validation(curve))
    expected = continuous.run()

    assert expected["status"] == "completed"
    assert expected["best_epoch"] == 6
    assert expected["history"][-1]["epoch"] == 8
    assert len(expected["learning_rate_reductions"]) == 2
    assert expected["learning_rate_reductions"][0]["at_epoch"] == 3
    assert expected["learning_rate_reductions"][0]["restored_epoch"] == 1
    assert expected["learning_rate_reductions"][0]["new_learning_rate"] == pytest.approx(2.5e-3)
    assert expected["learning_rate_reductions"][1]["at_epoch"] == 5
    assert expected["learning_rate_reductions"][1]["restored_epoch"] == 1
    assert expected["learning_rate_reductions"][1]["new_learning_rate"] == pytest.approx(6.25e-4)

    # Comprobar que la tasa exacta se aplicó a TODOS los grupos de parámetros en cada época.
    lr_by_epoch = dict(observed_lrs)
    for ep in (1, 2, 3):
        assert all(lr == pytest.approx(1e-2) for lr in lr_by_epoch[ep])
    for ep in (4, 5):
        assert all(lr == pytest.approx(2.5e-3) for lr in lr_by_epoch[ep])
    for ep in (6, 7, 8):
        assert all(lr == pytest.approx(6.25e-4) for lr in lr_by_epoch[ep])

    # 2. Reanudación tras interrupciones en los 3 puntos clave:
    #    (a) antes de la primera reducción (a mitad de la época 2),
    #    (b) DURANTE la primera reducción (en phase="lr_decay" tras validar época 3),
    #    (c) DESPUÉS de la primera reducción (a mitad de la época 4).
    expected_weights = {
        k: v.detach().clone()
        for k, v in continuous.predictor.state_dict().items()
        if torch.is_tensor(v)
    }

    for mode in ("before_decay", "during_decay", "after_decay"):
        folder = f"resumed-{mode}"
        first = build(folder)
        val_calls = {"n": 0}

        def eval_with_stop(
            source, *, stop=None, rows=None, _mode=mode, _stop=None, _calls=val_calls
        ):
            idx = _calls["n"]
            _calls["n"] += 1
            if _mode == "during_decay" and idx == 3:
                _stop.request_stop()
            val = curve[idx]
            return dict(session_mae=val, session_mse=val**2, labels=1, samples=1, session_count=1)

        if mode == "before_decay":
            stop = StopAtStep(3)
            stop.trainer = first
            monkeypatch.setattr(first, "evaluate", scripted_validation(curve))
            paused = first.run(stop=stop)
        elif mode == "during_decay":
            stop = StopRequest()
            monkeypatch.setattr(
                first,
                "evaluate",
                lambda source, *, stop=None, rows=None, _s=stop: eval_with_stop(
                    source, stop=stop, rows=rows, _stop=_s
                ),
            )
            paused = first.run(stop=stop)
            state = load_training_state(
                tmp_path / folder / "checkpoints", expected_identity=first.identity
            )
            assert state["cursor"] == dict(epoch=3, phase="lr_decay")
        else:
            # Pausar a mitad de la época 4 (después de la primera reducción en época 3).
            step_after_ep3 = expected["history"][3]["global_step"]
            stop = StopAtStep(step_after_ep3 + 2)
            stop.trainer = first
            monkeypatch.setattr(first, "evaluate", scripted_validation(curve))
            paused = first.run(stop=stop)

        assert paused["status"] == "paused"

        second = build(folder)
        remaining_start = len(paused["history"])
        monkeypatch.setattr(second, "evaluate", scripted_validation(curve[remaining_start:]))
        resumed = second.run(resume=True)

        assert resumed["status"] == "completed"
        assert resumed["history"] == expected["history"]
        assert resumed["best_epoch"] == expected["best_epoch"]
        assert resumed["best_score"] == expected["best_score"]
        assert resumed["learning_rate_reductions"] == expected["learning_rate_reductions"]
        assert resumed["global_step"] == expected["global_step"]
        assert resumed["stopping_report"] == expected["stopping_report"]
        for name, value in second.predictor.state_dict().items():
            if torch.is_tensor(value):
                assert torch.equal(value, expected_weights[name]), (mode, name)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_emit_average_matches_manual_mean_of_shared_weights_and_keeps_memories_from_best(
    shared, tmp_path, monkeypatch, dtype
):
    # Curva: ep 1=0.40, ep 2=0.10 (MEJOR), ep 3=0.35, ep 4=0.20 (2ª mejor), ep 5=0.15 (3ª -> 2ª)
    # Top-3 exacto: [2, 5, 4] (scores: 0.10, 0.15, 0.20).
    scores = [0.90, 0.40, 0.10, 0.35, 0.20, 0.15]
    sel = dict(
        metric=SESSION_MAE,
        stopping=FIXED_BUDGET,
        patience=5,
        min_delta=0.0,
        initial_state="reported",
        emit={"average": 3},
    )
    engine = make_trainer(
        shared,
        tmp_path / f"avg-{dtype}",
        selection=sel,
        epochs=5,
        dtype=dtype,
        optimizer_factory=lambda groups: DriftOptimizer(groups, lr=1e-2),
    )
    monkeypatch.setattr(engine, "evaluate", scripted_validation(scores))

    # Guardar una copia exacta de todos los parámetros tras cada época (1..5), moviendo
    # tanto pesos compartidos como memorias (`mac.persistent` y `mac.memory.initial_weights.*`).
    snapshots = {}

    def synthetic_pass(run, cursor, stop, save):
        ep = cursor["epoch"] + 1
        engine.global_step += 1
        with torch.no_grad():
            for _name, param in engine.predictor.named_parameters():
                # Desplazamiento distinto y no lineal por época para que la media de {2,4,5}
                # difiera claramente del valor en ep 2 (el mejor).
                param.add_(float(ep * ep) * 1e-3)
            engine.predictor._seal_parameters()
        snapshots[ep] = {
            k: v.detach().clone() if torch.is_tensor(v) else v
            for k, v in engine.predictor.state_dict().items()
        }
        return dict(session_mae=0.1, session_mse=0.01, mean_loss=0.1, updates=1, labels=1)

    monkeypatch.setattr(engine, "_train_pass", synthetic_pass)
    report = engine.run()

    assert report["status"] == "completed"
    assert report["best_epoch"] == 2
    assert report["averaged_epochs"] == [2, 5, 4]
    assert report["emit"]["mode"] == "average"
    assert report["emit"]["memories_and_online_state_epoch"] == 2

    roles = parameter_roles(engine.predictor)
    shared_names = set(roles["shared"])
    memory_names = set(roles["persistent_memory"]) | set(roles["initial_fast_weights"])
    assert shared_names and memory_names

    chosen_epochs = [2, 5, 4]
    emitted_state = engine.predictor.state_dict()
    for name, value in emitted_state.items():
        if not torch.is_tensor(value):
            assert value == snapshots[2][name]
            continue
        if name in shared_names:
            # Media aritmética exacta hecha a mano (acumulada en FP64).
            manual = (
                sum(snapshots[ep][name].to(torch.float64) for ep in chosen_epochs)
                / len(chosen_epochs)
            ).to(dtype=value.dtype)
            assert torch.equal(value, manual), f"Peso compartido {name} no coincide con la media"
            # Confirmar que difiere del mejor individual (si no, la prueba sería trivial).
            assert not torch.equal(value, snapshots[2][name])
        elif name in memory_names:
            # Memorias: tomadas literalmente de la mejor época (ep 2), NO promediadas.
            assert torch.equal(value, snapshots[2][name]), f"Memoria {name} no vino de la mejor"
            mean_mem = (
                sum(snapshots[ep][name].to(torch.float64) for ep in chosen_epochs)
                / len(chosen_epochs)
            ).to(dtype=value.dtype)
            assert not torch.equal(value, mean_mem), f"La prueba de memoria {name} es degenerada"
        else:
            # Búferes: tomados de la mejor época.
            assert torch.equal(value, snapshots[2][name]), f"Búfer {name} no vino de la mejor"

    # El estado en línea inicial (`initial_state`, pesos rápidos y momentos) proviene de la
    # mejor época (ep 2), nunca de un promedio de estados rápidos.
    best_reference = make_predictor(shared, dtype=dtype)
    best_reference.load_state_dict(snapshots[2])
    best_reference._seal_parameters()
    emitted_online = engine.predictor.initial_state(("US/A",))
    best_online = best_reference.initial_state(("US/A",))
    assert emitted_online.mac is not None and best_online.mac is not None
    for emitted_w, best_w in zip(
        emitted_online.mac.memory.weights, best_online.mac.memory.weights, strict=True
    ):
        assert torch.equal(emitted_w, best_w)
    for emitted_m, best_m in zip(
        emitted_online.mac.memory.momentum, best_online.mac.memory.momentum, strict=True
    ):
        assert torch.equal(emitted_m, best_m)

    # El `best` guardado en checkpoints contiene exactamente el estado emitido promediado,
    # y reanudar una ejecución completada vuelve a cargar ese estado exacto.
    saved_best = load_training_state(
        tmp_path / f"avg-{dtype}/checkpoints", expected_identity=engine.identity, selection="best"
    )
    for name, value in emitted_state.items():
        if torch.is_tensor(value):
            assert torch.equal(saved_best["model"][name], value)

    resumed_engine = make_trainer(
        shared,
        tmp_path / f"avg-{dtype}",
        selection=sel,
        epochs=5,
        dtype=dtype,
        optimizer_factory=lambda groups: DriftOptimizer(groups, lr=1e-2),
    )
    resumed_report = resumed_engine.run(resume=True)
    assert resumed_report["best_checkpoint"] == report["best_checkpoint"]
    for name, value in resumed_engine.predictor.state_dict().items():
        if torch.is_tensor(value):
            assert torch.equal(value, emitted_state[name])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_constant_zero_prediction_keeps_finite_gradients_with_adaptive_stopping(
    shared, tmp_path, dtype
):
    model = make_predictor(shared, dtype=dtype)
    with torch.no_grad():
        model.head.weight.zero_()
        model.head.bias.zero_()
        model._seal_parameters()

    sel = dict(
        PLATEAU_SELECTION,
        patience=1,
        smoothing=3,
        learning_rate_decay={"factor": 0.5, "reductions": 1},
        emit={"average": 2},
    )
    recorder = []

    class CheckingAdamW(torch.optim.AdamW):
        def step(self, closure=None):
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is not None:
                        assert torch.isfinite(p.grad).all()
                        recorder.append(float(p.grad.detach().abs().max()))
            return super().step(closure=closure)

    engine = make_trainer(
        shared,
        tmp_path / f"const-{dtype}",
        selection=sel,
        epochs=3,
        dtype=dtype,
        model=model,
        optimizer_factory=lambda groups: CheckingAdamW(groups, lr=1e-3),
    )
    report = engine.run()
    assert report["status"] == "completed"
    assert recorder and all(math.isfinite(v) for v in recorder)
    stopping_path = tmp_path / f"const-{dtype}/stopping_report.json"
    assert stopping_path.is_file()
    loaded = json.loads(stopping_path.read_text())
    assert loaded == report["stopping_report"]


def test_walk_forward_window_and_campaign_support_adaptive_stopping_without_leakage(tmp_path):
    # 1. Validación en comparison_campaign: fit admite opciones nuevas, reader las rechaza.
    doc = base_declaration()
    doc["stopping"]["fit"] = dict(
        metric=SESSION_RANK_IC,
        min_assets=3,
        stopping=VALIDATION_PLATEAU,
        patience=3,
        min_delta=0.0,
        max_epochs=12,
        initial_state="reported",
        smoothing=3,
        learning_rate_decay={"factor": 0.5, "reductions": 2},
        emit={"average": 3},
    )
    campaign.checked_declaration(doc)
    bad_reader = base_declaration()
    bad_reader["stopping"]["reader"] = dict(bad_reader["stopping"]["reader"], smoothing=3)
    with pytest.raises(ValueError):
        campaign.checked_declaration(bad_reader)

    # 2. Ventana walk-forward completa con IC de rangos, suavizado, decay, promedio e informe,
    #    y prueba sin fugas perturbando calibración/evaluación.
    stopping_rule = dict(
        metric=SESSION_RANK_IC,
        min_assets=3,
        stopping=VALIDATION_PLATEAU,
        patience=1,
        min_delta=0.0,
        max_epochs=3,
        initial_state="reported",
        smoothing=3,
        learning_rate_decay={"factor": 0.5, "reductions": 1},
        emit={"average": 2},
    )
    reports = {}
    with unfused_attention():
        for tag, perturb in (("clean", {}), ("future", dict(perturb_after=decision("2023-10-02")))):
            view, protocol = build_wf_views(tmp_path / tag, **perturb)
            rec = build_wf_recipe(tmp_path / tag)
            rep, _ = run_wf_window(
                view,
                protocol,
                rec,
                tmp_path / tag / "run",
                stopping=stopping_rule,
                optimizer_factory=lambda groups: DriftOptimizer(groups, lr=1e-3),
            )
            reports[tag] = rep

    clean, future = reports["clean"], reports["future"]
    assert clean["status"] == "completed" and future["status"] == "completed"
    # Sin fugas: perturbar calibración/evaluación no cambia ni un bit del ajuste, informe ni val.
    clean_fit = json.loads((tmp_path / "clean/run/fit/run.json").read_text())
    future_fit = json.loads((tmp_path / "future/run/fit/run.json").read_text())
    assert clean_fit["history"] == future_fit["history"]
    assert clean_fit["stopping_report"] == future_fit["stopping_report"]
    assert (
        clean["predictions"]["validation"]["sha256"]
        == future["predictions"]["validation"]["sha256"]
    )
    assert (
        clean["predictions"]["evaluation"]["sha256"]
        != future["predictions"]["evaluation"]["sha256"]
    )

    # Comprobar que el informe de parada existe junto a los resultados y tiene curva + diagnóstico.
    sr_ref = clean["fit"]["stopping_report"]
    sr_path = tmp_path / "clean/run" / sr_ref["path"]
    assert sr_path.is_file() and sha256(sr_path) == sr_ref["sha256"]
    sr = json.loads(sr_path.read_text())
    assert sr["metric"] == SESSION_RANK_IC
    assert len(sr["curve"]) == len(clean_fit["history"])
    # El entrenamiento registró session_rank_ic por época.
    assert all("session_rank_ic" in entry["train"] for entry in clean_fit["history"][1:])
    diag = sr["diagnosis"]
    assert diag["best_epoch"] == clean_fit["best_epoch"]
    assert diag["epochs_after_best"] == sr["curve"][-1]["epoch"] - clean_fit["best_epoch"]
    assert "stopped_by_plateau" in diag
    assert "best_epoch_train_validation_difference" in diag
    # `validation_selection_check` documenta el promedio de épocas sin fallar.
    assert clean["validation_selection_check"]["averaged_epochs"] == clean_fit["averaged_epochs"]

    # Comprobar que carry_titans sobre esta ventana reproduce exactamente la validación emitida.
    clean_view = tmp_path / "clean/views/fold-000/manifest.json"
    with unfused_attention():
        carried = wf.carry_titans(
            tmp_path / "clean/run",
            clean_view,
            clean_view,
            tmp_path / "carried",
            device="cpu",
            regenerate=True,
        )
    assert (
        carried["predictions"]["validation"]["sha256"]
        == clean["predictions"]["validation"]["sha256"]
    )
    assert pq.read_table(tmp_path / "carried/validation-predictions.parquet").equals(
        pq.read_table(tmp_path / "clean/run/validation-predictions.parquet")
    )
    assert wf_table(tmp_path / "clean/run", "validation")["sample_id"]


def test_train_session_rank_ic_matches_manual_spearman_and_resumes_bit_for_bit(shared, tmp_path):
    sel = dict(
        metric=SESSION_RANK_IC,
        min_assets=3,
        stopping=FIXED_BUDGET,
        patience=3,
        min_delta=0.0,
        initial_state="reported",
    )

    class SimpleRankScorer:
        def rows(self):
            return []

        def score(self, epoch, rows, metrics):
            return dict(session_rank_ic=0.1)

    def build(name):
        engine = make_trainer(
            shared,
            tmp_path / name,
            selection=sel,
            epochs=1,
            checkpoint_updates=2,
            optimizer_factory=RecordingOptimizer,
        )
        engine.epoch_scorer = SimpleRankScorer()
        return engine

    continuous = build("cont")
    expected = continuous.run()

    # Reanudar a mitad de época debe reproducir exactamente train.session_rank_ic.
    stop = StopAtStep(4)
    first = build("res")
    stop.trainer = first
    assert first.run(stop=stop)["status"] == "paused"
    second = build("res")
    resumed = second.run(resume=True)
    assert resumed["history"] == expected["history"]

    # Verificar el valor contra spearmanr manual sobre las predicciones emitidas en train.
    train_ic = expected["history"][1]["train"]["session_rank_ic"]
    # Extraer predicciones y etiquetas del recorrido manual con el mismo modelo.
    check = make_trainer(
        shared,
        tmp_path / "manual",
        selection=dict(SELECTION),
        epochs=1,
        optimizer_factory=RecordingOptimizer,
    )
    rows = []
    check.evaluate(shared["train"], rows=rows)
    by_session = {}
    for flow, at, pred, target, _ in rows:
        by_session.setdefault((flow.split("/", 1)[0], at), ([], []))[0].append(pred)
        by_session[(flow.split("/", 1)[0], at)][1].append(target)
    manual_ics = []
    for preds, targets in by_session.values():
        if len(preds) >= 3 and np.ptp(preds) > 0 and np.ptp(targets) > 0:
            r = float(spearmanr(targets, preds).statistic)
            if math.isfinite(r):
                manual_ics.append(r)
    expected_ic = float(np.asarray(manual_ics).mean()) if manual_ics else None
    assert train_ic == pytest.approx(expected_ic, abs=1e-12)


def test_window_defaults_identity_smoothing_best_check_and_non_adaptive_trainers(tmp_path):
    from mars_titan.training.candidate_run import CandidateRecipe
    from mars_titan.training.mars_titan_run import ReadoutRecipe

    # Otros entrenadores sin soporte adaptativo rechazan las opciones nuevas.
    with pytest.raises(ValueError):
        ReadoutRecipe(selection=dict(SELECTION, smoothing=3))
    with pytest.raises(ValueError):
        CandidateRecipe(selection=dict(SELECTION, emit={"average": 2}))

    # En run_titans_window, omitir opciones vs smoothing=1 / emit="best" / emit={"average": 1}
    # produce exactamente el mismo run_id, mejor época y predicciones Parquet.
    view, protocol = build_wf_views(tmp_path / "wf")
    rec = build_wf_recipe(tmp_path / "wf")
    base_rule = dict(
        metric=SESSION_RANK_IC,
        min_assets=3,
        stopping=VALIDATION_PLATEAU,
        patience=2,
        min_delta=0.0,
        max_epochs=2,
        initial_state="reported",
    )
    outputs = {}
    with unfused_attention():
        for name, rule in (
            ("plain", base_rule),
            ("defaults", dict(base_rule, smoothing=1, emit="best")),
            ("k1", dict(base_rule, emit={"average": 1})),
            ("smooth3", dict(base_rule, smoothing=3)),
        ):
            rep, _ = run_wf_window(
                view,
                protocol,
                rec,
                tmp_path / f"run-{name}",
                stopping=rule,
                optimizer_factory=lambda groups: DriftOptimizer(groups, lr=1e-3),
            )
            outputs[name] = rep

    assert outputs["plain"]["run_id"] == outputs["defaults"]["run_id"] == outputs["k1"]["run_id"]
    for part in ("validation", "calibration", "evaluation"):
        assert (
            outputs["plain"]["predictions"][part]["sha256"]
            == outputs["defaults"]["predictions"][part]["sha256"]
            == outputs["k1"]["predictions"][part]["sha256"]
        )
    # Con smoothing=3 y emit="best", run_id cambia y validation_selection_check conserva dif. 0.
    assert outputs["smooth3"]["run_id"] != outputs["plain"]["run_id"]
    assert outputs["smooth3"]["validation_selection_check"]["absolute_difference"] == 0.0
    assert "averaged_epochs" not in outputs["smooth3"]["validation_selection_check"]


def test_walk_forward_window_resumes_bit_for_bit_with_adaptive_stopping(tmp_path):
    view, protocol = build_wf_views(tmp_path / "wf")
    rec = build_wf_recipe(tmp_path / "wf")
    stopping_rule = dict(
        metric=SESSION_RANK_IC,
        min_assets=3,
        stopping=VALIDATION_PLATEAU,
        patience=1,
        min_delta=0.0,
        max_epochs=3,
        initial_state="reported",
        smoothing=3,
        learning_rate_decay={"factor": 0.5, "reductions": 1},
        emit={"average": 2},
    )

    with unfused_attention():
        expected, _ = run_wf_window(
            view,
            protocol,
            rec,
            tmp_path / "continuous",
            stopping=stopping_rule,
            optimizer_factory=lambda groups: DriftOptimizer(groups, lr=1e-3),
        )
        # Pausar a mitad del ajuste y reanudar la misma ventana walk-forward.
        optimizers = []

        def make_opt(groups):
            opt = DriftOptimizer(groups, lr=1e-3)
            optimizers.append(opt)
            return opt

        class StopAfterTwoSteps:
            @property
            def requested(self):
                return bool(optimizers) and optimizers[-1].calls >= 2

        paused, _ = run_wf_window(
            view,
            protocol,
            rec,
            tmp_path / "resumed",
            stopping=stopping_rule,
            stop=StopAfterTwoSteps(),
            optimizer_factory=make_opt,
        )
        assert paused["status"] == "paused"
        resumed, _ = run_wf_window(
            view,
            protocol,
            rec,
            tmp_path / "resumed",
            stopping=stopping_rule,
            optimizer_factory=make_opt,
        )

    assert resumed["status"] == "completed"
    assert resumed["run_id"] == expected["run_id"]
    assert resumed["fit"]["stopping_report"] == expected["fit"]["stopping_report"]
    for part in ("validation", "calibration", "evaluation"):
        assert resumed["predictions"][part]["sha256"] == expected["predictions"][part]["sha256"]
        assert pq.read_table(tmp_path / f"resumed/{part}-predictions.parquet").equals(
            pq.read_table(tmp_path / f"continuous/{part}-predictions.parquet")
        )
