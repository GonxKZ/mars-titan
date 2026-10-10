"""Adaptadores de Titans-MAC en el entrenador cronológico, comprobados hasta el paso.

El optimizador registra gradientes sin modificar pesos, así que cada brazo recorre los mismos
tramos que su padre y sus predicciones deben coincidir exactamente con las del padre.
"""

import copy
import json
from pathlib import Path

import pytest
import torch

from mars_titan.models.predictive_adaptation import (
    adapter_names,
    attach_adapters,
    base_digest,
    trainable_parameters,
)
from mars_titan.models.quantile_head import QUANTILE_HEAD
from mars_titan.models.titans.financial import VARIANTS, FinancialConfig, FinancialPredictor
from mars_titan.posttraining import chronological_matrix as cm
from mars_titan.posttraining.adapter_matrix import read_matrix
from mars_titan.training.checkpoints import load_training_state
from mars_titan.training.financial_run import (
    ChronologicalInference,
    ChronologicalRecipe,
    ChronologicalTrainer,
    parameter_roles,
)
from mars_titan.training.selection import FIXED_BUDGET
from tests.training.test_financial_run import (
    SELECTION,
    RecordingOptimizer,
    StopAtStep,
    corpus,
    named_records,
)

ROOT = Path(__file__).resolve().parents[2]
MATRIX_PATH = ROOT / "configs/posttraining/adapter-matrix-v3.json"


@pytest.fixture(autouse=True)
def explicit_fastpath():
    previous = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    yield
    torch.backends.mha.set_fastpath_enabled(previous)


@pytest.fixture(scope="module")
def matrix():
    document, _ = read_matrix(MATRIX_PATH)
    return document


@pytest.fixture(scope="module")
def shared(tmp_path_factory):
    return corpus(tmp_path_factory.mktemp("titans-adapters"))


def predictor(streams, variant, seed=42):
    """Padre de cuantiles en float64, como los brazos de la campaña con máscaras."""
    specification = streams["train"].specification()
    config = FinancialConfig(
        specification, variant=variant, hidden_size=32, seed=seed, head=QUANTILE_HEAD
    )
    return FinancialPredictor(config, dtype=torch.float64)


def clone(model):
    """Copia sellada de nuevo: la huella de firma usa la identidad de cada tensor."""
    result = copy.deepcopy(model)
    result._seal_parameters()
    return result


def adapted(model, matrix, points, seed=5):
    """Copia del padre con los adaptadores del brazo, sellada de nuevo."""
    result = copy.deepcopy(model)
    # Los subconjuntos de la variedad (sesgos y normalizaciones) se enumeran sobre el modelo.
    attach_adapters(result, cm.titans_targets(matrix, points, result), seed=seed)
    result._seal_parameters()
    return result


def arm_points(matrix, variant):
    return {arm["id"]: arm["points"] for arm in cm.arms(matrix, "titans_mac", variant=variant)}


def recipe(**options):
    values = dict(truncation=3, epochs=1, block_rows=2, selection=SELECTION, loss="pinball")
    values.update(options)
    if values.pop("fixed", False):
        values["selection"] = dict(values["selection"], stopping=FIXED_BUDGET)
    return ChronologicalRecipe(**values)


def posttrainer(streams, output, model, *, declared=None, **options):
    return ChronologicalTrainer(
        model,
        recipe(**options),
        train=streams["train"],
        validation=streams["validation"],
        output=output,
        optimizer_factory=RecordingOptimizer,
        posttraining=declared or dict(kind="technical_check", arm="fixture"),
        audit=True,
    )


def test_matrix_v3_declares_the_titans_arms_of_each_variant(matrix):
    # Por su coste, la variedad (#444) solo propone tres brazos en `mac_online`.
    expected = {
        "transformer_direct": ["head", "fusion", "head+fusion", "fusion_full_rank"],
        "mac_disabled": ["head", "fusion", "head+fusion", "fusion_full_rank"],
        "mac_frozen": [arm["id"] for arm in matrix["arms"]],
        "mac_online": [arm["id"] for arm in matrix["arms"]]
        + ["fusion_parallel_adapter", "bias", "persistent"],
    }
    assert {variant: list(arm_points(matrix, variant)) for variant in VARIANTS} == expected
    titans = matrix["architectures"]["chronological"]["titans_mac"]
    assert titans["frozen"] == ["mac.memory", "mac.persistent"]
    for paths in titans["targets"].values():
        assert not any(path.startswith(("mac.memory", "mac.persistent")) for path in paths)


@pytest.mark.parametrize("variant", VARIANTS)
def test_null_adapters_reproduce_the_parent_predictions_exactly(shared, matrix, variant):
    """Con U y las correcciones nulas, el brazo emite los mismos bits que el padre congelado.

    Congelar el padre ya cambia los últimos bits de transformer_direct en CPU: PyTorch
    descompone algunos productos de otra forma según `requires_grad`, también con SDPA Math.
    Por eso la igualdad exacta se exige frente al padre congelado y la del padre entrenable
    se acota con una tolerancia declarada.
    """
    _, streams = shared
    parent = predictor(streams, variant).eval()
    frozen = clone(parent).requires_grad_(False)
    reference = ChronologicalInference(frozen, recipe(), audit=True)
    expected = reference.evaluate(streams["validation"])
    trainable = ChronologicalInference(parent, recipe(), audit=True)
    trainable.evaluate(streams["validation"])
    for left, right in zip(trainable.audit, reference.audit, strict=True):
        assert left[:4] == right[:4]
        assert left[4] == pytest.approx(right[4], rel=1e-12, abs=1e-15)
    for arm, points in arm_points(matrix, variant).items():
        model = adapted(parent, matrix, points).eval()
        engine = ChronologicalInference(model, recipe(), audit=True)
        assert engine.evaluate(streams["validation"]) == expected, arm
        assert engine.audit == reference.audit, arm
        assert base_digest(model) == base_digest(parent)


def test_adapted_counts_match_the_declared_tensors(shared, matrix):
    _, streams = shared
    parent = predictor(streams, "mac_online")
    hidden = 32
    # head: 5 x 32 + 5. readout: dos matrices 32 x 32 de rango 4. fusion: 32 x (5 x 32 + 5).
    head, readout = 5 * hidden + 5, 2 * 4 * (hidden + hidden)
    fusion, fusion_full = 4 * (hidden + 5 * hidden + 5), hidden * (5 * hidden + 5)
    expected = dict(
        head=head,
        readout=readout,
        fusion=fusion,
        fusion_full_rank=fusion_full,
        **{"head+readout+fusion": head + readout + fusion},
    )
    for arm, count in expected.items():
        model = adapted(parent, matrix, arm_points(matrix, "mac_online")[arm])
        description = cm.describe(
            cm.titans_targets(matrix, arm_points(matrix, "mac_online")[arm]), model
        )
        assert trainable_parameters(model) == description["trainable_parameters"] == count, arm
        assert all(name.endswith(("delta", ".up", ".down")) for name in adapter_names(model)), arm


def test_gradients_reach_only_the_adapters_and_leave_memory_frozen(shared, matrix, tmp_path):
    _, streams = shared
    parent = predictor(streams, "mac_online")
    points = arm_points(matrix, "mac_online")["head+readout+fusion"]
    model = adapted(parent, matrix, points)
    before = base_digest(model)
    engine = posttrainer(streams, tmp_path / "run", model)
    assert list(engine.roles["adapters"]) == adapter_names(model)
    assert all(not names for role, names in engine.roles.items() if role != "adapters")
    frozen = engine.identity["frozen_parameters"]
    assert "mac.persistent" in frozen
    assert any(name.startswith("mac.memory.initial_weights.") for name in frozen)
    assert all(not name.startswith("mac.memory.") for name in adapter_names(model))
    report = engine.run()
    assert report["status"] == "completed"
    records = named_records(engine)
    assert records and all(set(record) == set(adapter_names(model)) for record in records)
    first = records[0]
    for name, value in first.items():
        assert value is not None and torch.isfinite(value).all(), name
        if name.endswith(".down"):
            # Con U nula el gradiente de V es exactamente cero en el primer paso.
            assert not value.abs().sum(), name
        else:
            assert value.abs().sum() > 0, name
    assert base_digest(model) == before
    assert model._parameter_id == engine.identity["initial_parameters_sha256"]


@pytest.mark.parametrize("variant", ["transformer_direct", "mac_online"])
def test_paired_arms_share_updates_labels_and_predictions(shared, matrix, tmp_path, variant):
    """Todos los brazos de un padre recorren los mismos tramos con las mismas etiquetas.

    La continuación completa no congela el padre, así que en transformer_direct sus
    predicciones solo coinciden con la tolerancia de la prueba de paridad.
    """
    _, streams = shared
    parent = predictor(streams, variant)
    runs, identities = {}, set()
    arms = {"full_continuation": None, **arm_points(matrix, variant)}
    for arm, points in arms.items():
        model = clone(parent) if points is None else adapted(parent, matrix, points)
        declared = dict(kind="technical_check", arm=arm)
        engine = posttrainer(streams, tmp_path / arm, model, declared=declared, fixed=True)
        identities.add(engine.run_id)
        runs[arm] = (engine, engine.run())
    assert len(identities) == len(arms)
    baseline, report = runs["full_continuation"]
    assert baseline.roles == parameter_roles(parent)
    assert baseline.identity["frozen_parameters"] == []

    def counts(result):
        keys = ("updates", "labels", "labels_in_loss", "segments")
        return [{k: e["train"][k] for k in keys} for e in result["history"] if e.get("train")]

    expected = [e for e in baseline.audit if e[0] != "prediction"]
    issued = [e for e in baseline.audit if e[0] == "prediction"]
    for arm, (engine, result) in runs.items():
        assert [e for e in engine.audit if e[0] != "prediction"] == expected, arm
        predictions = [e for e in engine.audit if e[0] == "prediction"]
        assert [e[:4] for e in predictions] == [e[:4] for e in issued], arm
        assert [e[4] for e in predictions] == pytest.approx(
            [e[4] for e in issued], rel=1e-12, abs=1e-15
        ), arm
        if variant == "mac_online":
            assert engine.audit == baseline.audit, arm
        assert counts(result) == counts(report) and counts(report), arm
        assert engine.optimizer.calls == baseline.optimizer.calls > 3, arm


@pytest.mark.parametrize("variant", ["transformer_direct", "mac_online"])
def test_adapter_gradients_follow_the_chain_rule_of_the_full_continuation(
    shared, matrix, tmp_path, variant
):
    """Con U nula, dL/dΔ = dL/dW, dL/dU = (α/r) dL/dW Vᵀ y dL/dV = 0 en cada paso.

    El registrador no cambia pesos, así que todos los pasos se evalúan en el mismo punto que
    los de la continuación completa. Sin recorte, el gradiente de cada corrección es la
    regla de la cadena aplicada al del tensor original.
    """
    _, streams = shared
    parent = predictor(streams, variant)
    points = arm_points(matrix, variant)
    points = points["head+readout+fusion" if "head+readout+fusion" in points else "head+fusion"]
    full = posttrainer(streams, tmp_path / "full", clone(parent), max_grad_norm=None)
    full.run()
    model = adapted(parent, matrix, points)
    arm = posttrainer(streams, tmp_path / "arm", model, max_grad_norm=None)
    arm.run()
    values = dict(model.named_parameters())
    reference, records = named_records(full), named_records(arm)
    assert len(records) == len(reference) > 3
    checked = set()
    for whole, adapted_record in zip(reference, records, strict=True):
        for name, gradient in adapted_record.items():
            module, rest = name.split(".parametrizations.")
            tensor = rest.split(".")[0]
            original = whole[f"{module}.{tensor}"]
            if name.endswith(".delta"):
                expected = original
            elif name.endswith(".up"):
                delta = model.get_submodule(module).parametrizations[tensor][0]
                expected = delta.scaling * original @ values[name[: -len("up")] + "down"].T
            else:
                expected = torch.zeros_like(gradient)
            torch.testing.assert_close(gradient, expected, rtol=1e-9, atol=1e-12)
            checked.add(f"{module}.{tensor}")
    targets = {f"{t.module}.{t.tensor}" for t in cm.titans_targets(matrix, points)}
    assert checked == targets


def test_resume_reproduces_the_continuous_adapter_run(shared, matrix, tmp_path):
    _, streams = shared
    parent = predictor(streams, "mac_online")
    points = arm_points(matrix, "mac_online")["head+readout"]
    options = dict(epochs=2, fixed=True, checkpoint_updates=2)
    continuous = posttrainer(
        streams, tmp_path / "continuous", adapted(parent, matrix, points), **options
    )
    expected = continuous.run()
    stop = StopAtStep(15)
    first = posttrainer(streams, tmp_path / "paused", adapted(parent, matrix, points), **options)
    stop.trainer = first
    assert first.run(stop=stop)["status"] == "paused"
    state = load_training_state(tmp_path / "paused/checkpoints", expected_identity=first.identity)
    assert state["run"]["fast"] and state["optimizer"]["calls"] == state["global_step"]
    assert set(adapter_names(first.predictor)) <= set(state["model"])
    second = posttrainer(streams, tmp_path / "paused", adapted(parent, matrix, points), **options)
    resumed = second.run(resume=True)
    assert resumed["status"] == "completed"
    assert first.audit + second.audit == continuous.audit
    joined = named_records(first) + named_records(second)
    assert len(joined) == len(named_records(continuous))
    for left, right in zip(joined, named_records(continuous), strict=True):
        for key, value in left.items():
            torch.testing.assert_close(value, right[key], rtol=0, atol=0)
    assert resumed["history"] == expected["history"]
    # Otra semilla de V es otro estado y su identidad no admite reanudar este ajuste.
    other = posttrainer(
        streams, tmp_path / "paused", adapted(parent, matrix, points, seed=6), **options
    )
    with pytest.raises(ValueError):
        other.run(resume=True)


def test_adapters_need_a_declared_posttraining_without_pairing(shared, matrix, tmp_path):
    _, streams = shared
    parent = predictor(streams, "mac_online")
    model = adapted(parent, matrix, arm_points(matrix, "mac_online")["head"])
    common = dict(
        train=streams["train"],
        validation=streams["validation"],
        optimizer_factory=RecordingOptimizer,
    )
    with pytest.raises(ValueError, match="postentrenamiento"):
        ChronologicalTrainer(model, recipe(), output=tmp_path / "a", **common)
    for declared in ({}, "head", ["head"]):
        with pytest.raises(ValueError, match="postentrenamiento"):
            ChronologicalTrainer(
                model, recipe(), output=tmp_path / "b", posttraining=declared, **common
            )
    with pytest.raises(ValueError, match="postentrenamiento"):
        ChronologicalTrainer(
            model,
            recipe(),
            output=tmp_path / "c",
            posttraining=dict(kind="technical_check"),
            pairing=dict(
                target_after=model._parameter_id,
                target_variant="mac_online",
                runtime_state_transferred=False,
            ),
            **common,
        )

    def everything(groups):
        everything = dict(model.named_parameters())
        return RecordingOptimizer([dict(params=list(everything.values()), role="all")])

    with pytest.raises(ValueError, match="ajuste externo"):
        ChronologicalTrainer(
            model,
            recipe(),
            output=tmp_path / "d",
            posttraining=dict(kind="technical_check"),
            **dict(common, optimizer_factory=everything),
        )
    frozen = clone(parent).requires_grad_(False)
    with pytest.raises(ValueError, match="ajuste externo"):
        ChronologicalTrainer(
            frozen, recipe(), output=tmp_path / "e", posttraining=dict(kind="x"), **common
        )


def test_roles_and_identity_keep_their_form_without_adapters(shared, tmp_path):
    _, streams = shared
    parent = predictor(streams, "mac_online")
    assert set(parameter_roles(parent)) == {"shared", "persistent_memory", "initial_fast_weights"}
    engine = ChronologicalTrainer(
        parent,
        recipe(),
        train=streams["train"],
        validation=streams["validation"],
        output=tmp_path / "run",
        optimizer_factory=RecordingOptimizer,
    )
    assert "posttraining" not in engine.identity
    assert "frozen_parameters" not in engine.identity
    json.dumps(engine.identity)
