"""Brazos de la variedad de #444 en la matriz de versión 3, comprobados hasta el paso.

Las pruebas que recorren un ajuste usan optimizadores que registran gradientes sin cambiar
pesos. Así cada brazo debe aplicar las mismas actualizaciones que los de #364 y su época
cero debe coincidir bit a bit con el padre congelado. Las trazas de #448 se activan en
algunas ejecuciones para demostrar que no cambian el cálculo.
"""

import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow.parquet as pq
import pytest
import torch

from mars_titan.data.input_policy import HISTORICAL_MASKED
from mars_titan.models.predictive_adaptation import (
    MODULE_ROOT,
    adapter_names,
    base_digest,
    trainable_parameters,
)
from mars_titan.models.quantile_head import QUANTILE_COLUMNS, QUANTILE_HEAD
from mars_titan.posttraining import adapter_matrix, adapter_variety
from mars_titan.posttraining import chronological_matrix as cm
from mars_titan.posttraining.adapter_traces import update_statistics
from mars_titan.posttraining.heldout import _adjustment, evaluate_partition
from mars_titan.posttraining.run import run_case
from mars_titan.training.checkpoints import StopRequest
from mars_titan.training.corpus_inputs import CorpusDataset
from mars_titan.training.financial_run import ChronologicalInference, parameter_roles
from tests.posttraining.masked_fixture import masked_ordered
from tests.posttraining.test_quantile_adaptation import setup
from tests.posttraining.test_titans_adapters import adapted, posttrainer, predictor
from tests.training.test_financial_run import corpus, named_records
from tests.training.test_mars_titan_run import native as native

ROOT = Path(__file__).resolve().parents[2]
MATRIX_PATH = ROOT / "configs/posttraining/adapter-matrix-v3.json"
CAMPAIGN = {
    "fusion": ["fusion_dora", "fusion_ia3", "fusion_parallel_adapter", "bias"],
    "readout": ["fusion_dora", "readout_dora", "fusion_ia3", "readout_ia3"],
}


@pytest.fixture(scope="module")
def matrix():
    return adapter_matrix.read_matrix(MATRIX_PATH)


@pytest.fixture(autouse=True)
def explicit_fastpath():
    previous = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    yield
    torch.backends.mha.set_fastpath_enabled(previous)


def ids(rows):
    return [row["id"] for row in rows]


def test_the_matrix_declares_each_variety_arm_where_its_point_exists(matrix):
    document, _ = matrix
    base = ["head", "fusion", "head+fusion", "fusion_full_rank"]
    full = [arm["id"] for arm in document["arms"]]
    recurrent = base + CAMPAIGN["fusion"]
    for family in ("rnn", "lstm", "gru", "dlinear"):
        assert ids(adapter_matrix.arms(document, family)) == recurrent
        assert ids(adapter_matrix.arms(document, family, reserve=True)) == [
            *base,
            "fusion_dora",
            "fusion_ia3",
            "fusion_parallel_adapter",
            "fusion_serial_adapter",
            "bias",
        ]
    transformer = [*full, *CAMPAIGN["readout"], "fusion_parallel_adapter", "bias"]
    assert ids(adapter_matrix.arms(document, "transformer")) == transformer
    reserve = ids(adapter_matrix.arms(document, "transformer", reserve=True))
    assert reserve[-3:] == ["fusion_serial_adapter", "bias", "norm"]
    titans = {
        variant: ids(cm.arms(document, "titans_mac", variant=variant))
        for variant in cm.TITANS_VARIANTS
    }
    assert titans["transformer_direct"] == titans["mac_disabled"] == recurrent
    reading = [*full, *CAMPAIGN["readout"], "fusion_parallel_adapter", "bias", "persistent"]
    assert titans["mac_frozen"] == titans["mac_online"] == reading
    for family in ("mars_titan", "cm_v1"):
        assert ids(cm.arms(document, family)) == [
            "core",
            "episodic_readout",
            "core+episodic_readout",
            "core_persistent",
        ]
        assert ids(cm.arms(document, family, bank=False)) == ["core", "core_persistent"]
    assert ids(cm.arms(document, "episodic_gru")) == ["head"]


def _mutated(document, change):
    result = copy.deepcopy(document)
    change(result["variety"])
    return result


def _arm(section, name):
    return next(arm for arm in section["arms"] if arm["id"] == name)


MUTATIONS = {
    "extra key": lambda v: v.update(extra=1),
    "missing point": lambda v: v["points"].pop("norm"),
    "selective form": lambda v: v["points"]["bias"].update(form="low_rank"),
    "unknown state": lambda v: v["points"]["bias"].update(invalidates=["weights"]),
    "unknown form": lambda v: _arm(v, "fusion_dora")["overrides"]["fusion"].update(form="vera"),
    "readout bottleneck": lambda v: v["arms"].append(
        dict(
            id="readout_parallel_adapter",
            points=["readout"],
            overrides=dict(
                readout=dict(
                    form="parallel_adapter", rank=4, alpha=4.0, invalidates=["attention_outputs"]
                )
            ),
            campaign=False,
        )
    ),
    "rank too large": lambda v: _arm(v, "fusion_dora")["overrides"]["fusion"].update(rank=65),
    "unnamed form": lambda v: _arm(v, "fusion_dora").update(id="fusion_magnitude"),
    "base arm id": lambda v: _arm(v, "bias").update(id="head"),
    "repeated arm": lambda v: v["arms"].append(copy.deepcopy(_arm(v, "bias"))),
    "two points": lambda v: _arm(v, "bias").update(points=["bias", "norm"]),
    "head point": lambda v: _arm(v, "bias").update(points=["head"], id="head_bias"),
    "selective override": lambda v: _arm(v, "bias").update(
        overrides=dict(bias=dict(form="residual", invalidates=["cached_parent_predictions"]))
    ),
    "proposal not boolean": lambda v: _arm(v, "bias").update(campaign="yes"),
    "missing reason": lambda v: v["inapplicable"].pop("persistent"),
    "short reason": lambda v: v["inapplicable"].update(norm="no aplica"),
    "unused reason": lambda v: (
        v["arms"].remove(_arm(v, "norm")),
        v["inapplicable"].update(norm=v["inapplicable"]["norm"]),
    ),
    "reader outside mac_online": lambda v: v["readers"].append("missing"),
    "repeated reader": lambda v: v["readers"].append("persistent"),
    "open variety": lambda v: v["arms"].extend(
        copy.deepcopy(_arm(v, "bias")) | dict(id=f"bias{i}") for i in range(12)
    ),
}


@pytest.mark.parametrize("name", MUTATIONS)
def test_invalid_variety_sections_are_rejected(matrix, name):
    document, _ = matrix
    with pytest.raises(ValueError):
        adapter_matrix.validate_matrix(_mutated(document, MUTATIONS[name]))


def test_only_version_three_admits_the_variety(matrix):
    document, _ = matrix
    second, _ = adapter_matrix.read_matrix(ROOT / "configs/posttraining/adapter-matrix-v2.json")
    with pytest.raises(ValueError):
        adapter_matrix.validate_matrix(second | dict(variety=document["variety"]))
    # Sin la sección la versión 3 sigue siendo válida y no declara brazos nuevos.
    plain = {key: value for key, value in document.items() if key != "variety"}
    adapter_matrix.validate_matrix(plain)
    assert ids(adapter_matrix.arms(plain, "gru")) == [
        "head",
        "fusion",
        "head+fusion",
        "fusion_full_rank",
    ]


def test_case_adapters_reject_mixed_or_unknown_points(matrix):
    document, digest = matrix
    case = next(
        item["case"]
        for item in adapter_matrix.cases(document, digest, "transformer", head=QUANTILE_HEAD)
        if item["id"] == "seed-42/bias"
    )
    adapter_matrix.validate_adapter(case["adapter"])
    mixed = copy.deepcopy(case["adapter"])
    mixed["points"] = dict(head=document["points"]["head"], **mixed["points"])
    with pytest.raises(ValueError):
        adapter_matrix.validate_adapter(mixed)
    unknown = copy.deepcopy(case["adapter"])
    unknown["points"] = dict(prompt=unknown["points"]["bias"])
    with pytest.raises(ValueError):
        adapter_matrix.validate_adapter(unknown)


@pytest.fixture(scope="module")
def edition(tmp_path_factory):
    root = tmp_path_factory.mktemp("variety-edition")
    view, ordered, report = masked_ordered(root / "data")
    return SimpleNamespace(view=view, ordered=ordered, report=report, root=root)


def _variety_cases(document, digest, kind):
    variety = {arm["id"] for arm in document["variety"]["arms"]}
    rows = adapter_matrix.cases(document, digest, kind, head=QUANTILE_HEAD, reserve=True)
    return {
        row["id"].split("/", 1)[1]: row["case"]
        for row in rows
        if row["case"]["seed"] == 42 and row["id"].split("/", 1)[1] in variety | {"fusion"}
    }


@pytest.mark.parametrize("kind", ["transformer", "gru"])
def test_variety_arms_reach_each_step_with_the_parent_at_epoch_zero(
    tmp_path, recorder, edition, matrix, kind
):
    """Cada brazo aplica las mismas actualizaciones que la fusión LoRA y empieza en el padre.

    Con el registrador no cambia ningún peso, así que la validación elegida es la época cero
    y sus predicciones deben ser las del padre bit a bit, también las reservadas.
    """
    document, digest = matrix
    parent, data, grid, scale, handles = setup(edition, tmp_path, kind)
    before = base_digest(parent.model)
    cases = _variety_cases(document, digest, kind)
    expected = {"fusion", "fusion_dora", "fusion_ia3", "fusion_parallel_adapter"}
    expected |= {"fusion_serial_adapter", "bias"}
    if kind == "transformer":
        expected |= {"readout_dora", "readout_ia3", "norm"}
    assert set(cases) == expected
    updates = document["budget"]["epochs"] * data.budget("real", 2)["updates"]
    held = {}
    for arm, case in cases.items():
        output = tmp_path / arm
        count = len(recorder.optimizers)
        report = run_case(
            data,
            output,
            case,
            grid,
            scale,
            parent=parent,
            batch_size=2,
            device="cpu",
            diagnostic=True,
        )
        assert report["status"] == "completed", arm
        assert report["global_step"] == report["total_steps"] == updates, arm
        optimizer = recorder.optimizers[count]
        assert len(optimizer.calls) == updates, arm
        assert report["trainable_parameters"] == sum(p.numel() for p in optimizer.parameters)
        assert (
            report["identity"]["adapter"]["trainable_parameters"] == report["trainable_parameters"]
        )
        assert report["selection"]["best_epoch"] == 0, arm
        table = pq.read_table(output / "validation-predictions.parquet").to_pydict()
        np.testing.assert_array_equal(table["prediction"], table["parent"], err_msg=arm)
        model, action_grid, neural = _adjustment(output / "run.json", report, parent, "cpu")
        # La evaluación reservada reconstruye el mismo brazo, ya congelado.
        rebuilt = sum(model.get_parameter(name).numel() for name in adapter_names(model))
        assert rebuilt == report["trainable_parameters"] and not trainable_parameters(model), arm
        path = tmp_path / f"{arm}-evaluation.parquet"
        evaluate_partition(
            CorpusDataset(edition.view, input_policy=HISTORICAL_MASKED),
            parent,
            "evaluation",
            path,
            stop=StopRequest(),
            device="cpu",
            model=model,
            grid=action_grid,
            neural=neural,
        )
        rows = pq.read_table(path).to_pydict()
        np.testing.assert_array_equal(rows["prediction"], rows["parent"], err_msg=arm)
        held[arm] = np.column_stack([rows[name] for name in QUANTILE_COLUMNS])
    # Todos los brazos reservan los mismos cuantiles: los del padre sin actualizaciones.
    first = next(iter(held.values()))
    assert all(np.array_equal(value, first) for value in held.values())
    assert base_digest(parent.model) == before
    for handle in handles:
        handle.close()


@pytest.fixture(scope="module")
def shared(tmp_path_factory):
    return corpus(tmp_path_factory.mktemp("variety-titans"))


def _titans_points(document, variant):
    return {
        arm["id"]: arm["points"]
        for arm in cm.arms(document, "titans_mac", variant=variant, reserve=True)
    }


def test_titans_variety_targets_avoid_the_neural_memory(shared, matrix):
    document, _ = matrix
    _, streams = shared
    parent = predictor(streams, "mac_online")
    for arm, points in _titans_points(document, "mac_online").items():
        targets = cm.titans_targets(document, points, parent)
        modules = {target.module for target in targets}
        assert not any(module.startswith("mac.memory") for module in modules), arm
        if arm == "persistent":
            assert [(t.module, t.tensor) for t in targets] == [("mac", "persistent")]
        else:
            assert all(not (t.module == "mac" and t.tensor == "persistent") for t in targets)
    model = adapted(parent, document, _titans_points(document, "mac_online")["bias"])
    assert "mac.persistent" not in {name.removesuffix(".original") for name in adapter_names(model)}
    norms = cm.titans_targets(document, {"norm": document["variety"]["points"]["norm"]}, parent)
    assert {t.module for t in norms} == {
        "price_encoder.blocks.0.norm1",
        "price_encoder.blocks.0.norm2",
        "price_encoder.norm",
    }


@pytest.mark.parametrize("arm", ["persistent", "fusion_parallel_adapter", "readout_ia3", "bias"])
def test_titans_variety_arms_reach_each_step_with_the_trace(shared, matrix, tmp_path, arm):
    """Mismas actualizaciones, gradiente solo en las correcciones y trazas sin efecto."""
    document, _ = matrix
    _, streams = shared
    parent = predictor(streams, "mac_online")
    points = _titans_points(document, "mac_online")[arm]
    engines, events = {}, []

    def trace(event, modules):
        events.append((event, update_statistics(modules["predictor"])))

    for name in ("plain", "traced"):
        model = adapted(parent, document, points)
        engine = posttrainer(streams, tmp_path / name, model, fixed=True)
        if name == "traced":
            engine.trace = trace
        engine.run()
        engines[name] = engine
    plain, traced = engines["plain"], engines["traced"]
    assert all(not names for role, names in plain.roles.items() if role != "adapters")
    assert plain.roles["adapters"] == adapter_names(plain.predictor)
    assert plain.audit == traced.audit
    assert plain.optimizer.calls == traced.optimizer.calls > 0
    for left, right in zip(named_records(plain), named_records(traced), strict=True):
        assert left.keys() == right.keys() == set(plain.roles["adapters"])
        assert all(torch.equal(left[key], right[key]) for key in left)
    assert [event["epoch"] for event, _ in events] == list(range(len(events)))
    # La actualización efectiva es nula sin pasos. En un adaptador por módulo la bajada es
    # aleatoria desde el inicio y solo la subida y su sesgo empiezan en cero.
    assert all(
        row["frobenius"] == 0.0
        for _, rows in events
        for row in rows
        if not row["module"].startswith(MODULE_ROOT) or row["tensor"].startswith("up")
    )
    if arm == "persistent":
        names = adapter_names(plain.predictor)
        assert names == ["mac.parametrizations.persistent.0.delta"]
    if arm == "fusion_parallel_adapter":
        assert all(name.startswith(MODULE_ROOT + ".") for name in plain.roles["adapters"])


@pytest.mark.parametrize("arm", ["fusion_parallel_adapter", "readout_dora", "bias", "persistent"])
def test_titans_variety_arms_reload_their_state_after_a_rebuild(shared, matrix, arm):
    """Al reanudar, el brazo se reconstruye desde la matriz y recupera sus correcciones.

    La reconstrucción usa otra semilla a propósito: el estado guardado debe sustituir por
    completo la inicialización, también la proyección aleatoria de los adaptadores por módulo.
    """
    document, _ = matrix
    _, streams = shared
    parent = predictor(streams, "mac_online")
    points = _titans_points(document, "mac_online")[arm]
    model = adapted(parent, document, points)
    description = cm.describe(cm.titans_targets(document, points, parent), model)
    assert description["trainable_parameters"] == trainable_parameters(model)
    with torch.no_grad():
        for name in adapter_names(model):
            model.get_parameter(name).uniform_(-0.05, 0.05)
    model._seal_parameters()
    rebuilt = adapted(parent, document, points, seed=99)
    assert adapter_names(rebuilt) == adapter_names(model)
    rebuilt.load_state_dict(copy.deepcopy(model.state_dict()))
    rebuilt._seal_parameters()
    engine = ChronologicalInference(model.eval(), posttrainer_recipe(), audit=True)
    twin = ChronologicalInference(rebuilt.eval(), posttrainer_recipe(), audit=True)
    assert engine.evaluate(streams["validation"]) == twin.evaluate(streams["validation"])
    assert engine.audit == twin.audit


def posttrainer_recipe():
    from tests.posttraining.test_titans_adapters import recipe

    return recipe()


def test_module_adapters_join_the_adapter_role(shared, matrix):
    document, _ = matrix
    _, streams = shared
    parent = predictor(streams, "mac_online")
    model = adapted(
        parent, document, _titans_points(document, "mac_online")["fusion_parallel_adapter"]
    )
    roles = parameter_roles(model)
    assert roles["adapters"] == [
        f"{MODULE_ROOT}.fusion.{name}" for name in ("down", "down_bias", "up", "up_bias")
    ]
    assert not any(name.startswith(MODULE_ROOT) for name in roles["shared"])


def test_variety_is_inapplicable_where_its_point_does_not_exist(matrix):
    document, _ = matrix
    for family in ("rnn", "lstm", "gru", "dlinear"):
        reserve = ids(adapter_matrix.arms(document, family, reserve=True))
        assert not {"norm", "persistent", "readout_dora", "readout_ia3"} & set(reserve)
    for variant in ("transformer_direct", "mac_disabled"):
        assert "persistent" not in ids(
            cm.arms(document, "titans_mac", variant=variant, reserve=True)
        )
    assert set(document["variety"]["inapplicable"]) == set(adapter_variety.PARTIAL)


def test_titans_subsets_skip_the_memory_gate_biases_of_the_recipe(shared, matrix):
    """Con las puertas con sesgo de la receta de la campaña, BitFit no toca la memoria.

    El padre de las demás pruebas no tiene esos sesgos, así que sin esta comprobación la
    exclusión de `mac.memory` no se ejercitaría.
    """
    from mars_titan.models.titans.config import GateBias
    from mars_titan.models.titans.financial import FinancialConfig, FinancialPredictor

    document, _ = matrix
    _, streams = shared
    config = FinancialConfig(
        streams["train"].specification(),
        variant="mac_online",
        hidden_size=32,
        seed=42,
        head=QUANTILE_HEAD,
        gate_bias=GateBias(),
        memory_residual_layer_norm=True,
    )
    model = FinancialPredictor(config, dtype=torch.float64)
    names = [name for name, _ in model.named_parameters()]
    assert any(name.startswith("mac.memory.") and name.endswith(".bias") for name in names)
    points = document["variety"]["points"]
    for name in ("bias", "norm"):
        targets = cm.titans_targets(document, {name: points[name]}, model)
        assert targets and not any(
            t.module.startswith(("mac.memory", "mac.persistent")) for t in targets
        )


def test_a_titans_case_must_be_an_arm_of_its_variant(matrix):
    """La ventana rechaza un brazo que la variante del padre no tiene o con otros puntos."""
    document, digest = matrix
    persistent = next(
        item["case"]["adapter"]
        for item in cm.cases(document, digest, "titans_mac", variant="mac_online")
        if item["id"].endswith("/persistent")
    )
    cm.variant_arm(document, "mac_frozen", persistent)
    for variant in ("transformer_direct", "mac_disabled"):
        with pytest.raises(ValueError, match="no pertenece a los puntos"):
            cm.variant_arm(document, variant, persistent)
    # Los brazos de reserva también son brazos de la variante.
    serial = next(
        item["case"]["adapter"]
        for item in cm.cases(document, digest, "titans_mac", variant="mac_disabled", reserve=True)
        if item["id"].endswith("/fusion_serial_adapter")
    )
    cm.variant_arm(document, "mac_disabled", serial)
    changed = copy.deepcopy(serial)
    changed["points"]["fusion"]["rank"] = 8
    with pytest.raises(ValueError, match="no pertenece a los puntos"):
        cm.variant_arm(document, "mac_disabled", changed)


def test_the_reader_core_learns_its_persistent_prompt_with_the_same_events(
    shared, tmp_path, native
):
    """MARS-TITAN con el prefijo persistente del núcleo como único destino.

    Recorre los mismos eventos, bancos y etiquetas que la continuación del lector, emite
    las mismas predicciones y solo envía gradiente a la corrección de la memoria persistente.
    """
    from tests.posttraining.test_readout_adapters import build, entries, named, without_predictions

    _, streams = shared
    full = build(streams, tmp_path / "full", native, "full_continuation")
    arm = build(streams, tmp_path / "arm", native, "core_persistent")
    core, readout = base_digest(arm.predictor), base_digest(arm.readout)
    reports = full.run(), arm.run()
    assert [report["status"] for report in reports] == ["completed", "completed"]
    assert without_predictions(arm.audit) == without_predictions(full.audit)
    assert entries(arm.audit, "prediction") == entries(full.audit, "prediction")
    assert arm.optimizer.calls == full.optimizer.calls > 0
    expected = {"core.mac.parametrizations.persistent.0.delta"}
    assert {name for names in arm.roles.values() for name in names} == expected
    records = named(arm)
    assert records and all(set(record) == expected for record in records)
    assert any(record[name].abs().sum() > 0 for record in records for name in expected)
    assert base_digest(arm.predictor) == core and base_digest(arm.readout) == readout
