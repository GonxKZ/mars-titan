"""Retirada de fuentes y sus rutas indirectas sin cambiar la población admitida."""

import copy
import importlib

import numpy as np
import pytest

from mars_titan.data.storage import sha256
from mars_titan.models.baselines.inputs import MODALITIES


def api():
    try:
        return importlib.import_module("mars_titan.data.information_views")
    except ModuleNotFoundError:
        pytest.fail("Falta el contrato ejecutable de vistas de información")


def specification():
    shapes = dict(prices=[2, 5], news=[2], charts=[1], fundamentals=[6], macro=[6])
    variables = []
    for block, width in (("prices", 10), ("news", 2), ("charts", 1)):
        variables.append(
            dict(
                name=block,
                source=block,
                block=block,
                value=list(range(width)),
                mask=[],
                age=[],
                dependencies=["prices"] if block == "charts" else [],
                transformation="control",
                history="prefijo disponible",
            )
        )
    for block in ("fundamentals", "macro"):
        for i, name in enumerate(("raw", "derived")):
            variables.append(
                dict(
                    name=f"{block}/{name}",
                    source=block,
                    block=block,
                    value=[i],
                    mask=[i + 2],
                    age=[i + 4],
                    dependencies=[f"{block}/raw"] if i else [],
                    transformation="numeric_context",
                    history="publicación efectiva",
                )
            )
    return dict(
        schema_version=1,
        kind="information_view",
        cohort_sha256="a" * 64,
        representation_sha256="b" * 64,
        shapes=shapes,
        variables=variables,
        allowed_sources=list(MODALITIES),
        allowed_variables=[v["name"] for v in variables],
    )


def batch(spec=None):
    spec = spec or specification()
    return dict(
        inputs={
            k: np.arange(1, 1 + 2 * int(np.prod(v)), dtype=np.float32).reshape(2, *v)
            for k, v in spec["shapes"].items()
        },
        target=np.array([0.1, -0.2]),
        sample_ids=["US/A/1", "US/B/1"],
        prediction_at=np.array([1, 1]),
        target_available_at=np.array([2, 2]),
        market=["US", "US"],
        weight=np.ones(2),
        confirmed_cursor=dict(consumed=2),
    )


def test_full_view_preserves_inputs_and_masks_are_grouped_with_their_value():
    view = api().InformationView(specification())
    raw = batch()
    full = view.apply(raw)
    for name in MODALITIES:
        np.testing.assert_array_equal(full["inputs"][name], raw["inputs"][name])
        assert not np.shares_memory(full["inputs"][name], raw["inputs"][name])
    reduced = view.without(variables=["macro/raw"])
    result = reduced.apply(raw)
    assert not result["inputs"]["macro"].any()
    assert result["sample_ids"] == raw["sample_ids"]
    np.testing.assert_array_equal(result["target"], raw["target"])
    assert raw["inputs"]["macro"].any()


def test_excluding_prices_removes_charts_and_all_transitive_dependencies():
    view = api().InformationView(specification()).without(sources=["prices"])
    raw = batch()
    result = view.apply(raw)
    assert not result["inputs"]["prices"].any()
    assert not result["inputs"]["charts"].any()
    raw["inputs"]["prices"] *= 99
    raw["inputs"]["charts"] += 900
    perturbed = view.apply(raw)
    for name in MODALITIES:
        np.testing.assert_array_equal(result["inputs"][name], perturbed["inputs"][name])


def test_excluded_mask_and_age_cannot_reenter_a_parent_memory_hmm_or_calibrator():
    view = api().InformationView(specification()).without(variables=["macro/raw"])
    raw = batch()
    for role in ("parent", "memory", "hmm", "calibration", "router", "normalization"):
        contract = view.artifact(role, "c" * 64)
        consumer = api().ViewConsumer(view, contract, lambda inputs: inputs["macro"].sum(axis=1))
        np.testing.assert_array_equal(consumer(view.apply(raw)), [0, 0])
        changed = copy.deepcopy(raw)
        changed["inputs"]["macro"] += 12345
        np.testing.assert_array_equal(consumer(view.apply(changed)), [0, 0])


@pytest.mark.parametrize("role", ["parent", "memory", "hmm", "calibration", "normalization"])
def test_consumer_rejects_a_state_fitted_on_another_view_even_with_same_dimensions(role):
    full = api().InformationView(specification())
    reduced = full.without(sources=["news"])
    with pytest.raises(ValueError, match="vista"):
        api().ViewConsumer(reduced, full.artifact(role, "c" * 64), lambda inputs: inputs)
    consumer = api().ViewConsumer(reduced, reduced.artifact(role, "c" * 64), lambda inputs: inputs)
    with pytest.raises(ValueError, match="vista"):
        consumer(full.apply(batch()))
    with pytest.raises(ValueError, match="vista"):
        consumer(batch())


def test_parent_feature_is_recalculated_after_masking_and_cannot_be_imported_as_raw_input():
    view = api().InformationView(specification()).without(sources=["news"])
    consumer = api().ViewConsumer(
        view, view.artifact("parent", "c" * 64), lambda inputs: inputs["news"].sum(axis=1)
    )
    result = api().attach_parent(view.apply(batch()), consumer)
    np.testing.assert_array_equal(result["parent"], [0, 0])
    np.testing.assert_array_equal(result["features"][:, -1], [0, 0])
    assert result["features"].shape == (2, 26)
    with pytest.raises(ValueError, match="indirect|ruta"):
        view.apply(batch() | {"parent": np.ones(2)})
    with pytest.raises(ValueError, match="indirect|ruta"):
        view.apply(batch() | {"features": np.ones((2, 20))})


def test_artifact_dependencies_require_the_same_view_and_cannot_disappear():
    full = api().InformationView(specification())
    view = full.without(sources=["news"])
    normalizer = view.artifact("normalization", "1" * 64)
    parent = view.artifact("parent", "2" * 64, dependencies=[normalizer])
    with pytest.raises(ValueError, match="dependencia"):
        api().ViewConsumer(view, parent, lambda inputs: inputs)
    api().ViewConsumer(view, parent, lambda inputs: inputs, dependencies=[normalizer])
    wrong = full.artifact("normalization", "1" * 64)
    with pytest.raises(ValueError, match="vista"):
        api().ViewConsumer(view, parent, lambda inputs: inputs, dependencies=[wrong])


@pytest.mark.parametrize("damage", ["uncovered", "overlap", "cycle", "unknown", "indirect"])
def test_invalid_layouts_and_unmasked_dependencies_are_rejected(damage):
    spec = specification()
    if damage == "uncovered":
        spec["variables"][-1]["age"] = []
    elif damage == "overlap":
        spec["variables"][-1]["age"] = [4]
    elif damage == "cycle":
        spec["variables"][-2]["dependencies"] = ["macro/derived"]
    elif damage == "unknown":
        spec["variables"][-1]["dependencies"] = ["unknown"]
    else:
        spec["allowed_sources"].remove("prices")
    with pytest.raises(ValueError):
        api().InformationView(spec)


def test_view_identity_binds_representation_layout_history_and_cohort(tmp_path):
    view = api().InformationView(specification())
    view.save(tmp_path / "view.json")
    loaded = api().InformationView.load(tmp_path / "view.json")
    assert view.identity == loaded.identity
    spec = view.manifest
    spec["variables"][0]["history"] = "otra ventana"
    assert view.identity != api().InformationView(spec).identity
    assert view.manifest == specification()
    spec = specification() | {"cohort_sha256": "d" * 64}
    assert view.identity != api().InformationView(spec).identity


def test_corrupt_values_and_excess_memory_are_rejected_without_relaxing_admission():
    view = api().InformationView(specification()).without(sources=["news"])
    raw = batch()
    raw["inputs"]["news"][0, 0] = np.nan
    with pytest.raises(ValueError, match="finit"):
        view.apply(raw)
    with pytest.raises(ValueError, match="presupuesto"):
        view.apply(batch(), max_bytes=4)
    with pytest.raises(ValueError, match="dimensi"):
        view.apply(
            batch() | {"inputs": batch()["inputs"] | {"news": np.ones((2, 1), dtype=np.float32)}}
        )


def test_consumers_receive_only_current_inputs_and_recheck_excluded_values():
    view = api().InformationView(specification()).without(sources=["news"])
    result = view.apply(batch())
    result["inputs"]["news"] += 50
    consumer = api().ViewConsumer(
        view, view.artifact("parent", "c" * 64), lambda inputs: (set(inputs), inputs["news"].sum())
    )
    names, value = consumer(result)
    assert names == set(MODALITIES)
    assert value == 0


def test_consumer_verifies_the_bound_artifact_before_delivering_inputs(tmp_path):
    payload = tmp_path / "state.bin"
    payload.write_bytes(b"estado de control")
    view = api().InformationView(specification())
    record = view.artifact("memory", sha256(payload))
    consumer = api().ViewConsumer(
        view, record, lambda inputs: inputs["news"].sum(), artifact_path=payload
    )
    assert consumer(view.apply(batch())) == 10
    payload.write_bytes(b"estado corrupto")
    with pytest.raises(ValueError, match="artefacto|huella"):
        consumer(view.apply(batch()))


def test_declared_fit_view_cannot_be_replaced_with_the_runtime_view():
    full = api().InformationView(specification())
    reduced = full.without(sources=["prices"])
    record = reduced.artifact("parent", "c" * 64)
    record["fit_view_sha256"] = full.identity
    with pytest.raises(ValueError, match="vista"):
        api().ViewConsumer(reduced, record, lambda inputs: inputs)


@pytest.mark.parametrize("field", ["prediction_at", "target_available_at", "input_available_at"])
def test_view_rejects_misaligned_timestamps(field):
    view = api().InformationView(specification())
    with pytest.raises(ValueError, match="alinead"):
        view.apply(batch() | {field: np.array([1])})
