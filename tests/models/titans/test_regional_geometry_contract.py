"""Geometría declarada de la fusión regional en cargas y conversiones válidas."""

import copy

import pytest
import torch
from test_financial_adapter import api, raw_batch
from test_frozen_financial import frozen_backend as frozen_backend
from test_market_memory import pt4
from test_regional_fusion import graph

from mars_titan.models.titans.regional_fusion import RegionalFusionConfig

DEVICES = ("cpu", pytest.param("cuda:0", marks=pytest.mark.cuda))
DTYPES = (torch.float32, torch.float64)


def predictor(device, dtype, *, regional=True):
    config = RegionalFusionConfig(topology="ring") if regional else None
    return (
        pt4(None, regional_fusion=config)
        .to(device=device, dtype=dtype)
        .eval()
        .requires_grad_(False)
    )


def module_on(device, dtype):
    return (
        graph(hidden=4, topology="ring").to(device=device, dtype=dtype).eval().requires_grad_(False)
    )


def assert_same(actual, expected):
    if isinstance(expected, torch.Tensor):
        assert actual.dtype == expected.dtype and actual.device == expected.device
        assert actual.shape == expected.shape and actual.layout == expected.layout
        if expected.layout == torch.sparse_coo:
            assert actual.is_coalesced() == expected.is_coalesced()
            assert torch.equal(actual._indices(), expected._indices())
            assert torch.equal(actual._values(), expected._values())
        else:
            assert torch.equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for name, value in expected.items():
            assert_same(actual[name], value)
    elif isinstance(expected, (tuple, list)):
        assert type(actual) is type(expected) and len(actual) == len(expected)
        for left, right in zip(actual, expected, strict=True):
            assert_same(left, right)
    else:
        assert type(actual) is type(expected) and actual == expected


def corrupt_edges(module, fault):
    if fault == "values":
        with torch.no_grad():
            module.edges[0, 2] = True
    elif fault == "dtype":
        module.edges = module.edges.to(torch.float32)
    elif fault == "shape":
        module.edges = module.edges[:4, :4].clone()
    else:
        raise AssertionError("El fixture no declara ese fallo de geometría")


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("fault", ["values", "dtype", "shape"])
def test_matching_corrupted_module_and_payload_cannot_redefine_declared_edges(device, dtype, fault):
    module = module_on(device, dtype)
    corrupt_edges(module, fault)
    before = copy.deepcopy(module.state_dict())
    payload = copy.deepcopy(before)
    payload["qkv.weight"].add_(0.125)
    with pytest.raises(ValueError):
        module.load_state_dict(payload)
    assert_same(module.state_dict(), before)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("altered_parent", [False, True])
def test_invalid_child_payload_is_rejected_before_copying_any_parent_weight(
    device, dtype, altered_parent
):
    model = predictor(device, dtype)
    payload = copy.deepcopy(model.state_dict())
    payload["regional_fusion.edges"][0, 2] = True
    payload["head.weight"].add_(0.125)
    if altered_parent:
        corrupt_edges(model.regional_fusion, "values")
    before = copy.deepcopy(model.state_dict())
    parameter_id = model._parameter_id
    with pytest.raises(ValueError):
        model.load_state_dict(payload)
    assert_same(model.state_dict(), before)
    assert model._parameter_id == parameter_id
    if not altered_parent:
        model.verify_parameter_identity()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "fault", ["values", "dtype", "shape", "configuration", "absent", "hidden", "type"]
)
def test_precision_conversion_does_not_seal_incompatible_regional_geometry(device, dtype, fault):
    model = predictor(device, dtype, regional=fault != "hidden")
    if fault in ("values", "dtype", "shape"):
        corrupt_edges(model.regional_fusion, fault)
    elif fault == "configuration":
        model.regional_fusion = (
            graph(hidden=model.config.hidden_size, topology="self")
            .to(device=device, dtype=dtype)
            .eval()
            .requires_grad_(False)
        )
    elif fault == "absent":
        model.regional_fusion = None
    elif fault == "hidden":
        model.regional_fusion = (
            graph(hidden=model.config.hidden_size, topology="ring")
            .to(device=device, dtype=dtype)
            .eval()
            .requires_grad_(False)
        )
    else:
        model.regional_fusion = torch.nn.Identity()
    parameter_id = model._parameter_id
    converted = torch.float64 if dtype == torch.float32 else torch.float32
    with pytest.raises(ValueError):
        model.to(device=device, dtype=converted)
    assert model._parameter_id == parameter_id


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("source,destination", [("cpu", "cuda:0"), ("cuda:0", "cpu")])
def test_device_conversion_does_not_legitimize_modified_boolean_edges(dtype, source, destination):
    model = predictor(source, dtype)
    corrupt_edges(model.regional_fusion, "values")
    parameter_id = model._parameter_id
    with pytest.raises(ValueError):
        model.to(device=destination)
    assert model._parameter_id == parameter_id


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("level", ["module", "parent"])
def test_genuine_reference_repairs_boolean_values_without_redefining_topology(device, dtype, level):
    model = module_on(device, dtype) if level == "module" else predictor(device, dtype)
    payload = copy.deepcopy(model.state_dict())
    module = model if level == "module" else model.regional_fusion
    corrupt_edges(module, "values")
    model.load_state_dict(payload)
    assert_same(model.state_dict(), payload)
    assert module.edges.dtype == torch.bool and module.edges.shape == (5, 5)
    assert not module.edges[0, 2]
    if level == "parent":
        model.verify_parameter_identity()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("level", ["module", "parent"])
@pytest.mark.parametrize("fault", ["dtype", "shape"])
def test_reference_cannot_repair_an_incompatible_buffer_type_or_shape_by_copy(
    device, dtype, level, fault
):
    model = module_on(device, dtype) if level == "module" else predictor(device, dtype)
    payload = copy.deepcopy(model.state_dict())
    marker = "qkv.weight" if level == "module" else "head.weight"
    payload[marker].add_(0.125)
    corrupt_edges(model if level == "module" else model.regional_fusion, fault)
    before = copy.deepcopy(model.state_dict())
    with pytest.raises(ValueError):
        model.load_state_dict(payload)
    assert_same(model.state_dict(), before)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("regional", [False, True])
def test_valid_load_and_none_preserve_identity_rng_and_the_next_prediction(device, dtype, regional):
    before_cpu = torch.get_rng_state().clone()
    before_cuda = torch.cuda.get_rng_state(0).clone() if device == "cuda:0" else None
    reference = predictor(device, dtype, regional=regional)
    restored = predictor(device, dtype, regional=regional)
    config_before = copy.deepcopy(reference.config.identity())
    extra_before = copy.deepcopy(reference.get_extra_state())
    keys_before = set(reference.state_dict())
    payload = copy.deepcopy(reference.state_dict())
    payload["head.weight"].add_(0.125)
    reference.load_state_dict(payload)
    restored.load_state_dict(payload)
    reference.verify_parameter_identity()
    restored.verify_parameter_identity()
    assert reference.config.identity() == restored.config.identity() == config_before
    assert reference.get_extra_state() == restored.get_extra_state() == extra_before
    assert set(restored.state_dict()) == keys_before
    assert_same(restored.state_dict(), reference.state_dict())
    if not regional:
        assert restored.regional_fusion is None and "regional_fusion" not in config_before
        assert not any(name.startswith("regional_fusion.") for name in keys_before)
    first_batch = api().DecisionBatch.from_corpus(
        raw_batch(), reference.config.inputs, device=device, dtype=dtype
    )
    first = reference.prepare(first_batch, reference.initial_state(first_batch.flow_ids))
    actual_first = restored.prepare(first_batch, restored.initial_state(first_batch.flow_ids))
    assert torch.equal(first.point_predictions, actual_first.point_predictions)
    next_batch = api().DecisionBatch.from_corpus(
        raw_batch(at=first_batch.prediction_at[0] + 10),
        reference.config.inputs,
        device=device,
        dtype=dtype,
    )
    expected = reference.prepare(next_batch, first.next_state)
    actual = restored.prepare(next_batch, actual_first.next_state)
    assert torch.equal(actual.point_predictions, expected.point_predictions)
    assert_same(
        restored.export_state_cpu(actual.next_state),
        reference.export_state_cpu(expected.next_state),
    )
    assert torch.equal(before_cpu, torch.get_rng_state())
    if before_cuda is not None:
        assert torch.equal(before_cuda, torch.cuda.get_rng_state(0))


def tensor_state(model):
    return {
        "parameters": {name: value.detach().clone() for name, value in model.named_parameters()},
        "buffers": {name: value.detach().clone() for name, value in model.named_buffers()},
    }


@pytest.mark.cuda
@pytest.mark.parametrize("level", ["module", "parent"])
def test_cuda_nonfinite_regional_payload_is_rejected_atomically_inside_deferred_checks(level):
    from mars_titan.models.titans.state import deferred_checks

    model = (
        module_on("cuda:0", torch.float64)
        if level == "module"
        else predictor("cuda:0", torch.float64)
    )
    before = copy.deepcopy(model.state_dict())
    payload = copy.deepcopy(before)
    prefix = "" if level == "module" else "regional_fusion."
    marker = "readout.weight" if level == "module" else "head.weight"
    payload[marker].add_(0.125)
    payload[prefix + "qkv.weight"][0, 0] = float("nan")
    parameter_id = getattr(model, "_parameter_id", None)
    with pytest.raises(ValueError), deferred_checks():
        model.load_state_dict(payload)
    assert_same(model.state_dict(), before)
    assert getattr(model, "_parameter_id", None) == parameter_id
    if level == "parent":
        model.verify_parameter_identity()


@pytest.mark.cuda
@pytest.mark.parametrize("level", ["module", "parent"])
def test_assigning_only_cuda_edges_into_a_cpu_target_is_rejected_before_copy(level):
    model = (
        module_on("cpu", torch.float64) if level == "module" else predictor("cpu", torch.float64)
    )
    before = copy.deepcopy(model.state_dict())
    payload = copy.deepcopy(before)
    prefix = "" if level == "module" else "regional_fusion."
    marker = "qkv.weight" if level == "module" else "head.weight"
    payload[prefix + "edges"] = payload[prefix + "edges"].to("cuda:0")
    payload[marker].add_(0.125)
    parameter_id = getattr(model, "_parameter_id", None)
    with pytest.raises(ValueError):
        model.load_state_dict(payload, assign=True)
    assert_same(model.state_dict(), before)
    assert getattr(model, "_parameter_id", None) == parameter_id
    if level == "parent":
        model.verify_parameter_identity()


@pytest.mark.cuda
@pytest.mark.parametrize("mode", ["assign", "partial_metadata"])
def test_assigning_a_cuda_regional_child_into_a_cpu_parent_is_rejected_before_copy(mode):
    model = predictor("cpu", torch.float64)
    before = copy.deepcopy(model.state_dict())
    payload = copy.deepcopy(before)
    for name, value in payload.items():
        if name.startswith("regional_fusion.") and isinstance(value, torch.Tensor):
            payload[name] = value.to("cuda:0")
    payload["head.weight"].add_(0.125)
    if mode == "partial_metadata":
        payload._metadata["regional_fusion"]["assign_to_params_buffers"] = True
    parameter_id = model._parameter_id
    with pytest.raises(ValueError):
        model.load_state_dict(payload, assign=mode == "assign")
    assert_same(model.state_dict(), before)
    assert model._parameter_id == parameter_id
    model.verify_parameter_identity()


@pytest.mark.cuda
@pytest.mark.parametrize("level", ["module", "parent"])
def test_mixed_device_payload_with_copy_preserves_the_cpu_target_device(level):
    model = (
        module_on("cpu", torch.float64) if level == "module" else predictor("cpu", torch.float64)
    )
    expected = copy.deepcopy(model.state_dict())
    prefix = "" if level == "module" else "regional_fusion."
    marker = "qkv.weight" if level == "module" else "head.weight"
    expected[marker].add_(0.125)
    payload = copy.deepcopy(expected)
    for name, value in payload.items():
        if isinstance(value, torch.Tensor) and (
            name.startswith(prefix) if prefix else name == "edges"
        ):
            payload[name] = value.to("cuda:0")
    model.load_state_dict(payload, assign=False)
    assert_same(model.state_dict(), expected)
    assert all(value.device.type == "cpu" for value in model.parameters())
    assert all(value.device.type == "cpu" for value in model.buffers())
    if level == "parent":
        model.verify_parameter_identity()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("level", ["module", "parent"])
def test_homogeneous_assignment_on_the_same_device_loads_the_declared_payload(device, level):
    model = (
        module_on(device, torch.float64) if level == "module" else predictor(device, torch.float64)
    )
    payload = copy.deepcopy(model.state_dict())
    marker = "qkv.weight" if level == "module" else "head.weight"
    payload[marker].add_(0.125)
    model.load_state_dict(payload, assign=True)
    assert_same(model.state_dict(), payload)
    if level == "parent":
        model.verify_parameter_identity()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "fault",
    [
        "absent_module",
        "hidden_module",
        "foreign_module",
        "different_configuration",
        "absent_child_configuration",
        "foreign_child_configuration",
        "foreign_parent_configuration",
    ],
)
def test_parent_prevalidates_its_regional_component_before_copying_any_weight(device, fault):
    model = predictor(device, torch.float64, regional=fault != "hidden_module")
    payload = copy.deepcopy(model.state_dict())
    if fault == "absent_module":
        model.regional_fusion = None
    elif fault == "hidden_module":
        model.regional_fusion = graph(hidden=model.config.hidden_size, topology="ring").to(device)
    elif fault == "foreign_module":
        model.regional_fusion = torch.nn.Identity()
    elif fault == "different_configuration":
        model.regional_fusion = graph(hidden=model.config.hidden_size, topology="self").to(device)
    elif fault == "absent_child_configuration":
        model.regional_fusion.config = None
    elif fault == "foreign_child_configuration":
        model.regional_fusion.config = object()
    else:
        object.__setattr__(model.config, "regional_fusion", object())
    if fault in ("absent_module", "hidden_module", "foreign_module", "different_configuration"):
        # El payload coincide con la estructura alterada y no depende de claves ausentes.
        payload = copy.deepcopy(model.state_dict())
    payload["head.weight"].add_(0.125)
    before = tensor_state(model)
    parameter_id = model._parameter_id
    with pytest.raises(ValueError):
        model.load_state_dict(payload)
    assert_same(tensor_state(model), before)
    assert model._parameter_id == parameter_id


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("level", ["module", "parent"])
@pytest.mark.parametrize("fault", ["nonpersistent", "unregistered", "sparse"])
def test_incompatible_physical_buffer_is_rejected_before_copying_weights(device, level, fault):
    model = (
        module_on(device, torch.float64) if level == "module" else predictor(device, torch.float64)
    )
    module = model if level == "module" else model.regional_fusion
    if fault == "nonpersistent":
        module._non_persistent_buffers_set.add("edges")
    elif fault == "unregistered":
        edges = module._buffers.pop("edges")
        module.edges = edges
    else:
        module.edges = module.edges.to_sparse()
    before = copy.deepcopy(model.state_dict())
    payload = copy.deepcopy(before)
    marker = "qkv.weight" if level == "module" else "head.weight"
    payload[marker].add_(0.125)
    original_edges = module.edges.detach().clone()
    parameter_id = getattr(model, "_parameter_id", None)
    with pytest.raises(ValueError):
        model.load_state_dict(payload)
    assert_same(model.state_dict(), before)
    assert_same(module.edges, original_edges)
    assert getattr(model, "_parameter_id", None) == parameter_id


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("fault", ["sparse_edges", "sparse_weight", "nan"])
def test_incompatible_regional_payload_is_rejected_before_copying_parent_weights(
    device, dtype, fault
):
    model = predictor(device, dtype)
    before = copy.deepcopy(model.state_dict())
    payload = copy.deepcopy(before)
    payload["head.weight"].add_(0.125)
    if fault == "sparse_edges":
        payload["regional_fusion.edges"] = payload["regional_fusion.edges"].to_sparse()
    elif fault == "sparse_weight":
        payload["regional_fusion.qkv.weight"] = payload["regional_fusion.qkv.weight"].to_sparse()
    else:
        payload["regional_fusion.qkv.weight"][0, 0] = float("nan")
    parameter_id = model._parameter_id
    with pytest.raises(ValueError):
        model.load_state_dict(payload)
    assert_same(model.state_dict(), before)
    assert model._parameter_id == parameter_id
    model.verify_parameter_identity()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("fault", ["hidden_dimension", "batch_limit", "qkv_shape", "readout_shape"])
def test_child_dimensions_and_shapes_must_match_the_parent_before_any_copy(device, fault):
    model = predictor(device, torch.float64)
    if fault == "hidden_dimension":
        model.regional_fusion = graph(hidden=model.config.hidden_size // 2, topology="ring").to(
            device
        )
    elif fault == "batch_limit":
        model.regional_fusion.max_batch = model.config.max_batch // 2
    else:
        linear = (
            model.regional_fusion.qkv if fault == "qkv_shape" else model.regional_fusion.readout
        )
        linear.weight = torch.nn.Parameter(linear.weight[:-1].clone(), requires_grad=False)
    before = copy.deepcopy(model.state_dict())
    payload = copy.deepcopy(before)
    payload["head.weight"].add_(0.125)
    parameter_id = model._parameter_id
    with pytest.raises(ValueError):
        model.load_state_dict(payload)
    assert_same(model.state_dict(), before)
    assert model._parameter_id == parameter_id


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "topology,expected,count",
    [
        (
            "self",
            [
                [True, False, False, False, False],
                [False, True, False, False, False],
                [False, False, True, False, False],
                [False, False, False, True, False],
                [False, False, False, False, True],
            ],
            5,
        ),
        (
            "ring",
            [
                [True, True, False, False, True],
                [True, True, True, False, False],
                [False, True, True, True, False],
                [False, False, True, True, True],
                [True, False, False, True, True],
            ],
            15,
        ),
        ("complete", [[True] * 5 for _ in range(5)], 25),
    ],
)
def test_declared_topologies_have_the_explicit_five_node_edges(device, topology, expected, count):
    module = graph(topology=topology).to(device)
    oracle = torch.tensor(expected, dtype=torch.bool, device=device)
    assert_same(module.edges, oracle)
    assert module.edges.count_nonzero().item() == count
    payload = copy.deepcopy(module.state_dict())
    module.load_state_dict(payload)
    assert_same(module.edges, oracle)
