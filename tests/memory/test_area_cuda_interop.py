"""Interoperabilidad pendiente entre padre CUDA y estado corrector CPU FP64."""

from dataclasses import replace

import pytest
from test_area_correction_window import AREAS, BRIDGED, KERNEL
from test_financial_session_associative import four_flow_source as four_flow_source
from test_financial_session_associative import module_backend as module_backend
from test_mars_titan_correction_parity import RECIPE, SOURCE, core, emitted
from test_mars_titan_session_parity import events
from torch.nn.attention import SDPBackend, sdpa_kernel

from mars_titan.memory.multiscale_mixture import MultiscaleMixtureConfig
from mars_titan.memory.role_binding import RoleBindingConfig
from mars_titan.training.mars_titan_correction import CorrectionInference


@pytest.mark.cuda
@pytest.mark.parametrize(
    "correction",
    [
        AREAS,
        KERNEL,
        BRIDGED,
        replace(BRIDGED, role_binding=RoleBindingConfig()),
        replace(
            AREAS,
            memory=replace(AREAS.memory, forgetting=0.0),
            multiscale=MultiscaleMixtureConfig(),
        ),
    ],
    ids=["areas", "kernel", "bridge", "role_bridge", "multiscale"],
)
def test_cuda_parent_and_cpu_memory_preserve_the_cpu_trajectory(
    four_flow_source, correction, monkeypatch
):
    if correction.routing is not None:
        from test_regime_routing import daily_routes

        from mars_titan.memory.regimes import RegimeRule

        monkeypatch.setattr(RegimeRule, "routes", daily_routes)
    results = {}
    with sdpa_kernel(SDPBackend.MATH):
        for device in ("cpu", "cuda:0"):
            predictor = core(four_flow_source).to(device)
            runner = CorrectionInference(
                predictor, RECIPE, correction, four_flow_source["codec"], audit=True
            )
            runner._pass(SOURCE, events(four_flow_source))
            results[device] = emitted(runner)
    left, right = results["cpu"], results["cuda:0"]
    assert [x[:2] for x in left] == [x[:2] for x in right]
    for cpu, cuda in zip(left, right, strict=True):
        assert cuda[2] == pytest.approx(cpu[2], rel=1e-9, abs=1e-12)


@pytest.mark.cuda
def test_cuda_parent_keeps_the_same_paired_lesion_columns(four_flow_source):
    import pyarrow as pa

    from mars_titan.memory.lesion_protocol import LesionProtocol
    from mars_titan.training.mars_titan_correction import prediction_rows

    correction = replace(
        AREAS,
        memory=replace(AREAS.memory, forgetting=0.0),
        multiscale=MultiscaleMixtureConfig(loss_scale=1.0),
    )
    protocol = LesionProtocol(seeds=(0,), fractions=((1, 2),))
    outputs = {}
    with sdpa_kernel(SDPBackend.MATH):
        for device in ("cpu", "cuda:0"):
            runner = CorrectionInference(
                core(four_flow_source).to(device),
                RECIPE,
                correction,
                four_flow_source["codec"],
                lesions=protocol,
            )
            rows = prediction_rows(runner)
            runner._pass(SOURCE, events(four_flow_source), rows=rows)
            outputs[device] = pa.concat_tables(rows.finish())
    for name in ["prediction", *("lesion_" + name for name in runner.lesion_plan.names)]:
        assert outputs["cuda:0"][name].to_pylist() == pytest.approx(
            outputs["cpu"][name].to_pylist(), rel=1e-9, abs=1e-12
        )
