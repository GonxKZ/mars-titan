"""Paridad CUDA E3 explícita con admisión exclusiva y padre congelado, sin optimizadores."""

import numpy as np
import pytest
import torch
from test_financial_session_associative import four_flow_source as four_flow_source
from test_financial_session_associative import module_backend as module_backend
from test_financial_session_associative import no_target_estimation as no_target_estimation
from test_mars_titan_correction_parity import PROXIMAL, RECIPE, SOURCE, emitted
from test_mars_titan_session_parity import events

from mars_titan.memory.salience_budget import SalienceBudgetConfig
from mars_titan.models.titans.financial import FinancialConfig, FinancialPredictor
from mars_titan.training.experiment_resources import GpuLease
from mars_titan.training.salience_inference import SalienceCorrectionInference


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
@pytest.mark.parametrize("rule", ("standardized_error", "absolute_error", "uniform"))
def test_cuda_salience_repeats_and_matches_the_paired_cpu_parent(four_flow_source, dtype, rule):
    with GpuLease(max_vram=256 * 1024**2):
        assert torch.cuda.is_available(), "Se requiere CUDA sin fallback"
        previous = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            configuration = FinancialConfig(
                four_flow_source["spec"], variant="mac_online", hidden_size=32, seed=42
            )
            cpu = FinancialPredictor(configuration, dtype=dtype, device="cpu").eval()
            cpu.requires_grad_(False)
            weights = cpu.state_dict()

            def walk(model):
                run = SalienceCorrectionInference(
                    model,
                    RECIPE,
                    PROXIMAL,
                    four_flow_source["codec"],
                    salience=SalienceBudgetConfig(rule=rule, writes_per_market=1),
                    audit=True,
                )
                result = run._pass(SOURCE, events(four_flow_source))
                return run, result

            reference, _ = walk(cpu)
            records = []
            for _ in range(2):
                parent = FinancialPredictor(configuration, dtype=dtype, device="cuda:0")
                parent.load_state_dict(weights)
                parent.eval().requires_grad_(False)
                actual, result = walk(parent)
                records.append(emitted(actual))
                expected = emitted(reference)
                assert [r[:2] for r in records[-1]] == [r[:2] for r in expected]
                rtol, atol = (5e-5, 3e-6) if dtype == torch.float32 else (2e-9, 2e-10)
                np.testing.assert_allclose(
                    [r[2] for r in records[-1]], [r[2] for r in expected], rtol=rtol, atol=atol
                )
                assert result["salience"]["observed"] == 16
                assert result["salience"]["selected"] == 4
            assert records[0] == records[1]
        finally:
            torch.backends.cuda.matmul.allow_tf32 = previous
