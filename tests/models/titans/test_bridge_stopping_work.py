"""El coste observado necesita una configuración adaptativa explícita."""

import pytest
import torch
from test_adaptive_bridge_contracts import adaptive_configuration, fixed_configuration

from mars_titan.models.titans.episodic_readout import EpisodicReadout


def test_fixed_reader_rejects_adaptive_work_instead_of_ignoring_it():
    reader = EpisodicReadout(fixed_configuration(adaptive_configuration()), dtype=torch.float64)
    with pytest.raises(ValueError, match="adaptativ"):
        reader.recurrence_work(1, selected_steps=1, evaluated_steps=1)
