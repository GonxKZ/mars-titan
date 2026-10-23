"""Conservar los dos ejes regionales en el catálogo identificado."""

import pytest

from mars_titan.models.titans.mechanism_catalog import MechanismCatalog
from mars_titan.models.titans.regional_execution import RegionalExecutionConfig
from mars_titan.models.titans.regional_hierarchy import RegionalHierarchyConfig


@pytest.mark.parametrize("execution", (RegionalExecutionConfig(), RegionalHierarchyConfig()))
def test_hierarchical_reader_and_pre_mac_execution_keep_both_axes(execution):
    catalog = MechanismCatalog(
        reader_contract="shared_evidence_bridge_hierarchy_v1",
        regional_execution=execution,
    )
    identity = catalog.identity()
    assert identity["reader_contract"] == "shared_evidence_bridge_hierarchy_v1"
    assert identity["regional_execution"] == execution.identity()
    assert identity["regional_trace"] == "common_ids_four_views_two_groups_one_root_effective_graph"
    assert MechanismCatalog.restore(identity).identity() == identity
    assert len(catalog.actions()) == 36
