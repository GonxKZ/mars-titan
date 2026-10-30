"""Recetas no canónicas de B3 comprobadas antes de abrir datos."""

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from mars_titan.memory.mars_titan_variant import check_components, load_declaration
from mars_titan.memory.role_binding import RoleBindingConfig
from mars_titan.training.mars_titan_correction import arm_components


def test_role_factor_is_connected_to_the_window_and_normalized():
    values = dict(rate=0.05, forgetting=0.01)
    arm = dict(
        associative_memory=dict(
            rule="proximal", key="codec", role_binding=asdict(RoleBindingConfig())
        )
    )
    checked, correction = check_components(load_declaration(), arm_components(arm, values))
    assert checked["associative_memory"]["role_binding"] == asdict(RoleBindingConfig())
    assert correction.role_binding == RoleBindingConfig()


@pytest.mark.parametrize(
    "role",
    [
        {},
        {"source": "current_sector", "mode": "bipolar", "seed": 0},
        {"source": "regime", "mode": "bipolar", "seed": 0, "guarantee": True},
    ],
)
def test_incomplete_or_unverified_role_contracts_are_rejected(role):
    arm = dict(
        associative_memory=dict(
            rule="proximal", key="codec", rate=0.05, forgetting=0.01, role_binding=role
        )
    )
    with pytest.raises(ValueError):
        check_components(load_declaration(), arm)


def test_all_role_arms_have_distinct_identities_and_common_recipe():
    document = json.loads(Path("configs/experiments/role-bound-keys-v1.json").read_text())
    assert document["default_enabled"] is False and document["final_test_access"] is False
    identities = []
    for arm in document["arms"].values():
        _, correction = check_components(
            load_declaration(), arm_components(arm["components"], dict(rate=0.05, forgetting=0.01))
        )
        identities.append(json.dumps(correction.identity(), sort_keys=True))
    assert len(identities) == len(set(identities)) == 8
