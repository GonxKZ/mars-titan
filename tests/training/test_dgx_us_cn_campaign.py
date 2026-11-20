"""Campaña DGX conjunta de EE. UU. y China: las reglas y los modelos de la v2 de EE. UU."""

import hashlib
import json
from pathlib import Path

from mars_titan.training import comparison_campaign as campaign
from mars_titan.training.comparison_cohorts import FULL_VIEW_RULE

ROOT = Path(__file__).resolve().parents[2]
DECLARATION = ROOT / "configs/experiments/dgx-us-cn-campaign-v1.json"
SINGLE = ROOT / "configs/experiments/dgx-us-campaign-v2.json"
OWN = {"market", "cohort", "description", "source_declaration_sha256"}


def test_the_joint_campaign_only_changes_the_market_and_its_cohort():
    document = campaign.load_declaration(DECLARATION)[0]
    single = json.loads(SINGLE.read_text())
    assert document["market"] == "US+CN" and document["final_test_access"] is False
    # 4.198 activos de EE. UU. y 810 de China: la vista completa de cada origen.
    assert document["cohort"] == dict(size=4198 + 810, rule=FULL_VIEW_RULE)
    assert single["market"] == "US" and single["cohort"] == dict(size=4198)
    assert set(document) == set(single)
    for key in set(single) - OWN:
        assert document[key] == single[key], key
    digest = hashlib.sha256(SINGLE.read_bytes()).hexdigest()
    assert document["source_declaration_sha256"] == digest
    assert len(campaign.plan(document, "mars_titan")) == 53 * 19 * 3


def test_each_job_of_the_joint_campaign_has_its_pair_in_the_us_campaign():
    document = campaign.load_declaration(DECLARATION)[0]
    single = campaign.load_declaration(SINGLE)[0]
    joint = [(job.key, job.runner, job.needs) for job in campaign.plan(document, "mars_titan")]
    alone = [(job.key, job.runner, job.needs) for job in campaign.plan(single, "mars_titan")]
    assert joint == alone
