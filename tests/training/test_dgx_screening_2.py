"""Segunda criba de la DGX: macro como condicionante y retirada de modalidades, con su control."""

import json
from pathlib import Path

import pytest

from mars_titan.training import comparison_campaign as campaign

ROOT = Path(__file__).resolve().parents[2]
DECLARATION = ROOT / "configs/experiments/dgx-us-screening-2-v1.json"
CAMPAIGN = ROOT / "configs/experiments/dgx-us-campaign-v2.json"
VARIANTS = dict(
    titans_mac_macro=dict(presence_modulation="macro_film_v1"),
    titans_mac_no_charts=dict(removed_modalities=dict(names=["charts"])),
    titans_mac_no_macro=dict(removed_modalities=dict(names=["macro"])),
    titans_mac_no_news=dict(removed_modalities=dict(names=["news"])),
    titans_mac_no_fundamentals=dict(removed_modalities=dict(names=["fundamentals"])),
)


@pytest.fixture(scope="module")
def document():
    return campaign.load_declaration(DECLARATION)[0]


def test_the_screening_shares_the_rules_and_the_control_of_the_campaign(document):
    v2 = json.loads(CAMPAIGN.read_text())
    assert document["final_test_access"] is False and document["seeds"] == [42]
    assert [origin["year"] for origin in document["origins"]] == [2013, 2014, 2015, 2016]
    for key in ("stopping", "budget", "cohort", "market"):
        assert document[key] == v2[key]
    assert document["execution"]["threads"] == v2["execution"]["threads"]
    assert document["recipes"]["core"] == v2["recipes"]["core"]
    # El control es el mismo modelo de la campaña, salvo su nivel: debe dar los mismos bits.
    control, reference = dict(document["models"]["titans_mac"]), dict(v2["models"]["titans_mac"])
    assert control.pop("tier") == "control" and reference.pop("tier") == "core"
    assert control == reference
    assert len(campaign.plan(document, "mars_titan")) == 6 * 4


@pytest.mark.parametrize("name", sorted(VARIANTS))
def test_each_variant_changes_one_declared_option_of_the_control(document, name):
    variant, control = dict(document["models"][name]), dict(document["models"]["titans_mac"])
    assert variant.pop("tier") != control.pop("tier")
    assert variant.pop("predictor_changes") == VARIANTS[name]
    assert variant == control and "predictor_changes" not in control
