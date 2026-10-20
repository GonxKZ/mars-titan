"""Todas las etapas de las declaraciones adicionales conservan la misma campaña."""

import json
from pathlib import Path

import pytest

from mars_titan.posttraining import campaign_stage as adapters
from mars_titan.posttraining import stage_comparison
from mars_titan.simulation import policy_plan
from mars_titan.training import campaign_data_policy as data_policy
from mars_titan.training import campaign_plan as plan
from mars_titan.training import campaign_publication as publication
from mars_titan.training import campaign_schedule_command as command
from mars_titan.training import modality_ablation_stage as ablation
from mars_titan.training.rolling_retention import load_schedule

ROOT = Path("configs")
CAMPAIGNS = {
    n: ROOT / f"baselines/historical-masked-campaign-a-{n}-v1.json" for n in ["sequence", "paper"]
}
FILES = {
    "adapters": ("posttraining", "historical-masked-adapter-stage"),
    "posttraining": ("posttraining", "historical-masked-adapter-comparison"),
    "ablation": ("evaluation", "historical-masked-ablation-stage"),
    "rl": ("simulation", "historical-masked-rl-stage"),
    "publication": ("evaluation", "historical-masked-publication"),
}
REGISTRY = dict(
    adapters="posttraining_adapter_matrix",
    ablation="modality_ablation",
    rl="rl_policy_comparison",
    publication="campaign_publication",
)


def path(kind, name):
    directory, prefix = FILES[kind]
    return ROOT / directory / f"{prefix}-a-{name}-v1.json"


def old_path(kind):
    directory, prefix = FILES[kind]
    return ROOT / directory / f"{prefix}-a-v2.json"


def read(source):
    return json.loads(source.read_text())


@pytest.mark.parametrize("name", ["sequence", "paper"])
def test_stages_and_publication_resolve_the_same_campaign_without_changing_the_design(name):
    campaign = plan.load_campaign(CAMPAIGNS[name])
    loaded = {
        "adapters": adapters.load_stage(path("adapters", name)),
        "ablation": ablation.load_stage(path("ablation", name)),
        "rl": policy_plan.load_stage(path("rl", name)),
    }
    for value in loaded.values():
        assert value["campaign"]["sha256"] == campaign["sha256"]
        assert value["campaign"]["path"] == campaign["path"]
    compared = stage_comparison.load_declaration(path("posttraining", name))
    assert compared["stage"]["sha256"] == loaded["adapters"]["sha256"]
    published = publication.load_publication(path("publication", name))
    assert published["campaign_config"]["sha256"] == campaign["sha256"]
    assert published["posttraining"]["stage"]["sha256"] == loaded["adapters"]["sha256"]
    for kind in FILES:
        before, after = read(old_path(kind)), read(path(kind, name))
        assert after["name"] != before["name"]
        for key in ("name", "campaign", "stage", "posttraining_comparison"):
            if key in before:
                after[key] = before[key]
        assert after == before


@pytest.mark.parametrize("name", ["sequence", "paper"])
def test_the_data_audit_includes_all_new_stages_without_replacing_the_old_registry(name):
    campaign = plan.load_campaign(CAMPAIGNS[name])
    names = {p.resolve() for p in data_policy.documents(campaign, plan.LATER_STAGES)}
    for kind, entry in REGISTRY.items():
        assert path(kind, name).resolve() in names
        assert (
            Path(plan.LATER_STAGES[entry]["additional_stages"][name]).resolve()
            == path(kind, name).resolve()
        )
        assert Path(plan.LATER_STAGES[entry]["joint_stage"]).resolve() == old_path(kind).resolve()
    assert old_path("adapters").resolve() not in names
    assert plan.plan_campaign(campaign)


@pytest.mark.parametrize("name", ["sequence", "paper"])
def test_complete_schedule_is_consumed_by_retention_only_for_its_own_campaign(
    name, tmp_path, capsys
):
    output = tmp_path / "schedule.json"
    argv = ["--campaign", str(CAMPAIGNS[name]), "--output", str(output)]
    for kind, option in [
        ("adapters", "--adapter-stage"),
        ("ablation", "--ablation-stage"),
        ("rl", "--rl-stage"),
    ]:
        argv.extend([option, str(path(kind, name))])
    assert command.main(argv) == 0
    summary = json.loads(capsys.readouterr().out)
    assert len(summary["windows"]) == 19
    assert summary["order"] == [
        "base_search",
        "selection",
        "selected_case_seeds",
        "online",
        "adapters",
        "chain",
        "ablation",
        "rl",
        "comparison",
        "release",
    ]
    campaign = plan.load_campaign(CAMPAIGNS[name])
    rows = load_schedule(output, campaign)
    assert len(rows) == 19 and sum(len(r["scopes"]) for r in rows) == 51
    assert sum(r["jobs"].get("adapters", 0) for r in summary["windows"]) > 0
    assert sum(r["jobs"].get("ablation", 0) for r in summary["windows"]) > 0
    assert sum(r["jobs"].get("rl", 0) for r in summary["windows"]) > 0
    other = "paper" if name == "sequence" else "sequence"
    with pytest.raises(ValueError, match="otra|pertenece"):
        load_schedule(output, plan.load_campaign(CAMPAIGNS[other]))


@pytest.mark.parametrize(
    "kind,option",
    [("adapters", "--adapter-stage"), ("ablation", "--ablation-stage"), ("rl", "--rl-stage")],
)
def test_historical_stages_are_still_rejected_for_a_new_campaign(kind, option):
    with pytest.raises(ValueError, match="otra campaña"):
        command.main(["--campaign", str(CAMPAIGNS["paper"]), option, str(old_path(kind))])


def test_publication_rejects_a_posttraining_comparison_from_the_other_campaign(tmp_path):
    document = read(path("publication", "paper"))
    for key in ("campaign", "comparison_matrix"):
        document[key] = str((path("publication", "paper").parent / document[key]).resolve())
    document["posttraining_comparison"] = str(path("posttraining", "sequence").resolve())
    target = tmp_path / "mixed.json"
    target.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="no deriva"):
        publication.load_publication(target)
