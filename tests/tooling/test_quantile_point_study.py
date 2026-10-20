"""Sellado de objetivos por modelo sin alterar las recetas históricas."""

import copy
import json

import pytest

from tests.tooling.test_neural_plasticity_study import api, view


def prepare(tmp_path):
    source = tmp_path / "source/view.json"
    source.parent.mkdir()
    source.write_text(json.dumps(view()))
    work = tmp_path / "point"
    return work, api().prepare(source, work, version="quantile_point_v1")


def test_every_loss_control_has_its_own_sealed_recipe_and_all_paired_seeds(tmp_path):
    work, result = prepare(tmp_path)
    assert api().verify_prepared(work) == result
    assert len(result["jobs"]) == 25
    assert result["counts"] == view()["counts"]
    common = json.loads((work / "recipes/recipe.json").read_text())
    assert common["recipe"]["loss"] == "pinball"
    for job in result["jobs"]:
        assert job["seed"] in (42, 43, 44, 45, 46)
        name = job.get("recipe", "recipes/recipe.json")
        recipe = json.loads((work / name).read_text())
        assert name in result["artifacts"]
        if job["arm"] in ("point_l1", "point_component"):
            expected = (
                "quantile_point_mae_v1"
                if job["arm"] == "point_l1"
                else "quantile_point_component_v1"
            )
            assert recipe["recipe"]["loss"] == expected
            recipe["recipe"]["loss"] = "pinball"
            assert recipe == common
        else:
            assert "recipe" not in job and recipe == common
    protocol = json.loads((work / "recipes/protocol.json").read_text())
    assert protocol["seeds"] == [42, 43, 44, 45, 46]
    assert result["configuration"]["validity_control"] == dict(
        reference="original",
        unit="all_feedback",
        criterion="identical_predictions_and_shared_parameters",
        required_before_interpretation=True,
    )


@pytest.mark.parametrize("fault", ["recipe", "job", "missing", "extra"])
def test_an_objective_cannot_be_swapped_omitted_or_injected_after_preparation(tmp_path, fault):
    work, result = prepare(tmp_path)
    if fault == "recipe":
        path = work / "recipes/point_l1.json"
        current = json.loads(path.read_text())
        current["recipe"]["loss"] = "pinball"
        path.write_text(json.dumps(current))
    else:
        if fault == "job":
            next(j for j in result["jobs"] if j["arm"] == "point_l1")["recipe"] = (
                "recipes/point_component.json"
            )
        elif fault == "missing":
            result["artifacts"].pop("recipes/point_component.json")
        else:
            result["artifacts"]["recipes/undeclared.json"] = "0" * 64
        (work / "prepared.json").write_text(json.dumps(result))
    with pytest.raises(ValueError):
        api().verify_prepared(work)


@pytest.mark.parametrize("fault", ["unknown_loss", "path_arm"])
def test_a_bad_point_catalogue_is_rejected_before_creating_the_work_directory(
    tmp_path, monkeypatch, fault
):
    module = api()
    config, catalog = module.study_files("quantile_point_v1")
    value = json.loads(catalog.read_text())
    if fault == "unknown_loss":
        value["arms"]["point_l1"]["recipe_loss"] = "mse"
    else:
        value["arms"]["../outside"] = copy.deepcopy(value["arms"]["point_l1"])
    bad = tmp_path / "catalog.json"
    bad.write_text(json.dumps(value))
    monkeypatch.setattr(module, "study_files", lambda version: (config, bad))
    source = tmp_path / "source/view.json"
    source.parent.mkdir()
    source.write_text(json.dumps(view()))
    with pytest.raises(ValueError):
        module.prepare(source, tmp_path / "bad", version="quantile_point_v1")
    assert not (tmp_path / "bad").exists()


@pytest.mark.parametrize("wrong", [False, True])
def test_run_reader_requires_the_recipe_of_its_control_instead_of_the_common_loss(tmp_path, wrong):
    from benchmarks.neural_plasticity_analysis import read_run
    from tests.tooling.test_neural_plasticity_analysis import run_fixture

    prepared, job, report, folder = run_fixture(tmp_path)
    job.update(arm="point_l1", recipe="recipes/point_l1.json")
    prepared["artifacts"][job["recipe"]] = "c" * 64
    report["request"]["recipe_sha256"] = (
        prepared["artifacts"]["recipes/recipe.json"] if wrong else "c" * 64
    )
    (folder / "run.json").write_text(json.dumps(report))
    if wrong:
        with pytest.raises(ValueError):
            read_run(tmp_path, prepared, job)
    else:
        actual, table, _ = read_run(tmp_path, prepared, job)
        assert actual == report and len(table) == 3


@pytest.mark.parametrize(
    "job",
    [
        dict(arm="../outside", recipe="recipes/../outside.json"),
        dict(arm="point_l1", recipe="/tmp/outside.json"),
        dict(arm="point_l1", recipe="recipes/point_component.json"),
        dict(arm="point_l1", recipe="recipes/recipe.json"),
        dict(recipe="recipes/point_l1.json"),
    ],
)
def test_a_job_recipe_cannot_escape_or_alias_another_arm(job):
    with pytest.raises(ValueError):
        api().recipe_name(job)
