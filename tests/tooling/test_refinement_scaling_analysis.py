"""Comparar cohortes anidadas sin cambiar población, objetivos ni unidad temporal."""

import copy
import json
from datetime import UTC, datetime, timedelta

import numpy as np
import pyarrow as pa
import pytest


def api():
    from benchmarks import refinement_scaling_analysis

    return refinement_scaling_analysis


OPTIONS = dict(
    bootstrap=dict(block_length=1, replicates=20, seed=20261018, confidence=0.95),
    sensitivity_block_lengths=[2],
)


def table(factor=1.0, *, extra=False, offset=0.0):
    observations = [(0, "A", 2), (0, "B", 6), (1, "A", 10)]
    if extra:
        observations += [(0, "C", 100), (1, "C", 100)]
    return pa.table(
        dict(
            sample_id=[f"US/{asset}/{day}" for day, asset, _ in observations],
            asset_id=[f"US/{asset}" for _, asset, _ in observations],
            market=["US"] * len(observations),
            prediction_at=pa.array(
                [
                    datetime(2023, 1, 3, tzinfo=UTC) + timedelta(days=day)
                    for day, _, _ in observations
                ],
                type=pa.timestamp("us", tz="UTC"),
            ),
            target=np.ones(len(observations)),
            prediction=[1 + error * factor + offset for _, _, error in observations],
            zero=np.zeros(len(observations)),
        )
    )


def tables():
    return {
        name: {
            route: table(factor, extra=name.startswith("n256"), offset=offset)
            for route, offset in (("policy", 0.0), ("k1", 2.0), ("k4cc", 1.0))
        }
        for name, factor in (
            ("n64-original", 1.0),
            ("n64-geometry", 0.5),
            ("n256-original", 0.75),
            ("n256-geometry", 0.2),
        )
    }


def by_name(report):
    return {row["name"]: row for row in report["contrasts"]}


def test_common_assets_use_equal_sessions_and_correct_factorial_sign():
    populations = api().score_populations(tables(), ["A", "B"])
    common = populations["common64"]
    assert common["n64-original:policy"].values.tolist() == [4.0, 10.0]
    result = api().compare_population(common, OPTIONS)
    values = by_name(result)
    row = values["geometry_n64_policy"]
    assert row["base_estimate"] == pytest.approx(7.0)
    assert row["estimate"] == pytest.approx(-3.5)
    assert row["relative_improvement_percent"] == pytest.approx(50.0)
    assert values["scale_original_policy"]["estimate"] == pytest.approx(-1.75)
    assert values["interaction_policy"]["estimate"] == pytest.approx(-0.35)
    assert values["n256-geometry_policy_minus_k1"]["estimate"] == pytest.approx(-2.0)
    assert values["n64-original_policy_minus_zero"]["estimate"] == pytest.approx(6.0)
    assert result["resampling"]["sessions"] == 2
    assert result["multiplicity"]["family_size"] == 27
    assert populations["full256"]["n256-original:policy"].values[0] > 20


@pytest.mark.parametrize("change", ["row", "target", "duplicate", "missing_route"])
def test_mismatched_rows_or_targets_are_not_silently_intersected(change):
    inputs = tables()
    selected = inputs["n256-geometry"]["policy"]
    if change == "row":
        inputs["n256-geometry"]["policy"] = selected.slice(1)
    elif change == "target":
        inputs["n256-geometry"]["policy"] = selected.set_column(
            selected.schema.get_field_index("target"), "target", pa.array([2.0] * len(selected))
        )
    elif change == "duplicate":
        inputs["n256-geometry"]["policy"] = pa.concat_tables([selected, selected.slice(0, 1)])
    else:
        inputs["n256-geometry"].pop("k1")
    with pytest.raises(ValueError):
        api().score_populations(inputs, ["A", "B"])


def test_common_assets_cannot_be_chosen_from_the_successful_rows():
    with pytest.raises(ValueError):
        api().score_populations(tables(), ["A"])


def test_seed_average_waits_for_all_declared_seeds_without_adding_dates():
    populations = api().score_populations(tables(), ["A", "B"])
    first = {42: populations, 43: copy.deepcopy(populations)}
    partial = api().average_seeds(first, [42, 43, 44])
    assert partial == {"status": "pending", "missing_seeds": [44]}
    first[43]["common64"]["n64-original:policy"].values[:] *= 2
    complete = first.copy()
    complete[44] = copy.deepcopy(populations)
    complete[44]["common64"]["n64-original:policy"].values[:] *= 3
    result = api().average_seeds(complete, [42, 43, 44])
    assert result["status"] == "completed"
    values = result["series"]["common64"]["n64-original:policy"]
    assert len(values.values) == 2
    np.testing.assert_array_equal(values.values, [8.0, 20.0])
    np.testing.assert_array_equal(
        populations["common64"]["n64-original:policy"].values, [4.0, 10.0]
    )
    with pytest.raises(ValueError):
        api().average_seeds(complete | {45: populations}, [42, 43, 44])


def put(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def test_learning_hold_precedes_any_result_read(tmp_path, monkeypatch):
    from mars_titan.training.learning_hold import LearningHoldError

    hold = tmp_path / "hold.json"
    put(hold, {"training_allowed": False})
    monkeypatch.setenv("MARS_TITAN_TRAINING_HOLD", str(hold))
    with pytest.raises(LearningHoldError):
        api().analyze({}, tmp_path / "output")
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("change", ["day", "models", "populations"])
def test_seed_average_rejects_different_sessions_or_model_sets(change):
    first = api().score_populations(tables(), ["A", "B"])
    other = copy.deepcopy(first)
    if change == "day":
        other["common64"]["n64-original:policy"].session_period[1] += 1
    elif change == "models":
        other["common64"]["unexpected"] = other["common64"]["zero"]
    else:
        other["unexpected"] = other["common64"]
    with pytest.raises(ValueError):
        api().average_seeds({42: first, 43: other}, [42, 43])


@pytest.fixture
def completed_studies(tmp_path, monkeypatch):
    import pyarrow.parquet as pq

    from mars_titan.data.storage import sha256

    job = dict(
        id="linear_v1-42/evaluate-k1",
        kind="evaluate",
        parent="linear_v1-42/select-k1",
        projection="linear_v1",
        seed=42,
        depth=1,
    )
    group = {"fixture": "two_readers_sealed"}
    monkeypatch.setattr(api(), "selection_group", lambda *_: group)
    works, prepared = {}, {}
    for name, cases in tables().items():
        work = tmp_path / name
        works[name] = work
        assets = ["A", "B", "C"] if name.startswith("n256-") else ["A", "B"]
        record = dict(
            configuration=OPTIONS
            | dict(
                cohort_size=256 if len(assets) == 3 else 64,
                seeds=[42],
                projections=["linear_v1"],
                reader_depths=[1],
            ),
            jobs=[dict(id="core", kind="core"), job],
            code={"fixture": "same_code"},
            environment={"fixture": "same_environment"},
            final_test_opened=False,
            assets=assets,
            cohort=dict(common_assets=["A", "B"]),
            counts={"fold-018": {"evaluation": len(cases["policy"])}},
        )
        prepared[name] = record
        put(work / "prepared.json", record)
        selected = work / "runs" / job["parent"] / "phase.json"
        put(selected, dict(status="completed", fixture="selection"))
        output = work / "runs" / job["id"]
        phases = {}
        for case, values in cases.items():
            folder = output if case == "policy" else output / "controls" / case
            folder = folder / "evaluation"
            folder.mkdir(parents=True)
            pq.write_table(values, folder / "predictions.parquet")
            phase = dict(
                status="completed",
                identity=dict(case=case, prepared_sha256=sha256(work / "prepared.json")),
                artifacts={"predictions.parquet": sha256(folder / "predictions.parquet")},
            )
            put(folder / "phase.json", phase)
            phases[case] = phase
        put(
            output / "run.json",
            dict(
                status="completed",
                final_test_opened=False,
                job=job,
                selection_group=group,
                selection_sha256=sha256(selected),
                phases=dict(evaluation=phases["policy"]),
                controls={case: dict(evaluation=phases[case]) for case in ("k1", "k4cc")},
            ),
        )
    monkeypatch.setattr(api(), "verify_prepared", lambda work: copy.deepcopy(prepared[work.name]))
    return works, prepared, job


@pytest.mark.parametrize(
    "change",
    ["group", "selection", "final_test", "predictions", "phase", "row_count", "preparation"],
)
def test_loading_rejects_changed_selection_phase_or_artifact(completed_studies, change):
    works, prepared, job = completed_studies
    work = works["n64-original"]
    assert api().load_tables(work, job, prepared[work.name]) is not None
    output = work / "runs" / job["id"]
    path = output / "run.json"
    report = json.loads(path.read_text())
    if change == "group":
        report["selection_group"] = {"changed": True}
    elif change == "selection":
        report["selection_sha256"] = "0" * 64
    elif change == "final_test":
        report["final_test_opened"] = True
    elif change == "predictions":
        (output / "evaluation/predictions.parquet").write_bytes(b"corrupted")
    elif change == "phase":
        report["phases"]["evaluation"]["status"] = "running"
    elif change == "preparation":
        phase_path = output / "evaluation/phase.json"
        phase = json.loads(phase_path.read_text())
        phase["identity"]["prepared_sha256"] = "0" * 64
        put(phase_path, phase)
        report["phases"]["evaluation"] = phase
    else:
        prepared[work.name]["counts"]["fold-018"]["evaluation"] += 1
    put(path, report)
    with pytest.raises(ValueError):
        api().load_tables(work, job, prepared[work.name])


@pytest.mark.parametrize("change", ["code", "assets", "common", "size", "seed", "final_test"])
def test_four_preparations_must_share_the_declared_comparison(completed_studies, change):
    works, prepared, _ = completed_studies
    assert api().preparations(works)
    record = prepared["n256-geometry"]
    if change == "code":
        record["code"] = {"changed": True}
    elif change == "assets":
        record["assets"] = ["A", "B", "D"]
    elif change == "common":
        record["cohort"]["common_assets"] = ["A"]
    elif change == "size":
        record["configuration"]["cohort_size"] = 64
    elif change == "seed":
        record["configuration"]["seeds"] = [43]
    else:
        record["final_test_opened"] = True
    with pytest.raises(ValueError):
        api().preparations(works)


@pytest.mark.parametrize("incomplete", [None, "missing", "running"])
def test_analysis_writes_only_complete_paired_groups(
    completed_studies, tmp_path, monkeypatch, incomplete
):
    import pyarrow.parquet as pq

    works, prepared, job = completed_studies
    monkeypatch.setattr(api(), "require_learning_allowed", lambda _: None)
    path = works["n256-geometry"] / "runs" / job["id"] / "run.json"
    if incomplete == "missing":
        path.unlink()
    elif incomplete == "running":
        report = json.loads(path.read_text())
        report["status"] = "running"
        put(path, report)
    output = tmp_path / "summary"
    result = api().analyze(works, output)
    assert json.loads((output / "results.json").read_text()) == result
    if incomplete:
        assert result["status"] == "partial"
        assert result["results"] == {}
        assert result["pending"] == [dict(job=job["id"], missing_studies=["n256-geometry"])]
        assert result["seed_means"]["linear_v1-k1"]["missing_seeds"] == [42]
    else:
        assert result["status"] == "completed" and result["pending"] == []
        common = result["results"][job["id"]]["populations"]["common64"]
        assert by_name(common["comparison"])["geometry_n64_policy"]["estimate"] == -3.5
        sessions = pq.read_table(output / job["id"].replace("/", "_") / common["sessions"]["path"])
        assert len(sessions) == 26
        assert result["seed_means"]["linear_v1-k1"]["status"] == "completed"
    with pytest.raises(FileExistsError):
        api().analyze(works, output)
