"""Protocolo de la criba de #746 y enumeración de sus trabajos, sin datos ni PyTorch."""

import copy
import json
import subprocess
import sys
from collections import Counter, defaultdict

import pytest

from mars_titan.training import variant_screening as screening
from mars_titan.training.screening_cohorts import ORIGIN_FOLDS


@pytest.fixture(scope="module")
def protocol():
    return screening.load_protocol()


@pytest.fixture(scope="module")
def jobs(protocol):
    return screening.screening_jobs(protocol)


def by_use(jobs):
    uses = defaultdict(set)
    for job in jobs:
        for label in job["used_by"]:
            uses[label].add(job["definition_sha256"])
    return uses


def test_the_registered_protocol_keeps_the_test_and_the_confirmation_closed(protocol):
    assert protocol["status"] == "registered_before_opening_s"
    assert protocol["default_enabled"] is False and protocol["final_test_access"] is False
    assert {o["year"]: o["fold"] for o in protocol["origins"]} == ORIGIN_FOLDS
    assert protocol["partitions"]["confirmation"]["opened_by_the_screening"] is False
    assert protocol["partitions"]["final_test"]["opened"] is False
    assert tuple(protocol["catalog"]) == screening.CATALOG
    assert protocol["fit"]["fit_months"] == 12 and protocol["fit"]["stopping"] == "common_epoch"
    assert protocol["fit"]["selection"] == protocol["selection"]["rule"] == screening.SELECTION_RULE
    amendment = protocol["selection"]["amendment"]
    assert amendment["registered_before_opening_s"] is True
    assert amendment["previous_rule"] == "best_validation_session_mae_within_the_fixed_budget"


def _changed(protocol, path, value):
    document = copy.deepcopy(protocol)
    target = document
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    return document


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("final_test_access",), True),
        (("default_enabled",), True),
        (("status",), "declared"),
        (("seeds",), [42, 43, 44, 45]),
        (("seeds",), [42, 42, 43, 44, 45]),
        (("fit", "fit_months"), 13),
        (("fit", "epochs"), 0),
        (("fit", "stopping"), "patience"),
        (("fit", "stopping"), "fixed_budget"),
        (("fit", "selection"), "best_validation_session_mae_within_the_fixed_budget"),
        (("selection", "metric"), "session_mae"),
        (("selection", "partition"), "evaluation"),
        (("selection", "applies_to"), ["fit"]),
        (("selection", "smoothing_radius"), 12),
        (("selection", "smoothing_radius"), 1.0),
        (("selection", "versions", "epochs"), [3, 6, 9, 13]),
        (("selection", "versions", "epochs"), [6, 3]),
        (("selection", "versions", "models"), ["F1/laplace_scale_head"] * 2),
        (("selection", "versions", "partitions"), ["validation", "evaluation"]),
        (("selection", "rule_check", "device"), "cuda"),
        (("selection", "rule_check", "measure_months"), 0),
        (("selection", "rule_check", "partition"), "evaluation"),
        (("selection", "amendment", "registered_before_opening_s"), False),
        (("selection", "amendment", "reason"), " "),
        (("partitions", "confirmation", "opened_by_the_screening"), True),
        (("partitions", "final_test", "opened"), True),
        (("budget", "concurrent_gpu_jobs"), 2),
        (("inference", "confidence"), 1.0),
        (("catalog", "A1", "runner"), "llm"),
        (("catalog", "B1", "primary"), ["area_memory"]),
    ],
)
def test_changes_that_open_data_or_alter_the_design_are_rejected(protocol, path, value):
    with pytest.raises(ValueError):
        screening.checked_protocol(_changed(protocol, path, value))


def test_the_origins_and_the_catalog_order_are_fixed(protocol):
    document = copy.deepcopy(protocol)
    document["origins"] = document["origins"][:3]
    with pytest.raises(ValueError, match="orígenes"):
        screening.checked_protocol(document)
    document = copy.deepcopy(protocol)
    document["catalog"] = dict(reversed(document["catalog"].items()))
    with pytest.raises(ValueError, match="catálogo"):
        screening.checked_protocol(document)


def test_identical_definitions_are_a_single_job(jobs):
    uses = by_use(jobs)
    shared = uses["references/scalar_reference"]
    assert uses["F1/absolute_error_control"] == shared == uses["F4/absolute_error_control"]
    assert uses["B1/area_memory"] == uses["B2/no_bridge"] == uses["B3/no_binding"]
    assert uses["B1/linear_codec"] == uses["E1/historical_retention"] == uses["E3/baseline"]
    assert uses["E3/baseline"] == uses["E4/baseline"]
    assert len({job["id"] for job in jobs}) == len(jobs)


def test_every_definition_runs_once_per_origin_and_seed(protocol, jobs):
    counts = Counter(job["definition_sha256"] for job in jobs)
    assert set(counts.values()) == {len(protocol["origins"]) * len(protocol["seeds"])}
    for job in jobs:
        assert job["id"].startswith(f"{job['origin']}-{job['seed']}-")
        assert job["fold"] == ORIGIN_FOLDS[job["origin"]]


def test_every_job_runs_after_its_parent_and_its_matched_variant(jobs):
    position = {job["id"]: index for index, job in enumerate(jobs)}
    for job in jobs:
        for field in screening.REQUIRES:
            if job[field] is not None:
                required = jobs[position[job[field]]]
                assert position[job[field]] < position[job["id"]]
                assert (required["origin"], required["seed"]) == (job["origin"], job["seed"])


def test_the_seeds_of_each_common_group_run_one_after_another(protocol, jobs):
    groups = defaultdict(list)
    for position, job in enumerate(jobs):
        if "common_group" in job:
            groups[job["common_group"]].append((position, job["seed"]))
    assert groups
    for members in groups.values():
        positions = [position for position, _ in members]
        assert positions == list(range(positions[0], positions[0] + len(protocol["seeds"])))
        assert [seed for _, seed in members] == protocol["seeds"]


def test_corrections_run_every_case_of_their_recipe_and_fits_the_declared_case(protocol, jobs):
    cases = defaultdict(set)
    for job in jobs:
        definition = job["definition"]
        if (job["origin"], job["seed"]) == (2019, 42):
            for label in job["used_by"]:
                cases[label].add(definition.get("search_case"))
    assert cases["B1/area_memory"] == {"eta5e-2", "eta25e-2"}
    assert cases["E3/standardized_error"] == {"eta5e-2", "eta25e-2"}
    assert cases["references/quantile_reference"] == {protocol["fit"]["search_case"]}
    assert cases["D1/nested_k2_j2_final"] == {protocol["fit"]["search_case"]}
    assert cases["E2/utility"] == {None}


def test_the_parent_is_the_reference_fit_of_the_same_origin_and_seed(jobs):
    reference = {
        (job["origin"], job["seed"]): job["id"]
        for job in jobs
        if "references/quantile_reference" in job["used_by"]
    }
    reader = {
        (job["origin"], job["seed"]): job["id"]
        for job in jobs
        if "references/reader_reference" in job["used_by"]
    }
    for job in jobs:
        label = job["used_by"][0]
        if label.startswith(("B1/", "E1/", "D1/")):
            assert job["parent"] == reference[(job["origin"], job["seed"])]
        if label.startswith("E2/"):
            assert job["parent"] == reader[(job["origin"], job["seed"])]


def test_the_matched_control_reads_the_variant_of_its_own_case(jobs):
    index = {job["id"]: job for job in jobs}
    matched = [job for job in jobs if job["used_by"] == ["E4/matched_constant"]]
    assert matched and all(job["after"] is not None for job in matched)
    for job in matched:
        variant = index[job["after"]]
        assert variant["used_by"] == ["E4/cusum"]
        assert variant["definition"]["search_case"] == job["definition"]["search_case"]
        assert job["definition"]["value"]["matched_to"] == variant["definition_sha256"]
    assert all(job["after"] is None for job in jobs if job not in matched)


def test_output_stages_and_pending_entries_have_no_jobs(protocol, jobs):
    labels = {label.partition("/")[0] for job in jobs for label in job["used_by"]}
    assert not labels & {"A1", "A2", "A3", "C4", "B4", "C2", "C3", "D2", "G1", "G2", "G3"}
    statuses = {row["entry"]: row["status"] for row in screening.pending(protocol)}
    assert statuses["C2"] == statuses["C3"] == "pending_implementation"
    assert statuses["G1"] == "after_the_screening_needs_its_own_approval"
    arms = screening.entry_arms(protocol, "A2")
    assert all(arm.runner == "output" for arm in arms.values())
    assert arms["abstention"].definition["source"] == "quantile_reference"


def test_every_contrast_endpoint_resolves(protocol):
    rows = screening.contrasts(protocol)
    assert rows
    for row in rows:
        assert row["role"] in ("primary", "required", "reported")
        assert row["metric"] in screening.METRICS
    primary = {row["entry"]: row for row in rows if row["role"] == "primary"}
    assert primary["A1"]["control"] == ("common", "zero")
    assert primary["C1"]["variant"] == ("C1", "talking_heads")
    reported = [row for row in rows if row["entry"] == "A3" and row["role"] == "reported"]
    assert reported[0]["control"] == ("A2", "abstention")
    regional = [row for row in rows if row["entry"] == "C6" and row["role"] == "reported"]
    assert regional[0]["control"] == ("references", "regional_reference")
    entries = {row["entry"] for row in rows}
    assert entries == {name for name, entry in protocol["catalog"].items() if "status" not in entry}


def test_an_unknown_endpoint_is_rejected(protocol):
    document = copy.deepcopy(protocol)
    document["catalog"]["B1"]["primary"] = ["area_memory", "no_such_control"]
    with pytest.raises(ValueError, match="no_such_control"):
        screening.contrasts(document)


def test_the_cost_estimate_follows_its_formula(protocol):
    rows = {2019: dict(fit=1000, validation=500, calibration=200, evaluation=800)}
    # Dos versiones y la predicción de cada época repiten tres veces la calibración y la
    # evaluación del ajuste. El lector solo predice una versión más.
    fit = dict(
        id="a",
        origin=2019,
        definition=dict(runner="fit", updates_multiplier=3),
        versions=[3, 6],
        epoch_predictions=True,
    )
    correction = dict(id="b", origin=2019, definition=dict(runner="correction"))
    reader = dict(
        id="c", origin=2019, definition=dict(runner="reader"), versions=[9], epoch_predictions=False
    )
    reference = protocol["cost_reference"]
    epochs = protocol["fit"]["epochs"]
    inference = 1000 / reference["inference_rows_per_second"]
    expected_fit = (
        3
        * epochs
        * (
            1000 * reference["fit_seconds_per_fit_row_epoch"]
            + 500 / reference["validation_rows_per_second"]
        )
        + inference
        + 3 * inference
    )
    expected_correction = 1500 / reference["inference_rows_per_second"]
    expected_reader = reference["reader_window_seconds"] + inference
    result = screening.cost_estimate(protocol, [fit, correction, reader], rows)
    total = expected_fit + expected_correction + expected_reader
    assert result["hours"] == pytest.approx(total / 3600)
    assert result["hours_by_runner"]["fit"] == pytest.approx(expected_fit / 3600)
    assert result["hours_by_runner"]["reader"] == pytest.approx(expected_reader / 3600)
    assert result["within_budget"] is (result["hours"] <= protocol["budget"]["max_total_gpu_hours"])


def test_declarations_must_stay_disabled(tmp_path, protocol):
    original = json.loads(
        (screening.ROOT / "configs/experiments/laplace-scale-head-v1.json").read_text()
    )
    folder = tmp_path / "configs/experiments"
    folder.mkdir(parents=True)
    (folder / "laplace-scale-head-v1.json").write_text(
        json.dumps(dict(original, default_enabled=True))
    )
    with pytest.raises(ValueError, match="desactivada"):
        screening.entry_arms(protocol, "F1", root=tmp_path)


def test_the_fit_limit_is_the_one_of_the_walk_forward_phases():
    from mars_titan.training.walk_forward_phases import MAX_FIT_MONTHS

    assert screening.MAX_FIT_MONTHS == MAX_FIT_MONTHS


def test_the_module_does_not_import_torch(source_environment):
    code = "import sys, mars_titan.training.variant_screening; print('torch' in sys.modules)"
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        env=source_environment,
    )
    assert result.stdout.strip() == "False"
