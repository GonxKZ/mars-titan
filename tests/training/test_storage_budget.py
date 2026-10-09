"""Escenarios de retención del presupuesto de disco sobre un plan pequeño y medidas fijas.

Las medidas son números inventados con el formato de `storage_measurements`, así que las
pruebas comprueban la contabilidad: qué ajustes se consideran elegidos, qué tramos lee cada
consumidor, la tabla común contada una vez y el pico con el original de cada ajuste.
"""

import json
from pathlib import Path

import pytest

from mars_titan.training import campaign_storage as storage
from mars_titan.training import storage_budget as budget

DECLARATION = Path("configs/baselines/historical-masked-campaign-storage.json")
COUNTS = dict(train=1000, validation=100, calibration=50, evaluation=200, flows=10)
LAYOUT = dict(current=60, large_groups=30, shared_rows=15, shared_rows_ordered=18)
SHARED = 20
AGGREGATE = 0.01


def job(arm, stage, seed, model="neural", candidate="a", family="neural_reference"):
    name = f"search-{candidate}" if stage == "search" else f"{stage}-s{seed}"
    return dict(
        id=f"US/fold-000/{arm}/{name}",
        scope="US",
        window="fold-000",
        arm=arm,
        family=family,
        model=model,
        stage=stage,
        kind="fit",
        seed=seed,
        case=None,
    )


JOBS = [
    job("gru", "search", 42, candidate="a"),
    job("gru", "search", 42, candidate="b"),
    job("gru", "finalist", 43),
    job("ridge", "search", 42, model="ridge", candidate="r1", family="tabular_reference"),
    job("ridge", "search", 42, model="ridge", candidate="r2", family="tabular_reference"),
    job("cm_v1_core_b", "search", 42, model="cm_v1_core", family="cm_v1"),
]
CAMPAIGN = dict(
    comparison_config=dict(
        arms=dict(gru=dict(output="quantile_head_v1"), ridge=dict(output="point"))
    )
)


def measured():
    writers = {name: dict(LAYOUT) for name in ("neural", "ridge", "titans")}
    return {
        "US/fold-000": {
            p: dict(rows=COUNTS[p], shared_rows_bytes=SHARED, writers=writers)
            for p in storage.HELD_OUT
        }
    }


def test_one_selected_job_per_arm_and_seed_without_results():
    selected = budget.selected_jobs(JOBS)
    assert selected == {
        "US/fold-000/gru/search-a",
        "US/fold-000/gru/finalist-s43",
        "US/fold-000/ridge/search-r1",
        "US/fold-000/cm_v1_core_b/search-a",
    }


def test_needed_partitions_follow_the_comparison_outputs():
    selected = budget.selected_jobs(JOBS)
    needed = {j["id"]: budget.needed_partitions(j, CAMPAIGN, selected) for j in JOBS}
    assert needed["US/fold-000/gru/search-a"] == ("calibration", "evaluation")
    assert needed["US/fold-000/gru/finalist-s43"] == ("calibration", "evaluation")
    assert needed["US/fold-000/gru/search-b"] == ()
    assert needed["US/fold-000/ridge/search-r1"] == ("evaluation",)
    assert needed["US/fold-000/ridge/search-r2"] == ()
    # El núcleo auxiliar de CM-v1 no es un brazo comparado.
    assert needed["US/fold-000/cm_v1_core_b/search-a"] == ()


@pytest.fixture
def estimate():
    value = storage.load_storage(DECLARATION)
    reports = {"US": {"fold-000": dict(counts=COUNTS)}}
    extras = dict(aggregate_bytes_per_row=AGGREGATE)
    return budget.base_estimate(CAMPAIGN, JOBS, reports, measured(), value, extras), value


def test_scenarios_keep_less_and_account_shared_tables_once(estimate):
    result, value = estimate
    totals = result["totals"]
    order = ["current_without_release", "current", "large_groups", "shared_rows"]
    kept = [totals[name]["retained_bytes"] for name in order]
    assert kept == sorted(kept, reverse=True) and len(set(kept)) == 4
    assert totals["needed_shared_rows"]["retained_bytes"] < totals["shared_rows"]["retained_bytes"]
    parts = totals["shared_rows"]["retained_parts"]["predictions"]
    # Seis ajustes con tres tramos de 15 bytes y tres tablas comunes.
    assert parts == 6 * 3 * 15 + 3 * SHARED
    needed = totals["needed_shared_rows"]["retained_parts"]["predictions"]

    def aggregate(*partitions):
        return sum(int(COUNTS[p] * AGGREGATE) for p in partitions)

    # Dos ajustes elegidos de la GRU con calibración y evaluación, Ridge con evaluación,
    # dos tablas comunes y agregados en lugar de las demás tablas.
    expected = 2 * (2 * 15 + aggregate("validation")) + 15 + 2 * SHARED
    expected += aggregate("validation", "calibration") + 3 * aggregate(*storage.HELD_OUT)
    assert needed == expected


def test_compaction_after_writing_keeps_the_original_in_the_peak(estimate):
    result, value = estimate
    totals = result["totals"]
    for name in budget.SCENARIOS:
        assert totals[name]["peak_bytes"] >= totals[name]["retained_bytes"]
    footprint = storage.job_footprint(
        JOBS[0], COUNTS, value, prediction_bytes=lambda _, p: LAYOUT["current"]
    )
    assert totals["shared_rows"]["peak_bytes"] >= (
        footprint["retained"]["predictions"] + footprint["transient_bytes"]
    )


def test_groups_count_jobs_and_rows_per_family_and_stage(estimate):
    result, _ = estimate
    groups = {(g["family"], g["stage"]): g for g in result["groups"]}
    assert groups[("neural_reference", "search")]["jobs"] == 2
    assert groups[("neural_reference", "finalist")]["jobs"] == 1
    assert groups[("cm_v1_core", "search")]["held_out_rows"] == 350
    assert result["selected_jobs"] == 4 and result["jobs"] == 6


def test_ablation_adds_evaluation_tables_and_indices_of_chronological_carries():
    from mars_titan.training.modality_ablation_stage import load_stage, plan_stage

    path = Path("configs/evaluation/historical-masked-ablation-stage-a.json")
    jobs = plan_stage(load_stage(path))
    windows = {(j["scope"], j["window"]) for j in jobs}
    reports = {s: {w: dict(counts=COUNTS) for t, w in windows if t == s} for s, _ in windows}
    sizes = {name: dict(LAYOUT) for name in ("neural", "ridge", "xgboost", "titans")}
    evaluation = dict(evaluation=dict(rows=COUNTS["evaluation"], writers=sizes))
    measured = {f"{s}/{w}": evaluation for s, w in windows}
    value = storage.load_storage(DECLARATION)
    result = budget.ablation_estimate(path, reports, measured, value)
    assert result["jobs"] == len(jobs) == 4185
    assert result["evaluation_rows"] == 4185 * COUNTS["evaluation"]
    chronological = result["models"]["titans_mac"]
    events = 2 * COUNTS["evaluation"] + COUNTS[value["index"]["warmup_partition"]]
    assert result["index_bytes"] == chronological * int(events * value["index"]["bytes_per_row"])
    for layout, totals in result["layouts"].items():
        fixed = 4185 * (LAYOUT[layout] + value["job_report_bytes"])
        assert totals["retained_without_indices"] == fixed
        assert totals["retained_bytes"] == fixed + result["index_bytes"]
        assert totals["peak_bytes"] > totals["retained_bytes"]


def test_adapter_scenarios_keep_compared_rows_and_count_the_original_while_compacting():
    from mars_titan.posttraining.campaign_stage import load_stage, plan_stage

    path = Path("configs/posttraining/historical-masked-adapter-stage-a.json")
    jobs = plan_stage(load_stage(path))
    windows = {(j["scope"], j["window"]) for j in jobs}
    reports = {s: {w: dict(counts=COUNTS) for t, w in windows if t == s} for s, _ in windows}
    adapter = dict(
        ordered_row_bytes=1,
        input_row_bytes=1,
        parent_cache_bytes=0,
        state_bytes=0,
        retained_states=0,
        job_report_bytes=0,
        row_bytes_current=100,
        row_bytes_large_groups=60,
        row_bytes_shared_rows=40,
    )
    extras = dict(adapter=adapter, aggregate_bytes_per_row=1)
    result = budget.adapter_estimate(path, reports, extras)["layouts"]
    held_out = sum(COUNTS[p] for p in storage.HELD_OUT)
    assert result["current"]["retained_bytes"] == len(jobs) * held_out * 100
    assert result["shared_rows"]["retained_bytes"] == len(jobs) * held_out * 40
    compared = COUNTS["calibration"] + COUNTS["evaluation"]
    needed = len(jobs) * (compared * 40 + COUNTS["validation"])
    assert result["needed_shared_rows"]["retained_bytes"] == needed
    # Al compactar, la tabla original del último ajuste coexiste con las de la ventana.
    gap = result["shared_rows"]["peak_bytes"] - result["shared_rows"]["retained_bytes"]
    assert gap >= held_out * 100
    assert result["large_groups"]["peak_bytes"] - result["large_groups"]["retained_bytes"] < gap


def test_open_search_keeps_the_rows_of_its_confirmed_cases_in_the_peak():
    search = [job("gru", "search", 42, candidate=c) for c in ("a", "b", "c")]
    # Un trabajo auxiliar posterior ya no ve las filas de la búsqueda cerrada.
    after = job("aux", "search", 42, candidate="z")
    value = storage.load_storage(DECLARATION)
    reports = {"US": {"fold-000": dict(counts=COUNTS)}}
    extras = dict(aggregate_bytes_per_row=AGGREGATE)
    totals = budget.base_estimate(CAMPAIGN, search, reports, measured(), value, extras)["totals"]
    footprint = storage.job_footprint(
        search[0], COUNTS, value, release=True, prediction_bytes=lambda _, p: LAYOUT["current"]
    )
    # Con el último caso abierto, el caso b aún guarda calibración y evaluación por fila.
    pending = 2 * LAYOUT["shared_rows"] - sum(
        int(COUNTS[p] * AGGREGATE) for p in ("calibration", "evaluation")
    )
    scenario = totals["needed_shared_rows"]
    transient = footprint["transient_bytes"] + footprint["retained"]["predictions"]
    assert scenario["peak_bytes"] == scenario["retained_bytes"] + transient + pending
    assert scenario["peak_job"] == search[-1]["id"]
    closed = budget.base_estimate(CAMPAIGN, [*search, after], reports, measured(), value, extras)
    scenario = closed["totals"]["needed_shared_rows"]
    assert scenario["peak_job"] == after["id"]
    assert scenario["peak_bytes"] == scenario["retained_bytes"] + transient


def test_formula_rebuilds_each_window_from_its_rows_and_measured_bytes_per_row():
    coefficients = budget.row_bytes(measured())
    assert coefficients["writers"]["neural"]["current"]["evaluation"] == 60 / COUNTS["evaluation"]
    assert coefficients["shared_rows"]["calibration"] == SHARED / COUNTS["calibration"]
    same = budget.formula_windows({"US": {"fold-000": COUNTS}}, coefficients)
    assert same == measured()
    double = {p: 2 * v for p, v in COUNTS.items()}
    scaled = budget.formula_windows({"US": {"fold-001": double}}, coefficients)["US/fold-001"]
    assert scaled["evaluation"]["writers"]["ridge"]["shared_rows"] == 2 * LAYOUT["shared_rows"]
    assert scaled["validation"]["shared_rows_bytes"] == 2 * SHARED
    # Un byte parcial cuenta entero: 201 filas a 0,3 bytes por fila ocupan 61 bytes.
    odd = dict(COUNTS, evaluation=COUNTS["evaluation"] + 1)
    grown = budget.formula_windows({"US": {"fold-002": odd}}, coefficients)["US/fold-002"]
    assert grown["evaluation"]["writers"]["neural"]["current"] == 61
    # Los bytes por fila son el máximo entre ventanas, nunca la media.
    window = measured()["US/fold-000"]
    writers = dict(window["evaluation"]["writers"], ridge=dict(LAYOUT, current=90))
    larger = dict(
        window, evaluation=dict(window["evaluation"], shared_rows_bytes=3 * SHARED, writers=writers)
    )
    # La ventana mayor va primero para que el resultado no dependa del orden.
    both = budget.row_bytes({"US/fold-000": larger, "US/fold-001": window})
    assert both["shared_rows"]["evaluation"] == 3 * SHARED / COUNTS["evaluation"]
    assert both["writers"]["ridge"]["current"]["evaluation"] == 90 / COUNTS["evaluation"]
    assert both["writers"]["neural"]["current"]["evaluation"] == 60 / COUNTS["evaluation"]
    assert both["windows"] == 2


def test_block_reading_removes_the_ordered_corpus_from_the_adapter_peak(tmp_path):
    from mars_titan.posttraining.campaign_stage import load_stage, plan_stage

    # A solo admite la lectura por bloques que necesita su ajuste con las filas nuevas. La
    # copia ordenada se compara con el plan de B, que aún la admite.
    path = Path("configs/posttraining/historical-masked-adapter-stage-b.json")
    declared = json.loads(path.read_text())
    assert declared["cohort_reading"]["source"] == "view_blocks"
    base = path.resolve().parent
    declared.update(
        campaign=str((base / declared["campaign"]).resolve()),
        matrix=str((base / declared["matrix"]).resolve()),
    )
    ordered = tmp_path / "ordered.json"
    ordered.write_text(
        json.dumps(dict(declared, cohort_reading=dict(source="ordered_corpus", retention="keep")))
    )
    jobs = plan_stage(load_stage(path))
    windows = {(j["scope"], j["window"]) for j in jobs}
    reports = {s: {w: dict(counts=COUNTS) for t, w in windows if t == s} for s, _ in windows}
    adapter = dict(
        ordered_row_bytes=1000,
        input_row_bytes=2000,
        parent_cache_bytes=0,
        state_bytes=0,
        retained_states=0,
        job_report_bytes=0,
        row_bytes_current=1,
        row_bytes_large_groups=1,
        row_bytes_shared_rows=1,
    )
    extras = dict(adapter=adapter, aggregate_bytes_per_row=1)
    blocks = budget.adapter_estimate(path, reports, extras)["layouts"]["current"]
    copied = budget.adapter_estimate(ordered, reports, extras)["layouts"]["current"]
    assert blocks["retained_bytes"] == copied["retained_bytes"]
    # Sin copia ordenada, el pico es lo conservado. Con ella, al menos la copia ordenada.
    assert blocks["peak_bytes"] == blocks["retained_bytes"]
    ordered_bytes = (COUNTS["train"] + COUNTS["validation"]) * 1000
    assert copied["peak_bytes"] >= blocks["peak_bytes"] + ordered_bytes
