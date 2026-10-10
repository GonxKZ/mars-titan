"""Contabilidad del disco de la retención v2 sobre un plan de dos ventanas y medidas fijas.

Las medidas son números inventados con el formato de `storage_measurements`. Se comprueba
qué conserva cada escenario tras liberar, cuándo se libera una entrada de las políticas, el
corpus de los adaptadores con y sin copia ordenada y el crecimiento por ventana.
"""

from pathlib import Path

import pytest

from mars_titan.training import campaign_storage as storage
from mars_titan.training import rolling_storage as rolling

DECLARATION = Path("configs/baselines/historical-masked-campaign-storage.json")
COUNTS = dict(train=1000, validation=100, calibration=50, evaluation=200, flows=10)
HELD = sum(COUNTS[p] for p in storage.HELD_OUT)
LAYOUT = dict(current=60, large_groups=30, shared_rows=15)
SHARED = 20
AGGREGATE = 0.5
WINDOWS = [dict(id=f"fold-00{i}", scopes={"US": f"fold-00{i}"}) for i in range(2)]


def job(window, model="neural", kind="fit"):
    return dict(
        id=f"US/{window}/{model}/{kind}",
        scope="US",
        window=window,
        arm=model,
        family=model,
        model=model,
        stage="search",
        kind=kind,
        seed=42,
        case=None,
    )


JOBS = [job(row["id"]) for row in WINDOWS]
ADAPTER = dict(
    ordered_row_bytes=3,
    input_row_bytes=5,
    parent_cache_bytes=7,
    state_bytes=11,
    retained_states=1,
    job_report_bytes=13,
    row_bytes_current=100,
    row_bytes_large_groups=60,
    row_bytes_shared_rows=40,
    comparison_aggregate_bytes_per_series=17,
)
EXTRAS = dict(aggregate_bytes_per_row=AGGREGATE, adapter=ADAPTER)


def measured():
    writers = {name: dict(LAYOUT) for name in set(storage.WRITERS.values())}
    window = {
        p: dict(rows=COUNTS[p], shared_rows_bytes=SHARED, writers=writers) for p in storage.HELD_OUT
    }
    return {f"US/{row['id']}": window for row in WINDOWS}


@pytest.fixture(scope="module")
def declared():
    return storage.load_storage(DECLARATION)


def estimate(declared, **options):
    counts = {"US": {row["id"]: COUNTS for row in WINDOWS}}
    return rolling.rolling_estimate(JOBS, WINDOWS, counts, measured(), declared, EXTRAS, **options)


def kept_per_fit(declared):
    """Lo que conserva un ajuste neuronal liberado: estado, informes, agregados y registros."""
    fixed = declared["job_report_bytes"] + declared["train_sessions_bytes"]
    fixed += declared["state_bytes"]["neural"]
    return fixed + int(AGGREGATE * HELD) + 3 * rolling.RECORD_BYTES


def test_released_rows_leave_states_reports_and_aggregates(declared):
    result = estimate(declared)["all_regenerated"]
    first, second = result["windows"]
    assert first["retained_bytes"] == kept_per_fit(declared)
    assert second["retained_bytes"] == result["retained_bytes"] == 2 * kept_per_fit(declared)
    # El pico de la base: lo conservado antes, el ajuste con sus tablas y su transitorio.
    written = 3 * LAYOUT["current"]
    footprint = storage.job_footprint(
        JOBS[1], COUNTS, declared, release=True, prediction_bytes=lambda *_: LAYOUT["current"]
    )
    assert second["base"] == (
        first["retained_bytes"] + footprint["retained_bytes"] + footprint["transient_bytes"]
    )
    assert footprint["retained"]["predictions"] == written
    # La regeneración de la liberación escribe las tablas y, si no repite, la compactada.
    assert second["release"] == second["rl"] + written + 3 * LAYOUT["shared_rows"]
    assert result["peak_bytes"] == max(
        row[phase] for row in result["windows"] for phase in rolling.PHASES
    )


def test_without_exact_regeneration_everything_stays_compacted(declared):
    result = estimate(declared)
    gap = result["none_regenerated"]["retained_bytes"] - result["all_regenerated"]["retained_bytes"]
    assert gap == 2 * (3 * LAYOUT["shared_rows"] + 3 * SHARED)


def test_a_policy_input_is_kept_compacted_until_its_last_reader(declared):
    reads = {JOBS[0]["id"]: 1}
    result = estimate(declared, policy_reads=reads)["all_regenerated"]
    first, second = result["windows"]
    compacted = 3 * LAYOUT["shared_rows"]
    assert first["retained_bytes"] == kept_per_fit(declared) + compacted + 3 * SHARED
    # En la última ventana que la lee se libera. La tabla común se cuenta hasta el final.
    assert second["retained_bytes"] == 2 * kept_per_fit(declared) + 3 * SHARED


def test_adapter_corpus_only_occupies_disk_with_an_ordered_copy():
    jobs = [dict(scope="US", window="fold-000", base_arm="gru", seed=s) for s in (42, 43)]
    blocks = rolling.adapter_window(jobs, COUNTS, EXTRAS, ordered_copy=False)
    ordered = rolling.adapter_window(jobs, COUNTS, EXTRAS, ordered_copy=True)
    fixed = ADAPTER["state_bytes"] + ADAPTER["job_report_bytes"]
    caches = 2 * ADAPTER["parent_cache_bytes"]
    assert blocks == ordered | dict(transient=0)
    assert blocks["written"] == 2 * (HELD * 100 + fixed) + caches
    assert blocks["compacted"] == 2 * (HELD * 40 + fixed) + caches
    corpus = (COUNTS["train"] + COUNTS["validation"]) * 3 + COUNTS["train"] * 5
    assert ordered["transient"] == corpus
    assert rolling.adapter_window([], COUNTS, EXTRAS, ordered_copy=True)["transient"] == 0


def test_the_stage_comparison_keeps_the_aggregates_of_every_compared_series():
    def stage_job(base_arm, seed, kind):
        return dict(scope="US", window="fold-001", base_arm=base_arm, seed=seed, kind=kind)

    jobs = [stage_job("gru", s, k) for s in (42, 43) for k in ("frozen", "fit", "fit")]
    # Ridge solo tiene el padre congelado y no entra en la comparación.
    jobs.append(stage_job("ridge", 42, "frozen"))
    # Seis trabajos comparados más el reentreno de la base y la cadena de cada semilla.
    assert rolling.comparison_series(jobs) == 6 + 2 * 2
    assert rolling.comparison_series(jobs[-1:]) == 0
    counted = rolling.adapter_window(jobs, COUNTS, EXTRAS, ordered_copy=False)
    free = dict(EXTRAS, adapter=dict(ADAPTER, comparison_aggregate_bytes_per_series=0))
    without = rolling.adapter_window(jobs, COUNTS, free, ordered_copy=False)
    for key in ("written", "compacted"):
        assert counted[key] - without[key] == 10 * 17
    assert counted["transient"] == without["transient"]


def test_adapters_ablation_and_tapes_add_to_their_window(declared):
    adapters = [dict(scope="US", window="fold-001", base_arm="neural", seed=42)]
    masked = [job("fold-000", model="titans_mac", kind="carry")]
    tapes = {"fold-001": 1000}
    result = estimate(declared, adapter_jobs=adapters, ablation_jobs=masked, tapes=tapes)
    plain = estimate(declared)
    for scenario in rolling.SCENARIOS:
        first, second = result[scenario]["windows"]
        base_first, base_second = plain[scenario]["windows"]
        kept = declared["job_report_bytes"] + rolling.RECORD_BYTES
        if scenario == "none_regenerated":
            kept += LAYOUT["shared_rows"]
        assert first["retained_bytes"] == base_first["retained_bytes"] + kept
        # Sin adaptadores en la primera ventana, la ablación crece sobre lo que dejó la base:
        # su evaluación con el índice en construcción del traslado cronológico y su informe.
        build = int(2 * COUNTS["evaluation"] * declared["index"]["build_bytes_per_row"])
        grown = LAYOUT["current"] + max(build, declared["job_report_bytes"])
        assert first["ablation"] - first["adapters"] == grown
        grown = rolling.adapter_window(adapters, COUNTS, EXTRAS, ordered_copy=False)
        # Los adaptadores escriben sobre lo que deja la base, que ya incluye la ablación previa.
        assert second["adapters"] == base_second["adapters"] + kept + grown["written"]
        assert second["retained_bytes"] == (
            base_second["retained_bytes"] + kept + grown["compacted"] + 1000
        )


def test_window_increment_is_the_largest_growth_over_what_was_kept(declared):
    result = estimate(declared)
    increment = rolling.window_increment(result)
    rows = result["all_regenerated"]["windows"]
    growth = [
        max(row[p] for p in rolling.PHASES) - before
        for row, before in zip(rows, [0] + [r["retained_bytes"] for r in rows[:-1]], strict=True)
    ]
    assert increment == dict(bytes=max(growth), window=rows[growth.index(max(growth))]["window"])


def test_the_storage_command_adds_the_rolling_walk_of_campaign_a(tmp_path, capsys):
    """La orden `storage` con `--schedule` recorre las etapas declaradas por ventana."""
    import json

    from mars_titan.training import storage_budget as budget
    from mars_titan.training.campaign_plan import load_campaign

    campaign = load_campaign("configs/baselines/historical-masked-campaign-a.json")
    resolved = campaign["comparison_config"]["resolved_scopes"]
    counts = {s: dict.fromkeys(resolved[s]["windows"], COUNTS) for s in campaign["scopes"]}
    names = sorted({w for s in campaign["scopes"] for w in resolved[s]["windows"]})
    schedule = dict(
        schema_version=1,
        campaign=campaign["sha256"],
        windows=[
            dict(window=w, scopes={s: w for s in campaign["scopes"] if w in counts[s]})
            for w in names
        ],
    )
    files = dict(counts=counts, schedule=schedule, extras=dict(EXTRAS, tape_bytes=17))
    files["row-bytes"] = budget.row_bytes(measured())
    for name, value in files.items():
        (tmp_path / f"{name}.json").write_text(json.dumps(value))
    arguments = ["--campaign", "configs/baselines/historical-masked-campaign-a.json"]
    arguments += ["--storage", str(DECLARATION), "--counts", str(tmp_path / "counts.json")]
    arguments += ["--row-bytes", str(tmp_path / "row-bytes.json")]
    arguments += ["--extras", str(tmp_path / "extras.json")]
    arguments += ["--schedule", str(tmp_path / "schedule.json"), "--adapter-blocks"]
    for stage, path in (
        ("ablation", "configs/evaluation/historical-masked-ablation-stage-a.json"),
        ("adapter", "configs/posttraining/historical-masked-adapter-stage-a.json"),
        ("rl", "configs/simulation/historical-masked-rl-stage-a.json"),
    ):
        arguments += [f"--{stage}-stage", path]
    arguments += ["--output", str(tmp_path / "estimate.json")]
    assert budget.main(arguments) == 0
    result = json.loads((tmp_path / "estimate.json").read_text())["rolling"]
    assert result["adapter_corpus"] == "blocks"
    for scenario in rolling.SCENARIOS:
        walk = result[scenario]
        assert [row["window"] for row in walk["windows"]] == names
        assert walk["peak_bytes"] >= max(row["retained_bytes"] for row in walk["windows"])
        assert result["increment"][scenario]["bytes"] > 0
    # Las cintas de las políticas y las entradas que leen se conservan en las dos.
    assert (
        result["none_regenerated"]["retained_bytes"] > result["all_regenerated"]["retained_bytes"]
    )
    printed = json.loads(capsys.readouterr().out.split("\n}\n", 1)[1])
    assert printed["all_regenerated"]["peak_at"] == result["all_regenerated"]["peak_at"]


def test_an_input_read_last_in_its_own_window_is_released_with_it(declared):
    result = estimate(declared, policy_reads={JOBS[1]["id"]: 1})["all_regenerated"]
    assert result["retained_bytes"] == 2 * kept_per_fit(declared)
