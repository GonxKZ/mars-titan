"""Identidades persistentes y conciliación de aliases acreditados por el resumen."""

import copy
import json
import sqlite3

import pytest

from mars_titan.observatory.collector import Collector, digest, public_run

SOURCE = dict(id="campaign", path="campaign", kind="archive", domain="real")
RELATIVE = "parents/gru/runs/seed-42/real/mae/run.json"
TASK = "gru/seed-42/real/mae"


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def receipt(**changes):
    return dict(
        status="completed",
        model="gru",
        identity={"case": {"kind": "gru", "seed": 42}},
        updated_at_utc="2026-01-01T00:00:00Z",
        **changes,
    )


def summary(task=TASK):
    return dict(status="completed", runs={task: {"path": RELATIVE}})


def legacy_cache(tmp_path, *, extra_attempt=False, canonical=True):
    """Crear las dos tablas anteriores, sin asociaciones añadidas por el arreglo."""
    folder = tmp_path / "campaign"
    path = folder / RELATIVE
    report = receipt()
    previous = dict(report, status="running")
    current_summary = summary()
    dump(path, report)
    dump(folder / "summary.json", current_summary)
    cache = tmp_path / "cache.sqlite"
    rows = []
    for task, raw in (({}, previous), ({"id": TASK}, report)):
        if task and not canonical:
            continue
        row = public_run(SOURCE, task, raw, {}, RELATIVE, path, None, False)
        rows.append(row)
    if extra_attempt:
        rows.append(dict(rows[0], attempt_id="attempt-0002"))
    with sqlite3.connect(cache) as db:
        db.execute("CREATE TABLE sources (path TEXT PRIMARY KEY, signature TEXT, body TEXT)")
        db.execute("CREATE TABLE runs (key TEXT PRIMARY KEY, body TEXT)")
        for candidate, raw in ((path, report), (folder / "summary.json", current_summary)):
            db.execute(
                "INSERT INTO sources VALUES (?, ?, ?)", (str(candidate), "old", json.dumps(raw))
            )
        for row in rows:
            db.execute(
                "INSERT INTO runs VALUES (?, ?)",
                (row["run_id"] + ":" + row["attempt_id"], json.dumps(row)),
            )
    return cache, rows


def collect(collector):
    return collector.collect([SOURCE])


def database_rows(db):
    return {
        name: list(db.execute(f"SELECT * FROM {name} ORDER BY 1, 2"))
        for (name,) in db.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }


def test_discovery_summary_and_task_rename_keep_one_identity(tmp_path):
    dump(tmp_path / "campaign" / RELATIVE, receipt())
    with Collector(tmp_path, tmp_path / "cache.sqlite") as collector:
        first = collect(collector)["runs"][0]
        for task in (TASK, "renamed-task", "renamed-again"):
            dump(tmp_path / "campaign/summary.json", summary(task))
            observed = collect(collector)["runs"]
            assert len(observed) == 1
            assert observed[0]["run_id"] == first["run_id"]
        (tmp_path / "campaign/summary.json").unlink()
        assert collect(collector)["runs"] == observed
        assert collector.db.execute("SELECT count(*) FROM run_origins").fetchone()[0] == 1


def test_attempts_share_the_first_observed_identity_without_losing_attempts(tmp_path):
    paths = [f"runs/gru/attempt-{index:04d}/run.json" for index in (1, 2)]
    dump(tmp_path / "campaign" / paths[0], receipt(attempt_id="attempt-0001"))
    with Collector(tmp_path, tmp_path / "cache.sqlite") as collector:
        first = collect(collector)["runs"][0]
        dump(tmp_path / "campaign" / paths[1], receipt(attempt_id="attempt-0002"))
        dump(
            tmp_path / "campaign/summary.json",
            {
                "runs": [{"id": TASK, "attempts": [{"path": path} for path in paths]}],
            },
        )
        result = collect(collector)
        assert len(result["runs"]) == 2
        assert {row["run_id"] for row in result["runs"]} == {first["run_id"]}
        assert {row["attempt_id"] for row in result["runs"]} == {"attempt-0001", "attempt-0002"}
        assert result["campaigns"][0]["registered_runs"] == 1


def test_summary_identifies_a_provisional_attempt_from_its_exact_path(tmp_path):
    paths = [f"runs/gru/attempt-{index:04d}/run.json" for index in (1, 2)]
    dump(tmp_path / "campaign" / paths[0], dict(receipt(), status="failed"))
    with Collector(tmp_path, tmp_path / "cache.sqlite") as collector:
        first = collect(collector)["runs"][0]
        assert first["attempt_id"] == "legacy"
        dump(tmp_path / "campaign" / paths[1], receipt())
        dump(
            tmp_path / "campaign/summary.json",
            {"runs": [{"id": TASK, "attempts": [{"path": path} for path in paths]}]},
        )
        result = collect(collector)
        assert len(result["runs"]) == 2
        assert {row["run_id"] for row in result["runs"]} == {first["run_id"]}
        assert {row["attempt_id"] for row in result["runs"]} == {"attempt-0001", "attempt-0002"}
        assert result["campaigns"][0]["counts"] == {"completed": 1, "not_started": 0}
        (tmp_path / "campaign/summary.json").unlink()
        assert collect(collector)["runs"] == result["runs"]


def test_two_table_cache_restores_the_provisional_attempt_before_reading_the_summary(tmp_path):
    cache, previous = legacy_cache(tmp_path, canonical=False)
    first = tmp_path / "campaign" / RELATIVE
    raw = dict(receipt(), status="running")
    dump(first, raw)
    with sqlite3.connect(cache) as db:
        db.execute("UPDATE sources SET body=? WHERE path=?", (json.dumps(raw), str(first)))
        db.execute("DELETE FROM sources WHERE path=?", (str(tmp_path / "campaign/summary.json"),))
    second = "runs/second/run.json"
    dump(tmp_path / "campaign" / second, receipt())
    dump(
        tmp_path / "campaign/summary.json",
        {"runs": [{"id": TASK, "attempts": [{"path": RELATIVE}, {"path": second}]}]},
    )
    with Collector(tmp_path, cache) as collector:
        result = collect(collector)
    assert len(result["runs"]) == 2
    assert {row["run_id"] for row in result["runs"]} == {previous[0]["run_id"]}
    assert {row["attempt_id"] for row in result["runs"]} == {"attempt-0001", "attempt-0002"}
    assert result["campaigns"][0]["counts"] == {"completed": 1, "not_started": 0}


def test_explicit_new_attempt_does_not_replace_legacy_without_a_summary_link(tmp_path):
    path = tmp_path / "campaign" / RELATIVE
    dump(path, receipt())
    with Collector(tmp_path, tmp_path / "cache.sqlite") as collector:
        collect(collector)
        dump(path, receipt(attempt_id="new-attempt"))
        result = collect(collector)["runs"]
    assert {row["attempt_id"] for row in result} == {"legacy", "new-attempt"}


def test_a_task_id_equal_to_another_receipt_directory_is_not_an_alias(tmp_path):
    for path in ("runs/one/run.json", "runs/two/run.json"):
        dump(tmp_path / "campaign" / path, receipt())
    dump(
        tmp_path / "campaign/summary.json",
        {"runs": {"runs/two": {"path": "runs/one/run.json"}}},
    )
    with Collector(tmp_path, tmp_path / "cache.sqlite") as collector:
        before = database_rows(collector.db)
        with pytest.raises(ValueError, match="identidad|origen"):
            collect(collector)
        assert database_rows(collector.db) == before


def test_two_receipts_cannot_declare_the_same_explicit_attempt(tmp_path):
    paths = ["runs/gru/attempt-0001/run.json", "runs/gru/attempt-0002/run.json"]
    for path in paths:
        dump(tmp_path / "campaign" / path, receipt(attempt_id="same-attempt"))
    dump(
        tmp_path / "campaign/summary.json",
        {"runs": [{"id": TASK, "attempts": [{"path": path} for path in paths]}]},
    )
    with Collector(tmp_path, tmp_path / "cache.sqlite") as collector:
        before = database_rows(collector.db)
        with pytest.raises(ValueError, match="intento"):
            collect(collector)
        assert database_rows(collector.db) == before


def test_explicit_attempts_reusing_one_receipt_path_keep_both_records(tmp_path):
    path = tmp_path / "campaign" / RELATIVE
    dump(path, receipt(attempt_id="first"))
    with Collector(tmp_path, tmp_path / "cache.sqlite") as collector:
        first = collect(collector)["runs"][0]
        dump(path, receipt(attempt_id="second"))
        observed = collect(collector)["runs"]
    assert len(observed) == 2
    assert {row["attempt_id"] for row in observed} == {"first", "second"}
    assert {row["run_id"] for row in observed} == {first["run_id"]}


def test_a_new_route_cannot_overwrite_an_attempt_previously_seen_in_another_route(tmp_path):
    path = tmp_path / "campaign" / RELATIVE
    dump(tmp_path / "campaign/summary.json", summary())
    dump(path, receipt(attempt_id="first"))
    with Collector(tmp_path, tmp_path / "cache.sqlite") as collector:
        collect(collector)
        dump(path, receipt(attempt_id="second"))
        collect(collector)
        before = database_rows(collector.db)
        other = "runs/new-path/run.json"
        dump(tmp_path / "campaign" / other, receipt(attempt_id="first"))
        dump(
            tmp_path / "campaign/summary.json",
            {"runs": [{"id": TASK, "attempts": [{"path": RELATIVE}, {"path": other}]}]},
        )
        with pytest.raises(ValueError, match="intento existente"):
            collect(collector)
        assert database_rows(collector.db) == before


@pytest.mark.parametrize("attempt", [None, "attempt-0042"])
def test_legacy_cache_with_only_the_confirmed_id_restores_its_attempt(tmp_path, attempt):
    cache, _ = legacy_cache(tmp_path)
    path = tmp_path / "campaign" / RELATIVE
    report = receipt() if attempt is None else receipt(attempt_id=attempt)
    dump(path, report)
    expected = public_run(SOURCE, {"id": TASK}, report, {}, RELATIVE, path, None, False)
    with sqlite3.connect(cache) as db:
        db.execute("DELETE FROM runs")
        db.execute(
            "INSERT INTO runs VALUES (?, ?)",
            (expected["run_id"] + ":" + expected["attempt_id"], json.dumps(expected)),
        )
        db.execute("UPDATE sources SET body=? WHERE path=?", (json.dumps(report), str(path)))
    with Collector(tmp_path, cache) as collector:
        assert collect(collector)["runs"] == [expected]
        path.unlink()
        (tmp_path / "campaign/summary.json").unlink()
        assert collect(collector)["runs"] == [expected]


def test_legacy_cached_path_blocks_an_ambiguous_alias_even_if_its_file_disappeared(tmp_path):
    cache, _ = legacy_cache(tmp_path, canonical=False)
    other = "runs/two/run.json"
    report = receipt()
    path = tmp_path / "campaign" / other
    dump(path, report)
    row = public_run(SOURCE, {}, report, {}, other, path, None, False)
    current_summary = summary("runs/two")
    dump(tmp_path / "campaign/summary.json", current_summary)
    with sqlite3.connect(cache) as db:
        db.execute("INSERT INTO sources VALUES (?, ?, ?)", (str(path), "old", json.dumps(report)))
        db.execute("INSERT INTO runs VALUES (?, ?)", (row["run_id"] + ":legacy", json.dumps(row)))
        db.execute(
            "UPDATE sources SET body=? WHERE path=?",
            (json.dumps(current_summary), str(tmp_path / "campaign/summary.json")),
        )
    path.unlink()
    with Collector(tmp_path, cache) as collector:
        before = database_rows(collector.db)
        with pytest.raises(ValueError, match="identidad|origen"):
            collect(collector)
        assert database_rows(collector.db) == before


def test_an_existing_canonical_key_must_belong_to_the_same_campaign(tmp_path):
    cache, previous = legacy_cache(tmp_path)
    invalid = previous[1]
    invalid["metadata"]["campaign"] = "foreign"
    with sqlite3.connect(cache) as db:
        db.execute(
            "UPDATE runs SET body=? WHERE key=?",
            (json.dumps(invalid), invalid["run_id"] + ":legacy"),
        )
    with Collector(tmp_path, cache) as collector:
        before = database_rows(collector.db)
        with pytest.raises(ValueError, match="campaña|identidad"):
            collect(collector)
        assert database_rows(collector.db) == before


def test_legacy_cache_keeps_confirmed_identity_and_distinct_attempts(tmp_path):
    cache, previous = legacy_cache(tmp_path, extra_attempt=True)
    confirmed = previous[1]["run_id"]
    with Collector(tmp_path, cache) as collector:
        result = collect(collector)
        assert len(result["runs"]) == 2
        assert {row["run_id"] for row in result["runs"]} == {confirmed}
        assert {row["attempt_id"] for row in result["runs"]} == {"legacy", "attempt-0002"}
        assert (
            next(row for row in result["runs"] if row["attempt_id"] == "legacy")["status"]
            == "completed"
        )
        assert (
            next(row for row in result["runs"] if row["attempt_id"] == "attempt-0002")["status"]
            == "running"
        )
        before = database_rows(collector.db)
        assert collect(collector)["runs"] == result["runs"]
        assert database_rows(collector.db) == before


def test_cached_summary_preserves_the_proof_when_task_is_renamed_before_migration(tmp_path):
    cache, previous = legacy_cache(tmp_path)
    dump(tmp_path / "campaign/summary.json", summary("new-task"))
    with Collector(tmp_path, cache) as collector:
        result = collect(collector)["runs"]
    assert len(result) == 1
    assert result[0]["run_id"] == previous[1]["run_id"]


def test_legacy_history_without_a_replacement_is_not_dropped(tmp_path):
    cache, previous = legacy_cache(tmp_path, canonical=False)
    (tmp_path / "campaign" / RELATIVE).unlink()
    (tmp_path / "campaign/summary.json").unlink()
    with Collector(tmp_path, cache) as collector:
        assert collect(collector)["runs"] == previous


def test_same_model_seed_and_source_hash_do_not_prove_an_alias(tmp_path):
    cache, previous = legacy_cache(tmp_path)
    unrelated = copy.deepcopy(previous[0])
    unrelated["run_id"] = SOURCE["id"] + "-" + digest("unrelated-path")[:20]
    with sqlite3.connect(cache) as db:
        db.execute(
            "INSERT INTO runs VALUES (?, ?)",
            (unrelated["run_id"] + ":legacy", json.dumps(unrelated)),
        )
    with Collector(tmp_path, cache) as collector:
        result = collect(collector)["runs"]
    assert len(result) == 2
    assert unrelated in result


def test_contradictory_existing_origins_roll_back_sources_runs_and_origins(tmp_path):
    paths = ["runs/one/run.json", "runs/two/run.json"]
    for path in paths:
        dump(tmp_path / "campaign" / path, receipt())
    with Collector(tmp_path, tmp_path / "cache.sqlite") as collector:
        collect(collector)
        before = database_rows(collector.db)
        dump(
            tmp_path / "campaign/summary.json",
            {"runs": [{"id": "claimed", "attempts": [{"path": path} for path in paths]}]},
        )
        with pytest.raises(ValueError, match="identidad|asociaci|origen"):
            collect(collector)
        assert database_rows(collector.db) == before


def test_two_tasks_cannot_claim_one_receipt_in_the_same_summary(tmp_path):
    dump(tmp_path / "campaign" / RELATIVE, receipt())
    dump(
        tmp_path / "campaign/summary.json",
        {"runs": {"one": {"path": RELATIVE}, "two": {"path": RELATIVE}}},
    )
    with Collector(tmp_path, tmp_path / "cache.sqlite") as collector:
        with pytest.raises(ValueError, match="identidad|asociaci|recibo"):
            collect(collector)


def test_identity_bindings_are_bounded_even_for_receipts_not_yet_created(tmp_path):
    dump(
        tmp_path / "campaign/summary.json",
        {"runs": {str(index): {"path": f"runs/{index}/run.json"} for index in range(3)}},
    )
    with Collector(tmp_path, tmp_path / "cache.sqlite", max_files=2) as collector:
        with pytest.raises(ValueError):
            collect(collector)
        assert collector.db.execute("SELECT count(*) FROM runs").fetchone()[0] == 0
