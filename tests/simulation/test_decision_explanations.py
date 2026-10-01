"""Explicar únicamente decisiones confirmadas con integridad y tiempos comprobados."""

import hashlib
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.explain_rl_decisions import explain


def sealed(value):
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return dict(payload=value, sha256=hashlib.sha256(encoded.encode()).hexdigest())


def fixture(tmp_path, *, sensitivity=False):
    directory = tmp_path / "trace"
    directory.mkdir()
    rows = []
    for identifier in (1, 2, 3):
        row = dict(
            decision_id=identifier,
            lane=0,
            world_sha256="a" * 64,
            context_sha256="b" * 64,
            episode=0,
            cursor=identifier,
            optimizer_step=0,
            decision_at=1672531200000000 + identifier,
            outcome_at=1672617600000000 + identifier,
            action=1,
            mode="warmup" if identifier == 1 else "sampled",
            learning_allowed=identifier != 1,
            critic=0.25,
            retrieved_count=0,
            reward=0.0,
            reward_valid=True,
            terminated=False,
            truncated=False,
            costs_delta=0.0,
            costs_present=False,
            memory_masked_action=0,
            memory_probability_l1=0.0,
            memory_action_changed=False,
            memory_sensitivity_present=False,
        )
        row.update({f"probability_{index}": 1 / 6 for index in range(6)})
        row.update({f"memory_masked_probability_{index}": 0.0 for index in range(6)})
        for index in range(4):
            row.update(
                {f"memory_id_{index}": 0, f"similarity_{index}": 0.0, f"matured_at_{index}": 0}
            )
        if sensitivity and identifier == 2:
            row.update(
                mode="greedy",
                learning_allowed=False,
                memory_sensitivity_present=True,
                memory_masked_action=5,
                memory_probability_l1=5 / 3,
                memory_action_changed=True,
            )
            row["memory_masked_probability_5"] = 1.0
        rows.append(row)
    temporary = directory / "pending.parquet"
    table = pa.Table.from_pylist(rows).replace_schema_metadata(
        {"trace_identity_sha256": "c" * 64, "schema_version": "1"}
    )
    pq.write_table(table, temporary)
    digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
    path = directory / f"trace-{digest}.parquet"
    temporary.rename(path)
    record = dict(path=path.name, sha256=digest, bytes=path.stat().st_size, first=1, last=3, rows=3)
    payload = dict(schema_version=1, identity_sha256="c" * 64, cursor=3, shards=[record])
    (directory / "trace-index.json").write_text(json.dumps(sealed(payload)))
    run = tmp_path / "run.json"
    run.write_text(
        json.dumps(dict(trace=dict(path="trace", identity_sha256="c" * 64, confirmed_cursor=2)))
    )
    return run, directory, path


def test_unconfirmed_tail_is_never_explained_and_warmup_is_identified(tmp_path):
    run, _, _ = fixture(tmp_path)
    text = explain(run)
    assert "Decisión 1" in text and "Decisión 2" in text and "Decisión 3" not in text
    assert "calentamiento" in text and "muestreo" in text
    assert "No estiman la probabilidad de rentabilidad" in text
    assert "Sensibilidad no registrada" in text
    assert "Recompensa posterior observada en 2023-01-02" in text


def test_shard_corruption_cannot_be_rendered_as_an_explanation(tmp_path):
    run, _, path = fixture(tmp_path)
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="integridad"):
        explain(run)


def test_recorded_sensitivity_describes_the_frozen_comparison(tmp_path):
    run, _, _ = fixture(tmp_path, sensitivity=True)
    text = explain(run, after=1)
    assert "Al ocultar los recuerdos" in text and "la acción por máximo fue 5" in text
    assert "distancia L1" in text and "sin identificación de causas económicas" in text
    assert "Sensibilidad no registrada" not in text


def test_foreign_identity_and_escaping_paths_are_rejected(tmp_path):
    run, _, _ = fixture(tmp_path)
    report = json.loads(run.read_text())
    report["trace"]["identity_sha256"] = "d" * 64
    run.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="identidad"):
        explain(run)
    report["trace"]["path"] = "../outside"
    run.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="ruta"):
        explain(run)


def test_requested_limit_and_cursor_must_be_bounded(tmp_path):
    run, _, _ = fixture(tmp_path)
    with pytest.raises(ValueError):
        explain(run, limit=201)
    with pytest.raises(ValueError):
        explain(run, after=3)


@pytest.mark.parametrize(
    "defect", ["future_memory", "past_outcome", "foreign_shard", "invalid_reward"]
)
def test_semantic_corruption_is_rejected_even_when_file_hashes_match(tmp_path, defect):
    run, directory, old = fixture(tmp_path)
    rows = pq.read_table(old).to_pylist()
    if defect == "future_memory":
        rows[1].update(retrieved_count=1, memory_id_0=1, matured_at_0=rows[1]["decision_at"] + 1)
    elif defect == "past_outcome":
        rows[1]["outcome_at"] = rows[1]["decision_at"]
    elif defect == "invalid_reward":
        rows[1].update(reward_valid=False, reward=123.0)
    temporary = directory / "altered.parquet"
    table = pa.Table.from_pylist(rows).replace_schema_metadata(
        {
            "trace_identity_sha256": ("d" if defect == "foreign_shard" else "c") * 64,
            "schema_version": "1",
        }
    )
    pq.write_table(table, temporary)
    digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
    path = directory / f"trace-{digest}.parquet"
    temporary.rename(path)
    payload = json.loads((directory / "trace-index.json").read_text())["payload"]
    payload["shards"][0].update(path=path.name, sha256=digest, bytes=path.stat().st_size)
    (directory / "trace-index.json").write_text(json.dumps(sealed(payload)))
    with pytest.raises(ValueError):
        explain(run)
