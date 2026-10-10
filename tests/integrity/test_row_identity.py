"""Identidad de filas entre brazos y reserva de 2024 sobre predicciones de juguete.

Reutiliza el estudio de `tests/evaluation/test_walk_forward_comparison.py`. Cada prueba
altera las predicciones escritas de un brazo concreto, sin modelos ni pasos de optimizador.
"""

import json
from datetime import datetime

import numpy as np
import pyarrow as pa
import pytest

from mars_titan.integrity import row_identity
from tests.evaluation.test_walk_forward_comparison import Study


def edited(tmp_path, change, scope="US"):
    study = Study(tmp_path / "study", scope=scope)
    study.changes.append(change)
    study.publish()
    return study, row_identity.check_sources(study.config_path, study.sources_path, scope)


def _take(value, index):
    if isinstance(value, pa.Array):
        return value.take(pa.array(index))
    if isinstance(value, np.ndarray):
        return value[index]
    return [value[i] for i in index]


def keep_rows(values, index):
    for key, value in list(values.items()):
        values[key] = _take(value, index)


def only(arm, seed, fold, partition):
    def decorate(change):
        def applied(a, s, f, p, values):
            if (a, s, f, p) == (arm, seed, fold, partition):
                change(values)

        return applied

    return decorate


def failing_segments(result):
    return {name for name, entry in result["shared_rows"].items() if not entry["passed"]}


@pytest.mark.parametrize("scope", ["US", "US+CN"])
def test_a_faithful_study_shares_rows_in_every_segment(tmp_path, scope):
    study = Study(tmp_path / "study", scope=scope)
    result = row_identity.check_sources(study.config_path, study.sources_path, scope)
    assert result["passed"], json.dumps(result, indent=1)[:2000]
    # Dos ventanas con evaluación para los cinco pares brazo-semilla y calibración para los
    # cuatro de cuantiles.
    assert set(result["shared_rows"]) == {
        f"fold-00{w}/{p}" for w in (0, 1) for p in ("calibration", "evaluation")
    }
    assert result["shared_rows"]["fold-000/evaluation"]["files"] == 5
    assert result["shared_rows"]["fold-000/calibration"]["files"] == 4
    assert all(entry["rows_in_2024"] == 0 for entry in result["files"])


def test_a_single_bit_in_one_target_breaks_the_shared_digest(tmp_path):
    @only("titans", 43, "fold-001", "evaluation")
    def flip(values):
        target = np.array(values["target"], dtype=np.float64)
        target[3] = np.nextafter(target[3], np.inf)
        values["target"] = target

    _, result = edited(tmp_path, flip)
    assert not result["passed"]
    assert failing_segments(result) == {"fold-001/evaluation"}
    groups = result["shared_rows"]["fold-001/evaluation"]["groups"]
    assert ["titans/43"] in groups


def test_a_missing_row_in_one_arm_is_detected(tmp_path):
    @only("ridge", 42, "fold-000", "evaluation")
    def drop(values):
        keep_rows(values, list(range(1, len(values["target"]))))

    _, result = edited(tmp_path, drop)
    assert failing_segments(result) == {"fold-000/evaluation"}
    rows = {
        (e["arm"], e["window"], e["partition"]): e["rows"]
        for e in result["files"]
        if e["window"] == "fold-000" and e["partition"] == "evaluation"
    }
    assert rows["ridge", "fold-000", "evaluation"] + 1 == rows["gru", "fold-000", "evaluation"]


def test_a_row_moved_into_2024_is_reported_in_every_arm(tmp_path):
    def move(arm, seed, fold, partition, values):
        if (fold, partition) == ("fold-001", "evaluation"):
            moments = values["prediction_at"].to_pylist()
            moments[0] = datetime.fromisoformat("2024-01-02T21:00:00+00:00")
            values["prediction_at"] = pa.array(moments, pa.timestamp("us", tz="UTC"))

    _, result = edited(tmp_path, move)
    assert not result["passed"]
    moved = [e for e in result["files"] if e["window"] == "fold-001" and not e["passed"]]
    assert len(moved) == 5
    assert all(e["rows_in_2024"] == 1 and e["rows_outside_segment"] == 1 for e in moved)
    # Todos los brazos tienen el mismo defecto, de modo que la huella común sigue igual.
    assert failing_segments(result) == set()


def test_duplicated_rows_and_non_finite_targets_are_counted(tmp_path):
    def duplicate(arm, seed, fold, partition, values):
        if (fold, partition) == ("fold-000", "evaluation"):
            keep_rows(values, [0, *range(len(values["target"]))])
            if arm == "gru" and seed == 42:
                target = np.array(values["target"], dtype=np.float64)
                target[-1] = np.nan
                values["target"] = target

    _, result = edited(tmp_path, duplicate)
    flagged = {
        (e["arm"], e["seed"]): e
        for e in result["files"]
        if e["window"] == "fold-000" and e["partition"] == "evaluation"
    }
    assert all(entry["duplicated_rows"] == 1 for entry in flagged.values())
    assert flagged["gru", 42]["non_finite_targets"] == 1
    assert flagged["ridge", 42]["non_finite_targets"] == 0


def test_the_digest_ignores_row_order_and_chunking():
    table = pa.table(
        dict(
            market=["US", "US", "CN"],
            asset_id=["b", "a", "z"],
            prediction_at=pa.array([2, 1, 1], pa.timestamp("us", tz="UTC")),
            target=[0.5, -0.25, 1.0],
        )
    )
    shuffled = pa.concat_tables([table.slice(2), table.slice(0, 2)])
    assert row_identity.row_digest(table)[0] == row_identity.row_digest(shuffled)[0]
    changed = table.set_column(3, "target", pa.array([0.5, -0.25, np.nextafter(1.0, 2.0)]))
    assert row_identity.row_digest(table)[0] != row_identity.row_digest(changed)[0]
    renamed = table.set_column(1, "asset_id", pa.array(["b", "a", "zz"]))
    assert row_identity.row_digest(table)[0] != row_identity.row_digest(renamed)[0]


def test_a_changed_prediction_file_is_rejected_by_its_digest(tmp_path):
    study = Study(tmp_path / "study", scope="US")
    path = study.sources_path.parent / "gru" / "42" / "fold-000" / "evaluation-predictions.parquet"
    path.write_bytes(path.read_bytes() + b"\0")
    with pytest.raises(ValueError, match="huella"):
        row_identity.check_sources(study.config_path, study.sources_path, "US")


def test_the_cli_writes_the_report_and_signals_failure(tmp_path):
    @only("gru", 43, "fold-000", "calibration")
    def drop(values):
        keep_rows(values, list(range(2, len(values["target"]))))

    study, _ = edited(tmp_path, drop)
    target = tmp_path / "rows.json"
    arguments = [str(study.config_path), str(study.sources_path), "US", "--output", str(target)]
    assert row_identity.main(arguments) == 1
    written = json.loads(target.read_text())
    assert written["kind"] == row_identity.KIND and written["failures"] == 1
