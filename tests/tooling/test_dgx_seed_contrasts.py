"""Los contrastes con varias semillas promedian por sesión y no esconden semillas que faltan."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "dgx_seed_contrasts", ROOT / "scripts" / "dgx_seed_contrasts.py"
)
contrasts = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(contrasts)

START = 1_357_160_700_000_000
DAY = 86_400_000_000
SESSIONS = {"validation": 30, "calibration": 20, "evaluation": 40}
ASSETS = 40


def write_model(state, model, seed, *, signal, noise_seed, sessions=SESSIONS, year=2013):
    """Emisiones sintéticas: las etiquetas dependen solo de la sesión y el ruido, de la semilla."""
    folder = state / "runs" / model / str(year) / f"seed-{seed}"
    folder.mkdir(parents=True)
    offset = 0
    for split, count in sessions.items():
        rows = []
        for session in range(count):
            labels = np.random.default_rng(10_000 + offset + session).standard_normal(ASSETS)
            noise = np.random.default_rng(noise_seed * 1000 + offset + session).standard_normal(
                ASSETS
            )
            prediction = signal * labels + noise
            instant = START + (offset + session) * DAY
            for asset in range(ASSETS):
                rows.append(
                    dict(
                        sample_id=f"US/A{asset:03d}/{instant}",
                        market="US",
                        prediction_at=instant,
                        target=float(labels[asset]),
                        prediction=float(prediction[asset]),
                    )
                )
        offset += count
        table = pa.Table.from_pylist(rows).cast(
            pa.schema(
                [
                    ("sample_id", pa.string()),
                    ("market", pa.string()),
                    ("prediction_at", pa.timestamp("us", tz="UTC")),
                    ("target", pa.float64()),
                    ("prediction", pa.float64()),
                ]
            )
        )
        pq.write_table(table, folder / f"{split}-predictions.parquet")


def build_state(tmp_path, seeds=(42, 43, 44)):
    state = tmp_path / "state"
    for seed in seeds:
        write_model(state, "base", seed, signal=0.3, noise_seed=seed)
        write_model(state, "mejor", seed, signal=0.6, noise_seed=seed)
        write_model(state, "igual", seed, signal=0.3, noise_seed=seed + 100)
    return state


def test_the_seeds_are_averaged_per_session_before_the_contrast(tmp_path):
    state = build_state(tmp_path)
    report = contrasts.build(state, 2013, [42, 43, 44], {"mejor": "base", "igual": "base"})
    row = report["contrasts"]["mejor"]
    assert row["control"] == "base" and row["seeds"] == [42, 43, 44]
    evaluation = row["evaluation"]
    per_seed = row["per_seed"]["evaluation"]
    # La media del contraste es la media de las diferencias de cada semilla.
    assert evaluation["mean"] == pytest.approx(np.mean(per_seed), rel=1e-12)
    assert evaluation["sessions"] == 40 and evaluation["mean"] > 0.1 and evaluation["t"] > 5
    assert row["signs"]["evaluation"] == "+++"
    # Un modelo con la misma señal que su control no se distingue de él.
    same = report["contrasts"]["igual"]["evaluation"]
    assert abs(same["t"]) < 2.5
    # La decisión junta validación y calibración, sin la evaluación.
    assert row["decision"]["sessions"] == 50
    assert report["models"]["mejor"]["evaluation"]["mean"] == pytest.approx(
        np.mean(report["models"]["mejor"]["per_seed"]["evaluation"]), rel=1e-12
    )


def test_identical_seeds_do_not_shrink_the_standard_error(tmp_path):
    state = tmp_path / "state"
    for seed in (42, 43, 44):
        # La misma semilla de ruido en las tres: tres copias del mismo ajuste.
        write_model(state, "base", seed, signal=0.3, noise_seed=1)
        write_model(state, "mejor", seed, signal=0.6, noise_seed=1)
    three = contrasts.build(state, 2013, [42, 43, 44], {"mejor": "base"})
    one = contrasts.build(state, 2013, [42], {"mejor": "base"})
    left, right = three["contrasts"]["mejor"]["evaluation"], one["contrasts"]["mejor"]["evaluation"]
    assert left["se"] == pytest.approx(right["se"], rel=1e-12)
    assert left["mean"] == pytest.approx(right["mean"], rel=1e-12)


def test_a_model_without_every_seed_is_reported_as_incomplete(tmp_path):
    state = build_state(tmp_path, seeds=(42, 43))
    write_model(state, "base", 44, signal=0.3, noise_seed=44)
    report = contrasts.build(state, 2013, [42, 43, 44], {"mejor": "base", "igual": "base"})
    assert report["contrasts"] == {}
    assert report["incomplete"] == {"igual": [44], "mejor": [44]}
    assert "base" in report["models"]


def test_seeds_that_do_not_share_sessions_are_rejected(tmp_path):
    state = build_state(tmp_path, seeds=(42, 43))
    shorter = dict(SESSIONS, evaluation=39)
    write_model(state, "base", 44, signal=0.3, noise_seed=44, sessions=shorter)
    write_model(state, "mejor", 44, signal=0.6, noise_seed=44, sessions=shorter)
    with pytest.raises(ValueError, match="no comparten sesiones"):
        contrasts.build(state, 2013, [42, 43, 44], {"mejor": "base"})


def test_the_command_writes_the_table_and_the_document(tmp_path, capsys):
    state = build_state(tmp_path)
    declared = tmp_path / "contrasts.json"
    declared.write_text(json.dumps(dict(controls={"mejor": "base", "igual": "base"})))
    output = tmp_path / "out"
    arguments = ["--state", str(state), "--year", "2013", "--contrasts", str(declared)]
    arguments += ["--seed", "42", "--seed", "43", "--seed", "44", "--output", str(output)]
    assert contrasts.main(arguments) == 0
    capsys.readouterr()
    document = json.loads((output / "seed-contrasts.json").read_text(encoding="utf-8"))
    assert document["kind"] == "dgx_seed_contrasts_v1" and document["final_test_opened"] is False
    assert set(document["contrasts"]) == {"mejor", "igual"}
    text = (output / "seed-contrasts.md").read_text(encoding="utf-8")
    assert "| mejor | base | 3 |" in text and "+++" in text
    with pytest.raises(SystemExit):
        contrasts.main([*arguments[:-1], str(output), "--seed", "42"])
