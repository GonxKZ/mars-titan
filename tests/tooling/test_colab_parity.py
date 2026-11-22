"""La paridad entre equipos compara las mismas filas y no cuenta las semillas como réplicas."""

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("colab_parity", ROOT / "scripts" / "colab_parity.py")
parity = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(parity)

START = 1_357_160_700_000_000
DAY = 86_400_000_000
SESSIONS = {"validation": 40, "calibration": 20, "evaluation": 60}


def write_run(folder, *, seed, signal, shift=0.0, noise_seed=0, assets=40, drop=None):
    """Escribir los tres tramos de una ejecución sintética con el orden de filas barajado."""
    folder.mkdir(parents=True)
    labels = np.random.default_rng(1000 + seed)
    extra = np.random.default_rng(5000 + noise_seed)
    offset = 0
    for split, sessions in SESSIONS.items():
        rows = []
        for session in range(sessions):
            target = labels.standard_normal(assets)
            other = labels.standard_normal(assets)
            prediction = signal * target + other + shift * extra.standard_normal(assets)
            for asset in range(assets):
                if drop is not None and (split, session, asset) == drop:
                    continue
                instant = START + (offset + session) * DAY
                rows.append(
                    dict(
                        sample_id=f"US/A{asset:03d}/{instant}",
                        market="US",
                        prediction_at=instant,
                        target=float(target[asset]),
                        prediction=float(prediction[asset]),
                    )
                )
        offset += sessions
        order = np.random.default_rng(seed + offset).permutation(len(rows))
        table = pa.Table.from_pylist([rows[index] for index in order]).cast(
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


def run(tmp_path, *arguments):
    output = tmp_path / "paridad.json"
    code = parity.main(
        [
            "--a",
            str(tmp_path / "a"),
            "--b",
            str(tmp_path / "b"),
            "--model",
            "m",
            "--origin",
            "2013",
            "--output",
            str(output),
            *arguments,
        ]
    )
    return code, json.loads(output.read_text(encoding="utf-8"))


def folder(tmp_path, machine, seed):
    return tmp_path / machine / "runs" / "m" / "2013" / f"seed-{seed}"


def test_two_copies_of_a_run_are_equivalent_with_rows_in_another_order(tmp_path, capsys):
    write_run(folder(tmp_path, "a", 42), seed=42, signal=0.3)
    write_run(folder(tmp_path, "b", 42), seed=42, signal=0.3)
    code, document = run(tmp_path, "--seed", "42")
    capsys.readouterr()
    assert code == 0
    summary = document["summary"]
    assert summary["verdict"] == "equivalente" and summary["same_rows_and_targets"] is True
    assert summary["pooled"]["mean_difference"] == 0.0 and summary["pooled"]["sessions"] == 120
    assert document["runs"][0]["splits"]["evaluation"]["prediction_max_abs_difference"] == 0.0


def test_a_missing_row_stops_the_comparison_with_its_own_exit_code(tmp_path, capsys):
    write_run(folder(tmp_path, "a", 42), seed=42, signal=0.3)
    write_run(folder(tmp_path, "b", 42), seed=42, signal=0.3, drop=("evaluation", 3, 7))
    code, document = run(tmp_path, "--seed", "42")
    capsys.readouterr()
    assert code == 1
    assert document["summary"]["verdict"] == "filas o etiquetas distintas"
    assert document["runs"][0]["splits"]["evaluation"]["same_rows"] is False


def test_a_machine_that_orders_worse_is_reported_as_different(tmp_path, capsys):
    for seed in (42, 43):
        write_run(folder(tmp_path, "a", seed), seed=seed, signal=0.6)
        write_run(folder(tmp_path, "b", seed), seed=seed, signal=0.6, shift=2.0, noise_seed=seed)
    code, document = run(tmp_path, "--seed", "42", "--seed", "43")
    capsys.readouterr()
    pooled = document["summary"]["pooled"]
    assert code == 2 and pooled["verdict"] == "distinto"
    assert pooled["mean_difference"] > 0.05 and pooled["interval_90"][0] > 0.002


def test_small_noise_with_few_sessions_is_inconclusive_not_equivalent(tmp_path, capsys):
    # Los dos equipos añaden a la misma base un ruido propio del mismo tamaño.
    write_run(folder(tmp_path, "a", 42), seed=42, signal=0.3, shift=0.2, noise_seed=8)
    write_run(folder(tmp_path, "b", 42), seed=42, signal=0.3, shift=0.2, noise_seed=9)
    code, document = run(tmp_path, "--seed", "42", "--margin", "0.002")
    capsys.readouterr()
    pooled = document["summary"]["pooled"]
    assert code == 2 and pooled["verdict"] == "no concluyente"
    assert pooled["interval_90"][0] < 0 < pooled["interval_90"][1]


def test_seeds_that_share_sessions_do_not_shrink_the_standard_error():
    rng = np.random.default_rng(0)
    series = rng.standard_normal(80) * 0.02
    keys = [f"US|{START + index * DAY}" for index in range(80)]

    def report(values):
        entry = dict(same_rows=True, same_targets=True, keys=keys, difference=list(values))
        return dict(splits={split: entry for split in parity.SPLITS})

    one = parity.summarize([report(series)], 0.002)["splits"]["evaluation"]
    three = parity.summarize([report(series)] * 3, 0.002)["splits"]["evaluation"]
    assert three["standard_error"] == pytest.approx(one["standard_error"], rel=1e-12)
    assert three["mean_difference"] == pytest.approx(one["mean_difference"], rel=1e-12)
    # Tres semillas con ruido independiente sí lo reducen, cerca de la raíz de tres.
    independent = [report(rng.standard_normal(80) * 0.02) for _ in range(3)]
    pooled = parity.summarize(independent, 0.002)["splits"]["evaluation"]
    assert 0.35 < pooled["standard_error"] / one["standard_error"] < 0.85


def test_ranks_average_ties_and_a_constant_prediction_orders_nothing():
    assert parity.average_ranks(np.array([3.0, 1.0, 3.0, 2.0])).tolist() == [3.5, 1.0, 3.5, 2.0]
    target = np.arange(40.0)
    assert parity.rank_correlation(np.zeros(40), target) == 0.0
    assert parity.rank_correlation(target * 2 + 1, target) == pytest.approx(1.0)
    assert parity.rank_correlation(-target, target) == pytest.approx(-1.0)


def test_sessions_below_the_minimum_of_assets_are_not_scored():
    rows = dict(
        market=np.array([1] * 29 + [1] * 30 + [0] * 31),
        markets=["CN", "US"],
        instant=np.array([1] * 29 + [2] * 30 + [2] * 31),
        target=np.arange(90.0),
        prediction=np.arange(90.0),
    )
    assert parity.session_ic(rows) == {"CN|2": pytest.approx(1.0), "US|2": pytest.approx(1.0)}


def test_the_standard_error_centres_each_block_and_matches_the_plain_one_without_dependence():
    rng = np.random.default_rng(3)
    values = np.r_[rng.standard_normal(400) + 5.0, rng.standard_normal(400) - 5.0]
    blocks = ["US"] * 400 + ["CN"] * 400
    plain = math.sqrt(np.var(values[:400]) * 400 + np.var(values[400:]) * 400) / 800
    assert parity.hac_standard_error(values, blocks) == pytest.approx(plain, rel=0.25)
    # Con un solo bloque, el salto de nivel entre mitades contaría como ruido.
    assert parity.hac_standard_error(values, ["US"] * 800) > 3 * plain


def test_the_equivalence_test_needs_the_whole_interval_inside_the_margin():
    assert parity.tost(0.0, 0.0005, 0.002) < 0.001
    assert parity.tost(0.0019, 0.0005, 0.002) > 0.4
    assert parity.verdict(0.0, 0.0005, 0.002)[0] == "equivalente"
    assert parity.verdict(0.004, 0.0005, 0.002)[0] == "distinto"
    assert parity.verdict(0.001, 0.002, 0.002)[0] == "no concluyente"
    assert parity.tost(0.001, 0.0, 0.002) == 0.0 and parity.tost(0.003, 0.0, 0.002) == 1.0
