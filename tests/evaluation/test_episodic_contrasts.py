"""WP5: los contrastes de cada órgano con su padre y su placebo sobre una emisión sintética."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from mars_titan.evaluation import episodic_contrasts as ec
from mars_titan.evaluation import organ_aggregation as og
from mars_titan.evaluation import posttraining_pilots as pilots
from mars_titan.memory import episodic_organs as eo

HERE = Path(__file__).resolve()
_spec = importlib.util.spec_from_file_location("wp5_organs", HERE.parent / "test_memory_organs.py")
organs_tests = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(organs_tests)
ORIGIN, SEED = organs_tests.ORIGIN, organs_tests.SEED


@pytest.fixture(scope="module")
def emitted(tmp_path_factory):
    root = tmp_path_factory.mktemp("wp5")
    data = organs_tests.campaign(root)
    keys = root / "claves"
    organs_tests.vk.build(data["manifest"], keys / data["view"], log=organs_tests.quiet)
    document = organs_tests.declaration(episodic_organs=organs_tests.episodic_section())
    output = root / "salida"
    organs_tests.emit(data, keys, output, document=document)
    return dict(root=root, output=output, document=document)


def _by_hand(output, model, variant, against):
    table = pq.read_table(output / "studies" / f"{ORIGIN}--seed-{SEED}.parquet").to_pylist()

    def values(name):
        return {
            (r["market"], r["instant"]): r["value"]
            for r in table
            if r["model"] == model and r["split"] == "evaluation" and r["variant"] == name
        }

    left, right = values(variant), values(against)
    common = sorted(set(left) & set(right))
    return float(np.mean([left[k] - right[k] for k in common])), len(common)


def test_each_organ_is_read_against_its_parent_and_its_placebo(emitted):
    summary = ec.compute(emitted["output"], ORIGIN, [SEED], registered=None)
    assert summary["status"] == "completed" and summary["seeds"] == [SEED]
    assert set(summary["results"]) == {"a", "b"}
    for model, arms in summary["results"].items():
        assert list(arms) == list(eo.EPISODIC_ARMS)
        for arm, result in arms.items():
            entry = result["seeds"][SEED]
            for name, variant, against in (
                ("vs_parent", arm, "parent"),
                ("placebo_vs_parent", f"{arm}_placebo", "parent"),
                ("vs_placebo", arm, f"{arm}_placebo"),
            ):
                mean, sessions = _by_hand(emitted["output"], model, variant, against)
                assert entry[name]["mean"] == pytest.approx(mean, abs=1e-12)
                assert entry[name]["sessions"] == sessions
                # Con una sola semilla, la media de semillas es la propia semilla.
                assert result["mean"][name]["mean"] == pytest.approx(mean, abs=1e-12)
            assert entry["selection"]["selected_on"] == "calibration"
            expected = ec.reading(entry)
            assert entry["reading"] == expected
            assert expected["own_signal"] == (
                ec._wins(entry["vs_parent"], ec.MIN_T) and ec._wins(entry["vs_placebo"], ec.MIN_T)
            )
    assert summary["results"]["a"]["dense_bank"]["seeds"][SEED]["uncertainty"]["evaluation"]


def test_the_reading_flags_a_placebo_that_beats_its_parent():
    def contrast(mean, t):
        return dict(mean=mean, t=t)

    calm = dict(
        vs_parent=contrast(0.01, 3.0),
        placebo_vs_parent=contrast(-0.002, -1.0),
        vs_placebo=contrast(0.012, 2.5),
    )
    assert ec.reading(calm) == dict(
        placebo_fails=True, own_signal=True, text="trae señal propia: gana al padre y a su placebo"
    )
    alarm = dict(calm, placebo_vs_parent=contrast(0.004, 2.1))
    assert not ec.reading(alarm)["placebo_fails"] and "alarma" in ec.reading(alarm)["text"]
    weak = dict(calm, vs_placebo=contrast(0.003, 1.9))
    assert ec.reading(weak)["text"] == "sin señal propia demostrada"


def test_the_seed_mean_uses_the_sessions_of_every_seed():
    left = {("US", 1): 0.1, ("US", 2): 0.3}
    right = {("US", 2): 0.5, ("US", 3): 0.0}
    assert ec.seed_mean([left, right]) == {("US", 2): pytest.approx(0.4)}


def test_the_order_publishes_once_and_refuses_another_declaration(emitted, tmp_path, capsys):
    arguments = ["medir", "--organs", str(emitted["output"]), "--origin", str(ORIGIN)]
    arguments += ["--seed", str(SEED), "--output", str(tmp_path / "wp5")]
    assert ec.main(arguments) == 1
    assert "no es la registrada para WP5" in capsys.readouterr().err
    summary = ec.measure(emitted["output"], ORIGIN, [SEED], tmp_path / "wp5", registered=None)
    folder = tmp_path / "wp5" / str(ORIGIN)
    assert json.loads((folder / "resumen.json").read_text())["results"] == json.loads(
        json.dumps(summary["results"])
    )
    assert "órgano − placebo" in (folder / "resumen.md").read_text(encoding="utf-8")
    again = ec.measure(emitted["output"], ORIGIN, [SEED], tmp_path / "wp5", registered=None)
    assert again["declaration_sha256"] == summary["declaration_sha256"]
    missing = ["medir", "--organs", str(tmp_path / "nada"), "--origin", str(ORIGIN)]
    missing += ["--seed", str(SEED), "--output", str(tmp_path / "otra")]
    assert ec.main(missing) == og.MISSING_EXIT


def test_the_registered_declaration_is_the_wp5_emission():
    path = HERE.parents[2] / "configs" / "experiments" / "posttraining-memory-organs-wp5-v10.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    digest = __import__("hashlib").sha256(pilots._canonical(document).encode()).hexdigest()
    assert digest == ec.REGISTERED_SHA256
