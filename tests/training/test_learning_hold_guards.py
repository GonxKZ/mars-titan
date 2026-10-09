"""Puntos de entrada que ajustan modelos frente a la protección local del aprendizaje.

Cada caso sustituye por un doble el primer paso posterior a la protección o la propia llamada
de ajuste. El doble registra la llamada y se detiene, así que ninguna prueba ajusta parámetros.
"""

import importlib
import importlib.metadata
import json
import os
import runpy
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mars_titan.training.learning_hold import LearningHoldError

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"


class Reached(Exception):
    """El punto de entrada superó la protección y llegó al paso sustituido."""


class Double:
    def __init__(self):
        self.calls = 0

    def __call__(self, *_args, **_kwargs):
        self.calls += 1
        raise Reached


def _module(name, attribute, monkeypatch):
    module = importlib.import_module(name)
    double = Double()
    monkeypatch.setattr(module, attribute, double)
    return module, double


def _script(name, attribute, monkeypatch):
    main = runpy.run_path(str(SCRIPTS / name))["main"]
    double = Double()
    monkeypatch.setitem(main.__globals__, attribute, double)
    return main, double


def _blocks():
    rng = np.random.default_rng(7)
    return lambda: iter([(rng.normal(size=(4, 3)), rng.normal(size=4))])


def _simple(name, attribute, call):
    def prepare(tmp_path, monkeypatch):
        module, double = _module(name, attribute, monkeypatch)
        return lambda: call(module, tmp_path / "out"), double

    return prepare


def _ridge(tmp_path, monkeypatch):
    ridge, double = _module(
        "mars_titan.models.baselines.ridge", "centered_normal_equations", monkeypatch
    )
    monkeypatch.setattr(ridge, "require_cuda", lambda: None)
    return lambda: ridge.fit_ridge_blocks(_blocks()), double


def _boosting(tmp_path, monkeypatch):
    boosting = importlib.import_module("mars_titan.models.baselines.boosting")
    double = Double()
    monkeypatch.setattr(
        boosting, "HistGradientBoostingRegressor", lambda **_: SimpleNamespace(fit=double)
    )
    return lambda: boosting.fit_boosting_batches(_blocks()), double


def _hmm(tmp_path, monkeypatch):
    double = Double()
    package, hmm = types.ModuleType("hmmlearn"), types.ModuleType("hmmlearn.hmm")
    hmm.GaussianHMM = lambda **_: SimpleNamespace(fit=double)
    monkeypatch.setitem(sys.modules, "hmmlearn", package)
    monkeypatch.setitem(sys.modules, "hmmlearn.hmm", hmm)
    monkeypatch.setattr(importlib.metadata, "version", lambda _name: "0.3.3")
    scenarios = importlib.import_module("mars_titan.simulation.adaptation_scenarios")
    features = [("train", np.random.default_rng(3).normal(size=(8, 3)))]
    return lambda: scenarios.fit_hmm(features), double


def _trainer(tmp_path, monkeypatch):
    from mars_titan.simulation.training import FinancialTrainer, TrainConfig
    from tests.simulation.test_training import environment

    config = TrainConfig(
        total_steps=12,
        batch_size=2,
        rollout_steps=4,
        ppo_epochs=2,
        replay_capacity=16,
        warmup_steps=2,
        target_interval=2,
        checkpoint_steps=4,
    )
    trainer = FinancialTrainer(environment(), "ppo", config, seed=42, device="cpu", diagnostic=True)
    double = Double()
    monkeypatch.setattr(trainer, "advance", double)
    return lambda: trainer.run(tmp_path / "out"), double


def _comparators(tmp_path, monkeypatch):
    campaign, double = _module("mars_titan.simulation.campaign", "TrainConfig", monkeypatch)
    config = json.loads((ROOT / "configs/simulation/comparators.json").read_text())

    def tape(partition, digest):
        return SimpleNamespace(
            partition=partition,
            assets=("AAA",),
            currency="USD",
            identity={"parent_id": "fixture"},
            domain="technical",
            sha256=digest * 64,
        )

    return (
        lambda: campaign.run_campaign(
            tape("train", "a"), [tape("validation", "b")], tmp_path / "out", config, diagnostic=True
        ),
        double,
    )


def _executable(tmp_path):
    binary = tmp_path / "mars-titan-ppo"
    binary.write_text("#!/bin/sh\nexit 0\n")
    binary.chmod(0o700)
    return binary


def _native_ppo_arguments(tmp_path, *, audit=False):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"schema_version": 2 if audit else 1}))
    sources = (
        ["--audit-run", str(tmp_path / "run"), "--audit-tape", str(tmp_path / "audit")]
        if audit
        else ["--train-tape", str(tmp_path / "train"), "--validation-tape", str(tmp_path / "val")]
    )
    return [
        "--config",
        str(config),
        "--output",
        str(tmp_path / "out"),
        "--binary",
        str(_executable(tmp_path)),
        "--diagnostic",
        *sources,
    ]


def _native_ppo(tmp_path, monkeypatch):
    main, double = _script("run_native_ppo.py", "run_child", monkeypatch)
    return lambda: main(_native_ppo_arguments(tmp_path)), double


def _native_benchmark(tmp_path, monkeypatch):
    main, double = _script("benchmark_native_ppo.py", "digest", monkeypatch)
    # El medidor fija variables de hilos y CUDA. Una copia evita que salgan de esta prueba.
    monkeypatch.setattr(os, "environ", dict(os.environ))
    arguments = ["--root", str(ROOT), "--binary", str(_executable(tmp_path))]
    arguments += ["--private", str(tmp_path / "out"), "--output", str(tmp_path / "result.json")]
    monkeypatch.setattr(sys, "argv", ["benchmark_native_ppo.py", *arguments])
    return main, double


def _adaptive_benchmark(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    main, double = _script("benchmark_adaptive_rl.py", "safe_destination", monkeypatch)
    arguments = ["--scenarios", str(tmp_path / "index.json"), "--binary", str(tmp_path / "bin")]
    return lambda: main([*arguments, "--output", str(tmp_path / "out")]), double


def _financial_comparators(tmp_path, monkeypatch):
    main, double = _script("run_financial_comparators.py", "read_manifest", monkeypatch)
    arguments = ["--train-tape", str(tmp_path / "train"), "--validation-tape", str(tmp_path / "v")]
    monkeypatch.setattr(sys, "argv", ["run", *arguments, "--output", str(tmp_path / "out")])
    return main, double


ENTRY_POINTS = {
    "reference_run": _simple(
        "mars_titan.training.reference_run",
        "_options",
        lambda m, out: m.run_reference_case(out.with_name("manifest.json"), out, {}),
    ),
    "reference_search": _simple(
        "mars_titan.training.reference_search",
        "_configuration",
        lambda m, out: m.run_search(out.with_name("c.json"), out.with_name("m.json"), out),
    ),
    "temporal_search": _simple(
        "mars_titan.training.temporal_search",
        "_inputs",
        lambda m, out: m.run_temporal_search(out.with_name("c.json"), out.with_name("v"), out),
    ),
    "reference_campaign": _simple(
        "mars_titan.training.reference_campaign",
        "_configuration",
        lambda m, out: m.run_reference_campaign(
            out.with_name("c.json"), out.with_name("m.json"), out
        ),
    ),
    "tabular_corpus": _simple(
        "mars_titan.training.tabular_corpus",
        "CorpusDataset",
        lambda m, out: m.run_tabular_reference(out.with_name("m.json"), out, kind="boosting"),
    ),
    "external_corpus": _simple(
        "mars_titan.training.external_corpus",
        "CorpusDataset",
        lambda m, out: m.run_external_reference(out.with_name("m.json"), out),
    ),
    "tabular_search": _simple(
        "mars_titan.training.tabular_search",
        "_configuration",
        lambda m, out: m.run_tabular_search(out.with_name("c.json"), out.with_name("m.json"), out),
    ),
    "predictive_run": _simple(
        "mars_titan.training.predictive_run",
        "_options",
        lambda m, out: m.run_predictive_case(out.with_name("o"), out.with_name("p"), out, {}),
    ),
    "predictive_study": _simple(
        "mars_titan.training.predictive_study",
        "_configuration",
        lambda m, out: m.run_predictive_study(
            out.with_name("c.json"), out.with_name("o"), out.with_name("p"), out
        ),
    ),
    "baseline_queue": _simple(
        "mars_titan.training.baseline_queue",
        "safe_destination",
        lambda m, out: m.run_queue(out.with_name("c.json"), out.with_name("r.json"), out),
    ),
    "klpo_queue": _simple(
        "mars_titan.training.klpo_queue",
        "_configuration",
        lambda m, out: m.run_queue(
            out.with_name("c.json"), out.with_name("r.json"), out.with_name("t.json"), out
        ),
    ),
    "real_campaign": _simple(
        "mars_titan.training.real_campaign",
        "prepare_campaign",
        lambda m, out: m.run_campaign(SimpleNamespace(state_dir=out), None),
    ),
    "financial_trainer": _trainer,
    "financial_comparators": _comparators,
    "adaptive_campaign": _simple(
        "mars_titan.simulation.adaptive_campaign",
        "safe_destination",
        lambda m, out: m.run_adaptive_campaign(out.with_name("s.json"), out.with_name("b"), out),
    ),
    "adaptation_scenarios_hmm": _simple(
        "mars_titan.simulation.adaptation_scenarios",
        "cases",
        lambda m, out: m.prepare_adaptation_scenarios({}, out, fit_markov=True),
    ),
    "fit_hmm": _hmm,
    "budget_training": _simple(
        "mars_titan.budget_training",
        "outside_source",
        lambda m, out: m.train_budget_grid(out.with_name("p"), out.with_name("r.json"), out),
    ),
    "gru_probe": _simple(
        "mars_titan.gru_probe",
        "validate_loss",
        lambda m, out: m.run_temporal_probe(
            out.with_name("p"), out.with_name("s"), out, out.with_name("r.json")
        ),
    ),
    "profiling": _simple(
        "mars_titan.profiling",
        "require_cuda",
        lambda m, out: m.profile_case(
            [out.with_name("a.parquet")], out, kind="mlp", batch_size=1, workers=0
        ),
    ),
    "reference_probe": _simple(
        "mars_titan.reference_probe",
        "prepare_probe",
        lambda m, out: m.run_reference_probe(
            out.with_name("p"), out.with_name("s"), out, out.with_name("r.json")
        ),
    ),
    "baseline_campaign": _simple(
        "mars_titan.models.baselines.campaign",
        "_load_config",
        lambda m, out: m.run_campaign(out.with_name("c.json"), out.with_name("p"), out, out),
    ),
    "fit_ridge_blocks": _ridge,
    "fit_boosting_batches": _boosting,
    "fit_external_boosting": _simple(
        "mars_titan.models.baselines.external_boosting",
        "_libraries",
        lambda m, out: m.fit_external_boosting(_blocks(), out, expected_rows=4),
    ),
    "run_native_ppo": _native_ppo,
    "benchmark_native_ppo": _native_benchmark,
    "benchmark_adaptive_rl": _adaptive_benchmark,
    "run_financial_comparators": _financial_comparators,
}


def _reach(call):
    # Sin este paso, la conversión de conftest omitiría un bloqueo indebido en vez de fallar.
    try:
        call()
    except LearningHoldError as error:
        message = f"La protección permitida detuvo el punto de entrada: {error}"
        raise AssertionError(message) from None


@pytest.mark.parametrize("name", ENTRY_POINTS)
def test_blocked_entry_point_stops_before_fitting_and_creates_no_output(
    name, learning_hold, tmp_path, monkeypatch
):
    call, double = ENTRY_POINTS[name](tmp_path, monkeypatch)
    learning_hold(False)
    with pytest.raises(LearningHoldError, match="Bloqueo de aprendizaje vigente"):
        call()
    assert double.calls == 0
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("allowed", [True, None], ids=["permitida", "ausente"])
@pytest.mark.parametrize("name", ENTRY_POINTS)
def test_permitted_entry_point_reaches_the_substituted_fit(
    name, allowed, learning_hold, tmp_path, monkeypatch
):
    call, double = ENTRY_POINTS[name](tmp_path, monkeypatch)
    learning_hold(allowed)
    with pytest.raises(Reached):
        _reach(call)
    assert double.calls == 1


def test_frozen_native_audit_is_not_blocked(learning_hold, tmp_path, monkeypatch):
    main, double = _script("run_native_ppo.py", "run_child", monkeypatch)
    learning_hold(False)
    with pytest.raises(Reached):
        _reach(lambda: main(_native_ppo_arguments(tmp_path, audit=True)))
    assert double.calls == 1


def test_scenario_generation_without_hmm_is_not_blocked(learning_hold, tmp_path, monkeypatch):
    scenarios, double = _module("mars_titan.simulation.adaptation_scenarios", "cases", monkeypatch)
    learning_hold(False)
    with pytest.raises(Reached):
        _reach(lambda: scenarios.prepare_adaptation_scenarios({}, tmp_path / "out"))
    assert double.calls == 1


def test_temporal_search_check_is_not_blocked(learning_hold, tmp_path, monkeypatch):
    search, double = _module("mars_titan.training.temporal_search", "_inputs", monkeypatch)
    learning_hold(False)
    with pytest.raises(Reached):
        _reach(lambda: search.check_temporal_search(tmp_path / "c.json", tmp_path / "v"))
    assert double.calls == 1


def test_target_preparation_runs_under_the_hold(learning_hold, tmp_path):
    from tests.training.test_corpus_targets import function, materialized

    manifest, prepared = materialized(tmp_path)
    learning_hold(False)
    result = function()(manifest, prepared, tmp_path / "supervised")
    assert result["counts"] == {"train": 1, "validation": 1}


def test_corpus_encoding_runs_under_the_hold(learning_hold, tmp_path):
    from tests.data.test_cohort_samples import Encoders
    from tests.data.test_corpus_encoding import encode, prepared_edition

    manifest, clock, macro = prepared_edition(tmp_path)
    learning_hold(False)
    result = encode(
        manifest,
        tmp_path / "encoded",
        macros={"US": macro},
        encoders=Encoders(),
        clocks={"US": clock},
        context=2,
    )
    assert result["samples"] == 5
