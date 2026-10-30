"""Recorrido cronológico con R1, U1 y U2 de #786, sin pasos de optimizador."""

import pytest
import torch

from mars_titan.models.titans.config import GateBias
from mars_titan.models.titans.differential_attention import DIFFERENTIAL_ATTENTION
from mars_titan.models.titans.financial import FinancialConfig, FinancialPredictor
from mars_titan.models.titans.huber_memory import HuberMemory
from mars_titan.models.titans.scale_invariant_time import ScaleInvariantTime
from mars_titan.training.checkpoints import load_training_state
from tests.training.test_financial_run import StopAtStep, corpus, named_records, trainer

NEUTRAL = {
    "huber": dict(memory_huber=HuberMemory(100.0)),
    "differential": dict(differential_attention=DIFFERENTIAL_ATTENTION),
    "time_scales": dict(memory_time_scales=ScaleInvariantTime(0.0)),
}
ACTIVE = dict(
    memory_huber=HuberMemory(0.3),
    memory_time_scales=ScaleInvariantTime(6.0),
    differential_attention=DIFFERENTIAL_ATTENTION,
)
LAMBDAS = [0.5, -0.25, 0.1, 0.3]


@pytest.fixture(autouse=True)
def explicit_fastpath():
    previous = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    yield
    torch.backends.mha.set_fastpath_enabled(previous)


@pytest.fixture(scope="module")
def shared(tmp_path_factory):
    return corpus(tmp_path_factory.mktemp("core-variants"))[1]


def predictor(streams, *, lambdas=None, **changes):
    specification = streams["train"].specification()
    config = FinancialConfig(
        specification, variant="mac_online", hidden_size=32, gate_bias=GateBias(), **changes
    )
    model = FinancialPredictor(config, dtype=torch.float64)
    if lambdas is not None:
        with torch.no_grad():
            model.mac.differential.weight.copy_(torch.tensor(lambdas, dtype=torch.float64))
        model._seal_parameters()
    return model


def assert_same_records(records, reference, *, exact=True):
    assert len(records) == len(reference) > 2
    for left, right in zip(records, reference, strict=True):
        assert left.keys() == right.keys()
        for name, value in left.items():
            if value is None or right[name] is None:
                assert value is right[name], name
            elif exact:
                torch.testing.assert_close(value, right[name], rtol=0, atol=0)
            else:
                torch.testing.assert_close(value, right[name], rtol=1e-10, atol=1e-14)


@pytest.mark.parametrize("changes", NEUTRAL.values(), ids=NEUTRAL)
def test_neutral_settings_reproduce_the_reference_run(shared, tmp_path, changes):
    # El recorte global usa la norma de todos los gradientes, también los de los parámetros
    # nuevos. Con una cota que no se alcanza se comparan los gradientes antes de recortar.
    unclipped = dict(max_grad_norm=1e6)
    control = trainer(shared, tmp_path / "control", model=predictor(shared), **unclipped)
    control.run()
    model = predictor(shared, **changes)
    engine = trainer(shared, tmp_path / "variant", model=model, **unclipped)
    assert engine.run()["status"] == "completed"
    assert engine.audit == control.audit
    records = named_records(engine)
    reference = named_records(control)
    added = [
        {name: record.pop(name) for name in set(record) - set(reference[0])} for record in records
    ]
    assert_same_records(records, reference)
    lambdas = [record.pop("mac.differential.weight", None) for record in added]
    if "differential_attention" in changes:
        assert any(value is not None and torch.count_nonzero(value) for value in lambdas)
    for record in added:
        assert all(value is None or not torch.count_nonzero(value) for value in record.values())


def test_active_variants_resume_after_interruption_like_the_continuous_run(shared, tmp_path):
    options = dict(epochs=2, fixed=True, checkpoint_updates=2)

    def engine(name):
        model = predictor(shared, lambdas=LAMBDAS, **ACTIVE)
        return trainer(shared, tmp_path / name, model=model, **options)

    continuous = engine("continuous")
    expected = continuous.run()
    first = engine("interrupted")
    stop = StopAtStep(15)
    stop.trainer = first
    assert first.run(stop=stop)["status"] == "paused"
    state = load_training_state(
        tmp_path / "interrupted/checkpoints", expected_identity=first.identity
    )
    assert state["run"] is not None
    second = engine("interrupted")
    resumed = second.run(resume=True)
    assert resumed["status"] == "completed"
    assert first.audit + second.audit == continuous.audit
    assert_same_records(named_records(first) + named_records(second), named_records(continuous))
    assert resumed["history"] == expected["history"]


def test_replay_by_flow_blocks_matches_the_whole_block(shared, tmp_path):
    records = []
    for name, rows in (("whole", None), ("blocks", 1)):
        model = predictor(shared, lambdas=LAMBDAS, **ACTIVE)
        engine = trainer(shared, tmp_path / name, model=model, accumulation_rows=rows)
        engine.run()
        records.append(named_records(engine))
    assert_same_records(*records, exact=False)
    whole = records[0]
    for name in (
        "mac.memory.huber_bias",
        "mac.differential.query",
        "mac.memory.alpha_projection.bias",
    ):
        assert any(
            record[name] is not None and torch.count_nonzero(record[name]) for record in whole
        )


def test_the_code_identity_of_the_run_names_the_new_modules(shared, tmp_path):
    engine = trainer(shared, tmp_path / "run", model=predictor(shared, **ACTIVE))
    code = engine._code()
    for module in ("huber_memory", "scale_invariant_time", "differential_attention"):
        assert f"mars_titan.models.titans.{module}" in code
    assert engine.identity["implementation"] == code
