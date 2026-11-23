"""Lector episódico con el término de IC por sesión en su pérdida, sin pasos que ajusten pesos."""

from collections import Counter

import pytest
import torch

from mars_titan.data.input_policy import HISTORICAL_MASKED
from mars_titan.memory import financial_observations as observations
from mars_titan.models.quantile_head import MEDIAN_GAIN, MEDIAN_INDEX
from mars_titan.models.titans.financial import FinancialConfig, FinancialPredictor
from mars_titan.models.titans.session_ic_objective import SESSION_IC_TERM
from mars_titan.training import mars_titan_run as mt
from mars_titan.training.corpus_inputs import CorpusDataset
from tests.models.titans.test_session_ic_term import reference
from tests.training.chronological_fixture import chronological_corpus, phases
from tests.training.test_financial_run import entries
from tests.training.test_mars_titan_run import (
    build,
    explicit_fastpath,  # noqa: F401
    gradients,
    native,  # noqa: F401
    recipe,
)

ASSETS = 5
TERM = dict(weight=2.0, minimum_rows=3)


@pytest.fixture(scope="module")
def sessions(tmp_path_factory):
    """Corpus con cinco activos por sesión, para que cada una tenga filas que ordenar."""
    root = tmp_path_factory.mktemp("reader-session-ic")
    dataset = CorpusDataset(
        chronological_corpus(root / "corpus", assets=ASSETS), input_policy=HISTORICAL_MASKED
    )
    result = {}
    for phase in phases():
        manifest = observations.prepare_observation_index(
            dataset, root / f"index-{phase.partition}", phase=phase
        )
        result[phase.partition] = observations.FinancialObservationSource(dataset, manifest)
    return result


class Spy:
    """Registra cada llamada al término sin cambiar su valor."""

    def __init__(self, inner):
        self.inner, self.calls = inner, []

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def term(self, point, target, sessions, liquidity=None):
        value = self.inner.term(point, target, sessions, liquidity)
        self.calls.append(
            (point.detach().clone(), target.detach().clone(), list(sessions), value.detach())
        )
        return value


def dispersed_parent(sessions, **changes):
    """Padre con la memoria de la receta de campaña, que conserva la dispersión entre activos.

    Con la memoria v1 del fixture la salida se iguala entre activos y la correlación
    regularizada trata esa dispersión como falta de información.
    """
    config = FinancialConfig(
        sessions["train"].specification(),
        variant="mac_online",
        hidden_size=32,
        seed=42,
        head="quantile_head_v1",
        memory_residual_layer_norm=True,
        gate_bias=dict(alpha_half_life=256.0, eta=0.5, theta=0.05),
        **changes,
    )
    return FinancialPredictor(config, dtype=torch.float64).eval().requires_grad_(False)


def fitted(sessions, output, native, **changes):  # noqa: F811
    options = dict(max_grad_norm=None, session_ic_term=TERM) | changes
    engine = build(
        sessions, output, native, model=dispersed_parent(sessions), plan=recipe(**options)
    )
    if engine.session_ic is not None:
        engine.session_ic = Spy(engine.session_ic)
    report = engine.run()
    assert report["status"] == "completed"
    return engine, report


def close(left, right, **tolerance):
    assert left.keys() == right.keys()
    for name, value in left.items():
        if value is None:
            assert right[name] is None, name
        else:
            torch.testing.assert_close(right[name], value, **tolerance)


@pytest.mark.native("MARS_TITAN_EPISODIC_NATIVE")
def test_each_update_scores_whole_sessions_with_the_points_the_reader_issued(
    sessions,
    tmp_path,
    native,  # noqa: F811
    learning_doubles,
):
    engine, report = fitted(sessions, tmp_path / "run", native, block_rows=2)
    spy = engine.session_ic
    updates = entries(engine.audit, "update")
    assert len(spy.calls) == len(updates) > 2
    issued = {
        (flow, at): value
        for _, _, flow, at, value, _ in entries(engine.audit, "prediction", "train")
    }
    seen = Counter()
    for (point, target, keys, value), update in zip(spy.calls, updates, strict=True):
        used = update[3]
        # Las filas del término son las etiquetas del tramo, con el punto que emitió el lector.
        assert sorted(keys) == sorted((flow.split("/", 1)[0], at) for flow, at in used)
        assert sorted(point.tolist()) == pytest.approx(sorted(issued[key] for key in used))
        expected = reference(point.tolist(), target.tolist(), keys, TERM["weight"], 3)
        assert value.item() == pytest.approx(expected, rel=1e-7, abs=1e-12)
        seen.update(keys)
    # Cada sesión entra entera en una sola actualización aunque sus filas vayan en tres bloques.
    assert max(seen.values()) == ASSETS and any(value.item() != 0 for *_, value in spy.calls)
    assert engine.identity["recipe"]["session_ic_term"]["name"] == SESSION_IC_TERM


@pytest.mark.native("MARS_TITAN_EPISODIC_NATIVE")
def test_the_gradient_does_not_depend_on_how_the_blocks_split_each_session(
    sessions,
    tmp_path,
    native,  # noqa: F811
    learning_doubles,
):
    records = {
        rows: gradients(fitted(sessions, tmp_path / f"rows-{rows}", native, block_rows=rows)[0])
        for rows in (1, 2, 8)
    }
    assert len(records[1]) == len(records[2]) == len(records[8]) > 3
    for rows in (1, 2):
        for left, right in zip(records[rows], records[8], strict=True):
            close(left, right, rtol=1e-9, atol=1e-12)


@pytest.mark.native("MARS_TITAN_EPISODIC_NATIVE")
def test_the_term_adds_a_gradient_that_is_linear_in_its_weight(
    sessions,
    tmp_path,
    native,  # noqa: F811
    learning_doubles,
):
    plain = gradients(fitted(sessions, tmp_path / "plain", native, session_ic_term=None)[0])
    one = gradients(
        fitted(
            sessions, tmp_path / "one", native, session_ic_term=dict(weight=1.0, minimum_rows=3)
        )[0]
    )
    two = gradients(fitted(sessions, tmp_path / "two", native)[0])
    assert len(plain) == len(one) == len(two) > 3
    changed = False
    for base, single, double in zip(plain, one, two, strict=True):
        assert base.keys() == single.keys() == double.keys()
        for name, value in base.items():
            if value is None:
                assert single[name] is None and double[name] is None
                continue
            # El optimizador registrador no mueve los pesos: la diferencia es solo el término.
            first, second = single[name] - value, double[name] - value
            torch.testing.assert_close(second, 2 * first, rtol=1e-7, atol=1e-12)
            changed = changed or bool(first.abs().sum() > 0)
    assert changed


@pytest.mark.native("MARS_TITAN_EPISODIC_NATIVE")
def test_the_reported_loss_is_the_declared_loss_plus_the_term(
    sessions,
    tmp_path,
    native,  # noqa: F811
    learning_doubles,
):
    engine, report = fitted(sessions, tmp_path / "with", native)
    _, plain = fitted(sessions, tmp_path / "without", native, session_ic_term=None)
    train, base = report["history"][1]["train"], plain["history"][1]["train"]
    assert train["labels_in_loss"] == base["labels_in_loss"] > 0
    calls = engine.session_ic.calls
    # Media por etiqueta: cada actualización pesa por sus etiquetas, también en el término.
    weighted = sum(value.item() * len(point) for point, _, _, value in calls)
    expected = base["mean_loss"] + weighted / train["labels_in_loss"]
    assert train["mean_loss"] == pytest.approx(expected, rel=1e-9)


def test_the_reader_recipe_declares_the_term_whole_and_without_liquidity():
    plain, declared = recipe(), recipe(session_ic_term=TERM)
    assert plain.session_ic() is None and "session_ic_term" not in plain.identity()
    identity = declared.identity()
    assert identity["session_ic_term"]["name"] == SESSION_IC_TERM
    assert {k: v for k, v in identity.items() if k != "session_ic_term"} == plain.identity()
    assert recipe(session_ic_term=None).identity() == plain.identity()
    for value in (
        dict(weight=1.0),
        dict(weight=0.0, minimum_rows=3),
        dict(weight=1.0, minimum_rows=3, otra=1),
        dict(weight=1.0, minimum_rows=3, liquidity_column=-1, liquidity_power=1.0),
        dict(weight=1.0, minimum_rows=3, liquidity_column=-1, liquidity_weight=2.0),
        dict(weight=1.0, minimum_rows=3, liquidity_column=-1, liquidity_weight=1.0),
        "session_ic_term_v1",
    ):
        with pytest.raises(ValueError):
            mt.ReadoutRecipe(session_ic_term=value)
    # `stability` no requiere entradas extra y entra en la identidad solo cuando > 0.
    stable = recipe(session_ic_term=dict(TERM, stability=0.5))
    assert stable.identity()["session_ic_term"]["stability"] == 0.5
    assert recipe(session_ic_term=dict(TERM, stability=0.0)).identity() == declared.identity()


@pytest.mark.native("MARS_TITAN_EPISODIC_NATIVE")
@pytest.mark.parametrize("term", [None, TERM])
def test_a_parent_with_the_median_gain_reads_as_its_equivalent_without_it(
    sessions,
    tmp_path,
    native,  # noqa: F811
    learning_doubles,
    term,
):
    """Un lector aplica la cabeza del padre: la ganancia debe llegar a lo que emite.

    Un padre con ganancia 1/4 emite lo mismo que otro sin ella con la fila de la mediana
    de su cabeza dividida entre cuatro. Si algún camino del lector calculara los cuantiles
    con los pesos de la cabeza, sin su ganancia, las dos lecturas se separarían.
    """
    gained, plain = dispersed_parent(sessions, median_gain=MEDIAN_GAIN), dispersed_parent(sessions)
    with torch.no_grad():
        gained.head.median_gain.fill_(-2.0)
        factor = torch.exp(gained.head.median_gain)
        plain.head.weight[MEDIAN_INDEX].mul_(factor)
        plain.head.bias[MEDIAN_INDEX].mul_(factor)
    gained._seal_parameters()
    plain._seal_parameters()

    def read(model, folder):
        engine = build(
            sessions,
            tmp_path / folder,
            native,
            model=model,
            plan=recipe(max_grad_norm=None, session_ic_term=term),
        )
        report = engine.run()
        assert report["status"] == "completed"
        return engine

    left, right = read(gained, "gained"), read(plain, "plain")
    assert left.identity != right.identity
    seen = 0
    for partition in ("train", "validation"):
        one = entries(left.audit, "prediction", partition)
        two = entries(right.audit, "prediction", partition)
        assert len(one) == len(two) > 0
        for first, second in zip(one, two, strict=True):
            assert first[2:4] == second[2:4]
            assert first[4] == pytest.approx(second[4], rel=1e-9, abs=1e-12)
            seen += first[4] != 0
    assert seen > 0
    # La actualización del lector también coincide: el gradiente pasa por la ganancia.
    for first, second in zip(gradients(left), gradients(right), strict=True):
        close(first, second, rtol=1e-8, atol=1e-11)
