"""Cota con información futura en la etapa de políticas y en su informe, sin aprender.

Las cintas son precios escritos en la prueba y los recibos del informe se escriben a mano.
Nada pasa por la protección de aprendizaje: la cota evalúa una regla fija y el informe solo
lee archivos confirmados.
"""

import math
from dataclasses import replace

import pytest

from mars_titan.data.storage import atomic_json, sha256
from mars_titan.integrity import oracle_bound
from mars_titan.simulation import campaign_stage, stage_report
from mars_titan.simulation.campaign_stage import PolicyTapes
from mars_titan.simulation.environment import ALLOCATIONS, FinancialEnv
from mars_titan.simulation.evaluation import evaluate, fixed_policy
from tests.integrity.test_oracle_bound import walk
from tests.simulation.test_financial_adversarial import custom

COSTS = [10, 0, 5, 20]
ZERO = COSTS.index(0)
ENVIRONMENT = dict(
    capital=1_000_000,
    cost_bps=10,
    participation=0.01,
    score_scale=0.01,
    ruin_penalty=-20,
    dividend_payment_lag_sessions=0,
)
STAGE = dict(policies=dict(environment=ENVIRONMENT, evaluation_costs_bps=COSTS))
OPTIONS = {k: v for k, v in ENVIRONMENT.items() if k != "dividend_payment_lag_sessions"}
TAPE_SHA = "a" * 64


def tapes(evaluation):
    return PolicyTapes(
        universe=(),
        train=(),
        validation=None,
        evaluation=evaluation,
        failure=None if evaluation is not None else dict(reason="sin cinta"),
        unfit=None,
        paths={},
        identity={},
    )


# Cálculo en la etapa


def test_the_bound_is_the_full_exposure_rule_on_the_future_scored_tape_without_costs():
    tape = custom(walk())
    found = campaign_stage.future_information_bound(tapes(tape), STAGE, "US", backend="python")
    assert found["kind"] == campaign_stage.BOUND_KIND
    assert found["excluded_from_comparisons"] is True and found["final_test_opened"] is False
    assert found["tape_sha256"] == tape.sha256
    assert found["oracle_tape_sha256"] == oracle_bound.oracle_tape(tape).sha256
    assert found["action"] == oracle_bound.FULL_EXPOSURE
    assert found["suspicious_fraction"] == oracle_bound.SUSPICIOUS_FRACTION
    episode = found["episode"]
    assert episode["cost_bps"] == 0 and episode["status"] == "completed"
    assert "equity" not in episode and episode["costs"] == 0
    expected = oracle_bound.bound(
        tape, env_options=dict(OPTIONS, cost_bps=0, backend="python", allocation=ALLOCATIONS[0])
    )
    assert episode["net_return"] == expected["policies"]["oracle_full_exposure"]["net_return"]


@pytest.mark.parametrize("seed", range(5))
def test_without_costs_buying_and_holding_one_asset_stays_below_the_bound(seed):
    # Una sola columna, como la cinta del índice de mercado: la cota solo está dentro en las
    # tenencias que suben y comprar y mantener las recorre todas.
    tape = custom(walk(sessions=120, assets=1, seed=seed))
    found = campaign_stage.future_information_bound(tapes(tape), STAGE, "US", backend="python")
    env = FinancialEnv(
        tape, backend="python", allocation=ALLOCATIONS[1], **dict(OPTIONS, cost_bps=0)
    )
    held = evaluate(env, fixed_policy("market_index"))["financial_validation"]["net_return"]
    assert math.log1p(held) < math.log1p(found["episode"]["net_return"])


@pytest.mark.native("MARS_TITAN_NATIVE_LIBRARY")
def test_the_native_accounting_gives_the_same_bound():
    tape = custom(walk())
    python = campaign_stage.future_information_bound(tapes(tape), STAGE, "US", backend="python")
    native = campaign_stage.future_information_bound(tapes(tape), STAGE, "US", backend="native")
    ours, theirs = python["episode"], native["episode"]
    assert ours.keys() == theirs.keys() and ours["status"] == theirs["status"]
    for key, value in ours.items():
        if isinstance(value, float):
            assert theirs[key] == pytest.approx(value, rel=1e-12, abs=1e-12), key
        else:
            assert theirs[key] == value, key


def stage_state(output):
    state = campaign_stage._Stage(STAGE, None, output, {}, {}, None)
    state.bound_backend = "python"
    return state


def test_each_evaluation_tape_computes_its_bound_once(tmp_path, monkeypatch):
    state, tape = stage_state(tmp_path), custom(walk(sessions=20, assets=3))
    first = state.bound(dict(market="US"), tapes(tape))
    assert first["path"] == f"bounds/{tape.sha256}.json"
    assert first["sha256"] == sha256(tmp_path / first["path"])

    def forbidden(*args, **kwargs):
        raise AssertionError("La cota de una cinta ya calculada se volvió a calcular")

    monkeypatch.setattr(campaign_stage, "future_information_bound", forbidden)
    assert state.bound(dict(market="US"), tapes(tape)) == first
    assert state.bound(dict(market="US"), tapes(None)) is None


def test_a_stored_bound_of_another_tape_is_rejected(tmp_path):
    state, tape = stage_state(tmp_path), custom(walk(sessions=20, assets=3))
    atomic_json(
        tmp_path / "bounds" / f"{tape.sha256}.json",
        dict(kind=campaign_stage.BOUND_KIND, tape_sha256=TAPE_SHA),
    )
    with pytest.raises(ValueError, match="no es la de su cinta"):
        state.bound(dict(market="US"), tapes(tape))


# Comprobación en el informe


def record(cost, net_return, status="completed", reason=None):
    return dict(cost_bps=cost, status=status, reason=reason, net_return=net_return)


def bound_document(net_return=3.0, status="completed", reason=None, **changes):
    return dict(
        dict(
            schema_version=1,
            kind=campaign_stage.BOUND_KIND,
            rule=oracle_bound.RULE,
            action=oracle_bound.FULL_EXPOSURE,
            suspicious_fraction=oracle_bound.SUSPICIOUS_FRACTION,
            tape_sha256=TAPE_SHA,
            oracle_tape_sha256="b" * 64,
            episode=record(0, net_return, status, reason),
            excluded_from_comparisons=True,
            final_test_opened=False,
        ),
        **changes,
    )


def job(arm="klpo_terminal", kind="fit", seed=43):
    return dict(
        id=f"fold-002/gru/{arm}/{kind}-s{seed}",
        scope="us_large",
        market="US",
        window="fold-002",
        predictor="gru",
        arm=arm,
        kind=kind,
        seed=seed,
    )


def receipt(free, *, of=None, reference=None, evaluation_tape=TAPE_SHA):
    """Recibo con el episodio sin costes dado y los episodios con costes muy por debajo."""
    episodes = [record(cost, -0.5) for cost in COSTS]
    episodes[ZERO] = free if isinstance(free, dict) else record(0, free)
    return dict(
        identity=dict(tapes=dict(evaluation=evaluation_tape)),
        evaluation=episodes,
        future_information_bound=reference,
        job=of or job(),
    )


def output_with(root, receipts, bound=None):
    """Salida con los recibos dados. Los que tienen cinta y no enlazan nada reciben la cota."""
    root.mkdir(parents=True, exist_ok=True)
    if bound is not None:
        atomic_json(root / "bounds" / "tape.json", bound)
        linked = dict(path="bounds/tape.json", sha256=sha256(root / "bounds" / "tape.json"))
        for item in receipts:
            if item["future_information_bound"] is None and item["identity"]["tapes"]["evaluation"]:
                item["future_information_bound"] = linked
    return dict(path=str(root.resolve()), receipts={r["job"]["id"]: r for r in receipts})


def checked(output, costs=COSTS):
    return stage_report.future_information(dict(evaluation_costs_bps=costs), output)


def test_episodes_far_below_the_bound_are_checked_without_flags(tmp_path):
    found = checked(output_with(tmp_path, [receipt(0.1)], bound_document(1.0)))
    assert found["status"] == "no_flagged_episodes" and found["cost_bps"] == 0
    assert found["checked"] == 1 and found["flagged"] == [] and found["unchecked"] == []
    assert found["rule"] == oracle_bound.RULE


def test_only_the_episode_without_costs_is_compared(tmp_path):
    # Con costes la cota rota cada sesión y deja de ser un máximo: comprar y mantener puede
    # superarla sin ningún fallo del entorno.
    held = receipt(0.1, of=job("market_index", "reference", None))
    held["evaluation"] = [record(cost, 0.1) for cost in COSTS]
    output = output_with(tmp_path, [held], bound_document(0.5))
    found = checked(output)
    assert found["checked"] == 1 and found["flagged"] == []
    with pytest.raises(ValueError, match="coste cero"):
        checked(output, costs=[5, 10, 20, 30])


def test_reaching_half_the_bound_growth_or_exceeding_it_is_flagged(tmp_path):
    # ln(1 + 1) es exactamente la mitad de ln(1 + 3): el umbral está incluido.
    assert math.log1p(1.0) == 0.5 * math.log1p(3.0)
    returns = {43: 1.0, 44: 3.0, 45: 3.5, 46: 0.9}
    receipts = [receipt(value, of=job(seed=seed)) for seed, value in returns.items()]
    found = checked(output_with(tmp_path, receipts, bound_document(3.0)))
    assert found["status"] == "review_required" and found["checked"] == 4
    flagged = {item["seed"]: item for item in found["flagged"]}
    assert set(flagged) == {43, 44, 45}
    assert flagged[43]["reason"] == "suspicious_fraction"
    assert flagged[43]["fraction_of_bound"] == 0.5
    # Igualar la cota no la supera, pero alcanza la fracción.
    assert flagged[44]["reason"] == "suspicious_fraction"
    assert flagged[45]["reason"] == "exceeds_bound"
    assert flagged[45]["log_growth"] == math.log1p(3.5)
    assert flagged[45]["bound_log_growth"] == math.log1p(3.0)
    item = flagged[43]
    assert (item["scope"], item["market"], item["window"]) == ("us_large", "US", "fold-002")
    assert (item["predictor"], item["arm"]) == ("gru", "klpo_terminal")
    assert item["bound"] == "bounds/tape.json"


@pytest.mark.parametrize(
    "arm, reads",
    [
        ("cash", True),
        ("hold_initial", True),
        ("rebalance_50", True),
        ("equal_weight_monthly", False),
        ("market_index", False),
    ],
)
def test_the_fraction_applies_only_to_policies_that_allocate_with_the_scores(tmp_path, arm, reads):
    assert stage_report.reads_scores(job(arm, "reference", None)) is reads
    near = receipt(2.0, of=job(arm, "reference", None))
    above = receipt(3.5, of=dict(job(arm, "reference", None), id="otra"))
    found = checked(output_with(tmp_path, [near, above], bound_document(3.0)))
    reasons = {item["log_growth"]: item["reason"] for item in found["flagged"]}
    expected = {math.log1p(3.5): "exceeds_bound"}
    if reads:
        expected[math.log1p(2.0)] = "suspicious_fraction"
    assert reasons == expected
    for kind in ("fit", "carry"):
        assert stage_report.reads_scores(job("market_index", kind)) is True


def test_without_positive_bound_growth_only_exceeding_is_flagged(tmp_path):
    # Un mercado plano no deja crecer a la cota: la caja la iguala sin superarla.
    found = checked(output_with(tmp_path, [receipt(0.0)], bound_document(0.0)))
    assert found["flagged"] == [] and found["checked"] == 1
    found = checked(output_with(tmp_path, [receipt(0.001)], bound_document(0.0)))
    assert [item["reason"] for item in found["flagged"]] == ["exceeds_bound"]
    assert found["flagged"][0]["fraction_of_bound"] is None


def test_unfinished_episodes_and_unfinished_bounds_are_not_compared(tmp_path):
    failed = receipt(record(0, None, "failed", "unpriced_exit"), of=job(seed=43))
    ruined = receipt(record(0, -1.0, "ruined", "ruined"), of=job(seed=44))
    found = checked(output_with(tmp_path / "a", [failed, ruined], bound_document(1.0)))
    assert found["checked"] == 0 and found["flagged"] == [] and found["unchecked"] == []
    unfinished = bound_document(None, status="failed", reason="unpriced_exit")
    found = checked(output_with(tmp_path / "b", [receipt(9.0)], unfinished))
    assert found["checked"] == 0 and found["flagged"] == []
    assert found["unchecked"] == [
        dict(
            scope="us_large",
            market="US",
            window="fold-002",
            predictor="gru",
            arm="klpo_terminal",
            seed=43,
            bound_status="failed",
            reason="unpriced_exit",
        )
    ]


def test_a_receipt_without_evaluation_tape_has_no_bound(tmp_path):
    failed = receipt(record(0, None, "failed", "sin cinta"), evaluation_tape=None)
    found = checked(output_with(tmp_path, [failed]))
    assert found == dict(found, checked=0, flagged=[], unchecked=[])


@pytest.mark.parametrize(
    "change",
    [
        dict(kind="other"),
        dict(tape_sha256="c" * 64),
        dict(action=4),
        dict(suspicious_fraction=0.9),
        dict(excluded_from_comparisons=False),
        dict(final_test_opened=True),
        dict(episode=record(10, 3.0)),
        dict(episode=None),
    ],
)
def test_a_bound_that_is_not_the_one_of_its_tape_is_rejected(tmp_path, change):
    output = output_with(tmp_path, [receipt(0.1)], bound_document(**change))
    with pytest.raises(ValueError, match="no es la de su cinta"):
        checked(output)


def test_a_changed_bound_file_is_rejected(tmp_path):
    output = output_with(tmp_path, [receipt(0.1)], bound_document(1.0))
    atomic_json(tmp_path / "bounds" / "tape.json", bound_document(0.15))
    with pytest.raises(ValueError, match="no es la de su cinta"):
        checked(output)


def test_receipts_must_link_the_bound_of_their_tape_inside_the_output(tmp_path):
    output = output_with(tmp_path / "a", [receipt(0.1)], bound_document(1.0))
    next(iter(output["receipts"].values())).pop("future_information_bound")
    # El recibo no tiene el campo: es otro error que enlazar ninguna cota.
    with pytest.raises(ValueError, match="s43 no enlaza su cota"):
        checked(output)
    with pytest.raises(ValueError, match="tiene cinta de evaluación y no enlaza su cota"):
        checked(output_with(tmp_path / "b", [receipt(0.1)]))
    outside = tmp_path / "bounds" / "tape.json"
    atomic_json(outside, bound_document(1.0))
    reference = dict(path="../bounds/tape.json", sha256=sha256(outside))
    with pytest.raises(ValueError, match="fuera de su salida"):
        checked(output_with(tmp_path / "c", [receipt(0.1, reference=reference)]))


def test_a_flagged_episode_takes_precedence_over_the_pending_survival_sensitivity():
    def section(flagged, affected, chosen=()):
        return dict(
            future_information=dict(flagged=flagged),
            validation_information=dict(flagged=list(chosen)),
            survival=dict(affected=affected),
        )

    assert stage_report.report_status([section([], [])]) == "complete"
    pending = [section([], []), section([], [dict(window="fold-002")])]
    assert stage_report.report_status(pending) == "pending_survival_sensitivity"
    flagged = [section([dict(reason="exceeds_bound")], []), section([], [dict(window="w")])]
    assert stage_report.report_status(flagged) == "environment_review_required"
    assert stage_report.report_status(list(reversed(flagged))) == "environment_review_required"
    # Un mejor estado de validación señalado también pasa por delante de la supervivencia.
    chosen = [section([], [dict(window="w")], [dict(reason="suspicious_fraction")])]
    assert stage_report.report_status(chosen) == "environment_review_required"


# Cota de la cinta de validación y mejor estado de cada ajuste

VALIDATION_SHA = "d" * 64


def test_the_validation_bound_is_the_bound_of_that_tape_and_shared_with_its_evaluation_role(
    tmp_path, monkeypatch
):
    tape = custom(walk(sessions=20, assets=3))
    as_validation = replace(tapes(None), validation=tape)
    found = campaign_stage.future_information_bound(
        as_validation, STAGE, "US", backend="python", partition="validation"
    )
    assert found == campaign_stage.future_information_bound(
        tapes(tape), STAGE, "US", backend="python"
    )
    # Una cinta que evalúa una ventana y valida la siguiente se calcula una sola vez.
    state = stage_state(tmp_path)
    first = state.bound(dict(market="US"), tapes(tape))

    def forbidden(*args, **kwargs):
        raise AssertionError("La cota de una cinta ya calculada se volvió a calcular")

    monkeypatch.setattr(campaign_stage, "future_information_bound", forbidden)
    assert state.bound(dict(market="US"), as_validation, "validation") == first


@pytest.mark.parametrize(
    "kind, selection, linked",
    [("fit", dict(best={}), True), ("fit", None, False), ("carry", None, False)]
    + [("reference", None, False)],
)
def test_only_a_fit_that_chose_a_state_links_the_bound_of_its_validation(
    tmp_path, monkeypatch, kind, selection, linked
):
    monkeypatch.setattr(campaign_stage, "check_report", lambda *args: None)
    state = stage_state(tmp_path)
    evaluation = custom(walk(sessions=20, assets=3, seed=1))
    validation = custom(walk(sessions=20, assets=3, seed=2))
    work = replace(tapes(evaluation), validation=validation)
    report = dict(transitions=0, updates=0, selection=selection, policy=None, evaluation=[])
    confirmed = state.confirm(job(kind=kind), {}, work, report, None)
    assert confirmed["future_information_bound"]["path"] == f"bounds/{evaluation.sha256}.json"
    linked_bound = confirmed["validation_future_information_bound"]
    if linked:
        assert linked_bound["path"] == f"bounds/{validation.sha256}.json"
        assert linked_bound["sha256"] == sha256(tmp_path / linked_bound["path"])
    else:
        assert linked_bound is None


def chosen(growth, *, ruins=0, of=None, validation_tape=VALIDATION_SHA):
    """Recibo de un ajuste con el mejor estado de validación dado."""
    item = receipt(0.1, of=of)
    item["identity"]["tapes"]["validation"] = validation_tape
    item["selection"] = dict(
        metric="ruin_count_then_mean_liquidated_log_growth",
        partition="validation",
        best=dict(ruin_count=ruins, mean_liquidated_log_growth=growth),
    )
    item["validation_future_information_bound"] = None
    return item


def validated(root, receipts, net_return=1.0, **changes):
    """Salida con la cota de evaluación y la de validación enlazadas a cada ajuste."""
    output = output_with(root, receipts, bound_document(3.0))
    document = bound_document(net_return, **dict(dict(tape_sha256=VALIDATION_SHA), **changes))
    atomic_json(root / "bounds" / "validation.json", document)
    linked = dict(path="bounds/validation.json", sha256=sha256(root / "bounds" / "validation.json"))
    for item in receipts:
        if item.get("selection") is not None:
            item["validation_future_information_bound"] = linked
    return output


POLICIES = dict(
    evaluation_costs_bps=COSTS,
    environment=ENVIRONMENT,
    selection=dict(metric="ruin_count_then_mean_liquidated_log_growth"),
)


def chosen_states(output):
    return stage_report.validation_information(POLICIES, output)


def test_best_states_far_below_the_validation_bound_are_checked_without_flags(tmp_path):
    found = chosen_states(validated(tmp_path, [chosen(0.05)]))
    assert found["status"] == "no_flagged_states" and found["checked"] == 1
    assert found["flagged"] == [] and found["unchecked"] == []
    assert (found["cost_bps"], found["bound_cost_bps"]) == (10, 0)
    assert found["metric"] == POLICIES["selection"]["metric"]


@pytest.mark.parametrize(
    "growth, reason", [(math.log(2) * 0.5, "suspicious_fraction"), (0.8, "exceeds_bound")]
)
def test_a_best_state_near_or_above_the_validation_bound_is_flagged(tmp_path, growth, reason):
    # La cota crece log(2): la mitad ya es sospechosa y más que ella la supera.
    found = chosen_states(validated(tmp_path, [chosen(growth)]))
    assert found["status"] == "review_required"
    (flag,) = found["flagged"]
    assert flag["reason"] == reason and flag["log_growth"] == growth
    assert flag["bound_log_growth"] == pytest.approx(math.log(2))
    assert flag["bound"] == "bounds/validation.json" and flag["arm"] == "klpo_terminal"


def test_ruined_states_are_not_compared_and_unfinished_bounds_are_unchecked(tmp_path):
    found = chosen_states(validated(tmp_path / "a", [chosen(5.0, ruins=1)]))
    assert found["checked"] == 0 and found["flagged"] == [] and found["unchecked"] == []
    unfinished = validated(
        tmp_path / "b", [chosen(5.0)], None, episode=record(0, None, "failed", "unpriced_exit")
    )
    found = chosen_states(unfinished)
    assert found["checked"] == 0 and found["flagged"] == []
    assert found["unchecked"][0]["reason"] == "unpriced_exit"


def test_only_fits_that_chose_a_state_link_the_validation_bound(tmp_path):
    output = validated(tmp_path / "a", [chosen(0.05)])
    linked = next(iter(output["receipts"].values()))
    linked.pop("validation_future_information_bound")
    with pytest.raises(ValueError, match="no enlaza la cota de su validación"):
        chosen_states(output)
    # Un ajuste con selección sin cota, o una referencia con ella, también fallan.
    output = validated(tmp_path / "b", [chosen(0.05)])
    next(iter(output["receipts"].values()))["validation_future_information_bound"] = None
    with pytest.raises(ValueError, match="Solo el recibo de un ajuste con selección"):
        chosen_states(output)
    reference = chosen(0.05, of=job(arm="market_index", kind="reference"))
    output = validated(tmp_path / "c", [reference])
    next(iter(output["receipts"].values()))["selection"] = None
    next(iter(output["receipts"].values()))["validation_future_information_bound"] = dict(
        path="bounds/validation.json", sha256="0" * 64
    )
    with pytest.raises(ValueError, match="Solo el recibo de un ajuste con selección"):
        chosen_states(output)


@pytest.mark.parametrize(
    "metric, field",
    [
        ("ruin_count_then_mean_liquidated_log_growth", "mean_liquidated_log_growth"),
        ("ruin_count_then_mean_log_growth", "mean_log_growth"),
    ],
)
def test_the_best_state_is_read_with_the_growth_of_the_declared_criterion(metric, field):
    best = dict(ruin_count=0, **{field: 0.25})
    assert campaign_stage.best_state(best, metric) == (0, 0.25)
    other = next(name for name in campaign_stage.SELECTION_GROWTH if name != metric)
    assert campaign_stage.best_state(best, other) is None
    for wrong in (
        dict(best, ruin_count=-1),
        dict(best, ruin_count=1.0),
        dict(best, **{field: math.inf}),
        dict(best, **{field: 1}),
        None,
    ):
        assert campaign_stage.best_state(wrong, metric) is None


def test_the_bound_of_another_tape_or_a_best_state_without_growth_is_rejected(tmp_path):
    other = validated(tmp_path / "a", [chosen(0.05, validation_tape=TAPE_SHA)])
    with pytest.raises(ValueError, match="no es la de su cinta de validation"):
        chosen_states(other)
    output = validated(tmp_path / "b", [chosen(0.05)])
    next(iter(output["receipts"].values()))["selection"]["best"].pop("ruin_count")
    with pytest.raises(ValueError, match="no publica sus ruinas"):
        chosen_states(output)
