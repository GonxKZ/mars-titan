"""Entrenador de la GRU candidata comprobado hasta el paso del optimizador, sin modificar pesos.

Las pruebas calculan pérdidas y gradientes, pero el optimizador solo registra llamadas.
Necesitan el enlace `_episodic_native` compilado con el candidato.
"""

import json
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
import torch

from mars_titan.evaluation.forecast_panel import ForecastPanel
from mars_titan.evaluation.session_metrics import SessionErrors
from mars_titan.evaluation.splits import stopping_rule
from mars_titan.memory.candidate_bank import (
    CandidateBankConfig,
    CandidateEpisodeBank,
    CandidateEpisodes,
)
from mars_titan.models.quantile_head import LEVELS, QUANTILE_COLUMNS, pinball_loss
from mars_titan.models.titans.financial_inputs import DecisionBatch, validated_cpu_batch
from mars_titan.training import candidate_run
from mars_titan.training.checkpoints import load_training_state, save_training_state
from mars_titan.training.learning_hold import LearningHoldError
from tests.training.chronological_fixture import decision
from tests.training.test_financial_run import RecordingOptimizer, StopAtStep, corpus

# El optimizador de las pruebas solo registra gradientes, así que basta una protección
# temporal permitida. El gancho global sigue omitiendo cualquier paso de torch.optim.
pytestmark = [
    pytest.mark.skipif(
        not os.environ.get("MARS_TITAN_EPISODIC_NATIVE"), reason="Falta el enlace nativo compilado"
    ),
    pytest.mark.usefixtures("learning_doubles"),
]

ROOT = Path(__file__).parents[2]
CONFIG = ROOT / "configs/candidate/chronological-training.json"
SELECTION = dict(metric="session_mae", patience=5, min_delta=1e-05, stopping="fixed_budget")
READ = {"query_weight", "query_bias", "value_weight", "value_bias"}


def modules():
    from mars_titan.models.candidate.episode_codec import FrozenCandidateCodec
    from mars_titan.models.candidate.frozen_consumer import FrozenCandidateConsumer
    from mars_titan.models.candidate.input_adapter import CandidateInputAdapter

    return CandidateInputAdapter, FrozenCandidateConsumer, FrozenCandidateCodec


@pytest.fixture(scope="module")
def shared(tmp_path_factory):
    return corpus(tmp_path_factory.mktemp("candidate"))


def adapter(streams, *, dtype=torch.float64, seed=42):
    return modules()[0](streams["train"].specification(), dtype=dtype, parameter_seed=seed)


def trainer(streams, output, *, model=None, heldout=None, factory=RecordingOptimizer, **options):
    recipe = dict(update_instants=3, epochs=1, block_rows=2, bank_capacity=4, selection=SELECTION)
    recipe.update(options)
    return candidate_run.CandidateChronologicalTrainer(
        model or adapter(streams),
        candidate_run.CandidateRecipe(**recipe),
        train=streams["train"],
        validation=streams["validation"],
        heldout=heldout,
        output=output,
        optimizer_factory=factory,
        audit=True,
    )


def entries(audit, kind, phase=None):
    return [e for e in audit if e[0] == kind and (phase is None or e[1] == phase)]


def named_records(engine):
    names = {id(p): name for name, p in engine.model.named_parameters().items()}
    return [{names[key]: value for key, value in r.items()} for r in engine.optimizer.records]


def active(value):
    return value is not None and bool(torch.isfinite(value).all()) and bool(value.abs().sum() > 0)


@pytest.mark.parametrize(
    "options",
    [
        dict(admission="m2"),
        dict(refinements=3),
        dict(refinements=True),
        dict(loss="mse"),
        dict(update_instants=0),
        dict(block_rows=257),
        dict(bank_capacity=0),
        dict(bank_seed=-1),
        dict(learning_rate=0.0),
        dict(weight_decay=-1.0),
        dict(max_grad_norm=float("inf")),
        dict(checkpoint_seconds=0.0),
        dict(selection=dict(SELECTION, metric="mae")),
        dict(selection=dict(SELECTION, stopping="unbounded")),
        dict(weight_decay_anchor="parent"),
    ],
)
def test_recipe_rejects_undeclared_or_invalid_declarations(options):
    with pytest.raises(ValueError):
        candidate_run.CandidateRecipe(**options)


def test_the_default_optimizer_anchors_the_decay_that_the_recipe_declares(shared, tmp_path):
    from mars_titan.training.anchored_decay import INITIAL, AnchoredAdamW

    _, streams = shared
    engine = trainer(
        streams, tmp_path / "anchored", factory=None, weight_decay=0.01, weight_decay_anchor=INITIAL
    )
    assert type(engine.optimizer) is AnchoredAdamW
    pairs = [
        (value, anchor)
        for group in engine.optimizer.param_groups
        for value, anchor in zip(group["params"], group["anchors"], strict=True)
    ]
    assert len(pairs) == len(engine.trainable) and all(
        torch.equal(value, anchor) and value.data_ptr() != anchor.data_ptr()
        for value, anchor in pairs
    )
    assert engine.identity["recipe"]["weight_decay_anchor"] == INITIAL
    assert engine.identity["anchored_decay"]["anchor"] == INITIAL
    plain = trainer(streams, tmp_path / "plain", factory=None, weight_decay=0.01)
    assert type(plain.optimizer) is torch.optim.AdamW and "anchored_decay" not in plain.identity


def test_declared_variants_follow_the_protocol_rule_and_budget(tmp_path):
    document = json.loads(CONFIG.read_text())
    protocol = json.loads((CONFIG.parent / document["protocol"]).read_text())
    rule = stopping_rule(protocol)
    recipes = {}
    cases = document["walk_forward"]["search_cases"]
    for variant in document["variants"]:
        for case in cases:
            recipe, loaded = candidate_run.load_recipe(CONFIG, variant=variant, search_case=case)
            assert recipe.epochs == rule["max_epochs"] == 30
            assert recipe.selection == {k: v for k, v in rule.items() if k != "max_epochs"}
            assert recipe.learning_rate == cases[case]["learning_rate"]
            assert loaded["status"] == "propuesta_sin_ejecutar"
            recipes[variant, case] = recipe
    principal = recipes[document["principal"], "lr1e-3"]
    assert (principal.admission, principal.refinements, principal.loss) == ("m1", 1, "pinball")
    assert recipes["m0_k1", "lr1e-3"].admission == "m0"
    assert {recipes[name, "lr1e-3"].refinements for name in ("m1_k2", "m1_k4")} == {2, 4}
    for case in cases:
        budget = {
            (r.epochs, r.update_instants, r.block_rows, r.learning_rate)
            for (_, name), r in recipes.items()
            if name == case
        }
        assert len(budget) == 1
    drifted = dict(document, protocol=str(CONFIG.parent / document["protocol"]))
    drifted["recipe"] = dict(document["recipe"], selection=dict(SELECTION, min_delta=0.0))
    path = tmp_path / "drifted.json"
    path.write_text(json.dumps(drifted))
    with pytest.raises(ValueError, match="protocolo"):
        candidate_run.load_recipe(path, variant="m1_k1", search_case="lr1e-3")
    with pytest.raises(ValueError):
        candidate_run.load_recipe(CONFIG, variant="m2_k1", search_case="lr1e-3")


def test_search_cases_match_titans_and_the_reader_and_cannot_be_skipped(tmp_path):
    # La GRU candidata busca las mismas dos tasas de aprendizaje que Titans-MAC y el lector.
    document = json.loads(CONFIG.read_text())
    titans = json.loads(
        (ROOT / "configs/titans/chronological-training-historical-masked.json").read_text()
    )
    assert document["walk_forward"]["search_cases"] == titans["walk_forward"]["search_cases"]
    assert document["walk_forward"]["warmup_months"] == titans["walk_forward"]["warmup_months"]
    assert "learning_rate" not in document["recipe"]
    for case in (None, "lr1e-2"):
        with pytest.raises(ValueError, match="casos de búsqueda"):
            candidate_run.load_recipe(CONFIG, variant="m1_k1", search_case=case)

    def written(change):
        changed = json.loads(json.dumps(document))
        changed["protocol"] = str(CONFIG.parent / document["protocol"])
        change(changed)
        path = tmp_path / "changed.json"
        path.write_text(json.dumps(changed))
        return path

    def variant_rate(changed):
        changed["variants"]["m1_k2"]["learning_rate"] = 0.01

    def base_rate(changed):
        changed["recipe"]["learning_rate"] = 0.001

    def other_keys(changed):
        changed["walk_forward"]["search_cases"]["lr1e-4"] = dict(max_grad_norm=0.5)

    def no_warmup(changed):
        del changed["walk_forward"]["warmup_months"]

    def long_warmup(changed):
        changed["walk_forward"]["warmup_months"] = 61

    for change in (variant_rate, base_rate, other_keys, no_warmup, long_warmup):
        with pytest.raises(ValueError):
            candidate_run.load_recipe(written(change), variant="m1_k1", search_case="lr1e-4")


@pytest.mark.parametrize("admission", ["m0", "m1"])
def test_roles_cover_the_parameters_and_m0_leaves_the_read_out(shared, tmp_path, admission):
    _, streams = shared
    engine = trainer(streams, tmp_path / "run", admission=admission)
    names = list(engine.model.named_parameters())
    covered = sorted(sum(engine.roles.values(), []))
    if admission == "m0":
        assert set(engine.inert) == READ and sorted(covered + engine.inert) == sorted(names)
    else:
        assert engine.inert == [] and covered == sorted(names)
    listed = {id(p) for g in engine.optimizer.param_groups for p in g["params"]}
    assert listed == {id(p) for p in engine.trainable}
    assert engine.identity["inert_parameters"] == engine.inert
    assert (engine.codec is None) == (admission == "m0")


def test_optimizer_must_cover_exactly_the_trainable_parameters(shared, tmp_path):
    _, streams = shared

    def partial(groups):
        return RecordingOptimizer(groups[:1])

    def extra(groups):
        model = adapter(streams, seed=7).model
        return RecordingOptimizer([*groups, dict(params=[model.named_parameters()["head_bias"]])])

    for factory in (partial, extra):
        with pytest.raises(ValueError, match="exactamente"):
            trainer(streams, tmp_path / "run", factory=factory)


@pytest.mark.parametrize("refinements", [1, 2, 4])
def test_gradients_reach_every_trainable_parameter(shared, tmp_path, refinements):
    _, streams = shared
    engine = trainer(streams, tmp_path / "m1", refinements=refinements)
    engine.run()
    records = named_records(engine)
    assert len(records) == engine.optimizer.calls > 3
    for name in engine.model.named_parameters():
        assert all(r[name] is None or bool(torch.isfinite(r[name]).all()) for r in records)
        assert any(active(r[name]) for r in records), name
    assert all(active(r["head_weight"]) and active(r["fusion_weight"]) for r in records)
    reference = trainer(streams, tmp_path / "m0", admission="m0", refinements=refinements)
    reference.run()
    records = named_records(reference)
    assert all(set(r) == set(engine.model.named_parameters()) - READ for r in records)
    for name in records[0]:
        assert all(r[name] is not None and bool(torch.isfinite(r[name]).all()) for r in records)
        assert any(active(r[name]) for r in records), name
    assert all(value.grad is None for name, value in reference.model.named_parameters().items())


@pytest.mark.parametrize("loss", ["pinball", "mae"])
def test_pinball_reaches_every_quantile_and_the_fallback_only_the_median(shared, tmp_path, loss):
    _, streams = shared
    engine = trainer(streams, tmp_path / loss, loss=loss)
    engine.run()
    for record in named_records(engine):
        rows = record["head_weight"].abs().sum(dim=1)
        if loss == "pinball":
            assert bool((rows > 0).all())
        else:
            assert rows[2] > 0 and bool((rows[[0, 1, 3, 4]] == 0).all())


def test_loop_uses_only_matured_labels_and_reads_the_bank_before_admitting(shared, tmp_path):
    _, streams = shared
    engine = trainer(streams, tmp_path / "run")
    report = engine.run()
    assert report["status"] == "completed"
    train = report["history"][1]["train"]
    updates = entries(engine.audit, "update")
    assert len(engine.optimizer.records) == len(updates) == train["updates"] > 3
    assert engine.optimizer.zero_calls == len(updates)
    used = [label for update in updates for label in update[3]]
    assert all(decision_at < maturity <= u[1] for u in updates for _, decision_at, maturity in u[3])
    assert len(used) == len(set(used)) == train["labels_in_loss"] == train["labels"]
    assert train["labels_without_graph"] == 0
    single = trainer(streams, tmp_path / "validation")
    single.evaluate(streams["validation"])
    for audit, partition in ((engine.audit, "train"), (single.audit, "validation")):
        admissions = entries(audit, "admit", partition)
        labels = entries(audit, "label", partition)
        assert admissions and sum(len(a[3]) for a in admissions) == len(labels)
        for prediction in entries(audit, "prediction", partition):
            at, seen = prediction[3], prediction[5]
            # La instantánea de un instante excluye las etiquetas que maduran en ese instante.
            assert seen == sum(len(a[3]) for a in admissions if a[2] < at)
        maturity = {(e[2], e[3]): e[4] for e in labels}
        assert all(decision_at < at for (_, decision_at), at in maturity.items())
    assert train["admitted"] == len(entries(engine.audit, "label", "train"))


def replay(engine, source, *, admission, refinements):
    """Repetir la fase con el consumidor congelado existente y un banco independiente."""
    adapter_type, consumer_type, codec_type = modules()
    frozen = adapter_type.restore(engine.adapter.export_state(), engine.specification)
    frozen.model.eval()
    for value in frozen.model.named_parameters().values():
        value.requires_grad_(False)
    consumer = consumer_type(frozen, refinements=refinements)
    codec = codec_type(frozen)
    bank = CandidateEpisodeBank(
        frozen.native,
        CandidateBankConfig(capacity=engine.recipe.bank_capacity, seed=engine.recipe.bank_seed),
        codec_id=codec.fingerprint(),
        representation_id=frozen.model.representation_id(),
        dtype=consumer.dtype,
        world="replay",
        partition=source.phase.partition,
        fold="replay",
    )
    pending, issued, quantiles = {}, {}, {}
    for event in source.batched_events(block_rows=engine.recipe.block_rows):
        memory = consumer.memory(bank.read_view() if admission == "m1" else None)
        matured = sorted(
            (at, flow, *pending.pop((flow, at)), value) for flow, at, value in event.labels
        )
        if event.at >= source.phase.decision_start:
            for raw in event.inputs:
                batch = validated_cpu_batch(raw, engine.specification)
                tensors = DecisionBatch.from_validated(batch, dtype=consumer.dtype)
                prepared = consumer.prepare(tensors, memory)
                encoded = codec.encode(batch)
                for row, (flow, at) in enumerate(
                    zip(batch.flow_ids, batch.prediction_at, strict=True)
                ):
                    issued[flow, at] = float(prepared.point_predictions[row])
                    quantiles[flow, at] = prepared.quantiles[row].numpy().copy()
                    pending[flow, at] = (
                        batch.input_available_at[row],
                        encoded.keys[row].copy(),
                        encoded.values[row].copy(),
                    )
        if admission == "m1" and matured:
            first = bank.seen + 1
            episodes = CandidateEpisodes(
                ids=torch.arange(first, first + len(matured), dtype=torch.int64),
                keys=torch.from_numpy(np.stack([m[3] for m in matured])),
                values=torch.from_numpy(np.stack([m[4] for m in matured])),
                times=torch.tensor([[m[0], m[2], event.at] for m in matured]),
                labels=torch.tensor([m[5] for m in matured], dtype=torch.float64),
            )
            bank = bank.propose(episodes, confirmed_at=event.at)
    return issued, quantiles, bank


@pytest.mark.parametrize(("admission", "refinements"), [("m0", 1), ("m1", 1), ("m1", 2)])
def test_validation_equals_the_frozen_consumer_replay(shared, tmp_path, admission, refinements):
    _, streams = shared
    engine = trainer(streams, tmp_path / "run", admission=admission, refinements=refinements)
    source = streams["validation"]
    destination = tmp_path / "validation.parquet"
    metrics = engine.evaluate(source, destination=destination)
    issued = {(e[2], e[3]): e[4] for e in entries(engine.audit, "prediction", "validation")}
    expected, quantiles, bank = replay(engine, source, admission=admission, refinements=refinements)
    assert len(issued) > 20 and issued == expected
    table = pq.read_table(destination)
    for sample, values in zip(
        table["sample_id"].to_pylist(),
        np.column_stack([table[c].to_numpy() for c in QUANTILE_COLUMNS]),
        strict=True,
    ):
        flow, at = sample.rsplit("/", 1)
        assert values.tobytes() == quantiles[flow, int(at)].tobytes()
    if admission == "m1":
        admitted = entries(engine.audit, "admit", "validation")
        assert admitted[-1][4] == bank.seen > bank.config.capacity
        assert metrics["admitted"] == bank.seen


def test_predictions_follow_the_common_contract(shared, tmp_path):
    _, streams = shared
    engine = trainer(streams, tmp_path / "run")
    path = tmp_path / "validation.parquet"
    metrics = engine.evaluate(streams["validation"], destination=path)
    table = pq.read_table(path)
    expected = [
        "sample_id",
        "asset_id",
        "market",
        "prediction_at",
        "target",
        "prediction",
        "zero",
        *QUANTILE_COLUMNS,
    ]
    assert table.column_names == expected and table.num_rows == metrics["samples"] > 0
    panel = ForecastPanel.from_arrow(table, quantile_columns=QUANTILE_COLUMNS, levels=LEVELS)
    assert panel.rows == table.num_rows
    assert table["prediction"].to_numpy().tobytes() == table["quantile_0500"].to_numpy().tobytes()
    assert all(
        sample.startswith(asset + "/")
        for sample, asset in zip(
            table["sample_id"].to_pylist(), table["asset_id"].to_pylist(), strict=True
        )
    )
    errors = SessionErrors()
    errors.update(
        table["market"].to_numpy(zero_copy_only=False),
        table["prediction_at"].cast("int64").to_numpy(),
        table["prediction"].to_numpy() - table["target"].to_numpy(),
    )
    assert errors.summary()["session_mae"] == pytest.approx(metrics["session_mae"], rel=1e-15)
    levels = torch.tensor(np.column_stack([table[c].to_numpy() for c in QUANTILE_COLUMNS]))
    assert bool((levels.diff(dim=1) >= 0).all())
    assert float(pinball_loss(levels, torch.tensor(table["target"].to_numpy()))) >= 0


def test_future_suffix_does_not_change_past_predictions_admissions_or_gradients(tmp_path):
    cut = decision("2022-12-09")
    results = []
    for name, perturb in (("base", None), ("future", cut)):
        _, streams = corpus(tmp_path / name, perturb_after=perturb)
        engine = trainer(streams, tmp_path / name / "run", update_instants=2)
        engine.run()
        results.append(engine)
    base, future = results

    def past(engine, kind, position):
        return [e for e in entries(engine.audit, kind, "train") if e[position] <= cut]

    assert past(base, "prediction", 3) == past(future, "prediction", 3)
    assert past(base, "admit", 2) == past(future, "admit", 2)
    assert entries(base.audit, "prediction", "train") != entries(
        future.audit, "prediction", "train"
    )
    steps = [u[2] for u in entries(base.audit, "update") if u[1] <= cut]
    assert steps and steps == [u[2] for u in entries(future.audit, "update") if u[1] <= cut]
    for left, right in zip(named_records(base)[: len(steps)], named_records(future), strict=False):
        assert left.keys() == right.keys()
        for key in left:
            if left[key] is None:
                assert right[key] is None
            else:
                torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)
    later = named_records(base)[len(steps)], named_records(future)[len(steps)]
    assert any(
        left is not None and not torch.equal(left, later[1][key]) for key, left in later[0].items()
    )


def test_refinements_do_not_count_as_memory_updates(shared, tmp_path):
    _, streams = shared
    reports = {}
    for refinements in (1, 4):
        engine = trainer(streams, tmp_path / str(refinements), refinements=refinements)
        reports[refinements] = engine.run()["history"][1]["train"]
    keys = ("admitted", "updates", "labels_in_loss", "segments", "predictions")
    assert {k: reports[1][k] for k in keys} == {k: reports[4][k] for k in keys}


def test_resume_after_interruption_reproduces_the_continuous_run(shared, tmp_path):
    _, streams = shared
    options = dict(epochs=2, checkpoint_updates=2)
    continuous = trainer(streams, tmp_path / "continuous", **options)
    expected = continuous.run()
    stop = StopAtStep(9)
    first = trainer(streams, tmp_path / "interrupted", **options)
    stop.trainer = first
    paused = first.run(stop=stop)
    assert paused["status"] == "paused"
    state = load_training_state(
        tmp_path / "interrupted/checkpoints", expected_identity=first.identity
    )
    assert state["cursor"]["phase"] == "train" and state["cursor"]["stage"] == "inputs"
    assert state["global_step"] >= 9 and state["optimizer"]["calls"] == state["global_step"]
    saved = state["run"]
    assert saved["bank"]["seen"] > 0 and saved["bank"]["reservoir_rng"]
    assert saved["staged"] is not None and len(saved["staged"]["ids"]) > 0
    assert state["parameters_sha256"] == first.model.parameter_fingerprint()
    second = trainer(streams, tmp_path / "interrupted", **options)
    resumed = second.run(resume=True)
    assert resumed["status"] == "completed"
    assert first.audit + second.audit == continuous.audit
    joined = named_records(first) + named_records(second)
    assert len(joined) == len(named_records(continuous)) == second.optimizer.calls
    for left, right in zip(joined, named_records(continuous), strict=True):
        for key in left:
            if left[key] is None:
                assert right[key] is None
            else:
                torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)
    assert resumed["history"] == expected["history"]
    assert resumed["best_epoch"] == expected["best_epoch"]
    for name in ("validation-predictions.parquet",):
        assert (tmp_path / "interrupted" / name).read_bytes() == (
            tmp_path / "continuous" / name
        ).read_bytes()
    retained = list((tmp_path / "interrupted/checkpoints").glob("state-*.pt"))
    assert 1 <= len(retained) <= 3


def test_hand_built_state_survives_a_checkpoint_round_trip(shared, tmp_path):
    _, streams = shared
    engine = trainer(streams, tmp_path / "run")
    run = candidate_run._Pass(bank=engine._new_bank("train"))
    generator = torch.Generator().manual_seed(5)

    def episodes(first, count, at):
        keys = torch.nn.functional.normalize(
            torch.randn(count, 128, generator=generator, dtype=torch.float64), dim=1
        )
        return CandidateEpisodes(
            ids=torch.arange(first, first + count, dtype=torch.int64),
            keys=keys,
            values=torch.randn(count, 256, generator=generator, dtype=torch.float64),
            times=torch.tensor([[at - 10 - i, at - 20 - i, at] for i in range(count)]),
            labels=torch.randn(count, generator=generator, dtype=torch.float64),
        )

    # Más episodios que plazas: el reservorio ya ha sustituido y su RNG no es el inicial.
    run.bank = run.bank.propose(episodes(1, 7, 1_000), confirmed_at=1_000)
    run.staged = episodes(8, 2, 2_000)
    for index, flow in enumerate(("US/A0000", "US/A0002")):
        run.pending[flow, 1_500 + index] = candidate_run._Pending(
            0.25 * index,
            available=1_400,
            key=run.staged.keys[index].clone(),
            value=run.staged.values[index].clone(),
        )
    run.errors.update(["US", "US"], [100, 200], [0.5, -0.25])
    run.counters.update(labels=2, admitted=7)
    state = dict(global_step=3, run=engine._export(run))
    save_training_state(tmp_path / "checkpoints", state, identity=engine.identity)
    loaded = load_training_state(tmp_path / "checkpoints", expected_identity=engine.identity)
    restored = engine._restore(loaded["run"])
    before, after = run.bank.snapshot(), restored.bank.snapshot()
    assert before.keys() == after.keys()
    for key, value in before.items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(value, after[key]), key
        else:
            assert value == after[key], key
    assert before["reservoir_rng"] != run.bank._native.causal_reservoir_state(73)
    for field in ("ids", "keys", "values", "times", "labels"):
        assert torch.equal(getattr(run.staged, field), getattr(restored.staged, field))
    assert restored.pending.keys() == run.pending.keys()
    for key, entry in run.pending.items():
        other = restored.pending[key]
        assert (entry.issued, entry.available) == (other.issued, other.available)
        assert torch.equal(entry.key, other.key) and torch.equal(entry.value, other.value)
    assert restored.errors.sessions == run.errors.sessions
    assert restored.counters == run.counters
    # Las propuestas posteriores sortean igual desde el estado original y el recuperado.
    following = episodes(10, 3, 3_000)
    left = run.bank.propose(following, confirmed_at=3_000).snapshot()
    right = restored.bank.propose(following, confirmed_at=3_000).snapshot()
    assert all(torch.equal(left[k], right[k]) for k in ("ids", "keys", "values", "labels"))
    run.graphs["US/A0000", 1] = torch.zeros(5)
    with pytest.raises(ValueError, match="barrera"):
        engine._export(run)


def test_completed_run_writes_predictions_and_restores_the_selected_state(shared, tmp_path):
    _, streams = shared
    engine = trainer(streams, tmp_path / "run", epochs=2)
    report = engine.run()
    assert report["status"] == "completed" and report["final_test_opened"] is False
    assert report["best_epoch"] == 0 and len(report["history"]) == 3
    record = report["predictions"]["validation"]
    path = tmp_path / "run" / record["path"]
    assert record["metrics"]["samples"] == pq.read_metadata(path).num_rows > 0
    fresh = adapter(streams)
    assert (
        candidate_run.restore_selected(tmp_path / "run", fresh)
        == (report["best_checkpoint"]["parameters_sha256"])
    )
    fresh.model.eval()
    for value in fresh.model.named_parameters().values():
        value.requires_grad_(False)
    consumer = modules()[1](fresh)
    assert consumer.model.parameter_fingerprint() == report["best_checkpoint"]["parameters_sha256"]
    other = adapter(streams, seed=7)
    with pytest.raises(ValueError, match="adaptador"):
        candidate_run.restore_selected(tmp_path / "run", other)


def test_resume_rejects_a_changed_recipe(shared, tmp_path):
    _, streams = shared
    engine = trainer(streams, tmp_path / "run", checkpoint_updates=1)
    stop = StopAtStep(2)
    stop.trainer = engine
    assert engine.run(stop=stop)["status"] == "paused"
    changed = trainer(streams, tmp_path / "run", refinements=2)
    with pytest.raises(ValueError, match="identidad"):
        changed.run(resume=True)
    with pytest.raises(ValueError):
        trainer(streams, tmp_path / "run").run()


@pytest.mark.parametrize("factory", ["recording", None])
def test_active_hold_stops_the_trainer_before_creating_outputs(
    shared, tmp_path, monkeypatch, learning_hold, factory
):
    _, streams = shared
    learning_hold(False)
    engine = trainer(streams, tmp_path / "run", **({} if factory else dict(factory=None)))

    def forbidden(*args, **kwargs):
        raise AssertionError("El entrenador llegó al paso del optimizador")

    # Sin esta sustitución, el gancho global de pruebas convertiría el paso en una omisión.
    monkeypatch.setattr(engine.optimizer, "step", forbidden)
    before = {name: value.clone() for name, value in engine.model.named_parameters().items()}
    with pytest.raises(LearningHoldError, match="GRU candidata"):
        engine.run()
    assert not (tmp_path / "run").exists()
    after = engine.model.named_parameters()
    assert all(torch.equal(before[name], after[name]) for name in before)


def test_rejects_incompatible_views_policies_and_scopes(shared, tmp_path):
    from tests.models.candidate.test_historical_inputs import specification as manual

    _, streams = shared
    strict = modules()[0](manual(historical=False), dtype=torch.float64)
    with pytest.raises(ValueError, match="histórica"):
        trainer(streams, tmp_path / "strict", model=strict)
    swapped = dict(train=streams["validation"], validation=streams["train"])
    with pytest.raises(ValueError, match="particiones"):
        trainer(swapped, tmp_path / "swapped")
    for heldout in (dict(calibration=streams["validation"]), dict(test=streams["validation"])):
        with pytest.raises(ValueError, match="particiones"):
            trainer(streams, tmp_path / "heldout", heldout=heldout)
    _, other = corpus(tmp_path / "other", perturb_after=decision("2022-12-09"))
    with pytest.raises(ValueError, match="particiones"):
        trainer(dict(train=streams["train"], validation=other["validation"]), tmp_path / "x")
    with pytest.raises(ValueError, match="entrada"):
        trainer(other, tmp_path / "y", model=adapter(streams))
    for scope in (dict(world=""), dict(fold="x" * 129)):
        with pytest.raises(ValueError, match="ámbito"):
            candidate_run.CandidateChronologicalTrainer(
                adapter(streams),
                candidate_run.CandidateRecipe(epochs=1, selection=SELECTION),
                train=streams["train"],
                validation=streams["validation"],
                output=tmp_path / "scope",
                optimizer_factory=RecordingOptimizer,
                **scope,
            )
    with pytest.raises(ValueError, match="fases declaradas"):
        trainer(streams, tmp_path / "run").evaluate(other["validation"])
    frozen = adapter(streams)
    frozen.model.named_parameters()["head_bias"].requires_grad_(False)
    with pytest.raises(ValueError, match="requerir gradiente"):
        trainer(streams, tmp_path / "frozen", model=frozen)


def test_admissions_follow_the_executor_order_and_identifiers(shared, tmp_path):
    _, streams = shared
    engine = trainer(streams, tmp_path / "run")
    bank = engine._new_bank("train")

    def pending(index):
        key = torch.zeros(128, dtype=torch.float64)
        key[index] = 1.0
        return candidate_run._Pending(
            0.0, available=10, key=key, value=torch.full((256,), float(index), dtype=torch.float64)
        )

    # Etiquetas que maduran juntas con decisiones distintas, en un orden de llegada cualquiera.
    admitted = [
        (30, "US/A0000", pending(0), 0.1),
        (20, "US/A0002", pending(1), 0.2),
        (20, "US/A0001", pending(2), 0.3),
    ]
    episodes = engine._episodes(bank, admitted, 40)
    assert episodes.ids.tolist() == [1, 2, 3]
    assert episodes.labels.tolist() == [0.3, 0.2, 0.1]
    assert episodes.times.tolist() == [[20, 10, 40], [20, 10, 40], [30, 10, 40]]
    following = engine._episodes(bank.propose(episodes, confirmed_at=40), admitted[:1], 50)
    assert following.ids.tolist() == [4]


def test_clipping_bounds_every_recorded_gradient(shared, tmp_path):
    _, streams = shared
    bound = 1e-6
    engine = trainer(streams, tmp_path / "run", max_grad_norm=bound)
    engine.run()
    for record in named_records(engine):
        norms = [value.norm() for value in record.values() if value is not None]
        assert float(torch.stack(norms).norm()) <= bound * (1 + 1e-6)


def test_saved_parameters_replace_the_live_module_values(shared, tmp_path):
    _, streams = shared
    engine = trainer(streams, tmp_path / "run", checkpoint_updates=1)
    stop = StopAtStep(2)
    stop.trainer = engine
    assert engine.run(stop=stop)["status"] == "paused"
    saved = engine.model.parameter_fingerprint()
    resumed = trainer(streams, tmp_path / "run", checkpoint_updates=1)
    other = adapter(streams, seed=7).model.named_parameters()
    # Valores escritos a mano después de fijar la identidad, sin pasos de optimizador.
    with torch.no_grad():
        for name, value in resumed.model.named_parameters().items():
            value.copy_(other[name])
    assert resumed.model.parameter_fingerprint() != saved
    assert resumed.run(resume=True)["status"] == "completed"
    assert resumed.model.parameter_fingerprint() == saved
    fresh = adapter(streams)
    with torch.no_grad():
        for name, value in fresh.model.named_parameters().items():
            value.copy_(other[name])
    assert candidate_run.restore_selected(tmp_path / "run", fresh) == saved
    assert fresh.model.parameter_fingerprint() == saved


def test_labels_without_a_live_graph_stay_out_of_the_loss(shared, tmp_path):
    from mars_titan.memory.financial_observations import ObservationEvent

    _, streams = shared
    engine = trainer(streams, tmp_path / "run", admission="m0")
    run = candidate_run._Pass()
    graph = torch.zeros(5, dtype=torch.float64, requires_grad=True)
    run.pending["US/A0000", 100] = candidate_run._Pending(0.5)
    run.pending["US/A0001", 100] = candidate_run._Pending(0.25, block=0)
    run.graphs["US/A0001", 100] = graph
    run.outstanding[0] = 1
    labels = (("US/A0000", 100, 0.125), ("US/A0001", 100, -0.125))
    event = ObservationEvent(200, (), labels, False)
    engine._labels(run, streams["train"], event, train=True)
    assert run.counters["labels"] == 2 and run.counters["labels_without_graph"] == 1
    assert len(run.predictions) == 1 and run.predictions[0] is graph
    assert run.targets == [-0.125] and run.used == [("US/A0001", 100, 200)]
    assert not run.pending and not run.graphs and not run.outstanding
    with pytest.raises(ValueError, match="pendiente"):
        engine._labels(run, streams["train"], event, train=True)


def test_heldout_partitions_write_the_common_prediction_files(tmp_path):
    from mars_titan.data.input_policy import HISTORICAL_MASKED
    from mars_titan.memory import financial_observations as api
    from mars_titan.memory.financial_session import FinancialPhase
    from mars_titan.training.corpus_inputs import CorpusDataset
    from tests.training.historical_temporal_fixture import historical_temporal_fixture
    from tests.training.test_historical_temporal import prepare

    prepare(historical_temporal_fixture(tmp_path / "source"), tmp_path / "views")
    dataset = CorpusDataset(
        tmp_path / "views/fold-000/manifest.json", input_policy=HISTORICAL_MASKED
    )
    streams = {}
    for name, start, end, _ in dataset.temporals["US"].partitioner.bounds:
        phase = FinancialPhase(name, int(start), int(start), int(end), int(end))
        index = api.prepare_observation_index(dataset, tmp_path / f"index-{name}", phase=phase)
        streams[name] = api.FinancialObservationSource(dataset, index)
    heldout = {name: streams[name] for name in candidate_run.HELDOUT}
    engine = trainer(streams, tmp_path / "run", heldout=heldout, update_instants=1)
    report = engine.run()
    assert report["status"] == "completed"
    assert set(report["predictions"]) == {"validation", "calibration", "evaluation"}
    for name, record in report["predictions"].items():
        table = pq.read_table(tmp_path / "run" / record["path"])
        panel = ForecastPanel.from_arrow(table, quantile_columns=QUANTILE_COLUMNS, levels=LEVELS)
        assert panel.rows == record["metrics"]["samples"] >= 1
        phase = streams[name].phase
        moments = table["prediction_at"].cast("int64").to_numpy()
        assert ((moments >= phase.decision_start) & (moments < phase.decision_end)).all()
    assert engine.identity["sources"].keys() == set(streams)
    swapped = dict(calibration=heldout["evaluation"], evaluation=heldout["calibration"])
    with pytest.raises(ValueError, match="particiones"):
        trainer(streams, tmp_path / "swapped", heldout=swapped)
