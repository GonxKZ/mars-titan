"""Pruebas del control en línea de transformer_compact en CPU, sin pasos que cambien pesos.

El ancla es una referencia Transformer seleccionada con los pesos iniciales de la semilla y
el banco es una ventana MARS-TITAN M1 ajustada con el registrador de gradientes sobre un
padre Titans-MAC. El optimizador del control solo registra llamadas y gradientes. Se
comprueban el orden entre emisión y aprendizaje, las etiquetas del banco, el tope, la
paridad con tope cero, la precisión y los rechazos.
"""

import json
import math
import shutil
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from mars_titan.data.input_policy import HISTORICAL_MASKED
from mars_titan.data.storage import sha256
from mars_titan.memory.financial_session import FinancialPhase
from mars_titan.models.quantile_head import MEDIAN_INDEX
from mars_titan.training import masked_campaign as engine
from mars_titan.training import online_reference as online
from mars_titan.training.campaign_plan import ONLINE
from mars_titan.training.carried_predictions import reference_model, selected_reference
from mars_titan.training.checkpoints import StopRequest
from mars_titan.training.corpus_inputs import CorpusDataset
from mars_titan.training.titans_walk_forward import _sources
from tests.training.test_carried_predictions import cpu as cpu
from tests.training.test_carried_predictions import neural_anchor
from tests.training.test_mars_titan_walk_forward import M1, mars, readout_recipe
from tests.training.test_titans_walk_forward import (
    learning_doubles_module as learning_doubles_module,
)
from tests.training.test_titans_walk_forward import recipe, unfused_attention, views, window

RULE = dict(
    optimizer="sgd",
    learning_rate=1e-3,
    block_rows=2,
    update_every=1,
    max_grad_norm=1.0,
    update_cap="episodic_bank_writes",
)


class Recorder:
    """Registra pasos y normas de gradiente sin modificar pesos."""

    def __init__(self, parameters, rule):
        self.parameters, self.rule, self.steps, self.norms = list(parameters), rule, 0, []

    def zero_grad(self, set_to_none=True):
        assert set_to_none is True
        for value in self.parameters:
            value.grad = None

    def step(self):
        self.steps += 1
        grads = [value.grad for value in self.parameters if value.grad is not None]
        assert grads
        self.norms.append(float(torch.sqrt(sum((g.double() ** 2).sum() for g in grads))))


class Factory:
    def __init__(self):
        self.instances = []

    def __call__(self, parameters, rule):
        self.instances.append(Recorder(parameters, rule))
        return self.instances[-1]


@pytest.fixture(autouse=True)
def strict_precision():
    previous = (
        torch.get_float32_matmul_precision(),
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
    )
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    yield
    torch.set_float32_matmul_precision(previous[0])
    torch.backends.cuda.matmul.allow_tf32 = previous[1]
    torch.backends.cudnn.allow_tf32 = previous[2]


@pytest.fixture(autouse=True)
def permitted_doubles(learning_doubles):
    return learning_doubles


@pytest.fixture(scope="module")
def base(tmp_path_factory, learning_doubles_module):
    root = tmp_path_factory.mktemp("online-reference")
    view, protocol = views(root / "base")
    with unfused_attention():
        window(view, protocol, recipe(root), root / "runs" / "titans")
        report, _ = mars(view, root / "runs" / "titans", readout_recipe(root), root / "m1", M1)
    assert report["status"] == "completed"
    return SimpleNamespace(root=root, view=view, bank=root / "m1", report=report)


@pytest.fixture
def anchor(base, tmp_path, cpu):
    neural_anchor(tmp_path / "anchor", base.view, kind="transformer")
    return tmp_path / "anchor"


def run(base, anchor, output, **options):
    factory = options.pop("optimizer_factory", None) or Factory()
    report = online.run_online_reference(
        anchor,
        base.view,
        options.pop("bank", base.bank),
        output,
        rule=options.pop("rule", RULE),
        input_policy=HISTORICAL_MASKED,
        reader_rows=2,
        optimizer_factory=factory,
        **options,
    )
    return report, factory


def sources(base, folder):
    dataset = CorpusDataset(base.view, input_policy=HISTORICAL_MASKED)
    phases = {
        name: FinancialPhase(**base.report["identity"]["phases"][name])
        for name in online.PARTITIONS
    }
    return dataset, _sources(dataset, phases, folder)


def model_of(anchor, view):
    report, state = selected_reference(anchor, view, input_policy=HISTORICAL_MASKED)
    return report["identity"], *reference_model(report["identity"], state, torch.device("cpu"))


def passed(base, anchor, folder, partition, *, cap, rule=RULE, source=None):
    identity, model, quantiles = model_of(anchor, base.view)
    _, built = sources(base, folder)
    recorder, audit = Recorder(model.parameters(), rule), []
    rows, metrics = online.online_pass(
        model,
        source or built[partition],
        rule=rule,
        cap=cap,
        case=identity["case"],
        quantiles=quantiles,
        optimizer=recorder,
        device=torch.device("cpu"),
        reader_rows=2,
        audit=audit,
    )
    return rows, metrics, recorder, audit


def frozen(anchor, view, partition):
    """Devuelve las predicciones del estado elegido sin actualizar, por activo e instante."""
    _, model, _ = model_of(anchor, view)
    model.eval()
    dataset = CorpusDataset(view, input_policy=HISTORICAL_MASKED)
    result = {}
    with torch.no_grad():
        for batch in dataset.batches(partition=partition, batch_size=3, epoch=0, seed=0):
            emitted = model(
                {k: torch.from_numpy(v) for k, v in batch["inputs"].items()},
                torch.from_numpy(batch["presence"]),
            )
            moments = batch["prediction_at"].astype("datetime64[us]").astype(np.int64)
            for sample, moment, value in zip(
                batch["sample_ids"], moments, emitted[:, MEDIAN_INDEX].numpy(), strict=True
            ):
                result[sample.rsplit("/", 1)[0], int(moment)] = float(value)
    return result


def predictions(rows):
    table = pa.concat_tables(rows.finish())
    moments = table["prediction_at"].cast(pa.int64()).to_pylist()
    return dict(
        zip(
            zip(table["asset_id"].to_pylist(), moments, strict=True),
            table["prediction"].to_pylist(),
            strict=True,
        )
    )


@pytest.mark.parametrize("partition", online.PARTITIONS)
def test_cap_zero_keeps_the_predictions_of_the_frozen_reference(base, anchor, tmp_path, partition):
    rows, metrics, recorder, audit = passed(base, anchor, tmp_path / "idx", partition, cap=0)
    assert recorder.steps == metrics["updates"] == metrics["labels_used"] == 0
    assert metrics["labels_beyond_cap"] == metrics["labels"] > 0
    assert not [entry for entry in audit if entry[0] == "step"]
    emitted, expected = predictions(rows), frozen(anchor, base.view, partition)
    assert set(emitted) == set(expected)
    for key, value in expected.items():
        assert math.isclose(emitted[key], value, rel_tol=1e-6, abs_tol=1e-7), key


@pytest.mark.parametrize("update_every", [1, 2])
def test_every_step_uses_labels_matured_after_their_prediction(
    base, anchor, tmp_path, update_every
):
    rule = dict(RULE, update_every=update_every)
    _, metrics, recorder, audit = passed(
        base, anchor, tmp_path / "idx", "evaluation", cap=10**6, rule=rule
    )
    emitted, matured, instants, used, stepped = {}, {}, [], [], set()
    for entry in audit:
        kind, at = entry[:2]
        if kind == "predict":
            # Las predicciones de un instante se emiten antes de sus pasos.
            assert at not in stepped
            for key in entry[2]:
                emitted.setdefault((key, at), at)
        elif kind == "label":
            decision = entry[2], entry[3]
            assert emitted[decision] == decision[1] < at
            matured[decision] = at
            if not instants or instants[-1] != at:
                instants.append(at)
        else:
            refs = entry[2]
            assert 1 <= len(refs) <= rule["block_rows"]
            # Cada paso usa etiquetas ya maduras, de predicciones emitidas antes.
            assert all(emitted[ref] < matured[ref] <= at for ref in refs)
            assert instants.index(at) % update_every == update_every - 1
            stepped.add(at)
            used += refs
    assert len(used) == len(set(used)) == metrics["labels_used"]
    assert recorder.steps == metrics["updates"] > 0
    assert metrics["update_instants"] == metrics["maturity_instants"] // update_every
    assert metrics["labels_used"] + metrics["labels_after_last_update"] == metrics["labels"]


def test_the_cap_limits_the_labels_used(base, anchor, tmp_path):
    _, metrics, recorder, audit = passed(base, anchor, tmp_path / "idx", "evaluation", cap=3)
    steps = [entry[2] for entry in audit if entry[0] == "step"]
    assert sum(len(refs) for refs in steps) == metrics["labels_used"] == 3
    assert recorder.steps == len(steps) and all(len(refs) <= 2 for refs in steps)
    assert metrics["labels_beyond_cap"] == metrics["labels"] - 3


class Shifted:
    """Envuelve el índice real y adelanta una etiqueta al instante de su decisión o antes."""

    def __init__(self, source, offset):
        self.source, self.offset = source, offset
        self.phase, self.metadata = source.phase, source.metadata

    def label_decisions(self):
        return self.source.label_decisions()

    def batched_events(self, **options):
        events = list(self.source.batched_events(**options))
        position = {event.at: index for index, event in enumerate(events)}
        origin, moved = next(
            (index, label)
            for index, event in enumerate(events)
            for label in event.labels
            if position[label[1]] + self.offset >= 0
        )
        target = position[moved[1]] + self.offset
        events[origin] = replace(
            events[origin], labels=tuple(x for x in events[origin].labels if x != moved)
        )
        events[target] = replace(events[target], labels=(*events[target].labels, moved))
        return iter(events)


class Unpredicted(Shifted):
    """Envuelve el índice real y añade la etiqueta de una decisión que nunca se predijo."""

    def batched_events(self, **options):
        events = list(self.source.batched_events(**options))
        index = next(i for i, event in enumerate(events) if event.labels)
        key, decision_at, value = events[index].labels[0]
        extra = ("US/NUNCA", decision_at, value)
        events[index] = replace(events[index], labels=(*events[index].labels, extra))
        return iter(events)


class Unlabelled(Shifted):
    """Envuelve el índice real y quita la etiqueta de una decisión, que se predice sin guardarse."""

    def __init__(self, source):
        super().__init__(source, 0)
        events = list(source.batched_events(start_cursor=0, block_rows=2))
        self.dropped = next(label for event in events for label in event.labels)[:2]

    def label_decisions(self):
        key, decision_at = self.dropped
        decisions = dict(self.source.label_decisions())
        decisions[key] = decisions[key][decisions[key] != decision_at]
        return decisions

    def batched_events(self, **options):
        for event in self.source.batched_events(**options):
            labels = tuple(x for x in event.labels if x[:2] != self.dropped)
            yield replace(event, labels=labels)


def test_a_label_of_a_decision_never_predicted_is_refused(base, anchor, tmp_path):
    _, built = sources(base, tmp_path / "idx")
    with pytest.raises(ValueError, match="predicción emitida antes de su maduración"):
        passed(
            base,
            anchor,
            tmp_path / "idx",
            "evaluation",
            cap=10**6,
            source=Unpredicted(built["evaluation"], 0),
        )


def test_a_decision_without_label_in_the_partition_is_predicted_and_not_kept(
    base, anchor, tmp_path
):
    _, built = sources(base, tmp_path / "idx")
    _, full, _, _ = passed(base, anchor, tmp_path / "idx", "evaluation", cap=10**6)
    source = Unlabelled(built["evaluation"])
    _, metrics, _, audit = passed(
        base, anchor, tmp_path / "idx", "evaluation", cap=10**6, source=source
    )
    assert metrics["labels"] == full["labels"] - 1
    assert metrics["predictions"] == full["predictions"]
    assert any(source.dropped[0] in e[2] for e in audit if e[:2] == ("predict", source.dropped[1]))


@pytest.mark.parametrize("offset", [0, -1])
def test_a_label_moved_before_its_prediction_is_refused(base, anchor, tmp_path, offset):
    _, built = sources(base, tmp_path / "idx")
    with pytest.raises(ValueError, match="predicción emitida antes de su maduración"):
        passed(
            base,
            anchor,
            tmp_path / "idx",
            "evaluation",
            cap=10**6,
            source=Shifted(built["evaluation"], offset),
        )


def test_the_window_writes_the_bank_rows_with_its_labels_and_cap(base, anchor, tmp_path):
    report, factory = run(base, anchor, tmp_path / "online")
    assert report["status"] == "completed" and report["final_test_opened"] is False
    assert report["kind"] == online.KIND and report["rule"] == RULE
    assert report["numerics"] == online.STRICT_FP32
    assert (
        report["anchor"]["checkpoint_sha256"]
        == json.loads((anchor / "run.json").read_text())["checkpoint"]["sha256"]
    )
    assert report["indices"] == {
        name: base.report["identity"]["indices"][name] for name in online.PARTITIONS
    }
    assert [item.steps for item in factory.instances] == [
        report["predictions"][name]["metrics"]["updates"] for name in online.PARTITIONS
    ]
    # Cada tramo parte de un modelo nuevo con el estado elegido.
    first, second = (item.parameters for item in factory.instances)
    assert all(a is not b and torch.equal(a, b) for a, b in zip(first, second, strict=True))
    for name in online.PARTITIONS:
        bank = base.report["predictions"][name]
        metrics = report["predictions"][name]["metrics"]
        assert metrics["labels"] == bank["metrics"]["labels"]
        assert metrics["cap"] == report["bank"]["admitted"][name] == bank["metrics"]["admitted"]
        assert metrics["labels_used"] <= metrics["cap"]
        rows = pq.read_table(tmp_path / "online" / f"{name}-predictions.parquet")
        bank_rows = pq.read_table(base.bank / bank["path"])
        for column in ("asset_id", "market", "prediction_at", "target"):
            assert sorted(rows[column].to_pylist()) == sorted(bank_rows[column].to_pylist())
        assert report["predictions"][name]["sha256"] == sha256(
            tmp_path / "online" / f"{name}-predictions.parquet"
        )


@pytest.mark.parametrize(
    "flags",
    [("highest", False, True), ("highest", True, False), ("high", False, False)],
    ids=str,
)
def test_reduced_precision_is_refused_before_reading_sources(base, anchor, tmp_path, flags):
    torch.set_float32_matmul_precision(flags[0])
    torch.backends.cuda.matmul.allow_tf32 = flags[1]
    torch.backends.cudnn.allow_tf32 = flags[2]
    with pytest.raises(ValueError, match="FP32 estricto"):
        run(base, anchor, tmp_path / "online")
    assert not (tmp_path / "online").exists()


def test_autocast_is_refused():
    with torch.autocast("cpu", dtype=torch.bfloat16), pytest.raises(ValueError, match="autocast"):
        online.strict_fp32()


@pytest.mark.parametrize("backend", ["generic", "matmul", "rnn", "conv"])
def test_tf32_requested_through_the_new_interface_is_refused(backend):
    backends = torch.backends
    target = dict(
        generic=backends,
        matmul=backends.cuda.matmul,
        rnn=backends.cudnn.rnn,
        conv=backends.cudnn.conv,
    )[backend]
    target.fp32_precision = "tf32"
    try:
        with pytest.raises(ValueError, match="FP32 estricto"):
            online.strict_fp32()
    finally:
        target.fp32_precision = "ieee"
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    assert online.strict_fp32() == online.STRICT_FP32


def test_a_generic_tf32_request_is_refused_even_when_every_backend_overrides_it():
    """Los indicadores antiguos se leen estrictos, así que solo la interfaz nueva lo detecta."""
    backends = torch.backends
    every = (backends.cuda.matmul, backends.cudnn, backends.cudnn.conv, backends.cudnn.rnn)
    for backend in every:
        backend.fp32_precision = "ieee"
    backends.fp32_precision = "tf32"
    try:
        assert (
            torch.get_float32_matmul_precision(),
            backends.cuda.matmul.allow_tf32,
            backends.cudnn.allow_tf32,
        ) == ("highest", False, False)
        with pytest.raises(ValueError, match="FP32 estricto"):
            online.strict_fp32()
    finally:
        backends.fp32_precision = "ieee"
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    assert online.strict_fp32() == online.STRICT_FP32


@pytest.mark.parametrize(
    "changes",
    [
        dict(learning_rate="pending"),
        dict(block_rows=0),
        dict(update_every=0),
        dict(max_grad_norm=math.inf),
        dict(optimizer="adamw"),
        dict(update_cap="steps"),
        dict(momentum=0.9),
    ],
    ids=str,
)
def test_the_rule_is_declared_without_pending_values(changes):
    with pytest.raises(ValueError, match="sin valores pendientes"):
        online.checked_rule(dict(RULE, **changes))
    assert online.checked_rule(dict(RULE)) == RULE


def bank_copy(base, folder, change):
    folder.mkdir(parents=True)
    report = json.loads((base.bank / "run.json").read_text())
    change(report)
    (folder / "run.json").write_text(json.dumps(report))
    return folder


@pytest.mark.parametrize(
    "change",
    [
        lambda r: r["request"].update(seed=43),
        lambda r: r["identity"]["variant"].update(components={"episodic_bank": "m3"}),
        lambda r: r.update(status="running"),
        lambda r: r["request"].update(view_sha256="0" * 64),
    ],
)
def test_the_bank_must_be_m1_in_the_same_view_and_seed(base, anchor, tmp_path, change):
    bank = bank_copy(base, tmp_path / "bank", change)
    with pytest.raises(ValueError, match="mars_titan_m1 con la misma vista y semilla"):
        run(base, anchor, tmp_path / "online", bank=bank)
    assert not (tmp_path / "online").exists()


def test_indices_and_labels_must_be_those_of_the_bank(base, anchor, tmp_path):
    other = bank_copy(
        base,
        tmp_path / "bank-index",
        lambda r: r["identity"]["indices"].update(evaluation="0" * 64),
    )
    with pytest.raises(ValueError, match="no es el del banco"):
        run(base, anchor, tmp_path / "a", bank=other)
    labels = bank_copy(
        base,
        tmp_path / "bank-labels",
        lambda r: r["predictions"]["calibration"]["metrics"].update(labels=10**6),
    )
    with pytest.raises(ValueError, match="mismas etiquetas maduras"):
        run(base, anchor, tmp_path / "b", bank=labels)


def test_only_a_transformer_reference_starts_the_control(base, tmp_path, cpu):
    neural_anchor(tmp_path / "gru", base.view, kind="gru")
    with pytest.raises(ValueError, match="parte de transformer_compact"):
        run(base, tmp_path / "gru", tmp_path / "online")


def test_the_engine_runs_the_control_and_pauses_between_instants(
    base, anchor, tmp_path, monkeypatch
):
    assert engine.EXECUTORS["neural", ONLINE] == dict(
        run=engine._online, device="cuda", resumable=False, report="online.json"
    )
    job_run = engine.JobRun(
        job=dict(id="US/fold-000/transformer_compact_online/online-s42"),
        case=dict(rule=RULE),
        view=base.view,
        view_sha256=sha256(base.view),
        folder=tmp_path / "attempt",
        policy=HISTORICAL_MASKED,
        batch_size=2,
        checkpoint_seconds=300,
        stop=StopRequest(),
        anchor=dict(folder=anchor, view=base.view),
        bank=dict(folder=base.bank),
    )
    stopped = StopRequest()
    stopped.requested = True
    with pytest.raises(engine.Paused):
        engine._online(replace(job_run, folder=tmp_path / "paused", stop=stopped))
    # El ejecutor de la campaña no recibe fábrica, así que el registrador sustituye al SGD.
    created = Factory()
    monkeypatch.setattr(online, "_sgd", created)
    report = engine._online(job_run)
    assert report["status"] == "completed" and len(created.instances) == 2


def campaign_state(tmp_path, view):
    state = object.__new__(engine._Campaign)
    state.campaign = dict(
        online_controls=dict(
            arms=dict(
                transformer_compact_online=dict(
                    parent_arm="transformer_compact", cap_arm="mars_titan_m1"
                )
            )
        )
    )
    state.output = tmp_path
    state.views = dict(US=dict(windows={"fold-000": dict(path=str(view), sha256="v")}))
    state.receipts = {
        "US/fold-000/transformer_compact/finalist-s43": dict(attempt="t/attempt-0001", sha256="t"),
        "US/fold-000/mars_titan_m1/finalist-s43": dict(attempt="m/attempt-0001", sha256="m"),
        "US/fold-000/mars_titan_m1_k2/finalist-s43": dict(attempt="k/attempt-0001", sha256="k"),
    }
    return state


def test_the_engine_resolves_the_anchor_and_the_bank_of_the_same_window_and_seed(tmp_path):
    state = campaign_state(tmp_path, tmp_path / "view.json")
    job = dict(
        id="US/fold-000/transformer_compact_online/online-s43",
        scope="US",
        window="fold-000",
        arm="transformer_compact_online",
        seed=43,
        kind=ONLINE,
        stage=ONLINE,
        depends=list(state.receipts),
        case=dict(rule=RULE),
    )
    case, anchor, sources = state.resolve(job)
    assert case == dict(rule=RULE)
    assert anchor == dict(
        folder=tmp_path / "t/attempt-0001",
        view=tmp_path / "view.json",
        job="US/fold-000/transformer_compact/finalist-s43",
        sha256="t",
    )
    assert sources == dict(
        source="US/fold-000/transformer_compact/finalist-s43",
        source_sha256="t",
        bank="US/fold-000/mars_titan_m1/finalist-s43",
        bank_sha256="m",
    )
    assert state.bank_of(job)["folder"] == tmp_path / "m/attempt-0001"
    with pytest.raises(ValueError, match="no depende de los estados elegidos"):
        state.resolve(dict(job, depends=[d for d in job["depends"] if "mars_titan_m1/" not in d]))
    assert state.bank_of(dict(job, kind="carry")) is None
    state.receipts[job["id"]] = dict(attempt="o/attempt-0001", sha256="o")
    assert state.selected("US", "fold-000", job["arm"], 43) == (
        job["id"],
        state.receipts[job["id"]],
    )


def test_label_decisions_come_from_the_maturity_events_of_the_index(base, tmp_path):
    _, built = sources(base, tmp_path / "idx")
    for name, source in built.items():
        decisions = source.label_decisions()
        found = {}
        for event in source.events():
            for key, decision_at, _ in event.labels:
                found.setdefault(key, []).append(decision_at)
        assert {k: sorted(v) for k, v in found.items()} == {
            k: v.tolist() for k, v in decisions.items()
        }, name
        assert (
            sum(len(v) for v in decisions.values())
            == (base.report["predictions"][name]["metrics"]["labels"])
        )


def test_a_copied_output_is_never_reused(base, anchor, tmp_path):
    run(base, anchor, tmp_path / "online")
    shutil.copytree(tmp_path / "online", tmp_path / "copy")
    with pytest.raises(ValueError, match="directorio nuevo"):
        run(base, anchor, tmp_path / "copy")
