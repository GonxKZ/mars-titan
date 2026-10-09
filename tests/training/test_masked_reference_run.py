"""Runner de referencias con máscaras en CPU, con un optimizador que no cambia pesos.

Estas pruebas recorren lectura, forward, pérdida, backward, checkpoint, recuperación
y retención. El optimizador inyectado solo registra gradientes y nunca modifica pesos.
CUDA queda sustituida por CPU únicamente dentro de la prueba, de forma explícita.
"""

import hashlib
import importlib
import json
import math
from types import SimpleNamespace

import numpy as np
import pyarrow.parquet as pq
import pytest
import torch

from mars_titan.data.input_policy import HISTORICAL_MASKED, MODALITIES, STRICT_INPUTS
from mars_titan.data.input_policy import policy_identity as input_identity
from mars_titan.data.storage import sha256
from mars_titan.models.baselines.multimodal import PRESENCE_FUSION
from mars_titan.training import checkpoints
from mars_titan.training.checkpoints import StopRequest, load_training_state
from mars_titan.training.corpus_inputs import CorpusDataset
from tests.training.historical_temporal_fixture import historical_temporal_fixture
from tests.training.test_historical_temporal import prepare
from tests.training.test_reference_run import training_corpus

engine = importlib.import_module("mars_titan.training.reference_run")
HELDOUT = ("validation", "calibration", "evaluation")
LEGACY_IDENTITY = {
    "torch",
    "cuda",
    "numpy",
    "pyarrow",
    "exchange_calendars",
    "python",
    "gpu",
    "cublas_workspace",
    "numerics",
    "code",
    "manifest_sha256",
    "case",
    "model_family",
    "batch_size",
    "dimensions",
    "context",
    "initialization",
    "weighting",
    "market_weights",
    "optimizer",
    "precision",
    "device",
    "input_cache",
}


def recording_optimizer(record):
    """Construir un sustituto de AdamW que registra gradientes y nunca actualiza pesos.

    No hereda de torch.optim.Optimizer porque no aplica ninguna actualización. Así la
    guarda global del bloqueo no necesita omitir estas pruebas y el contrato se comprueba
    aquí: cada llamada exige pesos idénticos a los recibidos al construirlo.
    """

    class RecordingOptimizer:
        def __init__(self, parameters, lr):
            self.parameters, self.lr, self.steps = list(parameters), lr, 0
            self.initial = [value.detach().clone() for value in self.parameters]
            record.optimizers += 1

        def zero_grad(self, set_to_none=True):
            if set_to_none is not True:
                raise AssertionError("El runner libera los gradientes entre lotes")
            for value in self.parameters:
                value.grad = None

        @torch.no_grad()
        def step(self):
            values = self.parameters
            assert all(torch.equal(a, b) for a, b in zip(values, self.initial, strict=True))
            record.calls.append(
                [None if p.grad is None else p.grad.detach().clone() for p in values]
            )
            self.steps += 1

        def state_dict(self):
            return dict(kind="recording_without_updates", lr=self.lr, steps=self.steps)

        def load_state_dict(self, state):
            if state.get("kind") != "recording_without_updates" or state.get("lr") != self.lr:
                raise AssertionError("El estado no corresponde al sustituto sin actualizaciones")
            self.steps = state["steps"]

    return RecordingOptimizer


@pytest.fixture
def cpu_runner(monkeypatch, learning_doubles):
    """Sustituir CUDA y AdamW solo en la prueba, sin aplicar actualizaciones.

    `learning_doubles` deja pasar la entrada de run_reference_case, que la protección del
    aprendizaje detendría. El sustituto de AdamW nunca modifica pesos.
    """
    record = SimpleNamespace(calls=[], optimizers=0)
    RecordingOptimizer = recording_optimizer(record)

    deterministic = torch.are_deterministic_algorithms_enabled()
    threads = torch.get_num_threads()
    for module in ("reference_run", "reference_campaign", "reference_search"):
        target = importlib.import_module(f"mars_titan.training.{module}")
        monkeypatch.setattr(target, "require_cuda", lambda **_: torch.device("cpu"))
    monkeypatch.setattr(torch.optim, "AdamW", RecordingOptimizer)
    monkeypatch.setattr(engine, "capture_rng", lambda _: checkpoints.capture_rng("cpu"))
    monkeypatch.setattr(
        engine, "restore_rng", lambda state, _: checkpoints.restore_rng(state, "cpu")
    )
    for name, value in dict(
        synchronize=None,
        reset_peak_memory_stats=None,
        max_memory_allocated=0,
        max_memory_reserved=0,
        get_device_name="cpu-technical-fixture",
    ).items():
        monkeypatch.setattr(torch.cuda, name, lambda *_, value=value: value)
    monkeypatch.setattr(torch, "set_num_threads", lambda _: None)
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.delenv("MARS_TITAN_INPUT_CACHE_MIB", raising=False)
    yield record
    torch.use_deterministic_algorithms(deterministic)
    torch.set_num_threads(threads)


def masked_view(tmp_path):
    fixture = historical_temporal_fixture(tmp_path / "source")
    prepare(fixture, tmp_path / "views")
    return tmp_path / "views/fold-000/manifest.json"


def architecture(kind):
    value = dict(hidden_size=32, layers=1, dropout=0.0)
    if kind == "transformer":
        value["transformer"] = dict(heads=2, feedforward_multiplier=2)
    return value


def masked_case(kind="gru", *, epochs=2, stopping="fixed_budget", patience=1):
    return dict(
        kind=kind,
        loss="mse",
        learning_rate=0.001,
        seed=42,
        epochs=epochs,
        huber_delta=0.01,
        architecture=architecture(kind),
        selection=dict(metric="session_mae", patience=patience, min_delta=0.0, stopping=stopping),
    )


def run_masked(manifest, output, case, **options):
    return engine.run_reference_case(
        manifest,
        output,
        case,
        batch_size=2,
        input_policy=HISTORICAL_MASKED,
        prediction_retention=engine.HELDOUT_FULL_TRAIN_SESSIONS,
        **options,
    )


def row_ids(dataset, partition):
    return [
        key
        for batch in dataset.batches(partition=partition, batch_size=2, epoch=0, seed=0)
        for key in batch["sample_ids"]
    ]


@pytest.mark.parametrize("kind", ["gru", "transformer"])
def test_masked_run_uses_presence_and_keeps_every_heldout_row(tmp_path, cpu_runner, kind):
    manifest = masked_view(tmp_path)
    output = tmp_path / "run"
    report = run_masked(manifest, output, masked_case(kind))
    assert report["status"] == "completed" and report["final_test_opened"] is False
    identity = report["identity"]
    assert {k: identity[k] for k in ("input_policy", "mask_contract")} == input_identity(
        HISTORICAL_MASKED
    )
    assert identity["mask_fusion"] == PRESENCE_FUSION
    assert identity["prediction_retention"] == engine.HELDOUT_FULL_TRAIN_SESSIONS
    assert "data/input_policy.py" in identity["code"]
    assert ("models/baselines/transformer.py" in identity["code"]) is (kind == "transformer")
    dataset = CorpusDataset(manifest, input_policy=HISTORICAL_MASKED)
    counts = dataset.manifest["counts"]
    assert report["samples"] == counts and min(counts.values()) >= 1
    steps = 2 * math.ceil(counts["train"] / 2)
    assert report["global_step"] == len(cpu_runner.calls) == steps
    names = [name for name, _ in torch.nn.Module.named_parameters(_model(report, dataset))]
    for gradients in cpu_runner.calls:
        by_name = dict(zip(names, gradients, strict=True))
        # El fixture no observa noticias ni fundamentales en ninguna fila.
        for absent in ("news", "fundamentals"):
            for key, value in by_name.items():
                if key.startswith(f"encoders.{absent}."):
                    assert value is not None and torch.count_nonzero(value) == 0
        assert by_name["encoders.charts.0.weight"].abs().sum() > 0
    for partition in HELDOUT:
        record = report["predictions"][partition]
        path = output / record["path"]
        assert record["path"] == f"{partition}-predictions.parquet"
        assert record["sha256"] == sha256(path) and record["bytes"] == path.stat().st_size
        table = pq.read_table(path)
        assert table["sample_id"].to_pylist() == row_ids(dataset, partition)
        assert record["metrics"]["samples"] == counts[partition]
    assert set(report["predictions"]) == set(HELDOUT)
    assert not (output / "train-predictions.parquet").exists()
    summary = report["train_summary"]
    sessions = pq.read_table(output / summary["path"]).to_pydict()
    assert summary["sha256"] == sha256(output / summary["path"])
    assert sum(sessions["samples"]) == summary["metrics"]["samples"] == counts["train"]
    train_ids = row_ids(dataset, "train")
    expected = hashlib.sha256("".join(f"{key}\n" for key in train_ids).encode()).hexdigest()
    assert summary["metrics"]["sample_ids_sha256"] == expected
    targets = [
        value
        for batch in dataset.batches(partition="train", batch_size=2, epoch=0, seed=0)
        for value in batch["target"]
    ]
    assert math.isclose(sum(sessions["zero_absolute_error"]), sum(abs(t) for t in targets))
    assert report["stop_reason"] == "budget_exhausted" and report["stopped_early"] is False


def _model(report, dataset):
    from mars_titan.models.baselines.multimodal import MultimodalReference

    identity = report["identity"]
    return MultimodalReference(
        identity["case"]["kind"],
        identity["dimensions"],
        context=identity["context"],
        mask_fusion=identity["mask_fusion"],
        **identity["case"]["architecture"],
    )


def test_fixed_budget_keeps_equal_updates_where_the_plateau_stops(tmp_path, cpu_runner):
    manifest = masked_view(tmp_path)
    plateau = run_masked(
        manifest, tmp_path / "plateau", masked_case(epochs=3, stopping="validation_plateau")
    )
    plateau_calls = len(cpu_runner.calls)
    fixed = run_masked(manifest, tmp_path / "fixed", masked_case(epochs=3))
    per_epoch = math.ceil(fixed["samples"]["train"] / 2)
    assert plateau["stopped_early"] is True and plateau["stop_reason"] == "validation_plateau"
    assert plateau["global_step"] == plateau_calls == 2 * per_epoch
    assert fixed["stopped_early"] is False and fixed["stop_reason"] == "budget_exhausted"
    assert fixed["global_step"] == len(cpu_runner.calls) - plateau_calls == 3 * per_epoch
    assert len(fixed["epochs"]) == 3 and fixed["plateau_epoch"] == 2
    assert "plateau_epoch" not in plateau
    assert fixed["selection"]["best_epoch"] == plateau["selection"]["best_epoch"] == 1
    selected = load_training_state(
        tmp_path / "fixed/checkpoints", expected_identity=fixed["identity"], selection="best"
    )
    assert selected["epoch"] == 1
    validation = [epoch["validation"]["session_mae"] for epoch in fixed["epochs"]]
    assert validation == [validation[0]] * 3
    assert fixed["predictions"]["validation"]["metrics"]["session_mae"] == validation[0]


def test_masked_pause_resumes_the_same_cursor_and_artifacts(tmp_path, cpu_runner, monkeypatch):
    manifest = masked_view(tmp_path)
    case = masked_case("dlinear", epochs=2)
    continuous = run_masked(manifest, tmp_path / "continuous", case)
    stop = StopRequest()
    real_save = engine.save_training_state

    def pause(directory, state, **kwargs):
        saved = real_save(directory, state, **kwargs)
        if state["global_step"] == 1:
            stop.request_stop()
        return saved

    monkeypatch.setattr(engine, "save_training_state", pause)
    paused = run_masked(manifest, tmp_path / "paused", case, checkpoint_steps=1, stop=stop)
    assert paused["status"] == "paused" and paused["global_step"] == 1
    assert "train_summary" not in paused and "predictions" not in paused
    monkeypatch.setattr(engine, "save_training_state", real_save)
    resumed = run_masked(manifest, tmp_path / "paused", case, checkpoint_steps=1, resume=True)
    assert resumed["status"] == "completed"
    assert resumed["global_step"] == continuous["global_step"]
    for name in ("predictions", "train_summary"):
        assert json.dumps(resumed[name], sort_keys=True).count("sha256") > 0
    for partition in HELDOUT:
        assert (
            resumed["predictions"][partition]["sha256"]
            == continuous["predictions"][partition]["sha256"]
        )
    assert resumed["train_summary"]["sha256"] == continuous["train_summary"]["sha256"]
    again = run_masked(manifest, tmp_path / "paused", case, checkpoint_steps=1, resume=True)
    assert again == resumed
    (tmp_path / "paused/train-sessions.parquet").write_bytes(b"changed")
    with pytest.raises(ValueError, match="predicciones"):
        run_masked(manifest, tmp_path / "paused", case, resume=True)


def test_masked_continuation_can_keep_the_parent_and_rejects_another_fusion(tmp_path, cpu_runner):
    manifest = masked_view(tmp_path)
    parent = tmp_path / "parent"
    run_masked(manifest, parent, masked_case())
    child_case = {**masked_case(), "loss": "mae", "learning_rate": 0.0001}
    child = run_masked(manifest, tmp_path / "child", child_case, initialize_from=parent)
    assert child["initialization"]["optimizer_policy"] == "new_adamw"
    assert child["initial_validation"]["samples"] == child["samples"]["validation"]
    assert child["selection"]["best_epoch"] == 0
    report = json.loads((parent / "run.json").read_text())
    report["identity"]["mask_fusion"] = "strict_original"
    (parent / "run.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="origen"):
        run_masked(manifest, tmp_path / "other", child_case, initialize_from=parent)
    assert not (tmp_path / "other").exists()


@pytest.mark.parametrize("explicit", [False, True])
def test_strict_default_keeps_its_identity_fields_and_prediction_files(
    tmp_path, cpu_runner, explicit
):
    manifest = training_corpus(tmp_path / "data")
    case = masked_case()
    case.pop("selection")
    options = dict(input_policy=STRICT_INPUTS) if explicit else {}
    report = engine.run_reference_case(manifest, tmp_path / "run", case, batch_size=5, **options)
    identity = report["identity"]
    assert set(identity) == LEGACY_IDENTITY
    assert list(identity["code"]) == list(engine._SOURCES)
    assert set(report["predictions"]) == {"train", "validation"}
    assert not {"train_summary", "stop_reason", "plateau_epoch"} & set(report)
    assert all(
        set(record) == {"path", "sha256", "metrics"} for record in report["predictions"].values()
    )
    state = load_training_state(tmp_path / "run/checkpoints", expected_identity=identity)
    assert state["model"]["fusion.0.weight"].shape == (32, 5 * 32)
    for partition in ("train", "validation"):
        table = pq.read_table(tmp_path / f"run/{partition}-predictions.parquet")
        assert len(table) == report["samples"][partition]


def test_strict_default_and_explicit_policy_produce_the_same_outputs(tmp_path, cpu_runner):
    manifest = training_corpus(tmp_path / "data")
    case = masked_case()
    case.pop("selection")
    default = engine.run_reference_case(manifest, tmp_path / "default", case, batch_size=5)
    explicit = engine.run_reference_case(
        manifest,
        tmp_path / "explicit",
        case,
        batch_size=5,
        input_policy=STRICT_INPUTS,
        prediction_retention=engine.FULL_TRAIN_VALIDATION,
    )
    assert default["identity"] == explicit["identity"]
    for partition in ("train", "validation"):
        assert pq.read_table(tmp_path / f"default/{partition}-predictions.parquet").equals(
            pq.read_table(tmp_path / f"explicit/{partition}-predictions.parquet")
        )


# Las opciones se rechazan antes de leer, sin ajuste. La protección se detendría antes.
@pytest.mark.usefixtures("learning_doubles")
@pytest.mark.parametrize(
    "change,options,message",
    [
        (lambda c: c.pop("architecture"), {}, "arquitectura"),
        (lambda c: c.update(kind="transformer"), {}, "arquitectura"),
        (
            lambda c: c.update(kind="transformer", architecture=architecture("transformer")),
            dict(batch_size=512),
            "lote|presupuesto",
        ),
        (
            lambda c: c.update(
                kind="transformer", architecture={**architecture("gru"), "transformer": None}
            ),
            {},
            "Transformer",
        ),
        (
            lambda c: c.update(
                kind="transformer",
                architecture={**architecture("gru"), "transformer": dict(heads=3)},
            ),
            {},
            "Transformer",
        ),
        (lambda c: None, dict(prediction_retention="train_full"), "retención"),
        (lambda c: None, dict(input_policy="historical"), "política"),
        (lambda c: c["selection"].update(stopping="never"), {}, "parada"),
    ],
)
def test_invalid_options_fail_before_reading_or_creating_a_run(tmp_path, change, options, message):
    case = masked_case()
    change(case)
    arguments = dict(batch_size=2, input_policy=HISTORICAL_MASKED) | options
    with pytest.raises(ValueError, match=message):
        engine.run_reference_case(tmp_path / "missing.json", tmp_path / "run", case, **arguments)
    assert not (tmp_path / "run").exists()


def test_strict_policy_cannot_read_the_masked_edition(tmp_path, cpu_runner):
    manifest = masked_view(tmp_path)
    case = masked_case()
    with pytest.raises(ValueError):
        engine.run_reference_case(manifest, tmp_path / "run", case, batch_size=2)
    assert not (tmp_path / "run").exists()


def test_presence_reaches_the_model_only_from_the_masked_reader(tmp_path, cpu_runner):
    manifest = masked_view(tmp_path)
    dataset = CorpusDataset(manifest, input_policy=HISTORICAL_MASKED)
    batch = next(dataset.batches(partition="train", batch_size=3, epoch=0, seed=1))
    seen = {}

    class Spy(torch.nn.Module):
        def forward(self, inputs, presence=None):
            seen.update(inputs=set(inputs), presence=presence)
            return torch.zeros(len(inputs["prices"]))

    engine._forward(Spy(), batch, "cpu")
    assert seen["inputs"] == set(MODALITIES)
    assert seen["presence"].dtype == torch.bool
    np.testing.assert_array_equal(seen["presence"].numpy(), batch["presence"])
    engine._forward(Spy(), {k: v for k, v in batch.items() if k != "presence"}, "cpu")
    assert seen["presence"] is None
