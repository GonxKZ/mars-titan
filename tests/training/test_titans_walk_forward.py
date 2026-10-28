"""Ventanas walk-forward de Titans-MAC hasta las predicciones por fila, sin pasos que ajusten.

El optimizador inyectado registra gradientes y llamadas sin modificar pesos. Las vistas
proceden de un corpus técnico con máscaras preparado con el contrato temporal v2.
"""

import copy
import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from mars_titan.data.input_policy import HISTORICAL_MASKED
from mars_titan.data.storage import sha256
from mars_titan.evaluation import scale_diagnostics as sd
from mars_titan.evaluation import walk_forward_comparison as comparison
from mars_titan.evaluation.forecast_panel import ForecastPanel
from mars_titan.evaluation.splits import build_folds, stopping_rule
from mars_titan.memory.financial_session import FinancialPhase
from mars_titan.models.quantile_head import QUANTILE_COLUMNS
from mars_titan.models.titans.financial import VARIANTS
from mars_titan.models.titans.financial_inputs import FINAL_TEST_US
from mars_titan.training import titans_walk_forward as wf
from mars_titan.training.corpus_inputs import CorpusDataset
from mars_titan.training.financial_run import ChronologicalTrainer
from mars_titan.training.learning_hold import LearningHoldError
from mars_titan.training.temporal_corpus import prepare_temporal_corpus
from mars_titan.training.walk_forward_phases import months_before
from mars_titan.training.window_indices import SHARED_INDICES_ENV
from tests.learning_hold_lowering import allowed_hold, lowers_learning_hold
from tests.learning_traces.parity import EVERY_STEP, metrics, same_rows, trace_rows
from tests.training.chronological_fixture import chronological_corpus, decision

CONFIGS = Path("configs")
RULE = dict(metric="session_mae", stopping="fixed_budget", patience=1, min_delta=1e-5)
PROTOCOL = dict(
    schema_version=2,
    market="US",
    train_start="2000-01-01",
    first_validation_start="2023-09-01",
    validation_months=1,
    calibration_months=1,
    evaluation_months=2,
    step_months=2,
    minimum_train_months=240,
    purge="label_interval",
    final_test_start="2024-01-01",
    final_test_end="2025-01-01",
    primary_metric="session_mae",
    selection=dict(RULE, max_epochs=2),
    seeds=[42],
)
EVALUATION_START = decision("2023-11-01")
INSIDE_EVALUATION = decision("2023-12-01")
QUANTILE_RECIPE = "chronological-training-quantile.json"
SCALAR_RECIPE = "chronological-training.json"


@pytest.fixture(autouse=True)
def explicit_fastpath():
    previous = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    yield
    torch.backends.mha.set_fastpath_enabled(previous)


@pytest.fixture(autouse=True)
def permitted_doubles(learning_doubles):
    """El ajuste está sustituido por un optimizador que no modifica pesos."""
    return learning_doubles


class Recorder:
    """Registrar gradientes por grupo y parámetro sin heredar de Optimizer."""

    def __init__(self, groups, *, retain_gradients=True):
        # Como en un optimizador de PyTorch, cada grupo lleva su tasa, que las trazas registran.
        self.param_groups = [
            dict(group, params=list(group["params"]), lr=group.get("lr", 0.0)) for group in groups
        ]
        self.calls, self.records = 0, []
        self.retain_gradients = retain_gradients

    def step(self):
        self.calls += 1
        if not self.retain_gradients:
            self.records.append(None)
            return
        self.records.append(
            [
                None if p.grad is None else p.grad.detach().clone()
                for group in self.param_groups
                for p in group["params"]
            ]
        )

    def zero_grad(self, set_to_none=True):
        assert set_to_none is True
        for group in self.param_groups:
            for value in group["params"]:
                value.grad = None

    def state_dict(self):
        return dict(calls=self.calls, roles=[group["role"] for group in self.param_groups])

    def load_state_dict(self, state):
        assert state["roles"] == [group["role"] for group in self.param_groups]
        self.calls = state["calls"]


class Factory:
    """Conservar los registradores de cada intento para comparar recorridos."""

    def __init__(self, *, retain_gradients=True):
        self.instances = []
        self.retain_gradients = retain_gradients

    def __call__(self, groups):
        self.instances.append(Recorder(groups, retain_gradients=self.retain_gradients))
        return self.instances[-1]

    @property
    def records(self):
        return [record for item in self.instances for record in item.records]


def views(root, **perturbation):
    """Preparar la vista de la única ventana del protocolo técnico."""
    parent = chronological_corpus(
        root / "parent", assets=3, first="2023-06-01", last="2023-12-29", **perturbation
    )
    protocol = root / "protocol.json"
    protocol.write_text(json.dumps(PROTOCOL))
    prepare_temporal_corpus(
        parent,
        protocol,
        None,
        None,
        root / "views",
        input_policy=HISTORICAL_MASKED,
        recover_annual_boundaries=True,
    )
    return root / "views" / "fold-000" / "manifest.json", protocol


def recipe(root, name=QUANTILE_RECIPE, **changes):
    document = json.loads((CONFIGS / "titans" / name).read_text())
    document["predictor"].update(hidden_size=32, dtype="float64")
    document["recipe"].update(truncation=3, block_rows=2, epochs=2, selection=dict(RULE))
    document["walk_forward"] = dict(warmup_months=1)
    for key, value in changes.items():
        if value is None:
            document.pop(key)
        else:
            document[key] = value
    path = root / f"recipe-{len(list(root.glob('recipe-*')))}.json"
    path.write_text(json.dumps(document))
    return path


def window(view, protocol, recipe_path, output, *, variant="mac_online", **options):
    factory = options.pop("optimizer_factory", None) or Factory()
    report = wf.run_titans_window(
        view,
        protocol,
        "fold-000",
        recipe_path,
        variant=variant,
        seed=42,
        output=output,
        device="cpu",
        indices=output.parent / "indices",
        optimizer_factory=factory,
        **options,
    )
    return report, factory


def table(output, partition):
    """Columnas por fila ordenadas por muestra, con los instantes en microsegundos UTC."""
    rows = pq.read_table(output / f"{partition}-predictions.parquet").sort_by("sample_id")
    data = {name: rows[name].to_pylist() for name in rows.column_names}
    data["prediction_at"] = rows["prediction_at"].cast(pa.int64()).to_pylist()
    return data


@pytest.fixture(scope="module")
def base(tmp_path_factory, learning_doubles_module):
    root = tmp_path_factory.mktemp("titans-walk-forward")
    view, protocol = views(root / "base")
    quantile, scalar = recipe(root), recipe(root, SCALAR_RECIPE)
    runs = {}
    with unfused_attention():
        for variant in VARIANTS:
            runs[variant] = window(
                view, protocol, quantile, root / "runs" / variant, variant=variant
            )
        runs["scalar"] = window(view, protocol, scalar, root / "runs" / "scalar")
    return dict(root=root, view=view, protocol=protocol, recipe=quantile, runs=runs)


@contextmanager
def unfused_attention():
    previous = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        yield
    finally:
        torch.backends.mha.set_fastpath_enabled(previous)


@pytest.fixture(scope="module")
def learning_doubles_module(tmp_path_factory):
    from mars_titan.training.learning_hold import HOLD_ENV

    path = allowed_hold(tmp_path_factory.mktemp("learning-hold") / "training-hold.json")
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv(HOLD_ENV, str(path))
        yield path


def micros(day):
    return int(np.datetime64(day, "us").astype(np.int64))


@pytest.mark.parametrize(
    "path",
    [
        "historical-masked-us-walk-forward-v2.json",
        "historical-masked-cn-walk-forward-v2.json",
        "historical-masked-joint-us-walk-forward-v2.json",
    ],
)
@pytest.mark.parametrize("warmup", [0, 12, 60])
def test_phases_cover_each_v2_fold_with_a_warmup_bounded_by_its_partition(path, warmup):
    protocol = json.loads((CONFIGS / "evaluation" / path).read_text())
    folds = build_folds(protocol)
    assert len(folds) == (19 if protocol["first_validation_start"] == "2004-04-01" else 13)
    for fold in folds:
        phases = wf.window_phases(fold, warmup)
        assert set(phases) == {"train", "validation", "calibration", "evaluation"}
        origin = micros("2000-01-01")
        train = phases["train"]
        assert (train.warmup_start, train.decision_start) == (origin, origin)
        assert (train.decision_end, train.close_at) == (micros(fold["train"][1]),) * 2
        for name in wf.PREDICTED:
            phase, (start, end) = phases[name], fold[name]
            assert type(phase) is FinancialPhase and phase.partition == name
            assert (phase.decision_start, phase.decision_end) == (micros(start), micros(end))
            assert phase.close_at == phase.decision_end <= FINAL_TEST_US
            year, month = int(start[:4]), int(start[5:7]) - warmup
            while month < 1:
                year, month = year - 1, month + 12
            expected = max(origin, micros(f"{year:04d}-{month:02d}-01"))
            assert phase.warmup_start == expected <= phase.decision_start
            assert phase.warmup_start >= micros(start) - warmup * 31 * 86_400_000_000
        assert phases["validation"].decision_start == train.decision_end
    # Con 60 meses, la primera validación US (abril de 2004) empieza a calentar en el origen.
    first = wf.window_phases(folds[0], warmup)["validation"]
    clipped = warmup == 60 and protocol["first_validation_start"] == "2004-04-01"
    assert (first.warmup_start == micros("2000-01-01")) == clipped


@pytest.mark.parametrize("fit", [1, 6, 12])
@pytest.mark.parametrize("warmup", [0, 12])
def test_bounded_fit_reads_its_last_months_after_their_warmup(fit, warmup):
    protocol = json.loads(
        (CONFIGS / "evaluation/historical-masked-us-walk-forward-v2.json").read_text()
    )
    origin = micros("2000-01-01")
    for fold in build_folds(protocol):
        full, bounded = wf.window_phases(fold, warmup), wf.window_phases(fold, warmup, fit)
        assert {name: bounded[name] for name in wf.PREDICTED} == {
            name: full[name] for name in wf.PREDICTED
        }
        train, first = bounded["train"], months_before(fold["train"][1], fit)
        assert train.decision_start == max(origin, micros(first))
        assert train.warmup_start == max(origin, micros(months_before(first, warmup)))
        assert (train.decision_end, train.close_at) == (full["train"].decision_end,) * 2
        assert train.decision_end - train.decision_start <= 366 * 86_400_000_000


@pytest.mark.parametrize("value", [0, 13, True, 1.0, "12"])
def test_bounded_fit_rejects_months_outside_one_year(value):
    fold = build_folds(PROTOCOL)[0]
    with pytest.raises(ValueError, match="fit_months"):
        wf.window_phases(fold, 1, value)
    document = dict(walk_forward=dict(warmup_months=1, fit_months=value))
    with pytest.raises(ValueError, match="fit_months"):
        wf.walk_forward_options(document, bounded_fit=True)


def test_only_paths_that_apply_a_bounded_fit_accept_it():
    document = dict(walk_forward=dict(warmup_months=12, fit_months=12))
    with pytest.raises(ValueError, match="ajuste acotado"):
        wf.walk_forward_options(document)
    assert wf.walk_forward_options(document, bounded_fit=True)["fit_months"] == 12
    assert wf.memory_policy(12, None) == wf.memory_policy(12)
    assert "fit_months" not in wf.memory_policy(12)
    bounded = wf.memory_policy(12, 12)
    assert bounded["fit_months"] == 12 and bounded["train"] != wf.memory_policy(12)["train"]


def test_bounded_fit_ignores_inputs_before_its_warmup(base, tmp_path):
    """Con un mes de ajuste y uno de calentamiento, nada anterior a julio llega al ajuste.

    Se alteran las entradas no de precio de las filas anteriores al 3 de julio, el primer día
    hábil del calentamiento.
    """
    bounded = recipe(tmp_path, walk_forward=dict(warmup_months=1, fit_months=1))
    reports, factories = {}, {}
    with unfused_attention():
        for name, changes in (
            ("clean", {}),
            ("before", dict(perturb_before=decision("2023-07-03"))),
        ):
            view, protocol = views(tmp_path / name, **changes)
            reports[name], factories[name] = window(
                view, protocol, bounded, tmp_path / name / "run"
            )
    train = reports["clean"]["identity"]["phases"]["train"]
    assert (train["warmup_start"], train["decision_start"], train["decision_end"]) == (
        micros("2023-07-01"),
        micros("2023-08-01"),
        micros("2023-09-01"),
    )
    assert reports["clean"]["identity"]["memory_policy"]["fit_months"] == 1
    clean, before = factories["clean"].records, factories["before"].records
    assert clean and len(clean) == len(before)
    for one, two in zip(clean, before, strict=True):
        for a, b in zip(one, two, strict=True):
            assert (a is None and b is None) or torch.equal(a, b)
    # El ajuste completo de la base empieza en el origen y aplica más actualizaciones.
    assert len(base["runs"]["mac_online"][1].records) > len(clean)


def test_controls_write_three_partitions_that_the_comparison_accepts(base, tmp_path):
    view, root = base["view"], base["root"]
    dataset = CorpusDataset(view, input_policy=HISTORICAL_MASKED)
    fold = build_folds(PROTOCOL)[0]
    for name, (report, factory) in base["runs"].items():
        quantile = name != "scalar"
        output = root / "runs" / name
        assert report["status"] == "completed" and report["final_test_opened"] is False
        assert factory.records and report["fit"]["global_step"] == len(factory.records)
        check = report["validation_selection_check"]
        assert check["absolute_difference"] == 0.0
        assert check["best_score"] == report["fit"]["best_score"]
        for partition in wf.PREDICTED:
            record = report["predictions"][partition]
            assert sha256(output / record["path"]) == record["sha256"]
            columns = comparison.COLUMNS + (QUANTILE_COLUMNS if quantile else ())
            rows = comparison._read_predictions(
                dict(path=output / record["path"], sha256=record["sha256"]), columns
            )
            comparison._check_segment(rows, fold, partition, ["US"], partition)
            frame = table(output, partition)
            count = len(frame["sample_id"])
            assert count == record["rows"] == dataset.manifest["counts"][partition]
            batches = list(dataset.batches(partition=partition, batch_size=512, epoch=0, seed=0))
            expected = sorted(
                (sample, float(target))
                for batch in batches
                for sample, target in zip(batch["sample_ids"], batch["target"], strict=True)
            )
            assert list(zip(frame["sample_id"], frame["target"], strict=True)) == expected
            assert frame["sample_id"] == [
                f"{asset}/{at}"
                for asset, at in zip(frame["asset_id"], frame["prediction_at"], strict=True)
            ]
            assert set(frame["market"]) == {"US"} and set(frame["zero"]) == {0.0}
            assert max(frame["prediction_at"]) < FINAL_TEST_US
            if quantile:
                assert frame["quantile_0500"] == frame["prediction"]
                assert set(QUANTILE_COLUMNS) <= set(frame)
            else:
                assert not set(QUANTILE_COLUMNS) & set(frame)
    _comparison_accepts(base, tmp_path)


def _comparison_accepts(base, tmp_path):
    """Recorrer `evaluate_walk_forward` con los Parquet escritos y sin abrir 2024."""
    arms = {f"titans_{variant}": variant for variant in VARIANTS}
    config = dict(
        json.loads((CONFIGS / "evaluation/historical-masked-2000-comparison.json").read_text()),
        scopes={"US": dict(protocols={"US": str(base["protocol"])}, windows="all")},
        metrics=dict(
            primary="mae",
            market_weighting="session",
            rank_ic_min_assets=3,
            quantile_head="quantile_head_v1",
        ),
        arms={
            "zero": dict(family="control", output="zero_control", seeds=[]),
            **{
                name: dict(family="titans_mac", output="quantile_head_v1", seeds=[42])
                for name in arms
            },
            "titans_mac_online_scalar": dict(family="titans_mac", output="point", seeds=[42]),
        },
    )
    # Los pares de retención nombran modelos que esta comparación reducida no conserva.
    config.pop("retention_interference")
    config["comparison"] = dict(
        config["comparison"],
        replicates=50,
        metrics=["mae", "mse", "pinball"],
        sensitivity_block_lengths=[5],
        families=dict(
            titans_controls=dict(
                kind="delta",
                base="titans_transformer_direct",
                variants=["titans_mac_disabled", "titans_mac_frozen", "titans_mac_online"],
            )
        ),
    )
    config_path = tmp_path / "comparison.json"
    config_path.write_text(json.dumps(config))
    runs = base["root"] / "runs"

    def entry(name, quantile):
        report = base["runs"][name][0]
        result = dict(input_policy=HISTORICAL_MASKED, view_sha256=sha256(base["view"]))
        for part in ("calibration", "evaluation") if quantile else ("evaluation",):
            record = report["predictions"][part]
            result[part] = dict(path=str(runs / name / record["path"]), sha256=record["sha256"])
        return result

    sources = dict(
        schema_version=1,
        kind=comparison.SOURCES_KIND,
        scope="US",
        input_policy=HISTORICAL_MASKED,
        windows={"fold-000": dict(view=dict(path=str(base["view"]), sha256=sha256(base["view"])))},
        arms={
            **{name: {"42": {"fold-000": entry(variant, True)}} for name, variant in arms.items()},
            "titans_mac_online_scalar": {"42": {"fold-000": entry("scalar", False)}},
        },
    )
    sources_path = tmp_path / "sources.json"
    sources_path.write_text(json.dumps(sources))
    report, sessions = comparison.evaluate_walk_forward(config_path, sources_path, "US")
    assert report["status"] == "completed" and report["final_test_opened"] is False
    assert set(report["arms"]) == {"zero", *arms, "titans_mac_online_scalar"}
    assert sessions.num_rows > 0


def test_paired_controls_start_from_the_mac_online_parameters(base):
    identities = {name: run[0]["identity"] for name, run in base["runs"].items()}
    fits = {
        name: json.loads((base["root"] / "runs" / name / "fit/run.json").read_text())["identity"]
        for name in VARIANTS
    }
    initial = {name: fits[name]["initial_parameters_sha256"] for name in VARIANTS}
    assert len({identities[name]["request"]["variant"] for name in VARIANTS}) == 4
    assert fits["mac_online"]["pairing_sha256"] is None
    assert all(fits[name]["pairing_sha256"] for name in VARIANTS if name != "mac_online")
    # Los controles con MAC copian todos los parámetros de mac_online. El directo no tiene MAC.
    assert initial["mac_disabled"] == initial["mac_frozen"] == initial["mac_online"]
    assert initial["transformer_direct"] != initial["mac_online"]
    policies = {json.dumps(identities[name]["memory_policy"]) for name in identities}
    assert len(policies) == 1
    assert json.loads(policies.pop())["warmup_months"] == 1
    assert {json.dumps(identities[name]["phases"]) for name in identities} == {
        json.dumps(identities["mac_online"]["phases"])
    }


def _compare(left, right, partition, *, until=None):
    """Igualdad exacta de predicciones y cuantiles emitidos, opcionalmente hasta un corte."""

    def emitted(output):
        frame = table(output, partition)
        columns = ["sample_id", "prediction_at", "prediction", *QUANTILE_COLUMNS]
        rows = list(zip(*(frame[name] for name in columns), strict=True))
        return [row for row in rows if until is None or row[1] <= until]

    first, second = emitted(left), emitted(right)
    assert first and second
    return first == second


def test_fit_and_memory_never_see_the_future_of_each_measured_partition(base, tmp_path):
    reference, recorded = base["runs"]["mac_online"]
    left = base["root"] / "runs" / "mac_online"
    view, protocol = views(tmp_path / "after-calibration", perturb_after=EVALUATION_START - 1)
    report, factory = window(view, protocol, base["recipe"], tmp_path / "after-calibration/run")
    right = tmp_path / "after-calibration/run"
    assert report["fit"]["best_score"] == reference["fit"]["best_score"]
    assert len(factory.records) == len(recorded.records)
    for one, two in zip(factory.records, recorded.records, strict=True):
        for a, b in zip(one, two, strict=True):
            assert (a is None and b is None) or torch.equal(a, b)
    assert _compare(left, right, "validation") and _compare(left, right, "calibration")
    assert not _compare(left, right, "evaluation")
    view, protocol = views(tmp_path / "inside", perturb_after=INSIDE_EVALUATION)
    window(view, protocol, base["recipe"], tmp_path / "inside/run")
    inside = tmp_path / "inside/run"
    assert _compare(left, inside, "evaluation", until=INSIDE_EVALUATION)
    assert not _compare(left, inside, "evaluation")


@pytest.mark.parametrize("variant", ["mac_online", "transformer_direct"])
def test_each_pass_restarts_memory_and_warms_up_only_before_its_partition(base, tmp_path, variant):
    """Entradas no de precio alteradas antes de octubre, el mes de calentamiento de evaluación."""
    view, protocol = views(tmp_path, perturb_before=decision("2023-10-02"))
    window(view, protocol, base["recipe"], tmp_path / "run", variant=variant)
    left, right = base["root"] / "runs" / variant, tmp_path / "run"
    assert _compare(left, right, "evaluation")
    # La calibración se calienta en septiembre: solo la memoria online recoge el cambio.
    assert _compare(left, right, "calibration") is (variant != "mac_online")


def test_window_resumes_from_coherent_barriers_and_never_repeats_a_completed_window(
    base, tmp_path, monkeypatch
):
    reference, recorded = base["runs"]["mac_online"]
    factory = Factory()

    class StopAtStep:
        @property
        def requested(self):
            return sum(item.calls for item in factory.instances[-1:]) >= 5

    output = tmp_path / "run"
    paused, _ = window(
        base["view"],
        base["protocol"],
        base["recipe"],
        output,
        optimizer_factory=factory,
        stop=StopAtStep(),
    )
    assert paused["status"] == "paused" and paused["predictions"] == {}

    class StopAfterValidation:
        @property
        def requested(self):
            return (output / "validation-predictions.parquet").exists()

    second, _ = window(
        base["view"],
        base["protocol"],
        base["recipe"],
        output,
        optimizer_factory=factory,
        stop=StopAfterValidation(),
    )
    assert second["status"] == "paused" and set(second["predictions"]) == {"validation"}
    final, _ = window(
        base["view"], base["protocol"], base["recipe"], output, optimizer_factory=factory
    )
    assert final["status"] == "completed" and len(final["attempts"]) == 3
    assert len(factory.records) == len(recorded.records)
    for one, two in zip(factory.records, recorded.records, strict=True):
        for a, b in zip(one, two, strict=True):
            assert (a is None and b is None) or torch.equal(a, b)
    left = base["root"] / "runs" / "mac_online"
    for partition in wf.PREDICTED:
        assert _compare(left, output, partition)
        assert (
            final["predictions"][partition]["rows"] == reference["predictions"][partition]["rows"]
        )

    def refuse(*_, **__):
        raise AssertionError("Una ventana completada no vuelve a abrir la vista")

    monkeypatch.setattr(wf, "CorpusDataset", refuse)
    again, _ = window(base["view"], base["protocol"], base["recipe"], output)
    assert again == final
    with pytest.raises(ValueError, match="petición"):
        window(base["view"], base["protocol"], base["recipe"], output, variant="mac_frozen")
    (output / "evaluation-predictions.parquet").write_bytes(b"PAR1 alterado PAR1")
    with pytest.raises(ValueError, match="predicciones confirmadas"):
        window(base["view"], base["protocol"], base["recipe"], output)


def test_window_refuses_while_the_learning_hold_blocks(base, tmp_path, learning_hold, monkeypatch):
    learning_hold(False)

    def refuse(*_, **__):
        raise AssertionError("La vista no debe abrirse con el bloqueo vigente")

    monkeypatch.setattr(wf, "CorpusDataset", refuse)
    with pytest.raises(LearningHoldError, match="Bloqueo de aprendizaje vigente"):
        window(base["view"], base["protocol"], base["recipe"], tmp_path / "run")
    assert not (tmp_path / "run").exists() and not (tmp_path / "indices").exists()


def test_window_fits_the_chosen_search_case_and_names_it_in_its_request(base, tmp_path):
    path = recipe(tmp_path)
    document = json.loads(path.read_text())
    document["recipe"].pop("learning_rate")
    document["walk_forward"]["search_cases"] = {
        "slow": dict(learning_rate=1e-4),
        "fast": dict(learning_rate=1e-3),
    }
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="Elige uno"):
        window(base["view"], base["protocol"], path, tmp_path / "missing")
    assert not (tmp_path / "missing").exists()
    with unfused_attention():
        report, _ = window(
            base["view"], base["protocol"], path, tmp_path / "run", search_case="slow"
        )
    assert report["status"] == "completed" and report["request"]["search_case"] == "slow"
    fit = json.loads((tmp_path / "run" / "fit/run.json").read_text())
    assert fit["identity"]["recipe"]["learning_rate"] == 1e-4
    # El optimizador sustituto no mueve pesos: solo cambia la identidad, no las filas.
    for partition in wf.PREDICTED:
        assert table(tmp_path / "run", partition) == table(
            base["root"] / "runs" / "mac_online", partition
        )


# Cambios de PT1 (#453) sobre la receta, los mismos que declara la campaña A v2.
PT1 = dict(
    gate_bias=dict(alpha_half_life=256.0, eta=0.15, theta=0.05),
    memory_stability=dict(alpha_floor=0.002, eta_ceiling=0.3, gradient_clip=4.0, gate_box=True),
)


@lowers_learning_hold(
    "tests/training/test_titans_walk_forward.py::"
    "test_window_builds_the_declared_predictor_changes_and_its_regeneration_repeats_them"
)
def test_window_builds_the_declared_predictor_changes_and_its_regeneration_repeats_them(
    base, tmp_path
):
    """PT1 viaja en la petición, entra en la identidad del predictor y se reconstruye al
    trasladar. Sin cambios, la petición conserva su forma anterior."""
    from mars_titan.training import prediction_regeneration as regeneration

    with unfused_attention():
        report, _ = window(
            base["view"], base["protocol"], base["recipe"], tmp_path / "run", predictor_changes=PT1
        )
    assert report["status"] == "completed" and report["request"]["predictor_changes"] == PT1
    plain = base["runs"]["mac_online"][0]
    assert "predictor_changes" not in plain["request"]
    fit = json.loads((tmp_path / "run" / "fit/run.json").read_text())["identity"]
    reference = json.loads((base["root"] / "runs/mac_online/fit/run.json").read_text())
    assert fit["predictor"]["memory_gate_bias"] == PT1["gate_bias"]
    assert fit["predictor"]["memory_stability"] == PT1["memory_stability"]
    assert "memory_stability" not in reference["identity"]["predictor"]
    # La η inicial de la caja cambia los bias iniciales de la memoria.
    assert fit["initial_parameters_sha256"] != reference["identity"]["initial_parameters_sha256"]
    # Sin PT1 en el predictor del traslado, el estado del ancla no correspondería a su
    # arquitectura. La regeneración repite las filas de la ventana sin diferencias.
    with unfused_attention():
        again = wf.carry_titans(
            tmp_path / "run",
            base["view"],
            base["view"],
            tmp_path / "again",
            device="cpu",
            regenerate=True,
        )
    result = regeneration.compare(
        regeneration.originals(tmp_path / "run", report), tmp_path / "again", again
    )
    assert result["identical"] is True, result


@pytest.mark.parametrize(
    "case",
    ["min_delta", "epochs", "walk_forward", "warmup", "window", "seed", "protocol", "variant"],
)
def test_window_rejects_declarations_that_break_the_protocol(base, tmp_path, case):
    view, protocol, recipe_path = base["view"], base["protocol"], base["recipe"]
    options = dict(window="fold-000", seed=42, variant="mac_online")
    if case in ("min_delta", "epochs"):
        document = json.loads(recipe_path.read_text())
        if case == "min_delta":
            document["recipe"]["selection"]["min_delta"] = 0.0
        else:
            document["recipe"]["epochs"] = 3
        recipe_path = tmp_path / "recipe.json"
        recipe_path.write_text(json.dumps(document))
    elif case in ("walk_forward", "warmup"):
        value = None if case == "walk_forward" else dict(warmup_months=-1)
        recipe_path = recipe(tmp_path, walk_forward=value)
    elif case == "window":
        options["window"] = "fold-001"
    elif case == "seed":
        options["seed"] = 43
    elif case == "protocol":
        protocol = tmp_path / "protocol.json"
        protocol.write_text(json.dumps(dict(PROTOCOL, seeds=[42, 7])))
    else:
        options["variant"] = "mac_bank"
    with pytest.raises(ValueError):
        wf.run_titans_window(
            view,
            protocol,
            options["window"],
            recipe_path,
            variant=options["variant"],
            seed=options["seed"],
            output=tmp_path / "run",
            device="cpu",
            optimizer_factory=Factory(),
        )
    assert not (tmp_path / "run" / "fit").exists()


@pytest.mark.parametrize("name", [SCALAR_RECIPE, QUANTILE_RECIPE])
@pytest.mark.parametrize(
    "protocol",
    [
        "historical-masked-us-walk-forward-v2.json",
        "historical-masked-cn-walk-forward-v2.json",
        "historical-masked-joint-us-walk-forward-v2.json",
    ],
)
def test_declared_recipes_apply_the_common_v2_rule_and_keep_their_other_fields(name, protocol):
    rule = stopping_rule(json.loads((CONFIGS / "evaluation" / protocol).read_text()))
    recipe_value, document, options = wf._recipe(CONFIGS / "titans" / name, rule)
    assert options == dict(warmup_months=12)
    assert recipe_value.selection == {k: v for k, v in rule.items() if k != "max_epochs"}
    assert recipe_value.epochs == rule["max_epochs"] == 30
    assert (recipe_value.truncation, recipe_value.learning_rate, recipe_value.block_rows) == (
        8,
        1e-3,
        128,
    )
    assert document["predictor"]["gate_bias"] == dict(alpha_half_life=256.0, eta=0.5, theta=0.05)
    assert "memory_residual_layer_norm" not in document["predictor"]
    assert document["status"] == "propuesta_sin_ejecutar"


def test_rows_sink_rejects_rows_outside_the_edition_or_without_their_output():
    rows = wf.PredictionRows(quantiles=True)
    with pytest.raises(ValueError):
        rows.append(("US/A0000", FINAL_TEST_US, 0.0, 0.0, [0.0] * 5))
    with pytest.raises(ValueError):
        rows.append(("US/A0000", FINAL_TEST_US - 1, 0.0, 0.0, None))
    with pytest.raises(ValueError):
        wf.PredictionRows(quantiles=False).append(("US/A0000", 1, 0.0, 0.0, [0.0] * 5))


class SkipFirstRow:
    """Destino que pierde la primera fila resuelta."""

    def __init__(self, rows):
        self.rows, self.skipped = rows, False

    def append(self, record):
        if self.skipped:
            self.rows.append(record)
        self.skipped = True


def inflated_validation(original):
    def altered(self, source, rows, *, stop=None):
        metrics = original(self, source, rows, stop=stop)
        if source.phase.partition == "validation":
            metrics = dict(metrics, session_mae=metrics["session_mae"] + 1e-3)
        return metrics

    return altered


class ShiftFirstTarget:
    """Destino que conserva el recuento pero cambia el objetivo de la primera fila."""

    def __init__(self, rows):
        self.rows, self.shifted = rows, False

    def append(self, record):
        if not self.shifted:
            flow, at, prediction, target, levels = record
            record, self.shifted = (flow, at, prediction, target + 1.0, levels), True
        self.rows.append(record)


def wrapped_sink(sink):
    def wrap(original):
        def altered(self, source, rows, *, stop=None):
            return original(self, source, sink(rows), stop=stop)

        return altered

    return wrap


@pytest.mark.parametrize(
    ("wrap", "message"),
    [
        (inflated_validation, "no reproduce la puntuación"),
        (wrapped_sink(SkipFirstRow), "no concilian"),
        (wrapped_sink(ShiftFirstTarget), "1 filas difieren de la vista"),
    ],
    ids=["validation", "dropped", "target"],
)
def test_window_fails_when_its_predictions_do_not_reconcile(
    base, tmp_path, monkeypatch, wrap, message
):
    monkeypatch.setattr(
        ChronologicalTrainer,
        "predict_partition",
        wrap(ChronologicalTrainer.predict_partition),
    )
    output = tmp_path / "run"
    with unfused_attention(), pytest.raises(ValueError, match=message):
        window(base["view"], base["protocol"], base["recipe"], output)
    report = json.loads((output / "run.json").read_text())
    assert report["status"] == "failed" and report["predictions"] == {}
    assert not (output / "validation-predictions.parquet").exists()


def test_view_must_keep_the_window_rows_and_the_closed_final_test(base):
    dataset = CorpusDataset(base["view"], input_policy=HISTORICAL_MASKED)
    protocol = json.loads(Path(base["protocol"]).read_text())
    fold = build_folds(protocol)[0]
    wf._check_view(dataset, protocol, fold)
    for change in (
        lambda manifest: manifest.update(final_test_opened=True),
        lambda manifest: manifest.pop("final_test_opened"),
        lambda manifest: manifest["counts"].update(evaluation=0),
        lambda manifest: manifest["counts"].pop("calibration"),
    ):
        manifest = copy.deepcopy(dataset.manifest)
        change(manifest)
        altered = SimpleNamespace(manifest=manifest)
        with pytest.raises(ValueError, match="cuatro tramos"):
            wf._check_view(altered, protocol, fold)
    with pytest.raises(ValueError, match="protocolo y la ventana"):
        wf._check_view(dataset, protocol, dict(fold, evaluation=["2023-11-01", "2024-01-02"]))


def test_window_refuses_an_output_with_files_from_another_run(base, tmp_path, monkeypatch):
    def refuse(*_, **__):
        raise AssertionError("La vista no debe abrirse sobre una salida ajena")

    monkeypatch.setattr(wf, "CorpusDataset", refuse)
    output = tmp_path / "run"
    output.mkdir()
    (output / "other.json").write_text("{}")
    with pytest.raises(ValueError, match="otra ejecución"):
        window(base["view"], base["protocol"], base["recipe"], output)
    assert sorted(path.name for path in output.iterdir()) == ["other.json"]


def test_shared_indices_keep_the_window_bit_for_bit(base, tmp_path, monkeypatch):
    """Índices compartidos por vista (#618): mismos gradientes, filas e índices que los propios."""
    reference, recorded = base["runs"]["mac_online"]
    shared = tmp_path / "shared"
    monkeypatch.setenv(SHARED_INDICES_ENV, str(shared))
    output, factory = tmp_path / "shared-run", Factory()
    with unfused_attention():
        report = wf.run_titans_window(
            base["view"],
            base["protocol"],
            "fold-000",
            base["recipe"],
            variant="mac_online",
            seed=42,
            output=output,
            device="cpu",
            optimizer_factory=factory,
        )
    assert report["status"] == "completed"
    assert report["identity"]["indices"] == reference["identity"]["indices"]
    assert report["fit"]["best_score"] == reference["fit"]["best_score"]
    assert len(factory.records) == len(recorded.records) > 0
    for one, two in zip(factory.records, recorded.records, strict=True):
        for a, b in zip(one, two, strict=True):
            assert (a is None and b is None) or torch.equal(a, b)
    for partition in wf.PREDICTED:
        assert _compare(base["root"] / "runs" / "mac_online", output, partition)
    view = shared / sha256(base["view"])[:16]
    assert not (output / "indices").exists()
    assert {path.name.split("-")[0] for path in view.iterdir() if path.is_dir()} == {
        "train",
        *wf.PREDICTED,
    }


def test_traces_keep_the_window_bit_for_bit_and_resume_without_duplicates(base, tmp_path):
    """Trazas en cada paso (#448): mismos gradientes y filas, y una reanudación sin duplicados."""
    reference, recorded = base["runs"]["mac_online"]
    output = tmp_path / "traced"
    report, factory = window(
        base["view"], base["protocol"], base["recipe"], output, traces=EVERY_STEP
    )
    assert len(factory.records) == len(recorded.records) > 0
    for one, two in zip(factory.records, recorded.records, strict=True):
        for a, b in zip(one, two, strict=True):
            assert (a is None and b is None) or torch.equal(a, b)
    for partition in wf.PREDICTED:
        assert _compare(base["root"] / "runs" / "mac_online", output, partition)
    assert report["fit"]["best_score"] == reference["fit"]["best_score"]
    manifest, rows = trace_rows(output / "traces")
    assert manifest["identity"]["family"] == "titans_mac" and manifest["exhausted_at_step"] is None
    assert {"loss", "grad_l2", "update_ratio", "grad_l2_total", "learning_rate"} <= metrics(
        rows, "train"
    )
    # La memoria neuronal de mac_online registra sus puertas y sus pesos rápidos.
    assert {"memory.alpha.mean", "memory.theta.mean", "memory.fast_weight_l2.0.mean"} <= metrics(
        rows, "train"
    )
    steps = {row["step"] for row in rows if row["phase"] == "train"}
    assert steps == set(range(1, len(recorded.records) + 1))
    scores = [row["value"] for row in rows if row["metric"] == "validation.score"]
    assert report["fit"]["best_score"] in scores
    resumed_factory = Factory()

    class StopAtStep:
        @property
        def requested(self):
            return sum(item.calls for item in resumed_factory.instances[-1:]) >= 5

    resumed = tmp_path / "resumed"
    paused, _ = window(
        base["view"],
        base["protocol"],
        base["recipe"],
        resumed,
        optimizer_factory=resumed_factory,
        stop=StopAtStep(),
        traces=EVERY_STEP,
    )
    assert paused["status"] == "paused"
    final, _ = window(
        base["view"],
        base["protocol"],
        base["recipe"],
        resumed,
        optimizer_factory=resumed_factory,
        traces=EVERY_STEP,
    )
    assert final["status"] == "completed"
    assert same_rows(rows, trace_rows(resumed / "traces")[1])


def test_titans_posttraining_traces_the_adapter_and_keeps_its_rows(base, tmp_path):
    """Trazas de #448 del postentrenamiento de Titans-MAC con el prefijo persistente adaptado."""
    from mars_titan.posttraining import chronological_matrix as cm
    from mars_titan.posttraining import chronological_windows as cw
    from mars_titan.posttraining.adapter_matrix import read_matrix

    matrix, digest = read_matrix(CONFIGS / "posttraining" / "adapter-matrix-v3.json")
    matrix["budget"].update(epochs=1)
    item = next(
        item
        for item in cm.cases(matrix, digest, "titans_mac", variant="mac_online")
        if item["case"]["seed"] == 42 and item["id"].endswith("/persistent")
    )
    parent = base["root"] / "runs" / "mac_online"
    common = dict(case=item["case"], matrix=matrix, digest=digest, device="cpu")
    with unfused_attention():
        plain = cw.run_titans_posttraining(
            parent, base["view"], tmp_path / "plain", optimizer_factory=Factory(), **common
        )
        traced = cw.run_titans_posttraining(
            parent,
            base["view"],
            tmp_path / "traced",
            optimizer_factory=Factory(),
            traces=EVERY_STEP,
            **common,
        )
    assert plain["status"] == traced["status"] == "completed"
    for partition in wf.PREDICTED:
        assert (
            plain["predictions"][partition]["sha256"] == traced["predictions"][partition]["sha256"]
        )
    manifest, rows = trace_rows(tmp_path / "traced" / "traces")
    assert manifest["identity"]["family"] == "posttraining_titans_mac"
    deltas = [row for row in rows if row["metric"] == "adapter.delta_l2"]
    assert deltas and all(row["value"] == 0.0 for row in deltas)
    groups = {row["group"] for row in rows if row["metric"] == "grad_l2"}
    # Solo la corrección del prefijo persistente recibe gradiente.
    assert groups == {"predictor.mac"}
    assert "memory.alpha.mean" in metrics(rows, "train")


COMMON = dict(RULE, stopping="common_epoch", max_epochs=2)
SCORING = dict(min_assets=3, predictions=True)


def test_common_epoch_window_waits_then_emits_its_epoch_and_versions(base, tmp_path, monkeypatch):
    output = tmp_path / "common"
    options = dict(stopping=COMMON, epoch_scoring=SCORING)
    with pytest.raises(ValueError, match="junto con la época común"):
        window(base["view"], base["protocol"], base["recipe"], output, versions=[2], **options)
    with pytest.raises(ValueError, match="cómo se puntúa"):
        window(base["view"], base["protocol"], base["recipe"], output, stopping=COMMON)
    with pytest.raises(ValueError, match="solo existe con la época común"):
        window(base["view"], base["protocol"], base["recipe"], output, epoch_scoring=SCORING)
    assert not output.exists()
    with unfused_attention():
        report, _ = window(base["view"], base["protocol"], base["recipe"], output, **options)
    assert report["status"] == "awaiting_joint_stop" and report["individual_stop_epoch"] == 2
    assert report["predictions"] == {} and report["request"]["epoch_scoring"] == SCORING
    fit = json.loads((output / "fit" / "run.json").read_text())
    assert sorted(fit["epoch_states"]) == ["1", "2"]
    for entry in fit["history"][1:]:
        scored = entry["validation"]
        # El IC del ajuste es el del análisis sobre la predicción guardada de esa época.
        panel = ForecastPanel.from_arrow(
            pq.read_table(output / scored["epoch_predictions"]["path"])
        )
        values, defined = sd.session_correlations(panel, panel.prediction, min_assets=3)
        assert scored["session_ic_values"] == values.tolist() and defined.any()
        assert scored["session_ic_defined"] == defined.tolist()
        assert scored["session_mae_values"] == sd.session_mae(panel, panel.prediction).tolist()
        assert scored["cohort_sha256"] == panel.cohort_sha256
        assert scored["session_time"] == panel.session_time.tolist()
        assert np.mean(scored["session_mae_values"]) == pytest.approx(scored["session_mae"])
        assert scored["session_ic"] == pytest.approx(values[defined].mean())
    loaded = []
    original = ChronologicalTrainer.load_epoch

    def recorded(self, report, epoch):
        loaded.append(epoch)
        return original(self, report, epoch)

    monkeypatch.setattr(ChronologicalTrainer, "load_epoch", recorded)
    with unfused_attention():
        for broken in ([1], [2, 2], [3]):
            with pytest.raises(ValueError, match="versiones"):
                window(
                    base["view"],
                    base["protocol"],
                    base["recipe"],
                    output,
                    joint_epoch=1,
                    versions=broken,
                    **options,
                )
        final, _ = window(
            base["view"],
            base["protocol"],
            base["recipe"],
            output,
            joint_epoch=1,
            versions=[2],
            **options,
        )
    assert final["status"] == "completed" and final["fit"]["best_epoch"] == 1
    assert loaded == [2] and final["released_state_bytes"] > 0
    versions = final["versions"]
    assert versions["epochs"] == [2] and sorted(versions["predictions"]["2"]) == [
        "calibration",
        "evaluation",
    ]
    for name, record in versions["predictions"]["2"].items():
        # Con el optimizador doble los pesos no cambian, así que cada versión repite las filas.
        version = pq.read_table(output / record["path"]).sort_by("sample_id")
        main = pq.read_table(output / f"{name}-predictions.parquet").sort_by("sample_id")
        assert version.equals(main) and record["rows"] == final["predictions"][name]["rows"]
    # Solo queda el estado elegido de la época común.
    assert len(list((output / "fit" / "checkpoints").glob("state-*.pt"))) == 1
    again, _ = window(
        base["view"],
        base["protocol"],
        base["recipe"],
        output,
        joint_epoch=1,
        versions=[2],
        **options,
    )
    assert again == final
    (output / versions["predictions"]["2"]["evaluation"]["path"]).write_bytes(b"PAR1 x PAR1")
    with pytest.raises(ValueError, match="predicciones confirmadas"):
        window(
            base["view"],
            base["protocol"],
            base["recipe"],
            output,
            joint_epoch=1,
            versions=[2],
            **options,
        )
