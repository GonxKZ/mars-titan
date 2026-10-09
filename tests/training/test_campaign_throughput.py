"""Estimación de horas por familia y variante y medición de caudal sin pasos de optimizador.

La estimación se comprueba con caudales y recuentos fijados. Las mediciones recorren en
CPU el corpus técnico: las referencias y los adaptadores por lotes, Titans-MAC y la GRU
candidata con su ajuste cronológico real y un optimizador que no modifica pesos. Deben
dejar los pesos intactos, no crear optimizadores de PyTorch y retirar su gancho. No se usa
la GPU ni se registran pérdidas.
"""

import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.optim import optimizer as optim_module

from mars_titan.data.storage import atomic_json
from mars_titan.models.baselines import multimodal
from mars_titan.models.quantile_head import QUANTILE_HEAD
from mars_titan.posttraining import adapter_matrix, campaign_stage
from mars_titan.training import campaign_plan as plan
from mars_titan.training import campaign_throughput as throughput
from mars_titan.training import experiment_resources
from mars_titan.training.campaign_plan import EPISODIC, NEURAL, TITANS, load_campaign
from tests.training.test_walk_forward_v2_views import PROTOCOLS, fixture, prepare

CAMPAIGNS = {
    v: Path(f"configs/baselines/historical-masked-campaign-{v.lower()}.json") for v in "AB"
}
STAGES = {
    v: Path(f"configs/posttraining/historical-masked-adapter-stage-{v.lower()}.json") for v in "AB"
}
CANDIDATE = Path("configs/candidate/chronological-training.json")
TITANS_RECIPE = Path("configs/titans/chronological-training-historical-masked.json")
ROWS = dict(train=36_000, validation=4_000, calibration=2_000, evaluation=8_000)
# Ajuste con validación inicial: 30 épocas de 360 s y 32 validaciones, calibración y
# evaluación a 400 filas/s. Traslado: calibración y evaluación.
FIT, CARRY = 30 * 360 + (32 * 4_000 + 10_000) / 400, 10_000 / 400
WINDOWS = dict(
    A={"US": (19, 0), "CN": (13, 0), "US+CN": (13, 0)},
    B={"US": (7, 12), "CN": (5, 8), "US+CN": (5, 8)},
)
DECLARED, HALVED = "accumulation_rows=null", "accumulation_rows=128"


def uniform(campaign, rows):
    """Mismas filas por tramo en todas las ventanas y caudales iguales en todos los brazos."""
    counts = {
        scope: {window: dict(rows) for window in resolved["windows"]}
        for scope, resolved in campaign["comparison_config"]["resolved_scopes"].items()
        if scope in campaign["scopes"]
    }
    rates = {
        NEURAL: {
            arm: {name: dict(train=100.0, inference=400.0) for name, _ in candidates}
            for arm, candidates in campaign["neural"]["candidates"].items()
        }
    }
    return counts, rates


def chronological(arms, *, slower=50.0, failing=None):
    """Caudales de una familia cronológica con la opción declarada y una más lenta."""
    options = {
        DECLARED: dict(train=100.0, peak_vram_allocated_bytes=7),
        HALVED: dict(train=slower, peak_vram_allocated_bytes=3),
    }
    if failing is not None:
        options[failing] = dict(status=throughput.OUT_OF_MEMORY, peak_vram_allocated_bytes=9)
    return {
        arm: dict(declared_option=DECLARED, inference=400.0, options=copy.deepcopy(options))
        for arm in arms
    }


def matrix_points(stage, slower=1.0):
    """Caudales de cada brazo de la matriz por padre candidato y brazo base.

    El último padre candidato de cada brazo es `slower` veces más lento.
    """
    rates = {}
    for arm, family in stage["families"].items():
        names = [name for name, _ in stage["campaign"]["neural"]["candidates"][arm]]
        rates[arm] = {
            name: {
                item["id"].split("/", 1)[1]: dict(train=100.0 / factor, inference=400.0 / factor)
                for item in adapter_matrix.cases(
                    stage["matrix"], stage["matrix_sha256"], family, head=QUANTILE_HEAD
                )
            }
            for name, factor in zip(names, [1.0] * (len(names) - 1) + [slower], strict=True)
        }
    return rates


def test_job_seconds_follow_epochs_validation_and_final_predictions():
    rates = {
        "gru-00": dict(train=100.0, inference=400.0),
        "gru-10": dict(train=50.0, inference=200.0),
    }
    fit = dict(kind="fit", candidate="gru-00")
    # 30 épocas de 360 s de ajuste y 10 s de validación, más 50.000 filas finales a 400/s.
    assert throughput.neural_job_seconds(fit, ROWS, rates, 30) == pytest.approx(30 * 370 + 125)
    finalist = dict(kind="fit", candidate=None)
    assert throughput.neural_job_seconds(finalist, ROWS, rates, 30) == pytest.approx(30 * 740 + 250)
    carried = dict(kind="carry", candidate=None)
    assert throughput.neural_job_seconds(carried, ROWS, rates, 30) == pytest.approx(10_000 / 200)


def test_validated_jobs_count_the_initial_and_final_validations():
    rate = dict(train=100.0, inference=400.0)
    seconds = throughput.validated_job_seconds(dict(kind="fit"), ROWS, rate, 30)
    assert seconds == pytest.approx(10_800 + (31 * 4_000 + 4_000 + 2_000 + 8_000) / 400)
    assert throughput.validated_job_seconds(dict(kind="carry"), ROWS, rate, 30) == 25


def test_titans_rows_add_the_warmup_months_with_the_density_of_each_partition():
    fold = load_campaign(CAMPAIGNS["A"])["comparison_config"]["resolved_scopes"]["US"]
    rows = throughput.titans_rows(ROWS, fold["windows"]["fold-000"], 12)
    # Calentamiento de 366 días frente a 183, 92 y 365 días de cada tramo.
    assert rows == dict(
        train=36_000,
        validation=4_000 + 8_000,
        calibration=2_000 + round(2_000 * 366 / 92),
        evaluation=8_000 + round(8_000 * 366 / 365),
    )
    assert throughput.titans_rows(ROWS, fold["windows"]["fold-000"], 0) == ROWS
    # El calentamiento no empieza antes del origen del tramo de ajuste.
    early = dict(fold["windows"]["fold-000"], validation=["2000-07-01", "2000-10-01"])
    assert throughput.titans_rows(ROWS, early, 12)["validation"] == 4_000 + round(4_000 * 182 / 92)


@pytest.mark.parametrize("variant", ["A", "B"])
def test_neural_hours_add_every_planned_job_and_leave_the_rest_unmeasured(variant):
    campaign = load_campaign(CAMPAIGNS[variant])
    counts, rates = uniform(campaign, ROWS)
    estimate = throughput.estimate_hours(campaign, counts, rates)
    neural = estimate["families"][NEURAL]
    fit, carry = (30 * 370 + 125) / 3600, 10_000 / 400 / 3600
    for scope, (trained, carried) in WINDOWS[variant].items():
        expected = 5 * (4 * trained * fit + 3 * carried * carry)
        assert neural["scopes"][scope]["hours"] == pytest.approx(expected)
        assert neural["scopes"][scope]["arms"]["gru"] == pytest.approx(expected / 5)
    assert neural["hours"] == pytest.approx(sum(s["hours"] for s in neural["scopes"].values()))
    assert estimate["families"][TITANS] == dict(status="not_measured")
    assert EPISODIC not in estimate["families"]
    assert estimate["tabular"] == "not_measured" and estimate["variant"] == variant
    assert estimate["total_gpu_hours"]["without_estimate"] == [TITANS]


@pytest.mark.parametrize("variant", ["A", "B"])
def test_titans_hours_include_warmup_and_compare_memory_options(variant):
    campaign = load_campaign(CAMPAIGNS[variant])
    counts, rates = uniform(campaign, ROWS)
    rates[TITANS] = chronological(campaign[TITANS]["arms"])
    estimate = throughput.estimate_hours(campaign, counts, rates)
    titans = estimate["families"][TITANS]
    assert titans["declared"] == DECLARED and set(titans["options"]) == {DECLARED, HALVED}
    declared, halved = (titans["options"][name] for name in (DECLARED, HALVED))
    # La opción declarada es aquí también la más rápida de las que caben en memoria.
    totals = estimate["total_gpu_hours"]
    neural = estimate["families"][NEURAL]["hours"]
    assert totals["declared_options"] == pytest.approx(neural + declared["hours"])
    assert totals["fastest_options"] == pytest.approx(totals["declared_options"])
    fits = sum(4 * 4 * trained for trained, _ in WINDOWS[variant].values())
    carries = sum(4 * 3 * carried for _, carried in WINDOWS[variant].values())
    assert (declared["training_jobs"], declared["prediction_jobs"]) == (fits, carries)
    # Con la mitad de caudal de ajuste, cada ajuste suma 30 * 36.000 / 100 s más.
    assert halved["hours"] - declared["hours"] == pytest.approx(fits * 3.0)
    assert (declared["peak_vram_allocated_bytes"], halved["peak_vram_allocated_bytes"]) == (7, 3)
    # Cada ventana de US suma sus ajustes o traslados con su propio calentamiento.
    resolved = campaign["comparison_config"]["resolved_scopes"]["US"]["windows"]
    rate, expected = dict(train=100.0, inference=400.0), 0.0
    for row in plan.schedule(list(resolved.values()), campaign["period"]):
        rows = throughput.titans_rows(ROWS, resolved[row["window"]], 12)
        kind, jobs = ("fit", 4) if row["trained"] else ("carry", 3)
        expected += jobs * throughput.validated_job_seconds(dict(kind=kind), rows, rate, 30)
    arm = declared["scopes"]["US"]["arms"]["titans_mac_online"]
    assert arm == pytest.approx(expected / 3600)
    assert (
        arm > (4 * WINDOWS[variant]["US"][0] * FIT + 3 * WINDOWS[variant]["US"][1] * CARRY) / 3600
    )


def test_an_option_out_of_memory_has_no_hours_and_the_totals_use_the_rest():
    campaign = load_campaign(CAMPAIGNS["A"])
    counts, rates = uniform(campaign, ROWS)
    rates[TITANS] = chronological(campaign[TITANS]["arms"])
    # Basta con que un único control no quepa para descartar la opción en la familia.
    online = rates[TITANS]["titans_mac_online"]["options"]
    online[DECLARED] = dict(status="out_of_memory", peak_vram_allocated_bytes=9)
    estimate = throughput.estimate_hours(campaign, counts, rates)
    titans = estimate["families"][TITANS]
    assert titans["options"][DECLARED] == dict(status="out_of_memory", peak_vram_allocated_bytes=9)
    totals = estimate["total_gpu_hours"]
    assert totals["declared_options"] is None and totals["without_estimate"] == []
    assert totals["fastest_options"] == pytest.approx(
        estimate["families"][NEURAL]["hours"] + titans["options"][HALVED]["hours"]
    )


@pytest.mark.parametrize("variant", ["A", "B"])
def test_candidate_hours_use_the_section_it_would_declare(variant):
    campaign = throughput.with_candidate(load_campaign(CAMPAIGNS[variant]), CANDIDATE)
    section = campaign[EPISODIC]
    assert section["arms"] == {"gru_episodic": "m1_k1"} and section["seed"] == 42
    assert section["declared_in_campaign"] is False
    counts, rates = uniform(campaign, ROWS)
    rates[EPISODIC] = chronological(["gru_episodic"])
    candidate = throughput.estimate_hours(campaign, counts, rates)["families"][EPISODIC]
    assert candidate["declared_in_campaign"] is False
    for scope, (trained, carried) in WINDOWS[variant].items():
        expected = (3 * trained * FIT + 3 * carried * CARRY) / 3600
        assert candidate["options"][DECLARED]["scopes"][scope]["hours"] == pytest.approx(expected)
    with pytest.raises(ValueError, match="ya declara"):
        throughput.with_candidate(campaign, CANDIDATE)
    other = throughput.with_candidate(load_campaign(CAMPAIGNS[variant]), CANDIDATE, "m1_k2")
    assert other[EPISODIC]["arms"] == {"gru_episodic": "m1_k2"}
    with pytest.raises(ValueError, match="GRU candidata"):
        throughput.with_candidate(load_campaign(CAMPAIGNS[variant]), CANDIDATE, "m9")


@pytest.mark.parametrize(
    ("variant", "fits", "carries", "parents"),
    [("A", 3915, 0, 45 * 15), ("B", 1479, 2436, 17 * 15)],
)
def test_posttraining_hours_cover_every_stage_job_and_each_parent_cache(
    variant, fits, carries, parents
):
    stage = campaign_stage.load_stage(STAGES[variant])
    campaign = load_campaign(CAMPAIGNS[variant])
    counts, rates = uniform(campaign, ROWS)
    # El segundo padre candidato de cada brazo es la mitad de rápido, también al predecir
    # la caché. Sin padre elegido, la estimación usa el más lento.
    rates[throughput.POSTTRAINING] = matrix_points(stage, slower=2.0)
    for candidates in rates[NEURAL].values():
        list(candidates.values())[-1]["inference"] = 200.0
    estimate = throughput.estimate_hours(campaign, counts, rates, stage=stage)
    assert throughput.POSTTRAINING in plan.LATER_STAGES
    matrix = estimate["families"][throughput.POSTTRAINING]
    assert (matrix["training_jobs"], matrix["prediction_jobs"]) == (fits, carries)
    assert matrix["parent_caches"] == parents
    # La matriz ajusta 5 épocas: 5 de 720 s, 7 validaciones, calibración y evaluación.
    fit = 5 * 720 + (7 * 4_000 + 10_000) / 200
    expected = fits * fit + carries * 10_000 / 200 + parents * (36_000 + 4_000) / 200
    assert matrix["hours"] == pytest.approx(expected / 3600)
    other = campaign_stage.load_stage(STAGES["B" if variant == "A" else "A"])
    with pytest.raises(ValueError, match="no parte de esta campaña"):
        throughput.estimate_hours(campaign, counts, rates, stage=other)
    unmeasured = throughput.estimate_hours(campaign, *uniform(campaign, ROWS), stage=stage)
    assert unmeasured["families"][throughput.POSTTRAINING] == dict(status="not_measured")


def test_variant_b_costs_less_than_a_with_the_same_rates():
    estimates = []
    for variant in "AB":
        campaign = throughput.with_candidate(load_campaign(CAMPAIGNS[variant]), CANDIDATE)
        counts, rates = uniform(campaign, ROWS)
        rates[TITANS] = chronological(campaign[TITANS]["arms"])
        rates[EPISODIC] = chronological(["gru_episodic"])
        stage = campaign_stage.load_stage(STAGES[variant])
        rates[throughput.POSTTRAINING] = matrix_points(stage)
        estimates.append(throughput.estimate_hours(campaign, counts, rates, stage=stage))
        assert json.loads(json.dumps(estimates[-1])) == estimates[-1]
    comparison = throughput._comparison(estimates)
    assert comparison["A"]["without_estimate"] == []
    assert 0 < comparison["b_over_a"]["declared_options"] < 1
    assert comparison["b_over_a"]["fastest_options"] == pytest.approx(
        comparison["B"]["fastest_options"] / comparison["A"]["fastest_options"]
    )
    assert throughput._comparison(estimates[:1]) is None


# Medición en CPU


@pytest.fixture
def cpu(monkeypatch):
    import mars_titan.data.embeddings as embeddings

    monkeypatch.setattr(embeddings, "require_cuda", lambda **_: torch.device("cpu"))
    for name, value in dict(
        synchronize=None, reset_peak_memory_stats=None, max_memory_allocated=0
    ).items():
        monkeypatch.setattr(torch.cuda, name, lambda *_, value=value: value)
    monkeypatch.setattr(torch, "set_num_threads", lambda _: None)
    deterministic = torch.are_deterministic_algorithms_enabled()
    yield
    torch.use_deterministic_algorithms(deterministic)


@pytest.fixture(scope="module")
def views(tmp_path_factory):
    root = tmp_path_factory.mktemp("throughput")
    data = fixture(root / "data", ("US",))
    prepare(data, PROTOCOLS["US"], root / "views")
    return root / "views"


class Recorded(multimodal.MultimodalReference):
    """Referencia que guarda su estado inicial para comprobar que la medición no lo cambia."""

    built = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        Recorded.built.append((self, copy.deepcopy(self.state_dict())))


@pytest.fixture
def recorded(monkeypatch):
    Recorded.built = []
    monkeypatch.setattr(multimodal, "MultimodalReference", Recorded)
    return Recorded.built


def unchanged(built):
    for model, initial in built:
        for name, value in model.state_dict().items():
            assert torch.equal(value, initial[name]), name
        assert all(parameter.grad is None for parameter in model.parameters())


def test_measurement_runs_forward_and_backward_without_changing_weights(views, cpu, recorded):
    campaign = load_campaign(CAMPAIGNS["A"])
    campaign["neural"] = dict(
        campaign["neural"],
        batch_size=1,
        candidates={"gru": campaign["neural"]["candidates"]["gru"][:1]},
    )
    hooks = len(optim_module._global_optimizer_pre_hooks)
    rates = throughput.measure_rates(
        campaign, views / "fold-018/manifest.json", batches=1, warmup=0
    )
    assert len(optim_module._global_optimizer_pre_hooks) == hooks
    (record,) = rates["gru"].values()
    assert record["train"] > 0 and record["inference"] > 0
    assert record["measured_train_rows"] == record["measured_inference_rows"] == 1
    assert not {"loss", "mae", "session_mae"} & set(record)
    assert len(recorded) == 1
    unchanged(recorded)


def test_matrix_measurement_covers_every_case_without_changing_the_parent(
    views, cpu, recorded, monkeypatch
):
    stage = campaign_stage.load_stage(STAGES["A"])
    campaign = stage["campaign"]
    campaign["neural"] = dict(
        campaign["neural"], candidates={"gru": campaign["neural"]["candidates"]["gru"][:1]}
    )
    budget = dict(stage["matrix"]["budget"], batch_size=1)
    stage.update(families={"gru": "gru"}, matrix=dict(stage["matrix"], budget=budget))
    from mars_titan.posttraining import run as posttraining_run

    built, build = [], posttraining_run.build_model
    monkeypatch.setattr(
        posttraining_run,
        "build_model",
        lambda parent, case, *rest: built.append(case["seed"]) or build(parent, case, *rest),
    )
    hooks = len(optim_module._global_optimizer_pre_hooks)
    rates = throughput.measure_posttraining(
        stage, views / "fold-018/manifest.json", batches=1, warmup=0
    )
    assert built == [42] * 5
    assert len(optim_module._global_optimizer_pre_hooks) == hooks
    ((name, points),) = rates["gru"].items()
    assert name == "gru-00"
    assert set(points) == {"full_continuation", "head", "fusion", "head+fusion", "fusion_full_rank"}
    for point, record in points.items():
        assert record["train"] > 0 and record["inference"] > 0, point
        assert record["measured_train_rows"] == record["measured_inference_rows"] == 1
        assert not {"loss", "mae", "session_mae"} & set(record)
    parameters = sum(value.numel() for value in recorded[0][0].parameters())
    assert points["full_continuation"]["trainable_parameters"] == parameters
    assert 0 < points["head"]["trainable_parameters"] < parameters
    unchanged(recorded)


def test_measurements_reject_invalid_counts(tmp_path):
    campaign = load_campaign(CAMPAIGNS["A"])
    stage = campaign_stage.load_stage(STAGES["A"])
    absent = Path("absent.json")
    for options in (dict(batches=0), dict(warmup=-1), dict(batches=1.5)):
        with pytest.raises(ValueError, match="enteros acotados"):
            throughput.measure_rates(campaign, absent, **options)
        with pytest.raises(ValueError, match="enteros acotados"):
            throughput.measure_posttraining(stage, absent, **options)
    for options in (dict(segments=0), dict(event_warmup=-1), dict(events=True)):
        with pytest.raises(ValueError, match="enteros acotados"):
            throughput.measure_titans(campaign, absent, tmp_path, **options)
        with pytest.raises(ValueError, match="enteros acotados"):
            throughput.measure_candidate(campaign, absent, tmp_path, **options)


class Stopped(Exception):
    pass


class Steps:
    """Entrenador mínimo: cada paso recorre diez filas y consulta la parada en su barrera."""

    def __init__(self, option, *, failing=None, shift=None, steps=100):
        self.weight = torch.ones(3, requires_grad=True)
        self.optimizer = throughput._NoStepOptimizer([dict(params=[self.weight], role="shared")])
        self.option, self.failing, self.shift, self.steps = option, failing, shift, steps
        self.train = object()

    def _train_pass(self, run, cursor, stop, save):
        assert cursor == dict(epoch=0, phase="train", event=0, stage="start")
        for _ in range(self.steps):
            if self.option == self.failing:
                raise torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 2 GiB")
            run.counters["observations"] += 10
            (2 * self.weight).sum().backward()
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            if self.option == self.shift:
                with torch.no_grad():
                    self.weight.add_(1)
            # Como los entrenadores, se consulta antes de guardar y otra vez después.
            if stop.requested:
                save(dict(cursor, event=1, stage="inputs"), run)
                if stop.requested:
                    raise Stopped
        return {}

    evaluated = []

    def evaluate(self, source, *, stop):
        assert source is self.train
        Steps.evaluated.append(self.option)
        # Un evento por entrada de `chronological_record`.
        for _ in range(50):
            if stop.requested:
                raise Stopped
        return {}


SETTINGS = dict(segments=3, segment_warmup=1, events=4, event_warmup=2)
OPTIONS = [{"accumulation_rows": None}, {"accumulation_rows": 128}]


def chronological_record(cpu_settings=SETTINGS, inputs=(5,) * 50, **trainer):
    Steps.evaluated = []
    return throughput._chronological(
        lambda option: Steps(option, **trainer),
        lambda _: SimpleNamespace(counters=dict(observations=0)),
        Stopped,
        OPTIONS,
        list(inputs),
        cpu_settings,
    )


def test_chronological_measurement_times_segments_and_events_between_barriers(cpu):
    record = chronological_record()
    assert record["declared_option"] == DECLARED and set(record["options"]) == {DECLARED, HALVED}
    for name, option in zip((DECLARED, HALVED), OPTIONS, strict=True):
        result = record["options"][name]
        # Un paso de calentamiento, tres medidos y el paso que encuentra la parada.
        assert result["measured_train_rows"] == 30 and result["step_calls_without_update"] == 5
        assert result["train"] > 0 and result["accumulation_rows"] == option["accumulation_rows"]
    assert record["measured_inference_rows"] == 20 and record["inference"] > 0
    # La inferencia se mide una vez, con el entrenador de la opción de la receta.
    assert Steps.evaluated == [OPTIONS[0]]


def test_memory_options_start_with_the_declared_recipe_without_repetitions():
    recipe = SimpleNamespace(accumulation_rows=128, recompute=True)
    assert throughput._options(recipe, throughput.TITANS_OPTIONS) == [
        {"accumulation_rows": 128},
        {"accumulation_rows": None},
    ]
    options = throughput._options(recipe, throughput.CANDIDATE_OPTIONS)
    assert options[0] == {"accumulation_rows": 128, "recompute": True}
    assert len(options) == 4 and sorted(map(throughput.option_name, options)) == sorted(
        map(throughput.option_name, throughput.CANDIDATE_OPTIONS)
    )
    other = SimpleNamespace(accumulation_rows=64, recompute=False)
    assert len(throughput._options(other, throughput.CANDIDATE_OPTIONS)) == 5
    assert throughput.option_name(options[0]) == "accumulation_rows=128,recompute=true"


def test_an_option_without_memory_is_recorded_and_the_rest_are_measured(cpu):
    record = chronological_record(failing=OPTIONS[0])
    failed = record["options"][DECLARED]
    assert failed["status"] == "out_of_memory" and "train" not in failed
    assert failed["error"] == "CUDA out of memory. Tried to allocate 2 GiB"
    assert record["options"][HALVED]["measured_train_rows"] == 30
    assert record["measured_inference_rows"] == 20


def test_chronological_measurement_rejects_changed_weights_and_short_passes(cpu):
    with pytest.raises(RuntimeError, match="modificado parámetros"):
        chronological_record(shift=OPTIONS[1])
    with pytest.raises(ValueError, match="tramos o eventos suficientes"):
        chronological_record(steps=3)
    with pytest.raises(ValueError, match="tramos o eventos suficientes"):
        chronological_record(dict(SETTINGS, events=60))
    # Eventos que solo resuelven etiquetas: no hay filas observadas que medir.
    with pytest.raises(ValueError, match="tramos o eventos suficientes"):
        chronological_record(inputs=(0,) * 50)


def test_the_step_guard_rejects_any_optimizer_step_and_is_removed():
    hooks = dict(optim_module._global_optimizer_pre_hooks)
    handle = throughput._forbid_steps()
    try:
        (forbid,) = (
            hook
            for key, hook in optim_module._global_optimizer_pre_hooks.items()
            if key not in hooks
        )
        with pytest.raises(RuntimeError, match="no admite pasos"):
            forbid(None, (), {})
    finally:
        handle.remove()
    assert dict(optim_module._global_optimizer_pre_hooks) == hooks


def titans_campaign(folder):
    """Campaña A con un único control de Titans y la receta de campaña reducida."""
    campaign = load_campaign(CAMPAIGNS["A"])
    recipe = json.loads(TITANS_RECIPE.read_text())
    recipe["predictor"].update(hidden_size=32)
    recipe["recipe"].update(truncation=3, block_rows=2)
    folder.mkdir(parents=True, exist_ok=True)
    atomic_json(folder / "titans.json", recipe)
    arms = {"titans_mac_online": campaign["comparison_config"]["arms"]["titans_mac_online"]}
    section = dict(recipe="titans.json", arms={"titans_mac_online": "mac_online"}, search_seed=42)
    campaign[TITANS] = plan._titans(
        section, arms, campaign["rule"], campaign["input_policy"], folder, 2
    )
    return campaign


@pytest.fixture
def fused():
    """Dejar fastpath activado, el valor por defecto de un proceso, y restaurarlo después."""
    previous = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(True)
    yield
    torch.backends.mha.set_fastpath_enabled(previous)


@pytest.fixture
def guarded_steps(monkeypatch):
    """Registrar en cada paso pedido si el gancho de la medición está activo."""
    seen, hooks = [], len(optim_module._global_optimizer_pre_hooks)
    step = throughput._NoStepOptimizer.step

    def recorded(self):
        seen.append(len(optim_module._global_optimizer_pre_hooks) - hooks)
        step(self)

    monkeypatch.setattr(throughput._NoStepOptimizer, "step", recorded)
    yield seen
    assert len(optim_module._global_optimizer_pre_hooks) == hooks


def test_titans_measurement_walks_the_chronological_trainer_without_changing_weights(
    views, cpu, tmp_path, guarded_steps, fused
):
    campaign = titans_campaign(tmp_path / "config")
    rates = throughput.measure_titans(
        campaign,
        views / "fold-000/manifest.json",
        tmp_path / "work",
        segments=1,
        segment_warmup=0,
        events=1,
        event_warmup=0,
    )
    assert torch.backends.mha.get_fastpath_enabled() is True
    (record,) = rates.values()
    assert record["variant"] == "mac_online" and record["measured_case"] == "lr1e-4"
    assert record["shared_by_cases"] == ["lr1e-4", "lr1e-3"]
    assert record["declared_option"] == DECLARED and set(record["options"]) == {DECLARED, HALVED}
    for result in record["options"].values():
        assert result["train"] > 0 and result["measured_train_rows"] > 0
        assert result["step_calls_without_update"] == 2
    assert record["inference"] > 0 and 0 < record["measured_inference_rows"]
    assert record["measured_inference_rows"] <= record["max_event_inputs"]
    assert not {"loss", "mean_loss", "session_mae"} & set(record)
    assert guarded_steps and set(guarded_steps) == {1}
    assert (tmp_path / "work/titans-indices").is_dir()
    assert not (tmp_path / "work/titans-unused").exists()


@pytest.mark.skipif(
    not os.environ.get("MARS_TITAN_EPISODIC_NATIVE"), reason="Falta el enlace nativo compilado"
)
def test_candidate_measurement_compares_accumulation_and_recomputation(
    views, cpu, tmp_path, guarded_steps
):
    campaign = throughput.with_candidate(load_campaign(CAMPAIGNS["A"]), CANDIDATE)
    rates = throughput.measure_candidate(
        campaign,
        views / "fold-000/manifest.json",
        tmp_path / "work",
        segments=1,
        segment_warmup=0,
        events=1,
        event_warmup=0,
    )
    (record,) = rates.values()
    assert record["variant"] == "m1_k1"
    names = [throughput.option_name(option) for option in throughput.CANDIDATE_OPTIONS]
    assert list(record["options"]) == names
    assert record["declared_option"] == "accumulation_rows=null,recompute=false"
    for result in record["options"].values():
        assert result["train"] > 0 and result["step_calls_without_update"] == 2
    assert record["inference"] > 0 and record["measured_inference_rows"] > 0
    assert guarded_steps and set(guarded_steps) == {1}
    assert not (tmp_path / "work/candidate-unused").exists()


class Lease:
    record = dict(device="cuda:0", note="doble")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


@pytest.fixture
def doubled(monkeypatch):
    """Sustituir la GPU y las mediciones por caudales fijos y registrar cada llamada."""
    calls = []
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setattr(experiment_resources, "GpuLease", Lease)
    monkeypatch.setattr(
        throughput, "_window_counts", lambda campaign, views: uniform(campaign, ROWS)[0]
    )

    def double(name, result):
        def measured(subject, view, *args, **kwargs):
            calls.append((name, Path(view), args, kwargs))
            return result(subject)

        monkeypatch.setattr(throughput, name, measured)

    double("measure_rates", lambda campaign: uniform(campaign, ROWS)[1][NEURAL])
    double("measure_titans", lambda campaign: chronological(campaign[TITANS]["arms"]))
    double("measure_candidate", lambda campaign: chronological(["gru_episodic"]))
    double("measure_posttraining", matrix_points)
    return calls


def test_campaign_report_measures_each_family_once_and_compares_both_variants(doubled, tmp_path):
    view = tmp_path / "view.json"
    report = throughput.measure_campaigns(
        [CAMPAIGNS["A"], CAMPAIGNS["B"]],
        {},
        view,
        stages=[STAGES["A"], STAGES["B"]],
        candidate=dict(recipe=CANDIDATE),
        work=tmp_path / "work",
        output=tmp_path / "report.json",
        segments=4,
    )
    assert [name for name, *_ in doubled] == [
        "measure_rates",
        "measure_titans",
        "measure_candidate",
        "measure_posttraining",
    ]
    settings = dict(throughput.SETTINGS, segments=4)
    assert report["settings"] == settings
    for name, path, args, kwargs in doubled:
        assert path == view
        if name in ("measure_titans", "measure_candidate"):
            assert args == (tmp_path / "work",)
            assert kwargs == dict(segments=4, segment_warmup=2, events=64, event_warmup=8)
        else:
            assert kwargs == dict(batches=50, warmup=5)
    families = {NEURAL, TITANS, EPISODIC, throughput.POSTTRAINING}
    assert [e["variant"] for e in report["estimates"]] == ["A", "B"]
    assert all(set(e["families"]) == families for e in report["estimates"])
    assert 0 < report["comparison"]["b_over_a"]["declared_options"] < 1
    assert report["optimizer_steps"] == 0 and report["scientific_training_started"] is False
    assert report["resources"] == Lease.record and report["final_test_opened"] is False
    assert json.loads((tmp_path / "report.json").read_text()) == json.loads(json.dumps(report))


def test_campaign_report_fails_before_measuring_without_its_prerequisites(
    doubled, tmp_path, monkeypatch
):
    paths = [CAMPAIGNS["A"], CAMPAIGNS["B"]]
    with pytest.raises(ValueError, match="directorio de trabajo"):
        throughput.measure_campaigns(paths, {}, tmp_path / "view.json")
    with pytest.raises(ValueError, match="parte de una campaña medida"):
        throughput.measure_campaigns(
            paths[:1], {}, tmp_path / "v.json", stages=[STAGES["B"]], work=tmp_path
        )
    with pytest.raises(ValueError, match="enteros acotados"):
        throughput.measure_campaigns(paths, {}, tmp_path / "v.json", work=tmp_path, events=0)
    load = throughput.load_campaign

    def fewer(path):
        campaign = load(path)
        if campaign["variant"] == "B":
            candidates = dict(campaign["neural"]["candidates"], gru=[])
            campaign["neural"] = dict(campaign["neural"], candidates=candidates)
        return campaign

    monkeypatch.setattr(throughput, "load_campaign", fewer)
    with pytest.raises(ValueError, match="mismos candidatos"):
        throughput.measure_campaigns(paths, {}, tmp_path / "v.json", work=tmp_path)
    monkeypatch.setattr(throughput, "load_campaign", load)
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG")
    with pytest.raises(ValueError, match="CUBLAS_WORKSPACE_CONFIG"):
        throughput.measure_campaigns(paths, {}, tmp_path / "v.json", work=tmp_path)
    assert doubled == []


def test_campaign_report_measures_only_the_declared_families(doubled, tmp_path):
    report = throughput.measure_campaigns(
        [CAMPAIGNS["B"]], {}, tmp_path / "view.json", work=tmp_path / "work"
    )
    assert [name for name, *_ in doubled] == ["measure_rates", "measure_titans"]
    (estimate,) = report["estimates"]
    assert set(estimate["families"]) == {NEURAL, TITANS} and report["comparison"] is None


def test_script_runs_the_single_throughput_command(doubled, tmp_path, capsys):
    import runpy

    script = runpy.run_path("scripts/run_masked_campaign.py", run_name="script")
    arguments = ["throughput", "--campaign", str(CAMPAIGNS["A"]), "--campaign", str(CAMPAIGNS["B"])]
    arguments += ["--stage", str(STAGES["A"]), "--stage", str(STAGES["B"])]
    arguments += ["--candidate-recipe", str(CANDIDATE), "--views", f"US={tmp_path}"]
    arguments += ["--first-view", str(tmp_path / "view.json"), "--work", str(tmp_path / "work")]
    arguments += ["--output", str(tmp_path / "report.json"), "--events", "16"]
    assert script["main"](arguments) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed == json.loads((tmp_path / "report.json").read_text())
    assert printed["settings"]["events"] == 16 and len(doubled) == 4


def test_script_checks_a_campaign_and_its_posttraining_stage_without_reading_data(capsys):
    import runpy

    script = runpy.run_path("scripts/run_masked_campaign.py", run_name="script")
    assert script["main"](["check", "--campaign", str(CAMPAIGNS["B"])]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "checked" and report["counts"]["prediction_jobs"] == 868
    assert script["main"](["posttraining", "check", "--stage", str(STAGES["B"])]) == 0
    stage = json.loads(capsys.readouterr().out)
    assert stage["status"] == "checked" and stage["variant"] == "B"
    assert (stage["counts"]["training_jobs"], stage["counts"]["prediction_jobs"]) == (1479, 2436)


def test_script_runs_the_posttraining_stage_only_without_the_hold(learning_hold, tmp_path):
    import runpy

    from mars_titan.training.learning_hold import LearningHoldError

    learning_hold(False)
    script = runpy.run_path("scripts/run_masked_campaign.py", run_name="script")
    arguments = ["posttraining", "run", "--stage", str(STAGES["A"]), "--views", f"US={tmp_path}"]
    arguments += [
        "--campaign-output",
        str(tmp_path / "campaign"),
        "--output",
        str(tmp_path / "out"),
    ]
    with pytest.raises(LearningHoldError):
        script["main"](arguments)
    assert not (tmp_path / "out").exists()
