"""Orden y análisis de los gemelos de escritura sobre un estado con la disposición de la campaña.

La orden repite cada ancla con su propia variante (paridad bit a bit) y construye los dos gemelos
(`titans_walk_forward.carry_titans` con `evaluated_variant`). Con el registrador de gradientes de
las pruebas los pesos no cambian, así que los gemelos repiten las ventanas cruzadas y el análisis
debe dar N = D = A e I = 0 exactamente. La estadística se comprueba además con series construidas:
la covarianza HAC conjunta frente al error de la criba, Fieller acotado o no, los contrastes del
factorial y el promedio de semillas sesión a sesión.
"""

import json
import os
import shutil

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mars_titan.training import titans_walk_forward as wf
from mars_titan.training import write_twins as twins
from mars_titan.training.window_indices import SHARED_INDICES_ENV
from tests.training.test_titans_walk_forward import (  # noqa: F401
    base,
    learning_doubles_module,
)

pytestmark = pytest.mark.usefixtures("learning_doubles")

# --- Orden de los gemelos sobre un estado con la disposición de la campaña ---------------------


def declaration(**changes):
    document = dict(
        schema_version=1,
        kind=twins.KIND,
        final_test_access=False,
        description="Gemelos de escritura del fixture de la ventana técnica.",
        market="US",
        min_assets=3,
        pairs=[dict(online="titans_mac", frozen="titans_mac_frozen")],
        runs=[dict(state="t", origin=2023, seeds=[42])],
        analysis=dict(
            primary="evaluation", secondary=["validation", "calibration"], confidence=0.95
        ),
    )
    document.update(changes)
    return document


def campaign_state(base, root):  # noqa: F811
    """Estado con la disposición de la campaña y copias de las ventanas del fixture.

    Son copias y no enlaces: los traslados rechazan cualquier enlace en sus rutas.
    """
    runs = base["root"] / "runs"
    for model, variant in (("titans_mac", "mac_online"), ("titans_mac_frozen", "mac_frozen")):
        shutil.copytree(runs / variant, root / "runs" / model / "2023" / "seed-42")
    shutil.copytree(base["view"].parent, root / "cohorts" / "2023")
    return root


@pytest.mark.parametrize(
    "changes",
    [
        dict(final_test_access=True),
        dict(kind="other"),
        dict(runs=[dict(state="t", origin=2024, seeds=[42])]),
        dict(runs=[dict(state="t", origin=2023, seeds=[43, 42])]),
        dict(runs=[dict(state="t", origin=2023, seeds=[42, 42])]),
        dict(pairs=[dict(online="titans_mac", frozen="titans_mac")]),
        dict(pairs=[]),
        dict(min_assets=1),
        dict(analysis=dict(primary="evaluation", secondary=["evaluation"], confidence=0.95)),
        dict(analysis=dict(primary="evaluation", secondary=[], confidence=1.0)),
        dict(extra=True),
    ],
)
def test_the_declaration_is_checked_before_opening_any_state(changes):
    with pytest.raises(ValueError):
        twins.checked_declaration(declaration(**changes))


def test_twins_reproduce_the_parity_and_the_crossed_cells_of_the_factorial(
    base,  # noqa: F811
    tmp_path,
):
    states = dict(t=campaign_state(base, tmp_path / "state"))
    output, logs = tmp_path / "twins", []
    (summary,) = twins.run_twins(declaration(), states, output, device="cpu", log=logs.append)
    jobs = summary["jobs"]
    assert set(jobs) == set(twins.JOBS)
    for name in ("parity_online", "parity_frozen"):
        parity = jobs[name]["parity"]
        assert parity["identical"] is True and set(parity["partitions"]) == set(wf.PREDICTED)
    assert jobs["online_fit_frozen_eval"]["write_policy"]["evaluated_variant"] == "mac_frozen"
    assert jobs["frozen_fit_online_eval"]["write_policy"]["evaluated_variant"] == "mac_online"
    assert summary["numerics"]["float32_matmul_precision"] == "highest"
    # Los índices compartidos solo se fijan durante la orden.
    assert SHARED_INDICES_ENV not in os.environ
    assert all(job["peak"]["host_max_rss_bytes"] > 0 for job in jobs.values())
    report = twins.analyze(declaration(), states, output)
    assert json.loads((output / "report.json").read_text()) == json.loads(json.dumps(report))
    for partition, result in report["pairs"]["titans_mac"]["partitions"].items():
        ic, contrasts = result["ic"], result["contrasts"]
        # El perfil reparte las sesiones del tramo entre sus trimestres.
        assert sum(item["sessions"] for item in result["quarters"].values()) == result["sessions"]
        assert set(result["trend"]) == {"N", "D", "A", "I"}
        # Oráculo: sin pasos de optimizador, cada gemelo es la ventana de la otra variante.
        assert ic["of"] == ic["ff"] and ic["fo"] == ic["oo"], partition
        assert contrasts["N"]["mean"] == contrasts["D"]["mean"] == contrasts["A"]["mean"]
        assert contrasts["I"]["mean"] == 0
        ratio = result["ttws"]
        assert ratio["estimate"] == (None if contrasts["D"]["mean"] == 0 else 1.0)
        assert result["sessions"] == contrasts["N"]["sessions"] > 0
    # Repetir la orden reutiliza los trabajos terminados y aparta el que quedó a medias.
    folder = output / "t--2023--seed-42--titans_mac"
    before = {name: job["receipt_sha256"] for name, job in jobs.items()}
    (folder / "online_fit_frozen_eval" / "carry.json").unlink()
    logs.clear()
    (again,) = twins.run_twins(declaration(), states, output, device="cpu", log=logs.append)
    assert sum("ya hecho" in line for line in logs) == 3
    assert again["jobs"]["parity_online"]["peak"] is None
    assert (folder / "online_fit_frozen_eval.interrumpido-1").is_dir()
    assert {
        name: job["receipt_sha256"]
        for name, job in again["jobs"].items()
        if name != "online_fit_frozen_eval"
    } == {name: digest for name, digest in before.items() if name != "online_fit_frozen_eval"}
    with pytest.raises(ValueError, match="otra declaración"):
        twins.run_twins(declaration(min_assets=4), states, output, device="cpu")
    # Un trabajo terminado cuyas tablas cambiaron no se reutiliza.
    produced = folder / "parity_online" / "evaluation-predictions.parquet"
    original = produced.read_bytes()
    produced.write_bytes(original[:-16] + bytes(16))
    with pytest.raises(ValueError, match="Han cambiado"):
        twins.run_twins(declaration(), states, output, device="cpu")
    produced.write_bytes(original)
    # Sin paridad exacta registrada no se interpreta nada.
    summary_path = folder / "twins.json"
    recorded = json.loads(summary_path.read_text())
    recorded["jobs"]["parity_frozen"]["parity"]["identical"] = False
    summary_path.write_text(json.dumps(recorded))
    with pytest.raises(ValueError, match="no se interpreta nada"):
        twins.analyze(declaration(), states, output)


def test_a_parity_that_is_not_bit_exact_stops_before_any_twin(
    base,  # noqa: F811
    tmp_path,
):
    states = dict(t=campaign_state(base, tmp_path / "state"))

    def flipped(anchor, anchor_view, view, output, **options):
        receipt = wf.carry_titans(anchor, anchor_view, view, output, **options)
        if options["evaluated_variant"] is None:
            path = output / "evaluation-predictions.parquet"
            table = pq.read_table(path)
            values = table["prediction"].to_numpy().copy()
            values[0] = np.nextafter(values[0], np.inf)
            column = table.schema.get_field_index("prediction")
            pq.write_table(table.set_column(column, "prediction", pa.array(values)), path)
        return receipt

    output = tmp_path / "twins"
    with pytest.raises(ValueError, match="paridad .* falla en evaluation"):
        twins.run_twins(declaration(), states, output, device="cpu", carry=flipped)
    folder = output / "t--2023--seed-42--titans_mac"
    failure = json.loads((folder / "parity-failure.json").read_text())
    assert failure["first"]["column"] == "prediction" and failure["first"]["differing_rows"] == 1
    assert not (folder / "online_fit_frozen_eval").exists()
    with pytest.raises(ValueError, match="Faltan los gemelos"):
        twins.analyze(declaration(), states, output)


class _CudaAllocator:
    """El contrato del asignador de PyTorch en la GPU, comprobado en la GB10 con torch 2.14: sus
    máximos no se pueden poner a cero antes de inicializar CUDA («Invalid device argument»), y
    el primer recorrido de la orden empieza sin contexto."""

    def __init__(self):
        self.started = False
        self.resets = 0

    def init(self):
        self.started = True

    def reset_peak_memory_stats(self, device):
        if not self.started:
            raise RuntimeError("Invalid device argument ")
        self.resets += 1

    def max_memory_allocated(self, device):
        return 3 * 1024**2

    def max_memory_reserved(self, device):
        return 5 * 1024**2


def test_on_the_gpu_each_run_resets_its_peaks_once_cuda_has_started(
    base,  # noqa: F811
    tmp_path,
    monkeypatch,
):
    import torch

    allocator = _CudaAllocator()
    for name in ("init", "reset_peak_memory_stats", "max_memory_allocated", "max_memory_reserved"):
        monkeypatch.setattr(torch.cuda, name, getattr(allocator, name))
    states = dict(t=campaign_state(base, tmp_path / "state"))

    def on_cpu(anchor, anchor_view, view, output, **options):
        return wf.carry_titans(anchor, anchor_view, view, output, **dict(options, device="cpu"))

    (summary,) = twins.run_twins(
        declaration(), states, tmp_path / "twins", device="cuda:0", carry=on_cpu
    )
    assert allocator.resets == len(twins.JOBS)
    reserved = {job["peak"]["cuda_max_reserved_bytes"] for job in summary["jobs"].values()}
    assert reserved == {5 * 1024**2}


# --- Estadística de los gemelos ---------------------------------------------------------------


def test_the_joint_hac_covariance_extends_the_screening_error_to_several_series():
    rng = np.random.default_rng(3)
    values = rng.standard_normal((300, 2)) @ np.array([[1.0, 0.4], [0.0, 0.8]])
    values[1:, 0] += 0.5 * values[:-1, 0]
    blocks = ["a"] * 180 + ["b"] * 120
    covariance, lags = twins.long_run_covariance(values, blocks)
    screening = twins._screening()
    for column in range(2):
        _, se, used = screening._newey_west_hac(values[:, column], blocks=blocks)
        assert covariance[column, column] == pytest.approx(se**2, rel=1e-12)
        assert used == lags
    assert covariance[0, 1] == covariance[1, 0] and covariance[0, 1] > 0


def test_the_ratio_has_a_bounded_fieller_interval_only_with_a_clear_denominator():
    rng = np.random.default_rng(5)
    denominator = 0.02 + 0.01 * rng.standard_normal(251)
    numerator = 0.5 * denominator + 0.002 * rng.standard_normal(251)
    strong = twins.ratio_statistics(numerator, denominator, 0.95)
    assert strong["estimate"] == pytest.approx(numerator.mean() / denominator.mean())
    low, high = strong["fieller"]["interval"]
    assert strong["fieller"]["bounded"] and low < strong["estimate"] < high and low < 0.5 < high
    assert 0 < strong["delta_se"] < high - low
    weak = twins.ratio_statistics(numerator, denominator - denominator.mean(), 0.95)
    assert weak["fieller"]["bounded"] is False and "no está acotado" in weak["fieller"]["reason"]
    zero = twins.ratio_statistics(numerator, np.zeros(251), 0.95)
    assert zero["estimate"] is None and zero["fieller"]["bounded"] is False


def test_session_ics_are_those_of_the_screening_with_its_thirty_assets(tmp_path):
    rng = np.random.default_rng(7)
    rows = []
    for session in range(6):
        assets = 40 if session != 2 else 20
        target = rng.standard_normal(assets)
        for asset in range(assets):
            rows.append(
                dict(
                    sample_id=f"s{session}-{asset:02d}",
                    market="US",
                    prediction_at=1_700_000_000_000_000 + session * 86_400_000_000,
                    target=float(target[asset]),
                    prediction=float(target[asset] + rng.standard_normal()),
                )
            )
    table = pa.Table.from_pylist(rows).set_column(
        2, "prediction_at", pa.array([r["prediction_at"] for r in rows], pa.timestamp("us", "UTC"))
    )
    pq.write_table(table, tmp_path / "evaluation-predictions.parquet")
    ics, _, _ = twins.session_ics(tmp_path, "evaluation", 30)
    screening = twins._screening()
    assert ics == screening.session_ic(screening.load_split(tmp_path, "evaluation"))
    assert len(ics) == 5
    assert len(twins.session_ics(tmp_path, "evaluation", 20)[0]) == 6


def test_the_first_difference_names_its_row_sample_and_values(tmp_path):
    table = pa.table(dict(sample_id=["b", "a", "c"], prediction=[0.5, 0.25, -1.0]))
    pq.write_table(table, tmp_path / "left.parquet")
    changed = table.set_column(1, "prediction", pa.array([0.5, np.nextafter(0.25, 1), -1.0]))
    pq.write_table(changed, tmp_path / "right.parquet")
    found = twins.first_difference(tmp_path / "left.parquet", tmp_path / "right.parquet")
    assert found["column"] == "prediction" and found["sample_id"] == "a"
    assert found["row"] == 0 and found["differing_rows"] == 1
    assert twins.first_difference(tmp_path / "left.parquet", tmp_path / "left.parquet") is None


def test_the_factorial_contrasts_separate_writing_from_fitting():
    rng = np.random.default_rng(11)
    noise = 0.002 * rng.standard_normal((4, 251))
    ic = dict(oo=0.10 + noise[0], of=0.07 + noise[1], ff=0.06 + noise[2], fo=0.05 + noise[3])
    result = twins.partition_statistics(ic, [("t", 2013, "US")] * 251, 0.95)
    means = {name: value["mean"] for name, value in result["contrasts"].items()}
    assert means["N"] == pytest.approx(0.03, abs=1e-3) and means["D"] == pytest.approx(
        0.04, abs=1e-3
    )
    assert means["A"] == pytest.approx(-0.01, abs=1e-3) and means["I"] == pytest.approx(
        0.04, abs=1e-3
    )
    assert means["I"] == pytest.approx(means["N"] - means["A"], abs=1e-15)
    assert result["readings"] == dict(N="positive", D="positive", A="negative", I="positive")
    assert result["ttws"]["estimate"] == pytest.approx(means["N"] / means["D"])
    assert result["ttws"]["interpretable"] is True
    assert result["contrasts"]["N"]["mde80"] == pytest.approx(
        2.8 * result["contrasts"]["N"]["se_hac"]
    )
    # Con el control congelado por encima, el cociente está acotado pero no se interpreta.
    reversed_ = dict(ic, ff=0.14 + noise[2])
    reversed_result = twins.partition_statistics(reversed_, [("t", 2013, "US")] * 251, 0.95)
    assert reversed_result["readings"]["D"] == "negative"
    assert reversed_result["ttws"]["fieller"]["bounded"] is True
    assert reversed_result["ttws"]["interpretable"] is False
    flat = dict(ic, of=ic["oo"] + 0.002 * rng.standard_normal(251))
    assert twins.partition_statistics(flat, [("t", 2013, "US")] * 251, 0.95)["readings"]["N"] == (
        "not_detected"
    )


def test_seeds_are_averaged_session_by_session_as_replicates():
    keys = [("US", 1), ("US", 2), ("US", 3)]
    first = {cell: dict(zip(keys, [0.1, 0.2, 0.3], strict=True)) for cell in twins.CELLS}
    second = {cell: dict(zip(keys, [0.3, 0.0, 0.1], strict=True)) for cell in twins.CELLS}
    second["of"] = dict(zip(keys, [0.5, 0.5, 0.5], strict=True))
    averaged = twins.seed_average({42: first, 43: second}, keys)
    np.testing.assert_allclose(averaged["oo"], [0.2, 0.1, 0.2])
    np.testing.assert_allclose(averaged["of"], [0.3, 0.35, 0.4])
    np.testing.assert_allclose(
        averaged["of"], twins.seed_average({43: second, 42: first}, keys)["of"]
    )


def test_the_profile_gives_each_contrast_by_calendar_quarter_and_its_trend_per_year():
    rng = np.random.default_rng(13)
    days = np.sort(rng.choice(365, size=251, replace=False))
    # 2013-01-01 a las 21:00 UTC, en microsegundos, más un día laborable por sesión.
    instants = 1_357_074_000_000_000 + days * 86_400_000_000
    year = days / 365.25
    noise = 0.002 * rng.standard_normal((4, 251))
    # Escribir gana 0,01 en enero y la ventaja cae 0,02 por año; la del control no cambia.
    ic = dict(
        oo=0.10 + noise[0],
        of=0.09 + 0.02 * year + noise[1],
        ff=0.08 + noise[2],
        fo=0.08 + noise[3],
    )
    # Dos bloques de mercado: el error HAC de cada trimestre usa los bloques de sus sesiones.
    blocks = [("t", 2013, "US" if i % 3 else "CN") for i in range(251)]
    result = twins.partition_statistics(ic, blocks, 0.95, instants=instants)
    contrasts = twins.factorial_contrasts(ic)
    months = instants.astype("datetime64[us]").astype("datetime64[M]").astype(np.int64) % 12
    labels = np.array([f"2013T{month // 3 + 1}" for month in months])
    assert list(result["quarters"]) == ["2013T1", "2013T2", "2013T3", "2013T4"]
    screening = twins._screening()
    for label, item in result["quarters"].items():
        rows = np.flatnonzero(labels == label)
        assert item["sessions"] == len(rows)
        for name in ("N", "D", "A", "I"):
            values = contrasts[name][rows]
            assert item[name]["mean"] == pytest.approx(values.mean(), abs=1e-15)
            _, se, _ = screening._newey_west_hac(values, blocks=[blocks[i] for i in rows])
            assert item[name]["se_hac"] == pytest.approx(se, rel=1e-12)
    trend = result["trend"]
    x = (instants - instants[0]) / (365.25 * 86_400_000_000)
    slope = np.polyfit(x, contrasts["N"], 1)[0]
    assert trend["N"]["slope_per_year"] == pytest.approx(slope, rel=1e-9)
    assert trend["N"]["slope_per_year"] == pytest.approx(-0.02, abs=0.005)
    centered = x - x.mean()
    residual = contrasts["N"] - contrasts["N"].mean() - slope * centered
    _, se_mean, _ = screening._newey_west_hac(centered * residual, blocks=blocks)
    expected = se_mean * len(x) / np.sum(centered**2)
    assert trend["N"]["se_hac"] == pytest.approx(expected, rel=1e-6)
    assert trend["N"]["t"] < -5 and abs(trend["D"]["t"]) < 3
    assert trend["N"]["sessions"] == 251
    # Sin instantes no hay perfil.
    assert "quarters" not in twins.partition_statistics(ic, blocks, 0.95)
