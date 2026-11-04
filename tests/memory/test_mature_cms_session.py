"""R2 (#786) en la sesión financiera y en la ventana de correcciones, con periodos cerrados."""

from dataclasses import replace

import pytest
import torch
from test_area_correction_window import decisions
from test_area_memory import proximal_step
from test_financial_session_associative import four_flow_source as four_flow_source
from test_financial_session_associative import frozen_consumer as frozen_consumer
from test_financial_session_associative import module_backend as module_backend
from test_financial_session_associative import no_target_estimation as no_target_estimation
from test_financial_session_associative import shared_native as shared_native
from test_financial_session_associative import trajectory
from test_mars_titan_correction_parity import RECIPE, core, core_pass, corrected_pass, emitted
from test_mars_titan_session_parity import events

from mars_titan.memory import multiscale_mixture
from mars_titan.memory.associative_memory import AssociativeMemoryConfig, MatureCorrection
from mars_titan.memory.multiscale_mixture import DAY_US, MultiscaleMixtureConfig
from mars_titan.training.financial_run import ChronologicalInference

CMS = MatureCorrection(
    AssociativeMemoryConfig("proximal", rate=0.25),
    multiscale=MultiscaleMixtureConfig(
        half_lives_days=(91.0,) * 3,
        weighting="uniform",
        weight_rate=0.0,
        write_periods=("session", "week", "month"),
    ),
)
# La fixture escribe del 6 al 9 de julio de 2021, dentro de una semana y de un mes.
WRITTEN_ROWS = 16


def compressed(period, at):
    """Calendario comprimido para cerrar periodos en la fixture: semanas de un día, meses de dos."""
    return at // DAY_US if period == "week" else at // (2 * DAY_US)


@pytest.mark.native("MARS_TITAN_EPISODIC_NATIVE")
@pytest.mark.parametrize("calendar", ["utc", "compressed"])
def test_session_window_and_recovery_keep_the_open_periods(
    shared_native, four_flow_source, frozen_consumer, tmp_path, calendar, monkeypatch
):
    if calendar == "compressed":
        monkeypatch.setattr(multiscale_mixture, "period_index", compressed)
    direct = trajectory(
        shared_native, four_flow_source, frozen_consumer, tmp_path / "direct", correction=CMS
    )
    recovered = trajectory(
        shared_native,
        four_flow_source,
        frozen_consumer,
        tmp_path / "recovered",
        correction=CMS,
        recover=True,
    )
    actual = [(a, t, v) for e in direct["events"] for _, a, t, v in e["predictions"]]
    assert actual == [(a, t, v) for e in recovered["events"] for _, a, t, v in e["predictions"]]
    assert direct["snapshot"] == recovered["snapshot"]
    window, _ = corrected_pass(four_flow_source, CMS)
    assert emitted(window) == actual
    assert direct["events"][-1]["memory"]["state_sha256"] == window.memory.export()["state_sha256"]
    applied = window.memory.applied_writes
    pending = window.memory.pending_rows
    assert [a + p for a, p in zip(applied, pending, strict=True)] == [WRITTEN_ROWS] * 3
    if calendar == "utc":
        assert applied == [WRITTEN_ROWS, 0, 0]
    else:
        # Con semanas de un día el experto semanal aplica cada día anterior al escribir.
        assert applied[1] == WRITTEN_ROWS - 4 and 0 < applied[2] < WRITTEN_ROWS


def test_window_matches_an_independent_recurrence_of_closed_periods(four_flow_source, monkeypatch):
    monkeypatch.setattr(multiscale_mixture, "period_index", compressed)
    plain = ChronologicalInference(core(four_flow_source), RECIPE, audit=True)
    core_pass(plain, events(four_flow_source))
    core_values = {(row[2], row[3]): row[4] for row in plain.audit if row[0] == "prediction"}
    half_lives = CMS.multiscale.half_lives_days
    matrices = [torch.zeros(64, 1, dtype=torch.float64) for _ in range(3)]
    open_periods = {1: None, 2: None}
    clock, pending, expected = None, {}, []
    for event in events(four_flow_source):
        aged = [
            matrix * (1.0 if clock is None else 2 ** (-(event.at - clock) / (DAY_US * h)))
            for matrix, h in zip(matrices, half_lives, strict=True)
        ]
        staged = sorted(
            (at, flow, label - core_values[flow, at], pending.pop((flow, at)))
            for flow, at, label in event.labels
        )
        if event.inputs:
            rows, keys = decisions(four_flow_source, event)
            parts = torch.cat([keys @ matrix for matrix in aged], dim=1)
            for (flow, at), key, predictions in zip(rows, keys, parts, strict=True):
                # Pesos uniformes: la corrección es la media de los tres expertos.
                expected.append((flow, at, core_values[flow, at] + predictions.mean().item()))
                pending[flow, at] = key
        if staged:
            values = torch.tensor([value for _, _, value, _ in staged], dtype=torch.float64)
            keys = torch.stack([key for *_, key in staged])
            matrices = [proximal_step(aged[0], keys, values, CMS.memory), aged[1], aged[2]]
            for index, period in ((1, "week"), (2, "month")):
                current, waiting = compressed(period, event.at), open_periods[index]
                if waiting is not None and waiting[0] != current:
                    # Cada evento tiene cuatro filas, así que pesar las filas por igual es pesar
                    # los eventos por igual.
                    chunk_keys, chunk_values = torch.cat(waiting[1]), torch.cat(waiting[2])
                    matrices[index] = proximal_step(
                        aged[index], chunk_keys, chunk_values, CMS.memory
                    )
                    waiting = None
                if waiting is None:
                    waiting = (current, [], [])
                waiting[1].append(keys)
                waiting[2].append(values)
                open_periods[index] = waiting
            clock = event.at
    inference, metrics = corrected_pass(four_flow_source, CMS)
    actual = emitted(inference)
    assert [row[:2] for row in actual] == [row[:2] for row in expected]
    torch.testing.assert_close(
        torch.tensor([row[2] for row in actual], dtype=torch.float64),
        torch.tensor([row[2] for row in expected], dtype=torch.float64),
        rtol=0,
        atol=1e-12,
    )
    # El experto mensual ha cerrado al menos un periodo, así que la recurrencia lo comprueba.
    assert any(torch.count_nonzero(matrix) for matrix in matrices[1:])
    assert metrics["multiscale"]["applied_writes"] == inference.memory.applied_writes
    assert metrics["multiscale"]["open_period_rows"] == inference.memory.pending_rows
    # Sin periodos las métricas de E1 conservan su forma.
    e1 = replace(CMS, multiscale=replace(CMS.multiscale, write_periods=None))
    assert set(corrected_pass(four_flow_source, e1)[1]["multiscale"]) == {
        "weights",
        "clock",
        "cohorts",
    }
