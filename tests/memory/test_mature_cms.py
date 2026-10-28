"""R2 de #786: periodos de escritura de E1, pasos proximales de periodo cerrado y madurez."""

import hashlib
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest
import torch

from mars_titan.memory import multiscale_mixture
from mars_titan.memory.area_memory import AreaLayout
from mars_titan.memory.associative_memory import (
    AssociativeMemoryConfig,
    MatureCorrection,
    _canonical,
)
from mars_titan.memory.multiscale_mixture import (
    DAY_US,
    EPOCH,
    MultiscaleMixtureConfig,
    period_index,
    state_tensor_bytes,
)
from mars_titan.memory.role_binding import RoleBindingConfig

PROXIMAL = AssociativeMemoryConfig("proximal", rate=0.5)
PLAIN = MatureCorrection(PROXIMAL)
HALF_LIFE = 91.0
CMS = MultiscaleMixtureConfig(
    half_lives_days=(HALF_LIFE,) * 3,
    weighting="uniform",
    weight_rate=0.0,
    write_periods=("session", "week", "month"),
)
CORRECTION = MatureCorrection(PROXIMAL, multiscale=CMS)
# Huellas de E1 calculadas con el código de develop anterior a R2 (1cbfcfde5).
E1_IDENTITY = "3940db5547c6ecb80c4f45db06d3c045a7de5262714ef8921ea15005c832e1b3"
E1_CORRECTION = "1eab85c72ec50b055e5ae7b6b26a1877f4b2d5ebfd3bcd770cef180edd6b7272"
E1_STATE = "1b0bf809da5cc8a567ea7e9b668e0801748d0beb63f24f8186066479bbd1d29c"
# Huellas de R2 calculadas al introducirlo (a60780a8b): la identidad y el estado tras la primera
# semana abierta de `history`.
R2_IDENTITY = "8926e348bdc3f7101ccedf0aa7c6128ee7332add35b319951101bce9a1bb12d1"
R2_STATE = "8993014ad37005c034bae126b607543092961535600d5b994416ad5eeb61a8af"
E1_FIELDS = {
    "schema_version",
    "configuration",
    "experts",
    "log_weights",
    "clock",
    "cohorts",
    "writes",
    "cursor",
    "matrix_sha256",
    "state_sha256",
}


def micros(*moment):
    return (datetime(*moment, tzinfo=UTC) - EPOCH) // timedelta(microseconds=1)


# Lunes, miércoles y viernes de una semana, el lunes siguiente y el primer día de febrero.
MOMENTS = (
    micros(2024, 1, 8, 15),
    micros(2024, 1, 10, 15),
    micros(2024, 1, 12, 15),
    micros(2024, 1, 15, 15),
    micros(2024, 2, 1, 15),
)


def unit_keys(rows, generator):
    keys = torch.randn(rows, 64, generator=generator, dtype=torch.float64)
    return keys / torch.linalg.vector_norm(keys, dim=1, keepdim=True)


def cohort(memory, at, generator, rows=2, values=None):
    if values is None:
        values = (torch.randn(rows, generator=generator, dtype=torch.float64) * 0.02).tolist()
    return PLAIN.feedback(
        ids=[memory.writes + index + 1 for index in range(rows)],
        decision_at=[at - 1] * rows,
        available_at=[at] * rows,
        keys=unit_keys(rows, generator),
        values=values,
    )


def write(memory, feedback, at):
    rows = feedback.ids.shape[0]
    issued = torch.zeros(rows, len(memory.mixture.half_lives_days), dtype=torch.float64)
    return memory.write(feedback, cutoff=at, issued_predictions=issued)


def proximal(matrix, keys, values, weights):
    """(I + η KᵀWK) A⁺ = A + η KᵀWv, la regla proximal sin olvido discreto."""
    weighted = keys * weights.unsqueeze(1)
    system = torch.eye(64, dtype=torch.float64) + PROXIMAL.rate * weighted.T @ keys
    return torch.linalg.solve(system, matrix + PROXIMAL.rate * weighted.T @ values.reshape(-1, 1))


def history(rows=(2, 2, 2, 2, 2), seed=3):
    """Memoria de R2 tras cada momento y los resultados escritos en cada uno."""
    generator = torch.Generator().manual_seed(seed)
    memory, states, written = CORRECTION.empty(), [], []
    for at, count in zip(MOMENTS, rows, strict=True):
        feedback = cohort(memory, at, generator, rows=count)
        memory = write(memory, feedback, at)
        states.append(memory)
        written.append(feedback)
    return states, written


def test_periods_follow_iso_weeks_from_monday_and_utc_months():
    sunday, monday = micros(2024, 1, 14, 23, 59, 59, 999999), micros(2024, 1, 15)
    assert period_index("week", micros(2024, 1, 8)) == period_index("week", sunday)
    assert period_index("week", monday) == period_index("week", sunday) + 1
    # El 1 de enero de 1970 fue jueves, así que su semana empezó el lunes 29 de diciembre.
    assert period_index("week", 0) == period_index("week", micros(1969, 12, 29)) == 0
    assert period_index("week", micros(1969, 12, 28, 23)) == -1
    leap = micros(2024, 2, 29, 23, 59, 59, 999999)
    assert period_index("month", leap) + 1 == period_index("month", micros(2024, 3, 1))
    assert period_index("month", micros(2024, 1, 31)) == 2024 * 12
    with pytest.raises(ValueError, match="Solo week y month"):
        period_index("session", 0)
    with pytest.raises(ValueError, match="microsegundos"):
        period_index("week", 1.5)
    with pytest.raises(ValueError, match="fuera del calendario"):
        period_index("month", 2**62)


class Name(str):
    """Un texto que solo se parece al nombre de un periodo."""


@pytest.mark.parametrize(
    "periods",
    [
        ("session", "session", "session"),
        ("week", "session", "month"),
        ("week", "month", "month"),
        ("session", "month", "week"),
        ("session", "week"),
        ("session", "day", "month"),
        "session",
        {"session", "week", "month"},
        ("session", 7, "month"),
        ("session", Name("week"), "month"),
    ],
    ids=[
        "all_session",
        "slow_first",
        "without_session",
        "unordered",
        "length",
        "unknown",
        "text",
        "set",
        "integer",
        "subclass",
    ],
)
def test_periods_must_be_one_per_expert_ordered_and_with_a_slow_one(periods):
    with pytest.raises(ValueError, match="R2 declara"):
        replace(CMS, write_periods=periods)


def test_r2_needs_the_proximal_rule_without_areas_or_roles():
    with pytest.raises(ValueError, match="regla proximal"):
        MatureCorrection(AssociativeMemoryConfig("delta", rate=0.5), multiscale=CMS)
    with pytest.raises(ValueError, match="regla proximal"):
        MatureCorrection(PROXIMAL, areas=AreaLayout(), multiscale=CMS)
    with pytest.raises(ValueError, match="regla proximal"):
        MatureCorrection(PROXIMAL, role_binding=RoleBindingConfig(), multiscale=CMS)
    # La memoria también lo comprueba, porque recibe el corrector sin la mezcla.
    for unmixed in (
        MatureCorrection(AssociativeMemoryConfig("delta", rate=0.5)),
        MatureCorrection(PROXIMAL, areas=AreaLayout()),
        MatureCorrection(PROXIMAL, role_binding=RoleBindingConfig()),
    ):
        with pytest.raises(ValueError, match="regla proximal"):
            multiscale_mixture.MultiscaleMemory(unmixed, CMS)


def test_without_periods_the_identity_and_state_of_e1_do_not_change():
    config = MultiscaleMixtureConfig()
    assert hashlib.sha256(_canonical(config.identity()).encode()).hexdigest() == E1_IDENTITY
    correction = MatureCorrection(PROXIMAL, multiscale=config)
    digest = hashlib.sha256(_canonical(correction.identity()).encode()).hexdigest()
    assert digest == E1_CORRECTION
    generator = torch.Generator().manual_seed(5)
    keys = torch.randn(3, 64, generator=generator, dtype=torch.float64)
    keys = keys / torch.linalg.vector_norm(keys, dim=1, keepdim=True)
    feedback = PLAIN.feedback(
        ids=[1, 2, 3],
        decision_at=[5, 6, 7],
        available_at=[10, 10, 10],
        keys=keys,
        values=[0.01, -0.02, 0.03],
    )
    payload = write(correction.empty(), feedback, 10).export()
    assert set(payload) == E1_FIELDS and payload["state_sha256"] == E1_STATE
    # E1 no guarda periodos abiertos.
    with pytest.raises(ValueError, match="no comparten"):
        multiscale_mixture.MultiscaleMemory(PLAIN, config, pending=(None, None, None))


def test_the_identity_names_the_periods_only_when_declared():
    identity = CMS.identity()
    assert hashlib.sha256(_canonical(identity).encode()).hexdigest() == R2_IDENTITY
    assert identity["write_periods"] == ["session", "week", "month"]
    assert identity["continuum"] == "mature_cms_v1"
    assert "write_periods" not in replace(CMS, write_periods=None).identity()
    assert CMS == replace(CMS, write_periods=["session", "week", "month"])


def test_a_slow_expert_closes_its_period_with_one_mean_proximal_step():
    states, written = history()
    matrices = [torch.zeros(64, 1, dtype=torch.float64) for _ in range(3)]
    waiting = {1: [], 2: []}
    expected_pending = ([0, 2, 2], [0, 4, 4], [0, 6, 6], [0, 2, 8], [0, 2, 2])
    clock = None
    for at, feedback, memory, pending in zip(
        MOMENTS, written, states, expected_pending, strict=True
    ):
        decay = 1.0 if clock is None else 2 ** (-(at - clock) / (DAY_US * HALF_LIFE))
        matrices = [matrix * decay for matrix in matrices]
        matrices[0] = proximal(matrices[0], feedback.keys, feedback.values, feedback.weights)
        for index, period in ((1, "week"), (2, "month")):
            if waiting[index] and period_index(period, waiting[index][-1][0]) != period_index(
                period, at
            ):
                # Los eventos tienen las mismas filas, así que pesar cada evento por igual
                # es pesar cada fila por igual.
                keys = torch.cat([keys for _, keys, _ in waiting[index]])
                values = torch.cat([values for *_, values in waiting[index]])
                weights = torch.full((keys.shape[0],), 1 / keys.shape[0], dtype=torch.float64)
                matrices[index] = proximal(matrices[index], keys, values, weights)
                waiting[index] = []
            waiting[index].append((at, feedback.keys, feedback.values))
        clock = at
        parts = memory.predict(feedback.keys, at=at)[1]
        expected = torch.cat([feedback.keys @ matrix for matrix in matrices], dim=1)
        torch.testing.assert_close(parts, expected, rtol=0, atol=1e-12)
        assert memory.pending_rows == pending
    # Hasta cerrar su primera semana el experto semanal no ha escrito nada.
    first_week = states[2].predict(written[2].keys, at=MOMENTS[2])[1]
    assert not torch.count_nonzero(first_week[:, 1:])


def test_each_event_of_a_closed_period_weighs_the_same_whatever_its_rows():
    states, written = history(rows=(3, 1, 2, 2, 2))
    first, second = written[0], written[1]
    keys = torch.cat([first.keys, second.keys, written[2].keys])
    values = torch.cat([first.values, second.values, written[2].values])
    # Tres eventos en la semana: cada uno pesa 1/3, repartido entre sus filas.
    weights = torch.tensor([1 / 9] * 3 + [1 / 3] + [1 / 6] * 2, dtype=torch.float64)
    expected = proximal(torch.zeros(64, 1, dtype=torch.float64), keys, values, weights)
    probe = written[3].keys
    parts = states[3].predict(probe, at=MOMENTS[3])[1]
    torch.testing.assert_close(parts[:, 1:2], probe @ expected, rtol=0, atol=1e-12)


def test_no_speed_reads_a_result_before_the_write_that_closes_its_period():
    generator = torch.Generator().manual_seed(9)
    left = right = CORRECTION.empty()
    probe = unit_keys(4, torch.Generator().manual_seed(1))
    for index, at in enumerate(MOMENTS):
        feedback = cohort(left, at, generator)
        values = feedback.values.clone()
        if index == 2:
            # Solo cambia el último resultado de la primera semana, ya maduro pero pendiente.
            values[0, 0] += 0.5
        left = write(left, feedback, at)
        right = write(right, replace(feedback, values=values), at)
        later = at + DAY_US // 2
        changed = (left.predict(probe, at=later)[1] != right.predict(probe, at=later)[1]).any(0)
        # La sesión lo lee desde su escritura, la semana desde el lunes siguiente y el mes
        # desde febrero.
        assert changed.tolist() == [index >= 2, index >= 3, index >= 4]
        for memory in (left, right):
            # Un experto lento no tiene cursor hasta su primer periodo cerrado.
            payload = memory.export()
            for expert in payload["experts"]:
                assert expert["cursor"] is None or expert["cursor"][0] <= memory.clock
            for waiting in payload["pending"][1:]:
                assert int(waiting["available_at"].max()) <= memory.clock


def test_restore_keeps_the_open_periods_and_the_next_write():
    states, _ = history()
    middle = states[2]
    payload = middle.export()
    assert set(payload) == E1_FIELDS | {"pending", "pending_sha256"}
    assert payload["pending"][0] is None and payload["pending"][2]["events"] == 3
    assert payload["state_sha256"] == R2_STATE
    # La huella del estado cubre la de sus periodos abiertos.
    assert middle._state_digest(dict(payload, pending_sha256="0" * 64)) != R2_STATE
    restored = CORRECTION.restore(payload)
    assert restored.export()["state_sha256"] == payload["state_sha256"]
    # Ni la memoria ni la restaurada comparten tensores con el estado exportado.
    digest = payload["pending_sha256"]
    for item in payload["pending"][1:]:
        item["values"].add_(1.0)
    assert middle.export()["pending_sha256"] == digest == restored.export()["pending_sha256"]
    generator = torch.Generator().manual_seed(11)
    feedback = cohort(middle, MOMENTS[3], generator)
    left, right = write(middle, feedback, MOMENTS[3]), write(restored, feedback, MOMENTS[3])
    assert left.export()["state_sha256"] == right.export()["state_sha256"]
    assert left.export()["pending_sha256"] == right.export()["pending_sha256"]


def keep_rows(item, rows, events=None):
    for name in ("ids", "decision_at", "available_at", "keys", "values", "weights"):
        item[name] = item[name][rows]
    if events is not None:
        item["events"] = events


def double_events(item):
    item.update(events=2 * item["events"], weights=2 * item["weights"])


# Tras la tercera escritura cada experto lento espera los seis resultados de tres eventos.
CORRUPTIONS = {
    "events": (lambda p: p["pending"][1].update(events=4), "sumar uno"),
    "events_and_weights": (lambda p: double_events(p["pending"][1]), "no es coherente"),
    "value": (lambda p: p["pending"][1]["values"].add_(1.0), "huellas"),
    "missing": (lambda p: p["pending"].__setitem__(1, None), "guarda su periodo abierto"),
    "session": (lambda p: p["pending"].__setitem__(0, p["pending"][1]), "no esperan"),
    "period": (lambda p: p["pending"][2].update(period=p["pending"][2]["period"] + 1), "coherente"),
    "float_period": (
        lambda p: p["pending"][2].update(period=float(p["pending"][2]["period"])),
        "enteros",
    ),
    "order": (lambda p: keep_rows(p["pending"][1], [1, 0, 2, 3, 4, 5]), "no es coherente"),
    "dropped_event": (lambda p: keep_rows(p["pending"][1], [2, 3, 4, 5], 2), "no es coherente"),
    "empty": (lambda p: keep_rows(p["pending"][1], []), "entre 1 y"),
    "last_id": (lambda p: p["pending"][1]["ids"][-1:].add_(100), "no es coherente"),
    "digest": (lambda p: p.pop("pending_sha256"), "configuración multiescala"),
    "digest_value": (lambda p: p.update(pending_sha256="0" * 64), "huellas"),
    "length": (lambda p: p["pending"].pop(), "un periodo abierto por experto"),
    "not_a_list": (lambda p: p.update(pending=None), "un periodo abierto por experto"),
    "field": (lambda p: p["pending"][1].pop("weights"), "todos sus campos"),
}


@pytest.mark.parametrize("name", CORRUPTIONS)
def test_restore_rejects_a_corrupted_open_period_for_its_reason(name):
    corrupt, reason = CORRUPTIONS[name]
    states, _ = history()
    payload = states[2].export()
    corrupt(payload)
    with pytest.raises(ValueError, match=reason):
        CORRECTION.restore(payload)


def test_a_memory_without_writes_has_neither_open_periods_nor_written_experts():
    states, _ = history()
    for corrupt in (
        lambda p: p["pending"].__setitem__(1, states[2].export()["pending"][1]),
        # El experto semanal de la quinta escritura ya cerró dos semanas.
        lambda p: p["experts"].__setitem__(1, states[4].export()["experts"][1]),
    ):
        payload = CORRECTION.empty().export()
        corrupt(payload)
        with pytest.raises(ValueError, match="no esperan"):
            CORRECTION.restore(payload)


def test_an_open_period_holds_at_most_the_cohort_limit(monkeypatch):
    monkeypatch.setattr(multiscale_mixture, "MAX_COHORT", 4)
    generator = torch.Generator().manual_seed(2)
    memory = CORRECTION.empty()
    for at in MOMENTS[:2]:
        memory = write(memory, cohort(memory, at, generator), at)
    # Dos eventos de dos filas llenan justo el límite de la semana y del mes.
    assert memory.pending_rows == [0, 4, 4]
    with pytest.raises(ValueError, match="como máximo 4"):
        write(memory, cohort(memory, MOMENTS[2], generator, rows=1), MOMENTS[2])


def test_the_open_period_keeps_each_event_in_canonical_order():
    generator = torch.Generator().manual_seed(4)
    memory = CORRECTION.empty()
    feedback = cohort(memory, MOMENTS[0], generator, rows=3)
    # Los mismos resultados llegan con los identificadores en orden inverso.
    reverse = [2, 1, 0]
    shuffled = PLAIN.feedback(
        ids=feedback.ids[reverse].tolist(),
        decision_at=feedback.decision_at[reverse].tolist(),
        available_at=feedback.available_at[reverse].tolist(),
        keys=feedback.keys[reverse],
        values=feedback.values[reverse].flatten().tolist(),
    )
    ordered, arrived = write(memory, feedback, MOMENTS[0]), write(memory, shuffled, MOMENTS[0])
    for waiting in arrived.export()["pending"][1:]:
        assert waiting["ids"].tolist() == [1, 2, 3]
    assert ordered.export()["state_sha256"] == arrived.export()["state_sha256"]


def test_state_bytes_count_the_open_periods_and_their_bound():
    states, _ = history()
    per_row = (3 + 64 + 2) * 8
    experts = (3 * 64 + 3) * 8
    assert states[2].numeric_state_bytes == experts + 2 * 6 * per_row
    assert states[3].numeric_state_bytes == experts + (2 + 8) * per_row
    bound = state_tensor_bytes(CORRECTION._unmixed(), CMS)
    assert bound == experts + 2 * 8192 * per_row
    assert state_tensor_bytes(PLAIN, replace(CMS, write_periods=None)) == experts


def test_empty_results_do_not_open_or_close_a_period():
    states, _ = history()
    memory = states[2]
    empty = PLAIN.feedback(
        ids=[],
        decision_at=[],
        available_at=[],
        keys=torch.empty(0, 64, dtype=torch.float64),
        values=[],
    )
    assert write(memory, empty, MOMENTS[3]) is memory
