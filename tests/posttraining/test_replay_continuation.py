"""Repaso de filas anteriores en la continuación por etapas (#55), sin pasos de optimizador.

Las pruebas puras fijan la sección de la matriz, los casos, el presupuesto, el orden y el
régimen. Las de datos usan las dos ventanas US de la campaña reducida de `staged_rows`
(padre en fold-000, filas nuevas de fold-001 desde 2022) y comparan con una lectura propia
de las etiquetas qué sesiones admite el repaso y qué lotes entrega cada época. El recorrido
de la etapa usa el optimizador del fixture `recorder`, que nunca cambia pesos.
"""

import copy
import json
import math
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mars_titan.data.input_policy import HISTORICAL_MASKED
from mars_titan.data.price_windows import CONTEXT_SESSIONS, PRICE_WINDOW_CHANNELS
from mars_titan.data.storage import atomic_json, sha256
from mars_titan.memory.regimes import UNCLASSIFIED
from mars_titan.models.quantile_head import QUANTILE_HEAD
from mars_titan.posttraining import (
    adapter_matrix,
    campaign_stage,
    matrix_runs,
    stage_comparison,
)
from mars_titan.posttraining import chronological_matrix as cm
from mars_titan.posttraining import replay_continuation as replay
from mars_titan.posttraining.inputs import PairedInputs
from mars_titan.posttraining.parent_cache import ParentCache
from mars_titan.posttraining.replay_inputs import ReplayInputs
from mars_titan.posttraining.run import case_code, validate_case
from mars_titan.training import campaign_chain
from tests.posttraining.campaign_fixture import CpuLease, base_campaign
from tests.posttraining.real_only import real_data_only
from tests.posttraining.test_adapter_matrix import parent
from tests.posttraining.test_staged_rows import labels, micros, views  # noqa: F401

CONFIGS = "configs/posttraining"
SECTION = replay.SECTION_NAME
SINCE = "2022-01-01"
REPLAY_MODULES = {
    "posttraining/replay_continuation.py",
    "posttraining/replay_inputs.py",
    "environments/view_cohorts.py",
    "memory/regimes.py",
}


@pytest.fixture(scope="module")
def matrix():
    return adapter_matrix.read_matrix(f"{CONFIGS}/adapter-matrix-v3.json")


def _mutated(document, change):
    value = copy.deepcopy(document)
    change(value)
    return value


def controls(items):
    return [item["control"] for item in items if item["control"] is not None]


def test_the_references_add_three_replay_controls_with_the_continuation_case(matrix):
    document, digest = matrix
    assert document[SECTION] == dict(ratio=0.5, campaign=["references"])
    expected = ["full_continuation", "anchored_continuation", *replay.CONTROLS]
    for family in adapter_matrix.FAMILIES:
        items = adapter_matrix.cases(document, digest, family, head=QUANTILE_HEAD)
        for seed in document["budget"]["seeds"]:
            own = {item["control"]: item for item in items if item["case"]["seed"] == seed}
            assert controls(own.values()) == expected
            full = own["full_continuation"]["case"]
            for control in replay.CONTROLS:
                case = own[control]["case"]
                # Mismo objetivo, presupuesto, λ, semilla y selección. Solo añade el repaso.
                assert case == dict(full, replay=dict(policy=control, ratio=0.5))
                assert case["selection"]["patience"] is None
                assert own[control]["id"] == f"seed-{seed}/{control}"
                validate_case(case)
                assert REPLAY_MODULES <= set(case_code(case, masked=True))
            assert not REPLAY_MODULES & set(case_code(full, masked=True))


def test_the_chronological_families_have_no_replay(matrix):
    document, digest = matrix
    for family, variant in [("titans_mac", "mac_online"), ("cm_v1", None), ("episodic_gru", None)]:
        found = cm.cases(document, digest, family, variant=variant, reserve=True)
        assert not set(controls(found)) & set(replay.CONTROLS), family


def test_without_proposal_the_controls_stay_in_reserve(matrix):
    document, digest = matrix
    value = adapter_matrix.validate_matrix(
        _mutated(document, lambda v: v[SECTION].update(campaign=[]))
    )
    assert not set(controls(adapter_matrix.cases(value, digest, "gru", head=QUANTILE_HEAD))) & set(
        replay.CONTROLS
    )
    reserve = adapter_matrix.cases(value, digest, "gru", head=QUANTILE_HEAD, reserve=True)
    assert set(replay.CONTROLS) <= set(controls(reserve))


INVALID = {
    "integer_ratio": lambda v: v[SECTION].update(ratio=1),
    "zero_ratio": lambda v: v[SECTION].update(ratio=0.0),
    "above_one": lambda v: v[SECTION].update(ratio=1.5),
    "nan_ratio": lambda v: v[SECTION].update(ratio=float("nan")),
    "text_ratio": lambda v: v[SECTION].update(ratio="0.5"),
    "extra_key": lambda v: v[SECTION].update(policy="replay_uniform"),
    "missing_campaign": lambda v: v[SECTION].pop("campaign"),
    "chronological_scope": lambda v: v[SECTION].update(campaign=["readers"]),
    "repeated_scope": lambda v: v[SECTION].update(campaign=["references", "references"]),
    "scope_string": lambda v: v[SECTION].update(campaign="references"),
    "with_patience": lambda v: v["selection"].update(patience=3),
}


@pytest.mark.parametrize("name", sorted(INVALID))
def test_invalid_sections_are_rejected(matrix, name):
    with pytest.raises(ValueError):
        adapter_matrix.validate_matrix(_mutated(matrix[0], INVALID[name]))


def test_only_version_three_admits_the_section(matrix):
    document, _ = adapter_matrix.read_matrix(f"{CONFIGS}/adapter-matrix-v2.json")
    with pytest.raises(ValueError, match="contrato"):
        adapter_matrix.validate_matrix(
            dict(copy.deepcopy(document), **{SECTION: matrix[0][SECTION]})
        )


def test_a_replay_case_is_only_a_supervised_full_continuation(matrix):
    document, digest = matrix
    items = {
        item["id"].split("/", 1)[1]: item["case"]
        for item in adapter_matrix.cases(document, digest, "gru", head=QUANTILE_HEAD)
        if item["case"]["seed"] == 42
    }
    case = items[replay.UNIFORM]
    for broken in (
        dict(items["fusion"], replay=case["replay"]),
        dict(case, weight_decay_anchor="initial_parameters"),
        dict(case, mode="mae"),
        dict(case, condition="real_resampled"),
    ):
        with pytest.raises(ValueError, match="repaso pertenece"):
            validate_case(broken)
    for spec in (
        dict(policy="replay_recent", ratio=0.5),
        dict(policy=replay.UNIFORM, ratio=0.0),
        dict(policy=replay.UNIFORM),
        dict(policy=replay.UNIFORM, ratio=0.5, seed=1),
    ):
        with pytest.raises(ValueError):
            validate_case(dict(case, replay=spec))


@pytest.mark.parametrize("family", ["gru", "transformer"])
def test_the_plan_gives_the_replay_controls_their_own_equal_budget(matrix, family, monkeypatch):
    document, digest = matrix
    model = parent(family)
    plan = adapter_matrix.plan(
        document, digest, family, model, updates_per_epoch=13, linear_features=1734
    )
    rows = {row["id"]: row for row in plan}
    epochs = document["budget"]["epochs"]
    full = rows["seed-42/full_continuation"]
    assert full["updates"] == epochs * 13
    for control in replay.CONTROLS:
        row = rows[f"seed-42/{control}"]
        # ceil(0,5 · 13) = 7 actualizaciones de repaso en cada época.
        assert row["updates"] == epochs * (13 + 7)
        assert row["trainable_parameters"] == full["trainable_parameters"]
        assert row["state_bytes"] == full["state_bytes"]
        assert row["invalidates"] == full["invalidates"]
    others = {row["updates"] for row in plan[1:] if row["control"] not in replay.CONTROLS}
    assert others == {epochs * 13}
    # Un repaso con otra proporción rompería la igualdad entre los tres controles.
    original = adapter_matrix.cases

    def uneven(*args, **kwargs):
        items = original(*args, **kwargs)
        for item in items:
            if item["control"] == replay.REPEAT:
                item["case"]["replay"] = dict(item["case"]["replay"], ratio=0.25)
        return items

    monkeypatch.setattr(adapter_matrix, "cases", uneven)
    with pytest.raises(ValueError, match="no comparten presupuesto"):
        adapter_matrix.plan(
            document, digest, family, model, updates_per_epoch=13, linear_features=1734
        )


def test_extra_updates_and_the_budget_estimate():
    assert replay.extra_updates(13, 0.5) == 7 and replay.extra_updates(14, 0.5) == 7
    assert replay.extra_updates(1, 0.01) == 1
    # 100 filas en lotes de 8 son al menos 13 actualizaciones, y 7 lotes de repaso completos.
    assert replay.estimated_epoch_rows(100, 0.5, 8) == 100 + 7 * 8
    for bad in (0, 1.0):
        with pytest.raises(ValueError):
            replay.extra_updates(bad, 0.5)


def test_each_session_is_split_into_its_own_batches():
    # Sesiones de 2, 2 y 3 filas en lotes de 4: tres actualizaciones, no ceil(7 / 4) = 2.
    assert replay.epoch_updates([2, 2, 3], 4) == 3
    assert replay.epoch_updates([5, 1], 2) == 4


def test_the_order_and_the_positions_draw_from_their_declared_streams():
    """Orden con [semilla, 5501] y posiciones con [semilla, época, 5502], como en el diseño."""
    assert (replay.SELECTION_STREAM, replay.ORDER_STREAM) == (5501, 5502)
    expected = np.random.default_rng([7, 5501]).permutation(20).tolist()
    assert replay.order(list(range(20)), replay.UNIFORM, seed=7) == expected
    rng = np.random.default_rng([7, 2, 5502])
    slots = np.sort(rng.choice(30, size=10, replace=False)).tolist()
    assert replay.positions(20, 10, seed=7, epoch=2) == (slots, rng.permutation(10).tolist())


def test_the_order_is_fixed_by_the_seed_and_balances_the_regimes():
    candidates = list(range(40))
    uniform = replay.order(candidates, replay.UNIFORM, seed=7)
    assert sorted(uniform) == candidates
    assert uniform == replay.order(candidates, replay.UNIFORM, seed=7)
    assert uniform != replay.order(candidates, replay.UNIFORM, seed=8)
    routes = [1] * 30 + [3] * 8 + [4] * 2
    balanced = replay.order(candidates, replay.REGIME, seed=7, routes=routes)
    assert sorted(balanced) == candidates
    # Una sesión de cada régimen por turno: las seis primeras cubren los tres regímenes dos
    # veces, aunque el primero tenga quince veces más sesiones que el último.
    first = [routes[i] for i in balanced[:6]]
    assert sorted(first) == [1, 1, 3, 3, 4, 4]
    assert Counter(routes[i] for i in balanced[:10]) == {1: 4, 3: 4, 4: 2}
    with pytest.raises(ValueError, match="ruta de cada sesión"):
        replay.order(candidates, replay.REGIME, seed=7)
    with pytest.raises(ValueError, match="no pertenece"):
        replay.order(candidates, replay.REGIME, seed=7, routes=[5] * 40)


def test_the_plan_reaches_the_declared_updates_exactly():
    rows = [10, 3, 17, 8, 1]
    # Lotes de 4: 3, 1, 5, 2 y 1. Con 7 actualizaciones la tercera sesión queda recortada.
    assert replay.plan([0, 1, 2, 3, 4], rows, batch_size=4, updates=7) == [(0, 3), (1, 1), (2, 3)]
    assert replay.plan([4, 3], rows, batch_size=4, updates=3) == [(4, 1), (3, 2)]
    with pytest.raises(ValueError, match="no bastan"):
        replay.plan([1, 4], rows, batch_size=4, updates=3)


def test_the_extra_visits_keep_the_order_of_the_new_rows():
    slots, permutation = replay.positions(10, 4, seed=3, epoch=0)
    assert slots == sorted(set(slots)) and len(slots) == 4 and all(0 <= s < 14 for s in slots)
    assert sorted(permutation) == [0, 1, 2, 3]
    assert replay.positions(10, 4, seed=3, epoch=0) == (slots, permutation)
    assert replay.positions(10, 4, seed=3, epoch=1) != (slots, permutation)


def _closes(count, start=1_000):
    steps = np.sin(np.arange(count) * 0.37) * 0.01 + np.where(np.arange(count) > 90, -0.004, 0.002)
    return np.arange(start, start + count, dtype=np.int64) * 10, 100 * np.exp(np.cumsum(steps))


def test_the_market_route_only_uses_closes_known_at_the_decision():
    available, closes = _closes(160)
    moments = [available[0] - 1, available[0], available[70] + 5, available[150]]
    routes = replay.market_routes(available, closes, moments, "US")
    # Antes del primer cierre y con un solo cierre no hay rendimientos.
    assert routes[:2] == [UNCLASSIFIED, UNCLASSIFIED]
    for at, route in zip(moments[2:], routes[2:], strict=True):
        known = int(np.searchsorted(available, at, side="right"))
        values = np.log(closes[max(0, known - CONTEXT_SESSIONS) : known])
        window = np.zeros((1, CONTEXT_SESSIONS, len(PRICE_WINDOW_CHANNELS)))
        window[0, -len(values) :, PRICE_WINDOW_CHANNELS.index("log_close_to_anchor_close")] = (
            values - values[0]
        )
        window[0, -len(values) :, -1] = 1.0
        assert route == replay.RULE.state("US", window, int(at)).route != UNCLASSIFIED
    # Cambiar cierres posteriores a una decisión no cambia su régimen.
    later = closes.copy()
    later[71:] *= 3.0
    assert replay.market_routes(available, later, moments[:3], "US") == routes[:3]
    with pytest.raises(ValueError, match="ordenados"):
        replay.market_routes(available[::-1], closes, moments, "US")


# Datos de la campaña reducida: padre en fold-000 y filas nuevas de fold-001.


class Predictor:
    """Padre técnico que depende de las entradas y de los bits de presencia."""

    def __call__(self, inputs, presence=None):
        return inputs["charts"].mean(axis=1) + 0.01 * presence.sum(axis=1)


@pytest.fixture(scope="module")
def window(views, tmp_path_factory):  # noqa: F811
    root = tmp_path_factory.mktemp("replay-window")
    opened = matrix_runs.MatrixWindow(
        views.windows["fold-001"]["path"],
        root / "window",
        encoding=views.windows["fold-001"]["sha256"],
        input_policy=HISTORICAL_MASKED,
        batch_size=4,
        stop=SimpleNamespace(requested=False),
        max_block_bytes=64 * 1024**2,
        since=micros(SINCE),
    )
    cache = ParentCache(root / "parent.sqlite", "a" * 64, "masked", Predictor())
    yield SimpleNamespace(window=opened, cache=cache, parent_view=views.windows["fold-000"]["path"])
    cache.close()
    opened.close()


def _inputs(window, policy, seed=42, ratio=0.5, batch_size=4):
    earlier = replay.SOURCES[policy] == replay.EARLIER
    return ReplayInputs(
        window.window.train,
        window.window.validation,
        window.cache,
        spec=replay.spec(policy, ratio),
        batch_size=batch_size,
        seed=seed,
        candidates=window.window.earlier(window.parent_view) if earlier else None,
        open_earlier=window.window.open_sessions if earlier else None,
    )


def test_earlier_sessions_are_the_ones_the_parent_fitted_completely(views, window):  # noqa: F811
    since = micros(SINCE)
    found = window.window.earlier(window.parent_view)
    fitted = {
        (item["key"], int(row))
        for item in labels(views.datasets["fold-000"])
        for row, partition in zip(item["rows"], item["partition"], strict=True)
        if partition == "train"
    }
    sessions = {}
    for item in labels(views.datasets["fold-001"]):
        mask = (item["partition"] == "train") & (item["decision"] < since)
        for row, at, mature in zip(
            item["rows"][mask], item["decision"][mask], item["maturity"][mask], strict=True
        ):
            known, late, count = sessions.get(int(at), (True, False, 0))
            sessions[int(at)] = (
                known and (item["key"], int(row)) in fitted,
                late or mature >= since,
                count + 1,
            )
    admitted = sorted(at for at, (known, late, _) in sessions.items() if known and not late)
    assert found["moments"] == admitted and admitted
    assert found["rows"] == [sessions[at][2] for at in admitted]
    assert found["markets"] == [["US"]] * len(admitted)
    # La validación y la calibración del padre están en el ajuste de fold-001, antes de las
    # filas nuevas, y el repaso las deja fuera.
    assert found["not_fitted"] == sum(not known for known, _, _ in sessions.values()) > 0
    assert found["late"] == sum(known and late for known, late, _ in sessions.values())
    parent_validation = [
        at
        for item in labels(views.datasets["fold-000"])
        for at, partition in zip(item["decision"], item["partition"], strict=True)
        if partition in ("validation", "calibration")
    ]
    assert min(parent_validation) > max(admitted)
    # Sin factor de mercado en la vista, ninguna sesión se clasifica.
    assert found["routes"] == [UNCLASSIFIED] * len(admitted)
    assert window.window.earlier(window.parent_view) is found


def test_sessions_that_mature_late_or_change_samples_are_not_admitted(views, window):  # noqa: F811
    from mars_titan.posttraining.staged_rows import partition_rows

    fitted, samples = partition_rows(views.datasets["fold-000"], ("train",))
    dataset = views.datasets["fold-001"]
    # Con la frontera dentro del ajuste del padre, las últimas sesiones anteriores maduran
    # después de ella y quedan fuera aunque el padre las ajustara.
    decisions = sorted({int(at) for _, decided, _ in fitted["train"].values() for at in decided})
    inside = decisions[len(decisions) // 2] + 1
    found = replay.earlier_sessions(dataset, inside, fitted["train"], samples)
    assert found["late"] > 0 and max(found["moments"]) < inside
    maturity = {
        int(at): int(mature)
        for item in labels(dataset)
        for at, mature in zip(item["decision"], item["maturity"], strict=True)
    }
    assert all(maturity[at] < inside for at in found["moments"])
    changed = {key: "0" * 64 for key in samples}
    with pytest.raises(ValueError, match="archivo de muestras"):
        replay.earlier_sessions(dataset, micros(SINCE), fitted["train"], changed)


def test_a_session_with_one_row_the_parent_did_not_fit_is_not_admitted(views):  # noqa: F811
    from mars_titan.posttraining.staged_rows import partition_rows

    fitted, samples = partition_rows(views.datasets["fold-000"], ("train",))
    dataset, since = views.datasets["fold-001"], micros(SINCE)
    found = replay.earlier_sessions(dataset, since, fitted["train"], samples)
    # Una sesión admitida con dos filas pierde una de las filas que ajustó el padre.
    pairs = zip(found["moments"], found["rows"], strict=True)
    session = next(at for at, count in pairs if count > 1)
    key = next(key for key, (_, decided, _) in fitted["train"].items() if session in decided)
    indices, decided, mature = fitted["train"][key]
    keep = decided != session
    partial = {**fitted["train"], key: (indices[keep], decided[keep], mature[keep])}
    again = replay.earlier_sessions(dataset, since, partial, samples)
    assert again["moments"] == [at for at in found["moments"] if at != session]
    assert again["not_fitted"] == found["not_fitted"] + 1


def comparable(batch):
    cursor = {k: v for k, v in batch["confirmed_cursor"].items() if k != "data_sha256"}
    arrays = {
        name: value.tobytes() for name, value in batch.items() if isinstance(value, np.ndarray)
    }
    inputs = {name: value.tobytes() for name, value in batch["inputs"].items()}
    return arrays, inputs, batch["sample_ids"], batch["origin"], cursor["visit"] >= 0


def _epoch(data, epoch, seed=42, cursor=None, batch_size=4):
    return list(
        data.batches(
            partition="train",
            condition="real",
            batch_size=batch_size,
            epoch=epoch,
            seed=seed,
            cursor=cursor,
        )
    )


@pytest.mark.parametrize("policy", replay.CONTROLS)
def test_each_epoch_keeps_the_new_rows_and_adds_the_same_planned_visits(window, policy):
    since = micros(SINCE)
    plain = PairedInputs(window.window.train, window.window.validation, window.cache)
    with _inputs(window, policy) as data:
        budget = data.budget("real", 4)
        assert budget["real_rows"] == plain.counts["train"]
        new = plain.budget("real", 4)["updates"]
        assert budget["updates"] == new + math.ceil(0.5 * new) == new + data.extra["updates"]
        assert (
            budget["extra_rows"] == data.extra["rows"] == sum(r for _, r in data.extra["sessions"])
        )
        visited = []
        for epoch in (0, 1):
            batches = _epoch(data, epoch)
            assert len(batches) == budget["updates"]
            real = [comparable(b) for b in batches if b["origin"] == "real"]
            assert [item[:4] for item in real] == [comparable(b)[:4] for b in _epoch(plain, epoch)]
            extra = [b for b in batches if b["origin"] != "real"]
            assert {b["origin"] for b in extra} == {
                "repeat" if policy == replay.REPEAT else "replay"
            }
            assert sum(len(b["target"]) for b in extra) == budget["extra_rows"]
            moments = Counter()
            for b in extra:
                moments[int(b["prediction_at"][0])] += len(b["target"])
                if policy == replay.REPEAT:
                    assert (b["prediction_at"] >= since).all()
                else:
                    assert (b["prediction_at"] < since).all()
            assert dict(moments) == dict(map(tuple, data.extra["sessions"]))
            visited.append([b["origin"] for b in batches])
        # Las mismas visitas adicionales en posiciones nuevas cada época.
        assert visited[0] != visited[1]
        with pytest.raises(ValueError, match="otra semilla"):
            _epoch(data, 0, seed=43)
        with pytest.raises(ValueError, match="otro lote"):
            data.budget("real", 8)


@pytest.mark.parametrize("policy", [replay.UNIFORM, replay.REPEAT])
def test_a_session_cut_by_the_plan_only_gives_its_first_rows(window, policy, monkeypatch):
    """Con lotes de una fila y una sola actualización adicional, la sesión planificada con
    más filas se lee recortada a su primera fila."""
    if replay.SOURCES[policy] == replay.EARLIER:
        found = window.window.earlier(window.parent_view)
        moments, rows = found["moments"], found["rows"]
    else:
        moments = [at for at, _ in window.window.train.index]
        rows = [count for _, count in window.window.train.index]
    widest = max(range(len(rows)), key=lambda i: (rows[i], -i))
    assert rows[widest] > 1

    # El orden pone primero esa sesión para que el plan la recorte.
    def first(candidates, *_, **__):
        return [widest, *(i for i in candidates if i != widest)]

    monkeypatch.setattr(replay, "order", first)
    plain = PairedInputs(window.window.train, window.window.validation, window.cache)
    new = plain.budget("real", 1)["updates"]
    with _inputs(window, policy, ratio=0.5 / new, batch_size=1) as data:
        assert data.extra == dict(updates=1, rows=1, sessions=[[moments[widest], 1]])
        batches = _epoch(data, 0, batch_size=1)
        extra = [b for b in batches if b["origin"] != "real"]
        assert len(batches) == new + 1 and len(extra) == 1
        assert len(extra[0]["target"]) == 1
        assert int(extra[0]["prediction_at"][0]) == moments[widest]


def test_the_plan_and_identity_depend_on_the_seed_and_policy(window):
    with _inputs(window, replay.UNIFORM) as first, _inputs(window, replay.UNIFORM) as again:
        assert first.extra == again.extra and first.sha256 == again.sha256
        identity = first.identity["replay"]
        assert identity["policy"] == replay.UNIFORM and identity["ratio"] == 0.5
        assert identity["source"] == replay.EARLIER and identity["since"] == micros(SINCE)
        assert identity["updates"] == first.extra["updates"]
        assert identity["candidates"]["count"] == len(
            window.window.earlier(window.parent_view)["moments"]
        )
        assert identity["plan_sha256"] == replay.digest(first.extra["sessions"])
        assert identity["routes"] is None
    with _inputs(window, replay.UNIFORM, seed=43) as other:
        assert other.extra["sessions"] != first.extra["sessions"]
        assert other.sha256 != first.sha256
    with _inputs(window, replay.REGIME) as regime, _inputs(window, replay.REPEAT) as repeat:
        assert regime.identity["replay"]["routes"]["rule"] == replay.RULE.identity()
        assert repeat.identity["replay"]["candidates"] is None
        assert len({first.sha256, regime.sha256, repeat.sha256}) == 3
    plain = PairedInputs(window.window.train, window.window.validation, window.cache)
    assert "replay" not in plain.identity and plain.sha256 != first.sha256


@pytest.mark.parametrize("policy", [replay.UNIFORM, replay.REPEAT])
def test_a_cursor_inside_the_epoch_resumes_the_same_batches(window, policy):
    with _inputs(window, policy) as data:
        batches = _epoch(data, 1)
        extra = [i for i, b in enumerate(batches) if b["origin"] != "real"]
        for index in (extra[0], extra[len(extra) // 2], len(batches) // 3):
            resumed = _epoch(data, 1, cursor=batches[index]["confirmed_cursor"])
            assert [comparable(b) for b in resumed] == [comparable(b) for b in batches[index + 1 :]]


def test_a_replayed_session_that_matures_after_the_first_new_row_is_rejected(window):
    with _inputs(window, replay.UNIFORM) as data:
        # Con una frontera anterior a la última sesión planificada, esa sesión ya no maduró.
        data.since = max(at for at, _ in data.extra["sessions"])
        with pytest.raises(ValueError, match="madura después"):
            _epoch(data, 0)


def test_the_source_must_hold_exactly_the_planned_sessions(window):
    candidates = window.window.earlier(window.parent_view)

    def wrong(moments):
        # Las mismas sesiones salvo la primera, sustituida por otra candidata sin planificar.
        other = next(at for at in candidates["moments"] if at not in moments)
        return window.window.open_sessions(sorted([*moments[1:], other]))

    with pytest.raises(ValueError, match="exactamente las sesiones planificadas"):
        ReplayInputs(
            window.window.train,
            window.window.validation,
            window.cache,
            spec=replay.spec(replay.UNIFORM, 0.5),
            batch_size=4,
            seed=42,
            candidates=window.window.earlier(window.parent_view),
            open_earlier=wrong,
        )
    with pytest.raises(ValueError, match="otra fuente"):
        ReplayInputs(
            window.window.train,
            window.window.validation,
            window.cache,
            spec=replay.spec(replay.REPEAT, 0.5),
            batch_size=4,
            seed=42,
            candidates=window.window.earlier(window.parent_view),
            open_earlier=window.window.open_sessions,
        )


def test_the_run_requires_cohorts_with_the_replay_of_its_case(window, matrix):
    from mars_titan.posttraining import run
    from mars_titan.posttraining.inputs import fit_normalization

    document, digest = matrix
    cases = {
        item["control"]: item["case"]
        for item in adapter_matrix.cases(document, digest, "gru", head=QUANTILE_HEAD)
        if item["case"]["seed"] == 42 and item["control"] is not None
    }
    plain = PairedInputs(window.window.train, window.window.validation, window.cache)
    normalization = fit_normalization(plain, batch_size=4)
    stub = SimpleNamespace(
        identity=dict(checkpoint_sha256="a" * 64, input_policy=HISTORICAL_MASKED), quantiles=True
    )

    def check(data, case):
        return run._validate_run(
            data,
            case,
            window.window.grid,
            normalization,
            parent=stub,
            batch_size=4,
            device="cpu",
            diagnostic=True,
            lease=None,
            checkpoint_seconds=60,
            max_updates=None,
        )

    with _inputs(window, replay.UNIFORM) as data:
        _, budget = check(data, cases[replay.UNIFORM])
        assert budget == data.budget("real", 4)
        for case in (
            cases["full_continuation"],
            cases[replay.REGIME],
            dict(cases[replay.UNIFORM], seed=43),
        ):
            with pytest.raises(ValueError, match="no llevan el repaso"):
                check(data, case)
    with pytest.raises(ValueError, match="no llevan el repaso"):
        check(plain, cases[replay.UNIFORM])
    _, budget = check(plain, cases["full_continuation"])
    assert budget == plain.budget("real", 4)


def test_the_regime_of_each_session_comes_from_the_declared_factor(views, tmp_path):  # noqa: F811
    dataset = views.datasets["fold-001"]
    asset = dataset.assets[0]
    path = dataset._file(asset, "prices")
    manifest = dict(
        dataset.manifest,
        market_factors={"US": dict(prices_path=str(path), prices_sha256=sha256(path))},
    )
    table = pq.read_table(path, columns=["close", "available_at"])
    available = table["available_at"].cast(pa.timestamp("us", tz="UTC")).cast(pa.int64()).to_numpy()
    closes = table["close"].to_numpy()
    sessions = dict(
        moments=[int(available[100]) + 1, int(available[200]) + 1], markets=[["US"]] * 2
    )
    routes = replay.factor_routes(manifest, sessions)
    assert routes == replay.market_routes(available, closes, sessions["moments"], "US")
    two = dict(sessions, markets=[["US"], ["CN", "US"]])
    assert replay.factor_routes(manifest, two)[1] == UNCLASSIFIED
    broken = dict(
        manifest, market_factors={"US": dict(prices_path=str(path), prices_sha256="0" * 64)}
    )
    with pytest.raises(ValueError, match="ha cambiado"):
        replay.factor_routes(broken, sessions)


# Etapa A con los controles de repaso.


@pytest.fixture(scope="module")
def staged(tmp_path_factory):
    base = base_campaign(tmp_path_factory.mktemp("replay-stage"), "A")
    value = json.loads(base.stage.read_text())
    value["additional_controls"] = [*value["additional_controls"], SECTION]
    path = base.stage.parent / "stage-replay.json"
    atomic_json(path, value)
    return SimpleNamespace(base=base, stage=path)


def test_a_stage_plans_the_replay_only_when_it_names_the_section(staged):
    loaded = campaign_stage.load_stage(staged.stage)
    jobs = campaign_stage.plan_stage(loaded)
    found = [job for job in jobs if job["control"] in replay.CONTROLS]
    assert sorted(job["control"] for job in found) == sorted(replay.CONTROLS)
    assert all(campaign_stage.candidate_kind(job) == "continuation" for job in found)
    plain = campaign_stage.plan_stage(campaign_stage.load_stage(staged.base.stage))
    assert len(plain) == len(jobs) - 3


def test_a_stage_cannot_name_a_section_its_matrix_does_not_declare(staged, tmp_path):
    value = json.loads(staged.stage.read_text())
    matrix = json.loads((staged.stage.parent / "matrix.json").read_text())
    matrix.pop(SECTION)
    atomic_json(tmp_path / "matrix.json", matrix)
    value.update(
        campaign=str(staged.stage.parent / "campaign.json"), matrix=str(tmp_path / "matrix.json")
    )
    atomic_json(tmp_path / "stage.json", value)
    with pytest.raises(ValueError, match="controles que su matriz declara"):
        campaign_stage.load_stage(tmp_path / "stage.json")


def test_variant_b_cannot_name_the_replay(tmp_path):
    from tests.posttraining.campaign_fixture import write_configs

    _, stage = write_configs(tmp_path / "config", "B")
    value = json.loads(stage.read_text())
    matrix = json.loads((stage.parent / "matrix.json").read_text())
    v3 = json.loads(open(f"{CONFIGS}/adapter-matrix-v3.json").read())
    v3.pop("variety")
    v3["budget"] = matrix["budget"]
    atomic_json(stage.parent / "matrix.json", v3)
    value["additional_controls"] = [SECTION]
    atomic_json(stage, value)
    with pytest.raises(ValueError, match="walk-forward por etapas"):
        campaign_stage.load_stage(stage)


def test_the_stage_runs_the_three_controls_with_equal_updates(staged, tmp_path, recorder):
    output = tmp_path / "stage"
    with real_data_only():
        summary = campaign_stage.run_stage(
            staged.stage,
            staged.base.views,
            staged.base.output,
            output,
            lease=CpuLease,
            stop=SimpleNamespace(requested=False),
            device="cpu",
        )
    assert summary["status"] == "completed"
    assert summary["planned"]["training_jobs"] == 9
    found = {
        str(path.parent.relative_to(output / "jobs")): json.loads(path.read_text())
        for path in (output / "jobs").rglob("receipt.json")
    }
    full = found["US/fold-001/gru__full_continuation/fit-s42"]
    replays = {control: found[f"US/fold-001/gru__{control}/fit-s42"] for control in replay.CONTROLS}
    new = full["updates"]
    extra = math.ceil(0.5 * new)
    assert {receipt["updates"] for receipt in replays.values()} == {new + extra}
    # Cada optimizador da exactamente los pasos de su plan, sin cambiar pesos.
    calls = sorted(len(item.calls) for item in recorder.optimizers)
    assert calls == sorted([new] * 6 + [new + extra] * 3)
    proof = json.loads((output / "windows-data/US/fold-001/fit-rows.json").read_text())
    for control, receipt in replays.items():
        assert "replay" not in full
        record = receipt["replay"]
        assert record["policy"] == control and record["updates"] == extra
        assert record["since"] == proof["start_us"]
        # Las filas nuevas de la prueba son las mismas. El repaso se declara aparte.
        assert receipt["fit_rows"]["rows"] == full["fit_rows"]["rows"] == proof["rows"]
        assert receipt["selection"]["best_epoch"] == 0
        assert receipt["score"] == full["score"]
        assert receipt["identity"]["case"]["replay"] == dict(policy=control, ratio=0.5)
    sessions = {control: r["replay"]["plan_sha256"] for control, r in replays.items()}
    # En estas ventanas todas las sesiones admitidas quedan sin clasificar, un solo grupo, así
    # que el reparto por régimen coincide con el uniforme. La repetición planifica otras.
    assert sessions[replay.UNIFORM] == sessions[replay.REGIME] != sessions[replay.REPEAT]
    second = campaign_chain.read_selection(output, "US", "fold-001", "gru", 42)
    kinds = {item["arm"]: item["kind"] for item in second["candidates"]}
    assert len(kinds) == 10
    assert {kinds[f"gru__{control}"] for control in replay.CONTROLS} == {"continuation"}


def test_the_comparison_contrasts_the_replay_with_the_repetition(staged):
    declaration = json.loads(open(f"{CONFIGS}/historical-masked-adapter-comparison-a.json").read())
    family = declaration["families"]["versus_repeat_new_rows"]
    assert family == dict(base=replay.REPEAT, variants=[replay.UNIFORM, replay.REGIME])
    stage = campaign_stage.load_stage(staged.stage)
    groups = stage_comparison._groups(stage)
    for control in replay.CONTROLS:
        assert groups["gru"][control] == f"gru__{control}"
    config = stage_comparison.derive_config(declaration, stage, "gru", groups)
    assert config["comparison"]["families"]["versus_repeat_new_rows"] == dict(
        kind="delta",
        base=f"gru__{replay.REPEAT}",
        variants=[f"gru__{replay.UNIFORM}", f"gru__{replay.REGIME}"],
    )
    for control in replay.CONTROLS:
        assert config["arms"][f"gru__{control}"]["family"] == "posttraining_control"
    # Sin los controles en la etapa, la familia no se deriva y no aparece ningún repaso.
    plain_stage = campaign_stage.load_stage(staged.base.stage)
    plain = stage_comparison._groups(plain_stage)
    assert all(plain["gru"][control] is None for control in replay.CONTROLS)
    config = stage_comparison.derive_config(declaration, plain_stage, "gru", plain)
    assert "versus_repeat_new_rows" not in config["comparison"]["families"]
    assert not any(control in arm for arm in config["arms"] for control in replay.CONTROLS)
