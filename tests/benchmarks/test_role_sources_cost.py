"""La medida de coste de las fuentes de B3 recorre cada fuente y tamaño sin ajustar nada."""

import json

import pytest

from benchmarks import role_sources_cost as cost
from mars_titan.memory.role_binding import SOURCES


def test_a_small_measure_covers_every_source_with_its_own_routes(tmp_path):
    output = tmp_path / "cost.json"
    cost.measure(output, sizes=(12, 20), warmup=1, repetitions=2)
    receipt = json.loads(output.read_text())
    assert receipt["issue"] == 750 and receipt["kind"] == "role_source_cost"
    assert receipt["optimizer_steps"] == 0 and receipt["training_runs"] == 0
    assert set(receipt["cases"]) == {f"{name}/{rows}" for name in SOURCES for rows in (12, 20)}
    assert set(receipt["rules"]) == set(SOURCES)
    assert "src/mars_titan/memory/role_sources.py" in receipt["sources_sha256"]
    for name, case in receipt["cases"].items():
        assert sum(case["occupancy"]) == int(name.split("/")[1])
        assert len(case["samples_ms"]) == 2 and case["min_ms"] <= case["p50_ms"]
    # Noticias en las filas pares y fundamentales en dos de cada tres: rutas 2, 3, 4, 1, ...
    assert receipt["cases"]["modality/12"]["occupancy"] == [0, 2, 2, 4, 4]
    assert receipt["cases"]["market/12"]["occupancy"] == [0, 12, 0, 0, 0]
    # Con menos de 20 activos el régimen no clasifica la cohorte.
    assert receipt["cases"]["regime/12"]["occupancy"] == [12, 0, 0, 0, 0]
    assert receipt["cases"]["regime/20"]["occupancy"][0] == 0
    sector = receipt["cases"]["sector/20"]["occupancy"]
    assert sector[0] == 0 and sector == receipt["cases"]["permuted_sector/20"]["occupancy"]
    with pytest.raises(FileExistsError):
        cost.measure(output, sizes=(12,), warmup=0, repetitions=1)


@pytest.mark.parametrize(
    "options",
    [
        dict(sizes=()),
        dict(sizes=(0,)),
        dict(sizes=(4097,)),
        dict(sizes=(8, 8)),
        dict(sizes=(True,)),
        dict(warmup=-1),
        dict(warmup=11),
        dict(repetitions=0),
        dict(repetitions=101),
        dict(repetitions=2.0),
    ],
)
def test_invalid_measures_are_refused_before_measuring(tmp_path, options):
    output = tmp_path / "cost.json"
    with pytest.raises(ValueError, match="presupuesto"):
        cost.measure(output, **dict(dict(sizes=(8,), warmup=0, repetitions=1), **options))
    assert not output.exists()
