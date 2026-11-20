"""Formas de las emisiones de ajuste sin acceder a valores ni predicciones."""

import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from benchmarks.comparison_efficiency_20261118.formas_emision import contar_formas, leer_indice
from mars_titan.data.storage import sha256


def test_la_lru_cuenta_el_orden_de_emision_y_el_resto_del_bloque():
    r = contar_formas([3, 5, 3, 0, 7], 4, (1, 2, 4))
    assert r["histograma"] == {1: 1, 3: 3, 4: 2}
    assert r["emisiones"] == 6 and r["bloques_completos"] == 2
    assert r["aciertos_lru"] == {1: 0, 2: 1, 4: 3}


def test_el_indice_solo_cuenta_entradas_y_comprueba_su_huella(tmp_path):
    archivo = tmp_path / "events.parquet"
    pq.write_table(pa.table({"event_at": [10, 10, 10, 20, 30], "kind": [0, 0, 1, 0, 2]}), archivo)
    manifest = dict(
        final_test_opened=False,
        identity={"phase": {"partition": "train"}},
        events_path=archivo.name,
        events_sha256=sha256(archivo),
    )
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    r = leer_indice(path)
    assert r["sesiones"] == 2 and r["filas"] == 3
    assert r["bloques"][512]["histograma"] == {1: 1, 2: 1}
    path.write_text(json.dumps(dict(manifest, events_sha256="0" * 64)))
    with pytest.raises(ValueError, match="huella"):
        leer_indice(path)
