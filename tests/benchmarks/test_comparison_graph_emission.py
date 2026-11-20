"""Errores de entrada y conservación del estado en el control del padre de M2."""

import hashlib
import json
import subprocess
import sys

import pyarrow as pa
import pytest
import torch

from benchmarks.comparison_efficiency_20261118 import estado_grafo_m2 as ensayo
from mars_titan.models.titans import MACConfig, MemoryConfig, TitansMAC
from mars_titan.models.titans.state import mac_tensors
from tests.models.titans.test_financial_adapter import specification


def origen_control(raiz):
    espec = specification()
    rutas = raiz / "runs"
    padre = rutas / "titans_mac/2023/seed-43"
    (padre / "fit").mkdir(parents=True)
    lector = rutas / "mars_titan_m2/2023/seed-43"
    lector.mkdir(parents=True)
    vista = dict(
        status="completed", final_test_opened=False, identity={"indices": {"train": "a" * 64}}
    )
    (lector / "run.json").write_text(
        json.dumps(
            dict(
                **vista,
                request={"parent": {"checkpoint_sha256": "b" * 64}},
                fit={"best_checkpoint": {"sha256": "d" * 64}},
            )
        )
    )
    (padre / "run.json").write_text(
        json.dumps(dict(**vista, fit={"best_checkpoint": {"sha256": "b" * 64}}))
    )
    (padre / "fit/run.json").write_text(
        json.dumps({"identity": {"predictor": {"inputs": espec.identity()}}})
    )
    ruta = raiz / "small.arrow"
    with pa.OSFile(str(ruta), "wb") as f, pa.ipc.new_file(f, ensayo.archivo.SCHEMA):
        pass
    datos = ruta.read_bytes()
    sello = dict(
        index="a" * 64,
        spec=espec.fingerprint(),
        verified={"digest": "c" * 64},
        block=512,
        bytes=len(datos),
        sha256=hashlib.sha256(datos).hexdigest(),
        events=0,
    )
    (raiz / "small.json").write_text(json.dumps(sello))
    return ruta, datos


def test_rechaza_el_arrow_que_conserva_el_tamano_y_cambia_de_huella(tmp_path, monkeypatch):
    ruta, datos = origen_control(tmp_path)
    ruta.write_bytes(bytes([datos[0] ^ 1]) + datos[1:])
    monkeypatch.setattr(
        ensayo.rw,
        "_parent",
        lambda *args: pytest.fail("La huella debe fallar antes del checkpoint"),
    )
    with pytest.raises(ensayo.archivo.InvalidArchive, match="huella"):
        ensayo.materiales(tmp_path, tmp_path, 0, "cpu")


def test_rechaza_el_reemplazo_del_archivo_durante_el_mapeo(tmp_path, monkeypatch):
    ruta, datos = origen_control(tmp_path)
    monkeypatch.setattr(ensayo.rw, "_parent", lambda *args: ({"identity": {"recipe": {}}}, None))
    monkeypatch.setattr(ensayo.rw, "_frozen_parent", lambda *a, **kw: None)
    abrir = pa.ipc.open_file

    def reemplazar(*a, **kw):
        lector = abrir(*a, **kw)
        nueva = tmp_path / "otra.arrow"
        nueva.write_bytes(datos)
        nueva.replace(ruta)
        return lector

    monkeypatch.setattr(pa.ipc, "open_file", reemplazar)
    with pytest.raises(ensayo.archivo.InvalidArchive, match="lectura"):
        ensayo.materiales(tmp_path, tmp_path, 0, "cpu")


def test_el_cli_rechaza_un_recibo_existente_antes_de_abrir_el_corpus(tmp_path, monkeypatch):
    recibo = tmp_path / "resultado.json"
    recibo.write_text('{"anterior": true}')
    monkeypatch.setattr(
        sys,
        "argv",
        ["control", "--estado", "estado", "--archivo", "archivo", "--salida", str(recibo)],
    )
    with pytest.raises(ValueError, match="El recibo ya existe"):
        ensayo.principal()
    assert recibo.read_text() == '{"anterior": true}'


def test_el_cli_impide_que_python_omita_las_comprobaciones_de_identidad():
    proceso = subprocess.run(
        [sys.executable, "-O", "-B", ensayo.__file__, "--help"], capture_output=True, text=True
    )
    assert proceso.returncode != 0
    assert "sin -O" in proceso.stderr


def test_dos_emisiones_cpu_de_512_filas_avanzan_el_estado_y_conservan_el_primer_error():
    mac = TitansMAC(
        MACConfig(
            MemoryConfig(2, depth=1, max_batch=512, max_tokens=1),
            max_segment=1,
            persistent_tokens=1,
        )
    )
    bloque = ensayo.Bloque(mac, por_grafo=False)
    token = torch.ones(512, 1, 2)
    inicial = mac.initial_state(512)
    with torch.no_grad():
        _, primero = bloque.pasar(token, inicial)
        guardado = [x.clone() for x in mac_tensors(primero)]
        salida, segundo = bloque.pasar(-token, primero)
        esperada, referencia = mac(-token, primero)
        assert ensayo.validar_error(bloque) == "El contador de pasos desbordaría int64"
    assert torch.equal(salida, esperada)
    assert torch.all(segundo.memory.steps == 2)
    for previo, original, obtenido, esperado in zip(
        guardado, mac_tensors(primero), mac_tensors(segundo), mac_tensors(referencia), strict=True
    ):
        assert torch.equal(previo, original)
        assert torch.equal(obtenido, esperado)
