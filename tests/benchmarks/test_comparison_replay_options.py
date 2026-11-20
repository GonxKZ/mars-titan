"""Huella de selección, aislamiento de datos y derivadas de las variantes del ensayo."""

from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from benchmarks.comparison_efficiency_20261118.opciones_replay import seleccionar, variante
from mars_titan.models.titans import MACConfig, MemoryConfig, TitansMAC
from mars_titan.training import financial_run as fr
from mars_titan.training.titans_walk_forward import unfused_attention
from tests.models.titans.test_financial_adapter import setup


def test_el_corte_cpu_rechaza_un_archivo_de_igual_tamano_con_otro_contenido(tmp_path, monkeypatch):
    from mars_titan.data.storage import sha256
    from mars_titan.memory import event_archive

    scripts = Path(__file__).resolve().parents[2] / "benchmarks/comparison_efficiency_20261118"
    monkeypatch.syspath_prepend(str(scripts))
    from replay_cpu import lotes_ajuste

    path = tmp_path / "corte.arrow"

    def escribir(instante):
        row = event_archive._encode(
            SimpleNamespace(at=instante, close_phase=True, inputs=(), labels=())
        )
        with pa.OSFile(str(path), "wb") as target, pa.ipc.new_file(target, row.schema) as writer:
            writer.write_batch(row)

    escribir(1)
    receipt = dict(events=1, block=1024, bytes=path.stat().st_size, sha256=sha256(path))
    escribir(2)
    assert path.stat().st_size == receipt["bytes"]
    with pytest.raises(event_archive.InvalidArchive, match="huella"):
        lotes_ajuste(path, receipt, None, 1, 2)


class CuentaOperaciones(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.cuentas = Counter()

    def __torch_dispatch__(self, op, types, args=(), kwargs=None):
        self.cuentas[str(op)] += 1
        return op(*args, **(kwargs or {}))


@pytest.mark.parametrize("modo", ["online", "frozen", "disabled"])
def test_una_mascara_por_modelo_conserva_huella_estado_y_gradiente(modo):
    original, lote = setup("mac_" + modo)
    base = original.prepare(lote, original.initial_state(lote.flow_ids), differentiable=True)
    with variante("mascara"):
        nuevo, _ = setup("mac_" + modo)
        nuevo.load_state_dict(original.state_dict(), strict=True)
        assert nuevo._parameter_id == original._parameter_id
        with CuentaOperaciones() as cuenta:
            actual = nuevo.prepare(lote, nuevo.initial_state(lote.flow_ids), differentiable=True)
        assert cuenta.cuentas["aten.triu.default"] == 0
        assert torch.equal(base.point_predictions, actual.point_predictions)
        for p, q in zip(
            base.next_state.mac.memory.weights, actual.next_state.mac.memory.weights, strict=True
        ):
            assert torch.equal(p, q)
        for resultado in (base, actual):
            resultado.point_predictions.square().sum().backward()
        for p, q in zip(original.parameters(), nuevo.parameters(), strict=True):
            assert (p.grad is None) == (q.grad is None)
            if p.grad is not None:
                assert torch.equal(p.grad, q.grad)


def test_otras_longitudes_calculan_su_mascara():
    with variante("mascara"):
        modelo = TitansMAC(MACConfig(MemoryConfig(4, max_tokens=4), max_segment=4))
        with CuentaOperaciones() as cuenta:
            modelo(torch.ones(2, 3, 4), modelo.initial_state(2))
        assert cuenta.cuentas["aten.triu.default"] == 1


def test_la_seleccion_identidad_guarda_la_huella_de_seleccion_y_copias_independientes():
    _, lote = setup()
    filas = list(range(len(lote.flow_ids)))
    base = lote.select(filas)
    with CuentaOperaciones() as cuenta:
        prueba = seleccionar(lote, filas)
    assert cuenta.cuentas["aten.index.Tensor"] == 0
    assert prueba.input_digest == base.input_digest != lote.input_digest
    for original, referencia, clon in zip(
        (*lote.inputs.values(), lote.presence),
        (*base.inputs.values(), base.presence),
        (*prueba.inputs.values(), prueba.presence),
        strict=True,
    ):
        assert torch.equal(referencia, clon) and referencia.stride() == clon.stride()
        assert clon.untyped_storage().data_ptr() != original.untyped_storage().data_ptr()
    assert seleccionar(lote, filas[::-1]).input_digest == lote.select(filas[::-1]).input_digest
    prueba.inputs["prices"].zero_()
    lote.verify()
    with pytest.raises(ValueError, match="cambiaron"):
        prueba.verify()


def test_la_reunion_de_perdidas_conserva_el_valor_y_los_gradientes_entre_grupos():
    modelo, lote = setup()
    receta = fr.ChronologicalRecipe(accumulation_rows=1)
    claves = list(zip(lote.flow_ids, lote.prediction_at, strict=True))
    run = SimpleNamespace(
        used=[(f, t, t) for f, t in claves],
        targets=[1.25, -0.5],
        penalties=[],
        starts=dict.fromkeys(lote.flow_ids),
        segment=[(lote, {}, False)],
    )
    with unfused_attention():
        motor = fr.ChronologicalTrainer.__new__(fr.ChronologicalTrainer)
        fr.ChronologicalInference.__init__(motor, modelo, receta)
        motor.teacher = None
        esperado = motor._replay(run)
        gradientes = [p.grad for p in modelo.parameters()]
        modelo.zero_grad(set_to_none=True)
        with variante("perdidas"):
            observado = motor._replay(run)
    assert observado.dtype == esperado.dtype and observado.device == esperado.device
    assert torch.equal(observado, esperado)
    for original, parametro in zip(gradientes, modelo.parameters(), strict=True):
        assert (original is None) == (parametro.grad is None)
        if original is not None:
            assert torch.equal(original, parametro.grad)
