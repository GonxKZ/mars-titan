"""Cargar referencias completas y recalcular sus predicciones sobre entradas efectivas."""

import copy
import math
from pathlib import Path

import numpy as np
import torch

from mars_titan.data.storage import sha256
from mars_titan.models.baselines.inputs import MODALITIES
from mars_titan.models.baselines.multimodal import MultimodalReference
from mars_titan.models.baselines.ridge import RidgeModel
from mars_titan.profiling import CostProbe
from mars_titan.training.experiment_resources import GpuLease
from mars_titan.training.predictive_parents import _parent, _signature
from mars_titan.training.reference_run import _confirmed_state

NEURAL = ("rnn", "lstm", "gru", "dlinear")


def require_device(device, diagnostic, lease):
    if diagnostic:
        if device != "cpu":
            raise ValueError("Los diagnósticos pequeños requieren CPU explícita")
    elif (
        device != "cuda:0"
        or not isinstance(lease, GpuLease)
        or lease.handle is None
        or lease.record is None
    ):
        raise ValueError("La ejecución científica requiere cuda:0 y un GpuLease activo")
    else:
        lease.check()


class FrozenParent:
    """Mantener un padre evaluable, separado de todas sus continuaciones."""

    def __init__(self, model, identity, shapes, device):
        self.model, self.identity, self.device = model, identity, device
        self.kind = identity["model"]
        self.shapes = {key: tuple(shape) for key, shape in shapes.items()}
        if self.kind in NEURAL:
            self.model.to(device).eval().requires_grad_(False)

    def predict(self, inputs):
        if not isinstance(inputs, dict) or set(inputs) != set(MODALITIES):
            raise ValueError("El padre necesita todas las modalidades")
        count = len(inputs["prices"])
        if (
            not 1 <= count <= 4096
            or sum(np.asarray(v).nbytes for v in inputs.values()) > 64 * 1024**2
            or any(
                v.shape != (count, *self.shapes[k]) or not np.isfinite(v).all()
                for k, v in inputs.items()
            )
        ):
            raise ValueError("Las dimensiones o valores no coinciden con el padre")
        predictions = []
        with torch.inference_mode():
            for start in range(0, count, 256):
                chunk = {k: v[start : start + 256] for k, v in inputs.items()}
                if self.kind in NEURAL:
                    tensors = {k: torch.tensor(v, device=self.device) for k, v in chunk.items()}
                    prediction = self.model(tensors).double().cpu().numpy()
                else:
                    matrix = np.concatenate(
                        [chunk[k].reshape(len(chunk[k]), -1) for k in MODALITIES], axis=1
                    )
                    prediction = self.model.predict(matrix)
                predictions.append(np.asarray(prediction, dtype=np.float64))
        result = np.concatenate(predictions)
        if result.shape != (count,) or not np.isfinite(result).all():
            raise ValueError("El padre no produce una predicción finita por fila")
        return result

    def continuation(self):
        if self.kind not in NEURAL:
            raise ValueError("La continuación neuronal requiere pesos de un padre neuronal")
        return copy.deepcopy(self.model).requires_grad_(True)


def _inference_contract(report, kind):
    contract = report.get("identity", {})
    names = ["training/corpus_inputs.py"]
    if kind in NEURAL:
        names.extend(
            (
                "models/baselines/dlinear.py",
                "models/baselines/multimodal.py"
                if contract.get("model_family") == "scientific_multimodal_reference"
                else "profiling.py",
            )
        )
    elif kind in {"ridge", "xgboost_external_cuda"}:
        implementation = "ridge" if kind == "ridge" else "external_boosting"
        names.extend(("models/baselines/inputs.py", f"models/baselines/{implementation}.py"))
    else:
        raise ValueError("La familia no tiene un contrato de inferencia admitido")
    hashes = contract.get("code", report.get("code", {}))
    if any(hashes.get(name) != sha256(Path(__file__).parents[1] / name) for name in names):
        raise ValueError("El padre no conserva el contrato de su código de inferencia")


def _neural_parent(report, source, report_path):
    shapes = source["shapes"]
    kind = report["identity"]["case"]["kind"]
    contract = report["identity"]
    dimensions = {key: shape[-1] for key, shape in shapes.items()}
    if contract.get("dimensions") != dimensions or contract.get("context") != shapes["prices"][0]:
        raise ValueError("Las dimensiones del padre no corresponden al corpus")
    case = contract["case"]
    family = contract.get("model_family")
    if family == "scientific_multimodal_reference" and "architecture" in case:
        model = MultimodalReference(
            kind, dimensions, context=contract["context"], **case["architecture"]
        )
    elif family == "legacy_cost_probe" and "architecture" not in case:
        model = CostProbe(kind, dimensions, context=contract["context"])
    else:
        raise ValueError("El padre no declara un constructor compatible")
    state = _confirmed_state(
        report_path.parent, contract, report["checkpoint"], report.get("selection")
    )
    if not case.get("selection") and (
        state.get("epoch") != case["epochs"]
        or state.get("confirmed_cursor") is not None
        or state.get("statistics", {}).get("samples") != 0
    ):
        raise ValueError("El checkpoint del padre no confirma una época completa")
    model.load_state_dict(state["model"], strict=True)
    if any(not torch.isfinite(value).all() for value in model.state_dict().values()):
        raise ValueError("El padre contiene pesos no finitos")
    return model


def load_parent(ordered, report_path, *, device="cuda:0", diagnostic=False, lease=None):
    """Exigir recibo completo, población idéntica y estado seleccionado íntegro."""
    require_device(device, diagnostic, lease)
    ordered, report_path = Path(ordered), Path(report_path)
    source, report, identity, (checkpoint, signature) = _parent(ordered, report_path)
    if source["scope"] != "full_corpus" or source["cohort_complete"] is not True:
        raise ValueError("El padre debe haberse ajustado a toda la población admitida")
    if identity["weighting"] != "natural":
        raise ValueError("El diseño emparejado fija la ponderación natural")
    shapes, kind = source["shapes"], identity["model"]
    _inference_contract(report, kind)
    features = sum(math.prod(shape) for shape in shapes.values())
    if diagnostic and max(source["counts"].values()) > 5000:
        raise ValueError("El diagnóstico CPU supera el presupuesto de 5000 filas")
    if kind in NEURAL:
        model = _neural_parent(report, source, report_path)
    elif kind == "ridge" and not diagnostic:
        model = RidgeModel.load(checkpoint)
        if len(model.mean) != features or report.get("features") != features:
            raise ValueError("Las dimensiones tabulares del padre no coinciden")
    elif kind == "xgboost_external_cuda" and not diagnostic:
        from mars_titan.models.baselines.external_boosting import ExternalBoostingModel

        model = ExternalBoostingModel.load(
            checkpoint, identity["checkpoint_sha256"], training_rows=source["counts"]["train"]
        )
        if model.features != features:
            raise ValueError("Las dimensiones tabulares del padre no coinciden")
    else:
        raise ValueError("El constructor de este padre no pertenece al diseño o dispositivo")
    if _signature(checkpoint) != signature:
        raise ValueError("El punto de control del padre ha cambiado durante la carga")
    return FrozenParent(model, identity, shapes, device)
