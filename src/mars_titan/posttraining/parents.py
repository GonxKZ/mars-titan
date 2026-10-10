"""Cargar referencias completas y recalcular sus predicciones sobre entradas efectivas."""

import math
from pathlib import Path

import numpy as np
import torch

from mars_titan.data.input_policy import STRICT_INPUTS, masked_inputs
from mars_titan.data.storage import sha256
from mars_titan.environments.cohorts import MAX_COHORT_ASSETS
from mars_titan.models.baselines.inputs import MODALITIES
from mars_titan.models.baselines.multimodal import (
    PRESENCE_FUSION,
    SCALAR_HEAD,
    STRICT_FUSION,
    MultimodalReference,
    transformer_batch_options,
)
from mars_titan.models.baselines.ridge import RidgeModel
from mars_titan.models.predictive_adaptation import adapted_copy, parent_copy
from mars_titan.models.quantile_head import CONTRACT, QUANTILE_HEAD, median
from mars_titan.profiling import CostProbe
from mars_titan.training.experiment_resources import GpuLease
from mars_titan.training.predictive_parents import _parent, _signature
from mars_titan.training.reference_run import _confirmed_state

NEURAL = ("rnn", "lstm", "gru", "dlinear", "transformer")
# Orden de los bits de presencia, el mismo de data.input_policy y de los padres tabulares.
PRESENCE_BITS = len(MODALITIES)
# Una cohorte completa de 8.192 filas con las 1.714 columnas float32 de la edición ocupa
# unos 56 MB. El padre la recorre en bloques de 256 filas y no copia más entradas.
MAX_INPUT_BYTES = 64 * 1024**2


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
        self.input_policy = identity.get("input_policy", STRICT_INPUTS)
        self.masked = masked_inputs(self.input_policy)
        self.shapes = {key: tuple(shape) for key, shape in shapes.items()}
        # Un padre de cuantiles aporta su mediana como predicción puntual del padre.
        self.quantiles = bool(getattr(model, "emits_quantiles", False))
        if self.quantiles != ("output_head" in identity):
            raise ValueError("La cabeza del padre no coincide con su identidad")
        if self.kind in NEURAL:
            self.model.to(device).eval().requires_grad_(False)

    def _presence(self, inputs, presence, count):
        if not self.masked:
            if presence is not None:
                raise ValueError("Un padre estricto no admite bits de presencia")
            return None
        if (
            not isinstance(presence, np.ndarray)
            or presence.dtype != np.bool_
            or presence.shape != (count, PRESENCE_BITS)
            or not presence[:, [MODALITIES.index("prices"), MODALITIES.index("charts")]].all()
        ):
            raise ValueError("El padre con máscaras necesita cinco bits válidos por fila")
        for index, name in enumerate(MODALITIES):
            if np.any(inputs[name].reshape(count, -1)[~presence[:, index]] != 0):
                raise ValueError("Un bloque ausente del padre contiene valores distintos de cero")
        return presence

    def predict(self, inputs, presence=None):
        if not isinstance(inputs, dict) or set(inputs) != set(MODALITIES):
            raise ValueError("El padre necesita todas las modalidades")
        count = len(inputs["prices"])
        if (
            not 1 <= count <= MAX_COHORT_ASSETS
            or sum(np.asarray(v).nbytes for v in inputs.values()) > MAX_INPUT_BYTES
            or any(
                v.shape != (count, *self.shapes[k]) or not np.isfinite(v).all()
                for k, v in inputs.items()
            )
        ):
            raise ValueError("Las dimensiones o valores no coinciden con el padre")
        presence = self._presence(inputs, presence, count)
        predictions = []
        with torch.inference_mode():
            for start in range(0, count, 256):
                chunk = {k: v[start : start + 256] for k, v in inputs.items()}
                bits = None if presence is None else presence[start : start + 256]
                if self.kind in NEURAL:
                    tensors = {k: torch.tensor(v, device=self.device) for k, v in chunk.items()}
                    extra = () if bits is None else (torch.tensor(bits, device=self.device),)
                    output = self.model(tensors, *extra)
                    output = median(output) if self.quantiles else output
                    prediction = output.double().cpu().numpy()
                else:
                    # Mismo orden que la matriz tabular: modalidades y, al final, los bits.
                    blocks = [chunk[k].reshape(len(chunk[k]), -1) for k in MODALITIES]
                    matrix = np.concatenate(blocks + ([] if bits is None else [bits]), axis=1)
                    prediction = self.model.predict(matrix)
                predictions.append(np.asarray(prediction, dtype=np.float64))
        result = np.concatenate(predictions)
        if result.shape != (count,) or not np.isfinite(result).all():
            raise ValueError("El padre no produce una predicción finita por fila")
        return result

    def _require_neural(self):
        # Un padre de cuantiles conserva su cabeza ordenada. El caso declara su pinball y
        # `run._validate_run` rechaza un objetivo que no corresponda a la cabeza.
        if self.kind not in NEURAL:
            raise ValueError("La continuación y los adaptadores requieren un padre neuronal")

    def continuation(self):
        self._require_neural()
        return parent_copy(self.model).requires_grad_(True)

    def adapted(self, targets, *, seed):
        """Copiar el padre congelado con correcciones nulas en los destinos declarados."""
        self._require_neural()
        return adapted_copy(self.model, targets, seed=seed)


def _inference_contract(report, kind, *, masked=False):
    contract = report.get("identity", {})
    names = ["training/corpus_inputs.py"] + (["data/input_policy.py"] if masked else [])
    if set(report.get("samples", {})) == {"train", "validation", "calibration", "evaluation"}:
        names.extend(
            ("training/temporal_corpus.py", "evaluation/splits.py", "evaluation/split_readiness.py")
        )
    if kind in NEURAL:
        names.extend(
            (
                "models/baselines/dlinear.py",
                "models/baselines/multimodal.py"
                if contract.get("model_family") == "scientific_multimodal_reference"
                else "profiling.py",
            )
        )
        if kind == "transformer":
            names.append("models/baselines/transformer.py")
        if contract.get("case", {}).get("head") == QUANTILE_HEAD:
            names.append("models/quantile_head.py")
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
    fusion = contract.get("mask_fusion", STRICT_FUSION)
    if (fusion == PRESENCE_FUSION) != masked_inputs(contract.get("input_policy", STRICT_INPUTS)):
        raise ValueError("La fusión del padre no corresponde a su política de entradas")
    head = case.get("head", SCALAR_HEAD)
    if (head == QUANTILE_HEAD) != ("output_head" in contract) or (
        "output_head" in contract and contract["output_head"] != CONTRACT
    ):
        raise ValueError("La cabeza del padre no corresponde a su contrato de salida")
    if family == "scientific_multimodal_reference" and "architecture" in case:
        model = MultimodalReference(
            kind,
            dimensions,
            context=contract["context"],
            mask_fusion=fusion,
            head=head,
            **case["architecture"],
            # Un Transformer ajustado con lotes de más de 256 ventanas guarda su lote máximo
            # en el estado, así que el padre se reconstruye con el lote de su identidad. Una
            # identidad sin lote conserva el contrato por defecto, y las demás familias no
            # dependen del lote.
            **transformer_batch_options(kind, contract.get("batch_size", 1)),
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
    if any(
        torch.is_tensor(value) and not torch.isfinite(value).all()
        for value in model.state_dict().values()
    ):
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
    masked = masked_inputs(identity.get("input_policy", STRICT_INPUTS))
    _inference_contract(report, kind, masked=masked)
    # Los padres tabulares con máscaras reciben los cinco bits después de las modalidades.
    features = sum(math.prod(shape) for shape in shapes.values()) + (PRESENCE_BITS if masked else 0)
    if diagnostic and max(source["counts"].values()) > 5000:
        raise ValueError("El diagnóstico CPU supera el presupuesto de 5000 filas")
    if kind in NEURAL:
        model = _neural_parent(report, source, report_path)
    elif kind == "ridge" and not diagnostic:
        model = RidgeModel.load(checkpoint)
        if (
            len(model.mean) != features
            or report.get("features") != features
            or (masked and report.get("feature_order") != [*MODALITIES, "presence"])
        ):
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
